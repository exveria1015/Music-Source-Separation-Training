# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'
__version__ = '1.0.5'

import argparse
from contextlib import nullcontext
import json
import os
import sys
import time

import numpy as np
from tqdm.auto import tqdm
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from ml_collections import ConfigDict
from typing import Any, List, Callable, Optional, Union
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from utils.settings import get_scheduler, parse_args_train, initialize_environment_ddp, \
    initialize_environment, get_model_from_config, wandb_init, uses_internal_model_loss, validate_train_setup, \
    apply_config_args
from utils.model_utils import normalize_batch, \
    save_eval_weights, save_last_weights, initialize_model_and_device

from valid import valid_multi_gpu, valid

import warnings

warnings.filterwarnings("ignore")


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def unwrap_parallel_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def model_for_validation(model: torch.nn.Module, ema_model: Optional[torch.nn.Module] = None) -> torch.nn.Module:
    if ema_model is not None:
        return ema_model
    return unwrap_parallel_model(model)


def synchronize_batch_success(success: bool, device: torch.device) -> bool:
    if not dist.is_initialized():
        return success

    reduce_device = torch.device("cpu") if dist.get_backend() == "gloo" else device
    flag = torch.tensor(1 if success else 0, device=reduce_device, dtype=torch.int)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def broadcast_float_from_main(value: Optional[float], device: torch.device) -> float:
    if not dist.is_initialized():
        return float(value) if value is not None else float("nan")

    broadcast_device = torch.device("cpu") if dist.get_backend() == "gloo" else device
    tensor_value = float(value) if value is not None else float("nan")
    value_tensor = torch.tensor([tensor_value], device=broadcast_device, dtype=torch.float64)
    dist.broadcast(value_tensor, src=0)
    return float(value_tensor.item())


def log_wandb(metrics: dict) -> None:
    if wandb.run is not None:
        wandb.log(metrics)


def append_epoch_log(args: argparse.Namespace, metrics: dict) -> None:
    if not is_main_process() or not getattr(args, "results_path", None):
        return
    os.makedirs(args.results_path, exist_ok=True)
    record = {"time": time.time(), **metrics}
    with open(os.path.join(args.results_path, "train_log.jsonl"), "a", encoding="utf-8") as out:
        out.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def create_grad_scaler(device: torch.device, enabled: bool):
    enabled = enabled and device.type == "cuda"
    if hasattr(torch, "amp"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def loss_is_finite(loss: torch.Tensor) -> bool:
    return bool(torch.isfinite(loss.detach()).all().item())


def gradients_are_finite(parameters) -> bool:
    for param in parameters:
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return False
    return True


def cuda_memory_stats(device: torch.device) -> dict:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    return {
        "cuda_allocated_mb": round(torch.cuda.memory_allocated(device) / 1024 / 1024, 2),
        "cuda_reserved_mb": round(torch.cuda.memory_reserved(device) / 1024 / 1024, 2),
    }


def normalize_active_stem_ids(active_stem_ids):
    if active_stem_ids is None:
        return None

    if torch.is_tensor(active_stem_ids):
        ids = active_stem_ids.detach().cpu()
        if ids.ndim == 0:
            return [int(ids.item())]
        if ids.dtype == torch.bool or ids.ndim == 2:
            if ids.ndim == 2:
                ids = ids.any(dim=0)
            else:
                ids = ids.to(torch.bool)
            ids = torch.nonzero(ids, as_tuple=False).flatten().tolist()
            return [int(i) for i in ids] if ids else None
        return [int(i) for i in ids.flatten().tolist()]

    return [int(i) for i in active_stem_ids]


def forward_step(x, y, active_stem_ids, get_internal_loss, model, multi_loss, device_ids):
    if get_internal_loss:
        active_stem_ids = normalize_active_stem_ids(active_stem_ids)
        loss = model(x, y, active_stem_ids=active_stem_ids)
        if isinstance(device_ids, (list, tuple)):
            loss = loss.mean()
        return loss
    else:
        y_ = model(x)
        return multi_loss(y_, y, x)



def train_one_epoch(model: torch.nn.Module, config: ConfigDict, args: argparse.Namespace,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, device_ids: List[int], epoch: int, use_amp: bool,
                    scaler: Any,
                    scheduler,
                    gradient_accumulation_steps: int, train_loader: torch.utils.data.DataLoader,
                    multi_loss: Callable[[torch.Tensor, torch.Tensor, torch.Tensor,], torch.Tensor], all_losses=None, world_size=None, ema_model=None, safe_mode=None) -> None:
    """
    Train the model for one epoch.

    Args:
        world_size:
        scheduler:
        model: The model to train.
        config: Configuration object containing training parameters.
        args: Command-line arguments with specific settings (e.g., model type).
        optimizer: Optimizer used for training.
        device: Device to run the model on (CPU or GPU).
        device_ids: List of GPU device IDs if using multiple GPUs.
        epoch: The current epoch number.
        use_amp: Whether to use automatic mixed precision (AMP) for training.
        scaler: Scaler for AMP to manage gradient scaling.
        gradient_accumulation_steps: Number of gradient accumulation steps before updating the optimizer.
        train_loader: DataLoader for the training dataset.
        multi_loss: The loss function to use during training.

    Returns:
        None
    """
    ddp = True if world_size else False
    should_print = is_main_process()
    model.train()
    if not ddp:
        model.to(device)
    if should_print:
        print(f'Train epoch: {epoch} Learning rate: {optimizer.param_groups[0]["lr"]}')
        sys.stdout.flush()
    loss_val = 0.
    total = 0
    all_losses = all_losses if all_losses is not None else {}
    all_losses[f'epoch_{epoch}'] = []
    optimizer.zero_grad(set_to_none=True)

    normalize = getattr(config.training, 'normalize', False)

    get_internal_loss = uses_internal_model_loss(args.model_type, args.use_standard_loss)
    amp_enabled = use_amp and getattr(device, "type", None) == "cuda"
    non_blocking = bool(getattr(args, 'pin_memory', False) and device.type == 'cuda')
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    accumulation_count = 0
    skipped_batches = 0
    nonfinite_losses = 0
    nonfinite_grads = 0
    optimizer_steps = 0
    scaler_skipped_steps = 0

    if ddp:
        pbar = tqdm(train_loader,
                    dynamic_ncols=True) if dist.get_rank() == 0 else train_loader
    else:
        pbar = tqdm(train_loader)

    for i, data in enumerate(pbar):
        if len(data)==3:
            batch, mixes, active_stem_ids = data
        elif len(data)==2:
            batch, mixes = data
            active_stem_ids = None
        else:
            raise ValueError(f'len data is {len(data)}')
        x = mixes.to(device, non_blocking=non_blocking)
        y = batch.to(device, non_blocking=non_blocking)

        if normalize:
            x, y = normalize_batch(x, y)
        next_accumulation_count = accumulation_count + 1
        should_step = next_accumulation_count >= gradient_accumulation_steps or (i == len(train_loader) - 1)
        sync_context = model.no_sync() if ddp and not should_step and hasattr(model, "no_sync") else nullcontext()
        with sync_context:
            forward_ok = True
            loss = None
            if safe_mode:
                try:
                    with autocast_context(device, amp_enabled):
                        loss = forward_step(x, y, active_stem_ids, get_internal_loss, model, multi_loss, device_ids)
                except Exception as e:
                    forward_ok = False
                    if should_print:
                        print(f'Error during training forward pass at epoch={epoch}, step={i}: {e}')
                    if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
                        torch.cuda.empty_cache()
            else:
                with autocast_context(device, amp_enabled):
                    loss = forward_step(x, y, active_stem_ids, get_internal_loss, model, multi_loss, device_ids)

            if loss is not None and loss.ndim != 0:
                loss = loss.mean()

            finite_loss = forward_ok and loss is not None and loss_is_finite(loss)
            if forward_ok and not finite_loss:
                nonfinite_losses += 1
                if should_print:
                    print(f'Non-finite loss at epoch={epoch}, step={i}; skip accumulated gradients.')

            if not synchronize_batch_success(finite_loss, device):
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                skipped_batches += 1
                continue

            loss /= gradient_accumulation_steps
            accumulation_count = next_accumulation_count
            scaler.scale(loss).backward()

        if should_step:

            scaler.unscale_(optimizer)

            grad_clip = getattr(config.training, 'grad_clip', 0)
            if grad_clip:
                grad_norm = nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                if not torch.isfinite(grad_norm):
                    nonfinite_grads += 1
                    skipped_batches += accumulation_count
                    optimizer.zero_grad(set_to_none=True)
                    if getattr(scaler, "is_enabled", lambda: False)():
                        scaler.update()
                    accumulation_count = 0
                    if should_print:
                        print(f'Non-finite gradient norm at epoch={epoch}, step={i}; optimizer step skipped.')
                    continue
            elif not gradients_are_finite(trainable_params):
                nonfinite_grads += 1
                skipped_batches += accumulation_count
                optimizer.zero_grad(set_to_none=True)
                if getattr(scaler, "is_enabled", lambda: False)():
                    scaler.update()
                accumulation_count = 0
                if should_print:
                    print(f'Non-finite gradients at epoch={epoch}, step={i}; optimizer step skipped.')
                continue

            scale_before = scaler.get_scale() if getattr(scaler, "is_enabled", lambda: False)() else None
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale() if scale_before is not None else None
            step_skipped_by_scaler = scale_before is not None and scale_after is not None and scale_after < scale_before

            if step_skipped_by_scaler:
                scaler_skipped_steps += 1
                skipped_batches += accumulation_count
                if should_print:
                    print(f'GradScaler skipped optimizer step at epoch={epoch}, step={i}.')
            else:
                optimizer_steps += 1

            if ema_model is not None and not step_skipped_by_scaler:
                if ddp:
                    ema_model.update_parameters(model.module)
                else:
                    ema_model.update_parameters(model)

            if scheduler.name in ['linear_scheduler'] and not step_skipped_by_scaler:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_count = 0
        if ddp:
            with torch.no_grad():
                reduce_device = torch.device("cpu") if dist.get_backend() == "gloo" else device
                loss_copy = loss.detach().to(reduce_device).clone()
                dist.all_reduce(loss_copy, op=dist.ReduceOp.SUM)
                loss_copy /= dist.get_world_size()
            if dist.get_rank() == 0:
                li = loss_copy.item() * gradient_accumulation_steps
                all_losses[f'epoch_{epoch}'].append(li)
                loss_val += li
                total += 1
                avg_loss = loss_val / max(total, 1)
                pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * avg_loss})
                sys.stdout.flush()
                log_wandb({'loss': 100 * li, 'avg_loss': 100 * avg_loss, 'i': i})
        else:
            li = loss.item() * gradient_accumulation_steps
            all_losses[f'epoch_{epoch}'].append(li)
            loss_val += li
            total += 1
            avg_loss = loss_val / max(total, 1)
            pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * avg_loss})
            log_wandb({'loss': 100 * li, 'avg_loss': 100 * avg_loss, 'i': i})
            loss.detach()

    if should_print:
        if total == 0:
            print(f'No training batches completed in epoch {epoch}.')
            return
        avg_loss = loss_val / total
        print(f'Training loss: {avg_loss}')
        stats = {
            'train_loss': avg_loss,
            'epoch': epoch,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'skipped_batches': skipped_batches,
            'nonfinite_losses': nonfinite_losses,
            'nonfinite_grads': nonfinite_grads,
            'optimizer_steps': optimizer_steps,
            'scaler_skipped_steps': scaler_skipped_steps,
        }
        stats.update(cuda_memory_stats(device))
        log_wandb(stats)
        append_epoch_log(args, stats)
        print(
            f'Train stats: optimizer_steps={optimizer_steps} skipped_batches={skipped_batches} '
            f'nonfinite_losses={nonfinite_losses} nonfinite_grads={nonfinite_grads} '
            f'scaler_skipped_steps={scaler_skipped_steps}'
        )


def compute_epoch_metrics(model: torch.nn.Module, args: argparse.Namespace, config: ConfigDict,
                          device: torch.device, device_ids: List[int], best_metric: float,
                          epoch: int, scheduler: torch.optim.lr_scheduler, optimizer,
                          all_time_all_metrics, all_losses, world_size=None, metrics_avg=None,
                          all_metrics=None, model_state_source: str = 'model') -> float:

    """
    Compute and log the metrics for the current epoch, and save model weights if the metric improves.

    Args:
        all_losses:
        all_metrics:
        metrics_avg:
        world_size:
        model: The model to evaluate.
        args: Command-line arguments containing configuration paths and other settings.
        config: Configuration dictionary containing training settings.
        device: The device (CPU or GPU) used for evaluation.
        device_ids: List of GPU device IDs when using multiple GPUs.
        best_metric: The best metric value seen so far.
        epoch: The current epoch number.
        scheduler: The learning rate scheduler to adjust the learning rate.
        optimizer:
        all_time_all_metrics:
    Returns:
        The updated best_metric.
    """

    ddp = True if world_size else False
    should_print = is_main_process()
    if not ddp:
        if torch.cuda.is_available() and len(device_ids) > 1:
            metrics_avg, all_metrics = valid_multi_gpu(model, args, config, args.device_ids, verbose=False)
        else:
            metrics_avg, all_metrics = valid(model, args, config, device, verbose=False)
        all_time_all_metrics[f"epoch_{epoch}"] = all_metrics

    if metrics_avg is None or args.metric_for_scheduler not in metrics_avg:
        raise KeyError(f"Metric '{args.metric_for_scheduler}' is missing from validation results: {metrics_avg}")

    metric_avg = float(metrics_avg[args.metric_for_scheduler])
    if not np.isfinite(metric_avg):
        if should_print:
            print(f"Validation metric '{args.metric_for_scheduler}' is not finite ({metric_avg}); skip checkpoint and scheduler update.")
        return best_metric

    if scheduler.name in ['ReduceLROnPlateau']:
        scheduler.step(metric_avg)

    if metric_avg > best_metric:

        if args.each_metrics_in_name:
            stem_parts = []
            for stem_name, values in all_metrics[args.metric_for_scheduler].items():
                if isinstance(values, dict):
                    values = list(values.values())
                stem_values = np.array(values, dtype=float)
                mean_val = stem_values.mean()
                std_val = stem_values.std()
                stem_parts.append(
                    f"{stem_name}_{args.metric_for_scheduler}_{mean_val:.4f}_std_{std_val:.4f}"
                )
            stem_info = "__".join(stem_parts)
            store_path = (
                f"{args.results_path}/model_{args.model_type}_ep_{epoch}_{stem_info}.ckpt"
            )
        else:
            store_path = (
                f"{args.results_path}/model_{args.model_type}_ep_{epoch}_{args.metric_for_scheduler}_{metric_avg:.4f}.ckpt"
            )
        best_metric = metric_avg
        if should_print:
            print(f'Store weights: {store_path}')
            save_eval_weights(
                store_path=store_path,
                model=model,
                device_ids=device_ids,
                epoch=epoch,
                all_time_all_metrics=all_time_all_metrics,
                all_losses=all_losses,
                best_metric=metric_avg,
                args=args,
                config=config,
                model_state_source=model_state_source,
            )

    if args.save_weights_every_epoch and should_print:
        metric_string = ''
        for m in metrics_avg:
            metric_string += '_{}_{:.4f}'.format(m, metrics_avg[m])
        store_path = f'{args.results_path}/model_{args.model_type}_ep_{epoch}{metric_string}.ckpt'
        save_eval_weights(
            store_path=store_path,
            model=model,
            device_ids=device_ids,
            epoch=epoch,
            all_time_all_metrics=all_time_all_metrics,
            all_losses=all_losses,
            best_metric=best_metric,
            args=args,
            config=config,
            model_state_source=model_state_source,
        )

    if should_print:
        log_wandb({'metric_main': metric_avg, 'best_metric': best_metric})
        append_epoch_log(args, {
            'epoch': epoch,
            'metric_main': metric_avg,
            'best_metric': best_metric,
            **{f'metric_{metric_name}': metrics_avg[metric_name] for metric_name in metrics_avg},
        })
        for metric_name in metrics_avg:
            log_wandb({f'metric_{metric_name}': metrics_avg[metric_name]})

    return best_metric


def train_model(args: Union[argparse.Namespace, None], rank=None, world_size=None) -> None:
    """
    Trains the model based on the provided arguments, including data preparation, optimizer setup,
    and loss calculation. The model is trained for multiple epochs with logging via wandb.

    Args:
        world_size:
        rank:
        args: Command-line arguments containing configuration paths, hyperparameters, and other settings.

    Returns:
        None
    """

    from utils.dataset import prepare_data
    from utils.model_utils import load_start_checkpoint
    from utils.model_utils import get_lora
    from utils.losses import choice_loss
    from utils.model_utils import get_optimizer, log_model_info

    args = parse_args_train(args)
    ddp = True if world_size else False
    if ddp and torch.cuda.is_available() and rank >= len(args.device_ids):
        raise ValueError(f"DDP rank {rank} has no matching device in --device_ids {args.device_ids}")
    if ddp:
        local_device_id = args.device_ids[rank] if torch.cuda.is_available() else None
        initialize_environment_ddp(rank, world_size, args.seed, args.results_path, device_id=local_device_id)
    else:
        initialize_environment(args.seed, args.results_path)
    model, config = get_model_from_config(args.model_type, args.config_path)
    if 'model_type' in config.training:
        args.model_type = config.training.model_type
    args = apply_config_args(args, config, mode='train')
    validate_train_setup(config, args, ddp=ddp)
    use_amp = getattr(config.training, 'use_amp', True)
    device_ids = args.device_ids
    if ddp:
        batch_size = config.training.batch_size
    else:
        batch_size = config.training.batch_size * len(device_ids)

    if not dist.is_initialized() or dist.get_rank() == 0:
        wandb_init(args, config, batch_size)

    train_loader = prepare_data(config, args, batch_size)
    if len(train_loader) == 0:
        raise RuntimeError("Training DataLoader is empty. Check data_path, dataset_type, batch_size, and metadata.")

    checkpoint = None
    if args.start_check_point:
        checkpoint = torch.load(args.start_check_point, weights_only=False, map_location='cpu')
        checkpoint_type = checkpoint.get('checkpoint_type', 'legacy') if isinstance(checkpoint, dict) else 'legacy'
        if checkpoint_type == 'eval' and (
            args.resume or args.load_optimizer or args.load_scheduler or args.load_scaler or args.load_ema
        ):
            raise ValueError(
                "Evaluation checkpoints contain model weights only. "
                "Use a training checkpoint such as last_<model_type>.ckpt for --resume or optimizer/scheduler/EMA loading."
            )
        load_start_checkpoint(args, model, checkpoint, type_='train')
    model = get_lora(args, config, model)

    if args.freeze_layers is not None:
        freeze_layers = []
        train_layers = []
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in args.freeze_layers):
                freeze_layers.append(name)
                print('Freezing layer:', name)
                param.requires_grad = False
            else:
                train_layers.append(name)
        print('Trainable layers: {}'.format(len(train_layers)))
        print('Frozen layers: {}'.format(len(freeze_layers)))

    if ddp:
        device = torch.device(f'cuda:{args.device_ids[rank]}' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        ddp_device_ids = [device.index] if device.type == 'cuda' else None
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=ddp_device_ids, find_unused_parameters=True)
        model_module = model.module
    else:
        device, model = initialize_model_and_device(model, args.device_ids)
        # If model is DataParallel, get underlying module
        model_module = model.module if hasattr(model, 'module') else model

    should_print = is_main_process()

    ema_model = None
    if hasattr(config.training, 'ema_momentum') and config.training.ema_momentum > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Initializing EMA with decay: {config.training.ema_momentum}")
        ema_model = AveragedModel(model_module, multi_avg_fn=get_ema_multi_avg_fn(config.training.ema_momentum))
        if args.start_check_point and args.load_ema and isinstance(checkpoint, dict) and checkpoint.get("ema_state_dict"):
            ema_model.load_state_dict(checkpoint["ema_state_dict"])
            if not dist.is_initialized() or dist.get_rank() == 0:
                print("Loaded EMA state from checkpoint.")
        
    if args.pre_valid:
        model_to_valid = model_for_validation(model, ema_model)
        if ddp:
            valid_multi_gpu(model_to_valid, args, config, args.device_ids, verbose=False)
        else:
            if torch.cuda.is_available() and len(args.device_ids) > 1:
                valid_multi_gpu(model_to_valid, args, config, args.device_ids, verbose=True)
            else:
                valid(model_to_valid, args, config, device, verbose=True)

    gradient_accumulation_steps = int(getattr(config.training, 'gradient_accumulation_steps', 1))

    # load optimizer
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    if args.start_check_point and "optimizer_state_dict" in checkpoint and args.load_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if args.start_check_point and "scheduler_state_dict" in checkpoint and args.load_scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # load num epoch
    if args.start_check_point and "epoch" in checkpoint and args.load_epoch:
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0

    if args.start_check_point and "best_metric" in checkpoint and args.load_best_metric:
        best_metric = checkpoint["best_metric"]
    else:
        best_metric = float('-inf')

    if args.start_check_point and "all_metrics" in checkpoint and args.load_all_metrics:
        all_time_all_metrics = checkpoint["all_metrics"]
    else:
        all_time_all_metrics = {}

    if args.start_check_point and "all_losses" in checkpoint and args.load_all_losses:
        all_losses = checkpoint["all_losses"]
    else:
        all_losses = {}

    multi_loss = choice_loss(args, config)
    scaler = create_grad_scaler(device, use_amp and torch.cuda.is_available())
    if args.start_check_point and args.load_scaler and isinstance(checkpoint, dict) and checkpoint.get("scaler_state_dict"):
        if getattr(scaler, "is_enabled", lambda: False)():
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if should_print:
                print("Loaded GradScaler state from checkpoint.")

    if args.set_per_process_memory_fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(1.0)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    safe_mode = args.safe_mode

    if should_print:
        if world_size:
            batch_size = config.training.batch_size
            ef_batch_size = batch_size * gradient_accumulation_steps * world_size
            num_gpu = world_size
        else:
            device_ids = args.device_ids
            batch_size = config.training.batch_size * len(device_ids)
            ef_batch_size = batch_size * gradient_accumulation_steps
            num_gpu = len(device_ids)

        print(
            f"Instruments: {config.training.instruments}\n"
            f"Metrics for training: {args.metrics}. Metric for scheduler: {args.metric_for_scheduler}\n"
            f"Patience: {getattr(config.training, 'patience', 'n/a')} "
            f"Reduce factor: {getattr(config.training, 'reduce_factor', 'n/a')}\n"
            f"Batch size: {batch_size} "
            f"Grad accum steps: {gradient_accumulation_steps} "
            f"Num gpus: {num_gpu} "
            f"Effective batch size: {ef_batch_size}\n"
            f"Dataset type: {args.dataset_type}\n"
            f"Optimizer: {config.training.optimizer}"
        )

        print(f'Train for: {config.training.num_epochs} epochs')
        log_model_info(model, args.results_path)

    for epoch in range(start_epoch, config.training.num_epochs):
        if ddp:
            train_loader.sampler.set_epoch(epoch)

        train_one_epoch(model, config, args, optimizer, device, device_ids, epoch,
                        use_amp, scaler, scheduler, gradient_accumulation_steps, train_loader, multi_loss, all_losses,
                        world_size, ema_model=ema_model, safe_mode=safe_mode)

        model_to_valid = model_for_validation(model, ema_model)
        validation_model_source = 'ema' if ema_model is not None else 'model'

        if should_print:
            save_last_weights(
                args, model, device_ids, optimizer, epoch, all_time_all_metrics, all_losses,
                best_metric, scheduler, config=config, ema_model=ema_model, scaler=scaler
            )
        if ddp:
            metrics_avg, all_metrics = valid_multi_gpu(model_to_valid, args, config, args.device_ids, verbose=False)
            metric_for_scheduler = None
            if rank == 0 and metrics_avg is not None and args.metric_for_scheduler in metrics_avg:
                metric_for_scheduler = float(metrics_avg[args.metric_for_scheduler])
            metric_for_scheduler = broadcast_float_from_main(metric_for_scheduler, device)
            if rank == 0:
                all_time_all_metrics[f"epoch_{epoch}"] = all_metrics
                best_metric = compute_epoch_metrics(
                    model=model_to_valid,
                    args=args,
                    config=config,
                    device=device,
                    device_ids=device_ids,
                    best_metric=best_metric,
                    epoch=epoch,
                    scheduler=scheduler,
                    optimizer=optimizer,
                    all_time_all_metrics=all_time_all_metrics,
                    all_losses=all_losses,
                    world_size=world_size,
                    metrics_avg=metrics_avg,
                    all_metrics=all_metrics,
                    model_state_source=validation_model_source,
                )
                save_last_weights(
                    args, model, device_ids, optimizer, epoch, all_time_all_metrics, all_losses,
                    best_metric, scheduler, config=config, ema_model=ema_model, scaler=scaler
                )
            elif scheduler.name in ['ReduceLROnPlateau'] and np.isfinite(metric_for_scheduler):
                scheduler.step(metric_for_scheduler)
        else:
            best_metric = compute_epoch_metrics(
                model=model_to_valid,
                args=args,
                config=config,
                device=device,
                device_ids=device_ids,
                best_metric=best_metric,
                epoch=epoch,
                scheduler=scheduler,
                optimizer=optimizer,
                all_time_all_metrics=all_time_all_metrics,
                all_losses=all_losses,
                model_state_source=validation_model_source,
            )
            if should_print:
                save_last_weights(
                    args, model, device_ids, optimizer, epoch, all_time_all_metrics, all_losses,
                    best_metric, scheduler, config=config, ema_model=ema_model, scaler=scaler
                )


if __name__ == "__main__":
    train_model(None)
