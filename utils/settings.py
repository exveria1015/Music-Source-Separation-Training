import os
import random
import time
import yaml
import wandb
import numpy as np
import torch
import argparse
import socket
import sys
from typing import Any, Dict, List, Tuple, Union
from omegaconf import ListConfig, OmegaConf
from ml_collections import ConfigDict
import torch.distributed as dist
from torch import nn
import soundfile as sf


INTERNAL_LOSS_MODEL_TYPES = {
    'bs_conformer',
    'bs_mamba2',
    'mel_band_conformer',
}

METRIC_ALIASES = {
    'neg_log_wmse': 'log_wmse',
}


def parse_with_overrides(
    parser: argparse.ArgumentParser,
    overrides: Union[argparse.Namespace, Dict, None],
) -> argparse.Namespace:
    """Parse CLI arguments and apply programmatic overrides when provided."""
    if overrides is None:
        args = parser.parse_args()
        provided = provided_arg_dests(parser, sys.argv[1:])
        args._provided_args = sorted(provided)
        return args

    args = parser.parse_args([])
    if isinstance(overrides, argparse.Namespace):
        override_dict = vars(overrides)
        provided = set(getattr(overrides, "_provided_args", override_dict.keys()))
    else:
        override_dict = dict(overrides)
        provided = set(override_dict.keys())
    args_dict = vars(args)
    args_dict.update(override_dict)
    parsed = argparse.Namespace(**args_dict)
    parsed._provided_args = sorted(provided)
    return parsed


def provided_arg_dests(parser: argparse.ArgumentParser, argv: List[str]) -> set:
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    provided = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            provided.add(dest)
    return provided


def normalize_device_ids(device_ids) -> List[int]:
    """Return device ids as a non-empty list of integers."""
    if device_ids is None:
        return [0]
    if isinstance(device_ids, int):
        return [device_ids]
    if isinstance(device_ids, str):
        parts = device_ids.replace(',', ' ').split()
        if not parts:
            raise ValueError("device_ids must not be empty")
        return [int(part) for part in parts]

    normalized = []
    for item in device_ids:
        if isinstance(item, str):
            normalized.extend(int(part) for part in item.replace(',', ' ').split())
        else:
            normalized.append(int(item))

    if not normalized:
        raise ValueError("device_ids must not be empty")
    return normalized


def uses_internal_model_loss(model_type: str, use_standard_loss: bool = False) -> bool:
    """Return whether a model computes its own training loss in forward()."""
    if use_standard_loss:
        return False
    return model_type in INTERNAL_LOSS_MODEL_TYPES or 'roformer' in model_type


def normalize_metric_name(metric_name: str) -> str:
    return METRIC_ALIASES.get(metric_name, metric_name)


def normalize_metric_names(metric_names: List[str]) -> List[str]:
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    normalized = []
    for metric_name in metric_names:
        metric_name = normalize_metric_name(metric_name)
        if metric_name not in normalized:
            normalized.append(metric_name)
    return normalized


def apply_legacy_lora_args(args: argparse.Namespace) -> argparse.Namespace:
    """Map legacy LoRA CLI names to the current explicit backend options."""
    if getattr(args, 'train_lora', False):
        args.train_lora_loralib = True
    legacy_checkpoint = getattr(args, 'lora_checkpoint', '')
    if legacy_checkpoint and not getattr(args, 'lora_checkpoint_loralib', ''):
        args.lora_checkpoint_loralib = legacy_checkpoint
    return args


def apply_resume_args(args: argparse.Namespace) -> argparse.Namespace:
    """Enable all training-state load flags for an explicit resume."""
    if getattr(args, 'resume', False):
        args.load_optimizer = True
        args.load_scheduler = True
        args.load_scaler = True
        args.load_ema = True
        args.load_epoch = True
        args.load_best_metric = True
        args.load_all_metrics = True
        args.load_all_losses = True
    return args


def arg_was_provided(args: argparse.Namespace, name: str) -> bool:
    return name in set(getattr(args, "_provided_args", []))


def config_arg_value(value: Any) -> Any:
    if isinstance(value, ListConfig):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def set_arg_from_config_if_unset(
    args: argparse.Namespace,
    arg_name: str,
    config: Union[ConfigDict, Dict],
    path: str,
) -> None:
    if not hasattr(args, arg_name) or arg_was_provided(args, arg_name):
        return
    value = config_value(config, path, None)
    if value is not None:
        setattr(args, arg_name, config_arg_value(value))


def apply_config_args(args: argparse.Namespace, config: Union[ConfigDict, Dict], mode: str) -> argparse.Namespace:
    """Use config.training defaults for CLI args that were not explicitly supplied."""
    common_mappings = [
        ("valid_path", "training.valid_path"),
        ("num_workers", "training.num_workers"),
        ("pin_memory", "training.pin_memory"),
        ("device_ids", "training.device_ids"),
    ]
    train_mappings = [
        ("data_path", "training.data_path"),
        ("dataset_type", "training.dataset_type"),
        ("persistent_workers", "training.persistent_workers"),
        ("prefetch_factor", "training.prefetch_factor"),
    ]

    for arg_name, path in common_mappings:
        set_arg_from_config_if_unset(args, arg_name, config, path)
    if mode == "train":
        for arg_name, path in train_mappings:
            set_arg_from_config_if_unset(args, arg_name, config, path)

    if hasattr(args, "metrics") and not arg_was_provided(args, "metrics"):
        metrics = config_value(config, "training.valid_metrics", None)
        if metrics is None:
            metrics = config_value(config, "training.metrics", None)
        if metrics is not None:
            args.metrics = config_arg_value(metrics)

    if hasattr(args, "metric_for_scheduler") and not arg_was_provided(args, "metric_for_scheduler"):
        metric_for_scheduler = config_value(config, "training.metric_for_scheduler", None)
        if metric_for_scheduler is not None:
            args.metric_for_scheduler = metric_for_scheduler

    if hasattr(args, "device_ids"):
        args.device_ids = normalize_device_ids(args.device_ids)
    if hasattr(args, "metrics"):
        args.metrics = normalize_metric_names(args.metrics)
    if hasattr(args, "metric_for_scheduler"):
        args.metric_for_scheduler = normalize_metric_name(args.metric_for_scheduler)
        if args.metric_for_scheduler not in args.metrics:
            args.metrics += [args.metric_for_scheduler]
    if mode == "train" and hasattr(args, "use_standard_loss") and uses_internal_model_loss(
        args.model_type,
        args.use_standard_loss,
    ):
        args.loss = [f'{args.model_type}_loss']
    return args


def config_value(config: Union[ConfigDict, Dict], path: str, default: Any = None) -> Any:
    current = config
    for part in path.split('.'):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        else:
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
    return current


def set_config_default(config: Union[ConfigDict, Dict], path: str, value: Any) -> None:
    parts = path.split('.')
    current = config
    for part in parts[:-1]:
        current = current[part] if isinstance(current, dict) else getattr(current, part)
    final_key = parts[-1]
    if isinstance(current, dict):
        current.setdefault(final_key, value)
    elif not hasattr(current, final_key):
        setattr(current, final_key, value)


def schema_scheduler_name(config: Union[ConfigDict, Dict]) -> str:
    return config_value(config, 'training.scheduler', 'ReduceLROnPlateau')


def schema_dataset_type(config: Union[ConfigDict, Dict], context: Dict[str, Any]) -> int:
    return int(context.get('dataset_type', 1))


CONFIG_SCHEMAS = {
    'common': [
        {'path': 'training.instruments', 'types': (list, tuple, ListConfig), 'non_empty': True},
        {'path': 'training.target_instrument', 'default': None, 'nullable': True},
        {'path': 'inference.num_overlap', 'types': int, 'min': 1},
        {'path': 'inference.batch_size', 'types': int, 'min': 1},
    ],
    'train': [
        {'path': 'training.batch_size', 'types': int, 'min': 1},
        {'path': 'training.num_epochs', 'types': int, 'min': 1},
        {'path': 'training.num_steps', 'types': int, 'min': 1,
         'when': lambda config, context: schema_dataset_type(config, context) != 5},
        {'path': 'training.lr', 'types': (int, float), 'min_exclusive': 0},
        {'path': 'training.optimizer', 'types': str,
         'choices': ['adam', 'adamw', 'radam', 'rmsprop', 'prodigy', 'adamw8bit', 'muon', 'adago']},
        {'path': 'training.patience', 'types': int, 'min': 0,
         'when': lambda config, context: schema_scheduler_name(config) == 'ReduceLROnPlateau'},
        {'path': 'training.reduce_factor', 'types': (int, float), 'min_exclusive': 0, 'max_exclusive': 1,
         'when': lambda config, context: schema_scheduler_name(config) == 'ReduceLROnPlateau'},
        {'path': 'training.num_warmup_steps', 'types': int, 'min': 0,
         'when': lambda config, context: schema_scheduler_name(config) == 'linear_scheduler'},
        {'path': 'training.ema_momentum', 'types': (int, float), 'min': 0, 'max_exclusive': 1,
         'optional': True},
        {'path': 'training.channels', 'types': int, 'min': 1, 'optional': True},
        {'path': 'training.file_types', 'types': (list, tuple, ListConfig), 'non_empty': True, 'optional': True},
        {'path': 'training.max_load_attempts', 'types': int, 'min': 1, 'optional': True},
        {'path': 'training.strict_sample_rate', 'types': bool, 'optional': True},
        {'path': 'training.class_balanced_stems', 'types': bool, 'optional': True},
        {'path': 'training.max_class_presence_ratio', 'types': (int, float), 'min_exclusive': 0, 'max': 1,
         'optional': True},
        {'path': 'training.read_metadata_procs', 'types': int, 'min': 1, 'optional': True},
        {'path': 'training.precompute_workers', 'types': int, 'min': 1, 'optional': True},
        {'path': 'training.max_precompute_batches', 'types': int, 'min': 1, 'optional': True},
        {'path': 'training.precompute_batch_for_chunks', 'types': int, 'min': 1, 'optional': True},
        {'path': 'training.num_precompute_chunks', 'types': int, 'min': 1, 'optional': True},
        {'path': 'audio.chunk_size', 'types': int, 'min': 1},
        {'path': 'audio.channels', 'types': int, 'min': 1, 'optional': True},
        {'path': 'audio.num_channels', 'types': int, 'min': 1, 'optional': True},
        {'path': 'audio.min_mean_abs', 'types': (int, float), 'min': 0, 'default': 0.0},
        {'path': 'augmentations.keep_original_mixture', 'types': bool, 'optional': True},
    ],
    'valid': [],
}


def validate_config_schema(config: Union[ConfigDict, Dict], schema_name: str, context: Dict[str, Any] = None) -> None:
    context = context or {}
    schema = CONFIG_SCHEMAS['common'] + CONFIG_SCHEMAS[schema_name]
    errors = []

    for field in schema:
        should_validate = field.get('when', lambda cfg, ctx: True)(config, context)
        if not should_validate:
            continue

        path = field['path']
        has_default = 'default' in field
        value = config_value(config, path, None)
        if value is None:
            if has_default:
                set_config_default(config, path, field['default'])
                value = field['default']
            elif field.get('optional', False) or field.get('nullable', False):
                continue
            else:
                errors.append(f"config.{path} is required")
                continue

        if value is None and field.get('nullable', False):
            continue

        expected_types = field.get('types')
        if expected_types is not None:
            if not isinstance(expected_types, tuple):
                expected_types = (expected_types,)
            if int in expected_types and isinstance(value, bool):
                errors.append(f"config.{path} must be {expected_types}, got bool")
                continue
            if not isinstance(value, expected_types):
                errors.append(f"config.{path} must be {expected_types}, got {type(value).__name__}")
                continue

        if field.get('non_empty') and len(value) == 0:
            errors.append(f"config.{path} must not be empty")

        if 'choices' in field and value not in field['choices']:
            errors.append(f"config.{path} must be one of {field['choices']}, got {value!r}")

        if 'min' in field and value < field['min']:
            errors.append(f"config.{path} must be >= {field['min']}, got {value!r}")
        if 'min_exclusive' in field and value <= field['min_exclusive']:
            errors.append(f"config.{path} must be > {field['min_exclusive']}, got {value!r}")
        if 'max' in field and value > field['max']:
            errors.append(f"config.{path} must be <= {field['max']}, got {value!r}")
        if 'max_exclusive' in field and value >= field['max_exclusive']:
            errors.append(f"config.{path} must be < {field['max_exclusive']}, got {value!r}")

    instruments = config_value(config, 'training.instruments')
    target_instrument = config_value(config, 'training.target_instrument')
    if instruments is not None and target_instrument is not None and target_instrument not in instruments:
        errors.append(f"config.training.target_instrument={target_instrument!r} is not in instruments {instruments}")

    scheduler_name = schema_scheduler_name(config)
    if scheduler_name not in ['linear_scheduler', 'ReduceLROnPlateau']:
        errors.append("config.training.scheduler must be one of ['linear_scheduler', 'ReduceLROnPlateau']")

    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(f"Invalid {schema_name} config:\n  - {joined}")


def validate_paths(paths, label: str) -> None:
    if paths is None:
        raise ValueError(f"{label} is required")
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    paths = list(paths)
    if not paths:
        raise ValueError(f"{label} must not be empty")
    missing = [str(path) for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"{label} contains missing paths: {missing}")


def require_positive_number(config: Union[ConfigDict, Dict], path: str, *, integer: bool = False) -> None:
    value = config_value(config, path)
    if value is None:
        raise ValueError(f"config.{path} is required")
    if integer and int(value) != value:
        raise ValueError(f"config.{path} must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"config.{path} must be > 0, got {value!r}")


def validate_common_config(config: Union[ConfigDict, Dict]) -> None:
    validate_config_schema(config, 'valid')


def validate_train_setup(config: Union[ConfigDict, Dict], args: argparse.Namespace, ddp: bool = False) -> None:
    validate_config_schema(config, 'train', context={'dataset_type': args.dataset_type})
    if not args.results_path:
        raise ValueError("--results_path is required")
    validate_paths(args.data_path, "--data_path")
    validate_paths(args.valid_path, "--valid_path")
    if args.dataset_type not in range(1, 8):
        raise ValueError(f"--dataset_type must be in 1..7, got {args.dataset_type}")
    if torch.cuda.is_available():
        available_devices = torch.cuda.device_count()
        invalid_device_ids = [device_id for device_id in args.device_ids if device_id < 0 or device_id >= available_devices]
        if invalid_device_ids:
            raise ValueError(f"Invalid CUDA device ids {invalid_device_ids}; available device count is {available_devices}.")
    if ddp and len(args.device_ids) < 1:
        raise ValueError("DDP training requires at least one device id")


def validate_valid_setup(config: Union[ConfigDict, Dict], args: argparse.Namespace) -> None:
    validate_config_schema(config, 'valid')
    validate_paths(args.valid_path, "--valid_path")
    if torch.cuda.is_available():
        available_devices = torch.cuda.device_count()
        invalid_device_ids = [device_id for device_id in args.device_ids if device_id < 0 or device_id >= available_devices]
        if invalid_device_ids:
            raise ValueError(f"Invalid CUDA device ids {invalid_device_ids}; available device count is {available_devices}.")


def parse_args_train(dict_args: Union[argparse.Namespace, Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for training configuration.

    This function constructs an argument parser for model, dataset, training, and logging
    options, merges overrides from a provided dictionary (if any), and returns the parsed
    arguments. If `dict_args` is None, the arguments are parsed from `sys.argv`.

    Args:
        dict_args (Dict | None): Optional dictionary of argument overrides. Keys should
            match the defined CLI options.

    Returns:
        argparse.Namespace: Parsed arguments namespace containing all configuration
        values required for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to start training")
    parser.add_argument("--resume", action='store_true',
                        help="Resume full training state from --start_check_point")
    parser.add_argument("--load_optimizer", action='store_true',
                        help="Load optimizer state from checkpoint (if available)")
    parser.add_argument("--load_scheduler", action='store_true',
                        help="Load scheduler state from checkpoint (if available)")
    parser.add_argument("--load_scaler", action='store_true',
                        help="Load AMP GradScaler state from checkpoint (if available)")
    parser.add_argument("--load_ema", action='store_true',
                        help="Load EMA state from checkpoint (if available)")
    parser.add_argument("--load_epoch", action='store_true', help="Load epoch number from checkpoint (if available)")
    parser.add_argument("--load_best_metric", action='store_true',
                        help="Load best metric from checkpoint (if available)")
    parser.add_argument("--load_all_metrics", action='store_true',
                        help="Load all metrics from checkpoint (if available)")
    parser.add_argument("--load_all_losses", action='store_true',
                        help="Load all losses from checkpoint (if available)")
    parser.add_argument("--safe_mode", action='store_true',
                        help="Ignore forward errors")
    parser.add_argument("--results_path", type=str,
                        help="path to folder where results will be stored (weights, metadata)")
    parser.add_argument("--data_path", nargs="+", type=str, help="Dataset data paths. You can provide several folders.")
    parser.add_argument("--dataset_type", type=int, default=1,
                        help="Dataset type. Must be one of: 1, 2, 3, 4, 5, 6, 7. Details here: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md")
    parser.add_argument("--valid_path", nargs="+", type=str,
                        help="validation data paths. You can provide several folders.")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--pin_memory", action='store_true', help="dataloader pin_memory")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0], help='list of gpu ids')
    parser.add_argument("--loss", type=str, nargs='+', choices=['masked_loss', 'mse_loss', 'l1_loss',
                                                                'multistft_loss', 'spec_masked_loss', 'spec_rmse_loss',
                                                                'log_wmse_loss', 'l1_snr_loss', 'l1_snr_db_loss',
                                                                'stft_l1_snr_db_loss', 'multi_l1_snr_db_loss'],
                        default=['masked_loss'], help="List of loss functions to use")
    parser.add_argument("--masked_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--mse_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--l1_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--log_wmse_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--multistft_loss_coef", type=float, default=0.001, help="Coef for loss")
    parser.add_argument("--spec_masked_loss_coef", type=float, default=1, help="Coef for loss")
    parser.add_argument("--spec_rmse_loss_coef", type=float, default=1, help="Coef for loss")
    parser.add_argument("--l1_snr_loss_coef", type=float, default=1., help="Coef for L1-SNR loss")
    parser.add_argument("--l1_snr_db_loss_coef", type=float, default=1., help="Coef for L1-SNR-DB loss")
    parser.add_argument("--stft_l1_snr_db_loss_coef", type=float, default=1., help="Coef for STFT-L1-SNR-DB loss")
    parser.add_argument("--multi_l1_snr_db_loss_coef", type=float, default=1., help="Coef for Multi-L1-SNR-DB loss")
    parser.add_argument("--wandb_key", type=str, default='', help='wandb API Key')
    parser.add_argument("--wandb_offline", action='store_true', help='local wandb')
    parser.add_argument("--pre_valid", action='store_true', help='Run validation before training')
    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"],
                        choices=['k_sdr', 'sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'neg_log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness', 'l1_snr'], help='List of metrics to use.')
    parser.add_argument("--metric_for_scheduler", default="sdr",
                        choices=['k_sdr', 'sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'neg_log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness', 'l1_snr'], help='Metric which will be used for scheduler.')
    parser.add_argument("--train_lora_peft", action='store_true', help="Training with LoRA from peft")
    parser.add_argument("--train_lora_loralib", action='store_true', help="Training with LoRA from loralib")
    parser.add_argument("--train_lora", action='store_true', help=argparse.SUPPRESS)
    parser.add_argument("--lora_checkpoint_peft", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--lora_checkpoint_loralib", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--lora_checkpoint", type=str, default='', help=argparse.SUPPRESS)
    parser.add_argument("--each_metrics_in_name", action='store_true',
                        help="All stems in naming checkpoints")
    parser.add_argument("--use_standard_loss", action='store_true',
                        help="Roformers will use provided loss instead of internal")
    parser.add_argument("--save_weights_every_epoch", action='store_true',
                        help="Weights will be saved every epoch with all metric values")
    parser.add_argument("--persistent_workers", action='store_true',
                        help="dataloader persistent_workers")
    parser.add_argument("--prefetch_factor", type=int, default=None,
                        help="dataloader prefetch_factor")
    parser.add_argument("--set_per_process_memory_fraction", action='store_true',
                        help="using only VRAM, no RAM")
    parser.add_argument("--load_only_compatible_weights", action='store_true',
                        help="using only VRAM, no RAM")
    parser.add_argument("--freeze_layers", nargs="+", type=str,
                        help="List of layers to freeze. Use prefixes e.g. layer1 - will freeze all layers whose names "
                             "starts with layer1. You can set mulitple parameters.")

    args = parse_with_overrides(parser, dict_args)
    args.device_ids = normalize_device_ids(args.device_ids)
    args = apply_resume_args(args)
    args = apply_legacy_lora_args(args)
    args.metrics = normalize_metric_names(args.metrics)
    args.metric_for_scheduler = normalize_metric_name(args.metric_for_scheduler)

    if args.metric_for_scheduler not in args.metrics:
        args.metrics += [args.metric_for_scheduler]

    if uses_internal_model_loss(args.model_type, args.use_standard_loss):
        args.loss = [f'{args.model_type}_loss']
    return args


def parse_args_valid(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for validation configuration.

    Builds the CLI for model selection, configuration paths, validation data
    locations, output/spectrogram saving options, device/runtime settings, and
    evaluation metrics. If `dict_args` is provided, its key–value pairs override
    or set the parsed arguments; otherwise arguments are read from `sys.argv`.

    Args:
        dict_args (Union[Dict, None]): Optional mapping of argument names to values
            used to override or supply CLI options programmatically.

    Returns:
        argparse.Namespace: Parsed arguments namespace containing all validation
        configuration values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of mdx23c, htdemucs, segm_models, mel_band_roformer,"
                             " bs_roformer, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint"
                                                                          " to valid weights")
    parser.add_argument("--valid_path", nargs="+", type=str, help="Validate path")
    parser.add_argument("--store_dir", type=str, default="", help="Path to store results as wav file")
    parser.add_argument("--draw_spectro", type=float, default=0,
                        help="If --store_dir is set then code will generate spectrograms for resulted stems as well."
                             " Value defines for how many seconds os track spectrogram will be generated.")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0], help='List of gpu ids')
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader num_workers")
    parser.add_argument("--pin_memory", action='store_true', help="Dataloader pin_memory")
    parser.add_argument("--extension", type=str, default='wav', help="Choose extension for validation")
    parser.add_argument("--use_tta", action='store_true',
                        help="Flag adds test time augmentation during inference (polarity and channel inverse)."
                             "While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"],
                        choices=['k_sdr', 'sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'neg_log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness', 'l1_snr'], help='List of metrics to use.')
    parser.add_argument("--lora_checkpoint_peft", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--lora_checkpoint_loralib", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--lora_checkpoint", type=str, default='', help=argparse.SUPPRESS)

    args = parse_with_overrides(parser, dict_args)
    args.device_ids = normalize_device_ids(args.device_ids)
    args = apply_legacy_lora_args(args)
    args.metrics = normalize_metric_names(args.metrics)

    return args


def parse_args_inference(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for inference configuration.

    Builds the CLI for model selection, configuration path, input/output handling,
    device/runtime options, test-time augmentation, and optional LoRA checkpoints.
    If `dict_args` is provided, its key–value pairs override or supply CLI options
    programmatically; otherwise, arguments are read from `sys.argv`.

    Args:
        dict_args (Union[Dict, None]): Optional mapping of argument names to values
            used to override or supply CLI options programmatically.

    Returns:
        argparse.Namespace: Parsed arguments namespace containing all inference
        configuration values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer,"
                             " scnet, scnet_unofficial, segm_models, swin_upernet, torchseg")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to valid weights")
    parser.add_argument("--input_folder", type=str, help="folder with mixtures to process")
    parser.add_argument("--store_dir", type=str, default="", help="path to store results as wav file")
    parser.add_argument("--draw_spectro", type=float, default=0,
                        help="Code will generate spectrograms for resulted stems."
                             " Value defines for how many seconds os track spectrogram will be generated.")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0], help='list of gpu ids')
    parser.add_argument("--extract_instrumental", action='store_true',
                        help="invert vocals to get instrumental if provided")
    parser.add_argument("--disable_detailed_pbar", action='store_true', help="disable detailed progress bar")
    parser.add_argument("--force_cpu", action='store_true', help="Force the use of CPU even if CUDA is available")
    parser.add_argument("--flac_file", action='store_true', help="Output flac file instead of wav")
    parser.add_argument("--pcm_type", type=str, choices=['PCM_16', 'PCM_24', 'FLOAT'], default='FLOAT',
                        help="PCM type for FLAC files (PCM_16 or PCM_24)")
    parser.add_argument("--use_tta", action='store_true',
                        help="Flag adds test time augmentation during inference (polarity and channel inverse)."
                        "While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    parser.add_argument("--bigshifts", type=int, default=1,
                        help="Number of circular time shifts to average during demix. Values <= 0 are treated as 1.")
    parser.add_argument("--lora_checkpoint_peft", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--lora_checkpoint", type=str, default='', help=argparse.SUPPRESS)
    parser.add_argument("--filename_template", type=str, default='{file_name}/{instr}',
                        help="Output filename template, without extension, using '/' for subdirectories. Default: '{file_name}/{instr}'")
    parser.add_argument("--lora_checkpoint_loralib", type=str, default='', help="Initial checkpoint to LoRA weights")
    args = parse_with_overrides(parser, dict_args)
    args.device_ids = normalize_device_ids(args.device_ids)
    args = apply_legacy_lora_args(args)
    args.pcm_type = validate_sndfile_subtype(args)

    return args


def validate_sndfile_subtype(args):
    codec = 'flac' if getattr(args, 'flac_file', False) else 'wav'
    subtype = args.pcm_type
    if subtype in sf.available_subtypes(codec):
        return subtype
    default = sf.default_subtype(codec)
    print(f"WARNING: codec {codec} doesn't support subtype {subtype}, defaulting to {default}")
    return default


def load_config(model_type: str, config_path: str) -> Union[ConfigDict, OmegaConf]:
    """
    Load a model configuration from a file.

    Based on `model_type`, returns either an OmegaConf (e.g., for 'htdemucs')
    or a YAML-parsed ConfigDict for other models.

    Args:
        model_type (str): Model identifier that determines the loader behavior
            (e.g., 'htdemucs', 'mdx23c', etc.).
        config_path (str): Path to the configuration file (YAML/OmegaConf).

    Returns:
        Union[ConfigDict, OmegaConf]: Loaded configuration object.

    Raises:
        FileNotFoundError: If `config_path` does not point to an existing file.
        ValueError: If the configuration cannot be parsed or is otherwise invalid.
    """
    try:
        with open(config_path, 'r') as f:
            if model_type == 'htdemucs':
                config = OmegaConf.load(config_path)
            else:
                config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def get_model_from_config(model_type: str, config_path: str) -> Tuple[nn.Module, Union[ConfigDict, OmegaConf]]:
    """
    Load and instantiate a model using a configuration file.

    Given a `model_type` and a path to a configuration, this function loads the
    configuration (YAML or OmegaConf) and constructs the corresponding model.

    Args:
        model_type (str): Identifier of the model family (e.g., 'mdx23c', 'htdemucs',
            'scnet', 'mel_band_conformer', etc.).
        config_path (str): Filesystem path to the configuration file used to
            initialize the model.

    Returns:
        Tuple[nn.Module, Union[ConfigDict, OmegaConf]]: A tuple containing the
        initialized PyTorch model and the loaded configuration object.

    Raises:
        ValueError: If `model_type` is unknown or model initialization fails.
        FileNotFoundError: If `config_path` does not exist (may be raised by the
            underlying config loader).
    """

    config = load_config(model_type, config_path)
    if 'model_type' in config.training:
        model_type = config.training.model_type
    if model_type == 'mdx23c':
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'htdemucs':
        from models.demucs4ht import get_model
        model = get_model(config)
    elif model_type == 'segm_models':
        from models.segm_models import Segm_Models_Net
        model = Segm_Models_Net(config)
    elif model_type == 'torchseg':
        from models.torchseg_models import Torchseg_Net
        model = Torchseg_Net(config)
    elif model_type == 'mel_band_roformer':
        from models.bs_roformer import MelBandRoformer
        model = MelBandRoformer(**dict(config.model))
    elif model_type == 'mel_band_conformer':
        from models.bs_roformer import MelBandConformer
        model = MelBandConformer(**dict(config.model))
    elif model_type == 'mel_band_roformer_experimental':
        from models.bs_roformer.mel_band_roformer_experimental import MelBandRoformer
        model = MelBandRoformer(**dict(config.model))
    elif model_type == 'bs_roformer':
        from models.bs_roformer import BSRoformer
        model = BSRoformer(**dict(config.model))
    elif model_type == 'bs_conformer':
        from models.bs_roformer import BSConformer
        model = BSConformer(**dict(config.model))
    elif model_type == 'bs_roformer_experimental':
        from models.bs_roformer.bs_roformer_experimental import BSRoformer
        model = BSRoformer(**dict(config.model))
    elif model_type == 'bs_mamba2':
        from models.bs_mamba2_code.bs_mamba2 import BSMamba2Model
        model = BSMamba2Model(**dict(config.model))
    elif model_type == 'swin_upernet':
        from models.upernet_swin_transformers import Swin_UperNet_Model
        model = Swin_UperNet_Model(config)
    elif model_type == 'bandit':
        from models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        model = MultiMaskMultiSourceBandSplitRNNSimple(**config.model)
    elif model_type == 'bandit_v2':
        from models.bandit_v2.bandit import Bandit
        model = Bandit(**config.kwargs)
    elif model_type == 'scnet_unofficial':
        from models.scnet_unofficial import SCNet
        model = SCNet(**config.model)
    elif model_type == 'scnet':
        from models.scnet import SCNet
        model = SCNet(**config.model)
    elif model_type == 'scnet_tran':
        from models.scnet.scnet_tran import SCNet_Tran
        model = SCNet_Tran(**config.model)
    elif model_type == 'apollo':
        from models.look2hear.models import BaseModel
        model = BaseModel.apollo(**config.model)
    elif model_type == 'experimental_mdx23c_stht':
        from models.mdx23c_tfc_tdf_v3_with_STHT import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'scnet_masked':
        from models.scnet.scnet_masked import SCNet
        model = SCNet(**config.model)
    elif model_type == 'conformer':
        from models.conformer_model import ConformerMSS, NeuralModel
        model = ConformerMSS(
            core=NeuralModel(**config.model),
            n_fft=config.stft.n_fft,
            hop_length=config.stft.hop_length,
            win_length=getattr(config.stft, 'win_length', config.stft.n_fft),
            center=config.stft.center
        )
    elif model_type == 'moises_light':
        from moises_light import MoisesLight
        model = MoisesLight(**dict(config.model))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, config


def get_scheduler(config, optimizer):
    scheduler_name = config.training.get('scheduler', 'ReduceLROnPlateau')
    if scheduler_name == 'linear_scheduler':
        from transformers import get_linear_schedule_with_warmup
        num_training_steps = config.training.num_epochs * config.training.num_steps
        num_warmup_steps = config.training.num_warmup_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_name == 'ReduceLROnPlateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=config.training.patience,
                                      factor=config.training.reduce_factor)
    else:
        available_schedulers = ['linear_scheduler', 'ReduceLROnPlateau']
        raise ValueError(
            f"Unknown scheduler '{scheduler_name}'. "
            f"Available options: {available_schedulers}. "
            f"Check your config.training.scheduler setting."
        )
    scheduler.name = scheduler_name
    return scheduler


def logging(logs: List[str], text: str, verbose_logging: bool = False) -> Union[List[str], None]:
    """
    Print a log message and optionally append it to an in-memory list.

    In Distributed Data Parallel (DDP) contexts, the message is printed only on
    rank 0; when DDP is uninitialized, it prints unconditionally. If
    `verbose_logging` is True, the message is also appended to `logs`.

    Args:
        logs (List[str]): Mutable list to which the message is appended when
            `verbose_logging` is True.
        text (str): The log message to print (rank 0 only under DDP) and
            optionally store.
        verbose_logging (bool, optional): If True, append `text` to `logs`.
            Defaults to False.

    Returns:
        List[str]: The function prints and may mutate `logs` in place.
    """
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(text)
        if verbose_logging:
            logs.append(text)
    return logs

def write_results_in_file(store_dir: str, logs: List[str]) -> None:
    """
    Write accumulated log messages to a results file.

    Creates (or overwrites) a `results.txt` file inside `store_dir` and writes
    each entry from `logs` as a separate line. In Distributed Data Parallel (DDP)
    scenarios, writing is intended to occur only on rank 0.

    Args:
        store_dir (str): Directory path where `results.txt` will be saved.
        logs (List[str]): Ordered collection of log lines to write.

    Returns:
        None
    """
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(store_dir, exist_ok=True)
        with open(f'{store_dir}/results.txt', 'w') as out:
            for item in logs:
                out.write(item + "\n")


def manual_seed(seed: int) -> None:
    """
    Initialize random seeds for reproducibility.

    Sets the seed across Python's `random`, NumPy, and PyTorch (CPU and CUDA)
    libraries, and updates the `PYTHONHASHSEED` environment variable. This helps
    ensure deterministic behavior where possible, though some GPU operations
    may still introduce nondeterminism.

    Args:
        seed (int): The seed value to use for all random number generators.

    Returns:
        None
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU
    torch.backends.cudnn.deterministic = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def initialize_environment(seed: int, results_path: str) -> None:
    """
    Initialize runtime environment settings.

    Sets random seeds for reproducibility, adjusts PyTorch cuDNN behavior,
    configures multiprocessing with the 'spawn' start method, and ensures
    the results directory exists.

    Args:
        seed (int): Random seed value for deterministic initialization.
        results_path (str): Filesystem path to create for saving results.

    Returns:
        None
    """

    if not results_path:
        raise ValueError("--results_path is required")

    manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    try:
        torch.multiprocessing.set_start_method('spawn')
    except Exception as e:
        pass
    os.makedirs(results_path, exist_ok=True)


def initialize_environment_ddp(
    rank: int,
    world_size: int,
    seed: int = 0,
    resuls_path: str = None,
    device_id: int = None,
    master_port: int = None,
) -> None:
    """
    Initialize environment for Distributed Data Parallel (DDP) training/validation.

    Sets up the DDP process group, seeds random number generators, configures
    multiprocessing to use the 'spawn' method, and creates a results directory
    if provided.

    Args:
        rank (int): Rank of the current process within the DDP group.
        world_size (int): Total number of processes participating in DDP.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        resuls_path (str, optional): Directory path to create for storing results.
            If None, no directory is created. Defaults to None.
        device_id (int, optional): CUDA device id assigned to this rank.
        master_port (int, optional): Shared DDP rendezvous port. If omitted,
            uses MASTER_PORT from the environment or a deterministic seed-based port.

    Returns:
        None
    """
    setup_ddp(rank, world_size, seed, device_id=device_id, master_port=master_port)
    manual_seed(seed)

    try:
        torch.multiprocessing.set_start_method('spawn', force=True)  # force=True prevent errors
    except RuntimeError as e:
        if "context has already been set" not in str(e):
            raise e
    if not (resuls_path is None):
        os.makedirs(resuls_path, exist_ok=True)


def gen_wandb_name(args, config) -> str:
    """
    Generate a descriptive name for a Weights & Biases (wandb) run.

    Combines the model type, a dash-joined list of training instruments,
    and the current date into a single string identifier.

    Args:
        args: Parsed arguments namespace containing at least `model_type`.
        config: Configuration object/dict with a `training.instruments` field.

    Returns:
        str: Formatted run name in the form
            "<model_type>_[<instrument1>-<instrument2>-...]_<YYYY-MM-DD>".
    """

    instrum = '-'.join(config['training']['instruments'])
    time_str = time.strftime("%Y-%m-%d")
    name = '{}_[{}]_{}'.format(args.model_type, instrum, time_str)
    return name


def wandb_init(args: argparse.Namespace, config: Union[ConfigDict, OmegaConf], batch_size: int) -> None:
    """
    Initialize Weights & Biases (wandb) for experiment tracking.

    Depending on the provided arguments, sets up wandb in one of three modes:
    - Offline mode when `args.wandb_offline` is True.
    - Disabled mode when no valid `wandb_key` is provided.
    - Online mode with authentication using `args.wandb_key`.

    Args:
        args (argparse.Namespace): Parsed arguments containing wandb options
            (`wandb_offline`, `wandb_key`, `device_ids`).
        config (Dict): Experiment configuration dictionary to log.
        batch_size (int): Training batch size to include in the run configuration.

    Returns:
        None
    """

    if args.wandb_offline:
        wandb.init(mode='offline',
                   project='msst',
                   name=gen_wandb_name(args, config),
                   config={'config': config, 'args': args, 'device_ids': args.device_ids, 'batch_size': batch_size}
                   )
    elif args.wandb_key is None or args.wandb_key.strip() == '':
        wandb.init(mode='disabled')
    else:
        wandb.login(key=args.wandb_key)
        wandb.init(
            project='msst',
            name=gen_wandb_name(args, config),
            config={'config': config, 'args': args, 'device_ids': args.device_ids, 'batch_size': batch_size}
        )


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))              # 0 → OS chooses free port
        return s.getsockname()[1]


def setup_ddp(
    rank: int,
    world_size: int,
    seed: int,
    device_id: int = None,
    master_port: int = None,
) -> None:
    """
    Initialize a Distributed Data Parallel (DDP) process group.

    Configures environment variables for the DDP master node, attempts to
    initialize the process group with the NCCL backend (preferred for GPUs),
    and falls back to the Gloo backend if NCCL is unavailable. Also sets the
    current CUDA device to match the process rank.

    Args:
        rank (int): Rank of the current process in the DDP group.
        world_size (int): Total number of processes participating in DDP.
        seed: Random seed used for deterministic fallback port selection.
        device_id (int, optional): CUDA device id assigned to this rank.
        master_port (int, optional): Explicit rendezvous port shared by all ranks.
    Returns:
        None
    """

    if master_port is None:
        master_port = os.environ.get('MASTER_PORT')
    if master_port is None:
        master_port = 10000 + (int(seed) % 50000)

    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ["USE_LIBUV"] = "0"

    if device_id is None:
        device_id = rank
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    if torch.cuda.is_available():
        try:
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            return
        except Exception as e:
            if dist.is_initialized():
                dist.destroy_process_group()
            if rank == 0:
                print(f'NCCL is not available ({e}). Using "gloo" backend.')

    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        if rank == 0 and not torch.cuda.is_available():
            print('CUDA is not available. Using "gloo" backend.')


def cleanup_ddp() -> None:
    """
    Finalize and clean up a Distributed Data Parallel (DDP) process group.

    Calls `torch.distributed.destroy_process_group()` to release resources
    associated with the current DDP environment.

    Returns:
        None
    """
    if dist.is_initialized():
        dist.destroy_process_group()
