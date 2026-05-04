# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import argparse
import copy
import hashlib
import math
import time
import os
import gc
import glob
import traceback
import torch
import librosa
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm
from ml_collections import ConfigDict
from typing import Tuple, Dict, List, Union, Any, Optional
import torch.distributed as dist
from pathlib import Path
from utils.settings import get_model_from_config, logging, normalize_device_ids, write_results_in_file, parse_args_valid, validate_valid_setup, apply_config_args
from utils.audio_utils import normalize_audio, denormalize_audio, read_audio_transposed, \
    draw_2_mel_spectrogram
from utils.model_utils import demix, prefer_target_instrument, apply_tta, load_start_checkpoint
from utils.metrics import get_metrics

import warnings

warnings.filterwarnings("ignore")


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def unwrap_parallel_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def distributed_tensor_device(preferred_device: torch.device) -> torch.device:
    if dist.is_initialized() and dist.get_backend() == "gloo":
        return torch.device("cpu")
    return preferred_device


def is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, RuntimeError) and ("out of memory" in message or "cuda oom" in message)


def output_base_path(store_dir: str, folder: str, instr: str) -> str:
    folder_path = Path(folder).resolve()
    digest = hashlib.sha1(str(folder_path).encode("utf-8")).hexdigest()[:8]
    safe_name = folder_path.name.replace(os.sep, "_")
    return str(Path(store_dir) / f"{safe_name}_{digest}_{instr}")


def demix_with_oom_retry(
    config: ConfigDict,
    model: torch.nn.Module,
    mix: np.ndarray,
    device: torch.device,
    model_type: str,
) -> Dict[str, np.ndarray]:
    try:
        return demix(config, model, mix, device, model_type=model_type)
    except RuntimeError as error:
        if not is_cuda_oom(error):
            raise
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            raise

        current_batch_size = int(getattr(config.inference, "batch_size", 1))
        if current_batch_size <= 1:
            raise

        torch.cuda.empty_cache()
        retry_config = copy.deepcopy(config)
        retry_config.inference.batch_size = max(1, current_batch_size // 2)
        if is_main_process():
            print(
                f"CUDA OOM during validation demix; retry with inference.batch_size="
                f"{retry_config.inference.batch_size}"
            )
        return demix(retry_config, model, mix, device, model_type=model_type)


def get_mixture_paths(
    args: argparse.Namespace,
    verbose: bool,
    config: ConfigDict,
    extension: str
) -> List[str]:
    """
    Collect validation mixture file paths from one or more root directories.

    Scans each directory in `args.valid_path` for files matching the pattern
    `<root>/*/mixture.<extension>` and returns a sorted list of absolute paths.
    In Distributed Data Parallel (DDP) runs, status messages are printed only
    on rank 0; otherwise they are printed unconditionally when `verbose=True`.

    Args:
        args (argparse.Namespace): Arguments with `valid_path` (str or List[str])
            specifying root directories to search.
        verbose (bool): If True, print collection details and summary.
        config (ConfigDict): Configuration used for informational logging
            (e.g., `inference.num_overlap`, `inference.batch_size`).
        extension (str): Audio file extension to match (with or without a leading dot).

    Returns:
        List[str]: Sorted list of discovered mixture file paths.
    """

    should_print = is_main_process()

    # --- read & normalize args.valid_path ---
    try:
        valid_path = args.valid_path
    except Exception as e:
        if should_print:
            print("No valid path in args")
        raise e

    if isinstance(valid_path, (str, os.PathLike)):
        valid_paths: List[str] = [valid_path]
    else:
        valid_paths = list(valid_path)

    # --- collect mixture files ---
    all_mixtures_path: List[str] = []
    extension = extension.lstrip(".")

    def find_mixture_files(root_dir):
        root_path = Path(root_dir)
        wav_files = list(root_path.rglob("mixture.wav"))
        flac_files = list(root_path.rglob("mixture.flac"))
        if not(extension in ['wav', 'flac']):
            ext_file = list(root_path.rglob(f"mixture.{extension}"))
        else:
            ext_file = []
        return wav_files + flac_files + ext_file

    for root in valid_paths:
        part = find_mixture_files(root)
        if not part and verbose and should_print:
            print(f"No validation data found in: {root}")
        all_mixtures_path.extend(str(path) for path in part)
    all_mixtures_path = sorted(dict.fromkeys(all_mixtures_path))

    if not all_mixtures_path:
        raise RuntimeError(f"No validation mixtures found in: {[str(path) for path in valid_paths]}")

    # --- verbose summary ---
    if verbose and should_print:
        # be robust to dict-like or attribute-like config
        inference = getattr(config, "inference", None)
        if inference is None and isinstance(config, dict):
            inference = config.get("inference", None)

        def _get(obj, name, default=None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)

        num_overlap = _get(inference, "num_overlap", "?")
        batch_size  = _get(inference, "batch_size",  "?")

        print(f"Total mixtures: {len(all_mixtures_path)}")
        print(f"Overlap: {num_overlap} Batch size: {batch_size}")

    return all_mixtures_path


def update_metrics_and_pbar(
    track_metrics: Dict[str, float],
    all_metrics: Dict[str, Dict[str, Union[Dict[str, float], List[float]]]],
    instr: str,
    pbar_dict: Dict[str, float],
    mixture_paths: Optional[Union[List[str], tqdm]],
    verbose: bool = False,
    path: Optional[str] = None,
) -> None:
    """
    Update accumulated metrics and (optionally) a tqdm progress bar.

    In non-DDP runs, appends each metric value to `all_metrics[metric_name][instr]`
    (a list). In DDP runs (when `torch.distributed` is initialized), stores values
    as `all_metrics[metric_name][instr][path]` (a dict keyed by file `path`);
    therefore `path` must be provided under DDP. When `verbose=True`, metric
    values are printed only on rank 0. Also updates `pbar_dict` and, if a tqdm
    instance is provided, calls `set_postfix` for live display.

    Args:
        track_metrics (Dict[str, float]): Mapping from metric name to its value for
            the current track/instrument.
        all_metrics (Dict[str, Dict[str, Union[Dict[str, float], List[float]]]]):
            Aggregator for all collected metrics, organized as
            `{metric_name: {instrument: list_or_dict}}`, where the inner container
            is a list (non-DDP) or dict keyed by `path` (DDP).
        instr (str): Instrument name associated with the current metrics.
        pbar_dict (Dict[str, float]): Dictionary holding the latest values to show
            in the tqdm postfix (updated in place).
        mixture_paths (Optional[Union[List[str], tqdm]]): If a tqdm progress bar is
            supplied, its `set_postfix` is called with `pbar_dict`.
        verbose (bool, optional): If True, print metric updates (rank 0 only in DDP).
            Defaults to False.
        path (Optional[str], optional): File path key required in DDP mode to index
            per-track metrics for `instr`. Ignored in non-DDP. Defaults to None.

    Returns:
        None
    """

    ddp_mode = dist.is_initialized()
    should_print = is_main_process()

    if ddp_mode and path is None:
        raise ValueError("`path` must be provided when torch.distributed is initialized.")

    for metric_name, metric_value in track_metrics.items():
        if verbose and should_print:
            print(f"Metric {metric_name:11s} value: {metric_value:.4f}")

        if metric_name not in all_metrics:
            all_metrics[metric_name] = {}
        if instr not in all_metrics[metric_name]:
            all_metrics[metric_name][instr] = {} if ddp_mode else []

        if ddp_mode:
            all_metrics[metric_name][instr][path] = metric_value  # type: ignore[index]
        else:
            all_metrics[metric_name][instr].append(metric_value)  # type: ignore[union-attr]

        pbar_dict[f"{metric_name}_{instr}"] = metric_value

    if mixture_paths is not None and hasattr(mixture_paths, "set_postfix"):
        try:
            mixture_paths.set_postfix(pbar_dict)
        except Exception:
            pass


def process_audio_files(
    mixture_paths: List[str],
    model: torch.nn.Module,
    args: Any,
    config: ConfigDict,
    device: torch.device,
    verbose: bool = False,
    is_tqdm: bool = True
) -> Dict[str, Dict[str, Union[Dict[str, float], List[float]]]]:
    """
    Run source separation on a list of mixtures and collect evaluation metrics.

    Performs optional resampling and normalization, demixes each track (with
    optional Test-Time Augmentation), saves separated stems (FLAC PCM_16 when
    peak ≤ 1.0 else WAV FLOAT), optionally renders spectrograms, computes the
    requested metrics, and aggregates them in a nested dictionary.

    In non-DDP runs, metrics are stored as lists:
        {metric_name: {instrument: [values...]}}
    In DDP runs (when `torch.distributed` is initialized), metrics are stored as
    dicts keyed by the track path:
        {metric_name: {instrument: {path: value, ...}}}

    Args:
        mixture_paths (List[str]): Absolute or relative paths to `mixture.<ext>` files.
        model (torch.nn.Module): Trained separator model in eval mode.
        args (Any): Runtime arguments (e.g., `metrics`, `model_type`, `use_tta`,
            `store_dir`, `draw_spectro`, `extension`).
        config (ConfigDict): Configuration with audio/inference/training settings
            (e.g., `audio.sample_rate`, `inference.batch_size`, `inference.num_overlap`,
            `inference.normalize`, `training.instruments`).
        device (torch.device): Device for inference (CPU/CUDA).
        verbose (bool, optional): Print per-track details and timings. Defaults to False.
        is_tqdm (bool, optional): Show a tqdm progress bar (rank 0 only under DDP). Defaults to True.

    Returns:
        Dict[str, Dict[str, Union[Dict[str, float], List[float]]]]: Aggregated metrics
        per metric and instrument; inner container is a list (non-DDP) or a dict keyed
        by track path (DDP).
    """

    ddp_mode = dist.is_initialized()
    should_print = is_main_process()

    instruments = prefer_target_instrument(config)
    use_tta = getattr(args, 'use_tta', False)
    store_dir = getattr(args, 'store_dir', '')

    # extension is used only for reading GT stems; outputs use FLAC/WAV rule unconditionally
    if 'inference' in config and 'extension' in config['inference']:
        extension = config['inference']['extension']
    else:
        extension = getattr(args, 'extension', 'wav')

    # --- init metrics container ---
    if ddp_mode:
        # behave like first: dict of dicts
        all_metrics: Dict[str, Dict[str, Dict]] = {
            metric: {instr: {} for instr in config.training.instruments}
            for metric in args.metrics
        }
    else:
        # behave like second: dict of lists
        all_metrics: Dict[str, Dict[str, List[float]]] = {
            metric: {instr: [] for instr in config.training.instruments}
            for metric in args.metrics
        }

    # --- tqdm wrapping as requested ---
    if is_tqdm and should_print:
        mixture_paths = tqdm(mixture_paths)


    def get_instruments(path: str) -> dict[str, str]:
        """Detect available instrument files and their extensions."""
        real_instruments: dict[str, str] = {}

        for instr in instruments:
            # Check supported extensions for each instrument
            for ext in [extension, "flac", "wav"]:
                file_path = Path(path) / f"{instr}.{ext}"
                if file_path.exists():
                    real_instruments[instr] = ext
                    break

        return real_instruments


    for path in mixture_paths:
        start_time = time.time()
        mix, sr = read_audio_transposed(path)
        mix_orig = mix.copy()
        folder = os.path.dirname(path)
        real_instruments = get_instruments(folder)
        if not real_instruments and verbose and should_print:
            print(f"No reference stems found for validation track: {folder}")
        # resample input to config SR if needed
        if 'audio' in config and 'sample_rate' in config.audio:
            target_sr = config.audio['sample_rate']
            if sr != target_sr:
                orig_length = mix.shape[-1]
                if verbose and should_print:
                    print(f'Warning: sample rate is different. In config: {target_sr} in file {path}: {sr}')
                mix = librosa.resample(mix, orig_sr=sr, target_sr=target_sr, res_type='kaiser_best')

        if verbose and should_print:
            print(f'Song: {os.path.abspath(folder)} Shape: {mix.shape}')

        # optional normalize
        if 'inference' in config and config.inference.get('normalize', False):
            mix, norm_params = normalize_audio(mix)
        else:
            norm_params = None

        waveforms_orig = demix_with_oom_retry(config, model, mix.copy(), device, model_type=args.model_type)

        if use_tta:
            waveforms_orig = apply_tta(config, model, mix, waveforms_orig, device, args.model_type)

        pbar_dict = {}

        for instr, track_extension in real_instruments.items():
            if verbose and should_print:
                print(f"Instr: {instr}")

            # read GT track
            if instr != 'other' or not getattr(config.training, 'other_fix', False):
                track, sr1 = read_audio_transposed(f"{folder}/{instr}.{track_extension}", instr, skip_err=True)
                if track is None:
                    continue
            else:
                # other = mix - vocals
                vocals_extension = real_instruments.get('vocals', track_extension)
                track, sr1 = read_audio_transposed(f"{folder}/vocals.{vocals_extension}")
                track = mix_orig - track

            if isinstance(waveforms_orig, dict):
                if instr not in waveforms_orig:
                    if verbose and should_print:
                        print(f"Model did not return instrument '{instr}' for {folder}; skip metric.")
                    continue
                estimates = waveforms_orig[instr]
            else:
                instrument_order = prefer_target_instrument(config)
                if instr not in instrument_order:
                    if verbose and should_print:
                        print(f"Model output has no instrument '{instr}' for {folder}; skip metric.")
                    continue
                estimate_index = instrument_order.index(instr)
                estimates = waveforms_orig[estimate_index]

            # back-resample estimates to original SR if input was resampled
            if 'audio' in config and 'sample_rate' in config.audio:
                target_sr = config.audio['sample_rate']
                if sr != target_sr:
                    estimates = librosa.resample(estimates, orig_sr=target_sr, target_sr=sr, res_type='kaiser_best')
                    estimates = librosa.util.fix_length(estimates, size=orig_length)

            # denormalize if needed
            if norm_params is not None and 'inference' in config and config.inference.get('normalize', False):
                estimates = denormalize_audio(estimates, norm_params)

            # --- saving (uniform rule) ---
            if store_dir:
                os.makedirs(store_dir, exist_ok=True)
                base = output_base_path(store_dir, folder, instr)
                peak = float(np.abs(estimates).max())
                if peak <= 1.0:
                    out_path = f"{base}.flac"
                    sf.write(out_path, estimates.T, sr, subtype='PCM_16')
                else:
                    out_path = f"{base}.wav"
                    sf.write(out_path, estimates.T, sr, subtype='FLOAT')

                draw_spec = getattr(args, 'draw_spectro', 0)
                if draw_spec and draw_spec > 0:
                    draw_2_mel_spectrogram(estimates.T, track.T, sr, draw_spec, base)

            # --- metrics ---
            k = config.training.get("k_sdr", 10)
            track_metrics = get_metrics(
                args.metrics,
                track,
                estimates,
                mix_orig,
                device=device,
                k=k
            )

            # --- update metrics + progress ---
            if ddp_mode:
                # behave like first: include path in call
                update_metrics_and_pbar(
                    track_metrics,
                    all_metrics,
                    instr,
                    pbar_dict,
                    mixture_paths=mixture_paths,
                    verbose=verbose and should_print,
                    path=path
                )
            else:
                # behave like second: no path argument
                update_metrics_and_pbar(
                    track_metrics,
                    all_metrics,
                    instr,
                    pbar_dict,
                    mixture_paths=mixture_paths,
                    verbose=verbose and should_print
                )

        if verbose and should_print:
            print(f"Time for song: {time.time() - start_time:.2f} sec")

    return all_metrics


def compute_metric_avg(
    store_dir: str,
    args,
    instruments: List[str],
    config: ConfigDict,
    all_metrics: Dict[str, Dict[str, Union[List[float], Dict[str, float]]]],
    start_time: float,
    track_count: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute average metrics across instruments (DDP-aware) and optionally log to file.

    For each metric, computes the mean value per instrument from its collected values
    (list in non-DDP, or dict-of-{path: value} in DDP), sums these instrument means,
    and divides by `len(instruments)` to obtain the final average (legacy behavior).
    Prints/logs only on rank 0 when `torch.distributed` is initialized; if `store_dir`
    is non-empty, writes a `results.txt` with logs.

    Args:
        store_dir (str): Directory to write `results.txt` when logging is enabled.
        args: Run arguments included in the log header when `store_dir` is provided.
        instruments (List[str]): Instruments to include in the averaging.
        config (ConfigDict): Config used for informational logging (e.g., overlap).
        all_metrics (Dict[str, Dict[str, Union[List[float], Dict[str, float]]]]):
            Nested metrics container:
              - non-DDP: {metric: {instrument: [values...]}}
              - DDP:     {metric: {instrument: {path: value, ...}}}
        start_time (float): Timestamp for reporting elapsed time.

    Returns:
        Dict[str, float]: Mapping from metric name to its average over instruments.
    """

    should_print = is_main_process()

    logs: List[str] = []
    verbose_logging = bool(store_dir) and should_print
    if verbose_logging:
        logs.append(str(args))

    logs = logging(logs, text=f"Num overlap: {config.inference.num_overlap}", verbose_logging=verbose_logging)
    if track_count is not None:
        logs = logging(logs, text=f"Validation tracks: {track_count}", verbose_logging=verbose_logging)

    metric_sum: Dict[str, float] = {}
    metric_count: Dict[str, int] = {}

    for instr in instruments:
        for metric_name in all_metrics:
            per_instr_container = all_metrics[metric_name]  # dict: instr -> (list | dict[path->val])

            values_obj = per_instr_container.get(instr, []) if isinstance(per_instr_container, dict) else []
            if isinstance(values_obj, dict):
                vals = list(values_obj.values())
            else:
                vals = list(values_obj)

            arr = np.asarray(vals, dtype=float)
            finite_arr = arr[np.isfinite(arr)]
            if finite_arr.size == 0:
                mean_val = float("nan")
                std_val = float("nan")
            else:
                mean_val = float(finite_arr.mean())
                std_val = float(finite_arr.std())

            logs = logging(
                logs,
                text=f"Instr {instr} {metric_name}: {mean_val:.4f} (Std: {std_val:.4f}, Count: {finite_arr.size})",
                verbose_logging=verbose_logging
            )
            if track_count is not None and finite_arr.size < track_count:
                logs = logging(
                    logs,
                    text=f"Missing/invalid {metric_name} samples for {instr}: {track_count - finite_arr.size}",
                    verbose_logging=verbose_logging
                )
            if np.isfinite(mean_val):
                metric_sum[metric_name] = metric_sum.get(metric_name, 0.0) + mean_val
                metric_count[metric_name] = metric_count.get(metric_name, 0) + 1

    metric_avg: Dict[str, float] = {}
    for metric_name in all_metrics:
        count = metric_count.get(metric_name, 0)
        metric_avg[metric_name] = metric_sum[metric_name] / count if count else float("nan")

    if len(instruments) > 1:
        for metric_name, avg in metric_avg.items():
            logs = logging(logs, text=f"Metric avg {metric_name:11s}: {avg:.4f}", verbose_logging=verbose_logging)

    logs = logging(logs, text=f"Elapsed time: {time.time() - start_time:.2f} sec", verbose_logging=verbose_logging)

    if store_dir:
        write_results_in_file(store_dir, logs)

    return metric_avg


def valid(
    model: torch.nn.Module,
    args,
    config: ConfigDict,
    device: torch.device,
    verbose: bool = False
) -> Tuple[dict, dict]:
    """
    Validate a trained model on a set of audio mixtures and compute metrics.

    This function performs validation by separating audio sources from mixtures,
    computing evaluation metrics, and optionally saving results to a file.

    Parameters:
    ----------
    model : torch.nn.Module
        The trained model for source separation.
    args : Namespace
        Command-line arguments or equivalent object containing configurations.
    config : dict
        Configuration dictionary with model and processing parameters.
    device : torch.device
        The device (CPU or CUDA) to run the model on.
    verbose : bool, optional
        If True, enables verbose output during processing. Default is False.

    Returns:
    -------
    dict
        A dictionary of average metrics across all instruments.
    """

    start_time = time.time()
    model = unwrap_parallel_model(model)
    model.eval().to(device)

    # dir to save files, if empty no saving
    store_dir = getattr(args, 'store_dir', '')
    # codec to save files
    if 'extension' in config['inference']:
        extension = config['inference']['extension']
    else:
        extension = getattr(args, 'extension', 'wav')

    all_mixtures_path = get_mixture_paths(args, verbose, config, extension)
    all_metrics = process_audio_files(all_mixtures_path, model, args, config, device, verbose, not verbose)
    instruments = prefer_target_instrument(config)

    return compute_metric_avg(store_dir, args, instruments, config, all_metrics, start_time, len(all_mixtures_path)), all_metrics


def validate_in_subprocess(
    proc_id: int,
    queue: torch.multiprocessing.Queue,
    all_mixtures_path: List[str],
    model: torch.nn.Module,
    args,
    config: ConfigDict,
    device: str,
    return_dict
) -> None:
    """
    Perform validation on a subprocess with multi-processing support. Each process handles inference on a subset of the mixture files
    and updates the shared metrics dictionary.

    Parameters:
    ----------
    proc_id : int
        The process ID (used to assign metrics to the correct key in `return_dict`).
    queue : torch.multiprocessing.Queue
        Queue to receive paths to the mixture files for processing.
    all_mixtures_path : List[str]
        List of paths to the mixture files to be processed.
    model : torch.nn.Module
        The model to be used for inference.
    args : dict
        Dictionary containing various argument configurations (e.g., metrics to calculate).
    config : ConfigDict
        Configuration object containing model settings and training parameters.
    device : str
        The device to use for inference (e.g., 'cpu', 'cuda:0').
    return_dict : torch.multiprocessing.Manager().dict
        Shared dictionary to store the results from each process.

    Returns:
    -------
    None
        The function modifies the `return_dict` in place, but does not return any value.
    """

    progress_bar = None
    try:
        m1 = unwrap_parallel_model(model).eval().to(device)
        if proc_id == 0:
            progress_bar = tqdm(total=len(all_mixtures_path))

        all_metrics = {
            metric: {instr: [] for instr in config.training.instruments}
            for metric in args.metrics
        }

        while True:
            _, path = queue.get()
            if path is None:
                break
            single_metrics = process_audio_files([path], m1, args, config, device, False, False)
            pbar_dict = {}
            for instr in config.training.instruments:
                for metric_name in all_metrics:
                    all_metrics[metric_name][instr] += single_metrics[metric_name][instr]
                    if len(single_metrics[metric_name][instr]) > 0:
                        pbar_dict[f"{metric_name}_{instr}"] = f"{single_metrics[metric_name][instr][0]:.4f}"
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(pbar_dict)
        return_dict[proc_id] = all_metrics
    except Exception:
        return_dict[proc_id] = {"__error__": traceback.format_exc()}
        raise
    finally:
        if progress_bar is not None:
            progress_bar.close()


def run_parallel_validation(
    verbose: bool,
    all_mixtures_path: List[str],
    config: ConfigDict,
    model: torch.nn.Module,
    device_ids: List[int],
    args,
    return_dict
) -> None:
    """
    Run parallel validation using multiple processes. Each process handles a subset of the mixture files and computes the metrics.
    The results are stored in a shared dictionary.

    Parameters:
    ----------
    verbose : bool
        Flag to print detailed information about the validation process.
    all_mixtures_path : List[str]
        List of paths to the mixture files to be processed.
    config : ConfigDict
        Configuration object containing model settings and validation parameters.
    model : torch.nn.Module
        The model to be used for inference.
    device_ids : List[int]
        List of device IDs (for multi-GPU setups) to use for validation.
    args : dict
        Dictionary containing various argument configurations (e.g., metrics to calculate).

    Returns:
    -------
        A shared dictionary containing the validation metrics from all processes.
    """

    device_ids = normalize_device_ids(device_ids)
    model = unwrap_parallel_model(model).to('cpu')

    queue = torch.multiprocessing.Queue()
    processes = []

    for i, device in enumerate(device_ids):
        if torch.cuda.is_available():
            device = f'cuda:{device}'
        else:
            device = 'cpu'
        p = torch.multiprocessing.Process(
            target=validate_in_subprocess,
            args=(i, queue, all_mixtures_path, model, args, config, device, return_dict)
        )
        p.start()
        processes.append(p)
    for i, path in enumerate(all_mixtures_path):
        queue.put((i, path))
    for _ in range(len(device_ids)):
        queue.put((None, None))  # sentinel value to signal subprocesses to exit
    for p in processes:
        p.join()

    errors = []
    for i, p in enumerate(processes):
        worker_result = return_dict.get(i)
        if isinstance(worker_result, dict) and "__error__" in worker_result:
            errors.append(f"worker {i}:\n{worker_result['__error__']}")
        elif p.exitcode != 0:
            errors.append(f"worker {i} exited with code {p.exitcode}")
        elif i not in return_dict:
            errors.append(f"worker {i} exited without returning metrics")

    if errors:
        raise RuntimeError("Parallel validation failed:\n" + "\n".join(errors))

    return


def block_bounds(num_tracks: int, world_size: int, rank: int) -> Tuple[int, int]:
    """
    Split a dataset of `num_tracks` items into `world_size` equal contiguous blocks
    and return the half-open interval [start, end) assigned to the given `rank`.

    This function enforces exact divisibility: `num_tracks` must be divisible
    by `world_size`, otherwise a ValueError is raised.

    Args:
        num_tracks (int): Total number of items to split (must be ≥ 0).
        world_size (int): Number of workers to divide the items into (must be > 0).
        rank (int): Zero-based worker index (0 ≤ rank < world_size).

    Returns:
        Tuple[int, int]: A pair `(start, end)` defining the block of indices for this rank.

    Raises:
        ValueError: If `num_tracks` is not divisible by `world_size`.

    Example:
         [block_bounds(12, 4, r) for r in range(4)]
        [(0, 3), (3, 6), (6, 9), (9, 12)]

         block_bounds(8, 2, 1)
        (4, 8)

         block_bounds(10, 3, 0)
        Traceback (most recent call last):
        ...
        ValueError: n (10) must be divisible by world_size (3)
    """
    if num_tracks % world_size != 0:
        raise ValueError(f"n ({num_tracks}) must be divisible by world_size ({world_size})")

    chunk = num_tracks // world_size
    start = rank * chunk
    end = start + chunk
    return start, end



def valid_multi_gpu(
    model: torch.nn.Module,
    args,
    config: ConfigDict,
    device_ids: Optional[List[int]] = None,
    verbose: bool = False
) -> Tuple[Dict[str, float], Dict]:
    """
    Validate a separator model across multiple GPUs with a unified API.

    Runs validation either in Distributed Data Parallel (DDP) mode—detected via
    `torch.distributed.is_initialized()`—or, if DDP is not active, via
    multi-processing / single-GPU execution using the provided `device_ids`.
    Collects per-track metrics, aggregates them into per-instrument/per-metric
    arrays, and computes per-metric averages.

    Behavior:
      * DDP mode: splits the dataset across ranks and gathers metrics; only rank 0
        returns results, while other ranks return `(None, None)`.
      * Non-DDP: launches parallel workers when `len(device_ids) > 1`, otherwise
        runs on a single device/CPU.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        args: Runtime arguments (e.g., metrics list, store dir).
        config (ConfigDict): Configuration with inference/training settings.
        device_ids (Optional[List[int]]): GPU device IDs for non-DDP parallelism.
            If None or length is 1, runs on a single device.
        verbose (bool, optional): If True, print progress/logs. Defaults to False.

    Returns:
        Tuple[Dict[str, float], Dict]: A pair `(metric_avg, all_metrics)` where
            - `metric_avg` maps metric name to its average score,
            - `all_metrics` is a nested dict `{metric: {instrument: List[float]}}`.
          In DDP mode, non-zero ranks return `(None, None)`.
    """

    start_time = time.time()
    if device_ids is None:
        device_ids = getattr(args, "device_ids", [0])
    device_ids = normalize_device_ids(device_ids)

    inference = getattr(config, "inference", None)
    if inference is None and isinstance(config, dict):
        inference = config.get("inference", {})
    extension = getattr(inference, "extension", None)
    if extension is None:
        if isinstance(inference, dict):
            extension = inference.get("extension", getattr(args, "extension", "wav"))
        else:
            extension = getattr(args, "extension", "wav")

    all_mixtures_path = get_mixture_paths(args, verbose, config, extension)

    ddp_mode = dist.is_initialized()

    if ddp_mode:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if torch.cuda.is_available():
            device_id = device_ids[rank] if rank < len(device_ids) else torch.cuda.current_device()
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")
        model = unwrap_parallel_model(model)
        model.to(device)
        model.eval()

        num_tracks = len(all_mixtures_path)
        pad_needed = (-num_tracks) % world_size
        if pad_needed and num_tracks > 0:
            all_mixtures_path += all_mixtures_path[:pad_needed]
        padded_num_tracks = len(all_mixtures_path)
        target_len = padded_num_tracks // world_size
        start, end = block_bounds(padded_num_tracks, world_size, rank)
        per_rank_data = all_mixtures_path[start:end]

        local_metrics = {
            metric: {instr: [] for instr in config.training.instruments}
            for metric in args.metrics
        }

        with torch.no_grad():
            single_metrics = process_audio_files(
                per_rank_data, model, args, config, device, verbose=verbose
            )
            for instr in config.training.instruments:
                for metric_name in args.metrics:
                    local_metrics[metric_name][instr] = single_metrics[metric_name][instr]

        all_metrics: Dict[str, Dict[str, List[float]]] = {m: {} for m in args.metrics}
        for metric in args.metrics:
            for instr in config.training.instruments:
                all_metrics[metric][instr] = []
                per_instr = local_metrics[metric][instr]
                if isinstance(per_instr, dict):
                    local_data = list(per_instr.values())
                else:
                    local_data = list(per_instr)

                gather_device = distributed_tensor_device(device)
                if len(local_data) == 0:
                    local_tensor = torch.full((target_len,), float("nan"), dtype=torch.float32, device=gather_device)
                else:
                    if len(local_data) < target_len:
                        local_data = local_data + [float("nan")] * (target_len - len(local_data))
                    local_tensor = torch.tensor(local_data, dtype=torch.float32, device=gather_device)

                gathered_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
                dist.all_gather(gathered_list, local_tensor)

                cat_vals = torch.cat(gathered_list).tolist()[:num_tracks]
                all_metrics[metric][instr] = cat_vals

        if dist.get_rank() == 0:
            instruments = prefer_target_instrument(config)
            metric_avg = compute_metric_avg(
                getattr(args, "store_dir", ""),
                args,
                instruments,
                config,
                all_metrics,
                start_time,
                num_tracks
            )
            return metric_avg, all_metrics

        return None, None

    # Not DDP
    store_dir = getattr(args, "store_dir", "")

    if len(device_ids) <= 1:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{device_ids[0]}")
        else:
            device = torch.device("cpu")
        return valid(model, args, config, device, verbose=verbose)

    return_dict = torch.multiprocessing.Manager().dict()
    run_parallel_validation(verbose, all_mixtures_path, config, model, device_ids, args, return_dict)

    all_metrics: Dict[str, Dict[str, List[float]]] = {m: {} for m in args.metrics}
    for metric in args.metrics:
        for instr in config.training.instruments:
            merged: List[float] = []
            for i in range(len(device_ids)):
                merged += return_dict[i][metric][instr]
            all_metrics[metric][instr] = merged

    instruments = prefer_target_instrument(config)
    metric_avg = compute_metric_avg(store_dir, args, instruments, config, all_metrics, start_time, len(all_mixtures_path))
    return metric_avg, all_metrics


def check_validation(dict_args):
    args = parse_args_valid(dict_args)
    torch.backends.cudnn.benchmark = True
    try:
        torch.multiprocessing.set_start_method('spawn')
    except Exception as e:
        pass
    model, config = get_model_from_config(args.model_type, args.config_path)
    if 'model_type' in config.training:
        args.model_type = config.training.model_type
    args = apply_config_args(args, config, mode='valid')
    validate_valid_setup(config, args)
    if args.start_check_point:
        checkpoint = torch.load(args.start_check_point, weights_only=False, map_location='cpu')
        load_start_checkpoint(args, model, checkpoint, type_='valid')
    if args.lora_checkpoint_peft:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_checkpoint_peft)
        model = model.merge_and_unload()
    print(f"Instruments: {config.training.instruments}")

    device_ids = args.device_ids
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_ids[0]}')
    else:
        device = 'cpu'
        print('CUDA is not available. Run validation on CPU. It will be very slow...')

    if torch.cuda.is_available() and len(device_ids) > 1:
        metrics = valid_multi_gpu(model, args, config, device_ids, verbose=False)
    else:
        metrics = valid(model, args, config, device, verbose=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return metrics


if __name__ == "__main__":
    check_validation(None)
