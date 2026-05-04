# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'


import os
import random
import json
import hashlib
import numpy as np
import torch
import soundfile as sf
from functools import partial
import pickle
import tempfile
import itertools
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from glob import glob
import audiomentations as AU
import pedalboard as PB
import warnings
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
warnings.filterwarnings("ignore")
import argparse


METADATA_CACHE_VERSION = 2
RANK_SEED_STRIDE = 1_000_003


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def rank_adjusted_seed(seed: int, rank: Optional[int] = None) -> int:
    if rank is None:
        rank = get_rank()
    return (int(seed) + int(rank) * RANK_SEED_STRIDE) % (2 ** 32)


def seed_data_rng(seed: int, rank: Optional[int] = None) -> None:
    data_seed = rank_adjusted_seed(seed, rank)
    random.seed(data_seed)
    np.random.seed(data_seed)


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def cfg_to_plain_dict(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "items"):
        return {key: value for key, value in config.items()}
    return dict(config)


def normalize_path_config(path_or_paths: Union[str, List[str], Tuple[str, ...]]) -> List[str]:
    paths = path_or_paths if isinstance(path_or_paths, (list, tuple)) else [path_or_paths]
    return [os.path.abspath(os.path.expanduser(str(path))) for path in paths]


def normalize_metadata_for_fingerprint(metadata: Any) -> Any:
    def normalize_entry(item):
        path, length = item
        return [os.path.abspath(os.path.expanduser(str(path))), int(length)]

    if isinstance(metadata, dict):
        return {
            str(instr): sorted(normalize_entry(item) for item in values)
            for instr, values in sorted(metadata.items())
        }
    return sorted(normalize_entry(item) for item in metadata)


def metadata_fingerprint(metadata: Any) -> str:
    normalized = normalize_metadata_for_fingerprint(metadata)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf8")).hexdigest()


def atomic_pickle_dump(obj: Any, path: str) -> None:
    target_path = os.path.abspath(path)
    target_dir = os.path.dirname(target_path) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=target_dir)
    os.close(fd)
    try:
        with open(tmp_path, 'wb') as out:
            pickle.dump(obj, out)
        os.replace(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def atomic_json_dump(obj: Any, path: str) -> None:
    target_path = os.path.abspath(path)
    target_dir = os.path.dirname(target_path) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=target_dir)
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf8") as out:
            json.dump(obj, out, indent=2)
        os.replace(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def fix_audio_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    if audio.shape[-1] < target_length:
        pad_width = [(0, 0)] * audio.ndim
        pad_width[-1] = (0, target_length - audio.shape[-1])
        audio = np.pad(audio, pad_width, mode='constant')
    elif audio.shape[-1] > target_length:
        audio = audio[..., :target_length]
    return np.ascontiguousarray(np.nan_to_num(audio, copy=False).astype(np.float32, copy=False))


def music_collate_fn(batch, min_size=1 * 44100, max_size=30 * 44100):
    """
    batch: list of elements from Dataset.__getitem__
    """
    if not batch:
        return torch.utils.data._utils.collate.default_collate(batch)

    item_len = len(batch[0])
    if item_len not in (2, 3):
        raise ValueError(f"music_collate_fn expects 2 or 3 values per item, got {item_len}")

    min_len_in_batch = min([t[1].shape[-1] for t in batch])
    min_len_in_batch = min(min_len_in_batch, max_size)
    min_size = min(min_size, min_len_in_batch)
    target_length = random.randint(min_size, min_len_in_batch)

    new_batch = []
    for item in batch:
        if len(item) != item_len:
            raise ValueError("music_collate_fn received mixed item sizes in one batch")
        stems, mix = item[:2]
        stems_crop = stems[..., :target_length]
        mix_crop = mix[..., :target_length]
        if item_len == 3:
            new_batch.append((stems_crop, mix_crop, item[2]))
        else:
            new_batch.append((stems_crop, mix_crop))

    return torch.utils.data._utils.collate.default_collate(new_batch)


def seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = (base_seed + worker_id) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def dataloader_kwargs(args: argparse.Namespace, collate_fn=None) -> dict:
    num_workers = max(int(args.num_workers), 0)
    pin_memory = bool(args.pin_memory and torch.cuda.is_available())
    base_seed = rank_adjusted_seed(int(getattr(args, 'seed', 0)))
    generator = torch.Generator()
    generator.manual_seed(base_seed)

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
        "generator": generator,
    }

    if num_workers > 0:
        kwargs["worker_init_fn"] = partial(seed_worker, base_seed=base_seed)
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        if args.prefetch_factor is not None:
            if args.prefetch_factor <= 0:
                raise ValueError("--prefetch_factor must be > 0 when provided")
            kwargs["prefetch_factor"] = args.prefetch_factor
    elif args.persistent_workers or args.prefetch_factor is not None:
        should_print = not dist.is_initialized() or dist.get_rank() == 0
        if should_print:
            print("persistent_workers/prefetch_factor are ignored because num_workers=0")

    return kwargs


def prepare_data(config: Union[ConfigDict, OmegaConf], args: argparse.Namespace, batch_size: int) -> DataLoader:
    """
    Build the training DataLoader. If torch.distributed.is_initialized() is True,
    construct a DDP DataLoader with DistributedSampler; otherwise, construct a regular DataLoader.

    Args:
        config: Dataset configuration passed to MSSDataset.
        args: Must provide data_path, results_path, dataset_type, and DataLoader settings.
        batch_size: Per-process mini-batch size.

    Returns:
        Configured DataLoader for the training split.
    """

    actionable_collate = None
    if 'augmentations' in config:
        if 'enable' in config['augmentations']:
            if config['augmentations']['enable']:
                if 'chunk_size_augm' in config['augmentations']:
                    if config['augmentations']['chunk_size_augm']:
                        if 'chunk_size_min' in config['augmentations'] and 'chunk_size_max' in config['augmentations']:
                            min_size = int(config['augmentations']['chunk_size_min'])
                            max_size = int(config['augmentations']['chunk_size_max'])
                            print("Use chunk size augmentation with range: {} up to {}".format(min_size, max_size))
                            try:
                                if max_size > config['audio']['chunk_size']:
                                    print('Warning: you need to increase config.audio.chunk_size from {} up to: {}'.format(
                                        config['audio']['chunk_size'], max_size
                                    ))
                            except:
                                pass
                            actionable_collate = partial(music_collate_fn, min_size=min_size, max_size=max_size)

    # DDP
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        seed_data_rng(int(getattr(args, 'seed', 0)), rank)

        if args.dataset_type != 5:
            ddp_batch = batch_size * world_size # maintain "num_steps" semantics across the whole world
        else:
            ddp_batch = batch_size

        trainset = MSSDataset(
            config,
            args.data_path,
            batch_size=ddp_batch,
            metadata_path=os.path.join(args.results_path, f"metadata_{args.dataset_type}.pkl"),
            dataset_type=args.dataset_type,
        )

        sampler = DistributedSampler(
            trainset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=int(getattr(args, 'seed', 0)),
        )

        train_loader = DataLoader(
            trainset,
            batch_size=batch_size,             # per-process batch size
            sampler=sampler,                   # sampler handles shuffling in DDP
            **dataloader_kwargs(args, actionable_collate),
        )
    else:
        trainset = MSSDataset(
            config,
            args.data_path,
            batch_size=batch_size,
            metadata_path=os.path.join(args.results_path, f"metadata_{args.dataset_type}.pkl"),
            dataset_type=args.dataset_type,
        )

        train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            **dataloader_kwargs(args, actionable_collate),
        )

    return train_loader


def load_chunk(path, length, chunk_size, offset=None, target_channels=2):
    """
    Returns array with shape (target_channels, chunk_size)
    """

    if length is None or length <= 0:
        return np.zeros((target_channels, chunk_size), dtype=np.float32)

    if chunk_size <= length:
        if offset is None:
            start = np.random.randint(length - chunk_size + 1)
        else:
            start = max(0, min(int(offset), max(length - chunk_size, 0)))
        x = sf.read(path, dtype='float32', start=start, frames=chunk_size)[0]
    else:
        if offset is None:
            start = 0
        else:
            start = max(0, int(offset))
        frames_to_read = length
        x = sf.read(path, dtype='float32', start=start, frames=frames_to_read)[0]

    if x.ndim == 1:
        x = x[:, None]

    ch = x.shape[1]
    if ch == target_channels:
        pass
    elif ch > target_channels:
        x = x[:, :target_channels]
    elif ch == 1 and target_channels > 1:
        x = np.repeat(x, target_channels, axis=1)
    else:
        raise ValueError(f"Path: {path}, num_channels: {ch}")

    return fix_audio_length(x.T, chunk_size)


def get_track_set_length(params):
    if len(params) == 4:
        path, instruments, file_types, dataset_type = params
        expected_sample_rate = None
        strict_sample_rate = False
    else:
        path, instruments, file_types, dataset_type, expected_sample_rate, strict_sample_rate = params
    should_print = (not dist.is_initialized() or dist.get_rank() == 0) and dataset_type != 7
    # Check lengths of all instruments (it can be different in some cases)
    lengths_arr = []
    sample_rates = set()
    for instr in instruments:
        length = -1
        for extension in file_types:
            path_to_audio_file = path + '/{}.{}'.format(instr, extension)
            if os.path.isfile(path_to_audio_file):
                try:
                    info = sf.info(path_to_audio_file)
                except Exception as e:
                    if should_print:
                        print(f'Cant read file "{path_to_audio_file}": {e}')
                    continue
                length = info.frames
                sample_rates.add(info.samplerate)
                break
        if length == -1:
            if should_print:
                print('Cant find file "{}" in folder {}'.format(instr, path))
            continue
        lengths_arr.append(length)
    if len(lengths_arr) == 0:
        if should_print:
            print(f'No usable stems found in folder {path}')
        return None

    if dataset_type in [6, 7]:
        for extension in file_types:
            path_to_mix_file = path + '/mixture.{}'.format(extension)
            if os.path.isfile(path_to_mix_file):
                try:
                    info = sf.info(path_to_mix_file)
                except Exception as e:
                    if should_print:
                        print(f'Cant read mixture file "{path_to_mix_file}": {e}')
                    continue
                lengths_arr.append(info.frames)
                sample_rates.add(info.samplerate)
                break

    file_label = 'audio files' if dataset_type in [6, 7] else 'stems'
    lengths_arr = np.array(lengths_arr)
    if lengths_arr.min() != lengths_arr.max() and should_print:
        print(f'Warning: lengths of {file_label} are different for path: {path}. ({lengths_arr.min()} != {lengths_arr.max()})')
    if len(sample_rates) > 1 and should_print:
        print(f'Warning: sample rates of {file_label} are different for path: {path}. ({sorted(sample_rates)})')
    if expected_sample_rate is not None:
        mismatched_rates = sorted(rate for rate in sample_rates if rate != expected_sample_rate)
        if mismatched_rates:
            msg = (
                f'Sample rate mismatch for path: {path}. '
                f'Expected {expected_sample_rate}, found {mismatched_rates}. '
                'Dataset loading does not resample audio.'
            )
            if strict_sample_rate:
                raise ValueError(msg)
            if should_print:
                print(f'Warning: {msg}')
    # We use minimum to allow overflow for soundfile read in non-equal length cases
    return path, lengths_arr.min()


# For multiprocessing
def get_track_length(params):
    if isinstance(params, (tuple, list)):
        path = params[0]
        expected_sample_rate = params[1] if len(params) > 1 else None
        strict_sample_rate = params[2] if len(params) > 2 else False
    else:
        path = params
        expected_sample_rate = None
        strict_sample_rate = False
    try:
        info = sf.info(path)
    except Exception:
        return None
    if expected_sample_rate is not None and info.samplerate != expected_sample_rate and strict_sample_rate:
        raise ValueError(
            f'Sample rate mismatch for path: {path}. '
            f'Expected {expected_sample_rate}, found {info.samplerate}. '
            'Dataset loading does not resample audio.'
        )
    length = info.frames
    if length <= 0:
        return None
    return (path, length)


def process_chunk_worker(args):
    task, instruments, file_types, min_mean_abs, target_channels = args
    track_path, track_length, offset, chunk_size = task

    try:
        for instrument in instruments:
            instrument_loud_enough = False
            for extension in file_types:
                path_to_audio_file = track_path + '/{}.{}'.format(instrument, extension)
                if os.path.isfile(path_to_audio_file):
                    try:
                        source = load_chunk(path_to_audio_file, length=track_length, offset=offset,
                                            chunk_size=chunk_size, target_channels=target_channels)
                        if np.abs(source).mean() >= min_mean_abs:
                            instrument_loud_enough = True
                            break
                    except Exception as e:
                        return (track_path, offset, False)

            if not instrument_loud_enough:
                return (track_path, offset, False)

        return (track_path, offset, True)

    except Exception:
        return (track_path, offset, False)


class MSSDataset(torch.utils.data.Dataset):
    def __init__(self, config, data_path, metadata_path="metadata.pkl", dataset_type=1, batch_size=None, verbose=True):
        self.verbose = verbose
        self.config = config
        self.dataset_type = dataset_type  # 1, 2, 3, 4 or 5
        self.data_path = data_path
        self.instruments = instruments = list(config.training.instruments)
        if batch_size is None:
            batch_size = config.training.batch_size
        self.batch_size = batch_size
        self.file_types = list(config.training.get('file_types', ['wav', 'flac']))
        self.metadata_path = metadata_path
        self.chunk_size = int(config.audio.chunk_size)
        self.min_mean_abs = float(config.audio.min_mean_abs)
        self.sample_rate = int(getattr(config.audio, 'sample_rate', getattr(config.training, 'samplerate', 44100)))
        self.target_channels = int(getattr(
            config.audio,
            'channels',
            getattr(config.audio, 'num_channels', getattr(config.training, 'channels', 2)),
        ))
        self.max_load_attempts = int(config.training.get('max_load_attempts', 50))
        self.strict_sample_rate = bool(config.training.get('strict_sample_rate', False))
        self.class_balanced_stems = bool(config.training.get('class_balanced_stems', False))
        if self.chunk_size <= 0:
            raise ValueError("config.audio.chunk_size must be > 0")
        if self.target_channels <= 0:
            raise ValueError("configured audio channel count must be > 0")
        if self.max_load_attempts <= 0:
            raise ValueError("config.training.max_load_attempts must be > 0")
        self._track_length_by_path = {}
        self._stem_aug_cache = {}
        self._pedalboard_aug_cache = {}
        self._mixture_mp3_aug = None
        self.class_to_tracks = {}
        self.available_classes = []

        should_print = is_main_process()

        # Augmentation block
        self.aug = False
        if 'augmentations' in config:
            if config['augmentations'].enable is True:
                if self.verbose and should_print:
                    print('Use augmentation for training')
                self.aug = True
        else:
            if self.verbose and should_print:
                print('There is no augmentations block in config. Augmentations disabled for training...')

        metadata = self.get_metadata()

        if self.dataset_type in [1, 4, 5, 6, 7]:
            if len(metadata) > 0:
                if self.verbose and should_print:
                    print('Found tracks in dataset: {}'.format(len(metadata)))
            else:
                if should_print:
                    print('No tracks found for training. Check paths you provided!')
                raise RuntimeError('No tracks found for training. Check paths you provided.')
        else:
            for instr in self.instruments:
                if self.verbose and should_print:
                    print('Found tracks for {} in dataset: {}'.format(instr, len(metadata[instr])))
        self.metadata = metadata
        if self.dataset_type in [1, 4, 5, 6, 7]:
            self._track_length_by_path = dict(metadata)
        self.do_chunks = config.training.get('precompute_chunks', False) and float(self.min_mean_abs) > 0
        # For dataset_type 5 - precompute all chunks
        if self.dataset_type == 5 or (self.dataset_type == 4 or self.dataset_type == 6) and self.do_chunks:
            self._initialize_chunks_metadata()
        if self.dataset_type == 7:
            self._build_class_to_tracks(filter_frequent=True)
        elif self.dataset_type in [4, 6] and self.class_balanced_stems:
            self._build_class_to_tracks(filter_frequent=False)
    def __len__(self):
        if self.dataset_type == 5:
            return len(self.chunks_metadata)
        return self.config.training.num_steps * self.batch_size


    def _get_mixture_mp3_aug(self):
        if self._mixture_mp3_aug is None:
            self._mixture_mp3_aug = AU.Mp3Compression(
                min_bitrate=self.config['augmentations']['mp3_compression_on_mixture_bitrate_min'],
                max_bitrate=self.config['augmentations']['mp3_compression_on_mixture_bitrate_max'],
                backend=self.config['augmentations']['mp3_compression_on_mixture_backend'],
                p=self.config['augmentations']['mp3_compression_on_mixture'],
            )
        return self._mixture_mp3_aug


    def _augmentation_config(self, instr):
        augmentations = self.config['augmentations']
        if 'all' in augmentations:
            augs = cfg_to_plain_dict(augmentations['all'])
        else:
            augs = dict()

        if instr in augmentations:
            for el in augmentations[instr]:
                augs[el] = augmentations[instr][el]
        return augs


    def _get_stem_aug(self, instr, name, factory):
        key = (instr, name)
        aug = self._stem_aug_cache.get(key)
        if aug is None:
            aug = factory()
            self._stem_aug_cache[key] = aug
        return aug


    def _get_pedalboard(self, instr, name, plugin_factory):
        key = (instr, name)
        board = self._pedalboard_aug_cache.get(key)
        if board is None:
            board = PB.Pedalboard([plugin_factory()])
            self._pedalboard_aug_cache[key] = board
        return board


    @staticmethod
    def _set_pedalboard_params(board, **params):
        plugin = board[0]
        for name, value in params.items():
            setattr(plugin, name, value)


    def _metadata_cache_config(self) -> Dict[str, Any]:
        return {
            'dataset_type': self.dataset_type,
            'data_path': normalize_path_config(self.data_path),
            'instruments': list(self.instruments),
            'file_types': sorted(self.file_types),
            'sample_rate': self.sample_rate,
            'strict_sample_rate': self.strict_sample_rate,
            'target_channels': self.target_channels,
        }


    def _metadata_cache_payload(self, metadata):
        return {
            'metadata_cache_version': METADATA_CACHE_VERSION,
            'config': self._metadata_cache_config(),
            'metadata': metadata,
        }


    def _metadata_fingerprint(self) -> str:
        return metadata_fingerprint(self.metadata)


    def _extract_metadata_from_cache(self, cache_data):
        if not isinstance(cache_data, dict) or 'metadata_cache_version' not in cache_data:
            return None

        if cache_data.get('metadata_cache_version') != METADATA_CACHE_VERSION:
            return None

        if cache_data.get('config') != self._metadata_cache_config():
            return None

        return cache_data.get('metadata')


    def _active_stem_mask(self, active_stem_ids: List[int]) -> torch.Tensor:
        mask = torch.zeros(len(self.instruments), dtype=torch.bool)
        if active_stem_ids:
            mask[active_stem_ids] = True
        return mask


    def _keep_original_mixture(self) -> bool:
        if 'augmentations' not in self.config:
            return False
        return bool(self.config.augmentations.get('keep_original_mixture', False))


    def _can_change_stems(self) -> bool:
        # For explicit-mixture training, keeping the original mixture means target
        # stems must remain unmodified as well.
        return not (self.dataset_type in [6, 7] and self._keep_original_mixture())


    def _load_mix_from_track(self, track_path: str, track_length: int, offset=None) -> Optional[np.ndarray]:
        should_print = is_main_process()
        for extension in self.file_types:
            path_to_mix_file = f"{track_path}/mixture.{extension}"
            if os.path.isfile(path_to_mix_file):
                try:
                    return load_chunk(
                        path_to_mix_file,
                        track_length,
                        self.chunk_size,
                        offset=offset,
                        target_channels=self.target_channels,
                    )
                except Exception as e:
                    if should_print:
                        print('Error loading mix: {} Path: {}'.format(e, path_to_mix_file))
                break
        return None


    def __getitem__(self, index):
        if self.dataset_type == 7:
            res, mix, active_stem_ids = self.load_class_balanced_aligned()
        elif self.dataset_type == 5:
            track_path, offset = self.chunks_metadata[index]
            res = self._load_chunk_by_offset(track_path, offset)
        elif self.dataset_type in [1, 2, 3]:
            res = self.load_random_mix()
        else:  # type 4 or 6
            if self.do_chunks:
                track_path, offset = self.chunks_metadata[np.random.randint(len(self.chunks_metadata))]
                if self.dataset_type == 6:
                    res, mix = self._load_chunk_by_offset(track_path, offset, return_mix=True)
                else:
                    res = self._load_chunk_by_offset(track_path, offset)
            else:
                if self.dataset_type == 6:
                    res, mix = self.load_aligned_data()
                else:
                    res, _ = self.load_aligned_data()

        # Randomly change loudness of each stem
        stem_values_changed = False
        if self.aug:
            if self._can_change_stems() and 'loudness' in self.config['augmentations']:
                if self.config['augmentations']['loudness']:
                    loud_values = np.random.uniform(
                        low=self.config['augmentations']['loudness_min'],
                        high=self.config['augmentations']['loudness_max'],
                        size=(len(res),)
                    )
                    loud_values = torch.tensor(loud_values, dtype=torch.float32)
                    res *= loud_values[:, None, None]
                    stem_values_changed = True
        if self.dataset_type in [6, 7]:
            if stem_values_changed and not self._keep_original_mixture():
                mix = res.sum(0)
        else:
            mix = res.sum(0)

        if self.aug:
            if 'mp3_compression_on_mixture' in self.config['augmentations']:
                if self.config['augmentations']['mp3_compression_on_mixture'] > 0:
                    apply_aug = self._get_mixture_mp3_aug()
                    mix_conv = mix.cpu().numpy().astype(np.float32)
                    required_shape = mix_conv.shape
                    mix = apply_aug(samples=mix_conv, sample_rate=self.sample_rate)
                    mix = fix_audio_length(mix, required_shape[-1])
                    mix = torch.tensor(mix, dtype=torch.float32)

        # If we need to optimize only given stem
        if self.config.training.target_instrument is not None:
            index = self.config.training.instruments.index(self.config.training.target_instrument)
            return res[index:index+1], mix

        if self.dataset_type==7:
            return res, mix, active_stem_ids

        return res, mix


    def _build_class_to_tracks(self, filter_frequent: bool = True):
        should_print = is_main_process()

        cache_path = self.metadata_path.replace('.pkl', '_class_to_tracks.json')
        cache_label = f"[dataset_type={self.dataset_type}]"

        total_tracks = len(self.metadata)
        max_ratio = self.config.training.get('max_class_presence_ratio', 0.4)
        cache_config = {
            "dataset_type": self.dataset_type,
            "data_path": normalize_path_config(self.data_path),
            "metadata_fingerprint": self._metadata_fingerprint(),
            "instruments": sorted(self.instruments),
            "file_types": sorted(self.file_types),
            "max_ratio": max_ratio,
            "total_tracks": total_tracks,
            "filter_frequent": filter_frequent,
        }

        if os.path.isfile(cache_path):
            if should_print:
                print(f"{cache_label} Loading class_to_tracks from cache")

            try:
                with open(cache_path, "r", encoding="utf8") as f:
                    cache = json.load(f)
            except Exception as e:
                cache = {}
                if should_print:
                    print(f"{cache_label} Cache unreadable ({e}), rebuilding")

            if (
                    cache.get("dataset_type") == cache_config["dataset_type"] and
                    cache.get("data_path") == cache_config["data_path"] and
                    cache.get("metadata_fingerprint") == cache_config["metadata_fingerprint"] and
                    cache.get("total_tracks") == cache_config["total_tracks"] and
                    cache.get("max_ratio") == cache_config["max_ratio"] and
                    cache.get("instruments") == cache_config["instruments"] and
                    cache.get("file_types") == cache_config["file_types"] and
                    cache.get("filter_frequent") == cache_config["filter_frequent"]
            ):
                self.class_to_tracks = cache["class_to_tracks"]
                self.available_classes = list(self.class_to_tracks.keys())

                if should_print:
                    print(
                        f"{cache_label} Loaded {len(self.available_classes)} classes from cache"
                    )
                return
            else:
                if should_print:
                    print(f"{cache_label} Cache invalid, rebuilding")

        class_to_tracks = {instr: [] for instr in self.instruments}

        track_iter = self.metadata
        if should_print:
            track_iter = tqdm(
                self.metadata,
                desc=f"{cache_label} Building class_to_tracks",
                total=total_tracks
            )

        for track_path, _ in track_iter:
            for instr in self.instruments:
                for ext in self.file_types:
                    path = f"{track_path}/{instr}.{ext}"
                    if os.path.isfile(path):
                        class_to_tracks[instr].append(track_path)
                        break

        filtered_class_to_tracks = {}

        for instr, tracks in class_to_tracks.items():
            count = len(tracks)
            ratio = count / total_tracks

            if count == 0:
                continue

            if filter_frequent and ratio > max_ratio:
                if should_print:
                    print(
                        f"{cache_label} Skip frequent stem '{instr}': "
                        f"{count}/{total_tracks} ({ratio:.1%})"
                    )
                continue

            filtered_class_to_tracks[instr] = tracks

        if len(filtered_class_to_tracks) == 0:
            raise RuntimeError(
                "No class-balanced stems are available. Check instruments, file_types, or max_class_presence_ratio."
            )

        self.class_to_tracks = filtered_class_to_tracks
        self.available_classes = list(filtered_class_to_tracks.keys())

        if should_print:
            print(f"{cache_label} Saving class_to_tracks cache")

        atomic_json_dump(
            {
                **cache_config,
                "class_to_tracks": filtered_class_to_tracks,
            },
            cache_path,
        )

        if should_print:
            print(
                f"Using {len(self.available_classes)} class-balanced stems "
                f"out of {len(self.instruments)} instruments"
            )


    def _sample_aligned_track(self):
        if self.class_balanced_stems:
            instr = random.choice(self.available_classes)
            track_path = random.choice(self.class_to_tracks[instr])
            track_length = self._track_length_by_path.get(track_path)
            if track_length is None:
                raise RuntimeError(f"Track length not found: {track_path}")
            return track_path, track_length

        return random.choice(self.metadata)

    def load_class_balanced_aligned(self):
        """
        1) Randomly choose instrument (class)
        2) Randomly choose track containing this instrument
        3) Load aligned chunk from this track
        """
        should_print = is_main_process()

        instr = random.choice(self.available_classes)
        track_path = random.choice(self.class_to_tracks[instr])

        track_length = self._track_length_by_path.get(track_path)

        if track_length is None:
            raise RuntimeError(f"Track length not found: {track_path}")

        if track_length >= self.chunk_size:
            offset = np.random.randint(track_length - self.chunk_size + 1)
        else:
            offset = None

        mix = self._load_mix_from_track(track_path, track_length, offset=offset)
        res = []
        active_stem_ids = []

        for idx, instr in enumerate(self.instruments):
            found = False
            for extension in self.file_types:
                path_to_audio_file = f"{track_path}/{instr}.{extension}"
                if os.path.isfile(path_to_audio_file):
                    try:
                        source = load_chunk(
                            path_to_audio_file,
                            track_length,
                            self.chunk_size,
                            offset=offset,
                            target_channels=self.target_channels,
                        )
                        active_stem_ids.append(idx)
                        found = True
                        break
                    except Exception as e:
                        if should_print:
                            print(e)

            if not found:
                source = self._zero_source()

            res.append(source)

        res = np.stack(res, axis=0)

        if mix is None:
            mix = np.sum(res, axis=0)

        if self.aug and self._can_change_stems():
            for i, instr in enumerate(self.instruments):
                res[i] = self.augm_data(res[i], instr)
            mix = np.sum(res, axis=0)

        return (
            torch.tensor(res, dtype=torch.float32),
            torch.tensor(mix, dtype=torch.float32),
            self._active_stem_mask(active_stem_ids)
        )


    def _chunks_cache_config(self) -> Dict[str, Any]:
        return {
            'dataset_type': self.dataset_type,
            'data_path': normalize_path_config(self.data_path),
            'metadata_fingerprint': self._metadata_fingerprint(),
            'chunk_size': self.chunk_size,
            'min_mean_abs': self.min_mean_abs,
            'instruments': sorted(self.instruments),
            'file_types': sorted(self.file_types),
            'target_channels': self.target_channels,
            'sample_rate': self.sample_rate,
            'strict_sample_rate': self.strict_sample_rate,
        }


    def _read_chunks_cache(self, chunks_cache_path: str, current_config: Dict[str, Any], should_print: bool):
        if os.path.exists(chunks_cache_path):
            try:
                with open(chunks_cache_path, 'rb') as f:
                    cached_chunks = pickle.load(f)
                cached_config = cached_chunks.get('config', {})
                if cached_config == current_config:
                    chunks_metadata = cached_chunks['chunks_metadata']
                    if self.verbose and should_print:
                        print(f'Loaded {len(chunks_metadata)} cached chunks from {chunks_cache_path}')
                    return chunks_metadata
                if self.verbose and should_print:
                    print('Config changed, recomputing chunks...')
                    print(f'Cached config: {cached_config}')
                    print(f'Current config: {current_config}')
            except Exception as e:
                if self.verbose and should_print:
                    print(f'Chunks cache corrupted ({e}), recomputing...')
        return None


    def _initialize_chunks_metadata(self):
        should_print = is_main_process()
        chunks_cache_path = self.metadata_path.replace('.pkl', '_chunks.pkl')
        current_config = self._chunks_cache_config()

        if dist.is_initialized():
            if dist.get_rank() == 0:
                self.chunks_metadata = self._read_chunks_cache(chunks_cache_path, current_config, should_print)
                if self.chunks_metadata is None:
                    self.chunks_metadata = self._precompute_and_cache_chunks(
                        chunks_cache_path, current_config)
            dist.barrier()
            if dist.get_rank() != 0:
                self.chunks_metadata = self._read_chunks_cache(chunks_cache_path, current_config, should_print)
                if self.chunks_metadata is None:
                    raise RuntimeError(
                        f'Chunks cache was not created by rank 0: {chunks_cache_path}'
                    )
        else:
            self.chunks_metadata = self._read_chunks_cache(chunks_cache_path, current_config, should_print)
            if self.chunks_metadata is None:
                self.chunks_metadata = self._precompute_and_cache_chunks(
                    chunks_cache_path, current_config)

        if self.verbose and should_print:
            print(f'Precomputed {len(self.chunks_metadata)} chunks')


    def _precompute_and_cache_chunks(self, cache_path, config):
        """Precompute all chunks and save to cache with config"""
        if self.dataset_type == 4 or self.dataset_type == 6:
            chunks_metadata = self._precompute_random_chunks()
        elif self.dataset_type == 5:
            chunks_metadata = self._precompute_chunks()
        else:
            raise ValueError('Only dataset type 4, 5, 6 can be precomputed')
        if len(chunks_metadata) == 0:
            raise RuntimeError(
                "No usable chunks were precomputed. Lower audio.min_mean_abs or check dataset files."
            )
        cache_data = {
            'chunks_metadata': chunks_metadata,
            'config': config
        }
        atomic_pickle_dump(cache_data, cache_path)

        return chunks_metadata


    def _precompute_chunks(self):
        """Precompute all chunks for dataset_type 5 with overlap 2 using multiprocessing"""
        should_print = (not dist.is_initialized() or dist.get_rank() == 0)

        tasks = []
        for track_path, track_length in self.metadata:
            if track_length < self.chunk_size:
                tasks.append((track_path, track_length, 0, track_length))
            else:
                step = self.chunk_size // 2
                num_chunks = (track_length - self.chunk_size) // step + 1
                for i in range(num_chunks):
                    offset = i * step
                    tasks.append((track_path, track_length, offset, self.chunk_size))

        if should_print:
            print(f"Total tasks to process: {len(tasks)}")

        if multiprocessing.cpu_count() > 1:
            chunks_metadata = self._process_tasks_parallel(tasks, should_print)
        else:
            chunks_metadata = self._process_tasks_sequential(tasks, should_print)

        if self.verbose and should_print:
            print(
                f'Created {len(chunks_metadata)} good chunks from {len(self.metadata)} tracks')

        return chunks_metadata

    def _precompute_random_chunks(self):
        """Precompute exact number of good chunks"""
        should_print = (not dist.is_initialized() or dist.get_rank() == 0)

        target_count = self.config.training.get('num_precompute_chunks', self.config.training.num_steps * self.batch_size * self.config.training.num_epochs)
        chunks_metadata = []

        if should_print:
            print(f"Generating exactly {target_count} good chunks...")

        max_batches = self.config.training.get('max_precompute_batches', max(10, int(target_count) * 20))
        batches_done = 0
        with tqdm(total=target_count, desc='Progress good chunks') as pbar:
            while len(chunks_metadata) < target_count:
                batches_done += 1
                if batches_done > max_batches:
                    raise RuntimeError(
                        f"Could not precompute {target_count} good chunks after {max_batches} batches. "
                        "Lower audio.min_mean_abs or inspect quiet/corrupt stems."
                    )
                batch_size = self.config.training.get('precompute_batch_for_chunks', 500)
                tasks = []
                need = target_count - len(chunks_metadata)
                for i in range(batch_size):
                    track_path, track_length = random.choice(self.metadata)
                    if track_length < self.chunk_size:
                        tasks.append((track_path, track_length, 0, track_length))
                    else:
                        offset = np.random.randint(track_length - self.chunk_size + 1)
                        tasks.append((track_path, track_length, offset, self.chunk_size))

                if multiprocessing.cpu_count() > 1:
                    good_chunks = self._process_tasks_parallel(tasks, False)
                else:
                    good_chunks = self._process_tasks_sequential(tasks, False)

                chunks_metadata.extend(good_chunks)
                pbar.update(min(len(good_chunks),need))

        chunks_metadata = chunks_metadata[:target_count]

        return chunks_metadata


    def _process_tasks_sequential(self, tasks, should_print):
        chunks_metadata = []

        pbar = tqdm(tasks, desc='Processing chunks') if should_print else tasks
        for task in pbar:
            track_path, track_length, offset, chunk_size = task
            if self._is_chunk_loud_enough(track_path, offset, chunk_size, track_length):
                chunks_metadata.append((track_path, offset))

        return chunks_metadata


    def _process_tasks_parallel(self, tasks, should_print):
        chunks_metadata = []
        if len(tasks) == 0:
            return chunks_metadata
        workers = int(self.config.training.get(
            'precompute_workers',
            max(1, multiprocessing.cpu_count() - 2)
        ))
        workers = max(1, min(workers, multiprocessing.cpu_count(), len(tasks)))

        with multiprocessing.Pool(processes=workers) as pool:

            worker_args = [(task, self.instruments, self.file_types, self.min_mean_abs, self.target_channels) for task in
                           tasks]

            results = []
            if should_print:
                with tqdm(total=len(tasks), desc='Processing chunks') as pbar:
                    for i, result in enumerate(pool.imap_unordered(process_chunk_worker, worker_args)):
                        results.append(result)
                        pbar.update(1)
            else:
                for result in pool.imap_unordered(process_chunk_worker, worker_args):
                    results.append(result)

            for result in results:
                track_path, offset, is_loud_enough = result
                if is_loud_enough:
                    chunks_metadata.append((track_path, offset))

        return chunks_metadata


    def _is_chunk_loud_enough(self, track_path, offset, chunk_size, track_length):

        try:
            for instrument in self.instruments:
                instrument_loud_enough = False
                for extension in self.file_types:
                    path_to_audio_file = track_path + '/{}.{}'.format(instrument, extension)
                    if os.path.isfile(path_to_audio_file):
                        try:
                            source = load_chunk(path_to_audio_file, length=track_length, offset=offset,
                                                chunk_size=chunk_size, target_channels=self.target_channels)
                            if np.abs(source).mean() >= self.min_mean_abs:
                                instrument_loud_enough = True
                                break
                        except Exception as e:
                            if not dist.is_initialized() or dist.get_rank() == 0:
                                print('Error loading: {} Path: {}'.format(e, path_to_audio_file))
                            return False

                if not instrument_loud_enough:
                    return False

            return True

        except Exception as e:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print('Error checking chunk loudness: {} Path: {}'.format(e, track_path))
            return False


    def read_from_metadata_cache(self, track_paths, instr=None):
        should_print = is_main_process()
        metadata = []
        if os.path.isfile(self.metadata_path):
            if self.verbose and should_print:
                print('Found metadata cache file: {}'.format(self.metadata_path))
            try:
                with open(self.metadata_path, 'rb') as f:
                    cache_data = pickle.load(f)
            except Exception as e:
                if should_print:
                    print(f'Metadata cache is unreadable ({e}); rebuilding.')
                return track_paths, metadata
        else:
            return track_paths, metadata

        old_metadata = self._extract_metadata_from_cache(cache_data)
        if old_metadata is None:
            if self.verbose and should_print:
                print('Metadata cache config/version mismatch; rebuilding.')
            return track_paths, metadata

        if instr:
            if not isinstance(old_metadata, dict) or instr not in old_metadata:
                return track_paths, metadata
            old_metadata = old_metadata[instr]

        # We will not re-read tracks existed in old metadata file
        track_paths_set = set(track_paths)
        for item in old_metadata:
            if not item or len(item) != 2:
                continue
            old_path, file_size = item
            if old_path in track_paths_set and file_size and file_size > 0:
                metadata.append((old_path, file_size))
                track_paths_set.remove(old_path)
        track_paths = sorted(track_paths_set)
        if len(metadata) > 0 and should_print:
            print('Old metadata was used for {} tracks.'.format(len(metadata)))
        return track_paths, metadata


    def get_metadata(self):
        read_metadata_procs = max(1, multiprocessing.cpu_count() - 2)
        should_print = is_main_process()
        if 'read_metadata_procs' in self.config['training']:
            read_metadata_procs = max(1, int(self.config['training']['read_metadata_procs']))

        if self.verbose and should_print:
            print(
                'Dataset type:', self.dataset_type,
                'Processes to use:', read_metadata_procs,
                '\nCollecting metadata for', str(self.data_path),
            )

        if self.dataset_type in [1, 4, 5, 6, 7]:  # Added type 7
            track_paths = []
            if type(self.data_path) == list:
                for tp in self.data_path:
                    tracks_for_folder = sorted(glob(tp + '/*'))
                    if len(tracks_for_folder) == 0 and should_print:
                        print('Warning: no tracks found in folder \'{}\'. Please check it!'.format(tp))
                    track_paths += tracks_for_folder
            else:
                track_paths += sorted(glob(self.data_path + '/*'))

            track_paths = [path for path in track_paths if os.path.basename(path)[0] != '.' and os.path.isdir(path)]
            track_paths, metadata = self.read_from_metadata_cache(track_paths, None)

            if read_metadata_procs <= 1:
                pbar = tqdm(track_paths) if should_print else track_paths
                for path in pbar:
                    result = get_track_set_length((
                        path,
                        self.instruments,
                        self.file_types,
                        self.dataset_type,
                        self.sample_rate,
                        self.strict_sample_rate,
                    ))
                    if result is not None:
                        metadata.append(result)
            else:
                with ThreadPoolExecutor(max_workers=read_metadata_procs) as executor:
                    futures = [
                        executor.submit(
                            get_track_set_length,
                            args
                        )
                        for args in zip(
                            track_paths,
                            itertools.repeat(self.instruments),
                            itertools.repeat(self.file_types),
                            itertools.repeat(self.dataset_type),
                            itertools.repeat(self.sample_rate),
                            itertools.repeat(self.strict_sample_rate),
                        )
                    ]

                    if should_print:
                        for f in tqdm(as_completed(futures), total=len(futures)):
                            result = f.result()
                            if result is not None:
                                metadata.append(result)
                    else:
                        for f in as_completed(futures):
                            result = f.result()
                            if result is not None:
                                metadata.append(result)

        elif self.dataset_type == 2:
            metadata = dict()
            for instr in self.instruments:
                metadata[instr] = []
                track_paths = []
                if type(self.data_path) == list:
                    for tp in self.data_path:
                        for extension in self.file_types:
                            track_paths += sorted(glob(tp + '/{}/*.{}'.format(instr, extension)))
                else:
                    for extension in self.file_types:
                        track_paths += sorted(glob(self.data_path + '/{}/*.{}'.format(instr, extension)))

                track_paths, metadata[instr] = self.read_from_metadata_cache(track_paths, instr)

                if read_metadata_procs <= 1:
                    pbar = tqdm(track_paths) if should_print else track_paths
                    for path in pbar:
                        result = get_track_length((path, self.sample_rate, self.strict_sample_rate))
                        if result is not None:
                            metadata[instr].append(result)
                else:
                    with multiprocessing.Pool(processes=read_metadata_procs) as p:
                        track_iter = p.imap(
                            get_track_length,
                            ((path, self.sample_rate, self.strict_sample_rate) for path in track_paths),
                        )
                        if should_print:
                            track_iter = tqdm(track_iter, total=len(track_paths))

                        for out in track_iter:
                            if out is not None:
                                metadata[instr].append(out)

        elif self.dataset_type == 3:
            import pandas as pd
            data_paths = self.data_path if isinstance(self.data_path, list) else [self.data_path]

            metadata = {instr: [] for instr in self.instruments}
            track_paths_by_instr = {instr: [] for instr in self.instruments}
            total_rows = 0
            skipped = 0
            for i in range(len(data_paths)):
                if self.verbose and should_print:
                    print('Reading tracks from: {}'.format(data_paths[i]))
                df = pd.read_csv(data_paths[i])
                total_rows += len(df)

                for instr in self.instruments:
                    part = df[df['instrum'] == instr].copy()
                    if should_print:
                        print('Tracks found for {}: {}'.format(instr, len(part)))
                    track_paths_by_instr[instr].extend(list(part['path'].values))

            for instr in self.instruments:
                track_paths = list(dict.fromkeys(track_paths_by_instr[instr]))
                track_paths, metadata[instr] = self.read_from_metadata_cache(track_paths, instr)

                pbar = tqdm(track_paths) if should_print else track_paths
                for path in pbar:
                    if not os.path.isfile(path):
                        if should_print:
                            print('Cant find track: {}'.format(path))
                        skipped += 1
                        continue
                    result = get_track_length((path, self.sample_rate, self.strict_sample_rate))
                    if result is None:
                        if should_print:
                            print('Problem with path: {}'.format(path))
                        skipped += 1
                        continue
                    metadata[instr].append(result)
            if skipped > 0 and should_print:
                print('Missing tracks: {} from {}'.format(skipped, total_rows))
        else:
            if should_print:
                print('Unknown dataset type: {}. Must be 1, 2, 3, 4, 5, 6 or 7'.format(self.dataset_type))
            raise ValueError('Unknown dataset type: {}. Must be 1, 2, 3, 4, 5, 6 or 7'.format(self.dataset_type))

        if isinstance(metadata, dict):
            empty_instruments = [instr for instr, values in metadata.items() if len(values) == 0]
            if empty_instruments:
                raise RuntimeError(f"No tracks found for instruments: {empty_instruments}")
        elif len(metadata) == 0:
            raise RuntimeError("No usable tracks found in dataset metadata.")

        # Save metadata
        atomic_pickle_dump(self._metadata_cache_payload(metadata), self.metadata_path)
        return metadata


    def _zero_source(self) -> np.ndarray:
        return np.zeros((self.target_channels, self.chunk_size), dtype=np.float32)


    def _find_stem_path(self, track_path: str, instr: str) -> Optional[str]:
        for extension in self.file_types:
            path_to_audio_file = f"{track_path}/{instr}.{extension}"
            if os.path.isfile(path_to_audio_file):
                return path_to_audio_file
        return None


    def _load_source_from_path(self, path: str, length: int, instr: str, offset=None) -> np.ndarray:
        should_print = is_main_process()
        try:
            return load_chunk(
                path,
                length,
                self.chunk_size,
                offset=offset,
                target_channels=self.target_channels,
            )
        except Exception as e:
            if should_print:
                print('Error: {} Path: {}'.format(e, path))
            return self._zero_source()


    def load_source(self, metadata, instr):
        should_print = is_main_process()
        source = self._zero_source()
        loud_enough = self.min_mean_abs <= 0

        for attempt in range(max(self.max_load_attempts, 1)):
            if self.dataset_type in [1, 4, 5, 6, 7]:
                track_path, track_length = random.choice(metadata)
                path_to_audio_file = self._find_stem_path(track_path, instr)
                if path_to_audio_file is None:
                    if should_print and attempt == 0:
                        print(f'Cant find stem "{instr}" in folder {track_path}; retrying.')
                    continue
                source = self._load_source_from_path(path_to_audio_file, track_length, instr)
            else:
                if instr not in metadata or len(metadata[instr]) == 0:
                    raise RuntimeError(f"No metadata entries for instrument '{instr}'")
                track_path, track_length = random.choice(metadata[instr])
                source = self._load_source_from_path(track_path, track_length, instr)

            loud_enough = np.abs(source).mean() >= self.min_mean_abs
            if loud_enough:
                break

        if not loud_enough and should_print:
            print(
                f"Could not find a loud enough chunk for '{instr}' after {self.max_load_attempts} attempts. "
                "Using the last sampled chunk."
            )
        if self.aug:
            source = self.augm_data(source, instr)
        return torch.tensor(source, dtype=torch.float32)


    def load_random_mix(self):
        res = []
        for instr in self.instruments:
            s1 = self.load_source(self.metadata, instr)
            # Mixup augmentation. Multiple mix of same type of stems
            if self.aug:
                if 'mixup' in self.config['augmentations']:
                    if self.config['augmentations'].mixup:
                        mixup = [s1]
                        for prob in self.config.augmentations.mixup_probs:
                            if random.uniform(0, 1) < prob:
                                s2 = self.load_source(self.metadata, instr)
                                mixup.append(s2)
                        mixup = torch.stack(mixup, dim=0)
                        loud_values = np.random.uniform(
                            low=self.config.augmentations.get(
                                'mixup_loudness_min',
                                self.config.augmentations.get('loudness_min', 1.0),
                            ),
                            high=self.config.augmentations.get(
                                'mixup_loudness_max',
                                self.config.augmentations.get('loudness_max', 1.0),
                            ),
                            size=(len(mixup),)
                        )
                        loud_values = torch.tensor(loud_values, dtype=torch.float32)
                        mixup *= loud_values[:, None, None]
                        s1 = mixup.mean(dim=0, dtype=torch.float32)
            res.append(s1)
        res = torch.stack(res)
        return res


    def _load_chunk_by_offset(self, track_path, offset, return_mix=False):
        """Load specific chunk by track path and offset"""
        res = []
        track_length = self._track_length_by_path.get(track_path)

        for instr in self.instruments:
            path_to_audio_file = self._find_stem_path(track_path, instr)
            if path_to_audio_file is None or track_length is None:
                source = self._zero_source()
            else:
                source = self._load_source_from_path(path_to_audio_file, track_length, instr, offset=offset)

            res.append(source)

        res = np.stack(res, axis=0)
        mix = None
        if return_mix and track_length is not None:
            mix = self._load_mix_from_track(track_path, track_length, offset=offset)
        if return_mix and mix is None:
            mix = res.sum(0)

        if self.aug and self._can_change_stems():
            for i, instr in enumerate(self.instruments):
                res[i] = self.augm_data(res[i], instr)
            if return_mix and not self._keep_original_mixture():
                mix = res.sum(0)

        if return_mix:
            return torch.tensor(res, dtype=torch.float32), torch.tensor(mix, dtype=torch.float32)
        return torch.tensor(res, dtype=torch.float32)


    def load_aligned_data(self):
        track_path, track_length = self._sample_aligned_track()
        should_print = is_main_process()
        attempts = 10
        while attempts:
            if track_length >= self.chunk_size:
                common_offset = np.random.randint(track_length - self.chunk_size + 1)
            else:
                common_offset = None
            res = []
            silent_chunks = 0
            for i in self.instruments:
                found = False
                for extension in self.file_types:
                    path_to_audio_file = f"{track_path}/{i}.{extension}"
                    if os.path.isfile(path_to_audio_file):
                        found = True
                        try:
                            source = load_chunk(
                                path_to_audio_file,
                                track_length,
                                self.chunk_size,
                                offset=common_offset,
                                target_channels=self.target_channels,
                            )
                        except Exception as e:
                            if should_print:
                                print(f"Error: {e} Path: {path_to_audio_file}")
                            source = self._zero_source()
                        break

                if not found:
                    source = self._zero_source()

                res.append(source)
                if np.abs(source).mean() < self.min_mean_abs:  # remove quiet chunks
                    silent_chunks += 1

            mix = self._load_mix_from_track(track_path, track_length, offset=common_offset)

            if silent_chunks == 0:
                break

            attempts -= 1
            if attempts <= 0 and should_print:
                print('Attempts max!', track_path)
            if common_offset is None:
                break

        try:
            res = np.stack(res, axis=0)
        except Exception as e:
            print('Error during stacking stems: {} Track Length: {} Track path: {}'.format(str(e), track_length,
                                                                                           track_path))
            res = np.zeros((len(self.instruments), self.target_channels, self.chunk_size), dtype=np.float32)
        if mix is None:
            mix = res.sum(0)
        if self.aug and self._can_change_stems():
            for i, instr in enumerate(self.instruments):
                res[i] = self.augm_data(res[i], instr)
            if not self.config.augmentations.get('keep_original_mixture', False):
                mix = res.sum(0)
        return torch.tensor(res, dtype=torch.float32), torch.tensor(mix, dtype=torch.float32)


    def augm_data(self, source, instr):
        # source.shape = (2, 261120) - first channels, second length
        source_shape = source.shape
        applied_augs = []
        augs = self._augmentation_config(instr)

        source = np.ascontiguousarray(source.astype(np.float32, copy=False))

        # Channel shuffle
        if 'channel_shuffle' in augs:
            if augs['channel_shuffle'] > 0:
                if random.uniform(0, 1) < augs['channel_shuffle']:
                    source = source[::-1].copy()
                    applied_augs.append('channel_shuffle')
        # Random inverse
        if 'random_inverse' in augs:
            if augs['random_inverse'] > 0:
                if random.uniform(0, 1) < augs['random_inverse']:
                    source = source[:, ::-1].copy()
                    applied_augs.append('random_inverse')
        # Random polarity (multiply -1)
        if 'random_polarity' in augs:
            if augs['random_polarity'] > 0:
                if random.uniform(0, 1) < augs['random_polarity']:
                    source = -source.copy()
                    applied_augs.append('random_polarity')
        # Random pitch shift
        if 'pitch_shift' in augs:
            if augs['pitch_shift'] > 0:
                if random.uniform(0, 1) < augs['pitch_shift']:
                    apply_aug = self._get_stem_aug(
                        instr,
                        'pitch_shift',
                        lambda: AU.PitchShift(
                            min_semitones=augs['pitch_shift_min_semitones'],
                            max_semitones=augs['pitch_shift_max_semitones'],
                            p=1.0,
                        ),
                    )
                    source = apply_aug(samples=source, sample_rate=self.sample_rate)
                    applied_augs.append('pitch_shift')
        # Random seven band parametric eq
        if 'seven_band_parametric_eq' in augs:
            if augs['seven_band_parametric_eq'] > 0:
                if random.uniform(0, 1) < augs['seven_band_parametric_eq']:
                    apply_aug = self._get_stem_aug(
                        instr,
                        'seven_band_parametric_eq',
                        lambda: AU.SevenBandParametricEQ(
                            min_gain_db=augs['seven_band_parametric_eq_min_gain_db'],
                            max_gain_db=augs['seven_band_parametric_eq_max_gain_db'],
                            p=1.0,
                        ),
                    )
                    source = apply_aug(samples=source, sample_rate=self.sample_rate)
                    applied_augs.append('seven_band_parametric_eq')
        # Random tanh distortion
        if 'tanh_distortion' in augs:
            if augs['tanh_distortion'] > 0:
                if random.uniform(0, 1) < augs['tanh_distortion']:
                    apply_aug = self._get_stem_aug(
                        instr,
                        'tanh_distortion',
                        lambda: AU.TanhDistortion(
                            min_distortion=augs['tanh_distortion_min'],
                            max_distortion=augs['tanh_distortion_max'],
                            p=1.0,
                        ),
                    )
                    source = apply_aug(samples=source, sample_rate=self.sample_rate)
                    applied_augs.append('tanh_distortion')
        # Random MP3 Compression
        if 'mp3_compression' in augs:
            if augs['mp3_compression'] > 0:
                if random.uniform(0, 1) < augs['mp3_compression']:
                    apply_aug = self._get_stem_aug(
                        instr,
                        'mp3_compression',
                        lambda: AU.Mp3Compression(
                            min_bitrate=augs['mp3_compression_min_bitrate'],
                            max_bitrate=augs['mp3_compression_max_bitrate'],
                            backend=augs['mp3_compression_backend'],
                            p=1.0,
                        ),
                    )
                    source = apply_aug(samples=source, sample_rate=self.sample_rate)
                    applied_augs.append('mp3_compression')
        # Random AddGaussianNoise
        if 'gaussian_noise' in augs:
            if augs['gaussian_noise'] > 0:
                if random.uniform(0, 1) < augs['gaussian_noise']:
                    apply_aug = self._get_stem_aug(
                        instr,
                        'gaussian_noise',
                        lambda: AU.AddGaussianNoise(
                            min_amplitude=augs['gaussian_noise_min_amplitude'],
                            max_amplitude=augs['gaussian_noise_max_amplitude'],
                            p=1.0,
                        ),
                    )
                    source = apply_aug(samples=source, sample_rate=self.sample_rate)
                    applied_augs.append('gaussian_noise')
        # Random TimeStretch
        if 'time_stretch' in augs:
            if augs['time_stretch'] > 0:
                if random.uniform(0, 1) < augs['time_stretch']:
                    apply_aug = self._get_stem_aug(
                        instr,
                        'time_stretch',
                        lambda: AU.TimeStretch(
                            min_rate=augs['time_stretch_min_rate'],
                            max_rate=augs['time_stretch_max_rate'],
                            leave_length_unchanged=True,
                            p=1.0,
                        ),
                    )
                    source = apply_aug(samples=source, sample_rate=self.sample_rate)
                    applied_augs.append('time_stretch')

        # Possible fix of shape
        if source_shape != source.shape:
            source = fix_audio_length(source, source_shape[-1])

        # Random Reverb
        if 'pedalboard_reverb' in augs:
            if augs['pedalboard_reverb'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_reverb']:
                    room_size = random.uniform(
                        augs['pedalboard_reverb_room_size_min'],
                        augs['pedalboard_reverb_room_size_max'],
                    )
                    damping = random.uniform(
                        augs['pedalboard_reverb_damping_min'],
                        augs['pedalboard_reverb_damping_max'],
                    )
                    wet_level = random.uniform(
                        augs['pedalboard_reverb_wet_level_min'],
                        augs['pedalboard_reverb_wet_level_max'],
                    )
                    dry_level = random.uniform(
                        augs['pedalboard_reverb_dry_level_min'],
                        augs['pedalboard_reverb_dry_level_max'],
                    )
                    width = random.uniform(
                        augs['pedalboard_reverb_width_min'],
                        augs['pedalboard_reverb_width_max'],
                    )
                    board = self._get_pedalboard(instr, 'pedalboard_reverb', PB.Reverb)
                    self._set_pedalboard_params(
                        board,
                        room_size=room_size,
                        damping=damping,
                        wet_level=wet_level,
                        dry_level=dry_level,
                        width=width,
                        freeze_mode=0.0,
                    )
                    source = board(source, self.sample_rate)
                    applied_augs.append('pedalboard_reverb')

        # Random Chorus
        if 'pedalboard_chorus' in augs:
            if augs['pedalboard_chorus'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_chorus']:
                    rate_hz = random.uniform(
                        augs['pedalboard_chorus_rate_hz_min'],
                        augs['pedalboard_chorus_rate_hz_max'],
                    )
                    depth = random.uniform(
                        augs['pedalboard_chorus_depth_min'],
                        augs['pedalboard_chorus_depth_max'],
                    )
                    centre_delay_ms = random.uniform(
                        augs['pedalboard_chorus_centre_delay_ms_min'],
                        augs['pedalboard_chorus_centre_delay_ms_max'],
                    )
                    feedback = random.uniform(
                        augs['pedalboard_chorus_feedback_min'],
                        augs['pedalboard_chorus_feedback_max'],
                    )
                    mix = random.uniform(
                        augs['pedalboard_chorus_mix_min'],
                        augs['pedalboard_chorus_mix_max'],
                    )
                    board = self._get_pedalboard(instr, 'pedalboard_chorus', PB.Chorus)
                    self._set_pedalboard_params(
                        board,
                        rate_hz=rate_hz,
                        depth=depth,
                        centre_delay_ms=centre_delay_ms,
                        feedback=feedback,
                        mix=mix,
                    )
                    source = board(source, self.sample_rate)
                    applied_augs.append('pedalboard_chorus')

        # Random Phazer
        if 'pedalboard_phazer' in augs:
            if augs['pedalboard_phazer'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_phazer']:
                    rate_hz = random.uniform(
                        augs['pedalboard_phazer_rate_hz_min'],
                        augs['pedalboard_phazer_rate_hz_max'],
                    )
                    depth = random.uniform(
                        augs['pedalboard_phazer_depth_min'],
                        augs['pedalboard_phazer_depth_max'],
                    )
                    centre_frequency_hz = random.uniform(
                        augs['pedalboard_phazer_centre_frequency_hz_min'],
                        augs['pedalboard_phazer_centre_frequency_hz_max'],
                    )
                    feedback = random.uniform(
                        augs['pedalboard_phazer_feedback_min'],
                        augs['pedalboard_phazer_feedback_max'],
                    )
                    mix = random.uniform(
                        augs['pedalboard_phazer_mix_min'],
                        augs['pedalboard_phazer_mix_max'],
                    )
                    board = self._get_pedalboard(instr, 'pedalboard_phazer', PB.Phaser)
                    self._set_pedalboard_params(
                        board,
                        rate_hz=rate_hz,
                        depth=depth,
                        centre_frequency_hz=centre_frequency_hz,
                        feedback=feedback,
                        mix=mix,
                    )
                    source = board(source, self.sample_rate)
                    applied_augs.append('pedalboard_phazer')

        # Random Distortion
        if 'pedalboard_distortion' in augs:
            if augs['pedalboard_distortion'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_distortion']:
                    drive_db = random.uniform(
                        augs['pedalboard_distortion_drive_db_min'],
                        augs['pedalboard_distortion_drive_db_max'],
                    )
                    board = self._get_pedalboard(instr, 'pedalboard_distortion', PB.Distortion)
                    self._set_pedalboard_params(board, drive_db=drive_db)
                    source = board(source, self.sample_rate)
                    applied_augs.append('pedalboard_distortion')

        # Random PitchShift
        if 'pedalboard_pitch_shift' in augs:
            if augs['pedalboard_pitch_shift'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_pitch_shift']:
                    semitones = random.uniform(
                        augs['pedalboard_pitch_shift_semitones_min'],
                        augs['pedalboard_pitch_shift_semitones_max'],
                    )
                    board = self._get_pedalboard(instr, 'pedalboard_pitch_shift', PB.PitchShift)
                    self._set_pedalboard_params(board, semitones=semitones)
                    source = board(source, self.sample_rate)
                    applied_augs.append('pedalboard_pitch_shift')

        # Random Resample
        if 'pedalboard_resample' in augs:
            if augs['pedalboard_resample'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_resample']:
                    target_sample_rate = random.uniform(
                        augs['pedalboard_resample_target_sample_rate_min'],
                        augs['pedalboard_resample_target_sample_rate_max'],
                    )
                    board = self._get_pedalboard(instr, 'pedalboard_resample', PB.Resample)
                    self._set_pedalboard_params(board, target_sample_rate=target_sample_rate)
                    source = board(source, self.sample_rate)
                    applied_augs.append('pedalboard_resample')

        # Random Bitcrash
        if 'pedalboard_bitcrash' in augs:
            if augs['pedalboard_bitcrash'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_bitcrash']:
                    bit_depth = random.uniform(
                        augs['pedalboard_bitcrash_bit_depth_min'],
                        augs['pedalboard_bitcrash_bit_depth_max'],
                    )
                    board = self._get_pedalboard(instr, 'pedalboard_bitcrash', PB.Bitcrush)
                    self._set_pedalboard_params(board, bit_depth=bit_depth)
                    source = board(source, self.sample_rate)
                    applied_augs.append('pedalboard_bitcrash')

        # Random MP3Compressor
        if 'pedalboard_mp3_compressor' in augs:
            if augs['pedalboard_mp3_compressor'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_mp3_compressor']:
                    vbr_quality = random.uniform(
                        augs['pedalboard_mp3_compressor_pedalboard_mp3_compressor_min'],
                        augs['pedalboard_mp3_compressor_pedalboard_mp3_compressor_max'],
                    )
                    board = self._get_pedalboard(instr, 'pedalboard_mp3_compressor', PB.MP3Compressor)
                    self._set_pedalboard_params(board, vbr_quality=vbr_quality)
                    source = board(source, self.sample_rate)
                    applied_augs.append('pedalboard_mp3_compressor')

        # print(applied_augs)
        return fix_audio_length(source, source_shape[-1])
