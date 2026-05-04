import argparse
import json
import pickle
import random

import numpy as np
import pytest
import soundfile as sf
import torch
from ml_collections import ConfigDict

import utils.dataset as dataset_module
from utils.dataset import (
    MSSDataset,
    dataloader_kwargs,
    get_track_set_length,
    metadata_fingerprint,
    music_collate_fn,
    normalize_path_config,
    rank_adjusted_seed,
)


def test_music_collate_fn_supports_active_stem_masks():
    random.seed(0)
    batch = [
        (
            torch.zeros(2, 2, 100),
            torch.zeros(2, 100),
            torch.tensor([True, False]),
        ),
        (
            torch.zeros(2, 2, 120),
            torch.zeros(2, 120),
            torch.tensor([False, True]),
        ),
    ]

    stems, mixes, active_stem_ids = music_collate_fn(batch, min_size=80, max_size=80)

    assert stems.shape == (2, 2, 2, 80)
    assert mixes.shape == (2, 2, 80)
    assert active_stem_ids.dtype == torch.bool
    assert active_stem_ids.tolist() == [[True, False], [False, True]]


def test_music_collate_fn_supports_standard_two_item_batches():
    random.seed(0)
    batch = [
        (torch.zeros(2, 2, 100), torch.zeros(2, 100)),
        (torch.zeros(2, 2, 120), torch.zeros(2, 120)),
    ]

    stems, mixes = music_collate_fn(batch, min_size=90, max_size=90)

    assert stems.shape == (2, 2, 2, 90)
    assert mixes.shape == (2, 2, 90)


def test_type6_metadata_uses_mixture_length_when_present(tmp_path):
    track_dir = tmp_path / "track"
    track_dir.mkdir()
    stem = np.zeros((100, 2), dtype=np.float32)
    mixture = np.zeros((80, 2), dtype=np.float32)
    sf.write(track_dir / "vocals.wav", stem, 44100)
    sf.write(track_dir / "mixture.wav", mixture, 44100)

    path, length = get_track_set_length((
        str(track_dir),
        ["vocals"],
        ["wav"],
        6,
        44100,
        True,
    ))

    assert path == str(track_dir)
    assert length == 80


def test_type7_metadata_checks_mixture_sample_rate(tmp_path):
    track_dir = tmp_path / "track"
    track_dir.mkdir()
    stem = np.zeros((100, 2), dtype=np.float32)
    mixture = np.zeros((100, 2), dtype=np.float32)
    sf.write(track_dir / "vocals.wav", stem, 44100)
    sf.write(track_dir / "mixture.wav", mixture, 48000)

    with pytest.raises(ValueError, match="Sample rate mismatch"):
        get_track_set_length((
            str(track_dir),
            ["vocals"],
            ["wav"],
            7,
            44100,
            True,
        ))


def test_type6_metadata_accepts_missing_mixture(tmp_path):
    track_dir = tmp_path / "track"
    track_dir.mkdir()
    stem = np.zeros((100, 2), dtype=np.float32)
    sf.write(track_dir / "vocals.wav", stem, 44100)

    path, length = get_track_set_length((
        str(track_dir),
        ["vocals"],
        ["wav"],
        6,
        44100,
        True,
    ))

    assert path == str(track_dir)
    assert length == 100


def test_metadata_fingerprint_changes_when_lengths_change(tmp_path):
    first = metadata_fingerprint([(tmp_path / "a", 100)])
    second = metadata_fingerprint([(tmp_path / "a", 101)])

    assert first != second


def test_rank_adjusted_seed_includes_rank():
    assert rank_adjusted_seed(123, rank=0) != rank_adjusted_seed(123, rank=1)


def test_dataloader_generator_seed_includes_ddp_rank(monkeypatch):
    monkeypatch.setattr(dataset_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dataset_module.dist, "get_rank", lambda: 2)
    args = argparse.Namespace(
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        seed=123,
    )

    kwargs = dataloader_kwargs(args)

    assert kwargs["generator"].initial_seed() == rank_adjusted_seed(123, rank=2)


def test_type7_loudness_augmentation_rebuilds_mixture():
    dataset = object.__new__(MSSDataset)
    dataset.dataset_type = 7
    dataset.aug = True

    config = ConfigDict()
    config.training = ConfigDict()
    config.training.instruments = ["vocals", "drums"]
    config.training.target_instrument = None
    config.augmentations = ConfigDict()
    config.augmentations.loudness = True
    config.augmentations.loudness_min = 2.0
    config.augmentations.loudness_max = 2.0
    dataset.config = config

    stems = torch.ones(2, 1, 4)
    original_mix = torch.zeros(1, 4)
    active_stem_ids = torch.tensor([True, True])
    dataset.load_class_balanced_aligned = lambda: (stems.clone(), original_mix.clone(), active_stem_ids)

    augmented_stems, mix, returned_active_stem_ids = dataset[0]

    assert torch.equal(mix, augmented_stems.sum(0))
    assert torch.equal(returned_active_stem_ids, active_stem_ids)


def _minimal_dataset_for_cache(tmp_path, *, data_path=None, metadata=None):
    dataset = object.__new__(MSSDataset)
    if metadata is not None:
        metadata = [(str(path), length) for path, length in metadata]
    dataset.dataset_type = 7
    dataset.data_path = str(data_path or tmp_path / "data")
    dataset.metadata = metadata or []
    dataset.metadata_path = str(tmp_path / "metadata_7.pkl")
    dataset.instruments = ["vocals"]
    dataset.file_types = ["wav"]
    dataset.chunk_size = 100
    dataset.min_mean_abs = 0.0
    dataset.target_channels = 2
    dataset.sample_rate = 44100
    dataset.strict_sample_rate = True
    dataset.verbose = False
    config = ConfigDict()
    config.training = ConfigDict()
    config.training.max_class_presence_ratio = 1.0
    dataset.config = config
    return dataset


def test_chunks_cache_config_changes_with_data_path_and_metadata(tmp_path):
    track_dir = tmp_path / "track"
    first = _minimal_dataset_for_cache(
        tmp_path,
        data_path=tmp_path / "data_a",
        metadata=[(track_dir, 100)],
    )
    second = _minimal_dataset_for_cache(
        tmp_path,
        data_path=tmp_path / "data_b",
        metadata=[(track_dir, 100)],
    )
    third = _minimal_dataset_for_cache(
        tmp_path,
        data_path=tmp_path / "data_a",
        metadata=[(track_dir, 101)],
    )

    first_config = first._chunks_cache_config()

    assert first_config["data_path"] == normalize_path_config(str(tmp_path / "data_a"))
    assert first_config != second._chunks_cache_config()
    assert first_config != third._chunks_cache_config()


def test_read_chunks_cache_rejects_stale_config(tmp_path):
    dataset = _minimal_dataset_for_cache(tmp_path, metadata=[(tmp_path / "track", 100)])
    cache_path = tmp_path / "metadata_7_chunks.pkl"
    current_config = dataset._chunks_cache_config()

    with open(cache_path, "wb") as out:
        pickle.dump({"config": {**current_config, "data_path": ["/stale"]}, "chunks_metadata": [("bad", 0)]}, out)
    assert dataset._read_chunks_cache(str(cache_path), current_config, should_print=False) is None

    expected_chunks = [(str(tmp_path / "track"), 0)]
    with open(cache_path, "wb") as out:
        pickle.dump({"config": current_config, "chunks_metadata": expected_chunks}, out)
    assert dataset._read_chunks_cache(str(cache_path), current_config, should_print=False) == expected_chunks


def test_class_to_tracks_cache_rebuilds_when_fingerprint_changes(tmp_path):
    track_dir = tmp_path / "track"
    track_dir.mkdir()
    (track_dir / "vocals.wav").write_bytes(b"placeholder")
    dataset = _minimal_dataset_for_cache(tmp_path, metadata=[(track_dir, 100)])
    cache_path = tmp_path / "metadata_7_class_to_tracks.json"
    stale_cache = {
        "dataset_type": 7,
        "data_path": normalize_path_config(dataset.data_path),
        "metadata_fingerprint": metadata_fingerprint([(track_dir, 101)]),
        "instruments": ["vocals"],
        "file_types": ["wav"],
        "max_ratio": 1.0,
        "total_tracks": 1,
        "filter_frequent": False,
        "class_to_tracks": {"vocals": ["/stale"]},
    }
    cache_path.write_text(json.dumps(stale_cache), encoding="utf8")

    dataset._build_class_to_tracks(filter_frequent=False)

    assert dataset.class_to_tracks == {"vocals": [str(track_dir)]}
