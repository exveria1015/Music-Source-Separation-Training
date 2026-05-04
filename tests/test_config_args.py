from pathlib import Path

from utils.settings import apply_config_args, load_config, parse_args_train


ROOT_DIR = Path(__file__).resolve().parents[1]
MOISES_LIGHT_CONFIG = ROOT_DIR / "configs" / "exveria1015" / "moises_light_bus8_fullband_wide_fromscratch.yaml"


def test_config_training_values_fill_unset_args(tmp_path):
    args = parse_args_train({
        "model_type": "moises_light",
        "config_path": str(MOISES_LIGHT_CONFIG),
        "results_path": str(tmp_path),
    })
    config = load_config(args.model_type, args.config_path)
    args.model_type = config.training.model_type

    args = apply_config_args(args, config, mode="train")

    assert args.dataset_type == 4
    assert args.metrics == ["sdr", "si_sdr", "bleedless", "fullness"]
    assert args.metric_for_scheduler == "si_sdr"
    assert args.num_workers == 8
    assert args.pin_memory is True
    assert args.persistent_workers is True
    assert args.prefetch_factor == 4


def test_explicit_args_override_config_training_values(tmp_path):
    args = parse_args_train({
        "model_type": "moises_light",
        "config_path": str(MOISES_LIGHT_CONFIG),
        "results_path": str(tmp_path),
        "dataset_type": 1,
        "metrics": ["sdr"],
        "metric_for_scheduler": "sdr",
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
    })
    config = load_config(args.model_type, args.config_path)
    args.model_type = config.training.model_type

    args = apply_config_args(args, config, mode="train")

    assert args.dataset_type == 1
    assert args.metrics == ["sdr"]
    assert args.metric_for_scheduler == "sdr"
    assert args.num_workers == 0
    assert args.pin_memory is False
    assert args.persistent_workers is False
    assert args.prefetch_factor is None
