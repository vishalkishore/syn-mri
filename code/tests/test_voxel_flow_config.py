from pathlib import Path

import pytest
import yaml

from src.voxel_flow.config import validate_config


@pytest.mark.parametrize(
    "field,value",
    [
        ("local_width", 48),
        ("mask_aware", True),
        ("sampling_steps", 0),
        ("report_semantic_dropout", 1.0),
        ("voxel_normalization", "full_mean_std"),
        ("architecture", "parallel_voxel_flow_v2_2"),
        ("local_decoder_channels", [32, 16]),
        ("world_size", 0),
        ("world_size", True),
        ("validation_batch_size", 0),
        ("validation_batch_size", True),
    ],
)
def test_invalid_config_fails_at_entry(field, value):
    path = Path(__file__).resolve().parents[1] / "configs/voxel_flow.yaml"

    config = yaml.safe_load(path.read_text())
    config[field] = value

    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize(
    "name",
    ["voxel_flow.yaml"],
)
def test_phase_decoder_configs_are_valid(name):
    path = Path(__file__).resolve().parents[1] / "configs" / name
    config = yaml.safe_load(path.read_text())

    validate_config(config)
