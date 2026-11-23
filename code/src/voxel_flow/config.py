"""Configuration entry checks and workspace-relative storage paths."""

from pathlib import Path
import math
import os

import yaml

ARCHITECTURE = "parallel_voxel_flow_v2_3"
CACHE_FORMAT = "packed_nonzero_float32_v1"
NORMALIZATION = "positive_brain_median"
LOSS_DEFAULTS = {
    "voxel_loss_reduction": "whole_volume",
    "voxel_brain_weight": 0.5,
    "contrast_loss_region": "whole_volume",
    "sobel_loss_region": "whole_volume",
}

CODE_ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    config = {**LOSS_DEFAULTS, **config}

    validate_config(config)

    workspace = Path(config.get("workspace_root", ".."))

    if not workspace.is_absolute():
        workspace = CODE_ROOT / workspace

    workspace = workspace.resolve()
    config["workspace_root"] = str(workspace)

    if config.get("compile_cnn_blocks", False):
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR", str(workspace / "cache/torchinductor")
        )
        os.environ.setdefault("TRITON_CACHE_DIR", str(workspace / "cache/triton"))

    for key in (
        "dataset_root",
        "source_manifest",
        "manifest",
        "voxel_cache_root",
        "report_token_cache",
        "run_dir",
    ):
        config[key] = str(workspace_path(config[key], config))

    return config


def workspace_path(path, config):
    path = Path(path)

    if not path.is_absolute():
        path = Path(config["workspace_root"]) / path

    path = path.resolve()

    if not path.is_relative_to(Path(config["workspace_root"]).resolve()):
        raise ValueError(f"Data, caches, and runs must stay in the workspace: {path}")

    if path.is_relative_to(CODE_ROOT):
        raise ValueError(
            f"Data, caches, and runs must be outside the code folder: {path}"
        )

    return path


def validate_config(config):
    if config.get("voxel_loss_reduction", "whole_volume") not in (
        "whole_volume",
        "mask_balanced",
    ):
        raise ValueError("voxel_loss_reduction must be whole_volume or mask_balanced")

    weight = config.get("voxel_brain_weight", 0.5)

    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("voxel_brain_weight must be in [0,1]")

    for key in ("contrast_loss_region", "sobel_loss_region"):
        if config.get(key, "whole_volume") not in ("whole_volume", "brain"):
            raise ValueError(f"{key} must be whole_volume or brain")

    if type(config.get("world_size", 1)) is not int or config.get("world_size", 1) < 1:
        raise ValueError("world_size must be a positive integer")

    validation_batch_size = config.get("validation_batch_size", config["batch_size"])

    if type(validation_batch_size) is not int or validation_batch_size < 1:
        raise ValueError("validation_batch_size must be a positive integer")

    if type(config.get("compile_cnn_blocks", False)) is not bool:
        raise ValueError("compile_cnn_blocks must be a boolean")

    if type(config.get("log_gradient_diagnostics", False)) is not bool:
        raise ValueError("log_gradient_diagnostics must be a boolean")

    required = {
        "architecture": ARCHITECTURE,
        "voxel_cache_format": CACHE_FORMAT,
        "voxel_normalization": NORMALIZATION,
        "native_shape": [240, 240, 168],
        "z_padding": [4, 4],
        "local_pack_factor": 4,
        "local_width": 96,
        "global_width": 384,
        "stages": 4,
        "blocks_per_stage": 2,
        "global_heads": 6,
        "global_rope_axis_dims": [20, 20, 24],
        "local_heads": 4,
        "local_expansion": 2,
        "global_ff_hidden": 1024,
        "global_decoder_channels": [128, 48, 16, 1],
        "local_decoder_channels": [32, 4],
        "condition_width": 384,
        "time_embedding_width": 256,
        "time_mlp_hidden": 1536,
        "spacing_mlp_hidden": 128,
        "attention_dropout": 0.0,
        "contrast_conditioning": True,
        "structure_feedback": True,
        "prediction_type": "clean",
        "flow_schedule": "linear",
        "timestep_sampling": "uniform",
        "loss_type": "voxel_velocity_contrast",
        "sampler": "heun_last_euler",
        "preload_cache": False,
    }

    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"Revision 2.3 requires {key}={value!r}")

    legacy = {
        "mask_aware",
        "mask_loss_weights",
        "normalization_file",
        "band_mean",
        "band_scale",
        "stage1",
        "label_schema",
    }

    if legacy.intersection(config):
        raise ValueError("Legacy Haar normalization/loss fields are not valid here")

    for key in (
        "batch_size",
        "gradient_accumulation_steps",
        "epochs",
        "sampling_steps",
        "bridge_query_chunk_size",
        "report_query_chunk_size",
        "checkpoint_every_updates",
        "validation_every_epochs",
        "preview_every_epochs",
        "evaluation_every_epochs",
    ):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")

    if type(config["num_workers"]) is not int or config["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")

    for key in (
        "source_noise_std",
        "training_denominator_floor",
        "learning_rate",
        "rope_coordinate_scale_mm",
        "absolute_coordinate_scale_mm",
        "rms_epsilon",
        "gradient_clip_norm",
    ):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")

    for key in ("contrast_loss_weight", "global_auxiliary_weight", "weight_decay"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")

    if (
        not math.isfinite(config.get("sobel_loss_weight", 0))
        or config.get("sobel_loss_weight", 0) < 0
    ):
        raise ValueError("sobel_loss_weight must be finite and nonnegative")

    if not 0 <= config.get("sobel_time_start", 0.3) < 1:
        raise ValueError("sobel_time_start must be in [0,1)")

    if (
        not math.isfinite(config.get("sobel_time_power", 1.5))
        or config.get("sobel_time_power", 1.5) <= 0
    ):
        raise ValueError("sobel_time_power must be finite and positive")

    for key in ("ema_decay", "report_semantic_dropout"):
        if not 0 <= config[key] < 1:
            raise ValueError(f"{key} must be in [0,1)")

    if not math.isfinite(config["guidance_scale"]):
        raise ValueError("guidance_scale must be finite")

    if config["evaluation_weights"] not in ("raw", "ema"):
        raise ValueError("evaluation_weights must be raw or ema")

    for key in ("time_frequency_base", "rope_base", "time_scale"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")

    if config["time_frequency_base"] <= 1 or config["rope_base"] <= 1:
        raise ValueError("Fourier/RoPE bases must exceed one")

    if len(config["adam_betas"]) != 2 or any(
        not 0 <= beta < 1 for beta in config["adam_betas"]
    ):
        raise ValueError("Adam betas must be two values in [0,1)")

    if any(
        not math.isfinite(value) or value <= 0
        for value in config["near_zero_thresholds"]
    ):
        raise ValueError("Near-zero thresholds must be finite and positive")

    if config["mixed_precision"] not in ("bf16", "fp32"):
        raise ValueError("mixed_precision must be bf16 or fp32")

    if config["evaluation_split"] != "validation":
        raise ValueError("Test evaluation requires explicit --split test")

    if config["max_updates"] is not None and (
        type(config["max_updates"]) is not int or config["max_updates"] < 1
    ):
        raise ValueError("max_updates must be a positive integer or null")

    if not config["contrast_offsets"] or any(
        type(x) is not int or not 0 < x < min(config["native_shape"])
        for x in config["contrast_offsets"]
    ):
        raise ValueError("Invalid contrast offsets")
