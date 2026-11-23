"""Generate the validation cohort and extract MedicalNet features on each GPU.

Launch with torchrun. Workers are independent inference shards, without DDP.
Each scan has a fixed seed independent of shard count and batch composition.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.voxel_flow.config import load_config
from src.voxel_flow.data import VoxelReportDataset
from src.voxel_flow.data import collate_batch
from src.voxel_flow.data import read_cache_manifest
from src.voxel_flow.evaluation import save_generated
from src.voxel_flow.flow import predict_clean
from src.voxel_flow.flow import generate_from_noise
from src.voxel_flow.model import ParallelVoxelFlow
from src.voxel_flow.training import batch_to_device
from src.voxel_flow.training import validate_checkpoint
from util import fid as volume_fid
from util.report_cache import ReportFeatureCache


def compute_fid_from_features(output):
    """Require exactly one real/generated feature pair for every validation row."""

    parts = []

    for path in sorted((output / "features").glob("*.npz")):
        with np.load(path) as data:
            parts.append((data["indices"], data["real"], data["generated"]))

    indices = np.concatenate([part[0] for part in parts])
    order = np.argsort(indices)
    expected = json.loads((output / "settings.json").read_text())["scans_per_group"]

    if not np.array_equal(indices[order], np.arange(expected)):
        raise ValueError(f"Features must cover all {expected} scans exactly once")

    real = np.concatenate([part[1] for part in parts])[order]
    generated = np.concatenate([part[2] for part in parts])[order]

    np.savez(
        output / "features.npz", indices=indices[order], real=real, generated=generated
    )

    real_mean, real_covariance = volume_fid.feature_mean_and_covariance(real)
    generated_mean, generated_covariance = volume_fid.feature_mean_and_covariance(
        generated
    )

    score = volume_fid.calculate_frechet_distance(
        real_mean,
        real_covariance,
        generated_mean,
        generated_covariance,
    )

    if not np.isfinite(score) or score < -1e-6:
        raise ValueError(f"Invalid Frechet distance: {score}")

    settings = json.loads((output / "settings.json").read_text())

    result = {
        "medicalnet_frechet_distance": score,
        "medicalnet_frechet_distance_display_x1000": 1000 * score,
        "real_scans": len(real),
        "generated_scans": len(generated),
        "feature_dimensions": real.shape[1],
        "feature_batch_sizes": sorted({len(part[0]) for part in parts}),
        "squared_feature_mean_distance": float(
            np.square(real_mean - generated_mean).sum()
        ),
        "settings": settings,
    }

    (output / "metrics.json").write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2), flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/voxel_flow.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--native", action="store_true")

    args = parser.parse_args()

    if args.aggregate:
        compute_fid_from_features(args.output)

        return

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))

    torch.set_num_threads(4)
    torch.cuda.set_device(rank)

    config = load_config(args.config)
    config["device"] = f"cuda:{rank}"
    checkpoint = args.checkpoint or Path(config["run_dir"]) / "epoch_200.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    validate_checkpoint(state, config)

    model = ParallelVoxelFlow(config)

    model.load_state_dict(
        {
            key.removeprefix("module."): value
            for key, value in state["ema_model"].items()
            if key.startswith("module.")
        }
    )

    model.to(config["device"]).eval().requires_grad_(False)

    updates = state["optimizer_updates"]

    del state

    extractor = volume_fid.get_feature_extractor(config["device"])
    frame, _ = read_cache_manifest(config["manifest"], config["report_token_cache"])
    frame = frame.loc[frame.split.eq("validation")].reset_index(drop=True)

    if len(frame) < 2:
        raise ValueError(f"FID needs at least 2 validation scans, found {len(frame)}")

    output = args.output

    (output / "generated").mkdir(parents=True, exist_ok=True)

    (output / "features").mkdir(exist_ok=True)

    if rank == 0:
        frame.to_csv(output / "selected_cases.csv", index=False)

        settings = {
            "metric": "3D MedicalNet ResNet50 Frechet feature distance",
            "checkpoint": str(checkpoint),
            "optimizer_updates": updates,
            "weights": "ema",
            "scans_per_group": len(frame),
            "unique_validation_patients": int(frame.patient_uid.nunique()),
            "sampling_steps": 16,
            "guidance_scale": config["guidance_scale"],
            "seed_per_case": "42 + validation row index",
            "extractor": "TencentMedicalNet/MedicalNet-Resnet50/resnet_50_23dataset.pth",
            "preprocessing": (
                "util.fid.preprocess_volume: per-volume 0.001/0.999 "
                "quantile clipping and min-max, centered 256^3 padding, "
                "average pool to 128^3"
            ),
            "axis_order": (
                "native X,Y,Z, identical for real and generated; no physical resampling"
            ),
            "limitations": (
                "Scan-level exploratory score; repeated patients and the "
                "sample count limit precision. Per-volume scaling hides "
                "intensity calibration errors. Not standard 2D Inception "
                "FID."
            ),
        }

        if args.native:
            settings.update(
                preprocessing=(
                    "Paper-based VoxStruct3D: closest RAS, positive-voxel "
                    "foreground, quantiles 0.10/0.99, linear [-1,1], center "
                    "crop/pad to 240x240x168; no interpolation"
                ),
                limitations=(
                    "VoxStruct3D evaluator preprocessing code is unpublished; "
                    "positive-voxel foreground and globally scaled existing "
                    "background are explicit assumptions. Scan-level cohort "
                    "has repeated patients."
                ),
            )

        (output / "settings.json").write_text(json.dumps(settings, indent=2))

    features = ReportFeatureCache(config["report_token_cache"], memory_map=True)
    dataset = VoxelReportDataset(
        frame, config["voxel_cache_root"], features, config["native_shape"]
    )
    indices = list(range(rank, len(dataset), world))
    completed = set()

    for path in (output / "features").glob(f"rank_{rank}_*.npz"):
        with np.load(path) as data:
            completed.update(data["indices"].tolist())

    indices = [index for index in indices if index not in completed]
    start = perf_counter()

    for offset in range(0, len(indices), args.batch_size):
        selected = indices[offset : offset + args.batch_size]
        feature_path = output / "features" / f"rank_{rank}_case_{selected[0]:04d}.npz"
        items = [dataset[index] for index in selected]
        target, condition, affine = batch_to_device(
            collate_batch(items), config["device"]
        )
        condition["report_drop"] = torch.zeros(
            len(items), dtype=torch.bool, device=target.device
        )

        noise = (
            torch.stack(
                [
                    torch.randn(
                        target.shape[1:],
                        device=target.device,
                        generator=torch.Generator(device=target.device).manual_seed(
                            42 + index
                        ),
                    )
                    for index in selected
                ]
            )
            * config["source_noise_std"]
        )

        generated, _ = generate_from_noise(
            lambda x, t: predict_clean(model, x, t, condition, affine, config),
            noise,
            steps=16,
        )

        torch.cuda.synchronize()

        print(
            json.dumps(
                {
                    "phase": "sampling_complete",
                    "rank": rank,
                    "batch_first_case": selected[0],
                    "seconds_since_start": round(perf_counter() - start, 1),
                }
            ),
            flush=True,
        )

        if not torch.isfinite(generated).all():
            raise ValueError("Nonfinite generated MRI")

        real_features = []
        generated_features = []

        for position, index in enumerate(selected):
            real_volume = target[position, 0].cpu().numpy()
            generated_volume = generated[position, 0].cpu().numpy()

            save_generated(
                output / "generated" / f"case_{index:03d}.nii.gz",
                generated_volume,
                items[position]["affine"].numpy(),
            )

            for volume, destination in (
                (real_volume, real_features),
                (generated_volume, generated_features),
            ):
                if args.native:
                    image = volume_fid.preprocess_native_volume(
                        volume, items[position]["affine"].numpy()
                    )
                else:
                    image = volume_fid.preprocess_volume(volume)

                image = image.unsqueeze(0).to(config["device"])

                destination.append(extractor(image).float().cpu().numpy()[0])

        np.savez(
            feature_path,
            indices=np.asarray(selected),
            real=np.asarray(real_features, dtype=np.float64),
            generated=np.asarray(generated_features, dtype=np.float64),
        )

        print(
            json.dumps(
                {
                    "rank": rank,
                    "completed": len(completed) + offset + len(selected),
                    "total": len(indices) + len(completed),
                    "seconds": round(perf_counter() - start, 1),
                }
            ),
            flush=True,
        )

    print(f"DONE rank {rank}", flush=True)


if __name__ == "__main__":
    with (
        torch.backends.cudnn.flags(enabled=True, benchmark=False),
        torch.compiler.set_stance("force_eager"),
    ):
        main()
