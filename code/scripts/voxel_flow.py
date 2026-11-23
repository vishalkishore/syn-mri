"""Train, sample, evaluate, or profile the separate native voxel-flow route."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.voxel_flow.config import load_config
from src.voxel_flow.config import workspace_path
from src.voxel_flow.data import CONTRASTS
from src.voxel_flow.data import VoxelReportDataset
from src.voxel_flow.data import collate_batch
from src.voxel_flow.data import read_cache_manifest
from src.voxel_flow.data import read_report_studies
from src.voxel_flow.data import validate_affine
from src.voxel_flow.evaluation import evaluate
from src.voxel_flow.evaluation import preview_slices
from src.voxel_flow.evaluation import save_generated
from src.voxel_flow.evaluation import save_preview
from src.voxel_flow.evaluation import select_cases
from src.voxel_flow.flow import make_noisy_state
from src.voxel_flow.flow import loss_needs_brain_mask
from src.voxel_flow.flow import predict_clean
from src.voxel_flow.flow import generate_from_noise
from src.voxel_flow.flow import voxel_flow_loss
from src.voxel_flow.model import ParallelVoxelFlow
from src.voxel_flow.training import batch_to_device
from src.voxel_flow.training import train
from src.voxel_flow.training import validate_checkpoint
from util.report_cache import ReportFeatureCache


def load_model(config, checkpoint, weights):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    validate_checkpoint(state, config)

    model = ParallelVoxelFlow(config)
    parameters = state["model"]

    if weights == "ema":
        parameters = {
            key.removeprefix("module."): value
            for key, value in state["ema_model"].items()
            if key.startswith("module.")
        }

    model.load_state_dict(parameters)
    model.to(config["device"]).eval()

    return model


def load_datasets(config):
    frame, metadata = read_cache_manifest(
        config["manifest"], config["report_token_cache"]
    )

    if (
        Path(config["manifest"]).resolve().parent
        != Path(config["voxel_cache_root"]).resolve()
    ):
        raise ValueError("manifest must belong to configured voxel_cache_root")

    if metadata["exploratory_pending_review"] != config["exploratory_pending_review"]:
        raise ValueError("Cache review policy differs from config")

    # One shared cache serves all datasets and loader workers; no re-encoding.
    features = ReportFeatureCache(config["report_token_cache"])

    dataset = VoxelReportDataset(
        frame,
        config["voxel_cache_root"],
        features,
        config["native_shape"],
        dataset_root=config["dataset_root"],
        include_brain_mask=loss_needs_brain_mask(config),
    )

    return frame, dataset, features


def sample_one_mri(args, config, model, output):
    if not args.study_uid or not args.geometry or not args.contrast:
        raise ValueError("sample requires --study-uid, --geometry, and --contrast")

    studies = read_report_studies(config["report_token_cache"])

    if args.study_uid not in set(studies.study_uid):
        raise ValueError("study-uid is absent from the existing report cache")

    geometry = nib.load(workspace_path(args.geometry, config))

    if geometry.shape != tuple(config["native_shape"]):
        raise ValueError("Reference header shape differs from native grid")

    validate_affine(geometry.affine)

    if nib.aff2axcodes(geometry.affine) != ("R", "A", "S") or not np.allclose(
        geometry.affine[:3, :3], np.diag(np.diag(geometry.affine[:3, :3])), atol=1e-5
    ):
        raise ValueError(
            "Reference geometry must follow the standardized anatomical RAS policy"
        )

    features = ReportFeatureCache(config["report_token_cache"])
    index = features.index[args.study_uid]

    tokens = features.tokens[
        int(features.offsets[index]) : int(features.offsets[index + 1])
    ]

    device = config["device"]

    condition = {
        "report_tokens": tokens[None].to(device),
        "report_mask": torch.ones(1, len(tokens), dtype=torch.bool, device=device),
        "contrast_id": torch.tensor([CONTRASTS.index(args.contrast)], device=device),
        "report_drop": torch.zeros(1, dtype=torch.bool, device=device),
    }

    affine = torch.from_numpy(geometry.affine)[None].to(device)
    generator = torch.Generator(device=device).manual_seed(config["evaluation_seed"])

    noise = (
        torch.randn(1, 1, *config["native_shape"], device=device, generator=generator)
        * config["source_noise_std"]
    )

    output.mkdir(parents=True, exist_ok=False)

    with torch.inference_mode():
        generated, calls = generate_from_noise(
            lambda x, t: predict_clean(model, x, t, condition, affine, config),
            noise,
            config["sampling_steps"],
        )

    array = generated[0, 0].cpu().numpy()

    save_generated(output / "sample.nii.gz", array, geometry.affine)

    save_preview(
        [("Generated native MRI", preview_slices(array))], output / "preview.png"
    )

    settings = {
        **config,
        "study_uid": args.study_uid,
        "contrast": args.contrast,
        "geometry_reference": str(args.geometry),
        "affine": geometry.affine.tolist(),
        "model_evaluations": calls,
        "weights": args.weights,
        "checkpoint": str(args.checkpoint),
        "intensity_units": "positive-brain-median normalized",
    }

    (output / "settings.json").write_text(json.dumps(settings, indent=2))

    print(
        json.dumps({"output_dir": str(output), "model_evaluations": calls}), flush=True
    )


def profile(config, output, report_length=256):
    if not torch.cuda.is_available() or torch.device(config["device"]).type != "cuda":
        raise ValueError("Native profiling requires CUDA")

    if report_length < 1:
        raise ValueError("report-length must be positive")

    output.mkdir(parents=True, exist_ok=False)

    device = config["device"]

    torch.manual_seed(config["seed"])

    model = ParallelVoxelFlow(config).to(device).train()

    if config.get("compile_cnn_blocks", False):
        model.compile_cnn_blocks()

    target = torch.randn(1, 1, *config["native_shape"], device=device)
    # Synthetic profiling geometry only; training always loads anatomical masks.
    brain_mask = None

    if loss_needs_brain_mask(config):
        brain_mask = torch.zeros_like(target, dtype=torch.bool)

        interior = tuple(
            slice(size // 4, 3 * size // 4) for size in config["native_shape"]
        )

        brain_mask[(slice(None), slice(None), *interior)] = True

    noise = torch.randn_like(target)
    times = torch.tensor([0.5], device=device)
    affine = torch.eye(4, dtype=torch.float64, device=device)[None]

    condition = {
        "report_tokens": torch.randn(1, report_length, 768, device=device),
        "report_mask": torch.ones(1, report_length, dtype=torch.bool, device=device),
        "contrast_id": torch.ones(1, dtype=torch.long, device=device),
        "report_drop": torch.zeros(1, dtype=torch.bool, device=device),
    }

    state = make_noisy_state(target, noise, times)

    def forward():
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=config["mixed_precision"] == "bf16"
        ):
            return model(state, times, condition, affine)

    # Warm up complete forward/backward, including checkpoint recomputation.
    warmup_loss, _ = voxel_flow_loss(
        forward(), target, times, config, brain_mask=brain_mask, affine=affine
    )

    warmup_loss.backward()

    del warmup_loss

    model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = perf_counter()
    outputs = forward()

    torch.cuda.synchronize()

    forward_seconds = perf_counter() - start

    loss, _ = voxel_flow_loss(
        outputs, target, times, config, brain_mask=brain_mask, affine=affine
    )

    start = perf_counter()

    loss.backward()

    torch.cuda.synchronize()

    backward_seconds = perf_counter() - start

    result = {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "native_shape": config["native_shape"],
        "report_tokens": report_length,
        "device": torch.cuda.get_device_name(torch.device(device)),
        "torch_version": str(torch.__version__),
        "forward_seconds": forward_seconds,
        "backward_including_checkpoint_recomputation_seconds": backward_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "loss": float(loss.detach()),
        "activation_checkpointing": config["activation_checkpointing"],
    }

    del outputs, loss

    model.zero_grad(set_to_none=True)
    model.eval()

    # Profiler records the kernels actually dispatched, not a backend preference.
    with (
        torch.inference_mode(),
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as trace,
    ):
        forward()

    result["attention_operators"] = [
        entry.key for entry in trace.key_averages() if "attention" in entry.key.lower()
    ]

    torch.cuda.synchronize()

    start = perf_counter()

    generated, calls = generate_from_noise(
        lambda x, t: predict_clean(model, x, t, condition, affine, config),
        noise,
        config["sampling_steps"],
    )

    torch.cuda.synchronize()

    result["sampling_seconds"] = perf_counter() - start
    result["sampling_model_evaluations"] = calls
    result["sampling_finite"] = bool(torch.isfinite(generated).all())

    (output / "profile.json").write_text(json.dumps(result, indent=2))

    (output / "config.yaml").write_text(yaml.safe_dump(config))

    print(json.dumps(result, indent=2), flush=True)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("command", choices=("train", "sample", "evaluate", "profile"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    parser.add_argument("--weights", choices=("raw", "ema"), default=None)
    parser.add_argument("--split", choices=("validation", "test"), default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--output-dir")
    parser.add_argument("--study-uid")
    parser.add_argument("--geometry")
    parser.add_argument("--contrast", choices=CONTRASTS)
    parser.add_argument("--report-length", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Training microbatch per GPU; accumulation preserves configured global "
            "batch"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total training epoch limit, including completed epochs",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small real-data epoch, validation, EMA evaluation, and preview",
    )

    args = parser.parse_args()

    torch.set_num_threads(4)

    config = load_config(args.config)

    if args.command != "train" and (
        args.batch_size is not None or args.epochs is not None
    ):
        parser.error("--batch-size and --epochs apply to training only")

    if args.command == "train":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))

        effective_batch = (
            config["batch_size"]
            * config.get("world_size", 1)
            * config["gradient_accumulation_steps"]
        )

        batch_size = (
            args.batch_size if args.batch_size is not None else config["batch_size"]
        )

        if batch_size < 1 or effective_batch % (batch_size * world_size):
            parser.error(
                "Per-GPU batch * world size must divide the configured effective batch"
            )

        config.update(
            world_size=world_size,
            batch_size=batch_size,
            gradient_accumulation_steps=effective_batch // (batch_size * world_size),
        )

        if args.epochs is not None:
            if args.epochs < 1:
                parser.error("--epochs must be positive")

            config["epochs"] = args.epochs

        if rank == 0:
            print(
                json.dumps(
                    {
                        "phase": "launch",
                        "world_size": world_size,
                        "batch_size_per_gpu": config["batch_size"],
                        "gradient_accumulation_steps": config[
                            "gradient_accumulation_steps"
                        ],
                        "effective_batch_size": effective_batch,
                    }
                ),
                flush=True,
            )

        if world_size > 1:
            local_rank = int(os.environ["LOCAL_RANK"])

            torch.cuda.set_device(local_rank)

            config["device"] = f"cuda:{local_rank}"

            # Rank 1 can wait while rank 0 writes full generation diagnostics.
            dist.init_process_group(
                "nccl",
                timeout=timedelta(hours=6),
                device_id=torch.device(config["device"]),
            )

        if args.smoke_test:
            if args.output_dir is None:
                parser.error("--smoke-test requires a fresh --output-dir")

            config.update(
                epochs=1,
                max_updates=None,
                checkpoint_every_updates=1,
                checkpoint_epochs=[1],
                validation_every_epochs=1,
                evaluation_every_epochs=1,
                preview_every_epochs=1,
                evaluation_samples=2,
                diversity_reports=1,
                diversity_samples_per_report=2,
            )

        if args.output_dir is not None:
            config["run_dir"] = str(workspace_path(args.output_dir, config))

        if rank == 0 and Path(config["run_dir"]).exists() and not args.resume:
            raise FileExistsError(f"Training run already exists: {config['run_dir']}")

        if any(
            value is not None
            for value in (
                args.steps,
                args.seed,
                args.num_samples,
                args.guidance_scale,
                args.split,
                args.weights,
            )
        ):
            parser.error("Sampling overrides apply to sample/evaluate/profile only")

        frame, dataset, features = load_datasets(config)

        train_rows, validation_rows = (
            frame[frame.split.eq("train")],
            frame[frame.split.eq("validation")],
        )

        if args.smoke_test:
            window = (
                config["batch_size"]
                * world_size
                * config["gradient_accumulation_steps"]
            )

            report_indices = train_rows.study_uid.map(features.index).to_numpy()
            report_lengths = np.diff(features.offsets.numpy())[report_indices]
            longest = int(np.argmax(report_lengths))

            train_rows = pd.concat(
                [
                    train_rows.iloc[[longest]],
                    train_rows.drop(train_rows.index[longest]).iloc[: window + 2],
                ]
            )

            validation_rows = select_cases(
                frame, "validation", 2, config["evaluation_seed"]
            )

            config["smoke_test"] = True
            config["smoke_train_cases"] = len(train_rows)
            config["smoke_validation_cases"] = len(validation_rows)
            config["smoke_max_report_tokens"] = int(report_lengths.max())

        if train_rows.empty or validation_rows.empty:
            raise ValueError(
                "Training needs nonempty patient-separated train/validation"
            )

        training_dataset = VoxelReportDataset(
            train_rows,
            config["voxel_cache_root"],
            features,
            dataset_root=config["dataset_root"],
            include_brain_mask=loss_needs_brain_mask(config),
        )

        validation_dataset = VoxelReportDataset(
            validation_rows,
            config["voxel_cache_root"],
            features,
            dataset_root=config["dataset_root"],
            include_brain_mask=loss_needs_brain_mask(config),
        )

        run_dir = Path(config["run_dir"])

        if args.resume:
            snapshot = pd.read_csv(
                run_dir / "manifest.csv", dtype=str, keep_default_na=False
            )

            if not frame.equals(snapshot):
                raise ValueError(
                    "Run cohort snapshot differs from current immutable cache"
                )

        def callbacks(epoch, raw, ema):
            if args.smoke_test:
                # Exercise the same checkpoint loading path as standalone evaluation.
                model = load_model(
                    config, run_dir / "last.pt", config["evaluation_weights"]
                )
            else:
                model = ema if config["evaluation_weights"] == "ema" else raw

            if epoch % config["evaluation_every_epochs"] == 0:
                evaluate(
                    model,
                    dataset,
                    frame,
                    config,
                    run_dir.parent / f"evaluation_epoch_{epoch:03d}",
                    weights=config["evaluation_weights"],
                )

            if epoch % config["preview_every_epochs"] == 0:
                batch = collate_batch([validation_dataset[0]])
                target, condition, affine = batch_to_device(batch, config["device"])

                generator = torch.Generator(device=config["device"]).manual_seed(
                    config["evaluation_seed"]
                )

                noise = (
                    torch.randn(
                        target.shape, device=config["device"], generator=generator
                    )
                    * config["source_noise_std"]
                )

                model.eval()

                result, calls = generate_from_noise(
                    lambda x, t: predict_clean(model, x, t, condition, affine, config),
                    noise,
                    config["sampling_steps"],
                )

                save_preview(
                    [
                        (
                            "Unpaired real reference",
                            preview_slices(target[0, 0].cpu().numpy()),
                        ),
                        ("Generated", preview_slices(result[0, 0].cpu().numpy())),
                    ],
                    run_dir / f"preview_epoch_{epoch:03d}.png",
                )

        # The trainer creates the run directory, preserving fresh-run protection.
        train(
            config,
            training_dataset,
            validation_dataset,
            resume=workspace_path(args.resume, config) if args.resume else None,
            epoch_callback=callbacks,
        )

        return

    args.weights = args.weights or config["evaluation_weights"]
    model = None

    if args.command in ("sample", "evaluate"):
        if not args.checkpoint:
            parser.error("--checkpoint is required")

        args.checkpoint = workspace_path(args.checkpoint, config)
        model = load_model(config, args.checkpoint, args.weights)

    for argument, key in (
        (args.steps, "sampling_steps"),
        (args.seed, "evaluation_seed"),
        (args.num_samples, "evaluation_samples"),
        (args.guidance_scale, "guidance_scale"),
    ):
        if argument is not None:
            config[key] = argument

    if config["sampling_steps"] < 1:
        raise ValueError("steps must be positive")

    default_output = Path(config["run_dir"]).parent / args.command
    output = workspace_path(args.output_dir or default_output, config)

    if args.command == "profile":
        profile(config, output, args.report_length or 256)
    elif args.command == "sample":
        sample_one_mri(args, config, model, output)
    else:
        frame, dataset, _ = load_datasets(config)
        config["checkpoint"] = str(args.checkpoint)

        metrics = evaluate(
            model,
            dataset,
            frame,
            config,
            output,
            args.split or "validation",
            args.num_samples,
            args.weights,
        )

        (output / "config.yaml").write_text(yaml.safe_dump(config))

        print(
            json.dumps(
                {
                    "output_dir": str(output),
                    "model_evaluations": metrics["model_evaluations"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
