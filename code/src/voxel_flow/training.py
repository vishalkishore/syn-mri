"""Single-device or DDP training with update-boundary checkpoints."""

from contextlib import nullcontext
from datetime import timedelta
import json
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim.swa_utils import AveragedModel
from torch.optim.swa_utils import get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import yaml

from src.voxel_flow.config import ARCHITECTURE
from src.voxel_flow.config import LOSS_DEFAULTS
from src.voxel_flow.data import collate_batch
from src.voxel_flow.flow import make_noisy_state
from src.voxel_flow.flow import voxel_flow_loss
from src.voxel_flow.model import ParallelVoxelFlow


def batch_to_device(batch, device):
    return (
        batch["volume"].to(device, non_blocking=True),
        {
            key: value.to(device, non_blocking=True)
            for key, value in batch["condition"].items()
        },
        batch["affine"].to(device, non_blocking=True),
    )


def make_loader(dataset, config, generator, indices=None):
    if indices is not None:
        dataset = Subset(dataset, indices.tolist())

    options = {}

    if config["num_workers"]:
        options.update(
            prefetch_factor=config["prefetch_factor"],
            persistent_workers=config["persistent_workers"],
        )

        if config.get("world_size", 1) > 1:
            options["multiprocessing_context"] = "spawn"

    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        collate_fn=collate_batch,
        generator=generator,
        **options,
    )


def cache_fingerprint(config):
    return {
        key: config[key]
        for key in (
            "voxel_cache_format",
            "voxel_normalization",
            "native_shape",
            "manifest",
            "report_token_cache",
        )
    }


@torch.no_grad()
def gradient_diagnostics(model, times, denominator_floor):
    """Measure unclipped parameter gradients and the window's time weighting."""

    groups = {}

    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            group = name.split(".")[0]

            groups.setdefault(group, []).append(
                torch.linalg.vector_norm(parameter.grad.detach(), dtype=torch.float32)
            )

    norms = (
        torch.stack(
            [
                torch.linalg.vector_norm(torch.stack(values))
                for values in groups.values()
            ]
        )
        .cpu()
        .tolist()
    )

    times = torch.cat(times).detach().float().cpu()
    weights = (1 - times).clamp_min(denominator_floor).reciprocal().square()
    edges = (0.0, 0.5, 0.9, 0.95, 1.0)

    return {
        "gradient_norm_by_module": dict(zip(groups, norms)),
        "time_min": float(times.min()),
        "time_max": float(times.max()),
        "mean_inverse_denominator_squared": float(weights.mean()),
        "time_bin_counts": {
            f"{left}-{right}": int(((times >= left) & (times < right)).sum())
            for left, right in zip(edges, edges[1:])
        },
    }


def validate_checkpoint(state, config):
    if state.get("architecture") != ARCHITECTURE:
        raise ValueError("Checkpoint is not parallel voxel flow revision 2.3")

    if state.get("cache_identity") != cache_fingerprint(config):
        raise ValueError("Checkpoint cache normalization/geometry/cohort differs")

    previous = state["config"]

    previous_batch = (
        previous["batch_size"]
        * state.get("world_size", 1)
        * previous["gradient_accumulation_steps"]
    )

    current_batch = (
        config["batch_size"]
        * config.get("world_size", 1)
        * config["gradient_accumulation_steps"]
    )

    if previous_batch != current_batch:
        raise ValueError("Resume must preserve the effective optimizer batch size")

    # Runtime/output controls may change on resume. Model and scientific settings
    # must not silently change the experiment at a checkpoint boundary.
    runtime = {
        "world_size",
        "batch_size",
        "gradient_accumulation_steps",
        "max_updates",
        "epochs",
        "device",
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "pin_memory",
        "run_dir",
        "workspace_root",
        "checkpoint_every_updates",
        "checkpoint_epochs",
        "validation_every_epochs",
        "preview_every_epochs",
        "evaluation_every_epochs",
        "cudnn_benchmark",
        "sampling_steps",
        "guidance_scale",
        "evaluation_weights",
        "evaluation_samples",
        "evaluation_seed",
        "diversity_reports",
        "diversity_samples_per_report",
        "near_zero_thresholds",
        "compile_cnn_blocks",
        "log_gradient_diagnostics",
        "validation_batch_size",
    }

    changed = [
        key
        for key, value in state["config"].items()
        if key not in runtime and config.get(key) != value
    ]

    changed.extend(
        key
        for key, default in LOSS_DEFAULTS.items()
        if key not in previous and config.get(key, default) != default
    )

    if changed:
        raise ValueError(f"Resume configuration differs: {changed}")


@torch.inference_mode()
@torch.backends.cudnn.flags(enabled=True, benchmark=False)
@torch.compiler.set_stance("force_eager")
def compute_validation_loss(model, dataset, config):
    model.eval()

    device = config["device"]
    generator = torch.Generator(device=device).manual_seed(config["seed"] + 100)
    worker_generator = torch.Generator().manual_seed(config["seed"] + 101)
    names = ["total", "voxel", "contrast", "global"]

    if config.get("voxel_loss_reduction", "whole_volume") == "mask_balanced":
        names.extend(["voxel_brain", "voxel_outside"])

    if config.get("sobel_loss_weight", 0) > 0:
        names.append("sobel")

    totals = {name: 0.0 for name in names}

    bins = [{"count": 0, **{name: 0.0 for name in names}} for _ in range(5)]

    count = 0
    started = perf_counter()

    print(json.dumps({"phase": "validation_start", "cases": len(dataset)}), flush=True)

    loader_config = dict(
        config, batch_size=config.get("validation_batch_size", config["batch_size"])
    )

    for batch in make_loader(dataset, loader_config, worker_generator):
        target, condition, affine = batch_to_device(batch, device)
        brain_mask = batch.get("brain_mask")

        if brain_mask is not None:
            brain_mask = brain_mask.to(device, non_blocking=True)

        # Draw each case's random noise and time in a fixed order, so the
        # validation result does not depend on the batch size.
        noise = torch.empty_like(target)
        times = torch.empty(len(target), device=device)

        for i in range(len(target)):
            torch.randn(
                target[i : i + 1].shape,
                device=device,
                generator=generator,
                out=noise[i : i + 1],
            )

            torch.rand(1, device=device, generator=generator, out=times[i : i + 1])

        noise.mul_(config["source_noise_std"])

        condition["report_drop"] = torch.zeros(
            len(target), dtype=torch.bool, device=device
        )

        with torch.autocast(
            device_type=target.device.type,
            dtype=torch.bfloat16,
            enabled=config["mixed_precision"] == "bf16",
        ):
            outputs = model(
                make_noisy_state(target, noise, times), times, condition, affine
            )

        # Keep equal case weights and the existing timestep-bin definitions.
        for i in range(len(target)):
            case_outputs = {
                key: outputs[key][i : i + 1] for key in ("clean", "global_clean")
            }

            loss, components = voxel_flow_loss(
                case_outputs,
                target[i : i + 1],
                times[i : i + 1],
                config,
                brain_mask=brain_mask[i : i + 1] if brain_mask is not None else None,
                affine=affine[i : i + 1],
            )

            values = {
                "total": float(loss),
                **{key: float(value) for key, value in components.items()},
            }

            if not all(
                torch.isfinite(torch.tensor(value)) for value in values.values()
            ):
                raise ValueError("Nonfinite validation loss")

            index = min(int(float(times[i]) * 5), 4)

            for key, value in values.items():
                totals[key] += value
                bins[index][key] += value

            bins[index]["count"] += 1
            count += 1

        print(
            json.dumps(
                {
                    "phase": "validation",
                    "cases_completed": count,
                    "cases": len(dataset),
                    "seconds": round(perf_counter() - started, 2),
                }
            ),
            flush=True,
        )

    if not count:
        raise ValueError("Validation split is empty")

    for entry in bins:
        if entry["count"]:
            for key in totals:
                entry[key] /= entry["count"]

    return {
        **{key: value / count for key, value in totals.items()},
        "time_bins": bins,
        "cases": count,
    }


def train(
    config,
    training_dataset,
    validation_dataset,
    resume=None,
    model=None,
    epoch_callback=None,
):
    device = torch.device(config["device"])
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1

    if world_size != config.get("world_size", 1):
        raise ValueError("Launch with torchrun using the configured world_size")

    # While rank 0 runs validation or sampling, the other ranks wait on a CPU
    # (Gloo) barrier. A waiting GPU (NCCL) barrier can disturb CUDA memory use.
    control_group = (
        dist.new_group(backend="gloo", timeout=timedelta(hours=6))
        if distributed
        else None
    )

    torch.manual_seed(config["seed"])

    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])

        torch.backends.cudnn.benchmark = config["cudnn_benchmark"]

    run_dir = Path(config["run_dir"])

    if resume is None and rank == 0:
        run_dir.mkdir(parents=True, exist_ok=False)

        if hasattr(training_dataset, "frame"):
            (run_dir / "manifest.csv").write_text(Path(config["manifest"]).read_text())
    elif resume is not None and (
        not run_dir.is_dir() or Path(resume).resolve().parent != run_dir.resolve()
    ):
        raise ValueError(
            "Resume checkpoint must belong to the configured run directory"
        )

    if distributed:
        dist.barrier()

    model = (model or ParallelVoxelFlow(config)).to(device)

    if resume is None and config.get("finetune_checkpoint"):
        source = torch.load(
            config["finetune_checkpoint"], map_location="cpu", weights_only=True
        )

        parameters = {
            key.removeprefix("module."): value
            for key, value in source["ema_model"].items()
            if key.startswith("module.")
        }

        model.load_state_dict(parameters, strict=True)

        if rank == 0:
            print(
                json.dumps(
                    {
                        "phase": "finetune_initialized",
                        "weights": "ema",
                        "source_updates": source["optimizer_updates"],
                        "learning_rate": config["learning_rate"],
                    }
                ),
                flush=True,
            )

        del source, parameters

    training_model = (
        DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            forward_sync_buffers=False,
        )
        if distributed
        else model
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=tuple(config["adam_betas"]),
        weight_decay=config["weight_decay"],
    )

    ema = AveragedModel(
        model, multi_avg_fn=get_ema_multi_avg_fn(config["ema_decay"]), device=device
    ).eval()

    if config.get("compile_cnn_blocks", False):
        # Compile after deepcopy so each callable is bound to its own model.
        model.compile_cnn_blocks()

        ema.module.compile_cnn_blocks()

    generators = {
        "shuffle": torch.Generator().manual_seed(config["seed"] + 1),
        "noise_time": torch.Generator(device=device).manual_seed(
            config["seed"] + 2 + 1000 * rank
        ),
        "report_drop": torch.Generator(device=device).manual_seed(
            config["seed"] + 3 + 1000 * rank
        ),
        "loader": torch.Generator().manual_seed(config["seed"] + 4 + 1000 * rank),
    }

    epoch, cursor, updates, microbatches = 0, 0, 0, 0
    order = torch.empty(0, dtype=torch.long)
    best = float("inf")

    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=True)

        validate_checkpoint(state, config)

        model.load_state_dict(state["model"])

        ema.load_state_dict(state["ema_model"])

        optimizer.load_state_dict(state["optimizer"])

        epoch, cursor = state["epoch"], state["next_sample"]
        updates, microbatches = state["optimizer_updates"], state["microbatches"]
        order, best = state["epoch_order"], state["best_validation_loss"]

        if len(order) and (
            len(order) != len(training_dataset)
            or not torch.equal(order.sort().values, torch.arange(len(training_dataset)))
        ):
            raise ValueError("Saved epoch order differs from training dataset")

        if not 0 <= cursor <= len(order):
            raise ValueError("Invalid saved training cursor")

        if state.get("world_size", 1) == world_size:
            rng = state["rank_rng"][rank] if "rank_rng" in state else state

            for key, generator in generators.items():
                generator.set_state(rng["generators"][key])

            torch.set_rng_state(rng["cpu_rng"])

            if device.type == "cuda":
                if "rank_rng" in state:
                    torch.cuda.set_rng_state(rng["cuda_rng"], device)
                else:
                    torch.cuda.set_rng_state_all(state["cuda_rng"])
        else:
            # Keep sample order/cursor; restart independent stochastic streams for
            # the new rank layout. Changing topology is not a bitwise replay.
            rng = state["rank_rng"][0] if "rank_rng" in state else state

            generators["shuffle"].set_state(rng["generators"]["shuffle"])

            restart_seed = config["seed"] + 1000003 * updates + 1000 * rank

            for offset, key in enumerate(("noise_time", "report_drop", "loader"), 2):
                generators[key].manual_seed(restart_seed + offset)

            torch.manual_seed(restart_seed)

            if device.type == "cuda":
                torch.cuda.manual_seed(restart_seed)

            if rank == 0:
                print(
                    json.dumps(
                        {
                            "phase": "resume_topology_change",
                            "previous_world_size": state.get("world_size", 1),
                            "world_size": world_size,
                            "effective_batch_size": (
                                config["batch_size"]
                                * world_size
                                * config["gradient_accumulation_steps"]
                            ),
                        }
                    ),
                    flush=True,
                )

    if rank == 0:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    def save(name):
        rng = {
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else None,
            "generators": {key: value.get_state() for key, value in generators.items()},
        }

        rank_rng = [None] * world_size if rank == 0 else None

        if distributed:
            dist.gather_object(rng, rank_rng, dst=0)
        else:
            rank_rng = [rng]

        if rank != 0:
            return

        state = {
            "architecture": ARCHITECTURE,
            "config": config,
            "cache_identity": cache_fingerprint(config),
            "model": model.state_dict(),
            "ema_model": ema.state_dict(),
            "ema_updates": int(ema.n_averaged),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "next_sample": cursor,
            "epoch_order": order,
            "optimizer_updates": updates,
            "microbatches": microbatches,
            "best_validation_loss": best,
            "world_size": world_size,
            "rank_rng": rank_rng,
            **rng,
        }

        torch.save(state, run_dir / name)

    mode = "a" if resume else "w"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    log_context = (run_dir / "train.jsonl").open(mode) if rank == 0 else nullcontext()

    with log_context as log:
        while epoch < config["epochs"]:
            epoch_started = perf_counter()

            if config["max_updates"] is not None and updates >= config["max_updates"]:
                break

            if not len(order):
                order = torch.randperm(
                    len(training_dataset), generator=generators["shuffle"]
                )

                cursor = 0

            remaining = order[cursor:]
            local_valid = len(remaining[rank::world_size])

            # Pad only for equal DDP step counts; padded examples have zero loss weight.
            padded_count = (
                (len(remaining) + world_size - 1) // world_size
            ) * world_size

            positions = torch.arange(padded_count)[rank::world_size]
            indices = remaining[positions % max(len(remaining), 1)]

            loader = make_loader(
                training_dataset, config, generators["loader"], indices
            )

            training_model.train()

            window_count = 0
            window_loss = torch.zeros((), dtype=torch.float64, device=device)
            window_times = []
            stopped = False

            for batch in loader:
                if window_count == 0:
                    optimizer.zero_grad(set_to_none=True)

                    window_examples = min(
                        config["batch_size"]
                        * world_size
                        * config["gradient_accumulation_steps"],
                        len(order) - cursor,
                    )

                target, condition, affine = batch_to_device(batch, device)
                brain_mask = batch.get("brain_mask")

                if brain_mask is not None:
                    brain_mask = brain_mask.to(device, non_blocking=True)

                valid = min(len(target), local_valid)
                local_valid -= valid

                global_count = min(
                    config["batch_size"] * world_size, len(order) - cursor
                )

                update_boundary = window_count + global_count == window_examples

                noise = (
                    torch.randn(
                        target.shape, device=device, generator=generators["noise_time"]
                    )
                    * config["source_noise_std"]
                )

                times = torch.rand(
                    len(target), device=device, generator=generators["noise_time"]
                )

                if config.get("log_gradient_diagnostics", False):
                    window_times.append(times[:valid].detach())

                condition["report_drop"] = (
                    torch.rand(
                        len(target), device=device, generator=generators["report_drop"]
                    )
                    < config["report_semantic_dropout"]
                )

                sync = (
                    training_model.no_sync()
                    if distributed and not update_boundary
                    else nullcontext()
                )

                with sync:
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=config["mixed_precision"] == "bf16",
                    ):
                        outputs = training_model(
                            make_noisy_state(target, noise, times),
                            times,
                            condition,
                            affine,
                        )

                    # A rank with no real tail example still participates in backward.
                    count = max(valid, 1)

                    case_outputs = {
                        key: outputs[key][:count] for key in ("clean", "global_clean")
                    }

                    loss, _ = voxel_flow_loss(
                        case_outputs,
                        target[:count],
                        times[:count],
                        config,
                        brain_mask=(
                            brain_mask[:count] if brain_mask is not None else None
                        ),
                        affine=affine[:count],
                    )

                    fraction = valid / window_examples

                    # DDP averages rank gradients; undo that average for case weighting.
                    (loss * (world_size * fraction)).backward()

                # Keep logging reductions on-device until the optimizer boundary.
                window_loss += loss.detach().double() * fraction
                window_count += global_count
                cursor += global_count
                microbatches += 1

                if window_count == window_examples:
                    if distributed:
                        dist.all_reduce(window_loss)

                        if config.get("log_gradient_diagnostics", False):
                            gathered_times = [None] * world_size

                            dist.all_gather_object(
                                gathered_times, torch.cat(window_times).cpu()
                            )

                            window_times = gathered_times

                    if not torch.isfinite(window_loss):
                        raise ValueError("Nonfinite voxel-flow loss")

                    diagnostics = (
                        gradient_diagnostics(
                            model, window_times, config["training_denominator_floor"]
                        )
                        if config.get("log_gradient_diagnostics", False)
                        else {}
                    )

                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config["gradient_clip_norm"],
                        error_if_nonfinite=True,
                    )

                    optimizer.step()

                    ema.update_parameters(model)

                    memory = {}

                    if device.type == "cuda":
                        peaks = torch.tensor(
                            [
                                torch.cuda.max_memory_allocated(device) / 2**30,
                                torch.cuda.max_memory_reserved(device) / 2**30,
                            ],
                            device=device,
                        )

                        if distributed:
                            dist.all_reduce(peaks, op=dist.ReduceOp.MAX)

                        memory = {
                            "peak_allocated_gib": float(peaks[0]),
                            "peak_reserved_gib": float(peaks[1]),
                        }

                    updates += 1

                    record = {
                        "epoch": epoch,
                        "optimizer_updates": updates,
                        "microbatches": microbatches,
                        "next_sample": cursor,
                        "window_examples": window_examples,
                        "loss": float(window_loss),
                        "gradient_norm": float(norm),
                        "ema_updates": int(ema.n_averaged),
                        "world_size": world_size,
                        "batch_size_per_gpu": config["batch_size"],
                        **memory,
                        **diagnostics,
                    }

                    if rank == 0:
                        log.write(json.dumps(record) + "\n")
                        log.flush()

                        print(json.dumps(record), flush=True)

                    window_count = 0

                    window_loss.zero_()

                    window_times = []

                    if updates % config["checkpoint_every_updates"] == 0:
                        save("last.pt")

                    if (
                        config["max_updates"] is not None
                        and updates >= config["max_updates"]
                    ):
                        stopped = True

                        break

            end_epoch = cursor == len(order)

            if end_epoch:
                save("last.pt")

                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "phase": "epoch_training_complete",
                                "epoch": epoch + 1,
                                "seconds": perf_counter() - epoch_started,
                            }
                        ),
                        flush=True,
                    )

            validation = None

            if stopped or (
                end_epoch and (epoch + 1) % config["validation_every_epochs"] == 0
            ):
                if rank == 0:
                    validation = compute_validation_loss(
                        model, validation_dataset, config
                    )

                    log.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "optimizer_updates": updates,
                                "validation": validation,
                            }
                        )
                        + "\n"
                    )

                    log.flush()

                if distributed:
                    shared = [validation]

                    dist.broadcast_object_list(shared, src=0, group=control_group)

                    validation = shared[0]

                if validation["total"] < best:
                    best = validation["total"]

                    save("best_loss.pt")

            if end_epoch and epoch_callback is not None and rank == 0:
                # Evaluation callbacks use their own random generators. Save and
                # restore the global random state so they cannot change training.
                cuda_devices = [device.index or 0] if device.type == "cuda" else []

                with (
                    torch.random.fork_rng(devices=cuda_devices),
                    torch.backends.cudnn.flags(enabled=True, benchmark=False),
                    torch.compiler.set_stance("force_eager"),
                ):
                    epoch_callback(epoch + 1, model, ema.module)

            if distributed:
                dist.barrier(group=control_group)

            if end_epoch:
                completed_epoch = epoch + 1
                epoch += 1
                cursor = 0
                order = torch.empty(0, dtype=torch.long)

                if completed_epoch in config["checkpoint_epochs"]:
                    save(f"epoch_{completed_epoch:03d}.pt")

            save("last.pt")

            if end_epoch and rank == 0:
                print(
                    json.dumps(
                        {
                            "phase": "epoch_complete",
                            "epoch": epoch,
                            "seconds": perf_counter() - epoch_started,
                        }
                    ),
                    flush=True,
                )

            if stopped:
                break

    return model, ema, optimizer
