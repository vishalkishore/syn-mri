"""Streaming native-resolution diagnostics; these do not validate clinical fidelity."""

import json
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from src.voxel_flow.data import collate_batch
from src.voxel_flow.flow import clean_to_velocity
from src.voxel_flow.flow import neighbor_difference_errors
from src.voxel_flow.flow import make_noisy_state
from src.voxel_flow.flow import predict_clean
from src.voxel_flow.flow import generate_from_noise
from src.voxel_flow.flow import voxel_flow_loss
from src.voxel_flow.training import batch_to_device
from util.haar import BAND_NAMES
from util.haar import haar_decompose

DIAGNOSTIC_TIMES = (0.0, 0.25, 0.5, 0.75, 0.95, 0.99)


def select_cases(frame, split, count, seed):
    if split not in ("validation", "test"):
        raise ValueError("Evaluation requires validation or explicitly selected test")

    if frame.groupby("patient_uid").split.nunique().gt(1).any():
        raise ValueError("Patient split leakage")

    if count < 2:
        raise ValueError(
            "Need at least two patients per group (four held-out patients)"
        )

    heldout = frame[frame.split.eq(split)].sort_values(
        ["patient_uid", "study_uid", "series_id"]
    )

    if heldout.patient_uid.nunique() < 2 * count:
        raise ValueError(
            "Too few held-out patients; explicitly reduce --num-samples (minimum 2)"
        )

    rng = np.random.default_rng(seed)
    rows = []

    for _, scans in heldout.groupby("patient_uid", sort=True):
        rows.append(scans.iloc[rng.integers(len(scans))])

    candidates = pd.DataFrame(rows)
    pairs = []
    contrast_pairs = []

    for contrast, group in candidates.groupby("contrast", sort=True):
        positions = rng.permutation(len(group))

        contrast_pairs.append(
            [
                (group.iloc[positions[i]], group.iloc[positions[i + 1]])
                for i in range(0, len(positions) - 1, 2)
            ]
        )

    # Round-robin contrast strata; every primary/reference pair has equal contrast.
    while len(pairs) < count and any(contrast_pairs):
        for group in contrast_pairs:
            if group and len(pairs) < count:
                pairs.append(group.pop())

    if len(pairs) != count:
        raise ValueError(
            "Too few independent contrast-matched patients; reduce --num-samples"
        )

    selected = pd.DataFrame([pair[0] for pair in pairs] + [pair[1] for pair in pairs])
    selected["evaluation_group"] = ["primary"] * count + ["reference"] * count

    return selected.reset_index(drop=True)


def average_pool(volume, factor):
    size_x, size_y, size_z = volume.shape

    return volume.reshape(
        size_x // factor, factor, size_y // factor, factor, size_z // factor, factor
    ).mean(axis=(1, 3, 5))


def wavelet_statistics(volume):
    coarse, details = haar_decompose(np.asarray(volume, dtype=np.float32)[None], 3)
    scalar = {"A3_energy": float(np.square(coarse.astype(np.float64)).mean())}

    for level, bands in enumerate(details, start=1):
        for name, band in zip(BAND_NAMES[1:], bands):
            scalar[f"H{level}_{name}_energy"] = float(
                np.square(band.astype(np.float64)).mean()
            )

    # Compare detail energy with coarse edges on one grid: 4x-pooled voxels.
    # H1 (2x grid) is pooled by 2 to reach it; H2 is already on the 4x grid.
    coarse_image = average_pool(volume, 4)
    center = coarse_image[:-1, :-1, :-1]

    boundary = sum(
        np.square(neighbor - center)
        for neighbor in (
            coarse_image[1:, :-1, :-1],
            coarse_image[:-1, 1:, :-1],
            coarse_image[:-1, :-1, 1:],
        )
    )

    for level in (1, 2):
        energy = np.zeros(coarse_image.shape, dtype=np.float64)

        for band in details[level - 1]:
            square = np.square(band.astype(np.float64))
            energy += average_pool(square, 2) if level == 1 else square

        valid = energy[:-1, :-1, :-1]
        correlation = 0.0

        if np.std(valid) > 0 and np.std(boundary) > 0:
            correlation = float(np.corrcoef(valid.ravel(), boundary.ravel())[0, 1])

        scalar[f"H{level}_boundary_correlation"] = correlation

        for q, value in zip((0.1, 0.5, 0.9), np.quantile(valid, [0.1, 0.5, 0.9])):
            scalar[f"H{level}_placement_q{q}"] = float(value)

        # Mean detail energy where coarse edges are weak, medium, or strong (tertiles).
        cuts = np.quantile(boundary, [1 / 3, 2 / 3])
        bins = np.searchsorted(cuts, boundary, side="right")

        for group in range(3):
            values = valid[bins == group]

            scalar[f"H{level}_boundary_bin{group}_mean_energy"] = (
                float(values.mean()) if len(values) else 0.0
            )

    return scalar


def volume_statistics(volume, affine, thresholds=(0.01, 0.05)):
    volume = np.asarray(volume, dtype=np.float32)

    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("Expected finite native volume")

    flat = volume.ravel()
    mean = float(flat.mean(dtype=np.float64))
    second_moment = float(np.square(flat.astype(np.float64)).mean())

    scalar = {
        "mean": mean,
        "variance": float(flat.var(dtype=np.float64)),
        "second_moment": second_moment,
        "negative_fraction": float((flat < 0).mean()),
    }

    for q, value in zip(
        (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0),
        np.quantile(flat, [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]),
    ):
        scalar[f"intensity_q{q}"] = float(value)

    for threshold in thresholds:
        scalar[f"near_zero_{threshold}"] = float((np.abs(flat) < threshold).mean())

    shell = np.zeros(volume.shape, dtype=bool)

    for axis in range(3):
        for edge in (0, -1):
            index = [slice(None)] * 3
            index[axis] = edge
            shell[tuple(index)] = True

    shell_values = volume[shell].astype(np.float64)
    scalar["outer_one_voxel_shell_mean_abs"] = float(np.abs(shell_values).mean())

    scalar["outer_one_voxel_shell_second_moment"] = float(
        np.square(shell_values).mean()
    )

    distributions = {
        "intensity": np.quantile(flat, (np.arange(128) + 0.5) / 128).tolist()
    }

    spacing = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)

    for offset in (1, 2, 4):
        for axis, name in enumerate("XYZ"):
            earlier, later = [slice(None)] * 3, [slice(None)] * 3
            earlier[axis], later[axis] = slice(None, -offset), slice(offset, None)
            difference = volume[tuple(later)] - volume[tuple(earlier)]
            prefix = f"difference_{name}_{offset}"
            scalar[prefix + "_mean"] = float(difference.mean(dtype=np.float64))

            scalar[prefix + "_rms"] = float(
                np.sqrt(np.square(difference.astype(np.float64)).mean())
            )

            distributions[prefix] = np.quantile(
                difference, (np.arange(128) + 0.5) / 128
            ).tolist()

            if offset == 1:
                scalar[f"gradient_{name}_rms_per_mm"] = (
                    scalar[prefix + "_rms"] / spacing[axis]
                )

    scalar.update(wavelet_statistics(volume))

    return {"scalar": scalar, "distributions": distributions}


def compare_groups(first, second):
    if len(first) != len(second):
        raise ValueError("Distribution comparisons need equal patient counts")

    result = {"patients_per_group": len(first), "scalar": {}, "distribution": {}}

    for key in first[0]["scalar"]:
        first_values = np.asarray([item["scalar"][key] for item in first])
        second_values = np.asarray([item["scalar"][key] for item in second])

        result["scalar"][key] = {
            "real_mean": float(first_values.mean()),
            "comparison_mean": float(second_values.mean()),
            "wasserstein": float(
                np.abs(np.sort(first_values) - np.sort(second_values)).mean()
            ),
        }

    for key in first[0]["distributions"]:
        first_values = np.sort(
            np.asarray([item["distributions"][key] for item in first]).ravel()
        )
        second_values = np.sort(
            np.asarray([item["distributions"][key] for item in second]).ravel()
        )

        result["distribution"][key] = {
            "wasserstein_equal_volume_quantiles": float(
                np.abs(first_values - second_values).mean()
            )
        }

    return result


def trajectory_rates(state, velocity):
    x, v = state.double(), velocity.double()

    return {
        "second_moment_rate": float(2 * (x * v).mean()),
        "centered_variance_rate": float(2 * ((x - x.mean()) * (v - v.mean())).mean()),
        "state_second_moment": float(x.square().mean()),
        "velocity_rms": float(v.square().mean().sqrt()),
    }


def save_generated(path, volume, affine):
    volume = np.asarray(volume, dtype=np.float32)
    header = nib.Nifti1Header()

    header.set_data_dtype(np.float32)
    header.set_xyzt_units("mm")

    header["descrip"] = b"positive-brain-median normalized generated MRI"
    image = nib.Nifti1Image(volume, affine, header)

    image.set_sform(affine, code=1)
    # qform cannot encode shear. Keep the exact sform and only retain a qform
    # when its quaternion representation preserves this affine.
    image.set_qform(affine, code=1)

    if not np.allclose(image.get_qform(), affine, rtol=0, atol=1e-5):
        image.set_qform(None, code=0)

    nib.save(image, path)


def preview_slices(volume):
    planes = []

    for axis in range(3):
        for fraction in (0.35, 0.5, 0.65):
            planes.append(
                np.take(
                    volume,
                    min(int(volume.shape[axis] * fraction), volume.shape[axis] - 1),
                    axis=axis,
                ).T.copy()
            )

    return planes


def save_preview(rows, path, limits=(-0.1, 2.5)):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 9, figsize=(18, 2.5 * len(rows)), squeeze=False)

    for line, (title, planes) in enumerate(rows):
        for index, plane in enumerate(planes):
            axes[line, index].imshow(
                plane, origin="lower", cmap="gray", vmin=limits[0], vmax=limits[1]
            )

            axes[line, index].axis("off")

        axes[line, 0].set_title(title, fontsize=7)

    fig.suptitle(f"Unpaired native volumes; fixed display [{limits[0]}, {limits[1]}]")
    fig.tight_layout()
    fig.savefig(path, dpi=100)

    plt.close(fig)


def save_grid_maps(records, output, thresholds):
    """Read one geometry stratum at a time; never align unlike native grids."""

    groups = {}

    for record in records:
        key = (
            record["kind"],
            record["contrast"],
            tuple(np.round(record["affine"].ravel(), 5)),
        )

        groups.setdefault(key, []).append(record)

    summaries = []

    for index, ((kind, contrast, _), group) in enumerate(groups.items()):
        # A single case has no estimate of between-patient spatial variance.
        if len(group) < 2:
            summaries.append(
                {
                    "kind": kind,
                    "contrast": contrast,
                    "count": 1,
                    "maps": None,
                    "reason": "Only one volume on this physical grid",
                }
            )

            continue

        total = square = occupancy = None

        for record in group:
            volume = nib.load(record["path"]).get_fdata(dtype=np.float32)

            if total is None:
                total = np.zeros(volume.shape, dtype=np.float64)
                square = np.zeros_like(total)
                occupancy = np.zeros((len(thresholds), *volume.shape), dtype=np.float64)

            total += volume
            square += np.square(volume.astype(np.float64))

            for j, threshold in enumerate(thresholds):
                occupancy[j] += np.abs(volume) < threshold

        mean = total / len(group)
        destination = output / f"grid_maps_{index:03d}.npz"

        np.savez(
            destination,
            mean=mean.astype(np.float32),
            variance=np.maximum(square / len(group) - mean**2, 0).astype(np.float32),
            near_zero_occupancy=(occupancy / len(group)).astype(np.float32),
            affine=group[0]["affine"],
            thresholds=thresholds,
        )

        summaries.append(
            {
                "kind": kind,
                "contrast": contrast,
                "count": len(group),
                "maps": destination.name,
            }
        )

    return summaries


@torch.inference_mode()
@torch.backends.cudnn.flags(enabled=True, benchmark=False)
@torch.compiler.set_stance("force_eager")
def evaluate(
    model, dataset, frame, config, output, split="validation", count=None, weights="ema"
):
    started = perf_counter()
    output = Path(output)

    selected = select_cases(
        frame, split, count or config["evaluation_samples"], config["evaluation_seed"]
    )

    output.mkdir(parents=True, exist_ok=False)

    selected.to_csv(output / "selected_cases.csv", index=False)

    settings = {
        "architecture": config["architecture"],
        "normalization": config["voxel_normalization"],
        "weights": weights,
        "seed": config["evaluation_seed"],
        "steps": config["sampling_steps"],
        "split": split,
        "patients_per_group": len(selected) // 2,
        "guidance_scale": config["guidance_scale"],
        "solver": config["sampler"],
        "display_limits": [-0.1, 2.5],
        "diagnostic_times": DIAGNOSTIC_TIMES,
        "geometry_note": (
            "Maps use identical affines within contrast strata. Unlike "
            "spacings/origins are not identical physical grids; no "
            "resampling is applied."
        ),
    }

    (output / "settings.json").write_text(json.dumps(settings, indent=2))

    model.eval()

    device = config["device"]

    lookup = {
        (row.study_uid, row.series_id): index for index, row in dataset.frame.iterrows()
    }

    generator = torch.Generator(device=device).manual_seed(config["evaluation_seed"])
    summaries = {"primary": [], "reference": [], "generated": []}

    scalar_rows, previews, trajectories, diagnostics, maps, review = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    diversity = []
    primary_total = len(selected) // 2
    evaluations_total = 0

    print(
        json.dumps({"phase": "evaluation_start", "primary_cases": primary_total}),
        flush=True,
    )

    for selection_index, row in selected.iterrows():
        print(
            json.dumps(
                {
                    "phase": "evaluation_case",
                    "case": selection_index,
                    "group": row.evaluation_group,
                }
            ),
            flush=True,
        )

        item = dataset[lookup[(row.study_uid, row.series_id)]]
        real = item["volume"][0].numpy()
        affine_array = item["affine"].numpy()
        case = f"case_{selection_index:03d}"
        real_path = output / f"{case}_real.nii.gz"

        save_generated(real_path, real, affine_array)

        real_summary = volume_statistics(
            real, affine_array, config["near_zero_thresholds"]
        )

        summaries[row.evaluation_group].append(real_summary)

        scalar_rows.append(
            {
                "case": case,
                "kind": row.evaluation_group,
                "patient_uid": row.patient_uid,
                **real_summary["scalar"],
            }
        )

        maps.append(
            {
                "kind": row.evaluation_group,
                "contrast": row.contrast,
                "affine": affine_array,
                "path": real_path,
            }
        )

        if row.evaluation_group == "reference":
            continue

        target, condition, affine = batch_to_device(collate_batch([item]), device)
        brain_mask = item.get("brain_mask")

        if brain_mask is not None:
            brain_mask = brain_mask[None].to(device)

        noise = (
            torch.randn(target.shape, device=device, generator=generator)
            * config["source_noise_std"]
        )

        condition["report_drop"] = torch.zeros(1, device=device, dtype=torch.bool)

        for time in DIAGNOSTIC_TIMES:
            times = torch.full((1,), time, device=device)
            state = make_noisy_state(target, noise, times)

            with torch.autocast(
                device_type=target.device.type,
                dtype=torch.bfloat16,
                enabled=config["mixed_precision"] == "bf16",
            ):
                predicted = model(state, times, condition, affine)

            loss, components = voxel_flow_loss(
                predicted, target, times, config, brain_mask=brain_mask, affine=affine
            )

            error = predicted["clean"].float() - target

            diagnostics.append(
                {
                    "case": case,
                    "time": time,
                    "capped_training_loss": float(loss),
                    **{
                        f"capped_{key}": float(value)
                        for key, value in components.items()
                    },
                    "voxel_mse": float(error.square().mean()),
                    "signed_contrast_mse": float(
                        neighbor_difference_errors(error).mean()
                    ),
                    "P4_global_mse": float(
                        (
                            F.avg_pool3d(predicted["global_clean"], 4)
                            - F.avg_pool3d(target, 4)
                        )
                        .square()
                        .mean()
                    ),
                    "uncapped_velocity_mse": float(
                        (error / (1 - time)).square().mean()
                    ),
                }
            )

            trajectories.append(
                {
                    "case": case,
                    "kind": "reference_path",
                    "time": time,
                    **trajectory_rates(state, target - noise),
                }
            )

        wanted = {
            min(round(time * config["sampling_steps"]), config["sampling_steps"] - 1)
            for time in DIAGNOSTIC_TIMES
        }

        def observer(time, state, clean, velocity):
            if round(time * config["sampling_steps"]) in wanted:
                clean_array = clean[0, 0].cpu().numpy()

                trajectories.append(
                    {
                        "case": case,
                        "kind": "generated",
                        "time": time,
                        **trajectory_rates(state, velocity),
                        "clean_rms_over_state_rms": float(
                            clean.double().square().mean().sqrt()
                            / state.double().square().mean().sqrt().clamp_min(1e-12)
                        ),
                        "endpoint_signed_contrast_energy": float(
                            neighbor_difference_errors(clean).mean()
                        ),
                        **wavelet_statistics(clean_array),
                    }
                )

        callback = lambda state, times: predict_clean(
            model, state, times, condition, affine, config
        )

        generated, calls = generate_from_noise(
            callback, noise, config["sampling_steps"], observer
        )

        print(
            json.dumps(
                {
                    "phase": "evaluation_sample",
                    "case": case,
                    "kind": "conditioned",
                    "model_evaluations": calls,
                }
            ),
            flush=True,
        )

        evaluations_total += calls
        generated_array = generated[0, 0].cpu().numpy()
        generated_path = output / f"{case}_generated.nii.gz"

        save_generated(generated_path, generated_array, affine_array)

        summary = volume_statistics(
            generated_array, affine_array, config["near_zero_thresholds"]
        )

        summaries["generated"].append(summary)

        scalar_rows.append(
            {
                "case": case,
                "kind": "generated",
                "patient_uid": row.patient_uid,
                **summary["scalar"],
            }
        )

        maps.append(
            {
                "kind": "generated",
                "contrast": row.contrast,
                "affine": affine_array,
                "path": generated_path,
            }
        )

        if len(previews) < 8:
            previews.extend(
                [
                    (f"{case} real {row.contrast}", preview_slices(real)),
                    (f"{case} generated", preview_slices(generated_array)),
                ]
            )

        review.append(
            {
                "case": case,
                "patient_uid": row.patient_uid,
                "study_uid": row.study_uid,
                "series_id": row.series_id,
                "contrast": row.contrast,
                "report": row.get("report", ""),
                "t1_visible_findings": "",
                **{
                    key: ""
                    for key in (
                        "tissue_boundaries",
                        "ventricles",
                        "cortex",
                        "left_right_consistency",
                        "artifacts",
                        "reviewer",
                    )
                },
            }
        )

        # Additional seeds for a fixed report and geometry; accumulate moments on CPU.
        if selection_index < config["diversity_reports"]:
            running_mean = generated_array.astype(np.float64)
            sum_squared_deviations = np.zeros_like(running_mean)

            for repeat in range(1, config["diversity_samples_per_report"]):
                repeated_noise = (
                    torch.randn(target.shape, device=device, generator=generator)
                    * config["source_noise_std"]
                )

                additional, calls = generate_from_noise(
                    callback, repeated_noise, config["sampling_steps"]
                )

                print(
                    json.dumps(
                        {
                            "phase": "evaluation_sample",
                            "case": case,
                            "kind": "diversity",
                            "repeat": repeat,
                        }
                    ),
                    flush=True,
                )

                evaluations_total += calls
                array = additional[0, 0].cpu().numpy()
                difference = array - running_mean
                running_mean += difference / (repeat + 1)
                sum_squared_deviations += difference * (array - running_mean)

                save_generated(
                    output / f"{case}_seed_{repeat}.nii.gz", array, affine_array
                )

            sample_count = config["diversity_samples_per_report"]

            diversity.append(
                {
                    "case": case,
                    "samples": sample_count,
                    "mean_location_variance_within_report": float(
                        sum_squared_deviations.mean() / sample_count
                    ),
                    "mean_volume_intensity": float(running_mean.mean()),
                }
            )

        # The same noise and acquisition, with semantic information changed only.
        ablations = []

        for mode in ("null", "shuffled"):
            changed = dict(condition)

            if mode == "null":
                changed["report_drop"] = torch.ones(1, dtype=torch.bool, device=device)
            else:
                shuffled_row = selected.iloc[(selection_index + 1) % primary_total]

                alternative = dataset[
                    lookup[(shuffled_row.study_uid, shuffled_row.series_id)]
                ]

                changed_report = collate_batch([alternative])["condition"]
                changed["report_tokens"] = changed_report["report_tokens"].to(device)
                changed["report_mask"] = changed_report["report_mask"].to(device)

            ablated, calls = generate_from_noise(
                lambda state, times: predict_clean(
                    model, state, times, changed, affine, config
                ),
                noise,
                config["sampling_steps"],
            )

            evaluations_total += calls

            print(
                json.dumps(
                    {
                        "phase": "evaluation_sample",
                        "case": case,
                        "kind": mode,
                        "model_evaluations": calls,
                    }
                ),
                flush=True,
            )

            ablations.append(
                {
                    "case": case,
                    "kind": mode,
                    "same_noise_voxel_mse_vs_correct_report": float(
                        (ablated - generated).double().square().mean()
                    ),
                }
            )

        diagnostics.extend(ablations)

    comparisons = {
        "real_vs_generated": compare_groups(
            summaries["primary"], summaries["generated"]
        ),
        "real_vs_real": compare_groups(summaries["primary"], summaries["reference"]),
    }

    # Matched contrast strata preserve equal per-volume weights within each comparison.
    contrasts = selected.iloc[:primary_total].contrast.tolist()
    comparisons["by_contrast"] = {}

    for contrast in sorted(set(contrasts)):
        indices = [i for i, value in enumerate(contrasts) if value == contrast]

        comparisons["by_contrast"][contrast] = {
            "real_vs_generated": compare_groups(
                [summaries["primary"][i] for i in indices],
                [summaries["generated"][i] for i in indices],
            ),
            "real_vs_real": compare_groups(
                [summaries["primary"][i] for i in indices],
                [summaries["reference"][i] for i in indices],
            ),
        }

    grid_maps = save_grid_maps(maps, output, config["near_zero_thresholds"])

    metrics = {
        "settings": settings,
        **comparisons,
        "model_evaluations": evaluations_total,
        "primary_sample_evaluations": 2 * config["sampling_steps"] - 1,
        "grid_maps": grid_maps,
        "diversity": diversity,
        "across_report_variance_of_volume_means": float(
            np.var([item["scalar"]["mean"] for item in summaries["generated"]])
        ),
        "collapsed_within_report": bool(
            diversity
            and all(
                item["mean_location_variance_within_report"] < 1e-12
                for item in diversity
            )
        ),
        "interpretation": (
            "Automated diagnostics only. No validated anatomy or "
            "clinical report-fidelity score is provided."
        ),
    }

    real_variance = np.mean(
        [item["scalar"]["variance"] for item in summaries["primary"]]
    )

    generated_variance = np.mean(
        [item["scalar"]["variance"] for item in summaries["generated"]]
    )

    metrics["generated_to_real_spatial_variance_ratio"] = (
        float(generated_variance / real_variance) if real_variance > 0 else None
    )

    (output / "distribution_summaries.json").write_text(
        json.dumps(summaries, allow_nan=False)
    )

    pd.DataFrame(scalar_rows).to_csv(output / "per_case_scalars.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "prediction_diagnostics.csv", index=False)

    pd.DataFrame(trajectories).to_csv(
        output / "trajectory_diagnostics.csv", index=False
    )

    pd.DataFrame(review).to_csv(output / "anatomy_report_review.csv", index=False)

    save_preview(previews, output / "preview.png")

    metrics["elapsed_seconds"] = perf_counter() - started

    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))

    print(
        json.dumps({"phase": "evaluation_complete", "output": str(output)}), flush=True
    )

    return metrics
