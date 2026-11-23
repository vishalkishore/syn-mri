"""Plan, build, verify, or benchmark lossless native MRI targets."""

import argparse
import json
from pathlib import Path
import resource
import shutil
import sys
from time import perf_counter

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.voxel_flow.config import CACHE_FORMAT
from src.voxel_flow.config import NORMALIZATION
from src.voxel_flow.config import load_config
from src.voxel_flow.config import workspace_path
from src.voxel_flow.data import load_voxel_target
from src.voxel_flow.data import normalize_by_brain_median
from src.voxel_flow.data import read_cache_manifest
from src.voxel_flow.data import read_report_studies
from src.voxel_flow.data import save_voxel_target
from src.voxel_flow.data import validate_cohort


def source_paths(row, source_root):
    image = Path(source_root) / row.image_path
    mask_path = row.get("mask_path", "")

    # The existing preprocessing cohort predates a mask_path column. Its saved
    # independent mask uses this established sibling name, not inferred support.
    mask = (
        Path(source_root) / mask_path
        if mask_path
        else image.with_name(image.name.replace(".nii.gz", "_mask.nii.gz"))
    )

    return image, mask


def selected_source(config, splits=None, limit=None):
    frame = pd.read_csv(config["source_manifest"], dtype=str, keep_default_na=False)

    validate_cohort(
        frame,
        read_report_studies(config["report_token_cache"]),
        config["exploratory_pending_review"],
    )

    if splits:
        frame = frame[frame.split.isin(splits)]

    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")

        frame = frame.groupby("split", sort=False).head(limit)

    if frame.empty:
        raise ValueError("No selected source scans")

    return frame.copy()


def estimate_cache_size(config, output, splits=None, limit=None):
    frame = selected_source(config, splits, limit)

    source_bytes = sum(
        source_paths(row, config["dataset_root"])[0].stat().st_size
        for _, row in frame.iterrows()
    )

    destination = output

    while not destination.exists():
        destination = destination.parent

    tokens = np.load(Path(config["report_token_cache"]) / "tokens.npy", mmap_mode="r")

    result = {
        "scans": len(frame),
        "patients": frame.patient_uid.nunique(),
        "split_counts": frame.split.value_counts().to_dict(),
        "source_image_bytes": source_bytes,
        "dense_float32_bytes": len(frame) * int(np.prod(config["native_shape"])) * 4,
        "destination_free_bytes": shutil.disk_usage(destination).free,
        "report_token_bytes": tokens.nbytes,
        "output_dir": str(output),
    }

    print(json.dumps(result, indent=2), flush=True)

    return result


def build_cache(config, output, splits=None, limit=None):
    frame = selected_source(config, splits, limit)

    output.mkdir(parents=True, exist_ok=False)

    (output / "targets").mkdir()

    sizes, fractions, source_sizes = [], [], []

    for index, (row_index, row) in enumerate(
        tqdm(frame.iterrows(), total=len(frame), desc="Voxel targets")
    ):
        image, mask = source_paths(row, config["dataset_root"])
        target, affine, median = normalize_by_brain_median(image, mask)
        relative = Path("targets") / f"{index:06d}.npz"
        path = output / relative

        save_voxel_target(path, target, affine, median)

        restored, restored_affine, restored_scale = load_voxel_target(
            path, config["native_shape"]
        )

        if (
            not np.array_equal(restored, target)
            or not np.array_equal(restored_affine, affine)
            or restored_scale != median
        ):
            raise ValueError(f"Lossless cache verification failed: {image}")

        count = int(np.count_nonzero(target))
        frame.loc[row_index, "voxel_cache_path"] = str(relative)
        frame.loc[row_index, "target_scale"] = float(median)
        frame.loc[row_index, "nonzero_count"] = count

        frame.loc[row_index, "mask_path"] = (
            str(mask.relative_to(config["dataset_root"]))
            if mask.is_relative_to(config["dataset_root"])
            else str(mask)
        )

        sizes.append(path.stat().st_size)

        source_sizes.append(image.stat().st_size)

        fractions.append(count / target.size)

    summary = {
        "scans": len(frame),
        "exact_round_trips": len(frame),
        "bytes_median": float(np.median(sizes)),
        "bytes_p95": float(np.quantile(sizes, 0.95)),
        "bytes_mean": float(np.mean(sizes)),
        "source_gzip_bytes_mean": float(np.mean(source_sizes)),
        "nonzero_fraction_mean": float(np.mean(fractions)),
    }

    metadata = {
        "format": CACHE_FORMAT,
        "shape": config["native_shape"],
        "normalization": NORMALIZATION,
        "normalization_definition": (
            "FP32 source / median(positive source inside saved brain mask); no clipping"
        ),
        "dataset_root": config["dataset_root"],
        "source_manifest": config["source_manifest"],
        "report_token_cache": config["report_token_cache"],
        "exploratory_pending_review": config["exploratory_pending_review"],
        "policy_ids": (
            frame.final_policy_hash.unique().tolist()
            if "final_policy_hash" in frame
            else []
        ),
        "summary": summary,
    }

    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Written last: a partial build has no training manifest.
    frame.to_csv(output / "manifest.csv", index=False)

    print(json.dumps(summary, indent=2), flush=True)

    return summary


def verify_cache(manifest, comparisons=None, source_root=None):
    frame, metadata = read_cache_manifest(manifest)
    root = Path(manifest).resolve().parent
    source_root = source_root or metadata["dataset_root"]
    limit = len(frame) if comparisons is None else comparisons

    if limit < 0:
        raise ValueError("comparisons must be nonnegative")

    for index, (_, row) in enumerate(
        tqdm(frame.iterrows(), total=len(frame), desc="Verify targets")
    ):
        target, affine, median = load_voxel_target(
            root / row.voxel_cache_path, metadata["shape"]
        )

        image_path, mask_path = source_paths(row, source_root)
        image = nib.load(image_path)

        if image.shape != target.shape or not np.array_equal(affine, image.affine):
            raise ValueError("Source/cache geometry mismatch")

        if float(row.target_scale) != median or int(
            float(row.nonzero_count)
        ) != np.count_nonzero(target):
            raise ValueError("Manifest/cache target metadata mismatch")

        if index < limit:
            expected, expected_affine, expected_median = normalize_by_brain_median(
                image_path, mask_path
            )

            if (
                not np.array_equal(expected, target)
                or not np.array_equal(expected_affine, affine)
                or expected_median != median
            ):
                raise ValueError("Source/cache target mismatch")

    result = {
        "verified_scans": len(frame),
        "exact_source_comparisons": min(limit, len(frame)),
    }

    print(json.dumps(result), flush=True)

    return result


class BenchmarkTargets(Dataset):
    def __init__(self, frame, cache_root, source_root, mode):
        self.frame = frame.reset_index(drop=True)
        self.cache_root = cache_root
        self.source_root = source_root
        self.mode = mode

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        start = perf_counter()
        row = self.frame.iloc[index]

        if self.mode == "compact":
            volume, _, _ = load_voxel_target(self.cache_root / row.voxel_cache_path)
        else:
            image, mask = source_paths(row, self.source_root)
            volume, _, _ = normalize_by_brain_median(image, mask)

        return (
            torch.from_numpy(volume[None]),
            perf_counter() - start,
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )


def benchmark(manifest, repeats=3, workers=0, source_root=None):
    if repeats < 1:
        raise ValueError("repeats must be positive")

    frame, metadata = read_cache_manifest(manifest)
    root = Path(manifest).resolve().parent

    results = {
        "note": (
            "First and repeated passes; OS caches are not flushed. RSS is "
            "per-process high-water RAM."
        ),
        "workers": workers,
        "modes": {},
    }

    for mode in ("source", "compact"):
        passes = []

        dataset = BenchmarkTargets(
            frame, root, source_root or metadata["dataset_root"], mode
        )

        for repeat in range(repeats):
            loader = DataLoader(
                dataset,
                batch_size=1,
                num_workers=workers,
                shuffle=False,
                generator=torch.Generator().manual_seed(42),
            )

            start = perf_counter()
            latencies, worker_ram = [], []

            for batch, elapsed, rss in loader:
                # Default collation materializes the complete dense batch.
                if batch.shape[1:] != (1, *metadata["shape"]):
                    raise ValueError("Unexpected benchmark batch")

                latencies.extend(elapsed.tolist())

                worker_ram.extend(rss.tolist())

            seconds = perf_counter() - start

            passes.append(
                {
                    "pass": repeat + 1,
                    "load_seconds_median": float(np.median(latencies)),
                    "load_seconds_p95": float(np.quantile(latencies, 0.95)),
                    "examples_per_second_including_batches": len(frame) / seconds,
                    "seconds": seconds,
                    "max_worker_rss_kib": max(worker_ram),
                }
            )

        results["modes"][mode] = passes

    (root / "benchmark.json").write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2), flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("command", choices=("plan", "build", "verify", "benchmark"))
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/voxel_flow.yaml"),
    )
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--source-root")
    parser.add_argument("--split", nargs="+", choices=("train", "validation", "test"))
    parser.add_argument("--limit", type=int, help="Maximum scans per selected split")
    parser.add_argument("--comparisons", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)

    args = parser.parse_args()

    torch.set_num_threads(2)

    config = load_config(args.config)
    output = workspace_path(args.output_dir or config["voxel_cache_root"], config)
    manifest = workspace_path(args.manifest or config["manifest"], config)
    source = workspace_path(args.source_root, config) if args.source_root else None

    if args.command == "plan":
        estimate_cache_size(config, output, args.split, args.limit)
    elif args.command == "build":
        build_cache(config, output, args.split, args.limit)
    elif args.command == "verify":
        verify_cache(manifest, args.comparisons, source)
    else:
        benchmark(manifest, args.repeats, args.workers, source)


if __name__ == "__main__":
    main()
