"""Select the 5,000-scan voxel-flow cohort: 4,500 train and 500 validation scans.

Sampling is without replacement inside the existing patient-level splits, so no
patient moves between splits. Test scans are never selected.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def select_cohort(frame, train_scans, validation_scans, train_seed, validation_seed):
    parts = []

    for split, count, seed in [
        ("train", train_scans, train_seed),
        ("validation", validation_scans, validation_seed),
    ]:
        available = frame[frame.split == split]

        if len(available) < count:
            raise ValueError(
                f"Split {split} has {len(available)} scans; {count} requested"
            )

        parts.append(available.sample(n=count, replace=False, random_state=seed))

    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input", type=Path, default=Path("runs/reports/report_inputs.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("cache/voxel_flow_inputs/manifest.csv")
    )
    parser.add_argument("--train-scans", type=int, default=4500)
    parser.add_argument("--validation-scans", type=int, default=500)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=43)

    args = parser.parse_args()

    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False)

    selected = select_cohort(
        frame,
        args.train_scans,
        args.validation_scans,
        args.train_seed,
        args.validation_seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite cohort: {args.output}")

    selected.to_csv(args.output, index=False)

    print(
        json.dumps(
            {
                "scans": len(selected),
                "patients": selected.patient_uid.nunique(),
                "split_counts": selected.split.value_counts().to_dict(),
                "output": str(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
