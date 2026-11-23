"""Join MRI series to reports and labels, then cache frozen BioMedCLIP report tokens.

Outputs in --output-dir:
- report_inputs.csv: retained T1 series with labels, report, and contrast.
- excluded.csv and report_inputs.schema.json.
- report_token_cache/: tokens.npy, pooled.npy, offsets.npy, studies.csv,
  metadata.json. The voxel-flow model reads this cache.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report_encoder import chunk_report
from src.report_encoder import load_biomedclip_text
from util.biomedclip_tokens import extract_report_tokens


def prepare_manifest(samples, labels, reports):
    if labels.study_uid.duplicated().any() or reports.study_uid.duplicated().any():
        raise ValueError("Study labels and reports must each have unique study IDs")

    columns = [column for column in labels.columns if column != "study_uid"]

    if len(columns) != 37:
        raise ValueError("Expected the 37 official pathology columns")

    labels = labels.copy()

    for column in columns:
        labels[column] = pd.to_numeric(labels[column], errors="raise")

    if not labels[columns].isin([0, 1]).all().all():
        raise ValueError("Pathology labels must be binary")

    # Preprocessing marks scans it could not standardize as held; they have no image.
    is_held = samples.get("status", pd.Series("", index=samples.index)).eq("held")
    held = samples[is_held].copy()
    held["exclusion_reason"] = "preprocessing_held"
    samples = samples[~is_held]

    suffix = samples.series_id.str.extract(r"-(\d+)$")[0].fillna("0").astype(int)

    if suffix.eq(1).any():
        raise ValueError("Suffix -1 has no contrast rule; expected base or -2")

    excluded = samples[suffix > 2].copy()
    excluded["exclusion_reason"] = "series_suffix_greater_than_2"
    excluded = pd.concat([held, excluded], ignore_index=True)
    retained = samples[suffix <= 2].copy()
    unannotated = retained[~retained.study_uid.isin(labels.study_uid)].copy()
    unannotated["exclusion_reason"] = "missing_official_pathology_annotation"
    excluded = pd.concat([excluded, unannotated], ignore_index=True)

    retained = retained[retained.study_uid.isin(labels.study_uid)].copy()

    retained = retained.drop(
        columns=["bands_path", "a2_regions_path", "contrast"], errors="ignore"
    )

    retained = retained.merge(labels, on="study_uid", validate="many_to_one")

    retained = retained.merge(
        reports[["study_uid", "report"]],
        on="study_uid",
        how="left",
        validate="many_to_one",
    )

    if retained.report.isna().any() or retained.report.str.strip().eq("").any():
        raise ValueError("Missing or empty report for a retained MRI")

    retained[columns] = retained[columns].astype(np.uint8)
    retained["no_label"] = retained[columns].sum(axis=1).eq(0).astype(np.uint8)
    # This historical cohort uses a user-verified suffix rule, not metadata contrast.
    post = retained.series_id.str.endswith("-2")
    retained["pre_contrast"] = (~post).astype(np.uint8)
    retained["post_contrast"] = post.astype(np.uint8)
    retained["contrast"] = np.where(post, "postcontrast", "noncontrast")
    retained["contrast_label_source"] = "user-verified series suffix: base=pre, -2=post"

    if retained.groupby("patient_uid").split.nunique().gt(1).any():
        raise ValueError("Patient split leakage")

    if retained.groupby("study_uid").patient_uid.nunique().gt(1).any():
        raise ValueError("Study maps to multiple patients")

    return retained, excluded, columns


def cache_report_tokens(reports, encoder_config, output_dir, device):
    """Store every valid contextual token of every report chunk, without padding."""

    encoder_config = dict(encoder_config, device=device)
    model, tokenizer = load_biomedclip_text(encoder_config)

    lengths = []

    for report in tqdm(reports.report, desc="Report token counts"):
        chunks = chunk_report(report, tokenizer)

        lengths.append(sum(len(tokenizer.encode(chunk)) for chunk in chunks))

    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    output_dir.mkdir(parents=True, exist_ok=False)

    tokens = np.lib.format.open_memmap(
        output_dir / "tokens.npy",
        mode="w+",
        dtype=np.float32,
        shape=(int(offsets[-1]), 768),
    )

    pooled = np.empty((len(reports), 512), dtype=np.float32)

    for index, row in enumerate(
        tqdm(reports.itertuples(), total=len(reports), desc="Frozen BioMedCLIP tokens")
    ):
        result = extract_report_tokens(model, tokenizer, row.report, device)
        valid = result["tokens"][result["attention_mask"]].numpy()

        if len(valid) != lengths[index] or not np.isfinite(valid).all():
            raise ValueError(f"Invalid contextual tokens: {row.study_uid}")

        tokens[offsets[index] : offsets[index + 1]] = valid
        pooled[index] = result["pooled"].numpy()

    tokens.flush()

    np.save(output_dir / "pooled.npy", pooled)
    np.save(output_dir / "offsets.npy", offsets)

    reports[["study_uid", "patient_uid", "split"]].to_csv(
        output_dir / "studies.csv", index=False
    )

    metadata = {
        "model": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "revision": encoder_config["biomedclip_revision"],
        "hidden_size": 768,
        "pooled_size": 512,
        "encoder_context": 256,
        "studies": len(reports),
        "valid_tokens": int(offsets[-1]),
        "dtype": "float32",
        "mask": (
            "Only tokenizer-valid tokens are packed; DataLoader restores padding "
            "masks. All chunks are retained, each independently encoded within "
            "256 tokens."
        ),
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/mr_rate_t1/source/pathology_labels/mrrate_labels.csv"),
    )
    parser.add_argument(
        "--reports", type=Path, default=Path("data/mr_rate_t1/reports.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/reports"))
    parser.add_argument(
        "--encoder-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/report_encoder.yaml",
    )
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    torch.set_num_threads(4)

    samples = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    labels = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
    reports = pd.read_csv(args.reports, dtype=str, keep_default_na=False)
    retained, excluded, columns = prepare_manifest(samples, labels, reports)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "report_inputs.csv"

    retained.to_csv(manifest_path, index=False)

    excluded.to_csv(args.output_dir / "excluded.csv", index=False)

    manifest_path.with_suffix(".schema.json").write_text(
        json.dumps(
            {
                "pathology_columns": columns,
                "binary_columns": columns
                + ["no_label", "pre_contrast", "post_contrast"],
                "definitions": (
                    "37 official study-level pathology labels; no_label=1 "
                    "only when all pathology labels are 0. Contrast flags "
                    "are scan-level, independent of pathology. Missing "
                    "annotations are excluded, not labeled negative."
                ),
            },
            indent=2,
        )
    )

    print(
        json.dumps(
            {
                "retained_series": len(retained),
                "retained_studies": retained.study_uid.nunique(),
                "split_counts": retained.split.value_counts().to_dict(),
                "excluded_series": len(excluded),
            }
        ),
        flush=True,
    )

    # One report per study; every series of a study shares the same report.
    studies = retained[["study_uid", "patient_uid", "split", "report"]]
    studies = studies.drop_duplicates("study_uid").reset_index(drop=True)

    cache_report_tokens(
        studies,
        yaml.safe_load(args.encoder_config.read_text()),
        args.output_dir / "report_token_cache",
        args.device,
    )


if __name__ == "__main__":
    main()
