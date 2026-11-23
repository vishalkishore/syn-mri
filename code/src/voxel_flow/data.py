"""Lossless voxel targets, cohort joins, and report-conditioned batches."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.voxel_flow.config import CACHE_FORMAT
from src.voxel_flow.config import NORMALIZATION
from util.mri import load_normalized_mri

CONTRASTS = ("unknown", "noncontrast", "postcontrast")


def validate_affine(affine):
    affine = np.asarray(affine)

    if (
        affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or not np.allclose(affine[3], [0, 0, 0, 1])
        or abs(np.linalg.det(affine[:3, :3])) < 1e-10
    ):
        raise ValueError("Invalid MRI affine")


def save_voxel_target(path, target, affine, source_median):
    target = np.asarray(target, dtype=np.float32)

    validate_affine(affine)

    if target.ndim != 3 or min(target.shape) < 1 or not np.isfinite(target).all():
        raise ValueError("Invalid voxel target")

    if not np.isfinite(source_median) or source_median <= 0:
        raise ValueError("Invalid source median")

    flat = target.ravel(order="C")
    support = flat != 0

    np.savez(
        path,
        values=flat[support],
        support_bits=np.packbits(support, bitorder="little"),
        shape=np.asarray(target.shape, dtype=np.int32),
        affine=np.asarray(affine, dtype=np.float64),
        source_median=np.asarray(source_median, dtype=np.float32),
    )


def load_voxel_target(path, expected_shape=None):
    with np.load(path, allow_pickle=False) as cache:
        shape_array = cache["shape"]

        if (
            shape_array.shape != (3,)
            or shape_array.dtype.kind not in "iu"
            or (shape_array <= 0).any()
        ):
            raise ValueError("Invalid compact voxel shape")

        shape = tuple(int(x) for x in shape_array)

        if expected_shape is not None and shape != tuple(expected_shape):
            raise ValueError("Wrong compact voxel target shape")

        count = int(np.prod(shape))
        bits = cache["support_bits"]

        if bits.dtype != np.uint8 or bits.shape != ((count + 7) // 8,):
            raise ValueError("Invalid or truncated support bit array")

        if count % 8 and int(bits[-1]) >> (count % 8):
            raise ValueError("Nonzero unused support tail bits")

        support = np.unpackbits(bits, count=count, bitorder="little").astype(bool)
        values = cache["values"]

        if (
            values.dtype != np.float32
            or values.ndim != 1
            or support.sum() != len(values)
            or not np.isfinite(values).all()
            or (values == 0).any()
        ):
            raise ValueError("Invalid compact voxel values")

        affine = cache["affine"].copy()

        validate_affine(affine)

        scale_array = cache["source_median"]

        if scale_array.shape != () or scale_array.dtype != np.float32:
            raise ValueError("Invalid source median encoding")

        scale = float(scale_array)

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid source median")

        target = np.zeros(count, dtype=np.float32)
        target[support] = values

    return target.reshape(shape), affine, scale


def load_brain_mask(path, shape, affine):
    """Load the independent anatomical mask; never infer it from MRI intensity."""

    image = nib.load(path)
    mask = np.asarray(image.dataobj)

    if mask.shape != tuple(shape) or not np.allclose(
        image.affine, affine, rtol=0, atol=1e-5
    ):
        raise ValueError(f"MRI/mask geometry mismatch: {path}")

    if not np.isin(mask, [0, 1]).all():
        raise ValueError(f"Brain mask must be binary: {path}")

    mask = mask.astype(bool)

    if not mask.any() or mask.all():
        raise ValueError(f"Brain mask must contain brain and outside voxels: {path}")

    return mask


def normalize_by_brain_median(image_path, mask_path):
    volume = load_normalized_mri(image_path)
    image = nib.load(image_path)
    mask_image = nib.load(mask_path)
    mask = np.asarray(mask_image.dataobj)

    if mask.shape != volume.shape or not np.allclose(
        mask_image.affine, image.affine, rtol=0, atol=1e-5
    ):
        raise ValueError(f"MRI/mask geometry mismatch: {image_path}")

    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Brain mask must be binary")

    positive = volume[(mask == 1) & (volume > 0)]

    if positive.size == 0:
        raise ValueError("Empty positive brain intensities")

    median = np.float32(np.median(positive))

    if not np.isfinite(median) or median <= 0:
        raise ValueError("Invalid positive brain median")

    return volume / median, image.affine.copy(), median


def validate_cohort(frame, report_studies, exploratory=False):
    required = {
        "patient_uid",
        "study_uid",
        "series_id",
        "split",
        "image_path",
        "status",
        "training_accepted",
        "classified_modality",
        "contrast",
    }

    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Nonempty MRI manifest requires columns {sorted(required)}")

    if (
        frame[list(required)].isna().any().any()
        or frame[list(required)].eq("").any().any()
    ):
        raise ValueError("Missing required MRI metadata")

    if frame.duplicated(["study_uid", "series_id"]).any():
        raise ValueError("Duplicate study/series")

    if frame.groupby("patient_uid").split.nunique().gt(1).any():
        raise ValueError("Patient split leakage")

    if frame.groupby("study_uid")[["patient_uid", "split"]].nunique().gt(1).any().any():
        raise ValueError("Study patient/split mismatch")

    if not frame.split.isin(["train", "validation", "test"]).all():
        raise ValueError("Unknown split")

    if not frame.classified_modality.eq("T1w").all():
        raise ValueError("Expected T1w modality")

    if not frame.contrast.isin(CONTRASTS).all():
        raise ValueError("Missing or invalid contrast metadata")

    accepted = frame.training_accepted.astype(str).str.casefold()

    if exploratory:
        if not (frame.status.eq("pending_review").all() and accepted.eq("false").all()):
            raise ValueError("Exploratory cohort must remain pending_review/unaccepted")
    elif not accepted.eq("true").all():
        raise ValueError("Reviewed training requires accepted scans")

    if (
        "visual_review" in frame
        and exploratory
        and not frame.visual_review.eq("pending").all()
    ):
        raise ValueError("Inconsistent exploratory visual review flags")

    keys = ["study_uid", "patient_uid", "split"]

    if (
        not set(keys).issubset(report_studies)
        or report_studies.study_uid.duplicated().any()
    ):
        raise ValueError("Invalid report-cache study metadata")

    joined = frame[keys].merge(
        report_studies[keys],
        on="study_uid",
        how="left",
        suffixes=("_mri", "_report"),
        validate="many_to_one",
    )

    for key in ("patient_uid", "split"):
        if not joined[f"{key}_mri"].eq(joined[f"{key}_report"]).all():
            raise ValueError(f"Missing report or report-cache {key} mismatch")


def read_report_studies(root):
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text())
    tokens = np.load(root / "tokens.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(root / "offsets.npy", mmap_mode="r", allow_pickle=False)
    studies = pd.read_csv(root / "studies.csv", dtype=str, keep_default_na=False)

    if (
        metadata["hidden_size"] != 768
        or tokens.ndim != 2
        or tokens.shape[1] != 768
        or tokens.dtype != np.float32
        or offsets.dtype != np.int64
        or offsets.shape != (len(studies) + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(tokens)
        or (np.diff(offsets) <= 0).any()
    ):
        raise ValueError("Invalid frozen report token cache")

    return studies


def read_cache_manifest(path, report_root=None):
    path = Path(path).resolve()
    metadata = json.loads((path.parent / "metadata.json").read_text())

    if (
        metadata.get("format") != CACHE_FORMAT
        or metadata.get("normalization") != NORMALIZATION
    ):
        raise ValueError("Incompatible voxel cache format or normalization")

    if metadata.get("shape") != [240, 240, 168]:
        raise ValueError("Incompatible native voxel geometry")

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    reports = read_report_studies(report_root or metadata["report_token_cache"])

    validate_cohort(frame, reports, metadata["exploratory_pending_review"])

    for value in frame.voxel_cache_path:
        relative = Path(value)

        if relative.is_absolute() or not (
            path.parent / relative
        ).resolve().is_relative_to(path.parent):
            raise ValueError("Voxel cache paths must be relative to cache root")

    return frame, metadata


class VoxelReportDataset(Dataset):
    def __init__(
        self,
        frame,
        cache_root,
        report_features,
        native_shape=(240, 240, 168),
        dataset_root=None,
        include_brain_mask=False,
    ):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(cache_root)
        self.report_features = report_features
        self.shape = tuple(native_shape)
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.include_brain_mask = include_brain_mask

        if include_brain_mask and self.dataset_root is None:
            raise ValueError("Mask supervision requires dataset_root")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]

        volume, affine, scale = load_voxel_target(
            self.root / row.voxel_cache_path, self.shape
        )

        if "target_scale" in row and not np.isclose(
            scale, float(row.target_scale), rtol=1e-7
        ):
            raise ValueError("Target scale differs from manifest")

        report = self.report_features.index[row.study_uid]
        start = int(self.report_features.offsets[report])
        stop = int(self.report_features.offsets[report + 1])
        tokens = self.report_features.tokens[start:stop]

        if not torch.isfinite(tokens).all():
            raise ValueError(f"Nonfinite report tokens: {row.study_uid}")

        supervision = {}

        if self.include_brain_mask:
            mask_path = row.get("mask_path", "")

            if not mask_path:
                raise ValueError(
                    f"Missing anatomical mask_path: {row.study_uid}/{row.series_id}"
                )

            mask = load_brain_mask(self.dataset_root / mask_path, self.shape, affine)

            if np.any(volume[~mask] != 0):
                raise ValueError(f"Nonzero target outside anatomical mask: {mask_path}")

            supervision["brain_mask"] = torch.from_numpy(mask[None])

        return {
            **supervision,
            "volume": torch.from_numpy(volume[None]),
            "affine": torch.from_numpy(affine),
            "condition": {
                "report_tokens": tokens,
                "contrast_id": torch.tensor(CONTRASTS.index(row.contrast)),
            },
            "case_id": f"{row.study_uid}/{row.series_id}",
            **{key: row[key] for key in ("patient_uid", "study_uid", "series_id")},
        }


def collate_batch(batch):
    tokens = [item["condition"]["report_tokens"] for item in batch]
    supervision = {}

    if "brain_mask" in batch[0]:
        supervision["brain_mask"] = torch.stack([item["brain_mask"] for item in batch])

    return {
        **supervision,
        "volume": torch.stack([item["volume"] for item in batch]),
        "affine": torch.stack([item["affine"] for item in batch]),
        "condition": {
            "report_tokens": pad_sequence(tokens, batch_first=True),
            "report_mask": pad_sequence(
                [torch.ones(len(x), dtype=torch.bool) for x in tokens], batch_first=True
            ),
            "contrast_id": torch.stack(
                [item["condition"]["contrast_id"] for item in batch]
            ),
        },
        **{
            key: [item[key] for item in batch]
            for key in ("case_id", "patient_uid", "study_uid", "series_id")
        },
    }
