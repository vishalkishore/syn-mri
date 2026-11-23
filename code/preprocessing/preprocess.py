"""MR-RATE: all eligible T1 series, metadata contrast labels, RAS brain masking.

Plan metadata candidates, then process masked RAS intermediates and rigidly
standardized axial model inputs with maximum-intensity normalization and training Haar statistics.
Run `python preprocess.py --help` for processing and optional pilot tools.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unicodedata
import zipfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from multiprocessing import get_context
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from filelock import FileLock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from util.haar import fit_haar_statistics
from util.haar import normalize_mri

REPO = "Forithmus/MR-RATE"
VERSION = "1.1"
IMAGE_INTERPOLATION_ORDER = (
    1  # Frozen after matched linear/cubic pilot; masks always nearest.
)

KEY = ["study_uid", "series_id"]
FIELDS = [
    "patient_uid",
    "study_uid",
    "series_id",
    "classified_modality",
    "array_shape",
    "array_spacing_mm",
    "SeriesDescription",
    "ProtocolName",
    "StudyDescription",
    "BodyPartExamined",
    "SeriesNumber",
    "acquisition_plane",
    "MRAcquisitionType",
    "is_center_modality",
    "is_derived",
    "is_localizer",
    "is_subtraction",
]

REPORT_FIELDS = [
    "study_uid",
    "report",
    "clinical_information",
    "technique",
    "findings",
    "impression",
]

POST = re.compile(
    r"\b(?:post\s*contrast|after\s+(?:iv\s+)?contrast|"
    r"contrast\s*enhanced|with\s+(?:iv\s+)?contrast)\b|"
    r"\+\s*c\b|"
    r"\bpost\b(?!\s+(?:op|operative|surgery|surgical|resection|"
    r"treatment|processed|processing|mortem)\b)"
)

PRE = re.compile(
    r"\b(?:pre\s*contrast|non\s*contrast|without\s+(?:iv\s+)?contrast|"
    r"no\s+(?:iv\s+)?contrast)\b"
)


def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()

        os.fsync(stream.fileno())

    os.replace(temporary, path)


def write_csv(path, frame):
    temporary = path.with_suffix(".csv.tmp")

    frame.to_csv(temporary, index=False)

    os.replace(temporary, path)


def text(value):
    normalized = unicodedata.normalize("NFKD", str(value).casefold().replace("ı", "i"))

    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )

    return re.sub(r"[-_\s]+", " ", normalized).strip()


def contrast_label(description, protocol):
    """Explicit phrases and observed MR-RATE +C/post series naming cues."""

    pre, post, evidence = False, False, []
    ambiguous = False

    for field, value in [
        ("SeriesDescription", description),
        ("ProtocolName", protocol),
    ]:
        normalized = text(value)

        # Appearance statements and negated instructions do not establish injection.
        if re.search(
            r"\b(?:no|without|not)\s+(?:iv\s+)?contrast\s+enhancement\b", normalized
        ):
            ambiguous = True

        if re.search(
            r"\bwith\s+and\s+without\s+(?:iv\s+)?contrast\b|\bpre\s*(?:contrast\s*)?(?:and|/|&)\s*post\s*contrast\b",
            normalized,
        ):
            ambiguous = True

        if re.search(
            r"\b(?:pre|post)\s*contrast\s+(?:not|cancelled|canceled|omitted)\b",
            normalized,
        ):
            ambiguous = True

        if re.search(
            r"\b(?:not|no|without)\s+(?:a\s+)?(?:pre|post)\s*contrast\b", normalized
        ):
            ambiguous = True

        if re.search(r"\b(?:not|no|without)\s*(?:\+\s*c\b|post\b)", normalized):
            ambiguous = True

        if re.search(r"\+\s*c\b\s+(?:not|cancelled|canceled|omitted)\b", normalized):
            ambiguous = True

        has_pre, has_post = bool(PRE.search(normalized)), bool(POST.search(normalized))
        pre |= has_pre
        post |= has_post

        if has_pre or has_post or ambiguous:
            evidence.append({"field": field, "value": value})

    if ambiguous or (pre and post):
        return "unknown", "conflicting_or_negated_text", evidence

    if pre:
        return "noncontrast", "explicit_metadata_text", evidence

    if post:
        return "postcontrast", "explicit_metadata_text", evidence

    return "unknown", "no_explicit_metadata_text", evidence


def brain_region(row, report):
    """Conservative examination-text routing, not a clinical anatomy verification."""

    evidence = []
    brain, other = False, False

    values = [
        (name, row.get(name, "")) for name in ["BodyPartExamined", "StudyDescription"]
    ]

    values.append(("technique", report.get("technique", "")))

    # Only a report's first non-empty line is considered as an examination title.
    lines = [
        line.strip() for line in report.get("report", "").splitlines() if line.strip()
    ]

    if lines:
        values.append(("report_first_line", lines[0]))

    for field, value in values:
        normalized = text(value)

        if field in {"BodyPartExamined", "StudyDescription"}:
            is_brain = bool(
                re.search(
                    r"\b(?:brain|head|cranial|cranium|intracranial|cerebral|beyin|kranial)\b",
                    normalized,
                )
            )

            is_other = bool(
                re.search(
                    r"\b(?:spine|spinal|cervical|thoracic|lumbar|orbit|orbits|whole body|servikal|torakal|lomber|omurga|orbita|orbital)\b",
                    normalized,
                )
            )
        else:
            is_brain = bool(
                re.search(
                    r"\b(?:brain|cranial|cerebral|beyin|kranial)\s+(?:mri|mr|emar|magnetic resonance)\b|"
                    r"\b(?:mri|mr|emar|magnetic resonance(?: imaging)?)\s*[,^:]?\s*(?:of\s+)?(?:the\s+)?(?:brain|beyin|kranial)\b",
                    normalized,
                )
            )

            is_other = bool(
                re.search(
                    r"\b(?:spine|spinal|cervical|thoracic|lumbar|orbit|orbits|servikal|torakal|lomber|omurga|orbita|orbital)\s+(?:mri|mr|emar)\b|"
                    r"\b(?:mri|mr|emar)\s*[,^:]?\s*(?:of\s+)?(?:the\s+)?(?:spine|cervical|thoracic|lumbar|orbits|servikal|torakal|lomber|omurga|orbita|orbital)\b",
                    normalized,
                )
            )

        brain |= is_brain
        other |= is_other

        if is_brain or is_other:
            evidence.append({"field": field, "value": value})

    region = (
        "brain"
        if brain and not other
        else "nonbrain"
        if other and not brain
        else "unknown"
    )

    return region, evidence


def unique_rows(frame, keys, name):
    frame = frame.drop_duplicates().copy()
    duplicates = frame.duplicated(keys, keep=False)

    if duplicates.any():
        raise ValueError(
            f"Conflicting duplicate {name} keys: {frame.loc[duplicates, keys].head().to_dict('records')}"
        )

    return frame


def vector(value, size, integer=False):
    parsed = ast.literal_eval(value)

    if not isinstance(parsed, (list, tuple)) or len(parsed) != size:
        raise ValueError("invalid vector length")

    array = np.asarray(parsed, dtype=float)

    if not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError("vector must contain positive finite numbers")

    if integer and (array != np.floor(array)).any():
        raise ValueError("shape must contain integers")

    return array


def build_plan(source, max_spacing, min_spacing=0.5):
    metadata_parts, report_parts = [], []

    for path in sorted((source / "metadata").glob("batch*_metadata.csv")):
        frame = pd.read_csv(
            path, dtype=str, keep_default_na=False, usecols=lambda c: c in FIELDS
        )

        required = set(FIELDS[:6])

        if not required.issubset(frame.columns):
            raise ValueError(
                f"Missing required metadata columns in {path.name}: {required - set(frame.columns)}"
            )

        frame = frame[frame["classified_modality"].str.casefold() == "t1w"].copy()

        for field in FIELDS:
            if field not in frame:
                frame[field] = ""

        frame["batch"] = path.name.split("_")[0]

        metadata_parts.append(frame)

    for path in sorted((source / "reports").glob("batch*_reports.csv")):
        frame = pd.read_csv(
            path, dtype=str, keep_default_na=False, usecols=lambda c: c in REPORT_FIELDS
        )

        if "study_uid" not in frame:
            raise ValueError(f"Missing report study_uid in {path.name}")

        for field in REPORT_FIELDS:
            if field not in frame:
                frame[field] = ""

        report_parts.append(frame[REPORT_FIELDS])

    if not metadata_parts or not report_parts:
        raise ValueError(
            "Expected metadata/batch*_metadata.csv and reports/batch*_reports.csv"
        )

    series = unique_rows(pd.concat(metadata_parts, ignore_index=True), KEY, "series")

    reports = unique_rows(
        pd.concat(report_parts, ignore_index=True), ["study_uid"], "report"
    )

    splits = pd.read_csv(source / "splits.csv", dtype=str, keep_default_na=False)

    splits = unique_rows(
        splits[["patient_uid", "study_uid", "split"]], ["study_uid"], "split"
    )

    splits["split"] = (
        splits["split"].str.casefold().str.strip().replace({"val": "validation"})
    )

    if not splits["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("Invalid or missing split name")

    # Check the full split table before filtering series, so leakage stays visible.
    if splits.groupby("patient_uid")["split"].nunique().gt(1).any():
        raise ValueError("Patient leakage across splits")

    for frame, columns in [
        (series, ["patient_uid", *KEY]),
        (reports, ["study_uid"]),
        (splits, ["patient_uid", "study_uid"]),
    ]:
        if frame[columns].eq("").any().any():
            raise ValueError("Empty identity in input tables")

    if series.groupby("study_uid")["patient_uid"].nunique().gt(1).any():
        raise ValueError("One study maps to multiple patients")

    # Each study has one split; all retained series inherit that study assignment.
    series = series.merge(
        splits.rename(columns={"patient_uid": "split_patient_uid"}),
        on="study_uid",
        how="left",
        validate="many_to_one",
    )

    mismatch = series["split_patient_uid"].notna() & series["patient_uid"].ne(
        series["split_patient_uid"]
    )

    if mismatch.any():
        raise ValueError("Patient identity differs between metadata and splits")

    series = series.drop(columns="split_patient_uid").fillna("")
    report_map = reports.set_index("study_uid").to_dict("index")
    records = []

    for row in series.to_dict("records"):
        for field in ["patient_uid", "study_uid", "series_id", "batch"]:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", row[field]) or row[field] in {
                ".",
                "..",
            }:
                raise ValueError(f"Unsafe path component in {field}")

        reasons = []
        report = report_map.get(row["study_uid"], {})

        if not report or not any(
            report.get(field, "").strip()
            for field in ["report", "findings", "impression"]
        ):
            reasons.append("missing_report")

        if not row["split"]:
            reasons.append("missing_split")

        region, evidence = brain_region(row, report)
        row["region"] = region
        row["region_evidence"] = json.dumps(evidence)

        if region != "brain":
            reasons.append(
                "nonbrain" if region == "nonbrain" else "unknown_brain_region"
            )

        for field in ["is_derived", "is_localizer", "is_subtraction"]:
            flag = text(row[field])

            if flag in {"true", "1", "yes"}:
                reasons.append(field)
            elif flag not in {"", "false", "0", "no"}:
                reasons.append(f"invalid_{field}")

        try:
            spacing = vector(row["array_spacing_mm"], 3)

            vector(row["array_shape"], 3, integer=True)

            if spacing.min() < min_spacing:
                reasons.append("metadata_spacing_too_small")

            if spacing.max() > max_spacing:
                reasons.append("metadata_spacing_too_large")
        except (ValueError, TypeError, SyntaxError):
            reasons.append("invalid_metadata_geometry")

        label, reason, evidence = contrast_label(
            row["SeriesDescription"], row["ProtocolName"]
        )

        row.update(
            contrast=label,
            contrast_reason=reason,
            contrast_evidence=json.dumps(evidence),
        )

        row["plan_status"] = "candidate" if not reasons else "held"
        row["plan_reason"] = ";".join(reasons)

        records.append(row)

    # Zero candidates is a valid audit outcome, not a reason to relax filters.
    return pd.DataFrame(
        records,
        columns=list(series.columns)
        + [
            "region",
            "region_evidence",
            "contrast",
            "contrast_reason",
            "contrast_evidence",
            "plan_status",
            "plan_reason",
        ],
    ), reports


def checked_image(path):
    image = nib.load(path)

    if len(image.shape) == 4 and image.shape[-1] == 1:
        image = nib.squeeze_image(image)

    if len(image.shape) != 3:
        raise ValueError("not_a_single_3d_volume")

    unit = image.header.get_xyzt_units()[0]
    factors = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}

    if unit not in factors:
        raise ValueError("unknown_spatial_unit")

    factor = factors[unit]
    affine = image.affine.copy()
    # Convert physical coordinates to millimeters without changing voxel values.
    affine[:3, :] *= factor

    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("invalid_affine")

    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)

    if not qcode and not scode:
        raise ValueError("no_coded_spatial_transform")

    corners = np.array(
        [
            [x, y, z, 1]
            for x in [0, image.shape[0] - 1]
            for y in [0, image.shape[1] - 1]
            for z in [0, image.shape[2] - 1]
        ]
    )

    # Compare physical positions across the FOV, where small rotations accumulate.
    if qcode and scode:
        distances = np.linalg.norm(
            (corners @ (qform - sform).T)[:, :3] * factor, axis=1
        )

        if not np.isfinite(distances).all() or distances.max() > 0.01:
            raise ValueError("qform_sform_conflict")

    spacing = nib.affines.voxel_sizes(affine)

    if not np.allclose(
        spacing, np.asarray(image.header.get_zooms()[:3]) * factor, rtol=1e-4, atol=1e-4
    ):
        raise ValueError("header_affine_spacing_conflict")

    return image, affine, spacing


DIRECTION_MARGIN_MIN = (
    0.1  # Conservative pilot guard, not a clinically validated cutoff.
)

ACQUISITION_DIRECTIONS = {
    "axial": ("R", "A", "S"),
    "coronal": ("R", "S", "A"),
    "sagittal": ("A", "S", "R"),
}


def acquisition_orientation(affine, row, slice_axis=None):
    """Resolve native acquisition axes, never using dimension lengths.

    Positive output directions: axial R/A/S, coronal R/S/A, sagittal A/S/R.
    A bare oblique label cannot establish an acquisition slice axis.
    """

    planes = {"axi": "axial", "cor": "coronal", "sag": "sagittal", "obl": "oblique"}
    label = text(row.get("acquisition_plane", ""))
    plane = planes.get(label, label)

    match = re.fullmatch(
        r"t1w-raw-(axi|cor|sag|obl)(?:-\d+)?", row.get("series_id", "")
    )

    id_plane = planes[match[1]] if match else ""

    evidence = {
        "metadata_plane": plane,
        "series_id_plane": id_plane,
        "native_header_slice_axis": slice_axis,
    }

    if not plane:
        plane = id_plane

    if id_plane and plane != id_plane:
        raise ValueError("orientation_unresolved: acquisition_plane_conflict")

    start = nib.orientations.io_orientation(affine)
    directions = ACQUISITION_DIRECTIONS

    if plane not in directions:
        raise ValueError("orientation_unresolved: missing_or_oblique_acquisition_plane")

    target = nib.orientations.axcodes2ornt(directions[plane])
    expected = int(np.flatnonzero(start[:, 0] == target[2, 0])[0])
    unit_axes = affine[:3, :3] / nib.affines.voxel_sizes(affine)
    # Require a unique dominant physical direction for each native axis.
    dominant = np.abs(unit_axes).argmax(axis=0)
    components = np.sort(np.abs(unit_axes), axis=0)
    margin = components[-1] - components[-2]

    evidence.update(
        direction_margin=margin.tolist(),
        direction_margin_min=DIRECTION_MARGIN_MIN,
        plane_labels_independent_confirmation=False,
    )

    if (
        (margin < DIRECTION_MARGIN_MIN).any()
        or not np.array_equal(dominant, start[:, 0])
        or len(set(dominant)) != 3
    ):
        raise ValueError("orientation_unresolved: ambiguous_affine_axes")

    if slice_axis is not None and slice_axis != expected:
        raise ValueError("orientation_unresolved: header_plane_conflict")

    evidence["native_slice_axis"] = expected

    evidence["slice_axis_basis"] = (
        "header_and_plane_agree"
        if slice_axis is not None
        else "plane_and_affine_inference"
    )

    return plane, evidence, nib.orientations.ornt_transform(start, target)


def check_orientation_coordinates(shape, affine, transform, output_affine):
    """Track original indices through array operations independently of affine mapping."""

    coordinates = []

    for axis in range(3):
        dimensions = [1, 1, 1]
        dimensions[axis] = shape[axis]
        indices = np.broadcast_to(np.arange(shape[axis]).reshape(dimensions), shape)

        coordinates.append(nib.orientations.apply_orientation(indices, transform))

    output_shape = coordinates[0].shape
    error = 0.0

    for x in [0, output_shape[0] // 2, output_shape[0] - 1]:
        for y in [0, output_shape[1] // 2, output_shape[1] - 1]:
            for z in [0, output_shape[2] // 2, output_shape[2] - 1]:
                index = (x, y, z)
                original = [array[index] for array in coordinates] + [1]

                error = max(
                    error,
                    float(
                        np.max(np.abs(affine @ original - output_affine @ [x, y, z, 1]))
                    ),
                )

    if error > 1e-4:
        raise ValueError(
            "orientation_unresolved: independent_world_position_check_failed"
        )

    return error


def audit_acquisition(root, plan=None, export=True):
    """Write small layout records for existing RAS images; do not rewrite MRIs."""

    if plan is None:
        plan = pd.read_csv(root / "plan.csv", dtype=str, keep_default_na=False)

    rows = []

    for row in plan.to_dict("records"):
        stem = f"{row['study_uid']}_{row['series_id']}"
        path = root / "records" / row["study_uid"] / f"{stem}.json"

        if not path.exists():
            continue

        record = {
            key: row[key]
            for key in [
                "patient_uid",
                "study_uid",
                "series_id",
                "split",
                "acquisition_plane",
            ]
        }

        record.update(
            image_path="",
            orientation_status="orientation_unresolved",
            training_accepted=False,
            visual_coverage_status="not_reviewed",
        )

        try:
            saved = json.loads(path.read_text())

            required = {
                "status",
                "image_path",
                "native_shape",
                "native_affine_mm",
                "output_shape",
                "output_affine_mm",
                "orientation_transform",
            }

            missing = required - saved.keys()

            if missing:
                raise ValueError(f"missing_saved_record_fields: {sorted(missing)}")

            if saved["status"] not in {"complete", "review"}:
                continue

            record.update(
                image_path=saved["image_path"],
                original_shape=saved["native_shape"],
                original_affine=saved["native_affine_mm"],
            )

            native_affine = np.array(saved["native_affine_mm"])

            plane, evidence, native_transform = acquisition_orientation(
                native_affine, row, saved.get("native_header_slice_axis")
            )

            image = nib.load(root / saved["image_path"])

            if image.shape != tuple(saved["output_shape"]) or not np.allclose(
                image.affine, saved["output_affine_mm"], atol=1e-4, rtol=0
            ):
                raise ValueError("orientation_unresolved: saved_geometry_mismatch")

            ras_transform = np.array(saved["orientation_transform"])

            tracked_affine = native_affine @ nib.orientations.inv_ornt_aff(
                ras_transform, saved["native_shape"]
            )

            if not np.allclose(tracked_affine, image.affine, atol=1e-4, rtol=0):
                raise ValueError("orientation_unresolved: native_transform_mismatch")

            target = nib.orientations.axcodes2ornt(ACQUISITION_DIRECTIONS[plane])

            transform = nib.orientations.ornt_transform(
                nib.orientations.io_orientation(image.affine), target
            )

            data = image.get_fdata(dtype=np.float32, caching="unchanged")
            output = nib.orientations.apply_orientation(data, transform)

            reverse = nib.orientations.ornt_transform(
                target, nib.orientations.io_orientation(image.affine)
            )

            if not np.isfinite(data).all() or not np.array_equal(
                nib.orientations.apply_orientation(output, reverse), data
            ):
                raise ValueError(
                    "orientation_unresolved: reverse_or_finite_check_failed"
                )

            mapping = nib.orientations.inv_ornt_aff(transform, image.shape)
            output_affine = image.affine @ mapping

            check_orientation_coordinates(
                image.shape, image.affine, transform, output_affine
            )

            order = np.argsort(transform[:, 0])

            if not np.allclose(
                nib.affines.voxel_sizes(output_affine),
                nib.affines.voxel_sizes(image.affine)[order],
            ):
                raise ValueError("orientation_unresolved: spacing_check_failed")

            record.update(
                original_shape=saved["native_shape"],
                original_affine=native_affine.tolist(),
                acquisition_plane=plane,
                acquisition_evidence=evidence,
                native_orientation_transform=native_transform.tolist(),
                axis_order=order.tolist(),
                axis_flips=(transform[order, 1] < 0).tolist(),
                orientation_transform=transform.tolist(),
                input_shape=list(image.shape),
                output_shape=list(output.shape),
                output_affine=output_affine.tolist(),
                output_spacing_mm=nib.affines.voxel_sizes(output_affine).tolist(),
                orientation_status="header_supported"
                if evidence["slice_axis_basis"] == "header_and_plane_agree"
                else "inferred",
                checks="reverse_array_independent_coordinates_spacing_finite_passed",
                nonzero_voxels=int(np.count_nonzero(output)),
                nonzero_voxels_per_slice=np.count_nonzero(output, axis=(0, 1)).tolist(),
                source_mask_touches_edge=saved.get("mask_touches_image_edge"),
                source_mask_available=False,
            )

            del data, output
        except (
            ValueError,
            OSError,
            EOFError,
            nib.filebasedimages.ImageFileError,
        ) as error:
            record["reason"] = f"{type(error).__name__}: {error}"

        destination = root / "acquisition_layout" / row["study_uid"] / f"{stem}.json"

        destination.parent.mkdir(parents=True, exist_ok=True)

        write_json(destination, record)

        rows.append(record)

    frame = pd.DataFrame(rows)

    if not export:
        return frame

    write_csv(root / "acquisition_audit.csv", frame)

    pilot = (
        frame[frame["orientation_status"].isin(["inferred", "header_supported"])]
        if len(frame)
        else frame
    )

    write_csv(root / "acquisition_pilot.csv", pilot)

    accepted = frame[frame["training_accepted"]] if len(frame) else frame
    # Retain the existing technical/brain-edge acceptance gate.
    samples = pd.read_csv(root / "samples.csv", dtype=str, keep_default_na=False)

    accepted = (
        samples.merge(
            accepted,
            on=["patient_uid", "study_uid", "series_id", "split", "image_path"],
            suffixes=("", "_layout"),
        )
        if len(accepted)
        else samples.iloc[:0]
    )

    write_csv(root / "acquisition_samples.csv", accepted)

    print(
        json.dumps(
            {
                "audited": len(frame),
                "accepted": len(accepted),
                "orientation_counts": frame.orientation_status.value_counts().to_dict()
                if len(frame)
                else {},
            },
            indent=2,
        )
    )


def axial_mask_grid(mask, affine, spacing=None):
    """Variable RAS grid covering actual transformed mask voxels plus 2 mm.

    spacing is the native spacing mapped to closest RAS axes BEFORE registration.
    Voxel footprints, not only centres, are included in the coverage calculation.
    """

    count = int(mask.sum())

    if not count:
        raise ValueError("empty_brain_mask")

    if spacing is None:
        order = np.argsort(nib.orientations.io_orientation(affine)[:, 0])
        spacing = nib.affines.voxel_sizes(affine)[order]

    spacing = np.asarray(spacing, dtype=float)
    center = np.empty(3)

    for axis in range(3):
        other_axes = tuple(index for index in range(3) if index != axis)
        weights = mask.sum(axis=other_axes)
        center[axis] = np.dot(np.arange(mask.shape[axis]), weights) / count

    center_world = nib.affines.apply_affine(affine, center)
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)

    for z in range(mask.shape[2]):
        x, y = np.nonzero(mask[:, :, z])

        if not len(x):
            continue

        world = nib.affines.apply_affine(
            affine, np.column_stack((x, y, np.full(len(x), z)))
        )

        lower = np.minimum(lower, world.min(axis=0))
        upper = np.maximum(upper, world.max(axis=0))

    # Include voxel edges, since center-only bounds can truncate a rotated mask.
    footprint = np.abs(affine[:3, :3]).sum(axis=1) / 2

    half_extent = (
        np.maximum(center_world - lower, upper - center_world) + footprint + 2.0
    )

    shape = np.ceil(2 * half_extent / spacing).astype(int)
    target_affine = np.diag([*spacing, 1.0])
    target_affine[:3, 3] = center_world - spacing * (shape - 1) / 2

    if (shape * spacing / 2 < half_extent - 1e-8).any():
        raise ValueError("axial_target_loses_mask_coverage")

    return target_affine, center_world, tuple(int(v) for v in shape)


def padded_grid(affine, required_shape, target_shape):
    """Expand a grid with balanced background; preserve existing voxel coordinates."""

    extra = np.asarray(target_shape, dtype=int) - np.asarray(required_shape, dtype=int)

    if (extra < 0).any():
        raise ValueError(
            f"required_shape_exceeds_padding_target: {tuple(required_shape)} > {tuple(target_shape)}"
        )

    before = extra // 2
    after = extra - before
    padded_affine = affine.copy()
    # Added leading voxels shift indices; compensate to preserve world positions.
    padded_affine[:3, 3] -= affine[:3, :3] @ before

    return padded_affine, before, after


def exact_native_grid(data, mask, affine, padding_shape=None):
    """Exact RAS reorder, crop empty background with 2 mm margin, and zero pad."""

    input_shape = data.shape

    transform = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(affine),
        nib.orientations.axcodes2ornt(("R", "A", "S")),
    )

    data = nib.orientations.apply_orientation(data, transform)
    mask = nib.orientations.apply_orientation(mask, transform)
    output_affine = affine @ nib.orientations.inv_ornt_aff(transform, input_shape)
    spacing = nib.affines.voxel_sizes(output_affine)
    # Round outward to retain a 2 mm crop margin where the source FOV allows it.
    margin = np.ceil(2 / spacing).astype(int)
    start = np.empty(3, dtype=int)
    stop = np.empty(3, dtype=int)

    for axis in range(3):
        other_axes = tuple(index for index in range(3) if index != axis)
        occupied = np.flatnonzero(mask.any(axis=other_axes))
        start[axis] = max(0, occupied[0] - margin[axis])
        stop[axis] = min(data.shape[axis], occupied[-1] + margin[axis] + 1)

    slices = tuple(slice(int(lower), int(upper)) for lower, upper in zip(start, stop))
    data, mask = data[slices], mask[slices]
    output_affine[:3, 3] = nib.affines.apply_affine(output_affine, start)
    required = np.array(data.shape)
    before = after = np.zeros(3, dtype=int)

    if padding_shape is not None:
        output_affine, before, after = padded_grid(
            output_affine, required, padding_shape
        )

        widths = list(zip(before, after))
        data, mask = np.pad(data, widths), np.pad(mask, widths)

    return (
        nib.Nifti1Image(data, output_affine),
        nib.Nifti1Image(mask.astype(np.uint8), output_affine),
        required,
        before,
        after,
    )


def sitk_image(data, affine):
    """NiBabel xyz/RAS to SimpleITK zyx/LPS, without interpolation."""

    import SimpleITK as sitk

    spacing = nib.affines.voxel_sizes(affine)
    direction = np.diag([-1.0, -1.0, 1.0]) @ affine[:3, :3] / spacing

    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4, rtol=0):
        raise ValueError("rigid_registration_requires_unsheared_native_grid")

    image = sitk.GetImageFromArray(
        np.asarray(data, dtype=np.float32).transpose(2, 1, 0)
    )

    image.SetSpacing(tuple(spacing))
    image.SetOrigin(tuple(np.array([-1.0, -1.0, 1.0]) * affine[:3, 3]))
    image.SetDirection(tuple(direction.ravel()))

    return image


def rigid_to_template(data, affine, mask, template, template_mask):
    """Estimate six-DOF rigid motion; return native-RAS to template-RAS matrix."""

    import SimpleITK as sitk

    fixed_data = template.get_fdata(dtype=np.float32) * template_mask
    fixed = sitk_image(fixed_data, template.affine)
    moving = sitk_image(data, affine)
    fixed_mask = sitk.Cast(sitk_image(template_mask, template.affine), sitk.sitkUInt8)
    moving_mask = sitk.Cast(sitk_image(mask, affine), sitk.sitkUInt8)
    _, fixed_center, _ = axial_mask_grid(template_mask, template.affine)
    _, moving_center, _ = axial_mask_grid(mask, affine)
    signs = np.array([-1.0, -1.0, 1.0])
    initial = sitk.Euler3DTransform()

    initial.SetCenter(tuple(signs * fixed_center))
    initial.SetTranslation(tuple(signs * (moving_center - fixed_center)))

    registration = sitk.ImageRegistrationMethod()

    registration.SetNumberOfThreads(1)
    registration.SetMetricAsMattesMutualInformation(32)
    registration.SetMetricFixedMask(fixed_mask)
    registration.SetMetricMovingMask(moving_mask)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.15, 42)
    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetOptimizerAsRegularStepGradientDescent(
        2.0, 0.001, 200, gradientMagnitudeTolerance=1e-6
    )

    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial, inPlace=True)
    registration.Execute(fixed, moving)

    # SimpleITK returns fixed->moving LPS; invert and conjugate for moving->fixed RAS.
    inverse = initial.GetInverse()
    translation = np.array(inverse.TransformPoint((0.0, 0.0, 0.0)))
    matrix = np.eye(4)

    matrix[:3, :3] = np.column_stack(
        [
            np.array(inverse.TransformPoint(tuple(np.eye(3)[a]))) - translation
            for a in range(3)
        ]
    )

    matrix[:3, 3] = translation
    conversion = np.diag([-1.0, -1.0, 1.0, 1.0])
    matrix = conversion @ matrix @ conversion
    rotation = matrix[:3, :3]

    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("registration_transform_not_rigid")

    error = 0.0

    for point in [
        moving_center,
        affine[:3, 3],
        nib.affines.apply_affine(affine, np.array(data.shape) - 1),
    ]:
        expected = signs * np.array(inverse.TransformPoint(tuple(signs * point)))

        error = max(
            error,
            float(np.max(np.abs(expected - nib.affines.apply_affine(matrix, point)))),
        )

    if error > 1e-5:
        raise ValueError("rigid_LPS_RAS_coordinate_check_failed")

    info = {
        "native_to_template_ras": matrix.tolist(),
        "transform_type": "rigid_6DOF",
        "rotation_determinant": float(np.linalg.det(rotation)),
        "transform_point_error_mm": error,
        "final_metric": float(registration.GetMetricValue()),
        "optimizer_stop": registration.GetOptimizerStopConditionDescription(),
        "seed": 42,
        "threads": 1,
        "template_id": "MNI152NLin2009cAsym",
        "iteration_limit_reached": "Maximum number of iterations"
        in registration.GetOptimizerStopConditionDescription(),
    }

    return matrix, info


def timed_download(*args, **kwargs):
    from huggingface_hub import hf_hub_download

    started = time.perf_counter()
    path = Path(hf_hub_download(*args, **kwargs))

    write_json(
        path.with_suffix(".timing.json"),
        {
            "download_seconds": time.perf_counter() - started,
        },
    )

    return path


def study_archives(groups, work, revision, workers=8, archives=None):
    """Prefetch a bounded window of study archives; consume in stable order."""

    if archives is not None:
        for (batch, study), group in groups:
            yield batch, study, group, archives / "mri" / batch / f"{study}.zip"

        return

    os.environ["HF_HUB_DISABLE_XET"] = "1"

    with tempfile.TemporaryDirectory(dir=work) as directory:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = deque()
            iterator = iter(groups)
            exhausted = False

            while pending or not exhausted:
                while len(pending) < workers and not exhausted:
                    item = next(iterator, None)

                    if item is None:
                        exhausted = True

                        break

                    (batch, study), group = item
                    folder = Path(directory) / batch / study

                    folder.mkdir(parents=True)

                    future = pool.submit(
                        timed_download,
                        REPO,
                        f"mri/{batch}/{study}.zip",
                        repo_type="dataset",
                        revision=revision,
                        local_dir=folder,
                        token=True,
                    )

                    pending.append((batch, study, group, folder, future))

                if pending:
                    batch, study, group, folder, future = pending.popleft()

                    # The consumer must finish or retain the archive before advancing.
                    yield batch, study, group, Path(future.result())

                    shutil.rmtree(folder)


def archive_jobs(jobs, function, workers, work):
    """Keep at most workers archives alive until their process jobs finish.

    Downloads delete their temporary archives when advanced. Hard links keep
    each submitted archive available without copying its contents.
    """

    with tempfile.TemporaryDirectory(dir=work) as directory:
        # Spawn avoids inheriting locks from active download threads.
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
        ) as pool:
            pending = deque()

            for index, job in enumerate(jobs):
                retained = Path(directory) / f"{index}.zip"

                try:
                    os.link(job[0], retained)
                except OSError:
                    shutil.copyfile(job[0], retained)

                timing = Path(job[0]).with_suffix(".timing.json")

                if timing.exists():
                    shutil.copyfile(timing, retained.with_suffix(".timing.json"))

                future = pool.submit(function, (retained, *job[1:]))

                pending.append((future, retained))

                if len(pending) == workers:
                    future, retained = pending.popleft()
                    result = future.result()

                    retained.unlink()

                    yield result

            while pending:
                future, retained = pending.popleft()
                result = future.result()

                retained.unlink()

                yield result


def preprocess_archive_job(job):
    process_archive(*job)


def final_result(record_path, root, policy_hash):
    """Return a committed final result only when its policy and image match."""

    try:
        record = json.loads(record_path.read_text())

        if record.get("final_policy_hash") != policy_hash:
            return None

        if record["status"] == "held":
            return record

        if record["status"] != "pending_review":
            return None

        if sha256(root / record["mask_path"]) != record["mask_sha256"]:
            return None

        if sha256(root / record["image_path"]) != record["sha256"]:
            return None

        return record
    except (OSError, ValueError, KeyError):
        return None


def complete_archive_job(job):
    (
        archive,
        study,
        group,
        root,
        destination,
        rigid,
        shape,
        template,
        mask,
        output,
        final_hash,
        policy,
        policy_hash,
    ) = job

    started = time.perf_counter()

    process_archive(archive, group.to_dict("records"), root, policy, policy_hash)

    native_seconds = time.perf_counter() - started
    started = time.perf_counter()
    layouts = audit_acquisition(root, group, export=False)

    eligible = (
        layouts[layouts.orientation_status.isin(["inferred", "header_supported"])]
        if len(layouts)
        else layouts
    )

    selected = (
        group[group.series_id.isin(eligible.series_id)]
        if len(eligible)
        else group.iloc[:0]
    )

    records = standardize_archive(
        (
            archive,
            study,
            selected,
            root,
            destination,
            rigid,
            shape,
            template,
            mask,
            output,
            final_hash,
        )
    )

    # Persist terminal native/orientation failures so a restart does not redownload them.
    for row in group[~group.series_id.isin(selected.series_id)].to_dict("records"):
        stem = f"{study}_{row['series_id']}"
        native = json.loads((root / "records" / study / f"{stem}.json").read_text())

        record = {
            key: row[key] for key in ["patient_uid", "study_uid", "series_id", "split"]
        }

        record.update(
            status="failed" if native.get("status") == "failed" else "held",
            image_path="",
            training_accepted=False,
            reason=native.get("reason") or "acquisition_orientation_unresolved",
            final_policy_hash=final_hash,
        )

        write_json(destination / f"{stem}.json", record)

        records.append(record)

    timing_path = archive.with_suffix(".timing.json")

    timing = (
        json.loads(timing_path.read_text())
        if timing_path.exists()
        else {"download_seconds": None}
    )

    timing.update(
        study_uid=study,
        native_processing_seconds=native_seconds,
        final_processing_seconds=time.perf_counter() - started,
        series=len(group),
    )

    write_json(destination / f"{study}.timing.json", timing)

    return records


def standardize_archive(job):
    (
        archive_path,
        study,
        group,
        root,
        destination,
        rigid,
        padding_shape,
        template_path,
        template_mask_path,
        output_dir,
    ) = job[:10]

    final_policy_hash = job[10] if len(job) > 10 else None

    from nibabel.processing import resample_from_to

    records = []
    shape_tag = "native_spacing"

    if output_dir is None:
        import matplotlib.pyplot as plt

    if rigid:
        template = nib.load(template_path)
        template_mask = nib.load(template_mask_path).get_fdata() > 0

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)

        with zipfile.ZipFile(archive_path) as archive:
            for row in group.to_dict("records"):
                stem = f"{study}_{row['series_id']}"
                record_path = destination / f"{stem}.json"

                if final_policy_hash is not None:
                    saved_final = final_result(record_path, root, final_policy_hash)

                    if saved_final is not None:
                        records.append(saved_final)

                        continue

                started = time.perf_counter()

                image_path, mask_path = (
                    temporary / "native.nii.gz",
                    temporary / "mask.nii.gz",
                )

                try:
                    for member, path in [
                        (f"{study}/img/{stem}.nii.gz", image_path),
                        (f"{study}/seg/{stem}_brain-mask.nii.gz", mask_path),
                    ]:
                        with archive.open(member) as source, path.open("wb") as output:
                            shutil.copyfileobj(source, output)

                    native, affine, _ = checked_image(image_path)
                    mask, mask_affine, _ = checked_image(mask_path)

                    if native.shape != mask.shape or not np.allclose(
                        affine, mask_affine, atol=1e-4, rtol=0
                    ):
                        raise ValueError("pilot_native_mask_geometry_mismatch")

                    native_data = native.get_fdata(dtype=np.float32)
                    mask_data = mask.get_fdata(dtype=np.float32)

                    if (
                        not np.isfinite(native_data).all()
                        or not np.isin(mask_data, [0, 1]).all()
                    ):
                        raise ValueError("pilot_invalid_native_data_or_mask")

                    saved = json.loads(
                        (root / "records" / study / f"{stem}.json").read_text()
                    )

                    layout = json.loads(
                        (
                            root / "acquisition_layout" / study / f"{stem}.json"
                        ).read_text()
                    )

                    plane, evidence, native_transform = acquisition_orientation(
                        affine, row, native.header.get_dim_info()[2]
                    )

                    data = native_data * mask_data

                    expected = nib.orientations.apply_orientation(
                        data, native_transform
                    )

                    model_image = nib.load(root / saved["image_path"])

                    model_data = nib.orientations.apply_orientation(
                        model_image.get_fdata(dtype=np.float32),
                        np.array(layout["orientation_transform"]),
                    )

                    if not np.array_equal(expected, model_data):
                        raise ValueError("pilot_recorded_transform_differs_from_native")

                    model_affine = affine @ nib.orientations.inv_ornt_aff(
                        native_transform, native.shape
                    )

                    if nib.aff2axcodes(model_affine) != ACQUISITION_DIRECTIONS[plane]:
                        raise ValueError("pilot_wrong_target_directions")

                    reverse = nib.orientations.ornt_transform(
                        nib.orientations.io_orientation(model_affine),
                        nib.orientations.io_orientation(affine),
                    )

                    if not np.array_equal(
                        nib.orientations.apply_orientation(model_data, reverse), data
                    ):
                        raise ValueError("pilot_reverse_failed")

                    error = check_orientation_coordinates(
                        native.shape, affine, native_transform, model_affine
                    )

                    inside = (
                        nib.orientations.apply_orientation(mask_data, native_transform)
                        > 0
                    )

                    shape = np.array(model_data.shape)
                    spacing = nib.affines.voxel_sizes(model_affine)
                    native_spacing = nib.affines.voxel_sizes(affine)

                    spacing_axis_order = np.argsort(
                        nib.orientations.io_orientation(affine)[:, 0]
                    ).astype(int)

                    output_spacing = native_spacing[spacing_axis_order]
                    motion = np.eye(4)
                    registration_info = {}

                    if rigid:
                        print(stem, "estimating rigid transform", flush=True)

                        motion, registration_info = rigid_to_template(
                            data, affine, mask_data > 0, template, template_mask
                        )

                    # Move the grid geometry first; sample voxel values only once below.
                    aligned_affine = motion @ model_affine

                    if rigid:
                        target_affine, center_world, target_shape = axial_mask_grid(
                            inside, aligned_affine, output_spacing
                        )

                        required_shape = np.array(target_shape)
                        before = after = np.zeros(3, dtype=int)

                        if padding_shape is not None:
                            target_affine, before, after = padded_grid(
                                target_affine, required_shape, padding_shape
                            )

                            target_shape = padding_shape

                        target_shape = np.array(target_shape)
                        image = nib.Nifti1Image(model_data, aligned_affine)

                        mask_image = nib.Nifti1Image(
                            inside.astype(np.uint8), aligned_affine
                        )

                        target = (tuple(target_shape), target_affine)

                        resized = resample_from_to(
                            image, target, order=IMAGE_INTERPOLATION_ORDER
                        )

                        # Nearest-neighbor sampling keeps the mask binary.
                        resized_mask = resample_from_to(mask_image, target, order=0)
                    else:
                        _, center_world, _ = axial_mask_grid(
                            inside, aligned_affine, output_spacing
                        )

                        resized, resized_mask, required_shape, before, after = (
                            exact_native_grid(
                                model_data, inside, model_affine, padding_shape
                            )
                        )

                        target_affine = resized.affine
                        target_shape = np.array(resized.shape)

                    resized_data = resized.get_fdata(dtype=np.float32)
                    output_mask = resized_mask.get_fdata() > 0

                    if not output_mask.any() or any(
                        output_mask.take(0, axis=a).any()
                        or output_mask.take(-1, axis=a).any()
                        for a in range(3)
                    ):
                        raise ValueError("resampled_mask_empty_or_touches_grid_edge")

                    if not np.isfinite(resized_data).all() or nib.aff2axcodes(
                        resized.affine
                    ) != ("R", "A", "S"):
                        raise ValueError("invalid_axial_output")

                    # Reapply the mask to remove interpolated values outside the brain.
                    resized_data, intensity_maximum = normalize_mri(
                        resized_data * output_mask, output_mask
                    )

                    resized = nib.Nifti1Image(resized_data, target_affine)
                    final_image_path = destination / f"{stem}_ras.nii.gz"
                    pending = destination / f"{stem}_ras.pending.nii.gz"

                    try:
                        nib.save(resized, pending)

                        verified = nib.load(pending)

                        if not np.array_equal(
                            verified.get_fdata(dtype=np.float32), resized_data
                        ):
                            raise ValueError("final_output_value_verification_failed")

                        if not np.allclose(
                            verified.affine, target_affine, rtol=0, atol=1e-4
                        ):
                            raise ValueError(
                                "final_output_geometry_verification_failed"
                            )

                        checksum = sha256(pending)

                        os.replace(pending, final_image_path)
                    finally:
                        pending.unlink(missing_ok=True)

                    final_mask_path = destination / f"{stem}_ras_mask.nii.gz"

                    nib.save(
                        nib.Nifti1Image(output_mask.astype(np.uint8), target_affine),
                        final_mask_path,
                    )

                    verified_mask = nib.load(final_mask_path)

                    if not np.array_equal(
                        verified_mask.get_fdata(), output_mask
                    ) or not np.allclose(
                        verified_mask.affine, target_affine, atol=1e-4, rtol=0
                    ):
                        raise ValueError("final_mask_verification_failed")

                    mask_checksum = sha256(final_mask_path)
                    occupancy = output_mask.sum(axis=(0, 1))
                    occupied = np.flatnonzero(occupancy)
                    center = (target_shape - 1) / 2

                    slices = [
                        int(occupied[0] + 0.1 * (occupied[-1] - occupied[0])),
                        round(center[2]),
                        int(occupied[0] + 0.9 * (occupied[-1] - occupied[0])),
                    ]

                    lo, hi = np.percentile(resized_data[output_mask], [1, 99])

                    if output_dir is None:
                        fig, axes = plt.subplots(
                            1, 3, figsize=(10, 4), layout="constrained"
                        )

                        for ax, k in zip(axes, slices):
                            extent = [
                                (-0.5 - center[0]) * output_spacing[0],
                                (target_shape[0] - 0.5 - center[0]) * output_spacing[0],
                                (-0.5 - center[1]) * output_spacing[1],
                                (target_shape[1] - 0.5 - center[1]) * output_spacing[1],
                            ]

                            ax.imshow(
                                resized_data[:, :, k].T,
                                cmap="gray",
                                origin="lower",
                                extent=extent,
                                vmin=lo,
                                vmax=hi,
                            )

                            ax.contour(
                                (np.arange(target_shape[0]) - center[0])
                                * output_spacing[0],
                                (np.arange(target_shape[1]) - center[1])
                                * output_spacing[1],
                                output_mask[:, :, k].T,
                                levels=[0.5],
                                colors=["lime"],
                                linewidths=0.5,
                            )

                            ax.set_xlim(-150, 150)
                            ax.set_ylim(-150, 150)
                            ax.set_aspect("equal")
                            ax.set_title(f"Axial slice {k}")

                        fig.suptitle(
                            f"{stem} ({plane} acquisition) → axial {shape_tag}\nnative spacing {output_spacing.round(4)} mm; brain-mask centred"
                        )

                        fig.savefig(destination / f"{stem}_comparison.png", dpi=120)

                        plt.close(fig)

                    record = {
                        "patient_uid": row["patient_uid"],
                        "study_uid": study,
                        "series_id": row["series_id"],
                        "split": row["split"],
                        "classified_modality": row["classified_modality"],
                        "contrast": row["contrast"],
                        "image_path": (destination / f"{stem}_ras.nii.gz")
                        .relative_to(root)
                        .as_posix(),
                        "mask_path": final_mask_path.relative_to(root).as_posix(),
                        "intensity_normalized": True,
                        "intensity_maximum": intensity_maximum,
                        "intensity_normalization": "brain_maximum_zero_background_no_clipping",
                        "acquisition_plane": plane,
                        "acquisition_evidence": evidence,
                        "source_shape": shape.tolist(),
                        "required_shape": required_shape.tolist(),
                        "padding_before": before.tolist(),
                        "padding_after": after.tolist(),
                        "target_shape": target_shape.tolist(),
                        "target_spacing_mm": output_spacing.tolist(),
                        "native_spacing_mm": native_spacing.tolist(),
                        "spacing_axis_order_native_to_RAS": spacing_axis_order.tolist(),
                        "model_input_affine_mm": model_affine.tolist(),
                        "target_affine_mm": target_affine.tolist(),
                        "native_orientation_transform": native_transform.tolist(),
                        "coordinate_max_error_mm": error,
                        "recorded_matches_native": True,
                        "reverse_exact": True,
                        "full_mask_fov_retained": True,
                        "output_plane": "axial" if rigid else "native_grid_closest_RAS",
                        "interpolation_applied": rigid,
                        "image_interpolation_order": IMAGE_INTERPOLATION_ORDER
                        if rigid
                        else None,
                        "registration_review_required": rigid,
                        "source_mask_touches_image_edge": saved[
                            "mask_touches_image_edge"
                        ],
                        "source_coverage_review_required": saved[
                            "mask_touches_image_edge"
                        ],
                        "output_axis_codes": ["R", "A", "S"],
                        "brain_center_world_mm": center_world.tolist(),
                        "mask_centred": rigid,
                        "head_pose_aligned": rigid,
                        "registration": registration_info,
                        "head_pose_alignment_status": "estimated_pending_review"
                        if rigid
                        else "not_applied",
                        "native_to_output_voxel": (
                            np.linalg.inv(target_affine) @ motion @ affine
                        ).tolist(),
                        "spacing_policy": "retain_native_spacing_mapped_to_pre_registration_RAS",
                        "coverage_claim": "geometric_mask_containment_only",
                        "mask_storage": "saved_with_standardized_image",
                        "mask_volume_change_percent": float(
                            (
                                np.count_nonzero(resized_mask.get_fdata())
                                * np.prod(output_spacing)
                                / (inside.sum() * np.prod(spacing))
                                - 1
                            )
                            * 100
                        ),
                        "status": "pending_review",
                        "visual_review": "pending",
                        "training_accepted": False,
                    }

                    record.update(
                        final_policy_hash=final_policy_hash,
                        sha256=checksum,
                        mask_sha256=mask_checksum,
                        processing_seconds=time.perf_counter() - started,
                    )

                    write_json(record_path, record)

                    records.append(record)

                    print(
                        stem, evidence["slice_axis_basis"], "checks passed", flush=True
                    )
                except (
                    KeyError,
                    ValueError,
                    OSError,
                    RuntimeError,
                    zipfile.BadZipFile,
                    nib.filebasedimages.ImageFileError,
                ) as error:
                    record = {
                        "patient_uid": row["patient_uid"],
                        "study_uid": study,
                        "series_id": row["series_id"],
                        "split": row["split"],
                        "status": "failed"
                        if isinstance(
                            error, (OSError, RuntimeError, zipfile.BadZipFile)
                        )
                        else "held",
                        "reason": f"{type(error).__name__}: {error}",
                        "image_path": "",
                        "training_accepted": False,
                    }

                    record.update(
                        final_policy_hash=final_policy_hash,
                        processing_seconds=time.perf_counter() - started,
                    )

                    write_json(record_path, record)

                    records.append(record)

                    print(stem, record["status"], record["reason"], flush=True)

    return records


def pilot_acquisition(
    root,
    rigid=False,
    max_series=20,
    archives=None,
    selected=None,
    output_dir=None,
    download_workers=8,
    padding_shape=None,
    process_workers=1,
):
    """Validate a limited subset against native headers and masks."""

    import matplotlib

    matplotlib.use("Agg")

    policy = json.loads((root / "policy.json").read_text())
    plan = pd.read_csv(root / "plan.csv", dtype=str, keep_default_na=False)

    layouts = pd.read_csv(
        root / "acquisition_pilot.csv", dtype=str, keep_default_na=False
    )

    subset = plan.merge(layouts[["study_uid", "series_id"]], on=KEY)

    if selected is not None:
        subset = subset.merge(selected[KEY], on=KEY)

    subset = subset.head(max_series)
    shape_tag = "native_spacing"

    destination = (
        output_dir
        if output_dir is not None
        else root / "previews" / ("rigid_pilot" if rigid else "axial_pilot") / shape_tag
    )

    if rigid:
        reference = Path("data/templates/MNI152NLin2009cAsym")
        template_path = reference / "mni_icbm152_t1_tal_nlin_asym_09c.nii"
        template_mask_path = reference / "mni_icbm152_t1_tal_nlin_asym_09c_mask.nii"
        template = nib.load(template_path)
        template_mask_image = nib.load(template_mask_path)

        if template.shape != template_mask_image.shape or not np.allclose(
            template.affine, template_mask_image.affine
        ):
            raise ValueError("template_mask_geometry_mismatch")

    destination.mkdir(parents=True, exist_ok=True)

    write_csv(destination / "manifest.csv", subset)

    shutil.copyfile(Path(__file__), destination / "preprocess_snapshot.py")

    write_json(
        destination / "provenance.json",
        {
            "code_sha256": sha256(Path(__file__)),
            "source_revision": policy["revision"],
            "shape_policy": "variable_actual_mask_coverage",
            "sampling": "rigid_linear_image_nearest_mask"
            if rigid
            else "exact_RAS_reorder_integer_crop_zero_pad_no_interpolation",
            "padding_shape": list(padding_shape) if padding_shape is not None else None,
            "image_interpolation_order": IMAGE_INTERPOLATION_ORDER if rigid else None,
            "mask_interpolation_order": 0 if rigid else None,
            "training_accepted": False,
            "rigid_registration": rigid,
            "template_sha256": sha256(template_path) if rigid else None,
            "template_mask_sha256": sha256(template_mask_path) if rigid else None,
        },
    )

    records = []
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    with closing(
        study_archives(
            subset.groupby(["batch", "study_uid"], sort=True),
            root / "work",
            policy["revision"],
            download_workers,
            archives,
        )
    ) as downloads:
        jobs = (
            (
                archive_path,
                study,
                group,
                root,
                destination,
                rigid,
                padding_shape,
                template_path if rigid else None,
                template_mask_path if rigid else None,
                output_dir,
            )
            for batch, study, group, archive_path in downloads
        )

        if process_workers == 1:
            for job in jobs:
                records.extend(standardize_archive(job))
        else:
            for result in archive_jobs(
                jobs, standardize_archive, process_workers, root / "work"
            ):
                records.extend(result)

    write_json(destination / "results.json", records)

    write_csv(
        destination / "candidates.csv",
        pd.DataFrame(records)
        if records
        else pd.DataFrame(
            columns=[
                "patient_uid",
                "study_uid",
                "series_id",
                "split",
                "image_path",
                "training_accepted",
            ]
        ),
    )


def prepare_volume(
    image_path, mask_path, row, max_spacing, max_memory_mib, min_spacing=0.5
):
    image, affine, spacing = checked_image(image_path)
    mask, mask_affine, _ = checked_image(mask_path)

    if image.shape != mask.shape or not np.allclose(
        affine, mask_affine, rtol=0, atol=1e-4
    ):
        raise ValueError("image_mask_geometry_mismatch")

    if not np.array_equal(image.shape, vector(row["array_shape"], 3, integer=True)):
        raise ValueError("metadata_header_shape_mismatch")

    if not np.allclose(
        spacing, vector(row["array_spacing_mm"], 3), rtol=1e-3, atol=1e-3
    ):
        raise ValueError("metadata_header_spacing_mismatch")

    if spacing.min() < min_spacing:
        raise ValueError("header_spacing_too_small")

    if spacing.max() > max_spacing:
        raise ValueError("header_spacing_too_large")

    if int(np.prod(image.shape, dtype=np.int64)) * 24 > max_memory_mib * 1024**2:
        raise ValueError("working_memory_budget_exceeded")

    data = image.get_fdata(dtype=np.float32, caching="unchanged")
    mask_data = mask.get_fdata(dtype=np.float32, caching="unchanged")

    if not np.isfinite(data).all() or not np.isfinite(mask_data).all():
        raise ValueError("nonfinite_image_or_mask")

    if not np.isin(mask_data, [0, 1]).all():
        raise ValueError("mask_not_binary_0_1")

    inside = mask_data.astype(bool)

    del mask_data

    if not inside.any():
        raise ValueError("empty_brain_mask")

    values = data[inside]

    if values.size < 2 or np.ptp(values) == 0:
        raise ValueError("constant_masked_image")

    intensity = {"min": float(values.min()), "max": float(values.max())}

    del values

    # Apply mask without cropping or changing voxel intensity within the mask.
    data[~inside] = 0
    start = nib.orientations.io_orientation(affine)
    target = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(start, target)
    output_affine = affine @ nib.orientations.inv_ornt_aff(transform, image.shape)
    output_data = nib.orientations.apply_orientation(data, transform)
    output_mask = nib.orientations.apply_orientation(inside, transform)
    reverse = nib.orientations.ornt_transform(target, start)

    if not np.array_equal(
        nib.orientations.apply_orientation(output_data, reverse), data
    ) or not np.array_equal(
        nib.orientations.apply_orientation(output_mask, reverse), inside
    ):
        raise ValueError("orientation_reverse_image_or_mask_failed")

    touches_edge = any(
        output_mask.take(0, axis=axis).any() or output_mask.take(-1, axis=axis).any()
        for axis in range(3)
    )

    header = nib.Nifti1Header()

    header.set_data_dtype(np.float32)
    header.set_xyzt_units("mm")

    output = nib.Nifti1Image(output_data, output_affine, header)

    # sform holds the exact geometry, including shear. Do not invent a qform.
    output.set_sform(
        output_affine,
        code=int(image.header["sform_code"] or image.header["qform_code"]),
    )

    output.set_qform(None, code=0)

    info = {
        "native_shape": list(image.shape),
        "native_affine_mm": affine.tolist(),
        "native_spacing_mm": spacing.tolist(),
        "native_axis_codes": list(nib.aff2axcodes(affine)),
        "native_header_slice_axis": image.header.get_dim_info()[2],
        "orientation_transform": transform.tolist(),
        "output_shape": list(output.shape),
        "output_affine_mm": output_affine.tolist(),
        "output_spacing_mm": nib.affines.voxel_sizes(output_affine).tolist(),
        "output_axis_codes": list(nib.aff2axcodes(output_affine)),
        "obliquity_degrees": np.rad2deg(nib.affines.obliquity(output_affine)).tolist(),
        "brain_voxels": int(inside.sum()),
        "mask_touches_image_edge": bool(touches_edge),
        "brain_intensity_range": intensity,
        "cropped": False,
        "resampled": False,
        "clinical_quality": "not_reviewed",
        "report_pairing": "study_level_unfiltered",
        "brain_region_status": "metadata_text_evidence_not_verified",
    }

    return output, info


def valid_saved_result(record_path, root, policy_hash):
    try:
        record = json.loads(record_path.read_text())
        path = root / record["image_path"]

        return (
            record["policy_hash"] == policy_hash
            and record["status"] in {"complete", "review"}
            and path.is_file()
            and sha256(path) == record["sha256"]
        )
    except (KeyError, ValueError, OSError):
        return False


def process_archive(archive_path, rows, root, policy, policy_hash):
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()

        for row in rows:
            stem = f"{row['study_uid']}_{row['series_id']}"
            record_path = root / "records" / row["study_uid"] / f"{stem}.json"

            record_path.parent.mkdir(parents=True, exist_ok=True)

            if valid_saved_result(record_path, root, policy_hash):
                continue

            record = {
                key: row[key]
                for key in [
                    "patient_uid",
                    "study_uid",
                    "series_id",
                    "split",
                    "contrast",
                    "contrast_reason",
                    "contrast_evidence",
                    "region_evidence",
                ]
            }

            record.update(
                policy_hash=policy_hash,
                processing_code_sha256=sha256(Path(__file__)),
                status="failed",
                image_path="",
                reason="",
            )

            try:
                wanted = [
                    f"{row['study_uid']}/img/{stem}.nii.gz",
                    f"{row['study_uid']}/seg/{stem}_brain-mask.nii.gz",
                ]

                if any(members.count(name) != 1 for name in wanted):
                    raise ValueError("missing_or_duplicate_archive_member")

                infos = [archive.getinfo(name) for name in wanted]

                required = (
                    sum(info.file_size for info in infos)
                    + int(np.prod(vector(row["array_shape"], 3))) * 8
                )

                if shutil.disk_usage(root).free < required + 64 * 1024**2:
                    raise OSError("insufficient_space_for_series")

                with tempfile.TemporaryDirectory(dir=root / "work") as directory:
                    temporary = Path(directory)
                    paths = [temporary / "image.nii.gz", temporary / "mask.nii.gz"]

                    for member, destination in zip(wanted, paths):
                        # Reading to EOF checks the ZIP member CRC. Never extractall.
                        with (
                            archive.open(member) as source,
                            destination.open("wb") as output,
                        ):
                            shutil.copyfileobj(source, output)

                    result, info = prepare_volume(
                        *paths,
                        row,
                        policy["max_spacing_mm"],
                        policy["max_memory_mib"],
                        policy["min_spacing_mm"],
                    )

                    destination = root / "images" / row["study_uid"] / f"{stem}.nii.gz"

                    destination.parent.mkdir(parents=True, exist_ok=True)

                    pending = destination.with_name(stem + ".pending.nii.gz")

                    try:
                        nib.save(result, pending)

                        saved = nib.load(pending)

                        if saved.shape != result.shape or not np.allclose(
                            saved.affine, result.affine, rtol=0, atol=1e-4
                        ):
                            raise ValueError("output_geometry_verification_failed")

                        if not np.array_equal(
                            saved.get_fdata(dtype=np.float32, caching="unchanged"),
                            np.asanyarray(result.dataobj),
                        ):
                            raise ValueError("output_value_verification_failed")

                        checksum = sha256(pending)

                        os.replace(pending, destination)

                        del result, saved
                    finally:
                        pending.unlink(missing_ok=True)

                record.update(info)
                record.update(
                    status="review" if info["mask_touches_image_edge"] else "complete",
                    reason="mask_touches_image_edge"
                    if info["mask_touches_image_edge"]
                    else "",
                    image_path=destination.relative_to(root).as_posix(),
                    sha256=checksum,
                )
            except (
                OSError,
                ValueError,
                zipfile.BadZipFile,
                RuntimeError,
                nib.filebasedimages.ImageFileError,
            ) as error:
                record.update(
                    status="failed", reason=f"{type(error).__name__}: {error}"
                )

            write_json(record_path, record)

            print(
                row["study_uid"],
                row["series_id"],
                record["status"],
                record["reason"],
                flush=True,
            )


def export_results(root, plan, policy_hash):
    records = []

    for row in plan.to_dict("records"):
        stem = f"{row['study_uid']}_{row['series_id']}"
        path = root / "records" / row["study_uid"] / f"{stem}.json"

        result = {
            "status": "held" if row["plan_status"] == "held" else "not_processed",
            "reason": row["plan_reason"],
        }

        if row["plan_status"] == "candidate" and path.exists():
            saved = json.loads(path.read_text())

            if saved.get("policy_hash") == policy_hash:
                result = saved

                if (
                    saved["status"] in {"complete", "review"}
                    and not (root / saved["image_path"]).is_file()
                ):
                    result = {"status": "failed", "reason": "missing_saved_output"}

        records.append({**row, **result})

    audit = pd.DataFrame(records) if records else plan.copy()

    for field in ["status", "reason", "image_path"]:
        if field not in audit:
            audit[field] = ""

    # This is a technical preprocessing subset, not clinical quality certification.
    ready = audit[audit["status"] == "complete"].copy()

    write_csv(root / "audit.csv", audit)
    write_csv(root / "samples.csv", ready)

    summary = {
        "status_counts": audit["status"].value_counts().to_dict(),
        "ready_series": len(ready),
        "ready_studies": ready["study_uid"].nunique(),
        "ready_patients": ready["patient_uid"].nunique(),
        "contrast_counts": ready["contrast"].value_counts().to_dict(),
    }

    write_json(root / "summary.json", summary)

    print(json.dumps(summary, indent=2), flush=True)


def dataset_haar_statistics(root):
    """Freeze statistics for the exact available final training subset, without copies."""

    manifest = pd.read_csv(
        root / "standardized_candidates.csv", dtype=str, keep_default_na=False
    )

    if manifest.groupby("patient_uid")["split"].nunique().gt(1).any():
        raise ValueError("Patient appears in multiple splits")

    # Held-out scans must not contribute to fitted normalization statistics.
    training = manifest[
        (manifest["split"] == "train")
        & manifest["image_path"].ne("")
        & manifest["status"].eq("pending_review")
        & manifest["training_accepted"].str.casefold().eq("true")
    ].sort_values(KEY)

    if training.empty:
        print("No accepted training scans; Haar statistics not calculated", flush=True)

        return None

    subset = []

    for row in training.to_dict("records"):
        subset.append(
            {
                key: row[key]
                for key in [
                    "patient_uid",
                    "study_uid",
                    "series_id",
                    "split",
                    "image_path",
                    "intensity_normalized",
                ]
            }
            | {
                "image_sha256": sha256(root / row["image_path"]),
                "intensity_maximum": row.get("intensity_maximum", ""),
            }
        )

    signature = hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest()
    output = root / "haar_statistics.json"

    if output.exists():
        saved = json.loads(output.read_text())

        if saved["training_subset_sha256"] != signature or saved[
            "haar_code_sha256"
        ] != sha256(Path(__file__).resolve().parents[1] / "util/haar.py"):
            raise ValueError("Frozen Haar subset/code changed; use a new dataset root")

        print(f"Using frozen Haar statistics: {output}", flush=True)

        return saved

    def volumes():
        for row, item in zip(training.to_dict("records"), subset):
            image = nib.load(root / row["image_path"])

            if image.shape != (240, 240, 168):
                raise ValueError("Haar model requires 240 x 240 x 168")

            data = image.get_fdata(dtype=np.float32)[None]
            normalized, maximum = normalize_mri(data, data != 0)

            if row["intensity_normalized"].lower() == "true":
                if not np.isclose(maximum, 1.0, atol=1e-6):
                    raise ValueError("Normalized image maximum differs from one")

                yield data
            else:
                item["intensity_maximum"] = maximum

                yield normalized

    # Stream one scan at a time; zero padding remains part of the model input.
    stats = fit_haar_statistics(volumes())

    stats.update(
        {
            "training_subset_sha256": signature,
            "subset": subset,
            "haar_code_sha256": sha256(
                Path(__file__).resolve().parents[1] / "util/haar.py"
            ),
            "shape": [240, 240, 168],
            "population_std": True,
            "padding_positions_included": True,
            "frozen": True,
            "training_acceptance": "separate_visual_and_registration_review_required",
        }
    )

    write_json(output, stats)

    write_csv(root / "haar_training_manifest.csv", pd.DataFrame(subset))

    print(
        f"Frozen 24 Haar pairs from {stats['training_scans']} training scans: {output}",
        flush=True,
    )

    return stats


def process_final_candidates(root, plan, candidates, policy, policy_hash, args):
    shape = tuple(args.target_shape)

    tag = (
        "native_spacing_padded_" if args.rigid else "native_grid_padded_"
    ) + "x".join(map(str, shape))

    destination = root / "standardized" / tag

    destination.mkdir(parents=True, exist_ok=True)

    reference = Path("data/templates/MNI152NLin2009cAsym")

    template = (
        reference / "mni_icbm152_t1_tal_nlin_asym_09c.nii" if args.rigid else None
    )

    mask = (
        reference / "mni_icbm152_t1_tal_nlin_asym_09c_mask.nii" if args.rigid else None
    )

    settings = {
        "native_policy_hash": policy_hash,
        "code_sha256": sha256(Path(__file__)),
        "rigid": args.rigid,
        "target_shape": list(shape),
        "image_interpolation_order": IMAGE_INTERPOLATION_ORDER if args.rigid else None,
        "template_sha256": sha256(template) if template else None,
        "template_mask_sha256": sha256(mask) if mask else None,
    }

    settings_path = destination / "provenance.json"

    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Final processing policy changed. Use a new output root.")

    write_json(settings_path, settings)

    shutil.copyfile(Path(__file__), destination / "preprocess_snapshot.py")

    final_hash = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode()
    ).hexdigest()

    all_candidates = plan[plan.plan_status.eq("candidate")]

    write_csv(destination / "manifest.csv", all_candidates)

    def pending_groups():
        for key, group in candidates.groupby(["batch", "study_uid"], sort=True):
            pending = []

            for row in group.to_dict("records"):
                path = destination / f"{row['study_uid']}_{row['series_id']}.json"

                if final_result(path, root, final_hash) is None:
                    pending.append(row)

            if pending:
                yield key, pd.DataFrame(pending)

    completed_this_run = 0

    try:
        with closing(
            study_archives(
                pending_groups(),
                root / "work",
                policy["revision"],
                args.download_workers,
                args.archives,
            )
        ) as downloads:
            jobs = (
                (
                    archive,
                    study,
                    group,
                    root,
                    destination,
                    args.rigid,
                    shape,
                    template,
                    mask,
                    destination,
                    final_hash,
                    policy,
                    policy_hash,
                )
                for batch, study, group, archive in downloads
            )

            results = (
                map(complete_archive_job, jobs)
                if args.process_workers == 1
                else archive_jobs(
                    jobs, complete_archive_job, args.process_workers, root / "work"
                )
            )

            for records in results:
                completed_this_run += len(records)

                write_json(
                    root / "progress.json",
                    {
                        "completed_this_run": completed_this_run,
                        "final_policy_hash": final_hash,
                        "completion_records": destination.relative_to(root).as_posix(),
                    },
                )
    finally:
        # Per-series atomic records are authoritative, even if a worker was interrupted.
        export_results(root, plan, policy_hash)

        native = pd.read_csv(root / "samples.csv", dtype=str, keep_default_na=False)

        write_csv(root / "native_samples.csv", native)

        records = []

        for row in all_candidates.sort_values(["batch", *KEY]).to_dict("records"):
            path = destination / f"{row['study_uid']}_{row['series_id']}.json"
            record = final_result(path, root, final_hash)

            if record is not None:
                records.append(record)

        frame = (
            pd.DataFrame(records)
            if records
            else pd.DataFrame(
                columns=[
                    "patient_uid",
                    "study_uid",
                    "series_id",
                    "split",
                    "status",
                    "image_path",
                    "training_accepted",
                ]
            )
        )

        write_json(destination / "results.json", records)

        write_csv(destination / "candidates.csv", frame)
        write_csv(root / "standardized_candidates.csv", frame)
        write_csv(root / "samples.csv", frame[frame.training_accepted.eq(True)])

        write_json(
            root / "progress.json",
            {
                "completed_series": len(frame),
                "eligible_series": len(all_candidates),
                "selected_series": len(candidates),
                "status_counts": frame.status.value_counts().to_dict(),
                "final_policy_hash": final_hash,
                "completion_records": destination.relative_to(root).as_posix(),
            },
        )

    # Freeze statistics only after the complete candidate set has terminal records.
    # A partial run must not freeze a training subset that grows on resume.
    if len(frame) == len(all_candidates):
        dataset_haar_statistics(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser(
        "plan",
        help="Download small tables, validate joins, and list every T1 candidate",
    )

    plan_parser.add_argument("--root", type=Path, required=True)
    plan_parser.add_argument(
        "--source",
        type=Path,
        help="Existing metadata/, reports/, and splits.csv; no table downloads",
    )
    plan_parser.add_argument(
        "--revision",
        help=(
            "Source revision; online defaults to current Hub revision resolved to "
            "its SHA"
        ),
    )
    plan_parser.add_argument(
        "--min-spacing",
        type=float,
        default=0.95,
        help="Minimum spacing on every axis (mm; default: 0.95)",
    )
    plan_parser.add_argument(
        "--max-spacing",
        type=float,
        default=1.05,
        help=(
            "Keep series with spacing <= this limit on every axis (mm; default: "
            "1.05); no resampling"
        ),
    )
    plan_parser.add_argument("--max-memory-mib", type=int, default=2048)

    process_parser = sub.add_parser(
        "process",
        help="Download, mask, rigidly align and pad; --no-rigid uses exact reordering",
    )

    process_parser.add_argument("--root", type=Path, required=True)
    process_parser.add_argument(
        "--process-workers",
        type=int,
        default=1,
        help=(
            "Parallel study preprocessing processes (default: 1); memory use grows "
            "with workers"
        ),
    )
    process_parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Parallel study downloads (default: 8)",
    )
    process_parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=(240, 240, 168),
        metavar=("X", "Y", "Z"),
        help="Zero-padding target; larger required grids are held, never resized",
    )
    process_parser.add_argument(
        "--archives",
        type=Path,
        help="Existing mri/batchXX/study.zip tree; no image downloads",
    )
    process_parser.add_argument(
        "--rigid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Rigid alignment with one linear resampling (default); --no-rigid uses "
            "exact reordering only"
        ),
    )
    process_parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Reuse verified native outputs after a code update; preserve the "
            "original plan policy"
        ),
    )
    process_parser.add_argument(
        "--native-only",
        action="store_true",
        help="Only mask/reorder; skip standardized output",
    )
    process_parser.add_argument("--split", choices=["train", "validation", "test"])
    process_parser.add_argument(
        "--max-series", type=int, help="Limit to the first N eligible MRI series"
    )
    process_parser.add_argument(
        "--max-studies", type=int, help="Limit this invocation to the first N study IDs"
    )

    orientation_parser = sub.add_parser(
        "orientation",
        help="Audit acquisition layout of existing images without copying volumes",
    )

    orientation_parser.add_argument("--root", type=Path, required=True)

    pilot_parser = sub.add_parser(
        "pilot", help="Standardize a limited subset of downloaded scans"
    )

    pilot_parser.add_argument("--root", type=Path, required=True)
    pilot_parser.add_argument("--download-workers", type=int, default=8)
    pilot_parser.add_argument("--max-series", type=int, default=20)
    pilot_parser.add_argument(
        "--rigid",
        action="store_true",
        help="Rigidly align to the downloaded ICBM152 2009c asymmetric T1 template",
    )

    stats_parser = sub.add_parser(
        "statistics",
        help=(
            "Freeze training-only 24-band Haar statistics for final standardized images"
        ),
    )

    stats_parser.add_argument("--root", type=Path, required=True)

    args = parser.parse_args()
    root = args.root.resolve()

    root.mkdir(parents=True, exist_ok=True)

    (root / "work").mkdir(exist_ok=True)

    # One writer per output root. Shared caches and source archives are never deleted.
    with FileLock(root / "pipeline.lock", timeout=0):
        if args.command == "statistics":
            dataset_haar_statistics(root)

            return

        if args.command == "pilot":
            if args.download_workers <= 0 or args.max_series <= 0:
                parser.error("Series count and target dimensions must be positive")

            pilot_acquisition(
                root,
                rigid=args.rigid,
                max_series=args.max_series,
                download_workers=args.download_workers,
            )

            return

        if args.command == "orientation":
            audit_acquisition(root)

            return

        if args.command == "plan":
            if (
                not np.isfinite([args.min_spacing, args.max_spacing]).all()
                or not 0 < args.min_spacing <= args.max_spacing
                or args.max_memory_mib <= 0
            ):
                parser.error(
                    "Require 0 < min-spacing <= max-spacing and positive memory limit"
                )

            source = args.source.resolve() if args.source else root / "source"
            revision = args.revision

            if args.source:
                provenance_path = source / "source_revision.json"

                if provenance_path.exists():
                    recorded = json.loads(provenance_path.read_text())

                    if revision and revision != recorded["revision"]:
                        raise ValueError(
                            (
                                "Requested local revision disagrees with "
                                "source_revision.json"
                            )
                        )

                    revision = recorded["revision"]
            else:
                # Avoid an independent, persistent Xet chunk cache for large downloads.
                os.environ["HF_HUB_DISABLE_XET"] = "1"

                from huggingface_hub import HfApi
                from huggingface_hub import snapshot_download

                revision = HfApi().dataset_info(REPO, revision=revision, token=True).sha

                snapshot_download(
                    REPO,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=source,
                    allow_patterns=["metadata/*.csv", "reports/*.csv", "splits.csv"],
                    token=True,
                )

                write_json(
                    source / "source_revision.json",
                    {"repo": REPO, "revision": revision},
                )

            paths = (
                sorted((source / "metadata").glob("batch*_metadata.csv"))
                + sorted((source / "reports").glob("batch*_reports.csv"))
                + [source / "splits.csv"]
            )

            plan, reports = build_plan(source, args.max_spacing, args.min_spacing)

            policy = {
                "version": VERSION,
                "code_sha256": sha256(Path(__file__)),
                "repo": REPO,
                "revision": revision,
                "source_hashes": {
                    path.relative_to(source).as_posix(): sha256(path) for path in paths
                },
                "min_spacing_mm": args.min_spacing,
                "max_spacing_mm": args.max_spacing,
                "max_memory_mib": args.max_memory_mib,
                "crop": False,
                "classifier": False,
                "keep_all_t1": True,
                "plan_sha256": hashlib.sha256(
                    plan.to_csv(index=False).encode()
                ).hexdigest(),
                "reports_sha256": hashlib.sha256(
                    reports.to_csv(index=False).encode()
                ).hexdigest(),
            }

            policy_hash = hashlib.sha256(
                json.dumps(policy, sort_keys=True).encode()
            ).hexdigest()

            if (root / "policy.json").exists() and json.loads(
                (root / "policy.json").read_text()
            ) != policy:
                raise ValueError("Policy or source changed. Use a new output root.")

            write_csv(root / "plan.csv", plan)
            write_csv(root / "reports.csv", reports)

            write_json(root / "policy.json", policy)

            export_results(root, plan, policy_hash)
        else:
            if args.process_workers <= 0:
                parser.error("--process-workers must be positive")

            if args.download_workers <= 0:
                parser.error("--download-workers must be positive")

            if min(args.target_shape) <= 0:
                parser.error("Target dimensions must be positive")

            if args.max_studies is not None and args.max_studies <= 0:
                parser.error("--max-studies must be positive")

            if args.max_series is not None and args.max_series <= 0:
                parser.error("--max-series must be positive")

            policy = json.loads((root / "policy.json").read_text())

            if (
                policy["code_sha256"] != sha256(Path(__file__))
                and not args.resume_existing
            ):
                raise ValueError(
                    "Code changed since plan. Use a new output root, or "
                    "--resume-existing to reuse verified native results."
                )

            if (
                sha256(root / "plan.csv") != policy["plan_sha256"]
                or sha256(root / "reports.csv") != policy["reports_sha256"]
            ):
                raise ValueError(
                    "Plan or report table changed since plan. Use a new output root."
                )

            policy_hash = hashlib.sha256(
                json.dumps(policy, sort_keys=True).encode()
            ).hexdigest()

            plan = pd.read_csv(root / "plan.csv", dtype=str, keep_default_na=False)
            candidates = plan[plan["plan_status"] == "candidate"]

            if args.split:
                candidates = candidates[candidates["split"] == args.split]

            if not args.archives and not re.fullmatch(
                r"[a-fA-F0-9]{40}", policy["revision"] or ""
            ):
                raise ValueError(
                    (
                        "Image downloads require a pinned Hub commit SHA; use --archives "
                        "for local input"
                    )
                )

            if args.max_series is not None:
                candidates = candidates.sort_values(
                    ["batch", "study_uid", "series_id"]
                ).head(args.max_series)

            if args.max_studies is not None:
                studies = (
                    candidates[["batch", "study_uid"]]
                    .drop_duplicates()
                    .sort_values(["batch", "study_uid"])
                    .head(args.max_studies)
                )

                candidates = candidates.merge(studies, on=["batch", "study_uid"])

            if not args.native_only:
                process_final_candidates(
                    root, plan, candidates, policy, policy_hash, args
                )

                return

            groups = candidates.groupby(["batch", "study_uid"], sort=True)

            pending_groups = (
                (key, group)
                for key, group in groups
                if not all(
                    valid_saved_result(
                        root / "records" / key[1] / f"{key[1]}_{row['series_id']}.json",
                        root,
                        policy_hash,
                    )
                    for row in group.to_dict("records")
                )
            )

            try:
                with closing(
                    study_archives(
                        pending_groups,
                        root / "work",
                        policy["revision"],
                        args.download_workers,
                        args.archives,
                    )
                ) as downloads:
                    if args.process_workers == 1:
                        for batch, study_uid, group, archive in downloads:
                            process_archive(
                                archive,
                                group.to_dict("records"),
                                root,
                                policy,
                                policy_hash,
                            )
                    else:
                        jobs = (
                            (
                                archive,
                                group.to_dict("records"),
                                root,
                                policy,
                                policy_hash,
                            )
                            for batch, study_uid, group, archive in downloads
                        )

                        for _ in archive_jobs(
                            jobs,
                            preprocess_archive_job,
                            args.process_workers,
                            root / "work",
                        ):
                            pass
            finally:
                # Completed records remain usable after an interrupted or failed run.
                export_results(root, plan, policy_hash)


if __name__ == "__main__":
    main()
