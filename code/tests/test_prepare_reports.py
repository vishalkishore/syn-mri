import numpy as np
import pandas as pd
import pytest

from scripts.prepare_reports import prepare_manifest
from scripts.select_cohort import select_cohort


def cohort():
    samples = pd.DataFrame(
        {
            "patient_uid": ["p1", "p1", "p2", "p3", "p4"],
            "study_uid": ["s1", "s1", "s2", "s3", "s4"],
            "series_id": ["t1w", "t1w-2", "t1w", "t1w-3", "t1w"],
            "split": ["train", "train", "validation", "train", "test"],
            "bands_path": ["old.npy"] * 5,
            "a2_regions_path": ["old_regions.npy"] * 5,
            "contrast": ["unknown"] * 5,
        }
    )

    columns = [f"pathology_{index}" for index in range(37)]
    labels = pd.DataFrame(np.zeros((3, 37), dtype=np.uint8), columns=columns)

    labels.insert(0, "study_uid", ["s1", "s2", "s3"])

    labels.loc[0, columns[0]] = 1

    reports = pd.DataFrame(
        {"study_uid": ["s1", "s2", "s3", "s4"], "report": ["Report"] * 4}
    )

    return samples, labels, reports


def test_suffix_pathology_and_no_label_rules(tmp_path):
    samples, labels, reports = cohort()
    retained, excluded, columns = prepare_manifest(samples, labels, reports)

    assert retained.study_uid.tolist() == ["s1", "s1", "s2"]
    assert retained[columns[0]].tolist() == [1, 1, 0]
    assert retained.no_label.tolist() == [0, 0, 1]
    assert retained.pre_contrast.tolist() == [1, 0, 1]
    assert retained.post_contrast.tolist() == [0, 1, 0]
    assert retained.contrast.tolist() == ["noncontrast", "postcontrast", "noncontrast"]

    assert set(excluded.exclusion_reason) == {
        "series_suffix_greater_than_2",
        "missing_official_pathology_annotation",
    }


def test_invalid_labels_missing_reports_and_leakage_are_rejected(tmp_path):
    samples, labels, reports = cohort()
    invalid = labels.copy()
    invalid.iloc[0, 1] = 2

    with pytest.raises(ValueError, match="binary"):
        prepare_manifest(samples, invalid, reports)

    with pytest.raises(ValueError, match="Missing or empty report"):
        prepare_manifest(samples, labels, reports[reports.study_uid != "s2"])

    samples.loc[2, "patient_uid"] = "p1"

    with pytest.raises(ValueError, match="Patient split leakage"):
        prepare_manifest(samples, labels, reports)


def test_cohort_selection_stays_inside_splits_and_skips_test():
    frame = pd.DataFrame(
        {
            "patient_uid": [f"p{index}" for index in range(10)],
            "split": ["train"] * 5 + ["validation"] * 3 + ["test"] * 2,
        }
    )

    selected = select_cohort(frame, 4, 2, train_seed=42, validation_seed=43)

    assert selected.split.value_counts().to_dict() == {"train": 4, "validation": 2}
    assert not selected.patient_uid.duplicated().any()

    again = select_cohort(frame, 4, 2, train_seed=42, validation_seed=43)

    assert again.equals(selected)

    with pytest.raises(ValueError, match="requested"):
        select_cohort(frame, 6, 2, train_seed=42, validation_seed=43)


def test_held_scans_are_excluded_with_reason():
    samples, labels, reports = cohort()
    samples["status"] = [
        "pending_review",
        "held",
        "pending_review",
        "pending_review",
        "pending_review",
    ]
    retained, excluded, _ = prepare_manifest(samples, labels, reports)

    assert retained.series_id.tolist() == ["t1w", "t1w"]
    assert "preprocessing_held" in set(excluded.exclusion_reason)
