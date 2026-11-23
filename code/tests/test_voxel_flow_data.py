import copy

import numpy as np
import pandas as pd
import pytest
import torch

from src.voxel_flow.config import load_config
from src.voxel_flow.data import collate_batch
from src.voxel_flow.data import load_voxel_target
from src.voxel_flow.data import save_voxel_target
from src.voxel_flow.data import validate_cohort
from src.voxel_flow.data import VoxelReportDataset


def cohort():
    frame = pd.DataFrame(
        [
            dict(
                patient_uid="p1",
                study_uid="s1",
                series_id="a",
                split="train",
                image_path="a.nii.gz",
                status="pending_review",
                training_accepted="False",
                classified_modality="T1w",
                contrast="noncontrast",
            ),
            dict(
                patient_uid="p2",
                study_uid="s2",
                series_id="a",
                split="validation",
                image_path="b.nii.gz",
                status="pending_review",
                training_accepted="False",
                classified_modality="T1w",
                contrast="postcontrast",
            ),
        ]
    )

    return frame, frame[["study_uid", "patient_uid", "split"]].copy()


def test_compact_exact_tail_negative_and_zero_roundtrip(tmp_path):
    target = np.zeros((3, 3, 3), dtype=np.float32)
    target.flat[[0, 5, 26]] = [-3.25, 1e-20, 2.75]
    path = tmp_path / "target.npz"

    save_voxel_target(path, target, np.eye(4), 0.25)

    restored, affine, median = load_voxel_target(path, target.shape)

    assert np.array_equal(target, restored)
    assert np.array_equal(affine, np.eye(4))
    assert median == 0.25


@pytest.mark.parametrize(
    "corrupt", ["short_bits", "dtype", "nan", "affine", "shape", "tail"]
)
def test_compact_malformed_inputs_rejected(tmp_path, corrupt):
    path = tmp_path / "target.npz"

    save_voxel_target(path, np.ones((3, 3, 3), dtype=np.float32), np.eye(4), 1.0)

    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}

    if corrupt == "short_bits":
        arrays["support_bits"] = arrays["support_bits"][:-1]
    elif corrupt == "dtype":
        arrays["values"] = arrays["values"].astype(np.float16)
    elif corrupt == "nan":
        arrays["values"][0] = np.nan
    elif corrupt == "affine":
        arrays["affine"][0] = 0
    elif corrupt == "shape":
        arrays["shape"] = np.array([3.0, 3.0, 3.0])
    else:
        arrays["support_bits"][-1] |= 128

    np.savez(path, **arrays)

    with pytest.raises(ValueError):
        load_voxel_target(path)


@pytest.mark.parametrize(
    "corrupt",
    ["duplicate", "leak", "study", "missing_report", "modality", "review", "contrast"],
)
def test_cohort_boundary_errors(corrupt):
    frame, reports = cohort()

    if corrupt == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif corrupt == "leak":
        frame.loc[1, "patient_uid"] = "p1"
    elif corrupt == "study":
        reports.loc[0, "patient_uid"] = "wrong"
    elif corrupt == "missing_report":
        reports = reports.iloc[:1]
    elif corrupt == "modality":
        frame.loc[0, "classified_modality"] = "T2w"
    elif corrupt == "review":
        frame.loc[0, "training_accepted"] = "True"
    else:
        frame.loc[0, "contrast"] = ""

    with pytest.raises(ValueError):
        validate_cohort(frame, reports, exploratory=True)


def test_series_share_token_interval_and_collation_pads_only_reports(tmp_path):
    from types import SimpleNamespace

    frame, _ = cohort()
    other = frame.iloc[0].copy()
    other.series_id = "b"
    frame = pd.concat([frame, other.to_frame().T], ignore_index=True)

    features = SimpleNamespace(
        index={"s1": 0, "s2": 1},
        offsets=torch.tensor([0, 2, 5]),
        tokens=torch.randn(5, 768),
    )

    for i in range(3):
        save_voxel_target(
            tmp_path / f"{i}.npz", np.ones((4, 4, 4), dtype=np.float32), np.eye(4), 1.0
        )

    frame["voxel_cache_path"] = [f"{i}.npz" for i in range(3)]
    dataset = VoxelReportDataset(frame, tmp_path, features, (4, 4, 4))
    first, second, third = [dataset[i] for i in range(3)]

    assert (
        first["condition"]["report_tokens"].data_ptr()
        == third["condition"]["report_tokens"].data_ptr()
    )

    batch = collate_batch([first, second])

    assert batch["volume"].shape == (2, 1, 4, 4, 4)

    assert batch["condition"]["report_mask"].tolist() == [
        [True, True, False],
        [True, True, True],
    ]

    assert set(batch["condition"]) == {"report_tokens", "report_mask", "contrast_id"}


def test_workspace_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    from pathlib import Path

    config_path = Path(__file__).resolve().parents[1] / "configs/voxel_flow.yaml"

    before = load_config(config_path)

    monkeypatch.chdir(tmp_path)

    assert load_config(config_path) == before
    assert not Path(before["run_dir"]).is_relative_to(config_path.parents[1])
