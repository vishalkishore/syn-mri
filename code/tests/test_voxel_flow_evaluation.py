import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from src.voxel_flow.data import VoxelReportDataset
from src.voxel_flow.data import save_voxel_target
from src.voxel_flow.evaluation import volume_statistics
from src.voxel_flow.evaluation import evaluate
from src.voxel_flow.evaluation import save_generated
from src.voxel_flow.evaluation import select_cases
from src.voxel_flow.evaluation import trajectory_rates
from test_voxel_flow_model import tiny_config
from test_voxel_flow_training import TinyPredictor


def evaluation_frame(count=8):
    return pd.DataFrame(
        [
            dict(
                patient_uid=f"p{i}",
                study_uid=f"s{i}",
                series_id="t1",
                split="validation",
                contrast="noncontrast",
                voxel_cache_path=f"{i}.npz",
            )
            for i in range(count)
        ]
    )


def test_selection_is_reproducible_disjoint_and_contrast_matched():
    frame = evaluation_frame()
    selected = select_cases(frame, "validation", 3, 42)

    assert selected.equals(
        select_cases(frame.sample(frac=1, random_state=3), "validation", 3, 42)
    )

    assert selected.patient_uid.nunique() == 6
    assert selected.contrast.iloc[:3].tolist() == selected.contrast.iloc[3:].tolist()

    with pytest.raises(ValueError):
        select_cases(frame.iloc[:3], "validation", 2, 42)

    leaky = pd.concat([frame, frame.iloc[:1].assign(split="train")])

    with pytest.raises(ValueError):
        select_cases(leaky, "validation", 2, 42)


def test_constant_and_ramp_metrics_and_physical_units():
    constant = volume_statistics(
        np.ones((16, 16, 16), dtype=np.float32) * 2, np.eye(4)
    )["scalar"]

    assert constant["variance"] == 0
    assert constant["second_moment"] == 4
    assert constant["H1_LLH_energy"] == 0

    ramp = np.broadcast_to(np.arange(16, dtype=np.float32)[:, None, None], (16, 16, 16))
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    summary = volume_statistics(ramp, affine)["scalar"]

    assert summary["difference_X_1_rms"] == 1
    assert summary["difference_X_4_rms"] == 4
    assert summary["gradient_X_rms_per_mm"] == 0.5
    assert summary["gradient_Z_rms_per_mm"] == 0
    assert summary["variance"] == np.var(np.arange(16.0))


def test_second_moment_and_centered_variance_rates_are_distinct():
    x = torch.ones(1, 1, 4, 4, 4) * 2
    v = torch.ones_like(x) * 3
    rates = trajectory_rates(x, v)

    assert rates["second_moment_rate"] == 12
    assert rates["centered_variance_rate"] == 0


def test_generated_nifti_preserves_unpadded_affine_including_shear(tmp_path):
    affine = np.array(
        [[1.0, 0.1, 0, -11.0], [0, 2.0, 0, 12.0], [0, 0, 3.0, -5.0], [0, 0, 0, 1.0]]
    )

    path = tmp_path / "generated.nii.gz"
    target = np.random.default_rng(42).normal(size=(8, 8, 8)).astype(np.float32)

    save_generated(path, target, affine)

    restored = nib.load(path)

    assert restored.shape == target.shape

    np.testing.assert_allclose(restored.affine, affine, atol=1e-6)

    assert int(restored.header["qform_code"]) == 0

    np.testing.assert_array_equal(restored.get_fdata(dtype=np.float32), target)


@pytest.mark.parametrize("masked", [False, True])
def test_evaluation_exports_unpaired_diagnostics_and_counts(tmp_path, masked):
    frame = evaluation_frame(4)
    root = tmp_path / "cache"

    root.mkdir()

    for i in range(4):
        target = np.random.default_rng(i).normal(size=(8, 8, 8)).astype(np.float32)

        if masked:
            target[0, 0, 0] = 0
            mask = np.ones_like(target, dtype=np.uint8)
            mask[0, 0, 0] = 0
            mask_path = root / f"{i}_mask.nii.gz"

            nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)

            frame.loc[i, "mask_path"] = str(mask_path)

        save_voxel_target(root / f"{i}.npz", target, np.eye(4), 1.0)

    features = SimpleNamespace(
        index={f"s{i}": i for i in range(4)},
        offsets=torch.arange(5) * 2,
        tokens=torch.randn(8, 768),
    )

    dataset = VoxelReportDataset(
        frame, root, features, (8, 8, 8), dataset_root=root, include_brain_mask=masked
    )

    config = tiny_config()

    config.update(
        native_shape=[8, 8, 8],
        sampling_steps=2,
        diversity_reports=1,
        diversity_samples_per_report=2,
        evaluation_samples=2,
        voxel_loss_reduction="mask_balanced" if masked else "whole_volume",
    )

    metrics = evaluate(
        TinyPredictor(),
        dataset,
        frame,
        config,
        tmp_path / "evaluation",
        count=2,
        weights="raw",
    )

    assert metrics["model_evaluations"] == 21  # 2*(correct,null,shuffled)*3 + 1*3
    assert metrics["real_vs_real"]["patients_per_group"] == 2

    output = tmp_path / "evaluation"

    for name in (
        "metrics.json",
        "selected_cases.csv",
        "per_case_scalars.csv",
        "preview.png",
        "prediction_diagnostics.csv",
        "trajectory_diagnostics.csv",
        "anatomy_report_review.csv",
    ):
        assert (output / name).is_file()

    pd.DataFrame(metrics["diversity"])

    # No paired patient PSNR/SSIM is used as a generation score.
    assert "psnr" not in json.dumps(metrics).lower()

    with pytest.raises(FileExistsError):
        evaluate(TinyPredictor(), dataset, frame, config, output, count=2)


def test_standalone_sample_reads_geometry_header_only(tmp_path, monkeypatch):
    from scripts import voxel_flow
    from types import SimpleNamespace

    config = tiny_config()

    config.update(
        native_shape=[8, 8, 8],
        sampling_steps=2,
        workspace_root=str(tmp_path),
        report_token_cache=str(tmp_path / "reports"),
    )

    reference = tmp_path / "header_only.nii.gz"
    affine = np.diag([1.0, 2.0, 3.0, 1.0])
    header_only = SimpleNamespace(shape=(8, 8, 8), affine=affine)
    original_load = nib.load

    monkeypatch.setattr(
        voxel_flow.nib,
        "load",
        lambda path: header_only if Path(path) == reference else original_load(path),
    )
    monkeypatch.setattr(
        voxel_flow,
        "read_report_studies",
        lambda root: pd.DataFrame({"study_uid": ["study"]}),
    )

    features = SimpleNamespace(
        index={"study": 0}, offsets=torch.tensor([0, 3]), tokens=torch.randn(3, 768)
    )

    monkeypatch.setattr(voxel_flow, "ReportFeatureCache", lambda root: features)

    args = SimpleNamespace(
        study_uid="study",
        geometry=str(reference),
        contrast="noncontrast",
        weights="raw",
        checkpoint="synthetic",
    )

    output = tmp_path / "sample"

    voxel_flow.sample_one_mri(args, config, TinyPredictor().eval(), output)

    generated = original_load(output / "sample.nii.gz")

    assert generated.shape == (8, 8, 8)

    np.testing.assert_allclose(generated.affine, affine)

    assert json.loads((output / "settings.json").read_text())["model_evaluations"] == 3
