"""Scientific invariants for anatomical-mask supervision."""

import nibabel as nib
import numpy as np
import pytest
import torch

from src.voxel_flow.data import load_brain_mask
from src.voxel_flow.flow import neighbor_difference_errors
from src.voxel_flow.flow import sobel_loss
from src.voxel_flow.flow import voxel_flow_loss


def config(**changes):
    return {
        "training_denominator_floor": 0.05,
        "contrast_offsets": [1, 2],
        "contrast_loss_weight": 0.0,
        "global_auxiliary_weight": 0.0,
        "voxel_loss_reduction": "mask_balanced",
        "voxel_brain_weight": 0.5,
        **changes,
    }


def test_balanced_per_patient_time_weight_and_backward():
    target = torch.zeros(2, 1, 8, 8, 8)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[0, :, :2] = True
    mask[1, :, :6] = True
    prediction = torch.where(mask, 2.0, 1.0).requires_grad_()
    times = torch.tensor([0.5, 0.99])

    _, terms = voxel_flow_loss(
        {"clean": prediction, "global_clean": target},
        target,
        times,
        config(),
        brain_mask=mask,
    )

    # Each patient has clean-space regional means 4 and 1, regardless of occupancy.
    torch.testing.assert_close(terms["voxel_brain"], torch.tensor(808.0))
    torch.testing.assert_close(terms["voxel_outside"], torch.tensor(202.0))
    torch.testing.assert_close(terms["voxel"], torch.tensor(505.0))

    terms["voxel"].backward()

    assert torch.isfinite(prediction.grad).all()
    assert (prediction.grad != 0).all()


def test_legacy_reduction_and_other_terms_unchanged():
    torch.manual_seed(7)

    target = torch.randn(2, 1, 8, 8, 8)
    prediction = torch.randn_like(target)
    outputs = {"clean": prediction, "global_clean": target}
    times = torch.tensor([0.2, 0.7])
    legacy = config(voxel_loss_reduction="whole_volume", contrast_loss_weight=1.0)
    _, old = voxel_flow_loss(outputs, target, times, legacy)

    expected = (
        ((prediction - target) / (1 - times[:, None, None, None, None])).square().mean()
    )

    torch.testing.assert_close(old["voxel"], expected, rtol=0, atol=0)

    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:, :, :4] = True

    _, new = voxel_flow_loss(
        outputs,
        target,
        times,
        dict(legacy, voxel_loss_reduction="mask_balanced"),
        brain_mask=mask,
    )

    for name in ("contrast", "global"):
        torch.testing.assert_close(old[name], new[name], rtol=0, atol=0)


def test_balanced_requires_anatomical_mask():
    target = torch.zeros(1, 1, 8, 8, 8)

    with pytest.raises(ValueError, match="brain_mask"):
        voxel_flow_loss(
            {"clean": target, "global_clean": target},
            target,
            torch.tensor([0.5]),
            config(),
        )


def test_brain_gradient_ignores_outside_errors_and_has_no_mask_edge():
    target = torch.zeros(1, 1, 12, 12, 12)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:, :, 2:10, 2:10, 2:10] = True
    prediction = torch.where(mask, 3.0, 100.0).requires_grad_()
    spacing = torch.tensor([[1.0, 2.0, 3.0]])

    contrast = neighbor_difference_errors(
        prediction, (1, 2), brain_mask=mask, spacing=spacing
    ).mean()

    sobel = sobel_loss(
        prediction, target, torch.ones(1), brain_mask=mask, spacing=spacing
    )

    assert contrast == 0
    assert sobel == 0

    (contrast + sobel).backward()

    assert torch.isfinite(prediction.grad).all()


def test_physical_ramp_signed_gradients_and_spacing():
    target = torch.zeros(1, 1, 12, 12, 12)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:, :, 2:10, 2:10, 2:10] = True
    ramp = torch.arange(12.0).reshape(1, 1, 12, 1, 1).expand_as(target)
    spacing = torch.tensor([[2.0, 1.0, 1.0]])
    contrast = neighbor_difference_errors(
        ramp, (1, 2), brain_mask=mask, spacing=spacing
    )

    torch.testing.assert_close(contrast, torch.tensor([0.125, 0, 0, 0.125, 0, 0]))

    sobel = sobel_loss(ramp, target, torch.ones(1), brain_mask=mask, spacing=spacing)

    torch.testing.assert_close(sobel, torch.tensor(1 / 12))

    assert sobel_loss(-ramp, ramp, torch.ones(1), brain_mask=mask, spacing=spacing) > 0

    assert (
        sobel_loss(ramp, target, torch.tensor([0.3]), brain_mask=mask, spacing=spacing)
        == 0
    )


def test_empty_gradient_interior_rejected():
    target = torch.zeros(1, 1, 8, 8, 8)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:, :, 3, 3, 3] = True

    with pytest.raises(ValueError, match="interior"):
        sobel_loss(
            target, target, torch.ones(1), brain_mask=mask, spacing=torch.ones(1, 3)
        )


@pytest.mark.parametrize("invalid", ["empty", "full", "nonbinary", "affine"])
def test_saved_mask_validation(tmp_path, invalid):
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[2:6, 2:6, 2:6] = 1
    affine = np.eye(4)

    if invalid == "empty":
        mask[:] = 0
    elif invalid == "full":
        mask[:] = 1
    elif invalid == "nonbinary":
        mask[3, 3, 3] = 2
    else:
        affine[0, 3] = 1

    path = tmp_path / "mask.nii.gz"

    nib.save(nib.Nifti1Image(mask, affine), path)

    with pytest.raises(ValueError):
        load_brain_mask(path, mask.shape, np.eye(4))


def test_mask_includes_zero_intensity_brain_voxels(tmp_path):
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[2:6, 2:6, 2:6] = 1
    path = tmp_path / "mask.nii.gz"

    nib.save(nib.Nifti1Image(mask, np.eye(4)), path)

    actual = load_brain_mask(path, mask.shape, np.eye(4))

    np.testing.assert_array_equal(actual, mask.astype(bool))


def test_dataset_collates_mask_only_as_supervision(tmp_path):
    from types import SimpleNamespace

    import pandas as pd

    from src.voxel_flow.data import VoxelReportDataset
    from src.voxel_flow.data import collate_batch
    from src.voxel_flow.data import save_voxel_target

    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[2:6, 2:6, 2:6] = 1
    volume = mask.astype(np.float32)
    volume[3, 3, 3] = 0  # This voxel remains part of the anatomical brain region.

    nib.save(nib.Nifti1Image(mask, np.eye(4)), tmp_path / "mask.nii.gz")

    save_voxel_target(tmp_path / "target.npz", volume, np.eye(4), 1.0)

    frame = pd.DataFrame(
        [
            dict(
                patient_uid="p",
                study_uid="s",
                series_id="t1",
                contrast="noncontrast",
                voxel_cache_path="target.npz",
                mask_path="mask.nii.gz",
            )
        ]
    )

    features = SimpleNamespace(
        index={"s": 0}, offsets=[0, 2], tokens=torch.ones(2, 768)
    )

    dataset = VoxelReportDataset(
        frame,
        tmp_path,
        features,
        (8, 8, 8),
        dataset_root=tmp_path,
        include_brain_mask=True,
    )

    batch = collate_batch([dataset[0]])

    assert batch["brain_mask"].dtype == torch.bool
    assert batch["brain_mask"].shape == batch["volume"].shape
    assert batch["brain_mask"][0, 0, 3, 3, 3]
    assert "brain_mask" not in batch["condition"]

    frame.loc[0, "mask_path"] = ""

    missing = VoxelReportDataset(
        frame,
        tmp_path,
        features,
        (8, 8, 8),
        dataset_root=tmp_path,
        include_brain_mask=True,
    )

    with pytest.raises(ValueError, match="mask_path"):
        missing[0]


def test_balanced_training_and_validation_smoke(tmp_path):
    from src.voxel_flow.training import train
    from src.voxel_flow.training import compute_validation_loss
    from test_voxel_flow_model import tiny_config
    from test_voxel_flow_training import TinyPredictor
    from test_voxel_flow_training import synthetic_dataset

    settings = tiny_config()

    settings.update(
        voxel_loss_reduction="mask_balanced",
        voxel_brain_weight=0.5,
        contrast_loss_region="brain",
        sobel_loss_region="brain",
        sobel_loss_weight=0.5,
        num_workers=0,
        pin_memory=False,
        batch_size=2,
        epochs=1,
        max_updates=None,
        gradient_accumulation_steps=1,
        run_dir=str(tmp_path / "training"),
    )

    dataset = synthetic_dataset()[:2]

    for item in dataset:
        item["brain_mask"] = torch.zeros_like(item["volume"], dtype=torch.bool)
        item["brain_mask"][:, 1:7, 1:7, 1:7] = True

    model = TinyPredictor()
    before = model.weight.detach().clone()

    train(settings, dataset, dataset, model=model)

    assert model.weight != before

    metrics = compute_validation_loss(model, dataset, settings)

    assert metrics["cases"] == 2
    assert (tmp_path / "training/last.pt").is_file()


def test_resume_cannot_silently_introduce_balanced_objective():
    from src.voxel_flow.config import ARCHITECTURE
    from src.voxel_flow.training import cache_fingerprint
    from src.voxel_flow.training import validate_checkpoint
    from test_voxel_flow_model import tiny_config

    legacy = tiny_config()

    state = dict(
        architecture=ARCHITECTURE,
        config=legacy,
        cache_identity=cache_fingerprint(legacy),
    )

    with pytest.raises(ValueError, match="voxel_loss_reduction"):
        validate_checkpoint(state, dict(legacy, voxel_loss_reduction="mask_balanced"))


@pytest.mark.parametrize(
    "key,value",
    [
        ("voxel_loss_reduction", "invalid"),
        ("voxel_brain_weight", float("nan")),
        ("voxel_brain_weight", 1.1),
        ("contrast_loss_region", "invalid"),
        ("sobel_loss_region", "invalid"),
    ],
)
def test_invalid_mask_loss_config(key, value):
    from pathlib import Path

    from src.voxel_flow.config import load_config
    from src.voxel_flow.config import validate_config

    settings = load_config(
        Path(__file__).resolve().parents[1] / "configs/voxel_flow.yaml"
    )

    settings[key] = value

    with pytest.raises(ValueError, match=key):
        validate_config(settings)
