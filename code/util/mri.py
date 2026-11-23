"""Standardized MRI validation shared by generation routes."""

import nibabel as nib
import numpy as np


def load_normalized_mri(image_path):
    """Read a brain-max-normalized MRI on the fixed anatomical RAS grid."""

    image = nib.load(image_path)

    if image.shape != (240, 240, 168) or nib.aff2axcodes(image.affine) != (
        "R",
        "A",
        "S",
    ):
        raise ValueError(f"Expected RAS MRI shape (240, 240, 168): {image_path}")

    if not np.allclose(
        image.affine[:3, :3],
        np.diag(np.diag(image.affine[:3, :3])),
        atol=1e-5,
    ):
        raise ValueError(f"Expected an anatomical RAS grid: {image_path}")

    volume = image.get_fdata(dtype=np.float32)

    if (
        not np.isfinite(volume).all()
        or volume.min() < 0
        or volume.max() > 1
        or not np.isclose(volume.max(), 1, atol=1e-6)
    ):
        raise ValueError(f"Expected brain-max-normalized MRI in [0, 1]: {image_path}")

    return volume
