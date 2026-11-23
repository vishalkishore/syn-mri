"""MedicalNet FID using the preprocessing in WDM-3D's BraTS evaluation.

Call ``preprocess_volume`` per scan before batching, then ``get_activations``,
``feature_mean_and_covariance``, and ``calculate_frechet_distance`` for each cohort.
Native real and generated scans must use the same preprocessing and axis order.
This module does not reorient or resample physical image grids.
"""

import importlib.util
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from huggingface_hub import hf_hub_download
from scipy import linalg
from torch.nn import functional as F


def get_feature_extractor(device=None):
    """Load the frozen WDM-3D ResNet-50 with MedicalNet pretrained weights."""

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_path = Path(__file__).resolve().parents[1] / "third_party/wdm_resnet.py"
    spec = importlib.util.spec_from_file_location("wdm_resnet", model_path)
    resnet = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(resnet)

    model = resnet.resnet50(
        sample_input_D=168,
        sample_input_H=260,
        sample_input_W=260,
        num_seg_classes=2,
        shortcut_type="B",
        no_cuda=False,
    )

    path = hf_hub_download(
        repo_id="TencentMedicalNet/MedicalNet-Resnet50",
        filename="resnet_50_23dataset.pth",
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    state = {
        key.removeprefix("module."): value
        for key, value in checkpoint["state_dict"].items()
    }

    # WDM replaces MedicalNet's segmentation head with parameter-free pooling.
    state = {
        key: value for key, value in state.items() if not key.startswith("conv_seg.")
    }

    model.load_state_dict(state, strict=True)

    return model.to(device).eval().requires_grad_(False)


def preprocess_volume(volume, img_size=128, *, preprocessed=False):
    """Return a float32 tensor shaped (1, S, S, S), where S is 128 or 256.

    Native input must be a finite 3D array with each axis at most 256 voxels.
    Clip at quantiles 0.001 and 0.999, min-max scale, center-pad to 256 cubed,
    then average-pool if S=128. For BraTS (240, 240, 155), padding matches WDM
    exactly; other shapes use the same centering rule, without resizing anatomy.

    Set preprocessed=True only for WDM-style saved volumes already on the target
    grid and intensity scale. These pass through unchanged, including generated
    intensity overshoot. Do not use this flag for native generated scans.
    """

    if img_size not in (128, 256):
        raise ValueError("img_size must be 128 or 256")

    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()

    volume = np.asarray(volume)

    if volume.ndim != 3 or min(volume.shape) < 1:
        raise ValueError(
            "Expected a nonempty 3D volume in the common cohort axis order"
        )

    if not np.isfinite(volume).all():
        raise ValueError("Volume contains nonfinite intensities")

    if preprocessed:
        if volume.shape != (img_size,) * 3:
            raise ValueError("Preprocessed volume must match the target cubic grid")

        return torch.tensor(volume, dtype=torch.float32).unsqueeze(0)

    if max(volume.shape) > 256:
        raise ValueError("Native volume axes must be at most 256 voxels")

    lower, upper = np.quantile(volume, [0.001, 0.999])

    if upper <= lower:
        raise ValueError("Volume has no intensity range after quantile clipping")

    normalized = (np.clip(volume, lower, upper) - lower) / (upper - lower)
    image = torch.zeros((1, 256, 256, 256), dtype=torch.float32)

    slices = tuple(
        slice((256 - size) // 2, (256 - size) // 2 + size) for size in volume.shape
    )

    image[(slice(None),) + slices] = torch.tensor(normalized, dtype=torch.float32)

    if img_size == 128:
        image = F.avg_pool3d(image, kernel_size=2, stride=2)

    return image


def preprocess_native_volume(volume, affine):
    """Paper-based native preprocessing; foreground is assumed strictly positive.

    Match the previous native FID run: RAS, foreground quantiles 0.10/0.99,
    global [-1,1] scaling, center crop/pad without interpolation. New padding
    is zero. Published evaluator code is unavailable for exact mask parity.
    """

    volume = np.asarray(volume)

    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("Expected a finite 3D MRI")

    image = nib.as_closest_canonical(nib.Nifti1Image(volume.astype(np.float32), affine))

    volume = image.get_fdata(dtype=np.float32)
    foreground = volume[volume > 0]

    if foreground.size < 2:
        raise ValueError("MRI has no positive foreground for normalization")

    lower, upper = np.quantile(foreground, [0.10, 0.99])

    if upper <= lower:
        raise ValueError("Foreground has no intensity range after clipping")

    normalized = 2 * (np.clip(volume, lower, upper) - lower) / (upper - lower) - 1
    shape = (240, 240, 168)
    output = np.zeros(shape, dtype=np.float32)
    source = []
    destination = []

    for size, target in zip(normalized.shape, shape):
        count = min(size, target)

        source.append(slice((size - count) // 2, (size - count) // 2 + count))

        destination.append(slice((target - count) // 2, (target - count) // 2 + count))

    output[tuple(destination)] = normalized[tuple(source)]

    return torch.from_numpy(output[None])


@torch.inference_mode()
def get_activations(model, data_loader, device=None):
    """Extract one feature row per preprocessed scan, including partial batches.

    Loader batches must be (N, 1, S, S, S) tensors or (tensor, metadata) pairs.
    Preprocessing must be applied per scan before collation.
    """

    device = (
        torch.device(device) if device is not None else next(model.parameters()).device
    )
    model = model.to(device).eval()
    activations = []

    for batch in data_loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]

        if (
            batch.ndim != 5
            or batch.shape[1] != 1
            or tuple(batch.shape[2:]) not in ((128,) * 3, (256,) * 3, (240, 240, 168))
        ):
            raise ValueError("Expected preprocessed batches shaped (N, 1, S, S, S)")

        if not torch.isfinite(batch).all():
            raise ValueError("Batch contains nonfinite intensities")

        features = model(batch.to(device=device, dtype=torch.float32))

        activations.append(features.cpu().numpy())

    if not activations:
        raise ValueError("Cannot extract features from an empty data loader")

    return np.concatenate(activations, axis=0).astype(np.float64)


def feature_mean_and_covariance(activations):
    """Compute sample mean and unbiased covariance over scan feature rows."""

    activations = np.asarray(activations, dtype=np.float64)

    if activations.ndim != 2 or activations.shape[0] < 2 or activations.shape[1] < 1:
        raise ValueError("Expected at least two scans in a 2D feature array")

    if not np.isfinite(activations).all():
        raise ValueError("Features contain nonfinite values")

    return activations.mean(axis=0), np.atleast_2d(np.cov(activations, rowvar=False))


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute Gaussian Frechet distance with WDM's singular-product fallback."""

    mu1 = np.atleast_1d(np.asarray(mu1, dtype=np.float64))
    mu2 = np.atleast_1d(np.asarray(mu2, dtype=np.float64))
    sigma1 = np.atleast_2d(np.asarray(sigma1, dtype=np.float64))
    sigma2 = np.atleast_2d(np.asarray(sigma2, dtype=np.float64))

    if (
        mu1.ndim != 1
        or mu1.size == 0
        or mu1.shape != mu2.shape
        or sigma1.shape != (mu1.size, mu1.size)
        or sigma1.shape != sigma2.shape
    ):
        raise ValueError("Means and covariances must have matching feature dimensions")

    if not all(np.isfinite(value).all() for value in (mu1, mu2, sigma1, sigma2)):
        raise ValueError("Means and covariances must contain only finite values")

    difference = mu1 - mu2
    covariance_mean = linalg.sqrtm(sigma1 @ sigma2)

    if not np.isfinite(covariance_mean).all():
        offset = np.eye(mu1.size) * eps
        covariance_mean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if not np.isfinite(covariance_mean).all():
        raise ValueError("Covariance matrix square root is nonfinite")

    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diag(covariance_mean).imag, 0, atol=1e-3):
            raise ValueError("Covariance matrix square root has an imaginary component")

        covariance_mean = covariance_mean.real

    return float(
        difference @ difference
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2 * np.trace(covariance_mean)
    )
