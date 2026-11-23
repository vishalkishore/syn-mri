"""Orthonormal 3D Haar from Report_conditioned_3D_MRI_generation.pdf, Appendix A.

Arrays use (..., channel, X, Y, Z); MRI input has one channel.
Band order: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH.
H = [1, -1]/sqrt(2). No normalization, padding, or axis reorientation.
Use float32 or float64. Each transformed spatial dimension must be even.
"""

import numpy as np


BAND_NAMES = ("LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH")


def haar_forward(volume):
    """Return (coarse, details), with 1 and 7 channels and halved spatial axes."""

    volume = np.asarray(volume)

    if volume.ndim < 4 or volume.shape[-4] != 1:
        raise ValueError("Expected (..., 1, X, Y, Z)")

    if volume.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("Haar requires float32 or float64")

    if any(size == 0 or size % 2 for size in volume.shape[-3:]):
        raise ValueError("Spatial dimensions must be positive and even; pad explicitly")

    if not np.isfinite(volume).all():
        raise ValueError("Volume contains NaN or Inf")

    x = volume[..., 0, :, :, :]
    a, b = x[..., 0::2, 0::2, 0::2], x[..., 0::2, 0::2, 1::2]
    c, d = x[..., 0::2, 1::2, 0::2], x[..., 0::2, 1::2, 1::2]
    e, f = x[..., 1::2, 0::2, 0::2], x[..., 1::2, 0::2, 1::2]
    g, h = x[..., 1::2, 1::2, 0::2], x[..., 1::2, 1::2, 1::2]

    bands = np.stack(
        [
            a + b + c + d + e + f + g + h,
            a - b + c - d + e - f + g - h,
            a + b - c - d + e + f - g - h,
            a - b - c + d + e - f - g + h,
            a + b + c + d - e - f - g - h,
            a - b + c - d - e + f - g + h,
            a + b - c - d - e - f + g + h,
            a - b - c + d - e + f + g - h,
        ],
        axis=-4,
    )

    bands *= volume.dtype.type(1 / np.sqrt(8))

    return bands[..., :1, :, :, :], bands[..., 1:, :, :, :]


def haar_inverse(coarse, details):
    """Reconstruct (..., 1, 2X, 2Y, 2Z) in the same physical Haar units."""

    coarse, details = np.asarray(coarse), np.asarray(details)

    if coarse.ndim < 4 or details.ndim != coarse.ndim:
        raise ValueError("Expected coarse/detail arrays with matching dimensions")

    if coarse.shape[-4] != 1 or details.shape[-4] != 7:
        raise ValueError("Expected 1 coarse channel and 7 detail channels")

    if (
        coarse.shape[:-4] != details.shape[:-4]
        or coarse.shape[-3:] != details.shape[-3:]
    ):
        raise ValueError("Coarse and detail grids must match")

    if coarse.dtype != details.dtype or coarse.dtype not in (
        np.dtype("float32"),
        np.dtype("float64"),
    ):
        raise TypeError("Coarse/details must share float32 or float64 dtype")

    if any(size == 0 for size in coarse.shape[-3:]):
        raise ValueError("Spatial dimensions must be positive")

    if not np.isfinite(coarse).all() or not np.isfinite(details).all():
        raise ValueError("Coefficients contain NaN or Inf")

    l = coarse[..., 0, :, :, :]
    b, c, d, e, f, g, h = [details[..., i, :, :, :] for i in range(7)]

    x = np.empty(
        coarse.shape[:-4] + (1,) + tuple(2 * s for s in coarse.shape[-3:]),
        dtype=coarse.dtype,
    )

    v = x[..., 0, :, :, :]
    v[..., 0::2, 0::2, 0::2] = l + b + c + d + e + f + g + h
    v[..., 0::2, 0::2, 1::2] = l - b + c - d + e - f + g - h
    v[..., 0::2, 1::2, 0::2] = l + b - c - d + e + f - g - h
    v[..., 0::2, 1::2, 1::2] = l - b - c + d + e - f - g + h
    v[..., 1::2, 0::2, 0::2] = l + b + c + d - e - f - g - h
    v[..., 1::2, 0::2, 1::2] = l - b + c - d - e + f - g + h
    v[..., 1::2, 1::2, 0::2] = l + b - c - d - e - f + g + h
    v[..., 1::2, 1::2, 1::2] = l - b - c + d - e + f + g - h
    x *= coarse.dtype.type(1 / np.sqrt(8))

    return x


def haar_decompose(volume, levels=3):
    """Return deepest coarse and (H1, H2, ...); split only coarse at each level."""

    if not isinstance(levels, int) or levels < 1:
        raise ValueError("levels must be a positive integer")

    if any(size % (2**levels) for size in np.shape(volume)[-3:]):
        raise ValueError("Spatial dimensions must be divisible by 2**levels")

    coarse = volume
    details = []

    for _ in range(levels):
        coarse, high = haar_forward(coarse)

        details.append(high)

    return coarse, tuple(details)


def haar_reconstruct(coarse, details):
    """Inverse pyramid; details must be ordered finest to coarsest, as returned."""

    if not details:
        raise ValueError("Expected at least one detail level")

    for high in reversed(details):
        coarse = haar_inverse(coarse, high)

    return coarse


def normalize_mri(volume, mask):
    """Normalize one masked MRI; return its array and reversible intensity scale."""

    volume, mask = np.asarray(volume), np.asarray(mask, dtype=bool)

    if volume.shape != mask.shape or not mask.any():
        raise ValueError("MRI and nonempty brain mask must have the same shape")

    if not np.isfinite(volume).all() or (volume < 0).any():
        raise ValueError("Hold MRI with nonfinite or negative intensities")

    if (volume[~mask] != 0).any():
        raise ValueError("Expected zero background outside the brain mask")

    maximum = float(volume[mask].max())

    if maximum == 0:
        raise ValueError("Hold MRI with zero brain maximum")

    return volume / maximum, maximum


def haar_model_bands(volume):
    """Three-level bands and reconstructed conditioning in physical Haar units."""

    a3, (h1, h2, h3) = haar_decompose(volume, levels=3)
    a2 = haar_inverse(a3, h3)
    a1 = haar_inverse(a2, h2)

    return {"A3": a3, "H3": h3, "H2": h2, "H1": h1, "A2": a2, "A1": a1}


def fit_haar_statistics(training_volumes):
    """Stream normalized training volumes; include all padded coefficient positions.

    Uses float64 parallel variance merging (population variance), one scan at a time.
    Caller must supply only the exact training subset and save its manifest.
    """

    moments = {}
    scans = 0

    for volume in training_volumes:
        if np.ndim(volume) != 4 or np.shape(volume)[0] != 1:
            raise ValueError("Statistics require one scan at a time: (1, X, Y, Z)")

        for name, band in haar_model_bands(volume).items():
            values = band.astype(np.float64).reshape(band.shape[-4], -1)
            n = values.shape[1]
            mean = values.mean(axis=1)
            m2 = np.square(values - mean[:, None]).sum(axis=1)

            count, old_mean, old_m2 = moments.get(
                name, (0, np.zeros_like(mean), np.zeros_like(mean))
            )

            delta = mean - old_mean
            total = count + n

            moments[name] = (
                total,
                old_mean + delta * n / total,
                old_m2 + m2 + delta**2 * count * n / total,
            )

        scans += 1

    if scans == 0:
        raise ValueError("No training images supplied")

    result = {}

    for name, (count, mean, m2) in moments.items():
        std = np.sqrt(m2 / count)

        result[name] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "scale": np.maximum(std, 1e-6).tolist(),
            "minimum_scale_used": (std < 1e-6).tolist(),
            "count": count,
        }

    return {"training_scans": scans, "bands": result}


def standardize_band(coefficients, statistics, inverse=False):
    """Apply a frozen per-channel pair; reverse before every inverse Haar step."""

    coefficients = np.asarray(coefficients)
    mean = np.asarray(statistics["mean"], dtype=coefficients.dtype)
    scale = np.asarray(statistics["scale"], dtype=coefficients.dtype)

    if (
        len(mean) != coefficients.shape[-4]
        or len(scale) != len(mean)
        or (scale <= 0).any()
    ):
        raise ValueError(
            "Statistics must match the band channels and have positive scales"
        )

    shape = (1,) * (coefficients.ndim - 4) + (len(mean), 1, 1, 1)
    mean, scale = mean.reshape(shape), scale.reshape(shape)

    return coefficients * scale + mean if inverse else (coefficients - mean) / scale
