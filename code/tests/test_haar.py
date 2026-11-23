"""Independent Appendix A checks, pyramid properties, and real template round trips."""

import itertools
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from util.haar import (
    BAND_NAMES,
    haar_forward,
    haar_inverse,
    haar_decompose,
    haar_reconstruct,
)


@pytest.mark.parametrize("position", list(itertools.product(range(2), repeat=3)))
def test_every_impulse_matches_independent_formula(position):
    x = np.zeros((1, 2, 2, 2), dtype=np.float64)
    x[(0,) + position] = 1
    coarse, details = haar_forward(x)
    actual = np.concatenate((coarse, details), axis=0)[:, 0, 0, 0]

    expected = [
        (-1.0) ** sum(p * i for p, i in zip(band, position)) / np.sqrt(8)
        for band in itertools.product(range(2), repeat=3)
    ]

    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_allclose(haar_inverse(coarse, details), x, atol=1e-15)

    assert BAND_NAMES == ("LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(1, 2, 2, 2), (1, 8, 16, 24), (3, 1, 8, 16, 24)])
def test_random_reconstruction_energy_and_immutability(dtype, shape):
    x = np.random.default_rng(42).normal(size=shape).astype(dtype)
    original = x.copy()
    coarse, details = haar_forward(x)
    c, d = coarse.copy(), details.copy()
    restored = haar_inverse(coarse, details)
    tolerance = 2e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(restored, x, atol=tolerance, rtol=tolerance)

    input_energy = np.square(x.astype(np.float64)).sum()

    output_energy = (
        np.square(coarse.astype(np.float64)).sum()
        + np.square(details.astype(np.float64)).sum()
    )

    assert output_energy == pytest.approx(input_energy, rel=tolerance)
    assert coarse.size + details.size == x.size
    assert restored.dtype == dtype

    assert (
        np.array_equal(x, original)
        and np.array_equal(coarse, c)
        and np.array_equal(details, d)
    )


def test_constant_and_single_high_band():
    x = np.full((1, 4, 6, 8), 3.0, dtype=np.float64)
    coarse, details = haar_forward(x)

    np.testing.assert_allclose(coarse, 3 * np.sqrt(8))

    assert not details.any()

    for band in range(7):
        high = np.zeros_like(details)
        high[band] = 1

        recovered_low, recovered_high = haar_forward(
            haar_inverse(np.zeros_like(coarse), high)
        )

        np.testing.assert_allclose(recovered_low, 0, atol=1e-15)
        np.testing.assert_allclose(recovered_high, high, atol=1e-15)


@pytest.mark.parametrize("shape", [(1, 240, 240, 168), (1, 224, 224, 160)])
def test_three_level_architecture_and_energy(shape):
    x = np.random.default_rng(42).normal(size=shape).astype(np.float32)
    coarse, details = haar_decompose(x)

    assert coarse.shape == (1,) + tuple(s // 8 for s in shape[-3:])

    for level, high in enumerate(details, 1):
        assert high.shape == (7,) + tuple(s // 2**level for s in shape[-3:])

    assert coarse.size + sum(d.size for d in details) == x.size

    np.testing.assert_allclose(
        haar_reconstruct(coarse, details), x, atol=3e-6, rtol=3e-6
    )

    energy = np.square(coarse.astype(np.float64)).sum() + sum(
        np.square(d.astype(np.float64)).sum() for d in details
    )

    assert energy == pytest.approx(np.square(x.astype(np.float64)).sum(), rel=2e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("masked", [False, True])
def test_real_template_full_volume(dtype, masked):
    root = Path(__file__).resolve().parents[2] / "data/templates/MNI152NLin2009cAsym"
    path = root / "mni_icbm152_t1_tal_nlin_asym_09c.nii"

    if not path.exists():
        pytest.skip("Real template integration test requires downloaded McGill T1")

    image = nib.load(path)
    x = image.get_fdata(dtype=dtype)

    if masked:
        mask = nib.load(root / "mni_icbm152_t1_tal_nlin_asym_09c_mask.nii")

        assert mask.shape == image.shape and np.allclose(mask.affine, image.affine)

        x *= mask.get_fdata() > 0

    # Explicit zero padding to a multiple of 8; no MRI resampling or normalization.
    padded = np.pad(x, [(0, (-s) % 8) for s in x.shape])[None]
    coarse, details = haar_decompose(padded)
    restored = haar_reconstruct(coarse, details)
    tolerance = 8 * np.finfo(dtype).eps * max(1.0, float(np.max(np.abs(x))))

    np.testing.assert_allclose(
        restored, padded, atol=tolerance, rtol=1e-6 if dtype == np.float32 else 1e-14
    )

    cropped = restored[0, : x.shape[0], : x.shape[1], : x.shape[2]]

    assert cropped.shape == image.shape
    assert np.max(np.abs(cropped - x)) <= tolerance

    energy = np.square(coarse.astype(np.float64)).sum() + sum(
        np.square(d.astype(np.float64)).sum() for d in details
    )

    assert energy == pytest.approx(
        np.square(x.astype(np.float64)).sum(),
        rel=2e-6 if dtype == np.float32 else 1e-14,
    )


def test_invalid_inputs():
    for shape in [(2, 2, 2), (2, 4, 4, 4), (1, 3, 4, 4), (1, 0, 4, 4)]:
        with pytest.raises(ValueError):
            haar_forward(np.zeros(shape, dtype=np.float32))

    with pytest.raises(TypeError):
        haar_forward(np.zeros((1, 2, 2, 2), dtype=np.int16))

    with pytest.raises(ValueError):
        haar_forward(np.full((1, 2, 2, 2), np.nan, dtype=np.float32))

    with pytest.raises(ValueError):
        haar_decompose(np.zeros((1, 4, 4, 4), dtype=np.float32), levels=3)

    with pytest.raises(ValueError):
        haar_inverse(np.zeros((1, 2, 2, 2)), np.zeros((7, 3, 2, 2)))

    with pytest.raises(TypeError):
        haar_inverse(np.zeros((1, 2, 2, 2), dtype=np.float32), np.zeros((7, 2, 2, 2)))


def test_inverse_arbitrary_coefficients_independent_formula():
    bands = np.random.default_rng(7).normal(size=8)

    restored = haar_inverse(
        bands[:1].reshape(1, 1, 1, 1), bands[1:].reshape(7, 1, 1, 1)
    )[0]

    for position in itertools.product(range(2), repeat=3):
        expected = sum(
            value * (-1.0) ** sum(p * i for p, i in zip(band, position))
            for value, band in zip(bands, itertools.product(range(2), repeat=3))
        ) / np.sqrt(8)

        assert restored[position] == pytest.approx(expected, abs=1e-15)


def test_noncontiguous_arrays():
    x = np.random.default_rng(8).normal(size=(2, 1, 8, 16, 24)).astype(np.float32)
    x = x[..., ::-1, :, ::-1]
    coarse, details = haar_decompose(x)

    np.testing.assert_allclose(
        haar_reconstruct(coarse, details), x, atol=2e-6, rtol=2e-6
    )


def test_frozen_statistics_population_and_staged_inverse():
    from util.haar import (
        normalize_mri,
        haar_model_bands,
        fit_haar_statistics,
        standardize_band,
    )

    rng = np.random.default_rng(90)
    volumes = []

    for shape, gain in [((1, 8, 8, 8), 10), ((1, 8, 16, 8), 30)]:
        v = rng.uniform(0, gain, shape)
        v[..., :2, :, :] = 0
        normalized, maximum = normalize_mri(v, v > 0)

        np.testing.assert_allclose(normalized * maximum, v)

        volumes.append(normalized)

    fitted = fit_haar_statistics(iter(volumes))

    assert sum(len(s["mean"]) for s in fitted["bands"].values()) == 24

    for name, stats in fitted["bands"].items():
        direct = np.concatenate(
            [
                haar_model_bands(v)[name].reshape(len(stats["mean"]), -1)
                for v in volumes
            ],
            axis=1,
        )

        np.testing.assert_allclose(stats["mean"], direct.mean(axis=1), atol=1e-14)
        np.testing.assert_allclose(stats["std"], direct.std(axis=1), atol=1e-14)

    bands = haar_model_bands(volumes[0])

    z = {
        name: standardize_band(band, fitted["bands"][name])
        for name, band in bands.items()
    }

    physical = {
        name: standardize_band(band, fitted["bands"][name], inverse=True)
        for name, band in z.items()
    }

    a2 = haar_inverse(physical["A3"], physical["H3"])

    np.testing.assert_allclose(
        standardize_band(a2, fitted["bands"]["A2"]), z["A2"], atol=1e-13
    )

    a1 = haar_inverse(a2, physical["H2"])

    np.testing.assert_allclose(
        standardize_band(a1, fitted["bands"]["A1"]), z["A1"], atol=1e-13
    )
    np.testing.assert_allclose(haar_inverse(a1, physical["H1"]), volumes[0], atol=1e-14)


def test_normalization_rejects_bad_mri_and_flags_constant_bands():
    from util.haar import normalize_mri, fit_haar_statistics

    for value in [0.0, -1.0, np.nan, np.inf]:
        with pytest.raises(ValueError):
            normalize_mri(
                np.full((1, 8, 8, 8), value), np.ones((1, 8, 8, 8), dtype=bool)
            )

    fitted = fit_haar_statistics([np.ones((1, 8, 8, 8), dtype=np.float64)])

    assert all(fitted["bands"]["H1"]["minimum_scale_used"])
    assert fitted["bands"]["H1"]["scale"] == [1e-6] * 7

    with pytest.raises(ValueError):
        fit_haar_statistics([])
