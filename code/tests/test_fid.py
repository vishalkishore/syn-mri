import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from util.fid import calculate_frechet_distance
from util.fid import get_activations
from util.fid import preprocess_volume
from util.fid import feature_mean_and_covariance


@pytest.mark.parametrize("img_size", [128, 256])
def test_preprocessing_matches_wdm_brats(img_size):
    # WDM uses quantiles over the whole image, including background.
    volume = np.zeros((240, 240, 155), dtype=np.float32)

    volume[20:200, 30:210, 15:140] = np.linspace(1, 100, 125, dtype=np.float32)[
        None, None, :
    ]

    volume[100, 100, 100] = 10000
    clipped = np.clip(volume, np.quantile(volume, 0.001), np.quantile(volume, 0.999))
    normalized = (clipped - clipped.min()) / (clipped.max() - clipped.min())
    expected = torch.zeros((1, 256, 256, 256))
    expected[:, 8:-8, 8:-8, 50:-51] = torch.tensor(normalized)

    if img_size == 128:
        expected = F.avg_pool3d(expected, 2, 2)

    actual = preprocess_volume(volume, img_size)

    assert actual.dtype == torch.float32

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    assert volume[100, 100, 100] == 10000


def test_native_project_grid_preserves_axes_and_center():
    volume = np.broadcast_to(np.arange(168), (240, 240, 168))
    actual = preprocess_volume(volume, img_size=256)

    assert actual.shape == (1, 256, 256, 256)

    torch.testing.assert_close(
        actual[0, 8:248, 8:248, 44:212],
        torch.tensor(volume / 167, dtype=torch.float32),
    )

    assert actual[:, :8].count_nonzero() == 0
    assert actual[:, :, :, :44].count_nonzero() == 0


def test_preprocessed_generated_values_pass_through():
    volume = torch.linspace(-0.1, 1.2, 128**3).reshape(128, 128, 128)
    actual = preprocess_volume(volume, preprocessed=True)

    torch.testing.assert_close(actual[0], volume, rtol=0, atol=0)


@pytest.mark.parametrize(
    "volume, kwargs, message",
    [
        (np.zeros((4, 4)), {}, "3D"),
        (np.zeros((0, 4, 4)), {}, "nonempty"),
        (np.zeros((4, 4, 4)), {}, "intensity range"),
        (np.full((4, 4, 4), np.nan), {}, "nonfinite"),
        (np.zeros((257, 4, 4)), {}, "at most 256"),
        (np.zeros((4, 4, 4)), {"img_size": 64}, "128 or 256"),
        (np.zeros((4, 4, 4)), {"preprocessed": True}, "target cubic grid"),
    ],
)
def test_invalid_preprocessing_inputs(volume, kwargs, message):
    with pytest.raises(ValueError, match=message):
        preprocess_volume(volume, **kwargs)


def test_activations_include_last_partial_batch_and_metadata():
    model = torch.nn.Sequential(
        torch.nn.Conv3d(1, 2, 1, bias=False),
        torch.nn.Dropout(0.9),
        torch.nn.AdaptiveAvgPool3d(1),
        torch.nn.Flatten(),
    )

    with torch.no_grad():
        model[0].weight.fill_(1)

    scans = torch.arange(3, dtype=torch.float32)[:, None, None, None, None]
    scans = scans.expand(3, 1, 128, 128, 128)
    loader = DataLoader(TensorDataset(scans, torch.arange(3)), batch_size=2)
    actual = get_activations(model, loader, device="cpu")

    np.testing.assert_allclose(actual, [[0, 0], [1, 1], [2, 2]])

    assert actual.dtype == np.float64
    assert not model.training

    with pytest.raises(ValueError, match="empty"):
        get_activations(model, [], device="cpu")

    with pytest.raises(ValueError, match="preprocessed batches"):
        get_activations(model, [torch.zeros(1, 1, 8, 8, 8)], device="cpu")


def test_feature_statistics_and_analytic_fid():
    features = np.array([[0, 1], [1, 3], [2, 2]], dtype=np.float32)
    mean, covariance = feature_mean_and_covariance(features)

    np.testing.assert_allclose(mean, [1, 2])
    np.testing.assert_allclose(covariance, [[1, 0.5], [0.5, 1]])

    assert calculate_frechet_distance(
        mean, covariance, mean, covariance
    ) == pytest.approx(0, abs=1e-12)

    # For diagonal covariances, the covariance term is sum((std1 - std2)**2).
    assert calculate_frechet_distance(
        [0, 0], np.diag([1, 4]), [3, 4], np.diag([4, 9])
    ) == pytest.approx(27)

    assert calculate_frechet_distance([0, 0], np.zeros((2, 2)), [1, 2], np.eye(2)) == 7


def test_invalid_statistics():
    with pytest.raises(ValueError, match="at least two"):
        feature_mean_and_covariance([[1, 2]])

    with pytest.raises(ValueError, match="nonfinite"):
        feature_mean_and_covariance([[1, np.nan], [2, 3]])

    with pytest.raises(ValueError, match="matching feature dimensions"):
        calculate_frechet_distance([0, 0], np.eye(3), [0, 0], np.eye(3))

    with pytest.raises(ValueError, match="finite"):
        calculate_frechet_distance([np.nan], [[1]], [0], [[1]])


def test_voxstruct_native_preprocessing():
    from util.fid import preprocess_native_volume

    volume = np.zeros((240, 240, 168), dtype=np.float32)
    volume[10:230, 10:230, 10:158] = np.linspace(1, 10, 148)[None, None, :]
    actual = preprocess_native_volume(volume, np.eye(4))
    lower, upper = np.quantile(volume[volume > 0], [0.10, 0.99])
    expected = 2 * (np.clip(volume, lower, upper) - lower) / (upper - lower) - 1

    assert actual.shape == (1, 240, 240, 168)

    np.testing.assert_allclose(actual[0].numpy(), expected, rtol=1e-6, atol=1e-6)

    flipped = volume[::-1].copy()
    affine = np.eye(4)
    affine[0, 0] = -1
    affine[0, 3] = 239

    torch.testing.assert_close(preprocess_native_volume(flipped, affine), actual)

    padded = preprocess_native_volume(volume[10:230, 10:230], np.eye(4))

    assert padded.shape == actual.shape
    assert padded[:, :10].count_nonzero() == 0
