import math

import pytest
import torch

from src.voxel_flow.geometry import noise_corrected_gradients
from src.voxel_flow.geometry import global_neighbors_of_local
from src.voxel_flow.geometry import group_into_neighborhoods
from src.voxel_flow.geometry import voxel_center_indices
from src.voxel_flow.geometry import space_to_channels
from src.voxel_flow.geometry import physical_positions
from src.voxel_flow.geometry import rotary_position_3d
from src.voxel_flow.geometry import channels_to_space


@pytest.mark.parametrize("factor", [2, 4])
def test_packing_phase_order_and_inverse(factor):
    x = torch.arange(2 * 8 * 8 * 8).reshape(1, 2, 8, 8, 8)
    packed = space_to_channels(x, factor)

    assert torch.equal(channels_to_space(packed, factor), x)

    for dx in range(factor):
        for dy in range(factor):
            for dz in range(factor):
                phase = (dx * factor + dy) * factor + dz

                assert packed[0, phase, 0, 0, 0] == x[0, 0, dx, dy, dz]


def test_grouping_matches_spatial_blocks():
    x = torch.arange(8 * 8 * 8).reshape(1, 1, 8, 8, 8)
    grouped = group_into_neighborhoods(x)

    assert torch.equal(grouped[0, 3, :, 0], x[0, 0, :4, 4:8, 4:8].flatten())


def test_centers_physical_spacing_and_padding():
    affine = torch.diag(torch.tensor([2.0, 3.0, 4.0, 1.0]))[None]
    affine[:, :3, 3] = torch.tensor([10.0, 20.0, 30.0])
    centers = voxel_center_indices((2, 2, 2), 4)

    torch.testing.assert_close(centers[0], torch.tensor([1.5, 1.5, -2.5]))
    torch.testing.assert_close(
        physical_positions(centers, affine)[0, 0], torch.tensor([13.0, 24.5, 20.0])
    )


def test_neighbors_bracket_every_query_without_duplicate_weight():
    shape = (8, 8, 8)
    indices, valid, offsets, ids = global_neighbors_of_local(shape, (2, 2, 2))

    assert valid.any(-1).all()
    assert offsets.shape[0] <= 512

    for row, allowed in zip(indices, valid):
        assert len(set(row[allowed].tolist())) == allowed.sum()

    assert valid[0].sum() == 1

    # Local cell (3,3,3) brackets global cells (0,1) on every axis.
    row = (3 * 8 + 3) * 8 + 3

    assert set(indices[row, valid[row]].tolist()) == set(range(8))


def test_rope_axis_pairs_and_translation_dot_product():
    torch.manual_seed(3)

    q = torch.randn(2, 3, 5, 64)
    k = torch.randn_like(q)
    positions = torch.randn(2, 5, 3)

    torch.testing.assert_close(rotary_position_3d(q, positions * 0), q)

    shift = torch.tensor([32.0, -11.0, 8.0])
    before = rotary_position_3d(q, positions) @ rotary_position_3d(
        k, positions
    ).transpose(-1, -2)

    after = rotary_position_3d(q, positions + shift) @ rotary_position_3d(
        k, positions + shift
    ).transpose(-1, -2)

    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)

    x_only = torch.tensor([[[16.0, 0.0, 0.0]]]).expand(2, 5, 3)
    rotated = rotary_position_3d(q, x_only)

    torch.testing.assert_close(rotated[..., 20:], q[..., 20:])


@pytest.mark.parametrize("oblique", [False, True])
def test_structure_linear_plane_and_physical_gradient(oblique):
    affine = torch.diag(torch.tensor([2.0, 3.0, 4.0, 1.0]))[None]

    if oblique:
        affine[0, 0, 1] = 0.5

    grid = torch.stack(
        torch.meshgrid(
            torch.arange(32.0), torch.arange(32.0), torch.arange(24.0), indexing="ij"
        )
    )

    gradient_index = torch.tensor([1.0, 2.0, 3.0])
    x = torch.einsum("i,ixyz->xyz", gradient_index, grid)[None, None]
    gradient = torch.linalg.inv(affine[0, :3, :3]).T @ gradient_index

    expected = torch.stack(
        [
            gradient[0] ** 2,
            gradient[1] ** 2,
            gradient[2] ** 2,
            math.sqrt(2) * gradient[0] * gradient[1],
            math.sqrt(2) * gradient[0] * gradient[2],
            math.sqrt(2) * gradient[1] * gradient[2],
        ]
    )

    actual = noise_corrected_gradients(x, torch.ones(1), affine)

    torch.testing.assert_close(
        actual, expected[None, None].expand_as(actual), atol=2e-5, rtol=2e-5
    )


@pytest.mark.parametrize("time", [0.0, 0.5, 0.9])
def test_structure_noise_expectation_with_sampling_uncertainty(time):
    torch.manual_seed(12)

    count = 64
    affine = torch.diag(torch.tensor([1.0, 1.5, 2.0, 1.0]))[None].expand(count, -1, -1)
    y = torch.arange(24.0).reshape(1, 1, 1, 1, 24).expand(1, 1, 32, 32, 24) * 0.2
    noise = torch.randn(count, 1, 32, 32, 24)

    result = noise_corrected_gradients(
        (1 - time) * noise + time * y, torch.full((count,), time), affine
    )

    expected = noise_corrected_gradients(y, torch.ones(1), affine[:1]) * time**2
    # Reduce cells within each independently drawn volume before estimating SE.
    estimates = result.mean(1)
    mean = estimates.mean(0)
    se = estimates.std(0) / math.sqrt(count)

    assert ((mean - expected.mean(1)[0]).abs() < 5 * se + 2e-5).all()
