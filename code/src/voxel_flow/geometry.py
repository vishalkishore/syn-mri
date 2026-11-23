"""Voxel rearrangements and physical RAS geometry. Spatial order is X,Y,Z."""

import math

import torch
from torch.nn import functional as F


def space_to_channels(x, factor):
    """Move each factor^3 block of voxels into channels.

    [B, C, X, Y, Z] -> [B, C * factor^3, X / factor, Y / factor, Z / factor].
    No values are lost; channels_to_space is the exact inverse.
    """

    batch, channels, size_x, size_y, size_z = x.shape

    if factor < 1 or any(size % factor for size in (size_x, size_y, size_z)):
        raise ValueError("Packing requires divisible spatial dimensions")

    x = x.reshape(
        batch,
        channels,
        size_x // factor,
        factor,
        size_y // factor,
        factor,
        size_z // factor,
        factor,
    )

    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)

    return x.reshape(
        batch,
        channels * factor**3,
        size_x // factor,
        size_y // factor,
        size_z // factor,
    )


def channels_to_space(x, factor):
    """Exact inverse of space_to_channels: channels back into factor^3 voxel blocks."""

    batch, channels, size_x, size_y, size_z = x.shape

    if factor < 1 or channels % factor**3:
        raise ValueError("Unpacking requires factor**3 phase channels")

    x = x.reshape(
        batch, channels // factor**3, factor, factor, factor, size_x, size_y, size_z
    )

    return x.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(
        batch, channels // factor**3, size_x * factor, size_y * factor, size_z * factor
    )


def voxel_center_indices(shape, stride, z_before=4, device=None):
    """Native-voxel index of the center of every cell in a grid with this stride.

    Returns [N, 3] in X, Y, Z order. Z is shifted back by the z_before padded slices,
    so the indices refer to the unpadded MRI.
    """

    axes = [
        stride * torch.arange(size, dtype=torch.float32, device=device)
        + (stride - 1) / 2
        for size in shape
    ]

    centers = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    centers[:, 2] -= z_before

    return centers


def physical_positions(centers, affine):
    """Convert [N, 3] voxel indices to [B, N, 3] RAS positions in mm.

    Each batch item uses its own affine.
    """

    return (
        torch.einsum("bij,nj->bni", affine[:, :3, :3].float(), centers.float())
        + affine[:, None, :3, 3].float()
    )


def group_into_neighborhoods(features):
    """Split the local grid into 4x4x4 neighborhoods: [B, C, X, Y, Z] -> [B, N, 64, C].

    Each neighborhood of local cells sits under one global token.
    """

    batch, channels, size_x, size_y, size_z = features.shape

    if any(size % 4 for size in (size_x, size_y, size_z)):
        raise ValueError("Local grid must be divisible by four")

    return (
        features.reshape(
            batch, channels, size_x // 4, 4, size_y // 4, 4, size_z // 4, 4
        )
        .permute(0, 2, 4, 6, 3, 5, 7, 1)
        .reshape(batch, -1, 64, channels)
    )


def global_neighbors_of_local(local_shape, global_shape):
    """For each local cell, find the 8 surrounding global tokens.

    Returns flat global indices [N, 8], a validity mask [N, 8], and the physical
    offsets between each local cell and its neighbors. At the grid edge, clipped
    neighbors repeat; the repeats are masked out, never given extra weight.
    """

    axes = [torch.arange(size) for size in local_shape]
    cell = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    # Centers in native-voxel units: a local cell spans 4 voxels (center 4*i + 1.5),
    # a global token spans 16 voxels (center 16*g + 7.5).
    lower = torch.floor((4 * cell + 1.5 - 7.5) / 16).long()
    corners = torch.tensor([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    neighbors = lower[:, None] + corners[None]

    for axis, size in enumerate(global_shape):
        neighbors[..., axis].clamp_(0, size - 1)

    indices = (neighbors[..., 0] * global_shape[1] + neighbors[..., 1]) * global_shape[
        2
    ] + neighbors[..., 2]

    valid = torch.ones_like(indices, dtype=torch.bool)

    for slot in range(1, 8):
        valid[:, slot] = ~(indices[:, :slot] == indices[:, slot : slot + 1]).any(dim=1)

    local_center = 4 * cell.float() + 1.5
    global_center = 16 * neighbors.float() + 7.5
    offsets = local_center[:, None] - global_center

    unique_offsets, offset_ids = torch.unique(
        offsets.reshape(-1, 3), dim=0, return_inverse=True
    )

    return indices, valid, unique_offsets, offset_ids.reshape(-1, 8)


def rotary_position_3d(
    features, positions, axis_dims=(20, 20, 24), base=10000.0, scale_mm=16.0
):
    """Apply 3D rotary position encoding (RoPE) to attention features.

    features: [B, heads, N, D]; positions: [B, N, 3] in mm. The head width D is
    split into one part per axis (axis_dims); each part encodes one axis.
    """

    if sum(axis_dims) != features.shape[-1] or any(d % 2 for d in axis_dims):
        raise ValueError("RoPE needs even axis widths summing to head width")

    pieces = []
    start = 0

    for axis, width in enumerate(axis_dims):
        inverse_frequencies = base ** (
            -2 * torch.arange(width // 2, device=features.device).float() / width
        )

        angles = (
            positions[:, None, :, axis : axis + 1].float()
            / scale_mm
            * inverse_frequencies
        )

        cosine = angles.cos().to(features.dtype)
        sine = angles.sin().to(features.dtype)

        pairs = features[..., start : start + width].reshape(
            *features.shape[:-1], width // 2, 2
        )

        first, second = pairs.unbind(-1)

        pieces.append(
            torch.stack(
                (first * cosine - second * sine, first * sine + second * cosine), dim=-1
            ).flatten(-2)
        )

        start += width

    return torch.cat(pieces, dim=-1)


def noise_corrected_gradients(x, times, affine, noise_std=1.0, z_padding=(4, 4)):
    """Local image-structure features for each global token, corrected for noise.

    Computes the average outer product of the physical image gradient over each
    16^3 cell, then subtracts the part that the noise in the current state adds.
    Output [B, tokens, 6]: xx, yy, zz, sqrt(2)*xy, sqrt(2)*xz, sqrt(2)*yz.
    A voxel contributes only when its three forward neighbors exist; padding
    never contributes.
    """

    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        center = x[:, 0, :-1, :-1, :-1]

        differences = torch.stack(
            [
                x[:, 0, 1:, :-1, :-1] - center,
                x[:, 0, :-1, 1:, :-1] - center,
                x[:, 0, :-1, :-1, 1:] - center,
            ],
            dim=1,
        )

        inverse = torch.linalg.inv(affine[:, :3, :3].float())

        gradients = torch.einsum(
            "bij,bjxyz->bixyz", inverse.transpose(-1, -2), differences
        )

        index_covariance = torch.ones(3, 3, device=x.device) + torch.eye(
            3, device=x.device
        )

        covariance = inverse.transpose(-1, -2) @ index_covariance @ inverse
        padding = (z_padding[0], z_padding[1] + 1, 0, 1, 0, 1)
        counts = F.avg_pool3d(F.pad(torch.ones_like(center[:, None]), padding), 16, 16)

        if torch.any(counts == 0):
            raise ValueError("Global cell has no native forward-difference anchors")

        channels = []
        noise = (1 - times.float()).square() * noise_std**2

        for first, second in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
            product = (
                gradients[:, first : first + 1] * gradients[:, second : second + 1]
            )

            average = F.avg_pool3d(F.pad(product, padding), 16, 16) / counts

            average = (
                average
                - (noise * covariance[:, first, second])[:, None, None, None, None]
            )

            if first != second:
                average = average * math.sqrt(2)

            channels.append(average)

        return torch.cat(channels, dim=1).flatten(2).transpose(1, 2)
