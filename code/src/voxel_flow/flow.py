"""Native linear flow objective and Heun integration with a final Euler step."""

import math

import torch
from torch.nn import functional as F


def make_noisy_state(target, noise, times):
    """Linear flow path: pure noise at t = 0, the real MRI at t = 1."""

    t = times.float()[:, None, None, None, None]

    return (1 - t) * noise.float() + t * target.float()


def loss_needs_brain_mask(config):
    """True when a configured loss term uses the saved brain mask."""

    return (
        config.get("voxel_loss_reduction", "whole_volume") == "mask_balanced"
        or config.get("contrast_loss_region", "whole_volume") == "brain"
        or config.get("sobel_loss_region", "whole_volume") == "brain"
    )


def region_mean(error_map, mask):
    """Average each MRI's selected voxels, without weighting larger regions more."""

    dims = (1, 2, 3, 4)
    counts = mask.sum(dims)

    if torch.any(counts == 0):
        raise ValueError("Every MRI must have a nonempty loss region/interior")

    return (error_map * mask).sum(dims) / counts


def neighbor_difference_errors(error, offsets=(1, 2, 4), brain_mask=None, spacing=None):
    """Squared error of differences between voxels 1, 2, and 4 apart on each axis.

    This penalizes wrong local contrast (edges and texture), not only wrong values.
    Returns one term per (offset, axis) pair.
    """

    terms = []

    for offset in offsets:
        for axis in (2, 3, 4):
            earlier = [slice(None)] * 5
            later = [slice(None)] * 5
            earlier[axis] = slice(None, -offset)
            later[axis] = slice(offset, None)
            difference = error[tuple(later)] - error[tuple(earlier)]

            if brain_mask is None:
                terms.append(difference.square().mean() / 2)
            else:
                # Require the entire line segment, not only its endpoints, in brain.
                kernel = [1, 1, 1]
                kernel[axis - 2] = offset + 1
                outside = (~brain_mask.bool()).float()
                valid = F.max_pool3d(outside, kernel, stride=1) == 0
                distance = spacing[:, axis - 2, None, None, None, None] * offset

                terms.append(
                    region_mean((difference / distance).square(), valid).mean() / 2
                )

    return torch.stack(terms)


def sobel_loss(
    prediction, target, times, start=0.3, power=1.5, brain_mask=None, spacing=None
):
    """Signed 3D gradient error; optional full-support brain interior, per mm.

    Legacy whole-volume mode retains voxel units and replicate padding. Brain
    mode uses a 3x3x3 interior and computes gradients before selecting locations.
    This clean-image term has its own time weight, without velocity scaling.
    """

    error = prediction.float() - target.float()
    derivative = error.new_tensor([-1, 0, 1])
    smoothing = error.new_tensor([1, 2, 1])
    padded = F.pad(error, (1, 1, 1, 1, 1, 1), mode="replicate")
    valid = None

    if brain_mask is not None:
        outside = F.pad((~brain_mask.bool()).float(), (1, 1, 1, 1, 1, 1), value=1)
        valid = F.max_pool3d(outside, 3, stride=1) == 0

    losses = []

    for axis in range(3):
        vectors = [derivative if index == axis else smoothing for index in range(3)]

        kernel = (
            vectors[0][:, None, None]
            * vectors[1][None, :, None]
            * vectors[2][None, None, :]
        ) / 32

        gradients = F.conv3d(padded, kernel[None, None])

        if valid is None:
            losses.append(gradients.square().mean(dim=(1, 2, 3, 4)))
        else:
            gradients = gradients / spacing[:, axis, None, None, None, None]

            losses.append(region_mean(gradients.square(), valid))

    weight = ((times.float() - start) / (1 - start)).clamp(0, 1).pow(power)

    return (torch.stack(losses).mean(0) * weight).mean()


def voxel_flow_loss(outputs, target, times, config, brain_mask=None, affine=None):
    if loss_needs_brain_mask(config) and brain_mask is None:
        raise ValueError("Selected loss requires the saved anatomical brain_mask")

    scale = (1 - times.float()).clamp_min(config["training_denominator_floor"])
    scale = scale[:, None, None, None, None]
    error = (outputs["clean"].float() - target.float()) / scale
    terms = {}

    if config.get("voxel_loss_reduction", "whole_volume") == "mask_balanced":
        error_map = error.square()
        brain = region_mean(error_map, brain_mask).mean()
        outside = region_mean(error_map, ~brain_mask.bool()).mean()
        weight = config.get("voxel_brain_weight", 0.5)
        voxel = weight * brain + (1 - weight) * outside

        terms.update(voxel_brain=brain, voxel_outside=outside)
    else:
        voxel = error.square().mean()

    physical_gradients = (
        config.get("contrast_loss_region", "whole_volume") == "brain"
        or config.get("sobel_loss_region", "whole_volume") == "brain"
    )

    spacing = None

    if physical_gradients:
        if affine is None:
            raise ValueError("Brain gradient losses require affine voxel spacing")

        # Tensor spatial axes follow the native NIfTI voxel axes (X, Y, Z).
        spacing = torch.linalg.vector_norm(affine[:, :3, :3].float(), dim=1)

    contrast = error.new_zeros(())

    if config["contrast_loss_weight"] > 0:
        contrast = neighbor_difference_errors(
            error,
            config["contrast_offsets"],
            brain_mask if config.get("contrast_loss_region") == "brain" else None,
            spacing,
        ).mean()

    coarse_error = (
        F.avg_pool3d(outputs["global_clean"].float(), 4, 4)
        - F.avg_pool3d(target.float(), 4, 4)
    ) / scale

    coarse = coarse_error.square().mean()

    total = (
        voxel
        + config["contrast_loss_weight"] * contrast
        + config["global_auxiliary_weight"] * coarse
    )

    terms.update(voxel=voxel, contrast=contrast, **{"global": coarse})

    if config.get("sobel_loss_weight", 0) > 0:
        with torch.autocast(device_type=target.device.type, enabled=False):
            sobel = sobel_loss(
                outputs["clean"],
                target,
                times,
                config.get("sobel_time_start", 0.3),
                config.get("sobel_time_power", 1.5),
                brain_mask if config.get("sobel_loss_region") == "brain" else None,
                spacing,
            )

        total = total + config["sobel_loss_weight"] * sobel
        terms["sobel"] = sobel

    return total, terms


def clean_to_velocity(clean, state, times):
    """Velocity that moves the current state straight to the predicted clean MRI."""

    if torch.any(times >= 1) or torch.any(times < 0):
        raise ValueError("Velocity requires 0 <= t < 1")

    return (clean.float() - state.float()) / (1 - times.float())[
        :, None, None, None, None
    ]


@torch.inference_mode()
def generate_from_noise(predict_clean, initial_noise, steps=64, observer=None):
    """Integrate from noise (t = 0) to an MRI (t = 1).

    Uses Heun steps, then a final Euler step that returns the last clean prediction.
    Returns the generated volume and the number of model evaluations (2 * steps - 1).
    observer(time, state, clean, velocity) is called once per step, if given.
    """

    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")

    x = initial_noise.float().clone()
    evaluations = 0

    for index in range(steps):
        time = index / steps
        following = (index + 1) / steps
        step_size = following - time
        times = torch.full((len(x),), time, device=x.device)
        clean = predict_clean(x, times).float()
        evaluations += 1
        velocity = clean_to_velocity(clean, x, times)

        if observer is not None:
            observer(time, x, clean, velocity)

        if index == steps - 1:
            x = clean

            continue

        predicted = x + step_size * velocity
        next_times = torch.full((len(x),), following, device=x.device)
        next_clean = predict_clean(predicted, next_times).float()
        evaluations += 1
        next_velocity = clean_to_velocity(next_clean, predicted, next_times)
        x = x + 0.5 * step_size * (velocity + next_velocity)

    return x, evaluations


def predict_clean(model, state, times, condition, affine, config):
    """Model prediction of the clean MRI, with optional report guidance.

    guidance_scale 1 uses the report as given. Other values mix the prediction
    with a no-report prediction: no_report + scale * (with_report - no_report).
    """

    with torch.autocast(
        device_type=state.device.type,
        dtype=torch.bfloat16,
        enabled=config["mixed_precision"] == "bf16",
    ):
        outputs = model(state, times, condition, affine)
        clean = outputs["clean"]
        scale = config["guidance_scale"]

        if not math.isfinite(scale):
            raise ValueError("guidance_scale must be finite")

        if scale != 1:
            null = dict(
                condition,
                report_drop=torch.ones(
                    len(state), device=state.device, dtype=torch.bool
                ),
            )

            null_clean = model(state, times, null, affine)["clean"]
            clean = null_clean + scale * (clean - null_clean)

    return clean.float()
