import copy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torch.optim.swa_utils import get_ema_multi_avg_fn

from src.voxel_flow.flow import clean_to_velocity
from src.voxel_flow.flow import neighbor_difference_errors
from src.voxel_flow.flow import generate_from_noise
from src.voxel_flow.flow import voxel_flow_loss
from src.voxel_flow.training import train
from src.voxel_flow.training import compute_validation_loss
from src.voxel_flow.training import validate_checkpoint
from test_voxel_flow_model import tiny_config


def test_loss_exact_constant_ramp_and_endpoint_floor():
    config = tiny_config()
    y = torch.zeros(1, 1, 8, 8, 8)
    outputs = {"clean": y, "global_clean": y}

    assert voxel_flow_loss(outputs, y, torch.tensor([0.5]), config)[0] == 0

    outputs = {"clean": y + 1, "global_clean": y}
    total, components = voxel_flow_loss(outputs, y, torch.tensor([0.99]), config)

    torch.testing.assert_close(total, torch.tensor(400.0), atol=0.001, rtol=1e-5)

    assert components["contrast"] == 0

    ramp = torch.arange(8.0).reshape(1, 1, 8, 1, 1).expand_as(y)
    terms = neighbor_difference_errors(ramp)

    torch.testing.assert_close(terms, torch.tensor([0.5, 0, 0, 2, 0, 0, 8, 0, 0]))


def test_velocity_subtracts_state_once():
    x, global_clean, local = [torch.randn(2, 1, 4, 4, 4) for _ in range(3)]
    t = torch.tensor([0.3, 0.8])
    actual = clean_to_velocity(global_clean + local, x, t)

    expected = (
        clean_to_velocity(global_clean, x, t)
        + local / (1 - t)[:, None, None, None, None]
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("steps", [1, 2, 64])
def test_oracle_sampler_count_endpoint_and_fp32(steps):
    target = torch.randn(2, 1, 4, 4, 4).bfloat16()
    seen = []

    def oracle(x, times):
        assert x.dtype == torch.float32
        assert times.max() < 1

        seen.append(times[0].item())

        return target

    result, count = generate_from_noise(oracle, torch.randn_like(target), steps)

    torch.testing.assert_close(result, target.float())

    assert count == len(seen) == 2 * steps - 1

    with pytest.raises(ValueError):
        generate_from_noise(oracle, target, 0)


def test_uneven_accumulation_and_ema_one_update_per_window():
    torch.manual_seed(4)

    model = nn.Linear(3, 2)
    split = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    split_optimizer = torch.optim.AdamW(split.parameters(), lr=0.01)
    ema = AveragedModel(split, multi_avg_fn=get_ema_multi_avg_fn(0.9))
    x, y = torch.randn(7, 3), torch.randn(7, 2)

    for start, stop in [(0, 5), (5, 7)]:
        optimizer.zero_grad()

        split_optimizer.zero_grad()

        ((model(x[start:stop]) - y[start:stop]).square().mean()).backward()

        for indices in torch.arange(start, stop).split(2):
            loss = (split(x[indices]) - y[indices]).square().mean()

            (loss * len(indices) / (stop - start)).backward()

        optimizer.step()

        split_optimizer.step()

        ema.update_parameters(split)

    assert int(ema.n_averaged) == 2

    for a, b in zip(model.parameters(), split.parameters()):
        torch.testing.assert_close(a, b)


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, times, condition, affine):
        clean = (
            x * self.weight
            + condition["report_drop"][:, None, None, None, None] * self.weight
        )

        return {"clean": clean, "global_clean": clean * 0.5}


def synthetic_dataset():
    result = []

    for i in range(7):
        result.append(
            {
                "volume": torch.full((1, 8, 8, 8), 0.1 * i),
                "affine": torch.eye(4, dtype=torch.float64),
                "condition": {
                    "report_tokens": torch.ones(2, 768),
                    "contrast_id": torch.tensor(1),
                },
                **{
                    key: str(i)
                    for key in ("case_id", "patient_uid", "study_uid", "series_id")
                },
            }
        )

    return result


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()

        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)

        for first, second in zip(a, b):
            assert_state_equal(first, second)
    else:
        assert a == b


def test_validation_batches_preserve_case_draws_and_metrics():
    config = tiny_config()

    config.update(num_workers=0, pin_memory=False, batch_size=1)

    dataset = synthetic_dataset()
    model = TinyPredictor()
    seen_states = []
    seen_times = []

    def capture_inputs(module, args):
        seen_states.append(args[0].clone())

        seen_times.append(args[1].clone())

    handle = model.register_forward_pre_hook(capture_inputs)

    try:
        single = compute_validation_loss(model, dataset, config)
        states = torch.cat(seen_states)
        times = torch.cat(seen_times)

        assert len(seen_states) == 7

        seen_states.clear()

        seen_times.clear()

        batched = compute_validation_loss(
            model, dataset, dict(config, validation_batch_size=3)
        )

        assert len(seen_states) == 3

        torch.testing.assert_close(torch.cat(seen_states), states, atol=0, rtol=0)
        torch.testing.assert_close(torch.cat(seen_times), times, atol=0, rtol=0)

        assert_state_equal(single, batched)

        generator = torch.Generator().manual_seed(config["seed"] + 100)
        expected_states = []
        expected_times = []

        for item in dataset:
            target = item["volume"][None]
            noise = torch.randn(target.shape, generator=generator)
            time = torch.rand(1, generator=generator)

            expected_states.append((1 - time) * noise + time * target)

            expected_times.append(time)

        torch.testing.assert_close(states, torch.cat(expected_states), atol=0, rtol=0)
        torch.testing.assert_close(times, torch.cat(expected_times), atol=0, rtol=0)
    finally:
        handle.remove()


def test_validation_bypasses_compiler_and_restores_cudnn_benchmark():
    config = tiny_config()

    config.update(num_workers=0, pin_memory=False, batch_size=2)

    model = TinyPredictor()

    def unexpected_compile(*args, **kwargs):
        pytest.fail("Validation must bypass compiled training regions")

    model.compile(backend=unexpected_compile)

    def check_flags(module, args):
        assert torch.backends.cudnn.enabled
        assert not torch.backends.cudnn.benchmark

    handle = model.register_forward_pre_hook(check_flags)

    try:
        with torch.backends.cudnn.flags(enabled=True, benchmark=True):
            result = compute_validation_loss(model, synthetic_dataset()[:2], config)

            assert torch.backends.cudnn.enabled
            assert torch.backends.cudnn.benchmark

        assert result["cases"] == 2
    finally:
        handle.remove()


def test_completed_training_is_saved_before_validation_failure(tmp_path, monkeypatch):
    config = tiny_config()

    config.update(
        batch_size=2,
        gradient_accumulation_steps=1,
        num_workers=0,
        pin_memory=False,
        epochs=1,
        max_updates=None,
        checkpoint_every_updates=100,
        validation_every_epochs=1,
        run_dir=str(tmp_path / "validation_failure"),
    )

    def failed_validation(*args):
        raise RuntimeError("Validation failed")

    monkeypatch.setattr(
        "src.voxel_flow.training.compute_validation_loss", failed_validation
    )

    dataset = synthetic_dataset()

    with pytest.raises(RuntimeError, match="Validation failed"):
        train(config, dataset, dataset[:2], model=TinyPredictor())

    state = torch.load(Path(config["run_dir"]) / "last.pt", weights_only=True)

    assert state["optimizer_updates"] == 4
    assert state["next_sample"] == len(dataset)

    monkeypatch.undo()

    train(
        config,
        dataset,
        dataset[:2],
        model=TinyPredictor(),
        resume=Path(config["run_dir"]) / "last.pt",
    )

    resumed = torch.load(Path(config["run_dir"]) / "last.pt", weights_only=True)

    assert resumed["optimizer_updates"] == 4
    assert resumed["epoch"] == 1


def test_update_boundary_restart_is_exact_including_short_windows(tmp_path):
    config = tiny_config()

    config.update(
        batch_size=2,
        gradient_accumulation_steps=2,
        num_workers=0,
        pin_memory=False,
        manifest="synthetic",
        report_token_cache="synthetic",
        epochs=3,
        max_updates=4,
        checkpoint_every_updates=1,
        checkpoint_epochs=[],
        run_dir=str(tmp_path / "continuous"),
    )

    dataset = synthetic_dataset()

    train(config, dataset, dataset[:2], model=TinyPredictor())

    split_config = dict(config, max_updates=1, run_dir=str(tmp_path / "resumed"))

    train(split_config, dataset, dataset[:2], model=TinyPredictor())

    split_config["max_updates"] = 4

    train(
        split_config,
        dataset,
        dataset[:2],
        resume=Path(split_config["run_dir"]) / "last.pt",
        model=TinyPredictor(),
    )

    first = torch.load(Path(config["run_dir"]) / "last.pt", weights_only=True)
    second = torch.load(Path(split_config["run_dir"]) / "last.pt", weights_only=True)

    for key in (
        "model",
        "optimizer",
        "ema_model",
        "epoch",
        "next_sample",
        "epoch_order",
        "optimizer_updates",
        "microbatches",
    ):
        assert_state_equal(first[key], second[key])

    for key in ("shuffle", "noise_time", "report_drop"):
        assert_state_equal(first["generators"][key], second["generators"][key])

    assert first["ema_updates"] == 4

    bad = dict(first, architecture="stage1")

    with pytest.raises(ValueError):
        validate_checkpoint(bad, config)

    bad = dict(first, architecture="parallel_voxel_flow_v2_2")

    with pytest.raises(ValueError, match="revision 2.3"):
        validate_checkpoint(bad, config)

    bad = dict(first, cache_identity={})

    with pytest.raises(ValueError):
        validate_checkpoint(bad, config)


def test_resume_allows_evaluation_changes_but_preserves_training_contract():
    from src.voxel_flow.config import ARCHITECTURE
    from src.voxel_flow.training import cache_fingerprint

    config = tiny_config()

    state = {
        "architecture": ARCHITECTURE,
        "config": config,
        "cache_identity": cache_fingerprint(config),
    }

    faster_evaluation = dict(
        config,
        sampling_steps=16,
        evaluation_samples=8,
        diversity_reports=2,
        diversity_samples_per_report=2,
        validation_batch_size=16,
    )

    validate_checkpoint(state, faster_evaluation)

    with pytest.raises(ValueError, match="source_noise_std"):
        validate_checkpoint(state, dict(faster_evaluation, source_noise_std=0.5))

    with pytest.raises(ValueError, match="effective optimizer batch"):
        validate_checkpoint(state, dict(config, world_size=2))


def test_gradient_diagnostics_preserve_gradients_and_report_capped_weights():
    from src.voxel_flow.training import gradient_diagnostics

    torch.manual_seed(9)

    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 1))

    model(torch.randn(4, 3)).square().mean().backward()

    original = [parameter.grad.clone() for parameter in model.parameters()]

    diagnostics = gradient_diagnostics(
        model, [torch.tensor([0.1, 0.5, 0.95, 0.99])], 0.05
    )

    expected = torch.sqrt(sum(gradient.square().sum() for gradient in original))

    measured = (
        sum(value**2 for value in diagnostics["gradient_norm_by_module"].values())
        ** 0.5
    )

    assert measured == pytest.approx(float(expected), rel=1e-5)

    assert diagnostics["time_bin_counts"] == {
        "0.0-0.5": 1,
        "0.5-0.9": 1,
        "0.9-0.95": 0,
        "0.95-1.0": 2,
    }

    assert diagnostics["mean_inverse_denominator_squared"] == pytest.approx(
        (1 / 0.9**2 + 4 + 400 + 400) / 4
    )

    for parameter, before in zip(model.parameters(), original):
        torch.testing.assert_close(parameter.grad, before, atol=0, rtol=0)


def test_endpoint_loss_gradient_is_capped_not_singular():
    config = tiny_config()
    target = torch.zeros(1, 1, 8, 8, 8)
    gradients = []

    for time in (0.5, 0.95, 0.99):
        prediction = torch.ones_like(target, requires_grad=True)
        outputs = {"clean": prediction, "global_clean": target}
        loss = voxel_flow_loss(outputs, target, torch.tensor([time]), config)[0]

        gradients.append(torch.autograd.grad(loss, prediction)[0])

    torch.testing.assert_close(gradients[1], gradients[2], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gradients[2], 100 * gradients[0])


def test_sobel_clean_loss_time_weight_and_gradients():
    from src.voxel_flow.flow import sobel_loss

    target = torch.zeros(1, 1, 8, 8, 8)
    ramp = torch.arange(8, dtype=torch.float32)[None, None, :, None, None]
    prediction = ramp.expand_as(target).clone().requires_grad_()

    assert sobel_loss(prediction, target, torch.tensor([0.3])) == 0

    full = sobel_loss(prediction, target, torch.tensor([1.0]))
    middle = sobel_loss(prediction, target, torch.tensor([0.65]))

    torch.testing.assert_close(middle, full * (0.5**1.5))
    # X derivative: six interior unit values and two boundary half values;
    # Y/Z derivatives vanish. Average over eight X positions and three axes.
    torch.testing.assert_close(full, torch.tensor((6 + 2 * 0.25) / 8 / 3))

    assert sobel_loss(target + 3, target, torch.tensor([1.0])) == 0
    assert sobel_loss(target, target, torch.tensor([1.0])) == 0

    full.backward()

    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0
