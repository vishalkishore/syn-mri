import copy
import os
from pathlib import Path

import pytest
import torch
import yaml

from src.voxel_flow.conditioning import VoxelConditioner
from src.voxel_flow.model import GlobalToLocalBridge
from src.voxel_flow.model import LocalReadout
from src.voxel_flow.model import ParallelVoxelFlow
from src.voxel_flow.model import OverlappingUpsample
from src.voxel_flow.model import ReportAttention
from src.voxel_flow.model import TokenRMSNorm
from src.voxel_flow.model import channel_rms_norm


def tiny_config():
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/voxel_flow.yaml").read_text()
    )

    # The tests expect the single-GPU settings of the original base config.
    config.pop("validation_batch_size")
    config.pop("log_gradient_diagnostics")

    config.update(
        world_size=1,
        batch_size=1,
        gradient_accumulation_steps=64,
        prefetch_factor=1,
        compile_cnn_blocks=False,
        native_shape=[32, 32, 24],
        local_width=8,
        global_width=24,
        condition_width=24,
        global_heads=2,
        local_heads=2,
        global_rope_axis_dims=[4, 4, 4],
        global_ff_hidden=32,
        global_decoder_channels=[16, 8, 4, 1],
        local_decoder_channels=[8, 4],
        time_embedding_width=8,
        time_mlp_hidden=48,
        spacing_mlp_hidden=8,
        stages=1,
        blocks_per_stage=1,
        bridge_query_chunk_size=37,
        report_query_chunk_size=31,
        activation_checkpointing=False,
        mixed_precision="fp32",
        device="cpu",
    )

    return config


def inputs(batch=1):
    x = torch.randn(batch, 1, 32, 32, 24)

    condition = {
        "report_tokens": torch.randn(batch, 5, 768),
        "report_mask": torch.tensor([[True, True, True, False, False]]).expand(
            batch, -1
        ),
        "contrast_id": torch.ones(batch, dtype=torch.long),
        "report_drop": torch.zeros(batch, dtype=torch.bool),
    }

    return (
        x,
        torch.full((batch,), 0.5),
        condition,
        torch.eye(4)[None].expand(batch, -1, -1),
    )


def test_rms_layouts_and_fp32_reductions():
    x = torch.randn(2, 8, 3, 4, 5)
    expected = x / torch.sqrt(x.square().mean(1, keepdim=True) + 1e-6)

    torch.testing.assert_close(channel_rms_norm(x, torch.ones(8), 1e-6), expected)

    norm = TokenRMSNorm(8)

    torch.testing.assert_close(norm(x.movedim(1, -1)).movedim(-1, 1), expected)


def test_semantic_dropout_removes_real_report_but_keeps_acquisition():
    conditioner = VoxelConditioner(tiny_config()).eval()
    _, times, condition, affine = inputs()
    condition["report_drop"][:] = True
    first = conditioner(times, condition, affine)
    changed = dict(condition, report_tokens=condition["report_tokens"] + 100)
    second = conditioner(times, changed, affine)

    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)

    assert first[2].sum() == 1

    changed["contrast_id"] = torch.full((1,), 2)

    assert not torch.equal(first[0], conditioner(times, changed, affine)[0])


def test_chunked_report_attention_padding_and_gradients():
    torch.manual_seed(4)

    attention = ReportAttention(8, 24, 2, 3)
    reference = copy.deepcopy(attention)
    reference.chunk_size = 100
    query = torch.randn(2, 11, 8, requires_grad=True)
    context = torch.randn(2, 7, 24, requires_grad=True)
    mask = torch.arange(7)[None] < torch.tensor([3, 5])[:, None]
    q2, c2 = query.detach().requires_grad_(), context.detach().requires_grad_()
    actual = attention(query, context, mask)
    expected = reference(q2, c2, mask)

    torch.testing.assert_close(actual, expected)

    actual.square().sum().backward()

    expected.square().sum().backward()

    torch.testing.assert_close(query.grad, q2.grad)
    torch.testing.assert_close(context.grad, c2.grad)

    modified = context.detach().masked_fill(~mask[..., None], 10000)

    torch.testing.assert_close(attention(query.detach(), modified, mask), actual)


def test_downward_attention_matches_explicit_reference_and_gradients():
    torch.manual_seed(3)

    bridge = GlobalToLocalBridge(tiny_config(), (8, 8, 8), (2, 2, 2))
    q = torch.randn(1, 7, 2, 4, requires_grad=True)
    k = torch.randn(1, 8, 2, 4, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    bias = torch.randn(1, len(bridge.offsets), 2, requires_grad=True)

    indices, valid, ids = (
        bridge.indices[100:107],
        bridge.valid[100:107],
        bridge.offset_ids[100:107],
    )

    actual = bridge.attend_chunk(q, k, v, bias, indices, valid, ids)
    rows = []

    for query in range(7):
        heads = []

        for head in range(2):
            logits = (q[0, query, head] * k[0, indices[query], head]).sum(-1) / 2
            logits = logits + bias[0, ids[query], head]
            weights = logits.masked_fill(~valid[query], -torch.inf).softmax(-1)

            heads.append((weights[:, None] * v[0, indices[query], head]).sum(0))

        rows.append(torch.cat(heads))

    expected = torch.stack(rows)[None]

    torch.testing.assert_close(actual, expected)

    ga = torch.autograd.grad(actual.square().sum(), (q, k, v, bias), retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), (q, k, v, bias))

    for a, b in zip(ga, ge):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_overlapping_upsampling_constant_interior_and_impulse():
    layer = OverlappingUpsample(1, 1)

    with torch.no_grad():
        layer.projection.weight.fill_(1)
        layer.projection.bias.zero_()

    output = layer(torch.ones(1, 1, 4, 4, 4))

    assert output.shape == (1, 1, 8, 8, 8)

    torch.testing.assert_close(output[..., 1:-1, 1:-1, 1:-1], torch.ones(1, 1, 6, 6, 6))

    impulse = torch.zeros(1, 1, 4, 4, 4)
    impulse[..., 1, 1, 1] = 1

    assert (layer(impulse) != 0).sum() == 64


def test_local_readout_phase_order_and_native_input_gradients():
    torch.manual_seed(17)

    readout = LocalReadout(tiny_config())
    local = torch.randn(1, 8, 2, 2, 2)
    padded = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
    base = torch.randn_like(padded, requires_grad=True)
    half = torch.randn(1, 4, 4, 4, 4)
    q = torch.randn(1, 24)

    with torch.no_grad():
        readout.head.bias.copy_(torch.arange(1.0, 9.0))

    initial = readout(local, padded, base, half, q)

    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                phase = (dx * 2 + dy) * 2 + dz

                torch.testing.assert_close(
                    initial[:, :, dx::2, dy::2, dz::2],
                    torch.full((1, 1, 4, 4, 4), phase + 1.0),
                )

    # Activate the final zero projection to check all three native input paths.
    with torch.no_grad():
        readout.native_refine[-1].weight.normal_(std=0.02)

    output = readout(local, padded, base, half, q)

    output.square().mean().backward()

    for gradient in (
        padded.grad,
        base.grad,
        readout.head.weight.grad,
        readout.native_refine[0].weight.grad,
        readout.native_refine[-1].weight.grad,
    ):
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_native_refinement_crosses_voxel_phase_groups():
    readout = LocalReadout(tiny_config())

    with torch.no_grad():
        readout.native_refine[0].weight.zero_()
        readout.native_refine[0].bias.zero_()

        readout.native_refine[0].weight[0, 0, 1, 1, 0] = 1
        readout.native_refine[-1].weight[0, 0, 1, 1, 1] = 1

    padded = torch.zeros(1, 1, 8, 8, 8)
    padded[0, 0, 3, 3, 3] = 1

    output = readout(
        torch.zeros(1, 8, 2, 2, 2),
        padded,
        torch.zeros_like(padded),
        torch.zeros(1, 4, 4, 4, 4),
        torch.zeros(1, 24),
    )

    expected = torch.zeros_like(padded)
    expected[0, 0, 3, 3, 4] = torch.nn.functional.silu(torch.tensor(1.0))

    torch.testing.assert_close(output, expected)


def test_zero_initialization_and_active_information_paths():
    torch.set_num_threads(2)

    model = ParallelVoxelFlow(tiny_config())
    x, times, condition, affine = inputs()
    outputs = model(x, times, condition, affine)

    assert outputs["clean"].shape == x.shape
    assert outputs["clean"].count_nonzero() == 0

    outputs["clean"].sum().backward()

    assert model.local_readout.head.weight.grad.abs().sum() > 0

    model.zero_grad()

    # Enable specified zero gates/heads only for this connectivity test.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "gate" in name or "modulation" in name or ".spatial." in name:
                parameter.add_(torch.randn_like(parameter) * 0.02)

        model.local_readout.head.weight.normal_(std=0.02)
        model.global_readout.upsamples[-1].projection.weight.normal_(std=0.02)

    condition["report_tokens"].requires_grad_()

    output = model(x, times, condition, affine)["clean"]

    output.square().mean().backward()

    assert condition["report_tokens"].grad.abs().sum() > 0

    for module in (
        model.upward[0].message[0],
        model.downward[0].key,
        model.global_readout.upsamples[0].projection,
        model.local_stem,
    ):
        assert module.weight.grad.abs().sum() > 0

    changed = dict(condition, report_tokens=condition["report_tokens"].detach() + 2)

    assert not torch.allclose(output, model(x, times, changed, affine)["clean"])


def test_checkpoint_recomputation_matches_uncheckpointed_backward():
    torch.set_num_threads(2)
    torch.manual_seed(31)

    config = tiny_config()

    config.update(stages=2, blocks_per_stage=2)

    plain = ParallelVoxelFlow(config)

    with torch.no_grad():
        for name, parameter in plain.named_parameters():
            if "gate" in name or "modulation" in name:
                parameter.add_(torch.randn_like(parameter) * 0.02)

        plain.local_readout.head.weight.normal_(std=0.02)
        plain.global_readout.upsamples[-1].projection.weight.normal_(std=0.02)

    config["activation_checkpointing"] = True
    checked = ParallelVoxelFlow(config)

    checked.load_state_dict(plain.state_dict())

    args = inputs()
    a, b = plain(*args)["clean"], checked(*args)["clean"]

    torch.testing.assert_close(a, b)

    (a - 1).square().mean().backward()

    (b - 1).square().mean().backward()

    for p, q in zip(plain.parameters(), checked.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.cuda
@pytest.mark.skipif(
    os.environ.get("VOXEL_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="Native GPU check is opt-in: VOXEL_CUDA_TESTS=1",
)
def test_native_forward_backward():
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/voxel_flow.yaml").read_text()
    )

    model = ParallelVoxelFlow(config).cuda().train()

    condition = {
        "report_tokens": torch.randn(1, 256, 768, device="cuda"),
        "report_mask": torch.ones(1, 256, dtype=torch.bool, device="cuda"),
        "contrast_id": torch.ones(1, dtype=torch.long, device="cuda"),
        "report_drop": torch.zeros(1, dtype=torch.bool, device="cuda"),
    }

    x = torch.randn(1, 1, 240, 240, 168, device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            x,
            torch.tensor([0.5], device="cuda"),
            condition,
            torch.eye(4, device="cuda")[None],
        )

    (output["clean"] - 0.1 * x).square().mean().backward()

    assert torch.isfinite(output["clean"]).all()
    assert output["clean"].dtype == torch.float32
    assert model.local_readout.head.weight.grad.abs().sum() > 0

    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


@pytest.mark.cuda
@pytest.mark.skipif(
    os.environ.get("VOXEL_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="Compiled CUDA comparison is opt-in",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_cnn_matches_outputs_and_gradients(dtype):
    from src.voxel_flow.model import ConditionedCNNBlock

    torch.manual_seed(51)

    eager = ConditionedCNNBlock(8, 24, 8).cuda()

    with torch.no_grad():
        eager.gate.weight.normal_(std=0.02)
        eager.modulation.weight.normal_(std=0.02)
        eager.spatial.weight.normal_(std=0.02)

    compiled = copy.deepcopy(eager)

    compiled.compile(fullgraph=True, dynamic=True)

    assert eager.state_dict().keys() == compiled.state_dict().keys()

    x = torch.randn(2, 8, 8, 8, 8, device="cuda", dtype=dtype, requires_grad=True)
    q = torch.randn(2, 24, device="cuda", requires_grad=True)
    context = torch.randn_like(x, requires_grad=True)
    other = [value.detach().clone().requires_grad_() for value in (x, q, context)]

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        expected = eager(x, q, context)
        actual = compiled(*other)

    tolerance = (
        dict(atol=2e-4, rtol=0.05)
        if dtype == torch.bfloat16
        else dict(atol=5e-6, rtol=1e-4)
    )

    torch.testing.assert_close(actual, expected, **tolerance)

    expected.float().square().mean().backward()

    actual.float().square().mean().backward()

    for first, second in zip((x, q, context), other):
        torch.testing.assert_close(first.grad, second.grad, **tolerance)

    for first, second in zip(eager.parameters(), compiled.parameters()):
        torch.testing.assert_close(first.grad, second.grad, **tolerance)
