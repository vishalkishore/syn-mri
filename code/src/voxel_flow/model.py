"""Parallel local-global voxel predictor. Only the native voxel state is integrated."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.voxel_flow.conditioning import VoxelConditioner
from src.voxel_flow.geometry import noise_corrected_gradients
from src.voxel_flow.geometry import global_neighbors_of_local
from src.voxel_flow.geometry import group_into_neighborhoods
from src.voxel_flow.geometry import voxel_center_indices
from src.voxel_flow.geometry import space_to_channels
from src.voxel_flow.geometry import physical_positions
from src.voxel_flow.geometry import rotary_position_3d
from src.voxel_flow.geometry import channels_to_space


def zero_init(module):
    nn.init.zeros_(module.weight)

    if module.bias is not None:
        nn.init.zeros_(module.bias)

    return module


def channel_rms_norm(features, weight, epsilon):
    dtype = features.dtype

    with torch.autocast(device_type=features.device.type, enabled=False):
        work = features.float()
        normalized = work * torch.rsqrt(work.square().mean(1, keepdim=True) + epsilon)

    return normalized.to(dtype) * weight.to(dtype)[None, :, None, None, None]


class TokenRMSNorm(nn.Module):
    def __init__(self, width, epsilon=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            work = x.float()

            normalized = work * torch.rsqrt(
                work.square().mean(-1, keepdim=True) + self.epsilon
            )

        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class ConditionedCNNBlock(nn.Module):
    def __init__(self, width, condition_width, spatial_width=0, epsilon=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

        self.spatial = (
            zero_init(nn.Conv3d(spatial_width, 2 * width, 1)) if spatial_width else None
        )

        self.modulation = zero_init(nn.Linear(condition_width, 2 * width))
        self.gate = zero_init(nn.Linear(condition_width, width))
        self.depthwise = nn.Conv3d(width, width, 3, padding=1, groups=width)
        self.expand = nn.Conv3d(width, 2 * width, 1)
        self.project = nn.Conv3d(2 * width, width, 1)

    def forward(self, features, condition, spatial_context=None):
        modulation = self.modulation(condition)[:, :, None, None, None]

        if self.spatial is not None:
            modulation = modulation + self.spatial(spatial_context)

        gamma, beta = modulation.chunk(2, dim=1)
        normalized = (1 + gamma) * channel_rms_norm(
            features, self.weight, self.epsilon
        ) + beta
        branch = self.project(F.silu(self.expand(self.depthwise(F.silu(normalized)))))

        return features + self.gate(condition)[:, :, None, None, None] * branch


class ReportAttention(nn.Module):
    def __init__(self, width, report_width, heads, chunk_size):
        super().__init__()

        self.heads = heads
        self.width = width
        self.chunk_size = chunk_size
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(report_width, width)
        self.value = nn.Linear(report_width, width)
        self.output = nn.Linear(width, width)

    def forward(self, tokens, report, mask):
        batch = len(tokens)
        head_width = self.width // self.heads

        keys = (
            self.key(report).reshape(batch, -1, self.heads, head_width).transpose(1, 2)
        )

        values = (
            self.value(report)
            .reshape(batch, -1, self.heads, head_width)
            .transpose(1, 2)
        )

        pieces = []

        for part in tokens.split(self.chunk_size, dim=1):
            queries = (
                self.query(part)
                .reshape(batch, -1, self.heads, head_width)
                .transpose(1, 2)
            )

            attended = F.scaled_dot_product_attention(
                queries, keys, values, attn_mask=mask[:, None, None, :], dropout_p=0.0
            )

            pieces.append(attended.transpose(1, 2).reshape(batch, -1, self.width))

        return self.output(torch.cat(pieces, dim=1))


class GlobalTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        width = config["global_width"]
        self.heads = config["global_heads"]
        self.axis_dims = config["global_rope_axis_dims"]
        self.rope_base = config["rope_base"]
        self.rope_scale = config["rope_coordinate_scale_mm"]

        self.norms = nn.ModuleList(
            [TokenRMSNorm(width, config["rms_epsilon"]) for _ in range(3)]
        )

        self.modulation = zero_init(nn.Linear(config["condition_width"], 9 * width))
        self.qkv = nn.Linear(width, 3 * width)
        self.self_output = nn.Linear(width, width)

        self.report = ReportAttention(
            width,
            config["condition_width"],
            self.heads,
            config["report_query_chunk_size"],
        )

        self.ff_gate = nn.Linear(width, config["global_ff_hidden"])
        self.ff_value = nn.Linear(width, config["global_ff_hidden"])
        self.ff_output = nn.Linear(config["global_ff_hidden"], width)

    def forward(self, tokens, condition, report, mask, positions):
        modulation = self.modulation(condition).unsqueeze(1).chunk(9, dim=-1)

        for index in range(3):
            scale, shift, gate = modulation[3 * index : 3 * index + 3]
            normalized = self.norms[index](tokens) * (1 + scale) + shift

            if index == 0:
                batch, count, width = normalized.shape

                query, key, value = (
                    self.qkv(normalized)
                    .reshape(batch, count, 3, self.heads, width // self.heads)
                    .permute(2, 0, 3, 1, 4)
                    .unbind(0)
                )

                query = rotary_position_3d(
                    query, positions, self.axis_dims, self.rope_base, self.rope_scale
                )

                key = rotary_position_3d(
                    key, positions, self.axis_dims, self.rope_base, self.rope_scale
                )

                attended = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0
                )

                branch = self.self_output(
                    attended.transpose(1, 2).reshape(batch, count, width)
                )
            elif index == 1:
                branch = self.report(normalized, report, mask)
            else:
                branch = self.ff_output(
                    F.silu(self.ff_gate(normalized)) * self.ff_value(normalized)
                )

            tokens = tokens + gate * branch

        return tokens


class LocalToGlobalBridge(nn.Module):
    def __init__(self, config):
        super().__init__()

        local, global_width = config["local_width"], config["global_width"]
        self.heads = config["local_heads"]
        self.local = local
        self.local_norm = TokenRMSNorm(local, config["rms_epsilon"])
        self.global_norm = TokenRMSNorm(global_width, config["rms_epsilon"])
        self.query = nn.Linear(global_width, local)
        self.key = nn.Linear(local, local)
        self.value = nn.Linear(local, local)
        self.relative_bias = nn.Parameter(torch.zeros(64, self.heads))

        self.message = nn.Sequential(
            nn.Linear(local + 6, global_width),
            nn.SiLU(),
            nn.Linear(global_width, global_width),
        )

        self.gate = zero_init(nn.Linear(config["condition_width"], global_width))

    def forward(self, local, global_tokens, structure, condition):
        grouped = group_into_neighborhoods(local)
        batch, count, cells, _ = grouped.shape
        width = self.local // self.heads

        query = self.query(self.global_norm(global_tokens)).reshape(
            batch, count, self.heads, width
        )

        keys = self.key(self.local_norm(grouped)).reshape(
            batch, count, cells, self.heads, width
        )

        values = self.value(grouped).reshape(batch, count, cells, self.heads, width)

        scores = torch.einsum("bnhd,bnkhd->bnhk", query, keys).float() / math.sqrt(
            width
        )

        scores = scores + self.relative_bias.T[None, None]

        attended = torch.einsum(
            "bnhk,bnkhd->bnhd", scores.softmax(-1).to(values.dtype), values
        ).flatten(-2)

        message = self.message(
            torch.cat([attended, structure.to(attended.dtype)], dim=-1)
        )

        return global_tokens + self.gate(condition)[:, None] * message


class GlobalToLocalBridge(nn.Module):
    def __init__(self, config, local_shape, global_shape):
        super().__init__()

        local, global_width = config["local_width"], config["global_width"]
        self.heads = config["local_heads"]
        self.width = local
        self.chunk_size = config["bridge_query_chunk_size"]
        self.checkpointing = config["activation_checkpointing"]
        self.norm_local = TokenRMSNorm(local, config["rms_epsilon"])
        self.norm_global = TokenRMSNorm(global_width, config["rms_epsilon"])
        self.query = nn.Linear(local, local)
        self.key = nn.Linear(global_width, local)
        self.value = nn.Linear(global_width, local)
        self.output = nn.Linear(local, local)

        self.bias = nn.Sequential(
            nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, self.heads)
        )

        indices, valid, offsets, offset_ids = global_neighbors_of_local(
            local_shape, global_shape
        )

        for name, value in (
            ("indices", indices),
            ("valid", valid),
            ("offsets", offsets),
            ("offset_ids", offset_ids),
        ):
            self.register_buffer(name, value, persistent=False)

    def attend_chunk(self, queries, keys, values, biases, indices, valid, offset_ids):
        # Gather inside the checkpointed chunk, never for the full local grid.
        gathered_keys = keys[:, indices]
        gathered_values = values[:, indices]

        scores = torch.einsum(
            "bqhd,bqkhd->bqhk", queries, gathered_keys
        ).float() / math.sqrt(queries.shape[-1])

        relative = biases[:, offset_ids].permute(0, 1, 3, 2)
        scores = (scores + relative).masked_fill(~valid[None, :, None], -torch.inf)
        probabilities = scores.softmax(-1).to(gathered_values.dtype)

        return torch.einsum("bqhk,bqkhd->bqhd", probabilities, gathered_values).flatten(
            -2
        )

    def forward(self, local, global_tokens, affine):
        batch = len(local)
        tokens = local.flatten(2).transpose(1, 2)

        queries = self.query(self.norm_local(tokens)).reshape(
            batch, -1, self.heads, self.width // self.heads
        )

        keys = self.key(self.norm_global(global_tokens)).reshape(
            batch, -1, self.heads, self.width // self.heads
        )

        values = self.value(global_tokens).reshape(
            batch, -1, self.heads, self.width // self.heads
        )

        physical_offsets = (
            torch.einsum("bij,nj->bni", affine[:, :3, :3].float(), self.offsets) / 16
        )

        biases = self.bias(physical_offsets)
        pieces = []

        for start in range(0, tokens.shape[1], self.chunk_size):
            stop = start + self.chunk_size

            args = (
                queries[:, start:stop],
                keys,
                values,
                biases,
                self.indices[start:stop],
                self.valid[start:stop],
                self.offset_ids[start:stop],
            )

            if self.checkpointing and self.training and torch.is_grad_enabled():
                pieces.append(checkpoint(self.attend_chunk, *args, use_reentrant=False))
            else:
                pieces.append(self.attend_chunk(*args))

        return self.output(torch.cat(pieces, dim=1)).transpose(1, 2).reshape_as(local)


class OverlappingUpsample(nn.Module):
    def __init__(self, input_width, output_width, zero_projection=False):
        super().__init__()

        self.projection = nn.Conv3d(input_width, output_width, 1)

        if zero_projection:
            zero_init(self.projection)

        self.spatial = nn.ConvTranspose3d(
            output_width,
            output_width,
            4,
            stride=2,
            padding=1,
            groups=output_width,
            bias=False,
        )

        kernel = torch.tensor([0.25, 0.75, 0.75, 0.25])
        kernel = kernel[:, None, None] * kernel[None, :, None] * kernel[None, None, :]

        with torch.no_grad():
            self.spatial.weight.copy_(kernel[None, None].expand_as(self.spatial.weight))

    def forward(self, features):
        return self.spatial(self.projection(features))


class GlobalReadout(nn.Module):
    def __init__(self, config):
        super().__init__()

        widths = [config["global_width"], *config["global_decoder_channels"]]

        self.upsamples = nn.ModuleList(
            [
                OverlappingUpsample(a, b, i == 3)
                for i, (a, b) in enumerate(zip(widths, widths[1:]))
            ]
        )

        self.blocks = nn.ModuleList(
            [
                ConditionedCNNBlock(
                    w, config["condition_width"], epsilon=config["rms_epsilon"]
                )
                for w in widths[1:-1]
            ]
        )

        self.checkpointing = config["activation_checkpointing"]

    def forward(self, features, condition):
        for upsample, block in zip(self.upsamples[:-1], self.blocks):
            features = upsample(features)

            if self.checkpointing and self.training and torch.is_grad_enabled():
                features = checkpoint(block, features, condition, use_reentrant=False)
            else:
                features = block(features, condition)

        half = features

        return self.upsamples[-1](features), half


class LocalReadout(nn.Module):
    """Predict eight voxel phases at half resolution, then refine native voxels."""

    def __init__(self, config):
        super().__init__()

        half, native = config["local_decoder_channels"]
        global_half = config["global_decoder_channels"][-2]
        self.half_up = OverlappingUpsample(config["local_width"], half)
        self.half_skip = nn.Conv3d(8, half, 1)
        self.fusion = nn.Conv3d(2 * half, half, 1)

        self.half_blocks = nn.ModuleList(
            [
                ConditionedCNNBlock(
                    half,
                    config["condition_width"],
                    half + global_half,
                    config["rms_epsilon"],
                )
                for _ in range(2)
            ]
        )

        self.head = zero_init(nn.Conv3d(half, 8, 1))

        self.native_refine = nn.Sequential(
            nn.Conv3d(3, native, 3, padding=1),
            nn.SiLU(),
            zero_init(nn.Conv3d(native, 1, 3, padding=1)),
        )

        self.checkpointing = config["activation_checkpointing"]

    def forward(self, local, padded, global_base, global_half, condition):
        upsampled = self.half_up(local)

        features = self.fusion(
            torch.cat([upsampled, self.half_skip(space_to_channels(padded, 2))], dim=1)
        )

        context = torch.cat([upsampled, global_half], dim=1)

        for block in self.half_blocks:
            if self.checkpointing and self.training and torch.is_grad_enabled():
                features = checkpoint(
                    block, features, condition, context, use_reentrant=False
                )
            else:
                features = block(features, condition, context)

        # Phase channel (dx * 2 + dy) * 2 + dz maps to that voxel in each 2^3 cell.
        initial = channels_to_space(self.head(features), 2)

        correction = self.native_refine(
            torch.cat([padded, global_base, initial], dim=1)
        )

        return initial.float() + correction.float()


class ParallelVoxelFlow(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = dict(config)
        self.native_shape = tuple(config["native_shape"])
        self.z_padding = tuple(config["z_padding"])

        padded_shape = (
            *self.native_shape[:2],
            self.native_shape[2] + sum(self.z_padding),
        )

        if any(s % 16 for s in padded_shape) or self.z_padding != (4, 4):
            raise ValueError("Padded grid must be divisible by 16 with Z padding [4,4]")

        self.local_shape = tuple(s // 4 for s in padded_shape)
        self.global_shape = tuple(s // 16 for s in padded_shape)
        self.checkpointing = config["activation_checkpointing"]
        self.conditioner = VoxelConditioner(config)
        local, global_width = config["local_width"], config["global_width"]
        self.local_stem = nn.Conv3d(64, local, 1)

        self.global_stem = nn.Sequential(
            nn.Conv3d(1, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(128, 128, 4, stride=2, padding=1, groups=128),
            nn.Conv3d(128, global_width, 1),
        )

        self.local_position = nn.Linear(3, local)
        self.global_position = nn.Linear(3, global_width)

        self.register_buffer(
            "local_centers", voxel_center_indices(self.local_shape, 4), persistent=False
        )
        self.register_buffer(
            "global_centers",
            voxel_center_indices(self.global_shape, 16),
            persistent=False,
        )

        self.upward = nn.ModuleList(
            [LocalToGlobalBridge(config) for _ in range(config["stages"])]
        )

        self.downward = nn.ModuleList(
            [
                GlobalToLocalBridge(config, self.local_shape, self.global_shape)
                for _ in range(config["stages"])
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        GlobalTransformerBlock(config)
                        for _ in range(config["blocks_per_stage"])
                    ]
                )
                for _ in range(config["stages"])
            ]
        )

        self.local_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConditionedCNNBlock(
                            local,
                            config["condition_width"],
                            local,
                            config["rms_epsilon"],
                        )
                        for _ in range(config["blocks_per_stage"])
                    ]
                )
                for _ in range(config["stages"])
            ]
        )

        self.local_report_norm = TokenRMSNorm(local, config["rms_epsilon"])

        self.local_report = ReportAttention(
            local,
            config["condition_width"],
            config["local_heads"],
            config["report_query_chunk_size"],
        )

        self.local_report_gate = zero_init(nn.Linear(config["condition_width"], local))
        self.global_readout = GlobalReadout(config)
        self.local_readout = LocalReadout(config)

    def compile_cnn_blocks(self):
        """Fuse CNN pointwise operations without changing checkpoint parameter names."""

        # Width, checkpoint, DDP, and inference variants share the same forward code.
        for block in self.modules():
            if isinstance(block, ConditionedCNNBlock):
                block.compile(fullgraph=True, dynamic=True, recompile_limit=64)

    def run_cnn_block(self, block, *args):
        if self.checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, *args, use_reentrant=False)

        return block(*args)

    def attend_local_report(self, local, condition, report, mask):
        tokens = local.flatten(2).transpose(1, 2)
        message = self.local_report(self.local_report_norm(tokens), report, mask)

        return (
            (tokens + self.local_report_gate(condition)[:, None] * message)
            .transpose(1, 2)
            .reshape_as(local)
        )

    def forward(self, x_t, times, condition, affine, return_diagnostics=False):
        if x_t.shape[1:] != (1, *self.native_shape):
            raise ValueError("Expected one native voxel state [B,1,X,Y,Z]")

        padded = F.pad(x_t.float(), (4, 4, 0, 0, 0, 0))
        condition_vector, report, mask = self.conditioner(times, condition, affine)
        local_positions = physical_positions(self.local_centers, affine)
        global_positions = physical_positions(self.global_centers, affine)
        local = self.local_stem(space_to_channels(padded, 4))

        position = self.local_position(
            local_positions / self.config["absolute_coordinate_scale_mm"]
        )

        local = local + position.transpose(1, 2).reshape_as(local)
        global_map = self.global_stem(F.avg_pool3d(padded, 4, 4))

        tokens = global_map.flatten(2).transpose(1, 2) + self.global_position(
            global_positions / self.config["absolute_coordinate_scale_mm"]
        )

        structure = noise_corrected_gradients(
            x_t, times, affine, self.config["source_noise_std"]
        )

        for upward, downward, global_blocks, local_blocks in zip(
            self.upward, self.downward, self.global_blocks, self.local_blocks
        ):
            tokens = self.run_cnn_block(
                upward, local, tokens, structure, condition_vector
            )

            for block in global_blocks:
                tokens = self.run_cnn_block(
                    block, tokens, condition_vector, report, mask, global_positions
                )

            context = self.run_cnn_block(downward, local, tokens, affine)

            for block in local_blocks:
                local = self.run_cnn_block(block, local, condition_vector, context)

        local = self.run_cnn_block(
            self.attend_local_report, local, condition_vector, report, mask
        )
        global_map = tokens.transpose(1, 2).reshape(len(x_t), -1, *self.global_shape)
        base, half = self.run_cnn_block(
            self.global_readout, global_map, condition_vector
        )
        residual = self.run_cnn_block(
            self.local_readout, local, padded, base, half, condition_vector
        )
        base = base[..., 4:-4].float()
        residual = residual[..., 4:-4].float()

        result = {
            "global_clean": base,
            "local_clean_residual": residual,
            "clean": base + residual,
        }

        if return_diagnostics:
            result["diagnostics"] = {
                "structure_rms": structure.square().mean().sqrt(),
                "global_feature_rms": tokens.float().square().mean().sqrt(),
                "local_feature_rms": local.float().square().mean().sqrt(),
            }

        return result
