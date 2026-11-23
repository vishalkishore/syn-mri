"""Report semantics, Fourier time, and acquisition conditioning."""

import math

import torch
from torch import nn


class VoxelConditioner(nn.Module):
    def __init__(self, config):
        super().__init__()

        width = config["condition_width"]
        time_width = config["time_embedding_width"]
        self.projection = nn.Linear(768, width)
        self.null_report = nn.Parameter(torch.randn(width) * 0.02)
        self.contrast = nn.Embedding(3, width)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_width, config["time_mlp_hidden"]),
            nn.SiLU(),
            nn.Linear(config["time_mlp_hidden"], width),
        )

        self.spacing_mlp = nn.Sequential(
            nn.Linear(3, config["spacing_mlp_hidden"]),
            nn.SiLU(),
            nn.Linear(config["spacing_mlp_hidden"], width),
        )

        self.register_buffer(
            "frequencies",
            torch.exp(
                -math.log(config["time_frequency_base"])
                * torch.arange(time_width // 2)
                / (time_width // 2)
            ),
        )

        self.time_scale = config["time_scale"]
        self.drop_probability = config["report_semantic_dropout"]

    def forward(self, times, condition, affine):
        report = self.projection(condition["report_tokens"])
        mask = condition["report_mask"]
        # Training supplies this draw before checkpointed blocks; the public model
        # also permits an explicit drop flag for deterministic tests and guidance.
        drop = condition.get("report_drop")

        if drop is None:
            drop = (
                torch.rand(len(times), device=times.device) < self.drop_probability
                if self.training
                else torch.zeros_like(times, dtype=torch.bool)
            )

        report = torch.where(
            drop[:, None, None], self.null_report.to(report.dtype), report
        )

        null_mask = torch.zeros_like(mask)
        null_mask[:, 0] = True
        mask = torch.where(drop[:, None], null_mask, mask)

        with torch.autocast(device_type=report.device.type, enabled=False):
            report_mean = report.float().masked_fill(~mask[..., None], 0).sum(
                1
            ) / mask.sum(1, keepdim=True)

            spacing = torch.linalg.vector_norm(affine[:, :3, :3].float(), dim=1)
            angles = self.time_scale * times.float()[:, None] * self.frequencies
            time_features = torch.cat([angles.cos(), angles.sin()], dim=-1)

        q = (
            self.time_mlp(time_features)
            + report_mean.to(report.dtype)
            + self.contrast(condition["contrast_id"])
            + self.spacing_mlp(spacing.log())
        )

        return q, report, mask
