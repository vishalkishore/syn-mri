from pathlib import Path
import json

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from src.voxel_flow.training import train
from test_voxel_flow_model import tiny_config
from test_voxel_flow_training import TinyPredictor
from test_voxel_flow_training import assert_state_equal
from test_voxel_flow_training import synthetic_dataset


class ConstantPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, times, condition, affine):
        clean = self.weight.expand_as(x)

        return {"clean": clean, "global_clean": clean}


def distributed_worker(rank, directory, masked=False):
    root = Path(directory)

    torch.set_num_threads(1)

    dist.init_process_group(
        "gloo", init_method=f"file://{root / 'rendezvous'}", rank=rank, world_size=2
    )

    try:
        config = tiny_config()

        config.update(
            world_size=2,
            batch_size=1,
            gradient_accumulation_steps=2,
            num_workers=0,
            pin_memory=False,
            epochs=2,
            max_updates=3,
            checkpoint_every_updates=1,
            checkpoint_epochs=[],
            log_gradient_diagnostics=True,
            run_dir=str(root / "continuous"),
        )

        dataset = synthetic_dataset()

        if masked:
            config.update(voxel_loss_reduction="mask_balanced", voxel_brain_weight=0.5)

            for index, item in enumerate(dataset):
                item["brain_mask"] = torch.zeros_like(item["volume"], dtype=torch.bool)
                item["brain_mask"][:, : index + 1] = True

        model, _, _ = train(config, dataset, dataset[:2], model=TinyPredictor())
        weights = [torch.zeros_like(model.weight) for _ in range(2)]

        dist.all_gather(weights, model.weight.detach())

        torch.testing.assert_close(weights[0], weights[1], atol=0, rtol=0)

        split = dict(config, max_updates=1, run_dir=str(root / "resumed"))

        train(split, dataset, dataset[:2], model=TinyPredictor())

        split["max_updates"] = 3

        train(
            split,
            dataset,
            dataset[:2],
            model=TinyPredictor(),
            resume=root / "resumed/last.pt",
        )

        dist.barrier()

        if rank == 0:
            first = torch.load(root / "continuous/last.pt", weights_only=True)
            second = torch.load(root / "resumed/last.pt", weights_only=True)

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

            for first_rng, second_rng in zip(first["rank_rng"], second["rank_rng"]):
                for key in ("shuffle", "noise_time", "report_drop"):
                    assert_state_equal(
                        first_rng["generators"][key], second_rng["generators"][key]
                    )

        weighted = dict(
            config,
            epochs=1,
            max_updates=None,
            gradient_accumulation_steps=4,
            training_denominator_floor=1.0,
            contrast_loss_weight=0.0,
            global_auxiliary_weight=0.0,
            run_dir=str(root / "weighted"),
        )

        train(weighted, dataset, dataset[:2], model=ConstantPredictor())

        if rank == 0:
            record = json.loads(
                (root / "weighted/train.jsonl").read_text().splitlines()[0]
            )

            assert record["window_examples"] == 7
            # d mean((w-y)^2)/dw = 2*(0.1 - mean(0, 0.1, ..., 0.6)).
            assert record["gradient_norm"] == pytest.approx(0.4, rel=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("masked", [False, True])
def test_ddp_uneven_tail_accumulation_and_exact_resume(tmp_path, masked):
    mp.spawn(distributed_worker, args=(str(tmp_path), masked), nprocs=2, join=True)


def topology_resume_worker(rank, directory):
    root = Path(directory)

    torch.set_num_threads(1)

    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'topology_rendezvous'}",
        rank=rank,
        world_size=2,
    )

    try:
        checkpoint = root / "continuous/last.pt"
        state = torch.load(checkpoint, weights_only=True)

        config = dict(
            state["config"],
            world_size=2,
            batch_size=1,
            gradient_accumulation_steps=2,
            epochs=3,
            max_updates=5,
        )

        dataset = synthetic_dataset()

        model, _, _ = train(
            config, dataset, dataset[:2], model=TinyPredictor(), resume=checkpoint
        )

        weights = [torch.zeros_like(model.weight) for _ in range(2)]

        dist.all_gather(weights, model.weight.detach())

        torch.testing.assert_close(weights[0], weights[1], atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_resume_ddp_to_single_and_back(tmp_path):
    mp.spawn(distributed_worker, args=(str(tmp_path),), nprocs=2, join=True)

    checkpoint = tmp_path / "continuous/last.pt"
    before = torch.load(checkpoint, weights_only=True)

    config = dict(
        before["config"],
        world_size=1,
        batch_size=1,
        gradient_accumulation_steps=4,
        max_updates=4,
    )

    dataset = synthetic_dataset()

    train(config, dataset, dataset[:2], model=TinyPredictor(), resume=checkpoint)

    single = torch.load(checkpoint, weights_only=True)

    assert single["world_size"] == 1
    assert single["optimizer_updates"] == 4
    assert single["ema_updates"] == 4
    assert single["epoch"] == 2
    assert len(single["rank_rng"]) == 1

    mp.spawn(topology_resume_worker, args=(str(tmp_path),), nprocs=2, join=True)

    restored = torch.load(checkpoint, weights_only=True)

    assert restored["world_size"] == 2
    assert restored["optimizer_updates"] == 5
    assert restored["ema_updates"] == 5
    assert restored["next_sample"] == 4
    assert len(restored["rank_rng"]) == 2
