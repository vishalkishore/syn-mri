"""Shared-memory frozen report features."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class ReportFeatureCache:
    """Load one shared copy for training, validation, and spawned loader workers."""

    def __init__(self, root, memory_map=False):
        root = Path(root)
        metadata = json.loads((root / "metadata.json").read_text())

        if metadata["hidden_size"] != 768 or metadata["pooled_size"] != 512:
            raise ValueError("Unexpected BioMedCLIP feature dimensions")

        if memory_map:
            # A copy-on-write memory map reads only the requested reports.
            # Changes in memory never reach the file on disk.
            self.tokens = torch.from_numpy(np.load(root / "tokens.npy", mmap_mode="c"))
        else:
            print("Loading frozen report tokens into shared CPU RAM", flush=True)

            self.tokens = torch.from_numpy(np.load(root / "tokens.npy")).share_memory_()

        self.pooled = torch.from_numpy(np.load(root / "pooled.npy")).share_memory_()
        self.offsets = torch.from_numpy(np.load(root / "offsets.npy")).share_memory_()
        studies = pd.read_csv(root / "studies.csv", dtype=str).study_uid.tolist()

        if len(studies) != len(set(studies)) or len(studies) != len(self.pooled):
            raise ValueError("Invalid report-to-feature mapping")

        if self.offsets[-1].item() != len(self.tokens):
            raise ValueError("Incomplete report token cache")

        self.index = {study: index for index, study in enumerate(studies)}

        print(
            f"Loaded {len(studies)} reports and {len(self.tokens)} valid tokens",
            flush=True,
        )
