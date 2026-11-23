"""Frozen BioMedCLIP report encoder, pinned downloads, and complete-text chunking."""

import os
from pathlib import Path
import re


def download_report_encoder(config):
    """Download the official encoder files at pinned revisions.

    The downloaded source code and weights are not modified.
    """

    os.environ["HF_HUB_DISABLE_XET"] = "1"

    from huggingface_hub import snapshot_download

    jobs = [
        (
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            config["biomedclip_revision"],
            Path(config["biomedclip_directory"]),
            ["open_clip_config.json", "open_clip_pytorch_model.bin"],
        ),
        (
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            config["biomedbert_revision"],
            Path(config["biomedbert_directory"]),
            ["config.json", "tokenizer_config.json", "vocab.txt"],
        ),
    ]

    for repo, revision, folder, files in jobs:
        snapshot_download(
            repo, revision=revision, local_dir=folder, allow_patterns=files
        )

    print("Pinned BioMedCLIP text artifacts downloaded")


def load_biomedclip_text(config):
    """Load the local pretrained text tower."""

    import open_clip

    from transformers import AutoTokenizer

    biomedclip_dir = Path(config["biomedclip_directory"])
    text_config = open_clip.get_model_config(f"local-dir:{biomedclip_dir}")["text_cfg"]
    text_config["hf_model_name"] = config["biomedbert_directory"]
    text_config["hf_tokenizer_name"] = config["biomedbert_directory"]
    text_config["hf_model_pretrained"] = False

    biomedclip_model = open_clip.create_model(
        f"local-dir:{biomedclip_dir}",
        text_cfg=text_config,
        require_pretrained=True,
    )

    text_model = biomedclip_model.text.requires_grad_(False).eval().to(config["device"])

    tokenizer = AutoTokenizer.from_pretrained(
        config["biomedbert_directory"],
        local_files_only=True,
    )

    return text_model, tokenizer


def chunk_report(report, tokenizer, context_length=256):
    """Keep sentences together when possible; split oversized sentences by words.

    Count special tokens with the actual encoder tokenizer; never truncate text.
    """

    if not isinstance(report, str) or not report.strip():
        raise ValueError("Empty report")

    def fits(text):
        return (
            len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
            <= context_length
        )

    chunks = []
    current = ""

    for sentence in re.split(r"(?<=[.!?])\s+|\n+", report.strip()):
        if not sentence.strip():
            continue

        parts = [sentence] if fits(sentence) else sentence.split()

        for part in parts:
            if not fits(part):
                raise ValueError(
                    "Single report word exceeds tokenizer context; hold report"
                )

            joined = (current + " " + part).strip()

            if fits(joined):
                current = joined
            else:
                chunks.append(current)

                current = part

    if current:
        chunks.append(current)

    return chunks
