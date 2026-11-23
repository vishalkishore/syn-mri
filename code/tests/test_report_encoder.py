"""Report chunking must preserve all words and respect the tokenizer context."""

import pytest

from src.report_encoder import chunk_report


class WordTokenizer:
    def encode(self, text, **kwargs):
        return [0] + list(range(len(text.split()))) + [1]


def test_chunking_no_loss_special_tokens_and_long_sentence():
    report = (
        "First sentence here. "
        + " ".join(f"word{i}" for i in range(40))
        + "!\nLast sentence."
    )

    chunks = chunk_report(report, WordTokenizer(), context_length=10)

    assert " ".join(chunks).split() == report.split()
    assert all(len(WordTokenizer().encode(c)) <= 10 for c in chunks)
    assert chunks[0].startswith("First sentence here.")

    with pytest.raises(ValueError):
        chunk_report("", WordTokenizer())


def test_download_only_requests_pinned_report_encoder_artifacts(tmp_path, monkeypatch):
    import huggingface_hub

    from src.report_encoder import download_report_encoder

    calls = []

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    config = {
        "biomedclip_directory": str(tmp_path / "clip"),
        "biomedclip_revision": "a" * 40,
        "biomedbert_directory": str(tmp_path / "bert"),
        "biomedbert_revision": "b" * 40,
    }

    download_report_encoder(config)

    assert [args[0] for args, _ in calls] == [
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    ]

    assert [kwargs["revision"] for _, kwargs in calls] == ["a" * 40, "b" * 40]
