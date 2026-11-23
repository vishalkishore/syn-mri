"""Frozen contextual BioMedCLIP tokens with the original text-tower pooling."""

import torch

from src.report_encoder import chunk_report


@torch.inference_mode()
def extract_report_tokens(text_model, tokenizer, report, device, batch_size=8):
    """Encode each sentence-preserving chunk within the 256-token BERT context.

    Return padded contextual tokens and their tokenizer mask, plus the mean
    projected CLS embedding used by the existing report cache. Padding removal
    for storage must use this mask; valid CLS/SEP tokens are retained.
    """

    text_model.requires_grad_(False).eval()

    chunks = chunk_report(report, tokenizer)

    inputs = tokenizer(
        chunks,
        padding="max_length",
        max_length=256,
        truncation=False,
        return_tensors="pt",
    )

    hidden = []
    pooled = []

    for start in range(0, len(chunks), batch_size):
        ids = inputs["input_ids"][start : start + batch_size].to(device)
        mask = inputs["attention_mask"][start : start + batch_size].to(device)

        output = text_model.transformer(
            input_ids=ids,
            attention_mask=mask,
            return_dict=True,
        )

        hidden.append(output.last_hidden_state.float().cpu())

        pooled.append(text_model.proj(text_model.pooler(output, mask)).float().cpu())

    return {
        "tokens": torch.cat(hidden).reshape(-1, 768),
        "attention_mask": inputs["attention_mask"].bool().flatten(),
        "pooled": torch.cat(pooled).mean(0),
    }
