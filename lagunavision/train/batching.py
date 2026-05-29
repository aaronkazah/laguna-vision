from __future__ import annotations

import math
from typing import Any


def pad_visual_batch(sequences: list[Any], label_sequences: list[Any]) -> dict[str, Any]:
    """Right-pad variable-length ``[Li, H]`` embedding rows into one batch.

    ``sequences[i]`` is the concatenated visual+text embeddings for sample ``i``;
    ``label_sequences[i]`` is its token-id targets (``-100`` where loss is masked).
    Padding rows get ``attention_mask=0`` and ``labels=-100`` so they never
    contribute to the causal-LM loss.
    """
    import torch

    if not sequences:
        raise ValueError("cannot pad an empty batch")
    if len(sequences) != len(label_sequences):
        raise ValueError("sequences and label_sequences must align")

    batch = len(sequences)
    max_len = max(int(seq.shape[0]) for seq in sequences)
    hidden = int(sequences[0].shape[1])
    reference = sequences[0]

    inputs_embeds = torch.zeros(batch, max_len, hidden, dtype=reference.dtype, device=reference.device)
    attention_mask = torch.zeros(batch, max_len, dtype=torch.long, device=reference.device)
    labels = torch.full((batch, max_len), -100, dtype=torch.long, device=reference.device)
    for index, (seq, label) in enumerate(zip(sequences, label_sequences)):
        length = int(seq.shape[0])
        inputs_embeds[index, :length] = seq
        attention_mask[index, :length] = 1
        labels[index, :length] = label
    return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask, "labels": labels}


def cosine_warmup_multiplier(
    step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.1
) -> float:
    """LambdaLR multiplier: linear warmup then cosine decay to ``min_ratio``."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
