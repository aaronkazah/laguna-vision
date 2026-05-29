from __future__ import annotations


def resolve_torch_device(device: str = "auto") -> str:
    import torch

    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
