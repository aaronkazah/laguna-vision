from __future__ import annotations

from pathlib import Path
from typing import Any


def _dtype_kwargs(dtype: str) -> dict[str, Any]:
    if not dtype or dtype == "auto":
        return {}

    import torch

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return {"torch_dtype": mapping[dtype.lower()]}
    except KeyError as exc:
        raise ValueError(f"Unsupported merge dtype {dtype!r}.") from exc


def merge_lora_adapter(
    *,
    base_model: str,
    lora_dir: Path,
    output_dir: Path,
    dtype: str = "auto",
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> dict[str, str]:
    if not lora_dir.exists():
        raise FileNotFoundError(f"Missing LoRA adapter directory: {lora_dir}")

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install merge dependencies with `pip install -e '.[llama]'`.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        **_dtype_kwargs(dtype),
    )
    model = PeftModel.from_pretrained(model, str(lora_dir))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output_dir), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(str(output_dir))
    return {
        "base_model": base_model,
        "lora_dir": str(lora_dir),
        "output_dir": str(output_dir),
        "status": "merged",
    }
