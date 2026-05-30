from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image

from lagunavision.visual_pipeline import LagunaVisionImagePipeline, VisualProjectorSpec


class EndpointHandler:
    """Hugging Face Inference Endpoint handler for Laguna Vision checkpoints."""

    def __init__(self, path: str = "") -> None:
        self.model_dir = Path(path or ".").resolve()
        self.checkpoint_dir = self._resolve_checkpoint_dir()
        self.pipeline: LagunaVisionImagePipeline | None = None

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        payload = data.get("inputs", data)
        if not isinstance(payload, dict):
            raise ValueError("inputs must be an object with image and question fields")

        payload = _normalize_payload(payload)
        question = str(payload.get("question") or "").strip()
        if not question:
            raise ValueError("question is required")

        image_value = payload.get("image")
        if image_value is None:
            raise ValueError(
                "image is required as base64, a data URI, an HTTPS URL, a local path, "
                "or OpenAI-style messages content with image_url"
            )

        context = str(payload.get("context") or "")
        max_new_tokens = int(payload.get("max_new_tokens") or os.environ.get("LAGUNA_MAX_NEW_TOKENS", "128"))

        image = _load_image(image_value)
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image.save(tmp.name)
            answer = asyncio.run(self._answer(Path(tmp.name), question, context, max_new_tokens))
        return {"answer": answer, "checkpoint": str(self.checkpoint_dir.relative_to(self.model_dir))}

    async def _answer(self, image: Path, question: str, context: str, max_new_tokens: int) -> str:
        if self.pipeline is None:
            self.pipeline = await self._load_pipeline()
        return await self.pipeline.answer_image(
            image=image,
            question=question,
            context=context,
            max_new_tokens=max_new_tokens,
        )

    async def _load_pipeline(self) -> LagunaVisionImagePipeline:
        spec_path = self.checkpoint_dir / "projector_spec.json"
        if not spec_path.exists():
            raise FileNotFoundError(f"missing projector_spec.json in {self.checkpoint_dir}")
        spec_row = json.loads(spec_path.read_text(encoding="utf-8"))
        spec = VisualProjectorSpec(
            input_dim=int(spec_row["input_dim"]),
            embedding_dim=int(spec_row["embedding_dim"]),
            hidden_dim=int(spec_row["hidden_dim"]),
            projector=spec_row.get("projector", "mlp"),
            visual_tokens=int(spec_row.get("visual_tokens", 64)),
            encoder=spec_row.get("encoder", "hf"),
            encoder_id=spec_row.get("encoder_id", ""),
            max_tiles=int(spec_row.get("max_tiles", 4)),
            patch_px=int(spec_row.get("patch_px", 32)),
        )
        lora_dir = self.checkpoint_dir / "lora"
        return await LagunaVisionImagePipeline.from_checkpoint(
            checkpoint=self.checkpoint_dir / "projector.pt",
            spec=spec,
            backbone_name=os.environ.get("LAGUNA_BACKBONE") or spec_row.get("backbone", "laguna"),
            model_id=os.environ.get("LAGUNA_MODEL_ID") or spec_row["model_id"],
            device=os.environ.get("LAGUNA_DEVICE", "auto"),
            vision_device=os.environ.get("LAGUNA_VISION_DEVICE", "auto"),
            lora_dir=lora_dir if lora_dir.exists() else None,
        )

    def _resolve_checkpoint_dir(self) -> Path:
        requested = os.environ.get("LAGUNA_CHECKPOINT_PATH", "latest").strip("/")
        candidates = [self.model_dir / requested, self.model_dir]
        for candidate in candidates:
            if (candidate / "projector.pt").exists() and (candidate / "projector_spec.json").exists():
                return candidate
        matches = sorted(self.model_dir.glob("**/projector.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
        if matches:
            return matches[0].parent
        raise FileNotFoundError(
            f"no Laguna Vision checkpoint found under {self.model_dir}; expected latest/projector.pt"
        )


def _load_image(value: Any) -> Image.Image:
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if not isinstance(value, str):
        raise ValueError("image must be bytes or a string")

    if value.startswith("data:image/"):
        _, encoded = value.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        response = requests.get(value, timeout=20)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    path = Path(value)
    if path.exists():
        return Image.open(path).convert("RGB")

    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("messages") is None:
        return payload

    question_parts: list[str] = []
    image_value: Any = payload.get("image")
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            question_parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text"} and item.get("text"):
                question_parts.append(str(item["text"]))
            elif item_type in {"image_url", "input_image"}:
                image_url = item.get("image_url")
                if isinstance(image_url, dict):
                    image_value = image_url.get("url") or image_url.get("image_url") or image_value
                else:
                    image_value = image_url or item.get("url") or image_value

    normalized = dict(payload)
    normalized.setdefault("question", "\n".join(part.strip() for part in question_parts if part.strip()))
    if image_value is not None:
        normalized["image"] = image_value
    return normalized
