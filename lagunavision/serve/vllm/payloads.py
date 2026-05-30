from __future__ import annotations

import base64
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lagunavision.serve.vllm.types import EncodedPromptEmbeds, GenerationInputs


@dataclass(frozen=True)
class LoadedImage:
    path: Path
    cleanup: bool


def _is_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _decode_data_uri(value: str) -> bytes:
    if "," not in value:
        raise ValueError("Malformed data URI image.")
    header, encoded = value.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Only base64 data URI images are supported.")
    return base64.b64decode(encoded, validate=True)


def load_image(image_value: Any, *, allow_local_files: bool = False) -> LoadedImage:
    if isinstance(image_value, bytes):
        payload = image_value
    elif isinstance(image_value, str):
        value = image_value.strip()
        if value.startswith("data:"):
            payload = _decode_data_uri(value)
        elif _is_http_url(value):
            with urllib.request.urlopen(value, timeout=30) as response:
                payload = response.read()
        else:
            candidate = Path(value).expanduser()
            try:
                exists = candidate.exists()
            except OSError:
                exists = False
            if exists:
                if not allow_local_files:
                    raise ValueError(
                        "Local image paths are disabled for this server. Send base64/data URI images "
                        "or start with --allow-local-files for trusted internal use."
                    )
                return LoadedImage(path=candidate, cleanup=False)
            payload = base64.b64decode(value, validate=True)
    else:
        raise ValueError("image must be a URL, data URI, base64 string, local path, or bytes.")

    tmp = tempfile.NamedTemporaryFile(prefix="laguna-vllm-image-", suffix=".image", delete=False)
    try:
        tmp.write(payload)
        return LoadedImage(path=Path(tmp.name), cleanup=True)
    finally:
        tmp.close()


def _extract_image_url(part: dict[str, Any]) -> str | None:
    url = part.get("url")
    if isinstance(url, str):
        return url
    image_url = part.get("image_url")
    if isinstance(image_url, str):
        return image_url
    if isinstance(image_url, dict):
        url = image_url.get("url")
        if isinstance(url, str):
            return url
    return None


def normalize_generation_inputs(payload: dict[str, Any]) -> GenerationInputs:
    raw_inputs = payload.get("inputs", payload)
    if not isinstance(raw_inputs, dict):
        raise ValueError("inputs must be an object containing image and question.")

    image = raw_inputs.get("image")
    question = raw_inputs.get("question") or raw_inputs.get("prompt")
    context = raw_inputs.get("context") or raw_inputs.get("ocr_text") or raw_inputs.get("text") or ""

    messages = raw_inputs.get("messages")
    if isinstance(messages, list):
        text_parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                text_parts.append(content)
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"text", "input_text"} and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif part.get("type") in {"image_url", "input_image"}:
                    image = image or _extract_image_url(part)
        question = question or "\n".join(piece for piece in text_parts if piece).strip()

    if not image:
        raise ValueError("Missing image; provide inputs.image or an image_url content part.")
    if not question:
        raise ValueError("Missing question; provide inputs.question or text message content.")

    max_new_tokens = raw_inputs.get("max_new_tokens", raw_inputs.get("max_tokens", payload.get("max_tokens")))
    temperature = raw_inputs.get("temperature", payload.get("temperature"))
    top_p = raw_inputs.get("top_p", payload.get("top_p"))
    stop = raw_inputs.get("stop", payload.get("stop"))
    request_id = raw_inputs.get("request_id", payload.get("request_id"))

    return GenerationInputs(
        image=image,
        question=str(question),
        context=str(context),
        max_new_tokens=int(max_new_tokens) if max_new_tokens is not None else None,
        temperature=float(temperature) if temperature is not None else None,
        top_p=float(top_p) if top_p is not None else None,
        stop=stop,
        request_id=str(request_id) if request_id is not None else None,
    )


def request_has_image(payload: dict[str, Any]) -> bool:
    try:
        normalize_generation_inputs(payload)
    except ValueError:
        return False
    return True


def build_vllm_completions_payload(
    *,
    model: str,
    prompt_embeds: EncodedPromptEmbeds,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float | None = None,
    stop: Any = None,
) -> dict[str, Any]:
    """Build a /v1/completions payload with prompt_embeds as a top-level field."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt_embeds": prompt_embeds.data,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if stop is not None:
        payload["stop"] = stop
    return payload

def extract_openai_answer(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    text = first.get("text")
    return text.strip() if isinstance(text, str) else ""
