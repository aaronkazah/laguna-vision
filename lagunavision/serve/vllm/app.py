from __future__ import annotations

import asyncio
from typing import Any

from lagunavision.serve.vllm.payloads import request_has_image
from lagunavision.serve.vllm.service import LagunaVisionVllmService
from lagunavision.serve.vllm.types import LagunaVisionVllmConfig


def create_app(config: LagunaVisionVllmConfig):
    try:
        from fastapi import FastAPI, HTTPException, Request
    except ImportError as exc:
        raise RuntimeError("Install serving dependencies with `pip install -e '.[vllm]'`.") from exc

    app = FastAPI(title="Laguna Vision vLLM Gateway", version="1.0")
    service: LagunaVisionVllmService | None = None
    service_lock = asyncio.Lock()

    async def get_service() -> LagunaVisionVllmService:
        nonlocal service
        if service is None:
            async with service_lock:
                if service is None:
                    service = await asyncio.to_thread(LagunaVisionVllmService.from_config, config)
        return service

    @app.on_event("startup")
    async def startup() -> None:
        await get_service()

    @app.get("/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        svc = await get_service()
        try:
            vllm_models = await svc.client.health()
        except Exception as exc:  # noqa: BLE001 - API boundary surfaces backend readiness.
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "status": "ok",
            "backend": "vllm",
            "model": svc.client.model,
            "checkpoint": svc.checkpoint,
            "vllm_models": vllm_models,
        }

    @app.post("/")
    @app.post("/generate")
    async def generate(request: Request) -> dict[str, Any]:
        svc = await get_service()
        payload = await request.json()
        try:
            return await svc.answer_payload(payload)
        except Exception as exc:  # noqa: BLE001 - return explicit request error.
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> dict[str, Any]:
        svc = await get_service()
        payload = await request.json()
        if payload.get("stream"):
            raise HTTPException(status_code=400, detail="Streaming is not supported by the Laguna Vision gateway.")
        if not request_has_image(payload):
            return await svc.client.chat_completion(payload)
        try:
            raw = await svc.answer_payload(payload, return_raw=True)
            # Wrap completions response into chat format for client compatibility
            choices = raw.get("choices", [])
            chat_choices = []
            for c in choices:
                chat_choices.append({
                    "index": c.get("index", 0),
                    "message": {"role": "assistant", "content": c.get("text", "")},
                    "finish_reason": c.get("finish_reason"),
                })
            return {
                "id": raw.get("id", ""),
                "object": "chat.completion",
                "created": raw.get("created", 0),
                "model": raw.get("model", svc.client.model),
                "choices": chat_choices,
                "usage": raw.get("usage", {}),
            }
        except Exception as exc:  # noqa: BLE001 - return explicit request error.
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
