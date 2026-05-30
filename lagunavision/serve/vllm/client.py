from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any


class VllmOpenAIClient:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        if not model:
            raise ValueError("model is required; use the vLLM served model name or LoRA module name.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout

    def _url(self, route: str) -> str:
        route = route.lstrip("/")
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/{route}"
        return f"{self.base_url}/v1/{route}"

    def _request_json(self, method: str, route: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self._url(route), data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM {method} {route} failed with HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach vLLM at {self.base_url}: {exc.reason}") from exc
        return json.loads(response_body) if response_body else {}

    async def health(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._request_json, "GET", "models")

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._request_json, "POST", "chat/completions", payload)

    async def completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._request_json, "POST", "completions", payload)
