from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from lagunavision.serve.vllm.app import create_app
from lagunavision.serve.vllm.merge import merge_lora_adapter
from lagunavision.serve.vllm.smoke import run_smoke_prompt_embeds
from lagunavision.serve.vllm.types import LagunaVisionVllmConfig
from lagunavision.serve.vllm.validate import (
    compare_vllm_gateway_with_hf_endpoint,
    validate_vllm_against_hf,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Laguna Vision through vLLM prompt embeddings")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="start the FastAPI vLLM gateway")
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--vllm-base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    serve.add_argument("--model", default=os.environ.get("VLLM_MODEL", ""))
    serve.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    serve.add_argument("--vision-device", default=os.environ.get("LAGUNA_VISION_DEVICE", "auto"))
    serve.add_argument("--embedding-dtype", default=os.environ.get("LAGUNA_VLLM_EMBEDDING_DTYPE", "float32"))
    serve.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("LAGUNA_MAX_NEW_TOKENS", "128")))
    serve.add_argument("--temperature", type=float, default=float(os.environ.get("LAGUNA_TEMPERATURE", "0")))
    serve.add_argument("--top-p", type=float, default=None)
    serve.add_argument("--timeout", type=float, default=float(os.environ.get("LAGUNA_VLLM_TIMEOUT", "120")))
    serve.add_argument("--host", default=os.environ.get("LAGUNA_VLLM_HOST", "0.0.0.0"))
    serve.add_argument("--port", type=int, default=int(os.environ.get("LAGUNA_VLLM_PORT", "8080")))
    serve.add_argument("--allow-local-files", action="store_true")

    validate = subparsers.add_parser("validate", help="compare vLLM output with the HF pipeline")
    validate.add_argument("--checkpoint", required=True)
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--output", required=True, type=Path)
    validate.add_argument("--vllm-base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    validate.add_argument("--model", default=os.environ.get("VLLM_MODEL", ""))
    validate.add_argument("--limit", type=int, default=None)
    validate.add_argument("--device", default=os.environ.get("LAGUNA_DEVICE", "auto"))
    validate.add_argument("--vision-device", default=os.environ.get("LAGUNA_VISION_DEVICE", "auto"))
    validate.add_argument("--max-new-tokens", type=int, default=32)

    compare = subparsers.add_parser(
        "compare-endpoints",
        help="compare a live HF endpoint with a live vLLM gateway",
    )
    compare.add_argument("--hf-endpoint", required=True)
    compare.add_argument("--vllm-gateway-url", required=True)
    compare.add_argument("--manifest", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument("--hf-token", default=os.environ.get("HF_ENDPOINT_TOKEN", ""))
    compare.add_argument("--vllm-token", default=os.environ.get("LAGUNA_VLLM_GATEWAY_TOKEN", ""))
    compare.add_argument("--limit", type=int, default=None)
    compare.add_argument("--max-new-tokens", type=int, default=32)
    compare.add_argument("--timeout", type=float, default=float(os.environ.get("LAGUNA_VLLM_TIMEOUT", "120")))

    smoke = subparsers.add_parser(
        "smoke-prompt-embeds",
        help="verify a running vLLM server accepts /v1/completions prompt_embeds",
    )
    smoke.add_argument("--vllm-base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    smoke.add_argument("--model", required=True)
    smoke.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    smoke.add_argument("--hf-model", default="")
    smoke.add_argument("--embedding-dim", type=int, default=0)
    smoke.add_argument("--token-count", type=int, default=4)
    smoke.add_argument("--max-tokens", type=int, default=8)
    smoke.add_argument("--timeout", type=float, default=float(os.environ.get("LAGUNA_VLLM_TIMEOUT", "120")))

    merge = subparsers.add_parser(
        "merge-lora",
        help="merge a Stage 2 LoRA adapter into a vLLM-loadable text model",
    )
    merge.add_argument("--base-model", default=os.environ.get("LAGUNA_MODEL_ID", "poolside/Laguna-XS.2"))
    merge.add_argument("--lora-dir", required=True, type=Path)
    merge.add_argument("--output-dir", required=True, type=Path)
    merge.add_argument("--dtype", default=os.environ.get("LAGUNA_MERGE_DTYPE", "auto"))
    merge.add_argument("--device-map", default=os.environ.get("LAGUNA_MERGE_DEVICE_MAP", "auto"))
    merge.add_argument("--trust-remote-code", action="store_true", default=True)

    args = parser.parse_args(argv)
    if args.command is None:
        args.command = "serve"
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "validate":
        summary = asyncio.run(
            validate_vllm_against_hf(
                checkpoint=args.checkpoint,
                manifest=args.manifest,
                output=args.output,
                vllm_base_url=args.vllm_base_url,
                model=args.model,
                limit=args.limit,
                device=args.device,
                vision_device=args.vision_device,
                max_new_tokens=args.max_new_tokens,
            )
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "compare-endpoints":
        summary = compare_vllm_gateway_with_hf_endpoint(
            hf_endpoint=args.hf_endpoint,
            vllm_gateway_url=args.vllm_gateway_url,
            manifest=args.manifest,
            output=args.output,
            hf_token=args.hf_token,
            vllm_token=args.vllm_token,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            timeout=args.timeout,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "smoke-prompt-embeds":
        result = run_smoke_prompt_embeds(
            base_url=args.vllm_base_url,
            model=args.model,
            api_key=args.api_key,
            hf_model=args.hf_model or None,
            embedding_dim=args.embedding_dim or None,
            token_count=args.token_count,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "merge-lora":
        result = merge_lora_adapter(
            base_model=args.base_model,
            lora_dir=args.lora_dir,
            output_dir=args.output_dir,
            dtype=args.dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps(result, indent=2))
        return

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install serving dependencies with `pip install -e '.[vllm]'`.") from exc

    config = LagunaVisionVllmConfig(
        checkpoint=args.checkpoint,
        vllm_base_url=args.vllm_base_url,
        model=args.model,
        api_key=args.api_key,
        vision_device=args.vision_device,
        embedding_dtype=args.embedding_dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        allow_local_files=args.allow_local_files,
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
