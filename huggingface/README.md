---
license: other
base_model: poolside/Laguna-XS.2
library_name: transformers
tags:
- vision-language
- laguna
- siglip
- llava
- lora
pipeline_tag: image-to-text
private: true
---

# Laguna Vision

Laguna Vision adds a visual input path to `poolside/Laguna-XS.2`. SigLIP encodes images, AnyRes tiling preserves screenshot/document detail, a resampler projector maps features into Laguna's embedding space, and LoRA adapters are trained with supervised visual-instruction data.

Method: **post-training multimodal adaptation via supervised fine-tuning**.

Current status: `latest` is an early 200k-example checkpoint. It serves successfully but is weakly grounded: **12 / 80** strict passes on the live capability matrix.

![Laguna Vision demo](assets/demo/lagunavision.gif)

## Vision pathway breakdown

Laguna can generate text, but it has no native pixel input. This checkpoint adds the missing bridge from screen/image pixels into Laguna tokens.

| Step | Implementation |
|---|---|
| Visual sensing | SigLIP vision encoder with AnyRes global/crop tiling |
| Token bridge | Resampler projector producing 256 visual tokens |
| Post-training | Stage 1 projector alignment, then Stage 2 projector + LoRA supervised tuning |
| Grounding audit | 80 deterministic live probes with raw payloads and extracted final answers |

## Checkpoint

| Field | Value |
|---|---|
| Path | `laguna-general-vision-200k-20260529-r2/stage2/step_000900` |
| Base model | `poolside/Laguna-XS.2` |
| Vision encoder | `google/siglip-so400m-patch14-384` |
| Visual path | AnyRes global view + up to 4 high-detail tiles |
| Visual tokens | 256 |
| Projector | resampler |
| Trainable weights | Stage 1: projector only; Stage 2: projector + LoRA |
| LoRA | rank 64, alpha 128, dropout 0.05 |
| Released run | 200k examples: 80k alignment + 120k instruction |
| Full recipe | 300k examples: 120k alignment + 180k instruction |

## Capability matrix

| Category | Result | Measures |
|---|---:|---|
| `basic_shape` | 2 / 10 | Single-object shape recognition. |
| `basic_color` | 3 / 10 | Single-object color recognition. |
| `color_shape_binding` | 1 / 10 | Binding color to shape. |
| `no_text_control` | 3 / 10 | Abstaining when no text is visible. |
| `tiny_ocr` | 0 / 10 | Exact small terminal text. |
| `dense_ui_localization` | 0 / 10 | Dense UI row/status localization. |
| `meme_semantics` | 3 / 10 | Simple visual relationship attribution. |
| `table_precision` | 0 / 10 | Precise document/table extraction. |

The answer audit should live at `evals/live_capability_eval_80/capability_probe.answers.rescored.jsonl`.

## What to keep in this model repo

| Path | Purpose |
|---|---|
| `README.md` | model card |
| `latest/` | stable adapter target |
| `<run_name>/stage1/step_*` and `<run_name>/stage2/step_*` | checkpoint lineage |
| `<run_name>/run_metadata/{recipe.json,run_state.json,job.log}` | run audit trail |
| `handler.py` and `requirements.txt` | endpoint runtime |
| `evals/live_capability_eval_80/` | probe images, manifest, summary, and raw answers |

Do not upload raw image archives, feature caches, access tokens, or full gated Laguna base weights.

## Endpoint

Use the default Hugging Face Dedicated Inference Endpoint Python runtime with this repo's `handler.py`.

| Setting | Value |
|---|---|
| Accelerator | A100 80GB for first deployment |
| Environment | `LAGUNA_CHECKPOINT_PATH=latest`, `LAGUNA_MODEL_ID=poolside/Laguna-XS.2`, `LAGUNA_MAX_NEW_TOKENS=128` |
| Secret | `HF_TOKEN` with base-model access if required |

In the Hugging Face Inference Endpoint UI, paste one of these objects into the JSON body editor. A plain text payload such as `{"inputs": "Hello world!"}` is not enough; Laguna Vision needs `inputs.image` plus `inputs.question`.

Quick HF UI test payload:

```json
{
  "inputs": {
    "image": "https://images.cocodataset.org/val2017/000000039769.jpg",
    "question": "What animals are in this image? Answer briefly.",
    "max_new_tokens": 64
  }
}
```

Same payload with `curl`:

```bash
HF_ENDPOINT=https://your-endpoint.endpoints.huggingface.cloud
HF_ENDPOINT_TOKEN=...

curl -s "${HF_ENDPOINT}" \
  -H "Authorization: Bearer ${HF_ENDPOINT_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "image": "https://images.cocodataset.org/val2017/000000039769.jpg",
      "question": "What animals are in this image? Answer briefly.",
      "max_new_tokens": 64
    }
  }'
```

Generic request:

```json
{
  "inputs": {
    "image": "https://example.com/image.jpg",
    "question": "What is shown in this image?",
    "max_new_tokens": 128
  }
}
```

Local image as a data URI:

```bash
IMAGE_DATA_URI="$(python3 - <<'PY'
import base64
from pathlib import Path

print("data:image/png;base64," + base64.b64encode(Path("path/to/image.png").read_bytes()).decode("ascii"))
PY
)"

curl -s "${HF_ENDPOINT}" \
  -H "Authorization: Bearer ${HF_ENDPOINT_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{
    \"inputs\": {
      \"image\": \"${IMAGE_DATA_URI}\",
      \"question\": \"What is shown in this image?\",
      \"max_new_tokens\": 64
    }
  }"
```

OpenAI-style multimodal request:

```json
{
  "inputs": {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is shown?"},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
      }
    ]
  }
}
```

Response:

```json
{"answer": "...", "checkpoint": "latest"}
```

## vLLM serving

For production vLLM serving, keep this HF endpoint as the reference path and run the repository gateway separately:

1. Merge `latest/lora` into `poolside/Laguna-XS.2` with `laguna-vision-vllm merge-lora`, or start vLLM with `--enable-lora --lora-modules laguna-vision=latest/lora`.
2. Start vLLM with `--trust-remote-code --enable-prompt-embeds`.
3. Start `laguna-vision-vllm serve --checkpoint latest --vllm-base-url http://127.0.0.1:8000/v1 --model laguna-vision`.
4. Compare this endpoint and the vLLM gateway with `laguna-vision-vllm compare-endpoints` on `evals/live_capability_eval_80/probe/manifest.jsonl`.

The gateway sends a single full embedding tensor to vLLM's `/v1/completions` API using top-level `prompt_embeds`; it does not send `prompt_embeds` as a chat content part. Prime validation on 2026-05-30 confirmed this vLLM API works on `vllm==0.10.2`. The real `poolside/Laguna-XS.2` backend needs an 80GB-class GPU or equivalent memory plan; a 1x A100 40GB pod reached the correct vLLM Transformers backend and then failed with CUDA OOM.

## Limitations

- Early checkpoint quality is uneven.
- OCR, counting, charts, tables, and precise UI localization are unreliable.
- The model can hallucinate when visual evidence is weak.
- Validate outputs before using them in user-facing or high-stakes workflows.
