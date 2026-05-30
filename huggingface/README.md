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

Laguna Vision is a vision-language adapter for `poolside/Laguna-XS.2`. It adds image and screenshot understanding by encoding visual inputs with `google/siglip-so400m-patch14-384`, projecting the resulting visual tokens into Laguna's embedding space, and applying a LoRA instruction adapter.

This repository contains the adapter artifacts and endpoint code needed to run the model. It does not duplicate the full Laguna base model weights.

The project was built for the [Poolside Research Hackathon](https://www.competehub.dev/en/competitions/lumaevt-toewzCfp1Ue1PcR) as a near-capability exploration for computer use: visual grounding for screenshots, UI state, code/debug images, documents, and other context that a text-only Laguna prompt cannot inspect directly.

## Current release

The `latest/` checkpoint points to `laguna-general-vision-200k-20260529-r2/stage2/step_000900`.

| Field | Value |
|---|---|
| Base model | `poolside/Laguna-XS.2` |
| Vision encoder | `google/siglip-so400m-patch14-384` |
| Checkpoint | `step_000900` |
| Stage 2 training examples | `120,000` |
| Visual tokens | `256` |
| Adapter files | `projector.pt`, `projector_spec.json`, `lora/adapter_model.safetensors`, `lora/adapter_config.json` |

Training used two stages:

| Stage | What trained | Examples | Purpose |
|---|---|---:|---|
| Stage 1 alignment | projector only | 80,000 | Map SigLIP visual features into Laguna's token space. |
| Stage 2 instruction | projector + LoRA | 120,000 | Learn visual QA, descriptions, OCR/docs/charts, screenshots/UI, and spatial/compositional answers. |

The full locked recipe is 300k examples, but this release uses a proportional 200k hackathon slice to fit the available time and GPU budget. The tradeoff is breadth over polish: endpoint and recipe are working, while exact visual detail quality still needs more training and evaluation.

## Current evaluation status

Serving works, but the latest checkpoint is only weakly grounded. In a live 80-case capability matrix against `latest/`, it passed **12 / 80** cases: some single-shape, single-color, no-text, and meme/spatial-attribution controls, plus one color+shape binding control. It still fails most color/shape bindings plus exact OCR, dense UI state, and table values.

The repository includes a deterministic capability probe:

```bash
laguna-vision capability-probe --output-dir data/capability_probe
HF_ENDPOINT_TOKEN=... laguna-vision eval-endpoint \
  --endpoint https://your-endpoint.endpoints.huggingface.cloud \
  --manifest data/capability_probe/manifest.jsonl \
  --output runs/evals/capability_probe.answers.jsonl \
  --summary-output runs/evals/capability_probe.summary.json
```

The default probe has **80 cases**, with **10 cases per category**. The evaluator keeps thinking/reasoning enabled but scores only the extracted final answer after removing visible thought blocks, answer labels, placeholder junk, and chat-template tags. It writes both the raw endpoint payload and final extracted answer for auditability.

Latest live category results:

| Category | Live result | Meaning |
|---|---:|---|
| `basic_shape` | 2 / 10 | Single-object shape recognition without requiring color. |
| `basic_color` | 3 / 10 | Single-object color recognition without requiring shape. |
| `color_shape_binding` | 1 / 10 | Binding the correct color to the correct shape. |
| `no_text_control` | 3 / 10 | No-text images should not hallucinate OCR. |
| `tiny_ocr` | 0 / 10 | Small terminal text remains weak. |
| `dense_ui_localization` | 0 / 10 | Dense UI row/status localization remains weak. |
| `meme_semantics` | 3 / 10 | Meme-style visual relationship attribution, such as which side is pushing the center character. |
| `table_precision` | 0 / 10 | Precise table value extraction remains weak. |

## Deployment

Use a Hugging Face Dedicated Inference Endpoint with the default Hugging Face Python runtime and this repository's `handler.py`.

Recommended first deployment:

| Setting | Value |
|---|---|
| Accelerator | NVIDIA A100 80GB |
| Inference engine | Default |
| Container arguments | blank |
| Container command | blank |

Environment variables:

```text
LAGUNA_CHECKPOINT_PATH=latest
LAGUNA_MODEL_ID=poolside/Laguna-XS.2
LAGUNA_MAX_NEW_TOKENS=128
```

If `poolside/Laguna-XS.2` is private or gated, add an endpoint secret named `HF_TOKEN` with access to that base model.

## API

Send requests to the endpoint root URL.

### Simple image-question format

```json
{
  "inputs": {
    "image": "https://example.com/image.jpg",
    "question": "What is shown in this image?",
    "max_new_tokens": 128
  }
}
```

`image` can be an HTTPS URL, a base64 string, or a data URI.

Example data URI request:

```json
{
  "inputs": {
    "image": "data:image/png;base64,...",
    "question": "Read the visible text."
  }
}
```

### OpenAI-style multimodal format

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
{
  "answer": "...",
  "checkpoint": "latest"
}
```

## Intended use

Laguna Vision is intended for exploratory image understanding tasks across natural images, screenshots, OCR-heavy images, documents, charts, UI captures, and spatial questions.

## Limitations

- This is an experimental adapter checkpoint, not a safety-reviewed production model.
- OCR, counting, charts, and precise spatial localization may be unreliable.
- The model can hallucinate or answer from language priors when visual evidence is weak.
- Outputs should be validated before use in user-facing or high-stakes workflows.
