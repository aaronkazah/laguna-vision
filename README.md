# Laguna Vision

Laguna XS.2 is text-only. For real computer use, that is a hard limit: the relevant state is often a screenshot, a browser page, a form, a chart, a terminal image, or an error dialog rather than text in the prompt.

Laguna Vision adds a visual input path to `poolside/Laguna-XS.2`. It uses SigLIP to encode images, an AnyRes tiling path for screenshots and documents, a resampler projector to map visual features into Laguna's embedding space, and LoRA adapters for supervised instruction tuning.

This is **post-training multimodal adaptation via supervised fine-tuning**: Stage 1 aligns the visual projector while Laguna is frozen; Stage 2 trains the projector plus LoRA adapters on image-question-answer and visual-instruction data.

The current `latest/` checkpoint is an early 200k-example run. It is not a finished vision model. It is useful because the full path is inspectable: data recipe, manifests, training scripts, checkpoint lineage, endpoint code, and raw eval answers.

## Vision pathway breakdown

Laguna can already generate text, but it has no sensor for pixels and no learned bridge from pixels into its token stream. Laguna Vision adds that missing interface end to end: pixels become SigLIP features, features become Laguna-space visual tokens, and the trained adapter is measured to see whether those tokens actually affect the answer.

| Step | Implementation | Why it matters |
|---|---|---|
| Visual sensing | SigLIP vision encoder with AnyRes global/crop tiling | Screenshots, documents, and UI captures need more detail than a single low-resolution image view. |
| Token bridge | Resampler projector producing 256 Laguna-space visual tokens | Laguna can condition on image evidence without replacing the language backbone. |
| Post-training | Stage 1 projector alignment, then Stage 2 projector + LoRA supervised tuning | The model itself is adapted; this is not a separate captioner bolted beside Laguna. |
| Training coverage | Natural images, dense captions, compositional QA, OCR, documents, charts, screenshots, UI data, and synthetic positional controls | The recipe is designed to transfer beyond one website, one UI, or one benchmark. |
| Grounding audit | 80 deterministic live probes with raw payloads and extracted final answers | The result is reproducible, inspectable, and honest about failures. |
| Target workflows | Coding and operations agents reading browser state, dashboards, forms, terminal screenshots, and error dialogs | The value is direct visual access to state that agents currently need humans to transcribe. |

## Current checkpoint and result

| Field | Value |
|---|---|
| Checkpoint | `laguna-general-vision-200k-20260529-r2/stage2/step_000900` (`latest/`) |
| Base model | `poolside/Laguna-XS.2` |
| Vision encoder | `google/siglip-so400m-patch14-384` |
| Visual path | AnyRes global view + up to 4 high-detail tiles |
| Visual tokens | 256 |
| Projector | resampler |
| Trainable weights | Stage 1: projector only; Stage 2: projector + LoRA |
| LoRA | rank 64, alpha 128, dropout 0.05 |
| Released run | 200k examples: 80k alignment + 120k instruction |
| Locked full recipe | 300k examples: 120k alignment + 180k instruction |
| Live eval | 12 / 80 strict passes |

The 200k run was the proportional slice of the locked 300k recipe. The tradeoff was breadth and reproducibility over polishing one narrow benchmark.

| Category | Result | Measures |
|---|---:|---|
| `basic_shape` | 2 / 10 | Single-object shape recognition. |
| `basic_color` | 3 / 10 | Single-object color recognition. |
| `color_shape_binding` | 1 / 10 | Binding the correct color to the correct shape. |
| `no_text_control` | 3 / 10 | Abstaining when no text is visible. |
| `tiny_ocr` | 0 / 10 | Exact small terminal text. |
| `dense_ui_localization` | 0 / 10 | Dense UI row/status localization. |
| `meme_semantics` | 3 / 10 | Simple visual relationship attribution. |
| `table_precision` | 0 / 10 | Precise document/table value extraction. |

The raw eval is committed at `evals/live_capability_eval_80/`. Read `evals/live_capability_eval_80/capability_probe.answers.rescored.jsonl` to inspect every prompt, expected answer, raw endpoint payload, extracted final answer, and score.

## Dataset recipe

Small local runs can use the built-in Hugging Face VQA mixture:

| Dataset | Purpose |
|---|---|
| `HuggingFaceM4/DocumentVQA` | documents and forms |
| `lmms-lab/textvqa` | natural images with embedded text |
| `howard-hou/OCR-VQA` | OCR-heavy QA |
| `HuggingFaceM4/ChartQA` | charts and diagrams |

The large recipe is `general-vision-300k-v1`:

| Stage | Source | Full recipe | 200k release |
|---|---|---:|---:|
| alignment | LLaVA pretrain | 120,000 | 80,000 |
| instruction | LLaVA instruct | 65,000 | 43,333 |
| instruction | ShareGPT4V dense descriptions | 35,000 | 23,333 |
| instruction | GQA balanced | 25,000 | 16,667 |
| instruction | DocumentVQA | 7,000 | 4,667 |
| instruction | TextVQA | 7,000 | 4,667 |
| instruction | OCR-VQA | 6,000 | 4,000 |
| instruction | ChartQA | 5,000 | 3,333 |
| instruction | WebSight | 10,000 | 6,667 |
| instruction | RICO ScreenQA | 7,000 | 4,667 |
| instruction | RICO Screen2Words | 4,000 | 2,666 |
| instruction | WebSRC | 4,000 | 2,667 |
| instruction | synthetic spatial OCR | 5,000 | 3,333 |

External assets required for the large recipe:

| Asset | Purpose |
|---|---|
| `liuhaotian/LLaVA-Pretrain` `images.zip` + `blip_laion_cc_sbu_558k.json` | stage-1 alignment |
| COCO `train2017` | image root for LLaVA instruct, ShareGPT4V, and COCO-backed QA |
| Public HF datasets above plus `HuggingFaceM4/WebSight`, RICO ScreenQA/Screen2Words, and WebSRC | documents, OCR, charts, screenshots, and web/UI data |

Keep raw datasets under a local or GPU-host run root such as `runs/laguna-vlm/datasets/raw` or `/data/laguna-vlm/datasets/raw`. Do not upload raw image archives, feature caches, tokens, or full gated Laguna base weights to Hugging Face.

## Install

```bash
git clone https://github.com/aaronkazah/laguna-vision.git
cd laguna-vision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,data,llama,publish]'
hf auth login
```

The CPU smoke test uses `sshleifer/tiny-gpt2`; Laguna training requires a machine that can load `poolside/Laguna-XS.2`.

## Reproduce the 200k run

```bash
export LAGUNA_VLM_ROOT="${PWD}/runs/laguna-vlm"
export RUN_NAME=laguna-general-vision-200k-repro
export DATA_DIR="${LAGUNA_VLM_ROOT}/datasets/general-vision-300k-v1-budget200k"
export FEATURE_CACHE_ROOT="${LAGUNA_VLM_ROOT}/feature_cache/general-vision-300k-v1-budget200k-siglip-so400m-tiles4"

laguna-vision general-materialize \
  --output-dir "${DATA_DIR}" \
  --train-budget 200000 \
  --download-assets \
  --coco-train2017-root "${LAGUNA_VLM_ROOT}/datasets/raw/coco/train2017" \
  --llava-pretrain-image-root "${LAGUNA_VLM_ROOT}/datasets/raw/llava_pretrain"
```

Run Stage 1 and Stage 2:

```bash
export ALLOW_LOCAL_OUTPUT=1
export NPROC=1        # set to visible GPU count on a multi-GPU host
export MAX_TILES=4

MANIFEST="${DATA_DIR}/alignment/train.jsonl" \
EVAL_MANIFEST="${DATA_DIR}/alignment/eval.jsonl" \
OUTPUT_DIR="${LAGUNA_VLM_ROOT}/checkpoints/${RUN_NAME}/stage1_alignment" \
FEATURE_CACHE_DIR="${FEATURE_CACHE_ROOT}/alignment" \
SAVE_EVERY=100 \
scripts/laguna_llava_stage1.sh

MANIFEST="${DATA_DIR}/instruction/train.jsonl" \
EVAL_MANIFEST="${DATA_DIR}/instruction/eval.jsonl" \
STAGE1_CHECKPOINT="${LAGUNA_VLM_ROOT}/checkpoints/${RUN_NAME}/stage1_alignment/projector.pt" \
OUTPUT_DIR="${LAGUNA_VLM_ROOT}/checkpoints/${RUN_NAME}/stage2_instruction" \
FEATURE_CACHE_DIR="${FEATURE_CACHE_ROOT}/instruction" \
SAVE_EVERY=100 \
scripts/laguna_llava_stage2.sh
```

Reuse the same `LAGUNA_VLM_ROOT` to resume; manifests, feature caches, and `step_*` checkpoints are durable.

## Small checks

```bash
scripts/local_smoke_train.sh

laguna-vision hf-materialize \
  --output-dir data/hf_vqa \
  --train-count 1000 \
  --eval-count 100
```

Train a small CUDA adapter:

```bash
laguna-vision train-visual-bridge \
  --backbone laguna \
  --manifest data/hf_vqa/train.jsonl \
  --eval-manifest data/hf_vqa/eval.jsonl \
  --output-dir checkpoints/laguna_hf_vqa \
  --encoder hf \
  --encoder-id google/siglip-so400m-patch14-384 \
  --projector resampler \
  --visual-tokens 256 \
  --max-tiles 1 \
  --batch-size 1 \
  --grad-accum 16 \
  --epochs 1 \
  --lora-rank 64 \
  --lora-alpha 128 \
  --device cuda \
  --vision-device cuda
```

Ask a checkpoint about an image:

```bash
laguna-vision ask-image \
  --checkpoint checkpoints/laguna_hf_vqa/projector.pt \
  --image path/to/image.png \
  --question "Explain this image." \
  --backbone laguna \
  --device cuda \
  --vision-device cuda
```

## Evaluate

```bash
laguna-vision capability-probe \
  --output-dir data/capability_probe

HF_ENDPOINT_TOKEN=... \
laguna-vision eval-endpoint \
  --endpoint https://your-endpoint.endpoints.huggingface.cloud \
  --manifest data/capability_probe/manifest.jsonl \
  --output runs/evals/capability_probe.answers.jsonl \
  --summary-output runs/evals/capability_probe.summary.json
```

The default probe has 80 cases: 8 categories, 10 cases each. Thinking/reasoning can remain enabled; scoring uses the extracted final answer after removing visible thought blocks, answer labels, placeholder text, and chat-template tags.

For local ablations:

```bash
laguna-vision eval-ablation \
  --manifest data/hf_vqa/eval.jsonl \
  --checkpoint checkpoints/laguna_hf_vqa/projector.pt \
  --output checkpoints/laguna_hf_vqa/ablation.jsonl \
  --threshold 0.15 \
  --device cuda \
  --vision-device cuda
```

`eval-ablation` compares blind text answers, OCR-only answers, trained visual answers, and an untrained visual-control arm.

## Hugging Face repo and endpoint

Publish a checkpoint:

```bash
HF_REPO_ID=your-org/laguna-vision \
CHECKPOINT_DIR=runs/laguna-vlm/checkpoints/laguna_hf_vqa/step_000250 \
HF_PRIVATE=1 \
scripts/publish_hf_checkpoint.sh
```

Recommended model repo contents:

| Path | Purpose |
|---|---|
| `README.md` | concise model card with checkpoint, recipe, eval, and limitations |
| `latest/` | stable adapter target |
| `<run_name>/stage1/step_*` and `<run_name>/stage2/step_*` | checkpoint lineage |
| `<run_name>/run_metadata/{recipe.json,run_state.json,job.log}` | run audit trail |
| `handler.py` and `requirements.txt` | endpoint runtime |
| `evals/live_capability_eval_80/` | probe images, manifest, summary, and raw answers |

Endpoint settings:

| Setting | Value |
|---|---|
| Runtime | Hugging Face Dedicated Inference Endpoint, default Python runtime |
| Accelerator | A100 80GB for the first deployment |
| Environment | `LAGUNA_CHECKPOINT_PATH=latest`, `LAGUNA_MODEL_ID=poolside/Laguna-XS.2`, `LAGUNA_MAX_NEW_TOKENS=128` |
| Secret | `HF_TOKEN` with base-model access if Laguna XS.2 is gated/private |

Simple request:

```json
{
  "inputs": {
    "image": "https://example.com/image.jpg",
    "question": "What is shown in this image?",
    "max_new_tokens": 128
  }
}
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

## Repository layout

```text
lagunavision/        Python package
scripts/             training, evaluation, deployment, and publishing entrypoints
tests/               tests
.github/workflows/   CI
```
