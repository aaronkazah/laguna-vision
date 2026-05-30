# Laguna Vision

Laguna Vision trains a lightweight visual adapter for `poolside/Laguna-XS.2`. It turns images and screenshots into visual tokens, projects them into Laguna's embedding space, and trains the projector plus optional LoRA adapters with supervised image-question-answer data.

The repository includes data materialization, training, evaluation, checkpoint publishing, and inference from local or private Hugging Face checkpoints.

This was built for the [Poolside Research Hackathon](https://www.competehub.dev/en/competitions/lumaevt-toewzCfp1Ue1PcR) as a near-capability exploration for computer use: giving Laguna enough visual grounding to inspect screenshots, UI states, error messages, documents, and visual context instead of relying on text-only prompts.

## Current release

`latest/` points to `laguna-general-vision-200k-20260529-r2/stage2/step_000900`, a 200k-example staged run:

| Stage | What trained | Examples | Purpose |
|---|---|---:|---|
| Stage 1 alignment | projector only | 80,000 | Teach SigLIP image features to become visual tokens Laguna can condition on. |
| Stage 2 instruction | projector + LoRA | 120,000 | Teach visual QA, descriptions, OCR/docs/charts, screenshots/UI, and spatial/compositional answers. |

The full locked recipe remains 300k examples, but the released checkpoint is the proportional 200k budget slice. That was the deliberate hackathon concession: spend the limited time and GPU budget on a broad end-to-end model, endpoint, and reproducible recipe rather than a narrower OCR-only overfit or a fully optimized final model.

Current behavior: the endpoint serves successfully, but the latest checkpoint is only weakly grounded. In the live 80-case capability matrix it passed **12 / 80** cases: some single-shape, single-color, no-text, and meme/spatial-attribution controls, plus one color+shape binding control. It still fails most color/shape bindings plus exact OCR, dense UI, table extraction, counting, and precise localization checks. It can give plausible generic descriptions while hallucinating the important details.

Core point: this is not tied to one cloud provider. Prime was only the hardware used for the hackathon run. The repository is meant to be cloned, installed, trained locally for small runs, and scaled on any compatible GPU machine for larger reproductions.

## Model

```text
image or screenshot
  -> SigLIP vision tower
  -> AnyRes tiles + 2D position features
  -> resampler projector
  -> Laguna XS.2 input embeddings
  -> answer tokens
```

The default production vision tower is `google/siglip-so400m-patch14-384`. It is stronger than the older CLIP baseline and works with the same adapter path for natural images, documents, charts, and screenshots.

The Python package is `lagunavision`; the distribution and CLI are `laguna-vision`.

## Datasets

The small default public training mixture uses image-bearing Hugging Face datasets:

| Dataset | Purpose |
|---|---|
| `HuggingFaceM4/DocumentVQA` | documents, forms, page layout |
| `lmms-lab/textvqa` | natural images with embedded text |
| `howard-hou/OCR-VQA` | OCR-heavy visual QA |
| `HuggingFaceM4/ChartQA` | charts and diagrams |

LLaVA-format JSON/JSONL is also supported through `laguna-vision llava-materialize`. Use it when the corresponding image archive is present on disk.

### Locked general-vision recipe

The production generalization recipe is locked as `general-vision-300k-v1`: 120k alignment examples followed by 180k instruction examples. It combines natural images, dense descriptions, compositional/spatial QA, OCR/docs/charts, screenshots/UI, and synthetic positional OCR controls. The goal is a broadly usable image model, not a narrow OCR/VQA overfit.

| Stage | Source | Exact dataset/file | Train |
|---|---|---:|---:|
| alignment | LLaVA pretrain | `liuhaotian/LLaVA-Pretrain` / `blip_laion_cc_sbu_558k.json` + `images.zip` | 120,000 |
| instruction | LLaVA instruct | `liuhaotian/LLaVA-Instruct-150K` / `llava_instruct_150k.json` + COCO `train2017` | 65,000 |
| instruction | ShareGPT4V dense descriptions | `Lin-Chen/ShareGPT4V` / `sharegpt4v_instruct_gpt4-vision_cap100k.json` + COCO `train2017` | 35,000 |
| instruction | GQA balanced | `lmms-lab/GQA` / `train_balanced_instructions` + `train_balanced_images` | 25,000 |
| instruction | Document/OCR/chart QA | `HuggingFaceM4/DocumentVQA`, `lmms-lab/textvqa`, `howard-hou/OCR-VQA`, `HuggingFaceM4/ChartQA` | 25,000 |
| instruction | Screenshots/UI/web | `HuggingFaceM4/WebSight`, `rootsautomation/RICO-ScreenQA`, `rootsautomation/RICO-Screen2Words`, `rootsautomation/websrc` | 25,000 |
| instruction | Positional OCR controls | local `lagunavision` synthetic generator | 5,000 |

Inspect the exact recipe without downloading data:

```bash
laguna-vision general-materialize \
  --output-dir /tmp/laguna-general-dry-run \
  --dry-run
```

Inspect a proportional 100k budget slice for a cheaper dry run:

```bash
laguna-vision general-materialize \
  --output-dir /tmp/laguna-general-100k-dry-run \
  --train-budget 100000 \
  --dry-run
```

That 100k slice preserves the full recipe ratio: 40k alignment examples and 60k instruction examples distributed across every source, rather than dropping whole capability areas.

The released 200k run used the same proportional recipe with a larger budget:

| Stage | Source | Train examples |
|---|---|---:|
| alignment | LLaVA pretrain | 80,000 |
| instruction | LLaVA instruct | 43,333 |
| instruction | ShareGPT4V dense descriptions | 23,333 |
| instruction | GQA balanced | 16,667 |
| instruction | DocumentVQA | 4,667 |
| instruction | TextVQA | 4,667 |
| instruction | OCR-VQA | 4,000 |
| instruction | ChartQA | 3,333 |
| instruction | WebSight | 6,667 |
| instruction | RICO ScreenQA | 4,667 |
| instruction | RICO Screen2Words | 2,666 |
| instruction | WebSRC | 2,667 |
| instruction | synthetic spatial OCR | 3,333 |

Raw data notes from the 200k run:

| Asset | Approximate size/count | Why it is needed |
|---|---:|---|
| LLaVA pretrain `images.zip` extraction | 558k images | stage-1 visual-language alignment |
| COCO `train2017.zip` | ~18GB compressed, 118,287 images | image root for LLaVA instruct, ShareGPT4V, and COCO-backed QA rows |
| Materialized manifests | small JSONL files under `/data/laguna-vlm/datasets/...` | reproducible selected 200k train/eval slice |

Keep raw public datasets under your run root, for example `runs/laguna-vlm/datasets/raw` locally or `/data/laguna-vlm/datasets/raw` on a GPU host. Do not upload the raw image archives to Hugging Face; publish only manifests, run metadata, checkpoints, and LoRA/projector artifacts.

Materialize the full recipe in a persistent run root:

```bash
export LAGUNA_VLM_ROOT="${PWD}/runs/laguna-vlm"

laguna-vision general-materialize \
  --output-dir "${LAGUNA_VLM_ROOT}/datasets/general-vision-300k-v1" \
  --download-assets \
  --coco-train2017-root "${LAGUNA_VLM_ROOT}/datasets/raw/coco/train2017" \
  --llava-pretrain-image-root "${LAGUNA_VLM_ROOT}/datasets/raw/llava_pretrain"
```

The command writes staged manifests at `alignment/train.jsonl`, `alignment/eval.jsonl`, `instruction/train.jsonl`, and `instruction/eval.jsonl`, plus `controls/wrong.jsonl` and `controls/blank.jsonl` for real-vs-wrong-vs-blank image checks.

How the real run works:

1. Choose a persistent run root; raw datasets, manifests, feature caches, logs, and checkpoints all live under that directory.
2. `general-materialize` downloads the public datasets and writes JSONL manifests. Raw data is not uploaded to Hugging Face.
3. `cache-visual-features` runs locally or in parallel across GPUs and stores SigLIP features under the run root.
4. Stage 1 trains only the visual projector for image-text alignment.
5. Stage 2 starts from the stage-1 projector and trains projector + LoRA adapters for instruction following.
6. Evaluation compares correct-image, wrong-image, and blank-image manifests so a checkpoint cannot pass by memorizing answer priors.
7. Publishing uploads only trained artifacts (`projector.pt`, `projector_spec.json`, `train_report.json`, and LoRA adapter files), not the raw dataset.

For a provider-neutral 200k reproduction, use a persistent run root and run the same staged recipe:

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

`MAX_TILES=4` means each image is encoded as a global view plus up to four high-detail crops. This costs more feature-cache/training time than one tile, but it is important for screenshots, documents, OCR, charts, and small objects.

Checkpoint/resume behavior:

| Path | Purpose |
|---|---|
| `${LAGUNA_VLM_ROOT}/datasets/...` | materialized manifests and raw/downloaded dataset assets |
| `${FEATURE_CACHE_ROOT}` | reusable SigLIP feature cache |
| `${LAGUNA_VLM_ROOT}/checkpoints/${RUN_NAME}/stage1_alignment` | stage-1 final and `step_*` checkpoints |
| `${LAGUNA_VLM_ROOT}/checkpoints/${RUN_NAME}/stage2_instruction` | stage-2 final and `step_*` checkpoints |
| `${LAGUNA_VLM_ROOT}/logs/${RUN_NAME}` | launcher/job logs |

Both stages can save every 100 optimizer steps. If a local or remote job is interrupted, rerun with the same run root; manifests, cached features, and `step_*` checkpoints are reused.

Offsite Hugging Face backup:

- Set `HF_REPO_ID`, for example `aaronkazah/laguna-vision`, when launching.
- During the run, every new `stage1/step_*` and `stage2/step_*` checkpoint is uploaded to the model repo at `HF_PATH_IN_REPO`/`RUN_NAME`.
- On exit, final stage checkpoints plus `recipe.json`, `run_state.json`, and `job.log` are uploaded.
- Raw datasets and full feature caches are not uploaded; they are recreated from the recipe and public dataset sources, or reused from your persistent run root.

No raw datasets, checkpoints, feature caches, tokens, or model weights are committed to GitHub.

## Install

```bash
git clone https://github.com/aaronkazah/laguna-vision.git
cd laguna-vision

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,data,llama,publish]'
```

Authenticate when using gated models or private checkpoint repos:

```bash
hf auth login
```

The install is local-first: it works on a laptop for the small smoke test and on any CUDA GPU host for Laguna training. `poolside/Laguna-XS.2` may require an approved Hugging Face token; the tiny smoke test below does not.

## Train locally

Run a small end-to-end check on CPU. This creates synthetic image data, trains a tiny adapter against `sshleifer/tiny-gpt2`, asks one image question, and runs the ablation evaluator:

```bash
scripts/local_smoke_train.sh
```

Generate deterministic capability probes:

```bash
laguna-vision capability-probe \
  --output-dir data/capability_probe
```

Materialize a small real-image split for local Laguna adapter training:

```bash
laguna-vision hf-materialize \
  --output-dir data/hf_vqa \
  --train-count 1000 \
  --eval-count 100
```

Train a local Laguna adapter. Use `--device cuda --vision-device cuda` on an NVIDIA GPU; for tiny development runs, swap to a small HF model and CPU like `scripts/local_smoke_train.sh`.

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

Reproduce the released 200k staged run on any compatible GPU machine by reusing the same run root from the recipe section:

```bash
export LAGUNA_VLM_ROOT="${PWD}/runs/laguna-vlm"
export RUN_NAME=laguna-general-vision-200k-repro
export DATA_DIR="${LAGUNA_VLM_ROOT}/datasets/general-vision-300k-v1-budget200k"
export FEATURE_CACHE_ROOT="${LAGUNA_VLM_ROOT}/feature_cache/general-vision-300k-v1-budget200k-siglip-so400m-tiles4"
export ALLOW_LOCAL_OUTPUT=1
export NPROC=1        # set to the number of visible GPUs on a multi-GPU host
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

The same commands run locally at smaller budgets by changing `--train-budget`, `NPROC`, `MAX_TILES`, and the manifest paths. The core recipe is two-stage and provider-neutral: projector alignment first, then projector + LoRA instruction tuning.

## Evaluate

The main eval is a deterministic capability probe designed to answer two questions: is the model actually seeing the image, and which visual categories are currently reliable vs. weak?

Generate the probe:

```bash
laguna-vision capability-probe \
  --output-dir data/capability_probe
```

Run it against a Hugging Face Endpoint or any compatible HTTP endpoint:

```bash
HF_ENDPOINT_TOKEN=... \
laguna-vision eval-endpoint \
  --endpoint https://your-endpoint.endpoints.huggingface.cloud \
  --manifest data/capability_probe/manifest.jsonl \
  --output runs/evals/capability_probe.answers.jsonl \
  --summary-output runs/evals/capability_probe.summary.json
```

The default probe has **80 cases**, with **10 cases per category**. The output JSONL stores every prompt, raw endpoint payload, raw answer string, extracted final answer, expected answer, score, and category. Thinking/reasoning remains enabled; scoring uses only the extracted final answer after removing visible thought blocks, answer labels, placeholder junk, and chat-template tags.

Latest live result for `latest/` on the Hugging Face endpoint: **12 / 80 passed (15.0%)**.

| Category | Live result | What it measures |
|---|---:|---|
| `basic_shape` | 2 / 10 | Single-object shape recognition without requiring color. |
| `basic_color` | 3 / 10 | Single-object color recognition without requiring shape. |
| `color_shape_binding` | 1 / 10 | Binding the correct color to the correct shape. |
| `no_text_control` | 3 / 10 | Abstaining when no OCR text exists. |
| `tiny_ocr` | 0 / 10 | Exact small terminal text. |
| `dense_ui_localization` | 0 / 10 | Small UI table state and row localization. |
| `meme_semantics` | 3 / 10 | Meme-style visual relationship attribution, such as which side is pushing the center character. |
| `table_precision` | 0 / 10 | Precise document/table value extraction. |

Use this as the first checkpoint sanity test. For the current `latest/`, there is measurable but unstable low-level shape/color signal; future checkpoints should first improve color coverage and color-shape binding before claiming stronger computer-use vision.

For local checkpoint evaluation, run the ablation scorer:

```bash
laguna-vision eval-ablation \
  --manifest data/hf_vqa/eval.jsonl \
  --checkpoint checkpoints/laguna_hf_vqa/projector.pt \
  --output checkpoints/laguna_hf_vqa/ablation.jsonl \
  --threshold 0.15 \
  --device cuda \
  --vision-device cuda
```

`eval-ablation` compares blind text answers, OCR-only answers, trained visual answers, and an untrained visual-control arm. The summary is written beside the JSONL output.

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

## Hugging Face checkpoints

Publish an early or final checkpoint privately:

```bash
HF_REPO_ID=your-org/laguna-vision \
CHECKPOINT_DIR=runs/laguna-vlm/checkpoints/laguna_hf_vqa/step_000250 \
HF_PRIVATE=1 \
scripts/publish_hf_checkpoint.sh
```

Load it directly for inference:

```bash
laguna-vision ask-image \
  --checkpoint hf://your-org/laguna-vision/latest \
  --image path/to/image.png \
  --question "Explain this image." \
  --backbone laguna \
  --device cuda \
  --vision-device cuda
```

## Scaling beyond a laptop

The same commands work on any GPU host: a workstation, rented box, Slurm node, Kubernetes job, or cloud VM. Put `LAGUNA_VLM_ROOT` on persistent storage, set `NPROC` to the number of visible GPUs, and rerun the stage scripts if interrupted.

Recommended large-run baseline:

| Resource | Value |
|---|---|
| GPU | 1-8 CUDA GPUs, 80GB+ preferred for Laguna |
| Dataset size | small smoke test locally; 200k to duplicate the released run |
| Vision tower | `google/siglip-so400m-patch14-384` |
| Checkpoint cadence | every 100 optimizer steps for resumability |
| Durable state | datasets, feature cache, checkpoints, logs, and eval outputs under `LAGUNA_VLM_ROOT` |

## Repository layout

```text
lagunavision/        Python package
scripts/             training, evaluation, deployment, and publishing entrypoints
tests/               unit and integration tests
.github/workflows/   CI
```
