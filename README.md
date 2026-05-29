# Laguna Vision

Laguna Vision trains a lightweight visual adapter for `poolside/Laguna-XS.2`. It turns images and screenshots into visual tokens, projects them into Laguna's embedding space, and trains the projector plus optional LoRA adapters with supervised image-question-answer data.

The repository includes data materialization, training, evaluation, checkpoint publishing, and inference from local or private Hugging Face checkpoints.

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

The default public training mixture uses image-bearing Hugging Face datasets:

| Dataset | Purpose |
|---|---|
| `HuggingFaceM4/DocumentVQA` | documents, forms, page layout |
| `lmms-lab/textvqa` | natural images with embedded text |
| `howard-hou/OCR-VQA` | OCR-heavy visual QA |
| `HuggingFaceM4/ChartQA` | charts and diagrams |

LLaVA-format JSON/JSONL is also supported through `laguna-vision llava-materialize`. Use it when the corresponding image archive is present on disk.

No raw datasets, checkpoints, feature caches, tokens, or model weights are committed to GitHub.

## Install

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,data,llama,publish]'
```

Authenticate when using gated models or private checkpoint repos:

```bash
hf auth login
```

## Train locally

Run a small end-to-end check before using remote GPUs:

```bash
scripts/local_smoke_train.sh
```

Materialize a small real-image split:

```bash
laguna-vision hf-materialize \
  --output-dir data/hf_vqa \
  --train-count 1000 \
  --eval-count 100
```

Train a local adapter:

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

## Evaluate

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
CHECKPOINT_DIR=/mnt/prime/laguna-vlm/checkpoints/laguna_hf_vqa/step_000250 \
HF_PRIVATE=1 \
scripts/publish_hf_checkpoint.sh
```

Load it directly for inference:

```bash
laguna-vision ask-image \
  --checkpoint hf://your-org/laguna-vision/step_000250 \
  --image path/to/image.png \
  --question "Explain this image." \
  --backbone laguna \
  --device cuda \
  --vision-device cuda
```

## Prime training

Use a persistent Prime disk for datasets, feature caches, logs, and checkpoints. The GPU pod can be replaced; the disk and Hugging Face checkpoint repo are the durable artifacts.

Start a detached run on an existing pod:

```bash
export PRIME_SSH_TARGET=ubuntu@<pod-ip>
export LAGUNA_VLM_ROOT=/mnt/prime/laguna-vlm
export MAX_RUNTIME=8h
export TRAIN_COUNT=30000
export EVAL_COUNT=1000
export HF_REPO_ID=your-org/laguna-vision
export PUBLISH_ON_EXIT=1
export PUBLISH_DURING_RUN=1

scripts/prime_start_detached_training.sh
```

The launcher copies the repo to the pod, installs it there, and starts the training process with `nohup`. The job continues if the laptop disconnects. Completed datasets and feature tensors are reused on the mounted disk; checkpoint directories are uploaded to Hugging Face during training and again at exit when `HF_REPO_ID` is set. `MAX_RUNTIME` stops the training process; Prime billing stops when the pod is terminated.

Resume from a published checkpoint on any compatible GPU pod:

```bash
export INIT_CHECKPOINT=hf://your-org/laguna-vision/laguna-vision-b300-20260529/step_000250
scripts/prime_start_detached_training.sh
```

Recommended remote GPU baseline:

| Resource | Value |
|---|---|
| GPU | 8x B300 262GB spot |
| Budget window | 4-8 hours |
| Dataset size | 30k train / 1k eval or larger |
| Vision tower | `google/siglip-so400m-patch14-384` |
| Checkpoint cadence | every 50 optimizer steps |

## Repository layout

```text
lagunavision/        Python package
scripts/             training, Prime, and publishing entrypoints
tests/               unit and integration tests
.github/workflows/   CI
```
