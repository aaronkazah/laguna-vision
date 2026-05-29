# Laguna Vision

Laguna Vision is an async-first visual bridge. It converts any image or screenshot into visual tokens, projects them into a frozen language backbone's input-embedding space, trains a lightweight connector, and evaluates the resulting checkpoint against frozen manifests.

The goal is general screen and image understanding: read text and document layout, reason about charts and natural images, and ground coarse spatial position. Developer screens (IDEs, terminals, stack traces, dashboards) are one held-out evaluation target, not a training specialization.

The default backbone is Laguna XS.2. The registry also includes Llama 3.2-3B, and any Hugging Face causal LM that exposes input embeddings works through the `hf` adapter, so the same data, tiling, encoder, connector, checkpoint, inference, and eval surfaces serve every backbone.

```text
frozen language backbone
  + visual tokens from any screen or image
  -> reads text, understands layout, grounds position
```

## Status

| Capability | Status |
|---|---|
| Public Hugging Face image/VQA data materialization | implemented |
| LLaVA JSON/JSONL and HF streaming materialization | implemented |
| Local saved images plus train/eval JSONL manifests | implemented |
| Dynamic AnyRes tiling for high-resolution screenshots | implemented |
| Normalized 2D tile position features | implemented |
| Frozen Hugging Face vision encoder dense patch tokens | implemented |
| Persistent on-disk visual feature cache | implemented |
| MLP connector and fixed-token resampler connector | implemented |
| Pluggable backbone registry (Laguna XS.2 default, any HF causal LM) | implemented |
| Visual-token injection via frozen-backbone input embeddings | implemented |
| Interactive image QA from a saved checkpoint (`ask-image`) | implemented |
| Checkpoint save/load/eval, warm-start, and step checkpoints | implemented |
| Leakage-controlled five-arm ablation eval | implemented |
| Multi-GPU data-parallel training (torchrun) | implemented |
| Prime persistent-disk production scripts | implemented |
| Verified Llama 3.2-3B image connector on H200 | passed controlled image ablation |
| Verified Laguna XS.2 early image checkpoint on H200 | accepts images; trained connector beats untrained control |
| Checkpoint publishing to Hugging Face Hub or releases | implemented |

## Training data policy

Laguna Vision does **not** train on code screenshots. Code screenshots are reserved for blind evaluation, validation probes, and demos.

Training uses public general image/text datasets so the visual bridge learns OCR, document layout, chart reading, natural-image text, and coarse spatial grounding without memorizing developer workflows.

No downloaded training data is checked into this repository. `.gitignore` excludes `/data/`, checkpoints, outputs, and model weight files. Checked-in data is limited to small example JSONL fixtures and test-generated synthetic fixtures.

## Data sources

| Dataset | Usage | Skill target |
|---|---|---|
| `liuhaotian/LLaVA-Pretrain` / official LLaVA JSON | supported by LLaVA materializer | image-caption grounding |
| `liuhaotian/LLaVA-Instruct-150K` / LLaVA v1.5 mixes | supported by LLaVA materializer | image instruction following |
| `HuggingFaceM4/DocumentVQA` | supported by materializer | document reading, forms, page layout |
| `lmms-lab/textvqa` | supported by materializer | natural images with embedded text |
| `howard-hou/OCR-VQA` | supported by materializer | book-cover OCR and visual QA |
| `HuggingFaceM4/ChartQA` | supported by materializer | chart and diagram reading |
| synthetic spatial OCR | generated locally | top/bottom/left/right layout grounding |

The materializer saves image bytes locally and writes JSONL manifests:

```json
{
  "id": "train_00000_lmms_lab_textvqa",
  "image": "train/images/00000.png",
  "question": "what is the brand of phone?",
  "ocr_text": "MIA NOKIA",
  "rubric": "vqa",
  "must_include": ["nokia"],
  "accepted_fix_terms": ["nokia"],
  "must_not_include": [],
  "source_dataset": "lmms-lab/textvqa"
}
```

Dataset size is controlled by CLI flags:

```bash
laguna-vision hf-materialize \
  --output-dir data/hf_vqa \
  --train-count 100000 \
  --eval-count 2000
```

Small local runs can use smaller counts; production-scale runs should use the scale targets below.

For the real Laguna/LLaVA path, materialize official LLaVA-format JSON directly. Local JSON/JSONL rows with `image` and `conversations` are converted into answer-only training examples:

```bash
laguna-vision llava-materialize \
  --source-json /mnt/prime/laguna-vlm/raw/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
  --image-root /mnt/prime/laguna-vlm/raw/LLaVA-Pretrain/images \
  --output-dir /mnt/prime/laguna-vlm/manifests/llava_pretrain \
  --eval-count 2048 \
  --image-mode reference

laguna-vision llava-materialize \
  --source-json /mnt/prime/laguna-vlm/raw/LLaVA-Instruct/llava_v1_5_mix665k.json \
  --image-root /mnt/prime/laguna-vlm/raw/LLaVA-Instruct/images \
  --output-dir /mnt/prime/laguna-vlm/manifests/llava_instruction \
  --eval-count 2048 \
  --image-mode reference
```

`answer` is stored explicitly in the manifest so instruction tuning trains on the assistant response, not on the older rubric keyword fallback.

## Architecture

```text
saved image
  -> global image + high-resolution AnyRes tiles
  -> frozen vision encoder dense patch tokens
  -> normalized 2D tile position features
  -> MLP projector or fixed-token resampler
  -> frozen backbone input embeddings
  -> answer loss on public VQA answer
```

The bridge path freezes the vision encoder and the language-model base weights. By default only the connector is trained. Passing `--lora-rank N` additionally attaches LoRA adapters to the frozen backbone, so the language model learns to use the visual tokens while the base weights stay frozen — the connector and the adapter train together and the adapter is saved alongside the checkpoint.

| Connector | Behavior |
|---|---|
| `mlp` | projects every visual patch token into LLM embedding space |
| `resampler` | compresses variable visual patch tokens into a fixed visual-token budget |

Backbones are selected from a registry, so the data, tiling, encoder, connector, checkpoint, and eval surfaces never change when the backbone does.

| Backbone | `--backbone` | Use |
|---|---|---|
| Laguna XS.2 | `laguna` (default) | production training target |
| Llama 3.2-3B Instruct | `llama` | smaller validation and confidence runs |
| any HF causal LM | `hf` with `--model-id` | local smoke tests, alternate backbones |

Every backbone is loaded as a frozen `AutoModelForCausalLM`; visual tokens are projected into its input-embedding space and prepended ahead of the text tokens, so no model-specific branching is needed.

### LLaVA methodology, not a LLaVA dependency

Laguna Vision does **not** make Laguna run inside LLaVA, Vicuna, or Llama. LLaVA is used as the public training recipe and data shape:

| LLaVA component | Laguna Vision equivalent |
|---|---|
| Llama/Vicuna language backbone | `poolside/Laguna-XS.2` |
| CLIP vision tower | frozen Hugging Face vision encoder |
| MLP vision-language connector | Laguna Vision `mlp` or `resampler` projector |
| image + user prompt + assistant answer JSON | `llava-materialize` output with explicit `answer` targets |
| stage 1 caption alignment + stage 2 instruction tuning | `laguna_llava_stage1.sh` + `laguna_llava_stage2.sh` |

This works because the supervised objective is model-agnostic: encode image patches, project them into the language model's embedding dimension, then train on assistant answers. LLaVA is "Llama-flavored" historically, but its data rows are just visual conversations, so Laguna can learn from the same format.

### Vision tower fidelity

`openai/clip-vit-large-patch14-336` is the default production tower because it is the LLaVA-class baseline: strong, widely supported, and fast enough to cache at scale. It is a good first choice for natural-image and instruction alignment.

It is **not sufficient by itself** to claim arbitrary high-fidelity screenshot understanding. Screenshots, documents, tables, terminals, and charts need:

| Requirement | Repo support |
|---|---|
| more pixels than one 336px crop | AnyRes tiling via `--max-tiles 4` or `--max-tiles 9` |
| text-rich training | DocVQA, TextVQA, OCR-VQA, ChartQA materialization |
| held-out screenshot gates | `web-probe`, developer screenshot eval, and five-arm ablation |
| repeatable proof | frozen eval manifests and saved image bytes |

The practical default is CLIP ViT-L/14 336 for stage 1 and early stage 2. If held-out OCR/screenshot eval saturates below target, switch the same pipeline to a stronger HF vision tower and a separate feature-cache directory.

## Quickstart

Create an environment:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,data,llama]'
```

For the official Meta checkpoint, authenticate Hugging Face before training:

```bash
huggingface-cli login
```

Materialize a small real HF train/eval split:

```bash
laguna-vision hf-materialize \
  --output-dir data/hf_vqa_tutorial \
  --train-count 40 \
  --eval-count 12
```

Train the visual bridge. On macOS, `--device auto` uses MPS when available. On GPU Linux, use `--device cuda --vision-device cuda`.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 laguna-vision train-visual-bridge \
  --manifest data/hf_vqa_tutorial/train.jsonl \
  --eval-manifest data/hf_vqa_tutorial/eval.jsonl \
  --output-dir checkpoints/hf_vqa_tutorial \
  --backbone hf \
  --model-id unsloth/Llama-3.2-3B-Instruct \
  --encoder hf \
  --encoder-id openai/clip-vit-base-patch32 \
  --projector resampler \
  --visual-tokens 64 \
  --max-tiles 9 \
  --max-items 40 \
  --epochs 1 \
  --device auto \
  --vision-device auto
```

Evaluate with the leakage-controlled ablation:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 laguna-vision eval-ablation \
  --manifest data/hf_vqa_tutorial/eval.jsonl \
  --checkpoint checkpoints/hf_vqa_tutorial/projector.pt \
  --output checkpoints/hf_vqa_tutorial/ablation.jsonl \
  --threshold 0.15 \
  --device auto \
  --vision-device auto
```

`eval-ablation` scores five arms on the same items, so visual signal is separated from the text prior and from OCR leakage:

| Arm | Inputs | Isolates |
|---|---|---|
| `text_only` | question | backbone's blind prior |
| `ocr_only` | question + detected text | what plain OCR already answers |
| `image_only` | question + visual tokens | the trained connector's visual signal |
| `image_ocr` | question + visual tokens + detected text | full upper bound |
| `image_untrained` | question + random connector, adapter off | control: tokens present, no training |

The headline metric is `capability_delta = image_only - text_only`; the run clears the gate when it beats `--threshold` (default `+0.15`). `connector_delta = image_only - image_untrained` confirms the trained weights — not the mere presence of prepended tokens — carry the signal: the control uses a random connector and disables any trained LoRA adapter, so it reflects the pre-training state. Both land in `ablation_summary.json`.

Ask a saved checkpoint about one image:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 laguna-vision ask-image \
  --checkpoint checkpoints/hf_vqa_tutorial/projector.pt \
  --image data/hf_vqa_tutorial/eval/images/00000.png \
  --question "What is shown in this image?" \
  --device auto \
  --vision-device auto
```

For a named registry checkpoint, add `--backbone llama` or `--backbone laguna`. If the checkpoint was trained with LoRA, `ask-image` loads the adapter from the sibling `lora/` directory automatically.

Inspect artifacts:

```bash
cat checkpoints/hf_vqa_tutorial/train_report.json
cat checkpoints/hf_vqa_tutorial/projector_spec.json
cat checkpoints/hf_vqa_tutorial/ablation_summary.json
head checkpoints/hf_vqa_tutorial/ablation.jsonl
```

## Distributed training

The trainer is single-process by default and scales to N GPUs under `torchrun`. The frozen backbone is replicated per rank; only the small connector's gradients are averaged across ranks, so the expensive frozen forward scales with GPU count. `torchrun` sets `WORLD_SIZE`/`RANK`/`LOCAL_RANK`, which the trainer reads automatically — no code change between one GPU and eight.

```bash
NPROC=8 scripts/train_visual_bridge_ddp.sh \
  --manifest data/hf_vqa/train.jsonl \
  --eval-manifest data/hf_vqa/eval.jsonl \
  --output-dir checkpoints/laguna_vqa \
  --backbone laguna \
  --encoder hf --encoder-id openai/clip-vit-base-patch32 \
  --projector resampler --visual-tokens 64 --max-tiles 9 \
  --batch-size 8 --grad-accum 4 --epochs 1 \
  --device cuda --vision-device cuda
```

Checkpointing is built in: pass `--save-every N` to write the connector and reports every N optimizer steps on rank 0, in addition to the final save. Use `--grad-accum` to raise the effective batch size and `--no-grad-checkpointing` to trade memory for speed when a GPU has headroom.

To fine-tune the backbone alongside the connector, add LoRA adapters with `--lora-rank` (and optionally `--lora-alpha`, `--lora-dropout`, and `--lora-targets q_proj v_proj`). Only the adapter and connector carry gradients; the base weights stay frozen. The adapter is saved to `lora/` next to `projector.pt` and is loaded automatically at eval and inference time.

```bash
NPROC=8 scripts/train_visual_bridge_ddp.sh \
  --manifest data/hf_vqa/train.jsonl \
  --output-dir checkpoints/laguna_vqa_lora \
  --backbone laguna \
  --encoder hf --encoder-id openai/clip-vit-base-patch32 \
  --projector resampler --visual-tokens 64 --max-tiles 9 \
  --batch-size 8 --grad-accum 4 --epochs 1 \
  --lora-rank 16 --lora-alpha 32 \
  --device cuda --vision-device cuda
```

For large LLaVA data, build the feature cache on the persistent disk before training. This avoids keeping hundreds of thousands of CLIP features in RAM and avoids re-encoding them every run:

```bash
NPROC=8 \
MANIFEST=/mnt/prime/laguna-vlm/manifests/llava_pretrain/train.jsonl \
FEATURE_CACHE_DIR=/mnt/prime/laguna-vlm/feature_cache/llava_stage1 \
scripts/laguna_llava_cache_features.sh
```

Use `--init-checkpoint` to warm-start stage 2 from stage 1, and `--init-lora-dir` to resume a LoRA adapter checkpoint.

### Production Laguna/LLaVA recipe

| Step | What happens | Storage invariant | Confirmation |
|---|---|---|---|
| 0. Prime disk | Create/attach a persistent Prime disk before downloading data | everything under `LAGUNA_VLM_ROOT` on the disk | `prime disks list -o json` shows the disk; pod has it mounted |
| 1. Data | Put LLaVA pretrain + instruction images/JSON on the disk and materialize manifests | `/mnt/prime/laguna-vlm/raw` and `/mnt/prime/laguna-vlm/manifests` | train/eval JSONL exists and image paths resolve |
| 2. Feature cache | Precompute vision-tower features in parallel across GPUs | `/mnt/prime/laguna-vlm/feature_cache` | cache file count approaches unique image count |
| 3. Stage 1 alignment | Frozen Laguna + frozen vision tower; train projector only on LLaVA image-caption alignment | `/mnt/prime/laguna-vlm/checkpoints/laguna_stage1_alignment` | loss decreases; step checkpoints appear |
| 4. Stage 2 instruction | Warm-start projector; train projector + Laguna LoRA on LLaVA instruction data | `/mnt/prime/laguna-vlm/checkpoints/laguna_stage2_instruction` | real-image `ask-image` improves at early checkpoints |
| 5. High-res screenshot/OCR pass | Optional follow-up using DocVQA/TextVQA/ChartQA/screenshots with `MAX_TILES=4..9` | same persistent disk | screenshots/OCR/charts pass held-out eval |
| 6. Cleanup | Terminate pod after confirming disk checkpoints | disk persists; pod is gone | `prime pods list -o json` shows no active pods |

Run scripts for the real path:

```bash
export LAGUNA_VLM_ROOT=/mnt/prime/laguna-vlm

scripts/laguna_llava_stage1.sh
scripts/laguna_llava_stage2.sh

CHECKPOINT=$LAGUNA_VLM_ROOT/checkpoints/laguna_stage2_instruction/step_000250/projector.pt \
IMAGE=$LAGUNA_VLM_ROOT/eval/real_images/your_image.jpg \
scripts/laguna_llava_early_eval.sh
```

The frozen backbone and base vision tower are downloaded from Hugging Face on each pod. The persistent disk holds raw data, manifests, feature caches, every step checkpoint, LoRA adapters, and eval outputs.

## Checkpoint artifacts

| File | Purpose |
|---|---|
| `projector.pt` | trainable connector weights |
| `lora/` | LoRA adapter weights (only when trained with `--lora-rank`) |
| `projector_spec.json` | backbone, model id, encoder id, connector type, token budget, tiling config, LoRA rank |
| `train_report.json` | losses, eval losses, train/eval counts, backbone, device info |
| `ablation.jsonl` | per-arm, per-item answers and scores |
| `ablation_summary.json` | per-arm pass rates, capability/connector deltas, gate result |

A publishable run should package the connector checkpoint, spec, training report, frozen eval manifests, ablation outputs, and exact dataset/source configuration.

### Publishing checkpoints

Early and final checkpoints can be stored in private Hugging Face Hub model repositories. Keep early checkpoints private until the model card, dataset licenses, eval report, and safety notes are ready.

```bash
python -m pip install -e '.[publish]'
huggingface-cli login

HF_REPO_ID=your-org/laguna-vision-early \
CHECKPOINT_DIR=/mnt/prime/laguna-vlm/checkpoints/laguna_stage2_instruction/step_000250 \
HF_PRIVATE=1 \
scripts/publish_hf_checkpoint.sh
```

The script uploads the checkpoint directory (`projector.pt`, `projector_spec.json`, `train_report.json`, optional `lora/`, and eval artifacts) under a path named after the checkpoint directory. Publish raw training data to a Hugging Face dataset repo only when the dataset license allows redistribution; otherwise publish the exact materialization commands, source dataset names, manifest schemas, and eval outputs.

## Verified validation runs

These runs prove the image path end-to-end: data materialization, image loading, visual-token projection, frozen-backbone injection, LoRA adapter save/load, checkpoint reload, and ablation scoring. They are not a claim that the checked-out early checkpoints are production-grade general VLMs.

| Run | Data | Train/eval | Connector | Visual tokens | Result |
|---|---|---:|---|---:|---:|
| Llama 3.2-3B scene bridge | generated scene images | 120 / 20 | resampler + LoRA r8 | 32 | passed: `image_only` 12/12, `text_only` 2/12, `image_untrained` 0/12, `capability_delta=+0.83` |
| Laguna XS.2 early scene bridge | generated scene images | 120 / 20 | resampler + LoRA r16 | 32 | image-capable early checkpoint: `image_only` 10/10, `image_untrained` 0/10, `connector_delta=+1.0`; strict `capability_delta=0.0` because text-only also solved the controlled prompts |
| public HF VQA smoke materialization | DocVQA/TextVQA/OCR-VQA/ChartQA sample | 24 / 8 | n/a | n/a | real image bytes and frozen manifests saved for sanity checks |
| generated scene validation | generated original scenes | 20 / 5 | MLP | variable | 5/5 |
| real HF VQA dense-token validation | public HF VQA mix | 80 / 20 | MLP | up to ~500 | 3/20 |
| real HF VQA resampler validation | public HF VQA mix | 80 / 20 | resampler | 64 | 2/20 |

The Laguna result should be read as an early checkpoint: it proves Laguna can receive and use visual embeddings through this connector, but it did not clear the strict text-prior capability gate on that controlled split. The Llama run did clear the gate and is the stronger current proof that the bridge adds vision rather than merely adding prompt tokens.

## Production-scale targets

| Stage | Target size | Purpose |
|---|---:|---|
| LLaVA stage 1 alignment | ~558k image-caption examples | visual/text embedding alignment |
| LLaVA stage 2 instruction | ~150k instruction examples plus v1.5-style VQA/task mix (~665k total) | visual instruction following |
| text-rich instruction | 50k-250k DocVQA/TextVQA/OCR-VQA/ChartQA examples | OCR, layout, charts, documents |
| synthetic spatial OCR | 1k-10k examples | spatial terms such as top-right, bottom terminal, sidebar |
| blind public VQA eval | 500-2,000 unseen examples | held-out VQA transfer |
| random web screenshot eval | 100 fresh 1920x1080 pages | broad screenshot understanding |
| code screenshot eval | 100-300 held-out screenshots | developer workflow transfer |

A checkpoint should be evaluated against text-only, OCR-only, image-only, and OCR+image baselines on frozen held-out manifests with saved image bytes.

## Additional commands

List configured public datasets:

```bash
laguna-vision datasets
```

Inspect tiling for a 1920x1080 screenshot:

```bash
laguna-vision tile --width 1920 --height 1080
```

Generate synthetic spatial OCR data:

```bash
laguna-vision spatial-ocr --output-dir data/spatial_ocr --count 1000
```

Generate public web-page screenshot probes:

```bash
python -m pip install -e '.[web]'
python -m playwright install chromium
laguna-vision web-probe --output-dir eval_web --limit 5 --width 1920 --height 1080
```

Generate held-out developer screenshot fixtures for local validation:

```bash
laguna-vision demo-eval --output-dir eval
```

Code screenshots are eval-only and must not be added to training manifests.

## Prime GPU runbook and cleanup

Production runs must use a Prime persistent disk. Do not make the only checkpoint copy on a disposable pod root disk.

```bash
prime availability list --gpu-count 8 -o json
prime availability list -o json

DISK_AVAILABILITY_ID=<disk-location-id> \
DISK_SIZE_GB=4000 \
scripts/prime_laguna_vlm_storage.sh

export DISK_ID=<created-disk-id>
POD_AVAILABILITY_ID=<8gpu-id> \
DISK_ID=$DISK_ID \
scripts/prime_laguna_vlm_storage.sh

prime pods create --id <8gpu-id> --name laguna-vlm-train --disk-size 500 --disks <disk-id> -y
ssh -i ~/.ssh/prime_intellect root@<ip> 'echo ok; nvidia-smi -L'
rsync -az -e "ssh -i ~/.ssh/prime_intellect" --exclude .git --exclude /data/ --exclude /checkpoints/ ./ root@<ip>:~/laguna-vision/
ssh -i ~/.ssh/prime_intellect root@<ip> 'cd ~/laguna-vision && bash scripts/pod_bootstrap.sh'
```

Current 8-GPU availability at the last check:

| Option | Availability id | Hourly price | Use |
|---|---:|---:|---|
| 8×B300 262GB spot | `bb2b5a` | `$21/hr` | best budget first attempt; checkpoint often and expect preemption |
| 8×B300 262GB | `a763ff` | `$60/hr` | fastest/largest, best for the real run |
| 8×H200 141GB | `3bda77` | `$32/hr` | strong cheaper fallback |
| 8×A100 80GB | `b22eb0` | `$11.94/hr` | cheapest fallback, low stock |

Expected wall-clock for a first serious Laguna run depends heavily on dataset download speed and feature-cache reuse. Plan roughly **4-8 hours** for disk setup, data download/materialization, and feature caching, then **8-18 hours** for stage 1+2 on 8×B300/H200. Early stage-2 checkpoints should be testable every `SAVE_EVERY=250` optimizer steps, typically after the first few hours of stage 2. Treat these as estimates until the first 1,000-step throughput is measured on the actual pod.

With a **$200 compute budget**, the best first attempt is the 8×B300 spot option when available. At `$21/hr`, $200 buys about 9.5 hours before storage/API overhead, so run a capped milestone instead of pretending it is enough for the full production run:

```bash
export LAGUNA_VLM_ROOT=/mnt/prime/laguna-vlm
export SAVE_EVERY=250

# Stage 1 budget pass: projector alignment on the largest slice that fits the time box.
MAX_ITEMS=300000 timeout 5h scripts/laguna_llava_stage1.sh

# Stage 2 early pass: warm-start from stage 1, train Laguna LoRA, and emit testable checkpoints.
MAX_ITEMS=150000 timeout 3h scripts/laguna_llava_stage2.sh

CHECKPOINT=$LAGUNA_VLM_ROOT/checkpoints/laguna_stage2_instruction/step_000250/projector.pt \
IMAGE=$LAGUNA_VLM_ROOT/eval/real_images/heldout.jpg \
scripts/laguna_llava_early_eval.sh
```

Stop early only if the checkpoint clears frozen held-out gates: real-image VQA improves over text-only, connector beats the untrained-control arm, random web screenshots are plausible, and OCR/document/chart probes improve. One convincing image is a demo; a frozen eval suite is evidence.

The Poolside Hackathon Prime Lab page is about Prime Lab credits and hosted Laguna RL training. It says participants get `$50` Lab credits for Prime Inference, Sandboxes, and On-Demand GPUs, while Hosted Training with Laguna XS.2 is separate from those credits and constrained to one concurrent run with `batch_size <= 128` rollouts. That hosted path uses `verifiers` environments and `prime train`; it is useful for RL/evaluation environments, but it is not a replacement for this supervised VLM connector/LoRA training unless the hosted job can run the multimodal bridge code and image datasets.

Before terminating, verify the persistent disk contains the checkpoints:

```bash
ssh -i ~/.ssh/prime_intellect root@<ip> 'find "$LAGUNA_VLM_ROOT/checkpoints" -maxdepth 3 -name projector.pt -print'
prime pods terminate <pod-id> -y
prime pods list -o json
prime disks list -o json
```

Do not leave the final pod command showing any active pods. Terminating the pod stops compute billing and wipes only the disposable pod disk; the attached Prime disk persists for resume.

## Repository layout

```text
lagunavision/       Python import package; distribution and repo are `laguna-vision`
  backbones/     backbone registry (Laguna XS.2, HF causal LM) and visual embedding injection
  data/          public dataset registry, materialization, manifest loading
  encoders/      PIL and Hugging Face vision encoders
  eval/          scoring, web probes, scene probes, leakage-controlled ablation
  positions/     normalized 2D tile features
  projectors/    MLP and fixed-token resampler connectors
  tiling/        dynamic AnyRes grid selection
  train/         data-parallel visual bridge training loop and batching
scripts/         torchrun launcher for multi-GPU training
```
