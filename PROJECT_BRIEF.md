# Laguna Vision Project Brief

This is the handoff file for reviewers, collaborators, and future agents. It describes what the repository is, what is intentionally not checked in, and what credentials or decisions are needed before production training.

## Repository identity

| Field | Value |
|---|---|
| Public repo | `https://github.com/aaronkazah/laguna-vision` |
| Python distribution | `laguna-vision` |
| CLI | `laguna-vision` (`lagunavision` remains as a compatibility alias) |
| Python import package | `lagunavision` |
| Primary target model | `poolside/Laguna-XS.2` |
| Goal | General image and screenshot understanding for Laguna |

The import package stays `lagunavision` because Python modules cannot contain hyphens. User-facing docs, packaging, and repo naming use `laguna-vision`.

## Current state

The repo contains the production scaffolding for supervised multimodal training:

| Area | Status |
|---|---|
| LLaVA-format data materialization | implemented |
| Public HF VQA/OCR/chart materialization | implemented |
| AnyRes image/screenshot tiling | implemented |
| Vision encoder feature cache | implemented |
| Laguna/Llama/HF causal-LM backbone registry | implemented |
| Connector and LoRA training | implemented |
| DDP training launcher | implemented |
| Early step checkpoints | implemented |
| Five-arm ablation eval | implemented |
| Prime persistent-disk runbook | implemented |
| Private Hugging Face checkpoint publishing | implemented |

The existing local toy checkpoints prove the image path is wired, but they are not a production general VLM. The real run is stage 1 alignment plus stage 2 instruction tuning on LLaVA-style data, followed by OCR/document/chart/screenshot eval.

## What to publish

Publish these to GitHub:

| Include | Reason |
|---|---|
| source code, scripts, tests, docs | lets developers inspect and reproduce the pipeline |
| tiny example JSONL fixtures | safe examples for local tests |
| run commands and config defaults | reproducibility |
| eval summaries and model cards | evidence and limitations |

Do not publish these to GitHub:

| Exclude | Where it belongs |
|---|---|
| raw training data | regenerate from licensed public sources or publish to HF Dataset only if redistribution is allowed |
| model weights/checkpoints | private Hugging Face model repo or GitHub Release only after license/safety review |
| Prime/HF/GitHub tokens | local keychain, `hf auth login`, GitHub Actions secrets, or Prime CLI auth |
| large feature caches | Prime persistent disk or object storage |

## Hugging Face convention

Use Hugging Face Hub for artifacts that are too large or too sensitive for GitHub:

| Artifact | Recommended location |
|---|---|
| early checkpoints | private HF model repo, e.g. `your-org/laguna-vision-early` |
| final checkpoint | public or private HF model repo after model card/license review |
| eval dataset manifests | GitHub if small and license-safe; HF Dataset if large |
| raw images | HF Dataset only if source license allows redistribution |

Login only on the machine that publishes or downloads private checkpoints:

```bash
hf auth login
HF_REPO_ID=your-org/laguna-vision-early \
CHECKPOINT_DIR=/mnt/prime/laguna-vlm/checkpoints/laguna_stage2_instruction/step_000250 \
HF_PRIVATE=1 \
scripts/publish_hf_checkpoint.sh
```

Private checkpoints can be used directly at inference time:

```bash
laguna-vision ask-image \
  --checkpoint hf://your-org/laguna-vision-early/step_000250 \
  --image path/to/image.png \
  --question "Explain this image." \
  --backbone laguna
```

For CI/CD publishing, store `HF_TOKEN` as a GitHub Actions secret. Never commit tokens.

## What is needed from the owner

| Need | Why |
|---|---|
| GitHub repo access to `aaronkazah/laguna-vision` | push the clean project history and enable CI |
| Decision: public vs private repo | determines whether docs mention unreleased model artifacts |
| Hugging Face org/repo names | where early and final checkpoints will live |
| HF write token or local `hf auth login` on the publishing machine | required for uploading private checkpoints; read auth is required for private inference |
| HF model-gated access approvals | required for gated backbones such as Llama and possibly Laguna |
| Prime CLI auth and SSH key | required only when launching GPU pods |
| Prime budget ceiling and spot tolerance | determines whether to use 8xB300 spot, B300 on-demand, or H200 |
| Dataset license decision | determines whether raw data can be mirrored or only regenerated |

## Developer workflow

Developers can run the code locally with tiny data, but production training needs Prime/HF credentials and GPU compute.

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,data,llama]'
laguna-vision --help
python -m pytest -q
```

Production training is intentionally script-driven:

```bash
export LAGUNA_VLM_ROOT=/mnt/prime/laguna-vlm
scripts/laguna_llava_stage1.sh
scripts/laguna_llava_stage2.sh
```

The scripts require a persistent Prime disk. Do not train directly onto a disposable pod root disk.

Run the local smoke before paying for GPUs:

```bash
scripts/local_smoke_train.sh
```

Then start a detached Prime run from a mounted persistent disk:

```bash
PRIME_SSH_TARGET=ubuntu@<pod-ip> \
LAGUNA_VLM_ROOT=/mnt/prime/laguna-vlm \
MAX_RUNTIME=9h \
scripts/prime_start_detached_training.sh
```

This survives laptop disconnects. `MAX_RUNTIME` stops training, but pod billing only stops automatically if the pod can run `prime pods terminate` with `TERMINATE_ON_EXIT=1 PRIME_POD_ID=<pod-id>`.

## Current budget recommendation

For a capped first milestone around `$200`, use 8xB300 spot when available and checkpoint aggressively. At `$21/hr`, it buys roughly 9.5 hours before storage/API overhead. That is enough for a serious early milestone, not a guaranteed complete general VLM.

If spot is unavailable or preemption risk is unacceptable, use 8xH200 at roughly `$32/hr` as the stable cost-conscious fallback. Use 8xB300 on-demand only when speed matters more than spend.

## Definition of done for "general image understanding"

A checkpoint is not considered generally useful because one uploaded image works. It needs frozen eval evidence:

| Gate | Pass condition |
|---|---|
| text prior control | `image_only` beats `text_only` |
| untrained control | trained connector beats random connector |
| natural image VQA | held-out public VQA improves |
| OCR/document/chart | text-rich eval improves |
| screenshot eval | fresh web/developer screenshots are plausible and grounded |
| artifact persistence | checkpoint, LoRA adapter, spec, report, and eval outputs are saved to persistent storage |
