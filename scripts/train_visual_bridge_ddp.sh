#!/usr/bin/env bash
# Launch N-way data-parallel visual-bridge training under torchrun.
#
# The frozen backbone is replicated per GPU; only the small connector's
# gradients are averaged across ranks, so the expensive frozen forward scales
# with GPU count. Set NPROC to the number of visible GPUs (default 8).
#
# All arguments after the script name pass straight through to
# `laguna-vision train-visual-bridge`, e.g.:
#
#   NPROC=8 scripts/train_visual_bridge_ddp.sh \
#     --manifest data/hf_vqa/train.jsonl \
#     --eval-manifest data/hf_vqa/eval.jsonl \
#     --output-dir checkpoints/laguna_vqa \
#     --backbone laguna \
#     --encoder hf --encoder-id openai/clip-vit-base-patch32 \
#     --projector resampler --visual-tokens 64 --max-tiles 9 \
#     --batch-size 8 --grad-accum 4 --epochs 1 \
#     --device cuda --vision-device cuda
set -euo pipefail

NPROC="${NPROC:-8}"

exec torchrun \
  --standalone \
  --nproc-per-node "${NPROC}" \
  -m lagunavision.cli train-visual-bridge "$@"
