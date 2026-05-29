#!/usr/bin/env bash
# Launch N-way data-parallel visual-bridge training under torchrun.
#
# The frozen backbone is replicated per GPU; only trainable adapter gradients are
# averaged across ranks. Set NPROC to the number of visible GPUs.
#
# All arguments after the script name pass straight through to
# `laguna-vision train-visual-bridge`, e.g.:
#
#   NPROC=8 scripts/train_visual_bridge_ddp.sh \
#     --manifest data/hf_vqa/train.jsonl \
#     --eval-manifest data/hf_vqa/eval.jsonl \
#     --output-dir checkpoints/laguna_vqa \
#     --backbone laguna \
#     --encoder hf --encoder-id google/siglip-so400m-patch14-384 \
#     --projector resampler --visual-tokens 256 --max-tiles 1 \
#     --batch-size 8 --grad-accum 4 --epochs 1 \
#     --device cuda --vision-device cuda
set -euo pipefail

NPROC="${NPROC:-8}"

exec torchrun \
  --standalone \
  --nproc-per-node "${NPROC}" \
  -m lagunavision.cli train-visual-bridge "$@"
