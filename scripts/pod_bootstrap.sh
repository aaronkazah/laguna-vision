#!/usr/bin/env bash
# Bootstrap a fresh CUDA pod for Laguna Vision training.
# Creates an isolated venv, installs the training + data extras, and reports
# the visible GPU topology so the caller can size batch/grad-accum before
# launching the distributed run.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! python3 - <<'PY' >/dev/null 2>&1
import ensurepip
import venv
PY
then
  if command -v apt-get >/dev/null && [ "$(id -u)" -eq 0 ]; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip
  else
    echo "python3-venv is required; install it, then rerun this script." >&2
    exit 1
  fi
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip wheel >/dev/null
python -m pip install -e '.[llama,data,publish]' hf_transfer

export HF_HUB_ENABLE_HF_TRANSFER=1

python - <<'PY'
import torch

count = torch.cuda.device_count()
print(f"cuda_available={torch.cuda.is_available()} device_count={count}")
if count:
    print(f"device0={torch.cuda.get_device_name(0)} bf16={torch.cuda.is_bf16_supported()}")
    free, total = torch.cuda.mem_get_info(0)
    print(f"gpu0_mem_total_gb={total / 1e9:.1f} free_gb={free / 1e9:.1f}")
PY
