#!/usr/bin/env bash
# Billable Prime helper for the real Laguna VLM run. It creates/reuses a
# persistent disk and prints the pod-create command that attaches it.
set -euo pipefail

DISK_NAME="${DISK_NAME:-laguna-vlm-data}"
DISK_SIZE_GB="${DISK_SIZE_GB:-4000}"
POD_NAME="${POD_NAME:-laguna-vlm-train}"

if [[ -z "${DISK_ID:-}" ]]; then
  : "${DISK_AVAILABILITY_ID:?Set DISK_AVAILABILITY_ID from 'prime availability list -o json' for the disk location.}"
  echo "Creating persistent Prime disk ${DISK_NAME} (${DISK_SIZE_GB} GB)..." >&2
  prime disks create --id "${DISK_AVAILABILITY_ID}" --size "${DISK_SIZE_GB}" --name "${DISK_NAME}" -y
  echo "Run 'prime disks list -o json' and export DISK_ID to the new disk id, then rerun this script." >&2
  exit 0
fi

: "${POD_AVAILABILITY_ID:?Set POD_AVAILABILITY_ID to the selected 8-GPU availability id.}"

echo "Create the training pod with the persistent disk attached:"
echo "prime pods create --id ${POD_AVAILABILITY_ID} --name ${POD_NAME} --disk-size ${POD_DISK_SIZE_GB:-500} --disks ${DISK_ID} -y"
echo
echo "After SSH, set LAGUNA_VLM_ROOT to the mounted persistent disk path before running training scripts."
