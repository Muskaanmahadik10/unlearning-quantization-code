#!/bin/bash
set -euo pipefail

case "${LABEL:?}" in
  l2_A)  export MODEL_TAG="llama2_7b_adamw";   export DATASET="tofu";      export CKPT_BASE="${CKPT_BASE:-checkpoints/llama2_7b_adamw}" ;;
  l3_A)  export MODEL_TAG="llama3_8b_adamw";   export DATASET="tofu";      export CKPT_BASE="${CKPT_BASE:-checkpoints/llama3_8b_adamw}" ;;
  l2m_A) export MODEL_TAG="llama2_muse_adamw"; export DATASET="muse_news"; export CKPT_BASE="${CKPT_BASE:-checkpoints/llama2_muse_adamw}" ;;
  l3m_A) export MODEL_TAG="llama3_muse_adamw"; export DATASET="muse_news"; export CKPT_BASE="${CKPT_BASE:-checkpoints/llama3_muse_adamw}" ;;
  *) echo "Unknown LABEL=$LABEL" >&2; exit 2 ;;
esac

if [[ "${STAGE:?}" == "gt" ]]; then
  bash gate2_v4_final/slurm/gate2_v4/gt_v4_slotfill.slurm
elif [[ "$STAGE" == "upqi" ]]; then
  bash gate2_v4_final/slurm/gate2_v4/upqi_v4_slotfill.slurm
else
  echo "Unknown STAGE=$STAGE" >&2
  exit 2
fi

