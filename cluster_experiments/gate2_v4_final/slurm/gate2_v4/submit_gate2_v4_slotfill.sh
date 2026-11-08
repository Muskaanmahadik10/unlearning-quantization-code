#!/bin/bash
set -euo pipefail

MODE="${SUBMIT_MODE:-test-only}"
STAGE="${STAGE:?STAGE=gt or upqi required}"
LABEL="${LABEL:?LABEL required: l2_A,l3_A,l2m_A,l3m_A}"
PARTITION="${PARTITION:-dev}"
ACCOUNT="${ACCOUNT:-anon_account}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-200G}"
TIME="${TIME:-04:00:00}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"

case "$LABEL" in
  l2_A)  MODEL_TAG="llama2_7b_adamw";   DATASET="tofu";      CKPT_BASE="${CKPT_BASE:-checkpoints/llama2_7b_adamw}" ;;
  l3_A)  MODEL_TAG="llama3_8b_adamw";   DATASET="tofu";      CKPT_BASE="${CKPT_BASE:-checkpoints/llama3_8b_adamw}" ;;
  l2m_A) MODEL_TAG="llama2_muse_adamw"; DATASET="muse_news"; CKPT_BASE="${CKPT_BASE:-checkpoints/llama2_muse_adamw}" ;;
  l3m_A) MODEL_TAG="llama3_muse_adamw"; DATASET="muse_news"; CKPT_BASE="${CKPT_BASE:-checkpoints/llama3_muse_adamw}" ;;
  *) echo "Unknown LABEL=$LABEL" >&2; exit 2 ;;
esac

SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-42-2024-5000}"
ALGOS_OVERRIDE="${ALGOS_OVERRIDE:-GA GDR NPO KLR}"
QUANTIZERS_OVERRIDE="${QUANTIZERS_OVERRIDE:-nf4 gptq_int4 rtn_int8}"
SLURM="gate2_v4_final/slurm/gate2_v4/${STAGE}_v4_slotfill.slurm"

echo "Gate 2 v4 slotfill"
echo "  mode:       $MODE"
echo "  stage:      $STAGE"
echo "  label:      $LABEL"
echo "  model:      $MODEL_TAG"
echo "  dataset:    $DATASET"
echo "  seeds:      $SEEDS_OVERRIDE"
echo "  algos:      $ALGOS_OVERRIDE"
echo "  quantizers: $QUANTIZERS_OVERRIDE"
echo "  partition:  $PARTITION"
echo "  account:    $ACCOUNT"
echo "  gpus:       $GPUS"
echo "  max-par:    $MAX_PARALLEL"
echo "  time:       $TIME"
echo "  mem:        $MEM"
echo "  slurm:      $SLURM"

SBATCH_ARGS=(
  "--partition=$PARTITION"
  "--account=$ACCOUNT"
  "--gres=gpu:$GPUS"
  "--cpus-per-task=$CPUS"
  "--mem=$MEM"
  "--time=$TIME"
  "--job-name=${LABEL}_${STAGE}_v4"
  "--export=ALL,PYTHON=$PYTHON,LABEL=$LABEL,MODEL_TAG=$MODEL_TAG,DATASET=$DATASET,CKPT_BASE=$CKPT_BASE,SEEDS_OVERRIDE=$SEEDS_OVERRIDE,ALGOS_OVERRIDE=$ALGOS_OVERRIDE,QUANTIZERS_OVERRIDE=$QUANTIZERS_OVERRIDE,MAX_PARALLEL=$MAX_PARALLEL"
)

if [[ "$MODE" == "test-only" ]]; then
  sbatch --test-only "${SBATCH_ARGS[@]}" "$SLURM"
elif [[ "$MODE" == "submit" ]]; then
  sbatch --parsable "${SBATCH_ARGS[@]}" "$SLURM"
else
  echo "SUBMIT_MODE must be test-only or submit" >&2
  exit 2
fi

