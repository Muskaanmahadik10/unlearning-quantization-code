#!/bin/bash
set -euo pipefail

MODE="${SUBMIT_MODE:-test-only}"
LABEL="${LABEL:?LABEL required: l2_A,l3_A,l2m_A,l3m_A}"
PARTITION="${PARTITION:-dev}"
ACCOUNT="${ACCOUNT:-anon_account}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-200G}"
TIME="${TIME:-04:00:00}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
SOURCE_LOCK="${SOURCE_LOCK:-gate2_v4_final/V4_SOURCE_LOCK.json}"
RESULTS_ROOT="${RESULTS_ROOT:-results/gate2_v4_bridge_standard}"
RANKINGS_PATH="${RANKINGS_PATH:-results/gate2_v4_bridge_standard/bridge_rankings_v4.jsonl}"
N_SAMPLES="${N_SAMPLES:-128}"
N_RETAIN="${N_RETAIN:-128}"
N_CALIBRATION="${N_CALIBRATION:-128}"
MANIFEST_N_SAMPLES="${MANIFEST_N_SAMPLES:-128}"
MANIFEST_N_RETAIN="${MANIFEST_N_RETAIN:-128}"
MANIFEST_N_CALIBRATION="${MANIFEST_N_CALIBRATION:-128}"
CONDITIONS_OVERRIDE="${CONDITIONS_OVERRIDE:-bf16 full_quantized}"
RANKING_SOURCES_OVERRIDE="${RANKING_SOURCES_OVERRIDE:-UPQI_abs random}"
K_VALUES_OVERRIDE="${K_VALUES_OVERRIDE:-1 2 4 8 16}"

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
SLURM="gate2_v4_final/slurm/gate2_v4/bridge_v4_slotfill.slurm"

echo "Gate 2 v4 bridge slotfill"
echo "  mode:       $MODE"
echo "  label:      $LABEL"
echo "  model:      $MODEL_TAG"
echo "  dataset:    $DATASET"
echo "  seeds:      $SEEDS_OVERRIDE"
echo "  algos:      $ALGOS_OVERRIDE"
echo "  quantizers: $QUANTIZERS_OVERRIDE"
echo "  conditions: $CONDITIONS_OVERRIDE"
echo "  rank srcs:  $RANKING_SOURCES_OVERRIDE"
echo "  k values:   $K_VALUES_OVERRIDE"
echo "  partition:  $PARTITION"
echo "  account:    $ACCOUNT"
echo "  gpus:       $GPUS"
echo "  max-par:    $MAX_PARALLEL"
echo "  time:       $TIME"
echo "  mem:        $MEM"
echo "  source:     $SOURCE_LOCK"
echo "  results:    $RESULTS_ROOT"
echo "  rankings:   $RANKINGS_PATH"
echo "  eval N:     forget=$N_SAMPLES retain=$N_RETAIN calib=$N_CALIBRATION"
echo "  manifest N: forget=$MANIFEST_N_SAMPLES retain=$MANIFEST_N_RETAIN calib=$MANIFEST_N_CALIBRATION"
echo "  slurm:      $SLURM"

SBATCH_ARGS=(
  "--partition=$PARTITION"
  "--account=$ACCOUNT"
  "--gres=gpu:$GPUS"
  "--cpus-per-task=$CPUS"
  "--mem=$MEM"
  "--time=$TIME"
  "--job-name=${LABEL}_bridge_v4"
  "--export=ALL,PYTHON=$PYTHON,SOURCE_LOCK=$SOURCE_LOCK,RESULTS_ROOT=$RESULTS_ROOT,RANKINGS_PATH=$RANKINGS_PATH,LABEL=$LABEL,MODEL_TAG=$MODEL_TAG,DATASET=$DATASET,CKPT_BASE=$CKPT_BASE,SEEDS_OVERRIDE=$SEEDS_OVERRIDE,ALGOS_OVERRIDE=$ALGOS_OVERRIDE,QUANTIZERS_OVERRIDE=$QUANTIZERS_OVERRIDE,MAX_PARALLEL=$MAX_PARALLEL,CONDITIONS_OVERRIDE=$CONDITIONS_OVERRIDE,RANKING_SOURCES_OVERRIDE=$RANKING_SOURCES_OVERRIDE,K_VALUES_OVERRIDE=$K_VALUES_OVERRIDE,N_SAMPLES=$N_SAMPLES,N_RETAIN=$N_RETAIN,N_CALIBRATION=$N_CALIBRATION,MANIFEST_N_SAMPLES=$MANIFEST_N_SAMPLES,MANIFEST_N_RETAIN=$MANIFEST_N_RETAIN,MANIFEST_N_CALIBRATION=$MANIFEST_N_CALIBRATION"
)

if [[ "$MODE" == "test-only" ]]; then
  sbatch --test-only "${SBATCH_ARGS[@]}" "$SLURM"
elif [[ "$MODE" == "submit" ]]; then
  sbatch --parsable "${SBATCH_ARGS[@]}" "$SLURM"
else
  echo "SUBMIT_MODE must be test-only or submit" >&2
  exit 2
fi
