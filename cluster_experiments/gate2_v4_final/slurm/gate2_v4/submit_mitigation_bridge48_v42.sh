#!/bin/bash
# Submit cost-capped 32-GPU Bridge48 confirmation.

set -euo pipefail
cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"
SUBMIT_MODE="${SUBMIT_MODE:-test-only}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:?SELECTION_MANIFEST required}"
RANKINGS_PATH="${RANKINGS_PATH:?RANKINGS_PATH required}"
SOURCE_LOCK="${SOURCE_LOCK:?SOURCE_LOCK required}"
RESULTS_ROOT="${RESULTS_ROOT:-results/gate2_v4_bridge_standard_v42_mitigation_bridge48_gqdp_k8}"
args=(
  --partition="${PARTITION:-32gpus}" --account="${ACCOUNT:-anon_account}"
  --nodes=4 --ntasks=4 --gpus-per-node=8 --cpus-per-task=64 --mem=1600G
  --time="${TIME:-01:30:00}" --job-name=mit_bridge48_gqdp_k8
  --output=logs/gate2_v4_bridge/mit_bridge48_gqdp_k8_%j.out
  --error=logs/gate2_v4_bridge/mit_bridge48_gqdp_k8_%j.err
  --export="ALL,SOURCE_LOCK=$SOURCE_LOCK,SELECTION_MANIFEST=$SELECTION_MANIFEST,RANKINGS_PATH=$RANKINGS_PATH,RESULTS_ROOT=$RESULTS_ROOT,MAX_WORKERS=32,STEP_MEM=200G"
)
script=gate2_v4_final/slurm/gate2_v4/mitigation_bridge48_v42_parallel_matrix.slurm
if [[ "$SUBMIT_MODE" == test-only ]]; then
  sbatch --test-only "${args[@]}" "$script"
else
  sbatch --parsable "${args[@]}" "$script"
fi
