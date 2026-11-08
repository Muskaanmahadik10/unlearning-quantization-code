#!/bin/bash
# Submit the budget-capped disjoint Bridge48 confirmation.

set -euo pipefail
cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"
SUBMIT_MODE="${SUBMIT_MODE:-test-only}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:?SELECTION_MANIFEST required}"
RANKINGS_PATH="${RANKINGS_PATH:?RANKINGS_PATH required}"
SOURCE_LOCK="${SOURCE_LOCK:?SOURCE_LOCK required}"
RESULTS_ROOT="${RESULTS_ROOT:-results/gate2_v4_bridge_standard_v42_mitigation_bridge48_confirmation}"
mkdir -p logs/gate2_v4_bridge

[[ -f "$SELECTION_MANIFEST" && -f "$RANKINGS_PATH" && -f "$SOURCE_LOCK" ]]
existing=$(find "$RESULTS_ROOT" -name bridge_v4.json -type f 2>/dev/null | wc -l)
partials=$(find "$RESULTS_ROOT" -name 'bridge_v4.json.partial' -type f 2>/dev/null | wc -l)
if [[ "$existing" -ne 0 || "$partials" -ne 0 ]]; then
  echo "STOP: Bridge48 output root is not empty finals=$existing partials=$partials" >&2
  exit 2
fi

args=(
  --partition="${PARTITION:-64gpus}" --account="${ACCOUNT:-anon_account}"
  --nodes=8 --ntasks=8 --gpus-per-node=8 --cpus-per-task=64 --mem=1600G
  --time="${TIME:-01:30:00}" --job-name=mitigation_v42_bridge48
  --output=logs/gate2_v4_bridge/mitigation_v42_bridge48_%j.out
  --error=logs/gate2_v4_bridge/mitigation_v42_bridge48_%j.err
  --export="ALL,SOURCE_LOCK=$SOURCE_LOCK,SELECTION_MANIFEST=$SELECTION_MANIFEST,RANKINGS_PATH=$RANKINGS_PATH,MAX_WORKERS=64,STEP_MEM=200G,RESULTS_ROOT=$RESULTS_ROOT"
)
script=gate2_v4_final/slurm/gate2_v4/mitigation_bridge48_confirmation_v42_parallel_matrix.slurm
echo "Bridge48 confirmation mode=$SUBMIT_MODE tasks=96 time=${TIME:-01:30:00} results=$RESULTS_ROOT"
if [[ "$SUBMIT_MODE" == test-only ]]; then
  sbatch --test-only "${args[@]}" "$script"
else
  sbatch --parsable "${args[@]}" "$script"
fi
