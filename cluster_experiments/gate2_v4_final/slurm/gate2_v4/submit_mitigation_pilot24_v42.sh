#!/bin/bash
# Submit the 192-condition Mitigation Pilot24 only after rankings are complete.

set -euo pipefail
cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"
PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
SUBMIT_MODE="${SUBMIT_MODE:-test-only}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:?SELECTION_MANIFEST required}"
RANKINGS_PATH="${RANKINGS_PATH:?RANKINGS_PATH required}"
SOURCE_LOCK="${SOURCE_LOCK:-$(cat gate2_v4_final/metric_locks/CURRENT_METRIC_V42_LOCK.txt)}"
args=(
  --partition="${PARTITION:-64gpus}" --account="${ACCOUNT:-anon_account}" --nodes=8 --ntasks=8 --gpus-per-node=8
  --cpus-per-task=64 --mem=1600G --time="${TIME:-08:00:00}" --job-name=mitigation_v42_pilot24
  --output=logs/gate2_v4_bridge/mitigation_v42_pilot24_%j.out
  --error=logs/gate2_v4_bridge/mitigation_v42_pilot24_%j.err
  --export="ALL,SOURCE_LOCK=$SOURCE_LOCK,SELECTION_MANIFEST=$SELECTION_MANIFEST,RANKINGS_PATH=$RANKINGS_PATH,MAX_WORKERS=64,STEP_MEM=200G,RESULTS_ROOT=${RESULTS_ROOT:-results/gate2_v4_bridge_standard_v42_mitigation_pilot24}"
)
script=gate2_v4_final/slurm/gate2_v4/mitigation_pilot24_v42_parallel_matrix.slurm
if [[ "$SUBMIT_MODE" == test-only ]]; then sbatch --test-only "${args[@]}" "$script"; else sbatch --parsable "${args[@]}" "$script"; fi
