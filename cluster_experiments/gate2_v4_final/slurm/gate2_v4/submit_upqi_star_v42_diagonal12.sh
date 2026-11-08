#!/bin/bash
# Submit or scheduler-test the 12-run corrected-UPQI convergence diagonal.

set -euo pipefail
cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"
SUBMIT_MODE="${SUBMIT_MODE:-test-only}"
ACCOUNT="${ACCOUNT:-anon_account}"
PARTITION="${PARTITION:-16gpus}"
SOURCE_LOCK="${SOURCE_LOCK:-$(cat gate2_v4_final/metric_locks/CURRENT_METRIC_V42_LOCK.txt)}"
[[ -f "$SOURCE_LOCK" ]] || { echo "Missing source lock: $SOURCE_LOCK" >&2; exit 2; }
[[ "$SUBMIT_MODE" == test-only || "$SUBMIT_MODE" == submit ]] || exit 2
mkdir -p logs/gate2_v4_upqi_star

args=(
  --partition="$PARTITION" --account="$ACCOUNT" --nodes=2 --ntasks=2 --gpus-per-node=8
  --cpus-per-task=64 --mem=1600G --time="${TIME:-04:00:00}" --job-name=upqi_v42_diag12
  --output=logs/gate2_v4_upqi_star/upqi_v42_diag12_%j.out
  --error=logs/gate2_v4_upqi_star/upqi_v42_diag12_%j.err
  --export="ALL,SOURCE_LOCK=$SOURCE_LOCK,SCOPE=diagonal12,MAX_WORKERS=12,STEP_MEM=200G,RESULTS_ROOT=${RESULTS_ROOT:-results/gate2_v4_standard_upqi_star_v42_diagonal12}"
)
script=gate2_v4_final/slurm/gate2_v4/upqi_star_v42_convergence_matrix.slurm
echo "UPQI-star v4.2 diagonal12 mode=$SUBMIT_MODE partition=$PARTITION"
if [[ "$SUBMIT_MODE" == test-only ]]; then
  sbatch --test-only "${args[@]}" "$script"
else
  sbatch --parsable "${args[@]}" "$script"
fi
