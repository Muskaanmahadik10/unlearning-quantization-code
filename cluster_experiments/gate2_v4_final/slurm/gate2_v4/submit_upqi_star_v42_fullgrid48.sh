#!/bin/bash
# Submit the 48-run K/N grid only when the diagonal cannot select a cheaper configuration.

set -euo pipefail
cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"
PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
SUBMIT_MODE="${SUBMIT_MODE:-test-only}"
DIAGONAL_DECISION_JSON="${DIAGONAL_DECISION_JSON:?DIAGONAL_DECISION_JSON required}"
SOURCE_LOCK="${SOURCE_LOCK:-$(cat gate2_v4_final/metric_locks/CURRENT_METRIC_V42_LOCK.txt)}"
decision="$($PYTHON -c "import json; print(json.load(open('$DIAGONAL_DECISION_JSON'))['decision'])")"
[[ "$decision" == UPQI_STAR_REFERENCE_ONLY_PASS_RUN_FULLGRID48 || "$decision" == UPQI_STAR_DIAGONAL_INCONCLUSIVE_RUN_FULLGRID48 ]] || {
  echo "Full grid not authorized by diagonal decision: $decision" >&2; exit 2;
}
args=(
  --partition="${PARTITION:-64gpus}" --account="${ACCOUNT:-anon_account}" --nodes=8 --ntasks=8 --gpus-per-node=8
  --cpus-per-task=64 --mem=1600G --time="${TIME:-05:00:00}" --job-name=upqi_v42_grid48
  --output=logs/gate2_v4_upqi_star/upqi_v42_grid48_%j.out
  --error=logs/gate2_v4_upqi_star/upqi_v42_grid48_%j.err
  --export="ALL,SOURCE_LOCK=$SOURCE_LOCK,SCOPE=fullgrid48,MAX_WORKERS=48,STEP_MEM=200G,RESULTS_ROOT=${RESULTS_ROOT:-results/gate2_v4_standard_upqi_star_v42_fullgrid48}"
)
script=gate2_v4_final/slurm/gate2_v4/upqi_star_v42_convergence_matrix.slurm
if [[ "$SUBMIT_MODE" == test-only ]]; then sbatch --test-only "${args[@]}" "$script"; else sbatch --parsable "${args[@]}" "$script"; fi
