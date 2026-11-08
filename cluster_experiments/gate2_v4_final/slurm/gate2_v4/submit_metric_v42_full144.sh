#!/bin/bash
# Submit Full144 exact metrics only after the wallet-derived cost gate passes.

set -euo pipefail
cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"
PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
SUBMIT_MODE="${SUBMIT_MODE:-test-only}"
ACCOUNT="${ACCOUNT:-anon_account}"
PARTITION="${PARTITION:-64gpus}"
SOURCE_LOCK="${SOURCE_LOCK:-$(cat gate2_v4_final/metric_locks/CURRENT_METRIC_V42_LOCK.txt)}"
COST_DECISION_JSON="${COST_DECISION_JSON:?COST_DECISION_JSON required}"
[[ -f "$SOURCE_LOCK" && -f "$COST_DECISION_JSON" ]] || exit 2
[[ "$SUBMIT_MODE" == test-only || "$SUBMIT_MODE" == submit ]] || exit 2

read -r COST_DECISION COST_FF COST_FR < <(
  "$PYTHON" - "$COST_DECISION_JSON" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data.get("decision", ""), data.get("target_fisher_forget", ""), data.get("target_fisher_retain", ""))
PY
)
[[ "$COST_DECISION" == FULL144_METRICS_WITHIN_BUDGET ]] || {
  echo "Full144 cost gate not passed: $COST_DECISION" >&2
  exit 2
}
mkdir -p logs/gate2_v4_metrics
args=(
  --partition="$PARTITION" --account="$ACCOUNT" --nodes=8 --ntasks=8 --gpus-per-node=8
  --cpus-per-task=64 --mem=1600G --time="${TIME:-12:00:00}" --job-name=metric_v42_full144
  --output=logs/gate2_v4_metrics/metric_v42_full144_%j.out
  --error=logs/gate2_v4_metrics/metric_v42_full144_%j.err
  --export="ALL,SOURCE_LOCK=$SOURCE_LOCK,MATRIX_SCOPE=full144,MAX_WORKERS=64,STEP_MEM=200G,RESULTS_ROOT=${RESULTS_ROOT:-results/gate2_v4_standard_metrics_v42_full144},FISHER_FORGET_SAMPLES=$COST_FF,FISHER_RETAIN_SAMPLES=$COST_FR"
)
script=gate2_v4_final/slurm/gate2_v4/metric_suite_v42_parallel_matrix.slurm
echo "Full144 metrics mode=$SUBMIT_MODE fisher=$COST_FF/$COST_FR cost_gate=$COST_DECISION"
if [[ "$SUBMIT_MODE" == test-only ]]; then
  sbatch --test-only "${args[@]}" "$script"
else
  sbatch --parsable "${args[@]}" "$script"
fi
