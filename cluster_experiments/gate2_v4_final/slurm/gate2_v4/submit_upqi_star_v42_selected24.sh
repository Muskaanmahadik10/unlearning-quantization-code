#!/bin/bash
# Submit corrected UPQI-star for the preregistered Pilot24 cells at the selected K/N.

set -euo pipefail
cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"
PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
SUBMIT_MODE="${SUBMIT_MODE:-test-only}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:?SELECTION_MANIFEST required}"
CONVERGENCE_DECISION_JSON="${CONVERGENCE_DECISION_JSON:?CONVERGENCE_DECISION_JSON required}"
SOURCE_LOCK="${SOURCE_LOCK:-$(cat gate2_v4_final/metric_locks/CURRENT_METRIC_V42_LOCK.txt)}"
read -r decision k n < <(
  "$PYTHON" - "$CONVERGENCE_DECISION_JSON" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
selected = data.get("selected") or {}
print(data.get("decision", ""), selected.get("k", ""), selected.get("n", ""))
PY
)
[[ "$decision" == UPQI_STAR_SELECT_DIAGONAL_CONFIG || "$decision" == UPQI_STAR_SELECT_FULLGRID_CONFIG ]] || {
  echo "Convergence gate has not selected a production K/N: $decision" >&2; exit 2;
}
[[ -n "$k" && -n "$n" ]] || exit 2
args=(
  --partition="${PARTITION:-32gpus}" --account="${ACCOUNT:-anon_account}" --nodes=4 --ntasks=4 --gpus-per-node=8
  --cpus-per-task=64 --mem=1600G --time="${TIME:-06:00:00}" --job-name=upqi_v42_selected24
  --output=logs/gate2_v4_upqi_star/upqi_v42_selected24_%j.out
  --error=logs/gate2_v4_upqi_star/upqi_v42_selected24_%j.err
  --export="ALL,SOURCE_LOCK=$SOURCE_LOCK,SELECTION_MANIFEST=$SELECTION_MANIFEST,N_QUADRATURE=$k,N_SAMPLES=$n,MAX_WORKERS=24,STEP_MEM=200G,RESULTS_ROOT=${RESULTS_ROOT:-results/gate2_v4_standard_upqi_star_v42_selected24}"
)
script=gate2_v4_final/slurm/gate2_v4/upqi_star_v42_selection_matrix.slurm
echo "Selected24 UPQI-star mode=$SUBMIT_MODE K=$k N=$n"
if [[ "$SUBMIT_MODE" == test-only ]]; then sbatch --test-only "${args[@]}" "$script"; else sbatch --parsable "${args[@]}" "$script"; fi
