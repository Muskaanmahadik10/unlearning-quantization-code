#!/bin/bash
# Submit the predeclared 12-cell Gate 2 v4.1 diversity-confirmation panel.

set -euo pipefail

cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"

PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
SOURCE_LOCK="${SOURCE_LOCK:?SOURCE_LOCK is required}"
RESULTS_ROOT="${RESULTS_ROOT:-results/gate2_v4_bridge_standard_v41_diversity12}"
ACCOUNT="${ACCOUNT:-anon_account}"
PARTITION="${PARTITION:-16gpus}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
SUBMIT_MODE="${SUBMIT_MODE:-test-only}"
CONFIRM_DIVERSITY12="${CONFIRM_DIVERSITY12:-NO}"
MATRIX="gate2_v4_final/slurm/gate2_v4/bridge_v4_parallel_matrix.slurm"

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 2; }
[[ -f "$SOURCE_LOCK" ]] || { echo "Missing source lock: $SOURCE_LOCK" >&2; exit 2; }
[[ -f "$MATRIX" ]] || { echo "Missing matrix launcher: $MATRIX" >&2; exit 2; }
[[ "$SUBMIT_MODE" == test-only || "$SUBMIT_MODE" == submit ]] || {
  echo "SUBMIT_MODE must be test-only or submit" >&2
  exit 2
}

if find "$RESULTS_ROOT" -name bridge_v4.json -type f -print -quit 2>/dev/null | grep -q .; then
  echo "STOP: result root already contains bridge_v4.json: $RESULTS_ROOT" >&2
  exit 2
fi

LOCK_SHA=$("$PYTHON" - "$SOURCE_LOCK" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["source_lock_sha256"])
PY
)

mkdir -p logs/gate2_v4_bridge

echo "Gate 2 v4.1 diversity confirmation"
echo "  mode:        $SUBMIT_MODE"
echo "  account:     $ACCOUNT"
echo "  partition:   $PARTITION"
echo "  tasks:       12"
echo "  workers:     12"
echo "  result root: $RESULTS_ROOT"
echo "  source lock: $SOURCE_LOCK"
echo "  lock sha:    $LOCK_SHA"
echo "  eval N:      forget=128 retain=128 calibration=128"

SBATCH_ARGS=(
  --partition="$PARTITION"
  --account="$ACCOUNT"
  --nodes=2
  --ntasks-per-node=1
  --gpus-per-node=8
  --cpus-per-task=64
  --mem=1600G
  --time="$TIME_LIMIT"
  --job-name=bridge_v41_div12
  --output=logs/gate2_v4_bridge/bridge_v41_div12_%j.out
  --error=logs/gate2_v4_bridge/bridge_v41_div12_%j.err
  --export="ALL,PYTHON=$PYTHON,SOURCE_LOCK=$SOURCE_LOCK,RESULTS_ROOT=$RESULTS_ROOT,MATRIX_SCOPE=diversity12,MAX_WORKERS=12,STEP_MEM=180G,N_SAMPLES=128,N_RETAIN=128,N_CALIBRATION=128,MANIFEST_N_SAMPLES=128,MANIFEST_N_RETAIN=128,MANIFEST_N_CALIBRATION=128"
)

if [[ "$SUBMIT_MODE" == test-only ]]; then
  sbatch --test-only "${SBATCH_ARGS[@]}" "$MATRIX"
  exit 0
fi

[[ "$CONFIRM_DIVERSITY12" == YES ]] || {
  echo "STOP: set CONFIRM_DIVERSITY12=YES for a real submission" >&2
  exit 2
}

JOB=$(sbatch --parsable "${SBATCH_ARGS[@]}" "$MATRIX")
JOB="${JOB%%;*}"
printf '%s\n' "$JOB" > bridge_v41_diversity12_jobid.txt
echo "DIVERSITY12_JOB=$JOB"
