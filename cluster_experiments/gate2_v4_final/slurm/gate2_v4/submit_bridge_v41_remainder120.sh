#!/bin/bash
# Submit only the 120 missing full-condition cells after accepted Pilot30.

set -euo pipefail

cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"

PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
SOURCE_LOCK="${SOURCE_LOCK:?SOURCE_LOCK is required}"
RESULTS_ROOT="${RESULTS_ROOT:-results/gate2_v4_bridge_standard_v41_remainder120}"
ACCOUNT="${ACCOUNT:-anon_account}"
PARTITION="${PARTITION:-32gpus}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
SUBMIT_MODE="${SUBMIT_MODE:-test-only}"
CONFIRM_REMAINDER120="${CONFIRM_REMAINDER120:-NO}"
ALLOW_RESUME="${ALLOW_RESUME:-NO}"
MAX_WORKERS="${MAX_WORKERS:-32}"
STEP_MEM="${STEP_MEM:-180G}"
MATRIX="gate2_v4_final/slurm/gate2_v4/bridge_v4_parallel_matrix.slurm"

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 2; }
[[ -f "$SOURCE_LOCK" ]] || { echo "Missing source lock: $SOURCE_LOCK" >&2; exit 2; }
[[ -f "$MATRIX" ]] || { echo "Missing matrix launcher: $MATRIX" >&2; exit 2; }
[[ "$SUBMIT_MODE" == test-only || "$SUBMIT_MODE" == submit ]] || {
  echo "SUBMIT_MODE must be test-only or submit" >&2
  exit 2
}
[[ "$MAX_WORKERS" =~ ^[1-9][0-9]*$ && "$MAX_WORKERS" -le 32 ]] || {
  echo "MAX_WORKERS must be in [1, 32]" >&2
  exit 2
}

mkdir -p "$RESULTS_ROOT" logs/gate2_v4_bridge
LOCK_SHA=$("$PYTHON" - "$SOURCE_LOCK" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["source_lock_sha256"])
PY
)

EXISTING=$(find "$RESULTS_ROOT" -name bridge_v4.json -type f | wc -l)
PARTIALS=$(find "$RESULTS_ROOT" -name 'bridge_v4.json.partial' -type f | wc -l)
[[ "$PARTIALS" -eq 0 ]] || {
  echo "STOP: partial bridge files require inspection before resume: $PARTIALS" >&2
  exit 2
}
if [[ "$EXISTING" -gt 0 ]]; then
  [[ "$ALLOW_RESUME" == YES ]] || {
    echo "STOP: result root contains $EXISTING finals; set ALLOW_RESUME=YES only to resume the same locked panel" >&2
    exit 2
  }
  [[ "$EXISTING" -le 120 ]] || {
    echo "STOP: result root contains more than 120 finals: $EXISTING" >&2
    exit 2
  }
  "$PYTHON" - "$RESULTS_ROOT" "$LOCK_SHA" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = sys.argv[2]
bad = []
paths = list(root.rglob("bridge_v4.json"))
for path in paths:
    try:
        found = json.loads(path.read_text(encoding="utf-8")).get("source_lock_sha256")
    except Exception as exc:
        bad.append(f"{path}:unreadable:{exc}")
        continue
    if found != expected:
        bad.append(f"{path}:lock={found}")
if bad:
    print("STOP: resume root contains results from another or unreadable lock", file=sys.stderr)
    print("\n".join(bad[:20]), file=sys.stderr)
    raise SystemExit(2)
print(f"RESUME_LOCK_AUDIT=PASS existing={len(paths)}")
PY
fi

echo "Gate 2 v4.1 full-factorial remainder"
echo "  mode:        $SUBMIT_MODE"
echo "  account:     $ACCOUNT"
echo "  partition:   $PARTITION"
echo "  tasks:       120 (exact complement of accepted full24)"
echo "  workers:     $MAX_WORKERS"
echo "  existing:    $EXISTING (resume=$ALLOW_RESUME)"
echo "  result root: $RESULTS_ROOT"
echo "  source lock: $SOURCE_LOCK"
echo "  lock sha:    $LOCK_SHA"
echo "  eval N:      forget=128 retain=128 calibration=128"

SBATCH_ARGS=(
  --partition="$PARTITION"
  --account="$ACCOUNT"
  --nodes=4
  --ntasks-per-node=1
  --gpus-per-node=8
  --cpus-per-task=64
  --mem=1600G
  --time="$TIME_LIMIT"
  --job-name=bridge_v41_rem120
  --output=logs/gate2_v4_bridge/bridge_v41_rem120_%j.out
  --error=logs/gate2_v4_bridge/bridge_v41_rem120_%j.err
  --export="ALL,PYTHON=$PYTHON,SOURCE_LOCK=$SOURCE_LOCK,RESULTS_ROOT=$RESULTS_ROOT,MATRIX_SCOPE=remainder120,MAX_WORKERS=$MAX_WORKERS,STEP_MEM=$STEP_MEM,N_SAMPLES=128,N_RETAIN=128,N_CALIBRATION=128,MANIFEST_N_SAMPLES=128,MANIFEST_N_RETAIN=128,MANIFEST_N_CALIBRATION=128"
)

if [[ "$SUBMIT_MODE" == test-only ]]; then
  sbatch --test-only "${SBATCH_ARGS[@]}" "$MATRIX"
  exit 0
fi

[[ "$CONFIRM_REMAINDER120" == YES ]] || {
  echo "STOP: set CONFIRM_REMAINDER120=YES for a real 32-GPU submission" >&2
  exit 2
}

JOB=$(sbatch --parsable "${SBATCH_ARGS[@]}" "$MATRIX")
JOB="${JOB%%;*}"
printf '%s\n' "$JOB" > bridge_v41_remainder120_jobid.txt
printf '%s\n' "$SOURCE_LOCK" > bridge_v41_remainder120_source_lock.txt
echo "REMAINDER120_JOB=$JOB"
