#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/work/anon_user/hessian_unlearning}"
PY="${PY:-/work/anon_user/.runtime/envs/unlearn-nano5/bin/python}"
MODE="${SUBMIT_MODE:-test-only}"
STAMP="$(date +%Y%m%d_%H%M%S)"
COMPAT="${COMPAT_WORKDIR:-$ROOT/_b4_gt_resume_compat_$STAMP}"
LOCK_REL="gate2_v4_final/V4_SOURCE_LOCK.json"
EXPECTED_LOCK="b4caaae79a48523b2166a0af4409e78508700a74fc03ee25b36a62bfb461aa7b"
LOCKED_VALIDATORS="$ROOT/gate2_v4_final/b4_resume_compat"

[[ -x "$PY" ]] || { echo "Missing Nano5 Python: $PY" >&2; exit 2; }
for name in audit_icml_core_144_v4_status.py validate_gate2_v4_cell_join.py validate_gt_v4.py; do
  [[ -f "$LOCKED_VALIDATORS/$name" ]] || { echo "Missing locked validator: $LOCKED_VALIDATORS/$name" >&2; exit 2; }
done

mkdir -p "$COMPAT"
cp -a "$ROOT/gate2_v4_final" "$COMPAT/gate2_v4_final"
cp -a "$LOCKED_VALIDATORS/"*.py "$COMPAT/gate2_v4_final/scripts/"

ln -s "$ROOT/checkpoints" "$COMPAT/checkpoints"
ln -s "$ROOT/data" "$COMPAT/data"
ln -s "$ROOT/manifests" "$COMPAT/manifests"
mkdir -p "$COMPAT/results" "$COMPAT/logs"
ln -s "$ROOT/results/gate2_v4_standard" "$COMPAT/results/gate2_v4_standard"
ln -s "$ROOT/logs/gate2_v4" "$COMPAT/logs/gate2_v4"

cd "$COMPAT"

"$PY" - "$LOCK_REL" "$EXPECTED_LOCK" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("gate2_v4_final/scripts").resolve()))
from gate2_v4_contract import load_source_lock, verify_source_lock_files

lock = load_source_lock(sys.argv[1])
assert lock["source_lock_sha256"] == sys.argv[2], lock["source_lock_sha256"]
verify_source_lock_files(lock)
print("PASS isolated b4 source tree", lock["source_lock_sha256"])
PY

echo "B4_COMPAT_WORKDIR=$COMPAT"
WORKDIR="$COMPAT" PY="$PY" SOURCE_LOCK="$LOCK_REL" SUBMIT_MODE="$MODE" \
PARTITION="${PARTITION:-dev}" TIME="${TIME:-02:00:00}" \
bash gate2_v4_final/slurm/gate2_v4/submit_nano5_finish_gt_v4.sh
