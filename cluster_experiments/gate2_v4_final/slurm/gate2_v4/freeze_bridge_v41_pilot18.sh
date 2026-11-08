#!/bin/bash
# Freeze the accepted Gate 2 v4.1 pilot18 evidence without deleting source files.

set -euo pipefail

cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"

PY="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
JOB="${PILOT_JOB:-317631}"
ROOT="${PILOT_ROOT:-results/gate2_v4_bridge_standard_v41_pilot18}"
AUDIT_SRC="${PILOT_AUDIT_DIR:-gate2_v4_bridge_v41_pilot18_audit_20260831_015601}"
EXPECTED_LOCK="${EXPECTED_BRIDGE_LOCK:-7e7af7446475d71cea49bdf98ff5a8439e4a213a72ba9f563e40cadd9238d229}"
STAMP="$(date +%Y%m%d_%H%M%S)"
FREEZE="gate2_v4_bridge_v41_pilot18_freeze_${STAMP}"
ARCHIVE="${FREEZE}.tar.gz"

[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 2; }
[[ -d "$ROOT" ]] || { echo "Missing pilot root: $ROOT" >&2; exit 2; }
[[ -d "$AUDIT_SRC" ]] || { echo "Missing pilot audit: $AUDIT_SRC" >&2; exit 2; }

mapfile -d '' RESULTS < <(find "$ROOT" -name bridge_v4.json -type f -print0 | sort -z)
PARTIALS="$(find "$ROOT" -name 'bridge_v4.json.partial' -type f | wc -l)"
[[ "${#RESULTS[@]}" -eq 18 ]] || { echo "Expected 18 results" >&2; exit 2; }
[[ "$PARTIALS" -eq 0 ]] || { echo "Found partial results" >&2; exit 2; }

"$PY" gate2_v4_final/scripts/validate_bridge_v4.py \
  --expected_source_lock_sha256 "$EXPECTED_LOCK" \
  "${RESULTS[@]}" > /tmp/bridge_v41_pilot18_freeze_validation_$$.jsonl
[[ "$(grep -c '"status": "PASS"' /tmp/bridge_v41_pilot18_freeze_validation_$$.jsonl)" -eq 18 ]]

mkdir -p "$FREEZE/results" "$FREEZE/logs" "$FREEZE/tooling" "$FREEZE/provenance"
cp -a "$ROOT" "$FREEZE/results/"
cp -a "$AUDIT_SRC" "$FREEZE/"
cp -a /tmp/bridge_v41_pilot18_freeze_validation_$$.jsonl "$FREEZE/provenance/strict_validation.jsonl"
rm -f /tmp/bridge_v41_pilot18_freeze_validation_$$.jsonl

find logs/gate2_v4_bridge -maxdepth 1 -type f -name "${JOB}_*" -exec cp -a {} "$FREEZE/logs/" \;

LOCK_PATH="$(tr -d '[:space:]' < gate2_v4_final/bridge_locks/CURRENT_NANO4_BRIDGE_V41_LOCK.txt)"
cp -a "$LOCK_PATH" "$FREEZE/provenance/"
cp -a gate2_v4_final/bridge_locks/CURRENT_NANO4_BRIDGE_V41_LOCK.txt "$FREEZE/provenance/"
cp -a results/gate2_v4_bridge_standard/bridge_rankings_v4.jsonl "$FREEZE/provenance/"

cp -a gate2_v4_final/scripts/run_gate2_whole_model_bridge_v4.py "$FREEZE/tooling/"
cp -a gate2_v4_final/scripts/validate_bridge_v4.py "$FREEZE/tooling/"
cp -a gate2_v4_final/scripts/validate_bridge_join_v4.py "$FREEZE/tooling/"
cp -a gate2_v4_final/scripts/analyze_bridge_v41_pilot18.py "$FREEZE/tooling/"
cp -a gate2_v4_final/slurm/gate2_v4/bridge_v4_parallel_matrix.slurm "$FREEZE/tooling/"
cp -a gate2_v4_final/slurm/gate2_v4/bridge_v4_slotfill.slurm "$FREEZE/tooling/"

sacct -X -j "$JOB" \
  --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Start,End,AllocTRES,NodeList \
  > "$FREEZE/provenance/slurm_accounting.txt"
wallet ANON_ACCOUNT > "$FREEZE/provenance/wallet_snapshot.txt" 2>&1 || true

find "$FREEZE" -type f \
  ! -path "$FREEZE/provenance/evidence_sha256.txt" \
  ! -path "$FREEZE/provenance/file_inventory.tsv" \
  -print0 | sort -z | xargs -0 sha256sum \
  > "$FREEZE/provenance/evidence_sha256.txt"
find "$FREEZE" -type f \
  ! -path "$FREEZE/provenance/file_inventory.tsv" \
  -printf '%s\t%p\n' | sort -k2 \
  > "$FREEZE/provenance/file_inventory.tsv"

tar -czf "$ARCHIVE" "$FREEZE"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
chmod -R a-w "$FREEZE"
chmod a-w "$ARCHIVE" "${ARCHIVE}.sha256"

echo "PILOT18_FREEZE=$FREEZE"
echo "PILOT18_ARCHIVE=$ARCHIVE"
cat "${ARCHIVE}.sha256"
echo "PILOT18_EVIDENCE_FREEZE=PASS"
