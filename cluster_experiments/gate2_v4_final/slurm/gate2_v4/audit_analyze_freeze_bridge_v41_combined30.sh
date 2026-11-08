#!/bin/bash
# Validate, join, analyze, and freeze Pilot18 plus Diversity12 evidence.

set -euo pipefail

cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"

PY="${PY:-/home/anon_user/.conda/envs/unlearn/bin/python}"
PILOT_ROOT="${PILOT_ROOT:-results/gate2_v4_bridge_standard_v41_pilot18}"
DIVERSITY_ROOT="${DIVERSITY_ROOT:-results/gate2_v4_bridge_standard_v41_diversity12}"
PILOT_LOCK_SHA="${PILOT_LOCK_SHA:-7e7af7446475d71cea49bdf98ff5a8439e4a213a72ba9f563e40cadd9238d229}"
DIVERSITY_LOCK_SHA="${DIVERSITY_LOCK_SHA:-2b895012ed0d43d6c6da188b773f415a8c628910a8069f3cd2585abf6d60c3bb}"
V4_LOCK_SHA="${V4_LOCK_SHA:-b4caaae79a48523b2166a0af4409e78508700a74fc03ee25b36a62bfb461aa7b}"
PILOT_JOB="${PILOT_JOB:-317631}"
DIVERSITY_JOB="${DIVERSITY_JOB:-317817}"
ANALYZER="gate2_v4_final/scripts/analyze_bridge_v41_combined30.py"
EXPECTED_ANALYZER="${EXPECTED_ANALYZER:-19f16e94752cff7d240fde3b1b60aa586b54b51e8a9e280099994e97b8cb91a6}"

[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 2; }
[[ -f "$ANALYZER" ]] || { echo "Missing analyzer: $ANALYZER" >&2; exit 2; }
[[ "$(sha256sum "$ANALYZER" | awk '{print $1}')" == "$EXPECTED_ANALYZER" ]] || {
  echo "STOP: combined analyzer hash mismatch" >&2
  exit 2
}

"$PY" -m py_compile \
  "$ANALYZER" \
  gate2_v4_final/scripts/validate_bridge_v4.py \
  gate2_v4_final/scripts/validate_bridge_join_v4.py

mapfile -d '' PILOT_RESULTS < <(find "$PILOT_ROOT" -name bridge_v4.json -type f -print0 | sort -z)
mapfile -d '' DIVERSITY_RESULTS < <(find "$DIVERSITY_ROOT" -name bridge_v4.json -type f -print0 | sort -z)
PILOT_PARTIALS=$(find "$PILOT_ROOT" -name 'bridge_v4.json.partial' -type f | wc -l)
DIVERSITY_PARTIALS=$(find "$DIVERSITY_ROOT" -name 'bridge_v4.json.partial' -type f | wc -l)

echo "PILOT_RESULTS=${#PILOT_RESULTS[@]}/18 PILOT_PARTIALS=$PILOT_PARTIALS"
echo "DIVERSITY_RESULTS=${#DIVERSITY_RESULTS[@]}/12 DIVERSITY_PARTIALS=$DIVERSITY_PARTIALS"
[[ "${#PILOT_RESULTS[@]}" -eq 18 && "$PILOT_PARTIALS" -eq 0 ]]
[[ "${#DIVERSITY_RESULTS[@]}" -eq 12 && "$DIVERSITY_PARTIALS" -eq 0 ]]

STAMP=$(date +%Y%m%d_%H%M%S)
AUDIT="gate2_v4_bridge_v41_combined30_audit_${STAMP}"
ANALYSIS="gate2_v4_bridge_v41_combined30_analysis_${STAMP}"
mkdir -p "$AUDIT" "$ANALYSIS"

"$PY" gate2_v4_final/scripts/validate_bridge_v4.py \
  --expected_source_lock_sha256 "$PILOT_LOCK_SHA" \
  "${PILOT_RESULTS[@]}" > "$AUDIT/pilot18_validation.jsonl"

"$PY" gate2_v4_final/scripts/validate_bridge_v4.py \
  --expected_source_lock_sha256 "$DIVERSITY_LOCK_SHA" \
  "${DIVERSITY_RESULTS[@]}" > "$AUDIT/diversity12_validation.jsonl"

[[ "$(grep -c '"status": "PASS"' "$AUDIT/pilot18_validation.jsonl")" -eq 18 ]]
[[ "$(grep -c '"status": "PASS"' "$AUDIT/diversity12_validation.jsonl")" -eq 12 ]]

"$PY" gate2_v4_final/scripts/validate_bridge_join_v4.py \
  --bridge_root "$PILOT_ROOT" \
  --v4_results_root results/gate2_v4_standard \
  --expected_bridge_source_lock_sha256 "$PILOT_LOCK_SHA" \
  --expected_v4_source_lock_sha256 "$V4_LOCK_SHA" \
  --out "$AUDIT/pilot18_join.tsv"

"$PY" gate2_v4_final/scripts/validate_bridge_join_v4.py \
  --bridge_root "$DIVERSITY_ROOT" \
  --v4_results_root results/gate2_v4_standard \
  --expected_bridge_source_lock_sha256 "$DIVERSITY_LOCK_SHA" \
  --expected_v4_source_lock_sha256 "$V4_LOCK_SHA" \
  --out "$AUDIT/diversity12_join.tsv"

find logs/gate2_v4_bridge -type f \
  \( -name "${PILOT_JOB}_*" -o -name "${DIVERSITY_JOB}_*" \) -print0 |
  xargs -0 -r grep -hE \
  'Traceback|CUDA out|OUT_OF_MEMORY|Segmentation fault|Killed|source lock.*fail|AssertionError|RuntimeError|ValueError' \
  > "$AUDIT/hard_errors.txt" || true

sacct -X -j "$PILOT_JOB,$DIVERSITY_JOB" \
  --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Start,End,AllocTRES,NodeList \
  > "$AUDIT/slurm_accounting.txt"

"$PY" "$ANALYZER" \
  --pilot_root "$PILOT_ROOT" \
  --diversity_root "$DIVERSITY_ROOT" \
  --expected_pilot_lock_sha256 "$PILOT_LOCK_SHA" \
  --expected_diversity_lock_sha256 "$DIVERSITY_LOCK_SHA" \
  --pilot_join_tsv "$AUDIT/pilot18_join.tsv" \
  --diversity_join_tsv "$AUDIT/diversity12_join.tsv" \
  --hard_error_file "$AUDIT/hard_errors.txt" \
  --out_dir "$ANALYSIS" | tee "$ANALYSIS.console.txt"

FREEZE="gate2_v4_bridge_v41_combined30_freeze_${STAMP}"
mkdir -p "$FREEZE/evidence" "$FREEZE/provenance" "$FREEZE/tooling"
cp -a "$AUDIT" "$FREEZE/evidence/"
cp -a "$ANALYSIS" "$ANALYSIS.console.txt" "$FREEZE/evidence/"
cp -a "$DIVERSITY_ROOT" "$FREEZE/evidence/diversity12_results"

find "$PILOT_ROOT" "$DIVERSITY_ROOT" -name bridge_v4.json -type f -print0 |
  sort -z | xargs -0 sha256sum > "$FREEZE/provenance/combined30_result_sha256.txt"
find logs/gate2_v4_bridge -type f \
  \( -name "${PILOT_JOB}_*" -o -name "${DIVERSITY_JOB}_*" \) -print0 |
  sort -z | xargs -0 -r sha256sum > "$FREEZE/provenance/combined30_log_sha256.txt"
cp -a gate2_v4_final/V4_SOURCE_LOCK.json "$FREEZE/provenance/"
find gate2_v4_final/bridge_locks -name '*.json' -type f -print0 |
  xargs -0 -r grep -l "$PILOT_LOCK_SHA\|$DIVERSITY_LOCK_SHA" |
  while read -r lock; do cp -a "$lock" "$FREEZE/provenance/"; done
cp -a "$ANALYZER" \
  gate2_v4_final/scripts/validate_bridge_v4.py \
  gate2_v4_final/scripts/validate_bridge_join_v4.py \
  "$FREEZE/tooling/"
sha256sum "$FREEZE/tooling/"* > "$FREEZE/provenance/tooling_sha256.txt"
wallet ANON_ACCOUNT > "$FREEZE/provenance/wallet_snapshot.txt" 2>&1 || true

tar -czf "${FREEZE}.tar.gz" "$FREEZE"
sha256sum "${FREEZE}.tar.gz" | tee "${FREEZE}.tar.gz.sha256"
chmod -R a-w "$FREEZE" "${FREEZE}.tar.gz" "${FREEZE}.tar.gz.sha256"

echo "COMBINED30_AUDIT=$AUDIT"
echo "COMBINED30_ANALYSIS=$ANALYSIS"
echo "COMBINED30_FREEZE=$FREEZE"
echo "COMBINED30_ARCHIVE=${FREEZE}.tar.gz"
cat "$ANALYSIS/COMBINED30_AUTOMATED_ANALYSIS.md"
cat "$ANALYSIS/combined30_decision.json"
