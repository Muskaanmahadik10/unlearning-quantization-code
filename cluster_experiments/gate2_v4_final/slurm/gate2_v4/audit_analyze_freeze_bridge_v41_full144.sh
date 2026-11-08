#!/bin/bash
# Validate, join, analyze, and freeze Pilot18 + Diversity12 + Remainder120.

set -euo pipefail

cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"

PY="${PY:-/home/anon_user/.conda/envs/unlearn/bin/python}"
PILOT_ROOT="${PILOT_ROOT:-results/gate2_v4_bridge_standard_v41_pilot18}"
DIVERSITY_ROOT="${DIVERSITY_ROOT:-results/gate2_v4_bridge_standard_v41_diversity12}"
REMAINDER_ROOT="${REMAINDER_ROOT:-results/gate2_v4_bridge_standard_v41_remainder120}"
PILOT_LOCK_SHA="${PILOT_LOCK_SHA:-7e7af7446475d71cea49bdf98ff5a8439e4a213a72ba9f563e40cadd9238d229}"
DIVERSITY_LOCK_SHA="${DIVERSITY_LOCK_SHA:-2b895012ed0d43d6c6da188b773f415a8c628910a8069f3cd2585abf6d60c3bb}"
REMAINDER_LOCK_SHA="${REMAINDER_LOCK_SHA:?REMAINDER_LOCK_SHA is required}"
V4_LOCK_SHA="${V4_LOCK_SHA:-b4caaae79a48523b2166a0af4409e78508700a74fc03ee25b36a62bfb461aa7b}"
PILOT_JOB="${PILOT_JOB:-317631}"
DIVERSITY_JOB="${DIVERSITY_JOB:-317817}"
REMAINDER_JOB="${REMAINDER_JOB:?REMAINDER_JOB is required}"
ANALYZER="gate2_v4_final/scripts/analyze_bridge_v41_full144.py"

[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 2; }
[[ -f "$ANALYZER" ]] || { echo "Missing analyzer: $ANALYZER" >&2; exit 2; }

"$PY" -m py_compile \
  "$ANALYZER" \
  gate2_v4_final/scripts/analyze_bridge_v41_combined30.py \
  gate2_v4_final/scripts/validate_bridge_v4.py \
  gate2_v4_final/scripts/validate_bridge_join_v4.py

collect_results() {
  local root="$1" expected="$2" label="$3"
  local count partials
  count=$(find "$root" -name bridge_v4.json -type f | wc -l)
  partials=$(find "$root" -name 'bridge_v4.json.partial' -type f | wc -l)
  echo "${label}_RESULTS=$count/$expected ${label}_PARTIALS=$partials"
  [[ "$count" -eq "$expected" && "$partials" -eq 0 ]]
}

collect_results "$PILOT_ROOT" 18 PILOT
collect_results "$DIVERSITY_ROOT" 12 DIVERSITY
collect_results "$REMAINDER_ROOT" 120 REMAINDER

require_completed_job() {
  local job="$1" label="$2" record state exit_code
  record=$(sacct -X -n -P -j "$job" --format=State,ExitCode | head -n 1)
  IFS='|' read -r state exit_code _ <<< "$record"
  echo "${label}_JOB_STATE=$state ${label}_JOB_EXIT=$exit_code"
  [[ "$state" == COMPLETED && "$exit_code" == 0:0 ]] || {
    echo "STOP: $label job $job is not COMPLETED 0:0" >&2
    exit 2
  }
}

require_completed_job "$PILOT_JOB" PILOT
require_completed_job "$DIVERSITY_JOB" DIVERSITY
require_completed_job "$REMAINDER_JOB" REMAINDER

STAMP=$(date +%Y%m%d_%H%M%S)
AUDIT="gate2_v4_bridge_v41_full144_audit_${STAMP}"
ANALYSIS="gate2_v4_bridge_v41_full144_analysis_${STAMP}"
mkdir -p "$AUDIT" "$ANALYSIS"

validate_panel() {
  local root="$1" lock_sha="$2" expected="$3" out="$4"
  mapfile -d '' files < <(find "$root" -name bridge_v4.json -type f -print0 | sort -z)
  "$PY" gate2_v4_final/scripts/validate_bridge_v4.py \
    --expected_source_lock_sha256 "$lock_sha" "${files[@]}" > "$out"
  [[ "$(grep -c '"status": "PASS"' "$out")" -eq "$expected" ]]
}

validate_panel "$PILOT_ROOT" "$PILOT_LOCK_SHA" 18 "$AUDIT/pilot18_validation.jsonl"
validate_panel "$DIVERSITY_ROOT" "$DIVERSITY_LOCK_SHA" 12 "$AUDIT/diversity12_validation.jsonl"
validate_panel "$REMAINDER_ROOT" "$REMAINDER_LOCK_SHA" 120 "$AUDIT/remainder120_validation.jsonl"

join_panel() {
  local root="$1" lock_sha="$2" out="$3"
  "$PY" gate2_v4_final/scripts/validate_bridge_join_v4.py \
    --bridge_root "$root" \
    --v4_results_root results/gate2_v4_standard \
    --expected_bridge_source_lock_sha256 "$lock_sha" \
    --expected_v4_source_lock_sha256 "$V4_LOCK_SHA" \
    --out "$out"
}

join_panel "$PILOT_ROOT" "$PILOT_LOCK_SHA" "$AUDIT/pilot18_join.tsv"
join_panel "$DIVERSITY_ROOT" "$DIVERSITY_LOCK_SHA" "$AUDIT/diversity12_join.tsv"
join_panel "$REMAINDER_ROOT" "$REMAINDER_LOCK_SHA" "$AUDIT/remainder120_join.tsv"

find logs/gate2_v4_bridge -type f \
  \( -name "*${PILOT_JOB}*" -o -name "*${DIVERSITY_JOB}*" -o -name "*${REMAINDER_JOB}*" \) \
  -print0 | xargs -0 -r grep -hE \
  'Traceback|CUDA out|OUT_OF_MEMORY|Segmentation fault|Killed|source lock.*fail|AssertionError|RuntimeError|ValueError' \
  > "$AUDIT/hard_errors.txt" || true

sacct -X -j "$PILOT_JOB,$DIVERSITY_JOB,$REMAINDER_JOB" \
  --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Start,End,AllocTRES,NodeList \
  > "$AUDIT/slurm_accounting.txt"

"$PY" "$ANALYZER" \
  --pilot_root "$PILOT_ROOT" \
  --diversity_root "$DIVERSITY_ROOT" \
  --remainder_root "$REMAINDER_ROOT" \
  --expected_pilot_lock_sha256 "$PILOT_LOCK_SHA" \
  --expected_diversity_lock_sha256 "$DIVERSITY_LOCK_SHA" \
  --expected_remainder_lock_sha256 "$REMAINDER_LOCK_SHA" \
  --pilot_join_tsv "$AUDIT/pilot18_join.tsv" \
  --diversity_join_tsv "$AUDIT/diversity12_join.tsv" \
  --remainder_join_tsv "$AUDIT/remainder120_join.tsv" \
  --hard_error_file "$AUDIT/hard_errors.txt" \
  --out_dir "$ANALYSIS" | tee "$ANALYSIS.console.txt"

FREEZE="gate2_v4_bridge_v41_full144_freeze_${STAMP}"
mkdir -p \
  "$FREEZE/evidence/results" \
  "$FREEZE/evidence/logs" \
  "$FREEZE/provenance" \
  "$FREEZE/tooling"
cp -a "$AUDIT" "$ANALYSIS" "$ANALYSIS.console.txt" "$FREEZE/evidence/"
cp -a "$PILOT_ROOT" "$DIVERSITY_ROOT" "$REMAINDER_ROOT" \
  "$FREEZE/evidence/results/"
find logs/gate2_v4_bridge -type f \
  \( -name "*${PILOT_JOB}*" -o -name "*${DIVERSITY_JOB}*" -o -name "*${REMAINDER_JOB}*" \) \
  -exec cp -a {} "$FREEZE/evidence/logs/" \;

find "$PILOT_ROOT" "$DIVERSITY_ROOT" "$REMAINDER_ROOT" \
  -name bridge_v4.json -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$FREEZE/provenance/full144_result_sha256.txt"
find logs/gate2_v4_bridge -type f \
  \( -name "*${PILOT_JOB}*" -o -name "*${DIVERSITY_JOB}*" -o -name "*${REMAINDER_JOB}*" \) \
  -print0 | sort -z | xargs -0 -r sha256sum \
  > "$FREEZE/provenance/full144_log_sha256.txt"
cp -a gate2_v4_final/V4_SOURCE_LOCK.json "$FREEZE/provenance/"
find gate2_v4_final/bridge_locks -name '*.json' -type f -print0 | \
  xargs -0 -r grep -l "$PILOT_LOCK_SHA\|$DIVERSITY_LOCK_SHA\|$REMAINDER_LOCK_SHA" | \
  while read -r lock; do cp -a "$lock" "$FREEZE/provenance/"; done
cp -a "$ANALYZER" \
  gate2_v4_final/scripts/analyze_bridge_v41_combined30.py \
  gate2_v4_final/scripts/validate_bridge_v4.py \
  gate2_v4_final/scripts/validate_bridge_join_v4.py \
  "$FREEZE/tooling/"
sha256sum "$FREEZE/tooling/"* > "$FREEZE/provenance/tooling_sha256.txt"
wallet ANON_ACCOUNT > "$FREEZE/provenance/wallet_snapshot.txt" 2>&1 || true

tar -czf "${FREEZE}.tar.gz" "$FREEZE"
sha256sum "${FREEZE}.tar.gz" | tee "${FREEZE}.tar.gz.sha256"
chmod -R a-w "$FREEZE" "${FREEZE}.tar.gz" "${FREEZE}.tar.gz.sha256"

echo "FULL144_AUDIT=$AUDIT"
echo "FULL144_ANALYSIS=$ANALYSIS"
echo "FULL144_FREEZE=$FREEZE"
echo "FULL144_ARCHIVE=${FREEZE}.tar.gz"
cat "$ANALYSIS/FULL144_AUTOMATED_ANALYSIS.md"
cat "$ANALYSIS/full144_decision.json"
