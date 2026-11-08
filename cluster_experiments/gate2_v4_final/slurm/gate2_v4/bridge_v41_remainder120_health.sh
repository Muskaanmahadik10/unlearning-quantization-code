#!/bin/bash
# Read-only health summary for the Gate 2 v4.1 Remainder120 job.

set -euo pipefail

cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"

JOB="${1:-$(cat bridge_v41_remainder120_jobid.txt 2>/dev/null || true)}"
ROOT="${RESULTS_ROOT:-results/gate2_v4_bridge_standard_v41_remainder120}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "Usage: $0 JOB_ID" >&2; exit 2; }

echo "=== TIME ==="
date '+NOW=%F %T %Z'
echo "=== JOB $JOB ==="
timeout 10s squeue -j "$JOB" \
  -o '%.10i %.10P %.30j %.2t %.19S %.10M %.11l %.4D %.12b %R' || true
sacct -X -j "$JOB" \
  --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Start,End,AllocTRES,NodeList

echo "=== RESULT COUNTS ==="
finals=$(find "$ROOT" -name bridge_v4.json -type f 2>/dev/null | wc -l)
partials=$(find "$ROOT" -name 'bridge_v4.json.partial' -type f 2>/dev/null | wc -l)
echo "FINALS=$finals/120"
echo "PARTIALS=$partials"
for quantizer in gptq_int4 nf4 rtn_int8; do
  count=$(find "$ROOT" -path "*/$quantizer/full_quantized/bridge_v4.json" -type f 2>/dev/null | wc -l)
  echo "$quantizer=$count/40"
done

echo "=== ACTIVE STEPS ==="
timeout 10s squeue --steps -j "$JOB" 2>/dev/null || true

echo "=== LATEST DECISIVE EVENTS ==="
find logs/gate2_v4_bridge -type f -name "${JOB}_*" -print0 2>/dev/null | \
  xargs -0 -r grep -hE \
  'Bridge condition=|Collecting GPTQ|Bridge v4 done|Traceback|CUDA out|OUT_OF_MEMORY|Segmentation fault|Killed|source lock.*fail|AssertionError|RuntimeError|ValueError' | \
  tail -n 60 || true

echo "=== HARD ERROR COUNT ==="
hard_errors=$(find logs/gate2_v4_bridge -type f -name "${JOB}_*" -print0 2>/dev/null | \
  xargs -0 -r grep -hE \
  'Traceback|CUDA out|OUT_OF_MEMORY|Segmentation fault|Killed|source lock.*fail|AssertionError|RuntimeError|ValueError' | \
  wc -l)
echo "HARD_ERRORS=$hard_errors"
