#!/bin/bash
set -euo pipefail
ROOT=${1:-/work/anon_user/hessian_unlearning}
REFERENCE_DIR=${2:?reference directory required}
cd "$ROOT"
SCRIPT=org_h200_execution_v1/slurm/org_e2b_b05_feature_current_reference_h200_dev_v2.slurm
test -f "$SCRIPT"
test -f "$REFERENCE_DIR/current_runtime_reference_manifest_v1.json"
mkdir -p results/org_h200_execution_v1/b05_feature_submissions_v2
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=results/org_h200_execution_v1/b05_feature_submissions_v2/current_reference_wave_${STAMP}.tsv
printf 'cell\tquantizer\tjob_id\n' > "$OUT"
for CELL in ordinary adverse; do
  for QUANTIZER in nf4 gptq_int4 rtn_int8; do
    RESPONSE=$(sbatch --parsable --export=ALL,B05_CELL="$CELL",B05_QUANTIZER="$QUANTIZER",B05_REFERENCE_DIR="$REFERENCE_DIR" "$SCRIPT")
    JOB=${RESPONSE%%;*}
    case "$JOB" in (*[!0-9]*|'') echo "invalid sbatch response: $RESPONSE" >&2; exit 1;; esac
    printf '%s\t%s\t%s\n' "$CELL" "$QUANTIZER" "$JOB" >> "$OUT"
  done
done
printf 'ORG_E2B_B05_CURRENT_REFERENCE_FEATURE_WAVE_SUBMITTED=PASS\nSUBMISSIONS=%s\n' "$OUT"
cat "$OUT"
