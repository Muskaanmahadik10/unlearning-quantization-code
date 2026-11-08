#!/bin/bash
set -euo pipefail
ROOT=${1:-/work/anon_user/hessian_unlearning}
cd "$ROOT"
SCRIPT=org_h200_execution_v1/slurm/org_e2b_b05_gptq_qdp_drift_diagnostic_h200_dev_v2.slurm
test -f "$SCRIPT"
for CELL in ordinary adverse; do
  RESPONSE=$(sbatch --parsable --export=ALL,B05_CELL="$CELL" "$SCRIPT")
  JOB=${RESPONSE%%;*}
  case "$JOB" in (*[!0-9]*|'') echo "invalid sbatch response: $RESPONSE" >&2; exit 1;; esac
  printf '%s\t%s\n' "$CELL" "$JOB"
done
printf 'ORG_E2B_B05_GPTQ_QDP_DRIFT_DIAGNOSTICS_V2_SUBMITTED=PASS\n'
