#!/bin/bash
set -euo pipefail
cd /work/anon_user/hessian_unlearning
PY=/usr/bin/python3
DIR=results/org_h200_execution_v1/b05_feature_cells_v2
REF=results/org_h200_execution_v1/b05_current_runtime_reference_v1
OUT=results/org_h200_execution_v1/b05_selector_locks/selector_lock_current_runtime_v2.json
JOBS=348869,349004,349005,349006,349007,349008
sacct -X -n -P -j "$JOBS" --format=JobIDRaw,State,ExitCode | "$PY" -c '
import sys
expected=set("348869 349004 349005 349006 349007 349008".split())
rows=[line.strip().split("|") for line in sys.stdin if line.strip()]
assert {r[0] for r in rows}==expected, "STOP: accounting incomplete"
assert all(r[1]=="COMPLETED" and r[2]=="0:0" for r in rows), "STOP: jobs unfinished or failed"
print("B05_SIX_JOB_ACCOUNTING=PASS")'
RECEIPTS=(
 "$DIR/production_ordinary_tofu_ga_seed42_gptq_int4_348869.json"
 "$DIR/production_ordinary_tofu_ga_seed42_nf4_349004.json"
 "$DIR/production_ordinary_tofu_ga_seed42_rtn_int8_349005.json"
 "$DIR/production_adverse_muse_gdr_seed2024_nf4_349006.json"
 "$DIR/production_adverse_muse_gdr_seed2024_gptq_int4_349007.json"
 "$DIR/production_adverse_muse_gdr_seed2024_rtn_int8_349008.json"
)
for FILE in "${RECEIPTS[@]}"; do sha256sum -c "$FILE.sha256"; done
"$PY" org_h200_execution_v1/scripts/test_certify_org_e2b_b05_feature_wave_v2.py
"$PY" org_h200_execution_v1/scripts/certify_org_e2b_b05_feature_wave_v2.py \
 --reference-manifest "$REF/current_runtime_reference_manifest_v1.json" \
 --receipts "${RECEIPTS[@]}" --output "$OUT"
sha256sum "$OUT" > "$OUT.sha256"
