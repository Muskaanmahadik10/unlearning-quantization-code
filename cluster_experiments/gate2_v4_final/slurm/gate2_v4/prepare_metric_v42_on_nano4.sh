#!/bin/bash
# Normalize, test, audit inputs, and create the immutable v4.2 metric source lock.

set -euo pipefail
cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"
PYTHON="${PYTHON:-/home/anon_user/.conda/envs/unlearn/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"

mapfile -t files < <(
  find gate2_v4_final/scripts gate2_v4_final/tests gate2_v4_final/slurm/gate2_v4 -maxdepth 1 -type f \
    \( -name '*v42*.py' -o -name '*v42*.slurm' -o -name '*v42*.sh' \) -print
)
printf '%s\n' "${files[@]}" | xargs -r sed -i 's/\r$//'

"$PYTHON" -m py_compile \
  gate2_v4_final/scripts/gate2_metric_v42_contracts.py \
  gate2_v4_final/scripts/gate2_metric_v42_math.py \
  gate2_v4_final/scripts/validate_gate2_metric_suite_v42.py \
  gate2_v4_final/scripts/run_gate2_metric_suite_v42.py \
  gate2_v4_final/scripts/validate_upqi_star_v42.py \
  gate2_v4_final/scripts/run_gate2_upqi_star_v42.py \
  gate2_v4_final/scripts/analyze_upqi_star_v42_convergence.py \
  gate2_v4_final/scripts/analyze_gate2_metric_suite_v42.py \
  gate2_v4_final/scripts/build_mitigation_pilot24_v42.py

"$PYTHON" -m unittest gate2_v4_final.tests.test_gate2_metric_v42_math -v

for script in gate2_v4_final/slurm/gate2_v4/*v42*.slurm gate2_v4_final/slurm/gate2_v4/*v42*.sh; do
  [[ -f "$script" ]] && bash -n "$script"
done

STATUS="gate2_v42_pre_metric_core144_status_${STAMP}.tsv"
"$PYTHON" gate2_v4_final/scripts/audit_icml_core_144_v4_status.py \
  --results_root results/gate2_v4_standard --out "$STATUS"
"$PYTHON" - "$STATUS" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig"), delimiter="\t"))
if len(rows) != 144:
    raise SystemExit(f"Core status has {len(rows)} rows, expected 144")
bad = [row for row in rows if row.get("next_action") != "lock_for_metrics"]
if bad:
    raise SystemExit(f"Core GT/UPQI/join gate failed for {len(bad)} cells: {bad[:3]}")
print("CORE144_GT_UPQI_JOIN=PASS rows=144")
PY

mkdir -p gate2_v4_final/metric_locks
LOCK="gate2_v4_final/metric_locks/V4_METRIC_V42_SOURCE_LOCK_NANO4_${STAMP}.json"
"$PYTHON" gate2_v4_final/scripts/write_gate2_v4_source_lock.py \
  --base_dir gate2_v4_final --manifest_dir manifests/gate2_v4 --output "$LOCK"
printf '%s\n' "$LOCK" > gate2_v4_final/metric_locks/CURRENT_METRIC_V42_LOCK.txt
LOCK_SHA="$($PYTHON -c "import json; print(json.load(open('$LOCK'))['source_lock_sha256'])")"

echo "METRIC_V42_PREPARE=PASS"
echo "METRIC_V42_LOCK=$LOCK"
echo "METRIC_V42_LOCK_SHA=$LOCK_SHA"
echo "CORE144_STATUS=$STATUS"
