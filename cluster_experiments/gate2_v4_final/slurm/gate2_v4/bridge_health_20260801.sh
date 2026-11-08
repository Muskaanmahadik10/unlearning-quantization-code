#!/bin/bash
set -euo pipefail

cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"

PY="${PY:-/home/anon_user/.conda/envs/unlearn/bin/python}"
BRIDGE_RESULTS_ROOT="${BRIDGE_RESULTS_ROOT:-results/gate2_v4_bridge_standard}"
BRIDGE_LOGS_ROOT="${BRIDGE_LOGS_ROOT:-logs/gate2_v4_bridge}"
BRIDGE_LOCK_SHA="${BRIDGE_LOCK_SHA:-}"
V4_RESULT_LOCK="${V4_RESULT_LOCK:-b4caaae79a48523b2166a0af4409e78508700a74fc03ee25b36a62bfb461aa7b}"

echo "=== bridge queue ==="
squeue -u "$USER" -o "%.8i %.10P %.30j %.10a %.2t %.24E %.11M %.11l %.4D %.4C %.12b %R" | grep -E "JOBID|bridge_v4" || true

echo "=== bridge raw counts ==="
echo -n "full_quantized: "
find "$BRIDGE_RESULTS_ROOT" -path "*/full_quantized/bridge_v4.json" 2>/dev/null | wc -l
echo -n "bf16 explicit:  "
find "$BRIDGE_RESULTS_ROOT" -path "*/bf16/bridge_v4.json" 2>/dev/null | wc -l
echo -n "protected:      "
find "$BRIDGE_RESULTS_ROOT" -path "*/protected_*_k*/bridge_v4.json" 2>/dev/null | wc -l

echo "=== protected ranking evidence spot check ==="
"$PY" - "$BRIDGE_RESULTS_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = list(root.glob("**/protected_*_k*/bridge_v4.json"))
bad = []
for path in paths:
    obj = json.load(open(path, encoding="utf-8"))
    ev = obj.get("ranking_evidence")
    if not isinstance(ev, dict) or not ev.get("rankings_sha256") or not ev.get("selected_rows"):
        bad.append(str(path))
print("protected_checked", len(paths))
print("protected_bad_evidence", len(bad))
for item in bad[:10]:
    print("BAD", item)
PY

echo "=== hard errors ==="
grep -R "Traceback\|CUDA out\|OUT_OF_MEMORY\|No checkpoint\|failed validation\|SystemExit" \
  "$BRIDGE_LOGS_ROOT" 2>/dev/null | tail -n 100 || true

echo "=== latest done ==="
grep -R "Bridge v4 done" "$BRIDGE_LOGS_ROOT" 2>/dev/null | tail -n 40 || true

if [[ -n "$BRIDGE_LOCK_SHA" ]]; then
  echo "=== bridge ICML audit ==="
  OUT_PREFIX="bridge_v4_health_$(date +%Y%m%d_%H%M%S)"
  "$PY" gate2_v4_final/scripts/audit_bridge_v4_icml.py \
    --bridge_root "$BRIDGE_RESULTS_ROOT" \
    --logs_root "$BRIDGE_LOGS_ROOT" \
    --expected_source_lock_sha256 "$BRIDGE_LOCK_SHA" \
    --out_prefix "$OUT_PREFIX" || true
  echo "OUT_PREFIX=$OUT_PREFIX"
fi

if [[ "${BRIDGE_DESTROYER_AUDIT:-0}" == "1" ]]; then
  echo "=== bridge destroyer audit ==="
  DESTROYER_OUT_PREFIX="bridge_v4_destroyer_health_$(date +%Y%m%d_%H%M%S)"
  DESTROYER_ARGS=(
    --bridge_root "$BRIDGE_RESULTS_ROOT"
    --logs_root "$BRIDGE_LOGS_ROOT"
    --expected_v4_source_lock_sha256 "$V4_RESULT_LOCK"
    --out_prefix "$DESTROYER_OUT_PREFIX"
  )
  if [[ -n "$BRIDGE_LOCK_SHA" ]]; then
    DESTROYER_ARGS+=(--expected_bridge_source_lock_sha256 "$BRIDGE_LOCK_SHA")
  fi
  "$PY" gate2_v4_final/scripts/audit_bridge_v4_destroyer.py "${DESTROYER_ARGS[@]}" || true
  echo "DESTROYER_OUT_PREFIX=$DESTROYER_OUT_PREFIX"
fi

echo "=== bridge join audit, if any bridge results exist ==="
if find "$BRIDGE_RESULTS_ROOT" -name bridge_v4.json -print -quit 2>/dev/null | grep -q .; then
  JOIN_OUT="bridge_v4_join_health_$(date +%Y%m%d_%H%M%S).tsv"
  "$PY" gate2_v4_final/scripts/validate_bridge_join_v4.py \
    --bridge_root "$BRIDGE_RESULTS_ROOT" \
    --v4_results_root results/gate2_v4_standard \
    --expected_bridge_source_lock_sha256 "$BRIDGE_LOCK_SHA" \
    --expected_v4_source_lock_sha256 "$V4_RESULT_LOCK" \
    --out "$JOIN_OUT" || true
  echo "JOIN_OUT=$JOIN_OUT"
fi
