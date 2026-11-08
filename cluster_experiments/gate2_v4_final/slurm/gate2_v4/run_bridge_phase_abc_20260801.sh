#!/bin/bash
set -euo pipefail

cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"

PY="${PY:-/home/anon_user/.conda/envs/unlearn/bin/python}"
V4_RESULT_LOCK="${V4_RESULT_LOCK:-b4caaae79a48523b2166a0af4409e78508700a74fc03ee25b36a62bfb461aa7b}"
V4_RESULTS_ROOT="${V4_RESULTS_ROOT:-results/gate2_v4_standard}"
BRIDGE_RESULTS_ROOT="${BRIDGE_RESULTS_ROOT:-results/gate2_v4_bridge_standard}"
BRIDGE_LOGS_ROOT="${BRIDGE_LOGS_ROOT:-logs/gate2_v4_bridge}"
V4_SOURCE_LOCK="${V4_SOURCE_LOCK:-gate2_v4_final/V4_SOURCE_LOCK.json}"
ALLOW_INCOMPLETE_GT="${ALLOW_INCOMPLETE_GT:-0}"

mkdir -p "$BRIDGE_RESULTS_ROOT" "$BRIDGE_LOGS_ROOT" gate2_v4_final/bridge_locks

echo "=== Phase A/B/C Gate 2 v4 bridge setup ==="
echo "PY=$PY"
echo "V4_RESULT_LOCK=$V4_RESULT_LOCK"
echo "V4_RESULTS_ROOT=$V4_RESULTS_ROOT"
echo "BRIDGE_RESULTS_ROOT=$BRIDGE_RESULTS_ROOT"

echo "=== static shell gates ==="
bash -n \
  gate2_v4_final/slurm/gate2_v4/bridge_v4_slotfill.slurm \
  gate2_v4_final/slurm/gate2_v4/bridge_v4_smoke_8gpu_bundle.slurm \
  gate2_v4_final/slurm/gate2_v4/bridge_v4_core64_8shard_bundle.slurm \
  gate2_v4_final/slurm/gate2_v4/submit_gate2_v4_bridge_slotfill.sh \
  gate2_v4_final/slurm/gate2_v4/submit_bridge_smoke_20260801.sh \
  gate2_v4_final/slurm/gate2_v4/submit_bridge_fullq_20260801.sh \
  gate2_v4_final/slurm/gate2_v4/bridge_health_20260801.sh

echo "=== static python gates ==="
"$PY" -m py_compile \
  gate2_v4_final/scripts/build_gate2_v4_bridge_rankings.py \
  gate2_v4_final/scripts/run_gate2_whole_model_bridge_v4.py \
  gate2_v4_final/scripts/validate_bridge_v4.py \
  gate2_v4_final/scripts/validate_bridge_join_v4.py \
  gate2_v4_final/scripts/audit_bridge_v4_icml.py \
  gate2_v4_final/scripts/audit_icml_core_144_v4_status.py

echo "=== Phase A: fixed v4 status audit ==="
STATUS_OUT="gate2_v4_bridge_phaseA_status_$(date +%Y%m%d_%H%M%S).tsv"
"$PY" gate2_v4_final/scripts/audit_icml_core_144_v4_status.py \
  --results_root "$V4_RESULTS_ROOT" \
  --source_lock_path "$V4_SOURCE_LOCK" \
  --out "$STATUS_OUT"
echo "STATUS_OUT=$STATUS_OUT"

"$PY" - "$STATUS_OUT" "$ALLOW_INCOMPLETE_GT" <<'PY'
import csv
import sys
from collections import Counter

path = sys.argv[1]
allow_incomplete = sys.argv[2] == "1"
rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
for key in ("gt_status", "upqi_status", "join_status", "next_action"):
    print(key, dict(Counter(row[key] for row in rows)))
bad = [
    row for row in rows
    if row["upqi_status"] != "clean_complete"
    or row["gt_status"] != "clean_complete"
    or row["join_status"] != "passed"
]
print("bad_or_incomplete_rows", len(bad))
for row in bad[:20]:
    print(
        row["cell_id"],
        row["gt_status"],
        row.get("gt_reason", ""),
        row["upqi_status"],
        row["join_status"],
        row.get("join_reason", ""),
    )
upqi_clean = sum(row["upqi_status"] == "clean_complete" for row in rows)
join_passed = sum(row["join_status"] == "passed" for row in rows)
if upqi_clean != 144:
    raise SystemExit(f"UPQI is not 144/144 clean: {upqi_clean}/144")
if join_passed != 144 and not allow_incomplete:
    raise SystemExit(f"Strict joins are not 144/144: {join_passed}/144")
if join_passed != 144:
    print(f"WARNING: strict joins are {join_passed}/144; GT-oracle bridge claims must exclude missing GT cells.")
PY

echo "=== Phase B: create separate bridge source lock ==="
BRIDGE_LOCK="gate2_v4_final/bridge_locks/V4_BRIDGE_SOURCE_LOCK_$(date +%Y%m%d_%H%M%S).json"
"$PY" gate2_v4_final/scripts/write_gate2_v4_source_lock.py \
  --base_dir gate2_v4_final \
  --manifest_dir manifests/gate2_v4 \
  --output "$BRIDGE_LOCK"

BRIDGE_LOCK_SHA=$("$PY" - "$BRIDGE_LOCK" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["source_lock_sha256"])
PY
)
BRIDGE_SCRIPT_SHA=$("$PY" - "$BRIDGE_LOCK" <<'PY'
import json
import sys
j = json.load(open(sys.argv[1], encoding="utf-8"))
print(j["files"]["gate2_v4_final/scripts/run_gate2_whole_model_bridge_v4.py"])
PY
)
echo "BRIDGE_LOCK=$BRIDGE_LOCK"
echo "BRIDGE_LOCK_SHA=$BRIDGE_LOCK_SHA"
echo "BRIDGE_SCRIPT_SHA=$BRIDGE_SCRIPT_SHA"

echo "=== Phase B: build bridge rankings from locked v4 evidence ==="
"$PY" gate2_v4_final/scripts/build_gate2_v4_bridge_rankings.py \
  --results_root "$V4_RESULTS_ROOT" \
  --out "$BRIDGE_RESULTS_ROOT/bridge_rankings_v4.jsonl" \
  --summary_out "$BRIDGE_RESULTS_ROOT/bridge_rankings_v4.summary.tsv" \
  --expected_source_lock_sha256 "$V4_RESULT_LOCK"

echo "=== Phase B: ranking sanity ==="
"$PY" - "$BRIDGE_RESULTS_ROOT/bridge_rankings_v4.jsonl" <<'PY'
import json
import sys
from collections import Counter

path = sys.argv[1]
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
src = Counter(row["ranking_source"] for row in rows)
cells = {row["cell_id"] for row in rows}
bad = [
    row for row in rows
    if not row.get("cell_id")
    or not row.get("layer_name")
    or not row.get("ranking_source")
    or row.get("score") is None
    or not row.get("evidence_sha256")
]
print("rows", len(rows))
print("cells", len(cells))
print("sources", dict(src))
print("bad_rows", len(bad))
assert len(cells) == 144, len(cells)
assert src["UPQI_abs"] == 144 * 224, src
assert src["UPQI_signed"] == 144 * 224, src
assert src["random"] == 144 * 224, src
assert src["displacement_norm"] == 144 * 224, src
assert not bad, bad[:3]
PY

echo "=== Phase C: bridge dry-run before model load ==="
PYTHON="$PY" SLURM_JOB_ID=dryrun_bridge_tofu DRY_RUN=1 SOURCE_LOCK="$BRIDGE_LOCK" \
MODEL_TAG=llama2_7b_adamw DATASET=tofu CKPT_BASE=checkpoints/llama2_7b_adamw \
SEEDS_OVERRIDE=42 ALGOS_OVERRIDE=GA QUANTIZERS_OVERRIDE="nf4 gptq_int4 rtn_int8" \
CONDITIONS_OVERRIDE=full_quantized GPUS_PER_JOB=1 MAX_PARALLEL=1 \
bash gate2_v4_final/slurm/gate2_v4/bridge_v4_slotfill.slurm

PYTHON="$PY" SLURM_JOB_ID=dryrun_bridge_muse DRY_RUN=1 SOURCE_LOCK="$BRIDGE_LOCK" \
MODEL_TAG=llama2_muse_adamw DATASET=muse_news CKPT_BASE=checkpoints/llama2_muse_adamw \
SEEDS_OVERRIDE=42 ALGOS_OVERRIDE=GA QUANTIZERS_OVERRIDE="nf4 gptq_int4 rtn_int8" \
CONDITIONS_OVERRIDE=full_quantized GPUS_PER_JOB=1 MAX_PARALLEL=1 \
bash gate2_v4_final/slurm/gate2_v4/bridge_v4_slotfill.slurm

echo "=== Phase C: dry-run log check ==="
grep -R "dry-run ok before model load\|Traceback\|SystemExit\|No checkpoint\|failed validation" \
  "$BRIDGE_LOGS_ROOT"/*dryrun_bridge* 2>/dev/null | tail -n 80 || true

echo "=== Phase A/B/C complete ==="
echo "Export these for next steps:"
echo "export BRIDGE_LOCK='$BRIDGE_LOCK'"
echo "export BRIDGE_LOCK_SHA='$BRIDGE_LOCK_SHA'"
echo "export V4_RESULT_LOCK='$V4_RESULT_LOCK'"
