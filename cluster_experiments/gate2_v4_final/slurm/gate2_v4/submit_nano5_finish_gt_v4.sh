#!/bin/bash
set -euo pipefail

cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"

PY="${PY:-/work/anon_user/.runtime/envs/unlearn-nano5/bin/python}"
SOURCE_LOCK="${SOURCE_LOCK:-gate2_v4_final/V4_SOURCE_LOCK.json}"
EXPECTED_LOCK="${EXPECTED_LOCK:-b4caaae79a48523b2166a0af4409e78508700a74fc03ee25b36a62bfb461aa7b}"
ACCOUNT="${ACCOUNT:-anon_account}"
PARTITION="${PARTITION:-normal}"
TIME="${TIME:-12:00:00}"
MODE="${SUBMIT_MODE:-test-only}"

export PY SOURCE_LOCK

"$PY" - "$SOURCE_LOCK" "$EXPECTED_LOCK" <<'PY'
import hashlib
import json
import pathlib
import sys

lock_path = pathlib.Path(sys.argv[1])
expected_lock = sys.argv[2]
lock = json.loads(lock_path.read_text(encoding="utf-8"))
assert lock["source_lock_sha256"] == expected_lock, lock["source_lock_sha256"]
runner = pathlib.Path("gate2_v4_final/scripts/run_gate2_ground_truth_v4.py")
actual = hashlib.sha256(runner.read_bytes()).hexdigest()
locked = lock["files"][runner.as_posix()]
assert actual == locked, (actual, locked)

partials = (
    pathlib.Path("results/gate2_v4_standard/llama3_8b_adamw_KLR_seed42/gptq_int4/ground_truth_v4.json.partial"),
    pathlib.Path("results/gate2_v4_standard/llama3_muse_adamw_NPO_seed5000/rtn_int8/ground_truth_v4.json.partial"),
)
for path in partials:
    assert path.exists(), path
    obj = json.loads(path.read_text(encoding="utf-8"))
    provenance = obj.get("provenance")
    assert isinstance(provenance, dict), (path, "missing_provenance")
    actual_lock = provenance.get("source_lock_sha256")
    assert actual_lock == expected_lock, (path, actual_lock)
    layers = obj.get("layers") or obj.get("per_layer") or obj.get("per_layer_checkpoint") or {}
    print("PASS partial", path, "layers", len(layers))
print("PASS canonical GT runner and result lock", expected_lock)
PY

submit_cell() {
  local label="$1"
  local seed="$2"
  local algo="$3"
  local quantizer="$4"

  STAGE=gt SUBMIT_MODE="$MODE" ACCOUNT="$ACCOUNT" LABEL="$label" \
  SEEDS_OVERRIDE="$seed" ALGOS_OVERRIDE="$algo" QUANTIZERS_OVERRIDE="$quantizer" \
  PARTITION="$PARTITION" GPUS=1 CPUS=8 MEM=200G TIME="$TIME" MAX_PARALLEL=1 \
  PYTHON="$PY" SOURCE_LOCK="$SOURCE_LOCK" \
  bash gate2_v4_final/slurm/gate2_v4/submit_gate2_v4_slotfill.sh
}

echo "=== Nano5 GT v4 finish: mode=$MODE account=$ACCOUNT partition=$PARTITION ==="
if [[ "$MODE" == "submit" ]]; then
  TOFU_OUTPUT="$(submit_cell l3_A 42 KLR gptq_int4)"
  printf '%s\n' "$TOFU_OUTPUT"
  TOFU_JOB="$(printf '%s\n' "$TOFU_OUTPUT" | tail -n 1)"

  MUSE_OUTPUT="$(submit_cell l3m_A 5000 NPO rtn_int8)"
  printf '%s\n' "$MUSE_OUTPUT"
  MUSE_JOB="$(printf '%s\n' "$MUSE_OUTPUT" | tail -n 1)"

  echo "TOFU_GT_JOB=$TOFU_JOB"
  echo "MUSE_GT_JOB=$MUSE_JOB"
else
  submit_cell l3_A 42 KLR gptq_int4
  submit_cell l3m_A 5000 NPO rtn_int8
  echo "GT test-only checks completed. Re-run with SUBMIT_MODE=submit after inspecting output."
fi
