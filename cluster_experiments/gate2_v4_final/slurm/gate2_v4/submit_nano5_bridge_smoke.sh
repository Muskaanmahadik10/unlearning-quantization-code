#!/bin/bash
set -euo pipefail

cd "${WORKDIR:-/work/anon_user/hessian_unlearning}"

PY="${PY:-/work/anon_user/.runtime/envs/unlearn-nano5/bin/python}"
BRIDGE_LOCK="${BRIDGE_LOCK:?Set BRIDGE_LOCK to the timestamped bridge lock created after 144/144 joins pass}"
ACCOUNT="${ACCOUNT:-anon_account}"
PARTITION="${PARTITION:-dev}"
TIME="${TIME:-02:00:00}"
MODE="${SUBMIT_MODE:-test-only}"

[[ -x "$PY" ]] || { echo "Missing Nano5 Python: $PY" >&2; exit 2; }
[[ -f "$BRIDGE_LOCK" ]] || { echo "Missing bridge lock: $BRIDGE_LOCK" >&2; exit 2; }

PY="$PY" BRIDGE_LOCK="$BRIDGE_LOCK" SUBMIT_MODE="$MODE" \
ACCOUNT="$ACCOUNT" PARTITION="$PARTITION" TIME="$TIME" \
GPUS=8 CPUS=64 MEM=0 \
bash gate2_v4_final/slurm/gate2_v4/submit_bridge_smoke_20260801.sh
