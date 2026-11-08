#!/bin/bash
# B0.2 offline install + one-H200 runtime qualification. Not production.
# V2 freezes the active toolchain PATH and supersedes V1 before execution.
#SBATCH --job-name=org_b02_h200_env_v2
#SBATCH --account=anon_account
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=org_h200_execution_v1/slurm/%x_%j.out
#SBATCH --error=org_h200_execution_v1/slurm/%x_%j.err

set -euo pipefail
umask 077

ROOT=/work/anon_user/hessian_unlearning
TARGET=/work/anon_user/.runtime/envs/org-packed-h200-v1
PROTECTED=/home/anon_user/.conda/envs/unlearn
SOURCE_PY="$PROTECTED/bin/python"
CANDIDATE=org_h200_execution_v1/ORG_E2B_B02_H200_PACKED_ENV_CANDIDATE_V1.json
CERTIFIER=org_h200_execution_v1/scripts/certify_org_e2b_b02_h200_env_v1.py
LAUNCHER=org_h200_execution_v1/slurm/org_e2b_b02_isolated_env_h200_dev_v2.slurm

fatal() {
  printf 'FATAL: %s\n' "$*" >&2
  exit 32
}

active_real_path() {
  local tool=$1
  local raw
  raw=$(command -v -- "$tool") || return 1
  readlink -f -- "$raw"
}

assert_system_tool() {
  local tool=$1
  local resolved
  resolved=$(active_real_path "$tool") || fatal "required tool is unavailable on deterministic PATH: $tool"
  case "$resolved" in
    /usr/bin/*|/bin/*) ;;
    *) fatal "$tool escaped the frozen system-tool roots: $resolved" ;;
  esac
  printf '%s=%s\n' "$tool" "$resolved"
}

cd "$ROOT"
: "${B02_STAGE_A:?Submit with --export=ALL,B02_STAGE_A=results/.../b02_stage_a_resolution_TIMESTAMP}"
test -d "$B02_STAGE_A"
test -x "$SOURCE_PY"
test -f "$CANDIDATE"
test -f "$CERTIFIER"
test -f "$LAUNCHER"
test ! -e "$TARGET"
test "$TARGET" != "$PROTECTED"
test "$TARGET" = /work/anon_user/.runtime/envs/org-packed-h200-v1

# Stage A, not the display driver, decides the toolkit. Refuse to guess.
CUDA_MODULE=$("$SOURCE_PY" -c '
import json, re, sys
r=json.load(open(sys.argv[1], encoding="utf-8"))
c=r.get("cuda_module_inventory", {})
a=r.get("authorization", {})
m=c.get("selected")
assert c.get("status") == "PASS"
assert a.get("stage_b_h200_environment_smoke_allowed") is True
assert isinstance(m, str) and re.fullmatch(r"cuda/12\.6(?:\.\d+)?", m)
print(m)
' "$B02_STAGE_A/stage_a_receipt.json")
test -n "$CUDA_MODULE"
if ! type module >/dev/null 2>&1; then
  test -f /etc/profile.d/lmod.sh
  # shellcheck source=/dev/null
  source /etc/profile.d/lmod.sh
fi
module purge
module load "$CUDA_MODULE"
case ":${LOADEDMODULES:-}:" in
  *":$CUDA_MODULE:"*) ;;
  *) fatal 'frozen CUDA module is not in LOADEDMODULES' ;;
esac
command -v nvcc >/dev/null
nvcc --version | grep -F 'release 12.6'
NVCC_REAL=$(readlink -f -- "$(command -v nvcc)")
export CUDA_HOME
CUDA_HOME=$(dirname "$(dirname "$NVCC_REAL")")
test -x "$CUDA_HOME/bin/nvcc"

unset CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV PYTHONHOME PYTHONPATH BNB_CUDA_VERSION
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# The submit shell may be an activated historical environment. Discard its
# PATH completely. The target prefix is deliberately first even before it is
# created; all remaining roots are frozen explicitly.
export PATH="$TARGET/bin:$CUDA_HOME/bin:/usr/bin:/bin"
hash -r
case ":$PATH:" in
  *":$PROTECTED/bin:"*) fatal 'protected historical environment leaked into PATH' ;;
esac
test "$(active_real_path nvcc)" = "$NVCC_REAL"
assert_system_tool gcc >/dev/null
assert_system_tool g++ >/dev/null
assert_system_tool ld >/dev/null
assert_system_tool nvidia-smi >/dev/null

RUN_DIR="results/org_h200_execution_v1/b02_h200_environment_v2_${SLURM_JOB_ID}"
test ! -e "$RUN_DIR"

# If anything after the atomic target claim fails (including cancellation,
# probe, finalization, or evidence verification), preserve the newly created
# environment under a unique failure name. Never remove or overwrite it.
B02_COMMITTED=0
quarantine_b02_target() {
  rc=${1:-1}
  if [[ "$B02_COMMITTED" -ne 1 && -e "$TARGET" && -f "$RUN_DIR/target_claim.json" ]]; then
    FAILED_TARGET="${TARGET}.FAILED-${SLURM_JOB_ID}-launcher-v2"
    if [[ -e "$FAILED_TARGET" ]]; then
      printf 'FATAL: failure quarantine already exists; target left in place: %s\n' "$FAILED_TARGET" >&2
    else
      mv -- "$TARGET" "$FAILED_TARGET"
      printf 'B02_FAILED_ENVIRONMENT_QUARANTINED=%s\n' "$FAILED_TARGET" >&2
    fi
  fi
  trap - EXIT HUP INT TERM
  exit "$rc"
}
trap 'quarantine_b02_target $?' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

"$SOURCE_PY" "$CERTIFIER" install \
  --candidate "$CANDIDATE" \
  --stage-a "$B02_STAGE_A" \
  --run-dir "$RUN_DIR" \
  --environment-root "$TARGET"

export PIP_NO_INDEX=1
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/cache/inductor"
export TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
export GPTQMODEL_TORCH_EXTENSIONS_DIR="$RUN_DIR/cache/gptqmodel"
export CUDA_CACHE_PATH="$RUN_DIR/cache/cuda"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$GPTQMODEL_TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH"

# Re-resolve after installation and freeze the six tools consumed by B0.3.
# `Path.resolve()` in the Python certifier produces these same real paths.
hash -r
TARGET_REAL=$(readlink -f -- "$TARGET")
NINJA_REAL=$(active_real_path ninja) || fatal 'hash-locked target ninja is unavailable'
case "$NINJA_REAL" in
  "$TARGET_REAL"/*) ;;
  *) fatal "ninja is not contained in the isolated target: $NINJA_REAL" ;;
esac
test "$(active_real_path nvcc)" = "$NVCC_REAL"
{
  printf 'PATH=%s\n' "$PATH"
  printf 'ninja=%s\n' "$NINJA_REAL"
  printf 'nvcc=%s\n' "$(active_real_path nvcc)"
  assert_system_tool gcc
  assert_system_tool g++
  assert_system_tool ld
  assert_system_tool nvidia-smi
} > "$RUN_DIR/toolchain_active_paths_v2.txt"
sha256sum "$RUN_DIR/toolchain_active_paths_v2.txt" \
  > "$RUN_DIR/toolchain_active_paths_v2.txt.sha256"

"$TARGET/bin/python" "$CERTIFIER" probe \
  --candidate "$CANDIDATE" \
  --stage-a "$B02_STAGE_A" \
  --run-dir "$RUN_DIR" \
  --report "$RUN_DIR/h200_probe.json"

# Prove that the paths recorded by the independent Python probe are identical
# to the six paths resolved by this deterministic launcher environment.
"$SOURCE_PY" - "$RUN_DIR/toolchain_active_paths_v2.txt" "$RUN_DIR/h200_probe.json" <<'PY'
import json
import os
import sys

expected = {}
with open(sys.argv[1], encoding="utf-8") as handle:
    for raw in handle:
        key, value = raw.rstrip("\n").split("=", 1)
        if key != "PATH":
            expected[key] = os.path.realpath(value)
with open(sys.argv[2], encoding="utf-8") as handle:
    report = json.load(handle)
assert report.get("status") == "PASS", report.get("failures")
assert set(expected) == {"gcc", "g++", "ld", "nvcc", "ninja", "nvidia-smi"}
observed = report.get("toolchain", {})
for name, wanted in expected.items():
    row = observed.get(name, {})
    assert row.get("available") is True, (name, row)
    got = os.path.realpath(row.get("path") or "")
    assert got == wanted, (name, wanted, got)
print("B02_V2_TOOLCHAIN_ACTIVE_PATH_EQUALITY=PASS")
PY

"$SOURCE_PY" "$CERTIFIER" finalize \
  --candidate "$CANDIDATE" \
  --stage-a "$B02_STAGE_A" \
  --run-dir "$RUN_DIR" \
  --launcher "$LAUNCHER"

sha256sum -c "$RUN_DIR/evidence_sha256.txt"
sha256sum -c "$RUN_DIR/toolchain_active_paths_v2.txt.sha256"
find "$RUN_DIR/cache" -type f -print0 | sort -z | xargs -0 -r sha256sum \
  > "$RUN_DIR/compiled_cache_sha256.txt"
sha256sum "$RUN_DIR/final.json" "$RUN_DIR/B02_H200_ENVIRONMENT_SOURCE_LOCK_V1.json" \
  "$RUN_DIR/ORG_E2B_PACKED_H200_ENVIRONMENT_LOCK_V1.json" \
  "$RUN_DIR/compiled_cache_sha256.txt" \
  "$RUN_DIR/toolchain_active_paths_v2.txt"
B02_COMMITTED=1
trap - EXIT HUP INT TERM
printf 'ORG_E2B_B02_ISOLATED_H200_ENVIRONMENT_V2=PASS\n'
printf 'FINAL=%s/final.json\n' "$RUN_DIR"
printf 'B03_BACKEND_CANARIES_AUTHORIZED=true\n'
printf 'E2B_ATLAS_PRODUCTION_AUTHORIZED=false\n'
