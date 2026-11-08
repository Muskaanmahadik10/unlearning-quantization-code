#!/bin/bash
# Strict B0.3b only: isolated B0.2 runtime + model-level NF4 fresh roundtrip.
# A PASS is not B0 completion and never authorizes E2B/E3 production.
#SBATCH --job-name=org_b03b_nf4_v3
#SBATCH --account=anon_account
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=org_h200_execution_v1/slurm/%x_%j.out
#SBATCH --error=org_h200_execution_v1/slurm/%x_%j.err

set -euo pipefail
umask 077

ROOT=/work/anon_user/hessian_unlearning
ENV_ROOT=/work/anon_user/.runtime/envs/org-packed-h200-v1
PYTHON="$ENV_ROOT/bin/python"
PROBE=org_h200_execution_v1/scripts/probe_org_e2b_nf4_model_roundtrip_v1.py
LAUNCHER=org_h200_execution_v1/slurm/org_e2b_b0_nf4_model_roundtrip_h200_dev_v3.slurm
SOURCE_MANIFEST=org_h200_execution_v1/ORG_E2B_B03B_NF4_MODEL_ROUNDTRIP_SOURCE_SHA256_V1.txt
LIVE_PROBE=org_h200_execution_v1/scripts/probe_org_e2b_live_gpu_runtime_v2.py
LOW_PROBE=org_h200_execution_v1/scripts/probe_org_e2b_nf4_fresh_reload_v2.py
LOW_ENGINE=org_h200_execution_v1/scripts/probe_org_e2b_nf4_fresh_reload_v1.py
LOW_LAUNCHER=org_h200_execution_v1/slurm/org_e2b_b0_nf4_fresh_reload_h200_dev_v2.slurm
LOW_MANIFEST=org_h200_execution_v1/ORG_E2B_B0_NF4_FRESH_RELOAD_TRANSFER_SHA256_V2.txt

cd "$ROOT"
: "${LIVE_RUNTIME_RECEIPT:?Pass exact job-345096 live V2 receipt path as historical provenance}"
: "${LOW_LEVEL_FRESH_RECEIPT:?Pass exact job-345122 low-level final receipt path as historical provenance}"
: "${ENVIRONMENT_LOCK_RECEIPT:?Pass exact B0.2 Stage-B environment-lock PASS receipt path}"
: "${ENVIRONMENT_LOCK_BUILDER:?Pass exact B0.2 Stage-B certifier source path}"
: "${ENVIRONMENT_LOCK_SOURCE_MANIFEST:?Pass exact B0.2 source-manifest path}"
for path in "$PYTHON" "$PROBE" "$LAUNCHER" "$SOURCE_MANIFEST" "$LIVE_PROBE" \
  "$LOW_PROBE" "$LOW_ENGINE" "$LOW_LAUNCHER" "$LOW_MANIFEST" \
  "$LIVE_RUNTIME_RECEIPT" "$LOW_LEVEL_FRESH_RECEIPT" \
  "$ENVIRONMENT_LOCK_RECEIPT" "$ENVIRONMENT_LOCK_BUILDER" \
  "$ENVIRONMENT_LOCK_SOURCE_MANIFEST"; do
  test -f "$path" -o -x "$path"
done
sha256sum -c "$SOURCE_MANIFEST"

unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONHOME PYTHONPATH BNB_CUDA_VERSION
export PYTHONDONTWRITEBYTECODE=1
CUDA_MODULE=$(/usr/bin/python3 -c '
import json,re,sys
receipt=json.load(open(sys.argv[1],encoding="utf-8"))
module=receipt.get("toolchain",{}).get("cuda_module")
assert receipt.get("schema_version")=="org_e2b_packed_h200_environment_lock_v1"
assert receipt.get("overall_status")=="PASS"
assert isinstance(module,str) and re.fullmatch(r"cuda/12\.6(?:\.\d+)?",module)
print(module)
' "$ENVIRONMENT_LOCK_RECEIPT")
if ! type module >/dev/null 2>&1; then
  test -f /etc/profile.d/lmod.sh
  # shellcheck source=/dev/null
  source /etc/profile.d/lmod.sh
fi
module purge
module load "$CUDA_MODULE"
case ":${LOADEDMODULES:-}:" in
  *":$CUDA_MODULE:"*) ;;
  *) printf 'FATAL: frozen CUDA module is not loaded\n' >&2; exit 31 ;;
esac
NVCC_REAL=$(readlink -f "$(command -v nvcc)")
export CUDA_HOME
CUDA_HOME=$(dirname "$(dirname "$NVCC_REAL")")
export PATH="$ENV_ROOT/bin:$CUDA_HOME/bin:/usr/bin:/bin"
export PYTHONPATH="$ROOT/org_h200_execution_v1/scripts"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8

RUN_DIR="results/org_h200_execution_v1/b03b_nf4_model_roundtrip/strict_v3_${SLURM_JOB_ID}"
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/cache/inductor"
export TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
export CUDA_CACHE_PATH="$RUN_DIR/cache/cuda"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
BASELINE="$RUN_DIR/baseline"
CHECKPOINT="$RUN_DIR/packed_checkpoint"
REFERENCE="$RUN_DIR/numerical_reference.safetensors"
PRODUCER="$RUN_DIR/producer.json"
CONSUMER="$RUN_DIR/consumer.json"
FINAL="$RUN_DIR/final.json"

COMMON=(
  --execution-mode strict
  --environment-lock-receipt "$ENVIRONMENT_LOCK_RECEIPT"
  --environment-lock-builder "$ENVIRONMENT_LOCK_BUILDER"
  --environment-lock-source-manifest "$ENVIRONMENT_LOCK_SOURCE_MANIFEST"
  --live-runtime-receipt "$LIVE_RUNTIME_RECEIPT"
  --low-level-fresh-receipt "$LOW_LEVEL_FRESH_RECEIPT"
  --live-probe-script "$LIVE_PROBE"
  --low-level-probe-script "$LOW_PROBE"
  --low-level-engine-script "$LOW_ENGINE"
  --low-level-launcher "$LOW_LAUNCHER"
  --low-level-source-manifest "$LOW_MANIFEST"
  --source-manifest "$SOURCE_MANIFEST"
  --launcher "$LAUNCHER"
  --expected-live-job-id 345096
  --expected-fresh-job-id 345122
  --provenance-torch-version 2.5.1+cu124
  --provenance-bnb-version 0.45.0
  --expected-machine x86_64
  --expected-gpu-name-regex H200
  --expected-compute-capability 9.0
  --expected-environment-root "$ENV_ROOT"
  --expected-torch-version 2.8.0+cu126
  --expected-bnb-version 0.47.0
  --expected-transformers-version 4.56.2
  --expected-accelerate-version 1.10.1
  --expected-safetensors-version 0.6.2
  --expected-cuda-version 12.6
)

"$PYTHON" "$PROBE" produce "${COMMON[@]}" \
  --baseline-dir "$BASELINE" --checkpoint-dir "$CHECKPOINT" \
  --reference "$REFERENCE" --report "$PRODUCER" &
PRODUCER_PID=$!
wait "$PRODUCER_PID"
if kill -0 "$PRODUCER_PID" 2>/dev/null; then
  printf 'FAIL: producer remains alive: %s\n' "$PRODUCER_PID" >&2
  exit 1
fi
"$PYTHON" "$PROBE" reload "${COMMON[@]}" \
  --baseline-dir "$BASELINE" --checkpoint-dir "$CHECKPOINT" \
  --reference "$REFERENCE" --producer-report "$PRODUCER" \
  --expected-producer-pid "$PRODUCER_PID" --report "$CONSUMER"
"$PYTHON" "$PROBE" finalize "${COMMON[@]}" \
  --baseline-dir "$BASELINE" --checkpoint-dir "$CHECKPOINT" \
  --reference "$REFERENCE" --producer-report "$PRODUCER" \
  --consumer-report "$CONSUMER" --report "$FINAL"

"$PYTHON" - "$FINAL" "$ENVIRONMENT_LOCK_RECEIPT" <<'PY'
import hashlib, json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
with open(sys.argv[2], "rb") as handle:
    environment_sha = hashlib.sha256(handle.read()).hexdigest()
assert report["overall_status"] == "PASS"
assert report["qualification"] == "STRICT_B0_3B_CANARY_PASS"
assert report["certification"]["strict_b0_3b_model_roundtrip_certified"] is True
assert report["certification"]["eligible_as_strict_b0_3b_prerequisite_for_b0_4"] is True
assert report["scope_guards"]["strict_environment_lock_sha256"] == environment_sha
assert report["certification"]["b0_representation_certified"] is False
assert report["certification"]["production_authorized"] is False
assert report["certification"]["scientific_claim_authorized"] is False
PY
sha256sum -c "$SOURCE_MANIFEST"
find "$RUN_DIR" -type f ! -name evidence_sha256.txt -print0 | sort -z | xargs -0 sha256sum > "$RUN_DIR/evidence_sha256.txt"
sha256sum -c "$RUN_DIR/evidence_sha256.txt"
printf 'ORG_E2B_B03B_NF4_MODEL_ROUNDTRIP_STRICT_V3=PASS\nFINAL=%s\n' "$FINAL"
printf 'B0_REPRESENTATION_CERTIFIED=false\nPRODUCTION_AUTHORIZED=false\nSCIENTIFIC_CLAIM_AUTHORIZED=false\n'
