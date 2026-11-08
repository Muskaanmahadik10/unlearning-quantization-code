#!/bin/bash
#SBATCH --job-name=org_h200_canary8
#SBATCH --account=anon_account
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=700G
#SBATCH --time=02:00:00
#SBATCH --output=logs/org_h200/%x_%j.out
#SBATCH --error=logs/org_h200/%x_%j.err

set -euo pipefail
cd /work/anon_user/hessian_unlearning
mkdir -p logs/org_h200 results/org_h200_execution_v1/{parity,cache}

PYTHON=/home/anon_user/.conda/envs/unlearn/bin/python
BUNDLE_SCRIPT=org_gb200_execution_v1/scripts/run_org_bridge_bundle_v2.py
LEGACY_SCRIPT=gate2_v4_final/scripts/run_gate2_whole_model_bridge_v4.py
PARITY128_SCRIPT=org_gb200_execution_v1/scripts/test_org_gptq_production128_parity_v5.py
BOUNDARY_SCRIPT=org_gb200_execution_v1/scripts/test_org_gptq_cpu_boundary_parity_v4.py
VALIDATOR=org_h200_execution_v1/scripts/validate_org_h200_adversarial_canary_v1.py
LOCK=org_h200_execution_v1/locks/ORG_H200_ADVERSARIAL_CANARY8_LOCK_V1.json
RANKINGS=org_gb200_execution_v1/preregistration/core72_v1/org_core72_rankings_v1.jsonl
BUNDLED_ROOT=results/org_h200_execution_v1/gate2_v4_bridge_standard_h200_canary8_v1
LEGACY_ROOT=results/org_h200_execution_v1/gate2_v4_bridge_standard_h200_legacy_canary_v1
REPEAT_ROOT=results/org_h200_execution_v1/gate2_v4_bridge_standard_h200_repeat_canary_v1
PARITY128_OUT=results/org_h200_execution_v1/parity/gptq_production128_parity_h200_${SLURM_JOB_ID}.json
BOUNDARY_OUT=results/org_h200_execution_v1/parity/gptq_boundary224_h200_${SLURM_JOB_ID}.json
VALIDATION_OUT=results/org_h200_execution_v1/parity/h200_adversarial_canary_validation_${SLURM_JOB_ID}.json

test -x "$PYTHON"
test -f "$LOCK"
mkdir -p "$BUNDLED_ROOT" "$LEGACY_ROOT" "$REPEAT_ROOT"
export PYTHONNOUSERSITE=1
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONHOME
export PYTHONPATH=gate2_v4_final/scripts:org_gb200_execution_v1/scripts

BUNDLE_SHA="$($PYTHON -c "import json; print(json.load(open('$LOCK'))['files']['$BUNDLE_SCRIPT'])")"
LEGACY_SHA="$($PYTHON -c "import json; print(json.load(open('$LOCK'))['files']['$LEGACY_SCRIPT'])")"
LOCK_SHA="$($PYTHON -c "import json; print(json.load(open('$LOCK'))['source_lock_sha256'])")"
SELECTORS=(GQDP_signed_v42 GQDP_signed_v42_transport_quantizer_cyclic_v1 legacy_UPQI_signed_v4 matched_random_type_depth_v1_seed00 matched_random_type_depth_v1_seed01)

sha256sum "$0" | tee "logs/org_h200/${SLURM_JOB_ID}_launcher.sha256"
uname -a | tee "logs/org_h200/${SLURM_JOB_ID}_uname.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap \
  --format=csv,noheader | tee "logs/org_h200/${SLURM_JOB_ID}_gpus.txt"
"$PYTHON" - <<'PY' | tee "logs/org_h200/${SLURM_JOB_ID}_runtime.txt"
import bitsandbytes, torch, transformers
print("python=" + __import__("sys").version.replace("\n", " "))
print("torch=" + torch.__version__)
print("cuda=" + str(torch.version.cuda))
print("transformers=" + transformers.__version__)
print("bitsandbytes=" + bitsandbytes.__version__)
PY

gpu_env() {
  local gpu="$1"
  local root="results/org_h200_execution_v1/cache/${SLURM_JOB_ID}/gpu${gpu}"
  mkdir -p "$root/triton" "$root/inductor"
  printf '%s\t%s\n' "$root/triton" "$root/inductor"
}

run_bundle() {
  local gpu="$1" model_dir="$2" model_tag="$3" dataset="$4" algo="$5" seed="$6" quantizer="$7" root="$8" label="$9"
  local forget="manifests/gate2_v4/${dataset}_forget_core128_seed20260625.json"
  local retain="manifests/gate2_v4/${dataset}_retain_gptq_calib128_seed20260625.json"
  local cache triton inductor
  cache="$(gpu_env "$gpu")"
  IFS=$'\t' read -r triton inductor <<< "$cache"
  local extra=()
  if [[ "$quantizer" == gptq_int4 ]]; then extra=(--calibration_manifest_path "$retain"); fi
  CUDA_VISIBLE_DEVICES="$gpu" TRITON_CACHE_DIR="$triton" TORCHINDUCTOR_CACHE_DIR="$inductor" \
    /usr/bin/time -v "$PYTHON" "$BUNDLE_SCRIPT" \
      --model_dir "$model_dir" --model_tag "$model_tag" --dataset "$dataset" --algo "$algo" --seed "$seed" \
      --quantizer "$quantizer" --ranking_sources "${SELECTORS[@]}" --k_values 8 --rankings_path "$RANKINGS" \
      --manifest_path "$forget" --retain_manifest_path "$retain" "${extra[@]}" \
      --results_root "$root" --source_lock_path "$LOCK" --expected_script_sha256 "$BUNDLE_SHA" \
      --n_samples 128 --n_retain 128 --n_calibration 128 --manifest_n_samples 128 \
      --manifest_n_retain 128 --manifest_n_calibration 128 --max_length 512 --dtype bfloat16 \
      >"logs/org_h200/${SLURM_JOB_ID}_${label}.log" 2>&1
}

run_legacy() {
  local gpu="$1" model_dir="$2" model_tag="$3" dataset="$4" algo="$5" seed="$6" quantizer="$7" label="$8"
  local forget="manifests/gate2_v4/${dataset}_forget_core128_seed20260625.json"
  local retain="manifests/gate2_v4/${dataset}_retain_gptq_calib128_seed20260625.json"
  local cache triton inductor
  cache="$(gpu_env "$gpu")"
  IFS=$'\t' read -r triton inductor <<< "$cache"
  local common=(
    --model_dir "$model_dir" --model_tag "$model_tag" --dataset "$dataset" --algo "$algo" --seed "$seed"
    --quantizer "$quantizer" --rankings_path "$RANKINGS" --manifest_path "$forget" --retain_manifest_path "$retain"
    --results_root "$LEGACY_ROOT" --source_lock_path "$LOCK" --expected_script_sha256 "$LEGACY_SHA"
    --n_samples 128 --n_retain 128 --n_calibration 128 --manifest_n_samples 128
    --manifest_n_retain 128 --manifest_n_calibration 128 --max_length 512 --dtype bfloat16
  )
  {
    CUDA_VISIBLE_DEVICES="$gpu" TRITON_CACHE_DIR="$triton" TORCHINDUCTOR_CACHE_DIR="$inductor" \
      "$PYTHON" "$LEGACY_SCRIPT" "${common[@]}" --condition full_quantized
    local source
    for source in "${SELECTORS[@]}"; do
      CUDA_VISIBLE_DEVICES="$gpu" TRITON_CACHE_DIR="$triton" TORCHINDUCTOR_CACHE_DIR="$inductor" \
        "$PYTHON" "$LEGACY_SCRIPT" "${common[@]}" --condition protected --ranking_source "$source" --k 8
    done
  } >"logs/org_h200/${SLURM_JOB_ID}_${label}.log" 2>&1
}

pids=()
run_bundle 0 checkpoints/llama3_8b_adamw/unlearned/NPO/seed_5000 llama3_8b_adamw tofu NPO 5000 gptq_int4 "$BUNDLED_ROOT" bundle_gptq & pids+=("$!")
run_bundle 1 checkpoints/llama3_muse_adamw/unlearned/GA/seed_2024 llama3_muse_adamw muse_news GA 2024 nf4 "$BUNDLED_ROOT" bundle_nf4 & pids+=("$!")
run_bundle 2 checkpoints/llama2_7b_adamw/unlearned/GA/seed_2024 llama2_7b_adamw tofu GA 2024 rtn_int8 "$BUNDLED_ROOT" bundle_rtn & pids+=("$!")

(
  cache="$(gpu_env 3)"; IFS=$'\t' read -r triton inductor <<< "$cache"
  CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR="$triton" TORCHINDUCTOR_CACHE_DIR="$inductor" \
    /usr/bin/time -v "$PYTHON" "$PARITY128_SCRIPT" \
      --model_dir checkpoints/llama3_8b_adamw/unlearned/NPO/seed_5000 \
      --calibration_manifest manifests/gate2_v4/tofu_retain_gptq_calib128_seed20260625.json \
      --n_samples 128 --output "$PARITY128_OUT"
  sha256sum "$PARITY128_OUT" > "${PARITY128_OUT}.sha256"
) >"logs/org_h200/${SLURM_JOB_ID}_parity128.log" 2>&1 & pids+=("$!")

run_legacy 4 checkpoints/llama3_muse_adamw/unlearned/GA/seed_2024 llama3_muse_adamw muse_news GA 2024 nf4 legacy_nf4 & pids+=("$!")
run_legacy 5 checkpoints/llama2_7b_adamw/unlearned/GA/seed_2024 llama2_7b_adamw tofu GA 2024 rtn_int8 legacy_rtn & pids+=("$!")

(
  cache="$(gpu_env 6)"; IFS=$'\t' read -r triton inductor <<< "$cache"
  CUDA_VISIBLE_DEVICES=6 TRITON_CACHE_DIR="$triton" TORCHINDUCTOR_CACHE_DIR="$inductor" \
    /usr/bin/time -v "$PYTHON" "$BOUNDARY_SCRIPT" \
      --model_dir checkpoints/llama3_8b_adamw/unlearned/NPO/seed_5000 \
      --calibration_manifest manifests/gate2_v4/tofu_retain_gptq_calib128_seed20260625.json \
      --n_samples 1 --output "$BOUNDARY_OUT"
  sha256sum "$BOUNDARY_OUT" > "${BOUNDARY_OUT}.sha256"
) >"logs/org_h200/${SLURM_JOB_ID}_boundary224.log" 2>&1 & pids+=("$!")

run_bundle 7 checkpoints/llama2_7b_adamw/unlearned/GA/seed_2024 llama2_7b_adamw tofu GA 2024 rtn_int8 "$REPEAT_ROOT" repeat_rtn & pids+=("$!")

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -ne 0 ]]; then echo ORG_H200_ADVERSARIAL_CANARY8=WORKER_FAIL; exit 1; fi

"$PYTHON" "$VALIDATOR" \
  --bundled_root "$BUNDLED_ROOT" --legacy_root "$LEGACY_ROOT" --repeat_root "$REPEAT_ROOT" \
  --source_lock "$LOCK" --parity128 "$PARITY128_OUT" --boundary224 "$BOUNDARY_OUT" \
  --output "$VALIDATION_OUT"
find "$BUNDLED_ROOT" "$LEGACY_ROOT" "$REPEAT_ROOT" -type f -name bridge_v4.json -print0 \
  | sort -z | xargs -0 sha256sum > "results/org_h200_execution_v1/parity/h200_canary_result_sha256_${SLURM_JOB_ID}.txt"
echo ORG_H200_ADVERSARIAL_CANARY8=PASS
echo VALIDATION="$VALIDATION_OUT"
