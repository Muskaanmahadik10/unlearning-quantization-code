#!/bin/bash
#SBATCH --job-name=org_bundle_canary2
#SBATCH --account=anon_account
#SBATCH --partition=gb200-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=02:00:00
#SBATCH --output=logs/org_gb200/%x_%j.out
#SBATCH --error=logs/org_gb200/%x_%j.err

set -euo pipefail
cd /work/anon_user/hessian_unlearning
mkdir -p logs/org_gb200 results/org_gb200_execution_v1/gate2_v4_bridge_standard_bundled_canary2_v2
module purge
module load cuda/13.0
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH PYTHONHOME
V=/work/anon_user/.runtime/envs/unlearn-gb200-aarch64
H=/work/anon_user/.runtime/sysroots/python311-devel-aarch64/extracted/usr/include/python3.11
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$V/bin:$CUDA_HOME/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export CPATH="$H:/usr/include/python3.11${CPATH:+:$CPATH}"
export C_INCLUDE_PATH="$H:/usr/include/python3.11${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export TRITON_CACHE_DIR="results/org_gb200_execution_v1/cache/${SLURM_JOB_ID}/triton"
export TORCHINDUCTOR_CACHE_DIR="results/org_gb200_execution_v1/cache/${SLURM_JOB_ID}/inductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

PYTHON="$V/bin/python"
SCRIPT=org_gb200_execution_v1/scripts/run_org_bridge_bundle_v2.py
LOCK=org_gb200_execution_v1/locks/ORG_BUNDLED_CANARY_GB200_LOCK_V2.json
RANKINGS=org_gb200_execution_v1/preregistration/core72_v1/org_core72_rankings_v1.jsonl
RESULTS=results/org_gb200_execution_v1/gate2_v4_bridge_standard_bundled_canary2_v2
SCRIPT_SHA="$($PYTHON -c "import json; print(json.load(open('$LOCK'))['files']['$SCRIPT'])")"
SELECTORS=(GQDP_signed_v42 GQDP_signed_v42_transport_quantizer_cyclic_v1 legacy_UPQI_signed_v4 matched_random_type_depth_v1_seed00 matched_random_type_depth_v1_seed01)

run_cell() {
  local gpu="$1" model_dir="$2" model_tag="$3" dataset="$4" algo="$5" seed="$6" quantizer="$7"
  local forget="manifests/gate2_v4/${dataset}_forget_core128_seed20260625.json"
  local retain="manifests/gate2_v4/${dataset}_retain_gptq_calib128_seed20260625.json"
  local extra=()
  if [[ "$quantizer" == gptq_int4 ]]; then extra=(--calibration_manifest_path "$retain"); fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SCRIPT" \
    --model_dir "$model_dir" --model_tag "$model_tag" --dataset "$dataset" --algo "$algo" --seed "$seed" \
    --quantizer "$quantizer" --ranking_sources "${SELECTORS[@]}" --k_values 8 --rankings_path "$RANKINGS" \
    --manifest_path "$forget" --retain_manifest_path "$retain" "${extra[@]}" \
    --results_root "$RESULTS" --source_lock_path "$LOCK" --expected_script_sha256 "$SCRIPT_SHA" \
    --n_samples 128 --n_retain 128 --n_calibration 128 --manifest_n_samples 128 \
    --manifest_n_retain 128 --manifest_n_calibration 128 --max_length 512 --dtype bfloat16 \
    >"logs/org_gb200/${SLURM_JOB_ID}_${model_tag}_${algo}_${seed}_${quantizer}_bundle2.log" 2>&1
}

run_cell 0 checkpoints/llama3_8b_adamw/unlearned/NPO/seed_5000 llama3_8b_adamw tofu NPO 5000 gptq_int4 & p0=$!
(
  run_cell 1 checkpoints/llama3_muse_adamw/unlearned/GA/seed_2024 llama3_muse_adamw muse_news GA 2024 nf4
  PARITY_SCRIPT=org_gb200_execution_v1/scripts/test_org_gptq_production128_parity_v5.py
  PARITY_OUT=results/org_gb200_execution_v1/parity/gptq_production128_parity_v5_${SLURM_JOB_ID}.json
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$PARITY_SCRIPT" \
    --model_dir checkpoints/llama3_8b_adamw/unlearned/NPO/seed_5000 \
    --calibration_manifest manifests/gate2_v4/tofu_retain_gptq_calib128_seed20260625.json \
    --n_samples 128 --output "$PARITY_OUT" \
    >"logs/org_gb200/${SLURM_JOB_ID}_gptq_production128_parity.log" 2>&1
  sha256sum "$PARITY_OUT" > "${PARITY_OUT}.sha256"
) & p1=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
if [[ "$status" -ne 0 ]]; then echo ORG_BUNDLED_CANARY2_V2=FAIL; exit "$status"; fi
count="$(find "$RESULTS" -type f -name bridge_v4.json | wc -l)"
test "$count" -eq 12
echo ORG_BUNDLED_CANARY2_V2=PASS
echo "RESULT_COUNT=$count"
echo "RESULTS_ROOT=$RESULTS"
