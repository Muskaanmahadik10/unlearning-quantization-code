#!/bin/bash
#SBATCH --job-name=org_gptq_bound4
#SBATCH --account=anon_account
#SBATCH --partition=gb200-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=logs/org_gb200/%x_%j.out
#SBATCH --error=logs/org_gb200/%x_%j.err

set -euo pipefail
cd /work/anon_user/hessian_unlearning
mkdir -p logs/org_gb200 results/org_gb200_execution_v1/parity
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

SCRIPT=org_gb200_execution_v1/scripts/test_org_gptq_cpu_boundary_parity_v4.py
OUT=results/org_gb200_execution_v1/parity/gptq_cpu_boundary_parity_v4_${SLURM_JOB_ID}.json
sha256sum "$SCRIPT" > "${OUT}.script.sha256"
"$V/bin/python" "$SCRIPT" \
  --model_dir checkpoints/llama3_8b_adamw/unlearned/NPO/seed_5000 \
  --calibration_manifest manifests/gate2_v4/tofu_retain_gptq_calib128_seed20260625.json \
  --n_samples 1 \
  --output "$OUT"
sha256sum "$OUT" > "${OUT}.sha256"
