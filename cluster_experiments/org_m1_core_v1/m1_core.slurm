#!/bin/bash
#SBATCH --account=anon_account
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --job-name=org_m1_core_v1
#SBATCH --output=/work/anon_user/hessian_unlearning/logs/org_m1_core_v1/m1_core_%A_%a.out
#SBATCH --error=/work/anon_user/hessian_unlearning/logs/org_m1_core_v1/m1_core_%A_%a.err
set -euo pipefail
cd /work/anon_user/hessian_unlearning
module purge
module load cuda/12.6
M1_ENV=/work/anon_user/.runtime/envs/org-packed-h200-v1
export PATH="$M1_ENV/bin:/usr/local/cuda-12.6/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ORG_E2B_EXPECTED_ENVIRONMENT="$M1_ENV"
export PYTHONPATH="$PWD/org_m1_core_v1:$PWD/org_m1_core_v1/support"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
test -n "${M1_PREFLIGHT:-}"
test -n "${M1_ENDPOINT_CPU_RECEIPT:-}"
test -n "${M1_ENDPOINT_CPU_SHA256:-}"
sha256sum -c org_m1_core_v1/SOURCE_SHA256.txt
"$M1_ENV/bin/python" org_m1_core_v1/m1_core_v1.py \
  --index "$SLURM_ARRAY_TASK_ID" --preflight "$M1_PREFLIGHT" \
  --output "results/org_m1_core_v1/core_${SLURM_ARRAY_JOB_ID}/cell_${SLURM_ARRAY_TASK_ID}"
sha256sum -c org_m1_core_v1/SOURCE_SHA256.txt
