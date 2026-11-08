#!/bin/bash
#SBATCH -A anon_account
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 00:30:00
#SBATCH -J org_b05_parallel
#SBATCH -o logs/org_h200/org_b05_parallel_%A_%a.out
#SBATCH -e logs/org_h200/org_b05_parallel_%A_%a.err
set -euo pipefail
cd /work/anon_user/hessian_unlearning
module purge
module load cuda/12.6
V=/work/anon_user/.runtime/envs/org-packed-h200-v1
export PATH="$V/bin:/usr/local/cuda-12.6/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ORG_E2B_EXPECTED_ENVIRONMENT="$V"
export PYTHONPATH="$PWD/org_h200_execution_v1/scripts:$PWD/gate2_v4_final/scripts:$PWD/org_gb200_execution_v1/scripts"
sha256sum -c org_h200_execution_v1/ORG_B05_PARALLEL_HANDOFF_V1_SHA256.txt
"$V/bin/python" org_h200_execution_v1/scripts/org_b05_parallel_handoff_v1.py run --index "$SLURM_ARRAY_TASK_ID"
