#!/bin/bash
#SBATCH --job-name=org_m2_train_u
#SBATCH --account=anon_account
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/work/anon_user/hessian_unlearning/logs/org_m2_u_training_v1/train_%j.out
#SBATCH --error=/work/anon_user/hessian_unlearning/logs/org_m2_u_training_v1/train_%j.err
set -euo pipefail
cd /work/anon_user/hessian_unlearning
module purge
module load cuda/12.6
V=/work/anon_user/.runtime/envs/org-packed-h200-v1
export PATH="$V/bin:/usr/local/cuda-12.6/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1 PYTHONHASHSEED=20260909
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME BNB_CUDA_VERSION
"$V/bin/python" -I -B org_m2_u_training_v1/m2_train_u_v1.py \
  --output "results/org_m2_u_training_v1/job_${SLURM_JOB_ID}/U"
