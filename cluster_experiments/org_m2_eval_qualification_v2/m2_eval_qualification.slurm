#!/bin/bash
#SBATCH --job-name=org_m2_eval_qual
#SBATCH --account=anon_account
#SBATCH --partition=dev
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=00:15:00
#SBATCH --output=/work/anon_user/hessian_unlearning/logs/org_m2_eval_qualification_v2/eval_%A_%a.out
#SBATCH --error=/work/anon_user/hessian_unlearning/logs/org_m2_eval_qualification_v2/eval_%A_%a.err
set -euo pipefail
cd /work/anon_user/hessian_unlearning
module purge
module load cuda/12.6
V=/work/anon_user/.runtime/envs/org-packed-h200-v1
export PATH="$V/bin:/usr/local/cuda-12.6/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1 PYTHONHASHSEED=20260909
unset PYTHONPATH PYTHONHOME BNB_CUDA_VERSION
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
case "$SLURM_ARRAY_TASK_ID" in
  0) ROLE=B ;;
  1) ROLE=O ;;
  2) ROLE=R ;;
  3) ROLE=U ;;
  *) exit 2 ;;
esac
"$V/bin/python" -I -B org_m2_eval_qualification_v2/m2_eval_qualification_v1.py \
  --output "results/org_m2_eval_qualification_v2/job_${SLURM_ARRAY_JOB_ID}/${ROLE}"
