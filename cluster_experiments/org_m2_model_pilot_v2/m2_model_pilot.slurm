#!/bin/bash
#SBATCH -A anon_account
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 00:30:00
#SBATCH --array=0-1%2
#SBATCH -J org_m2_pilot_v2
#SBATCH -o logs/org_m2_model_pilot_v2/pilot_%A_%a.out
#SBATCH -e logs/org_m2_model_pilot_v2/pilot_%A_%a.err
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
P=org_m2_model_pilot_v2
sha256sum -c "$P/SOURCE_SHA256.txt"
case "$SLURM_ARRAY_TASK_ID" in
  0) ROLE=R ;;
  1) ROLE=U ;;
  *) exit 64 ;;
esac
OUT="results/org_m2_model_pilot_v2/job_${SLURM_ARRAY_JOB_ID}/${ROLE}"
"$V/bin/python" -I -B "$P/m2_model_pilot_v1.py" --role "$ROLE" --output "$OUT"
sha256sum -c "$P/SOURCE_SHA256.txt"
