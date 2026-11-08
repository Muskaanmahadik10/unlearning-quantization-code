#!/bin/bash
#SBATCH -A anon_account
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 00:10:00
#SBATCH -J org_b05_rtn_diag
#SBATCH -o logs/org_h200/org_b05_rtn_diag_%j.out
#SBATCH -e logs/org_h200/org_b05_rtn_diag_%j.err
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
sha256sum -c org_h200_execution_v1/ORG_B05_REAL_MODEL_SMOKE_SHA256.txt
OUT=results/org_h200_execution_v1/b05_real_model_smoke/rtn_reload_diag_${SLURM_JOB_ID}.json
"$V/bin/python" org_h200_execution_v1/scripts/diagnose_org_b05_rtn_reload_v1.py \
 --smoke-root results/org_h200_execution_v1/b05_real_model_smoke/rtn_int8_349160 \
 --plan results/org_h200_execution_v1/b05_selector_locks/behavior_plan_v1.json --output "$OUT"
sha256sum "$OUT" > "$OUT.sha256"
