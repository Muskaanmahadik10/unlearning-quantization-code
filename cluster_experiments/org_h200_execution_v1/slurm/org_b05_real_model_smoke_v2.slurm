#!/bin/bash
#SBATCH -A anon_account
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 00:20:00
#SBATCH -J org_b05_smoke_v2
#SBATCH -o logs/org_h200/org_b05_smoke_v2_%j.out
#SBATCH -e logs/org_h200/org_b05_smoke_v2_%j.err
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
sha256sum -c org_h200_execution_v1/ORG_B05_RTN_RELOAD_REPAIR_V2_SHA256.txt
"$V/bin/python" org_h200_execution_v1/scripts/test_org_b05_real_model_smoke_v1.py
"$V/bin/python" org_h200_execution_v1/scripts/test_org_b05_real_model_smoke_v2.py
OUT=results/org_h200_execution_v1/b05_real_model_smoke/rtn_int8_v2_${SLURM_JOB_ID}
"$V/bin/python" org_h200_execution_v1/scripts/run_org_b05_real_model_smoke_v2.py \
 --plan results/org_h200_execution_v1/b05_selector_locks/behavior_plan_v1.json \
 --cell ordinary_tofu_ga_seed42 --quantizer rtn_int8 --output "$OUT"
sha256sum "$OUT/final.json" > "$OUT/final.json.sha256"
