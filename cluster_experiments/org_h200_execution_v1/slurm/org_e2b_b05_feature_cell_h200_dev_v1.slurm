#!/bin/bash
#SBATCH -A anon_account
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 02:00:00
#SBATCH -J org_b05_feat
#SBATCH -o logs/org_h200/org_b05_feat_%j.out
#SBATCH -e logs/org_h200/org_b05_feat_%j.err

set -euo pipefail
ROOT=/work/anon_user/hessian_unlearning
ENV_ROOT=/work/anon_user/.runtime/envs/org-packed-h200-v1
cd "$ROOT"
module purge
module load cuda/12.6
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ORG_E2B_EXPECTED_ENVIRONMENT="$ENV_ROOT"
export PATH="$ENV_ROOT/bin:/usr/local/cuda-12.6/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64"
export PYTHONPATH="$ROOT/gate2_v4_final/scripts:$ROOT/org_gb200_execution_v1/scripts:$ROOT/org_h200_execution_v1/scripts"
mkdir -p logs/org_h200 results/org_h200_execution_v1/b05_feature_cells

: "${B05_CELL:?set B05_CELL to ordinary or adverse}"
: "${B05_QUANTIZER:?set B05_QUANTIZER to nf4, gptq_int4, or rtn_int8}"
: "${B05_MODE:?set B05_MODE to smoke or production}"

case "$B05_CELL" in
  ordinary)
    CELL_ID=ordinary_tofu_ga_seed42
    CHECKPOINT_REL=checkpoints/llama2_7b_adamw/unlearned/GA/seed_42
    FORGET=manifests/gate2_v4/tofu_forget_core128_seed20260625.json
    RETAIN=manifests/gate2_v4/tofu_retain_gptq_calib128_seed20260625.json
    METRIC_CELL=llama2_7b_adamw_GA_seed42
    ;;
  adverse)
    CELL_ID=adverse_muse_gdr_seed2024
    CHECKPOINT_REL=checkpoints/llama2_muse_adamw/unlearned/GDR/seed_2024
    FORGET=manifests/gate2_v4/muse_news_forget_core128_seed20260625.json
    RETAIN=manifests/gate2_v4/muse_news_retain_gptq_calib128_seed20260625.json
    METRIC_CELL=llama2_muse_adamw_GDR_seed2024
    ;;
  *) echo "invalid B05_CELL=$B05_CELL" >&2; exit 2 ;;
esac

case "$B05_MODE" in
  smoke) N=2 ;;
  production) N=128 ;;
  *) echo "invalid B05_MODE=$B05_MODE" >&2; exit 2 ;;
esac

SCRIPT=org_h200_execution_v1/scripts/run_org_e2b_b05_feature_cell_v1.py
SCRIPT_SHA=$(/usr/bin/sha256sum "$SCRIPT" | /usr/bin/awk '{print $1}')
ORACLE_SCRIPT=org_h200_execution_v1/scripts/build_org_e2b_b05_legacy_tensor_oracle_v1.py
ORACLE_SCRIPT_SHA=$(/usr/bin/sha256sum "$ORACLE_SCRIPT" | /usr/bin/awk '{print $1}')
B04_FINAL=results/org_h200_execution_v1/b04_packed_public/dual_cert_gptqv2_348012_rtnv1_347943/final.json
NF4_B04_FINAL=results/org_h200_execution_v1/b04_public_representation/v1_346698/final.json
ENV_LOCK=results/org_h200_execution_v1/b02_h200_environment_v2_345586/ORG_E2B_PACKED_H200_ENVIRONMENT_LOCK_V1.json
LEDGER=org_h200_execution_v1/locks/ORG_H200_CORE72_CHECKPOINT_LEDGER_V1.json
LEDGER_CERT=org_h200_execution_v1/locks/ORG_H200_CORE72_CHECKPOINT_CERTIFICATION_V1.json
METRIC=results/gate2_v4_standard_metrics_v42_full144/${METRIC_CELL}/${B05_QUANTIZER}/metrics_v42.json
OUT=results/org_h200_execution_v1/b05_feature_cells/${B05_MODE}_${CELL_ID}_${B05_QUANTIZER}_${SLURM_JOB_ID}.json
ORACLE=results/org_h200_execution_v1/b05_feature_cells/${B05_MODE}_${CELL_ID}_${B05_QUANTIZER}_${SLURM_JOB_ID}.legacy_oracle.json

test -x "$ENV_ROOT/bin/python"
test -f "$B04_FINAL"
test -f "$NF4_B04_FINAL"
test -f "$ENV_LOCK"
test -f "$LEDGER"
test -f "$LEDGER_CERT"
test -f "$METRIC"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader

LEGACY_PYTHON=/home/anon_user/.conda/envs/unlearn/bin/python
test -x "$LEGACY_PYTHON"
"$LEGACY_PYTHON" "$ORACLE_SCRIPT" \
  --cell-id "$CELL_ID" \
  --checkpoint-dir "$CHECKPOINT_REL" \
  --calibration-manifest "$RETAIN" \
  --quantizer "$B05_QUANTIZER" \
  --n-calibration "$N" \
  --expected-script-sha256 "$ORACLE_SCRIPT_SHA" \
  --output "$ORACLE"

"$ENV_ROOT/bin/python" "$SCRIPT" \
  --cell-id "$CELL_ID" \
  --checkpoint-dir "$CHECKPOINT_REL" \
  --checkpoint-relative "$CHECKPOINT_REL" \
  --checkpoint-ledger "$LEDGER" \
  --checkpoint-certification "$LEDGER_CERT" \
  --b04-final "$B04_FINAL" \
  --nf4-b04-final "$NF4_B04_FINAL" \
  --environment-lock "$ENV_LOCK" \
  --forget-manifest "$FORGET" \
  --retain-manifest "$RETAIN" \
  --calibration-manifest "$RETAIN" \
  --metric-reference "$METRIC" \
  --legacy-tensor-oracle "$ORACLE" \
  --quantizer "$B05_QUANTIZER" \
  --mode "$B05_MODE" \
  --n-samples "$N" --n-retain "$N" --n-calibration "$N" \
  --expected-script-sha256 "$SCRIPT_SHA" \
  --output "$OUT"

/usr/bin/sha256sum "$ORACLE" "$OUT" > "${OUT}.sha256"
printf 'ORG_E2B_B05_FEATURE_SLURM=PASS\nOUTPUT=%s\n' "$OUT"
