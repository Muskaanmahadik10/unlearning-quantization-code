#!/bin/bash
#SBATCH -A anon_account
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 02:00:00
#SBATCH -J org_b05_fv2
#SBATCH -o logs/org_h200/org_b05_fv2_%j.out
#SBATCH -e logs/org_h200/org_b05_fv2_%j.err

set -euo pipefail
ROOT=/work/anon_user/hessian_unlearning
ENV_ROOT=/work/anon_user/.runtime/envs/org-packed-h200-v1
cd "$ROOT"
module purge
module load cuda/12.6
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ORG_E2B_EXPECTED_ENVIRONMENT="$ENV_ROOT"
export PATH="$ENV_ROOT/bin:/usr/local/cuda-12.6/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64
export PYTHONPATH="$ROOT/gate2_v4_final/scripts:$ROOT/org_gb200_execution_v1/scripts:$ROOT/org_h200_execution_v1/scripts"
mkdir -p logs/org_h200 results/org_h200_execution_v1/b05_feature_cells_v2

: "${B05_CELL:?set B05_CELL to ordinary or adverse}"
: "${B05_QUANTIZER:?set B05_QUANTIZER}"
: "${B05_REFERENCE_DIR:?set B05_REFERENCE_DIR}"

case "$B05_CELL" in
  ordinary)
    CELL_ID=ordinary_tofu_ga_seed42
    CHECKPOINT_REL=checkpoints/llama2_7b_adamw/unlearned/GA/seed_42
    FORGET=manifests/gate2_v4/tofu_forget_core128_seed20260625.json
    RETAIN=manifests/gate2_v4/tofu_retain_gptq_calib128_seed20260625.json
    ;;
  adverse)
    CELL_ID=adverse_muse_gdr_seed2024
    CHECKPOINT_REL=checkpoints/llama2_muse_adamw/unlearned/GDR/seed_2024
    FORGET=manifests/gate2_v4/muse_news_forget_core128_seed20260625.json
    RETAIN=manifests/gate2_v4/muse_news_retain_gptq_calib128_seed20260625.json
    ;;
  *) echo "invalid B05_CELL=$B05_CELL" >&2; exit 2 ;;
esac
case "$B05_QUANTIZER" in nf4|gptq_int4|rtn_int8) ;; *) echo "invalid quantizer" >&2; exit 2;; esac

SCRIPT=org_h200_execution_v1/scripts/run_org_e2b_b05_feature_cell_v2.py
SCRIPT_SHA=$(/usr/bin/sha256sum "$SCRIPT" | /usr/bin/awk '{print $1}')
REFERENCE="$B05_REFERENCE_DIR/${CELL_ID}_${B05_QUANTIZER}_current_reference_v1.json"
REFERENCE_MANIFEST="$B05_REFERENCE_DIR/current_runtime_reference_manifest_v1.json"
B04_FINAL=results/org_h200_execution_v1/b04_packed_public/dual_cert_gptqv2_348012_rtnv1_347943/final.json
NF4_B04_FINAL=results/org_h200_execution_v1/b04_public_representation/v1_346698/final.json
ENV_LOCK=results/org_h200_execution_v1/b02_h200_environment_v2_345586/ORG_E2B_PACKED_H200_ENVIRONMENT_LOCK_V1.json
LEDGER=org_h200_execution_v1/locks/ORG_H200_CORE72_CHECKPOINT_LEDGER_V1.json
LEDGER_CERT=org_h200_execution_v1/locks/ORG_H200_CORE72_CHECKPOINT_CERTIFICATION_V1.json
OUT=results/org_h200_execution_v1/b05_feature_cells_v2/production_${CELL_ID}_${B05_QUANTIZER}_${SLURM_JOB_ID}.json

for F in "$SCRIPT" "$REFERENCE" "$REFERENCE_MANIFEST" "$B04_FINAL" "$NF4_B04_FINAL" "$ENV_LOCK" "$LEDGER" "$LEDGER_CERT" "$FORGET" "$RETAIN"; do test -f "$F"; done
test -x "$ENV_ROOT/bin/python"
"$ENV_ROOT/bin/python" - "$REFERENCE_MANIFEST" "$REFERENCE" "$CELL_ID" "$B05_QUANTIZER" <<'PY'
import hashlib,json,pathlib,sys
m=pathlib.Path(sys.argv[1]); r=pathlib.Path(sys.argv[2]); cell=sys.argv[3]; q=sys.argv[4]
d=json.loads(m.read_text()); x=json.loads(r.read_text())
assert d["status"]=="PASS" and d["pre_outcome"] is True and d["b05_feature_v2_authorized"] is True
assert d["protected_outcomes_inspected"] is False and len(d["references"])==6
match=[z for z in d["references"] if z["cell_id"]==cell and z["quantizer"]==q]
assert len(match)==1 and hashlib.sha256(r.read_bytes()).hexdigest()==match[0]["sha256"]
assert x["cell_id"]==cell and x["quantizer"]==q and x["pre_outcome"] is True
print("B05_V2_REFERENCE_PREVERIFY=PASS")
PY
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader

"$ENV_ROOT/bin/python" "$SCRIPT" \
  --cell-id "$CELL_ID" --checkpoint-dir "$CHECKPOINT_REL" --checkpoint-relative "$CHECKPOINT_REL" \
  --checkpoint-ledger "$LEDGER" --checkpoint-certification "$LEDGER_CERT" \
  --b04-final "$B04_FINAL" --nf4-b04-final "$NF4_B04_FINAL" --environment-lock "$ENV_LOCK" \
  --forget-manifest "$FORGET" --retain-manifest "$RETAIN" --calibration-manifest "$RETAIN" \
  --metric-reference "$REFERENCE" --legacy-tensor-oracle "$REFERENCE" \
  --quantizer "$B05_QUANTIZER" --mode production \
  --n-samples 128 --n-retain 128 --n-calibration 128 \
  --expected-script-sha256 "$SCRIPT_SHA" --output "$OUT"

/usr/bin/sha256sum "$REFERENCE_MANIFEST" "$REFERENCE" "$OUT" > "${OUT}.sha256"
printf 'ORG_E2B_B05_FEATURE_CURRENT_REFERENCE_V2_SLURM=PASS\nOUTPUT=%s\n' "$OUT"
