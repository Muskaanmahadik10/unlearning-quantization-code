#!/bin/bash
# Read-only scheduler accounting and CPU tests. No submission or model loading.
set -euo pipefail
cd /work/anon_user/hessian_unlearning
sha256sum -c org_h200_execution_v1/ORG_B05_BEHAVIOR_EVALUATOR_V1_SHA256.txt
V=/work/anon_user/.runtime/envs/org-packed-h200-v1
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 "$V/bin/python" \
 org_h200_execution_v1/scripts/test_org_b05_behavior_evaluator_v1.py --torch
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 "$V/bin/python" \
 org_h200_execution_v1/scripts/test_org_b05_behavior_rtn_canary_v1.py
bash -n org_h200_execution_v1/slurm/org_b05_behavior_rtn_canary_v1.slurm
printf '\n=== ALLOCATION ACCOUNTING (TOP LEVEL ONLY) ===\n'
timeout 30s sacct -X -P -n -u anon_user -S 2026-09-06T00:00:00 \
 --format=JobIDRaw,JobName%60,State,ElapsedRaw,AllocTRES%300
printf '\nB05_BEHAVIOR_READINESS=TESTS_COMPLETE_ACCOUNTING_REVIEW_REQUIRED\n'
printf 'GPU_SUBMISSIONS=0\nPRODUCTION_AUTHORIZED=false\n'
