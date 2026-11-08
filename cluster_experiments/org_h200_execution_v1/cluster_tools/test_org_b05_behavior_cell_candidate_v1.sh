#!/bin/bash
set -euo pipefail
cd /work/anon_user/hessian_unlearning
sha256sum -c org_h200_execution_v1/ORG_B05_BEHAVIOR_CELL_CANDIDATE_V1_SHA256.txt
V=/work/anon_user/.runtime/envs/org-packed-h200-v1
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1
"$V/bin/python" org_h200_execution_v1/scripts/test_org_b05_behavior_evaluator_v1.py --torch
"$V/bin/python" org_h200_execution_v1/scripts/test_org_b05_behavior_wave_contract_v1.py
"$V/bin/python" org_h200_execution_v1/scripts/test_org_b05_behavior_cell_v1.py
bash -n org_h200_execution_v1/slurm/org_b05_behavior_cell_candidate_v1.slurm
printf 'B05_CELL_CANDIDATE_CPU_TESTS=PASS\nGPU_SUBMISSIONS=0\nPRODUCTION_AUTHORIZED=false\n'
