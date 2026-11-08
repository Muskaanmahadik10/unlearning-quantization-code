#!/bin/bash
set -euo pipefail
cd /work/anon_user/hessian_unlearning
sha256sum -c org_h200_execution_v1/ORG_B05_FINISH_WAVE_V1_SHA256.txt
PY=/work/anon_user/.runtime/envs/org-packed-h200-v1/bin/python
"$PY" org_h200_execution_v1/scripts/test_org_b05_finish_wave_v1.py
"$PY" org_h200_execution_v1/scripts/test_org_b05_gptq_pair_repair_v2.py
bash -n org_h200_execution_v1/slurm/org_b05_finish_wave_v1.slurm
"$PY" org_h200_execution_v1/scripts/org_b05_finish_wave_v1.py prepare
"$PY" org_h200_execution_v1/scripts/submit_org_b05_finish_wave_v1.py
