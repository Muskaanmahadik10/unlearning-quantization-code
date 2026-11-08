#!/bin/bash
set -euo pipefail
cd /work/anon_user/hessian_unlearning
sha256sum -c org_h200_execution_v1/ORG_B05_GPTQ_PAIR_REPAIR_V2_SHA256.txt
PY=/work/anon_user/.runtime/envs/org-packed-h200-v1/bin/python
"$PY" org_h200_execution_v1/scripts/test_org_b05_gptq_pair_repair_v2.py
bash -n org_h200_execution_v1/slurm/org_b05_gptq_pair_repair_v2.slurm
"$PY" org_h200_execution_v1/scripts/org_b05_gptq_pair_repair_v2.py prepare
"$PY" - <<'PY'
from pathlib import Path
import subprocess
root=Path('results/org_h200_execution_v1/b05_gptq_pair_repair/349840_1_v2')
with (root/'submission_attempt.txt').open('x') as f:
    f.write('One GPTQ pair retry; reconcile this attempt before any resubmission.\n')
try:
    r=subprocess.run(['sbatch','--parsable','org_h200_execution_v1/slurm/org_b05_gptq_pair_repair_v2.slurm'],capture_output=True,text=True,timeout=30)
except subprocess.TimeoutExpired:
    print('SUBMISSION_UNCERTAIN: reconcile queue/accounting; do not resubmit'); raise
(root/'submission_response.txt').write_text(r.stdout+'\n'+r.stderr)
if r.returncode:
    print(r.stdout,r.stderr); raise SystemExit(r.returncode)
job=r.stdout.strip().split(';')[0]
if not job.isdigit(): raise RuntimeError('Unexpected response; reconcile before retry')
print('GPTQ_PAIR_REPAIR_JOB='+job)
PY
