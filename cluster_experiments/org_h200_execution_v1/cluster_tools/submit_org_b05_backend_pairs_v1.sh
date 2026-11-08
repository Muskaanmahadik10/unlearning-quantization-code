#!/bin/bash
set -euo pipefail
cd /work/anon_user/hessian_unlearning
sha256sum -c org_h200_execution_v1/ORG_B05_REMAINING_BACKENDS_V1_SHA256.txt
PY=/work/anon_user/.runtime/envs/org-packed-h200-v1/bin/python
"$PY" org_h200_execution_v1/scripts/test_org_b05_remaining_backends_v1.py
bash -n org_h200_execution_v1/slurm/org_b05_backend_pair_parallel_v1.slurm
"$PY" org_h200_execution_v1/scripts/org_b05_remaining_backends_v1.py prepare
"$PY" - <<'PY'
import pathlib,subprocess
root=pathlib.Path('results/org_h200_execution_v1/b05_remaining_backends_v1')
with (root/'submission_attempt.txt').open('x') as f:
    f.write('Two-task backend validation submission attempted; reconcile before retry.\n')
try:
    r=subprocess.run(['sbatch','--parsable','org_h200_execution_v1/slurm/org_b05_backend_pair_parallel_v1.slurm'],
                     capture_output=True,text=True,timeout=30)
except subprocess.TimeoutExpired:
    print('SUBMISSION_UNCERTAIN: do not resubmit'); raise
(root/'submission_response.txt').write_text(r.stdout+'\n'+r.stderr)
print(r.stdout,r.stderr)
if r.returncode: raise SystemExit(r.returncode)
job=r.stdout.strip().split(';')[0]
if not job.isdigit(): raise RuntimeError('Unexpected response; reconcile before retry')
print('B05_BACKEND_PAIR_ARRAY='+job)
print('TASK_0=ordinary_nf4_pair\nTASK_1=ordinary_gptq_pair')
PY
