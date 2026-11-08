"""CPU admission followed by one guarded two-GPU O/R array, on explicit request."""
import argparse
import datetime
import io
import os
import re
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m2_stage_common_v1 import *

def parse_job(text):
    m=re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?\s*',text)
    require(m is not None,'ambiguous sbatch acknowledgement; no automatic retry');return m[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');p.add_argument('--accept-unverified-user-quota',action='store_true');a=p.parse_args()
    require(not os.environ.get('SLURM_JOB_ID'),'run on login node')
    require(not (RESULT/'launch_or_v1').exists(),'prior submission claim exists; reconcile before retry')
    torch,np,values,release=setup(False);validate_plan(release)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=RESULT/('cpu_'+stamp);out.mkdir(parents=True,exist_ok=False)
    suite=unittest.TestSuite()
    modules=['test_m2_metric_assembly_stdlib_v1','test_m2_metric_assembly_torch_v1','test_m2_training_release_stdlib_v1','test_m2_real_evaluator_torch_v1']
    for name in modules:suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (out/'CPU_TESTS.txt').write_text(stream.getvalue(),encoding='utf-8');print(stream.getvalue(),flush=True)
    require(result.wasSuccessful() and not result.skipped and result.testsRun==release['cluster_tests'],'CPU admission test failure')
    from m2_real_data_v1 import prepare
    binding=prepare(torch,values,out)
    require(not torch.cuda.is_initialized(),'CPU admission initialized GPU')
    # Per-user quota cannot be inferred from df. Preserve the diagnostic and
    # require the explicit risk acknowledgement rather than claiming quota PASS.
    quota={'verified':False,'risk_acknowledged':a.accept_unverified_user_quota}
    try:
        q=subprocess.run(['quota','-s'],capture_output=True,text=True,timeout=10)
        quota.update(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)
    except (OSError,subprocess.TimeoutExpired) as e:quota['diagnostic']=type(e).__name__
    save(out/'quota.json',quota)
    free=shutil.disk_usage(ROOT).free;require(free>=160*1024**3,'shared storage below 160 GiB; no submit')
    save(out/'final.json',{'status':'TRAINING_CPU_ADMISSION_PASS','tests':result.testsRun,'source_lock_sha256':source(),
      'data_binding_sha256':sha(out/'DATA_BINDING.json'),'gpu_submissions':0,'real_model_outcomes_inspected':False,
      'shared_free_bytes':free,'user_quota_verified':False,'training_only_gpu_hours_cap':4,'training_only_ntd_cap':120})
    print('M2_TRAINING_CPU_ADMISSION=PASS\nCPU_OUTPUT='+str(out),flush=True)
    if not a.submit:print('GPU_SUBMISSIONS=0');return
    require(a.accept_unverified_user_quota,'no quota certification; inspect quota.json and explicitly acknowledge before launch')
    q=subprocess.run(['squeue','-h','-u','anon_user','-o','%i|%j|%T'],capture_output=True,text=True,timeout=25)
    require(q.returncode==0 and 'org_m2_train_or' not in q.stdout,'queue unavailable or duplicate training')
    test=subprocess.run(['sbatch','--test-only',str(PKG/'m2_train_or.slurm')],capture_output=True,text=True,timeout=35,cwd=ROOT)
    (out/'scheduler_test.txt').write_text(test.stdout+test.stderr,encoding='utf-8')
    require(test.returncode==0,'scheduler test failed; no submission')
    claim=RESULT/'launch_or_v1';claim.mkdir(exist_ok=False)
    save(claim/'request.json',{'source_lock_sha256':source(),'cpu_receipt':str(out/'final.json'),'cpu_receipt_sha256':sha(out/'final.json'),
      'roles':['O','R'],'max_gpu_hours':4,'max_ntd':120,'user_quota_verified':False,'user_accepted_unverified_quota':True})
    (ROOT/'logs/org_m2_training_release_v1').mkdir(parents=True,exist_ok=True)
    env=dict(os.environ)
    for k in ('CUDA_VISIBLE_DEVICES','PYTHONPATH','PYTHONHOME','BNB_CUDA_VERSION'):env.pop(k,None)
    try:
        r=subprocess.run(['sbatch','--parsable',str(PKG/'m2_train_or.slurm')],capture_output=True,text=True,timeout=35,cwd=ROOT,env=env)
        (claim/'sbatch_stdout.txt').write_text(r.stdout);(claim/'sbatch_stderr.txt').write_text(r.stderr)
        require(r.returncode==0,'sbatch failed: '+r.stderr);jid=parse_job(r.stdout)
        save(claim/'submitted.json',{'job_id':jid,'roles':['O','R'],'gpus_per_task':1,'max_gpu_hours':4,'max_ntd':120})
        print('M2_OR_TRAINING_ARRAY='+jid+'\nTASK_0=O\nTASK_1=R\nMAX_NTD=120\nU_AUTOMATIC=false')
    except Exception as e:
        save(claim/'SUBMISSION_UNKNOWN.json',{'error':str(e),'retry_allowed':False});raise

if __name__=='__main__':main()
