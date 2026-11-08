"""CPU-gated single U submission. Timeout never permits automatic resubmission."""
import argparse
import datetime
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m2_u_common_v1 import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');p.add_argument('--accept-unverified-user-quota',action='store_true');a=p.parse_args()
    require(not os.environ.get('SLURM_JOB_ID'),'run admission on login node')
    claim=RESULT/'launch_v1';require(not claim.exists(),'prior claim exists: reconcile, do not resubmit')
    torch,np,values,r=setup(False);parent_metadata(r)
    out=RESULT/('cpu_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    suite=unittest.TestSuite()
    for name in ('test_m2_training_torch_v1','test_m2_u_stdlib_v1','test_m2_u_torch_v1'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
    stream=io.StringIO();t=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (out/'CPU_TESTS.txt').write_text(stream.getvalue(),encoding='utf-8');print(stream.getvalue(),flush=True)
    require(t.wasSuccessful() and not t.skipped and t.testsRun==r['cpu_test_count'],'U CPU tests failed')
    require(not torch.cuda.is_initialized(),'CPU gate initialized CUDA')
    free=shutil.disk_usage(ROOT).free;require(free>=160*1024**3,'shared free space below 160 GiB')
    save(out/'final.json',{'status':'U_CPU_ADMISSION_PASS','source_lock_sha256':source(),'parent_model_sha256':r['parent_model_sha256'],
      'tests':t.testsRun,'shared_free_bytes':free,'user_quota_verified':False,'GPU_submissions':0,'real_model_outcomes_inspected':False})
    print('M2_U_CPU_ADMISSION=PASS\nCPU_OUTPUT='+str(out),flush=True)
    if not a.submit:print('GPU_SUBMISSIONS=0');return
    require(a.accept_unverified_user_quota,'personal quota unknown; explicit acknowledgement required')
    q=subprocess.run(['squeue','-h','-u','anon_user','-o','%i|%j|%T'],capture_output=True,text=True,timeout=25)
    require(q.returncode==0 and 'org_m2_train_u' not in q.stdout,'queue unavailable or duplicate U job')
    test=subprocess.run(['sbatch','--test-only',str(PKG/'m2_train_u.slurm')],capture_output=True,text=True,timeout=35,cwd=ROOT)
    (out/'scheduler_test.txt').write_text(test.stdout+test.stderr,encoding='utf-8');require(test.returncode==0,'scheduler test failed')
    claim.mkdir(exist_ok=False)
    save(claim/'request.json',{'source_lock_sha256':source(),'max_gpu_hours':2,'max_ntd':60,'accepted_unverified_quota':True,
      'cpu_receipt':str(out/'final.json'),'cpu_receipt_sha256':sha(out/'final.json'),'parent_model_sha256':r['parent_model_sha256']})
    (ROOT/'logs/org_m2_u_training_v1').mkdir(parents=True,exist_ok=True)
    from m2_training_release_v1 import parse_job
    env=dict(os.environ)
    for k in ('CUDA_VISIBLE_DEVICES','PYTHONPATH','PYTHONHOME','BNB_CUDA_VERSION'):env.pop(k,None)
    try:
        s=subprocess.run(['sbatch','--parsable',str(PKG/'m2_train_u.slurm')],capture_output=True,text=True,timeout=35,cwd=ROOT,env=env)
        (claim/'sbatch_stdout.txt').write_text(s.stdout);(claim/'sbatch_stderr.txt').write_text(s.stderr)
        require(s.returncode==0,'sbatch failed: '+s.stderr);jid=parse_job(s.stdout)
        save(claim/'submitted.json',{'job_id':jid,'role':'U','max_gpu_hours':2,'max_ntd':60})
        print('M2_U_TRAINING_JOB='+jid+'\nMAX_ADDITIONAL_NTD=60\nEVALUATION_LAUNCH_AUTHORIZED=false')
    except Exception as e:
        save(claim/'SUBMISSION_UNKNOWN.json',{'error':str(e),'automatic_retry':False});raise
if __name__=='__main__':main()
