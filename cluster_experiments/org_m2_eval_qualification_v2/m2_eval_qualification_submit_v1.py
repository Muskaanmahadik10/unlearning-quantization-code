"""CPU tests and evidence checks first; one explicit, fail-closed array submit."""
import argparse
import datetime
import io
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m2_eval_qualification_common_v1 import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');p.add_argument('--stage',choices=['canary','remaining'],required=True);a=p.parse_args()
    from m2_eval_repair_guard_v2 import STAGES,reconcile_prior,verify_canary
    indices=STAGES[a.stage];limit_hours=len(indices)/4;limit_ntd=limit_hours*30
    require(not os.environ.get('SLURM_JOB_ID'),'run on login node')
    require(not (RESULT/('launch_'+a.stage)).exists(),'prior submission claim exists; reconcile without resubmitting')
    torch,np,values,r=setup(False)
    prior=reconcile_prior()
    canary=verify_canary() if a.stage=='remaining' else None
    committed=1 if canary is None else canary['gpu_hours']+limit_hours
    require(prior['gpu_hours']+committed<=r['bucket_cap_gpu_hours'],'qualification allocation exceeded; revise budget before submission')
    out=RESULT/('cpu_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    for role in ROLES:checkpoint(role,r,full=True)
    from m2_stage_common_v1 import validate_binding
    binding=read(ROOT/r['training_cpu']/'DATA_BINDING.json');validate_binding(binding)
    tokens=ROOT/r['training_cpu']/'EVALUATION_TOKENS.json'
    require(sha(tokens)==binding['tokens_sha256']==r['evaluation_tokens_sha256'],'full evaluation token binding changed')
    panels=fixture(values[5]);require(canonical(panels)==r['fixture_canonical_sha256'],'fixture definition changed')
    save(out/'FIXTURE_PANELS.json',panels)
    save(out/'PRIOR_FAILURE_ACCOUNTING.json',prior)
    if canary is not None:save(out/'ACCEPTED_CANARY.json',canary)
    suite=unittest.TestSuite()
    for name in ('test_m2_metric_assembly_stdlib_v1','test_m2_metric_assembly_torch_v1','test_m2_real_evaluator_torch_v1','test_m2_eval_qualification_v1','test_m2_eval_qualification_torch_v1','test_m2_eval_repair_guard_v2','test_m2_eval_oracle_repair_torch_v2'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
    import contextlib
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (out/'CPU_TESTS.txt').write_text(stream.getvalue(),encoding='utf-8');print(stream.getvalue(),flush=True)
    require(result.wasSuccessful() and not result.skipped and result.testsRun==r['cluster_tests'],'CPU admission tests failed')
    require(not torch.cuda.is_initialized(),'CPU gate initialized GPU')
    free=shutil.disk_usage(ROOT).free;require(free>=160*1024**3,'shared free space below inherited threshold')
    save(out/'final.json',{'status':'CPU_QUALIFICATION_ADMISSION_PASS','tests':result.testsRun,'source_lock_sha256':source(),
       'fixture_sha256':sha(out/'FIXTURE_PANELS.json'),'all_three_checkpoints_rehashed':True,'shared_free_bytes':free,
       'quota_verified':False,'source_of_quota_risk_acceptance':'accepted training request; new output is small receipts only',
       'gpu_submissions':0,'scientific_evaluation_authorized':False})
    print('M2_EVALUATOR_CPU_ADMISSION=PASS\nCPU_OUTPUT='+str(out),flush=True)
    if not a.submit:print('GPU_SUBMISSIONS=0');return
    q=subprocess.run(['squeue','-h','-u','anon_user','-o','%i|%j|%T'],capture_output=True,text=True,timeout=25)
    require(q.returncode==0 and 'org_m2_eval_qual' not in q.stdout,'queue unavailable or duplicate evaluator job')
    test=subprocess.run(['sbatch','--test-only','--array='+','.join(map(str,indices)),str(PKG/'m2_eval_qualification.slurm')],capture_output=True,text=True,cwd=ROOT,timeout=35)
    (out/'scheduler_test.txt').write_text(test.stdout+test.stderr,encoding='utf-8');require(test.returncode==0,'scheduler test failed; no submission')
    claim=RESULT/('launch_'+a.stage);claim.mkdir(exist_ok=False)
    save(claim/'request.json',{'source_lock_sha256':source(),'cpu_receipt':str(out/'final.json'),'cpu_receipt_sha256':sha(out/'final.json'),
      'roles':list(ROLES),'indices':list(indices),'stage':a.stage,'max_gpu_hours':limit_hours,'max_ntd':limit_ntd,'scope':SCOPE})
    (ROOT/'logs/org_m2_eval_qualification_v2').mkdir(parents=True,exist_ok=True)
    env=dict(os.environ)
    for k in ('CUDA_VISIBLE_DEVICES','PYTHONPATH','PYTHONHOME','BNB_CUDA_VERSION'):env.pop(k,None)
    try:
        s=subprocess.run(['sbatch','--parsable','--array='+','.join(map(str,indices)),str(PKG/'m2_eval_qualification.slurm')],capture_output=True,text=True,cwd=ROOT,env=env,timeout=35)
        (claim/'sbatch_stdout.txt').write_text(s.stdout,encoding='utf-8');(claim/'sbatch_stderr.txt').write_text(s.stderr,encoding='utf-8')
        require(s.returncode==0,'submission failed: '+s.stderr);jid=job_id(s.stdout)
        save(claim/'submitted.json',{'job_id':jid,'roles':list(ROLES),'indices':list(indices),'stage':a.stage,'max_gpu_hours':limit_hours,'max_ntd':limit_ntd})
        print('M2_EVALUATOR_REPAIR_'+a.stage.upper()+'_ARRAY='+jid+'\nINDICES='+','.join(map(str,indices))+'\nMAX_ADDITIONAL_NTD='+str(limit_ntd)+'\nSCIENTIFIC_EVALUATION_AUTHORIZED=false')
    except Exception as e:
        save(claim/'SUBMISSION_UNKNOWN.json',{'error':str(e),'automatic_retry':False});raise

if __name__=='__main__':main()
