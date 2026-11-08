"""CPU checks and single, durable submission claim. No automatic retry."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m2_pilot_contract_v1 import *


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--submit',action='store_true')
    args=ap.parse_args()
    require(not (ROOT/'results/org_m2_model_pilot_v1/launch_v1').exists(),'V1 has a submission claim: reconcile it before any V2 launch')
    # No model load or GPU query during this pre-submission CPU step.
    from m2_model_pilot_v1 import setup
    torch,np,values=setup(False)
    plan=values[0]
    suite=unittest.TestSuite()
    for n in ('test_m2_manifest_repair_v2','test_m2_model_pilot_v1','test_m2_paged_cpu_v1','test_m2_training_contract_v1',
              'test_m2_training_torch_v1','test_m2_training_preflight_v1'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(n))
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+str(os.getpid())
    cpu=ROOT/'results/org_m2_model_pilot_v2'/('cpu_'+stamp);cpu.mkdir(parents=True,exist_ok=False)
    with (cpu/'CPU_TESTS.txt').open('x',encoding='utf-8') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    if not result.wasSuccessful():print((cpu/'CPU_TESTS.txt').read_text(),file=sys.stderr)
    require(result.wasSuccessful() and not result.skipped and result.testsRun==plan['cluster_tests'], 'pilot CPU tests failed; see '+str(cpu/'CPU_TESTS.txt'))
    require(not torch.cuda.is_initialized(),'GPU initialized during CPU preflight')
    save(cpu/'final.json',{'status':'PILOT_COMPONENT_CPU_PASS','tests':result.testsRun,'source_lock_sha256':source(),
                          'gpu_submissions':0,'production_authorized':False})
    print('M2_PILOT_CPU_TESTS=PASS\nCPU_OUTPUT='+str(cpu),flush=True)
    if not args.submit:
        print('GPU_SUBMISSIONS=0');return
    require(not os.environ.get('SLURM_JOB_ID'),'submit from login node, not nested allocation')
    # Query failures stop before the submission claim. Do not suppress stderr.
    queue=subprocess.run(['squeue','-h','-u','anon_user','-o','%i|%j|%T'],capture_output=True,text=True,timeout=25)
    require(queue.returncode==0,'scheduler query failed: '+queue.stderr)
    require(not any('org_m2_pilot_v2' in line for line in queue.stdout.splitlines()),'pilot job already queued')
    claim=ROOT/'results/org_m2_model_pilot_v2/launch_v1'
    claim.mkdir(exist_ok=False)
    save(claim/'request.json',{'status':'SUBMISSION_REQUEST_RECORDED','source_lock_sha256':source(),
         'cpu_receipt':str(cpu/'final.json'),'cpu_receipt_sha256':sha(cpu/'final.json'),
         'max_gpu_hours':1,'max_ntd':30,'training_scientific_states_authorized':False,
         'pilot_only_user_release':'user lets go after full-model pilot proposal',
         'shared_free_bytes_checked':shutil.disk_usage(ROOT).free,'user_quota_certified':False})
    (ROOT/'logs/org_m2_model_pilot_v2').mkdir(exist_ok=True)
    try:
        result=subprocess.run(['sbatch','--parsable',str(PKG/'m2_model_pilot.slurm')],
                              capture_output=True,text=True,timeout=35,cwd=ROOT,env=submission_env(os.environ))
        with (claim/'sbatch_stdout.txt').open('x') as f:f.write(result.stdout)
        with (claim/'sbatch_stderr.txt').open('x') as f:f.write(result.stderr)
        require(result.returncode==0,'submission rejected: '+result.stderr)
        jid=job_id(result.stdout)
        save(claim/'submitted.json',{'job_id':jid,'tasks':plan['tasks'],'gpu_hours_max':1,'ntd_max':30})
        print('M2_FULL_MODEL_PILOT_ARRAY='+jid+'\nTASK_0=supervised_R\nTASK_1=NPO_B_teacher_surrogate\nNTD_MAX=30\nPRODUCTION_AUTHORIZED=false')
    except Exception as e:
        save(claim/'SUBMISSION_UNKNOWN.json',{'error':str(e),'automatic_retry_allowed':False})
        raise


if __name__=='__main__':main()
