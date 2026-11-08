"""Four concurrent real BF16 models, two fresh calibration-fixture passes each."""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m2_eval_qualification_common_v1 import *

def admission(output):
    r=release();source();jid=job_id(os.environ.get('SLURM_ARRAY_JOB_ID',''))
    ix=os.environ.get('SLURM_ARRAY_TASK_ID','')
    require(ix in ('0','1','2','3') and os.environ.get('SLURM_JOB_NAME')=='org_m2_eval_qual' and os.environ.get('SLURM_JOB_PARTITION')=='dev','Slurm allocation identity')
    role=ROLES[int(ix)]
    require(output.resolve()==(RESULT/('job_'+jid)/role).resolve(),'output escape/role')
    from m2_eval_repair_guard_v2 import claim_for_index,STAGES
    c=claim_for_index(int(ix));req=read(c/'request.json')
    require(req['stage'] in STAGES and req['indices']==list(STAGES[req['stage']]) and int(ix) in req['indices'],'stage/index mismatch')
    require(req['source_lock_sha256']==source() and req['roles']==list(ROLES) and req['max_gpu_hours']==len(req['indices'])/4 and req['max_ntd']==len(req['indices'])*7.5,'request drift')
    cpu=Path(req['cpu_receipt'])
    require(cpu.resolve().is_relative_to(RESULT.resolve()) and sha(cpu)==req['cpu_receipt_sha256'],'CPU receipt identity')
    final=read(cpu)
    require(final['status']=='CPU_QUALIFICATION_ADMISSION_PASS' and final['tests']==r['cluster_tests'] and final['source_lock_sha256']==source(),'CPU gate not passed')
    require(sha(cpu.parent/'FIXTURE_PANELS.json')==final['fixture_sha256'],'fixture bytes drift')
    for _ in range(100):
        if (c/'submitted.json').is_file() or (c/'SUBMISSION_UNKNOWN.json').exists():break
        time.sleep(.2)
    require(not (c/'SUBMISSION_UNKNOWN.json').exists(),'submission ambiguous')
    s=read(c/'submitted.json');require(s['job_id']==jid and s['roles']==list(ROLES) and s['stage']==req['stage'] and s['indices']==req['indices'],'unrecorded allocation')
    return role,cpu.parent,r

def worker(args):
    role,cpu,r=admission(args.output)
    torch,np,values,r=setup(True)
    from m2_model_pilot_v1 import load_model
    from m2_u_model_io_v1 import install_model
    import m2_paged_checkpoint_v1 as ck
    from m2_real_evaluator_v1 import primitive_suite
    from m2_metric_assembly_v1 import upstream_namespace
    from m2_eval_qualification_oracle_v1 import ObservedModel,validate_result,generation_batch_trial
    from transformers import AutoTokenizer
    _,base,_,_,_,cal,design=values
    panels=read(cpu/'FIXTURE_PANELS.json');require(panels==fixture(cal),'fixture calibration rebind')
    cp,metadata=checkpoint(role,r)
    print('M2_EVALUATOR_MODEL_LOAD='+role+':'+args.phase,flush=True)
    start=time.perf_counter();model=load_model(torch,base)
    require(canonical(ck.model_rows(torch,model))==r['base_state_sha256'],'loaded B state drift')
    if role!='B':install_model(torch,model,cp,metadata,r['models'][role]['state_sha256'])
    model.gradient_checkpointing_disable();model.eval()
    for p in model.parameters():p.requires_grad_(False)
    require(not model.is_gradient_checkpointing and not model.training,'evaluation mode')
    before=ck.model_rows(torch,model);aliases=ck.aliases(model)
    expect=r['base_state_sha256'] if role=='B' else r['models'][role]['state_sha256']
    require(canonical(before)==expect,'inference state identity')
    tokenizer=AutoTokenizer.from_pretrained(base['snapshot_path'],local_files_only=True,trust_remote_code=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right';require(tokenizer.pad_token_id==128009,'tokenizer pad identity')
    load_seconds=time.perf_counter()-start
    from m2_eval_oracle_repair_v2 import probability_precheck
    pre=probability_precheck(torch,model,tokenizer,panels,ROOT/'org_m2_metric_integration_v1/upstream')
    save(args.output/(args.phase+'_probability_precheck.json'),pre)
    require(pre['status']=='EARLY_PROBABILITY_PARITY_PASS',str(pre.get('error',pre['status'])))
    print('M2_EARLY_PROBABILITY_PARITY='+role+':'+args.phase+':PASS',flush=True)
    observed=ObservedModel(torch,model);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with torch.no_grad():
        result=primitive_suite(torch,observed,tokenizer,panels,design['evaluation']['generation'],ROOT/'org_m2_metric_integration_v1/upstream',batch_size=2)
    torch.cuda.synchronize();evaluation_seconds=time.perf_counter()-start
    save(args.output/(args.phase+'_observations.json'),{'role':role,'scope':SCOPE,'source_lock_sha256':source(),'result':result,'forward_calls':observed.forward_calls,'generations':observed.generations,'layout_diagnostics':observed.layout_diagnostics})
    oracle=validate_result(torch,result,panels,observed,tokenizer,upstream_namespace(ROOT/'org_m2_metric_integration_v1/upstream'))
    print('M2_EVALUATOR_PRIMITIVES='+role+':'+args.phase+':PASS\nM2_BATCH8_CALIBRATION_CHECK=START',flush=True)
    batch_trial=generation_batch_trial(torch,model,tokenizer,cal['records'][:8],design['evaluation']['generation'])
    require(ck.model_rows(torch,model)==before and ck.aliases(model)==aliases,'evaluation mutated model/aliases')
    require(all(p.grad is None for p in model.parameters()),'evaluation produced gradients')
    checkpoint(role,r);source()
    save(args.output/(args.phase+'.json'),{'status':'MODEL_FIXTURE_PRIMITIVES_PASS','scope':SCOPE,'role':role,'phase':args.phase,'pid':os.getpid(),
      'model_sha256':expect,'source_lock_sha256':source(),'fixture_sha256':sha(cpu/'FIXTURE_PANELS.json'),
      'result':result,'oracle':oracle,'layout_diagnostics':observed.layout_diagnostics,'probability_precheck':pre,'generations':observed.generations,'forward_calls':observed.forward_calls,'batch_trial':batch_trial,
      'timings':observed.timings,'load_seconds':load_seconds,'evaluation_seconds':evaluation_seconds,
      'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'peak_host_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'gpu':{'name':torch.cuda.get_device_name(0),'uuid':str(torch.cuda.get_device_properties(0).uuid),'torch':torch.__version__,'cuda':torch.version.cuda},
      'state_and_aliases_unchanged':True,'sealed_or_official_holdout_scored':False,'scientific_outcomes':False})
    print('M2_EVALUATOR_FIXTURE='+role+':'+args.phase+':PASS',flush=True)

def final_check(output):
    a,b=read(output/'a.json'),read(output/'b.json')
    require(a['status']==b['status']=='MODEL_FIXTURE_PRIMITIVES_PASS' and a['phase']=='a' and b['phase']=='b' and a['pid']!=b['pid'],'fresh processes not demonstrated')
    for k in ('scope','role','model_sha256','source_lock_sha256','fixture_sha256','generations','gpu','state_and_aliases_unchanged','sealed_or_official_holdout_scored','scientific_outcomes'):
        compare(a[k],b[k],0)
    delta=compare(a['result'],b['result']);compare(a['forward_calls'],b['forward_calls'])
    for k in ('scope','records','single_ids','batch8_ids','greedy_ids_exact','batch8_candidate','automatic_batch_selection'):
        compare(a['batch_trial'][k],b['batch_trial'][k],0)
    return {'status':'REAL_MODEL_EVALUATOR_FIXTURE_PASS','role':a['role'],'scope':SCOPE,
      'source_lock_sha256':a['source_lock_sha256'],'fixture_sha256':a['fixture_sha256'],'model_sha256':a['model_sha256'],
      'a_sha256':sha(output/'a.json'),'b_sha256':sha(output/'b.json'),'max_repeat_delta':delta,
      'greedy_ids_exact':True,'batch8_fixture_candidate':a['batch_trial']['batch8_candidate'],
      'sealed_outcomes_inspected':False,'scientific_evaluation_authorized':False,'M3_authorized':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['all','a','b'],default='all');a=p.parse_args()
    role,_,_=admission(a.output)
    if a.phase!='all':worker(a);return
    a.output.mkdir(parents=True,exist_ok=False)
    try:
        for phase in ('a','b'):
            subprocess.run([sys.executable,'-I','-B',__file__,'--output',str(a.output),'--phase',phase],check=True)
        save(a.output/'final.json',final_check(a.output))
        print('M2_REAL_EVALUATOR_QUALIFICATION='+role+':PASS\nSCIENTIFIC_EVALUATION_AUTHORIZED=false',flush=True)
    except Exception as e:
        save(a.output/'FAILURE.json',{'status':'FAIL','type':type(e).__name__,'error':str(e),'automatic_retry':False});raise

if __name__=='__main__':main()
