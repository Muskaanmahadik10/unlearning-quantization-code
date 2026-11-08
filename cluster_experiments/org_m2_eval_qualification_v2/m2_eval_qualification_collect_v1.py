"""Verify all four terminal tasks and indexed receipts, export small evidence."""
import datetime
import statistics
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m2_eval_qualification_common_v1 import *

def assembly_check(branch,reference,ns):
    from m2_metric_assembly_v1 import assemble,METRICS
    require(reference['role']=='R' and reference['scope']==branch['scope']==SCOPE,'wrong reference/scope')
    for k in ('source_lock_sha256','fixture_sha256'):require(reference[k]==branch[k],'reference identity '+k)
    p=branch['result']['primitives'];rp=reference['result']['primitives']
    # The accepted math's synthetic_cpu scope describes its fixture contract,
    # not the device that generated these scores. Never label fixtures science.
    a=assemble(p,{'role':'synthetic_R','forget_truth':rp['forget_truth'],'mia_auc':reference['result']['mia_auc']},scope='synthetic_cpu')
    def average(v):return sum(v)/len(v)
    official={'forget_truth_ratio':p['forget_truth']['agg_value'],
      'forget_quality':ns['ks_test'](None,pre_compute={'forget':p['forget_truth']},reference_logs={'retain_model_logs':{'retain':rp['forget_truth']}})['agg_value'],
      'forget_Q_A_Prob':average(list(p['forget_probability'].values())),
      'forget_Q_A_ROUGE':average(list(p['forget_rouge'].values())),
      'model_utility':ns['hm_aggregate'](None,pre_compute={k:{'agg_value':v} for k,v in p['utility'].items()})['agg_value'],
      'privleak':ns['privleak'](None,pre_compute={'forget':{'agg_value':a['mia_auc']}},reference_logs={'retain_model_logs':{'retain':{'agg_value':reference['result']['mia_auc']}}})['agg_value'],
      'extraction_strength':average(list(p['extraction'].values()))}
    compare(a['metrics'],{k:float(v) for k,v in official.items()},1e-12)
    require(set(a['metrics'])==set(METRICS),'seven metric coverage')
    a['scope']=SCOPE;a['actual_reference_role']='R';a['reference_model_sha256']=reference['model_sha256']
    a['reference_receipt_canonical_sha256']=canonical(reference);a['metric_assembly_upstream_parity']=True
    return a

def main():
    require(not os.environ.get('SLURM_JOB_ID'),'collect on login node after terminal accounting')
    torch,np,values,r=setup(False)
    from m2_eval_qualification_v1 import final_check
    from m2_metric_assembly_v1 import upstream_namespace
    from m2_eval_repair_guard_v2 import STAGES,reconcile_prior,verify_canary,fetch_accounting
    prior=reconcile_prior();verify_canary();claims={};cpus={};jobs={};paths={};times={};accounting_text=''
    for stage,indices in STAGES.items():
        claim=RESULT/('launch_'+stage);request=read(claim/'request.json');submitted=read(claim/'submitted.json')
        require(not (claim/'SUBMISSION_UNKNOWN.json').exists(),'ambiguous submission')
        require(request['source_lock_sha256']==source() and submitted['roles']==list(ROLES),'submission identity')
        require(request['stage']==submitted['stage']==stage and request['indices']==submitted['indices']==list(indices),'stage identity')
        cpu=Path(request['cpu_receipt']);require(cpu.resolve().is_relative_to(RESULT.resolve()) and sha(cpu)==request['cpu_receipt_sha256'],'CPU receipt')
        jid=job_id(submitted['job_id']);raw,tt=fetch_accounting(jid,indices);times.update(tt);accounting_text+=raw
        claims[stage]=claim;cpus[stage]=cpu;jobs[stage]=jid
        for i in indices:paths[ROLES[i]]=RESULT/('job_'+jid)/ROLES[i]
    branches={}
    for role,path in paths.items():
        cpu=cpus['canary' if role=='B' else 'remaining']
        require(read(path/'final.json')==final_check(path),'final certificate inconsistent')
        b=read(path/'a.json');require(b['role']==role and b['source_lock_sha256']==source(),'branch role/source')
        require(b['fixture_sha256']==read(cpu)['fixture_sha256'],'branch fixture binding')
        expected=r['base_state_sha256'] if role=='B' else r['models'][role]['state_sha256']
        require(b['model_sha256']==expected,'accepted model state mismatch')
        checkpoint(role,r);branches[role]=b
    ns=upstream_namespace(ROOT/'org_m2_metric_integration_v1/upstream')
    assembled={role:assembly_check(b,branches['R'],ns) for role,b in branches.items()}
    stem='m2_eval_qualification_repair_v2_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=ROOT/'audit_exports'/stem;out.mkdir(parents=True,exist_ok=False)
    def copyfile(src,dest):
        require(src.is_file() and not src.is_symlink() and src.stat().st_size<32*1024**2,'unsafe/large evidence')
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(src.read_bytes())
    (out/'sacct.psv').write_text(accounting_text,encoding='utf-8')
    save(out/'PRIOR_FAILURE_ACCOUNTING.json',prior)
    for role,path in paths.items():
        for name in ('a.json','b.json','final.json','a_observations.json','b_observations.json','a_probability_precheck.json','b_probability_precheck.json'):copyfile(path/name,out/role/name)
        jid=jobs['canary' if role=='B' else 'remaining']
        for ext in ('out','err'):copyfile(ROOT/'logs/org_m2_eval_qualification_v2'/('eval_'+jid+'_'+str(ROLES.index(role))+'.'+ext),out/(role+'.'+ext))
    for stage in STAGES:
        cpu=cpus[stage];claim=claims[stage]
        for name in ('final.json','CPU_TESTS.txt','FIXTURE_PANELS.json','scheduler_test.txt','PRIOR_FAILURE_ACCOUNTING.json'):
            copyfile(cpu.parent/name,out/'cpu'/stage/name)
        for name in ('request.json','submitted.json'):copyfile(claim/name,out/stage/name)
    copyfile(cpus['remaining'].parent/'ACCEPTED_CANARY.json',out/'ACCEPTED_CANARY.json')
    save(out/'fixture_seven_metrics.json',assembled)
    timing={}
    for role,b in branches.items():
        gs=[t for t in b['timings'] if t['kind']=='generate'];bt=b['batch_trial']
        timing[role]={'fixture_inference_seconds':b['evaluation_seconds'],'load_seconds':b['load_seconds'],
          'generation_seconds_per_record_median':statistics.median(x['seconds'] for x in gs),
          'generation_seconds_per_record_max':max(x['seconds'] for x in gs),'peak_gpu_bytes':b['peak_gpu_bytes'],
          'batch8_fixture_candidate':bt['batch8_candidate'],'batch8_seconds':bt['batch8_seconds'],'observed_batch8_speedup':bt['observed_speedup'],
          'all_record_generation_count':4617,'automatic_forecast_or_budget_authorization':False,
          'warning':'Eight calibration records do not bound full benchmark runtime or prove universal batch invariance.'}
    save(out/'timing.json',timing)
    save(out/'final.json',{'status':'FOUR_MODEL_EVALUATOR_QUALIFICATION_EVIDENCE_PASS','job_ids':jobs,'roles':list(ROLES),
      'source_lock_sha256':source(),'scope':SCOPE,'seven_metric_fixture_parity':True,'fresh_greedy_ids_exact':True,
      'gpu_hours':sum(times.values())/3600,'ntd_at_30':sum(times.values())/120,
      'prior_failed_gpu_hours':prior['gpu_hours'],'prior_failed_ntd':prior['ntd_at_30'],
      'qualification_all_attempts_gpu_hours':prior['gpu_hours']+sum(times.values())/3600,
      'all_four_batch8_fixture_checks_pass':all(b['batch_trial']['greedy_ids_exact'] for b in branches.values()),
      'sealed_or_official_holdout_scored':False,'scientific_evaluation_authorized':False,'M3_authorized':False,
      'next':'Review full-data timing budget and development/sealed orchestration; no automatic next GPU submission.'})
    save(out/'EVIDENCE_MANIFEST.json',{'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]})
    archive=Path(str(out)+'.tar.gz')
    with tarfile.open(archive,'x:gz') as tf:
        for p in sorted(out.rglob('*')):
            if p.is_file():tf.add(p,arcname=p.relative_to(out).as_posix(),recursive=False)
    with Path(str(archive)+'.sha256').open('x',encoding='utf-8',newline='\n') as f:f.write(sha(archive)+'  '+archive.name+'\n')
    print('M2_EVALUATOR_QUALIFICATION_EVIDENCE=PASS\nGPU_SUBMISSIONS=0\nSCIENTIFIC_EVALUATION_AUTHORIZED=false\nRETURN='+str(archive)+'\nSIDECAR='+str(archive)+'.sha256')

if __name__=='__main__':main()
