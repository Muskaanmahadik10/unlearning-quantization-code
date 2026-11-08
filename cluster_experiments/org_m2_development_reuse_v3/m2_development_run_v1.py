"""Guarded twelve-shard development evaluation with fresh-process repeats."""
import argparse,datetime,hashlib,io,json,os,re,shutil,subprocess,sys,tarfile,time,unittest,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m2_development_math_v1 import require,ROLES,COUNTS,GATES,select,shard_rows,analyze
PKG=Path(__file__).resolve().parent
ROOT=Path('/work/anon_user/hessian_unlearning')
QUAL=ROOT/'org_m2_eval_qualification_v2'
ENV=Path('/work/anon_user/.runtime/envs/org-packed-h200-v1')
RESULT=ROOT/'results/org_m2_development_reuse_v3'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unique(pairs):
    d={}
    for k,v in pairs:require(k not in d,'duplicate JSON');d[k]=v
    return d
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'),object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite')))
def save(p,x):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def verify(root,records):
    require(records and len({r['path'] for r in records})==len(records),'manifest coverage')
    for r in records:
        p=root/r['path'];require(not Path(r['path']).is_absolute() and '..' not in Path(r['path']).parts and '\\' not in r['path'] and ':' not in r['path'],'unsafe path')
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve()),'input path')
        require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'input drift '+str(p))
def source():verify(PKG,read(PKG/'SELF_LOCK.json')['files']);return sha(PKG/'SELF_LOCK.json')
def prerequisites(gpu=False):
    source();p=read(PKG/'PLAN.json')
    require(p['tasks']==12 and p['minutes_per_task']==120 and p['max_gpu_hours']==24 and p['max_ntd']==720,'budget/task contract')
    require(p['sealed_launch_authorized'] is False and p['packed_launch_authorized'] is False,'scope expansion')
    require(sha(QUAL/'SELF_LOCK.json')==p['qualification_source_sha256'],'qualification source')
    verify(QUAL,read(QUAL/'SELF_LOCK.json')['files'])
    evidence=ROOT/p['qualification_evidence'];require(sha(evidence/'EVIDENCE_MANIFEST.json')==p['qualification_manifest_sha256'],'qualification evidence manifest')
    verify(evidence,read(evidence/'EVIDENCE_MANIFEST.json')['files'])
    require(sha(evidence/'final.json')==p['qualification_final_sha256'],'accepted qualification')
    sys.path.insert(0,str(QUAL))
    from m2_eval_qualification_common_v1 import setup,checkpoint
    torch,np,values,r=setup(gpu)
    sys.path.insert(0,str(PKG))
    design=values[-1];require(design['development_gates']==GATES,'threshold change')
    partition=read(ROOT/'org_m2_model_pilot_v2/OBSERVED_LABEL_PARTITION_V2.json')
    require(sha(ROOT/'org_m2_model_pilot_v2/OBSERVED_LABEL_PARTITION_V2.json')==design['data']['partition_sha256'],'partition binding')
    tokenpath=ROOT/r['training_cpu']/'EVALUATION_TOKENS.json'
    require(sha(tokenpath)==r['evaluation_tokens_sha256'],'token drift')
    panels=select(read(tokenpath),partition)
    require(canonical({k:[x['qa_sha256'] for x in v] for k,v in panels.items()})==p['selection_sha256'],'development QA binding')
    return torch,values,r,panels,p

def accounting(raw,jid):
    rows={}
    for line in raw.splitlines():
        if not line.strip():continue
        parts=line.rstrip('|').split('|');require(len(parts)==5,'accounting columns')
        j,state,code,sec,tres=map(str.strip,parts)
        require(j in {jid+'_'+str(i) for i in range(12)} and j not in rows,'extra/duplicate task')
        require(state=='COMPLETED' and code=='0:0' and sec.isdigit() and 0<int(sec)<=7320,'unsuccessful/unknown task')
        require([x for x in tres.split(',') if x.startswith('gres/gpu=')]==['gres/gpu=1'],'GPU allocation')
        rows[j]=int(sec)
    require(len(rows)==12,'missing tasks');return rows


def prior_accounting(raw):
    rows={}
    for line in raw.splitlines():
        if not line.strip():continue
        parts=line.rstrip('|').split('|');require(len(parts)==5,'prior accounting columns')
        j,state,code,sec,tres=map(str.strip,parts)
        require(j in {'360986_'+str(i) for i in range(12)} and j not in rows,'prior task coverage')
        require(state=='TIMEOUT' and sec.isdigit() and 480<=int(sec)<=600,'prior attempt not confirmed timeout')
        require(re.fullmatch(r'[0-9]+:[0-9]+',code) is not None,'prior exit code')
        require([x for x in tres.split(',') if x.startswith('gres/gpu=')]==['gres/gpu=1'],'prior GPU allocation')
        rows[j]={'state':state,'exit_code':code,'elapsed_seconds':int(sec)}
    require(len(rows)==12,'missing prior tasks');return rows

def recovery_guard():
    p=read(PKG/'PLAN.json');old=ROOT/'org_m2_development_v1'
    require(sha(old/'SELF_LOCK.json')==p['prior_source_sha256'],'prior source identity')
    verify(old,read(old/'SELF_LOCK.json')['files'])
    a=subprocess.run(['sacct','-X','-n','-P','-j','360986','--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES'],capture_output=True,text=True,timeout=30)
    require(a.returncode==0,'prior accounting unavailable')
    rows=prior_accounting(a.stdout)
    d=ROOT/'results/org_m2_development_v1/job_360986'
    require(d.is_dir() and not d.is_symlink(),'prior output missing')
    candidates=[x for x in d.rglob('*.json') if x.name in ('a.json','b.json','final.json')]
    accepted=read(PKG/'ACCEPTED_FIRST_PASSES.json')['files']
    require({x.relative_to(ROOT).as_posix() for x in candidates}=={x['path'] for x in accepted},'completed phase receipts exist with unexpected coverage')
    verify(ROOT,accepted)
    return {'job_id':'360986','tasks':rows,'gpu_hours':sum(x['elapsed_seconds'] for x in rows.values())/3600,
      'prior_complete_phase_receipts':12,'old_results_modified':False,'sacct':a.stdout}

def progress(message):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+message,flush=True)

def submit(enabled):
    require(not os.environ.get('SLURM_JOB_ID'),'login node only')
    claim=RESULT/'launch_v1';require(not claim.exists(),'prior submission claim; do not resubmit')
    prior=recovery_guard()
    torch,values,r,panels,p=prerequisites(False)
    from m2_eval_qualification_common_v1 import checkpoint
    for role in ROLES:checkpoint(role,r,full=True)
    out=RESULT/('cpu_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    suite=unittest.TestSuite()
    for name in ('test_m2_development_v1','test_m2_development_torch_v1','test_m2_development_recovery_v2'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (out/'CPU_TESTS.txt').write_text(stream.getvalue(),encoding='utf-8');print(stream.getvalue(),flush=True)
    require(result.wasSuccessful() and not result.skipped and result.testsRun==p['cluster_test_count'],'new orchestration CPU tests failed')
    require(not torch.cuda.is_initialized(),'CPU gate initialized GPU')
    require(shutil.disk_usage(ROOT).free>=20*1024**3,'insufficient receipt free space; user quota unknown')
    save(out/'final.json',{'status':'DEVELOPMENT_CPU_ADMISSION_PASS','tests':result.testsRun,'source_lock_sha256':source(),
      'selection_sha256':p['selection_sha256'],'counts':COUNTS,'generation_batch_size':1,'sealed_outcomes_inspected':False})
    save(out/'prior_attempt.json',prior)
    print('M2_DEVELOPMENT_CPU_ADMISSION=PASS',flush=True)
    if not enabled:print('GPU_SUBMISSIONS=0');return
    q=subprocess.run(['squeue','-h','-r','-u','anon_user','-o','%i|%j|%T'],capture_output=True,text=True,timeout=25)
    require(q.returncode==0 and not q.stdout.strip(),'queue must be empty before 12-task wave; do not alter running jobs')
    # Revalidate known qualification job accounting on the cluster, not only copied logs.
    a=subprocess.run(['sacct','-X','-n','-P','-j','360917,360949,360958','--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES'],capture_output=True,text=True,timeout=30)
    require(a.returncode==0,'qualification accounting unavailable')
    actual={}
    for line in a.stdout.splitlines():
        parts=line.rstrip('|').split('|');require(len(parts)==5,'qualification accounting columns')
        j,s,c,t,tr=map(str.strip,parts);require(j not in actual and t.isdigit() and 'gres/gpu=1' in tr.split(','),'qualification accounting')
        actual[j]=[s,c,int(t)]
    require(actual==p['qualification_accounting'],'qualification attempts changed')
    require(p['recovery_budget_authorized'] is True and p['max_gpu_hours']==24,'recovery budget not authorized')
    require(recovery_guard()==prior,'prior evidence changed before submission')
    script=PKG/'m2_development.slurm'
    test=subprocess.run(['sbatch','--test-only',str(script)],capture_output=True,text=True,timeout=35,cwd=ROOT)
    (out/'scheduler_test.txt').write_text(test.stdout+test.stderr,encoding='utf-8');require(test.returncode==0,'scheduler test failed')
    claim.mkdir(exist_ok=False);save(claim/'request.json',{'source_lock_sha256':source(),'cpu_receipt':str(out/'final.json'),
       'cpu_sha256':sha(out/'final.json'),'max_ntd':720,'indices':list(range(12))})
    (ROOT/'logs/org_m2_development_reuse_v3').mkdir(parents=True,exist_ok=True)
    env=dict(os.environ)
    for k in ('CUDA_VISIBLE_DEVICES','PYTHONPATH','PYTHONHOME','BNB_CUDA_VERSION'):env.pop(k,None)
    try:
        s=subprocess.run(['sbatch','--parsable',str(script)],capture_output=True,text=True,timeout=35,cwd=ROOT,env=env)
        (claim/'sbatch_stdout.txt').write_text(s.stdout,encoding='utf-8');(claim/'sbatch_stderr.txt').write_text(s.stderr,encoding='utf-8')
        require(s.returncode==0,'submission failed; reconcile claim '+s.stderr)
        match=re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?\s*',s.stdout);require(match is not None,'ambiguous job id')
        jid=match[1];save(claim/'submitted.json',{'job_id':jid,'indices':list(range(12)),'max_ntd':720})
        print('M2_DEVELOPMENT_ARRAY='+jid+'\nTASKS=12\nMAX_ADDITIONAL_NTD=720\nSEALED_AND_PACKED_LAUNCH_AUTHORIZED=false')
    except Exception as e:save(claim/'SUBMISSION_UNKNOWN.json',{'error':str(e),'automatic_retry':False});raise

def admission():
    ix=os.environ.get('SLURM_ARRAY_TASK_ID','');jid=os.environ.get('SLURM_ARRAY_JOB_ID','')
    require(ix in [str(i) for i in range(12)] and re.fullmatch('[1-9][0-9]*',jid) is not None,'array identity')
    require(os.environ.get('SLURM_JOB_NAME')=='org_m2_dev' and os.environ.get('SLURM_JOB_PARTITION')=='dev','allocation scope')
    c=RESULT/'launch_v1';req=read(c/'request.json');require(req['source_lock_sha256']==source() and req['max_ntd']==720 and req['indices']==list(range(12)),'request identity')
    cp=Path(req['cpu_receipt']);require(cp.resolve().is_relative_to(RESULT.resolve()) and sha(cp)==req['cpu_sha256'],'CPU identity')
    require(read(cp)['status']=='DEVELOPMENT_CPU_ADMISSION_PASS','CPU not admitted')
    for _ in range(100):
        if (c/'submitted.json').exists() or (c/'SUBMISSION_UNKNOWN.json').exists():break
        time.sleep(.2)
    require(not (c/'SUBMISSION_UNKNOWN.json').exists() and read(c/'submitted.json')['job_id']==jid,'unrecorded allocation')
    i=int(ix);return i,ROLES[i//4],i%4,RESULT/('job_'+jid)/('task_'+ix)

def worker(phase):
    i,role,shard,out=admission();progress('PHASE='+phase+':PREFLIGHT_START')
    torch,values,r,panels,p=prerequisites(True)
    progress('PHASE='+phase+':PREFLIGHT_COMPLETE')
    from m2_model_pilot_v1 import load_model
    from m2_u_model_io_v1 import install_model
    from m2_eval_qualification_common_v1 import checkpoint
    import m2_paged_checkpoint_v1 as ck
    from m2_development_score_v1 import score
    from transformers import AutoTokenizer
    base=values[1];cp,meta=checkpoint(role,r);model=load_model(torch,base)
    require(canonical(ck.model_rows(torch,model))==r['base_state_sha256'],'base load drift')
    expected=r['models'][role]['state_sha256'];install_model(torch,model,cp,meta,expected)
    model.gradient_checkpointing_disable();model.eval()
    for x in model.parameters():x.requires_grad_(False)
    before=ck.model_rows(torch,model);aliases=ck.aliases(model);require(canonical(before)==expected,'model identity')
    tok=AutoTokenizer.from_pretrained(base['snapshot_path'],local_files_only=True,trust_remote_code=False)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tok.padding_side='right';require(tok.pad_token_id==128009,'tokenizer pad')
    selected={k:shard_rows(v,shard) for k,v in panels.items()}
    progress('PHASE='+phase+':MODEL_READY')
    paneldir=out/('panels_'+phase);paneldir.mkdir(exist_ok=False)
    def panel_saved(name,rows):
        save(paneldir/(name+'.json'),{'status':'PANEL_ONLY_NOT_CERTIFIED','phase':phase,'index':i,'role':role,
          'source_lock_sha256':source(),'model_sha256':expected,'selection_sha256':p['selection_sha256'],'rows':rows})
        progress('PHASE='+phase+':PANEL_SAVED='+name)
    start=time.perf_counter()
    with torch.no_grad():data=score(torch,model,tok,selected,values[-1]['evaluation']['generation'],ROOT/'org_m2_metric_integration_v1/upstream',on_panel=panel_saved)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    require(before==ck.model_rows(torch,model) and aliases==ck.aliases(model) and all(x.grad is None for x in model.parameters()),'model mutated')
    source();save(out/(phase+'.json'),{'status':'DEVELOPMENT_SHARD_PASS','index':i,'role':role,'shard':shard,'phase':phase,'pid':os.getpid(),
      'source_lock_sha256':source(),'model_sha256':expected,'selection_sha256':p['selection_sha256'],'result':data,
      'evaluation_seconds':seconds,'sealed_or_holdout_scored':False,'state_unchanged':True})

def final_check(out):
    from m2_eval_qualification_common_v1 import compare
    a,b=read(out/'a.json'),read(out/'b.json')
    rec=next(x for x in read(PKG/'ACCEPTED_FIRST_PASSES.json')['files'] if x['index']==a['index'])
    require(sha(out/'a.json')==rec['sha256'] and a['source_lock_sha256']==read(PKG/'PLAN.json')['prior_source_sha256'] and b['source_lock_sha256']==source(),'cross-version source provenance')
    require(a['pid']!=b['pid'] and a['phase']=='a' and b['phase']=='b','fresh processes')
    for k in ('status','index','role','shard','model_sha256','selection_sha256','sealed_or_holdout_scored','state_unchanged'):compare(a[k],b[k],0)
    delta=compare(a['result']['panels'],b['result']['panels'])
    # Token IDs are integers: compare enforces exact identity, not score tolerance.
    compare(a['result']['oracle_forward_calls'],b['result']['oracle_forward_calls'])
    return {'status':'DEVELOPMENT_FRESH_REPEAT_PASS','index':a['index'],'role':a['role'],'a_sha256':sha(out/'a.json'),'b_sha256':sha(out/'b.json'),'max_repeat_delta':delta}

def run():
    _,_,_,out=admission();out.mkdir(parents=True,exist_ok=False)
    try:
        accepted=read(PKG/'ACCEPTED_FIRST_PASSES.json')['files'];ix=int(os.environ['SLURM_ARRAY_TASK_ID'])
        rec=next(x for x in accepted if x['index']==ix);verify(ROOT,[rec])
        shutil.copyfile(ROOT/rec['path'],out/'a.json');require(sha(out/'a.json')==rec['sha256'],'copied first pass drift')
        subprocess.run([sys.executable,'-I','-B',__file__,'--phase','b'],check=True)
        sys.path.insert(0,str(QUAL));save(out/'final.json',final_check(out));print('M2_DEVELOPMENT_SHARD=PASS',flush=True)
    except Exception as e:save(out/'FAILURE.json',{'type':type(e).__name__,'message':str(e),'automatic_retry':False});raise

def collect():
    require(not os.environ.get('SLURM_JOB_ID'),'collect on login node')
    _,values,r,panels,p=prerequisites(False)
    claim=RESULT/'launch_v1';require(not (claim/'SUBMISSION_UNKNOWN.json').exists(),'ambiguous submission')
    req=read(claim/'request.json');require(req['source_lock_sha256']==source(),'request source')
    jid=read(claim/'submitted.json')['job_id'];require(re.fullmatch('[1-9][0-9]*',jid) is not None,'job id')
    acc=subprocess.run(['sacct','-X','-n','-P','-j',jid,'--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES'],capture_output=True,text=True,timeout=30)
    require(acc.returncode==0,'accounting unavailable');secs=accounting(acc.stdout,jid)
    states={role:{k:{} for k in COUNTS} for role in ROLES};paths=[]
    for i in range(12):
        d=RESULT/('job_'+jid)/('task_'+str(i));f=final_check(d);require(read(d/'final.json')==f,'final binding')
        a=read(d/'a.json');role=ROLES[i//4];shard=i%4
        require(a['index']==i and a['role']==role and a['shard']==shard and a['status']=='DEVELOPMENT_SHARD_PASS' and a['source_lock_sha256']==p['prior_source_sha256'] and a['model_sha256']==r['models'][role]['state_sha256'] and a['selection_sha256']==p['selection_sha256'],'shard identity')
        require(a['state_unchanged'] is True and a['sealed_or_holdout_scored'] is False,'state/scope')
        for panel,rows in panels.items():
            expected={x['qa_sha256']:x['index'] for x in shard_rows(rows,shard)};observed=a['result']['panels'][panel]
            require({q:x['index'] for q,x in observed.items()}==expected and not set(states[role][panel])&set(observed),'QA coverage')
            states[role][panel].update({q:x['metrics'] for q,x in observed.items()})
        paths.append(d)
    decision=analyze(states)
    stem='m2_development_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=ROOT/'audit_exports'/stem;out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(PKG/'ACCEPTED_FIRST_PASSES.json',out/'ACCEPTED_FIRST_PASSES.json')
    save(out/'decision.json',decision);save(out/'aligned_rows.json',states)
    for i,d in enumerate(paths):
        target=out/('task_'+str(i));target.mkdir()
        for n in ('a.json','b.json','final.json'):shutil.copyfile(d/n,target/n)
        for phase in ('b',):
            shutil.copytree(d/('panels_'+phase),target/('panels_'+phase))
        for ext in ('out','err'):shutil.copyfile(ROOT/'logs/org_m2_development_reuse_v3'/('dev_'+jid+'_'+str(i)+'.'+ext),out/('task_'+str(i)+'.'+ext))
    for n in ('request.json','submitted.json'):shutil.copyfile(claim/n,out/n)
    cpu=Path(req['cpu_receipt']);require(cpu.resolve().is_relative_to(RESULT.resolve()) and sha(cpu)==req['cpu_sha256'],'CPU evidence binding')
    shutil.copyfile(cpu.parent/'prior_attempt.json',out/'prior_attempt.json')
    shutil.copyfile(cpu,out/'cpu_final.json');shutil.copyfile(cpu.parent/'CPU_TESTS.txt',out/'CPU_TESTS.txt')
    (out/'sacct.psv').write_text(acc.stdout,encoding='utf-8')
    save(out/'final.json',{'status':'DEVELOPMENT_EVIDENCE_COMPLETE','development_gate':decision['status'],'job_id':jid,'source_lock_sha256':source(),
      'gpu_hours':sum(secs.values())/3600,'ntd_at_30':sum(secs.values())/120,'sealed_launch_authorized':False,'packed_launch_authorized':False,'M3_authorized':False})
    save(out/'EVIDENCE_MANIFEST.json',{'files':[{'path':x.relative_to(out).as_posix(),'bytes':x.stat().st_size,'sha256':sha(x)} for x in sorted(out.rglob('*')) if x.is_file()]})
    archive=Path(str(out)+'.tar.gz')
    with tarfile.open(archive,'x:gz') as t:
        for x in sorted(out.rglob('*')):
            if x.is_file():t.add(x,arcname=x.relative_to(out).as_posix(),recursive=False)
    with Path(str(archive)+'.sha256').open('x',encoding='utf-8',newline='\n') as f:f.write(sha(archive)+'  '+archive.name+'\n')
    print('M2_DEVELOPMENT_EVIDENCE=COMPLETE\nDEVELOPMENT_GATE='+decision['status']+'\nGPU_SUBMISSIONS=0\nRETURN='+str(archive)+'\nSIDECAR='+str(archive)+'.sha256')

def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');p.add_argument('--collect',action='store_true');p.add_argument('--phase',choices=['all','a','b']);a=p.parse_args()
    require(sum((a.submit,a.collect,a.phase is not None))<=1,'incompatible modes')
    if a.collect:collect()
    elif a.phase=='all':run()
    elif a.phase:worker(a.phase)
    else:submit(a.submit)
if __name__=='__main__':main()
