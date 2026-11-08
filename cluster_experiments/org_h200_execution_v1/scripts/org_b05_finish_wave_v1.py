"""Complete the frozen six-cell B05 wave without duplicate condition execution."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import run_org_b05_behavior_cell_v2 as patched
import org_b05_remaining_backends_v1 as prior
import org_b05_gptq_pair_repair_v2 as repair
from org_b05_behavior_evaluator_v1 import compare_reloads, summarize
from certify_org_e2b_b05_feature_wave_v1 import EXPECTED

cell=patched.runner
life=patched.lifecycle
ROOT=Path('results/org_h200_execution_v1/b05_finish_wave_v1')
SOURCE=Path('org_h200_execution_v1/ORG_B05_FINISH_WAVE_V1_SHA256.txt')
PLAN=prior.PLAN
PHASES=prior.PHASES
RTN=prior.RTN
FINALIZER_SECONDS=900

def verify_sources():
    for line in SOURCE.read_text().splitlines():
        h,p=line.split(None,1)
        life.require(life.sha256_file(Path(p.strip()))==h,'source drift '+p)

def verify_record(path):
    r=life.read(path)
    life.require(r.get('status') in ('PASS','EXECUTION_COMPLETE'),'failed prerequisite '+str(path))
    for n,h in r['evidence'].items():
        life.require(life.sha256_file(path.parent/n)==h,'evidence drift '+n)
    return r

def inventory(folder):
    return {str(p.relative_to(folder)):life.sha256_file(p) for p in folder.rglob('*.json')}

def verify_inventory(folder, hashes):
    for n,h in hashes.items():
        life.require(life.sha256_file(folder/n)==h,'condition JSON drift '+n)

def validate(folder, artifacts=False):
    prior.validate(folder)
    for phase in PHASES:
        dest=folder/'emulated' if phase=='emulated' else folder
        r=life.read(dest/(phase+'_behavior.json'))
        for role in ('forget','retain'):
            rows=r[role]['rows']
            life.require(len(rows)==128,'128 sample requirement')
            compare_reloads(rows,rows)
    if artifacts:
        hashes=life.read(folder/'artifact_hashes.json')
        actual={str(p.relative_to(folder/'artifact')) for p in (folder/'artifact').rglob('*') if p.is_file()}
        life.require(actual==set(hashes),'artifact universe')
        for n,h in hashes.items():
            life.require(life.sha256_file(folder/'artifact'/n)==h,'artifact drift '+n)

def make_tasks(plan, reused):
    life.require({(c['cell_id'],c['quantizer']) for c in plan['cells']}==EXPECTED and len(plan['cells'])==6,'cell universe')
    tasks=[]; conditions={}
    for c in sorted(plan['cells'],key=lambda c:(c['cell_id'],c['quantizer'])):
        cid,q=c['cell_id'],c['quantizer']; key=cid+'/'+q
        _,targets,sets=cell.context(SimpleNamespace(plan=PLAN,cell=cid,quantizer=q))
        conditions[key]=sets
        for name in cell.CONDITIONS:
            if sets[name]['alias_of'] or (key+'/'+name) in reused: continue
            tasks.append({'cell':cid,'quantizer':q,'condition':name,
                'group':'gptq' if q=='gptq_int4' else 'other',
                'limit_seconds':3600 if q=='gptq_int4' else 1800,
                'folder':str(ROOT/'conditions'/(cid+'__'+q)/name)})
    life.require(len(tasks)<=23,'unexpected unfinished task count')
    return tasks,conditions

def accounting(text,expected):
    found={}
    for line in text.splitlines():
        parts=[x.strip() for x in line.split('|')]
        if len(parts)<4 or parts[0] not in expected: continue
        j,state,sec,tres=parts[:4]
        life.require(j not in found,'duplicate accounting')
        life.require(state.split()[0] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'),'nonterminal task')
        fields=dict(x.split('=',1) for x in tres.split(',') if '=' in x)
        life.require(fields.get('gres/gpu')=='1','expected one GPU')
        life.require(int(sec)>=0,'invalid elapsed')
        found[j]={'state':state,'seconds':int(sec)}
    life.require(set(found)==set(expected),'incomplete accounting')
    return found

def query_jobs(ids):
    return subprocess.run(['sacct','-X','-n','-P','-j',','.join(ids),
        '--format=JobID%40,State%30,ElapsedRaw,AllocTRES%300'],capture_output=True,text=True,check=True,timeout=45).stdout

def checkpoint_inputs(plan):
    files={}
    for c in plan['cells']:
        feature=life.read(Path(c['feature_receipt']['path']))
        life.require(life.sha256_file(Path(c['feature_receipt']['path']))==c['feature_receipt']['sha256'],'feature identity')
        prereq=feature['prerequisites']
        for key in ('checkpoint_ledger','checkpoint_certification'):
            rec=prereq[key]
            life.require(life.sha256_file(Path(rec['path']))==rec['sha256'],'checkpoint prerequisite drift')
        ledger=life.read(Path(prereq['checkpoint_ledger']['path']))
        cert=life.read(Path(prereq['checkpoint_certification']['path']))
        life.require(cert['status']=='PASS' and cert['ledger_file_sha256']==prereq['checkpoint_ledger']['sha256'],'checkpoint certification linkage')
        chosen=[r for r in ledger['checkpoints'] if r['checkpoint_dir']==c['checkpoint']['certified_relative_path']]
        life.require(len(chosen)==1,'checkpoint ledger identity')
        for rec in chosen[0]['files']:
            p=Path(rec['path'])
            life.require(p.resolve().is_relative_to(Path(c['checkpoint']['path']).resolve()),'checkpoint path boundary')
            if str(p) in files: continue
            life.require(p.stat().st_size==rec['size'] and life.sha256_file(p)==rec['sha256'],'checkpoint byte drift '+str(p))
            files[str(p)]={'sha256':rec['sha256'],'size':rec['size'],'mtime_ns':p.stat().st_mtime_ns}
    return files

def prepare():
    verify_sources()
    life.require(not ROOT.exists(),'already prepared; no duplicate submissions')
    pair=life.read(repair.ROOT/'backend_validation.json')
    life.require(pair['status']=='PASS','backend pair prerequisite')
    for q in ('nf4','gptq'):
        life.require(life.sha256_file(Path(pair[q+'_receipt']))==pair[q+'_sha256'],'pair receipt drift')
        verify_record(Path(pair[q+'_receipt']))
    rtn=verify_record(RTN/'final.json')
    life.require(rtn['transport']['status']=='PASS','ordinary RTN prerequisite')
    reused={}
    _,_,sets=cell.context(SimpleNamespace(plan=PLAN,cell='ordinary_tofu_ga_seed42',quantizer='rtn_int8'))
    for name in cell.CONDITIONS:
        if sets[name]['alias_of']: continue
        folder=RTN/name; validate(folder,True)
        reused['ordinary_tofu_ga_seed42/rtn_int8/'+name]={'folder':str(folder),'json_hashes':inventory(folder)}
    for q,rkey in (('nf4','nf4'),('gptq_int4','gptq')):
        folder=Path(pair[rkey+'_receipt']).parent; validate(folder,True)
        reused['ordinary_tofu_ga_seed42/'+q+'/pair']={'folder':str(folder),'json_hashes':inventory(folder)}
    plan=life.read(PLAN)
    tasks,conditions=make_tasks(plan,reused)
    checkpoints=checkpoint_inputs(plan)
    ceiling=sum(t['limit_seconds'] for t in tasks)+FINALIZER_SECONDS
    life.require(ceiling<=58500,'additional wave exceeds 16.25 GPU hours')
    # Metadata preflight also reserves room for all condition artifacts and logs.
    life.require(shutil.disk_usage(Path.cwd()).free>=250*1024**3,'need 250 GiB available workspace headroom')
    text=query_jobs(['349981']); last=accounting(text,{'349981'})
    life.require(last['349981']['state']=='COMPLETED','repair job not completed')
    used=life.read(repair.ROOT/'authorization.json')['accounted_gpu_seconds']+last['349981']['seconds']
    ROOT.mkdir(parents=True)
    (ROOT/'prior_accounting.psv').write_text(text)
    auth={'schema_version':'org_b05_behavior_wave_authorization_v1','status':'APPROVED',
        'cells':sorted(conditions),'plan_sha256':life.sha256_file(PLAN),
        'source_manifest_sha256':life.sha256_file(SOURCE),'max_job_gpu_hours':2,
        'non_negligible_nll':1e-6,'cumulative_accounting_reviewed':True,
        'accounted_prior_gpu_seconds':used,'additional_gpu_seconds_ceiling':ceiling,
        'additional_ntd_ceiling':ceiling/3600*30,
        'approval_record':'User requested completing all remaining B05 in one parallel wave; disclosed extra ceiling 16.25 GPU-hours / 487.50 NTD. No Atlas execution authorized.',
        'atlas_production_authorized':False}
    life.atomic(ROOT/'authorization.json',auth)
    life.atomic(ROOT/'wave.json',{'tasks':tasks,'conditions':conditions,'reused':reused,'checkpoint_files':checkpoints,
        'authorization_sha256':life.sha256_file(ROOT/'authorization.json')})
    print('B05_FINISH_PREPARED=PASS\nREMAINING_TASKS='+str(len(tasks)))
    print('ADDITIONAL_GPU_HOURS_MAX='+str(ceiling/3600)+'\nADDITIONAL_NTD_MAX='+str(ceiling/3600*30))

def load_wave():
    verify_sources(); w=life.read(ROOT/'wave.json')
    life.require(life.sha256_file(ROOT/'authorization.json')==w['authorization_sha256'],'authorization drift')
    a=life.read(ROOT/'authorization.json')
    life.require(life.sha256_file(PLAN)==a['plan_sha256'],'plan drift')
    life.require(life.sha256_file(SOURCE)==a['source_manifest_sha256'],'source manifest drift')
    for name,rec in w['checkpoint_files'].items():
        stat=Path(name).stat()
        life.require(stat.st_size==rec['size'] and stat.st_mtime_ns==rec['mtime_ns'],'checkpoint changed during wave '+name)
    return w

def run(group,index):
    w=load_wave(); task=[t for t in w['tasks'] if t['group']==group][index]
    folder=Path(task['folder']); folder.mkdir(parents=True)
    for phase in PHASES:
        dest=folder/'emulated' if phase=='emulated' else folder
        if phase=='emulated': dest.mkdir()
        print('B05_FINISH_PHASE='+task['cell']+'/'+task['quantizer']+'/'+task['condition']+'/'+phase,flush=True)
        subprocess.run([sys.executable,str(Path(patched.__file__)),'--plan',str(PLAN),
            '--authorization',str(ROOT/'authorization.json'),'--source-manifest',str(SOURCE),
            '--cell',task['cell'],'--quantizer',task['quantizer'],'--condition',task['condition'],
            '--phase',phase,'--output',str(dest)],check=True)
    validate(folder)
    life.atomic(folder/'condition_complete.json',{'status':'PASS','task':task,
        'job_id':os.environ.get('SLURM_ARRAY_JOB_ID','')+'_'+os.environ.get('SLURM_ARRAY_TASK_ID',''),
        'evidence':inventory(folder),'production_authorized':False})
    print('B05_FINISH_CONDITION=PASS',flush=True)

def aggregate():
    w=load_wave(); sub=life.read(ROOT/'submissions.json')
    expected={j+'_'+str(i) for group,j in sub['arrays'].items() for i in range(sum(t['group']==group for t in w['tasks']))}
    # Slurm dependencies can release before accounting's database view catches up.
    for attempt in range(4):
        text=query_jobs(list(sub['arrays'].values()))
        try:
            stats=accounting(text,expected)
            break
        except RuntimeError as exc:
            if str(exc) not in ('incomplete accounting','nonterminal task') or attempt==3: raise
            time.sleep(3)
    (ROOT/'wave_accounting.psv').write_text(text)
    failures=[j+':'+r['state'] for j,r in stats.items() if r['state']!='COMPLETED']
    for t in w['tasks']:
        if not (Path(t['folder'])/'condition_complete.json').is_file(): failures.append('missing:'+t['folder'])
    if failures:
        life.atomic(ROOT/'final.json',{'status':'INCOMPLETE','failures':failures,'atlas_production_authorized':False})
        raise RuntimeError('B05 incomplete; preserved evidence, no automatic retries')
    records={}; reports={}; table=['cell\tquantizer\tcondition\talias_of\tpacked_forget_nll\tpacked_retain_nll\temulated_forget_nll\temulated_retain_nll']
    for key,sets in w['conditions'].items():
        packed={}; emulated={}; cid,q=key.split('/')
        for name in cell.CONDITIONS:
            alias=sets[name]['alias_of']
            if alias:
                packed[name]=packed[alias]; emulated[name]=emulated[alias]
                records[key+'/'+name]={'alias_of':key+'/'+alias}
            else:
                ident=key+'/'+name
                if ident in w['reused']:
                    r=w['reused'][ident]; folder=Path(r['folder']); verify_inventory(folder,r['json_hashes'])
                else:
                    folder=Path(next(t['folder'] for t in w['tasks'] if (t['cell'],t['quantizer'],t['condition'])==(cid,q,name)))
                    r=verify_record(folder/'condition_complete.json')
                validate(folder,True)
                packed[name]=life.read(folder/'reload_b_behavior.json')
                emulated[name]=life.read(folder/'emulated/emulated_behavior.json')
                records[ident]={'folder':str(folder),'json_hashes':inventory(folder)}
            vals=[summarize(data[name][role]['rows'])['token_mean_nll'] for data in (packed,emulated) for role in ('forget','retain')]
            table.append('\t'.join([cid,q,name,alias or '',*map(str,vals)]))
        reports[key]=cell.transport(packed,emulated)
    life.require(len(records)==36 and len(reports)==6,'final coverage')
    observed=sum(r['seconds'] for r in stats.values())
    life.require(observed<=sum(t['limit_seconds'] for t in w['tasks']),'allocation overrun; audit required')
    for name,rec in w['checkpoint_files'].items():
        life.require(life.sha256_file(Path(name))==rec['sha256'],'post-wave checkpoint drift')
    (ROOT/'condition_results.tsv').write_text('\n'.join(table)+'\n')
    passed=all(r['status']=='PASS' for r in reports.values())
    life.atomic(ROOT/'final.json',{'status':'EXECUTION_COMPLETE','cells':6,'conditions':36,
        'integrity':'PASS','transport':'PASS' if passed else 'STOP_OR_INCONCLUSIVE',
        'cell_transport':reports,'records':records,'wave_gpu_seconds':observed,
        'wave_ntd_at_user_rate':observed/3600*30,'finalizer_accounting_pending':True,
        'atlas_lock_preparation_allowed':passed,'atlas_production_authorized':False,
        'plan_sha256':life.sha256_file(PLAN),'wave_sha256':life.sha256_file(ROOT/'wave.json'),
        'condition_results_sha256':life.sha256_file(ROOT/'condition_results.tsv')})
    final=ROOT/'final.json'
    (ROOT/'final.json.sha256').write_text(life.sha256_file(final)+'  '+str(final)+'\n')
    print('B05_EXECUTION=COMPLETE\nCELLS=6\nCONDITIONS=36\nINTEGRITY=PASS')
    print('TRANSPORT='+('PASS' if passed else 'STOP_OR_INCONCLUSIVE'))
    print('ATLAS_LOCK_PREPARATION_ALLOWED='+str(passed)+'\nATLAS_PRODUCTION_AUTHORIZED=false')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('prepare','run','aggregate'))
    p.add_argument('--group',choices=('gptq','other')); p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.mode=='run': run(a.group,a.index)
    else: globals()[a.mode]()
