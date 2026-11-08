"""Read-only source/results audit; writes a new evidence export, never submits jobs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tarfile
import uuid

HERE = Path(__file__).resolve().parent
ARMS = ['N0','Z_a','M_a','R_a','Z_b','M_b','R_b','Z_ab','M_ab','R_ab']
def require(ok, message):
    if not ok: raise ValueError(message)
def pairs(items):
    result = {}
    for k,v in items:
        require(k not in result, 'duplicate JSON key'); result[k]=v
    return result
def read(p):
    def bad(v): raise ValueError('nonfinite JSON: '+v)
    return json.loads(p.read_text(), object_pairs_hook=pairs, parse_constant=bad)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p, obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')
def finite(v): return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0
def safe(root,rel):
    p=(root/rel).resolve(); require(p.is_relative_to(root.resolve()),'path escape'); return p

def accounting(text):
    required = {**{f'357253_{i}':'CANCELLED' for i in (3,4,5)},
                **{f'357308_{i}':'COMPLETED' for i in (3,4,5)},
                '357360_0':'FAILED','357360_1':'COMPLETED','357360_2':'COMPLETED','357529':'COMPLETED'}
    rows={}
    for line in text.splitlines():
        c=[x.strip() for x in line.split('|')]
        if c[0] not in required: continue
        require(len(c)>=5 and c[0] not in rows,'duplicate/incomplete accounting')
        state=c[1].split()[0].rstrip('+')
        require(state==required[c[0]], 'unexpected state '+c[0])
        if state=='COMPLETED': require(c[2]=='0:0','nonzero completed exit')
        if state=='FAILED': require(c[2]=='1:0','unexpected failure exit')
        require(c[3].isdigit() and 0<int(c[3])<=1200,'elapsed invalid')
        require(re.search(r'(?:^|,)gres/gpu=1(?:,|$)',c[4]) is not None,'allocation not one GPU')
        rows[c[0]]={'state':state,'exit_code':c[2],'elapsed_seconds':int(c[3]),'alloc_tres':c[4]}
    require(set(rows)==set(required),'missing accounting jobs')
    return rows

def phase_check(record, cell, source_sha, phase):
    require(record['status']=='PASS' and record['cell']==cell['cell'] and record['phase']==phase,'phase identity')
    require(record['source_lock_sha256']==source_sha,'phase source mismatch')
    require(record['requantized'] is False and record['core_launch_authorized'] is False,'scope mismatch')
    require(set(record['arms'])==set(ARMS),'arm coverage')
    require(finite(record['seconds']) and finite(record['load_seconds']),'invalid phase timing')
    for arm,r in record['arms'].items():
        names=[] if arm=='N0' else [cell[s][:-7] for s in ('a','b') if s in arm.split('_')[1]]
        dose=None if arm=='N0' else {'Z':0.,'M':.5,'R':1.}[arm[0]]
        require(r['status']=='PASS' and r['targets']==names and r['dose']==dose,'arm contract '+arm)
        require(finite(r['seconds']),'arm timing')
        require((r['behavior'] is not None)==(arm in ('N0','Z_ab')),'smoke behavior scope')

def behavior_check(a,b):
    require(set(a)==set(b) and len(a)==2,'forget/retain coverage')
    maximum=0.
    for role in a:
        x,y=a[role]['rows'],b[role]['rows']
        require(len(x)==len(y)==128,'sample count')
        require(len({r['index'] for r in x})==128,'duplicate samples')
        for p,q in zip(x,y):
            require(all(p[k]==q[k] for k in ('index','sample_sha256','encoding_sha256','tokens')),'sample identity')
            require(finite(p['loss_sum']) and finite(q['loss_sum']) and p['tokens']>0,'invalid loss')
            d=abs(p['loss_sum']/p['tokens']-q['loss_sum']/q['tokens'])
            require(d<=1e-6,'behavior reload drift'); maximum=max(maximum,d)
    return maximum

def run(root):
    require(root.resolve()==Path('/work/anon_user/hessian_unlearning'),'wrong cluster root')
    cfg=read(HERE/'INPUTS.json')
    require(sha(HERE/'INPUTS.json')==read(HERE/'SELF_LOCK.json')['inputs_sha256'],'input config changed')
    require(sha(Path(__file__))==read(HERE/'SELF_LOCK.json')['script_sha256'],'collector changed')
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=root/'audit_exports'/('m1_smoke_reconciliation_'+tag); out.mkdir(parents=True,exist_ok=False)
    try:
        for package,lock_sha in cfg['packages'].items():
            base=root/package; require(sha(base/'SOURCE_LOCK.json')==lock_sha,'unapproved package version')
            for member in read(base/'SOURCE_LOCK.json')['files']:
                require(sha(safe(base,member['path']))==member['sha256'],'source file drift '+member['path'])
        design=read(root/cfg['design'])
        require(sha(root/cfg['design'])==cfg['design_sha256'],'design drift')
        proc=subprocess.run(['sacct','-X','-n','-P','-j','357253,357308,357360,357529',
                             '--format=JobID%80,State%60,ExitCode,ElapsedRaw,AllocTRES%500'],capture_output=True,text=True,timeout=30)
        write(out/'accounting_raw.json',{'stdout':proc.stdout,'stderr':proc.stderr,'returncode':proc.returncode})
        require(proc.returncode==0,'accounting unavailable')
        jobs=accounting(proc.stdout)
        audit=[]
        for entry in cfg['accepted']:
            index=entry['index']; cell=design['cells'][index]; folder=root/entry['folder']
            source_sha=cfg['packages'][entry['package']]
            final=read(folder/'final.json')
            require(final['status']=='PASS' and final['scope']=='M1_SIX_CELL_SMOKE_ONLY' and final['cell']==cell['cell'],'final identity')
            require(final['source_lock_sha256']==source_sha and final['core_launch_authorized'] is False,'final source/scope')
            require(set(final['phases'])=={'produce','reload'},'phase coverage')
            phases={}
            for part in ('produce','reload'):
                path=folder/(part+'.json'); require(sha(path)==final['phases'][part],'phase hash drift')
                record=read(path); phase_check(record,cell,source_sha,part); phases[part]=record
                write(out/'receipts'/str(index)/(part+'.json'),record)
            a,b=phases['produce'],phases['reload']
            require(a['runtime']['pid']!=b['runtime']['pid'] and a['runtime']['process_uuid']!=b['runtime']['process_uuid'],'process not fresh')
            require(a['runtime']['gpu_uuid']==b['runtime']['gpu_uuid'],'GPU changed')
            require(a['decode_reports']==b['decode_reports'],'decode drift')
            for arm in ARMS:
                x,y=a['arms'][arm],b['arms'][arm]
                for field in ('patch_sha256','logits_sha256'): require(x[field]==y[field],'arm artifact identity')
                if arm!='N0': require(sha(folder/(arm+'.safetensors'))==x['patch_sha256'],'patch bytes changed')
                require(sha(folder/(arm+'_logits.pt'))==x['logits_sha256'],'synthetic bytes changed')
            deltas={arm:behavior_check(a['arms'][arm]['behavior'],b['arms'][arm]['behavior']) for arm in ('N0','Z_ab')}
            write(out/'receipts'/str(index)/'final.json',final)
            audit.append({'index':index,'cell':cell['cell'],'job':entry['job'],'status':'PASS','source_lock_sha256':source_sha,
                          'source_folder':entry['folder'],'behavior_max_nll_delta':deltas,
                          'produce_seconds':a['seconds'],'reload_seconds':b['seconds'],
                          'arm_timings':{p:{arm:phases[p]['arms'][arm]['seconds'] for arm in ARMS} for p in phases}})
        require(len(audit)==6 and {r['index'] for r in audit}==set(range(6)),'six-cell completeness')
        seconds=sum(r['elapsed_seconds'] for r in jobs.values())
        require(seconds/3600<7,'M1 budget exhausted')
        report={'status':'SIX_CELL_SMOKE_RECONCILIATION_PASS','cells':audit,'jobs':jobs,
                'gpu_hours_all_smoke_attempts':seconds/3600,'ntd_at_user_rate':seconds/120,
                'remaining_m1_core_envelope_gpu_hours':7-seconds/3600,
                'source_version_reconciliation':cfg['source_equivalence'],
                'scope':'SMOKE_INTEGRITY_ONLY_NOT_CORE_EXECUTION_OR_SCIENTIFIC_SUCCESS',
                'core_launch_authorized':False,'gpu_submissions':0,'independent_backup_complete':False}
        write(out/'final.json',report)
        write(out/'collector_inputs.json',cfg)
        print('M1_SIX_CELL_RECONCILIATION=PASS\nM1_CORE_LAUNCH_AUTHORIZED=false\nGPU_SUBMISSIONS=0')
    except Exception as e:
        write(out/'FAILURE.json',{'status':'FAIL','error':str(e),'type':type(e).__name__,'core_launch_authorized':False})
        print('EVIDENCE='+str(out)); raise
    files=sorted(p for p in out.rglob('*') if p.is_file())
    write(out/'evidence_manifest.json',{'files':[{'path':p.relative_to(out).as_posix(),'sha256':sha(p)} for p in files]})
    archive=Path(str(out)+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar: tar.add(out,arcname=out.name)
    Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii')
    print('RETURN='+str(archive)+'\nSIDECAR='+str(archive)+'.sha256')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
