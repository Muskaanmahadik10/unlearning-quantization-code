"""Immutable calibration-only evaluator admission; never a scientific outcome gate."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys

ROOT=Path('/work/anon_user/hessian_unlearning')
PKG=Path(__file__).resolve().parent
RESULT=ROOT/'results/org_m2_eval_qualification_v2'
ENV=Path('/work/anon_user/.runtime/envs/org-packed-h200-v1')
ROLES=('B','O','R','U')
PANELS=('forget','retain','ra','wf','holdout','retain_all')
SCOPE='ENGINEERING_CALIBRATION_FIXTURE_NOT_BENCHMARK'

def require(ok,msg):
    if not ok:raise ValueError(msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def unique(pairs):
    d={}
    for k,v in pairs:
        require(k not in d,'duplicate JSON key');d[k]=v
    return d

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def save(p,x):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')

def confined(root,s):
    require(isinstance(s,str) and s and '\\' not in s and ':' not in s,'unsafe path')
    p=PurePosixPath(s)
    require(not p.is_absolute() and '..' not in p.parts and str(p)==s,'noncanonical path')
    q=root/s
    require(q.is_file() and not q.is_symlink() and q.resolve().is_relative_to(root.resolve()),'missing/escaped input '+s)
    return q

def verify(root,rows):
    require(rows and len({r['path'] for r in rows})==len(rows),'empty/duplicate manifest')
    for r in rows:
        p=confined(root,r['path'])
        require(type(r['bytes']) is int and r['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',r['sha256']) is not None,'manifest schema')
        require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'input drift '+r['path'])

def source():verify(PKG,read(PKG/'SELF_LOCK.json')['files']);return sha(PKG/'SELF_LOCK.json')

def release():
    r=read(PKG/'RELEASE.json')
    require(r['roles']==list(ROLES) and r['gpus_per_task']==1 and r['minutes_per_task']==15,'resource drift')
    require(r['gpu_hours_cap']==1 and r['ntd_cap']==30 and r['rate']==30,'budget expansion')
    require(r['scope']==SCOPE and r['scientific_evaluation_authorized'] is False and r['M3_authorized'] is False,'scope expansion')
    require(r['records']==3 and r['scoring_batch_size']==2 and r['generation_batch_size']==1 and r['fresh_passes']==2,'coverage drift')
    return r

def dependencies():
    source();r=release()
    for dep in r['dependencies']:
        d=ROOT/dep['path']
        require(sha(d/'SELF_LOCK.json')==dep['sha256'],'dependency lock drift')
        verify(d,read(d/'SELF_LOCK.json')['files'])
    verify(ROOT,r['accepted_evidence'])
    # Import accepted modules only after their exact bytes are authenticated.
    sys.path[:0]=[str(PKG),str(ROOT/'org_m2_u_training_v1'),str(ROOT/'org_m2_training_release_v1'),
                  str(ROOT/'org_m2_model_pilot_v2'),str(ROOT/'org_m2_metric_integration_v1')]
    return r

def checkpoint(role,r,full=False):
    require(role in ROLES,'unknown role')
    if role=='B':return None,None
    entry=r['models'][role];cp=ROOT/entry['checkpoint']
    require(sha(cp/'MANIFEST.json')==entry['manifest_sha256'],'checkpoint manifest drift '+role)
    rows=read(cp/'MANIFEST.json')['files']
    require(len(rows)==4 and {x['path'] for x in rows}=={'model.safetensors','optimizer.safetensors','rng.safetensors','metadata.json'},'checkpoint file coverage')
    verify(cp,rows if full else [x for x in rows if x['path'] in ('model.safetensors','metadata.json')])
    m=read(cp/'metadata.json');i=m['identity']
    require(m['schema']=='m2_paged_resume_v1' and i['role']==role and i['parent_role']==('O' if role=='U' else 'B'),'checkpoint lineage')
    require(i['source_lock_sha256']==entry['training_source_sha256'] and i['base_sha256']==r['base_state_sha256'] and i['data_binding_sha256']==r['data_binding_sha256'],'checkpoint source/data')
    require(m['cursor']==entry['updates'] and len(m['optimizer_steps'])==146 and set(m['optimizer_steps'].values())=={entry['updates']},'incomplete training')
    require(canonical(m['model_tensors'])==entry['state_sha256'],'model tensor digest')
    return cp,m

def setup(gpu):
    r=dependencies()
    from m2_stage_common_v1 import setup as accepted_setup
    torch,np,values,_=accepted_setup(gpu)
    return torch,np,values,r

def fixture(cal):
    """Whole accepted QA token records populate synthetic panel/option slots.

    No claim that the other QA pairs are real wrong answers/nonmembers/utility
    examples. They exercise indexed orchestration without opening sealed data.
    """
    require(len(cal['records'])==128,'calibration coverage')
    records=cal['records'][:3]
    require(len({x['qa_sha256'] for x in records})==3,'fixture duplicate QA')
    token=lambda x:{k:copy.deepcopy(x[k]) for k in ('input_ids','labels','attention_mask','prompt_ids')}
    panels={}
    for panel in PANELS:
        rr=[]
        for j,rec in enumerate(records):
            ts={'answer':[token(rec)]}
            if panel in ('forget','retain'):ts['paraphrase']=[token(rec)]
            if panel in ('forget','retain','ra','wf'):
                ts['wrong']=[token(records[(j+1+k%2)%3]) for k in range(6)]
            rr.append({'index':str(j),'qa_sha256':rec['qa_sha256'],'tokens':ts})
        panels[panel]=rr
    return panels

def compare(a,b,tolerance=1e-6,path='root'):
    """Strict structure/IDs/strings; only finite floating scores get tolerance."""
    if isinstance(a,dict):
        require(isinstance(b,dict) and set(a)==set(b),'key mismatch '+path)
        return max([compare(a[k],b[k],tolerance,path+'/'+str(k)) for k in a]+[0])
    if isinstance(a,list):
        require(isinstance(b,list) and len(a)==len(b),'row coverage '+path)
        return max([compare(x,y,tolerance,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))]+[0])
    if isinstance(a,float):
        require(type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tolerance,f'score mismatch {path}: expected={a!r} observed={b!r} delta={abs(a-b) if type(b) in (int,float) else None!r} tolerance={tolerance!r}')
        return abs(a-b)
    require(type(a)==type(b) and a==b,'identity/value mismatch '+path);return 0

def job_id(s):
    m=re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?\s*',s)
    require(m is not None,'ambiguous submission; do not retry');return m[1]

def accounting(text,jid):
    out={}
    for line in text.splitlines():
        if not line.strip():continue
        row=[x.strip() for x in line.rstrip('|').split('|')]
        require(len(row)==5,'accounting columns')
        j,s,e,t,tr=row
        require(j in [jid+'_'+str(i) for i in range(4)] and j not in out and s=='COMPLETED' and e=='0:0','nonterminal/failed/duplicate task')
        require(t.isdigit() and 0<int(t)<=900 and [x for x in tr.split(',') if x.startswith('gres/gpu=')]==['gres/gpu=1'],'unknown allocation/time')
        out[j]=int(t)
    require(len(out)==4,'four-role accounting incomplete');return out
