"""Locked dependencies and scope for the O/R training-only release."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys

ROOT=Path('/work/anon_user/hessian_unlearning')
PKG=Path(__file__).resolve().parent
ENV=Path('/work/anon_user/.runtime/envs/org-packed-h200-v1')
PILOT=ROOT/'org_m2_model_pilot_v2'
METRIC=ROOT/'org_m2_metric_integration_v1'
RESULT=ROOT/'results/org_m2_training_release_v1'

def require(ok,msg):
    if not ok:raise ValueError(msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def unique(rows):
    d={}
    for k,v in rows:
        require(k not in d,'duplicate JSON key');d[k]=v
    return d

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,x):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')

def verify(root,rows):
    require(rows and len({r['path'] for r in rows})==len(rows),'manifest coverage')
    for r in rows:
        s=r['path'];p=PurePosixPath(s)
        require(s and str(p)==s and not p.is_absolute() and '..' not in p.parts and '\\' not in s and ':' not in s,'manifest path')
        q=root/s
        require(q.is_file() and not q.is_symlink() and q.resolve().is_relative_to(root.resolve()),'missing/escaped source '+s)
        require(type(r['bytes']) is int and q.stat().st_size==r['bytes'] and sha(q)==r['sha256'],'source drift '+s)

def source():verify(PKG,read(PKG/'SELF_LOCK.json')['files']);return sha(PKG/'SELF_LOCK.json')

def dependencies():
    source();release=read(PKG/'RELEASE.json')
    for name,root in [('pilot',PILOT),('metric',METRIC)]:
        require(sha(root/'SELF_LOCK.json')==release[name+'_source_sha256'],'dependency identity '+name)
        verify(root,read(root/'SELF_LOCK.json')['files'])
    verify(ROOT,release['accepted_cluster_evidence'])
    # Original modules are reused byte-for-byte; their globals keep their own
    # frozen pilot namespace. No hidden monkeypatching of pilot authorization.
    sys.path[:0]=[str(PKG),str(PILOT),str(METRIC)]
    return release

def setup(gpu=False):
    require(sys.platform=='linux' and Path(sys.prefix)==ENV and sys.flags.isolated and sys.dont_write_bytecode,'certified python -I -B required')
    release=dependencies()
    from m2_model_pilot_v1 import setup as prior_setup
    torch,np,values=prior_setup(gpu)
    return torch,np,values,release

def validate_plan(p):
    require(p['roles']==['O','R'] and p['gpus_per_task']==1 and p['minutes_per_task']==120,'resource scope')
    require(p['gpu_hours_cap']==4 and p['ntd_cap']==120 and p['rate']==30,'budget drift')
    require(p['U_automatic'] is False and p['evaluation_launch_authorized'] is False and p['M3_authorized'] is False,'scope expansion')
    return p

def selected_role(index):
    require(str(index) in ('0','1'),'training array index');return ('O','R')[int(index)]

def validate_request(r):
    require(r['roles']==['O','R'] and r['max_gpu_hours']==4 and r['max_ntd']==120,'request scope/budget')
    require(r['user_quota_verified'] is False and r['user_accepted_unverified_quota'] is True,'quota acknowledgement')

def validate_binding(b):
    require(b['counts']=={'forget':400,'retain':400,'ra':100,'wf':117,'holdout':400,'retain_all':3600},'evaluation row coverage')
    require(b['official_forget_exact_qa_matches_training_forget'] and b['retain_eval_subset_of_training_retain'],'dataset linkage')
    require(b['holdout_training_overlap']==0 and b['all_variant_token_parity'] is True,'token/nonmember binding')
    require(b['evaluated_model_outcomes']==0,'CPU binding must not inspect outcomes')

def accounting(text,jid):
    rows={}
    for line in text.splitlines():
        if not line.strip():continue
        parts=[x.strip() for x in line.rstrip('|').split('|')]
        require(len(parts)==5,'accounting schema')
        j,s,e,t,tr=parts
        require(j in (jid+'_0',jid+'_1') and j not in rows and s=='COMPLETED' and e=='0:0' and t.isdigit(),'job not successfully terminal')
        require([x for x in tr.split(',') if x.startswith('gres/gpu=')]==['gres/gpu=1'],'GPU allocation')
        rows[j]=int(t)
    require(len(rows)==2,'two-job accounting incomplete');return rows
