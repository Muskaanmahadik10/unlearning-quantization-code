"""Separate U authorization; frozen O/R dependencies are never modified."""
import os
from pathlib import Path
import sys
ROOT=Path('/work/anon_user/hessian_unlearning')
PKG=Path(__file__).resolve().parent
OR_SOURCE=ROOT/'org_m2_training_release_v1'
sys.path.insert(0,str(OR_SOURCE))
from m2_stage_common_v1 import require,sha,read,save,canonical,verify
RESULT=ROOT/'results/org_m2_u_training_v1'
PARENT=ROOT/'results/org_m2_training_release_v1/job_360638/O/final_checkpoint'

def source():verify(PKG,read(PKG/'SELF_LOCK.json')['files']);return sha(PKG/'SELF_LOCK.json')
def validate_release(r):
    require(r['role']=='U' and r['parent_role']==r['teacher_role']=='O','U lineage scope')
    require(r['gpus']==1 and r['minutes']==120 and r['gpu_hours_cap']==2 and r['ntd_cap']==60 and r['rate']==30,'U resource/budget')
    require(r['fresh_optimizer'] is True and r['automatic_evaluation'] is False and r['M3_authorized'] is False,'U authorization scope')
    require(r['updates']==130 and r['parent_updates']==625,'schedule scope')
    return r

def dependencies():
    source();r=validate_release(read(PKG/'RELEASE.json'))
    require(sha(OR_SOURCE/'SELF_LOCK.json')==r['or_source_sha256'],'O/R source lock changed')
    import m2_stage_common_v1 as old
    old.dependencies();verify(ROOT,r['accepted_cluster_evidence'])
    return r

def parent_metadata(r,rehash=True):
    require(sha(PARENT/'MANIFEST.json')==r['parent_checkpoint_manifest_sha256'],'parent manifest drift')
    files=read(PARENT/'MANIFEST.json')['files']
    require(len(files)==4 and {x['path'] for x in files}=={'model.safetensors','optimizer.safetensors','rng.safetensors','metadata.json'},'parent file coverage')
    if rehash:verify(PARENT,files)
    m=read(PARENT/'metadata.json');ident=m['identity']
    require(m['schema']=='m2_paged_resume_v1' and ident['role']=='O' and ident['parent_role']=='B' and ident['source_lock_sha256']==r['or_source_sha256'],'parent lineage')
    require(m['cursor']==625 and len(m['optimizer_steps'])==146 and set(m['optimizer_steps'].values())=={625},'parent full schedule')
    require(canonical(m['model_tensors'])==r['parent_model_sha256'] and len(m['model_tensors'])==147,'parent model identity')
    require(ident['base_sha256']==r['base_model_sha256'] and ident['data_binding_sha256']==r['data_binding_sha256'],'parent data/base')
    return m

def setup(gpu):
    r=dependencies()
    import m2_stage_common_v1 as old
    torch,np,values,_=old.setup(gpu)
    sched=values[4]['U'];cfg=values[6]['training']['U']
    require(sched['total_updates']==130 and sched['sample_count']==4000 and cfg['expected_retain_draws']==4000,'U update/sample schedule')
    require((cfg['beta'],cfg['alpha_retain'],cfg['gamma_forget'])==(.1,1,1),'NPO coefficients')
    return torch,np,values,r

def accounting(text,jid):
    rows=[x.rstrip('|').split('|') for x in text.splitlines() if x.strip()]
    require(len(rows)==1 and len(rows[0])==5,'one-job accounting coverage')
    j,s,e,t,tr=[x.strip() for x in rows[0]]
    require(j==jid and s=='COMPLETED' and e=='0:0' and t.isdigit(),'U not successfully terminal')
    require([x for x in tr.split(',') if x.startswith('gres/gpu=')]==['gres/gpu=1'],'U GPU accounting')
    return int(t)
