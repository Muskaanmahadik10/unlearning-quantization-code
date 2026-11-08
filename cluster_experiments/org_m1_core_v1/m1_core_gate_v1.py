"""Production gates: pinned smoke reconciliation, complete all-arm fresh reload."""
from pathlib import Path
from m1_contract_v1 import PACKAGE, PROJECT, read, require, sha, ARMS
from reconcile_m1_smokes_v1 import behavior_check

def reconciliation_gate():
    pin=read(PACKAGE/'CORE_PLAN.json')['reconciliation']
    p=PROJECT/pin['path']
    require(sha(p)==pin['sha256'],'reconciliation receipt changed')
    r=read(p)
    require(r['status']=='SIX_CELL_SMOKE_RECONCILIATION_PASS' and len(r['cells'])==6,'smoke reconciliation failed')
    require(r['gpu_hours_all_smoke_attempts']+4<=7,'core maximum exceeds M1 cap')
    return r

def validate_core(folder,index):
    design=read(PACKAGE/'M1_EXPERIMENT_DESIGN_V1.json'); cell=design['cells'][index]
    f=read(folder/'final.json')
    require(f['status']=='PASS' and f['scope']=='M1_CORE_CELL_ONLY' and f['cell']==cell['cell'],'core final identity')
    require(f['source_lock_sha256']==sha(PACKAGE/'SOURCE_LOCK.json'),'core source mismatch')
    require(set(f['phases'])=={'produce','reload'},'core phase coverage')
    records={}
    for phase in ('produce','reload'):
        p=folder/(phase+'.json'); require(sha(p)==f['phases'][phase],'core phase hash')
        r=read(p); records[phase]=r
        require(r['status']=='PASS' and r['phase']==phase and r['cell']==cell['cell'],'phase identity')
        require(r['source_lock_sha256']==f['source_lock_sha256'] and r['requantized'] is False,'phase source')
        require(set(r['arms'])==set(ARMS),'all ten arms required')
        for arm,v in r['arms'].items():
            from m1_contract_v1 import arm_targets
            targets,dose=arm_targets(cell,arm)
            require(v['status']=='PASS' and v['targets']==targets and v['dose']==dose,'arm identity')
            require(v['behavior'] is not None,'production behavior missing')
            if arm!='N0': require(sha(folder/(arm+'.safetensors'))==v['patch_sha256'],'patch drift')
            require(sha(folder/(arm+'_logits.pt'))==v['logits_sha256'],'logit drift')
    a,b=records['produce'],records['reload']
    require(a['runtime']['pid']!=b['runtime']['pid'] and a['runtime']['process_uuid']!=b['runtime']['process_uuid'],'fresh process failed')
    require(a['runtime']['gpu_uuid']==b['runtime']['gpu_uuid'],'GPU changed')
    require(a['decode_reports']==b['decode_reports'],'decoded/original identity changed')
    for arm in ARMS:
        require(a['arms'][arm]['patch_sha256']==b['arms'][arm]['patch_sha256'],'phase patch mismatch')
        require(a['arms'][arm]['logits_sha256']==b['arms'][arm]['logits_sha256'],'phase logits mismatch')
        behavior_check(a['arms'][arm]['behavior'],b['arms'][arm]['behavior'])
    return {'index':index,'cell':cell['cell'],'status':'PASS','final_sha256':sha(folder/'final.json')}
