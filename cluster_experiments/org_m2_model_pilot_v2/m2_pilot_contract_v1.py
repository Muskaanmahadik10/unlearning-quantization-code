"""Offline contracts for a bounded M2 model-level pilot, not production."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time

ROOT=Path('/work/anon_user/hessian_unlearning')
ENV=Path('/work/anon_user/.runtime/envs/org-packed-h200-v1')
PKG=Path(__file__).resolve().parent


def require(ok, message):
    if not ok: raise ValueError(message)


def unique(pairs):
    result={}
    for k,v in pairs:
        require(k not in result,'duplicate JSON key');result[k]=v
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=unique,
                      parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def canonical(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(path, obj):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def confined(root, rel):
    require(isinstance(rel,str) and rel and '\\' not in rel and ':' not in rel,'invalid path')
    p=PurePosixPath(rel)
    require(not p.is_absolute() and '..' not in p.parts and p.as_posix()==rel,'noncanonical path')
    target=root/rel
    require(target.resolve().is_relative_to(root.resolve()) and target.is_file(),'missing/escaped input '+rel)
    return target


def manifest_schema(rows, hash_only=False):
    require(isinstance(rows,list) and rows,'empty/invalid manifest')
    paths=[]
    for r in rows:
        require(isinstance(r,dict) and isinstance(r.get('path'),str),'invalid manifest row/path')
        require(isinstance(r.get('sha256'),str) and re.fullmatch('[0-9a-f]{64}',r['sha256']) is not None,'missing/invalid manifest SHA256')
        if hash_only:
            require(set(r)=={'path','sha256'},'unexpected hash-only manifest fields')
        else:
            require(type(r.get('bytes')) is int and r['bytes']>=0,'missing/invalid manifest byte count')
        paths.append(r['path'])
    require(len(paths)==len(set(paths)),'duplicate manifest paths')


def verify(root, rows, hash_only=False):
    manifest_schema(rows,hash_only)
    for r in rows:
        p=confined(root,r['path'])
        if not hash_only:require(p.stat().st_size==r['bytes'],'input size drift '+str(p))
        require(sha(p)==r['sha256'],'input hash drift '+str(p))


def source():
    verify(PKG,read(PKG/'SELF_LOCK.json')['files'])
    return sha(PKG/'SELF_LOCK.json')


def pilot_indices(role):
    require(role in ('R','U'),'pilot role must be R or U')
    # A declared coverage exercise, not a contiguous scientific training run.
    return [0,1,2,112 if role=='R' else 12]


def job_id(text):
    m=re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?\s*',text)
    require(m is not None,'ambiguous submission; do not retry');return m[1]


def budget(plan):
    require(plan['tasks']==['R','U'] and plan['gpus_per_task']==1 and plan['minutes_per_task']==30,'resource drift')
    require(plan['gpu_hours_max']==1 and plan['ntd_max']==30 and plan['production_authorized'] is False,'budget/authority drift')
    return 1.0


def bind_inputs():
    source()
    plan=read(PKG/'PILOT_PLAN.json');budget(plan)
    base=read(PKG/'BASE_INPUT_IDENTITY_LOCK.json')
    binding=read(PKG/'RUNTIME_DATA_BINDING.json')
    cpu=read(PKG/'cpu_evidence/final.json')
    require(cpu['status']=='CPU_COMPONENTS_AND_TOKENIZER_PASS' and cpu['tests']==23 and
            cpu['training_authorized'] is False,'CPU return not accepted')
    for name in ('training_tokens','training_schedule'):
        require(sha(PKG/('cpu_evidence/'+name+'.json'))==cpu[name+'_sha256'],'CPU output drift')
    require(sha(PKG/'BASE_INPUT_IDENTITY_LOCK.json')==cpu['base_lock_sha256'],'base identity drift')
    design=read(PKG/'CONTROLLED_OUR_DESIGN_V2.json')
    partition=read(PKG/'OBSERVED_LABEL_PARTITION_V2.json')
    require(sha(PKG/'OBSERVED_LABEL_PARTITION_V2.json')==design['data']['partition_sha256'],'partition drift')
    tokens=read(PKG/'cpu_evidence/training_tokens.json')
    from m2_ambiguity_contract_v2 import bind_calibration_tokens
    cal=read(PKG/'CALIBRATION_TOKEN_BINDING_V1.json')
    recal=bind_calibration_tokens(partition,tokens)
    require(cal['records']==recal['records'] and cal['selected_tokens_sha256']==recal['selected_tokens_sha256'], 'calibration drift')
    require(cal['cpu_final_sha256']==sha(PKG/'cpu_evidence/final.json'),'calibration CPU binding')
    from m2_training_core_v1 import schedule
    sched=read(PKG/'cpu_evidence/training_schedule.json')
    require(sched=={r:schedule(r,design['training'][r]) for r in ('O','R','U')},'schedule drift')
    return plan,base,binding,tokens,sched,cal,design


def cluster_preflight(full_environment=True):
    require(sys.platform=='linux' and Path(sys.prefix)==ENV and sys.flags.isolated and sys.dont_write_bytecode,
            'use certified Linux python with -I -B')
    require(ROOT.is_dir(),'cluster root missing')
    values=bind_inputs();plan,base,binding=values[:3]
    lockpath=ROOT/binding['environment_lock_path']
    require(sha(lockpath)==binding['environment_lock_sha256'],'environment lock drift')
    lock=read(lockpath)
    require(lock['overall_status']=='PASS' and sha(sys.executable)==lock['interpreter_identity']['executable_sha256'], 'interpreter identity')
    envmanifest=Path(lock['environment_tree_manifest']['path'])
    require(sha(envmanifest)==lock['environment_tree_manifest']['sha256'],'environment manifest drift')
    entries=read(envmanifest)
    if full_environment:
        verify(ENV,entries['files'])
        for r in entries['symlinks']:
            p=ENV/r['path']
            require(not Path(r['path']).is_absolute() and '..' not in Path(r['path']).parts and
                    p.parent.resolve().is_relative_to(ENV.resolve()) and p.is_symlink() and os.readlink(p)==r['target'], 'environment symlink drift')
            # The certified venv's interpreter may deliberately link outside its
            # prefix. Its target bytes are separately bound above, not rejected.
    setup=Path(binding['setup_run'])
    require(sha(setup/'overlay_file_manifest.json')==binding['overlay_manifest_sha256'],'overlay manifest drift')
    verify(setup,read(setup/'overlay_file_manifest.json')['files'],hash_only=True)
    verify(Path(base['snapshot_path']),base['files'])
    verify(ROOT/'org_m2_evaluator_setup_v1',binding['data_files'])
    require(shutil.disk_usage(ROOT).free>=plan['shared_free_bytes_min'],'insufficient shared free space')
    sys.path.insert(0,str(setup/'site-packages'))
    require({n:importlib.metadata.version(n) for n in binding['effective_versions']}==binding['effective_versions'],'runtime version drift')
    return values


def outcome_compare(a,b):
    for key in ('model_tensors','optimizer_tensors','optimizer_steps','optimizer_groups','aliases',
                'buffer_schema','rng_probe','selected_update_indices','update_results','identity','role','final_model_sha256'):
        require(a[key]==b[key],'fresh resume mismatch '+key)
    require(a['status']==b['status']=='PILOT_BRANCH_PASS' and a['phase']=='produce' and b['phase']=='resume','branch status/phase')
    require(a['pid']!=b['pid'],'fresh process not demonstrated')
    require(a['teacher_initial_sha256']==a['teacher_final_sha256']==b['teacher_initial_sha256']==b['teacher_final_sha256'], 'teacher drift')
    require(a['gpu']==b['gpu'],'GPU identity changed inside one task')
    require(a['base_model_sha256']==b['base_model_sha256'] and a['base_model_sha256']!=a['final_model_sha256'], 'no observed student update/base drift')
    return True


def submission_env(environ):
    result=dict(environ)
    for name in ('CUDA_VISIBLE_DEVICES','PYTHONPATH','PYTHONHOME','BNB_CUDA_VERSION'):
        result.pop(name,None)
    return result


def allocation_guard(role,output,environ=None):
    e=os.environ if environ is None else environ
    require(role in ('R','U'),'pilot role')
    jid=job_id(e.get('SLURM_ARRAY_JOB_ID',''))
    require(e.get('SLURM_JOB_ID') and e.get('SLURM_ARRAY_TASK_ID')==str(['R','U'].index(role)), 'Slurm role/index identity')
    require(e.get('SLURM_JOB_NAME')=='org_m2_pilot_v2' and e.get('SLURM_JOB_PARTITION')=='dev', 'allocation name/partition')
    require(e.get('CUDA_VISIBLE_DEVICES','').strip() not in ('','-1','NoDevFiles'),'GPU allocation visibility missing')
    expected=ROOT/'results/org_m2_model_pilot_v2'/('job_'+jid)/role
    require(Path(output).resolve()==expected.resolve(),'wrong pilot output')
    claim=ROOT/'results/org_m2_model_pilot_v2/launch_v1'
    request=read(claim/'request.json')
    require(request['source_lock_sha256']==source() and request['max_gpu_hours']==1 and request['max_ntd']==30,'pilot request/source drift')
    cp=Path(request['cpu_receipt'])
    require(cp.resolve().is_relative_to((ROOT/'results/org_m2_model_pilot_v2').resolve()) and sha(cp)==request['cpu_receipt_sha256'],'CPU receipt drift')
    cpu=read(cp)
    require(cpu['status']=='PILOT_COMPONENT_CPU_PASS' and cpu['tests']==read(PKG/'PILOT_PLAN.json')['cluster_tests'] and cpu['source_lock_sha256']==source(),'CPU admission not passed')
    # sbatch can launch an allocation before its acknowledgement is persisted.
    for _ in range(50):
        if (claim/'submitted.json').is_file() or (claim/'SUBMISSION_UNKNOWN.json').exists():break
        time.sleep(.2)
    submitted=read(claim/'submitted.json')
    require(submitted['job_id']==jid and submitted['tasks']==['R','U'] and submitted['gpu_hours_max']==1 and submitted['ntd_max']==30,'unrecorded/duplicate allocation')
    return jid
