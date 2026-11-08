"""Fail-closed serial-to-condition-parallel handoff for job 349643 only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import run_org_b05_behavior_cell_v1 as cell
from org_b05_behavior_evaluator_v1 import compare_reloads

BASE=Path('results/org_h200_execution_v1')
OLD=BASE/'b05_behavior_cell_candidate/ordinary_tofu_ga_seed42_rtn_int8_349643'
NEW=BASE/'b05_behavior_parallel/repair_349643_v1'
PLAN=BASE/'b05_selector_locks/behavior_plan_v1.json'
SOURCE=Path('org_h200_execution_v1/ORG_B05_PARALLEL_HANDOFF_V1_SHA256.txt')
PHASES=('produce','reload_a','reload_b','emulated')


def terminal():
    queue=subprocess.run(['squeue','-h','-j','349643','-o','%i %T'],capture_output=True,text=True,check=True,timeout=20)
    cell.life.require(not queue.stdout.strip(),'parent still in queue; wait for cleanup')
    result=subprocess.run(['sacct','-X','-n','-P','-j','349643','--format=JobIDRaw,State,ElapsedRaw,AllocTRES'],
                          capture_output=True,text=True,check=True,timeout=20)
    rows=[r.split('|') for r in result.stdout.splitlines() if r.startswith('349643|')]
    cell.life.require(len(rows)==1 and rows[0][1].split()[0] in
        ('COMPLETED','CANCELLED','FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'),
        'parent not confirmed terminal; stop, do not launch')
    return result.stdout


def phase_dir(folder,phase): return folder/'emulated' if phase=='emulated' else folder


def complete(folder,phase):
    root=phase_dir(folder,phase)
    if not (root/(phase+'.json')).exists(): return False
    cell.life.require(cell.life.read(root/(phase+'.json')).get('status')=='PASS','bad phase receipt')
    result=cell.life.read(root/(phase+'_behavior.json'))
    for role in ('forget','retain'):
        rows=result[role]['rows']; cell.life.require(len(rows)==128,'sample count')
        compare_reloads(rows,rows)
    return True


def copy_phase(old,new,phase):
    src=phase_dir(old,phase); dst=phase_dir(new,phase); dst.mkdir(parents=True,exist_ok=True)
    names=[phase+'.json',phase+'_behavior.json']
    if phase=='produce': names+=['source.json','state.json','artifact_hashes.json','produce_logits.pt','runtime_buffers.json']
    elif phase in ('reload_a','reload_b') and (src/(phase+'_logits.pt')).exists(): names+=[phase+'_logits.pt']
    for name in names:
        shutil.copy2(src/name,dst/name)
    if phase=='produce':
        hashes=cell.life.read(src/'artifact_hashes.json')
        actual={str(p.relative_to(src/'artifact')) for p in (src/'artifact').rglob('*') if p.is_file()}
        cell.life.require(actual==set(hashes),'parent artifact universe')
        for name,h in hashes.items(): cell.life.require(cell.life.sha256_file(src/'artifact'/name)==h,'parent artifact drift')
        # Copy, not link: old evidence cannot be changed by future consumers.
        shutil.copytree(src/'artifact',dst/'artifact')


def prepare():
    accounting=terminal()
    cell.life.require(not (OLD/'final.json').exists(),'serial final already exists; validate it instead of handing off')
    cell.life.require(not NEW.exists(),'handoff already exists; inspect instead of resubmitting')
    NEW.mkdir(parents=True)
    (NEW/'parent_accounting.psv').write_text(accounting)
    a=SimpleNamespace(plan=PLAN,cell='ordinary_tofu_ga_seed42',quantizer='rtn_int8')
    _,_,sets=cell.context(a)
    tasks=[]; reused={}
    for name in cell.CONDITIONS:
        if sets[name]['alias_of']: continue
        old=OLD/name; new=NEW/name; new.mkdir()
        reused[name]=[]
        for phase in PHASES:
            if phase in ('reload_a','reload_b') and 'produce' not in reused[name]: continue
            if complete(old,phase):
                copy_phase(old,new,phase); reused[name].append(phase)
        if len(reused[name])<4: tasks.append(name)
    auth={'schema_version':'org_b05_behavior_wave_authorization_v1','status':'APPROVED',
          'cells':['ordinary_tofu_ga_seed42/rtn_int8'],'plan_sha256':cell.life.sha256_file(PLAN),
          'source_manifest_sha256':cell.life.sha256_file(SOURCE),'max_job_gpu_hours':2,
          'parallel_task_limit_gpu_hours':0.5,'parallel_task_count':len(tasks),
          'parallel_additional_gpu_hours_ceiling':len(tasks)*0.5,
          'cumulative_cap_gpu_hours':7.5,'previous_cap_gpu_hours':5,
          'non_negligible_nll':1e-6,'cumulative_accounting_reviewed':True,
          'approval_record':'User explicitly authorized condition parallelization and budget change, 2026-09-07; only replacement of job 349643.',
          'full_wave_authorized':False}
    cell.life.atomic(NEW/'authorization.json',auth)
    cell.life.atomic(NEW/'handoff.json',{'tasks':tasks,'reused':reused,'parent_job':349643,
         'copied_evidence':{str(p.relative_to(NEW)):cell.life.sha256_file(p) for p in NEW.rglob('*') if p.is_file()}})
    print('REUSED_PHASES='+str(reused)); print('PARALLEL_TASKS='+str(tasks))


def check_handoff():
    h=cell.life.read(NEW/'handoff.json')
    for name,digest in h['copied_evidence'].items():
        cell.life.require(cell.life.sha256_file(NEW/name)==digest,'copied evidence changed: '+name)
    return h


def run(index):
    h=check_handoff(); name=h['tasks'][index]; root=NEW/name
    with (root/'execution_claim.txt').open('x') as f: f.write(os.environ.get('SLURM_JOB_ID','unknown'))
    for phase in PHASES:
        if phase in h['reused'][name]: continue
        dest=phase_dir(root,phase); dest.mkdir(exist_ok=True)
        subprocess.run([sys.executable,str(Path(cell.__file__)), '--plan',str(PLAN),'--authorization',str(NEW/'authorization.json'),
            '--source-manifest',str(SOURCE),'--cell','ordinary_tofu_ga_seed42','--quantizer','rtn_int8',
            '--condition',name,'--phase',phase,'--output',str(dest)],check=True)
    cell.life.atomic(root/'condition_complete.json',{'status':'PASS','condition':name})
    print('B05_PARALLEL_CONDITION=PASS '+name)


def aggregate():
    check_handoff()
    _,_,sets=cell.context(SimpleNamespace(plan=PLAN,cell='ordinary_tofu_ga_seed42',quantizer='rtn_int8'))
    packed={}; emulated={}; records={}
    for name in cell.CONDITIONS:
        alias=sets[name]['alias_of']
        if alias:
            packed[name]=packed[alias]; emulated[name]=emulated[alias]; records[name]={'alias_of':alias,'record':records[alias]}; continue
        root=NEW/name
        cell.life.require(all(complete(root,p) for p in PHASES),'incomplete condition '+name)
        receipts=[cell.life.read(root/(p+'.json')) for p in PHASES[:3]]
        cell.life.require(len({r['pid'] for r in receipts})==3,'fresh processes')
        baseline=cell.life.read(root/'produce_behavior.json')
        for phase in PHASES[1:3]:
            candidate=cell.life.read(root/(phase+'_behavior.json'))
            for role in ('forget','retain'): compare_reloads(baseline[role]['rows'],candidate[role]['rows'])
        packed[name]=candidate; emulated[name]=cell.life.read(root/'emulated/emulated_behavior.json')
        records[name]={'path':str(root/'reload_b_behavior.json'),'sha256':cell.life.sha256_file(root/'reload_b_behavior.json')}
    report=cell.transport(packed,emulated)
    cell.life.atomic(NEW/'final.json',{'status':'EXECUTION_COMPLETE','transport':report,'conditions':records,
        'production_authorized':False,'evidence':{str(p.relative_to(NEW)):cell.life.sha256_file(p) for p in NEW.rglob('*.json')}})
    print('B05_PARALLEL_AGGREGATION=COMPLETE\nTRANSPORT='+report['status'])


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('prepare','run','aggregate')); p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.mode=='prepare': prepare()
    elif a.mode=='run': run(a.index)
    else: aggregate()
