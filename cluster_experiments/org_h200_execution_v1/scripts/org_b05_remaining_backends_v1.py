"""Two bounded backend validations; outputs reserved for the remaining B05 wave."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import run_org_b05_behavior_cell_v1 as cell
from org_b05_behavior_evaluator_v1 import compare_reloads

ROOT=Path('results/org_h200_execution_v1/b05_remaining_backends_v1')
PLAN=Path('results/org_h200_execution_v1/b05_selector_locks/behavior_plan_v1.json')
SOURCE=Path('org_h200_execution_v1/ORG_B05_REMAINING_BACKENDS_V1_SHA256.txt')
RTN=Path('results/org_h200_execution_v1/b05_behavior_parallel/repair_349643_v1')
PHASES=('produce','reload_a','reload_b','emulated')
PAIRS=(('ordinary_tofu_ga_seed42','nf4'),('ordinary_tofu_ga_seed42','gptq_int4'))


def allocation_seconds(text):
    expected={'349643','349745_0','349745_1','349745_2'}; seen={}
    for line in text.splitlines():
        parts=[p.strip() for p in line.split('|')]
        if len(parts)<4 or parts[0] not in expected: continue
        job,state,seconds,tres=parts[:4]
        cell.life.require(job not in seen,'duplicate accounting')
        cell.life.require(state.split()[0] in ('COMPLETED','CANCELLED','FAILED','TIMEOUT'),'nonterminal parent')
        fields=dict(v.split('=',1) for v in tres.split(',') if '=' in v)
        cell.life.require(fields.get('gres/gpu')=='1','expected one GPU per allocation')
        seen[job]=int(seconds)
    cell.life.require(set(seen)==expected,'incomplete parent accounting')
    return sum(seen.values())


def prepare():
    cell.life.require(not ROOT.exists(),'prepared previously; inspect, do not resubmit')
    final=cell.life.read(RTN/'final.json')
    cell.life.require(final['status']=='EXECUTION_COMPLETE' and final['transport']['status']=='PASS','ordinary RTN prerequisite')
    for name,h in final['evidence'].items():
        cell.life.require(cell.life.sha256_file(RTN/name)==h,'RTN evidence drift')
    result=subprocess.run(['sacct','-X','-n','-P','-j','349643,349745',
        '--format=JobID%30,State,ElapsedRaw,AllocTRES%300'],capture_output=True,text=True,check=True,timeout=30)
    used=10173+allocation_seconds(result.stdout)
    cell.life.require(used+7200<=7.5*3600,'two validation allocations exceed 7.5 cumulative cap')
    tasks=[]
    for cid,q in PAIRS:
        c,targets,sets=cell.context(SimpleNamespace(plan=PLAN,cell=cid,quantizer=q))
        cell.life.require(sets['pair']['alias_of'] is None,'pair alias')
        tasks.append({'cell':cid,'quantizer':q,'condition':'pair'})
    ROOT.mkdir(parents=True)
    (ROOT/'prior_accounting.psv').write_text(result.stdout)
    auth={'schema_version':'org_b05_behavior_wave_authorization_v1','status':'APPROVED',
        'cells':[f'{c}/{q}' for c,q in PAIRS],'plan_sha256':cell.life.sha256_file(PLAN),
        'source_manifest_sha256':cell.life.sha256_file(SOURCE),'max_job_gpu_hours':2,
        'task_limit_gpu_hours':1,'authorized_conditions':['pair'],
        'cumulative_accounting_reviewed':True,'accounted_gpu_seconds':used,
        'cumulative_cap_gpu_hours':7.5,'additional_gpu_hours_ceiling':2,
        'non_negligible_nll':1e-6,
        'approval_record':'User requested next phase ASAP after ordinary RTN transport PASS; two bounded backend validations only within existing 7.5 cap.',
        'remaining_full_wave_authorized':False}
    cell.life.atomic(ROOT/'authorization.json',auth)
    cell.life.atomic(ROOT/'tasks.json',{'tasks':tasks,'authorization_sha256':cell.life.sha256_file(ROOT/'authorization.json'),
         'ordinary_rtn_final_sha256':cell.life.sha256_file(RTN/'final.json')})
    print('B05_BACKEND_VALIDATION_PREPARED=PASS\nADDITIONAL_GPU_HOURS_MAX=2\nADDITIONAL_NTD_MAX=60')


def run(index):
    manifest=cell.life.read(ROOT/'tasks.json')
    cell.life.require(cell.life.sha256_file(ROOT/'authorization.json')==manifest['authorization_sha256'],'authorization drift')
    task=manifest['tasks'][index]
    cell.life.require((task['cell'],task['quantizer']) in PAIRS and task['condition']=='pair','task scope')
    folder=ROOT/(task['cell']+'__'+task['quantizer'])/'pair'
    folder.mkdir(parents=True)  # Exclusive creation: no duplicate writer/retry.
    for phase in PHASES:
        dest=folder if phase!='emulated' else folder/'emulated'
        if phase=='emulated': dest.mkdir()
        subprocess.run([sys.executable,str(Path(cell.__file__)), '--plan',str(PLAN),
            '--authorization',str(ROOT/'authorization.json'),'--source-manifest',str(SOURCE),
            '--cell',task['cell'],'--quantizer',task['quantizer'],'--condition','pair',
            '--phase',phase,'--output',str(dest)],check=True)
    validate(folder)
    cell.life.atomic(folder/'condition_complete.json',{'status':'PASS','task':task,
        'job_id':os.environ.get('SLURM_JOB_ID'),'production_authorized':False,
        'evidence':{str(p.relative_to(folder)):cell.life.sha256_file(p) for p in folder.rglob('*.json')}})
    print('B05_BACKEND_PAIR=PASS '+task['quantizer'])


def validate(folder):
    receipts=[cell.life.read((folder if p!='emulated' else folder/'emulated')/(p+'.json')) for p in PHASES]
    cell.life.require(all(r['status']=='PASS' for r in receipts),'phase failure')
    cell.life.require(len({r['pid'] for r in receipts[:3]})==3,'fresh process identity')
    baseline=cell.life.read(folder/'produce_behavior.json')
    for p in ('reload_a','reload_b'):
        result=cell.life.read(folder/(p+'_behavior.json'))
        for role in ('forget','retain'):
            cell.life.require(len(result[role]['rows'])==128,'sample count')
            compare_reloads(baseline[role]['rows'],result[role]['rows'])


def certify():
    for cid,q in PAIRS:
        folder=ROOT/(cid+'__'+q)/'pair'
        report=cell.life.read(folder/'condition_complete.json')
        cell.life.require(report['status']=='PASS','condition status')
        for name,h in report['evidence'].items():
            cell.life.require(cell.life.sha256_file(folder/name)==h,'condition evidence drift')
        validate(folder)
    cell.life.atomic(ROOT/'backend_validation.json',{'status':'PASS','scope':'ordinary_nf4_gptq_pair',
        'full_transport_evaluated':False,'remaining_wave_authorized':False,
        'pair_receipts':{q:cell.life.sha256_file(ROOT/(cid+'__'+q)/'pair/condition_complete.json') for cid,q in PAIRS}})
    print('B05_TWO_BACKEND_VALIDATIONS=PASS\nREMAINING_WAVE_AUTHORIZED=false')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('prepare','run','certify')); p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.mode=='prepare': prepare()
    elif a.mode=='run': run(a.index)
    else: certify()
