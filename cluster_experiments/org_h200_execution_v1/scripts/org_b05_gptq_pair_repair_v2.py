"""One GPTQ pair retry, immutable NF4 reuse, capped by actual B05 accounting."""
import argparse
from pathlib import Path
import subprocess
import run_org_b05_behavior_cell_v2 as patched
import org_b05_remaining_backends_v1 as prior

life=patched.lifecycle
OLD=prior.ROOT
ROOT=Path('results/org_h200_execution_v1/b05_gptq_pair_repair/349840_1_v2')
SOURCE=Path('org_h200_execution_v1/ORG_B05_GPTQ_PAIR_REPAIR_V2_SHA256.txt')
NF4=OLD/'ordinary_tofu_ga_seed42__nf4/pair'

def verify_nf4():
    r=life.read(NF4/'condition_complete.json')
    life.require(r['status']=='PASS','NF4 prerequisite')
    for n,h in r['evidence'].items():
        life.require(life.sha256_file(NF4/n)==h,'NF4 evidence drift')
    prior.validate(NF4)
    return life.sha256_file(NF4/'condition_complete.json')

def seconds(text):
    seen={}
    for line in text.splitlines():
        row=[x.strip() for x in line.split('|')]
        if len(row)<4 or row[0] not in ('349840_0','349840_1'): continue
        j,state,elapsed,tres=row[:4]
        life.require(j not in seen,'duplicate accounting')
        life.require(state==('COMPLETED' if j=='349840_0' else 'FAILED'),'unexpected prior job state')
        fields=dict(x.split('=',1) for x in tres.split(',') if '=' in x)
        life.require(fields.get('gres/gpu')=='1','GPU allocation mismatch')
        life.require(int(elapsed)>0,'invalid elapsed')
        seen[j]=int(elapsed)
    life.require(set(seen)=={'349840_0','349840_1'},'missing accounting')
    return sum(seen.values())

def prepare():
    life.require(not ROOT.exists(),'already prepared; reconcile submission before retry')
    nf4hash=verify_nf4()
    oldauth=life.read(OLD/'authorization.json')
    oldtasks=life.read(OLD/'tasks.json')
    life.require(life.sha256_file(OLD/'authorization.json')==oldtasks['authorization_sha256'],'old authorization drift')
    life.require(oldauth['plan_sha256']==life.sha256_file(prior.PLAN),'plan changed')
    result=subprocess.run(['sacct','-X','-n','-P','-j','349840',
        '--format=JobID%30,State,ElapsedRaw,AllocTRES%300'],capture_output=True,text=True,check=True,timeout=30)
    used=oldauth['accounted_gpu_seconds']+seconds(result.stdout)
    life.require(used+3600<=7.5*3600,'retry exceeds cumulative 7.5 GPU-hour cap')
    ROOT.mkdir(parents=True)
    (ROOT/'prior_accounting.psv').write_text(result.stdout)
    auth=dict(oldauth)
    auth.update(cells=['ordinary_tofu_ga_seed42/gptq_int4'],source_manifest_sha256=life.sha256_file(SOURCE),
        accounted_gpu_seconds=used,additional_gpu_hours_ceiling=1,
        approval_record='Bounded GPTQ-only repair within existing cumulative budget; NF4 reused unchanged.',
        nf4_receipt_sha256=nf4hash)
    life.atomic(ROOT/'authorization.json',auth)
    life.atomic(ROOT/'tasks.json',{'tasks':[{'cell':'ordinary_tofu_ga_seed42','quantizer':'gptq_int4','condition':'pair'}],
        'authorization_sha256':life.sha256_file(ROOT/'authorization.json')})
    print('GPTQ_REPAIR_PREPARED=PASS\nRETRY_GPU_HOURS_MAX=1\nRETRY_NTD_MAX=30')

def run():
    life.require(verify_nf4()==life.read(ROOT/'authorization.json')['nf4_receipt_sha256'],'NF4 reference changed')
    prior.ROOT=ROOT
    prior.SOURCE=SOURCE
    prior.run(0)

def certify():
    nf4hash=verify_nf4()
    folder=ROOT/'ordinary_tofu_ga_seed42__gptq_int4/pair'
    r=life.read(folder/'condition_complete.json')
    life.require(r['status']=='PASS','GPTQ not passed')
    for n,h in r['evidence'].items():
        life.require(life.sha256_file(folder/n)==h,'GPTQ evidence drift')
    prior.validate(folder)
    life.atomic(ROOT/'backend_validation.json',{'status':'PASS','scope':'ordinary_nf4_gptq_pair',
        'nf4_receipt':str(NF4/'condition_complete.json'),'nf4_sha256':nf4hash,
        'gptq_receipt':str(folder/'condition_complete.json'),'gptq_sha256':life.sha256_file(folder/'condition_complete.json'),
        'full_transport_evaluated':False,'remaining_wave_authorized':False})
    print('B05_TWO_BACKEND_VALIDATIONS=PASS\nREMAINING_WAVE_AUTHORIZED=false')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('prepare','run','certify'))
    globals()[p.parse_args().mode]()
