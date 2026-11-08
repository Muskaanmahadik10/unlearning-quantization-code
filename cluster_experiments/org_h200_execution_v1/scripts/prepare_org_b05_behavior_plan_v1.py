"""Prepare the exact B0.5 behavior task plan without evaluating outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
from certify_org_e2b_b05_feature_wave_v1 import read, sha256_file, atomic
from certify_org_e2b_b05_feature_wave_v2 import EXPECTED, validate, require


def prepare(lock_path, output):
    lock = read(lock_path)
    require(lock.get('schema_version') == 'org_e2b_b05_selector_lock_v2' and lock.get('status') == 'PASS', 'selector lock status')
    require(lock.get('protected_outcomes_inspected') is False, 'selector outcome flag')
    ref_record = lock['reference_manifest']
    require(sha256_file(Path(ref_record['path'])) == ref_record['sha256'], 'reference manifest hash')
    refs = read(Path(ref_record['path']))
    require(set(lock['cells']) == {f'{c}/{q}' for c,q in EXPECTED}, 'six-cell universe')
    tasks = []
    for cell_key, record in sorted(lock['cells'].items()):
        path = Path(record['receipt']['path'])
        require(sha256_file(path) == record['receipt']['sha256'], 'feature receipt hash')
        key, checked = validate(path, refs)
        require(f'{key[0]}/{key[1]}' == cell_key and checked['conditions'] == record['conditions'], 'locked selector mismatch')
        item = read(path)
        prereq = item['prerequisites']
        tasks.append({
            'cell_id': key[0], 'quantizer': key[1], 'checkpoint': item['checkpoint'],
            'feature_receipt': record['receipt'], 'conditions': record['conditions'],
            'manifests': {k: prereq[k] for k in ('forget_manifest','retain_manifest','calibration_manifest')},
            'environment_lock': prereq['environment_lock'],
            'condition_count': 6,
            'unique_packed_sets': sorted({tuple(x['protected_layers']) for n,x in record['conditions'].items() if n != 'bf16'}),
            'feature_intervention': 'gate2_v4_backend.quantize_dequantize',
            'packed_intervention': {'nf4':'bitsandbytes_NF4','gptq_int4':'GPTQModel_INT4','rtn_int8':'B04_packed_RTN_INT8'}[key[1]],
            'cross_implementation_agreement_required': True,
        })
    report = {
        'schema_version':'org_b05_behavior_plan_v1','status':'PLAN_PREPARED',
        'selector_lock':{'path':str(lock_path.resolve()),'sha256':sha256_file(lock_path)},
        'cells':tasks, 'cell_count':6, 'condition_count':36,
        'protected_outcomes_generated':False, 'gpu_hours':0,
        'behavior_launcher_certified':False, 'behavior_launch_authorized':False,
        'pending_execution_contracts':[
            'behavior evaluator and sample-selection contract',
            'generalize B04 module inventory from public 112 to historical 224 targets',
            'producer plus two fresh consumers and per-sample repeatability checks',
            'packed versus emulated contrasts and frozen non-negligibility definition',
            'cumulative B05 accounting against protocol cap before further allocation',
        ],
    }
    atomic(output,report)
    print('B05_BEHAVIOR_PLAN=PREPARED\nCELLS=6\nCONDITIONS=36\nGPU_HOURS=0\nBEHAVIOR_LAUNCH_AUTHORIZED=false')
    print(f'OUTPUT={output}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selector-lock',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    prepare(a.selector_lock,a.output)
