"""Ordinary RTN pair behavior canary; two fresh consumers, all manifest samples.

Requires an externally reviewed budget authorization. Never authorizes production.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import run_org_b05_real_model_smoke_v1 as base
import run_org_b05_real_model_smoke_v2 as repaired
from org_b05_behavior_evaluator_v1 import evaluate, compare_reloads


def authorize(a):
    x = base.read(a.authorization)
    base.require(x.get('schema_version') == 'org_b05_behavior_canary_budget_v1', 'budget schema')
    base.require(x.get('status') == 'APPROVED' and x.get('scope') == 'ordinary_rtn_pair_canary', 'budget scope')
    base.require(x.get('plan_sha256') == base.sha256_file(a.plan), 'budget plan identity')
    base.require(x.get('runner_sha256') == base.sha256_file(Path(__file__)), 'budget runner identity')
    base.require(x.get('max_allocated_gpu_hours') == 0.5, 'canary allocation cap')
    base.require(x.get('cumulative_accounting_reviewed') is True, 'cumulative accounting not reviewed')
    base.require(bool(x.get('approval_record')), 'missing explicit approval record')
    return x


def worker(a):
    import transformers as tf
    from run_gate2_metric_suite_v42 import manifest_samples
    c, _, _ = base.entry(base.read(a.plan), 'ordinary_tofu_ga_seed42', 'rtn_int8')
    tokenizer = tf.AutoTokenizer.from_pretrained(c['checkpoint']['path'], local_files_only=True,
                                                trust_remote_code=False)
    samples = {}
    for role in ('forget', 'retain'):
        _, samples[role] = manifest_samples(c['manifests'][role+'_manifest']['path'], 128,
                                           forget_contract=(role == 'forget'))
    original = repaired.original_synthetic

    def forward_and_evaluate(torch, model):
        synthetic = original(torch, model)
        # Repaired worker calls this twice, independently, to test repeatability.
        result = {role: evaluate(model, tokenizer, rows) for role, rows in samples.items()}
        path = a.output/(a.phase+'_behavior.json')
        if not path.exists():
            if a.phase != 'produce':
                expected = base.read(a.output/'produce_behavior.json')
                for role in samples:
                    compare_reloads(expected[role]['rows'], result[role]['rows'])
            base.atomic(path, result)
        else:
            expected = base.read(path)
            for role in samples:
                compare_reloads(expected[role]['rows'], result[role]['rows'])
            base.atomic(a.output/(a.phase+'_behavior_repeat.json'), {'status': 'PASS'})
        return synthetic

    try:
        repaired.original_synthetic = forward_and_evaluate
        repaired.worker(SimpleNamespace(plan=a.plan, output=a.output, phase=a.phase,
                                       cell='ordinary_tofu_ga_seed42', quantizer='rtn_int8'))
    finally:
        repaired.original_synthetic = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--authorization', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=('all','produce','reload_a','reload_b'), default='all')
    a = p.parse_args()
    authorize(a)
    if a.phase != 'all':
        worker(a)
        return
    base.require(not a.output.exists(), 'output exists')
    a.output.mkdir(parents=True)
    phases = ('produce','reload_a','reload_b')
    for phase in phases:
        subprocess.run([sys.executable, __file__, '--plan', str(a.plan), '--authorization',
                        str(a.authorization), '--output', str(a.output), '--phase', phase], check=True)
    receipts = [base.read(a.output/(phase+'.json')) for phase in phases]
    base.require(len({r['pid'] for r in receipts}) == 3, 'process independence')
    for phase in phases:
        base.require(base.read(a.output/(phase+'_behavior_repeat.json'))['status'] == 'PASS', 'repeat check')
    baseline = base.read(a.output/'produce_behavior.json')
    for phase in phases[1:]:
        result = base.read(a.output/(phase+'_behavior.json'))
        for role in ('forget','retain'):
            compare_reloads(baseline[role]['rows'], result[role]['rows'])
    base.atomic(a.output/'final.json', {
        'schema_version': 'org_b05_behavior_rtn_canary_v1', 'status': 'PASS',
        'scope': 'ordinary_rtn_pair_only', 'sample_counts': {'forget':128,'retain':128},
        'plan_sha256': base.sha256_file(a.plan),
        'authorization_sha256': base.sha256_file(a.authorization),
        'receipts': receipts, 'production_authorized': False,
        'evidence': {p.name:base.sha256_file(p) for p in a.output.glob('*.json')}})
    print('B05_RTN_BEHAVIOR_CANARY=PASS\nPRODUCTION_AUTHORIZED=false')


if __name__ == '__main__':
    main()
