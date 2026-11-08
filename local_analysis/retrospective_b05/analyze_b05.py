"""Retrospective, descriptive reanalysis of the frozen B05 contrast export.

No model execution, threshold selection, inferential tests or admission changes.
Uses decimal arithmetic on the exported values and explicit paired identities.
"""
import argparse
import csv
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

COHORTS = {'adverse_muse_gdr_seed2024', 'ordinary_tofu_ga_seed42'}
QUANTIZERS = {'gptq_int4', 'nf4', 'rtn_int8'}
CONDITIONS = {'safe_looking_singleton', 'risky_looking_singleton', 'pair'}
ROUTES = {'prediction': 'linear_prediction', 'packed': 'packed_delta', 'emulated': 'emulated_delta'}
EPS = Decimal('0.000001')
FLOORS = ('0.000001', '0.001', '0.01')


def sign(value, floor=EPS):
    return 1 if value > floor else -1 if value < -floor else 0


def favorable(forget, retain, floor=EPS):
    return forget > floor and retain < -floor


def load(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if fields is None or len(fields) != len(set(fields)):
            raise ValueError('missing or duplicate columns')
        if not {'cell', 'quantizer', 'condition', 'role', 'selected_layers', *ROUTES.values()} <= set(fields):
            raise ValueError('missing required columns')
        rows = list(reader)
    expected = {(c, q, k, r) for c in COHORTS for q in QUANTIZERS for k in CONDITIONS for r in ('forget', 'retain')}
    seen = set()
    for r in rows:
        key = tuple(r[k] for k in ('cell', 'quantizer', 'condition', 'role'))
        if key in seen or key not in expected:
            raise ValueError('duplicate or unexpected row ' + str(key))
        seen.add(key)
        if not r['selected_layers']:
            raise ValueError('missing intervention identity')
        for name in ROUTES.values():
            try:
                value = Decimal(r[name])
            except (InvalidOperation, TypeError):
                raise ValueError('invalid numeric value ' + name)
            if not value.is_finite():
                raise ValueError('nonfinite numeric value ' + name)
            r[name] = value
    if seen != expected:
        raise ValueError('incomplete factorial coverage')
    return rows


def analyze(rows):
    paired = {}
    decisions = []
    for r in rows:
        key = tuple(r[k] for k in ('cell', 'quantizer', 'condition'))
        paired.setdefault(key, {})[r['role']] = r
        p, e, h = (r[k] for k in ('packed_delta', 'emulated_delta', 'linear_prediction'))
        pe = sign(p) != 0 and sign(p) == sign(e)
        pp = sign(p) != 0 and sign(p) == sign(h)
        decisions.append({**r, 'packed_emulated_agree': pe, 'prediction_packed_agree': pp})
    pairs = []
    for key, role in sorted(paired.items()):
        if role['forget']['selected_layers'] != role['retain']['selected_layers']:
            raise ValueError('forget/retain intervention identity mismatch ' + str(key))
        item = dict(zip(('cell', 'quantizer', 'condition'), key))
        item['selected_layers'] = role['forget']['selected_layers']
        for route, column in ROUTES.items():
            item[route + '_forget_delta'] = role['forget'][column]
            item[route + '_retain_delta'] = role['retain'][column]
            item[route + '_joint_favorable'] = favorable(role['forget'][column], role['retain'][column])
        pairs.append(item)
    sensitivity = []
    for raw_floor in FLOORS:
        floor = Decimal(raw_floor)
        eligible = [r for r in rows if abs(r['packed_delta']) > floor and abs(r['emulated_delta']) > floor]
        opposite = sum((r['packed_delta'] > 0) != (r['emulated_delta'] > 0) for r in eligible)
        sensitivity.append({'strict_absolute_floor_nll': raw_floor, 'eligible_contrasts': len(eligible), 'opposite_signs': opposite})
    groups = []
    for cohort in sorted(COHORTS):
        for quantizer in sorted(QUANTIZERS):
            subset = [r for r in pairs if r['cell'] == cohort and r['quantizer'] == quantizer]
            groups.append({'cell': cohort, 'quantizer': quantizer, 'interventions': len(subset), **{
                route: sum(r[route + '_joint_favorable'] for r in subset) for route in ROUTES}})
    summary = {
        'analysis_status': 'RETROSPECTIVE_DESCRIPTIVE_NOT_NEW_ADMISSION_RULE',
        'rows': len(rows), 'paired_interventions': len(pairs), 'checkpoint_units': len(COHORTS),
        'joint_direction_rule': 'forget_delta > 0.000001 AND retain_delta < -0.000001',
        'contrast': 'protected minus full condition, answer-token-weighted mean NLL',
        'sign_partition': {
            'both_agree': sum(r['packed_emulated_agree'] and r['prediction_packed_agree'] for r in decisions),
            'packed_emulated_only': sum(r['packed_emulated_agree'] and not r['prediction_packed_agree'] for r in decisions),
            'prediction_packed_only': sum(not r['packed_emulated_agree'] and r['prediction_packed_agree'] for r in decisions),
            'neither': sum(not r['packed_emulated_agree'] and not r['prediction_packed_agree'] for r in decisions)},
        'joint_favorable': {route: sum(r[route + '_joint_favorable'] for r in pairs) for route in ROUTES},
        'packed_and_emulated_joint': sum(r['packed_joint_favorable'] and r['emulated_joint_favorable'] for r in pairs),
        'predicted_and_packed_joint': sum(r['prediction_joint_favorable'] and r['packed_joint_favorable'] for r in pairs),
        'predicted_but_not_packed_joint': sum(r['prediction_joint_favorable'] and not r['packed_joint_favorable'] for r in pairs),
        'magnitude_sensitivity': sensitivity, 'by_checkpoint_quantizer': groups,
        'independence': '18 selected interventions nested in 6 combinations from 2 historical checkpoints; no population rate or confidence interval',
        'frozen_original_decisions_changed': False,
    }
    return summary, pairs, decisions


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(source, out):
    summary, pairs, decisions = analyze(load(source))
    summary['source_sha256'] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    summary['source_name'] = Path(source).name
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    write_csv(out/'paired_interventions.csv', pairs)
    write_csv(out/'contrast_decisions.csv', decisions)
    write_csv(out/'magnitude_sensitivity.csv', summary['magnitude_sensitivity'])
    write_csv(out/'by_checkpoint_quantizer.csv', summary['by_checkpoint_quantizer'])
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))
