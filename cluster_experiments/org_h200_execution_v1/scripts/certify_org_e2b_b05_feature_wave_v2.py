"""Validate current-runtime feature receipts and freeze their selectors, CPU only."""
import argparse
import math
from pathlib import Path
from certify_org_e2b_b05_feature_wave_v1 import EXPECTED, read, atomic, sha256_file
from run_org_e2b_b05_feature_cell_v1 import canonical_sha256, select_sets


def require(ok, message):
    if not ok:
        raise ValueError(message)


def linked(record):
    path = Path(record['path'])
    require(sha256_file(path) == record['sha256'], f'linked hash mismatch: {path}')
    return path


def validate(path, manifest):
    item = read(path)
    require(item.get('schema_version') == 'org_e2b_b05_feature_cell_v2' and item.get('status') == 'PASS', 'receipt status/schema')
    require(item.get('mode') == 'production' and item.get('production_eligible') is True, 'feature mode')
    require(item.get('outcome_bearing') is False and item.get('protected_outcomes_inspected') is False, 'outcome flags')
    require(item.get('b05_behavior_authorized') is False and item.get('atlas_production_authorized') is False, 'authorization flags')
    require(item.get('sample_counts') == dict(forget=128, retain=128, calibration=128), 'sample counts')
    key = (item['cell_id'], item['quantizer'])
    require(key in EXPECTED, 'cell identity')
    matches = [r for r in manifest['references'] if (r['cell_id'], r['quantizer']) == key]
    require(len(matches) == 1, 'reference manifest identity')
    reference_path = linked(item['current_runtime_reference'])
    require(sha256_file(reference_path) == matches[0]['sha256'], 'reference manifest hash')
    ref = read(reference_path)
    require(ref.get('pre_outcome') is True and ref.get('protected_outcomes_inspected') is False, 'reference flags')
    require((ref['cell_id'], ref['quantizer']) == key, 'reference identity')
    require(item['answer_tokens'] == ref['answer_tokens'], 'answer tokens')
    for record in item['prerequisites'].values():
        linked(record)
    for label in ('frozen_v1_computation', 'v2_wrapper'):
        linked(item[label])
    rows = item['per_layer']
    require(item.get('target_count') == 224 and len(rows) == 224, 'row count')
    names = [r['layer'] for r in rows]
    rr = {r['layer']: r for r in ref['per_layer']}
    require(len(set(names)) == 224 and len(ref['per_layer']) == 224 and set(names) == set(rr), 'matrix universe')
    require(canonical_sha256(rows) == item['per_layer_sha256'], 'row digest')
    for row in rows:
        old = rr[row['layer']]
        require(row['displacement_sha256'] == old['displacement_sha256'], 'displacement drift')
        for field, reference_field in [('forget_qdp_signed', 'gqdp_signed'), ('retain_qdp_signed', 'retain_qdp_signed')]:
            a, b = float(row[field]), float(old[reference_field])
            require(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-10, f'{field} drift')
        delta = float(row['retain_independent_abs_delta'])
        require(math.isfinite(delta) and 0 <= delta <= 1e-10, 'independent retain QDP')
        require(row['protection_forget_benefit'] == -row['forget_qdp_signed'], 'benefit algebra')
        require(row['protection_retain_cost'] == -row['retain_qdp_signed'], 'cost algebra')
        require(isinstance(row['n_parameters'], int) and row['n_parameters'] > 0, 'parameter count')
    total = sum(r['n_parameters'] for r in rows)
    require(total == item['target_parameters'], 'parameter sum')
    require(select_sets(rows, total) == item['conditions'], 'selector recomputation')
    return key, {'receipt': {'path': str(path.resolve()), 'sha256': sha256_file(path)},
                 'conditions': item['conditions'], 'per_layer_sha256': item['per_layer_sha256'],
                 'target_parameters': total}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-manifest', type=Path, required=True)
    p.add_argument('--receipts', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = read(a.reference_manifest)
    require(manifest.get('schema_version') == 'org_e2b_b05_current_runtime_reference_manifest_v1' and manifest.get('status') == 'PASS', 'manifest status')
    require(manifest.get('pre_outcome') is True and manifest.get('protected_outcomes_inspected') is False, 'manifest flags')
    require(len(a.receipts) == 6 and len(manifest['references']) == 6, 'six cells required')
    cells = {}
    for path in a.receipts:
        key, result = validate(path, manifest)
        require(key not in cells, 'duplicate cell')
        cells[key] = result
    require(set(cells) == EXPECTED, 'cell universe')
    atomic(a.output, {'schema_version': 'org_e2b_b05_selector_lock_v2', 'status': 'PASS',
        'feature_cell_count': 6, 'condition_count_per_cell': 6,
        'cells': {f'{k[0]}/{k[1]}': cells[k] for k in sorted(cells)},
        'reference_manifest': {'path': str(a.reference_manifest.resolve()), 'sha256': sha256_file(a.reference_manifest)},
        'validator_sha256': sha256_file(Path(__file__)),
        'selector_definitions_frozen': True, 'protected_outcomes_inspected': False,
        'feature_prerequisite_pass': True, 'behavior_launcher_certified': False,
        'b05_behavior_authorized': False, 'atlas_production_authorized': False})
    print('B05_V2_SELECTOR_LOCK=PASS\nFEATURE_CELLS=6\nBEHAVIOR_LAUNCHER_CERTIFIED=false')
    print(f'OUTPUT={a.output}')


if __name__ == '__main__':
    main()
