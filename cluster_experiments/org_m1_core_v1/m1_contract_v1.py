"""M1 smoke-only contract. Original B05 artifacts and prior gates are immutable."""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re

PROJECT = Path('/work/anon_user/hessian_unlearning')
PACKAGE = Path(__file__).resolve().parent
ENVIRONMENT = Path('/work/anon_user/.runtime/envs/org-packed-h200-v1')
MANIFEST_SHA = '90fa4ef9d8ac7b2056be3afd3758bb19c28c41f67006c3f8319c06b65c96c3ab'
VERIFICATION_SHA = 'ae390501102a4f3160f5f932ac41159e91f891232909eddb5d1653c84424bdfe'
ARMS = ('N0', 'Z_a', 'M_a', 'R_a', 'Z_b', 'M_b', 'R_b', 'Z_ab', 'M_ab', 'R_ab')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def bad(_):
        raise ValueError('nonfinite JSON')
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=bad)


def write(path, value):
    path = Path(path)
    side = Path(str(path) + '.sha256')
    require(not path.exists() and not side.exists(), 'output exists: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    with side.open('x', encoding='ascii', newline='\n') as f:
        f.write(sha(path) + '  ' + path.name + '\n')


def safe_path(root, relative):
    require(isinstance(relative, str) and relative and '\\' not in relative and ':' not in relative, 'unsafe path')
    p = PurePosixPath(relative)
    require(not p.is_absolute() and '..' not in p.parts and str(p) == relative, 'unsafe path')
    require(not any(ord(c) < 32 for c in relative), 'unsafe control character')
    root = Path(root).resolve()
    candidate = root
    for part in p.parts:
        candidate /= part
        require(not candidate.is_symlink(), 'symlink path')
    require(candidate.resolve().is_relative_to(root), 'path escape')
    return candidate


def verify_file(root, relative, expected):
    require(bool(re.fullmatch('[0-9a-f]{64}', expected)), 'invalid SHA')
    p = safe_path(root, relative)
    before = p.stat()
    require(p.is_file() and sha(p) == expected, 'source hash mismatch: ' + relative)
    after = p.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'file changed while hashing')
    return p


def package_check():
    rows = read(PACKAGE / 'SOURCE_LOCK.json')['files']
    for row in rows:
        verify_file(PACKAGE, row['path'], row['sha256'])
    return sha(PACKAGE / 'SOURCE_LOCK.json')


def interpreter_check(executable, prefix, lock):
    # A valid venv executable may be a symlink to its certified base interpreter.
    expected = lock['interpreter_identity']
    require(Path(prefix).resolve() == ENVIRONMENT.resolve(), 'wrong Python sys.prefix')
    require(Path(executable).absolute() == Path(expected['executable']).absolute(), 'wrong venv entrypoint')
    require(sha(executable) == expected['executable_sha256'], 'Python executable byte drift')


def arm_targets(cell, arm):
    require(arm in ARMS, 'unknown arm')
    require(cell['a'] != cell['b'] and all(cell[k].endswith('.weight') for k in ('a', 'b')), 'invalid pair')
    if arm == 'N0':
        return [], None
    family, subset = arm.split('_')
    return [cell[k].removesuffix('.weight') for k in subset], {'Z': 0.0, 'M': 0.5, 'R': 1.0}[family]


def validate_source_receipt(manifest, receipt):
    require(receipt.get('status') == 'PASS' and receipt.get('source_hashes_verified') is True, 'source verification required')
    require(receipt.get('root') == PROJECT.as_posix() and receipt.get('manifest_sha256') == MANIFEST_SHA, 'receipt identity')
    expected = {r['path']: r for r in manifest['files']}
    rows = receipt.get('files', [])
    require(len(rows) == len(expected) == 1038, 'receipt coverage')
    require(len({r['path'] for r in rows}) == len(rows), 'duplicate source records')
    require({r['path'] for r in rows} == set(expected), 'receipt path universe')
    for row in rows:
        e = expected[row['path']]
        require(row['status'] == 'PASS' and row['expected_sha256'] == e['sha256'], 'source file identity')
        require(type(row['bytes']) is int and row['bytes'] >= 0, 'source size')
        require(e['bytes'] is None or e['bytes'] == row['bytes'], 'source size drift')
    return expected


def preflight(output):
    require(Path.cwd().resolve() == PROJECT, 'run from exact cluster project root')
    source_lock = package_check()
    require(sha(PACKAGE / 'preservation_manifest_v2.json') == MANIFEST_SHA, 'manifest pin')
    require(sha(PACKAGE / 'verification.json') == VERIFICATION_SHA, 'receipt pin')
    pins = validate_source_receipt(read(PACKAGE / 'preservation_manifest_v2.json'), read(PACKAGE / 'verification.json'))
    policy = read(PACKAGE / 'RETENTION_DECISION.json')
    require(policy['independent_backup_complete'] is False and policy['independent_backup_deferred_by_user'] is True, 'retention decision')
    design = read(PACKAGE / 'M1_EXPERIMENT_DESIGN_V1.json')
    require(len(design['cells']) == 6 and design['task_counts']['core'] == 60, 'design coverage')
    # Check only M1's current inputs here, not all historical 1,038 paths again.
    needed = set(r['path'] for r in design['sources'])
    for cell in design['cells']:
        prefix = cell['source_checkpoint']['certified_relative_path'] + '/'
        needed.update(p for p in pins if p.startswith(prefix))
        for ref in cell['manifests'].values():
            needed.add(str(Path(ref['path']).relative_to(PROJECT)).replace('\\', '/'))
        folder = cell['baseline_folder']
        needed.update(p for p in pins if p.startswith(folder + '/') and not p.startswith(folder + '/emulated/'))
        arm_targets(cell, 'R_ab')
    for relative in sorted(needed):
        require(relative in pins, 'unfrozen M1 prerequisite: ' + relative)
        verify_file(PROJECT, relative, pins[relative]['sha256'])
    plan = read(PROJECT / 'results/org_h200_execution_v1/b05_selector_locks/behavior_plan_v1.json')
    for c in plan['cells']:
        for ref in (c['environment_lock'], c['feature_receipt']):
            verify_file(PROJECT, str(Path(ref['path']).relative_to(PROJECT)), ref['sha256'])
    result = {'status': 'PASS', 'scope': 'M1_SMOKE_ONLY', 'source_lock_sha256': source_lock,
              'design_sha256': sha(PACKAGE / 'M1_EXPERIMENT_DESIGN_V1.json'),
              'verified_input_count': len(needed), 'independent_backup_complete': False,
              'retention_decision_sha256': sha(PACKAGE / 'RETENTION_DECISION.json'),
              'core_launch_authorized': False, 'gpu_submissions': 0}
    write(output, result)
    print('M1_CPU_PREFLIGHT=PASS', flush=True)


def matching_unprotected(reference, current, protected):
    def filtered(rows):
        return {k: v for k, v in rows.items() if not any(k == p or k.startswith(p + '.') for p in protected)}
    require(filtered(reference) == filtered(current), 'unprotected operational state changed')


def planned_core_cost(smoke_gpu_hours, group_minutes):
    require(type(smoke_gpu_hours) in (int, float) and math.isfinite(smoke_gpu_hours) and smoke_gpu_hours >= 0, 'invalid spent hours')
    require(len(group_minutes) == 3 and all(type(v) is int and v > 0 for v in group_minutes), 'three time limits required')
    total = smoke_gpu_hours + 6 * sum(group_minutes) / 60
    return {'gpu_hours_ceiling': total, 'within_core_7_gpu_hours': total <= 7,
            'concurrency_target': 18, 'core_launch_authorized': False}
