"""Read-only inputs for a same-artifact diagnostic; stdlib only, no GPU imports."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import datetime


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root, env, output):
    root, env = root.resolve(), env.resolve()
    sources = sorted(env.glob('lib/python*/site-packages/gptqmodel/nn_modules/qlinear/*.py'))
    if not any(p.name == 'torch.py' for p in sources):
        raise RuntimeError('Installed GPTQ torch decoder source missing; no GPU launch')
    final = root / 'results/org_h200_execution_v1/b05_finish_wave_v1/final.json'
    if not final.is_file():
        raise RuntimeError('B05 final report missing')
    bases = [root / 'results/org_h200_execution_v1' / p for p in (
        'b05_finish_wave_v1', 'b05_behavior_parallel',
        'b05_remaining_backends_v1', 'b05_gptq_pair_repair')]
    records, members = [], [(p, 'installed_qlinear/' + p.name) for p in sources]
    members.append((final, 'b05_final.json'))
    for base in bases:
        if not base.exists():
            continue
        for manifest in sorted(base.rglob('artifact_hashes.json')):
            hashes = json.loads(manifest.read_text())
            artifact = manifest.parent / 'artifact'
            files = []
            for name, expected in hashes.items():
                file = (artifact / name).resolve()
                if not file.is_relative_to(artifact.resolve()):
                    raise RuntimeError('Unsafe artifact member: ' + name)
                files.append({'name': name, 'expected_sha256': expected,
                              'exists': file.is_file(),
                              'bytes': file.stat().st_size if file.is_file() else None})
            records.append({'condition_root': str(manifest.parent),
                            'manifest_sha256': sha(manifest), 'files': files})
            for name in ('artifact_hashes.json', 'source.json', 'state.json',
                         'artifact/config.json', 'artifact/quantize_config.json'):
                p = manifest.parent / name
                if p.is_file():
                    members.append((p, 'cluster/' + p.relative_to(root).as_posix()))
    if not records:
        raise RuntimeError('No saved artifact manifests found; no GPU launch')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    report = output / 'inventory.json'
    report.write_text(json.dumps({'status': 'INPUT_COLLECTION_COMPLETE',
        'gpu_hours': 0, 'production_authorized': False,
        'artifact_bytes_verified': False,
        'note': 'Existence and sizes only; full artifact hashes must be verified before diagnostic execution.',
        'artifacts': records,
        'sources': {str(p): sha(p) for p in sources}}, indent=2))
    members.append((report, 'inventory.json'))
    archive = output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'x:gz') as tar:
        for p, name in members:
            tar.add(p, arcname=name, recursive=False)
    sidecar = Path(str(archive) + '.sha256')
    with sidecar.open('x') as f:
        f.write(sha(archive) + '  ' + archive.name + '\n')
    print('B05_SAME_ARTIFACT_INPUTS=COMPLETE\nGPU_HOURS=0')
    print('RETURN=' + str(archive))
    print('SIDECAR=' + str(sidecar))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--env', type=Path, required=True)
    a = p.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    collect(a.root, a.env, a.root / ('results/org_h200_execution_v1/same_artifact_inputs_' + stamp))
