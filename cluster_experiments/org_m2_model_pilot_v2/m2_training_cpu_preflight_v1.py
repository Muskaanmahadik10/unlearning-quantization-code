"""Offline CPU-only preparation; no LLM loader, training CLI or GPU submission."""
import argparse
import ast
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import tarfile
import traceback
import unittest
import uuid

ROOT = Path('/work/anon_user/hessian_unlearning')
ENV = Path('/work/anon_user/.runtime/envs/org-packed-h200-v1')
PKG = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def pairs(rows):
    out = {}
    for k, v in rows:
        require(k not in out, 'duplicate JSON key: ' + k)
        out[k] = v
    return out


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def save(path, obj):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def confined(base, rel):
    require(isinstance(rel, str) and rel and '\\' not in rel, 'invalid relative path')
    p = Path(rel)
    require(not p.is_absolute() and '..' not in p.parts, 'path traversal')
    target = base / p
    require(target.resolve().is_relative_to(base.resolve()) and target.is_file(), 'missing/escaped input: ' + rel)
    return target


def verify(base, rows):
    require(rows and len({r['path'] for r in rows}) == len(rows), 'empty/duplicate manifest')
    for r in rows:
        p = confined(base, r['path'])
        if 'bytes' in r:
            require(p.stat().st_size == r['bytes'], 'input size drift: ' + r['path'])
        require(sha(p) == r['sha256'], 'input hash drift: ' + r['path'])


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_collator(data_utils):
    # Keep the pinned class unchanged; avoid importing the project's plugin registry.
    tree = ast.parse((PKG / 'upstream_collators.py').read_text(encoding='utf-8'))
    removed = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == 'data.utils']
    require(len(removed) == 1 and [(a.name, a.asname) for a in removed[0].names] == [('IGNORE_INDEX', None)],
            'upstream collator import changed')
    tree.body = [n for n in tree.body if n is not removed[0]]
    namespace = {'IGNORE_INDEX': data_utils.IGNORE_INDEX}
    exec(compile(tree, str(PKG / 'upstream_collators.py'), 'exec'), namespace)
    return namespace['DataCollatorForSupervisedDataset']


def jsonl(path):
    return [json.loads(line, object_pairs_hook=pairs) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def qa_identity(row):
    return hashlib.sha256(json.dumps([row['question'], row['answer']], ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def variants(row):
    require(isinstance(row.get('question'), str) and isinstance(row.get('answer'), str), 'invalid QA row')
    yield 'answer', row['question'], row['answer']
    if 'paraphrased_answer' in row:
        require(isinstance(row['paraphrased_answer'], str), 'invalid paraphrase')
        yield 'paraphrased_answer', row['question'], row['paraphrased_answer']
    if 'perturbed_answer' in row:
        require(isinstance(row['perturbed_answer'], list) and row['perturbed_answer'], 'invalid perturbations')
        for i, a in enumerate(row['perturbed_answer']):
            require(isinstance(a, str), 'invalid perturbation string')
            yield 'perturbed_' + str(i), row['question'], a
    if 'paraphrased_question' in row:
        require(isinstance(row['paraphrased_question'], str), 'invalid paraphrased question')
        yield 'paraphrased_question', row['paraphrased_question'], row['answer']


def archive_run(run, stamp):
    files = sorted(p for p in run.iterdir() if p.is_file())
    evidence = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
    save(run / 'EVIDENCE_MANIFEST.json', {'files': evidence})
    export = ROOT / 'audit_exports' / ('m2_training_cpu_' + stamp + '.tar.gz')
    export.parent.mkdir(exist_ok=True)
    with tarfile.open(export, 'x:gz') as t:
        for p in files + [run / 'EVIDENCE_MANIFEST.json']:
            t.add(p, arcname=p.name, recursive=False)
    with tarfile.open(export) as t:
        for row in evidence:
            require(hashlib.sha256(t.extractfile(row['path']).read()).hexdigest() == row['sha256'], 'archive readback failed')
    with Path(str(export) + '.sha256').open('x', encoding='ascii', newline='\n') as f:
        f.write(sha(export) + '  ' + export.name + '\n')
    print('RETURN=' + str(export) + '\nSIDECAR=' + str(export) + '.sha256', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-cpu', action='store_true', required=True)
    ap.parse_args()
    require(platform.system() == 'Linux' and Path(sys.prefix) == ENV, 'use the certified Nano4 interpreter with -I -B')
    require(sys.flags.isolated and sys.dont_write_bytecode, 'invoke with -I -B')
    require(ROOT.is_dir(), 'cluster root absent')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    run = ROOT / 'results/org_m2_training_cpu_preflight_v1' / stamp
    run.mkdir(parents=True, exist_ok=False)
    try:
        source = read(PKG / 'SELF_LOCK.json')['files']
        verify(PKG, source)
        binding = read(PKG / 'RUNTIME_DATA_BINDING.json')
        base = read(PKG / 'BASE_INPUT_IDENTITY_LOCK.json')
        design = read(PKG / 'CONTROLLED_OUR_DESIGN_V1.json')
        env_lock_path = ROOT / binding['environment_lock_path']
        require(sha(env_lock_path) == binding['environment_lock_sha256'], 'environment lock changed')
        env_lock = read(env_lock_path)
        require(sha(sys.executable) == env_lock['interpreter_identity']['executable_sha256'], 'interpreter bytes changed')
        setup = Path(binding['setup_run'])
        require(sha(setup / 'overlay_file_manifest.json') == binding['overlay_manifest_sha256'], 'overlay manifest changed')
        verify(setup, read(setup / 'overlay_file_manifest.json')['files'])
        overlay = setup / 'site-packages'
        require(overlay.is_dir(), 'existing overlay missing; no automatic install')
        verify(Path(base['snapshot_path']), base['files'])
        data_pkg = ROOT / 'org_m2_evaluator_setup_v1'
        require(sha(data_pkg / 'DATA_PROVENANCE.json') == binding['data_provenance_sha256'], 'data provenance changed')
        verify(data_pkg, binding['data_files'])
        os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                          OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                          HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        sys.path[:0] = [str(PKG), str(overlay)]
        versions = {n: importlib.metadata.version(n) for n in binding['effective_versions']}
        require(versions == binding['effective_versions'], 'effective package versions changed')
        import torch
        from transformers import AutoTokenizer
        import m2_training_core_v1 as c
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        save(run / 'runtime.json', {'versions': versions, 'prefix': sys.prefix, 'cuda_visible_devices': '',
             'model_weight_bytes_rehashed': True, 'model_weight_tensors_loaded': False,
             'overlay_files_rehashed': True, 'source_lock_sha256': sha(PKG / 'SELF_LOCK.json')})
        suite = unittest.TestSuite()
        for module in ('test_m2_training_contract_v1', 'test_m2_training_torch_v1', 'test_m2_training_preflight_v1'):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
        with (run / 'CPU_TESTS.txt').open('x', encoding='utf-8') as f:
            result = unittest.TextTestRunner(stream=f, verbosity=2).run(suite)
        require(result.wasSuccessful() and not result.skipped and result.testsRun == binding['expected_cluster_tests'],
                'CPU tests failed or coverage changed; see ' + str(run / 'CPU_TESTS.txt'))
        print('M2_TRAINING_COMPONENT_TORCH_CPU_TESTS=PASS', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(base['snapshot_path'], local_files_only=True, trust_remote_code=False)
        original_pad = tokenizer.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        oracle = load_module('m2_pinned_data_oracle', PKG / 'upstream_data_utils.py')
        batch_oracle = official_collator(oracle)(tokenizer=tokenizer, padding_side='right')
        template = {'apply_chat_template': True, 'system_prompt': design['prompt']['system'],
                    'date_string': design['prompt']['date']}
        summary, training_tokens, violations = [], {}, []
        data = {Path(r['path']).stem: jsonl(data_pkg / r['path']) for r in binding['data_files']}
        ids = {name: [qa_identity(r) for r in data[name]] for name in ('full','forget10','retain90','holdout10')}
        require([len(ids[n]) for n in ('full','forget10','retain90','holdout10')] == [4000,400,3600,400], 'QA count drift')
        require(all(len(v) == len(set(v)) for v in ids.values()), 'duplicate QA')
        require(set(ids['full']) == set(ids['forget10']) | set(ids['retain90']) and
                not set(ids['forget10']) & set(ids['retain90']), 'full/forget/retain partition drift')
        require(not set(ids['holdout10']) & set(ids['full']), 'exact-QA nonmember overlap')
        for name, rows in data.items():
            count, max_length, max_prompt, eos_masked = 0, 0, 0, 0
            token_hash = hashlib.sha256()
            batch_ours, batch_official = [], []
            for i, row in enumerate(rows):
                for kind, question, answer in variants(row):
                    # No truncation: enumerate full lengths, then reject oversize training.
                    ours = c.encode_qa(tokenizer, question, answer, cap=1_000_000)
                    expected = oracle.preprocess_chat_instance(tokenizer, template, [question], [answer], max_length=512)
                    for k in ('input_ids','labels','attention_mask'):
                        require(ours[k] == expected[k].tolist(), f'upstream tokenizer mismatch {name}:{i}:{kind}:{k}')
                    generation = oracle.preprocess_chat_instance(tokenizer, template, [question], [answer],
                                                                max_length=512, predict_with_generate=True)
                    require(ours['prompt_ids'] == generation['input_ids'].tolist(), 'generation prompt mismatch')
                    if name in ('full','forget10','retain90') and len(ours['input_ids']) > 512:
                        violations.append({'file': name, 'row': i, 'kind': kind, 'tokens': len(ours['input_ids'])})
                    if name == 'full' and kind == 'answer':
                        training_tokens[ids['full'][i]] = ours
                    token_hash.update(json.dumps([i,kind,ours], sort_keys=True, separators=(',', ':')).encode())
                    max_length = max(max_length, len(ours['input_ids']))
                    max_prompt = max(max_prompt, len(ours['prompt_ids']))
                    eos_masked += sum(x == tokenizer.pad_token_id for x in ours['input_ids'])
                    batch_ours.append(ours); batch_official.append(expected); count += 1
                    if len(batch_ours) == 8:
                        actual, reference = c.collate(torch,batch_ours,tokenizer.pad_token_id), batch_oracle(batch_official)
                        require(all(torch.equal(actual[k],reference[k]) for k in actual), 'upstream collator mismatch')
                        batch_ours.clear(); batch_official.clear()
            if batch_ours:
                actual, reference = c.collate(torch,batch_ours,tokenizer.pad_token_id), batch_oracle(batch_official)
                require(all(torch.equal(actual[k],reference[k]) for k in actual), 'partial collator mismatch')
            summary.append({'file':name,'records':len(rows),'answer_variants':count,'max_tokens':max_length,
                            'max_prompt_tokens':max_prompt,'token_stream_sha256':token_hash.hexdigest(),
                            'unpadded_positions_masked_by_official_pad_rule':eos_masked})
            print(f'M2_TOKENIZER_FILE={name}:PASS variants={count}', flush=True)
        save(run / 'tokenizer_audit.json', {'files': summary, 'original_pad_token_id': original_pad,
             'effective_pad_token_id': tokenizer.pad_token_id, 'eos_token_id': tokenizer.eos_token_id,
             'oversize_training': violations, 'upstream_input_label_generation_and_collator_parity': True,
             'evaluation_variants_scanned_not_scored': True, 'author_nonmembership_certified': False})
        require(not violations, 'training exceeds 512 tokens; no rows were truncated or dropped')
        save(run / 'training_tokens.json', {'by_exact_qa_sha256': training_tokens,
             'role_order': {'O':ids['full'],'R':ids['retain90'],'U':ids['forget10']}})
        schedules = {role:c.schedule(role,design['training'][role]) for role in ('O','R','U')}
        save(run / 'training_schedule.json', schedules)
        verify(PKG, source)
        verify(data_pkg, binding['data_files'])
        verify(Path(base['snapshot_path']), base['files'])
        require(not torch.cuda.is_initialized(), 'unexpected CUDA initialization')
        save(run / 'final.json', {'status':'CPU_COMPONENTS_AND_TOKENIZER_PASS',
             'tests':result.testsRun,'source_lock_sha256':sha(PKG/'SELF_LOCK.json'),
             'base_revision':base['revision'],'base_lock_sha256':sha(PKG/'BASE_INPUT_IDENTITY_LOCK.json'),
             'training_tokens_sha256':sha(run/'training_tokens.json'),
             'training_schedule_sha256':sha(run/'training_schedule.json'),
             'author_mapping_certified':False,'grouped_split_frozen':False,'full_model_launcher_certified':False,
             'cuda_optimizer_and_resume_certified':False,'training_authorized':False,
             'gpu_submissions':0,'gpu_hours':0,'production_authorized':False})
        print('M2_TRAINING_CPU_PREFLIGHT=PASS', flush=True)
    except Exception as e:
        save(run/'FAILURE.json', {'status':'FAIL','exception_type':type(e).__name__,'error':str(e),
             'traceback':traceback.format_exc(),'training_authorized':False,'production_authorized':False})
        print('M2_TRAINING_CPU_PREFLIGHT=FAIL\nFAILURE='+str(run/'FAILURE.json'), flush=True)
    archive_run(run,stamp)
    print('GPU_SUBMISSIONS=0\nTRAINING_AUTHORIZED=false', flush=True)
    if (run/'FAILURE.json').exists():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
