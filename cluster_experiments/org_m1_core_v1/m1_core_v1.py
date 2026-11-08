"""M1 core: ten fully evaluated arms per cell; checked native lifecycle retained."""
import argparse
import gc
import importlib.metadata
import importlib
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from m1_contract_v1 import (PROJECT, PACKAGE, ENVIRONMENT, ARMS, require, read, sha,
                           write, safe_path, verify_file, package_check, arm_targets,
                           matching_unprotected, interpreter_check)

sys.path.insert(0, str(PACKAGE / 'support'))


def digest(torch, tensor):
    return sha_tensor_bytes(tensor.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())


def sha_tensor_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def state(torch, model):
    result = {}
    # Includes quantizer's serialized state and nonpersistent runtime buffers.
    values = dict(model.state_dict())
    for name, tensor in model.named_buffers():
        values.setdefault(name, tensor)
    for name, tensor in values.items():
        if torch.is_tensor(tensor):
            result[name] = {'sha256': digest(torch, tensor), 'shape': list(tensor.shape),
                            'dtype': str(tensor.dtype), 'device': str(tensor.device)}
    return result


def module_contract(model):
    return {n: {'class': type(m).__module__ + '.' + type(m).__qualname__, 'training': m.training,
                'compute_dtype': str(getattr(m, 'compute_dtype', None))}
            for n, m in model.named_modules()}


def decode(torch, module, quantizer):
    require(module.bias is None, 'M1 target bias not supported by frozen protocol')
    if quantizer == 'rtn_int8':
        require(module.qweight.dtype == torch.int8 and module.scale.dtype == torch.float32, 'RTN storage')
        require(tuple(module.scale.shape) == (module.out_features, 1), 'RTN scale axis')
        # Exactly the accepted native RTN decoder, including scale-to-BF16 cast.
        weight = module.qweight.to(torch.bfloat16) * module.scale.to(torch.bfloat16)
        reference = module.qweight.to(torch.bfloat16).mul(module.scale.to(torch.bfloat16))
        require(torch.equal(weight, reference), 'RTN decode repeat')
    elif quantizer == 'nf4':
        import bitsandbytes as bnb
        require(isinstance(module, bnb.nn.Linear4bit), 'NF4 class')
        qs = module.weight.quant_state
        require(qs.quant_type == 'nf4' and not qs.nested and qs.dtype == torch.bfloat16, 'unsupported NF4 quant state')
        weight = bnb.functional.dequantize_4bit(module.weight.data, quant_state=qs)
        # Independently check fixed positions against the saved codebook and block scales.
        flat = module.weight.data.reshape(-1)
        for pos in (0, 1, 2, 63, 64, module.in_features - 1, weight.numel() // 2, weight.numel() - 1):
            byte = int(flat[pos // 2].item())
            code = (byte >> 4) if pos % 2 == 0 else byte & 15
            scalar = (qs.code[code].float() * qs.absmax[pos // qs.blocksize].float()).to(qs.dtype)
            require(torch.equal(weight.reshape(-1)[pos], scalar), 'NF4 scalar decode oracle')
    else:
        from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
        require(type(module) is TorchQuantLinear and not module.adapter, 'unsupported GPTQ backend/adapter')
        require(module.bits == 4 and module.qweight.dtype == torch.int32 and len(module.g_idx) == module.in_features, 'GPTQ layout')
        weight = module.dequantize_weight(num_itr=1).T.contiguous()
        # A native-matmul parity probe below checks orientation and current decoder path.
        require(torch.equal(weight, module.dequantize_weight(num_itr=1).T.contiguous()), 'GPTQ decode repeat')
    require(tuple(weight.shape) == (module.out_features, module.in_features), 'decoder orientation/shape')
    require(bool(torch.isfinite(weight).all()), 'nonfinite decoded weight')
    return weight.detach().float()


def original_weights(cell, names):
    from safetensors import safe_open
    folder = Path(cell['source_checkpoint']['path'])
    index_path = folder / 'model.safetensors.index.json'
    index = read(index_path)['weight_map'] if index_path.exists() else {n + '.weight': 'model.safetensors' for n in names}
    result = {}
    for name in names:
        key = name + '.weight'
        require(key in index, 'missing original weight index')
        with safe_open(str(safe_path(folder, index[key])), framework='pt', device='cpu') as f:
            result[name] = f.get_tensor(key)
    return result


from m1_endpoint_v3 import patch_weight


def synthetic(torch, model):
    ids = torch.tensor([[1, 10, 20, 30, 40, 50, 60, 2] * 4], device='cuda:0')
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits.float()
    require(bool(torch.isfinite(logits).all()), 'nonfinite synthetic logits')
    return logits.cpu()


def environment_check(torch, tf, cell):
    require(torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == 'NVIDIA H200', 'one isolated H200 required')
    plan = read(PROJECT / 'results/org_h200_execution_v1/b05_selector_locks/behavior_plan_v1.json')
    old = [r for r in plan['cells'] if r['cell_id'] + '/' + r['quantizer'] == cell['cell']]
    require(len(old) == 1, 'B05 cell identity')
    ref = old[0]['environment_lock']
    require(sha(ref['path']) == ref['sha256'], 'environment lock drift')
    lock = read(ref['path'])
    require(lock['overall_status'] == 'PASS', 'environment not certified')
    interpreter_check(sys.executable, sys.prefix, lock)
    for package, record in lock['packages'].items():
        require(importlib.metadata.version(package) == record['version'], 'package version drift: ' + package)
        require(sha(record['module_path']) == record['module_sha256'], 'package source drift: ' + package)
        loaded = importlib.import_module(package)
        require(Path(loaded.__file__).resolve() == Path(record['module_path']).resolve(), 'shadowed package: ' + package)
    for record in (lock['gptqmodel_kernel'],):
        require(sha(record['module_path']) == record['module_sha256'], 'GPTQ decoder source drift')
    kernel_base = Path(lock['gptqmodel_kernel']['module_path']).with_name('__init__.py')
    require(sha(kernel_base) == 'c6e509b29f92becabfa09e36feddf5c98c7ca0f84cc72cb5e3b90550c74ec347', 'GPTQ base decoder source drift')
    require(sha(lock['bitsandbytes_native']['library_path']) == lock['bitsandbytes_native']['library_sha256'], 'NF4 native library drift')
    props = torch.cuda.get_device_properties(0)
    device_uuid = str(getattr(props, 'uuid', ''))
    require(device_uuid and device_uuid != 'None', 'GPU UUID unavailable')
    return old[0], {'gpu_uuid': device_uuid, 'device': props.name,
                    'pid': os.getpid(), 'process_uuid': str(uuid.uuid4()),
                    'job_id': os.environ.get('SLURM_JOB_ID'), 'hostname': os.uname().nodename,
                    'packages': {k: v['version'] for k, v in lock['packages'].items()}}


def phase(args):
    started = time.monotonic()
    package_sha = package_check()
    from m1_core_gate_v1 import reconciliation_gate
    reconciliation_gate()
    cpu_path = Path(os.environ['M1_ENDPOINT_CPU_RECEIPT'])
    require(sha(cpu_path) == os.environ['M1_ENDPOINT_CPU_SHA256'], 'CPU regression receipt drift')
    cpu = read(cpu_path)
    require(cpu['status'] == 'PASS' and cpu['scope'] == 'EXACT_ENDPOINT_TORCH_CPU_ONLY'
            and cpu['source_lock_sha256'] == package_sha, 'CPU regression binding failed')
    ready = read(args.preflight)
    require(ready['status'] == 'PASS' and ready['scope'] == 'M1_SMOKE_ONLY' and ready['source_lock_sha256'] == package_sha, 'preflight/source mismatch')
    design = read(PACKAGE / 'M1_EXPERIMENT_DESIGN_V1.json')
    cell = design['cells'][args.index]
    require(ready['design_sha256'] == sha(PACKAGE / 'M1_EXPERIMENT_DESIGN_V1.json'), 'design drift')
    quantizer = cell['cell'].split('/')[1]
    folder = PROJECT / cell['baseline_folder']
    verify_file(PROJECT, cell['baseline_artifact_hashes_source'], cell['baseline_artifact_hashes_source_sha256'])
    import torch
    import transformers as tf
    import run_org_e2b_b04_packed_public_canary_v1 as packed
    import run_org_e2b_b04_representation_canary_v1 as nf4
    import b05_same_artifact_v1 as accepted
    from diagnose_org_b05_rtn_reload_v1 import corrected_load
    from org_b05_behavior_evaluator_v1 import evaluate, compare_reloads
    from run_gate2_metric_suite_v42 import manifest_samples
    from safetensors.torch import save_file, load_file
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    old, runtime = environment_check(torch, tf, cell)
    feature = read(old['feature_receipt']['path'])
    require(sha(old['feature_receipt']['path']) == old['feature_receipt']['sha256'], 'feature pin')
    targets = [r['layer'].removesuffix('.weight') for r in feature['per_layer']]
    require(len(set(targets)) == len(targets) == 224, 'target universe')
    selected, _ = arm_targets(cell, 'R_ab')
    require(set(selected) <= set(targets), 'pair outside universe')
    guard = nf4.NetworkGuard(); guard.install()
    try:
        hashes = accepted.verify(folder)
        pins = read(PACKAGE / 'preservation_manifest_v2.json')['files']
        source_prefix = cell['source_checkpoint']['certified_relative_path'] + '/'
        source_pins = [r for r in pins if r['path'].startswith(source_prefix)]
        require(bool(source_pins), 'missing pinned U checkpoint')
        for r in source_pins:
            verify_file(PROJECT, r['path'], r['sha256'])
        for ref in cell['manifests'].values():
            require(sha(ref['path']) == ref['sha256'], 'data manifest drift')
        artifact = folder / 'artifact'
        load_start = time.monotonic()
        if quantizer == 'gptq_int4':
            wrapper, model = packed.load_gptq(torch, artifact)
        elif quantizer == 'nf4':
            model = tf.AutoModelForCausalLM.from_pretrained(artifact, local_files_only=True, trust_remote_code=False,
                       torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='eager')
        else:
            model = corrected_load(torch, tf, packed, artifact, targets)
        model.eval(); model.config._attn_implementation = 'eager'
        load_seconds = time.monotonic() - load_start
        require(all(t.device.type == 'cuda' for t in list(model.parameters()) + list(model.buffers())), 'offload forbidden')
        baseline = synthetic(torch, model)
        require(float((baseline - synthetic(torch, model)).abs().max()) <= 1e-6, 'native sentinel repeat')
        frozen = state(torch, model)
        frozen_modules = module_contract(model)
        frozen_config = model.config.to_dict()
        require({k: v['sha256'] for k, v in frozen.items() if k in model.state_dict()} == read(folder / 'state.json'), 'B05 baseline persistent state drift')
        originals = original_weights(cell, selected)
        native = {n: model.get_submodule(n) for n in selected}
        decoded, reports = {}, {}
        for n, m in native.items():
            before = state(torch, m)
            decoded[n] = decode(torch, m, quantizer)
            require(state(torch, m) == before, 'decoder mutated module state')
            x = torch.linspace(-1, 1, m.in_features * 3, device='cuda:0', dtype=torch.float32).reshape(3, m.in_features).to(torch.bfloat16)
            with torch.inference_mode():
                observed = m(x)
                if quantizer == 'gptq_int4':
                    reference = torch.matmul(x, decoded[n].to(x.dtype).T.contiguous())
                else:
                    reference = torch.nn.functional.linear(x, decoded[n].to(x.dtype))
            require(bool(torch.isfinite(observed).all() and torch.isfinite(reference).all()), 'decode forward nonfinite')
            delta = float((observed.float() - reference.float()).abs().max())
            # NF4 native-vs-dense equality is intentionally not required: this is the sham contrast.
            if quantizer != 'nf4':
                require(delta == 0.0, 'native decoder/forward orientation mismatch')
            reports[n] = {'decoded_sha256': digest(torch, decoded[n]), 'original_sha256': digest(torch, originals[n]),
                          'decode_dtype': str(decoded[n].dtype), 'shape': list(decoded[n].shape),
                          'module_probe_native_dense_max_delta': delta,
                          'decode_to_bf16_max_delta': float((decoded[n] - decoded[n].to(torch.bfloat16).float()).abs().max())}
        # Real Torch endpoint/shape/nonfinite failures, executed on H200 before arm evaluation.
        toy_q = torch.tensor([[0.0, 0.25], [-0.5, 1.0]], device='cuda:0')
        toy_u = torch.tensor([[0.5, -0.25], [0.0, 0.5]], device='cuda:0')
        for dose in (0.0, 0.5, 1.0):
            require(torch.equal(patch_weight(torch, toy_q, toy_u, dose),
                    ((1 - dose) * toy_q + dose * toy_u).to(torch.bfloat16)), 'tiny interpolation oracle')
        for bad_q, bad_u, dose in ((toy_q, toy_u[:1], 0.5), (toy_q, toy_u, 0.25), (toy_q * float('nan'), toy_u, 1.0)):
            try:
                patch_weight(torch, bad_q, bad_u, dose)
            except ValueError:
                pass
            else:
                raise ValueError('tiny negative control unexpectedly passed')
        tokenizer = tf.AutoTokenizer.from_pretrained(old['checkpoint']['path'], local_files_only=True, trust_remote_code=False)
        samples = {r: manifest_samples(cell['manifests'][r + '_manifest']['path'], 128, forget_contract=r == 'forget')[1] for r in ('forget', 'retain')}
        results = {}
        previous = read(args.output / 'produce.json') if args.phase == 'reload' else None
        for arm in ARMS:
            arm_started = time.monotonic()
            names, dose = arm_targets(cell, arm)
            patch_path = args.output / (arm + '.safetensors')
            patch_hash = None
            try:
                if names:
                    if args.phase == 'produce':
                        patches = {n: patch_weight(torch, decoded[n], originals[n].to('cuda:0'), dose).cpu().contiguous() for n in names}
                        require(not patch_path.exists(), 'patch already exists')
                        save_file(patches, str(patch_path))
                    else:
                        require(sha(patch_path) == previous['arms'][arm]['patch_sha256'], 'patch file drift')
                    patches = load_file(str(patch_path), device='cuda:0')
                    require(set(patches) == set(names), 'patch tensor universe')
                    for n in names:
                        expected = patch_weight(torch, decoded[n], originals[n].to('cuda:0'), dose)
                        require(torch.equal(patches[n], expected), 'serialized patch mismatch')
                        replacement = torch.nn.Linear(native[n].in_features, native[n].out_features, bias=False, device='cuda:0', dtype=torch.bfloat16)
                        replacement.weight = torch.nn.Parameter(patches[n], requires_grad=False)
                        replacement.eval(); packed.set_module(model, n, replacement)
                    patch_hash = sha(patch_path)
                matching_unprotected(frozen, state(torch, model), names)
                matching_unprotected(frozen_modules, module_contract(model), names)
                require(model.config.to_dict() == frozen_config, 'model configuration changed')
                logits = synthetic(torch, model)
                logits_path = args.output / (arm + '_logits.pt')
                if args.phase == 'produce':
                    require(not logits_path.exists(), 'synthetic file exists')
                    torch.save(logits, logits_path)
                else:
                    require(sha(logits_path) == previous['arms'][arm]['logits_sha256'], 'synthetic reference drift')
                    prior = torch.load(logits_path, weights_only=True, map_location='cpu')
                    require(float((prior - logits).abs().max()) <= 1e-6, 'fresh-process synthetic drift: ' + arm)
                # Production: full forget and retain manifests for EVERY arm in both processes.
                behavior = {r: evaluate(model, tokenizer, s) for r, s in samples.items()}
                if arm == 'N0':
                    historical = read(folder / 'reload_b_behavior.json')
                    for role in samples:
                        compare_reloads(historical[role]['rows'], behavior[role]['rows'])
                if args.phase == 'reload' and behavior is not None:
                    for role in samples:
                        compare_reloads(previous['arms'][arm]['behavior'][role]['rows'], behavior[role]['rows'])
                matching_unprotected(frozen, state(torch, model), names)
                results[arm] = {'status': 'PASS', 'targets': names, 'dose': dose, 'patch_sha256': patch_hash,
                                'logits_sha256': sha(logits_path), 'behavior': behavior, 'seconds': time.monotonic() - arm_started}
            finally:
                for n in names:
                    packed.set_module(model, n, native[n])
            require(state(torch, model) == frozen, 'baseline state not restored after ' + arm)
            require(module_contract(model) == frozen_modules and model.config.to_dict() == frozen_config, 'baseline class/config not restored')
            require(float((synthetic(torch, model) - baseline).abs().max()) <= 1e-6, 'baseline sentinel not restored')
            print('M1_SMOKE_ARM=' + args.phase + ':' + arm + ':PASS', flush=True)
        require(accepted.verify(folder) == hashes and guard.attempts == 0, 'artifact/network mutation')
        for r in source_pins:
            verify_file(PROJECT, r['path'], r['sha256'])
        for ref in cell['manifests'].values():
            require(sha(ref['path']) == ref['sha256'], 'post-smoke data manifest drift')
        if previous is not None:
            require(previous['decode_reports'] == reports, 'fresh decoder/original identity drift')
        write(args.output / (args.phase + '.json'), {'status': 'PASS', 'phase': args.phase, 'cell': cell['cell'],
              'source_lock_sha256': package_sha, 'runtime': runtime, 'decode_reports': reports,
              'arms': results, 'load_seconds': load_seconds, 'seconds': time.monotonic() - started,
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'requantized': False, 'core_launch_authorized': False})
    finally:
        guard.restore()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=int, required=True, choices=range(6))
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=('all', 'produce', 'reload'), default='all')
    args = p.parse_args()
    require(Path.cwd().resolve() == PROJECT, 'wrong project directory')
    require(args.output.resolve().is_relative_to(PROJECT / 'results/org_m1_core_v1'), 'output outside new M1 results root')
    if args.phase != 'all':
        phase(args); return
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    for part in ('produce', 'reload'):
        subprocess.run([sys.executable, __file__, '--index', str(args.index), '--preflight', str(args.preflight),
                        '--output', str(args.output), '--phase', part], check=True)
    a, b = (read(args.output / (part + '.json')) for part in ('produce', 'reload'))
    require(a['runtime']['process_uuid'] != b['runtime']['process_uuid'] and a['runtime']['pid'] != b['runtime']['pid'], 'fresh process required')
    require(a['runtime']['gpu_uuid'] == b['runtime']['gpu_uuid'], 'GPU identity changed during smoke')
    require(set(a['arms']) == set(b['arms']) == set(ARMS), 'smoke arm completeness')
    write(args.output / 'final.json', {'status': 'PASS', 'scope': 'M1_CORE_CELL_ONLY', 'cell': a['cell'],
          'seconds': time.monotonic() - started, 'source_lock_sha256': a['source_lock_sha256'],
          'phases': {part: sha(args.output / (part + '.json')) for part in ('produce', 'reload')},
          'core_launch_authorized': False, 'independent_backup_complete': False})
    print('M1_CORE_CELL=PASS\nM1_CORE_LAUNCH_AUTHORIZED=false', flush=True)


if __name__ == '__main__':
    main()
