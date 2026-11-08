"""Mandatory real PyTorch CPU regression before any endpoint-repair GPU job."""
import argparse
import os
from pathlib import Path
import tempfile
from m1_contract_v1 import package_check, read, require, sha, write
from m1_endpoint_v3 import patch_weight


def run(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU regression requires GPUs hidden')
    source_sha = package_check()
    import torch
    from safetensors.torch import save_file, load_file
    require(not torch.cuda.is_initialized(), 'CUDA must remain uninitialized')
    q = torch.tensor([[1., -1., 2.**20, -2.**20, 0., 2.**-30]], dtype=torch.float32, device='cpu')
    u = torch.tensor([[2.**-30, -2.**-30, 2.**-20, -2.**-20, 1., 0.]], dtype=torch.float32, device='cpu')
    old = (q + (u - q)).to(torch.bfloat16)
    require(not torch.equal(old, u.to(torch.bfloat16)), 'cancellation regression did not reproduce')
    checks = ['old_formula_cancellation_reproduced']
    for dtype in (torch.float32, torch.bfloat16):
        a, b = q.to(dtype), u.to(dtype)
        a_before, b_before = a.clone(), b.clone()
        for dose in (0., .5, 1.):
            result = patch_weight(torch, a, b, dose)
            expected = (b.to(torch.bfloat16) if dose == 1. else a.to(torch.bfloat16) if dose == 0.
                        else (a.float() + .5 * (b.float() - a.float())).to(torch.bfloat16))
            require(torch.equal(result, expected), 'endpoint/interior regression')
            require(result.device.type == 'cpu' and result.dtype == torch.bfloat16, 'regression output type')
            require(result.data_ptr() not in (a.data_ptr(), b.data_ptr()), 'patch aliases inputs')
            with tempfile.TemporaryDirectory(prefix='m1_endpoint_cpu_') as temp:
                p = Path(temp) / 'patch.safetensors'
                save_file({'weight': result}, str(p))
                require(torch.equal(load_file(str(p))['weight'], expected), 'serialized endpoint mismatch')
            checks.append(str(dtype) + ':' + str(dose) + ':exact_independent_serialized')
        require(torch.equal(a, a_before) and torch.equal(b, b_before), 'input mutation')
    for badq, badu, dose in ((q, u[:, :1], 1.), (q, u, .25), (q * float('nan'), u, 1.), (q, u * float('inf'), 0.)):
        try:
            patch_weight(torch, badq, badu, dose)
        except ValueError:
            checks.append('invalid_input_rejected')
        else:
            raise ValueError('invalid input unexpectedly accepted')
    require(not torch.cuda.is_initialized(), 'CPU regression initialized CUDA')
    report = {'status': 'PASS', 'scope': 'EXACT_ENDPOINT_TORCH_CPU_ONLY', 'checks': checks,
              'source_lock_sha256': source_sha, 'torch': torch.__version__, 'gpu_hours': 0,
              'cuda_initialized': False, 'core_launch_authorized': False}
    write(output, report)
    print('M1_ENDPOINT_TORCH_CPU_REGRESSION=PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
