"""Isolated RTN loader repair; preserve V1 evidence and strengthen reload checks."""
from pathlib import Path
import run_org_b05_real_model_smoke_v1 as base
from diagnose_org_b05_rtn_reload_v1 import corrected_load, buffers

original_worker = base.worker
original_synthetic = base.synthetic


def worker(a):
    import torch
    import run_org_e2b_b04_packed_public_canary_v1 as packed
    old_loader = packed.load_rtn
    old_synthetic = base.synthetic

    def checked_forward(torch, model):
        x = original_synthetic(torch, model)
        before = buffers(torch, model)
        repeated = original_synthetic(torch, model)
        after = buffers(torch, model)
        base.require(before == after, 'nonpersistent buffers changed on repeated forward')
        base.require(float((x-repeated).abs().max()) <= 1e-6, 'repeat forward drift')
        if a.phase == 'produce':
            base.atomic(a.output/'nonpersistent_buffers.json', after)
        else:
            base.require(after == base.read(a.output/'nonpersistent_buffers.json'),
                         'nonpersistent buffer dtype/value drift on reload')
        return x

    try:
        packed.load_rtn = lambda torch, tf, artifact, names: corrected_load(torch, tf, packed, artifact, names)
        base.synthetic = checked_forward
        original_worker(a)
    finally:
        packed.load_rtn = old_loader
        base.synthetic = old_synthetic


if __name__ == '__main__':
    # V1 orchestrator must launch this repaired entrypoint in each fresh process.
    base.__file__ = str(Path(__file__).resolve())
    base.worker = worker
    base.main()
