"""Exact Z/R endpoints; unchanged interior interpolation and strict assertions."""
from m1_contract_v1 import require


def patch_weight(torch, quantized, original, dose):
    require(dose in (0.0, 0.5, 1.0), 'unapproved dose')
    require(quantized.shape == original.shape, 'patch shape')
    require(bool(torch.isfinite(quantized).all() and torch.isfinite(original).all()), 'nonfinite patch inputs')
    if dose == 1.0:
        w = original.to(torch.bfloat16).clone()
    elif dose == 0.0:
        w = quantized.to(torch.bfloat16).clone()
    else:
        w = (quantized.float() + dose * (original.float() - quantized.float())).to(torch.bfloat16)
    require(bool(torch.isfinite(w).all()), 'nonfinite patch')
    if dose == 1.0:
        require(torch.equal(w, original.to(torch.bfloat16)), 'restoration endpoint not exact')
    if dose == 0.0:
        require(torch.equal(w, quantized.to(torch.bfloat16)), 'sham endpoint')
    return w
