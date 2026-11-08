#!/usr/bin/env python3
"""Exercise the Gate 2 v4 bridge runtime before loading a model checkpoint."""

from __future__ import annotations

import inspect
import json
import platform
from importlib import metadata
from typing import Any, Dict

import torch

from gate2_v4_backend import (
    collect_layer_hessians,
    get_weight_layers,
    quantize_dequantize,
)


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "MISSING"


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.float()).all().item())


def exercise_quantizer(name: str, weight: torch.Tensor) -> Dict[str, Any]:
    hessian = None
    if name == "gptq_int4":
        hessian = torch.eye(weight.shape[1], dtype=torch.float32, device=weight.device)
    output = quantize_dequantize(weight, name, hessian=hessian)
    if output.shape != weight.shape:
        raise RuntimeError(f"{name} shape mismatch: {tuple(output.shape)} != {tuple(weight.shape)}")
    if output.device != weight.device:
        raise RuntimeError(f"{name} device mismatch: {output.device} != {weight.device}")
    if not finite_tensor(output):
        raise RuntimeError(f"{name} produced non-finite values")
    delta = (output.float() - weight.float()).abs()
    if float(delta.max().item()) <= 0.0:
        raise RuntimeError(f"{name} produced a zero perturbation")
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "max_abs_delta": float(delta.max().item()),
        "mean_abs_delta": float(delta.mean().item()),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    signature = inspect.signature(collect_layer_hessians)
    if "max_samples" not in signature.parameters:
        raise SystemExit("collect_layer_hessians lacks max_samples")
    if "max_length" in signature.parameters:
        raise SystemExit("unexpected collect_layer_hessians max_length parameter")
    annotations = inspect.signature(get_weight_layers).return_annotation
    weight = torch.linspace(-3.0, 3.0, steps=64 * 64, device="cuda", dtype=torch.float32).reshape(64, 64)
    quantizers = {
        name: exercise_quantizer(name, weight)
        for name in ("nf4", "rtn_int8", "gptq_int4")
    }
    report = {
        "status": "PASS",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"),
        "bitsandbytes": package_version("bitsandbytes"),
        "cuda_build": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "collect_layer_hessians_parameters": list(signature.parameters),
        "get_weight_layers_return_annotation": str(annotations),
        "quantizers": quantizers,
    }
    if report["bitsandbytes"] == "MISSING":
        raise SystemExit("bitsandbytes package metadata is missing")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("BRIDGE_RUNTIME_PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
