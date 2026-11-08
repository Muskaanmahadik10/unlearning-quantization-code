#!/usr/bin/env python3
"""Production-depth GPTQ parity on 21 layers across 128 samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
V4_DIR = REPO_ROOT / "gate2_v4_final" / "scripts"
for directory in (SCRIPT_DIR, V4_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gate2_v4_backend import collect_layer_hessians, get_weight_layers, quantize_dequantize  # noqa: E402
from gate2_v4_contract import assert_manifest_sha, load_json, sha256_file  # noqa: E402
from org_gptq_shared_cpu_finalize_collector_v3 import (  # noqa: E402
    collect_shared_hessians_gpu_cpu_finalize,
)
from run_gate2_ground_truth_v4 import model_dtype  # noqa: E402


def raw_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--calibration_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples != 128:
        raise SystemExit("This production-depth gate requires exactly 128 samples")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    manifest = load_json(args.calibration_manifest)
    assert_manifest_sha(manifest)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) < args.n_samples:
        raise SystemExit("Calibration manifest lacks 128 samples")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=model_dtype(args.dtype), device_map="auto", trust_remote_code=True
    )
    model.eval()
    device = next(model.parameters()).device
    all_names = [name for name, _ in get_weight_layers(model)]
    if len(all_names) != 224:
        raise SystemExit(f"Expected 224 model layers, got {len(all_names)}")
    names = [
        name for name in all_names
        if any(f"model.layers.{block}." in name for block in (0, 15, 31))
    ]
    if len(names) != 21:
        raise SystemExit(f"Expected 21 representative layers, got {len(names)}")

    start = time.time()
    legacy = collect_layer_hessians(
        model, tokenizer, samples, names, device=device, max_samples=args.n_samples
    )
    legacy_seconds = time.time() - start
    start = time.time()
    fast, layer_to_group, token_counts = collect_shared_hessians_gpu_cpu_finalize(
        model, tokenizer, samples, names, device=device, max_samples=args.n_samples
    )
    fast_seconds = time.time() - start

    hessian_rows = []
    for name in names:
        old = legacy[name].float()
        new = fast[name].float()
        delta = (old - new).abs()
        hessian_rows.append({
            "layer_name": name,
            "canonical_input_layer": layer_to_group[name],
            "reference_sha256": raw_sha256(old),
            "candidate_sha256": raw_sha256(new),
            "exact": bool(torch.equal(old, new)),
            "max_abs_delta": float(delta.max().item()),
            "mean_abs_delta": float(delta.mean().item()),
        })

    params = dict(model.named_parameters())
    quantized_rows = []
    quantize_start = time.time()
    for index, name in enumerate(names, start=1):
        weight = params[name].data.detach()
        old = quantize_dequantize(weight, "gptq_int4", hessian=legacy[name])
        new = quantize_dequantize(weight, "gptq_int4", hessian=fast[name])
        delta = (old.float() - new.float()).abs()
        denominator = torch.linalg.vector_norm(old.float()).clamp_min(1e-30)
        quantized_rows.append({
            "layer_name": name,
            "reference_sha256": raw_sha256(old),
            "candidate_sha256": raw_sha256(new),
            "exact": bool(torch.equal(old, new)),
            "mismatch_fraction": float((old != new).float().mean().item()),
            "relative_l2_delta": float((torch.linalg.vector_norm(delta) / denominator).item()),
            "max_abs_delta": float(delta.max().item()),
            "mean_abs_delta": float(delta.mean().item()),
        })
        print(f"PRODUCTION128_TENSOR={index}/21 {name}", flush=True)
        del old, new, delta, denominator
        torch.cuda.empty_cache()
    quantize_seconds = time.time() - quantize_start

    hessian_exact = sum(int(row["exact"]) for row in hessian_rows)
    quantized_exact = sum(int(row["exact"]) for row in quantized_rows)
    status = "PASS" if hessian_exact == 21 and quantized_exact == 21 else "FAIL"
    result = {
        "schema_version": "org_gptq_production128_parity_v5",
        "status": status,
        "pass_contract": "all 21 representative Hessians and GPTQ tensors bit-identical at production n=128",
        "model_dir": args.model_dir,
        "calibration_manifest": args.calibration_manifest,
        "calibration_manifest_file_sha256": sha256_file(args.calibration_manifest),
        "script_sha256": sha256_file(Path(__file__)),
        "collector_sha256": sha256_file(SCRIPT_DIR / "org_gptq_shared_cpu_finalize_collector_v3.py"),
        "n_samples": args.n_samples,
        "model_layer_count": len(all_names),
        "representative_blocks": [0, 15, 31],
        "representative_layer_count": len(names),
        "hessian_exact_count": hessian_exact,
        "quantized_exact_count": quantized_exact,
        "legacy_seconds": legacy_seconds,
        "fast_seconds": fast_seconds,
        "quantize_seconds": quantize_seconds,
        "shared_group_token_counts": token_counts,
        "hessian_comparisons": hessian_rows,
        "quantized_comparisons": quantized_rows,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    tmp.replace(output)
    print(f"ORG_GPTQ_PRODUCTION128_PARITY={status}")
    print(f"HESSIAN_EXACT={hessian_exact}/21")
    print(f"QUANTIZED_EXACT={quantized_exact}/21")
    print(f"LEGACY_SECONDS={legacy_seconds:.6f}")
    print(f"FAST_SECONDS={fast_seconds:.6f}")
    print(f"QUANTIZE_SECONDS={quantize_seconds:.6f}")
    print(f"OUTPUT={output}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

