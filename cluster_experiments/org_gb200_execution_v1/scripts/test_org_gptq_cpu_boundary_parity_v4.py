#!/usr/bin/env python3
"""Locate the CPU/GPU arithmetic boundary that changes GPTQ tensors.

The job computes the frozen legacy reference once and, in one additional
forward pass, constructs two shared-input candidates:

* fast: accumulate Gram matrices on GPU, then copy and normalize on CPU;
* cpu_step: copy each Gram matrix and accumulate/normalize on CPU.

No production result is written by this diagnostic.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
V4_DIR = REPO_ROOT / "gate2_v4_final" / "scripts"
for directory in (SCRIPT_DIR, V4_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gate2_v4_backend import (  # noqa: E402
    collect_layer_hessians,
    get_module_by_name,
    get_weight_layers,
    quantize_dequantize,
)
from gate2_v4_contract import assert_manifest_sha, load_json, sha256_file  # noqa: E402
from gate2_v4_dataset import encode_qa_preserve_answer  # noqa: E402
from run_gate2_ground_truth_v4 import model_dtype  # noqa: E402
from test_org_gptq_collector_parity_v1 import canonical_input_layer  # noqa: E402


def raw_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def dual_shared_collect(
    model: Any,
    tokenizer: Any,
    calibration_data: Sequence[Mapping[str, Any]],
    target_layer_names: Sequence[str],
    *,
    device: torch.device,
    max_samples: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, str], dict[str, int]]:
    layer_to_group = {name: canonical_input_layer(name) for name in target_layer_names}
    groups = list(OrderedDict.fromkeys(layer_to_group.values()))
    module_map = {
        name: get_module_by_name(model, name.replace(".weight", "")) for name in groups
    }
    gpu_accum = {
        name: torch.zeros(module.in_features, module.in_features, dtype=torch.float32, device=device)
        for name, module in module_map.items()
    }
    cpu_step_accum = {
        name: torch.zeros(module.in_features, module.in_features, dtype=torch.float32, device="cpu")
        for name, module in module_map.items()
    }
    n_tokens = {name: 0 for name in groups}
    hooks = []
    for name, module in module_map.items():
        def make_hook(group_name: str):
            def hook_fn(_module: Any, args: tuple[Any, ...]) -> None:
                x = args[0].detach().to(device=device, dtype=torch.float32)
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                gram = x.T @ x
                gpu_accum[group_name].add_(gram)
                cpu_step_accum[group_name].add_(gram.cpu())
                n_tokens[group_name] += int(x.shape[0])
            return hook_fn
        hooks.append(module.register_forward_pre_hook(make_hook(name)))
    try:
        model.eval()
        total = min(max_samples, len(calibration_data))
        for index, sample in enumerate(list(calibration_data)[:max_samples], start=1):
            question = sample.get("question", sample.get("prompt", ""))
            answer = sample.get("answer", sample.get("response", sample.get("completion", "")))
            encoded = encode_qa_preserve_answer(tokenizer, str(question), str(answer), max_length=512)
            inputs = {
                "input_ids": encoded["input_ids"].unsqueeze(0).to(device),
                "attention_mask": encoded["attention_mask"].unsqueeze(0).to(device),
            }
            with torch.no_grad():
                model(**inputs)
            logging.info("dual-boundary calibration %d/%d", index, total)
    finally:
        for hook in hooks:
            hook.remove()

    fast_groups: dict[str, torch.Tensor] = {}
    for name in groups:
        if n_tokens[name] <= 0:
            raise SystemExit(f"No tokens collected for {name}")
        # The fast candidate crosses the CPU boundary only once per group.
        fast_groups[name] = gpu_accum[name].cpu()
        fast_groups[name].div_(n_tokens[name])
        # This candidate reproduces the legacy boundary on every sample.
        cpu_step_accum[name].div_(n_tokens[name])
    del gpu_accum
    gc.collect()
    torch.cuda.empty_cache()
    return (
        {name: fast_groups[layer_to_group[name]] for name in target_layer_names},
        {name: cpu_step_accum[layer_to_group[name]] for name in target_layer_names},
        layer_to_group,
        n_tokens,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--calibration_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def compare_hessians(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    names: Sequence[str],
) -> dict[str, Any]:
    exact_count = 0
    max_abs = 0.0
    mean_abs_max = 0.0
    failures = []
    for name in names:
        old = reference[name].float()
        new = candidate[name].float()
        delta = (old - new).abs()
        exact = bool(torch.equal(old, new))
        maximum = float(delta.max().item())
        mean = float(delta.mean().item())
        exact_count += int(exact)
        max_abs = max(max_abs, maximum)
        mean_abs_max = max(mean_abs_max, mean)
        if not exact:
            failures.append({"layer_name": name, "max_abs_delta": maximum, "mean_abs_delta": mean})
    return {
        "exact_count": exact_count,
        "failure_count": len(failures),
        "max_abs_delta": max_abs,
        "mean_abs_delta_max": mean_abs_max,
        "failures": failures,
    }


def compare_quantized(
    model: Any,
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    names: Sequence[str],
) -> dict[str, Any]:
    params = dict(model.named_parameters())
    rows = []
    for index, name in enumerate(names, start=1):
        weight = params[name].data.detach()
        old = quantize_dequantize(weight, "gptq_int4", hessian=reference[name])
        new = quantize_dequantize(weight, "gptq_int4", hessian=candidate[name])
        delta = (old.float() - new.float()).abs()
        denominator = torch.linalg.vector_norm(old.float()).clamp_min(1e-30)
        rows.append({
            "layer_name": name,
            "reference_sha256": raw_sha256(old),
            "candidate_sha256": raw_sha256(new),
            "exact": bool(torch.equal(old, new)),
            "mismatch_fraction": float((old != new).float().mean().item()),
            "relative_l2_delta": float((torch.linalg.vector_norm(delta) / denominator).item()),
            "max_abs_delta": float(delta.max().item()),
            "mean_abs_delta": float(delta.mean().item()),
        })
        print(f"BOUNDARY_TENSOR={index}/{len(names)} {name}", flush=True)
        del old, new, delta, denominator
        torch.cuda.empty_cache()
    return {
        "exact_count": sum(int(row["exact"]) for row in rows),
        "failure_count": sum(int(not row["exact"]) for row in rows),
        "comparisons": rows,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    manifest = load_json(args.calibration_manifest)
    assert_manifest_sha(manifest)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) < args.n_samples:
        raise SystemExit("Calibration manifest lacks requested samples")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=model_dtype(args.dtype), device_map="auto", trust_remote_code=True
    )
    model.eval()
    device = next(model.parameters()).device
    names = [name for name, _ in get_weight_layers(model)]
    if len(names) != 224:
        raise SystemExit(f"Expected 224 layers, got {len(names)}")

    start = time.time()
    legacy = collect_layer_hessians(model, tokenizer, samples, names, device=device, max_samples=args.n_samples)
    legacy_seconds = time.time() - start
    start = time.time()
    fast, cpu_step, layer_to_group, token_counts = dual_shared_collect(
        model, tokenizer, samples, names, device=device, max_samples=args.n_samples
    )
    dual_seconds = time.time() - start

    representative = [
        name for name in names
        if any(f"model.layers.{block}." in name for block in (0, 15, 31))
    ]
    fast_h = compare_hessians(legacy, fast, names)
    cpu_h = compare_hessians(legacy, cpu_step, names)
    start = time.time()
    fast_q = compare_quantized(model, legacy, fast, representative)
    fast_q_seconds = time.time() - start
    start = time.time()
    cpu_q = compare_quantized(model, legacy, cpu_step, representative)
    cpu_q_seconds = time.time() - start

    fast_pass = fast_h["exact_count"] == 224 and fast_q["exact_count"] == 21
    cpu_step_pass = cpu_h["exact_count"] == 224 and cpu_q["exact_count"] == 21
    status = "PASS_FAST" if fast_pass else ("PASS_CPU_ONLY" if cpu_step_pass else "FAIL")
    result = {
        "schema_version": "org_gptq_cpu_boundary_parity_v4",
        "status": status,
        "model_dir": args.model_dir,
        "calibration_manifest": args.calibration_manifest,
        "calibration_manifest_file_sha256": sha256_file(args.calibration_manifest),
        "script_sha256": sha256_file(Path(__file__)),
        "n_samples": args.n_samples,
        "target_layer_count": len(names),
        "representative_tensor_count": len(representative),
        "unique_shared_input_group_count": len(set(layer_to_group.values())),
        "shared_group_token_counts": token_counts,
        "legacy_seconds": legacy_seconds,
        "dual_candidate_seconds": dual_seconds,
        "fast_quantize_seconds": fast_q_seconds,
        "cpu_step_quantize_seconds": cpu_q_seconds,
        "fast": {"hessian": fast_h, "quantized": fast_q},
        "cpu_step": {"hessian": cpu_h, "quantized": cpu_q},
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
    print(f"ORG_GPTQ_CPU_BOUNDARY={status}")
    print(f"FAST_HESSIAN_EXACT={fast_h['exact_count']}/224")
    print(f"FAST_QUANTIZED_EXACT={fast_q['exact_count']}/21")
    print(f"CPU_STEP_HESSIAN_EXACT={cpu_h['exact_count']}/224")
    print(f"CPU_STEP_QUANTIZED_EXACT={cpu_q['exact_count']}/21")
    print(f"LEGACY_SECONDS={legacy_seconds:.6f}")
    print(f"DUAL_CANDIDATE_SECONDS={dual_seconds:.6f}")
    print(f"OUTPUT={output}")
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

