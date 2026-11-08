#!/usr/bin/env python3
"""Compare legacy and shared-input GPU GPTQ Hessian collectors on frozen samples."""

from __future__ import annotations

import argparse
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
if str(V4_DIR) not in sys.path:
    sys.path.insert(0, str(V4_DIR))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gate2_v4_backend import (  # noqa: E402
    collect_layer_hessians,
    get_module_by_name,
    get_weight_layers,
)
from gate2_v4_contract import assert_manifest_sha, load_json, sha256_file  # noqa: E402
from gate2_v4_dataset import encode_qa_preserve_answer  # noqa: E402
from run_gate2_ground_truth_v4 import model_dtype  # noqa: E402

LOGGER = logging.getLogger("org_gptq_collector_parity")
PAIR_REPLACEMENTS = (
    (".self_attn.k_proj.weight", ".self_attn.q_proj.weight"),
    (".self_attn.v_proj.weight", ".self_attn.q_proj.weight"),
    (".mlp.up_proj.weight", ".mlp.gate_proj.weight"),
)


def canonical_input_layer(layer_name: str) -> str:
    for suffix, replacement in PAIR_REPLACEMENTS:
        if layer_name.endswith(suffix):
            return layer_name[: -len(suffix)] + replacement
    return layer_name


def collect_shared_hessians_gpu(
    model: Any,
    tokenizer: Any,
    calibration_data: Sequence[Mapping[str, Any]],
    target_layer_names: Sequence[str],
    *,
    device: torch.device,
    max_samples: int,
) -> tuple[dict[str, torch.Tensor], dict[str, str], dict[str, int]]:
    layer_to_group = {name: canonical_input_layer(name) for name in target_layer_names}
    groups = list(OrderedDict.fromkeys(layer_to_group.values()))
    module_map = {
        name: get_module_by_name(model, name.replace(".weight", "")) for name in groups
    }
    accumulators = {
        name: torch.zeros(
            module.in_features,
            module.in_features,
            dtype=torch.float32,
            device=device,
        )
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
                accumulators[group_name].addmm_(x.T, x)
                n_tokens[group_name] += int(x.shape[0])
            return hook_fn
        hooks.append(module.register_forward_pre_hook(make_hook(name)))
    try:
        model.eval()
        total = min(max_samples, len(calibration_data))
        for index, sample in enumerate(list(calibration_data)[:max_samples], start=1):
            question = sample.get("question", sample.get("prompt", ""))
            answer = sample.get("answer", sample.get("response", sample.get("completion", "")))
            encoded = encode_qa_preserve_answer(
                tokenizer, str(question), str(answer), max_length=512
            )
            inputs = {
                "input_ids": encoded["input_ids"].unsqueeze(0).to(device),
                "attention_mask": encoded["attention_mask"].unsqueeze(0).to(device),
            }
            with torch.no_grad():
                model(**inputs)
            LOGGER.info("shared-GPU calibration %d/%d", index, total)
    finally:
        for hook in hooks:
            hook.remove()
    for name in groups:
        if n_tokens[name] <= 0:
            raise SystemExit(f"No tokens collected for {name}")
        accumulators[name].div_(n_tokens[name])
    return (
        {name: accumulators[layer_to_group[name]] for name in target_layer_names},
        layer_to_group,
        n_tokens,
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--calibration_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_samples", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    manifest = load_json(args.calibration_manifest)
    assert_manifest_sha(manifest)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) < args.n_samples:
        raise SystemExit("Calibration manifest lacks requested samples")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=model_dtype(args.dtype),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    all_layers = [name for name, _ in get_weight_layers(model)]
    wanted_suffixes = (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    )
    selected = [name for name in all_layers if name in wanted_suffixes]
    if len(selected) != len(wanted_suffixes):
        raise SystemExit(f"Expected {len(wanted_suffixes)} parity layers, found {selected}")

    start = time.time()
    legacy = collect_layer_hessians(
        model,
        tokenizer,
        samples,
        selected,
        device=device,
        max_samples=args.n_samples,
    )
    legacy_seconds = time.time() - start
    start = time.time()
    shared, layer_to_group, shared_tokens = collect_shared_hessians_gpu(
        model,
        tokenizer,
        samples,
        selected,
        device=device,
        max_samples=args.n_samples,
    )
    shared_seconds = time.time() - start
    comparisons = []
    for name in selected:
        old = legacy[name].float().cpu()
        new = shared[name].float().cpu()
        delta = (old - new).abs()
        comparisons.append(
            {
                "layer_name": name,
                "canonical_input_layer": layer_to_group[name],
                "shape": list(old.shape),
                "legacy_sha256": tensor_sha256(old),
                "shared_sha256": tensor_sha256(new),
                "max_abs_delta": float(delta.max().item()),
                "mean_abs_delta": float(delta.mean().item()),
                "allclose_atol_1e-5_rtol_1e-5": bool(
                    torch.allclose(old, new, atol=1e-5, rtol=1e-5)
                ),
            }
        )
    result = {
        "schema_version": "org_gptq_collector_parity_v1",
        "status": "PASS" if all(row["allclose_atol_1e-5_rtol_1e-5"] for row in comparisons) else "FAIL",
        "model_dir": args.model_dir,
        "calibration_manifest": args.calibration_manifest,
        "calibration_manifest_file_sha256": sha256_file(args.calibration_manifest),
        "script_sha256": sha256_file(Path(__file__)),
        "dependency_sha256": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in (
                V4_DIR / "gate2_v4_backend.py",
                V4_DIR / "gate2_v4_contract.py",
                V4_DIR / "gate2_v4_dataset.py",
                V4_DIR / "run_gate2_ground_truth_v4.py",
            )
        },
        "n_samples": args.n_samples,
        "selected_layers": selected,
        "unique_shared_input_groups": sorted(set(layer_to_group.values())),
        "shared_group_token_counts": shared_tokens,
        "legacy_seconds": legacy_seconds,
        "shared_gpu_seconds": shared_seconds,
        "comparisons": comparisons,
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
    print(f"ORG_GPTQ_COLLECTOR_PARITY={result['status']}")
    print(f"LEGACY_SECONDS={legacy_seconds:.6f}")
    print(f"SHARED_GPU_SECONDS={shared_seconds:.6f}")
    print(f"OUTPUT={output}")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
