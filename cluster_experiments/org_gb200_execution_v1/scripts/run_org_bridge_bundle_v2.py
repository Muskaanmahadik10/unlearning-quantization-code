#!/usr/bin/env python3
"""Run one ORG cell as a shared-state bundle with cached quantized tensors."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
import time
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

import run_gate2_whole_model_bridge_v4 as bridge  # noqa: E402
from gate2_v4_backend import get_weight_layers, quantize_dequantize  # noqa: E402
from gate2_v4_contract import (  # noqa: E402
    assert_forget_manifest_ready,
    assert_manifest_sha,
    load_json,
    provenance,
    require_source_lock,
    save_json,
    sha256_file,
)
from run_gate2_ground_truth_v4 import model_dtype  # noqa: E402
from org_gptq_shared_cpu_finalize_collector_v3 import (  # noqa: E402
    collect_shared_hessians_gpu_cpu_finalize,
)
from validate_bridge_v4 import validate_obj  # noqa: E402

LOGGER = logging.getLogger("org_bridge_bundle_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--quantizer", choices=("nf4", "gptq_int4", "rtn_int8"), required=True)
    parser.add_argument("--ranking_sources", nargs="+", required=True)
    parser.add_argument("--k_values", nargs="+", type=int, required=True)
    parser.add_argument("--rankings_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--retain_manifest_path", required=True)
    parser.add_argument("--calibration_manifest_path")
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--source_lock_path", required=True)
    parser.add_argument("--expected_script_sha256", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--n_retain", type=int, default=128)
    parser.add_argument("--n_calibration", type=int, default=128)
    parser.add_argument("--manifest_n_samples", type=int, default=128)
    parser.add_argument("--manifest_n_retain", type=int, default=128)
    parser.add_argument("--manifest_n_calibration", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def request_args(base: argparse.Namespace, condition: str, source: str | None, k: int) -> argparse.Namespace:
    values = vars(base).copy()
    values.update(condition=condition, ranking_source=source, k=k)
    return argparse.Namespace(**values)


def load_cell_rankings(path: Path, cell_id: str, sources: Sequence[str]) -> tuple[dict[str, list[dict]], str]:
    wanted = set(sources)
    grouped = {source: [] for source in sources}
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        for binary_line in raw:
            digest.update(binary_line)
            if not binary_line.strip():
                continue
            row = json.loads(binary_line)
            source = str(row.get("ranking_source", ""))
            if row.get("cell_id") == cell_id and source in wanted:
                grouped[source].append(row)
    for source, rows in grouped.items():
        rows.sort(key=lambda row: (int(row.get("rank", 10**9)), str(row.get("layer_name", ""))))
        if len(rows) != 224 or len({row["layer_name"] for row in rows}) != 224:
            raise SystemExit(f"Ranking {cell_id} {source} is not a 224-layer permutation")
    return grouped, digest.hexdigest()


def ranking_evidence(
    *, source: str, k: int, rows: Sequence[Mapping[str, Any]], rankings_path: Path,
    rankings_sha256: str,
) -> tuple[list[str], dict[str, Any]]:
    selected = list(rows[:k])
    layers = [str(row["layer_name"]) for row in selected]
    evidence_rows = []
    for row in selected:
        evidence_rows.append(
            {
                "layer_name": str(row.get("layer_name", "")),
                "rank": int(row.get("rank", 0)),
                "score": float(row.get("score", 0.0)),
                "score_direction": str(row.get("score_direction", "")),
                "source_lock_sha256": str(row.get("source_lock_sha256", "")),
                "manifest_sha256": str(row.get("manifest_sha256", "")),
                "evidence_path": str(row.get("evidence_path", "")),
                "evidence_sha256": str(row.get("evidence_sha256", "")),
                "org_selector_evidence": row.get("org_selector_evidence"),
            }
        )
    evidence = {
        "ranking_source": source,
        "k": k,
        "protected_layer_count": len(layers),
        "rankings_path": str(rankings_path),
        "rankings_sha256": rankings_sha256,
        "ranking_schema_version": str(rows[0].get("schema_version", "")),
        "selected_rows": evidence_rows,
        "selected_source_locks": sorted({row["source_lock_sha256"] for row in evidence_rows}),
        "selected_manifest_sha256s": sorted({row["manifest_sha256"] for row in evidence_rows}),
        "selected_evidence_paths": sorted({row["evidence_path"] for row in evidence_rows}),
    }
    return layers, evidence


def raw_tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def compression_for(
    target_layers: Sequence[str], protected_layers: Sequence[str], layer_cache: Mapping[str, Mapping[str, Any]],
    quantizer: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    protected = set(protected_layers)
    per_layer = []
    target_params = protected_params = quantized_params = 0
    for name in target_layers:
        cached = layer_cache[name]
        n_params = int(cached["n_params"])
        target_params += n_params
        is_protected = name in protected
        if is_protected:
            protected_params += n_params
        else:
            quantized_params += n_params
        per_layer.append(
            {
                "layer_name": name,
                "protected": is_protected,
                "quantized": not is_protected,
                "n_params": n_params,
                "epsilon_l2": 0.0 if is_protected else float(cached["epsilon_l2"]),
                "max_abs_delta": 0.0 if is_protected else float(cached["max_abs_delta"]),
                "original_tensor_sha256": cached["original_tensor_sha256"],
                "quantized_tensor_sha256": cached["quantized_tensor_sha256"],
            }
        )
    bits = 4 if quantizer in ("nf4", "gptq_int4") else 8
    compression = {
        "target_params": float(target_params),
        "protected_params": float(protected_params),
        "quantized_params": float(quantized_params),
        "protected_parameter_fraction": float(protected_params / target_params),
        "quantized_parameter_fraction": float(quantized_params / target_params),
        "quantizer_bits": float(bits),
        "effective_average_bits": float((protected_params * 16 + quantized_params * bits) / target_params),
    }
    return per_layer, compression


def install_condition(
    model: Any, target_layers: Sequence[str], protected_layers: Sequence[str], layer_cache: Mapping[str, Mapping[str, Any]],
) -> None:
    protected = set(protected_layers)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name in target_layers:
            source = layer_cache[name]["original"] if name in protected else layer_cache[name]["quantized"]
            weight = params[name]
            weight.data.copy_(source.to(device=weight.device, dtype=weight.dtype))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    bridge.reject_bad_bridge_root(Path(args.results_root))
    if sha256_file(Path(__file__)) != args.expected_script_sha256:
        raise SystemExit("Bundle script SHA mismatch")
    source_info = require_source_lock(
        script_path=Path(__file__), source_lock_path=args.source_lock_path,
        expected_script_sha256=args.expected_script_sha256,
    )
    source_lock_sha256 = str(source_info["source_lock_sha256"])

    forget_obj = load_json(args.manifest_path)
    assert_forget_manifest_ready(forget_obj, n_samples=args.manifest_n_samples)
    bridge.require_manifest_sample_count(
        forget_obj, expected=args.manifest_n_samples, requested=args.n_samples, label="forget"
    )
    forget_desc = bridge.manifest_descriptor(args.manifest_path)
    _, forget_samples, _ = bridge.load_manifest_samples(args.manifest_path, args.n_samples)
    retain_obj = load_json(args.retain_manifest_path)
    assert_manifest_sha(retain_obj)
    bridge.require_manifest_sample_count(
        retain_obj, expected=args.manifest_n_retain, requested=args.n_retain, label="retain"
    )
    _, retain_samples, _ = bridge.validate_retain_manifest(args.retain_manifest_path, args.n_retain)
    retain_desc = bridge.manifest_descriptor(args.retain_manifest_path)
    calibration_desc = None
    calibration_samples = None
    if args.quantizer == "gptq_int4":
        if not args.calibration_manifest_path:
            raise SystemExit("GPTQ bundle requires calibration manifest")
        calibration_obj = load_json(args.calibration_manifest_path)
        assert_manifest_sha(calibration_obj)
        bridge.require_manifest_sample_count(
            calibration_obj, expected=args.manifest_n_calibration,
            requested=args.n_calibration, label="calibration",
        )
        _, calibration_samples, _ = bridge.validate_retain_manifest(
            args.calibration_manifest_path, args.n_calibration
        )
        calibration_desc = bridge.manifest_descriptor(args.calibration_manifest_path)

    cid = bridge.cell_id(args)
    rankings_path = Path(args.rankings_path)
    grouped_rankings, rankings_sha256 = load_cell_rankings(rankings_path, cid, args.ranking_sources)
    LOGGER.info("Loading model %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=model_dtype(args.dtype), device_map="auto", trust_remote_code=True
    )
    model.eval()
    device = next(model.parameters()).device
    target_layers = [name for name, _ in get_weight_layers(model)]
    if len(target_layers) != 224:
        raise SystemExit(f"Expected 224 target layers, got {len(target_layers)}")

    bundle_start = time.time()
    bf16_start = time.time()
    bf16_metrics = bridge.run_metrics(
        model, tokenizer, forget_samples, retain_samples, device=device, max_length=args.max_length
    )
    bf16_seconds = time.time() - bf16_start
    hessians: Mapping[str, torch.Tensor] = {}
    hessian_seconds = 0.0
    hessian_groups: Mapping[str, str] = {}
    hessian_tokens: Mapping[str, int] = {}
    if args.quantizer == "gptq_int4":
        start = time.time()
        hessians, hessian_groups, hessian_tokens = collect_shared_hessians_gpu_cpu_finalize(
            model, tokenizer, calibration_samples or [], target_layers,
            device=device, max_samples=args.n_calibration,
        )
        hessian_seconds = time.time() - start

    params = dict(model.named_parameters())
    layer_cache: dict[str, dict[str, Any]] = {}
    quantize_start = time.time()
    for index, name in enumerate(target_layers, start=1):
        weight = params[name]
        original = weight.data.detach().clone()
        quantized = quantize_dequantize(original, args.quantizer, hessian=hessians.get(name))
        delta = (quantized - original).float()
        original_cpu = original.to(device="cpu")
        quantized_cpu = quantized.to(device="cpu")
        layer_cache[name] = {
            "original": original_cpu,
            "quantized": quantized_cpu,
            "n_params": int(weight.numel()),
            "epsilon_l2": float(torch.linalg.vector_norm(delta).cpu().item()),
            "max_abs_delta": float(delta.abs().max().cpu().item()),
            "original_tensor_sha256": raw_tensor_sha256(original_cpu),
            "quantized_tensor_sha256": raw_tensor_sha256(quantized_cpu),
        }
        del original, quantized, delta
        if index % 16 == 0:
            LOGGER.info("Cached quantized tensors %d/%d", index, len(target_layers))
    quantize_seconds = time.time() - quantize_start
    del hessians
    gc.collect()
    torch.cuda.empty_cache()
    precompute = {
        "bf16_seconds": bf16_seconds,
        "hessian_seconds": hessian_seconds,
        "quantize_cache_seconds": quantize_seconds,
        "unique_hessian_groups": len(set(hessian_groups.values())) if hessian_groups else 0,
        "hessian_group_token_counts": dict(hessian_tokens),
    }

    requests: list[tuple[argparse.Namespace, list[str], Mapping[str, Any] | None]] = []
    full_args = request_args(args, "full_quantized", None, 0)
    requests.append((full_args, [], None))
    for k in args.k_values:
        if k < 0 or k > len(target_layers):
            raise SystemExit(f"Invalid k={k}")
        for source in args.ranking_sources:
            protected, evidence = ranking_evidence(
                source=source, k=k, rows=grouped_rankings[source], rankings_path=rankings_path,
                rankings_sha256=rankings_sha256,
            )
            requests.append((request_args(args, "protected", source, k), protected, evidence))

    written = skipped = 0
    for condition_args, protected_layers, evidence in requests:
        path = bridge.output_path(condition_args)
        if bridge.result_complete_ok(
            path, source_lock_sha256, condition_args,
            forget_manifest_desc=forget_desc, retain_manifest_desc=retain_desc,
            calibration_manifest_desc=calibration_desc,
        ) and not args.force:
            skipped += 1
            continue
        condition_start = time.time()
        install_condition(model, target_layers, protected_layers, layer_cache)
        per_layer, compression = compression_for(
            target_layers, protected_layers, layer_cache, args.quantizer
        )
        metrics = bridge.run_metrics(
            model, tokenizer, forget_samples, retain_samples, device=device, max_length=args.max_length
        )
        result = bridge.build_result(
            condition_args,
            source_lock_sha256=source_lock_sha256,
            forget_manifest_desc=forget_desc,
            forget_manifest_obj=forget_obj,
            retain_manifest_desc=retain_desc,
            calibration_manifest_desc=calibration_desc,
            target_layers=target_layers,
            protected_layers=protected_layers,
            per_layer=per_layer,
            compression=compression,
            bf16_metrics=bf16_metrics,
            condition_metrics=metrics,
            ranking_evidence=evidence,
            elapsed_s=time.time() - condition_start,
            source_info=source_info,
        )
        result["provenance"] = provenance(
            script_path=Path(__file__), argv=sys.argv, source_info=source_info,
            repo_root=Path.cwd(),
            extra={"execution_mode": "shared_state_cached_quantized_tensor_bundle_v2_cpu_finalized"},
        )
        result["bundle_precomputation"] = precompute
        result["bundle_elapsed_before_write_s"] = time.time() - bundle_start
        errors = validate_obj(result, path=path, expected_source_lock_sha256=source_lock_sha256)
        if errors:
            raise SystemExit(f"Bundled result failed validation: {errors}")
        save_json(path, result)
        LOGGER.info("Wrote %s", path)
        written += 1
    print("ORG_BRIDGE_BUNDLE=PASS")
    print(f"CELL_ID={cid}")
    print(f"WRITTEN={written}")
    print(f"SKIPPED={skipped}")
    print(f"BUNDLE_SECONDS={time.time() - bundle_start:.6f}")
    print(f"HESSIAN_SECONDS={hessian_seconds:.6f}")
    print(f"QUANTIZE_CACHE_SECONDS={quantize_seconds:.6f}")


if __name__ == "__main__":
    main()

