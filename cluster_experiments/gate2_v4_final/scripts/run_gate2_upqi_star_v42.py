#!/usr/bin/env python3
"""Corrected source-locked UPQI-star producer for one Gate 2 v4 cell."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import gate2_v4_backend as backend  # noqa: E402
import run_gate2_upqi_v4 as legacy  # noqa: E402
from gate2_metric_v42_math import (  # noqa: E402
    GAUSS_LEGENDRE_CONTRACT,
    UPQI_ABS_CONTRACT,
    UPQI_SIGNED_CONTRACT,
    UPQI_STAR_SCHEMA,
    gauss_legendre_unit_interval,
    integrate_gradient_samples,
    upqi_scores,
)
from gate2_v4_contract import (  # noqa: E402
    assert_forget_manifest_ready,
    load_json,
    loss_contract,
    provenance,
    reject_legacy_result_path,
    require_source_lock,
    save_json,
    sha256_file,
)
from validate_upqi_star_v42 import validate_obj  # noqa: E402


LOGGER = logging.getLogger("gate2_upqi_star_v42")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--dataset", required=True, choices=("tofu", "muse_news"))
    parser.add_argument("--quantizer", required=True, choices=("nf4", "gptq_int4", "rtn_int8"))
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--legacy_upqi_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--calibration_manifest_path")
    parser.add_argument("--source_lock_path", required=True)
    parser.add_argument("--expected_script_sha256", required=True)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--n_calibration", type=int, default=128)
    parser.add_argument("--n_quadrature", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max_layers", type=int)
    parser.add_argument(
        "--layer_indices",
        help="Comma-separated zero-based indices into the canonical target-layer order.",
    )
    parser.add_argument("--stop_after_layers", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def reference(path: str) -> dict[str, Any]:
    obj = load_json(path)
    return {
        "path": str(Path(path)),
        "sha256": sha256_file(path),
        "schema_version": obj.get("schema_version"),
        "source_lock_sha256": obj.get("provenance", {}).get("source_lock_sha256"),
        "sample_manifest_sha256": obj.get("sample_manifest", {}).get("manifest_sha256"),
    }


def gt_layer_map(path: str) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    obj = load_json(path)
    layers = obj.get("layers")
    if obj.get("schema_version") != "gate2_gt_v4.0" or not isinstance(layers, Mapping):
        raise ValueError(f"GT reference is not a complete v4 result: {path}")
    return obj, layers


def compute_layers(
    model: Any,
    tokenizer: Any,
    samples: Sequence[Mapping[str, Any]],
    target_layers: Sequence[str],
    global_layer_indices: Mapping[str, int],
    gt_layers: Mapping[str, Mapping[str, Any]],
    *,
    quantizer: str,
    hessians: Optional[Mapping[str, torch.Tensor]],
    nodes: Sequence[float],
    weights: Sequence[float],
    batch_size: int,
    max_length: int,
    device: torch.device,
    resume: Sequence[Mapping[str, Any]],
    checkpoint: Any,
    stop_after_layers: Optional[int],
) -> list[dict[str, Any]]:
    params = dict(model.named_parameters())
    completed = {str(row["layer"]): dict(row) for row in resume}
    output = [completed[name] for name in target_layers if name in completed]
    computed = 0
    for index, name in enumerate(target_layers):
        if name in completed:
            continue
        if name not in gt_layers:
            raise KeyError(f"GT reference missing layer {name}")
        param = params[name]
        original = param.data.clone()
        hessian = hessians[name].to(param.device) if hessians and name in hessians else None
        eps = legacy.displacement_for_layer(param, quantizer, hessian)
        gradients: list[torch.Tensor] = []
        losses: list[float] = []
        token_counts: list[int] = []
        try:
            for node in nodes:
                interpolated = original.float() + float(node) * eps.to(param.device)
                param.data.copy_(interpolated.to(param.data.dtype))
                grad, loss, tokens = legacy.mean_loss_grad_for_layer(
                    model,
                    tokenizer,
                    samples,
                    name,
                    device=device,
                    batch_size=batch_size,
                    max_length=max_length,
                )
                gradients.append(grad)
                losses.append(float(loss))
                token_counts.append(int(tokens))
        finally:
            param.data.copy_(original)
        integrated = integrate_gradient_samples(gradients, weights)
        scores = upqi_scores(integrated, eps)
        gt_delta = float(gt_layers[name]["delta_forget_ce"])
        row = {
            "layer": name,
            "layer_index": index,
            "global_layer_index": int(global_layer_indices[name]),
            "block_idx": backend.extract_block_idx(name),
            "submodule": backend.classify_submodule(name),
            **scores,
            "epsilon_l2": float(torch.linalg.vector_norm(eps.double()).item()),
            "epsilon_max": float(torch.max(torch.abs(eps.double())).item()),
            "integrated_grad_l2": float(torch.linalg.vector_norm(integrated).item()),
            "weighted_mean_path_loss": float(sum(weight * loss for weight, loss in zip(weights, losses))),
            "n_answer_tokens": int(min(token_counts)),
            "gt_delta_forget_ce": gt_delta,
            "completeness_residual": float(scores["upqi_signed"] - gt_delta),
        }
        output.append(row)
        computed += 1
        checkpoint(output)
        LOGGER.info(
            "UPQI-star %d/%d %s signed=%g abs=%g residual=%g",
            index + 1,
            len(target_layers),
            name,
            row["upqi_signed"],
            row["upqi_abs"],
            row["completeness_residual"],
        )
        if stop_after_layers and computed >= stop_after_layers:
            raise SystemExit(f"Stopped after {computed} layers for resume proof")
    by_name = {str(row["layer"]): dict(row) for row in output}
    return [by_name[name] for name in target_layers]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    start = time.time()
    out_path = Path(args.output_dir) / "upqi_star_v42.json"
    partial_path = out_path.with_suffix(".json.partial")
    reject_legacy_result_path(out_path)
    source_info = require_source_lock(
        script_path=__file__, source_lock_path=args.source_lock_path, expected_script_sha256=args.expected_script_sha256
    )
    manifest = load_json(args.manifest_path)
    assert_forget_manifest_ready(manifest)
    if len(manifest.get("samples", [])) < args.n_samples:
        raise ValueError(f"Forget manifest has fewer than requested {args.n_samples} samples")
    samples = [row for row in manifest["samples"][: args.n_samples] if isinstance(row, Mapping)]
    if len(samples) != args.n_samples:
        raise ValueError("Forget sample subset contains invalid rows")
    calibration_manifest = None
    calibration_samples: list[Mapping[str, Any]] = []
    if args.quantizer == "gptq_int4":
        if not args.calibration_manifest_path:
            raise SystemExit("gptq_int4 requires --calibration_manifest_path")
        calibration_manifest = load_json(args.calibration_manifest_path)
        backend_samples = calibration_manifest.get("samples", [])
        calibration_samples = [row for row in backend_samples if isinstance(row, Mapping)]
        if len(calibration_samples) != args.n_calibration:
            raise ValueError("Calibration manifest count mismatch")
    gt_obj, gt_layers = gt_layer_map(args.gt_path)
    nodes_array, weights_array = gauss_legendre_unit_interval(args.n_quadrature)
    nodes = [float(value) for value in nodes_array]
    weights = [float(value) for value in weights_array]
    gt_ref = reference(args.gt_path)
    legacy_ref = reference(args.legacy_upqi_path)

    requested_layer_indices = None
    if args.layer_indices:
        try:
            requested_layer_indices = [int(value) for value in args.layer_indices.split(",")]
        except ValueError as exc:
            raise SystemExit("--layer_indices must be comma-separated integers") from exc
        if not requested_layer_indices:
            raise SystemExit("--layer_indices cannot be empty")
        if requested_layer_indices != sorted(set(requested_layer_indices)):
            raise SystemExit("--layer_indices must be unique and strictly increasing")
    if args.max_layers and requested_layer_indices is not None:
        raise SystemExit("--max_layers and --layer_indices are mutually exclusive")
    if args.batch_size <= 0 or args.max_length <= 0:
        raise SystemExit("--batch_size and --max_length must be positive")

    resume_contract = {
        "model_dir": str(Path(args.model_dir)),
        "model_tag": args.model_tag,
        "algo": args.algo,
        "seed": str(args.seed),
        "dataset": args.dataset,
        "quantizer": args.quantizer,
        "dtype": args.dtype,
        "n_samples": args.n_samples,
        "n_calibration": args.n_calibration if args.quantizer == "gptq_int4" else None,
        "n_quadrature": args.n_quadrature,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_layers": args.max_layers,
        "requested_layer_indices": requested_layer_indices,
        "sample_manifest_sha256": manifest.get("manifest_sha256"),
        "calibration_manifest_sha256": None
        if calibration_manifest is None
        else calibration_manifest.get("manifest_sha256"),
        "ground_truth_sha256": gt_ref["sha256"],
        "legacy_upqi_sha256": legacy_ref["sha256"],
        "source_lock_sha256": source_info.get("source_lock_sha256"),
    }

    def build(rows: Sequence[Mapping[str, Any]], n_layers: int) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "schema_version": UPQI_STAR_SCHEMA,
            "stage": "upqi_star_v42",
            "result_root_policy": "v4_only_no_v2_v3_reuse",
            "method": "layerwise_one_at_a_time_integrated_gradients_along_quantization_displacement",
            "quadrature_contract_id": GAUSS_LEGENDRE_CONTRACT,
            "upqi_signed_contract_id": UPQI_SIGNED_CONTRACT,
            "upqi_abs_contract_id": UPQI_ABS_CONTRACT,
            "model_dir": args.model_dir,
            "model_tag": args.model_tag,
            "algo": args.algo,
            "seed": str(args.seed),
            "dataset": args.dataset,
            "quantizer": args.quantizer,
            "dtype": args.dtype,
            "n_samples": args.n_samples,
            "n_calibration": args.n_calibration if args.quantizer == "gptq_int4" else None,
            "n_quadrature": args.n_quadrature,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "quadrature_nodes": nodes,
            "quadrature_weights": weights,
            "n_layers": n_layers,
            "n_layers_completed": len(rows),
            "target_layer_selection": layer_selection,
            "loss_contract": loss_contract(args.n_samples, manifest),
            "sample_manifest": {
                "path": args.manifest_path,
                "manifest_sha256": manifest.get("manifest_sha256"),
                "subset_policy": "first_n_from_locked_core128_manifest",
            },
            "calibration_manifest": None if calibration_manifest is None else {
                "path": args.calibration_manifest_path,
                "manifest_sha256": calibration_manifest.get("manifest_sha256"),
            },
            "ground_truth_reference": gt_ref,
            "legacy_upqi_v4_reference": legacy_ref,
            "resume_contract": resume_contract,
            "per_layer": [dict(row) for row in rows],
            "timing": {"total_seconds": float(time.time() - start)},
            "provenance": provenance(
                script_path=__file__, argv=sys.argv, source_info=source_info, repo_root=REPO_ROOT,
                extra={"upqi_star_schema": UPQI_STAR_SCHEMA},
            ),
        }
        obj["validation"] = validate_obj(obj, out_path)
        return obj

    if out_path.exists() and not args.force:
        report = validate_obj(load_json(out_path), out_path)
        if report["passed"]:
            LOGGER.info("Existing UPQI-star result is valid: %s", out_path)
            return
        raise SystemExit(f"Existing UPQI-star output invalid; quarantine or use --force: {report}")
    if args.dry_run:
        LOGGER.info("dry-run contract pass before model load: %s", out_path)
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=legacy.model_dtype(args.dtype), device_map="auto", trust_remote_code=True
    )
    device = next(model.parameters()).device
    all_target_layers = [name for name, _ in backend.get_weight_layers(model)]
    selected_global_indices = list(range(len(all_target_layers)))
    if args.max_layers:
        selected_global_indices = selected_global_indices[: args.max_layers]
    elif requested_layer_indices is not None:
        bad = [index for index in requested_layer_indices if index < 0 or index >= len(all_target_layers)]
        if bad:
            raise SystemExit(
                f"--layer_indices outside canonical range 0..{len(all_target_layers) - 1}: {bad}"
            )
        selected_global_indices = requested_layer_indices
    target_layers = [all_target_layers[index] for index in selected_global_indices]
    global_layer_indices = {
        name: index for index, name in zip(selected_global_indices, target_layers)
    }
    layer_selection = {
        "policy": "all" if len(target_layers) == len(all_target_layers) else (
            "explicit_global_indices" if requested_layer_indices is not None else "canonical_prefix"
        ),
        "global_indices": selected_global_indices,
        "n_total_available_layers": len(all_target_layers),
    }
    resume: list[dict[str, Any]] = []
    if partial_path.exists() and not args.force:
        partial = load_json(partial_path)
        if partial.get("schema_version") == UPQI_STAR_SCHEMA and partial.get("resume_contract") == resume_contract:
            resume = [dict(row) for row in partial.get("per_layer", []) if isinstance(row, Mapping)]
            LOGGER.info("Resuming %d UPQI-star layers", len(resume))
        else:
            raise SystemExit(
                f"Partial UPQI-star result does not match the exact resume contract: {partial_path}. "
                "Quarantine it or rerun with --force."
            )
    remaining = [name for name in target_layers if name not in {row.get("layer") for row in resume}]
    hessians = None
    if args.quantizer == "gptq_int4" and remaining:
        hessians = backend.collect_layer_hessians(
            model, tokenizer, calibration_samples, remaining, device=device, max_samples=args.n_calibration
        )

    def checkpoint(rows: Sequence[Mapping[str, Any]]) -> None:
        obj = build(rows, len(target_layers))
        obj["validation"] = {"passed": False, "errors": ["partial_incomplete"], "warnings": []}
        save_json(partial_path, obj)

    rows = compute_layers(
        model,
        tokenizer,
        samples,
        target_layers,
        global_layer_indices,
        gt_layers,
        quantizer=args.quantizer,
        hessians=hessians,
        nodes=nodes,
        weights=weights,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        resume=resume,
        checkpoint=checkpoint,
        stop_after_layers=args.stop_after_layers,
    )
    result = build(rows, len(target_layers))
    if not result["validation"]["passed"]:
        save_json(out_path.with_suffix(".json.failed"), result)
        raise SystemExit(f"UPQI-star validation failed: {result['validation']}")
    save_json(out_path, result)
    partial_path.unlink(missing_ok=True)
    LOGGER.info("UPQI-star v4.2 done: %s", out_path)


if __name__ == "__main__":
    main()
