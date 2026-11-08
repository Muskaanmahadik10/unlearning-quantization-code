#!/usr/bin/env python3
"""Compute exact endpoint GQDP and reviewer-safe comparators for one v4 cell."""

from __future__ import annotations

import argparse
import gc
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import gate2_v4_backend as backend  # noqa: E402
from gate2_metric_v42_math import GQDP_CONTRACT, METRIC_SUITE_SCHEMA, endpoint_metrics  # noqa: E402
from gate2_v4_contract import (  # noqa: E402
    assert_forget_manifest_ready,
    assert_manifest_sha,
    load_json,
    loss_contract,
    provenance,
    reject_legacy_result_path,
    require_source_lock,
    save_json,
    sha256_file,
)
from gate2_v4_dataset import AnswerOnlyQADatasetV4, collate_answer_only  # noqa: E402
from validate_gate2_metric_suite_v42 import validate_obj  # noqa: E402


LOGGER = logging.getLogger("gate2_metric_suite_v42")


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
    parser.add_argument("--upqi_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--retain_manifest_path", required=True)
    parser.add_argument("--calibration_manifest_path")
    parser.add_argument("--original_weights", required=True)
    parser.add_argument("--source_lock_path", required=True)
    parser.add_argument("--expected_script_sha256", required=True)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--n_retain", type=int, default=128)
    parser.add_argument("--n_calibration", type=int, default=128)
    parser.add_argument("--fisher_forget_samples", type=int, default=0)
    parser.add_argument("--fisher_retain_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max_layers", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def model_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def manifest_samples(path: str, expected: int, *, forget_contract: bool) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    manifest = load_json(path)
    if forget_contract:
        assert_forget_manifest_ready(manifest, n_samples=expected)
    else:
        assert_manifest_sha(manifest)
        if len(manifest.get("samples", [])) != expected:
            raise ValueError(f"Manifest {path} has {len(manifest.get('samples', []))} samples; expected {expected}")
    samples = [row for row in manifest.get("samples", []) if isinstance(row, Mapping)]
    if len(samples) != expected:
        raise ValueError(f"Manifest {path} has {len(samples)} usable samples; expected {expected}")
    return manifest, samples


def make_loader(samples: Sequence[Mapping[str, Any]], tokenizer: Any, *, batch_size: int, max_length: int) -> DataLoader:
    dataset = AnswerOnlyQADatasetV4(samples, tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_answer_only(rows, tokenizer.pad_token_id),
    )


def mean_gradients(
    model: Any,
    tokenizer: Any,
    samples: Sequence[Mapping[str, Any]],
    target_layers: Sequence[str],
    *,
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], int]:
    device = next(model.parameters()).device
    loader = make_loader(samples, tokenizer, batch_size=batch_size, max_length=max_length)
    model.eval()
    model.zero_grad(set_to_none=True)
    token_total = 0
    for batch_index, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        tokens = int((shift_labels != -100).sum().item())
        if tokens <= 0:
            raise RuntimeError("Metric-suite batch has zero answer tokens")
        loss_sum = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        loss_sum.backward()
        token_total += tokens
        if batch_index % 16 == 0:
            LOGGER.info("mean-gradient batches=%d answer_tokens=%d", batch_index, token_total)
    if token_total <= 0:
        raise RuntimeError("No answer tokens for mean gradient")
    param_dict = dict(model.named_parameters())
    gradients: dict[str, torch.Tensor] = {}
    for name in target_layers:
        grad = param_dict[name].grad
        if grad is None:
            raise RuntimeError(f"Missing gradient for {name}")
        value = (grad.detach().float().cpu() / float(token_total)).contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite mean gradient for {name}")
        gradients[name] = value
    model.zero_grad(set_to_none=True)
    return gradients, token_total


def load_original_state(path: str) -> Mapping[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, Mapping) and isinstance(state.get("state_dict"), Mapping):
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("original_weights is not a state mapping")
    return state


def per_sample_fisher_and_fqp(
    model: Any,
    tokenizer: Any,
    samples: Sequence[Mapping[str, Any]],
    target_layers: Sequence[str],
    displacement: Mapping[str, torch.Tensor],
    *,
    max_length: int,
) -> tuple[dict[str, float], dict[str, float]]:
    device = next(model.parameters()).device
    loader = make_loader(samples, tokenizer, batch_size=1, max_length=max_length)
    params = dict(model.named_parameters())
    eps_sq = {name: displacement[name].to(device=params[name].device, dtype=torch.float32).square() for name in target_layers}
    fisher = {name: 0.0 for name in target_layers}
    fqp = {name: 0.0 for name in target_layers}
    model.eval()
    for sample_index, batch in enumerate(loader, start=1):
        model.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        tokens = int((shift_labels != -100).sum().item())
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        loss.backward()
        for name in target_layers:
            grad = params[name].grad
            if grad is None:
                raise RuntimeError(f"Missing per-sample gradient for {name}")
            grad_sq = grad.detach().float().square()
            fisher[name] += float(grad_sq.sum().item())
            fqp[name] += float((grad_sq * eps_sq[name]).sum().item())
        if sample_index % 16 == 0:
            LOGGER.info("per-sample Fisher/FQP %d/%d", sample_index, len(samples))
    count = float(len(samples))
    model.zero_grad(set_to_none=True)
    del eps_sq
    torch.cuda.empty_cache()
    return ({name: value / count for name, value in fisher.items()}, {name: value / count for name, value in fqp.items()})


def per_sample_fisher_scalar(
    model: Any,
    tokenizer: Any,
    samples: Sequence[Mapping[str, Any]],
    target_layers: Sequence[str],
    *,
    max_length: int,
) -> dict[str, float]:
    device = next(model.parameters()).device
    loader = make_loader(samples, tokenizer, batch_size=1, max_length=max_length)
    params = dict(model.named_parameters())
    fisher = {name: 0.0 for name in target_layers}
    model.eval()
    for sample_index, batch in enumerate(loader, start=1):
        model.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        output.loss.backward()
        for name in target_layers:
            grad = params[name].grad
            if grad is None:
                raise RuntimeError(f"Missing retain Fisher gradient for {name}")
            fisher[name] += float(grad.detach().float().square().sum().item())
        if sample_index % 16 == 0:
            LOGGER.info("retain per-sample Fisher %d/%d", sample_index, len(samples))
    count = float(len(samples))
    model.zero_grad(set_to_none=True)
    return {name: value / count for name, value in fisher.items()}


def reference_metadata(path: str) -> dict[str, Any]:
    obj = load_json(path)
    return {
        "path": str(Path(path)),
        "sha256": sha256_file(path),
        "schema_version": obj.get("schema_version"),
        "source_lock_sha256": obj.get("provenance", {}).get("source_lock_sha256"),
        "sample_manifest_sha256": obj.get("sample_manifest", {}).get("manifest_sha256"),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    start = time.time()
    out_path = Path(args.output_dir) / "metrics_v42.json"
    reject_legacy_result_path(out_path)
    source_info = require_source_lock(
        script_path=__file__,
        source_lock_path=args.source_lock_path,
        expected_script_sha256=args.expected_script_sha256,
    )
    forget_manifest, forget_samples = manifest_samples(args.manifest_path, args.n_samples, forget_contract=True)
    retain_manifest, retain_samples = manifest_samples(args.retain_manifest_path, args.n_retain, forget_contract=False)
    calibration_manifest: Optional[Mapping[str, Any]] = None
    calibration_samples: list[Mapping[str, Any]] = []
    if args.quantizer == "gptq_int4":
        if not args.calibration_manifest_path:
            raise SystemExit("gptq_int4 requires --calibration_manifest_path")
        calibration_manifest, calibration_samples = manifest_samples(
            args.calibration_manifest_path, args.n_calibration, forget_contract=False
        )
    if out_path.exists() and not args.force:
        report = validate_obj(load_json(out_path), out_path)
        if report["passed"]:
            LOGGER.info("Existing metric suite is valid: %s", out_path)
            return
        raise SystemExit(f"Existing metric suite is invalid; quarantine or use --force: {report}")
    if args.dry_run:
        LOGGER.info("dry-run contract pass before model load: %s", out_path)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=model_dtype(args.dtype),
        device_map="auto",
        trust_remote_code=True,
    )
    device = next(model.parameters()).device
    target_layers = [name for name, _ in backend.get_weight_layers(model)]
    if args.max_layers:
        target_layers = target_layers[: args.max_layers]
    params = dict(model.named_parameters())
    LOGGER.info("Computing endpoint mean forget gradients for %d layers", len(target_layers))
    forget_gradients, forget_tokens = mean_gradients(
        model, tokenizer, forget_samples, target_layers, batch_size=args.batch_size, max_length=args.max_length
    )
    timing = {"mean_forget_gradient_seconds": float(time.time() - start)}

    hessians = None
    hessian_start = time.time()
    if args.quantizer == "gptq_int4":
        LOGGER.info("Collecting GPTQ Hessians from %d locked calibration samples", len(calibration_samples))
        hessians = backend.collect_layer_hessians(
            model, tokenizer, calibration_samples, target_layers, device=device, max_samples=args.n_calibration
        )
    timing["gptq_hessian_seconds"] = float(time.time() - hessian_start)
    original_state = load_original_state(args.original_weights)
    displacement: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    endpoint_start = time.time()
    n_blocks = max(backend.extract_block_idx(name) for name in target_layers) + 1
    for index, name in enumerate(target_layers):
        param = params[name]
        hessian = hessians[name].to(param.device) if hessians and name in hessians else None
        quantized = backend.quantize_dequantize(param.data, args.quantizer, hessian=hessian)
        eps = (quantized.float() - param.data.float()).detach().cpu().contiguous()
        displacement[name] = eps
        if name not in original_state:
            raise KeyError(f"{name} missing from original_weights")
        original = original_state[name].to(device=param.device, dtype=param.data.dtype)
        original_quantized = backend.quantize_dequantize(original, args.quantizer, hessian=hessian)
        bcr_true = float((original_quantized != quantized).float().mean().item())
        metrics = endpoint_metrics(param.data, forget_gradients[name], eps)
        metrics.update({
            "layer": name,
            "layer_index": index,
            "block_idx": backend.extract_block_idx(name),
            "submodule": backend.classify_submodule(name),
            "layer_position": float(backend.extract_block_idx(name) / max(n_blocks - 1, 1)),
            "bcr_true": bcr_true,
        })
        rows.append(metrics)
        if (index + 1) % 32 == 0:
            LOGGER.info("endpoint metrics %d/%d", index + 1, len(target_layers))
        del original, original_quantized, quantized
        if hessian is not None:
            del hessian
    del original_state, hessians
    timing["endpoint_metric_seconds"] = float(time.time() - endpoint_start)
    gc.collect()
    torch.cuda.empty_cache()

    fisher_forget: dict[str, float] = {}
    fqp_forget: dict[str, float] = {}
    fisher_forget_start = time.time()
    if args.fisher_forget_samples > 0:
        fisher_samples = forget_samples[: args.fisher_forget_samples]
        fisher_forget, fqp_forget = per_sample_fisher_and_fqp(
            model, tokenizer, fisher_samples, target_layers, displacement, max_length=args.max_length
        )
    timing["fisher_forget_seconds"] = float(time.time() - fisher_forget_start)
    fisher_retain: dict[str, float] = {}
    fisher_retain_start = time.time()
    if args.fisher_retain_samples > 0:
        fisher_retain = per_sample_fisher_scalar(
            model,
            tokenizer,
            retain_samples[: args.fisher_retain_samples],
            target_layers,
            max_length=args.max_length,
        )
    timing["fisher_retain_seconds"] = float(time.time() - fisher_retain_start)
    for row in rows:
        name = row["layer"]
        row["fisher_forget_ps"] = fisher_forget.get(name)
        row["fqp_forget"] = fqp_forget.get(name)
        row["fisher_retain_ps"] = fisher_retain.get(name)

    result: dict[str, Any] = {
        "schema_version": METRIC_SUITE_SCHEMA,
        "stage": "metric_suite_v42",
        "result_root_policy": "v4_only_no_v2_v3_reuse",
        "model_dir": args.model_dir,
        "model_tag": args.model_tag,
        "algo": args.algo,
        "seed": str(args.seed),
        "dataset": args.dataset,
        "quantizer": args.quantizer,
        "dtype": args.dtype,
        "n_layers": len(target_layers),
        "n_samples": args.n_samples,
        "n_retain": args.n_retain,
        "n_calibration": args.n_calibration if args.quantizer == "gptq_int4" else None,
        "fisher_forget_samples": min(args.fisher_forget_samples, len(forget_samples)),
        "fisher_retain_samples": min(args.fisher_retain_samples, len(retain_samples)),
        "gradient_answer_tokens": forget_tokens,
        "gradient_aggregation": "answer_token_weighted_mean_gradient",
        "gqdp_contract_id": GQDP_CONTRACT,
        "bcr_contract": "fraction_exact_dequantized_values_changed_original_vs_unlearned_shared_quantizer_context",
        "fisher_contract": "empirical_per_sample_answer_ce_gradient_square" if args.fisher_forget_samples else None,
        "fqp_contract": "sum_empirical_fisher_diagonal_times_quantization_displacement_squared" if args.fisher_forget_samples else None,
        "loss_contract": loss_contract(args.n_samples, forget_manifest),
        "sample_manifest": {"path": args.manifest_path, "manifest_sha256": forget_manifest.get("manifest_sha256")},
        "retain_manifest": {"path": args.retain_manifest_path, "manifest_sha256": retain_manifest.get("manifest_sha256")},
        "calibration_manifest": None if calibration_manifest is None else {
            "path": args.calibration_manifest_path,
            "manifest_sha256": calibration_manifest.get("manifest_sha256"),
        },
        "original_weights": {"path": args.original_weights, "sha256": sha256_file(args.original_weights)},
        "ground_truth_reference": reference_metadata(args.gt_path),
        "legacy_upqi_v4_reference": reference_metadata(args.upqi_path),
        "per_layer": rows,
        "timing": {**timing, "total_seconds": float(time.time() - start)},
        "provenance": provenance(
            script_path=__file__, argv=sys.argv, source_info=source_info, repo_root=REPO_ROOT,
            extra={"metric_suite_schema": METRIC_SUITE_SCHEMA},
        ),
    }
    result["validation"] = validate_obj(result, out_path)
    if not result["validation"]["passed"]:
        save_json(out_path.with_suffix(".json.failed"), result)
        raise SystemExit(f"Metric suite validation failed: {result['validation']}")
    save_json(out_path, result)
    LOGGER.info("Metric suite v4.2 done: %s", out_path)


if __name__ == "__main__":
    main()
