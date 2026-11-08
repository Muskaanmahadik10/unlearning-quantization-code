#!/usr/bin/env python3
"""Gate 2 v4 ground truth with hard truth-ratio and source-lock contracts."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gate2_v4_contract import (  # noqa: E402
    GT_SCHEMA,
    assert_forget_manifest_ready,
    canonical_json,
    finite_number,
    load_json,
    logmeanexp,
    loss_contract,
    provenance,
    reject_legacy_result_path,
    require_source_lock,
    save_json,
    sha256_text,
)
from gate2_v4_dataset import encode_answer_only, encode_qa_preserve_answer, normalize_answer, normalize_question  # noqa: E402
from validate_gt_v4 import validate_obj  # noqa: E402


def load_v2_backend() -> Any:
    for path in (
        SCRIPT_DIR / "gate2_v4_backend.py",
        SCRIPT_DIR / "run_gate2_ground_truth_v2.py",
        REPO_ROOT / "scripts" / "pipeline" / "run_gate2_ground_truth_v2.py",
    ):
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("gate2_v4_quant_backend", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ModuleNotFoundError("No Gate 2 quantization backend found in v4 or legacy locations")


gt_v1 = load_v2_backend()


LOGGER = logging.getLogger("gate2_gt_v4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gate 2 GT v4 for one cell")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--dataset", required=True, choices=["tofu", "muse_news"])
    parser.add_argument("--quantizer", required=True, choices=["nf4", "gptq_int4", "rtn_int8"])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--calibration_manifest_path")
    parser.add_argument("--source_lock_path", required=True)
    parser.add_argument("--expected_script_sha256", required=True)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--n_calibration", type=int, default=128)
    parser.add_argument("--kl_subset_size", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument("--max_layers", type=int, default=None)
    parser.add_argument("--stop_after_layers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def model_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.unsqueeze(0).to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask", "labels"}}


def qa_inputs(model_tokenizer: Any, sample: Mapping[str, Any], device: torch.device, max_length: int) -> Dict[str, torch.Tensor]:
    encoded = encode_qa_preserve_answer(
        model_tokenizer,
        normalize_question(sample),
        normalize_answer(sample),
        max_length,
    )
    return {
        "input_ids": encoded["input_ids"].unsqueeze(0).to(device),
        "attention_mask": encoded["attention_mask"].unsqueeze(0).to(device),
    }


@torch.no_grad()
def answer_stats(model: Any, tokenizer: Any, question: str, answer: str, device: torch.device, max_length: int) -> Dict[str, Any]:
    encoded = encode_answer_only(tokenizer, question, answer, max_length)
    labels = encoded["labels"].unsqueeze(0).to(device)
    inputs = to_device(encoded, device)
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits.float()
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    answer_mask = shift_labels != -100
    if not bool(answer_mask.any()):
        raise ValueError("zero answer tokens after causal shift")
    gather_labels = shift_labels.clone()
    gather_labels[gather_labels == -100] = 0
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, gather_labels.unsqueeze(-1)).squeeze(-1)
    answer_log_probs = token_log_probs[answer_mask]
    mean_logprob = float(answer_log_probs.mean().item())
    ce = float((-answer_log_probs).mean().item())
    return {
        "answer_logprob": mean_logprob,
        "answer_ce": ce,
        "n_answer_tokens": int(answer_log_probs.numel()),
    }


@torch.no_grad()
def compute_forget_eval_v4(
    model: Any,
    tokenizer: Any,
    samples: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    max_length: int,
    store_per_sample: bool = False,
) -> Dict[str, Any]:
    answer_logprobs: List[float] = []
    answer_ces: List[float] = []
    tr_logs: List[float] = []
    per_sample: List[Dict[str, Any]] = []
    bad_sample_count = 0
    nonfinite_answer_count = 0
    nonfinite_perturbation_count = 0
    min_answer_tokens: Optional[int] = None
    min_perturbations: Optional[int] = None
    for idx, sample in enumerate(samples):
        try:
            question = normalize_question(sample)
            answer = normalize_answer(sample)
            perturbations = sample.get("perturbations")
            if not isinstance(perturbations, list) or not perturbations:
                raise ValueError("missing perturbations")
            actual = answer_stats(model, tokenizer, question, answer, device, max_length)
            if not finite_number(actual.get("answer_logprob")) or not finite_number(actual.get("answer_ce")):
                nonfinite_answer_count += 1
                raise ValueError("non-finite answer stats")
            perturb_lps = [
                answer_stats(model, tokenizer, question, str(row["answer"]), device, max_length)["answer_logprob"]
                for row in perturbations
                if isinstance(row, Mapping) and str(row.get("answer", "")).strip()
            ]
            if not perturb_lps:
                raise ValueError("no valid perturbation logprobs")
            if not all(finite_number(value) for value in perturb_lps):
                nonfinite_perturbation_count += 1
                raise ValueError("non-finite perturbation stats")
            mean_perturbation_logprob = logmeanexp(perturb_lps)
            tr_log = mean_perturbation_logprob - float(actual["answer_logprob"])
            if not math.isfinite(tr_log):
                raise ValueError("non-finite truth-ratio log")
            answer_logprobs.append(float(actual["answer_logprob"]))
            answer_ces.append(float(actual["answer_ce"]))
            tr_logs.append(float(tr_log))
            min_answer_tokens = int(actual["n_answer_tokens"]) if min_answer_tokens is None else min(min_answer_tokens, int(actual["n_answer_tokens"]))
            min_perturbations = len(perturb_lps) if min_perturbations is None else min(min_perturbations, len(perturb_lps))
            if store_per_sample:
                per_sample.append({
                    "idx": idx,
                    "sample_hash": sample.get("sample_hash"),
                    "answer_logprob": float(actual["answer_logprob"]),
                    "answer_ce": float(actual["answer_ce"]),
                    "mean_perturbation_logprob": float(mean_perturbation_logprob),
                    "truth_ratio_log": float(tr_log),
                    "n_perturbations": len(perturb_lps),
                    "n_answer_tokens": int(actual["n_answer_tokens"]),
                    "reportable": True,
                })
        except Exception as exc:
            bad_sample_count += 1
            if store_per_sample:
                per_sample.append({"idx": idx, "sample_hash": sample.get("sample_hash"), "reportable": False, "error": str(exc)})
    if bad_sample_count:
        raise ValueError(f"GT v4 truth-ratio contract failed for {bad_sample_count}/{len(samples)} samples")
    if len(tr_logs) != len(samples):
        raise ValueError("truth-ratio count mismatch")
    return {
        "answer_logprob": float(np.mean(answer_logprobs)),
        "answer_ce": float(np.mean(answer_ces)),
        "truth_ratio_log": float(logmeanexp(tr_logs)),
        "truth_ratio_log_mean_sample": float(np.mean(tr_logs)),
        "truth_ratio_reportable": True,
        "truth_ratio_n": len(tr_logs),
        "bad_sample_count": bad_sample_count,
        "n_samples": len(samples),
        "n_valid_answer_samples": len(answer_logprobs),
        "truth_ratio_sentry": {
            "all_samples_reportable": bad_sample_count == 0 and len(tr_logs) == len(samples),
            "n_samples": len(samples),
            "n_reportable": len(tr_logs),
            "min_answer_tokens": min_answer_tokens,
            "min_perturbations": min_perturbations,
            "nonfinite_answer_logprob_count": nonfinite_answer_count,
            "nonfinite_perturbation_logprob_count": nonfinite_perturbation_count,
        },
        "per_sample": per_sample if store_per_sample else [],
    }


def manifest_sample_hashes(manifest: Mapping[str, Any]) -> List[str]:
    return [str(sample.get("sample_hash", "")) for sample in manifest.get("samples", []) if isinstance(sample, Mapping)]


def sample_hashes_sha256(hashes: Sequence[str]) -> str:
    return sha256_text(canonical_json(list(hashes)))


def bf16_truth_ratio_evidence(bf16_metrics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in bf16_metrics.get("per_sample", []) if isinstance(bf16_metrics.get("per_sample"), list) else []:
        if not isinstance(row, Mapping):
            continue
        rows.append({
            "idx": int(row.get("idx", len(rows))),
            "sample_hash": row.get("sample_hash"),
            "answer_token_count": row.get("n_answer_tokens"),
            "n_perturbations": row.get("n_perturbations"),
            "answer_logprob_bf16": row.get("answer_logprob"),
            "mean_perturbation_logprob_bf16": row.get("mean_perturbation_logprob"),
            "truth_ratio_log_bf16": row.get("truth_ratio_log"),
            "reportable": row.get("reportable") is True,
        })
    return rows


@torch.no_grad()
def collect_reference_logits(model: Any, tokenizer: Any, samples: Sequence[Mapping[str, Any]], device: torch.device, max_length: int) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for sample in samples:
        inputs = qa_inputs(tokenizer, sample, device, max_length)
        out.append(model(**inputs).logits.detach().cpu())
    return out


@torch.no_grad()
def kl_ref_to_quant(model: Any, tokenizer: Any, sample: Mapping[str, Any], ref_logits: torch.Tensor, device: torch.device, max_length: int) -> float:
    inputs = qa_inputs(tokenizer, sample, device, max_length)
    quant_logits = model(**inputs).logits
    ref_log_probs = F.log_softmax(ref_logits.to(device).float(), dim=-1)
    quant_log_probs = F.log_softmax(quant_logits.float(), dim=-1)
    return float(F.kl_div(quant_log_probs, ref_log_probs, reduction="none", log_target=True).sum(dim=-1).mean().item())


def build_result(
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    calibration_manifest: Optional[Mapping[str, Any]],
    source_info: Mapping[str, Any],
    target_layers: Sequence[str],
    total_model_layers: int,
    bf16_metrics: Mapping[str, Any],
    layers: Mapping[str, Mapping[str, Any]],
    total_seconds: float,
) -> Dict[str, Any]:
    sample_hashes = manifest_sample_hashes(manifest)
    bf16_evidence = bf16_truth_ratio_evidence(bf16_metrics)
    obj: Dict[str, Any] = {
        "schema_version": GT_SCHEMA,
        "stage": "gt_v4",
        "result_root_policy": "v4_only_no_v2_v3_reuse",
        "model_dir": args.model_dir,
        "model_tag": args.model_tag,
        "algo": args.algo,
        "seed": str(args.seed),
        "dataset": args.dataset,
        "quantizer": args.quantizer,
        "dtype": args.dtype,
        "n_samples": int(args.n_samples),
        "n_calibration": int(args.n_calibration) if args.quantizer == "gptq_int4" else None,
        "target": "delta_forget_ce",
        "secondary_targets": ["delta_answer_logprob", "delta_truth_ratio_log", "KL_forget_clamped"],
        "truth_ratio_policy": "hard_required",
        "n_layers": len(target_layers),
        "n_layers_completed": len(layers),
        "total_model_layers": total_model_layers,
        "target_layer_names": list(target_layers),
        "loss_contract": loss_contract(args.n_samples, manifest),
        "sample_manifest": {
            "path": str(Path(args.manifest_path)),
            "schema_version": manifest.get("schema_version"),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "dataset": manifest.get("dataset"),
            "split": manifest.get("split"),
            "purpose": manifest.get("purpose"),
            "selection_contract_id": manifest.get("selection_contract_id"),
            "sample_hashes_sha256": sample_hashes_sha256(sample_hashes),
        },
        "calibration_manifest": None if calibration_manifest is None else {
            "path": str(Path(args.calibration_manifest_path)),
            "schema_version": calibration_manifest.get("schema_version"),
            "manifest_sha256": calibration_manifest.get("manifest_sha256"),
            "dataset": calibration_manifest.get("dataset"),
            "split": calibration_manifest.get("split"),
            "purpose": calibration_manifest.get("purpose"),
            "selection_contract_id": calibration_manifest.get("selection_contract_id"),
        },
        "bf16_baseline": dict(bf16_metrics),
        "truth_ratio_sentry": {
            **dict(bf16_metrics.get("truth_ratio_sentry", {})),
            "sample_hashes_sha256": sample_hashes_sha256([str(row.get("sample_hash", "")) for row in bf16_evidence]),
        },
        "bf16_per_sample_truth_ratio_evidence": bf16_evidence,
        "layers": OrderedDict((name, dict(layers[name])) for name in target_layers if name in layers),
        "timing": {"total_seconds": float(total_seconds)},
        "methodology": {
            "answer_mask": "first unmasked label is first answer token; prefix fully masked",
            "truth_ratio": "required; logmeanexp(perturbation_answer_logprob) - answer_logprob",
            "delta_forget_ce": "CE_quantized - CE_bf16 on locked forget manifest",
            "delta_answer_logprob": "answer_logprob_quantized - answer_logprob_bf16",
            "single_layer_intervention": "one layer quantized at a time; all other weights restored",
        },
        "provenance": provenance(
            script_path=__file__,
            argv=sys.argv,
            source_info=source_info,
            repo_root=REPO_ROOT,
            extra={"gt_schema": GT_SCHEMA},
        ),
    }
    obj["validation"] = validate_obj(obj, Path(args.output_dir) / "ground_truth_v4.json")
    return obj


def existing_complete_ok(path: Path, args: argparse.Namespace, manifest: Mapping[str, Any], source_info: Mapping[str, Any]) -> bool:
    if not path.exists() or args.force:
        return False
    try:
        obj = load_json(path)
        report = validate_obj(obj, path)
    except Exception:
        return False
    return (
        report["passed"]
        and obj.get("sample_manifest", {}).get("manifest_sha256") == manifest.get("manifest_sha256")
        and obj.get("provenance", {}).get("source_lock_sha256") == source_info.get("source_lock_sha256")
        and obj.get("provenance", {}).get("script_sha256") == source_info.get("script_sha256")
        and obj.get("model_tag") == args.model_tag
        and obj.get("algo") == args.algo
        and str(obj.get("seed")) == str(args.seed)
        and obj.get("quantizer") == args.quantizer
    )


def load_calibration_data(args: argparse.Namespace, calibration_manifest: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    try:
        retain_data = gt_v1.load_json_data(args.data_dir, args.dataset, "retain")
        calibration_data = gt_v1.select_manifest_samples(calibration_manifest, retain_data)
        if len(calibration_data) == args.n_calibration:
            return calibration_data
        LOGGER.warning("Raw retain selection returned %d samples, expected %d", len(calibration_data), args.n_calibration)
    except (Exception, SystemExit) as exc:
        LOGGER.warning("Falling back to calibration manifest samples because raw retain load failed: %s", exc)
    samples = calibration_manifest.get("samples")
    if not isinstance(samples, list):
        raise SystemExit("Calibration manifest has no samples")
    calibration_data = [row for row in samples if isinstance(row, Mapping)]
    if len(calibration_data) != args.n_calibration:
        raise SystemExit(f"Calibration manifest contains {len(calibration_data)} usable samples, expected {args.n_calibration}")
    return calibration_data


def partial_layers_ok(path: Path, args: argparse.Namespace, manifest: Mapping[str, Any], source_info: Mapping[str, Any], target_layers: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    if not path.exists() or args.force:
        return {}
    try:
        obj = load_json(path)
    except Exception as exc:
        LOGGER.warning("Ignoring unreadable GT v4 partial %s: %s", path, exc)
        return {}
    if obj.get("schema_version") != GT_SCHEMA or obj.get("stage") != "gt_v4":
        return {}
    if obj.get("sample_manifest", {}).get("manifest_sha256") != manifest.get("manifest_sha256"):
        return {}
    if obj.get("provenance", {}).get("source_lock_sha256") != source_info.get("source_lock_sha256"):
        return {}
    if obj.get("provenance", {}).get("script_sha256") != source_info.get("script_sha256"):
        return {}
    layers = obj.get("layers")
    if not isinstance(layers, Mapping):
        return {}
    target_set = set(target_layers)
    out: Dict[str, Dict[str, Any]] = {}
    for name, entry in layers.items():
        if name in target_set and isinstance(entry, Mapping):
            if entry.get("truth_ratio_reportable_bf16") is True and entry.get("truth_ratio_reportable_quantized") is True:
                out[name] = dict(entry)
    return out


def main() -> None:
    setup_logging()
    t0 = time.time()
    args = parse_args()
    out_path = Path(args.output_dir) / "ground_truth_v4.json"
    partial_path = out_path.with_suffix(".json.partial")
    reject_legacy_result_path(out_path)
    source_info = require_source_lock(
        script_path=__file__,
        source_lock_path=args.source_lock_path,
        expected_script_sha256=args.expected_script_sha256,
    )
    manifest = load_json(args.manifest_path)
    assert_forget_manifest_ready(manifest, n_samples=args.n_samples)
    calibration_manifest = None
    if args.quantizer == "gptq_int4":
        if not args.calibration_manifest_path:
            raise SystemExit("gptq_int4 requires --calibration_manifest_path")
        calibration_manifest = load_json(args.calibration_manifest_path)
    if existing_complete_ok(out_path, args, manifest, source_info):
        LOGGER.info("Existing GT v4 satisfies locked request: %s", out_path)
        return
    if args.dry_run:
        LOGGER.info("dry-run ok before model load: %s", out_path)
        return
    samples = manifest["samples"]
    torch_dtype = model_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)
    device = next(model.parameters()).device
    all_target_layers = [name for name, _ in gt_v1.get_weight_layers(model)]
    target_layers = all_target_layers[: args.max_layers] if args.max_layers else list(all_target_layers)
    resume_layers = partial_layers_ok(partial_path, args, manifest, source_info, target_layers)
    if resume_layers:
        LOGGER.info("Resuming GT v4 partial with %d completed layers", len(resume_layers))
    remaining_layers = [name for name in target_layers if name not in resume_layers]
    LOGGER.info("Remaining GT v4 layers needing compute: %d/%d", len(remaining_layers), len(target_layers))
    partial_obj = load_json(partial_path) if resume_layers and partial_path.exists() else None
    if (
        partial_obj
        and isinstance(partial_obj.get("bf16_baseline"), Mapping)
        and isinstance(partial_obj.get("bf16_baseline", {}).get("per_sample"), list)
        and len(partial_obj.get("bf16_baseline", {}).get("per_sample", [])) == int(args.n_samples)
    ):
        bf16_metrics = dict(partial_obj["bf16_baseline"])
    else:
        LOGGER.info("Computing BF16 baseline with hard truth-ratio contract")
        bf16_metrics = compute_forget_eval_v4(model, tokenizer, samples, device, max_length=args.max_length, store_per_sample=True)
    if bf16_metrics.get("truth_ratio_reportable") is not True:
        raise SystemExit("BF16 truth ratio not reportable")
    layers: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    for name in target_layers:
        if name in resume_layers:
            layers[name] = resume_layers[name]
    if not remaining_layers:
        obj = build_result(args=args, manifest=manifest, calibration_manifest=calibration_manifest, source_info=source_info, target_layers=target_layers, total_model_layers=len(all_target_layers), bf16_metrics=bf16_metrics, layers=layers, total_seconds=time.time() - t0)
        if not obj["validation"]["passed"]:
            raise SystemExit(f"GT v4 validation failed: {obj['validation']}")
        save_json(out_path, obj)
        partial_path.unlink(missing_ok=True)
        LOGGER.info("GT v4 done from complete partial: %s", out_path)
        return
    kl_subset = list(samples[: min(args.kl_subset_size, len(samples))])
    LOGGER.info("Collecting BF16 reference logits on %d samples", len(kl_subset))
    bf16_logits = collect_reference_logits(model, tokenizer, kl_subset, device, max_length=args.max_length)
    hessians = None
    if args.quantizer == "gptq_int4":
        calibration_data = load_calibration_data(args, calibration_manifest)
        LOGGER.info("Collecting GPTQ Hessians for %d remaining layers from locked retain manifest", len(remaining_layers))
        hessians = gt_v1.collect_layer_hessians(model, tokenizer, calibration_data, remaining_layers, device=device, max_samples=args.n_calibration)
    param_dict = dict(model.named_parameters())

    def checkpoint() -> None:
        obj = build_result(args=args, manifest=manifest, calibration_manifest=calibration_manifest, source_info=source_info, target_layers=target_layers, total_model_layers=len(all_target_layers), bf16_metrics=bf16_metrics, layers=layers, total_seconds=time.time() - t0)
        obj["validation"] = {"passed": False, "errors": ["partial_incomplete"], "warnings": []}
        save_json(partial_path, obj)
        LOGGER.info("checkpoint saved: %d/%d layers", len(layers), len(target_layers))

    computed = 0
    for idx, layer_name in enumerate(target_layers):
        if layer_name in layers:
            LOGGER.info("GT v4 %d/%d %s already checkpointed; skipping", idx + 1, len(target_layers), layer_name)
            continue
        param = param_dict[layer_name]
        original_weight = param.data.clone()
        H = hessians[layer_name].to(device) if hessians and layer_name in hessians else None
        try:
            dequantized = gt_v1.quantize_dequantize(original_weight, args.quantizer, hessian=H)
            epsilon = dequantized.float() - original_weight.float()
            param.data.copy_(dequantized.to(param.data.dtype))
            quant_metrics = compute_forget_eval_v4(model, tokenizer, samples, device, max_length=args.max_length)
            kl_values = [kl_ref_to_quant(model, tokenizer, sample, bf16_logits[j], device, args.max_length) for j, sample in enumerate(kl_subset)]
            kl_raw = float(np.mean(kl_values)) if kl_values else 0.0
        finally:
            param.data.copy_(original_weight)
        entry = {
            "layer": layer_name,
            "layer_index": idx,
            "CE_bf16": float(bf16_metrics["answer_ce"]),
            "CE_quantized": float(quant_metrics["answer_ce"]),
            "delta_forget_ce": float(quant_metrics["answer_ce"] - bf16_metrics["answer_ce"]),
            "answer_logprob_bf16": float(bf16_metrics["answer_logprob"]),
            "answer_logprob_quantized": float(quant_metrics["answer_logprob"]),
            "delta_answer_logprob": float(quant_metrics["answer_logprob"] - bf16_metrics["answer_logprob"]),
            "truth_ratio_log_bf16": float(bf16_metrics["truth_ratio_log"]),
            "truth_ratio_log_quantized": float(quant_metrics["truth_ratio_log"]),
            "delta_truth_ratio_log": float(quant_metrics["truth_ratio_log"] - bf16_metrics["truth_ratio_log"]),
            "truth_ratio_reportable_bf16": True,
            "truth_ratio_reportable_quantized": bool(quant_metrics.get("truth_ratio_reportable")),
            "truth_ratio_n_bf16": int(bf16_metrics["truth_ratio_n"]),
            "truth_ratio_n_quantized": int(quant_metrics["truth_ratio_n"]),
            "bad_sample_count_bf16": int(bf16_metrics["bad_sample_count"]),
            "bad_sample_count_quantized": int(quant_metrics["bad_sample_count"]),
            "truth_ratio_layer_sentry": {
                "n_reportable_quantized": int(quant_metrics["truth_ratio_n"]),
                "n_samples_quantized": int(quant_metrics["n_samples"]),
                "min_answer_tokens_quantized": quant_metrics.get("truth_ratio_sentry", {}).get("min_answer_tokens"),
                "min_perturbations_quantized": quant_metrics.get("truth_ratio_sentry", {}).get("min_perturbations"),
                "nonfinite_answer_logprob_count": int(quant_metrics.get("truth_ratio_sentry", {}).get("nonfinite_answer_logprob_count", 0)),
                "nonfinite_perturbation_logprob_count": int(quant_metrics.get("truth_ratio_sentry", {}).get("nonfinite_perturbation_logprob_count", 0)),
            },
            "n_valid_answer_samples_bf16": int(bf16_metrics["n_valid_answer_samples"]),
            "n_valid_answer_samples_quantized": int(quant_metrics["n_valid_answer_samples"]),
            "n_samples_bf16": int(bf16_metrics["n_samples"]),
            "n_samples_quantized": int(quant_metrics["n_samples"]),
            "KL_forget_raw": kl_raw,
            "KL_forget_clamped": max(0.0, kl_raw),
            "block_idx": gt_v1.extract_block_idx(layer_name),
            "submodule": gt_v1.classify_submodule(layer_name),
            "weight_diff_max": float(epsilon.abs().max().item()),
            "epsilon_l2": float(epsilon.norm(2).item()),
        }
        for key, value in entry.items():
            if key not in {"layer", "submodule"} and isinstance(value, float) and not finite_number(value):
                raise ValueError(f"Non-finite GT v4 value {key} for {layer_name}")
        layers[layer_name] = entry
        computed += 1
        LOGGER.info("GT v4 %d/%d %s delta_ce=%g tr_delta=%g", idx + 1, len(target_layers), layer_name, entry["delta_forget_ce"], entry["delta_truth_ratio_log"])
        if args.checkpoint_every and computed % args.checkpoint_every == 0:
            checkpoint()
        if args.stop_after_layers and computed >= args.stop_after_layers:
            checkpoint()
            raise SystemExit(f"Stopped after {computed} layers for resume proof")
    obj = build_result(args=args, manifest=manifest, calibration_manifest=calibration_manifest, source_info=source_info, target_layers=target_layers, total_model_layers=len(all_target_layers), bf16_metrics=bf16_metrics, layers=layers, total_seconds=time.time() - t0)
    if not obj["validation"]["passed"]:
        save_json(out_path.with_suffix(".json.failed"), obj)
        raise SystemExit(f"GT v4 validation failed: {obj['validation']}")
    save_json(out_path, obj)
    partial_path.unlink(missing_ok=True)
    LOGGER.info("GT v4 done: %s", out_path)


if __name__ == "__main__":
    main()
