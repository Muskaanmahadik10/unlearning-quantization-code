#!/usr/bin/env python3
"""Run Gate 2 v4 whole-model bridge conditions.

The bridge evaluates whether layerwise v4 fragility signals predict deployed
whole-model quantization behavior. It writes only to
results/gate2_v4_bridge_standard by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gate2_v4_backend import (  # noqa: E402
    collect_layer_hessians,
    get_weight_layers,
    quantize_dequantize,
)
from gate2_v4_contract import (  # noqa: E402
    LOSS_CONTRACT_ID,
    MASK_CONTRACT_ID,
    NO_FALLBACK_LABEL_CONTRACT_ID,
    TRUTH_RATIO_CONTRACT_ID,
    assert_forget_manifest_ready,
    assert_manifest_sha,
    load_json,
    loss_contract,
    manifest_fingerprint,
    provenance,
    require_source_lock,
    save_json,
    sha256_file,
)
from gate2_v4_dataset import encode_answer_only, normalize_answer, normalize_question  # noqa: E402
from run_gate2_ground_truth_v4 import (  # noqa: E402
    compute_forget_eval_v4,
    model_dtype,
)
from validate_bridge_v4 import BRIDGE_SCHEMA, validate_obj  # noqa: E402

LOGGER = logging.getLogger("gate2_bridge_v4")
BRIDGE_ROOT_REQUIRED_FRAGMENT = "gate2_v4_bridge_standard"
LEGACY_FRAGMENTS = ("gate2_v2", "gate2_v3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--quantizer", choices=["nf4", "gptq_int4", "rtn_int8"], required=True)
    parser.add_argument("--condition", choices=["bf16", "full_quantized", "protected"], required=True)
    parser.add_argument("--ranking_source", default=None)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--rankings_path", default=None)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--retain_manifest_path", required=True)
    parser.add_argument("--calibration_manifest_path", default=None)
    parser.add_argument("--results_root", default="results/gate2_v4_bridge_standard")
    parser.add_argument("--source_lock_path", required=True)
    parser.add_argument("--expected_script_sha256", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--n_retain", type=int, default=128)
    parser.add_argument("--n_calibration", type=int, default=128)
    parser.add_argument("--manifest_n_samples", type=int, default=128)
    parser.add_argument("--manifest_n_retain", type=int, default=128)
    parser.add_argument("--manifest_n_calibration", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def reject_bad_bridge_root(path: Path) -> None:
    s = str(path)
    if BRIDGE_ROOT_REQUIRED_FRAGMENT not in s:
        raise SystemExit(f"Bridge results_root must contain {BRIDGE_ROOT_REQUIRED_FRAGMENT}: {path}")
    if any(frag in s for frag in LEGACY_FRAGMENTS):
        raise SystemExit(f"Refusing legacy bridge result path: {path}")


def cell_id(args: argparse.Namespace) -> str:
    return f"{args.model_tag}_{args.algo}_seed{args.seed}/{args.quantizer}"


def condition_slug(condition: str, ranking_source: Optional[str], k: int) -> str:
    if condition == "protected":
        if not ranking_source:
            raise SystemExit("--ranking_source is required for protected condition")
        safe_source = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in ranking_source)
        return f"protected_{safe_source}_k{k}"
    return condition


def output_path(args: argparse.Namespace) -> Path:
    slug = condition_slug(args.condition, args.ranking_source, args.k)
    return Path(args.results_root) / f"{args.model_tag}_{args.algo}_seed{args.seed}" / args.quantizer / slug / "bridge_v4.json"


def current_script_sha() -> str:
    return sha256_file(Path(__file__))


def verify_source_lock(args: argparse.Namespace) -> Mapping[str, Any]:
    lock = require_source_lock(
        script_path=Path(__file__),
        source_lock_path=Path(args.source_lock_path),
        expected_script_sha256=args.expected_script_sha256,
    )
    actual = current_script_sha()
    if actual != args.expected_script_sha256:
        raise SystemExit(f"Script SHA mismatch: actual={actual} expected={args.expected_script_sha256}")
    return lock


def manifest_descriptor(path: str) -> Dict[str, Any]:
    obj = load_json(path)
    assert_manifest_sha(obj)
    return {
        "path": path,
        "sha256": manifest_fingerprint(obj),
        "schema_version": obj.get("schema_version"),
        "dataset": obj.get("dataset"),
        "purpose": obj.get("purpose"),
        "n_samples": len(obj.get("samples", [])) if isinstance(obj.get("samples"), list) else None,
    }


def load_manifest_samples(path: str, n: int) -> Tuple[Mapping[str, Any], List[Mapping[str, Any]], str]:
    obj = load_json(path)
    assert_manifest_sha(obj)
    sha = manifest_fingerprint(obj)
    samples = obj.get("samples")
    if not isinstance(samples, list) or not samples:
        raise SystemExit(f"Manifest has no samples: {path}")
    return obj, list(samples[:n]), sha


def require_manifest_sample_count(
    manifest: Mapping[str, Any],
    *,
    expected: int,
    requested: int,
    label: str,
) -> None:
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise SystemExit(f"{label} manifest has no samples")
    if int(expected) <= 0:
        raise SystemExit(f"{label} expected manifest sample count must be positive")
    if len(samples) != int(expected):
        raise SystemExit(f"{label} manifest sample count {len(samples)} != {expected}")
    if int(requested) <= 0 or int(requested) > len(samples):
        raise SystemExit(
            f"{label} evaluation sample count {requested} outside [1, {len(samples)}]"
        )


def validate_retain_manifest(path: str, n: int) -> Tuple[Mapping[str, Any], List[Mapping[str, Any]], str]:
    obj, samples, sha = load_manifest_samples(path, n)
    bad = []
    for idx, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            bad.append(f"{idx}:not_object")
            continue
        if not str(sample.get("question", "")).strip():
            bad.append(f"{idx}:missing_question")
        if not str(sample.get("answer", "")).strip():
            bad.append(f"{idx}:missing_answer")
    if bad:
        raise SystemExit(f"Retain manifest invalid: {path}; examples={bad[:5]}")
    return obj, samples, sha


@torch.no_grad()
def compute_retain_eval(
    model: Any,
    tokenizer: Any,
    samples: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    max_length: int,
) -> Dict[str, Any]:
    ce_values: List[float] = []
    logprob_values: List[float] = []
    token_counts: List[int] = []
    for sample in samples:
        question = normalize_question(sample)
        answer = normalize_answer(sample)
        encoded = encode_answer_only(tokenizer, question, answer, max_length)
        input_ids = encoded["input_ids"].unsqueeze(0).to(device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)
        labels = encoded["labels"].unsqueeze(0).to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :].float()
        shifted_labels = labels[:, 1:]
        mask = shifted_labels != -100
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(-1, shifted_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        answer_log_probs = gathered[mask]
        n_tokens = int(mask.sum().item())
        answer_logprob = answer_log_probs.mean()
        ce = -answer_log_probs.mean()
        ce_values.append(float(ce.detach().cpu().item()))
        logprob_values.append(float(answer_logprob.detach().cpu().item()))
        token_counts.append(n_tokens)
    return {
        "answer_ce": float(sum(ce_values) / len(ce_values)),
        "answer_logprob": float(sum(logprob_values) / len(logprob_values)),
        "n_samples": len(ce_values),
        "answer_token_mean": float(sum(token_counts) / len(token_counts)),
        "answer_token_min": int(min(token_counts)),
        "perplexity_proxy": float(math.exp(min(80.0, sum(ce_values) / len(ce_values)))),
    }


def summarize_forget(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    sentry = metrics.get("truth_ratio_sentry", {}) if isinstance(metrics.get("truth_ratio_sentry"), Mapping) else {}
    per_sample = metrics.get("per_sample", [])
    if not isinstance(per_sample, list):
        raise ValueError("forget per_sample evidence must be a list")
    if len(per_sample) != int(metrics["n_samples"]):
        raise ValueError(
            "forget per_sample evidence count mismatch: "
            f"{len(per_sample)} != {int(metrics['n_samples'])}"
        )
    token_counts: List[int] = []
    for idx, row in enumerate(per_sample):
        if not isinstance(row, Mapping) or not bool(row.get("reportable", False)):
            raise ValueError(f"forget sample {idx} is not reportable")
        n_answer_tokens = row.get("n_answer_tokens")
        if not isinstance(n_answer_tokens, int) or n_answer_tokens <= 0:
            raise ValueError(f"forget sample {idx} has invalid answer token count")
        token_counts.append(n_answer_tokens)
    answer_token_mean = float(sum(token_counts) / len(token_counts))
    answer_token_min = int(min(token_counts))
    sentry_min = sentry.get("min_answer_tokens")
    if not isinstance(sentry_min, int) or sentry_min != answer_token_min:
        raise ValueError(
            "forget answer-token sentry mismatch: "
            f"sentry={sentry_min!r} computed={answer_token_min}"
        )
    return {
        "answer_ce": float(metrics["answer_ce"]),
        "answer_logprob": float(metrics["answer_logprob"]),
        "truth_ratio_log": float(metrics["truth_ratio_log"]),
        "truth_ratio_reportable": bool(metrics.get("truth_ratio_reportable", True)),
        "n_samples": int(metrics["n_samples"]),
        "answer_token_mean": answer_token_mean,
        "answer_token_min": answer_token_min,
        "truth_ratio_sentry": sentry,
        "per_sample": per_sample,
    }


def run_metrics(
    model: Any,
    tokenizer: Any,
    forget_samples: Sequence[Mapping[str, Any]],
    retain_samples: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    max_length: int,
) -> Dict[str, Any]:
    forget = compute_forget_eval_v4(
        model,
        tokenizer,
        forget_samples,
        device=device,
        max_length=max_length,
        store_per_sample=True,
    )
    retain = compute_retain_eval(
        model,
        tokenizer,
        retain_samples,
        device=device,
        max_length=max_length,
    )
    return {"forget": summarize_forget(forget), "retain": retain}


def load_ranking_rows(path: str, cid: str, ranking_source: str) -> List[Dict[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("cell_id") == cid and row.get("ranking_source") == ranking_source:
                rows.append(row)
    rows = sorted(rows, key=lambda r: (int(r.get("rank", 10**9)), str(r.get("layer_name", ""))))
    unique_rows: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        layer = str(row.get("layer_name", ""))
        if not layer:
            continue
        if layer in seen:
            continue
        seen.add(layer)
        unique_rows.append(dict(row))
    return unique_rows


def load_rankings(path: str, cid: str, ranking_source: str) -> List[str]:
    return [str(row["layer_name"]) for row in load_ranking_rows(path, cid, ranking_source)]


def protected_layers_for_condition(args: argparse.Namespace, target_layers: Sequence[str]) -> List[str]:
    if args.condition == "bf16":
        return list(target_layers)
    if args.condition == "full_quantized":
        return []
    if args.k < 0:
        raise SystemExit("--k must be non-negative")
    if args.ranking_source == "all_layers":
        return list(target_layers)
    if args.ranking_source == "none":
        return []
    if not args.rankings_path:
        raise SystemExit("--rankings_path is required for protected condition")
    if not args.ranking_source:
        raise SystemExit("--ranking_source is required for protected condition")
    ranked = load_rankings(args.rankings_path, cell_id(args), args.ranking_source)
    if not ranked:
        raise SystemExit(f"No rankings found for {cell_id(args)} source={args.ranking_source}")
    target_set = set(target_layers)
    bad = [x for x in ranked if x not in target_set]
    if bad:
        raise SystemExit(f"Ranking contains layers outside target layer order: {bad[:5]}")
    return ranked[: min(args.k, len(target_layers))]


def ranking_evidence_for_condition(
    args: argparse.Namespace,
    protected_layers: Sequence[str],
) -> Optional[Dict[str, Any]]:
    if args.condition != "protected":
        return None
    evidence: Dict[str, Any] = {
        "ranking_source": args.ranking_source,
        "k": int(args.k),
        "protected_layer_count": len(protected_layers),
        "selected_rows": [],
    }
    if args.ranking_source in ("all_layers", "none"):
        evidence["special_selector"] = args.ranking_source
        return evidence
    if not args.rankings_path:
        raise SystemExit("--rankings_path is required for protected ranking evidence")
    ranking_path = Path(args.rankings_path)
    rows = load_ranking_rows(str(ranking_path), cell_id(args), str(args.ranking_source))
    rows_by_layer = {str(row.get("layer_name", "")): row for row in rows}
    selected_rows: List[Dict[str, Any]] = []
    for layer in protected_layers:
        row = rows_by_layer.get(str(layer))
        if row is None:
            raise SystemExit(f"Protected layer missing from ranking rows: {layer}")
        selected_rows.append(
            {
                "layer_name": str(row.get("layer_name", "")),
                "rank": int(row.get("rank", 0)),
                "score": float(row.get("score", 0.0)),
                "score_direction": str(row.get("score_direction", "")),
                "source_lock_sha256": str(row.get("source_lock_sha256", "")),
                "manifest_sha256": str(row.get("manifest_sha256", "")),
                "evidence_path": str(row.get("evidence_path", "")),
                "evidence_sha256": str(row.get("evidence_sha256", "")),
            }
        )
    evidence.update(
        {
            "rankings_path": str(ranking_path),
            "rankings_sha256": sha256_file(ranking_path),
            "ranking_schema_version": str(rows[0].get("schema_version", "")) if rows else "",
            "selected_rows": selected_rows,
            "selected_source_locks": sorted({str(row.get("source_lock_sha256", "")) for row in selected_rows}),
            "selected_manifest_sha256s": sorted({str(row.get("manifest_sha256", "")) for row in selected_rows}),
            "selected_evidence_paths": sorted({str(row.get("evidence_path", "")) for row in selected_rows}),
        }
    )
    return evidence


def collect_hessians_if_needed(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    target_layers: Sequence[str],
    protected_layers: Sequence[str],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if args.quantizer != "gptq_int4":
        return {}
    if not args.calibration_manifest_path:
        raise SystemExit("gptq_int4 bridge requires --calibration_manifest_path")
    _, calib_samples, _ = validate_retain_manifest(args.calibration_manifest_path, args.n_calibration)
    remaining = [layer for layer in target_layers if layer not in set(protected_layers)]
    if not remaining:
        return {}
    LOGGER.info("Collecting GPTQ Hessians for %d bridge layers", len(remaining))
    return collect_layer_hessians(
        model,
        tokenizer,
        calib_samples,
        remaining,
        device=device,
        max_samples=args.n_calibration,
    )


def apply_condition_quantization(
    args: argparse.Namespace,
    model: Any,
    target_layers: Sequence[str],
    protected_layers: Sequence[str],
    hessians: Mapping[str, torch.Tensor],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    protected = set(protected_layers)
    param_dict = dict(model.named_parameters())
    per_layer: List[Dict[str, Any]] = []
    target_params = 0
    protected_params = 0
    quantized_params = 0
    quant_bits = 4 if args.quantizer in ("nf4", "gptq_int4") else 8
    for layer_name in target_layers:
        if layer_name not in param_dict:
            raise SystemExit(f"Target layer missing from model parameters: {layer_name}")
        weight = param_dict[layer_name]
        n_params = int(weight.numel())
        target_params += n_params
        is_protected = layer_name in protected
        if is_protected:
            protected_params += n_params
            per_layer.append(
                {
                    "layer_name": layer_name,
                    "protected": True,
                    "quantized": False,
                    "n_params": n_params,
                    "epsilon_l2": 0.0,
                    "max_abs_delta": 0.0,
                }
            )
            continue
        original = weight.data.detach().clone()
        hessian = hessians.get(layer_name)
        quantized = quantize_dequantize(original, args.quantizer, hessian=hessian)
        delta = (quantized - original).float()
        weight.data.copy_(quantized.to(dtype=weight.dtype, device=weight.device))
        quantized_params += n_params
        per_layer.append(
            {
                "layer_name": layer_name,
                "protected": False,
                "quantized": True,
                "n_params": n_params,
                "epsilon_l2": float(torch.linalg.vector_norm(delta).detach().cpu().item()),
                "max_abs_delta": float(delta.abs().max().detach().cpu().item()),
            }
        )
    if target_params <= 0:
        raise SystemExit("No target params found")
    compression = {
        "target_params": float(target_params),
        "protected_params": float(protected_params),
        "quantized_params": float(quantized_params),
        "protected_parameter_fraction": float(protected_params / target_params),
        "quantized_parameter_fraction": float(quantized_params / target_params),
        "quantizer_bits": float(quant_bits),
        "effective_average_bits": float((protected_params * 16.0 + quantized_params * quant_bits) / target_params),
    }
    return per_layer, compression


def evaluation_contract(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "model_dir": str(args.model_dir),
        "n_forget": int(args.n_samples),
        "n_retain": int(args.n_retain),
        "n_calibration": int(args.n_calibration),
        "manifest_n_forget": int(args.manifest_n_samples),
        "manifest_n_retain": int(args.manifest_n_retain),
        "manifest_n_calibration": int(args.manifest_n_calibration),
        "max_length": int(args.max_length),
        "dtype": str(args.dtype),
    }


def request_matches(
    obj: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    forget_manifest_desc: Mapping[str, Any],
    retain_manifest_desc: Mapping[str, Any],
    calibration_manifest_desc: Optional[Mapping[str, Any]],
) -> bool:
    expected_fields = {
        "model_tag": args.model_tag,
        "dataset": args.dataset,
        "algo": args.algo,
        "seed": str(args.seed),
        "quantizer": args.quantizer,
        "condition": args.condition,
        "ranking_source": args.ranking_source,
        "k": int(args.k),
    }
    if any(obj.get(key) != value for key, value in expected_fields.items()):
        return False
    if obj.get("evaluation_contract") != evaluation_contract(args):
        return False
    for key, expected in (
        ("forget_manifest", forget_manifest_desc),
        ("retain_manifest", retain_manifest_desc),
    ):
        actual = obj.get(key)
        if not isinstance(actual, Mapping) or actual.get("sha256") != expected.get("sha256"):
            return False
    actual_calibration = obj.get("calibration_manifest")
    if calibration_manifest_desc is None:
        return actual_calibration is None
    return isinstance(actual_calibration, Mapping) and actual_calibration.get("sha256") == calibration_manifest_desc.get("sha256")


def result_complete_ok(
    path: Path,
    expected_lock: str,
    args: argparse.Namespace,
    *,
    forget_manifest_desc: Mapping[str, Any],
    retain_manifest_desc: Mapping[str, Any],
    calibration_manifest_desc: Optional[Mapping[str, Any]],
) -> bool:
    if not path.exists():
        return False
    obj = load_json(path)
    errors = validate_obj(obj, path=path, expected_source_lock_sha256=expected_lock)
    if errors:
        LOGGER.info("Existing bridge result is not clean: %s errors=%s", path, ",".join(errors))
        return False
    if not request_matches(
        obj,
        args,
        forget_manifest_desc=forget_manifest_desc,
        retain_manifest_desc=retain_manifest_desc,
        calibration_manifest_desc=calibration_manifest_desc,
    ):
        LOGGER.info("Existing bridge result does not satisfy the current request: %s", path)
        return False
    LOGGER.info("Existing bridge result satisfies request: %s", path)
    return True


def reusable_full_quantized_bf16_reference(
    args: argparse.Namespace,
    expected_lock: str,
    *,
    forget_manifest_desc: Mapping[str, Any],
    retain_manifest_desc: Mapping[str, Any],
    calibration_manifest_desc: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    if args.condition != "protected":
        return None
    reference_path = (
        Path(args.results_root)
        / f"{args.model_tag}_{args.algo}_seed{args.seed}"
        / args.quantizer
        / "full_quantized"
        / "bridge_v4.json"
    )
    if not reference_path.exists():
        return None
    reference_obj = load_json(reference_path)
    errors = validate_obj(reference_obj, path=reference_path, expected_source_lock_sha256=expected_lock)
    if errors:
        LOGGER.info("Cannot reuse BF16 reference from %s: %s", reference_path, ",".join(errors))
        return None
    reference_args = argparse.Namespace(**vars(args))
    reference_args.condition = "full_quantized"
    reference_args.ranking_source = None
    reference_args.k = 0
    if not request_matches(
        reference_obj,
        reference_args,
        forget_manifest_desc=forget_manifest_desc,
        retain_manifest_desc=retain_manifest_desc,
        calibration_manifest_desc=calibration_manifest_desc,
    ):
        LOGGER.info("Cannot reuse BF16 reference with a different evaluation contract: %s", reference_path)
        return None
    bf16_reference = reference_obj.get("bf16_reference")
    return bf16_reference if isinstance(bf16_reference, Mapping) else None


def build_result(
    args: argparse.Namespace,
    *,
    source_lock_sha256: str,
    forget_manifest_desc: Mapping[str, Any],
    forget_manifest_obj: Mapping[str, Any],
    retain_manifest_desc: Mapping[str, Any],
    calibration_manifest_desc: Optional[Mapping[str, Any]],
    target_layers: Sequence[str],
    protected_layers: Sequence[str],
    per_layer: Sequence[Mapping[str, Any]],
    compression: Mapping[str, float],
    bf16_metrics: Mapping[str, Any],
    condition_metrics: Mapping[str, Any],
    ranking_evidence: Optional[Mapping[str, Any]],
    elapsed_s: float,
    source_info: Mapping[str, Any],
) -> Dict[str, Any]:
    forget_b = bf16_metrics["forget"]
    forget_c = condition_metrics["forget"]
    retain_b = bf16_metrics["retain"]
    retain_c = condition_metrics["retain"]
    delta_ce = float(forget_c["answer_ce"] - forget_b["answer_ce"])
    result = {
        "schema_version": BRIDGE_SCHEMA,
        "stage": "whole_model_bridge_v4",
        "result_root_policy": "bridge_v4_only_no_v2_v3_reuse",
        "cell_id": cell_id(args),
        "model_tag": args.model_tag,
        "dataset": args.dataset,
        "algo": args.algo,
        "seed": str(args.seed),
        "quantizer": args.quantizer,
        "condition": args.condition,
        "ranking_source": args.ranking_source,
        "k": int(args.k),
        "source_lock_sha256": source_lock_sha256,
        "script_sha256": args.expected_script_sha256,
        "contracts": {
            "loss_contract_id": LOSS_CONTRACT_ID,
            "mask_contract_id": MASK_CONTRACT_ID,
            "truth_ratio_contract_id": TRUTH_RATIO_CONTRACT_ID,
            "no_fallback_label_contract_id": NO_FALLBACK_LABEL_CONTRACT_ID,
        },
        "evaluation_contract": evaluation_contract(args),
        "loss_contract": loss_contract(args.n_samples, forget_manifest_obj),
        "forget_manifest": dict(forget_manifest_desc),
        "retain_manifest": dict(retain_manifest_desc),
        "calibration_manifest": dict(calibration_manifest_desc) if calibration_manifest_desc else None,
        "target_layer_names": list(target_layers),
        "protected_layer_names": list(protected_layers),
        "protected_layer_count": len(protected_layers),
        "quantized_layer_count": len(target_layers) - len(protected_layers),
        "ranking_evidence": dict(ranking_evidence) if ranking_evidence else None,
        "bf16_reference": bf16_metrics,
        "condition_metrics": condition_metrics,
        "deltas": {
            "delta_forget_ce": delta_ce,
            "delta_answer_logprob": float(forget_c["answer_logprob"] - forget_b["answer_logprob"]),
            "delta_truth_ratio_log": float(forget_c["truth_ratio_log"] - forget_b["truth_ratio_log"]),
            "harmful_recovery": max(0.0, -delta_ce),
            "over_forgetting": max(0.0, delta_ce),
            "delta_retain_ce": float(retain_c["answer_ce"] - retain_b["answer_ce"]),
            "delta_retain_logprob": float(retain_c["answer_logprob"] - retain_b["answer_logprob"]),
        },
        "compression": dict(compression),
        "per_layer_quantization": list(per_layer),
        "elapsed_s": elapsed_s,
        "provenance": provenance(
            script_path=Path(__file__),
            argv=sys.argv,
            source_info=source_info,
            repo_root=Path.cwd(),
        ),
    }
    return result


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    out_path = output_path(args)
    reject_bad_bridge_root(Path(args.results_root))
    source_info = verify_source_lock(args)
    source_lock_sha256 = str(source_info.get("source_lock_sha256", ""))

    forget_manifest_obj = load_json(args.manifest_path)
    assert_forget_manifest_ready(
        forget_manifest_obj,
        n_samples=args.manifest_n_samples,
    )
    require_manifest_sample_count(
        forget_manifest_obj,
        expected=args.manifest_n_samples,
        requested=args.n_samples,
        label="forget",
    )
    forget_manifest_desc = manifest_descriptor(args.manifest_path)
    _, forget_samples, _ = load_manifest_samples(args.manifest_path, args.n_samples)
    retain_manifest_obj = load_json(args.retain_manifest_path)
    assert_manifest_sha(retain_manifest_obj)
    require_manifest_sample_count(
        retain_manifest_obj,
        expected=args.manifest_n_retain,
        requested=args.n_retain,
        label="retain",
    )
    _, retain_samples, _ = validate_retain_manifest(args.retain_manifest_path, args.n_retain)
    retain_manifest_desc = manifest_descriptor(args.retain_manifest_path)
    calibration_desc = None
    if args.calibration_manifest_path:
        calibration_manifest_obj = load_json(args.calibration_manifest_path)
        assert_manifest_sha(calibration_manifest_obj)
        require_manifest_sample_count(
            calibration_manifest_obj,
            expected=args.manifest_n_calibration,
            requested=args.n_calibration,
            label="calibration",
        )
        calibration_desc = manifest_descriptor(args.calibration_manifest_path)

    if result_complete_ok(
        out_path,
        source_lock_sha256,
        args,
        forget_manifest_desc=forget_manifest_desc,
        retain_manifest_desc=retain_manifest_desc,
        calibration_manifest_desc=calibration_desc,
    ) and not args.force:
        return

    if args.condition == "protected" and not args.ranking_source:
        raise SystemExit("protected condition requires --ranking_source")
    if args.dry_run:
        LOGGER.info("dry-run ok before model load: %s", out_path)
        return

    start = time.time()
    dtype = model_dtype(args.dtype)
    LOGGER.info("Loading model %s dtype=%s", args.model_dir, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    first_param = next(model.parameters())
    device = first_param.device

    target_layers = [name for name, _ in get_weight_layers(model)]
    if not target_layers:
        raise SystemExit("No v4 target layers found")
    protected_layers = protected_layers_for_condition(args, target_layers)
    ranking_evidence = ranking_evidence_for_condition(args, protected_layers)
    LOGGER.info("Bridge condition=%s protected=%d/%d", args.condition, len(protected_layers), len(target_layers))

    bf16_metrics = reusable_full_quantized_bf16_reference(
        args,
        source_lock_sha256,
        forget_manifest_desc=forget_manifest_desc,
        retain_manifest_desc=retain_manifest_desc,
        calibration_manifest_desc=calibration_desc,
    )
    if bf16_metrics is None:
        LOGGER.info("Computing BF16 bridge reference")
        bf16_metrics = run_metrics(
            model,
            tokenizer,
            forget_samples,
            retain_samples,
            device=device,
            max_length=args.max_length,
        )
    else:
        LOGGER.info("Reusing BF16 reference from matching full_quantized bridge result")
    hessians = collect_hessians_if_needed(
        args,
        model,
        tokenizer,
        target_layers,
        protected_layers,
        device=device,
    )
    per_layer, compression = apply_condition_quantization(
        args,
        model,
        target_layers,
        protected_layers,
        hessians,
    )
    LOGGER.info("Computing bridge condition metrics")
    condition_metrics = run_metrics(
        model,
        tokenizer,
        forget_samples,
        retain_samples,
        device=device,
        max_length=args.max_length,
    )
    result = build_result(
        args,
        source_lock_sha256=source_lock_sha256,
        forget_manifest_desc=forget_manifest_desc,
        forget_manifest_obj=forget_manifest_obj,
        retain_manifest_desc=retain_manifest_desc,
        calibration_manifest_desc=calibration_desc,
        target_layers=target_layers,
        protected_layers=protected_layers,
        per_layer=per_layer,
        compression=compression,
        bf16_metrics=bf16_metrics,
        condition_metrics=condition_metrics,
        ranking_evidence=ranking_evidence,
        elapsed_s=time.time() - start,
        source_info=source_info,
    )
    errors = validate_obj(result, path=out_path, expected_source_lock_sha256=source_lock_sha256)
    if errors:
        raise SystemExit(f"Bridge result failed validation before write: {errors}")
    save_json(out_path, result)
    LOGGER.info("Bridge v4 done: %s", out_path)


if __name__ == "__main__":
    main()
