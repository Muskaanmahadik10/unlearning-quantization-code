#!/usr/bin/env python3
"""Torch-free validator for Gate 2 v4 whole-model bridge results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gate2_v4_contract import (  # noqa: E402
    LOSS_CONTRACT_ID,
    MASK_CONTRACT_ID,
    NO_FALLBACK_LABEL_CONTRACT_ID,
    TRUTH_RATIO_CONTRACT_ID,
    QUANTIZERS,
    finite_number,
    load_json,
)

BRIDGE_SCHEMA = "gate2_bridge_v4.1"
BRIDGE_ROOT_REQUIRED_FRAGMENT = "gate2_v4_bridge_standard"
LEGACY_FRAGMENTS = ("gate2_v2", "gate2_v3")
CONDITIONS = {"bf16", "full_quantized", "protected"}


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def finite_path(obj: Mapping[str, Any], dotted: str) -> bool:
    current: Any = obj
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return finite(current)


def validate_bridge_path(path: Optional[Path], errors: List[str]) -> None:
    if path is None:
        return
    s = str(path)
    if BRIDGE_ROOT_REQUIRED_FRAGMENT not in s:
        errors.append("bad_bridge_result_root")
    if any(frag in s for frag in LEGACY_FRAGMENTS):
        errors.append("legacy_path_forbidden")


def require_string(obj: Mapping[str, Any], key: str, errors: List[str]) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        errors.append(f"missing_{key}")
        return ""
    return value


def manifest_ok(obj: Mapping[str, Any], key: str, errors: List[str], required: bool = True) -> None:
    value = obj.get(key)
    if not isinstance(value, Mapping):
        if required:
            errors.append(f"missing_{key}")
        return
    if not isinstance(value.get("path"), str) or not value.get("path"):
        errors.append(f"{key}_missing_path")
    if not isinstance(value.get("sha256"), str) or not value.get("sha256"):
        errors.append(f"{key}_missing_sha256")


def validate_metrics(prefix: str, metrics: Mapping[str, Any], errors: List[str]) -> None:
    forget = metrics.get("forget")
    retain = metrics.get("retain")
    if not isinstance(forget, Mapping):
        errors.append(f"{prefix}_missing_forget_metrics")
        return
    if not isinstance(retain, Mapping):
        errors.append(f"{prefix}_missing_retain_metrics")
        return
    for key in ("answer_ce", "answer_logprob", "truth_ratio_log"):
        if not finite(forget.get(key)):
            errors.append(f"{prefix}_forget_{key}_nonfinite")
    if forget.get("truth_ratio_reportable") is not True:
        errors.append(f"{prefix}_truth_ratio_nonreportable")
    for key in ("answer_ce", "answer_logprob"):
        if not finite(retain.get(key)):
            errors.append(f"{prefix}_retain_{key}_nonfinite")
    if not isinstance(forget.get("n_samples"), int) or int(forget.get("n_samples", 0)) <= 0:
        errors.append(f"{prefix}_forget_n_samples_bad")
    if not isinstance(retain.get("n_samples"), int) or int(retain.get("n_samples", 0)) <= 0:
        errors.append(f"{prefix}_retain_n_samples_bad")
    if not isinstance(forget.get("answer_token_min"), int) or int(forget.get("answer_token_min", 0)) <= 0:
        errors.append(f"{prefix}_forget_answer_token_min_bad")
    forget_token_mean = forget.get("answer_token_mean")
    if not finite(forget_token_mean) or float(forget_token_mean) <= 0:
        errors.append(f"{prefix}_forget_answer_token_mean_bad")
    elif isinstance(forget.get("answer_token_min"), int) and float(forget_token_mean) < int(forget["answer_token_min"]):
        errors.append(f"{prefix}_forget_answer_token_mean_below_min")
    per_sample = forget.get("per_sample")
    if not isinstance(per_sample, list) or len(per_sample) != int(forget.get("n_samples", -1)):
        errors.append(f"{prefix}_forget_per_sample_count_bad")
    else:
        token_counts: List[int] = []
        for row in per_sample:
            if not isinstance(row, Mapping) or not bool(row.get("reportable", False)):
                errors.append(f"{prefix}_forget_per_sample_reportability_bad")
                break
            n_answer_tokens = row.get("n_answer_tokens")
            if not isinstance(n_answer_tokens, int) or n_answer_tokens <= 0:
                errors.append(f"{prefix}_forget_per_sample_token_count_bad")
                break
            token_counts.append(n_answer_tokens)
        if len(token_counts) == len(per_sample) and token_counts:
            computed_mean = sum(token_counts) / len(token_counts)
            computed_min = min(token_counts)
            if not finite(forget_token_mean) or abs(float(forget_token_mean) - computed_mean) > 1e-9:
                errors.append(f"{prefix}_forget_answer_token_mean_mismatch")
            if int(forget.get("answer_token_min", -1)) != computed_min:
                errors.append(f"{prefix}_forget_answer_token_min_mismatch")
    if not isinstance(retain.get("answer_token_min"), int) or int(retain.get("answer_token_min", 0)) <= 0:
        errors.append(f"{prefix}_retain_answer_token_min_bad")


def validate_obj(obj: Mapping[str, Any], path: Optional[Path] = None, expected_source_lock_sha256: Optional[str] = None) -> List[str]:
    errors: List[str] = []
    validate_bridge_path(path, errors)

    if obj.get("schema_version") != BRIDGE_SCHEMA:
        errors.append("bad_schema_version")
    if obj.get("stage") != "whole_model_bridge_v4":
        errors.append("bad_stage")
    if obj.get("result_root_policy") != "bridge_v4_only_no_v2_v3_reuse":
        errors.append("bad_result_root_policy")

    for key in ("cell_id", "model_tag", "dataset", "algo", "seed", "quantizer"):
        require_string(obj, key, errors)
    if obj.get("quantizer") not in QUANTIZERS:
        errors.append("bad_quantizer")

    condition = obj.get("condition")
    if condition not in CONDITIONS:
        errors.append("bad_condition")

    lock = obj.get("source_lock_sha256")
    if not isinstance(lock, str) or not lock:
        errors.append("missing_source_lock_sha256")
    if expected_source_lock_sha256 and lock != expected_source_lock_sha256:
        errors.append("source_lock_mismatch")
    provenance = obj.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("missing_provenance")
    elif provenance.get("source_lock_sha256") != lock:
        errors.append("provenance_source_lock_mismatch")

    contracts = obj.get("contracts")
    if not isinstance(contracts, Mapping):
        errors.append("missing_contracts")
    else:
        expected = {
            "loss_contract_id": LOSS_CONTRACT_ID,
            "mask_contract_id": MASK_CONTRACT_ID,
            "truth_ratio_contract_id": TRUTH_RATIO_CONTRACT_ID,
            "no_fallback_label_contract_id": NO_FALLBACK_LABEL_CONTRACT_ID,
        }
        for key, value in expected.items():
            if contracts.get(key) != value:
                errors.append(f"bad_{key}")

    evaluation = obj.get("evaluation_contract")
    if not isinstance(evaluation, Mapping):
        errors.append("missing_evaluation_contract")
    else:
        if not isinstance(evaluation.get("model_dir"), str) or not evaluation.get("model_dir"):
            errors.append("evaluation_model_dir_bad")
        for key in ("n_forget", "n_retain", "n_calibration", "max_length"):
            if not isinstance(evaluation.get(key), int) or int(evaluation.get(key, 0)) <= 0:
                errors.append(f"evaluation_{key}_bad")
        if evaluation.get("dtype") not in ("bfloat16", "float16", "float32"):
            errors.append("evaluation_dtype_bad")

    manifest_ok(obj, "forget_manifest", errors)
    manifest_ok(obj, "retain_manifest", errors)
    if obj.get("quantizer") == "gptq_int4":
        manifest_ok(obj, "calibration_manifest", errors)
    else:
        manifest_ok(obj, "calibration_manifest", errors, required=False)

    target_layers = obj.get("target_layer_names")
    protected_layers = obj.get("protected_layer_names")
    per_layer = obj.get("per_layer_quantization")
    if not isinstance(target_layers, list) or not target_layers or not all(isinstance(x, str) for x in target_layers):
        errors.append("bad_target_layer_names")
        target_layers = []
    if not isinstance(protected_layers, list) or not all(isinstance(x, str) for x in protected_layers):
        errors.append("bad_protected_layer_names")
        protected_layers = []
    if len(set(protected_layers)) != len(protected_layers):
        errors.append("duplicate_protected_layers")
    if any(x not in set(target_layers) for x in protected_layers):
        errors.append("protected_layer_not_in_target_order")
    if not isinstance(per_layer, list) or len(per_layer) != len(target_layers):
        errors.append("bad_per_layer_quantization_length")
    else:
        for idx, row in enumerate(per_layer):
            if not isinstance(row, Mapping):
                errors.append("bad_per_layer_quantization_row")
                continue
            if row.get("layer_name") != target_layers[idx]:
                errors.append("per_layer_order_mismatch")
                break
            if row.get("protected") not in (True, False):
                errors.append("per_layer_protected_not_bool")
            if row.get("quantized") not in (True, False):
                errors.append("per_layer_quantized_not_bool")
            if not finite(row.get("n_params")) or float(row.get("n_params")) <= 0:
                errors.append("per_layer_n_params_bad")

    protected_count = obj.get("protected_layer_count")
    quantized_count = obj.get("quantized_layer_count")
    if isinstance(target_layers, list):
        n_layers = len(target_layers)
        if protected_count != len(protected_layers):
            errors.append("protected_layer_count_mismatch")
        if not isinstance(quantized_count, int):
            errors.append("quantized_layer_count_bad")
        elif quantized_count + len(protected_layers) != n_layers:
            errors.append("quantized_plus_protected_count_mismatch")
        if condition == "bf16" and len(protected_layers) != n_layers:
            errors.append("bf16_must_protect_all_layers")
        if condition == "full_quantized" and len(protected_layers) != 0:
            errors.append("full_quantized_must_protect_no_layers")
        if condition == "protected":
            if not isinstance(obj.get("ranking_source"), str) or not obj.get("ranking_source"):
                errors.append("protected_missing_ranking_source")
            if not isinstance(obj.get("k"), int) or int(obj.get("k", -1)) < 0:
                errors.append("protected_bad_k")
            ranking_evidence = obj.get("ranking_evidence")
            if not isinstance(ranking_evidence, Mapping):
                errors.append("protected_missing_ranking_evidence")
            else:
                if ranking_evidence.get("ranking_source") != obj.get("ranking_source"):
                    errors.append("ranking_evidence_source_mismatch")
                if ranking_evidence.get("k") != obj.get("k"):
                    errors.append("ranking_evidence_k_mismatch")
                selected_rows = ranking_evidence.get("selected_rows")
                special_selector = obj.get("ranking_source") in ("all_layers", "none")
                if special_selector and ranking_evidence.get("special_selector") != obj.get("ranking_source"):
                    errors.append("ranking_evidence_special_selector_mismatch")
                if obj.get("ranking_source") == "all_layers" and protected_layers != target_layers:
                    errors.append("all_layers_selector_must_protect_all_layers")
                if obj.get("ranking_source") == "none" and protected_layers:
                    errors.append("none_selector_must_protect_no_layers")
                if not isinstance(selected_rows, list):
                    errors.append("ranking_evidence_selected_rows_bad")
                elif special_selector and selected_rows:
                    errors.append("ranking_evidence_special_selector_rows_must_be_empty")
                elif not special_selector and len(selected_rows) != len(protected_layers):
                    errors.append("ranking_evidence_selected_count_mismatch")
                elif not special_selector:
                    selected_layers = [str(row.get("layer_name", "")) for row in selected_rows if isinstance(row, Mapping)]
                    if selected_layers != protected_layers:
                        errors.append("ranking_evidence_layer_order_mismatch")
                    for row in selected_rows:
                        if not isinstance(row, Mapping):
                            errors.append("ranking_evidence_row_bad")
                            continue
                        if not isinstance(row.get("layer_name"), str) or not row.get("layer_name"):
                            errors.append("ranking_evidence_row_missing_layer_name")
                        if not finite(row.get("score")):
                            errors.append("ranking_evidence_row_score_nonfinite")
                        if not isinstance(row.get("rank"), int) or int(row.get("rank", 0)) <= 0:
                            errors.append("ranking_evidence_row_bad_rank")
                if not special_selector:
                    if not isinstance(ranking_evidence.get("rankings_path"), str) or not ranking_evidence.get("rankings_path"):
                        errors.append("ranking_evidence_missing_rankings_path")
                    if not isinstance(ranking_evidence.get("rankings_sha256"), str) or not ranking_evidence.get("rankings_sha256"):
                        errors.append("ranking_evidence_missing_rankings_sha256")

    bf16 = obj.get("bf16_reference")
    metrics = obj.get("condition_metrics")
    deltas = obj.get("deltas")
    if not isinstance(bf16, Mapping):
        errors.append("missing_bf16_reference")
    else:
        validate_metrics("bf16", bf16, errors)
    if not isinstance(metrics, Mapping):
        errors.append("missing_condition_metrics")
    else:
        validate_metrics("condition", metrics, errors)
    if not isinstance(deltas, Mapping):
        errors.append("missing_deltas")
    else:
        for key in ("delta_forget_ce", "delta_answer_logprob", "delta_truth_ratio_log", "harmful_recovery", "over_forgetting", "delta_retain_ce"):
            if not finite(deltas.get(key)):
                errors.append(f"{key}_nonfinite")
        if finite(deltas.get("delta_forget_ce")):
            delta_ce = float(deltas["delta_forget_ce"])
            expected_harmful = max(0.0, -delta_ce)
            expected_over = max(0.0, delta_ce)
            if finite(deltas.get("harmful_recovery")) and abs(float(deltas["harmful_recovery"]) - expected_harmful) > 1e-8:
                errors.append("harmful_recovery_identity_mismatch")
            if finite(deltas.get("over_forgetting")) and abs(float(deltas["over_forgetting"]) - expected_over) > 1e-8:
                errors.append("over_forgetting_identity_mismatch")

    compression = obj.get("compression")
    if not isinstance(compression, Mapping):
        errors.append("missing_compression")
    else:
        for key in ("target_params", "protected_params", "quantized_params", "protected_parameter_fraction", "effective_average_bits"):
            if not finite(compression.get(key)):
                errors.append(f"compression_{key}_nonfinite")
        if finite(compression.get("protected_parameter_fraction")):
            frac = float(compression["protected_parameter_fraction"])
            if frac < -1e-12 or frac > 1.0 + 1e-12:
                errors.append("protected_parameter_fraction_out_of_range")

    return sorted(set(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--expected_source_lock_sha256", default=None)
    args = parser.parse_args()
    bad = 0
    for p in args.paths:
        path = Path(p)
        obj = load_json(path)
        errors = validate_obj(obj, path=path, expected_source_lock_sha256=args.expected_source_lock_sha256)
        status = "PASS" if not errors else "FAIL"
        print(json.dumps({"path": str(path), "status": status, "errors": errors}, sort_keys=True))
        if errors:
            bad += 1
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
