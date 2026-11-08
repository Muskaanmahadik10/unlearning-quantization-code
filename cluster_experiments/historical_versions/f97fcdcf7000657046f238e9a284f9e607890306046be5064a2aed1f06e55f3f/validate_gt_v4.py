#!/usr/bin/env python3
"""Torch-free strict validator for Gate 2 GT v4 JSON."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gate2_v4_contract import (  # noqa: E402
    GT_SCHEMA,
    LOSS_CONTRACT_ID,
    MASK_CONTRACT_ID,
    NO_FALLBACK_LABEL_CONTRACT_ID,
    TRUTH_RATIO_CONTRACT_ID,
    canonical_json,
    finite_number,
    load_json,
    reject_legacy_result_path,
    save_json,
    sha256_text,
)


REQUIRED_LAYER_KEYS = (
    "CE_bf16",
    "CE_quantized",
    "delta_forget_ce",
    "answer_logprob_bf16",
    "answer_logprob_quantized",
    "delta_answer_logprob",
    "truth_ratio_log_bf16",
    "truth_ratio_log_quantized",
    "delta_truth_ratio_log",
    "KL_forget_raw",
    "KL_forget_clamped",
    "epsilon_l2",
    "weight_diff_max",
)


def validate_obj(obj: Mapping[str, Any], path: str | Path = "<memory>") -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        reject_legacy_result_path(path)
    except Exception as exc:
        errors.append(str(exc))
    if obj.get("schema_version") != GT_SCHEMA:
        errors.append(f"wrong_schema:{obj.get('schema_version')}")
    if obj.get("stage") != "gt_v4":
        errors.append("wrong_or_missing_stage")
    if obj.get("result_root_policy") != "v4_only_no_v2_v3_reuse":
        errors.append("missing_v4_only_policy")
    if obj.get("truth_ratio_policy") != "hard_required":
        errors.append("truth_ratio_policy_not_hard_required")
    if obj.get("target") != "delta_forget_ce":
        errors.append("wrong_primary_target")
    if "delta_log_AP" in obj.get("secondary_targets", []):
        errors.append("forbidden_delta_log_AP_target")
    loss = obj.get("loss_contract", {})
    if loss.get("loss_contract_id") != LOSS_CONTRACT_ID:
        errors.append("wrong_loss_contract")
    if loss.get("mask_contract_id") != MASK_CONTRACT_ID:
        errors.append("wrong_mask_contract")
    if loss.get("label_fallback_contract_id") != NO_FALLBACK_LABEL_CONTRACT_ID:
        errors.append("wrong_label_fallback_contract")
    if loss.get("truth_ratio_contract_id") != TRUTH_RATIO_CONTRACT_ID:
        errors.append("wrong_truth_ratio_contract")
    if not obj.get("sample_manifest", {}).get("manifest_sha256"):
        errors.append("missing_sample_manifest_sha256")
    sample_hashes_sha = obj.get("sample_manifest", {}).get("sample_hashes_sha256")
    if not sample_hashes_sha:
        errors.append("missing_sample_hashes_sha256")
    if obj.get("quantizer") == "gptq_int4" and not obj.get("calibration_manifest", {}).get("manifest_sha256"):
        errors.append("missing_gptq_calibration_manifest")
    if obj.get("quantizer") != "gptq_int4" and obj.get("calibration_manifest") is not None:
        errors.append("non_gptq_has_calibration_manifest")
    provenance = obj.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("missing_provenance")
    else:
        for key in ("script_sha256", "source_lock_sha256", "argv"):
            if key not in provenance:
                errors.append(f"missing_provenance_{key}")
    baseline = obj.get("bf16_baseline")
    if not isinstance(baseline, Mapping):
        errors.append("missing_bf16_baseline")
    else:
        if baseline.get("truth_ratio_reportable") is not True:
            errors.append("bf16_truth_ratio_not_reportable")
        if int(baseline.get("truth_ratio_n", -1)) != int(baseline.get("n_samples", -2)):
            errors.append("bf16_truth_ratio_count_mismatch")
        if int(baseline.get("bad_sample_count", -1)) != 0:
            errors.append("bf16_bad_sample_count_nonzero")
        for key in ("answer_ce", "answer_logprob", "truth_ratio_log"):
            if not finite_number(baseline.get(key)):
                errors.append(f"bf16_nonfinite_{key}")
    sentry = obj.get("truth_ratio_sentry")
    if not isinstance(sentry, Mapping):
        errors.append("missing_truth_ratio_sentry")
    else:
        if sentry.get("all_samples_reportable") is not True:
            errors.append("truth_ratio_sentry_not_all_reportable")
        if int(sentry.get("n_samples", -1)) != int(obj.get("n_samples", -2)):
            errors.append("truth_ratio_sentry_n_samples_mismatch")
        if int(sentry.get("n_reportable", -1)) != int(obj.get("n_samples", -2)):
            errors.append("truth_ratio_sentry_reportable_count_mismatch")
        if int(sentry.get("min_answer_tokens", 0)) <= 0:
            errors.append("truth_ratio_sentry_nonpositive_min_answer_tokens")
        if int(sentry.get("min_perturbations", 0)) <= 0:
            errors.append("truth_ratio_sentry_nonpositive_min_perturbations")
        if int(sentry.get("nonfinite_answer_logprob_count", -1)) != 0:
            errors.append("truth_ratio_sentry_nonfinite_answer_logprob")
        if int(sentry.get("nonfinite_perturbation_logprob_count", -1)) != 0:
            errors.append("truth_ratio_sentry_nonfinite_perturbation_logprob")
    evidence = obj.get("bf16_per_sample_truth_ratio_evidence")
    if not isinstance(evidence, list):
        errors.append("missing_bf16_per_sample_truth_ratio_evidence")
        evidence = []
    if len(evidence) != int(obj.get("n_samples", -1)):
        errors.append("bf16_per_sample_evidence_count_mismatch")
    evidence_hashes: List[str] = []
    for idx, row in enumerate(evidence):
        if not isinstance(row, Mapping):
            errors.append(f"bad_bf16_evidence_row:{idx}")
            continue
        if row.get("reportable") is not True:
            errors.append(f"bf16_evidence_not_reportable:{idx}")
        sample_hash = row.get("sample_hash")
        if not sample_hash:
            errors.append(f"bf16_evidence_missing_sample_hash:{idx}")
        else:
            evidence_hashes.append(str(sample_hash))
        if int(row.get("answer_token_count", 0)) <= 0:
            errors.append(f"bf16_evidence_nonpositive_answer_tokens:{idx}")
        if int(row.get("n_perturbations", 0)) <= 0:
            errors.append(f"bf16_evidence_nonpositive_perturbations:{idx}")
        for key in ("answer_logprob_bf16", "mean_perturbation_logprob_bf16", "truth_ratio_log_bf16"):
            if not finite_number(row.get(key)):
                errors.append(f"bf16_evidence_nonfinite_{key}:{idx}")
    if evidence_hashes and sample_hashes_sha:
        actual_hash = sha256_text(canonical_json(evidence_hashes))
        if actual_hash != sample_hashes_sha:
            errors.append("bf16_evidence_sample_hashes_mismatch")
    if isinstance(sentry, Mapping) and evidence_hashes and sentry.get("sample_hashes_sha256"):
        actual_hash = sha256_text(canonical_json(evidence_hashes))
        if actual_hash != sentry.get("sample_hashes_sha256"):
            errors.append("truth_ratio_sentry_sample_hashes_mismatch")
    target_names = obj.get("target_layer_names")
    layers = obj.get("layers")
    if not isinstance(target_names, list):
        errors.append("target_layer_names_not_list")
        target_names = []
    if not isinstance(layers, Mapping):
        errors.append("layers_not_mapping")
        layers = {}
    if int(obj.get("n_layers_completed", -1)) != int(obj.get("n_layers", -2)):
        errors.append("n_layers_completed_mismatch")
    if len(layers) != int(obj.get("n_layers", -1)):
        errors.append("layer_count_mismatch")
    if list(layers.keys()) != target_names:
        errors.append("layer_order_mismatch")
    for name, entry in layers.items():
        if not isinstance(entry, Mapping):
            errors.append(f"bad_layer_entry:{name}")
            continue
        for key in REQUIRED_LAYER_KEYS:
            if not finite_number(entry.get(key)):
                errors.append(f"nonfinite_{key}:{name}")
        if "delta_log_AP" in entry or "delta_AP" in entry:
            errors.append(f"forbidden_ap_alias:{name}")
        if entry.get("truth_ratio_reportable_bf16") is not True:
            errors.append(f"layer_bf16_truth_ratio_not_reportable:{name}")
        if entry.get("truth_ratio_reportable_quantized") is not True:
            errors.append(f"layer_quant_truth_ratio_not_reportable:{name}")
        layer_sentry = entry.get("truth_ratio_layer_sentry")
        if not isinstance(layer_sentry, Mapping):
            errors.append(f"missing_truth_ratio_layer_sentry:{name}")
        else:
            if int(layer_sentry.get("n_reportable_quantized", -1)) != int(entry.get("n_samples_quantized", -2)):
                errors.append(f"layer_sentry_reportable_count_mismatch:{name}")
            if int(layer_sentry.get("n_samples_quantized", -1)) != int(obj.get("n_samples", -2)):
                errors.append(f"layer_sentry_n_samples_mismatch:{name}")
            if int(layer_sentry.get("min_answer_tokens_quantized", 0)) <= 0:
                errors.append(f"layer_sentry_nonpositive_answer_tokens:{name}")
            if int(layer_sentry.get("min_perturbations_quantized", 0)) <= 0:
                errors.append(f"layer_sentry_nonpositive_perturbations:{name}")
            if int(layer_sentry.get("nonfinite_answer_logprob_count", -1)) != 0:
                errors.append(f"layer_sentry_nonfinite_answer_logprob:{name}")
            if int(layer_sentry.get("nonfinite_perturbation_logprob_count", -1)) != 0:
                errors.append(f"layer_sentry_nonfinite_perturbation_logprob:{name}")
        if int(entry.get("truth_ratio_n_bf16", -1)) != int(entry.get("n_samples_bf16", -2)):
            errors.append(f"layer_bf16_truth_ratio_count_mismatch:{name}")
        if int(entry.get("truth_ratio_n_quantized", -1)) != int(entry.get("n_samples_quantized", -2)):
            errors.append(f"layer_quant_truth_ratio_count_mismatch:{name}")
        if int(entry.get("bad_sample_count_quantized", -1)) != 0:
            errors.append(f"quantized_bad_sample_count_nonzero:{name}")
        if finite_number(entry.get("KL_forget_raw")) and finite_number(entry.get("KL_forget_clamped")):
            if float(entry["KL_forget_clamped"]) < -1e-12:
                errors.append(f"negative_kl_clamped:{name}")
            if abs(max(0.0, float(entry["KL_forget_raw"])) - float(entry["KL_forget_clamped"])) > 1e-9:
                errors.append(f"kl_clamp_identity_mismatch:{name}")
        if finite_number(entry.get("delta_answer_logprob")) and finite_number(entry.get("delta_forget_ce")):
            if abs(float(entry["delta_answer_logprob"]) + float(entry["delta_forget_ce"])) > 1e-5:
                errors.append(f"logprob_ce_identity_mismatch:{name}")
    embedded = obj.get("validation")
    if isinstance(embedded, Mapping) and embedded.get("passed") is not True:
        warnings.append("embedded_validation_not_passed")
    return {"path": str(path), "passed": not errors, "errors": errors, "warnings": warnings, "n_layers": len(layers)}


def validate_path(path: Path) -> Dict[str, Any]:
    return validate_obj(load_json(path), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GT v4 JSON")
    parser.add_argument("--path")
    parser.add_argument("--base_dir")
    parser.add_argument("--output_json")
    parser.add_argument("--output_tsv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.path) == bool(args.base_dir):
        raise SystemExit("Pass exactly one of --path or --base_dir")
    paths = [Path(args.path)] if args.path else sorted(Path(args.base_dir).rglob("ground_truth_v4.json"))
    reports = [validate_path(path) for path in paths]
    if args.output_json:
        save_json(args.output_json, reports)
    if args.output_tsv:
        with Path(args.output_tsv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "passed", "errors", "warnings", "n_layers"], delimiter="\t")
            writer.writeheader()
            for report in reports:
                row = dict(report)
                row["errors"] = ";".join(row["errors"])
                row["warnings"] = ";".join(row["warnings"])
                writer.writerow(row)
    for report in reports:
        print(f"{'PASS' if report['passed'] else 'FAIL'} {report['path']} errors={len(report['errors'])}")
    if not all(report["passed"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
