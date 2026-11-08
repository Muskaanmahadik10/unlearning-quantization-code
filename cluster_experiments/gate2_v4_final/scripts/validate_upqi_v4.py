#!/usr/bin/env python3
"""Torch-free strict validator for Gate 2 UPQI v4 JSON."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gate2_v4_contract import (  # noqa: E402
    LOSS_CONTRACT_ID,
    MASK_CONTRACT_ID,
    NO_FALLBACK_LABEL_CONTRACT_ID,
    TRUTH_RATIO_CONTRACT_ID,
    UPQI_ABS_CONTRACT_ID,
    UPQI_SCHEMA,
    finite_number,
    load_json,
    reject_legacy_result_path,
    save_json,
)


def validate_obj(obj: Mapping[str, Any], path: str | Path = "<memory>") -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        reject_legacy_result_path(path)
    except Exception as exc:
        errors.append(str(exc))
    if obj.get("schema_version") != UPQI_SCHEMA:
        errors.append(f"wrong_schema:{obj.get('schema_version')}")
    if obj.get("stage") != "upqi_v4":
        errors.append("wrong_or_missing_stage")
    if obj.get("upqi_abs_contract_id") != UPQI_ABS_CONTRACT_ID:
        errors.append("wrong_upqi_abs_contract")
    loss = obj.get("loss_contract", {})
    if loss.get("loss_contract_id") != LOSS_CONTRACT_ID:
        errors.append("wrong_loss_contract")
    if loss.get("mask_contract_id") != MASK_CONTRACT_ID:
        errors.append("wrong_mask_contract")
    if loss.get("label_fallback_contract_id") != NO_FALLBACK_LABEL_CONTRACT_ID:
        errors.append("wrong_label_fallback_contract")
    if loss.get("truth_ratio_contract_id") != TRUTH_RATIO_CONTRACT_ID:
        errors.append("wrong_truth_ratio_contract")
    if obj.get("dtype") != "bfloat16":
        errors.append(f"dtype_not_bfloat16:{obj.get('dtype')}")
    if not obj.get("sample_manifest", {}).get("manifest_sha256"):
        errors.append("missing_sample_manifest_sha256")
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
    per_layer = obj.get("per_layer")
    if not isinstance(per_layer, list):
        errors.append("per_layer_not_list")
        per_layer = []
    if int(obj.get("n_layers_completed", -1)) != int(obj.get("n_layers", -2)):
        errors.append("n_layers_completed_mismatch")
    if len(per_layer) != int(obj.get("n_layers", -1)):
        errors.append("n_layers_mismatch")
    for i, row in enumerate(per_layer):
        if not isinstance(row, Mapping):
            errors.append(f"bad_layer_entry:{i}")
            continue
        for key in ("upqi_signed", "upqi_abs", "epsilon_l2", "epsilon_max", "mean_grad_l2", "mean_loss"):
            if not finite_number(row.get(key)):
                errors.append(f"nonfinite_{key}:{i}")
        if finite_number(row.get("upqi_abs")) and finite_number(row.get("upqi_signed")):
            if float(row["upqi_abs"]) < abs(float(row["upqi_signed"])) - 1e-6:
                errors.append(f"signed_exceeds_abs:{i}")
        if finite_number(row.get("n_answer_tokens")) and int(row["n_answer_tokens"]) <= 0:
            errors.append(f"zero_answer_tokens:{i}")
        if finite_number(row.get("epsilon_l2")) and float(row["epsilon_l2"]) == 0.0:
            warnings.append(f"zero_displacement:{i}")
    embedded = obj.get("validation")
    if isinstance(embedded, Mapping) and embedded.get("passed") is not True:
        warnings.append("embedded_validation_not_passed")
    return {"path": str(path), "passed": not errors, "errors": errors, "warnings": warnings, "n_layers": len(per_layer)}


def validate_path(path: Path) -> Dict[str, Any]:
    return validate_obj(load_json(path), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate UPQI v4 JSON")
    parser.add_argument("--path")
    parser.add_argument("--base_dir")
    parser.add_argument("--output_json")
    parser.add_argument("--output_tsv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.path) == bool(args.base_dir):
        raise SystemExit("Pass exactly one of --path or --base_dir")
    paths = [Path(args.path)] if args.path else sorted(Path(args.base_dir).rglob("upqi_v4.json"))
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

