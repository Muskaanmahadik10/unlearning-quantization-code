#!/usr/bin/env python3
"""Torch-free strict validator for Gate 2 v4.2 metric-suite JSON."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gate2_metric_v42_contracts import GQDP_CONTRACT, METRIC_SUITE_SCHEMA  # noqa: E402
from gate2_v4_contract import load_json, reject_legacy_result_path  # noqa: E402


REQUIRED_METRICS = (
    "gqdp_signed",
    "gqdp_abs",
    "gqdp_cosine",
    "gqdp_per_param",
    "magnitude_product",
    "grad_l1",
    "grad_l2",
    "grad_mean_abs",
    "epsilon_l1",
    "epsilon_l2",
    "epsilon_mean_abs",
    "epsilon_max",
    "weight_l1",
    "weight_l2",
    "weight_mean_abs",
    "quantization_mse",
    "sqnr_db",
    "layer_position",
    "bcr_true",
)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_obj(obj: Mapping[str, Any], path: str | Path = "<memory>") -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        reject_legacy_result_path(path)
    except Exception as exc:
        errors.append(str(exc))
    if obj.get("schema_version") != METRIC_SUITE_SCHEMA:
        errors.append(f"wrong_schema:{obj.get('schema_version')}")
    if obj.get("stage") != "metric_suite_v42":
        errors.append("wrong_stage")
    if obj.get("gqdp_contract_id") != GQDP_CONTRACT:
        errors.append("wrong_gqdp_contract")
    if obj.get("gradient_aggregation") != "answer_token_weighted_mean_gradient":
        errors.append("wrong_gradient_aggregation")
    if obj.get("bcr_contract") != "fraction_exact_dequantized_values_changed_original_vs_unlearned_shared_quantizer_context":
        errors.append("wrong_bcr_contract")
    for container_name in ("sample_manifest", "retain_manifest"):
        if not obj.get(container_name, {}).get("manifest_sha256"):
            errors.append(f"missing_{container_name}_sha256")
    if obj.get("quantizer") == "gptq_int4" and not obj.get("calibration_manifest", {}).get("manifest_sha256"):
        errors.append("missing_gptq_calibration_manifest")
    if obj.get("quantizer") != "gptq_int4" and obj.get("calibration_manifest") is not None:
        errors.append("non_gptq_has_calibration_manifest")
    for key in ("source_lock_sha256", "script_sha256"):
        if not obj.get("provenance", {}).get(key):
            errors.append(f"missing_provenance_{key}")
    layers = obj.get("per_layer")
    if not isinstance(layers, list):
        errors.append("per_layer_not_list")
        layers = []
    if len(layers) != int(obj.get("n_layers", -1)):
        errors.append("layer_count_mismatch")
    names: list[str] = []
    for index, row in enumerate(layers):
        if not isinstance(row, Mapping):
            errors.append(f"bad_layer:{index}")
            continue
        name = row.get("layer")
        if not isinstance(name, str) or not name:
            errors.append(f"bad_layer_name:{index}")
        else:
            names.append(name)
        if int(row.get("layer_index", -1)) != index:
            errors.append(f"bad_layer_index:{index}")
        for metric in REQUIRED_METRICS:
            if not finite(row.get(metric)):
                errors.append(f"nonfinite_{metric}:{index}")
        if finite(row.get("gqdp_abs")) and finite(row.get("gqdp_signed")):
            if abs(float(row["gqdp_abs"]) - abs(float(row["gqdp_signed"]))) > 1e-10:
                errors.append(f"gqdp_abs_identity:{index}")
        if int(row.get("n_parameters", 0)) <= 0:
            errors.append(f"bad_n_parameters:{index}")
        for optional in ("fisher_forget_ps", "fqp_forget", "fisher_retain_ps"):
            if optional in row and row[optional] is not None and not finite(row[optional]):
                errors.append(f"nonfinite_{optional}:{index}")
    if len(names) != len(set(names)):
        errors.append("duplicate_layer_names")
    if int(obj.get("fisher_forget_samples", 0)) == 0:
        warnings.append("fisher_forget_not_computed")
    if int(obj.get("fisher_retain_samples", 0)) == 0:
        warnings.append("fisher_retain_not_computed")
    return {"path": str(path), "passed": not errors, "errors": errors, "warnings": warnings, "n_layers": len(layers)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path")
    parser.add_argument("--base_dir")
    parser.add_argument("--output_tsv")
    args = parser.parse_args()
    if bool(args.path) == bool(args.base_dir):
        raise SystemExit("Pass exactly one of --path or --base_dir")
    paths = [Path(args.path)] if args.path else sorted(Path(args.base_dir).rglob("metrics_v42.json"))
    reports = [validate_obj(load_json(path), path) for path in paths]
    if args.output_tsv:
        with Path(args.output_tsv).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("path", "passed", "errors", "warnings", "n_layers"), delimiter="\t")
            writer.writeheader()
            for report in reports:
                writer.writerow({**report, "errors": ";".join(report["errors"]), "warnings": ";".join(report["warnings"])})
    for report in reports:
        print(f"{'PASS' if report['passed'] else 'FAIL'} {report['path']} errors={len(report['errors'])}")
    if not reports or not all(report["passed"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
