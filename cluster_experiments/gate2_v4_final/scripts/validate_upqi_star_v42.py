#!/usr/bin/env python3
"""Torch-free validator for corrected Gate 2 UPQI-star v4.2 outputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping

from gate2_metric_v42_contracts import (
    GAUSS_LEGENDRE_CONTRACT,
    UPQI_ABS_CONTRACT,
    UPQI_SIGNED_CONTRACT,
    UPQI_STAR_SCHEMA,
)
from gate2_v4_contract import load_json, reject_legacy_result_path


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_obj(obj: Mapping[str, Any], path: str | Path = "<memory>") -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        reject_legacy_result_path(path)
    except Exception as exc:
        errors.append(str(exc))
    if obj.get("schema_version") != UPQI_STAR_SCHEMA:
        errors.append(f"wrong_schema:{obj.get('schema_version')}")
    if obj.get("stage") != "upqi_star_v42":
        errors.append("wrong_stage")
    if obj.get("quadrature_contract_id") != GAUSS_LEGENDRE_CONTRACT:
        errors.append("wrong_quadrature_contract")
    if obj.get("upqi_signed_contract_id") != UPQI_SIGNED_CONTRACT:
        errors.append("wrong_signed_contract")
    if obj.get("upqi_abs_contract_id") != UPQI_ABS_CONTRACT:
        errors.append("wrong_abs_contract")
    nodes = obj.get("quadrature_nodes")
    weights = obj.get("quadrature_weights")
    order = int(obj.get("n_quadrature", -1))
    if not isinstance(nodes, list) or not isinstance(weights, list) or len(nodes) != order or len(weights) != order:
        errors.append("quadrature_shape_mismatch")
    elif not all(finite(value) for value in nodes + weights):
        errors.append("nonfinite_quadrature")
    elif abs(sum(float(value) for value in weights) - 1.0) > 1e-12:
        errors.append("quadrature_weights_not_unit_sum")
    if not obj.get("sample_manifest", {}).get("manifest_sha256"):
        errors.append("missing_sample_manifest_sha256")
    if obj.get("quantizer") == "gptq_int4" and not obj.get("calibration_manifest", {}).get("manifest_sha256"):
        errors.append("missing_gptq_calibration_manifest")
    if not obj.get("ground_truth_reference", {}).get("sha256"):
        errors.append("missing_ground_truth_reference")
    if not obj.get("legacy_upqi_v4_reference", {}).get("sha256"):
        errors.append("missing_legacy_upqi_reference")
    resume_contract = obj.get("resume_contract")
    if not isinstance(resume_contract, Mapping):
        errors.append("missing_resume_contract")
        resume_contract = {}
    for key in ("batch_size", "max_length"):
        if key in obj:
            if int(obj.get(key, 0)) <= 0:
                errors.append(f"nonpositive_{key}")
            if resume_contract.get(key) != obj.get(key):
                errors.append(f"resume_contract_{key}_mismatch")
    for key in ("source_lock_sha256", "script_sha256"):
        if not obj.get("provenance", {}).get(key):
            errors.append(f"missing_provenance_{key}")
    rows = obj.get("per_layer")
    if not isinstance(rows, list):
        errors.append("per_layer_not_list")
        rows = []
    if len(rows) != int(obj.get("n_layers", -1)):
        errors.append("layer_count_mismatch")
    selection = obj.get("target_layer_selection")
    selected_global_indices: list[int] | None = None
    if selection is not None:
        if not isinstance(selection, Mapping):
            errors.append("target_layer_selection_not_mapping")
        else:
            raw_indices = selection.get("global_indices")
            total = selection.get("n_total_available_layers")
            if not isinstance(raw_indices, list) or not all(
                isinstance(value, int) and not isinstance(value, bool) for value in raw_indices
            ):
                errors.append("bad_target_global_indices")
            else:
                selected_global_indices = list(raw_indices)
                if selected_global_indices != sorted(set(selected_global_indices)):
                    errors.append("target_global_indices_not_unique_sorted")
                if len(selected_global_indices) != int(obj.get("n_layers", -1)):
                    errors.append("target_global_indices_count_mismatch")
                if not isinstance(total, int) or total <= 0:
                    errors.append("bad_total_available_layers")
                elif any(index < 0 or index >= total for index in selected_global_indices):
                    errors.append("target_global_index_out_of_range")
    names: list[str] = []
    for index, row in enumerate(rows):
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
        if selected_global_indices is not None:
            if int(row.get("global_layer_index", -1)) != selected_global_indices[index]:
                errors.append(f"bad_global_layer_index:{index}")
        for key in (
            "upqi_signed",
            "upqi_abs",
            "epsilon_l2",
            "epsilon_max",
            "integrated_grad_l2",
            "gt_delta_forget_ce",
            "completeness_residual",
        ):
            if not finite(row.get(key)):
                errors.append(f"nonfinite_{key}:{index}")
        if finite(row.get("upqi_abs")) and finite(row.get("upqi_signed")):
            if float(row["upqi_abs"]) + 1e-10 < abs(float(row["upqi_signed"])):
                errors.append(f"signed_exceeds_abs:{index}")
        if int(row.get("n_answer_tokens", 0)) <= 0:
            errors.append(f"nonpositive_answer_tokens:{index}")
    if len(names) != len(set(names)):
        errors.append("duplicate_layer_names")
    residuals = [abs(float(row["completeness_residual"])) for row in rows if isinstance(row, Mapping) and finite(row.get("completeness_residual"))]
    if residuals and max(residuals) > 0.1:
        warnings.append("large_completeness_residual_review_kn_convergence")
    return {"path": str(path), "passed": not errors, "errors": errors, "warnings": warnings, "n_layers": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path")
    parser.add_argument("--base_dir")
    parser.add_argument("--output_tsv")
    args = parser.parse_args()
    if bool(args.path) == bool(args.base_dir):
        raise SystemExit("Pass exactly one of --path or --base_dir")
    paths = [Path(args.path)] if args.path else sorted(Path(args.base_dir).rglob("upqi_star_v42.json"))
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
