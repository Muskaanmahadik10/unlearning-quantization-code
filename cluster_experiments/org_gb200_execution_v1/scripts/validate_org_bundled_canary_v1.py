#!/usr/bin/env python3
"""Validate bundled canary outputs and compare NF4 against the legacy runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
V4_DIR = REPO_ROOT / "gate2_v4_final" / "scripts"
if str(V4_DIR) not in sys.path:
    sys.path.insert(0, str(V4_DIR))

from gate2_v4_contract import load_json, sha256_file  # noqa: E402
from validate_bridge_v4 import validate_obj  # noqa: E402

ATOL = 1e-6
RTOL = 1e-6
EXPECTED_NEW = 12
EXPECTED_OLD_NF4 = 6


def close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), abs_tol=ATOL, rel_tol=RTOL)


def numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            output.update(numeric_leaves(child, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            name = f"{prefix}[{index}]"
            output.update(numeric_leaves(child, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)
    return output


def compare_numeric(label: str, old: Any, new: Any, failures: list[str], report: dict[str, Any]) -> None:
    left = numeric_leaves(old)
    right = numeric_leaves(new)
    if set(left) != set(right):
        failures.append(f"{label}:numeric_paths_differ")
        return
    deltas = {key: abs(left[key] - right[key]) for key in left}
    bad = [key for key in left if not close(left[key], right[key])]
    report[label] = {
        "numeric_leaf_count": len(left),
        "max_abs_delta": max(deltas.values(), default=0.0),
        "bad_count": len(bad),
        "bad_examples": bad[:10],
    }
    if bad:
        failures.append(f"{label}:numeric_mismatch:{len(bad)}")


def result_files(root: Path) -> list[Path]:
    return sorted(root.rglob("bridge_v4.json"))


def identity(obj: Mapping[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(obj["cell_id"]),
        str(obj["quantizer"]),
        str(obj["condition"]),
        str(obj.get("ranking_source")),
        int(obj["k"]),
    )


def canonical_selected_rows(obj: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    evidence = obj.get("ranking_evidence")
    if not isinstance(evidence, Mapping):
        return []
    rows = evidence.get("selected_rows")
    if not isinstance(rows, list):
        return []
    return [
        (
            row.get("layer_name"), row.get("rank"), row.get("score"),
            row.get("score_direction"), row.get("source_lock_sha256"),
            row.get("manifest_sha256"), row.get("evidence_path"), row.get("evidence_sha256"),
        )
        for row in rows if isinstance(row, Mapping)
    ]


def tensor_fingerprint_map(obj: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    return {
        str(row["layer_name"]): (
            str(row.get("original_tensor_sha256", "")),
            str(row.get("quantized_tensor_sha256", "")),
        )
        for row in obj["per_layer_quantization"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy_root", required=True)
    parser.add_argument("--bundled_root", required=True)
    parser.add_argument("--bundle_lock", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    old_root = Path(args.legacy_root)
    new_root = Path(args.bundled_root)
    lock = load_json(args.bundle_lock)
    expected_lock = str(lock["source_lock_sha256"])
    failures: list[str] = []
    report: dict[str, Any] = {
        "schema_version": "org_bundled_canary_validation_v1",
        "atol": ATOL,
        "rtol": RTOL,
        "legacy_root": str(old_root),
        "bundled_root": str(new_root),
        "bundle_lock": str(args.bundle_lock),
        "bundle_lock_file_sha256": sha256_file(args.bundle_lock),
        "validator_sha256": sha256_file(Path(__file__)),
        "comparisons": {},
    }
    new_paths = result_files(new_root)
    old_paths = result_files(old_root)
    if len(new_paths) != EXPECTED_NEW:
        failures.append(f"new_result_count:{len(new_paths)}")
    new_objects: dict[tuple[str, str, str, str, int], Mapping[str, Any]] = {}
    for path in new_paths:
        obj = load_json(path)
        errors = validate_obj(obj, path=path, expected_source_lock_sha256=expected_lock)
        if errors:
            failures.append(f"new_validation:{path}:{','.join(errors)}")
        key = identity(obj)
        if key in new_objects:
            failures.append(f"duplicate_new_identity:{key}")
        new_objects[key] = obj

    old_nf4 = []
    for path in old_paths:
        obj = load_json(path)
        if obj.get("quantizer") == "nf4":
            errors = validate_obj(obj, path=path)
            if errors:
                failures.append(f"legacy_validation:{path}:{','.join(errors)}")
            old_nf4.append(obj)
    if len(old_nf4) != EXPECTED_OLD_NF4:
        failures.append(f"legacy_nf4_count:{len(old_nf4)}")

    for old in old_nf4:
        key = identity(old)
        new = new_objects.get(key)
        label = "/".join(map(str, key))
        if new is None:
            failures.append(f"missing_new_match:{label}")
            continue
        for field in (
            "cell_id", "model_tag", "dataset", "algo", "seed", "quantizer", "condition",
            "ranking_source", "k", "target_layer_names", "protected_layer_names",
            "protected_layer_count", "quantized_layer_count", "evaluation_contract", "contracts",
        ):
            if old.get(field) != new.get(field):
                failures.append(f"{label}:identity_field:{field}")
        if canonical_selected_rows(old) != canonical_selected_rows(new):
            failures.append(f"{label}:selected_ranking_rows")
        compare_numeric(f"{label}:bf16", old["bf16_reference"], new["bf16_reference"], failures, report["comparisons"])
        compare_numeric(f"{label}:condition", old["condition_metrics"], new["condition_metrics"], failures, report["comparisons"])
        compare_numeric(f"{label}:deltas", old["deltas"], new["deltas"], failures, report["comparisons"])
        compare_numeric(f"{label}:compression", old["compression"], new["compression"], failures, report["comparisons"])
        old_layers = [
            {key: row.get(key) for key in ("layer_name", "protected", "quantized", "n_params", "epsilon_l2", "max_abs_delta")}
            for row in old["per_layer_quantization"]
        ]
        new_layers = [
            {key: row.get(key) for key in ("layer_name", "protected", "quantized", "n_params", "epsilon_l2", "max_abs_delta")}
            for row in new["per_layer_quantization"]
        ]
        if [row["layer_name"] for row in old_layers] != [row["layer_name"] for row in new_layers]:
            failures.append(f"{label}:per_layer_order")
        compare_numeric(f"{label}:per_layer", old_layers, new_layers, failures, report["comparisons"])

    # Cached original and quantized tensors must be invariant across conditions within each cell.
    by_cell: dict[str, list[Mapping[str, Any]]] = {}
    for obj in new_objects.values():
        by_cell.setdefault(str(obj["cell_id"]), []).append(obj)
    for cell, objects in by_cell.items():
        base_fingerprints = tensor_fingerprint_map(objects[0])
        base_bf16 = objects[0]["bf16_reference"]
        for obj in objects[1:]:
            if tensor_fingerprint_map(obj) != base_fingerprints:
                failures.append(f"{cell}:cached_tensor_fingerprint_drift")
            if obj["bf16_reference"] != base_bf16:
                failures.append(f"{cell}:bf16_reference_drift")
        if any(not original or not quantized for original, quantized in base_fingerprints.values()):
            failures.append(f"{cell}:missing_tensor_fingerprint")
        report["comparisons"][f"{cell}:cache_identity"] = {
            "conditions": len(objects),
            "layers": len(base_fingerprints),
            "status": "PASS",
        }

    report["new_result_count"] = len(new_paths)
    report["legacy_nf4_result_count"] = len(old_nf4)
    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print(f"ORG_BUNDLED_CANARY_VALIDATION={report['status']}")
    print(f"NEW_RESULTS={len(new_paths)}")
    print(f"LEGACY_NF4_RESULTS={len(old_nf4)}")
    print(f"FAILURE_COUNT={len(failures)}")
    print(f"OUTPUT={output}")
    if failures:
        for failure in failures[:20]:
            print(f"FAILURE={failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
