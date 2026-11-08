#!/usr/bin/env python3
"""Fail-closed validation for the eight-GPU H200 adversarial canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ORG_SCRIPTS = REPO_ROOT / "org_gb200_execution_v1" / "scripts"
V4_DIR = REPO_ROOT / "gate2_v4_final" / "scripts"
for directory in (ORG_SCRIPTS, V4_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from gate2_v4_contract import load_json, sha256_file  # noqa: E402
from validate_bridge_v4 import validate_obj  # noqa: E402
from validate_org_bundled_canary_v1 import (  # noqa: E402
    canonical_selected_rows,
    compare_numeric,
    identity,
    tensor_fingerprint_map,
)


def result_objects(root: Path) -> list[tuple[Path, Mapping[str, Any]]]:
    return [(path, load_json(path)) for path in sorted(root.rglob("bridge_v4.json"))]


def comparable_layers(obj: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = ("layer_name", "protected", "quantized", "n_params", "epsilon_l2", "max_abs_delta")
    return [{key: row.get(key) for key in fields} for row in obj["per_layer_quantization"]]


def compare_result(
    label: str,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    failures: list[str],
    comparisons: dict[str, Any],
) -> None:
    for field in (
        "cell_id", "model_tag", "dataset", "algo", "seed", "quantizer", "condition",
        "ranking_source", "k", "target_layer_names", "protected_layer_names",
        "protected_layer_count", "quantized_layer_count", "evaluation_contract", "contracts",
    ):
        if reference.get(field) != candidate.get(field):
            failures.append(f"{label}:identity_field:{field}")
    if canonical_selected_rows(reference) != canonical_selected_rows(candidate):
        failures.append(f"{label}:selected_ranking_rows")
    for field in ("bf16_reference", "condition_metrics", "deltas", "compression"):
        compare_numeric(f"{label}:{field}", reference[field], candidate[field], failures, comparisons)
    compare_numeric(
        f"{label}:per_layer", comparable_layers(reference), comparable_layers(candidate), failures, comparisons
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundled_root", required=True)
    parser.add_argument("--legacy_root", required=True)
    parser.add_argument("--repeat_root", required=True)
    parser.add_argument("--source_lock", required=True)
    parser.add_argument("--parity128", required=True)
    parser.add_argument("--boundary224", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    lock = load_json(args.source_lock)
    lock_sha = str(lock["source_lock_sha256"])
    bundled = result_objects(Path(args.bundled_root))
    legacy = result_objects(Path(args.legacy_root))
    repeat = result_objects(Path(args.repeat_root))
    failures: list[str] = []
    comparisons: dict[str, Any] = {}
    report: dict[str, Any] = {
        "schema_version": "org_h200_adversarial_canary_validation_v1",
        "source_lock": args.source_lock,
        "source_lock_file_sha256": sha256_file(args.source_lock),
        "validator_sha256": sha256_file(Path(__file__)),
        "bundled_root": args.bundled_root,
        "legacy_root": args.legacy_root,
        "repeat_root": args.repeat_root,
        "comparisons": comparisons,
    }

    if len(bundled) != 18:
        failures.append(f"bundled_result_count:{len(bundled)}")
    if len(legacy) != 12:
        failures.append(f"legacy_result_count:{len(legacy)}")
    if len(repeat) != 6:
        failures.append(f"repeat_result_count:{len(repeat)}")

    collections = (("bundled", bundled), ("legacy", legacy), ("repeat", repeat))
    maps: dict[str, dict[tuple[str, str, str, str, int], Mapping[str, Any]]] = {}
    for collection_name, rows in collections:
        mapped: dict[tuple[str, str, str, str, int], Mapping[str, Any]] = {}
        for path, obj in rows:
            errors = validate_obj(obj, path=path, expected_source_lock_sha256=lock_sha)
            if errors:
                failures.append(f"{collection_name}_validation:{path}:{','.join(errors)}")
            key = identity(obj)
            if key in mapped:
                failures.append(f"{collection_name}_duplicate:{key}")
            mapped[key] = obj
        maps[collection_name] = mapped

    bundled_quantizers = sorted(str(obj.get("quantizer")) for _, obj in bundled)
    for quantizer in ("gptq_int4", "nf4", "rtn_int8"):
        if bundled_quantizers.count(quantizer) != 6:
            failures.append(f"bundled_{quantizer}_count:{bundled_quantizers.count(quantizer)}")

    # Bundled caching must be invariant across all six conditions in each cell.
    by_cell: dict[str, list[Mapping[str, Any]]] = {}
    for _, obj in bundled:
        by_cell.setdefault(str(obj["cell_id"]), []).append(obj)
    if len(by_cell) != 3:
        failures.append(f"bundled_cell_count:{len(by_cell)}")
    for cell, objects in by_cell.items():
        if len(objects) != 6:
            failures.append(f"{cell}:condition_count:{len(objects)}")
            continue
        fingerprints = tensor_fingerprint_map(objects[0])
        bf16 = objects[0]["bf16_reference"]
        for obj in objects[1:]:
            if tensor_fingerprint_map(obj) != fingerprints:
                failures.append(f"{cell}:cached_tensor_fingerprint_drift")
            if obj["bf16_reference"] != bf16:
                failures.append(f"{cell}:bf16_reference_drift")
        comparisons[f"{cell}:cache_invariance"] = {"conditions": len(objects), "layers": len(fingerprints)}

    # Legacy separate-process execution must match bundled NF4 and RTN results.
    for key, old in maps["legacy"].items():
        new = maps["bundled"].get(key)
        label = "legacy_vs_bundle/" + "/".join(map(str, key))
        if new is None:
            failures.append(f"missing_bundled_match:{key}")
        else:
            compare_result(label, old, new, failures, comparisons)

    # Independent RTN bundle repeat must reproduce all values and fingerprints.
    for key, repeated in maps["repeat"].items():
        original = maps["bundled"].get(key)
        label = "repeat_vs_bundle/" + "/".join(map(str, key))
        if original is None:
            failures.append(f"missing_repeat_match:{key}")
            continue
        compare_result(label, original, repeated, failures, comparisons)
        if tensor_fingerprint_map(original) != tensor_fingerprint_map(repeated):
            failures.append(f"{label}:tensor_fingerprint_mismatch")

    parity128 = load_json(args.parity128)
    if parity128.get("status") != "PASS":
        failures.append("production128_parity_status")
    if parity128.get("n_samples") != 128:
        failures.append("production128_sample_count")
    if parity128.get("hessian_exact_count") != 21:
        failures.append("production128_hessian_exact")
    if parity128.get("quantized_exact_count") != 21:
        failures.append("production128_quantized_exact")

    boundary = load_json(args.boundary224)
    if boundary.get("status") != "PASS_FAST":
        failures.append("boundary224_status")
    if boundary.get("fast", {}).get("hessian", {}).get("exact_count") != 224:
        failures.append("boundary224_hessian_exact")
    if boundary.get("fast", {}).get("quantized", {}).get("exact_count") != 21:
        failures.append("boundary224_quantized_exact")

    report.update(
        bundled_result_count=len(bundled),
        legacy_result_count=len(legacy),
        repeat_result_count=len(repeat),
        parity128_file_sha256=sha256_file(args.parity128),
        boundary224_file_sha256=sha256_file(args.boundary224),
        failures=failures,
        status="PASS" if not failures else "FAIL",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print(f"ORG_H200_ADVERSARIAL_VALIDATION={report['status']}")
    print(f"BUNDLED_RESULTS={len(bundled)}")
    print(f"LEGACY_RESULTS={len(legacy)}")
    print(f"REPEAT_RESULTS={len(repeat)}")
    print(f"FAILURE_COUNT={len(failures)}")
    print(f"OUTPUT={output}")
    if failures:
        for failure in failures[:30]:
            print(f"FAILURE={failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
