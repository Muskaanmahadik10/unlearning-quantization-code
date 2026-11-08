#!/usr/bin/env python3
"""Outcome-blind integrity validator for H200 E2A canary or production."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for directory in (REPO_ROOT / "gate2_v4_final/scripts", REPO_ROOT / "org_gb200_execution_v1/scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from gate2_v4_contract import load_json, load_source_lock, sha256_file, verify_source_lock_files  # noqa: E402
from validate_bridge_v4 import validate_obj  # noqa: E402
from validate_org_bundled_canary_v1 import tensor_fingerprint_map  # noqa: E402

K_VALUES = (2, 4, 16, 32)


def all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source_lock", required=True)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--mode", choices=("canary1", "production"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    lock = load_source_lock(args.source_lock)
    verify_source_lock_files(lock)
    lock_sha = str(lock["source_lock_sha256"])
    expected_role = "e2a_canary" if args.mode == "canary1" else "e2a_production"
    if (
        lock.get("lock_role") != expected_role
        or lock.get("adaptive_followup_authorized") is not True
        or lock.get("original_e2_authorized") is not False
        or lock.get("e1_scientific_gate_status") != "STOP"
        or lock.get("e2a_k_values") != list(K_VALUES)
    ):
        raise SystemExit("E2A source-lock role or adaptive lineage mismatch")
    if args.mode == "production" and lock.get("e2a_canary_status") != "PASS":
        raise SystemExit("E2A production lock does not certify the canary")

    with Path(args.cells).open("r", encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle, delimiter="\t"), key=lambda row: int(row["execution_index"]))
    if len(rows) != 72:
        raise SystemExit("E2A requires the frozen 72-cell table")
    expected_rows = rows[:1] if args.mode == "canary1" else rows
    expected_ids = {row["cell_id"] for row in expected_rows}
    selectors = set(manifest["selectors"])
    root = Path(args.results_root)
    failures: list[str] = []

    receipts = sorted((root / "_receipts" / args.mode).rglob("*.json"))
    if len(receipts) != len(expected_rows):
        failures.append(f"receipt_count:{len(receipts)}")
    receipt_cells = set()
    for path in receipts:
        receipt = load_json(path)
        if receipt.get("schema_version") != "org_h200_e2a_cell_receipt_v1" or receipt.get("status") != "PASS" or receipt.get("result_count") != 45:
            failures.append(f"bad_receipt:{path}")
        if receipt.get("execution_label") != args.mode or tuple(receipt.get("k_values", [])) != K_VALUES:
            failures.append(f"receipt_contract:{path}")
        if receipt.get("source_lock_sha256") != lock_sha or receipt.get("manifest_sha256") != manifest["manifest_sha256"]:
            failures.append(f"receipt_identity:{path}")
        receipt_cells.add(receipt.get("cell_id"))
        for name, expected_hash in receipt.get("result_sha256", {}).items():
            if not Path(name).is_file() or sha256_file(name) != expected_hash:
                failures.append(f"receipt_hash:{name}")
    if receipt_cells != expected_ids:
        failures.append("receipt_cell_set")

    total_results = 0
    for row in expected_rows:
        cell_id = row["cell_id"]
        paths = sorted((root / cell_id).rglob("bridge_v4.json"))
        total_results += len(paths)
        if len(paths) != 45:
            failures.append(f"cell_result_count:{cell_id}:{len(paths)}")
            continue
        seen_full = 0
        seen_sources = {k: set() for k in K_VALUES}
        base_fingerprints = None
        for path in paths:
            obj = load_json(path)
            errors = validate_obj(obj, path=path, expected_source_lock_sha256=lock_sha)
            if errors:
                failures.append(f"schema:{path}:{','.join(errors)}")
            if obj.get("cell_id") != cell_id or obj.get("quantizer") != row["quantizer"]:
                failures.append(f"cell_identity:{path}")
            if not all_finite({key: obj.get(key) for key in ("bf16_reference", "condition_metrics", "deltas", "compression", "elapsed_s")}):
                failures.append(f"nonfinite:{path}")
            if obj.get("condition") == "full_quantized" and obj.get("k") == 0:
                seen_full += 1
            elif obj.get("condition") == "protected" and obj.get("k") in K_VALUES:
                k = int(obj["k"])
                seen_sources[k].add(obj.get("ranking_source"))
                if obj.get("protected_layer_count") != k:
                    failures.append(f"protected_count:{path}")
            else:
                failures.append(f"condition:{path}")
            fingerprints = tensor_fingerprint_map(obj)
            if not fingerprints or any(not first or not second for first, second in fingerprints.values()):
                failures.append(f"fingerprints_missing:{path}")
            if base_fingerprints is None:
                base_fingerprints = fingerprints
            elif fingerprints != base_fingerprints:
                failures.append(f"fingerprint_drift:{cell_id}")
        if seen_full != 1 or any(seen_sources[k] != selectors for k in K_VALUES):
            failures.append(f"condition_set:{cell_id}")

    expected_results = len(expected_rows) * 45
    if total_results != expected_results:
        failures.append(f"total_result_count:{total_results}")
    if args.mode == "canary1":
        all_results = list(root.rglob("bridge_v4.json"))
        if len(all_results) != 45:
            failures.append(f"unexpected_canary_results:{len(all_results)}")

    report = {
        "schema_version": "org_h200_e2a_integrity_v1",
        "status": "PASS" if not failures else "FAIL",
        "outcome_values_inspected": False,
        "mode": args.mode, "cell_count": len(expected_rows),
        "result_count": total_results, "expected_result_count": expected_results,
        "receipt_count": len(receipts), "k_values": list(K_VALUES),
        "source_lock_sha256": lock_sha, "manifest_sha256": manifest["manifest_sha256"],
        "failures": failures, "validator_sha256": sha256_file(Path(__file__)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print(f"ORG_H200_E2A_VALIDATION={report['status']}")
    print(f"MODE={args.mode}")
    print(f"CELLS={len(expected_rows)}")
    print(f"RESULTS={total_results}/{expected_results}")
    print(f"RECEIPTS={len(receipts)}")
    print(f"FAILURE_COUNT={len(failures)}")
    print(f"OUTPUT={output}")
    if failures:
        for failure in failures[:30]:
            print(f"FAILURE={failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

