#!/usr/bin/env python3
"""Outcome-blind integrity validator for an H200 E1 execution wave."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
V4_DIR = REPO_ROOT / "gate2_v4_final" / "scripts"
ORG_DIR = REPO_ROOT / "org_gb200_execution_v1" / "scripts"
for directory in (V4_DIR, ORG_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from gate2_v4_contract import load_json, load_source_lock, sha256_file, verify_source_lock_files  # noqa: E402
from validate_bridge_v4 import validate_obj  # noqa: E402
from validate_org_bundled_canary_v1 import tensor_fingerprint_map  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source_lock", required=True)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--wave", choices=("stress16", "remaining56"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = load_json(args.manifest)
    lock = load_source_lock(args.source_lock)
    verify_source_lock_files(lock)
    lock_sha = str(lock["source_lock_sha256"])
    with Path(args.cells).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    wave_rows = {row["cell_id"]: row for row in rows if row["execution_wave"] == args.wave}
    expected_ids = set(manifest["stress16_cell_ids"] if args.wave == "stress16" else manifest["remaining56_cell_ids"])
    failures = []
    if set(wave_rows) != expected_ids:
        failures.append("wave_cell_identity")
    selectors = set(manifest["selectors"])
    root = Path(args.results_root)
    receipts = sorted((root / "_receipts" / args.wave).rglob("*.json"))
    if len(receipts) != len(expected_ids):
        failures.append(f"receipt_count:{len(receipts)}")
    receipt_cells = set()
    for path in receipts:
        receipt = load_json(path)
        if receipt.get("status") != "PASS" or receipt.get("result_count") != 12:
            failures.append(f"bad_receipt:{path}")
        if receipt.get("source_lock_sha256") != lock_sha or receipt.get("manifest_sha256") != manifest["manifest_sha256"]:
            failures.append(f"receipt_identity:{path}")
        receipt_cells.add(receipt.get("cell_id"))
        for name, expected_hash in receipt.get("result_sha256", {}).items():
            if not Path(name).is_file() or sha256_file(name) != expected_hash:
                failures.append(f"receipt_hash:{name}")
    if receipt_cells != expected_ids:
        failures.append("receipt_cell_set")

    total_results = 0
    for cell_id in sorted(expected_ids):
        cell_root = root / cell_id
        paths = sorted(cell_root.rglob("bridge_v4.json"))
        total_results += len(paths)
        if len(paths) != 12:
            failures.append(f"cell_result_count:{cell_id}:{len(paths)}")
            continue
        seen_full = 0
        seen_sources = set()
        base_fingerprints = None
        for path in paths:
            obj = load_json(path)
            errors = validate_obj(obj, path=path, expected_source_lock_sha256=lock_sha)
            if errors:
                failures.append(f"schema:{path}:{','.join(errors)}")
            if obj.get("cell_id") != cell_id:
                failures.append(f"cell_identity:{path}")
            if obj.get("condition") == "full_quantized":
                seen_full += 1
            elif obj.get("condition") == "protected" and obj.get("k") == 8:
                seen_sources.add(obj.get("ranking_source"))
                if obj.get("protected_layer_count") != 8:
                    failures.append(f"protected_count:{path}")
            else:
                failures.append(f"condition:{path}")
            fingerprints = tensor_fingerprint_map(obj)
            if not fingerprints or any(not a or not b for a, b in fingerprints.values()):
                failures.append(f"fingerprints_missing:{path}")
            if base_fingerprints is None:
                base_fingerprints = fingerprints
            elif fingerprints != base_fingerprints:
                failures.append(f"fingerprint_drift:{cell_id}")
        if seen_full != 1 or seen_sources != selectors:
            failures.append(f"condition_set:{cell_id}")
    expected_results = len(expected_ids) * 12
    if total_results != expected_results:
        failures.append(f"total_result_count:{total_results}")
    report = {
        "schema_version": "org_h200_e1_wave_integrity_v1",
        "status": "PASS" if not failures else "FAIL",
        "outcome_values_inspected": False,
        "wave": args.wave,
        "cell_count": len(expected_ids),
        "result_count": total_results,
        "expected_result_count": expected_results,
        "receipt_count": len(receipts),
        "source_lock_sha256": lock_sha,
        "manifest_sha256": manifest["manifest_sha256"],
        "failures": failures,
        "validator_sha256": sha256_file(Path(__file__)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print(f"ORG_H200_E1_WAVE_VALIDATION={report['status']}")
    print(f"WAVE={args.wave}")
    print(f"CELLS={len(expected_ids)}")
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
