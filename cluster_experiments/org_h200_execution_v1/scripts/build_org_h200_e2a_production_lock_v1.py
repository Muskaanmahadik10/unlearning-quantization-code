#!/usr/bin/env python3
"""Promote a certified E2A canary lock to an E2A production lock."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from build_org_h200_e2_lock_v1 import canonical_hash, load_json, sha256_file, verify_checksum_manifest, verify_source_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary_lock", default="org_h200_execution_v1/locks/ORG_H200_E2A_CANARY_LOCK_V1.json")
    parser.add_argument("--canary_validation", required=True)
    parser.add_argument("--canary_outputs_sha256", required=True)
    parser.add_argument("--canary_sacct", required=True)
    parser.add_argument("--canary_job_id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = [Path(args.canary_lock), Path(args.canary_validation), Path(args.canary_outputs_sha256), Path(args.canary_sacct)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Missing E2A promotion input:\n" + "\n".join(missing))
    parent = verify_source_lock(paths[0])
    parent_sha = str(parent["source_lock_sha256"])
    if (
        parent.get("lock_role") != "e2a_canary"
        or parent.get("e1_scientific_gate_status") != "STOP"
        or parent.get("original_e2_authorized") is not False
        or parent.get("adaptive_followup_authorized") is not True
        or parent.get("e2a_k_values") != [2, 4, 16, 32]
    ):
        raise SystemExit("E2A canary lock contract mismatch")

    validation = load_json(paths[1])
    if (
        validation.get("schema_version") != "org_h200_e2a_integrity_v1"
        or validation.get("status") != "PASS" or validation.get("mode") != "canary1"
        or validation.get("cell_count") != 1 or validation.get("result_count") != 45
        or validation.get("receipt_count") != 1
        or validation.get("source_lock_sha256") != parent_sha
        or validation.get("outcome_values_inspected") is not False or validation.get("failures")
    ):
        raise SystemExit("E2A canary validation is not an exact outcome-blind PASS")

    targets = verify_checksum_manifest(paths[2], 46)
    results = [path for path in targets if path.name == "bridge_v4.json"]
    receipts = [path for path in targets if "/_receipts/canary1/" in path.as_posix()]
    if len(results) != 45 or len(receipts) != 1:
        raise SystemExit("E2A canary artifact inventory mismatch")
    receipt = load_json(receipts[0])
    if (
        receipt.get("schema_version") != "org_h200_e2a_cell_receipt_v1"
        or receipt.get("status") != "PASS" or receipt.get("execution_label") != "canary1"
        or receipt.get("result_count") != 45 or receipt.get("source_lock_sha256") != parent_sha
        or receipt.get("k_values") != [2, 4, 16, 32]
    ):
        raise SystemExit("E2A canary receipt mismatch")

    with paths[3].open("r", encoding="utf-8", newline="") as handle:
        rows = {row.get("JobID"): row for row in csv.DictReader(handle, delimiter="|")}
    for suffix in ("", ".batch"):
        row = rows.get(args.canary_job_id + suffix, {})
        if row.get("State") != "COMPLETED" or row.get("ExitCode") != "0:0":
            raise SystemExit(f"E2A canary Slurm state mismatch: {args.canary_job_id + suffix}")

    locked = dict(parent["files"])
    for path in paths:
        key, digest = path.as_posix(), sha256_file(path)
        if key in locked and locked[key] != digest:
            raise SystemExit(f"Conflicting locked file identity: {key}")
        locked[key] = digest

    output = Path(args.output)
    if output.exists():
        existing = load_json(output)
        recorded = existing.pop("source_lock_sha256", None)
        if recorded != canonical_hash(existing) or existing.get("files") != locked:
            raise SystemExit("Existing E2A production lock differs; use a new version")
        print("ORG_H200_E2A_PRODUCTION_LOCK_V1=REUSED_PASS")
        print(f"SOURCE_LOCK={output}")
        print(f"SOURCE_LOCK_SHA256={recorded}")
        return

    lock = {
        "schema_version": "gate2_v4_source_lock_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "H200 E2A production after excluded adaptive canary",
        "lock_role": "e2a_production",
        "hardware": "NVIDIA H200",
        "parent_source_lock": paths[0].as_posix(),
        "parent_source_lock_sha256": parent_sha,
        "e1_scientific_gate_status": "STOP",
        "original_e2_authorized": False,
        "adaptive_followup_authorized": True,
        "e2a_canary_status": "PASS",
        "e2a_canary_job_id": args.canary_job_id,
        "e2a_k_values": [2, 4, 16, 32],
        "canary_results_excluded_from_e2a": True,
        "production_results_root": "results/org_h200_execution_v1/gate2_v4_bridge_standard_h200_e2a_core72_v1",
        "expected_cells": 72,
        "expected_results_per_cell": 45,
        "expected_results": 3240,
        "files": locked,
    }
    lock["source_lock_sha256"] = canonical_hash(lock)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(__import__("json").dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print("ORG_H200_E2A_PRODUCTION_LOCK_V1=PASS")
    print(f"SOURCE_LOCK={output}")
    print(f"SOURCE_LOCK_SHA256={lock['source_lock_sha256']}")
    print("CANARY_RESULTS=45")
    print("CANARY_RECEIPTS=1")


if __name__ == "__main__":
    main()

