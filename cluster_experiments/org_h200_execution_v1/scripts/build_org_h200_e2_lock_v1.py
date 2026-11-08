#!/usr/bin/env python3
"""Build the immutable H200 E2 lock after all E1 gates pass."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_V2_PARENT = "dd1528d1d5378fb9a85353ae0960685862b15623291f41c4991fb3be949712e0"
EXPECTED_E1_JOBS = ("340622", "340623", "340655", "340656", "340657", "340658", "340659", "340660", "340661")
OVERLAY_FILES = (
    "org_h200_execution_v1/ORG_H200_E1_ANALYSIS_METHOD_AMENDMENT_V1.md",
    "org_h200_execution_v1/ORG_H200_E2_DOSE_RESPONSE_SCIENTIFIC_CONTRACT_V1.md",
    "org_h200_execution_v1/ORG_H200_E2_EXECUTION_AMENDMENT_V1.md",
    "org_h200_execution_v1/ORG_H200_E2_EXECUTION_RUNBOOK_V1.md",
    "org_h200_execution_v1/scripts/analyze_org_h200_e1_gate_v1.py",
    "org_h200_execution_v1/scripts/build_org_h200_e2_lock_v1.py",
    "org_h200_execution_v1/scripts/build_org_h200_e2_production_lock_v1.py",
    "org_h200_execution_v1/scripts/run_org_h200_e2_cell_v1.py",
    "org_h200_execution_v1/scripts/test_org_h200_e2_contract_v1.py",
    "org_h200_execution_v1/scripts/validate_org_h200_e2_v1.py",
    "org_h200_execution_v1/slurm/org_h200_e2_canary1_v1.slurm",
    "org_h200_execution_v1/slurm/org_h200_e2_production8_v1.slurm",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return value


def verify_source_lock(path: Path) -> dict[str, Any]:
    lock = load_json(path)
    recorded = lock.pop("source_lock_sha256", None)
    if recorded != canonical_hash(lock):
        raise SystemExit(f"Source-lock canonical hash mismatch: {path}")
    for name, expected in lock.get("files", {}).items():
        target = Path(name)
        if not target.is_file() or sha256_file(target) != expected:
            raise SystemExit(f"Source-lock file mismatch: {target}")
    lock["source_lock_sha256"] = recorded
    return lock


def verify_checksum_manifest(path: Path, expected_count: int) -> list[Path]:
    targets = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line)
        if not match:
            raise SystemExit(f"Malformed checksum line in {path}: {line}")
        expected, name = match.groups()
        target = Path(name)
        if not target.is_file() or sha256_file(target) != expected.lower():
            raise SystemExit(f"Checksum verification failed: {target}")
        targets.append(target)
    if len(targets) != expected_count or len(set(targets)) != expected_count:
        raise SystemExit(f"Checksum count mismatch: {len(targets)} != {expected_count}")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent_lock", default="org_h200_execution_v1/locks/ORG_H200_E1_PRODUCTION_LOCK_V3.json")
    parser.add_argument("--stress_validation", required=True)
    parser.add_argument("--remaining_validation", required=True)
    parser.add_argument("--scientific_gate", required=True)
    parser.add_argument("--e1_outputs_sha256", required=True)
    parser.add_argument("--e1_sacct", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    parent_path = Path(args.parent_lock)
    evidence = [Path(args.stress_validation), Path(args.remaining_validation), Path(args.scientific_gate), Path(args.e1_outputs_sha256), Path(args.e1_sacct)]
    required = [parent_path, *evidence, *(Path(name) for name in OVERLAY_FILES)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing E2 lock input:\n" + "\n".join(missing))

    parent = verify_source_lock(parent_path)
    parent_sha = str(parent["source_lock_sha256"])
    if parent.get("parent_source_lock_sha256") != EXPECTED_V2_PARENT:
        raise SystemExit("Unexpected E1 V3 parent lineage")
    if parent.get("hardware") != "NVIDIA H200" or parent.get("retry_results_root") != "results/org_h200_execution_v1/gate2_v4_bridge_standard_h200_e1_core72_v3":
        raise SystemExit("Unexpected E1 V3 execution contract")

    stress = load_json(Path(args.stress_validation))
    remaining = load_json(Path(args.remaining_validation))
    expected_reports = (
        (stress, "stress16", 16, 192, 16),
        (remaining, "remaining56", 56, 672, 56),
    )
    for report, wave, cells, results, receipts in expected_reports:
        if (
            report.get("status") != "PASS"
            or report.get("wave") != wave
            or report.get("cell_count") != cells
            or report.get("result_count") != results
            or report.get("receipt_count") != receipts
            or report.get("source_lock_sha256") != parent_sha
            or report.get("failures")
        ):
            raise SystemExit(f"E1 {wave} validation is not an exact PASS")

    gate = load_json(Path(args.scientific_gate))
    if (
        gate.get("schema_version") != "org_h200_e1_scientific_gate_v1"
        or gate.get("status") != "PASS"
        or gate.get("e2_authorized") is not True
        or gate.get("integrity_pass") is not True
        or gate.get("analyzed_cells") != 72
        or gate.get("source_lock_sha256") != parent_sha
        or not gate.get("gate_results")
        or not all(gate["gate_results"].values())
    ):
        raise SystemExit("E1 scientific gate does not authorize E2")
    method_path = Path("org_h200_execution_v1/ORG_H200_E1_ANALYSIS_METHOD_AMENDMENT_V1.md")
    if gate.get("method_contract_sha256") != sha256_file(method_path):
        raise SystemExit("E1 scientific gate method identity mismatch")

    output_targets = verify_checksum_manifest(Path(args.e1_outputs_sha256), 936)
    receipts = [path for path in output_targets if "/_receipts/" in path.as_posix()]
    results = [path for path in output_targets if path.name == "bridge_v4.json"]
    if len(receipts) != 72 or len(results) != 864:
        raise SystemExit("E1 frozen output inventory mismatch")

    with Path(args.e1_sacct).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="|"))
    by_id = {row.get("JobID"): row for row in rows}
    for job_id in EXPECTED_E1_JOBS:
        main = by_id.get(job_id, {})
        step = by_id.get(f"{job_id}.0", {})
        if main.get("State") != "COMPLETED" or main.get("ExitCode") != "0:0":
            raise SystemExit(f"E1 job {job_id} did not complete cleanly")
        if step.get("State") != "COMPLETED" or step.get("ExitCode") != "0:0" or step.get("NTasks") != "8":
            raise SystemExit(f"E1 job step {job_id}.0 mismatch")

    locked = dict(parent["files"])
    additions = [parent_path, *evidence, *(Path(name) for name in OVERLAY_FILES)]
    for path in additions:
        key = path.as_posix()
        digest = sha256_file(path)
        if key in locked and locked[key] != digest:
            raise SystemExit(f"Conflicting locked file identity: {key}")
        locked[key] = digest

    output = Path(args.output)
    if output.exists():
        existing = load_json(output)
        recorded = existing.pop("source_lock_sha256", None)
        if recorded != canonical_hash(existing) or existing.get("files") != locked:
            raise SystemExit("Existing E2 lock differs; preserve it and use a new version")
        print("ORG_H200_E2_LOCK_V1=REUSED_PASS")
        print(f"SOURCE_LOCK={output}")
        print(f"SOURCE_LOCK_SHA256={recorded}")
        return

    lock = {
        "schema_version": "gate2_v4_source_lock_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "H200 E2 fresh-budget dose-response after preregistered E1 gate",
        "lock_role": "e2_canary",
        "hardware": "NVIDIA H200",
        "parent_source_lock": parent_path.as_posix(),
        "parent_source_lock_sha256": parent_sha,
        "e1_scientific_gate_status": "PASS",
        "e2_authorized": True,
        "e2_k_values": [2, 4, 16, 32],
        "canary_results_excluded_from_e2": True,
        "production_results_root": "results/org_h200_execution_v1/gate2_v4_bridge_standard_h200_e2_core72_v1",
        "expected_cells": 72,
        "expected_results_per_cell": 45,
        "expected_results": 3240,
        "e1_outputs_manifest_sha256": sha256_file(Path(args.e1_outputs_sha256)),
        "files": locked,
    }
    lock["source_lock_sha256"] = canonical_hash(lock)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print("ORG_H200_E2_LOCK_V1=PASS")
    print(f"SOURCE_LOCK={output}")
    print(f"SOURCE_LOCK_SHA256={lock['source_lock_sha256']}")
    print(f"PARENT_SOURCE_LOCK_SHA256={parent_sha}")
    print(f"E1_RESULTS_LOCKED={len(results)}")
    print(f"E1_RECEIPTS_LOCKED={len(receipts)}")


if __name__ == "__main__":
    main()
