#!/usr/bin/env python3
"""Build the immutable H200 E2A canary lock from the observed E1 STOP."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from build_org_h200_e2_lock_v1 import (
    EXPECTED_E1_JOBS,
    EXPECTED_V2_PARENT,
    canonical_hash,
    load_json,
    sha256_file,
    verify_checksum_manifest,
    verify_source_lock,
)


OVERLAY_FILES = (
    "org_h200_execution_v1/ORG_H200_E1_COMPLETE_EVIDENCE_LEDGER_20260905.md",
    "org_h200_execution_v1/ORG_H200_E2A_ADAPTIVE_TRADEOFF_SCIENTIFIC_CONTRACT_V1.md",
    "org_h200_execution_v1/ORG_H200_E2A_EXECUTION_RUNBOOK_V1.md",
    "org_h200_execution_v1/scripts/build_org_h200_e2_lock_v1.py",
    "org_h200_execution_v1/scripts/build_org_h200_e2a_lock_v1.py",
    "org_h200_execution_v1/scripts/build_org_h200_e2a_production_lock_v1.py",
    "org_h200_execution_v1/scripts/run_org_h200_e2a_cell_v1.py",
    "org_h200_execution_v1/scripts/validate_org_h200_e2a_v1.py",
    "org_h200_execution_v1/scripts/analyze_org_h200_e2a_v1.py",
    "org_h200_execution_v1/scripts/test_org_h200_e2a_contract_v1.py",
    "org_h200_execution_v1/slurm/org_h200_e2a_canary1_v1.slurm",
    "org_h200_execution_v1/slurm/org_h200_e2a_production8_v1.slurm",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent_lock", default="org_h200_execution_v1/locks/ORG_H200_E1_PRODUCTION_LOCK_V3.json")
    parser.add_argument("--stress_validation", required=True)
    parser.add_argument("--remaining_validation", required=True)
    parser.add_argument("--scientific_gate", required=True)
    parser.add_argument("--tradeoff_audit", required=True)
    parser.add_argument("--tradeoff_tsv", required=True)
    parser.add_argument("--e1_outputs_sha256", required=True)
    parser.add_argument("--e1_sacct", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    parent_path = Path(args.parent_lock)
    evidence = [
        Path(args.stress_validation), Path(args.remaining_validation),
        Path(args.scientific_gate), Path(args.tradeoff_audit),
        Path(args.tradeoff_tsv), Path(args.e1_outputs_sha256), Path(args.e1_sacct),
    ]
    required = [parent_path, *evidence, *(Path(name) for name in OVERLAY_FILES)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing E2A lock input:\n" + "\n".join(missing))

    parent = verify_source_lock(parent_path)
    parent_sha = str(parent["source_lock_sha256"])
    if parent.get("parent_source_lock_sha256") != EXPECTED_V2_PARENT:
        raise SystemExit("Unexpected E1 V3 parent lineage")
    if (
        parent.get("hardware") != "NVIDIA H200"
        or parent.get("retry_results_root")
        != "results/org_h200_execution_v1/gate2_v4_bridge_standard_h200_e1_core72_v3"
    ):
        raise SystemExit("Unexpected E1 V3 execution contract")

    for path, wave, cells, results, receipts in (
        (Path(args.stress_validation), "stress16", 16, 192, 16),
        (Path(args.remaining_validation), "remaining56", 56, 672, 56),
    ):
        report = load_json(path)
        if (
            report.get("status") != "PASS" or report.get("wave") != wave
            or report.get("cell_count") != cells or report.get("result_count") != results
            or report.get("receipt_count") != receipts
            or report.get("source_lock_sha256") != parent_sha or report.get("failures")
        ):
            raise SystemExit(f"E1 {wave} validation is not an exact PASS")

    gate_path = Path(args.scientific_gate)
    gate = load_json(gate_path)
    expected_gates = {
        "recovery_signal": True,
        "native_gqdp_primary": True,
        "native_gqdp_robustness": True,
        "retain_utility": False,
        "integrity": True,
    }
    if (
        gate.get("schema_version") != "org_h200_e1_scientific_gate_v1"
        or gate.get("status") != "STOP" or gate.get("e2_authorized") is not False
        or gate.get("integrity_pass") is not True or gate.get("analyzed_cells") != 72
        or gate.get("source_lock_sha256") != parent_sha
        or gate.get("gate_results") != expected_gates
    ):
        raise SystemExit("E1 gate is not the exact retain-utility-triggered STOP required by E2A")
    method = Path("org_h200_execution_v1/ORG_H200_E1_ANALYSIS_METHOD_AMENDMENT_V1.md")
    if gate.get("method_contract_sha256") != sha256_file(method):
        raise SystemExit("E1 scientific-gate method identity mismatch")

    audit_path = Path(args.tradeoff_audit)
    audit = load_json(audit_path)
    if (
        audit.get("schema_version") != "org_h200_e1_tradeoff_audit_v1"
        or audit.get("status") != "PASS"
        or audit.get("post_gate_descriptive_analysis") is not True
        or audit.get("confirmatory_gate_unchanged") is not True
        or audit.get("e2_authorized") is not False
        or audit.get("source_lock_sha256") != parent_sha
        or audit.get("gate_sha256") != sha256_file(gate_path)
        or audit.get("cell_count") != 72 or audit.get("checkpoint_count") != 40
    ):
        raise SystemExit("E1 trade-off audit mismatch")
    correlation = audit.get("cell_level", {}).get("pearson_native_D_vs_safety_slack")
    if not isinstance(correlation, (int, float)) or correlation >= 0.0:
        raise SystemExit("E1 adaptive trade-off trigger is absent")
    with Path(args.tradeoff_tsv).open("r", encoding="utf-8", newline="") as handle:
        tradeoff_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(tradeoff_rows) != 72 or len({row.get("cell_id") for row in tradeoff_rows}) != 72:
        raise SystemExit("E1 trade-off TSV must contain 72 unique cells")

    output_targets = verify_checksum_manifest(Path(args.e1_outputs_sha256), 936)
    receipts = [path for path in output_targets if "/_receipts/" in path.as_posix()]
    results = [path for path in output_targets if path.name == "bridge_v4.json"]
    if len(receipts) != 72 or len(results) != 864:
        raise SystemExit("E1 frozen output inventory mismatch")

    with Path(args.e1_sacct).open("r", encoding="utf-8", newline="") as handle:
        by_id = {row.get("JobID"): row for row in csv.DictReader(handle, delimiter="|")}
    for job_id in EXPECTED_E1_JOBS:
        main = by_id.get(job_id, {})
        step = by_id.get(f"{job_id}.0", {})
        if main.get("State") != "COMPLETED" or main.get("ExitCode") != "0:0":
            raise SystemExit(f"E1 job {job_id} did not complete cleanly")
        if step.get("State") != "COMPLETED" or step.get("ExitCode") != "0:0" or step.get("NTasks") != "8":
            raise SystemExit(f"E1 job step {job_id}.0 mismatch")

    locked = dict(parent["files"])
    for path in [parent_path, *evidence, *(Path(name) for name in OVERLAY_FILES)]:
        key, digest = path.as_posix(), sha256_file(path)
        if key in locked and locked[key] != digest:
            raise SystemExit(f"Conflicting locked file identity: {key}")
        locked[key] = digest

    output = Path(args.output)
    if output.exists():
        existing = load_json(output)
        recorded = existing.pop("source_lock_sha256", None)
        if recorded != canonical_hash(existing) or existing.get("files") != locked:
            raise SystemExit("Existing E2A lock differs; preserve it and use a new version")
        print("ORG_H200_E2A_LOCK_V1=REUSED_PASS")
        print(f"SOURCE_LOCK={output}")
        print(f"SOURCE_LOCK_SHA256={recorded}")
        return

    lock = {
        "schema_version": "gate2_v4_source_lock_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "H200 E2A post-E1 adaptive low-dose trade-off study",
        "lock_role": "e2a_canary",
        "hardware": "NVIDIA H200",
        "parent_source_lock": parent_path.as_posix(),
        "parent_source_lock_sha256": parent_sha,
        "e1_scientific_gate_status": "STOP",
        "original_e2_authorized": False,
        "adaptive_followup_authorized": True,
        "adaptive_trigger": "retain_utility_fail_after_recovery_and_native_gqdp_pass",
        "e2a_primary_k_values": [2, 4],
        "e2a_secondary_k_values": [16, 32],
        "e2a_k_values": [2, 4, 16, 32],
        "canary_results_excluded_from_e2a": True,
        "production_results_root": "results/org_h200_execution_v1/gate2_v4_bridge_standard_h200_e2a_core72_v1",
        "expected_cells": 72,
        "expected_results_per_cell": 45,
        "expected_results": 3240,
        "e1_outputs_manifest_sha256": sha256_file(Path(args.e1_outputs_sha256)),
        "e1_gate_sha256": sha256_file(gate_path),
        "e1_tradeoff_audit_sha256": sha256_file(audit_path),
        "files": locked,
    }
    lock["source_lock_sha256"] = canonical_hash(lock)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(__import__("json").dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print("ORG_H200_E2A_LOCK_V1=PASS")
    print(f"SOURCE_LOCK={output}")
    print(f"SOURCE_LOCK_SHA256={lock['source_lock_sha256']}")
    print("E1_GATE=STOP_RETAIN_UTILITY_ONLY")
    print("E2A_ADAPTIVE_FOLLOWUP=AUTHORIZED")


if __name__ == "__main__":
    main()
