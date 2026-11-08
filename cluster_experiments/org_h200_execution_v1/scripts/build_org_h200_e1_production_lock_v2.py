#!/usr/bin/env python3
"""Build the source lock for the single-node H200 E1 launcher repair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PARENT_LOCK = Path("org_h200_execution_v1/locks/ORG_H200_E1_PRODUCTION_LOCK_V1.json")
SACCT = Path("results/org_h200_execution_v1/certification/job340568_sacct.txt")
FAILURE_RESULTS = Path("results/org_h200_execution_v1/certification/job340568_failure_artifacts.sha256")
FAILURE_LOGS = Path("results/org_h200_execution_v1/certification/job340568_logs.sha256")
OVERLAY_FILES = (
    "org_h200_execution_v1/H200_E1_STRESS16_LAUNCHER_AMENDMENT_V2.md",
    "org_h200_execution_v1/H200_E1_STRESS8_REPAIR_RUNBOOK_V2.md",
    "org_h200_execution_v1/scripts/build_org_h200_e1_production_lock_v2.py",
    "org_h200_execution_v1/slurm/org_h200_e1_stress8_v2.slurm",
    "org_h200_execution_v1/slurm/org_h200_e1_remaining8_v2.slurm",
)
EXPECTED_RECEIPTS = {
    "results/org_h200_execution_v1/gate2_v4_bridge_standard_h200_e1_core72_v1/_receipts/stress16/llama2_muse_adamw_GDR_seed42__rtn_int8.json",
    "results/org_h200_execution_v1/gate2_v4_bridge_standard_h200_e1_core72_v1/_receipts/stress16/llama2_muse_adamw_NPO_seed5000__nf4.json",
}


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
        raise SystemExit(f"Parent source-lock canonical hash mismatch: {path}")
    for name, expected in lock.get("files", {}).items():
        target = Path(name)
        if not target.is_file() or sha256_file(target) != expected:
            raise SystemExit(f"Parent source-lock file mismatch: {target}")
    lock["source_lock_sha256"] = recorded
    return lock


def verify_checksum_manifest(path: Path, expected_count: int) -> list[Path]:
    targets: list[Path] = []
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
        raise SystemExit(f"Checksum count mismatch for {path}: {len(targets)} != {expected_count}")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    required = [PARENT_LOCK, SACCT, FAILURE_RESULTS, FAILURE_LOGS, *(Path(x) for x in OVERLAY_FILES)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing v2 lock input:\n" + "\n".join(missing))

    parent = verify_source_lock(PARENT_LOCK)
    parent_sha = str(parent["source_lock_sha256"])
    if parent_sha != "76ed029a3ab71309dc842e6fdadea9220cfd37da92564eb80bd8bb5a4e0cf912":
        raise SystemExit("Unexpected parent E1 production lock")
    if parent.get("hardware") != "NVIDIA H200" or parent.get("canary_scientific_values_inspected") is not False:
        raise SystemExit("Parent lock scientific boundary mismatch")

    with SACCT.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="|"))
    by_id = {row.get("JobID"): row for row in rows}
    main_row = by_id.get("340568", {})
    batch_row = by_id.get("340568.batch", {})
    step_row = by_id.get("340568.0", {})
    if main_row.get("State") != "FAILED" or main_row.get("ExitCode") != "137:0" or main_row.get("DerivedExitCode") != "0:9":
        raise SystemExit("Job 340568 main accounting evidence mismatch")
    if batch_row.get("State") != "FAILED" or batch_row.get("ExitCode") != "137:0":
        raise SystemExit("Job 340568 batch accounting evidence mismatch")
    if not step_row.get("State", "").startswith("CANCELLED") or step_row.get("ExitCode") != "0:9":
        raise SystemExit("Job 340568 step accounting evidence mismatch")
    if step_row.get("NNodes") != "2" or step_row.get("NTasks") != "16":
        raise SystemExit("Job 340568 allocation evidence mismatch")

    result_targets = verify_checksum_manifest(FAILURE_RESULTS, 78)
    log_targets = verify_checksum_manifest(FAILURE_LOGS, 35)
    receipt_names = {path.as_posix() for path in result_targets if "/_receipts/stress16/" in path.as_posix()}
    if receipt_names != EXPECTED_RECEIPTS:
        raise SystemExit(f"Unexpected job 340568 receipt set: {sorted(receipt_names)}")
    for name in sorted(EXPECTED_RECEIPTS):
        receipt = load_json(Path(name))
        if receipt.get("status") != "PASS" or receipt.get("wave") != "stress16" or receipt.get("result_count") != 12:
            raise SystemExit(f"Invalid completed-cell receipt: {name}")
        if receipt.get("source_lock_sha256") != parent_sha:
            raise SystemExit(f"Completed-cell receipt source-lock mismatch: {name}")

    locked = dict(parent["files"])
    additions = [PARENT_LOCK, SACCT, FAILURE_RESULTS, FAILURE_LOGS]
    additions.extend(Path(name) for name in OVERLAY_FILES)
    additions.extend(result_targets)
    additions.extend(log_targets)
    for path in additions:
        key = path.as_posix()
        digest = sha256_file(path)
        if key in locked and locked[key] != digest:
            raise SystemExit(f"Conflicting locked file identity: {key}")
        locked[key] = digest

    output = Path(args.output)
    if output.exists():
        existing = load_json(output)
        existing_sha = existing.pop("source_lock_sha256", None)
        if existing_sha != canonical_hash(existing):
            raise SystemExit("Existing v2 source-lock canonical hash mismatch")
        if existing.get("files") != locked:
            raise SystemExit("Existing v2 lock differs; preserve it and use a new version")
        print("ORG_H200_E1_PRODUCTION_LOCK_V2=REUSED_PASS")
        print(f"SOURCE_LOCK={output}")
        print(f"SOURCE_LOCK_SHA256={existing_sha}")
        return

    lock = {
        "schema_version": "gate2_v4_source_lock_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Outcome-blind H200 Core72 E1 with single-node fault-isolated launchers",
        "hardware": "NVIDIA H200",
        "parent_source_lock": PARENT_LOCK.as_posix(),
        "parent_source_lock_sha256": parent_sha,
        "failed_job_id": "340568",
        "failed_job_classification": "infrastructure_or_launcher_sigkill_no_worker_traceback",
        "failed_job_outputs_excluded_from_e1": True,
        "canary_scientific_values_inspected": False,
        "retry_results_root": "results/org_h200_execution_v1/gate2_v4_bridge_standard_h200_e1_core72_v2",
        "files": locked,
    }
    lock["source_lock_sha256"] = canonical_hash(lock)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print("ORG_H200_E1_PRODUCTION_LOCK_V2=PASS")
    print(f"SOURCE_LOCK={output}")
    print(f"SOURCE_LOCK_SHA256={lock['source_lock_sha256']}")
    print(f"PARENT_SOURCE_LOCK_SHA256={parent_sha}")
    print(f"FAILED_RESULT_EVIDENCE={len(result_targets)}")
    print(f"FAILED_LOG_EVIDENCE={len(log_targets)}")


if __name__ == "__main__":
    main()
