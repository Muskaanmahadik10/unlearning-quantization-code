#!/usr/bin/env python3
"""Build the source lock for the H200 E1 srun-wait repair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PARENT_LOCK = Path("org_h200_execution_v1/locks/ORG_H200_E1_PRODUCTION_LOCK_V2.json")
SACCT = Path("results/org_h200_execution_v1/certification/jobs340595_340596_sacct.txt")
FAILURE_RESULTS = Path("results/org_h200_execution_v1/certification/jobs340595_340596_failure_artifacts.sha256")
FAILURE_LOGS = Path("results/org_h200_execution_v1/certification/jobs340595_340596_logs.sha256")
OVERLAY_FILES = (
    "org_h200_execution_v1/H200_E1_SRUN_WAITTIME_AMENDMENT_V3.md",
    "org_h200_execution_v1/scripts/build_org_h200_e1_production_lock_v3.py",
    "org_h200_execution_v1/slurm/org_h200_srun_wait0_canary_v3.slurm",
    "org_h200_execution_v1/slurm/org_h200_e1_stress8_v3.slurm",
    "org_h200_execution_v1/slurm/org_h200_e1_remaining8_v3.slurm",
)
EXPECTED_PARENT_SHA = "dd1528d1d5378fb9a85353ae0960685862b15623291f41c4991fb3be949712e0"
EXPECTED_JOB_IDS = ("340595", "340596")


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


def verify_failed_job_rows(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="|"))
    by_id = {row.get("JobID"): row for row in rows}
    for job_id in EXPECTED_JOB_IDS:
        main = by_id.get(job_id, {})
        batch = by_id.get(f"{job_id}.batch", {})
        step = by_id.get(f"{job_id}.0", {})
        if main.get("State") != "FAILED" or main.get("ExitCode") != "137:0" or main.get("DerivedExitCode") != "0:9":
            raise SystemExit(f"Job {job_id} main accounting evidence mismatch")
        if main.get("NNodes") != "1":
            raise SystemExit(f"Job {job_id} node-count evidence mismatch")
        if batch.get("State") != "FAILED" or batch.get("ExitCode") != "137:0":
            raise SystemExit(f"Job {job_id} batch accounting evidence mismatch")
        if not step.get("State", "").startswith("CANCELLED") or step.get("ExitCode") != "0:9":
            raise SystemExit(f"Job {job_id} step accounting evidence mismatch")
        if step.get("NNodes") != "1" or step.get("NTasks") != "8":
            raise SystemExit(f"Job {job_id} task-allocation evidence mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    required = [PARENT_LOCK, SACCT, FAILURE_RESULTS, FAILURE_LOGS, *(Path(x) for x in OVERLAY_FILES)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing v3 lock input:\n" + "\n".join(missing))

    parent = verify_source_lock(PARENT_LOCK)
    parent_sha = str(parent["source_lock_sha256"])
    if parent_sha != EXPECTED_PARENT_SHA:
        raise SystemExit("Unexpected parent E1 V2 production lock")
    if parent.get("hardware") != "NVIDIA H200" or parent.get("canary_scientific_values_inspected") is not False:
        raise SystemExit("Parent lock scientific boundary mismatch")

    verify_failed_job_rows(SACCT)
    result_targets = verify_checksum_manifest(FAILURE_RESULTS, 100)
    log_targets = verify_checksum_manifest(FAILURE_LOGS, 36)

    receipts = [path for path in result_targets if "/_receipts/stress16/" in path.as_posix()]
    if len(receipts) != 6:
        raise SystemExit(f"Expected six completed-cell receipts, found {len(receipts)}")
    for path in receipts:
        receipt = load_json(path)
        if receipt.get("status") != "PASS" or receipt.get("wave") != "stress16" or receipt.get("result_count") != 12:
            raise SystemExit(f"Invalid completed-cell receipt: {path}")
        if receipt.get("source_lock_sha256") != parent_sha:
            raise SystemExit(f"Completed-cell receipt source-lock mismatch: {path}")

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
            raise SystemExit("Existing v3 source-lock canonical hash mismatch")
        if existing.get("files") != locked:
            raise SystemExit("Existing v3 lock differs; preserve it and use a new version")
        print("ORG_H200_E1_PRODUCTION_LOCK_V3=REUSED_PASS")
        print(f"SOURCE_LOCK={output}")
        print(f"SOURCE_LOCK_SHA256={existing_sha}")
        return

    lock = {
        "schema_version": "gate2_v4_source_lock_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Outcome-blind H200 Core72 E1 with unlimited independent-task srun wait",
        "hardware": "NVIDIA H200",
        "parent_source_lock": PARENT_LOCK.as_posix(),
        "parent_source_lock_sha256": parent_sha,
        "failed_job_ids": list(EXPECTED_JOB_IDS),
        "failed_job_classification": "slurm_waittime_killed_valid_heterogeneous_workers",
        "failed_job_outputs_excluded_from_e1": True,
        "canary_scientific_values_inspected": False,
        "scheduler_canary_required": True,
        "scheduler_canary_contract": "eight_unequal_duration_tasks_all_complete_with_srun_wait_0",
        "retry_results_root": "results/org_h200_execution_v1/gate2_v4_bridge_standard_h200_e1_core72_v3",
        "files": locked,
    }
    lock["source_lock_sha256"] = canonical_hash(lock)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print("ORG_H200_E1_PRODUCTION_LOCK_V3=PASS")
    print(f"SOURCE_LOCK={output}")
    print(f"SOURCE_LOCK_SHA256={lock['source_lock_sha256']}")
    print(f"PARENT_SOURCE_LOCK_SHA256={parent_sha}")
    print(f"FAILED_RESULT_EVIDENCE={len(result_targets)}")
    print(f"FAILED_LOG_EVIDENCE={len(log_targets)}")
    print(f"COMPLETED_RECEIPTS_QUARANTINED={len(receipts)}")


if __name__ == "__main__":
    main()
