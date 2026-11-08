#!/usr/bin/env python3
"""Build the immutable production lock for the H200 Core72 E1 study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FILES = (
    "org_h200_execution_v1/PREREGISTRATION_AMENDMENT_H200_PRIMARY_V1.md",
    "org_h200_execution_v1/ORG_H200_E1_SCIENTIFIC_CONTRACT_V1.md",
    "org_h200_execution_v1/H200_E1_PREPRODUCTION_RUNBOOK_V1.md",
    "org_h200_execution_v1/scripts/build_org_h200_core72_manifest_v2.py",
    "org_h200_execution_v1/scripts/build_checkpoint_identity_ledger_v1.py",
    "org_h200_execution_v1/scripts/certify_checkpoint_ledger_v1.py",
    "org_h200_execution_v1/scripts/run_org_h200_e1_cell_v1.py",
    "org_h200_execution_v1/scripts/validate_org_h200_e1_wave_v1.py",
    "org_h200_execution_v1/scripts/build_org_h200_e1_production_lock_v1.py",
    "org_h200_execution_v1/slurm/org_h200_e1_stress16_v1.slurm",
    "org_gb200_execution_v1/scripts/run_org_bridge_bundle_v2.py",
    "org_gb200_execution_v1/scripts/org_gptq_shared_cpu_finalize_collector_v3.py",
    "org_gb200_execution_v1/scripts/validate_org_bundled_canary_v1.py",
    "org_gb200_execution_v1/preregistration/core72_v1/org_core72_manifest_v1.json",
    "org_gb200_execution_v1/preregistration/core72_v1/org_core72_cells.tsv",
    "org_gb200_execution_v1/preregistration/core72_v1/org_core72_rankings_v1.jsonl",
    "org_gb200_execution_v1/preregistration/core72_v1/output_sha256.json",
    "org_h200_execution_v1/preregistration/core72_h200_v2/org_h200_core72_manifest_v2.json",
    "org_h200_execution_v1/preregistration/core72_h200_v2/org_h200_core72_cells_v2.tsv",
    "org_h200_execution_v1/locks/ORG_H200_CORE72_CHECKPOINT_LEDGER_V1.json",
    "org_h200_execution_v1/locks/ORG_H200_CORE72_CHECKPOINT_CERTIFICATION_V1.json",
    "org_h200_execution_v1/locks/ORG_H200_CORE72_PREPRODUCTION_V1.sha256",
    "org_h200_execution_v1/locks/ORG_H200_ADVERSARIAL_CANARY8_LOCK_V2.json",
    "results/org_h200_execution_v1/parity/h200_adversarial_canary_validation_v2_340505.json",
    "results/org_h200_execution_v1/parity/h200_canary_result_sha256_v2_340505.txt",
    "results/org_h200_execution_v1/certification/job340505_sacct.txt",
    "results/org_h200_execution_v1/certification/job340505_certification.sha256",
    "gate2_v4_final/scripts/run_gate2_whole_model_bridge_v4.py",
    "gate2_v4_final/scripts/gate2_v4_backend.py",
    "gate2_v4_final/scripts/gate2_v4_contract.py",
    "gate2_v4_final/scripts/gate2_v4_dataset.py",
    "gate2_v4_final/scripts/run_gate2_ground_truth_v4.py",
    "gate2_v4_final/scripts/validate_bridge_v4.py",
    "manifests/gate2_v4/tofu_forget_core128_seed20260625.json",
    "manifests/gate2_v4/tofu_retain_gptq_calib128_seed20260625.json",
    "manifests/gate2_v4/muse_news_forget_core128_seed20260625.json",
    "manifests/gate2_v4/muse_news_retain_gptq_calib128_seed20260625.json",
    "org_h200_execution_v1/slurm/org_h200_e1_remaining32_v1.slurm",
    "org_h200_execution_v1/slurm/org_h200_e1_remaining8_v1.slurm",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return value


def verify_checksum_manifest(path: Path) -> int:
    count = 0
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
        count += 1
    if not count:
        raise SystemExit(f"Empty checksum manifest: {path}")
    return count


def rss_bytes(text: str) -> int:
    match = re.fullmatch(r"([0-9.]+)([KMGTP]?)", text.strip(), re.IGNORECASE)
    if not match:
        raise SystemExit(f"Cannot parse MaxRSS: {text}")
    value, suffix = match.groups()
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
    return int(float(value) * scale[suffix.upper()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [Path(name) for name in FILES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Missing production-lock inputs:\n" + "\n".join(missing))

    manifest = load_json(Path(FILES[17]))
    embedded_hash = manifest.pop("manifest_sha256", None)
    if embedded_hash != canonical_hash(manifest):
        raise SystemExit("H200 Core72 manifest canonical hash mismatch")
    if manifest.get("expected_cells") != 72 or len(manifest.get("stress16_cell_ids", [])) != 16 or len(manifest.get("remaining56_cell_ids", [])) != 56:
        raise SystemExit("H200 Core72 manifest counts mismatch")
    if not manifest.get("outcome_blind") or not manifest.get("canary_results_excluded_from_e1"):
        raise SystemExit("H200 Core72 outcome-blind boundary missing")

    with Path(FILES[18]).open("r", encoding="utf-8", newline="") as handle:
        cell_rows = list(csv.DictReader(handle, delimiter="\t"))
    manifest_cell_ids = set(manifest["stress16_cell_ids"]) | set(manifest["remaining56_cell_ids"])
    table_cell_ids = {row["cell_id"] for row in cell_rows}
    if len(cell_rows) != 72 or len(table_cell_ids) != 72 or table_cell_ids != manifest_cell_ids:
        raise SystemExit("H200 Core72 cell table does not match the frozen manifest")
    expected_checkpoint_dirs = {
        f"checkpoints/{row['model_tag']}/unlearned/{row['algo']}/seed_{row['seed']}"
        for row in cell_rows
    }

    ledger_path = Path(FILES[19])
    ledger = load_json(ledger_path)
    ledger_hash = ledger.pop("ledger_sha256", None)
    if ledger_hash != canonical_hash(ledger):
        raise SystemExit("Checkpoint ledger canonical hash mismatch")
    if ledger.get("checkpoint_count") != 40 or ledger.get("file_count") != 400 or ledger.get("total_bytes") != 588580226924:
        raise SystemExit("Checkpoint ledger inventory mismatch")
    ledger_checkpoint_dirs = {entry.get("checkpoint_dir") for entry in ledger.get("checkpoints", [])}
    if len(ledger_checkpoint_dirs) != 40 or ledger_checkpoint_dirs != expected_checkpoint_dirs:
        missing = sorted(expected_checkpoint_dirs - ledger_checkpoint_dirs)
        extra = sorted(ledger_checkpoint_dirs - expected_checkpoint_dirs)
        raise SystemExit(f"Checkpoint ledger/cell-table mismatch: missing={missing} extra={extra}")
    receipt = load_json(Path(FILES[20]))
    if receipt.get("status") != "PASS" or receipt.get("ledger_file_sha256") != sha256_file(ledger_path):
        raise SystemExit("Checkpoint ledger certification mismatch")
    if receipt.get("checkpoint_count") != 40 or receipt.get("file_count") != 400 or receipt.get("total_bytes") != 588580226924:
        raise SystemExit("Checkpoint certification totals mismatch")
    if receipt.get("verifier_sha256") != sha256_file(Path(FILES[5])):
        raise SystemExit("Checkpoint verifier identity mismatch")

    preproduction_count = verify_checksum_manifest(Path(FILES[21]))
    canary_result_count = verify_checksum_manifest(Path(FILES[24]))
    certification_count = verify_checksum_manifest(Path(FILES[26]))
    if preproduction_count != 3 or canary_result_count != 36 or certification_count != 6:
        raise SystemExit("Certification checksum counts mismatch")
    canary = load_json(Path(FILES[23]))
    if canary.get("status") != "PASS" or canary.get("failures") or canary.get("bundled_result_count") != 18 or canary.get("legacy_result_count") != 12 or canary.get("repeat_result_count") != 6:
        raise SystemExit("H200 adversarial canary validation mismatch")

    with Path(FILES[25]).open("r", encoding="utf-8", newline="") as handle:
        sacct_rows = list(csv.DictReader(handle, delimiter="|"))
    main_rows = [row for row in sacct_rows if row.get("JobID") == "340505"]
    batch_rows = [row for row in sacct_rows if row.get("JobID") == "340505.batch"]
    if len(main_rows) != 1 or main_rows[0].get("State") != "COMPLETED" or main_rows[0].get("ExitCode") != "0:0":
        raise SystemExit("Canary Slurm terminal state mismatch")
    if len(batch_rows) != 1 or rss_bytes(batch_rows[0].get("MaxRSS", "")) > 595 * 1024**3:
        raise SystemExit("Canary memory-reserve gate failed")

    locked = {path.as_posix(): sha256_file(path) for path in paths}
    output = Path(args.output)
    if output.exists():
        existing = load_json(output)
        if existing.get("files") != locked:
            raise SystemExit("Existing E1 production lock differs; preserve it and use a new version")
        print("ORG_H200_E1_PRODUCTION_LOCK_V1=REUSED_PASS")
        print(f"SOURCE_LOCK={output}")
        print(f"SOURCE_LOCK_SHA256={existing['source_lock_sha256']}")
        return
    lock = {
        "schema_version": "gate2_v4_source_lock_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Outcome-blind H200 Core72 E1 production with certified runtime and checkpoint bytes",
        "hardware": "NVIDIA H200",
        "manifest_sha256": embedded_hash,
        "checkpoint_ledger_sha256": ledger_hash,
        "canary_job_id": "340505",
        "canary_scientific_values_inspected": False,
        "files": locked,
    }
    payload = json.dumps(lock, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    lock["source_lock_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print("ORG_H200_E1_PRODUCTION_LOCK_V1=PASS")
    print(f"SOURCE_LOCK={output}")
    print(f"SOURCE_LOCK_SHA256={lock['source_lock_sha256']}")
    print(f"MANIFEST_SHA256={embedded_hash}")
    print(f"CHECKPOINT_LEDGER_SHA256={ledger_hash}")


if __name__ == "__main__":
    main()
