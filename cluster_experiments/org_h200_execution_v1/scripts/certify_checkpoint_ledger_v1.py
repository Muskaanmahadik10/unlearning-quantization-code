#!/usr/bin/env python3
"""Fully rehash an H200 checkpoint ledger and emit a lockable PASS receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    ledger_path = Path(args.ledger)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    recorded_ledger_hash = ledger.pop("ledger_sha256", None)
    if recorded_ledger_hash != canonical_hash(ledger):
        raise SystemExit("Ledger canonical hash mismatch")
    checked = 0
    checked_bytes = 0
    for checkpoint in ledger["checkpoints"]:
        for record in checkpoint["files"]:
            path = Path(record["path"])
            if not path.is_file():
                raise SystemExit(f"Missing checkpoint file: {path}")
            if path.stat().st_size != record["size"]:
                raise SystemExit(f"Checkpoint size mismatch: {path}")
            if sha256_file(path) != record["sha256"]:
                raise SystemExit(f"Checkpoint SHA mismatch: {path}")
            checked += 1
            checked_bytes += int(record["size"])
            if checked % 10 == 0:
                print(f"LEDGER_FILE_VERIFIED={checked}/{ledger['file_count']}", flush=True)
    if checked != ledger["file_count"] or checked_bytes != ledger["total_bytes"]:
        raise SystemExit("Ledger verification totals mismatch")
    receipt = {
        "schema_version": "org_h200_checkpoint_ledger_certification_v1",
        "status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "ledger_path": ledger_path.as_posix(),
        "ledger_file_sha256": sha256_file(ledger_path),
        "ledger_sha256": recorded_ledger_hash,
        "checkpoint_count": ledger["checkpoint_count"],
        "file_count": checked,
        "total_bytes": checked_bytes,
        "verifier_sha256": sha256_file(Path(__file__)),
    }
    output = Path(args.receipt)
    if output.exists():
        raise SystemExit("Verification receipt already exists; preserve it and use a new version")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print("ORG_H200_CHECKPOINT_LEDGER_CERTIFICATION=PASS")
    print(f"CHECKPOINTS={receipt['checkpoint_count']}")
    print(f"FILES={checked}")
    print(f"TOTAL_BYTES={checked_bytes}")
    print(f"RECEIPT={output}")


if __name__ == "__main__":
    main()

