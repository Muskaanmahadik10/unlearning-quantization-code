#!/usr/bin/env python3
"""Hash every model/config/tokenizer byte used by the frozen H200 Core72 panel."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INCLUDE_PATTERNS = (
    "*.safetensors", "*.safetensors.index.json", "pytorch_model*.bin", "model*.bin",
    "config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json",
    "added_tokens.json", "merges.txt", "vocab.json", "vocab.txt", "*.model",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def required_files(checkpoint: Path) -> list[Path]:
    files = [
        path for path in checkpoint.rglob("*")
        if path.is_file() and any(fnmatch.fnmatch(path.name, pattern) for pattern in INCLUDE_PATTERNS)
    ]
    if not any(path.name == "config.json" for path in files):
        raise SystemExit(f"Missing config.json: {checkpoint}")
    if not any(path.suffix == ".safetensors" or path.name.startswith("pytorch_model") or path.name.startswith("model") and path.suffix == ".bin" for path in files):
        raise SystemExit(f"Missing model weight files: {checkpoint}")
    return sorted(files)


def checkpoint_dir(row: dict[str, str]) -> Path:
    return Path("checkpoints") / row["model_tag"] / "unlearned" / row["algo"] / f"seed_{row['seed']}"


def build(cells_path: Path) -> dict[str, Any]:
    with cells_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 72:
        raise SystemExit(f"Expected 72 cells, found {len(rows)}")
    checkpoints = sorted({checkpoint_dir(row).as_posix() for row in rows})
    entries = []
    total_bytes = 0
    for index, name in enumerate(checkpoints, 1):
        root = Path(name)
        if not root.is_dir():
            raise SystemExit(f"Missing checkpoint directory: {root}")
        files = []
        for path in required_files(root):
            size = path.stat().st_size
            files.append({"path": path.as_posix(), "size": size, "sha256": sha256_file(path)})
            total_bytes += size
        entries.append({"checkpoint_dir": root.as_posix(), "files": files})
        print(f"CHECKPOINT_HASHED={index}/{len(checkpoints)} {root}", flush=True)
    ledger = {
        "schema_version": "org_h200_checkpoint_identity_ledger_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cells_tsv": cells_path.as_posix(),
        "cells_tsv_sha256": sha256_file(cells_path),
        "checkpoint_count": len(entries),
        "file_count": sum(len(entry["files"]) for entry in entries),
        "total_bytes": total_bytes,
        "checkpoints": entries,
    }
    ledger["ledger_sha256"] = canonical_hash(ledger)
    return ledger


def verify(ledger_path: Path) -> None:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    expected_hash = ledger.pop("ledger_sha256", None)
    if expected_hash != canonical_hash(ledger):
        raise SystemExit("Ledger canonical hash mismatch")
    failures = []
    for checkpoint in ledger["checkpoints"]:
        for record in checkpoint["files"]:
            path = Path(record["path"])
            if not path.is_file():
                failures.append(f"missing:{path}")
            elif path.stat().st_size != record["size"]:
                failures.append(f"size:{path}")
            elif sha256_file(path) != record["sha256"]:
                failures.append(f"sha256:{path}")
    if failures:
        raise SystemExit("Checkpoint ledger verification failed:\n" + "\n".join(failures[:50]))
    print("ORG_H200_CHECKPOINT_LEDGER_VERIFY=PASS")
    print(f"CHECKPOINTS={ledger['checkpoint_count']}")
    print(f"FILES={ledger['file_count']}")
    print(f"TOTAL_BYTES={ledger['total_bytes']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", required=True)
    parser.add_argument("--output")
    parser.add_argument("--verify")
    args = parser.parse_args()
    if bool(args.output) == bool(args.verify):
        raise SystemExit("Choose exactly one of --output or --verify")
    if args.verify:
        verify(Path(args.verify))
        return
    ledger = build(Path(args.cells))
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Checkpoint ledger already exists; preserve it and use --verify")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print("ORG_H200_CHECKPOINT_LEDGER_BUILD=PASS")
    print(f"CHECKPOINTS={ledger['checkpoint_count']}")
    print(f"FILES={ledger['file_count']}")
    print(f"TOTAL_BYTES={ledger['total_bytes']}")
    print(f"LEDGER_SHA256={ledger['ledger_sha256']}")
    print(f"OUTPUT={output}")


if __name__ == "__main__":
    main()
