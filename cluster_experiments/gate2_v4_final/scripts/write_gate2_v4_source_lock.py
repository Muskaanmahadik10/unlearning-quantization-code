#!/usr/bin/env python3
"""Write V4_SOURCE_LOCK.json for scripts, Slurm files, and manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gate2_v4_contract import SOURCE_LOCK_SCHEMA, save_json, sha256_file, source_lock_hash  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Gate 2 v4 source lock")
    parser.add_argument("--base_dir", default="gate2_v4_final")
    parser.add_argument("--manifest_dir", default="manifests/gate2_v4")
    parser.add_argument("--output", default="gate2_v4_final/V4_SOURCE_LOCK.json")
    return parser.parse_args()


def collect_files(base_dir: Path, manifest_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for pattern in ("scripts/*.py", "slurm/gate2_v4/*.slurm", "slurm/gate2_v4/*.sh"):
        for path in sorted(base_dir.glob(pattern)):
            files[str(path).replace("\\", "/")] = sha256_file(path)
    if manifest_dir.exists():
        for path in sorted(manifest_dir.glob("*.json")):
            files[str(path).replace("\\", "/")] = sha256_file(path)
    return files


def main() -> None:
    args = parse_args()
    lock = {
        "schema_version": SOURCE_LOCK_SCHEMA,
        "policy": "v4 production refuses to run when script SHA differs from this lock",
        "base_dir": str(Path(args.base_dir)),
        "manifest_dir": str(Path(args.manifest_dir)),
        "files": collect_files(Path(args.base_dir), Path(args.manifest_dir)),
    }
    lock["source_lock_sha256"] = source_lock_hash(lock)
    save_json(args.output, lock)
    print(f"Wrote {args.output} files={len(lock['files'])} sha={lock['source_lock_sha256']}")


if __name__ == "__main__":
    main()

