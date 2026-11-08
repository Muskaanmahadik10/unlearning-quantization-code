#!/usr/bin/env python3
"""Check every repository file against FILE_MANIFEST.csv (size and SHA-256).

Read-only; standard library only. Exit status 0 means every listed file matches
and no unlisted files are present.
"""
import csv
import hashlib
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
with open(root / "FILE_MANIFEST.csv", encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))

problems = 0
listed = set()
for r in rows:
    listed.add(r["path"])
    p = root / r["path"]
    if not p.is_file():
        print("MISSING", r["path"])
        problems += 1
        continue
    data = p.read_bytes()
    if len(data) != int(r["bytes"]) or hashlib.sha256(data).hexdigest() != r["sha256"]:
        print("CHANGED", r["path"])
        problems += 1

unlisted = sorted(
    p.relative_to(root).as_posix()
    for p in root.rglob("*")
    if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts and p.name != "FILE_MANIFEST.csv"
    and p.relative_to(root).as_posix() not in listed
)
for u in unlisted:
    print("NOT IN MANIFEST", u)

print(f"{len(rows)} files checked: {'all match' if not problems else f'{problems} problem(s)'}; "
      f"{len(unlisted)} unlisted file(s)")
sys.exit(1 if problems or unlisted else 0)
