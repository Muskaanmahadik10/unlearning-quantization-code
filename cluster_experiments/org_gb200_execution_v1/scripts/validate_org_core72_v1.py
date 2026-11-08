#!/usr/bin/env python3
"""Fail-closed validation of frozen ORG Core72 preregistration inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash_without_self(obj: dict) -> str:
    clean = dict(obj)
    clean.pop("manifest_sha256", None)
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    root = Path(args.dir)
    manifest_path = root / "org_core72_manifest_v1.json"
    canary_path = root / "org_canary2_manifest_v1.json"
    rankings_path = root / "org_core72_rankings_v1.jsonl"
    hashes_path = root / "output_sha256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canary = json.loads(canary_path.read_text(encoding="utf-8"))
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))

    assert canonical_hash_without_self(manifest) == manifest["manifest_sha256"]
    assert canonical_hash_without_self(canary) == canary["manifest_sha256"]
    assert canary["parent_manifest_sha256"] == manifest["manifest_sha256"]
    for name, expected in hashes.items():
        assert sha256_file(root / name) == expected, name

    cells = manifest["cells"]
    assert len(cells) == 72 and len({row["cell_id"] for row in cells}) == 72
    assert len(canary["cells"]) == 2
    assert {row["cell_id"] for row in canary["cells"]} <= {row["cell_id"] for row in cells}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with rankings_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                grouped[(row["cell_id"], row["ranking_source"])].append(row)
    assert sum(map(len, grouped.values())) == 177408
    expected_sources = set(cells[0]["selectors"].split(";"))
    assert len(expected_sources) == 11
    for cell in cells:
        cid = cell["cell_id"]
        assert set(cell["selectors"].split(";")) == expected_sources
        native_top = None
        for source in expected_sources:
            rows = sorted(grouped[(cid, source)], key=lambda row: int(row["rank"]))
            assert len(rows) == 224
            assert [int(row["rank"]) for row in rows] == list(range(1, 225))
            assert len({row["layer_name"] for row in rows}) == 224
            if source == "GQDP_signed_v42":
                native_top = [row["layer_name"] for row in rows[:8]]
        assert native_top is not None
        # Controls cannot silently collapse to the native top-8 selection.
        for source in expected_sources:
            if source.startswith("matched_random_type_depth_v1_seed"):
                rows = sorted(grouped[(cid, source)], key=lambda row: int(row["rank"]))
                assert Counter(row["layer_name"] for row in rows[:8]) != Counter(native_top)

    print("ORG_CORE72_VALIDATION=PASS")
    print("CELLS=72")
    print("SELECTORS_PER_CELL=11")
    print("RANKING_ROWS=177408")
    print(f"MANIFEST_SHA256={manifest['manifest_sha256']}")
    print(f"CANARY_SHA256={canary['manifest_sha256']}")


if __name__ == "__main__":
    main()
