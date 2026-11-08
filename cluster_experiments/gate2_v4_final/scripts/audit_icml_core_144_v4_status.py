#!/usr/bin/env python3
"""Strict status audit for the Gate 2 v4 ICML core."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gate2_v4_contract import SEEDS, core_cell_id, iter_core_cells, load_json, load_source_lock, source_lock_hash  # noqa: E402
from validate_gate2_v4_cell_join import validate_pair  # noqa: E402
from validate_gt_v4 import validate_obj as validate_gt_obj  # noqa: E402
from validate_upqi_v4 import validate_obj as validate_upqi_obj  # noqa: E402


def source_lock_matches_current(obj: Mapping[str, Any], expected_source_lock_sha: str | None) -> bool:
    if not expected_source_lock_sha:
        return True
    return obj.get("provenance", {}).get("source_lock_sha256") == expected_source_lock_sha


def complete_gt(path: Path, expected_source_lock_sha: str | None) -> Tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        obj = load_json(path)
        report = validate_gt_obj(obj, path)
    except Exception as exc:
        return False, f"bad_json:{exc}"
    if report["passed"] and not source_lock_matches_current(obj, expected_source_lock_sha):
        return False, "dirty_complete:source_lock_not_current"
    return (True, "clean_complete") if report["passed"] else (False, "dirty_complete:" + ",".join(report["errors"][:5]))


def complete_upqi(path: Path, expected_source_lock_sha: str | None) -> Tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        obj = load_json(path)
        report = validate_upqi_obj(obj, path)
    except Exception as exc:
        return False, f"bad_json:{exc}"
    if report["passed"] and not source_lock_matches_current(obj, expected_source_lock_sha):
        return False, "dirty_complete:source_lock_not_current"
    return (True, "clean_complete") if report["passed"] else (False, "dirty_complete:" + ",".join(report["errors"][:5]))


def partial_status(path: Path, stage: str) -> Tuple[str, str]:
    if not path.exists():
        return "", ""
    try:
        obj = load_json(path)
    except Exception:
        return "partial", "unreadable"
    if stage == "gt":
        done = obj.get("n_layers_completed", len(obj.get("layers", {})) if isinstance(obj.get("layers"), Mapping) else 0)
    else:
        done = obj.get("n_layers_completed", len(obj.get("per_layer", [])) if isinstance(obj.get("per_layer"), list) else 0)
    return "partial", f"{done}/{obj.get('n_layers', '?')}"


def row_for(root: Path, cell: Mapping[str, str], expected_source_lock_sha: str | None) -> Dict[str, str]:
    cell_dir = root / cell["cell_id"]
    gt_path = cell_dir / "ground_truth_v4.json"
    upqi_path = cell_dir / "upqi_v4.json"
    gt_ok, gt_reason = complete_gt(gt_path, expected_source_lock_sha)
    upqi_ok, upqi_reason = complete_upqi(upqi_path, expected_source_lock_sha)
    gt_partial_path = cell_dir / "ground_truth_v4.json.partial"
    upqi_partial_path = cell_dir / "upqi_v4.json.partial"
    gt_status = (
        "clean_complete"
        if gt_ok
        else "dirty_complete"
        if gt_path.exists()
        else "partial"
        if gt_partial_path.exists()
        else "missing"
    )
    upqi_status = (
        "clean_complete"
        if upqi_ok
        else "dirty_complete"
        if upqi_path.exists()
        else "partial"
        if upqi_partial_path.exists()
        else "missing"
    )
    if gt_status == "partial":
        _, gt_reason = partial_status(gt_partial_path, "gt")
    if upqi_status == "partial":
        _, upqi_reason = partial_status(upqi_partial_path, "upqi")
    join_status = "not_ready"
    join_reason = "gt_or_upqi_incomplete"
    if gt_ok and upqi_ok:
        join = validate_pair(gt_path, upqi_path)
        join_status = "passed" if join["passed"] else "failed"
        join_reason = "strict_join_passed" if join["passed"] else ",".join(join["errors"][:8])
    return {
        **cell,
        "gt_status": gt_status,
        "gt_reason": gt_reason,
        "gt_path": str(gt_path),
        "upqi_status": upqi_status,
        "upqi_reason": upqi_reason,
        "upqi_path": str(upqi_path),
        "join_status": join_status,
        "join_reason": join_reason,
        "next_action": "lock_for_metrics" if join_status == "passed" else ("submit_gt_v4" if not gt_ok else "submit_upqi_v4" if not upqi_ok else "fix_join_contract"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Gate 2 v4 core-144 status")
    parser.add_argument("--results_root", default="results/gate2_v4_standard")
    parser.add_argument("--source_lock_path", default="gate2_v4_final/V4_SOURCE_LOCK.json")
    parser.add_argument("--seeds", default=",".join(SEEDS))
    parser.add_argument("--out", default="icml_core_144_v4_status.tsv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [x.strip() for x in args.seeds.split(",") if x.strip()]
    expected_source_lock_sha = None
    if args.source_lock_path:
        lock = load_source_lock(args.source_lock_path)
        expected_source_lock_sha = lock.get("source_lock_sha256", source_lock_hash(lock))
    rows = [row_for(Path(args.results_root), cell, expected_source_lock_sha) for cell in iter_core_cells(seeds)]
    fields = [
        "cell_id", "label", "model_tag", "dataset", "algo", "seed", "quantizer",
        "gt_status", "gt_reason", "gt_path", "upqi_status", "upqi_reason", "upqi_path",
        "join_status", "join_reason", "next_action",
    ]
    with Path(args.out).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print("=== Gate 2 v4 core-144 status ===")
    print(f"rows: {len(rows)}")
    print(f"GT: {dict(Counter(r['gt_status'] for r in rows))}")
    print(f"UPQI: {dict(Counter(r['upqi_status'] for r in rows))}")
    print(f"join: {dict(Counter(r['join_status'] for r in rows))}")
    print(f"next_action: {dict(Counter(r['next_action'] for r in rows))}")
    print(f"wrote: {args.out}")
    by_setting = defaultdict(Counter)
    for row in rows:
        by_setting[row["label"]][row["next_action"]] += 1
    for label, counts in sorted(by_setting.items()):
        print(label, dict(counts))


if __name__ == "__main__":
    main()
