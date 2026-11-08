#!/usr/bin/env python3
"""Validate joins among v4 bridge, GT, and UPQI outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gate2_v4_contract import load_json  # noqa: E402
from validate_bridge_v4 import validate_obj as validate_bridge_obj  # noqa: E402
from validate_gt_v4 import validate_obj as validate_gt_obj  # noqa: E402
from validate_upqi_v4 import validate_obj as validate_upqi_obj  # noqa: E402

LEGACY_FRAGMENTS = ("gate2_v2", "gate2_v3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge_root", default="results/gate2_v4_bridge_standard")
    parser.add_argument("--v4_results_root", default="results/gate2_v4_standard")
    parser.add_argument("--out", default="bridge_v4_join_audit.tsv")
    parser.add_argument(
        "--expected_source_lock_sha256",
        default=None,
        help="Backward-compatible single-lock mode. Prefer the bridge/v4-specific lock flags.",
    )
    parser.add_argument("--expected_bridge_source_lock_sha256", default=None)
    parser.add_argument("--expected_v4_source_lock_sha256", default=None)
    parser.add_argument("--bf16_tolerance", type=float, default=1e-5)
    return parser.parse_args()


def result_cell_dir(results_root: Path, bridge: Mapping[str, Any]) -> Path:
    return results_root / f"{bridge['model_tag']}_{bridge['algo']}_seed{bridge['seed']}" / str(bridge["quantizer"])


def bridge_files(root: Path) -> Iterable[Path]:
    yield from root.glob("*/*/*/bridge_v4.json")


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def metric(obj: Mapping[str, Any], dotted: str) -> Optional[float]:
    current: Any = obj
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return float(current) if finite(current) else None


def gt_bf16_metric(gt: Mapping[str, Any], key: str) -> Optional[float]:
    candidates = [
        f"bf16_baseline.{key}",
        f"bf16_reference.forget.{key}",
        f"bf16_forget.{key}",
    ]
    for dotted in candidates:
        value = metric(gt, dotted)
        if value is not None:
            return value
    return None


def manifests_match(bridge: Mapping[str, Any], other: Mapping[str, Any], key: str) -> bool:
    b = bridge.get(key)
    if key == "forget_manifest":
        candidates = [other.get("forget_manifest"), other.get("sample_manifest"), other.get("manifest")]
    else:
        candidates = [other.get(key)]
    if not isinstance(b, Mapping):
        return False
    bsha = b.get("sha256") or b.get("manifest_sha256")
    for cand in candidates:
        if isinstance(cand, Mapping):
            osha = cand.get("sha256") or cand.get("manifest_sha256")
            if bsha and osha and bsha == osha:
                return True
    return False


def compare_bf16_to_gt(bridge: Mapping[str, Any], gt: Mapping[str, Any], tol: float) -> List[str]:
    issues: List[str] = []
    pairs = [
        ("answer_ce", "bf16_reference.forget.answer_ce"),
        ("answer_logprob", "bf16_reference.forget.answer_logprob"),
        ("truth_ratio_log", "bf16_reference.forget.truth_ratio_log"),
    ]
    for gt_key, bridge_key in pairs:
        g = gt_bf16_metric(gt, gt_key)
        b = metric(bridge, bridge_key)
        if g is None or b is None:
            issues.append(f"bf16_{gt_key}_missing")
        elif abs(g - b) > tol:
            issues.append(f"bf16_{gt_key}_mismatch")
    return issues


def main() -> None:
    args = parse_args()
    bridge_root = Path(args.bridge_root)
    v4_root = Path(args.v4_results_root)
    if any(frag in str(bridge_root) for frag in LEGACY_FRAGMENTS):
        raise SystemExit(f"Refusing legacy bridge_root: {bridge_root}")
    if any(frag in str(v4_root) for frag in LEGACY_FRAGMENTS):
        raise SystemExit(f"Refusing legacy v4_results_root: {v4_root}")
    expected_bridge_lock = args.expected_bridge_source_lock_sha256 or args.expected_source_lock_sha256
    expected_v4_lock = args.expected_v4_source_lock_sha256 or args.expected_source_lock_sha256

    rows: List[Dict[str, Any]] = []
    hard_fail = 0
    for path in sorted(bridge_files(bridge_root)):
        bridge = load_json(path)
        issues = validate_bridge_obj(bridge, path=path, expected_source_lock_sha256=expected_bridge_lock)
        cell_dir = result_cell_dir(v4_root, bridge)
        upqi_path = cell_dir / "upqi_v4.json"
        gt_path = cell_dir / "ground_truth_v4.json"
        gt_status = "missing"
        upqi_status = "missing"
        if upqi_path.exists():
            upqi = load_json(upqi_path)
            upqi_report = validate_upqi_obj(upqi, path=upqi_path)
            upqi_errors = list(upqi_report.get("errors", [])) if isinstance(upqi_report, Mapping) else ["upqi_validator_bad_report"]
            if expected_v4_lock and upqi.get("provenance", {}).get("source_lock_sha256") != expected_v4_lock:
                upqi_errors.append("source_lock_mismatch")
            if upqi_errors:
                issues.extend(f"upqi_{e}" for e in upqi_errors)
                upqi_status = "dirty"
            else:
                upqi_status = "clean"
            if not manifests_match(bridge, upqi, "forget_manifest"):
                issues.append("bridge_upqi_forget_manifest_mismatch")
        else:
            issues.append("upqi_missing")
        if gt_path.exists():
            gt = load_json(gt_path)
            gt_report = validate_gt_obj(gt, path=gt_path)
            gt_errors = list(gt_report.get("errors", [])) if isinstance(gt_report, Mapping) else ["gt_validator_bad_report"]
            if expected_v4_lock and gt.get("provenance", {}).get("source_lock_sha256") != expected_v4_lock:
                gt_errors.append("source_lock_mismatch")
            if gt_errors:
                gt_status = "dirty"
                issues.extend(f"gt_{e}" for e in gt_errors)
            else:
                gt_status = "clean"
            if not manifests_match(bridge, gt, "forget_manifest"):
                issues.append("bridge_gt_forget_manifest_mismatch")
            issues.extend(compare_bf16_to_gt(bridge, gt, args.bf16_tolerance))
            if str(bridge.get("ranking_source", "")).startswith("GT_oracle") and gt_status != "clean":
                issues.append("gt_oracle_ranking_without_clean_gt")
        else:
            gt_status = "missing"
            if str(bridge.get("ranking_source", "")).startswith("GT_oracle"):
                issues.append("gt_oracle_ranking_gt_missing")
        status = "PASS" if not issues else "FAIL"
        if issues:
            hard_fail += 1
        rows.append(
            {
                "status": status,
                "cell_id": bridge.get("cell_id", ""),
                "condition": bridge.get("condition", ""),
                "ranking_source": bridge.get("ranking_source", ""),
                "k": bridge.get("k", ""),
                "gt_status": gt_status,
                "upqi_status": upqi_status,
                "issues": ";".join(sorted(set(issues))),
                "bridge_path": str(path),
                "gt_path": str(gt_path) if gt_path.exists() else "",
                "upqi_path": str(upqi_path) if upqi_path.exists() else "",
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "status",
            "cell_id",
            "condition",
            "ranking_source",
            "k",
            "gt_status",
            "upqi_status",
            "issues",
            "bridge_path",
            "gt_path",
            "upqi_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {out} rows={len(rows)} fail={hard_fail}")
    if hard_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
