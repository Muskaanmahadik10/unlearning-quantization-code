#!/usr/bin/env python3
"""Strict GT/UPQI v4 cell join validator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gate2_v4_contract import load_json, save_json  # noqa: E402
from validate_gt_v4 import validate_obj as validate_gt_obj  # noqa: E402
from validate_upqi_v4 import validate_obj as validate_upqi_obj  # noqa: E402


def manifest_sha(obj: Mapping[str, Any], key: str) -> Any:
    block = obj.get(key)
    return block.get("manifest_sha256") if isinstance(block, Mapping) else None


def layer_names_gt(gt: Mapping[str, Any]) -> List[str]:
    target_names = gt.get("target_layer_names")
    if isinstance(target_names, list):
        return [name for name in target_names if isinstance(name, str) and name]
    return list(gt.get("layers", {}).keys()) if isinstance(gt.get("layers"), Mapping) else []


def layer_names_upqi(upqi: Mapping[str, Any]) -> List[str]:
    out = []
    for row in upqi.get("per_layer", []) if isinstance(upqi.get("per_layer"), list) else []:
        if isinstance(row, Mapping):
            out.append(str(row.get("layer")))
    return out


def validate_pair(gt_path: str | Path, upqi_path: str | Path) -> Dict[str, Any]:
    gt_path = Path(gt_path)
    upqi_path = Path(upqi_path)
    errors: List[str] = []
    if not gt_path.exists():
        errors.append("missing_gt")
        gt = None
    else:
        gt = load_json(gt_path)
        report = validate_gt_obj(gt, gt_path)
        errors.extend(f"gt:{err}" for err in report["errors"])
    if not upqi_path.exists():
        errors.append("missing_upqi")
        upqi = None
    else:
        upqi = load_json(upqi_path)
        report = validate_upqi_obj(upqi, upqi_path)
        errors.extend(f"upqi:{err}" for err in report["errors"])
    if gt and upqi:
        for key in ("model_dir", "dataset", "quantizer", "dtype", "n_samples"):
            if gt.get(key) != upqi.get(key):
                errors.append(f"{key}_mismatch")
        if manifest_sha(gt, "sample_manifest") != manifest_sha(upqi, "sample_manifest"):
            errors.append("sample_manifest_sha_mismatch")
        if gt.get("quantizer") == "gptq_int4":
            if manifest_sha(gt, "calibration_manifest") != manifest_sha(upqi, "calibration_manifest"):
                errors.append("calibration_manifest_sha_mismatch")
        if gt.get("loss_contract") != upqi.get("loss_contract"):
            errors.append("loss_contract_mismatch")
        if gt.get("provenance", {}).get("source_lock_sha256") != upqi.get("provenance", {}).get("source_lock_sha256"):
            errors.append("source_lock_mismatch")
        gt_layer_names = layer_names_gt(gt)
        upqi_layer_names = layer_names_upqi(upqi)
        gt_layers = gt.get("layers")
        if isinstance(gt_layers, Mapping):
            missing_gt_layers = [name for name in gt_layer_names if name not in gt_layers]
            extra_gt_layers = [str(name) for name in gt_layers.keys() if str(name) not in set(gt_layer_names)]
            if missing_gt_layers:
                errors.append("gt_missing_target_layers")
            if extra_gt_layers:
                errors.append("gt_extra_layers")
        if gt_layer_names != upqi_layer_names:
            errors.append("layer_order_mismatch")
        if int(gt.get("n_layers", -1)) != int(upqi.get("n_layers", -2)):
            errors.append("n_layers_mismatch")
    return {
        "gt_path": str(gt_path),
        "upqi_path": str(upqi_path),
        "passed": not errors,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one joined GT/UPQI v4 cell")
    parser.add_argument("--results_root", default="results/gate2_v4_standard")
    parser.add_argument("--cell")
    parser.add_argument("--gt_path")
    parser.add_argument("--upqi_path")
    parser.add_argument("--output_json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cell:
        cell_dir = Path(args.results_root) / args.cell
        gt_path = cell_dir / "ground_truth_v4.json"
        upqi_path = cell_dir / "upqi_v4.json"
    elif args.gt_path and args.upqi_path:
        gt_path = Path(args.gt_path)
        upqi_path = Path(args.upqi_path)
    else:
        raise SystemExit("Pass --cell or both --gt_path/--upqi_path")
    report = validate_pair(gt_path, upqi_path)
    if args.output_json:
        save_json(args.output_json, report)
    print(f"{'PASS' if report['passed'] else 'FAIL'} {gt_path.parent} errors={len(report['errors'])}")
    if not report["passed"]:
        for error in report["errors"][:40]:
            print(error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
