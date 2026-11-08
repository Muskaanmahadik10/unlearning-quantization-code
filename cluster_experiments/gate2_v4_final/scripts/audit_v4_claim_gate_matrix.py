#!/usr/bin/env python3
"""Build the Gate 2 v4 claim-gate matrix for paper readiness."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gate2_v4_contract import SEEDS, iter_core_cells, load_json  # noqa: E402
from validate_gate2_v4_cell_join import validate_pair  # noqa: E402
from validate_gt_v4 import validate_obj as validate_gt_obj  # noqa: E402
from validate_upqi_v4 import validate_obj as validate_upqi_obj  # noqa: E402


def stage_status(path: Path, partial_name: str, validate_fn: Any) -> Tuple[str, str]:
    if path.exists():
        try:
            report = validate_fn(load_json(path), path)
        except Exception as exc:
            return "dirty_complete", f"bad_json:{exc}"
        if report["passed"]:
            return "clean_complete", "validator_passed"
        return "dirty_complete", ";".join(report["errors"][:10])
    partial = path.with_name(partial_name)
    if partial.exists():
        try:
            obj = load_json(partial)
            done = obj.get("n_layers_completed")
            total = obj.get("n_layers")
            return "partial", f"{done}/{total}"
        except Exception as exc:
            return "partial", f"unreadable:{exc}"
    return "missing", "missing"


def official_metrics_status(metrics_root: Path, cell_id: str) -> Tuple[str, str]:
    if not metrics_root.exists():
        return "official_metrics_missing", "metrics_root_missing"
    cell_dir = metrics_root / cell_id
    candidates = [
        cell_dir / "official_metrics_v4.json",
        cell_dir / "metrics_v4_standard.json",
        cell_dir / "unlearning_quality_v4.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                obj = load_json(path)
            except Exception as exc:
                return "official_metrics_dirty", f"bad_json:{exc}"
            if obj.get("validation", {}).get("passed") is True or obj.get("passed") is True:
                return "official_metrics_passed", str(path)
            return "official_metrics_dirty", str(path)
    return "official_metrics_missing", "missing"


def row_for(results_root: Path, metrics_root: Path, cell: Mapping[str, str]) -> Dict[str, str]:
    cell_dir = results_root / cell["cell_id"]
    gt_path = cell_dir / "ground_truth_v4.json"
    upqi_path = cell_dir / "upqi_v4.json"
    gt_status, gt_reason = stage_status(gt_path, "ground_truth_v4.json.partial", validate_gt_obj)
    upqi_status, upqi_reason = stage_status(upqi_path, "upqi_v4.json.partial", validate_upqi_obj)
    join_status = "join_not_ready"
    join_reason = "gt_or_upqi_not_clean"
    if gt_status == "clean_complete" and upqi_status == "clean_complete":
        join = validate_pair(gt_path, upqi_path)
        join_status = "join_passed" if join["passed"] else "join_failed"
        join_reason = "strict_join_passed" if join["passed"] else ";".join(join["errors"][:10])
    metrics_status, metrics_reason = official_metrics_status(metrics_root, cell["cell_id"])
    paper_ready = (
        gt_status == "clean_complete"
        and upqi_status == "clean_complete"
        and join_status == "join_passed"
        and metrics_status == "official_metrics_passed"
    )
    if paper_ready:
        gate = "paper_ready"
    elif join_status == "join_passed":
        gate = metrics_status
    elif upqi_status == "clean_complete":
        gate = join_status
    elif gt_status == "clean_complete":
        gate = f"upqi_{upqi_status}"
    else:
        gate = f"gt_{gt_status}"
    return {
        **cell,
        "gt_status": gt_status,
        "gt_reason": gt_reason,
        "upqi_status": upqi_status,
        "upqi_reason": upqi_reason,
        "join_status": join_status,
        "join_reason": join_reason,
        "official_metrics_status": metrics_status,
        "official_metrics_reason": metrics_reason,
        "claim_gate": gate,
        "paper_ready": "YES" if paper_ready else "NO",
    }


def write_tsv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cell_id", "label", "model_tag", "dataset", "algo", "seed", "quantizer",
        "gt_status", "gt_reason", "upqi_status", "upqi_reason",
        "join_status", "join_reason", "official_metrics_status", "official_metrics_reason",
        "claim_gate", "paper_ready",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gate_counts = Counter(row["claim_gate"] for row in rows)
    ready_counts = Counter(row["paper_ready"] for row in rows)
    by_dataset = defaultdict(Counter)
    for row in rows:
        by_dataset[row["dataset"]][row["claim_gate"]] += 1
    lines = [
        "# Gate 2 v4 Claim-Gate Matrix",
        "",
        f"Rows: {len(rows)}",
        f"Paper-ready counts: {dict(ready_counts)}",
        f"Claim-gate counts: {dict(gate_counts)}",
        "",
        "## By Dataset",
        "",
    ]
    for dataset, counts in sorted(by_dataset.items()):
        lines.append(f"- `{dataset}`: {dict(counts)}")
    lines.append("")
    lines.append("`paper_ready` requires clean GT, clean UPQI, strict join pass, and official metrics pass.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Gate 2 v4 claim-gate matrix")
    parser.add_argument("--results_root", default="results/gate2_v4_standard")
    parser.add_argument("--metrics_root", default="results/gate2_v4_metrics")
    parser.add_argument("--seeds", default=",".join(SEEDS))
    parser.add_argument("--output_tsv", default="claim_gate_matrix_v4.tsv")
    parser.add_argument("--output_md", default="claim_gate_summary_v4.md")
    parser.add_argument("--allow_incomplete", action="store_true", help="Write the matrix without failing when cells are not paper-ready")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [seed.strip() for seed in args.seeds.split(",") if seed.strip()]
    rows = [row_for(Path(args.results_root), Path(args.metrics_root), cell) for cell in iter_core_cells(seeds)]
    write_tsv(Path(args.output_tsv), rows)
    write_md(Path(args.output_md), rows)
    print(f"rows: {len(rows)}")
    print(f"paper_ready: {dict(Counter(row['paper_ready'] for row in rows))}")
    print(f"claim_gate: {dict(Counter(row['claim_gate'] for row in rows))}")
    print(f"wrote: {args.output_tsv}")
    print(f"wrote: {args.output_md}")
    if any(row["paper_ready"] != "YES" for row in rows) and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
