#!/usr/bin/env python3
"""Analyze corrected UPQI-star K/N convergence without treating layers as replicates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gate2_v4_contract import load_json, save_json  # noqa: E402
from validate_upqi_star_v42 import validate_obj  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--expected_cells", type=int, default=3)
    parser.add_argument("--expected_results", type=int, default=12)
    parser.add_argument("--scope", choices=("diagonal", "fullgrid"), default="diagonal")
    parser.add_argument("--reference_k", type=int, default=16)
    parser.add_argument("--reference_n", type=int, default=128)
    parser.add_argument("--min_spearman", type=float, default=0.95)
    parser.add_argument("--min_top8_overlap", type=float, default=0.75)
    parser.add_argument("--max_residual_ratio", type=float, default=1.25)
    return parser.parse_args()


def cell_id(obj: Mapping[str, Any]) -> str:
    return f"{obj['model_tag']}_{obj['algo']}_seed{obj['seed']}/{obj['quantizer']}"


def rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    output = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            output[order[position]] = average
        cursor = end
    return output


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else 0.0


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return pearson(rank(left), rank(right))


def top_overlap(left: Mapping[str, float], right: Mapping[str, float], k: int) -> float:
    left_top = {name for name, _ in sorted(left.items(), key=lambda item: (-item[1], item[0]))[:k]}
    right_top = {name for name, _ in sorted(right.items(), key=lambda item: (-item[1], item[0]))[:k]}
    return len(left_top & right_top) / float(k)


def layer_scores(obj: Mapping[str, Any], key: str, *, negate: bool = False) -> dict[str, float]:
    output: dict[str, float] = {}
    for row in obj.get("per_layer", []):
        value = float(row[key])
        output[str(row["layer"])] = -value if negate else value
    return output


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(root.rglob("upqi_star_v42.json"))
    structural_errors: list[str] = []
    records: dict[tuple[str, int, int], tuple[Path, Mapping[str, Any]]] = {}
    for path in paths:
        obj = load_json(path)
        report = validate_obj(obj, path)
        if not report["passed"]:
            structural_errors.append(f"invalid:{path}:{';'.join(report['errors'])}")
            continue
        key = (cell_id(obj), int(obj["n_quadrature"]), int(obj["n_samples"]))
        if key in records:
            structural_errors.append(f"duplicate:{key}")
        records[key] = (path, obj)
    cells = sorted({key[0] for key in records})
    if len(paths) != args.expected_results:
        structural_errors.append(f"result_count:{len(paths)}:{args.expected_results}")
    if len(cells) != args.expected_cells:
        structural_errors.append(f"cell_count:{len(cells)}:{args.expected_cells}")

    comparison_rows: list[dict[str, Any]] = []
    candidate_pass: dict[tuple[int, int], list[bool]] = defaultdict(list)
    candidate_pairs = sorted({(key[1], key[2]) for key in records})
    for cell in cells:
        reference_key = (cell, args.reference_k, args.reference_n)
        if reference_key not in records:
            structural_errors.append(f"missing_reference:{cell}")
            continue
        _, reference = records[reference_key]
        reference_signed = layer_scores(reference, "upqi_signed", negate=True)
        reference_abs = layer_scores(reference, "upqi_abs")
        reference_residual = statistics.median(
            abs(float(row["completeness_residual"])) for row in reference["per_layer"]
        )
        for k, n in candidate_pairs:
            key = (cell, k, n)
            if key not in records:
                continue
            path, obj = records[key]
            signed = layer_scores(obj, "upqi_signed", negate=True)
            absolute = layer_scores(obj, "upqi_abs")
            names = sorted(set(reference_signed) & set(signed))
            signed_rho = spearman([signed[name] for name in names], [reference_signed[name] for name in names])
            abs_rho = spearman([absolute[name] for name in names], [reference_abs[name] for name in names])
            signed_top8 = top_overlap(signed, reference_signed, 8)
            abs_top8 = top_overlap(absolute, reference_abs, 8)
            residual = statistics.median(abs(float(row["completeness_residual"])) for row in obj["per_layer"])
            residual_limit = max(reference_residual * args.max_residual_ratio, 1e-12)
            passed = (
                len(names) == int(reference["n_layers"])
                and signed_rho >= args.min_spearman
                and abs_rho >= args.min_spearman
                and signed_top8 >= args.min_top8_overlap
                and abs_top8 >= args.min_top8_overlap
                and residual <= residual_limit
            )
            candidate_pass[(k, n)].append(passed)
            comparison_rows.append(
                {
                    "cell_id": cell,
                    "k": k,
                    "n": n,
                    "reference_k": args.reference_k,
                    "reference_n": args.reference_n,
                    "layer_count": len(names),
                    "signed_spearman": signed_rho,
                    "abs_spearman": abs_rho,
                    "signed_top8_overlap": signed_top8,
                    "abs_top8_overlap": abs_top8,
                    "median_abs_completeness_residual": residual,
                    "reference_median_abs_residual": reference_residual,
                    "candidate_pass": passed,
                    "path": str(path),
                }
            )
    write_tsv(out_dir / "upqi_star_convergence_cells.tsv", comparison_rows)

    complete_candidates = [
        pair for pair, values in candidate_pass.items()
        if len(values) == args.expected_cells and all(values)
    ]
    selected = min(complete_candidates, key=lambda pair: (pair[0] * pair[1], pair[1], pair[0])) if complete_candidates else None
    reference_pair = (args.reference_k, args.reference_n)
    if structural_errors:
        decision = "UPQI_STAR_INVALID_STOP_REPAIR"
    elif selected is None and args.scope == "fullgrid":
        decision = "UPQI_STAR_FULLGRID_INCONCLUSIVE_STOP_REVIEW"
    elif selected is None:
        decision = "UPQI_STAR_DIAGONAL_INCONCLUSIVE_RUN_FULLGRID48"
    elif selected == reference_pair and args.scope == "diagonal":
        decision = "UPQI_STAR_REFERENCE_ONLY_PASS_RUN_FULLGRID48"
    elif args.scope == "fullgrid":
        decision = "UPQI_STAR_SELECT_FULLGRID_CONFIG"
    else:
        decision = "UPQI_STAR_SELECT_DIAGONAL_CONFIG"
    result = {
        "decision": decision,
        "result_count": len(paths),
        "cell_count": len(cells),
        "structural_errors": structural_errors,
        "passing_candidates": [{"k": k, "n": n} for k, n in complete_candidates],
        "selected": None if selected is None else {"k": selected[0], "n": selected[1]},
        "thresholds": {
            "min_spearman": args.min_spearman,
            "min_top8_overlap": args.min_top8_overlap,
            "max_residual_ratio": args.max_residual_ratio,
        },
        "claim_boundary": "Convergence selects a numerical configuration; it does not establish predictor superiority or mitigation efficacy.",
    }
    save_json(out_dir / "upqi_star_convergence_decision.json", result)
    markdown = [
        "# UPQI-star v4.2 Convergence Gate",
        "",
        f"Decision: **{decision}**",
        "",
        f"- Valid results: {len(records)}/{args.expected_results}",
        f"- Cells: {len(cells)}/{args.expected_cells}",
        f"- Structural errors: {len(structural_errors)}",
        f"- Passing K/N candidates: {complete_candidates or 'none'}",
        f"- Selected: {selected or 'none'}",
        "",
        "This gate compares layer rankings and completeness residuals against K=16, N=128. "
        "If only the reference passes, run the full K/N grid before choosing a cheaper production configuration.",
    ]
    (out_dir / "UPQI_STAR_CONVERGENCE.md").write_text("\n".join(markdown) + "\n", encoding="utf-8", newline="\n")
    print(f"UPQI_STAR_CONVERGENCE_DECISION={decision}")
    print(f"WROTE={out_dir}")
    raise SystemExit(0 if not structural_errors else 2)


if __name__ == "__main__":
    main()
