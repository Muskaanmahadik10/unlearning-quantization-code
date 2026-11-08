#!/usr/bin/env python3
"""Analyze matched Pilot24 mitigation conditions using cells as paired units."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


GUIDED = ("GQDP_signed_v42", "UPQI_star_signed_v42", "displacement_l2_v42")
RANDOM = "deterministic_random_v42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mitigation_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--retain_abs_margin", type=float, default=0.01)
    parser.add_argument("--bootstrap_iters", type=int, default=10000)
    parser.add_argument("--random_seed", type=int, default=20260901)
    return parser.parse_args()


def load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cell_id(obj: Mapping[str, Any]) -> str:
    return str(obj.get("cell_id") or f"{obj['model_tag']}_{obj['algo']}_seed{obj['seed']}/{obj['quantizer']}")


def bootstrap_ci(values: Sequence[float], *, iters: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = [statistics.fmean(rng.choice(values) for _ in values) for _ in range(iters)]
    estimates.sort()
    return estimates[int(0.025 * (iters - 1))], estimates[int(0.975 * (iters - 1))]


def signflip_p(values: Sequence[float], *, iters: int, seed: int) -> float:
    observed = abs(statistics.fmean(values))
    rng = random.Random(seed)
    extreme = 1
    for _ in range(iters):
        test = abs(statistics.fmean(value if rng.random() < 0.5 else -value for value in values))
        extreme += test >= observed - 1e-15
    return extreme / float(iters + 1)


def bh_adjust(rows: list[dict[str, Any]]) -> None:
    order = sorted(range(len(rows)), key=lambda index: float(rows[index]["p_value"]))
    running = 1.0
    adjusted = [1.0] * len(rows)
    total = len(rows)
    for reverse in range(total - 1, -1, -1):
        index = order[reverse]
        running = min(running, float(rows[index]["p_value"]) * total / (reverse + 1))
        adjusted[index] = min(1.0, running)
    for row, value in zip(rows, adjusted):
        row["q_value"] = value


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(Path(args.manifest))
    expected_cells = {str(row["cell_id"]): row for row in manifest.get("cells", [])}
    errors: list[str] = []
    if len(expected_cells) != 24:
        errors.append(f"manifest_cell_count:{len(expected_cells)}:24")
    results: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    paths = sorted(Path(args.mitigation_root).rglob("bridge_v4.json"))
    for path in paths:
        obj = load_json(path)
        if obj.get("condition") != "protected":
            errors.append(f"nonprotected_result:{path}")
            continue
        key = (cell_id(obj), str(obj.get("ranking_source")), int(obj.get("k", -1)))
        if key in results:
            errors.append(f"duplicate:{key}")
        results[key] = obj
    if len(paths) != 192:
        errors.append(f"result_count:{len(paths)}:192")

    pair_rows: list[dict[str, Any]] = []
    for cell, cell_row in expected_cells.items():
        for k in (4, 8):
            random_key = (cell, RANDOM, k)
            if random_key not in results:
                errors.append(f"missing_random:{cell}:k{k}")
                continue
            random_obj = results[random_key]
            random_harm = float(random_obj["deltas"]["harmful_recovery"])
            random_retain = abs(float(random_obj["deltas"]["delta_retain_ce"]))
            for selector in GUIDED:
                key = (cell, selector, k)
                if key not in results:
                    errors.append(f"missing_guided:{cell}:{selector}:k{k}")
                    continue
                guided = results[key]
                guided_harm = float(guided["deltas"]["harmful_recovery"])
                guided_retain = abs(float(guided["deltas"]["delta_retain_ce"]))
                pair_rows.append(
                    {
                        "cell_id": cell,
                        "panel": cell_row["panel"],
                        "model_tag": cell_row["model_tag"],
                        "dataset": cell_row["dataset"],
                        "algo": cell_row["algo"],
                        "seed": cell_row["seed"],
                        "quantizer": cell_row["quantizer"],
                        "selector": selector,
                        "k": k,
                        "guided_harmful_recovery": guided_harm,
                        "random_harmful_recovery": random_harm,
                        "harmful_advantage_over_random": random_harm - guided_harm,
                        "guided_abs_retain_shift": guided_retain,
                        "random_abs_retain_shift": random_retain,
                        "retain_noninferior": guided_retain <= random_retain + args.retain_abs_margin,
                    }
                )
    write_tsv(out_dir / "mitigation_pilot24_matched_pairs.tsv", pair_rows)

    summaries: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(str(row["selector"]), int(row["k"]))].append(row)
    for index, ((selector, k), rows) in enumerate(sorted(grouped.items())):
        advantages = [float(row["harmful_advantage_over_random"]) for row in rows]
        low, high = bootstrap_ci(advantages, iters=args.bootstrap_iters, seed=args.random_seed + index)
        summaries.append(
            {
                "selector": selector,
                "k": k,
                "matched_cells": len(rows),
                "mean_harmful_advantage": statistics.fmean(advantages),
                "median_harmful_advantage": statistics.median(advantages),
                "ci95_low": low,
                "ci95_high": high,
                "paired_win_rate": statistics.fmean(value > 0 for value in advantages),
                "retain_noninferior_fraction": statistics.fmean(bool(row["retain_noninferior"]) for row in rows),
                "p_value": signflip_p(advantages, iters=args.bootstrap_iters, seed=args.random_seed + 1000 + index),
            }
        )
    bh_adjust(summaries)
    for row in summaries:
        row["mitigation_gate"] = (
            "PASS"
            if int(row["matched_cells"]) == 24
            and float(row["q_value"]) < 0.05
            and float(row["ci95_low"]) > 0
            and float(row["retain_noninferior_fraction"]) >= 0.80
            else "FAIL"
        )
    write_tsv(out_dir / "mitigation_pilot24_summary.tsv", summaries)
    passing = [row for row in summaries if row["mitigation_gate"] == "PASS"]
    if errors:
        decision = "MITIGATION_PILOT24_INVALID_STOP_REPAIR"
    elif passing:
        decision = "MITIGATION_PILOT24_PASS_EXPAND_BRIDGE48"
    else:
        decision = "MITIGATION_PILOT24_NO_GENERAL_ADVANTAGE_STOP_EXPANSION"
    decision_obj = {
        "decision": decision,
        "result_count": len(paths),
        "matched_pair_count": len(pair_rows),
        "errors": errors,
        "passing_selector_k": [{"selector": row["selector"], "k": row["k"]} for row in passing],
        "retain_noninferiority_margin_abs_ce": args.retain_abs_margin,
        "claim_boundary": "Pilot24 can gate Bridge48 expansion; official benchmark claims remain closed.",
    }
    with (out_dir / "mitigation_pilot24_decision.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(decision_obj, handle, indent=2, sort_keys=True)
        handle.write("\n")
    markdown = [
        "# Gate 2 v4.2 Mitigation Pilot24 Analysis",
        "",
        f"Decision: **{decision}**",
        "",
        f"- Protected results: {len(paths)}/192",
        f"- Matched guided-vs-random pairs: {len(pair_rows)}/144",
        f"- Structural errors: {len(errors)}",
        f"- Passing selector/k combinations: {[(row['selector'], row['k']) for row in passing] or 'none'}",
        "",
        "A pass requires a positive 95% cell-bootstrap interval, BH-adjusted p<0.05, and retain noninferiority in at least 80% of cells.",
    ]
    (out_dir / "MITIGATION_PILOT24_ANALYSIS.md").write_text("\n".join(markdown) + "\n", encoding="utf-8", newline="\n")
    print(f"MITIGATION_PILOT24_DECISION={decision}")
    print(f"WROTE={out_dir}")
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
