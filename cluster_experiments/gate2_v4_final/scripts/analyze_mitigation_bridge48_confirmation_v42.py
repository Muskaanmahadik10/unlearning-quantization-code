#!/usr/bin/env python3
"""Analyze the preregistered disjoint Bridge48 GQDP confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


GUIDED = "GQDP_signed_v42"
RANDOM = "deterministic_random_v42"
K_VALUE = 8


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


def bootstrap_ci(values: Sequence[float], iters: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(iters))
    return estimates[int(0.025 * (iters - 1))], estimates[int(0.975 * (iters - 1))]


def signflip_p(values: Sequence[float], iters: int, seed: int) -> float:
    observed = abs(statistics.fmean(values))
    rng = random.Random(seed)
    extreme = 1
    for _ in range(iters):
        value = abs(statistics.fmean(item if rng.random() < 0.5 else -item for item in values))
        extreme += value >= observed - 1e-15
    return extreme / (iters + 1)


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


def bh_adjust(rows: list[dict[str, Any]]) -> None:
    order = sorted(range(len(rows)), key=lambda index: float(rows[index]["p_value"]))
    running = 1.0
    for reverse in range(len(order) - 1, -1, -1):
        index = order[reverse]
        running = min(running, float(rows[index]["p_value"]) * len(rows) / (reverse + 1))
        rows[index]["q_value"] = min(1.0, running)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(Path(args.manifest))
    expected = {str(row["cell_id"]): row for row in manifest.get("cells", [])}
    errors: list[str] = []
    if manifest.get("decision") != "BRIDGE48_CONFIRMATION_SELECTION_READY":
        errors.append(f"manifest_decision:{manifest.get('decision')}")
    if len(expected) != 48:
        errors.append(f"manifest_cell_count:{len(expected)}:48")

    results: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    paths = sorted(Path(args.mitigation_root).rglob("bridge_v4.json"))
    if len(paths) != 96:
        errors.append(f"result_count:{len(paths)}:96")
    for path in paths:
        obj = load_json(path)
        key = (cell_id(obj), str(obj.get("ranking_source")), int(obj.get("k", -1)))
        if obj.get("condition") != "protected":
            errors.append(f"nonprotected:{path}")
        if key in results:
            errors.append(f"duplicate:{key}")
        results[key] = obj

    pairs: list[dict[str, Any]] = []
    for cell, metadata in expected.items():
        guided = results.get((cell, GUIDED, K_VALUE))
        random_result = results.get((cell, RANDOM, K_VALUE))
        if guided is None or random_result is None:
            errors.append(f"missing_pair:{cell}")
            continue
        guided_harm = float(guided["deltas"]["harmful_recovery"])
        random_harm = float(random_result["deltas"]["harmful_recovery"])
        guided_retain = abs(float(guided["deltas"]["delta_retain_ce"]))
        random_retain = abs(float(random_result["deltas"]["delta_retain_ce"]))
        pairs.append(
            {
                **{key: metadata[key] for key in ("cell_id", "model_tag", "dataset", "algo", "seed", "quantizer")},
                "k": K_VALUE,
                "guided_harmful_recovery": guided_harm,
                "random_harmful_recovery": random_harm,
                "harmful_advantage_over_random": random_harm - guided_harm,
                "guided_abs_retain_shift": guided_retain,
                "random_abs_retain_shift": random_retain,
                "retain_noninferior": guided_retain <= random_retain + args.retain_abs_margin,
            }
        )
    write_tsv(out_dir / "bridge48_confirmation_matched_pairs.tsv", pairs)

    advantages = [float(row["harmful_advantage_over_random"]) for row in pairs]
    primary: dict[str, Any] = {"matched_cells": len(pairs)}
    if advantages:
        low, high = bootstrap_ci(advantages, args.bootstrap_iters, args.random_seed)
        primary.update(
            {
                "mean_harmful_advantage": statistics.fmean(advantages),
                "median_harmful_advantage": statistics.median(advantages),
                "ci95_low": low,
                "ci95_high": high,
                "paired_win_rate": statistics.fmean(value > 0 for value in advantages),
                "retain_noninferior_fraction": statistics.fmean(bool(row["retain_noninferior"]) for row in pairs),
                "p_value": signflip_p(advantages, args.bootstrap_iters, args.random_seed + 1),
            }
        )
    primary_pass = bool(
        not errors
        and len(pairs) == 48
        and float(primary.get("ci95_low", float("-inf"))) > 0
        and float(primary.get("p_value", 1.0)) < 0.05
        and float(primary.get("retain_noninferior_fraction", 0.0)) >= 0.80
    )
    primary["confirmation_gate"] = "PASS" if primary_pass else "FAIL"
    write_tsv(out_dir / "bridge48_confirmation_primary.tsv", [primary])

    subgroup_rows: list[dict[str, Any]] = []
    for factor in ("model_tag", "dataset", "algo", "seed", "quantizer"):
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in pairs:
            grouped[str(row[factor])].append(row)
        for index, (level, rows) in enumerate(sorted(grouped.items())):
            values = [float(row["harmful_advantage_over_random"]) for row in rows]
            low, high = bootstrap_ci(values, args.bootstrap_iters, args.random_seed + 100 + len(subgroup_rows))
            subgroup_rows.append(
                {
                    "factor": factor,
                    "level": level,
                    "n_cells": len(rows),
                    "mean_harmful_advantage": statistics.fmean(values),
                    "median_harmful_advantage": statistics.median(values),
                    "ci95_low": low,
                    "ci95_high": high,
                    "paired_win_rate": statistics.fmean(value > 0 for value in values),
                    "retain_noninferior_fraction": statistics.fmean(bool(row["retain_noninferior"]) for row in rows),
                    "p_value": signflip_p(values, args.bootstrap_iters, args.random_seed + 1000 + len(subgroup_rows) + index),
                }
            )
    if subgroup_rows:
        bh_adjust(subgroup_rows)
    write_tsv(out_dir / "bridge48_confirmation_subgroup_sensitivity.tsv", subgroup_rows)

    if errors:
        decision = "BRIDGE48_CONFIRMATION_INVALID_STOP_REPAIR"
    elif primary_pass:
        decision = "BRIDGE48_CONFIRMATION_PASS_OFFICIAL_EVAL_READY"
    else:
        decision = "BRIDGE48_CONFIRMATION_NO_TRANSPORT_STOP_GENERAL_CLAIM"
    decision_obj = {
        "decision": decision,
        "result_count": len(paths),
        "matched_pair_count": len(pairs),
        "errors": errors,
        "primary": primary,
        "retain_noninferiority_margin_abs_ce": args.retain_abs_margin,
        "claim_boundary": "A pass supports GQDP mitigation transport to this factorial envelope; official benchmark success remains unestablished.",
    }
    (out_dir / "bridge48_confirmation_decision.json").write_text(
        json.dumps(decision_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    markdown = [
        "# Gate 2 v4.2 Bridge48 Confirmation Analysis",
        "",
        f"Decision: **{decision}**",
        "",
        f"- Results: {len(paths)}/96",
        f"- Matched cells: {len(pairs)}/48",
        f"- Structural errors: {len(errors)}",
        f"- Primary confirmation: {primary['confirmation_gate']}",
        "",
        "The primary unit is the matched cell. Subgroup tests are multiplicity-adjusted sensitivity analyses.",
        "Official TOFU/MUSE claims remain closed until maintained evaluator results pass.",
    ]
    (out_dir / "BRIDGE48_CONFIRMATION_ANALYSIS.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"BRIDGE48_CONFIRMATION_DECISION={decision}")
    print(f"WROTE={out_dir}")
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
