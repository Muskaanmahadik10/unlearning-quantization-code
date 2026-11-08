#!/usr/bin/env python3
"""Analyze disjoint Bridge48 GQDP-versus-random confirmation at the cell level."""

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
K = 8


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
        estimate = abs(statistics.fmean(value if rng.random() < 0.5 else -value for value in values))
        extreme += estimate >= observed - 1e-15
    return extreme / float(iters + 1)


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
    expected = {str(row["cell_id"]): row for row in manifest.get("cells", [])}
    errors: list[str] = []
    if manifest.get("decision") != "BRIDGE48_SELECTION_READY_FOR_GQDP_K8_CONFIRMATION":
        errors.append("manifest_not_accepted")
    if len(expected) != 48:
        errors.append(f"manifest_cell_count:{len(expected)}:48")

    results: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    paths = sorted(Path(args.mitigation_root).rglob("bridge_v4.json"))
    for path in paths:
        obj = load_json(path)
        key = (cell_id(obj), str(obj.get("ranking_source")), int(obj.get("k", -1)))
        if obj.get("condition") != "protected":
            errors.append(f"nonprotected:{path}")
        if key in results:
            errors.append(f"duplicate:{key}")
        results[key] = obj
    if len(paths) != 96:
        errors.append(f"result_count:{len(paths)}:96")

    pairs: list[dict[str, Any]] = []
    for cell, metadata in sorted(expected.items()):
        guided = results.get((cell, GUIDED, K))
        random_result = results.get((cell, RANDOM, K))
        if guided is None or random_result is None:
            errors.append(f"missing_pair:{cell}")
            continue
        guided_harm = float(guided["deltas"]["harmful_recovery"])
        random_harm = float(random_result["deltas"]["harmful_recovery"])
        guided_retain = abs(float(guided["deltas"]["delta_retain_ce"]))
        random_retain = abs(float(random_result["deltas"]["delta_retain_ce"]))
        pairs.append({
            "cell_id": cell,
            "model_tag": metadata["model_tag"],
            "dataset": metadata["dataset"],
            "algo": metadata["algo"],
            "seed": metadata["seed"],
            "quantizer": metadata["quantizer"],
            "guided_harmful_recovery": guided_harm,
            "random_harmful_recovery": random_harm,
            "harmful_advantage_over_random": random_harm - guided_harm,
            "guided_abs_retain_shift": guided_retain,
            "random_abs_retain_shift": random_retain,
            "retain_noninferior": guided_retain <= random_retain + args.retain_abs_margin,
        })
    write_tsv(out_dir / "bridge48_matched_pairs.tsv", pairs)

    summary: dict[str, Any] = {"matched_cells": len(pairs)}
    if pairs:
        values = [float(row["harmful_advantage_over_random"]) for row in pairs]
        low, high = bootstrap_ci(values, iters=args.bootstrap_iters, seed=args.random_seed)
        summary.update({
            "mean_harmful_advantage": statistics.fmean(values),
            "median_harmful_advantage": statistics.median(values),
            "ci95_low": low,
            "ci95_high": high,
            "paired_win_rate": statistics.fmean(value > 0 for value in values),
            "retain_noninferior_fraction": statistics.fmean(bool(row["retain_noninferior"]) for row in pairs),
            "p_value": signflip_p(values, iters=args.bootstrap_iters, seed=args.random_seed + 1),
        })
    gate = (
        not errors
        and len(pairs) == 48
        and float(summary.get("ci95_low", 0.0)) > 0
        and float(summary.get("p_value", 1.0)) < 0.05
        and float(summary.get("retain_noninferior_fraction", 0.0)) >= 0.80
    )
    summary["confirmation_gate"] = "PASS" if gate else "FAIL"
    write_tsv(out_dir / "bridge48_summary.tsv", [summary])

    subgroup_rows: list[dict[str, Any]] = []
    for factor in ("model_tag", "dataset", "algo", "seed", "quantizer"):
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in pairs:
            grouped[str(row[factor])].append(row)
        for level, rows in sorted(grouped.items()):
            values = [float(row["harmful_advantage_over_random"]) for row in rows]
            subgroup_rows.append({
                "factor": factor,
                "level": level,
                "n_cells": len(rows),
                "mean_harmful_advantage": statistics.fmean(values),
                "median_harmful_advantage": statistics.median(values),
                "win_rate": statistics.fmean(value > 0 for value in values),
                "retain_noninferior_fraction": statistics.fmean(bool(row["retain_noninferior"]) for row in rows),
                "status": "DESCRIPTIVE_NOT_CONFIRMATORY",
            })
    write_tsv(out_dir / "bridge48_subgroups.tsv", subgroup_rows)

    if errors:
        decision = "BRIDGE48_CONFIRMATION_INVALID_STOP_REPAIR"
    elif gate:
        decision = "BRIDGE48_GQDP_CONFIRMATION_PASS_PROCEED_OFFICIAL_EVAL"
    else:
        decision = "BRIDGE48_GQDP_CONFIRMATION_FAIL_NO_GENERAL_CLAIM"
    decision_obj = {
        "decision": decision,
        "result_count": len(paths),
        "matched_pair_count": len(pairs),
        "errors": errors,
        "retain_noninferiority_margin_abs_ce": args.retain_abs_margin,
        "main_confirmation": summary,
        "claim_boundary": "A pass supports GQDP target-layer mitigation transport only; official benchmark claims remain closed.",
    }
    with (out_dir / "bridge48_decision.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(decision_obj, handle, indent=2, sort_keys=True)
        handle.write("\n")
    (out_dir / "BRIDGE48_CONFIRMATION_ANALYSIS.md").write_text(
        "\n".join([
            "# Gate 2 v4.2 Bridge48 Confirmation",
            "",
            f"Decision: **{decision}**",
            "",
            f"- Results: {len(paths)}/96",
            f"- Matched cells: {len(pairs)}/48",
            f"- Structural errors: {len(errors)}",
            f"- Main confirmation gate: {summary['confirmation_gate']}",
            "- Subgroup rows are descriptive and do not independently establish transport.",
            "- Official TOFU/MUSE claims remain closed.",
        ]) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"BRIDGE48_CONFIRMATION_DECISION={decision}")
    print(f"WROTE={out_dir}")
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
