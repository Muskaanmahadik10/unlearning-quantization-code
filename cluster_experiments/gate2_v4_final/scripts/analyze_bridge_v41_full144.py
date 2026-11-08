#!/usr/bin/env python3
"""Lock-aware analysis of the complete Gate 2 v4.1 full144 bridge."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from analyze_bridge_v41_combined30 import (
    EPS,
    bh_adjust,
    expected_diversity_keys,
    expected_pilot_keys,
    group_summary,
    identity,
    load_panel,
    mitigation_and_boundaries,
    sample_inference,
)
from analyze_bridge_v41_pilot18 import mean, median, pearson, spearman, write_tsv


SETTINGS = (
    ("llama2_7b_adamw", "tofu"),
    ("llama3_8b_adamw", "tofu"),
    ("llama2_muse_adamw", "muse_news"),
    ("llama3_muse_adamw", "muse_news"),
)
ALGOS = ("GA", "GDR", "NPO", "KLR")
SEEDS = ("42", "2024", "5000")
QUANTIZERS = ("nf4", "gptq_int4", "rtn_int8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot_root", required=True)
    parser.add_argument("--diversity_root", required=True)
    parser.add_argument("--remainder_root", required=True)
    parser.add_argument("--expected_pilot_lock_sha256", required=True)
    parser.add_argument("--expected_diversity_lock_sha256", required=True)
    parser.add_argument("--expected_remainder_lock_sha256", required=True)
    parser.add_argument("--pilot_join_tsv", required=True)
    parser.add_argument("--diversity_join_tsv", required=True)
    parser.add_argument("--remainder_join_tsv", required=True)
    parser.add_argument("--hard_error_file", required=True)
    parser.add_argument("--bootstrap_reps", type=int, default=10000)
    parser.add_argument("--permutation_reps", type=int, default=20000)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def expected_full_keys() -> set[Tuple[str, str, str, str, str, str, str, int]]:
    return {
        (model, dataset, algo, seed, quantizer, "full_quantized", "none", 0)
        for model, dataset in SETTINGS
        for algo in ALGOS
        for seed in SEEDS
        for quantizer in QUANTIZERS
    }


def read_join(path: Path, expected: int) -> Tuple[List[Dict[str, str]], bool]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return rows, len(rows) == expected and all(row.get("status") == "PASS" for row in rows)


def marginal_effects(
    rows: Sequence[Mapping[str, Any]], field: str, outcome: str
) -> List[Dict[str, Any]]:
    grand = mean([float(row[outcome]) for row in rows])
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(float(row[outcome]))
    return [
        {
            "factor": field,
            "level": level,
            "outcome": outcome,
            "n_cells": len(values),
            "level_mean": mean(values),
            "grand_mean": grand,
            "effect_from_grand": mean(values) - grand,
            "level_median": median(values),
            "level_min": min(values),
            "level_max": max(values),
        }
        for level, values in sorted(grouped.items())
    ]


def interaction_effects(
    rows: Sequence[Mapping[str, Any]], left: str, right: str, outcome: str
) -> List[Dict[str, Any]]:
    grand = mean([float(row[outcome]) for row in rows])
    left_means = {
        level: mean([float(row[outcome]) for row in rows if str(row[left]) == level])
        for level in sorted({str(row[left]) for row in rows})
    }
    right_means = {
        level: mean([float(row[outcome]) for row in rows if str(row[right]) == level])
        for level in sorted({str(row[right]) for row in rows})
    }
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row[left]), str(row[right]))].append(float(row[outcome]))
    output: List[Dict[str, Any]] = []
    for (left_level, right_level), values in sorted(grouped.items()):
        cell_mean = mean(values)
        output.append(
            {
                "left_factor": left,
                "left_level": left_level,
                "right_factor": right,
                "right_level": right_level,
                "outcome": outcome,
                "n_cells": len(values),
                "cell_mean": cell_mean,
                "additive_expectation": left_means[left_level] + right_means[right_level] - grand,
                "interaction_residual": (
                    cell_mean - left_means[left_level] - right_means[right_level] + grand
                ),
            }
        )
    return output


def setting_label(row: Mapping[str, Any]) -> str:
    return f"{row['model_tag']}::{row['dataset']}"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        load_panel(Path(args.pilot_root), "pilot18", args.expected_pilot_lock_sha256),
        load_panel(Path(args.diversity_root), "diversity12", args.expected_diversity_lock_sha256),
        load_panel(Path(args.remainder_root), "remainder120", args.expected_remainder_lock_sha256),
    ]
    paths = [item for panel in panels for item in panel[0]]
    objects = [item for panel in panels for item in panel[1]]
    rows = [item for panel in panels for item in panel[2]]
    validation_errors = [item for panel in panels for item in panel[3]]
    partials = [item for panel in panels for item in panel[4]]
    full_rows = [row for row in rows if row["condition"] == "full_quantized"]
    protected_rows = [row for row in rows if row["condition"] == "protected"]
    for row in full_rows:
        row["setting"] = setting_label(row)

    full_keys = [identity(obj) for obj in objects if obj["condition"] == "full_quantized"]
    protected_keys = [identity(obj) for obj in objects if obj["condition"] == "protected"]
    duplicate_full = sorted(key for key, count in Counter(full_keys).items() if count > 1)
    duplicate_protected = sorted(key for key, count in Counter(protected_keys).items() if count > 1)
    missing_full = sorted(expected_full_keys() - set(full_keys))
    extra_full = sorted(set(full_keys) - expected_full_keys())
    pilot_keys = {identity(obj) for obj in panels[0][1]}
    diversity_keys = {identity(obj) for obj in panels[1][1]}
    remainder_keys = {identity(obj) for obj in panels[2][1]}
    expected_pilot = expected_pilot_keys()
    expected_diversity = expected_diversity_keys()
    accepted_full = {
        key for key in expected_pilot | expected_diversity if key[5] == "full_quantized"
    }
    expected_remainder = expected_full_keys() - accepted_full
    panel_inventory_errors = {
        "missing_pilot": sorted(expected_pilot - pilot_keys),
        "extra_pilot": sorted(pilot_keys - expected_pilot),
        "missing_diversity": sorted(expected_diversity - diversity_keys),
        "extra_diversity": sorted(diversity_keys - expected_diversity),
        "missing_remainder": sorted(expected_remainder - remainder_keys),
        "extra_remainder": sorted(remainder_keys - expected_remainder),
    }

    pilot_joins, pilot_join_pass = read_join(Path(args.pilot_join_tsv), 18)
    diversity_joins, diversity_join_pass = read_join(Path(args.diversity_join_tsv), 12)
    remainder_joins, remainder_join_pass = read_join(Path(args.remainder_join_tsv), 120)
    hard_error_lines = sum(
        bool(line.strip())
        for line in Path(args.hard_error_file).read_text(encoding="utf-8").splitlines()
    )

    write_tsv(out_dir / "full144_all150_results.tsv", rows)
    write_tsv(out_dir / "full144_full_conditions.tsv", full_rows)
    write_tsv(out_dir / "full144_protected_controls.tsv", protected_rows)

    inference = [
        sample_inference(obj, args.bootstrap_reps, args.permutation_reps)
        for obj in objects
        if obj["condition"] == "full_quantized"
    ]
    panel_by_cell = {str(row["cell_id"]): str(row["panel"]) for row in full_rows}
    q_values = bh_adjust([float(row["sign_flip_p"]) for row in inference])
    for row, q_value in zip(inference, q_values):
        row["panel"] = panel_by_cell[str(row["cell_id"])]
        row["sign_flip_q_bh"] = q_value
        low = float(row["bootstrap_ci95_low"])
        high = float(row["bootstrap_ci95_high"])
        if high < 0.0 and q_value < 0.05:
            row["inference_label"] = "supported_harmful_recovery"
        elif low > 0.0 and q_value < 0.05:
            row["inference_label"] = "supported_over_forgetting"
        else:
            row["inference_label"] = "inconclusive"
    write_tsv(out_dir / "full144_paired_inference_bh144.tsv", inference)

    for field in ("panel", "setting", "model_tag", "dataset", "algo", "seed", "quantizer"):
        write_tsv(out_dir / f"full144_summary_by_{field}.tsv", group_summary(full_rows, field))

    outcomes = ("delta_forget_ce", "harmful_recovery", "abs_delta_retain_ce")
    marginal_rows: List[Dict[str, Any]] = []
    for outcome in outcomes:
        for field in ("setting", "algo", "seed", "quantizer"):
            marginal_rows.extend(marginal_effects(full_rows, field, outcome))
    write_tsv(out_dir / "full144_balanced_marginal_effects.tsv", marginal_rows)

    interaction_rows: List[Dict[str, Any]] = []
    for outcome in outcomes:
        for left, right in (
            ("algo", "quantizer"),
            ("setting", "quantizer"),
            ("setting", "algo"),
        ):
            interaction_rows.extend(interaction_effects(full_rows, left, right, outcome))
    write_tsv(out_dir / "full144_balanced_interaction_residuals.tsv", interaction_rows)

    mitigation_rows, none_drift, all_zero, mitigation_pass = mitigation_and_boundaries(rows)
    write_tsv(out_dir / "full144_pilot_mitigation_controls.tsv", mitigation_rows)

    panel_counts = Counter(str(row["panel"]) for row in rows)
    technical_pass = (
        len(paths) == 150
        and len(full_rows) == 144
        and len(protected_rows) == 6
        and panel_counts == {"pilot18": 18, "diversity12": 12, "remainder120": 120}
        and not partials
        and not validation_errors
        and not duplicate_full
        and not duplicate_protected
        and not missing_full
        and not extra_full
        and not any(panel_inventory_errors.values())
    )
    joins_pass = pilot_join_pass and diversity_join_pass and remainder_join_pass
    hard_log_pass = hard_error_lines == 0
    boundary_pass = none_drift <= 1e-12 and all_zero <= 1e-12
    complete_factorial = (
        len(full_rows) == 144
        and {str(row["setting"]) for row in full_rows}
        == {f"{model}::{dataset}" for model, dataset in SETTINGS}
        and {str(row["algo"]) for row in full_rows} == set(ALGOS)
        and {str(row["seed"]) for row in full_rows} == set(SEEDS)
        and {str(row["quantizer"]) for row in full_rows} == set(QUANTIZERS)
    )
    gates = {
        "technical_150_result_gate": technical_pass,
        "strict_150_join_gate": joins_pass,
        "hard_log_gate": hard_log_pass,
        "boundary_control_gate": boundary_pass,
        "complete_full144_factorial_gate": complete_factorial,
    }
    decision = (
        "FULL144_EVIDENCE_READY_FOR_INTERPRETATION"
        if all(gates.values())
        else "STOP_AND_REPAIR_FULL144_EVIDENCE"
    )

    recoveries = [float(row["harmful_recovery"]) for row in full_rows]
    retain_shifts = [float(row["abs_delta_retain_ce"]) for row in full_rows]
    truth_deltas = [float(row["delta_truth_ratio_log"]) for row in full_rows]
    inference_counts = Counter(str(row["inference_label"]) for row in inference)
    decision_obj = {
        "decision": decision,
        "gates": gates,
        "counts": {
            "all_results": len(rows),
            "full_conditions": len(full_rows),
            "protected_controls": len(protected_rows),
            "pilot18": panel_counts["pilot18"],
            "diversity12": panel_counts["diversity12"],
            "remainder120": panel_counts["remainder120"],
            "partials": len(partials),
            "validation_errors": len(validation_errors),
            "join_rows": len(pilot_joins) + len(diversity_joins) + len(remainder_joins),
            "hard_error_lines": hard_error_lines,
        },
        "inventory": {
            "missing_full": missing_full,
            "extra_full": extra_full,
            "duplicate_full": duplicate_full,
            "duplicate_protected": duplicate_protected,
            **panel_inventory_errors,
        },
        "signal": {
            "harmful_positive": sum(value > EPS for value in recoveries),
            "overforget_positive": sum(float(row["over_forgetting"]) > EPS for row in full_rows),
            "paired_inference_labels": dict(inference_counts),
        },
        "correlations_descriptive_only": {
            "pearson_recovery_vs_abs_retain_shift": pearson(recoveries, retain_shifts),
            "spearman_recovery_vs_abs_retain_shift": spearman(recoveries, retain_shifts),
            "pearson_recovery_vs_truth_ratio_delta": pearson(recoveries, truth_deltas),
            "spearman_recovery_vs_truth_ratio_delta": spearman(recoveries, truth_deltas),
        },
        "controls": {
            "none_k0_vs_full_max_drift": none_drift,
            "all_layers_k224_max_abs_delta": all_zero,
            "single_setting_mitigation_pass": mitigation_pass,
        },
        "claim_boundary": (
            "A passing result completes the target-layer full-quantized factorial bridge only. "
            "It does not establish universal mitigation, official TOFU/MUSE benchmark performance, "
            "exact GQDP superiority, or deployment-wide quantization."
        ),
    }
    (out_dir / "full144_decision.json").write_text(
        json.dumps(decision_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "full144_validation_errors.txt").write_text(
        "\n".join(validation_errors) + ("\n" if validation_errors else ""),
        encoding="utf-8",
    )

    lines = [
        "# Gate 2 v4.1 Full144 Bridge Analysis",
        "",
        f"Decision: **{decision}**",
        "",
        "## Evidence",
        "",
        f"- All results: {len(rows)}/150 (144 full plus 6 frozen controls)",
        f"- Full conditions: {len(full_rows)}/144",
        f"- Strict joins: {len(pilot_joins) + len(diversity_joins) + len(remainder_joins)}/150",
        f"- Partials: {len(partials)}",
        f"- Validation errors: {len(validation_errors)}",
        f"- Hard-error lines: {hard_error_lines}",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in gates.items())
    lines.extend(
        [
            "",
            "## Signal",
            "",
            f"- Harmful-recovery point estimates: {sum(value > EPS for value in recoveries)}/144",
            f"- Over-forgetting point estimates: {sum(float(row['over_forgetting']) > EPS for row in full_rows)}/144",
            f"- Paired inference after BH over 144 cells: {dict(inference_counts)}",
            "",
            "## Interpretation Boundary",
            "",
            "This analyzer completes the balanced target-layer full-quantized bridge. Marginal and interaction tables are cell-level factorial descriptions. Per-sample paired tests support individual-cell direction; they do not turn token, sample, or layer rows into independent experimental replicates.",
            "",
            "The six mitigation/control results remain a single Llama2-TOFU GA/42 demonstration. Official TOFU/MUSE metrics, broad matched mitigation, exact GQDP comparisons, and deployment-wide claims remain separate gates.",
        ]
    )
    (out_dir / "FULL144_AUTOMATED_ANALYSIS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"RESULTS={len(rows)}/150 FULL={len(full_rows)}/144 PROTECTED={len(protected_rows)}/6")
    print(f"JOINS={len(pilot_joins) + len(diversity_joins) + len(remainder_joins)}/150")
    print(f"VALIDATION_ERRORS={len(validation_errors)} PARTIALS={len(partials)} HARD_ERRORS={hard_error_lines}")
    print(f"FULL144_DECISION={decision}")
    print(f"WROTE {out_dir / 'FULL144_AUTOMATED_ANALYSIS.md'}")
    print(f"WROTE {out_dir / 'full144_decision.json'}")
    if decision != "FULL144_EVIDENCE_READY_FOR_INTERPRETATION":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
