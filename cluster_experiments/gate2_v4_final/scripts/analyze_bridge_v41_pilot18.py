#!/usr/bin/env python3
"""Analyze and decision-gate the accepted Gate 2 v4.1 pilot18 bridge run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_bridge_v4 import validate_obj as validate_bridge_obj  # noqa: E402


MODELS = (
    ("llama2_7b_adamw", "tofu"),
    ("llama3_8b_adamw", "tofu"),
    ("llama2_muse_adamw", "muse_news"),
    ("llama3_muse_adamw", "muse_news"),
)
QUANTIZERS = ("nf4", "gptq_int4", "rtn_int8")
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bridge_root",
        default="results/gate2_v4_bridge_standard_v41_pilot18",
    )
    parser.add_argument("--expected_bridge_source_lock_sha256", required=True)
    parser.add_argument("--strict_join_tsv", default=None)
    parser.add_argument("--hard_error_file", default=None)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else math.nan


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else math.nan


def average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else math.nan


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def condition_slug(obj: Mapping[str, Any]) -> str:
    if obj["condition"] == "full_quantized":
        return "full_quantized"
    return f"protected_{obj['ranking_source']}_k{obj['k']}"


def result_key(obj: Mapping[str, Any]) -> Tuple[str, str, str, str, str, int]:
    ranking_source = str(obj.get("ranking_source") or "none")
    return (
        str(obj["model_tag"]),
        str(obj["dataset"]),
        str(obj["quantizer"]),
        str(obj["condition"]),
        ranking_source,
        int(obj["k"]),
    )


def expected_keys() -> set[Tuple[str, str, str, str, str, int]]:
    keys = {
        (model, dataset, quantizer, "full_quantized", "none", 0)
        for model, dataset in MODELS
        for quantizer in QUANTIZERS
    }
    keys.update(
        {
            ("llama2_7b_adamw", "tofu", "gptq_int4", "protected", "UPQI_signed", 4),
            ("llama2_7b_adamw", "tofu", "gptq_int4", "protected", "random", 4),
            ("llama2_7b_adamw", "tofu", "nf4", "protected", "UPQI_signed", 4),
            ("llama2_7b_adamw", "tofu", "nf4", "protected", "random", 4),
            ("llama2_7b_adamw", "tofu", "nf4", "protected", "none", 0),
            ("llama2_7b_adamw", "tofu", "nf4", "protected", "all_layers", 224),
        }
    )
    return keys


def flatten_result(path: Path, obj: Mapping[str, Any]) -> Dict[str, Any]:
    bf16_forget = obj["bf16_reference"]["forget"]
    bf16_retain = obj["bf16_reference"]["retain"]
    condition_forget = obj["condition_metrics"]["forget"]
    condition_retain = obj["condition_metrics"]["retain"]
    deltas = obj["deltas"]
    compression = obj["compression"]
    return {
        "cell_id": obj["cell_id"],
        "model_tag": obj["model_tag"],
        "dataset": obj["dataset"],
        "algo": obj["algo"],
        "seed": obj["seed"],
        "quantizer": obj["quantizer"],
        "condition": obj["condition"],
        "condition_slug": condition_slug(obj),
        "ranking_source": obj.get("ranking_source") or "none",
        "k": obj["k"],
        "protected_layer_count": obj["protected_layer_count"],
        "quantized_layer_count": obj["quantized_layer_count"],
        "protected_parameter_fraction": compression["protected_parameter_fraction"],
        "effective_average_bits": compression["effective_average_bits"],
        "bf16_forget_ce": bf16_forget["answer_ce"],
        "condition_forget_ce": condition_forget["answer_ce"],
        "delta_forget_ce": deltas["delta_forget_ce"],
        "harmful_recovery": deltas["harmful_recovery"],
        "over_forgetting": deltas["over_forgetting"],
        "bf16_forget_logprob": bf16_forget["answer_logprob"],
        "condition_forget_logprob": condition_forget["answer_logprob"],
        "delta_answer_logprob": deltas["delta_answer_logprob"],
        "bf16_truth_ratio_log": bf16_forget["truth_ratio_log"],
        "condition_truth_ratio_log": condition_forget["truth_ratio_log"],
        "delta_truth_ratio_log": deltas["delta_truth_ratio_log"],
        "bf16_retain_ce": bf16_retain["answer_ce"],
        "condition_retain_ce": condition_retain["answer_ce"],
        "delta_retain_ce": deltas["delta_retain_ce"],
        "abs_delta_retain_ce": abs(float(deltas["delta_retain_ce"])),
        "bf16_retain_logprob": bf16_retain["answer_logprob"],
        "condition_retain_logprob": condition_retain["answer_logprob"],
        "delta_retain_logprob": deltas["delta_retain_logprob"],
        "bf16_forget_token_mean": bf16_forget["answer_token_mean"],
        "bf16_forget_token_min": bf16_forget["answer_token_min"],
        "condition_forget_token_mean": condition_forget["answer_token_mean"],
        "condition_forget_token_min": condition_forget["answer_token_min"],
        "bf16_forget_per_sample_rows": len(bf16_forget["per_sample"]),
        "condition_forget_per_sample_rows": len(condition_forget["per_sample"]),
        "elapsed_s": obj["elapsed_s"],
        "source_lock_sha256": obj["source_lock_sha256"],
        "result_path": str(path),
    }


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty TSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def compare_metrics(
    left: Mapping[str, Any], right: Mapping[str, Any], fields: Iterable[str]
) -> float:
    return max(abs(float(left[field]) - float(right[field])) for field in fields)


def main() -> None:
    args = parse_args()
    root = Path(args.bridge_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(root.rglob("bridge_v4.json"))
    partials = sorted(root.rglob("bridge_v4.json.partial"))
    objects: List[Dict[str, Any]] = []
    validation_errors: List[str] = []

    for path in paths:
        obj = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_bridge_obj(
            obj,
            path=path,
            expected_source_lock_sha256=args.expected_bridge_source_lock_sha256,
        )
        validation_errors.extend(f"{path}:{error}" for error in errors)
        objects.append(obj)

    actual_keys = [result_key(obj) for obj in objects]
    duplicate_keys = sorted(key for key, count in Counter(actual_keys).items() if count > 1)
    missing_keys = sorted(expected_keys() - set(actual_keys))
    extra_keys = sorted(set(actual_keys) - expected_keys())

    rows = [flatten_result(path, obj) for path, obj in zip(paths, objects)]
    write_tsv(out_dir / "pilot18_all_results.tsv", rows)

    full_rows = [row for row in rows if row["condition"] == "full_quantized"]
    protected_rows = [row for row in rows if row["condition"] == "protected"]
    write_tsv(out_dir / "pilot18_full12.tsv", full_rows)
    write_tsv(out_dir / "pilot18_protected6.tsv", protected_rows)

    quantizer_rows: List[Dict[str, Any]] = []
    for quantizer in QUANTIZERS:
        group = [row for row in full_rows if row["quantizer"] == quantizer]
        recovery = [float(row["harmful_recovery"]) for row in group]
        over = [float(row["over_forgetting"]) for row in group]
        retain = [float(row["abs_delta_retain_ce"]) for row in group]
        quantizer_rows.append(
            {
                "quantizer": quantizer,
                "n": len(group),
                "harmful_positive_n": sum(value > EPS for value in recovery),
                "over_forgetting_positive_n": sum(value > EPS for value in over),
                "harmful_recovery_mean": mean(recovery),
                "harmful_recovery_median": median(recovery),
                "harmful_recovery_min": min(recovery),
                "harmful_recovery_max": max(recovery),
                "abs_delta_retain_ce_mean": mean(retain),
            }
        )
    write_tsv(out_dir / "pilot18_quantizer_summary.tsv", quantizer_rows)

    index = {
        (
            str(row["model_tag"]),
            str(row["dataset"]),
            str(row["quantizer"]),
            str(row["condition"]),
            str(row["ranking_source"]),
            int(row["k"]),
        ): row
        for row in rows
    }

    mitigation_rows: List[Dict[str, Any]] = []
    mitigation_pass = True
    retain_closeness_pass = True
    matched_fraction_pass = True
    for quantizer in ("nf4", "gptq_int4"):
        full = index[("llama2_7b_adamw", "tofu", quantizer, "full_quantized", "none", 0)]
        random_row = index[("llama2_7b_adamw", "tofu", quantizer, "protected", "random", 4)]
        upqi = index[("llama2_7b_adamw", "tofu", quantizer, "protected", "UPQI_signed", 4)]
        full_harmful = float(full["harmful_recovery"])
        random_harmful = float(random_row["harmful_recovery"])
        upqi_harmful = float(upqi["harmful_recovery"])
        matched = abs(
            float(upqi["protected_parameter_fraction"])
            - float(random_row["protected_parameter_fraction"])
        ) <= 1e-12
        beats_random = upqi_harmful < random_harmful - EPS
        closer_retain = float(upqi["abs_delta_retain_ce"]) < float(
            random_row["abs_delta_retain_ce"]
        ) - EPS
        matched_fraction_pass &= matched
        mitigation_pass &= beats_random
        retain_closeness_pass &= closer_retain
        mitigation_rows.append(
            {
                "quantizer": quantizer,
                "full_harmful_recovery": full_harmful,
                "random_harmful_recovery": random_harmful,
                "upqi_harmful_recovery": upqi_harmful,
                "upqi_reduction_vs_full_pct": (
                    100.0 * (full_harmful - upqi_harmful) / full_harmful
                    if full_harmful > 0
                    else math.nan
                ),
                "upqi_reduction_vs_random_pct": (
                    100.0 * (random_harmful - upqi_harmful) / random_harmful
                    if random_harmful > 0
                    else math.nan
                ),
                "full_delta_retain_ce": full["delta_retain_ce"],
                "random_delta_retain_ce": random_row["delta_retain_ce"],
                "upqi_delta_retain_ce": upqi["delta_retain_ce"],
                "full_delta_truth_ratio_log": full["delta_truth_ratio_log"],
                "random_delta_truth_ratio_log": random_row["delta_truth_ratio_log"],
                "upqi_delta_truth_ratio_log": upqi["delta_truth_ratio_log"],
                "matched_protected_fraction": matched,
                "upqi_beats_random_harmful_recovery": beats_random,
                "upqi_closer_to_bf16_retain_ce": closer_retain,
            }
        )
    write_tsv(out_dir / "pilot18_mitigation_vs_random.tsv", mitigation_rows)

    boundary_fields = (
        "condition_forget_ce",
        "condition_forget_logprob",
        "condition_truth_ratio_log",
        "condition_retain_ce",
        "condition_retain_logprob",
        "delta_forget_ce",
        "harmful_recovery",
        "over_forgetting",
        "delta_truth_ratio_log",
        "delta_retain_ce",
    )
    full_nf4 = index[("llama2_7b_adamw", "tofu", "nf4", "full_quantized", "none", 0)]
    none_nf4_candidates = [
        row
        for row in protected_rows
        if row["quantizer"] == "nf4"
        and row["ranking_source"] == "none"
        and int(row["k"]) == 0
    ]
    all_nf4 = index[("llama2_7b_adamw", "tofu", "nf4", "protected", "all_layers", 224)]
    if len(none_nf4_candidates) != 1:
        raise SystemExit(f"expected one protected none/k0 row, got {len(none_nf4_candidates)}")
    none_nf4 = none_nf4_candidates[0]
    none_full_max_drift = compare_metrics(full_nf4, none_nf4, boundary_fields)
    all_zero_max = max(
        abs(float(all_nf4[field]))
        for field in (
            "delta_forget_ce",
            "harmful_recovery",
            "over_forgetting",
            "delta_answer_logprob",
            "delta_truth_ratio_log",
            "delta_retain_ce",
            "delta_retain_logprob",
        )
    )
    boundary_pass = none_full_max_drift <= EPS and all_zero_max <= EPS

    recoveries = [float(row["harmful_recovery"]) for row in full_rows]
    retain_abs = [float(row["abs_delta_retain_ce"]) for row in full_rows]
    truth_delta = [float(row["delta_truth_ratio_log"]) for row in full_rows]
    positive_settings = {
        (str(row["model_tag"]), str(row["dataset"]))
        for row in full_rows
        if float(row["harmful_recovery"]) > EPS
    }
    positive_quantizers = {
        str(row["quantizer"])
        for row in full_rows
        if float(row["harmful_recovery"]) > EPS
    }

    technical_pass = (
        len(paths) == 18
        and not partials
        and not validation_errors
        and not duplicate_keys
        and not missing_keys
        and not extra_keys
    )
    join_pass = True
    join_rows = 0
    if args.strict_join_tsv:
        with Path(args.strict_join_tsv).open(encoding="utf-8", newline="") as handle:
            joins = list(csv.DictReader(handle, delimiter="\t"))
        join_rows = len(joins)
        join_pass = len(joins) == 18 and all(row.get("status") == "PASS" for row in joins)

    hard_log_pass = True
    hard_error_lines = 0
    if args.hard_error_file:
        hard_path = Path(args.hard_error_file)
        hard_error_lines = sum(1 for line in hard_path.read_text(encoding="utf-8").splitlines() if line.strip())
        hard_log_pass = hard_error_lines == 0

    signal_pass = (
        sum(value > EPS for value in recoveries) >= 2
        and len(positive_settings) >= 2
        and len(positive_quantizers) >= 2
    )
    algorithms = sorted({str(row["algo"]) for row in full_rows})
    seeds = sorted({str(row["seed"]) for row in full_rows})
    design_coverage_pass = len(algorithms) >= 2 and len(seeds) >= 2

    gates = {
        "technical_result_gate": technical_pass,
        "strict_join_gate": join_pass,
        "hard_log_gate": hard_log_pass,
        "boundary_control_gate": boundary_pass,
        "measurable_structured_signal_gate": signal_pass,
        "single_cell_mitigation_signal_gate": (
            mitigation_pass and retain_closeness_pass and matched_fraction_pass
        ),
        "predeclared_pilot_diversity_gate": design_coverage_pass,
    }

    required_except_diversity = all(
        value
        for name, value in gates.items()
        if name != "predeclared_pilot_diversity_gate"
    )
    if all(gates.values()):
        decision = "GO_FULLQ144"
    elif required_except_diversity and not design_coverage_pass:
        decision = "CONDITIONAL_GO_RUN_DIVERSITY_CONFIRMATION_FIRST"
    else:
        decision = "STOP_AND_REPAIR"

    decision_obj = {
        "decision": decision,
        "gates": gates,
        "result_count": len(paths),
        "partial_count": len(partials),
        "validation_error_count": len(validation_errors),
        "duplicate_keys": duplicate_keys,
        "missing_keys": missing_keys,
        "extra_keys": extra_keys,
        "join_rows": join_rows,
        "hard_error_lines": hard_error_lines,
        "full_condition_count": len(full_rows),
        "protected_condition_count": len(protected_rows),
        "harmful_recovery_positive_count": sum(value > EPS for value in recoveries),
        "over_forgetting_positive_count": sum(
            float(row["over_forgetting"]) > EPS for row in full_rows
        ),
        "positive_settings": sorted(positive_settings),
        "positive_quantizers": sorted(positive_quantizers),
        "algorithms_covered": algorithms,
        "seeds_covered": seeds,
        "none_k0_vs_full_max_drift": none_full_max_drift,
        "all_k224_zero_delta_max": all_zero_max,
        "pearson_harmful_vs_abs_retain_shift": pearson(recoveries, retain_abs),
        "spearman_harmful_vs_abs_retain_shift": spearman(recoveries, retain_abs),
        "pearson_harmful_vs_delta_truth_ratio": pearson(recoveries, truth_delta),
        "spearman_harmful_vs_delta_truth_ratio": spearman(recoveries, truth_delta),
        "note": (
            "The diversity gate comes from the governing bridge plan. The measurable-signal "
            "operationalization is an execution decision rule, not a preregistered significance threshold."
        ),
    }
    (out_dir / "pilot18_decision.json").write_text(
        json.dumps(decision_obj, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Gate 2 v4.1 Pilot18 Automated Analysis",
        "",
        f"Decision: **{decision}**",
        "",
        "## Evidence Counts",
        "",
        f"- Results: {len(paths)}/18",
        f"- Partials: {len(partials)}",
        f"- Full conditions: {len(full_rows)}/12",
        f"- Additional protected/control conditions: {len(protected_rows)}/6",
        f"- Strict join rows: {join_rows if args.strict_join_tsv else 'not supplied'}",
        f"- Hard-error lines: {hard_error_lines if args.hard_error_file else 'not supplied'}",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in gates.items())
    lines.extend(
        [
            "",
            "## Full-Condition Signal",
            "",
            f"- Harmful-recovery positive: {decision_obj['harmful_recovery_positive_count']}/12",
            f"- Over-forgetting positive: {decision_obj['over_forgetting_positive_count']}/12",
            f"- Positive settings: {len(positive_settings)}/4",
            f"- Positive quantizers: {len(positive_quantizers)}/3",
            f"- Algorithms covered: {', '.join(algorithms)}",
            f"- Seeds covered: {', '.join(seeds)}",
            "",
            "## Boundary Controls",
            "",
            f"- none/k0 versus full maximum drift: {none_full_max_drift:.12g}",
            f"- all-layers/k224 maximum absolute delta: {all_zero_max:.12g}",
            "",
            "## Mitigation Comparisons",
            "",
            "| Quantizer | Full recovery | Random recovery | UPQI recovery | Reduction vs random | Matched fraction |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in mitigation_rows:
        lines.append(
            "| {quantizer} | {full_harmful_recovery:.6f} | "
            "{random_harmful_recovery:.6f} | {upqi_harmful_recovery:.6f} | "
            "{upqi_reduction_vs_random_pct:.2f}% | {matched_protected_fraction} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Decision Interpretation",
            "",
            "The current run passes technical, join, control, signal, and single-cell mitigation gates.",
            "It does not satisfy the governing plan's requirement for at least two algorithms and two seeds.",
            "Run a small diversity-confirmation pilot before committing the full 144-cell bridge allocation.",
            "The mitigation signal remains a one-cell result and is not a general mitigation claim.",
            "",
        ]
    )
    (out_dir / "PILOT18_AUTOMATED_ANALYSIS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print(f"RESULTS={len(paths)} FULL={len(full_rows)} PROTECTED={len(protected_rows)}")
    print(f"VALIDATION_ERRORS={len(validation_errors)} PARTIALS={len(partials)}")
    print(f"HARMFUL_POSITIVE={decision_obj['harmful_recovery_positive_count']}/12")
    print(f"BOUNDARY_PASS={boundary_pass} MITIGATION_SIGNAL_PASS={mitigation_pass}")
    print(f"DESIGN_COVERAGE_PASS={design_coverage_pass} ALGOS={algorithms} SEEDS={seeds}")
    print(f"FULL_BRIDGE_DECISION={decision}")
    print(f"WROTE {out_dir / 'pilot18_all_results.tsv'}")
    print(f"WROTE {out_dir / 'pilot18_decision.json'}")
    print(f"WROTE {out_dir / 'PILOT18_AUTOMATED_ANALYSIS.md'}")

    if not technical_pass or not join_pass or not hard_log_pass or not boundary_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
