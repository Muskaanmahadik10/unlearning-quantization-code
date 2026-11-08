#!/usr/bin/env python3
"""Audit and analyze the Gate 2 v4.1 Pilot18 plus Diversity12 panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_bridge_v41_pilot18 import (  # noqa: E402
    flatten_result,
    mean,
    median,
    pearson,
    spearman,
    write_tsv,
)
from validate_bridge_v4 import validate_obj as validate_bridge_obj  # noqa: E402


EPS = 1e-8
MODELS = (
    ("llama2_7b_adamw", "tofu"),
    ("llama3_8b_adamw", "tofu"),
    ("llama2_muse_adamw", "muse_news"),
    ("llama3_muse_adamw", "muse_news"),
)
QUANTIZERS = ("nf4", "gptq_int4", "rtn_int8")
DIVERSITY_SPEC = (
    ("llama2_7b_adamw", "tofu", "NPO", "2024", "nf4"),
    ("llama2_7b_adamw", "tofu", "KLR", "5000", "gptq_int4"),
    ("llama2_7b_adamw", "tofu", "GDR", "2024", "rtn_int8"),
    ("llama3_8b_adamw", "tofu", "NPO", "5000", "nf4"),
    ("llama3_8b_adamw", "tofu", "KLR", "2024", "gptq_int4"),
    ("llama3_8b_adamw", "tofu", "GDR", "5000", "rtn_int8"),
    ("llama2_muse_adamw", "muse_news", "NPO", "2024", "nf4"),
    ("llama2_muse_adamw", "muse_news", "KLR", "5000", "gptq_int4"),
    ("llama2_muse_adamw", "muse_news", "GDR", "2024", "rtn_int8"),
    ("llama3_muse_adamw", "muse_news", "NPO", "5000", "nf4"),
    ("llama3_muse_adamw", "muse_news", "KLR", "2024", "gptq_int4"),
    ("llama3_muse_adamw", "muse_news", "GDR", "5000", "rtn_int8"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot_root", required=True)
    parser.add_argument("--diversity_root", required=True)
    parser.add_argument("--expected_pilot_lock_sha256", required=True)
    parser.add_argument("--expected_diversity_lock_sha256", required=True)
    parser.add_argument("--pilot_join_tsv", required=True)
    parser.add_argument("--diversity_join_tsv", required=True)
    parser.add_argument("--hard_error_file", required=True)
    parser.add_argument("--bootstrap_reps", type=int, default=10000)
    parser.add_argument("--permutation_reps", type=int, default=20000)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def identity(obj: Mapping[str, Any]) -> Tuple[str, str, str, str, str, str, str, int]:
    return (
        str(obj["model_tag"]),
        str(obj["dataset"]),
        str(obj["algo"]),
        str(obj["seed"]),
        str(obj["quantizer"]),
        str(obj["condition"]),
        str(obj.get("ranking_source") or "none"),
        int(obj["k"]),
    )


def expected_pilot_keys() -> set[Tuple[str, str, str, str, str, str, str, int]]:
    keys = {
        (model, dataset, "GA", "42", quantizer, "full_quantized", "none", 0)
        for model, dataset in MODELS
        for quantizer in QUANTIZERS
    }
    keys.update(
        {
            ("llama2_7b_adamw", "tofu", "GA", "42", "nf4", "protected", "UPQI_signed", 4),
            ("llama2_7b_adamw", "tofu", "GA", "42", "nf4", "protected", "random", 4),
            ("llama2_7b_adamw", "tofu", "GA", "42", "gptq_int4", "protected", "UPQI_signed", 4),
            ("llama2_7b_adamw", "tofu", "GA", "42", "gptq_int4", "protected", "random", 4),
            ("llama2_7b_adamw", "tofu", "GA", "42", "nf4", "protected", "none", 0),
            ("llama2_7b_adamw", "tofu", "GA", "42", "nf4", "protected", "all_layers", 224),
        }
    )
    return keys


def expected_diversity_keys() -> set[Tuple[str, str, str, str, str, str, str, int]]:
    return {
        (model, dataset, algo, seed, quantizer, "full_quantized", "none", 0)
        for model, dataset, algo, seed, quantizer in DIVERSITY_SPEC
    }


def load_panel(
    root: Path, panel: str, expected_lock: str
) -> Tuple[List[Path], List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[Path]]:
    paths = sorted(root.rglob("bridge_v4.json"))
    partials = sorted(root.rglob("bridge_v4.json.partial"))
    objects: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for path in paths:
        obj = json.loads(path.read_text(encoding="utf-8"))
        found = validate_bridge_obj(
            obj, path=path, expected_source_lock_sha256=expected_lock
        )
        errors.extend(f"{path}:{error}" for error in found)
        row = flatten_result(path, obj)
        row["panel"] = panel
        objects.append(obj)
        rows.append(row)
    return paths, objects, rows, errors, partials


def read_join(path: Path, expected: int) -> Tuple[List[Dict[str, str]], bool]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return rows, len(rows) == expected and all(row.get("status") == "PASS" for row in rows)


def bh_adjust(p_values: Sequence[float]) -> List[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def sample_inference(
    obj: Mapping[str, Any], bootstrap_reps: int, permutation_reps: int
) -> Dict[str, Any]:
    before = obj["bf16_reference"]["forget"]["per_sample"]
    after = obj["condition_metrics"]["forget"]["per_sample"]
    if len(before) != 128 or len(after) != 128:
        raise ValueError(f"{obj['cell_id']}: expected 128 paired samples")
    deltas: List[float] = []
    tr_deltas: List[float] = []
    for index, (left, right) in enumerate(zip(before, after)):
        if not left.get("reportable") or not right.get("reportable"):
            raise ValueError(f"{obj['cell_id']}: unreportable sample {index}")
        if left.get("sample_hash") != right.get("sample_hash"):
            raise ValueError(f"{obj['cell_id']}: sample hash mismatch {index}")
        deltas.append(float(right["answer_ce"]) - float(left["answer_ce"]))
        tr_deltas.append(float(right["truth_ratio_log"]) - float(left["truth_ratio_log"]))

    values = np.asarray(deltas, dtype=np.float64)
    truth_values = np.asarray(tr_deltas, dtype=np.float64)
    digest = hashlib.sha256(str(obj["cell_id"]).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(bootstrap_reps, len(values)))
    boot = values[indices].mean(axis=1)
    truth_boot = truth_values[indices].mean(axis=1)
    signs = rng.choice(
        np.asarray((-1.0, 1.0)), size=(permutation_reps, len(values))
    )
    permuted = (signs * values).mean(axis=1)
    observed = float(values.mean())
    p_value = float(
        (np.count_nonzero(np.abs(permuted) >= abs(observed)) + 1)
        / (permutation_reps + 1)
    )
    standard_deviation = float(values.std(ddof=1))
    return {
        "cell_id": obj["cell_id"],
        "panel": "pilot18" if str(obj["algo"]) == "GA" else "diversity12",
        "model_tag": obj["model_tag"],
        "dataset": obj["dataset"],
        "algo": obj["algo"],
        "seed": obj["seed"],
        "quantizer": obj["quantizer"],
        "n_pairs": len(values),
        "mean_delta_forget_ce": observed,
        "harmful_recovery": max(0.0, -observed),
        "over_forgetting": max(0.0, observed),
        "bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
        "mean_delta_truth_ratio_log": float(truth_values.mean()),
        "truth_ratio_ci95_low": float(np.quantile(truth_boot, 0.025)),
        "truth_ratio_ci95_high": float(np.quantile(truth_boot, 0.975)),
        "paired_effect_dz": observed / standard_deviation if standard_deviation else math.nan,
        "sign_flip_p": p_value,
    }


def group_summary(rows: Sequence[Mapping[str, Any]], field: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    output: List[Dict[str, Any]] = []
    for value in sorted(grouped):
        members = grouped[value]
        recovery = [float(row["harmful_recovery"]) for row in members]
        retain = [float(row["abs_delta_retain_ce"]) for row in members]
        output.append(
            {
                field: value,
                "n": len(members),
                "harmful_positive": sum(item > EPS for item in recovery),
                "overforget_positive": sum(float(row["over_forgetting"]) > EPS for row in members),
                "recovery_mean": mean(recovery),
                "recovery_median": median(recovery),
                "recovery_min": min(recovery),
                "recovery_max": max(recovery),
                "abs_retain_shift_mean": mean(retain),
            }
        )
    return output


def mitigation_and_boundaries(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], float, float, bool]:
    index = {
        (
            str(row["model_tag"]),
            str(row["dataset"]),
            str(row["algo"]),
            str(row["seed"]),
            str(row["quantizer"]),
            str(row["condition"]),
            str(row["ranking_source"]),
            int(row["k"]),
        ): row
        for row in rows
    }
    comparisons: List[Dict[str, Any]] = []
    mitigation_pass = True
    for quantizer in ("nf4", "gptq_int4"):
        prefix = ("llama2_7b_adamw", "tofu", "GA", "42", quantizer)
        full = index[prefix + ("full_quantized", "none", 0)]
        random_row = index[prefix + ("protected", "random", 4)]
        upqi = index[prefix + ("protected", "UPQI_signed", 4)]
        matched = abs(
            float(upqi["protected_parameter_fraction"])
            - float(random_row["protected_parameter_fraction"])
        ) <= 1e-12
        beats_random = float(upqi["harmful_recovery"]) < float(random_row["harmful_recovery"]) - EPS
        closer_retain = float(upqi["abs_delta_retain_ce"]) < float(random_row["abs_delta_retain_ce"]) - EPS
        mitigation_pass &= matched and beats_random and closer_retain
        comparisons.append(
            {
                "quantizer": quantizer,
                "full_harmful_recovery": full["harmful_recovery"],
                "random_harmful_recovery": random_row["harmful_recovery"],
                "upqi_harmful_recovery": upqi["harmful_recovery"],
                "upqi_reduction_vs_random_pct": (
                    100.0
                    * (float(random_row["harmful_recovery"]) - float(upqi["harmful_recovery"]))
                    / float(random_row["harmful_recovery"])
                    if float(random_row["harmful_recovery"]) > EPS
                    else math.nan
                ),
                "matched_protected_fraction": matched,
                "upqi_closer_retain_than_random": closer_retain,
            }
        )

    prefix = ("llama2_7b_adamw", "tofu", "GA", "42", "nf4")
    full = index[prefix + ("full_quantized", "none", 0)]
    none = index[prefix + ("protected", "none", 0)]
    all_layers = index[prefix + ("protected", "all_layers", 224)]
    metric_fields = (
        "harmful_recovery",
        "over_forgetting",
        "delta_forget_ce",
        "delta_truth_ratio_log",
        "delta_retain_ce",
    )
    none_drift = max(abs(float(full[field]) - float(none[field])) for field in metric_fields)
    all_zero = max(abs(float(all_layers[field])) for field in metric_fields)
    return comparisons, none_drift, all_zero, mitigation_pass


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pilot = load_panel(Path(args.pilot_root), "pilot18", args.expected_pilot_lock_sha256)
    diversity = load_panel(Path(args.diversity_root), "diversity12", args.expected_diversity_lock_sha256)
    pilot_paths, pilot_objects, pilot_rows, pilot_errors, pilot_partials = pilot
    diversity_paths, diversity_objects, diversity_rows, diversity_errors, diversity_partials = diversity
    objects = pilot_objects + diversity_objects
    rows = pilot_rows + diversity_rows
    full_rows = [row for row in rows if row["condition"] == "full_quantized"]
    protected_rows = [row for row in rows if row["condition"] == "protected"]

    pilot_keys = [identity(obj) for obj in pilot_objects]
    diversity_keys = [identity(obj) for obj in diversity_objects]
    duplicate_keys = sorted(
        key for key, count in Counter(pilot_keys + diversity_keys).items() if count > 1
    )
    missing_pilot = sorted(expected_pilot_keys() - set(pilot_keys))
    extra_pilot = sorted(set(pilot_keys) - expected_pilot_keys())
    missing_diversity = sorted(expected_diversity_keys() - set(diversity_keys))
    extra_diversity = sorted(set(diversity_keys) - expected_diversity_keys())

    pilot_joins, pilot_join_pass = read_join(Path(args.pilot_join_tsv), 18)
    diversity_joins, diversity_join_pass = read_join(Path(args.diversity_join_tsv), 12)
    hard_error_lines = sum(
        bool(line.strip())
        for line in Path(args.hard_error_file).read_text(encoding="utf-8").splitlines()
    )

    write_tsv(out_dir / "combined30_all_results.tsv", rows)
    write_tsv(out_dir / "combined30_full24.tsv", full_rows)
    write_tsv(out_dir / "combined30_protected6.tsv", protected_rows)
    write_tsv(out_dir / "combined30_diversity12.tsv", diversity_rows)

    inference = [
        sample_inference(obj, args.bootstrap_reps, args.permutation_reps)
        for obj in objects
        if obj["condition"] == "full_quantized"
    ]
    q_values = bh_adjust([float(row["sign_flip_p"]) for row in inference])
    for row, q_value in zip(inference, q_values):
        row["sign_flip_q_bh"] = q_value
        low = float(row["bootstrap_ci95_low"])
        high = float(row["bootstrap_ci95_high"])
        if high < 0.0 and q_value < 0.05:
            row["inference_label"] = "supported_harmful_recovery"
        elif low > 0.0 and q_value < 0.05:
            row["inference_label"] = "supported_over_forgetting"
        else:
            row["inference_label"] = "inconclusive"
    write_tsv(out_dir / "combined30_full24_paired_inference.tsv", inference)

    group_outputs: Dict[str, List[Dict[str, Any]]] = {}
    for field in ("panel", "model_tag", "dataset", "algo", "seed", "quantizer"):
        group_outputs[field] = group_summary(full_rows, field)
        write_tsv(out_dir / f"combined30_summary_by_{field}.tsv", group_outputs[field])

    ga_index = {
        (str(row["model_tag"]), str(row["dataset"]), str(row["quantizer"])): row
        for row in pilot_rows
        if row["condition"] == "full_quantized"
    }
    matched_rows: List[Dict[str, Any]] = []
    for row in diversity_rows:
        ga = ga_index[(str(row["model_tag"]), str(row["dataset"]), str(row["quantizer"]))]
        matched_rows.append(
            {
                "model_tag": row["model_tag"],
                "dataset": row["dataset"],
                "quantizer": row["quantizer"],
                "diversity_algo": row["algo"],
                "diversity_seed": row["seed"],
                "ga42_harmful_recovery": ga["harmful_recovery"],
                "diversity_harmful_recovery": row["harmful_recovery"],
                "difference_diversity_minus_ga42": float(row["harmful_recovery"]) - float(ga["harmful_recovery"]),
                "ga42_abs_retain_shift": ga["abs_delta_retain_ce"],
                "diversity_abs_retain_shift": row["abs_delta_retain_ce"],
            }
        )
    write_tsv(out_dir / "combined30_diversity_vs_matched_ga42.tsv", matched_rows)

    mitigation_rows, none_drift, all_zero, mitigation_pass = mitigation_and_boundaries(rows)
    write_tsv(out_dir / "combined30_mitigation_controls.tsv", mitigation_rows)

    validation_errors = pilot_errors + diversity_errors
    technical_pass = (
        len(pilot_paths) == 18
        and len(diversity_paths) == 12
        and not pilot_partials
        and not diversity_partials
        and not validation_errors
        and not duplicate_keys
        and not missing_pilot
        and not extra_pilot
        and not missing_diversity
        and not extra_diversity
    )
    boundary_pass = none_drift <= 1e-12 and all_zero <= 1e-12
    algorithms = sorted({str(row["algo"]) for row in full_rows})
    seeds = sorted({str(row["seed"]) for row in full_rows})
    datasets = sorted({str(row["dataset"]) for row in full_rows})
    quantizers = sorted({str(row["quantizer"]) for row in full_rows})
    models = sorted({str(row["model_tag"]) for row in full_rows})
    coverage_pass = (
        algorithms == ["GA", "GDR", "KLR", "NPO"]
        and seeds == ["2024", "42", "5000"]
        and datasets == ["muse_news", "tofu"]
        and quantizers == ["gptq_int4", "nf4", "rtn_int8"]
        and len(models) == 4
    )

    non_ga_positive = [
        row for row in diversity_rows if float(row["harmful_recovery"]) > EPS
    ]
    positive_algorithms = sorted({str(row["algo"]) for row in non_ga_positive})
    positive_settings = sorted(
        {(str(row["model_tag"]), str(row["dataset"])) for row in non_ga_positive}
    )
    positive_quantizers = sorted({str(row["quantizer"]) for row in non_ga_positive})
    generalization_pass = (
        len(positive_algorithms) >= 2
        and len(positive_settings) >= 2
        and len(positive_quantizers) >= 2
    )
    joins_pass = pilot_join_pass and diversity_join_pass
    hard_log_pass = hard_error_lines == 0

    gates = {
        "technical_30_result_gate": technical_pass,
        "strict_30_join_gate": joins_pass,
        "hard_log_gate": hard_log_pass,
        "boundary_control_gate": boundary_pass,
        "single_cell_mitigation_signal_gate": mitigation_pass,
        "factor_level_coverage_gate": coverage_pass,
        "non_ga42_generalization_gate": generalization_pass,
    }
    if all(gates.values()):
        decision = "GO_FULLQ144"
    elif not technical_pass or not joins_pass or not hard_log_pass or not boundary_pass:
        decision = "STOP_AND_REPAIR_EVIDENCE"
    else:
        decision = "STOP_FULLQ144_SIGNAL_NOT_GENERALIZED"

    recoveries = [float(row["harmful_recovery"]) for row in full_rows]
    retain_shifts = [float(row["abs_delta_retain_ce"]) for row in full_rows]
    truth_deltas = [float(row["delta_truth_ratio_log"]) for row in full_rows]
    inference_counts = Counter(str(row["inference_label"]) for row in inference)
    decision_obj = {
        "decision": decision,
        "gates": gates,
        "counts": {
            "results": len(rows),
            "full": len(full_rows),
            "protected": len(protected_rows),
            "pilot": len(pilot_rows),
            "diversity": len(diversity_rows),
            "partials": len(pilot_partials) + len(diversity_partials),
            "validation_errors": len(validation_errors),
            "join_rows": len(pilot_joins) + len(diversity_joins),
            "hard_error_lines": hard_error_lines,
        },
        "coverage": {
            "algorithms": algorithms,
            "seeds": seeds,
            "datasets": datasets,
            "models": models,
            "quantizers": quantizers,
            "is_complete_factorial": False,
            "limitation": "Diversity12 confounds non-GA algorithm with quantizer and samples only one seed per setting; full144 is required for factorial attribution.",
        },
        "signal": {
            "harmful_positive_full24": sum(value > EPS for value in recoveries),
            "overforget_positive_full24": sum(float(row["over_forgetting"]) > EPS for row in full_rows),
            "non_ga_positive_algorithms": positive_algorithms,
            "non_ga_positive_settings": positive_settings,
            "non_ga_positive_quantizers": positive_quantizers,
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
        },
        "inventory": {
            "duplicate_keys": duplicate_keys,
            "missing_pilot": missing_pilot,
            "extra_pilot": extra_pilot,
            "missing_diversity": missing_diversity,
            "extra_diversity": extra_diversity,
        },
        "statistical_note": "Per-sample inference uses deterministic paired bootstrap CIs and sign-flip tests with BH correction. It is supportive, not a preregistered confirmatory analysis.",
    }
    (out_dir / "combined30_decision.json").write_text(
        json.dumps(decision_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "combined30_validation_errors.txt").write_text(
        "\n".join(validation_errors) + ("\n" if validation_errors else ""),
        encoding="utf-8",
    )

    lines = [
        "# Gate 2 v4.1 Combined Pilot30 Analysis",
        "",
        f"Decision: **{decision}**",
        "",
        "## Evidence",
        "",
        f"- Results: {len(rows)}/30",
        f"- Full-quantized conditions: {len(full_rows)}/24",
        f"- Protected/control conditions: {len(protected_rows)}/6",
        f"- Strict joins: {len(pilot_joins) + len(diversity_joins)}/30",
        f"- Partials: {len(pilot_partials) + len(diversity_partials)}",
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
            f"- Harmful-recovery positive: {sum(value > EPS for value in recoveries)}/24",
            f"- Over-forgetting positive: {sum(float(row['over_forgetting']) > EPS for row in full_rows)}/24",
            f"- Non-GA positive algorithms: {', '.join(positive_algorithms) or 'none'}",
            f"- Non-GA positive settings: {len(positive_settings)}/4",
            f"- Non-GA positive quantizers: {', '.join(positive_quantizers) or 'none'}",
            f"- Paired inference labels: {dict(inference_counts)}",
            "",
            "## Interpretation Boundary",
            "",
            "The combined panel covers every algorithm, seed, dataset, model, and quantizer level, but it is not a complete factorial design. In Diversity12, NPO is paired with NF4, KLR with GPTQ-int4, and GDR with RTN-int8. Algorithm and quantizer effects therefore remain confounded until full144. Group summaries and correlations are descriptive, not independent causal estimates.",
            "",
            "The six protection/control results remain one Llama2-TOFU GA/42 setting. They support a mitigation demonstration but not a universal mitigation claim.",
            "",
            "## Full144 Rule",
            "",
            "Proceed only when all seven gates pass. Otherwise repair evidence failures or stop expansion when the non-GA/42 signal does not generalize.",
        ]
    )
    (out_dir / "COMBINED30_AUTOMATED_ANALYSIS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"RESULTS={len(rows)} FULL={len(full_rows)} PROTECTED={len(protected_rows)}")
    print(f"VALIDATION_ERRORS={len(validation_errors)} PARTIALS={len(pilot_partials) + len(diversity_partials)}")
    print(f"JOINS={len(pilot_joins) + len(diversity_joins)}/30 HARD_ERRORS={hard_error_lines}")
    print(f"NON_GA_POSITIVE_ALGOS={positive_algorithms}")
    print(f"COMBINED30_DECISION={decision}")
    print(f"WROTE {out_dir / 'COMBINED30_AUTOMATED_ANALYSIS.md'}")
    print(f"WROTE {out_dir / 'combined30_decision.json'}")
    if decision == "STOP_AND_REPAIR_EVIDENCE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
