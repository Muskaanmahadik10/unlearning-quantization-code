#!/usr/bin/env python3
"""Independent, torch-free publication audit for a frozen Gate 2 v4.1 Full144 archive.

This program intentionally does not import bridge producers, validators, or analyzers.
It treats the frozen archive as hostile input and rebuilds inventory, integrity,
pairing, control, statistical, claim, and resource gates with the Python standard
library only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import tarfile
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA = "gate2_bridge_v4.1"
DECISION_INVALID = "BRIDGE_INVALID_STOP_REPAIR"
DECISION_METRICS = "BRIDGE_CERTIFIED_PROCEED_METRICS"
DECISION_MITIGATION = "BRIDGE_CERTIFIED_MITIGATION_ELIGIBLE"
PANELS = {
    "pilot18": ("gate2_v4_bridge_standard_v41_pilot18", 18),
    "diversity12": ("gate2_v4_bridge_standard_v41_diversity12", 12),
    "remainder120": ("gate2_v4_bridge_standard_v41_remainder120", 120),
}
SETTINGS = (
    ("llama2_7b_adamw", "tofu"),
    ("llama3_8b_adamw", "tofu"),
    ("llama2_muse_adamw", "muse_news"),
    ("llama3_muse_adamw", "muse_news"),
)
ALGOS = ("GA", "GDR", "NPO", "KLR")
SEEDS = ("42", "2024", "5000")
QUANTIZERS = ("nf4", "gptq_int4", "rtn_int8")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HARD_LOG_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)", re.I),
    re.compile(r"CUDA out of memory|OUT_OF_MEMORY|oom-kill", re.I),
    re.compile(r"Segmentation fault|\bKilled\b", re.I),
    re.compile(r"source lock.*fail|AssertionError", re.I),
    re.compile(r"srun: error|slurmstepd: error", re.I),
    re.compile(r"\b(CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)\b", re.I),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_archive", required=True)
    parser.add_argument("--pilot_lock_sha256", required=True)
    parser.add_argument("--diversity_lock_sha256", required=True)
    parser.add_argument("--remainder_lock_sha256", required=True)
    parser.add_argument("--v4_lock_sha256", required=True)
    parser.add_argument("--pilot_job", required=True)
    parser.add_argument("--diversity_job", required=True)
    parser.add_argument("--remainder_job", required=True)
    parser.add_argument("--pre_full_status_tsv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bootstrap_reps", type=int, default=10000)
    parser.add_argument("--permutation_reps", type=int, default=20000)
    parser.add_argument("--random_seed", type=int, default=20260831)
    return parser.parse_args(argv)


def fail_constant(value: str) -> None:
    raise ValueError(f"non-standard non-finite JSON constant: {value}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle, parse_constant=fail_constant)
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON must be an object")
    return obj


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def embedded_hash(obj: Mapping[str, Any], field: str) -> str:
    stripped = dict(obj)
    stripped.pop(field, None)
    return hashlib.sha256(canonical_json(stripped).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base: int, label: str) -> int:
    digest = hashlib.sha256(f"{base}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def nonfinite_paths(value: Any, prefix: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            errors.extend(nonfinite_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(nonfinite_paths(child, f"{prefix}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(prefix)
    return errors


@contextmanager
def archive_root(source: Path) -> Iterator[Path]:
    if source.is_dir():
        yield source
        return
    if not source.is_file() or not tarfile.is_tarfile(source):
        raise ValueError(f"frozen archive is neither a directory nor a tar archive: {source}")
    with tempfile.TemporaryDirectory(prefix="full144_audit_") as temp:
        destination = Path(temp)
        with tarfile.open(source, "r:*") as archive:
            root = destination.resolve()
            for member in archive.getmembers():
                candidate = (destination / member.name).resolve()
                if root != candidate and root not in candidate.parents:
                    raise ValueError(f"unsafe archive member: {member.name}")
            archive.extractall(destination)
        children = [item for item in destination.iterdir() if item.is_dir()]
        yield children[0] if len(children) == 1 else destination


def find_unique(root: Path, name: str, *, directory: bool = False) -> Path:
    matches = [p for p in root.rglob(name) if p.is_dir() == directory]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name!r}, found {len(matches)}")
    return matches[0]


def identity(obj: Mapping[str, Any]) -> tuple[str, str, str, str, str, str, str, int]:
    return (
        str(obj.get("model_tag", "")), str(obj.get("dataset", "")),
        str(obj.get("algo", "")), str(obj.get("seed", "")),
        str(obj.get("quantizer", "")), str(obj.get("condition", "")),
        str(obj.get("ranking_source") or "none"), int(obj.get("k", -1)),
    )


def expected_full_keys() -> set[tuple[str, str, str, str, str, str, str, int]]:
    return {
        (model, dataset, algo, seed, quantizer, "full_quantized", "none", 0)
        for model, dataset in SETTINGS for algo in ALGOS for seed in SEEDS for quantizer in QUANTIZERS
    }


def expected_pilot_keys() -> set[tuple[str, str, str, str, str, str, str, int]]:
    keys = {
        (model, dataset, "GA", "42", quantizer, "full_quantized", "none", 0)
        for model, dataset in SETTINGS for quantizer in QUANTIZERS
    }
    keys.update({
        ("llama2_7b_adamw", "tofu", "GA", "42", "nf4", "protected", "UPQI_signed", 4),
        ("llama2_7b_adamw", "tofu", "GA", "42", "nf4", "protected", "random", 4),
        ("llama2_7b_adamw", "tofu", "GA", "42", "gptq_int4", "protected", "UPQI_signed", 4),
        ("llama2_7b_adamw", "tofu", "GA", "42", "gptq_int4", "protected", "random", 4),
        ("llama2_7b_adamw", "tofu", "GA", "42", "nf4", "protected", "none", 0),
        ("llama2_7b_adamw", "tofu", "GA", "42", "nf4", "protected", "all_layers", 224),
    })
    return keys


def expected_diversity_keys() -> set[tuple[str, str, str, str, str, str, str, int]]:
    spec = (
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
    return {(m, d, a, s, q, "full_quantized", "none", 0) for m, d, a, s, q in spec}


def expected_panel_keys(panel: str) -> set[tuple[str, str, str, str, str, str, str, int]]:
    pilot = expected_pilot_keys()
    diversity = expected_diversity_keys()
    if panel == "pilot18":
        return pilot
    if panel == "diversity12":
        return diversity
    accepted = {key for key in pilot | diversity if key[5] == "full_quantized"}
    return expected_full_keys() - accepted


def metric_row_errors(prefix: str, metrics: Mapping[str, Any], require_truth: bool) -> list[str]:
    errors: list[str] = []
    keys = ["answer_ce", "answer_logprob", "answer_token_mean", "answer_token_min", "n_samples"]
    if require_truth:
        keys.append("truth_ratio_log")
    for key in keys:
        if key not in metrics or (key not in ("n_samples", "answer_token_min") and not finite_number(metrics[key])):
            errors.append(f"{prefix}_{key}_bad")
    if metrics.get("n_samples") != 128:
        errors.append(f"{prefix}_n_samples_not_128")
    if not isinstance(metrics.get("answer_token_min"), int) or int(metrics.get("answer_token_min", 0)) <= 0:
        errors.append(f"{prefix}_answer_token_min_bad")
    return errors


def sample_pair_errors(prefix: str, before: Any, after: Any, *, truth: bool) -> tuple[list[str], list[float]]:
    errors: list[str] = []
    deltas: list[float] = []
    if not isinstance(before, list) or not isinstance(after, list):
        return [f"{prefix}_per_sample_missing"], deltas
    if len(before) != 128 or len(after) != 128:
        errors.append(f"{prefix}_per_sample_count:{len(before)}/{len(after)}")
        return errors, deltas
    seen: set[str] = set()
    for index, (left, right) in enumerate(zip(before, after)):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            errors.append(f"{prefix}_row_not_object:{index}")
            continue
        left_hash, right_hash = left.get("sample_hash"), right.get("sample_hash")
        if not isinstance(left_hash, str) or not HEX64.fullmatch(left_hash):
            errors.append(f"{prefix}_sample_hash_bad:{index}")
        if left_hash != right_hash:
            errors.append(f"{prefix}_sample_identity_mismatch:{index}")
        if left_hash in seen:
            errors.append(f"{prefix}_duplicate_sample:{index}")
        seen.add(str(left_hash))
        if left.get("reportable") is not True or right.get("reportable") is not True:
            errors.append(f"{prefix}_unreportable:{index}")
        for side, row in (("bf16", left), ("condition", right)):
            if not finite_number(row.get("answer_ce")):
                errors.append(f"{prefix}_{side}_answer_ce_bad:{index}")
            if not isinstance(row.get("n_answer_tokens"), int) or int(row.get("n_answer_tokens", 0)) <= 0:
                errors.append(f"{prefix}_{side}_answer_tokens_bad:{index}")
            if truth and not finite_number(row.get("truth_ratio_log")):
                errors.append(f"{prefix}_{side}_truth_ratio_bad:{index}")
        if finite_number(left.get("answer_ce")) and finite_number(right.get("answer_ce")):
            deltas.append(float(right["answer_ce"]) - float(left["answer_ce"]))
    return errors, deltas


def validate_result(path: Path, obj: Mapping[str, Any], panel: str, expected_lock: str) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    errors.extend(f"nonfinite:{item}" for item in nonfinite_paths(obj))
    if obj.get("schema_version") != SCHEMA:
        errors.append("bad_schema")
    if obj.get("stage") != "whole_model_bridge_v4" or obj.get("result_root_policy") != "bridge_v4_only_no_v2_v3_reuse":
        errors.append("bad_stage_or_root_policy")
    if obj.get("source_lock_sha256") != expected_lock:
        errors.append("wrong_source_lock")
    provenance = obj.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("source_lock_sha256") != expected_lock:
        errors.append("provenance_lock_mismatch")
    key = identity(obj)
    model, dataset, algo, seed, quantizer, condition, ranking_source, k = key
    if (model, dataset) not in SETTINGS or algo not in ALGOS or seed not in SEEDS or quantizer not in QUANTIZERS:
        errors.append("bad_factor_level")
    expected_model_dir = f"checkpoints/{model}/unlearned/{algo}/seed_{seed}"
    evaluation = obj.get("evaluation_contract")
    if not isinstance(evaluation, Mapping):
        errors.append("missing_evaluation_contract")
        evaluation = {}
    if evaluation.get("model_dir") != expected_model_dir:
        errors.append("checkpoint_identity_path_mismatch")
    for key_name in ("n_forget", "n_retain", "manifest_n_forget", "manifest_n_retain"):
        if evaluation.get(key_name) != 128:
            errors.append(f"evaluation_{key_name}_not_128")
    if evaluation.get("n_calibration") != 128 or evaluation.get("manifest_n_calibration") != 128:
        errors.append("evaluation_calibration_not_128")
    manifest_facts: dict[str, str] = {}
    for label in ("forget", "retain"):
        descriptor = obj.get(f"{label}_manifest")
        if not isinstance(descriptor, Mapping):
            errors.append(f"missing_{label}_manifest")
            continue
        if descriptor.get("dataset") != dataset or descriptor.get("n_samples") != 128:
            errors.append(f"{label}_manifest_contract_mismatch")
        sha = str(descriptor.get("sha256", ""))
        path_value = str(descriptor.get("path", ""))
        if not HEX64.fullmatch(sha) or not path_value:
            errors.append(f"{label}_manifest_provenance_bad")
        manifest_facts[label] = f"{path_value}|{sha}"
    calibration = obj.get("calibration_manifest")
    if quantizer == "gptq_int4":
        if not isinstance(calibration, Mapping) or calibration.get("n_samples") != 128 or not HEX64.fullmatch(str(calibration.get("sha256", ""))):
            errors.append("gptq_calibration_manifest_bad")
        else:
            manifest_facts["calibration"] = f"{calibration.get('path')}|{calibration.get('sha256')}"
    target = obj.get("target_layer_names")
    protected = obj.get("protected_layer_names")
    layers = obj.get("per_layer_quantization")
    if not isinstance(target, list) or len(target) != 224 or len(set(target)) != 224:
        errors.append("target_layer_inventory_bad")
        target = []
    if not isinstance(protected, list) or len(set(protected)) != len(protected):
        errors.append("protected_layer_inventory_bad")
        protected = []
    if not isinstance(layers, list) or len(layers) != len(target):
        errors.append("per_layer_inventory_bad")
        layers = []
    material = 0
    for index, row in enumerate(layers):
        if not isinstance(row, Mapping) or row.get("layer_name") != target[index]:
            errors.append(f"per_layer_order_bad:{index}")
            continue
        if not finite_number(row.get("epsilon_l2")) or not finite_number(row.get("max_abs_delta")):
            errors.append(f"per_layer_delta_bad:{index}")
        elif bool(row.get("quantized")) and (float(row["epsilon_l2"]) > 0 or float(row["max_abs_delta"]) > 0):
            material += 1
    if condition == "full_quantized" and (ranking_source != "none" or k != 0 or protected or material == 0):
        errors.append("full_quantized_contract_bad")
    if condition == "protected":
        evidence = obj.get("ranking_evidence")
        if not isinstance(evidence, Mapping) or evidence.get("ranking_source") != ranking_source or evidence.get("k") != k:
            errors.append("ranking_evidence_contract_bad")
        else:
            selected = evidence.get("selected_rows")
            if ranking_source in ("none", "all_layers"):
                if selected not in ([], None) or evidence.get("special_selector") != ranking_source:
                    errors.append("special_selector_evidence_bad")
            elif not isinstance(selected, list) or len(selected) != k:
                errors.append("ranking_selected_count_bad")
            else:
                selected_layers = [row.get("layer_name") for row in selected if isinstance(row, Mapping)]
                if selected_layers != protected or len(set(selected_layers)) != k:
                    errors.append("ranking_selected_layers_bad")
                for row in selected:
                    if not isinstance(row, Mapping) or not isinstance(row.get("rank"), int) or int(row.get("rank", 0)) <= 0 or not finite_number(row.get("score")):
                        errors.append("ranking_selected_row_bad")
                        break
                if not HEX64.fullmatch(str(evidence.get("rankings_sha256", ""))):
                    errors.append("ranking_file_hash_bad")
    bf16 = obj.get("bf16_reference")
    condition_metrics = obj.get("condition_metrics")
    if not isinstance(bf16, Mapping) or not isinstance(condition_metrics, Mapping):
        errors.append("missing_metric_branches")
        bf16, condition_metrics = {}, {}
    forget_b, forget_c = bf16.get("forget", {}), condition_metrics.get("forget", {})
    retain_b, retain_c = bf16.get("retain", {}), condition_metrics.get("retain", {})
    if not all(isinstance(item, Mapping) for item in (forget_b, forget_c, retain_b, retain_c)):
        errors.append("metric_branch_shape_bad")
        forget_b = forget_c = retain_b = retain_c = {}
    errors.extend(metric_row_errors("bf16_forget", forget_b, True))
    errors.extend(metric_row_errors("condition_forget", forget_c, True))
    errors.extend(metric_row_errors("bf16_retain", retain_b, False))
    errors.extend(metric_row_errors("condition_retain", retain_c, False))
    for branch_name, branch in (
        ("bf16_forget", forget_b), ("condition_forget", forget_c),
        ("bf16_retain", retain_b), ("condition_retain", retain_c),
    ):
        sample_rows = branch.get("per_sample")
        if isinstance(sample_rows, list) and sample_rows:
            token_counts = [row.get("n_answer_tokens") for row in sample_rows if isinstance(row, Mapping)]
            if len(token_counts) == len(sample_rows) and all(isinstance(value, int) and value > 0 for value in token_counts):
                if not math.isclose(float(branch.get("answer_token_mean", math.nan)), statistics.fmean(token_counts), abs_tol=1e-9):
                    errors.append(f"{branch_name}_answer_token_mean_mismatch")
                if branch.get("answer_token_min") != min(token_counts):
                    errors.append(f"{branch_name}_answer_token_min_mismatch")
    forget_pair_errors, sample_deltas = sample_pair_errors(
        "forget", forget_b.get("per_sample"), forget_c.get("per_sample"), truth=True
    )
    retain_pair_errors, _ = sample_pair_errors(
        "retain", retain_b.get("per_sample"), retain_c.get("per_sample"), truth=False
    )
    errors.extend(forget_pair_errors)
    errors.extend(retain_pair_errors)
    delta = obj.get("deltas")
    if not isinstance(delta, Mapping):
        errors.append("missing_deltas")
        delta = {}
    aggregate_delta = float(forget_c.get("answer_ce", math.nan)) - float(forget_b.get("answer_ce", math.nan)) if finite_number(forget_c.get("answer_ce")) and finite_number(forget_b.get("answer_ce")) else math.nan
    if not finite_number(delta.get("delta_forget_ce")) or not math.isclose(float(delta.get("delta_forget_ce", math.nan)), aggregate_delta, abs_tol=1e-8):
        errors.append("delta_forget_identity_bad")
    if sample_deltas and not math.isclose(statistics.fmean(sample_deltas), aggregate_delta, abs_tol=1e-6):
        errors.append("sample_aggregate_delta_mismatch")
    forget_hashes = [str(row.get("sample_hash", "")) for row in forget_b.get("per_sample", []) if isinstance(row, Mapping)]
    retain_hashes = [str(row.get("sample_hash", "")) for row in retain_b.get("per_sample", []) if isinstance(row, Mapping)]
    return sorted(set(errors)), {
        "panel": panel, "result_path": str(path), "cell_id": str(obj.get("cell_id", "")),
        "model_tag": model, "dataset": dataset, "algo": algo, "seed": seed,
        "quantizer": quantizer, "condition": condition, "ranking_source": ranking_source, "k": k,
        "source_lock_sha256": str(obj.get("source_lock_sha256", "")),
        "checkpoint_identity": str(evaluation.get("model_dir", "")),
        "forget_manifest": manifest_facts.get("forget", ""), "retain_manifest": manifest_facts.get("retain", ""),
        "calibration_manifest": manifest_facts.get("calibration", ""),
        "forget_rows": len(forget_b.get("per_sample", [])) if isinstance(forget_b.get("per_sample"), list) else 0,
        "retain_rows": len(retain_b.get("per_sample", [])) if isinstance(retain_b.get("per_sample"), list) else 0,
        "forget_sample_set_sha256": hashlib.sha256(canonical_json(forget_hashes).encode("utf-8")).hexdigest() if forget_hashes else "",
        "retain_sample_set_sha256": hashlib.sha256(canonical_json(retain_hashes).encode("utf-8")).hexdigest() if retain_hashes else "",
        "materially_changed_layers": material, "delta_forget_ce": delta.get("delta_forget_ce"),
        "harmful_recovery": delta.get("harmful_recovery"), "delta_retain_ce": delta.get("delta_retain_ce"),
        "elapsed_s": obj.get("elapsed_s"),
        "file_sha256": sha256_file(path), "status": "PASS" if not errors else "FAIL",
        "errors": ";".join(sorted(set(errors))), "sample_deltas": sample_deltas,
    }


def numeric_projection(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: numeric_projection(child) for key, child in sorted(value.items()) if key not in {"elapsed_s", "provenance"}}
    if isinstance(value, list):
        return [numeric_projection(child) for child in value]
    return value


def max_numeric_drift(left: Any, right: Any) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return math.inf
        return max((max_numeric_drift(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return math.inf
        return max((max_numeric_drift(a, b) for a, b in zip(left, right)), default=0.0)
    if finite_number(left) and finite_number(right):
        return abs(float(left) - float(right))
    return 0.0 if left == right else math.inf


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position)); upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def infer_values(values: Sequence[float], label: str, args: argparse.Namespace) -> dict[str, Any]:
    clean = [float(value) for value in values]
    if not clean:
        return {"n_blocks": 0, "mean_effect": math.nan, "median_effect": math.nan, "ci95_low": math.nan, "ci95_high": math.nan, "sign_flip_p": math.nan, "effect_dz": math.nan, "loo_min_mean": math.nan, "loo_max_mean": math.nan, "loo_sign_stable": False}
    rng = random.Random(stable_seed(args.random_seed, label))
    boot = [statistics.fmean(rng.choice(clean) for _ in clean) for _ in range(args.bootstrap_reps)]
    observed = statistics.fmean(clean)
    exceed = 0
    for _ in range(args.permutation_reps):
        permuted = statistics.fmean(value * (-1 if rng.getrandbits(1) else 1) for value in clean)
        exceed += abs(permuted) >= abs(observed)
    standard_deviation = statistics.stdev(clean) if len(clean) > 1 else 0.0
    loo = [statistics.fmean(clean[:index] + clean[index + 1:]) for index in range(len(clean))] if len(clean) > 1 else [observed]
    return {
        "n_blocks": len(clean), "mean_effect": observed, "median_effect": statistics.median(clean),
        "ci95_low": percentile(boot, 0.025), "ci95_high": percentile(boot, 0.975),
        "sign_flip_p": (exceed + 1) / (args.permutation_reps + 1),
        "effect_dz": observed / standard_deviation if standard_deviation else math.nan,
        "loo_min_mean": min(loo), "loo_max_mean": max(loo),
        "loo_sign_stable": all(value == 0 or math.copysign(1, value) == math.copysign(1, observed) for value in loo),
    }


def bh_adjust(rows: list[dict[str, Any]], family_key: str = "test_family") -> None:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if finite_number(row.get("sign_flip_p")):
            groups[str(row[family_key])].append(index)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: float(rows[index]["sign_flip_p"]))
        running = 1.0
        for reverse in range(len(ordered) - 1, -1, -1):
            index = ordered[reverse]
            rank = reverse + 1
            running = min(running, float(rows[index]["sign_flip_p"]) * len(ordered) / rank)
            rows[index]["q_bh"] = min(1.0, running)


def cell_inference(full_records: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in full_records:
        stats = infer_values(record["sample_deltas"], f"cell:{record['cell_id']}", args)
        rows.append({
            "row_type": "cell", "test_family": "cell_bh144", "contrast": record["cell_id"],
            "model_tag": record["model_tag"], "dataset": record["dataset"], "algo": record["algo"],
            "seed": record["seed"], "quantizer": record["quantizer"], **stats,
        })
    bh_adjust(rows)
    for row in rows:
        q = float(row.get("q_bh", 1.0)); low = float(row["ci95_low"]); high = float(row["ci95_high"])
        row["inference_label"] = "supported_harmful_recovery" if high < 0 and q < 0.05 else "supported_over_forgetting" if low > 0 and q < 0.05 else "inconclusive"
    return rows


def aggregate_map(full_records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, str, str], float]:
    return {(str(r["model_tag"]), str(r["dataset"]), str(r["algo"]), str(r["seed"]), str(r["quantizer"])): float(r["delta_forget_ce"]) for r in full_records}


def factorial_inference(full_records: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    data = aggregate_map(full_records)
    factors = {"model": [m for m, _ in SETTINGS], "dataset": ["tofu", "muse_news"], "algo": list(ALGOS), "quantizer": list(QUANTIZERS)}
    index_position = {"model": 0, "dataset": 1, "algo": 2, "seed": 3, "quantizer": 4}
    rows: list[dict[str, Any]] = []
    for factor in ("quantizer", "algo", "model", "dataset"):
        levels = sorted(set(factors[factor]))
        position = index_position[factor]
        for left_index in range(len(levels)):
            for right_index in range(left_index + 1, len(levels)):
                left, right = levels[left_index], levels[right_index]
                differences: list[float] = []
                for key, value in data.items():
                    if key[position] != left:
                        continue
                    paired = list(key); paired[position] = right
                    paired_key = tuple(paired)
                    if paired_key in data:
                        differences.append(value - data[paired_key])
                label = f"main:{factor}:{left}-{right}"
                rows.append({"row_type": "main_effect", "test_family": "main_effects", "contrast": label, "factor_a": factor, "level_a1": left, "level_a2": right, "factor_b": "", "level_b1": "", "level_b2": "", **infer_values(differences, label, args)})
    interaction_pairs = (("quantizer", "algo"), ("quantizer", "model"), ("quantizer", "dataset"), ("algo", "model"), ("algo", "dataset"), ("model", "dataset"))
    for factor_a, factor_b in interaction_pairs:
        pos_a, pos_b = index_position[factor_a], index_position[factor_b]
        levels_a, levels_b = sorted(set(factors[factor_a])), sorted(set(factors[factor_b]))
        for a_index in range(len(levels_a)):
            for a_right in range(a_index + 1, len(levels_a)):
                for b_index in range(len(levels_b)):
                    for b_right in range(b_index + 1, len(levels_b)):
                        a1, a2 = levels_a[a_index], levels_a[a_right]
                        b1, b2 = levels_b[b_index], levels_b[b_right]
                        differences: list[float] = []
                        for key in data:
                            if key[pos_a] != a1 or key[pos_b] != b1:
                                continue
                            k12 = list(key); k12[pos_b] = b2
                            k21 = list(key); k21[pos_a] = a2
                            k22 = list(key); k22[pos_a] = a2; k22[pos_b] = b2
                            if tuple(k12) in data and tuple(k21) in data and tuple(k22) in data:
                                differences.append((data[key] - data[tuple(k12)]) - (data[tuple(k21)] - data[tuple(k22)]))
                        label = f"interaction:{factor_a}x{factor_b}:{a1}-{a2}:{b1}-{b2}"
                        rows.append({"row_type": "interaction", "test_family": "interactions", "contrast": label, "factor_a": factor_a, "level_a1": a1, "level_a2": a2, "factor_b": factor_b, "level_b1": b1, "level_b2": b2, **infer_values(differences, label, args)})
    bh_adjust(rows)
    for row in rows:
        row["claim_status"] = "SUPPORTED" if float(row.get("q_bh", 1.0)) < 0.05 and (float(row["ci95_high"]) < 0 or float(row["ci95_low"]) > 0) else "DESCRIPTIVE_OR_INCONCLUSIVE"
    return rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def slurm_records(root: Path, jobs: Mapping[str, str]) -> tuple[list[dict[str, Any]], list[str]]:
    accounting = find_unique(root, "slurm_accounting.txt")
    text = accounting.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for panel, job in jobs.items():
        line = next((line for line in text.splitlines() if re.match(rf"^\s*{re.escape(job)}\s", line)), "")
        state = "COMPLETED" if re.search(r"\bCOMPLETED\b", line) else "UNKNOWN"
        exit_code = "0:0" if re.search(r"\b0:0\b", line) else "UNKNOWN"
        elapsed_match = re.search(r"\b(\d+-)?\d{2}:\d{2}:\d{2}\b", line)
        elapsed = elapsed_match.group(0) if elapsed_match else ""
        partition_match = re.search(r"\b(8gpus|16gpus|32gpus|64gpus|dev|normal2?|4nodes)\b", line)
        partition = partition_match.group(1) if partition_match else ""
        gpu_match = re.match(r"(\d+)gpus", partition)
        allocated = int(gpu_match.group(1)) if gpu_match else 0
        seconds = elapsed_seconds(elapsed) if elapsed else 0
        if state != "COMPLETED" or exit_code != "0:0":
            errors.append(f"slurm_parent_bad:{panel}:{job}:{state}:{exit_code}")
        rows.append({"panel": panel, "job_id": job, "partition": partition, "state": state, "exit_code": exit_code, "elapsed": elapsed, "allocated_gpus": allocated, "allocated_gpu_hours": allocated * seconds / 3600 if allocated else "", "accounting_line": line})
    return rows, errors


def elapsed_seconds(value: str) -> int:
    days = 0
    if "-" in value:
        day, value = value.split("-", 1); days = int(day)
    hours, minutes, seconds = (int(item) for item in value.split(":"))
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def scan_logs(root: Path, jobs: Iterable[str]) -> list[str]:
    findings: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or not any(job in path.name for job in jobs):
            continue
        if path.suffix.lower() not in {".log", ".out", ".err", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), 1):
            if any(pattern.search(line) for pattern in HARD_LOG_PATTERNS):
                findings.append(f"{path}:{line_number}:{line.strip()}")
    return findings


def audit(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_locks = {"pilot18": args.pilot_lock_sha256, "diversity12": args.diversity_lock_sha256, "remainder120": args.remainder_lock_sha256}
    jobs = {"pilot18": str(args.pilot_job), "diversity12": str(args.diversity_job), "remainder120": str(args.remainder_job)}
    global_errors: list[str] = []
    records: list[dict[str, Any]] = []
    objects: dict[tuple[str, str, str, str, str, str, str, int], dict[str, Any]] = {}
    with archive_root(Path(args.frozen_archive)) as root:
        for panel, (root_name, expected_count) in PANELS.items():
            try:
                panel_root = find_unique(root, root_name, directory=True)
            except ValueError as exc:
                global_errors.append(str(exc)); continue
            paths = sorted(panel_root.rglob("bridge_v4.json"))
            partials = sorted(panel_root.rglob("bridge_v4.json.partial"))
            if len(paths) != expected_count:
                global_errors.append(f"{panel}_result_count:{len(paths)}/{expected_count}")
            if partials:
                global_errors.append(f"{panel}_partials:{len(partials)}")
            actual_keys: list[tuple[str, str, str, str, str, str, str, int]] = []
            for path in paths:
                try:
                    obj = load_json(path)
                    errors, record = validate_result(path, obj, panel, expected_locks[panel])
                except Exception as exc:
                    errors = [f"json_or_validation_exception:{type(exc).__name__}:{exc}"]
                    record = {"panel": panel, "result_path": str(path), "cell_id": "", "status": "FAIL", "errors": ";".join(errors), "sample_deltas": []}
                    obj = {}
                record["status"] = "PASS" if not errors else "FAIL"
                record["errors"] = ";".join(errors)
                records.append(record)
                if obj:
                    key = identity(obj); actual_keys.append(key)
                    if key in objects:
                        global_errors.append(f"duplicate_identity:{key}")
                    objects[key] = obj
            expected_keys = expected_panel_keys(panel)
            for missing in sorted(expected_keys - set(actual_keys)):
                global_errors.append(f"missing_{panel}:{missing}")
            for extra in sorted(set(actual_keys) - expected_keys):
                global_errors.append(f"extra_{panel}:{extra}")
        manifest_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
        for record in records:
            for label in ("forget", "retain"):
                value = str(record.get(f"{label}_manifest", ""))
                if value:
                    manifest_groups[(str(record.get("dataset", "")), label)].add(value)
        for group, values in manifest_groups.items():
            if len(values) != 1:
                global_errors.append(f"manifest_inconsistency:{group}:{len(values)}")
        manifest_cache: dict[str, tuple[Path, dict[str, Any]]] = {}
        for record in records:
            for label in ("forget_manifest", "retain_manifest", "calibration_manifest"):
                descriptor = str(record.get(label, ""))
                if not descriptor:
                    continue
                manifest_path, manifest_sha = descriptor.split("|", 1)
                cache_key = f"{manifest_path}|{manifest_sha}"
                if cache_key in manifest_cache:
                    continue
                matches = [path for path in root.rglob(Path(manifest_path).name) if path.is_file()]
                if len(matches) != 1:
                    global_errors.append(f"frozen_manifest_file_count:{manifest_path}:{len(matches)}")
                    continue
                try:
                    manifest_obj = load_json(matches[0])
                    actual = embedded_hash(manifest_obj, "manifest_sha256")
                    if actual != manifest_sha or manifest_obj.get("manifest_sha256") != actual:
                        global_errors.append(f"frozen_manifest_hash_mismatch:{manifest_path}")
                    samples = manifest_obj.get("samples")
                    if not isinstance(samples, list) or len(samples) != 128:
                        global_errors.append(f"frozen_manifest_sample_count:{manifest_path}")
                    manifest_cache[cache_key] = (matches[0], manifest_obj)
                except Exception as exc:
                    global_errors.append(f"frozen_manifest_bad:{manifest_path}:{exc}")
        for dataset in ("tofu", "muse_news"):
            for sample_kind in ("forget", "retain"):
                digests = {
                    str(record.get(f"{sample_kind}_sample_set_sha256", ""))
                    for record in records if record.get("dataset") == dataset
                }
                if len(digests) != 1 or "" in digests:
                    global_errors.append(f"sample_set_inconsistency:{dataset}:{sample_kind}:{len(digests)}")
        full_records = [record for record in records if record.get("condition") == "full_quantized"]
        full_objects = {key: obj for key, obj in objects.items() if key[5] == "full_quantized"}
        for checkpoint in {(key[0], key[1], key[2], key[3]) for key in full_objects}:
            matches = [(key, obj) for key, obj in full_objects.items() if key[:4] == checkpoint]
            if len(matches) != 3:
                global_errors.append(f"bf16_quantizer_replication_count:{checkpoint}:{len(matches)}")
                continue
            reference = numeric_projection(matches[0][1].get("bf16_reference"))
            for key, obj in matches[1:]:
                if max_numeric_drift(reference, numeric_projection(obj.get("bf16_reference"))) > 1e-12:
                    global_errors.append(f"bf16_reproducibility_drift:{checkpoint}:{key[4]}")
        base = ("llama2_7b_adamw", "tofu", "GA", "42", "nf4")
        full_key = base + ("full_quantized", "none", 0)
        none_key = base + ("protected", "none", 0)
        all_key = base + ("protected", "all_layers", 224)
        none_drift = math.inf; all_drift = math.inf
        if all(key in objects for key in (full_key, none_key, all_key)):
            full_obj, none_obj, all_obj = objects[full_key], objects[none_key], objects[all_key]
            none_fields = {name: full_obj.get(name) for name in ("bf16_reference", "condition_metrics", "deltas", "compression", "per_layer_quantization")}
            none_other = {name: none_obj.get(name) for name in ("bf16_reference", "condition_metrics", "deltas", "compression", "per_layer_quantization")}
            none_drift = max_numeric_drift(numeric_projection(none_fields), numeric_projection(none_other))
            all_drift = max_numeric_drift(numeric_projection(all_obj.get("condition_metrics")), numeric_projection(all_obj.get("bf16_reference")))
            if none_drift > 1e-12: global_errors.append(f"none_k0_boundary_drift:{none_drift}")
            if all_drift > 1e-12: global_errors.append(f"all_k224_boundary_drift:{all_drift}")
        else:
            global_errors.append("boundary_controls_missing")
        join_count = 0
        for panel in PANELS:
            try:
                join = find_unique(root, f"{panel.replace('18','18').replace('12','12').replace('120','120')}_join.tsv")
            except ValueError:
                candidates = list(root.rglob({"pilot18": "pilot18_join.tsv", "diversity12": "diversity12_join.tsv", "remainder120": "remainder120_join.tsv"}[panel]))
                if len(candidates) != 1:
                    global_errors.append(f"join_file_missing:{panel}"); continue
                join = candidates[0]
            rows = read_tsv(join); join_count += len(rows)
            if len(rows) != PANELS[panel][1] or any(row.get("status") != "PASS" for row in rows):
                global_errors.append(f"join_gate_bad:{panel}:{len(rows)}")
        if join_count != 150:
            global_errors.append(f"join_total:{join_count}/150")
        status_rows = read_tsv(Path(args.pre_full_status_tsv))
        if len(status_rows) != 144:
            global_errors.append(f"pre_full_status_count:{len(status_rows)}/144")
        for index, row in enumerate(status_rows):
            normalized = " ".join(str(value) for value in row.values()).lower()
            if "clean_complete" not in normalized or "passed" not in normalized or "lock_for_metrics" not in normalized:
                global_errors.append(f"pre_full_status_bad:{index}")
        slurm_rows, slurm_errors = slurm_records(root, jobs)
        global_errors.extend(slurm_errors)
        log_findings = scan_logs(root, jobs.values())
        global_errors.extend(f"hard_log:{item}" for item in log_findings)
        lock_files = [path for path in root.rglob("*.json") if "provenance" in path.parts and "LOCK" in path.name]
        found_lock_shas = set()
        for path in lock_files:
            try:
                lock_obj = load_json(path)
                embedded = str(lock_obj.get("source_lock_sha256", ""))
                if embedded_hash(lock_obj, "source_lock_sha256") != embedded:
                    global_errors.append(f"source_lock_internal_hash_mismatch:{path}")
                found_lock_shas.add(embedded)
            except Exception as exc:
                global_errors.append(f"lock_json_bad:{path}:{exc}")
        for lock in (*expected_locks.values(), args.v4_lock_sha256):
            if lock not in found_lock_shas:
                global_errors.append(f"frozen_lock_missing:{lock}")
    integrity_fields = [key for key in records[0].keys() if key != "sample_deltas"] if records else ["status", "errors"]
    write_tsv(out_dir / "full144_cell_integrity.tsv", records, integrity_fields)
    preliminary_integrity = (
        len(records) == 150
        and len(full_records) == 144
        and not global_errors
        and all(record.get("status") == "PASS" for record in records)
    )
    cell_rows = cell_inference(full_records, args) if preliminary_integrity else []
    factor_rows = factorial_inference(full_records, args) if preliminary_integrity else []
    inference_rows = cell_rows + factor_rows
    write_tsv(out_dir / "full144_factorial_inference.tsv", inference_rows)
    structural_pass = preliminary_integrity
    supported = [row for row in cell_rows if row.get("inference_label") == "supported_harmful_recovery"]
    supported_cells = {row["contrast"] for row in supported}
    supported_source = [record for record in full_records if record.get("cell_id") in supported_cells]
    spans = {
        "settings": len({(r["model_tag"], r["dataset"]) for r in supported_source}),
        "algorithms": len({r["algo"] for r in supported_source}),
        "quantizers": len({r["quantizer"] for r in supported_source}),
    }
    mitigation_eligible = len(supported) >= 12 and min(spans.values(), default=0) >= 2
    decision = DECISION_INVALID if not structural_pass else DECISION_MITIGATION if mitigation_eligible else DECISION_METRICS
    claim_rows = [
        {"claim": "balanced_target_layer_quantization_bridge", "status": "PASS" if structural_pass else "REJECT", "reason": "All frozen structural gates" if structural_pass else "One or more structural gates failed"},
        {"claim": "exact_GQDP_superiority", "status": "REJECT", "reason": "Exact source-locked GQDP endpoint files are not part of Full144"},
        {"claim": "general_mitigation", "status": "REJECT", "reason": "Requires matched multi-cell mitigation Pilot24 and gated expansion"},
        {"claim": "official_benchmark_success", "status": "REJECT", "reason": "Requires maintained official TOFU/MUSE evaluation"},
        {"claim": "certified_deletion", "status": "REJECT", "reason": "Bridge proxies do not certify deletion"},
        {"claim": "deployment_wide_behavior", "status": "REJECT", "reason": "Only declared target layers were quantized"},
        {"claim": "paper_ready", "status": "REJECT", "reason": "Predictor, mitigation, official metrics, provenance, and venue-fit gates remain"},
    ]
    write_tsv(out_dir / "full144_claim_gate.tsv", claim_rows)
    for row in slurm_rows:
        panel_records = [record for record in records if record.get("panel") == row["panel"]]
        row["result_count"] = len(panel_records)
        row["summed_cell_gpu_hours"] = sum(float(record.get("elapsed_s", 0.0)) for record in panel_records if finite_number(record.get("elapsed_s"))) / 3600.0
        allocated_hours = row.get("allocated_gpu_hours")
        row["cell_time_over_allocated_gpu_time"] = (
            row["summed_cell_gpu_hours"] / float(allocated_hours)
            if finite_number(allocated_hours) and float(allocated_hours) > 0 else ""
        )
    write_tsv(out_dir / "full144_resource_efficiency.tsv", slurm_rows)
    result_hashes = sorted((str(record.get("result_path", "")), str(record.get("file_sha256", ""))) for record in records)
    decision_obj = {
        "decision": decision,
        "structural_gate_pass": structural_pass,
        "counts": {"results": len(records), "full_conditions": len(full_records), "controls": len(records) - len(full_records), "joins": join_count, "integrity_failures": sum(r.get("status") != "PASS" for r in records), "global_errors": len(global_errors), "supported_harmful_recovery_cells": len(supported)},
        "mitigation_eligibility_span": spans,
        "boundary_controls": {"none_k0_vs_full_max_drift": none_drift if math.isfinite(none_drift) else None, "all_layers_k224_vs_bf16_max_drift": all_drift if math.isfinite(all_drift) else None},
        "limitations": ["Checkpoint identity is path-based; no checkpoint-content hash exists in bridge JSON.", "Retain per-sample certification fails when producer outputs contain aggregate-only retain metrics.", "Full144 is target-layer bridge evidence, not official benchmark or deployment evidence."],
        "next_phase": (
            "repair_only_failed_stage1_evidence_gates" if decision == DECISION_INVALID
            else "exact_gqdp_and_predictor_baselines_then_mitigation_pilot24"
        ),
        "wallet_policy_after_stage1": {
            "snapshot_name": "B0",
            "exact_gqdp_and_predictor_fraction": 0.30,
            "mitigation_pilot24_and_gated_bridge48_fraction": 0.40,
            "official_evaluation_and_reproducibility_fraction": 0.30,
            "unused_transfer_order": ["official_evaluation", "mitigation_expansion"],
        },
        "errors": sorted(set(global_errors + [f"{r.get('cell_id')}:{r.get('errors')}" for r in records if r.get("status") != "PASS"])),
        "result_set_sha256": hashlib.sha256(canonical_json(sorted((str(record.get("panel", "")), str(record.get("cell_id", "")), str(record.get("condition", "")), str(record.get("ranking_source", "")), str(record.get("k", "")), str(record.get("file_sha256", ""))) for record in records)).encode("utf-8")).hexdigest(),
        "statistical_seed": args.random_seed,
        "bootstrap_reps": args.bootstrap_reps,
        "permutation_reps": args.permutation_reps,
    }
    (out_dir / "full144_next_phase_decision.json").write_text(json.dumps(decision_obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# Full144 Brutal Publication Audit", "", f"Decision: **{decision}**", "",
        "## Structural Certification", "", f"- Results: {len(records)}/150", f"- Full factorial: {len(full_records)}/144", f"- Strict joins: {join_count}/150", f"- Integrity failures: {decision_obj['counts']['integrity_failures']}", f"- Global gate errors: {len(global_errors)}", "",
        "## Statistical Evidence", "", f"- BH-supported harmful-recovery cells: {len(supported)}/144", f"- Span: {spans}", "- Experimental units are paired cells and matched checkpoint blocks. Samples, tokens, and layers are not promoted to independent research replicates.", "",
        "## Mandatory Claim Boundary", "", "Full144 can certify only the balanced target-layer quantization bridge. Exact GQDP superiority, general mitigation, official benchmark success, deletion, deployment-wide behavior, and paper readiness remain rejected until their separate gates pass.", "",
        "## Known Evidence Gap", "", "The current v4.1 producer stores 128 paired forget rows but retain metrics are aggregate-only. Under the requested audit standard, missing 128-row retain evidence is a structural failure and must not be relabeled as a pass.", "",
        "## Errors", "",
    ]
    lines.extend(f"- {error}" for error in decision_obj["errors"][:200])
    if not decision_obj["errors"]:
        lines.append("- None")
    (out_dir / "FULL144_BRUTAL_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for output in ("FULL144_BRUTAL_AUDIT.md", "full144_cell_integrity.tsv", "full144_factorial_inference.tsv", "full144_claim_gate.tsv", "full144_resource_efficiency.tsv", "full144_next_phase_decision.json"):
        if not (out_dir / output).is_file():
            raise RuntimeError(f"required output missing: {output}")
    return decision, decision_obj


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.bootstrap_reps < 100 or args.permutation_reps < 100:
        raise SystemExit("bootstrap and permutation repetitions must each be at least 100")
    decision, result = audit(args)
    print(f"FULL144_PUBLICATION_DECISION={decision}")
    print(f"RESULTS={result['counts']['results']}/150 FULL={result['counts']['full_conditions']}/144 JOINS={result['counts']['joins']}/150")
    print(f"INTEGRITY_FAILURES={result['counts']['integrity_failures']} GLOBAL_ERRORS={result['counts']['global_errors']}")
    print(f"OUTPUT_DIR={args.out_dir}")
    if decision == DECISION_INVALID:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
