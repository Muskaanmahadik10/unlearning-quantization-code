#!/usr/bin/env python3
"""Torch-free Gate 2 v4 contracts, hashing, and validation helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


MANIFEST_SCHEMA = "gate2_manifest_v4.0"
MANIFEST_SELECTION_CONTRACT_ID = "deterministic_hash_without_replacement_seed_dataset_split_index_sample_v4_0"
GT_SCHEMA = "gate2_gt_v4.0"
UPQI_SCHEMA = "gate2_upqi_v4.0"
LOSS_CONTRACT_ID = "answer_only_causal_lm_v4"
MASK_CONTRACT_ID = "first_answer_token_prefix_excluded_v4"
NO_FALLBACK_LABEL_CONTRACT_ID = "no_synthetic_label_fallback_v4_0"
TRUTH_RATIO_CONTRACT_ID = "log_space_mean_perturbation_over_answer_probability_required_v4_0"
UPQI_ABS_CONTRACT_ID = "path_integral_abs_grad_times_epsilon_v4_0"
SOURCE_LOCK_SCHEMA = "gate2_v4_source_lock_v1"
RESULTS_ROOT_REQUIRED_FRAGMENT = "gate2_v4_standard"

ALGOS = ("GA", "GDR", "NPO", "KLR")
QUANTIZERS = ("nf4", "gptq_int4", "rtn_int8")
SEEDS = ("42", "2024", "5000")

CORE_SETTINGS = {
    "l2_A": {"model_tag": "llama2_7b_adamw", "dataset": "tofu"},
    "l3_A": {"model_tag": "llama3_8b_adamw", "dataset": "tofu"},
    "l2m_A": {"model_tag": "llama2_muse_adamw", "dataset": "muse_news"},
    "l3m_A": {"model_tag": "llama3_muse_adamw", "dataset": "muse_news"},
}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def logmeanexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("logmeanexp requires at least one value")
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("logmeanexp received non-finite value")
    m = max(float(v) for v in values)
    return m + math.log(sum(math.exp(float(v) - m) for v in values) / float(len(values)))


def sample_hash(sample: Mapping[str, Any]) -> str:
    payload = {
        "id": sample.get("id", sample.get("idx", sample.get("sample_id"))),
        "question": sample.get("question", sample.get("prompt", "")),
        "answer": sample.get("answer", sample.get("completion", "")),
    }
    return sha256_text(canonical_json(payload))


def perturbation_hash(text: str) -> str:
    return sha256_text(str(text).strip())


def manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    stripped = dict(manifest)
    stripped.pop("manifest_sha256", None)
    return sha256_text(canonical_json(stripped))


def assert_manifest_sha(manifest: Mapping[str, Any]) -> None:
    got = manifest.get("manifest_sha256")
    actual = manifest_fingerprint(manifest)
    if got != actual:
        raise ValueError(f"manifest_sha256 mismatch: got={got} actual={actual}")


def assert_forget_manifest_ready(manifest: Mapping[str, Any], *, n_samples: Optional[int] = None) -> None:
    assert_manifest_sha(manifest)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"wrong manifest schema: {manifest.get('schema_version')}")
    if manifest.get("selection_contract_id") != MANIFEST_SELECTION_CONTRACT_ID:
        raise ValueError("wrong or missing selection_contract_id")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest has no samples")
    if n_samples is not None and len(samples) != int(n_samples):
        raise ValueError(f"manifest sample count {len(samples)} != {n_samples}")
    for idx, sample in enumerate(samples):
        perturbations = sample.get("perturbations")
        if not isinstance(perturbations, list) or not perturbations:
            raise ValueError(f"sample {idx} has no perturbations")
        if int(sample.get("n_perturbations", len(perturbations))) <= 0:
            raise ValueError(f"sample {idx} has nonpositive n_perturbations")
        if not sample.get("sample_hash"):
            raise ValueError(f"sample {idx} missing sample_hash")
        for pidx, perturbation in enumerate(perturbations):
            if not str(perturbation.get("answer", "")).strip():
                raise ValueError(f"sample {idx} perturbation {pidx} missing answer")
            if not perturbation.get("perturbation_hash"):
                raise ValueError(f"sample {idx} perturbation {pidx} missing hash")


def loss_contract(n_samples: int, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "loss_contract_id": LOSS_CONTRACT_ID,
        "mask_contract_id": MASK_CONTRACT_ID,
        "label_fallback_contract_id": NO_FALLBACK_LABEL_CONTRACT_ID,
        "truth_ratio_contract_id": TRUTH_RATIO_CONTRACT_ID,
        "objective": "mean negative log likelihood over answer tokens only",
        "causal_shift": "first unmasked label is first answer token; prefix tokens are excluded",
        "truth_ratio": "computed in log-space and required for every GT sample/layer",
        "n_samples": int(n_samples),
        "manifest_schema": manifest.get("schema_version"),
        "manifest_sha256": manifest.get("manifest_sha256"),
    }


def source_lock_hash(lock: Mapping[str, Any]) -> str:
    stripped = dict(lock)
    stripped.pop("source_lock_sha256", None)
    return sha256_text(canonical_json(stripped))


def load_source_lock(path: str | Path) -> Mapping[str, Any]:
    lock = load_json(path)
    if lock.get("schema_version") != SOURCE_LOCK_SCHEMA:
        raise ValueError(f"wrong source lock schema: {lock.get('schema_version')}")
    expected = lock.get("source_lock_sha256")
    actual = source_lock_hash(lock)
    if expected and expected != actual:
        raise ValueError(f"source lock hash mismatch: got={expected} actual={actual}")
    return lock


def verify_source_lock_files(lock: Mapping[str, Any]) -> None:
    files = lock.get("files")
    if not isinstance(files, Mapping) or not files:
        raise SystemExit("source lock has no files to verify")
    problems: List[str] = []
    for raw_path, expected_sha in sorted(files.items()):
        rel_path = Path(str(raw_path))
        if not rel_path.exists():
            problems.append(f"missing:{raw_path}")
            continue
        actual_sha = sha256_file(rel_path)
        if actual_sha != str(expected_sha):
            problems.append(f"sha_mismatch:{raw_path}:current={actual_sha}:locked={expected_sha}")
    if problems:
        detail = "\n".join(problems[:20])
        more = "" if len(problems) <= 20 else f"\n... {len(problems) - 20} more"
        raise SystemExit(f"source lock file verification failed:\n{detail}{more}")


def require_source_lock(
    *,
    script_path: str | Path,
    source_lock_path: str | Path,
    expected_script_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    script_path = Path(script_path).resolve()
    lock = load_source_lock(source_lock_path)
    verify_source_lock_files(lock)
    script_sha = sha256_file(script_path)
    rel = str(script_path).replace("\\", "/")
    files = lock.get("files", {})
    locked_sha = None
    for key, value in files.items():
        if key.replace("\\", "/") in rel:
            locked_sha = value
            break
    if expected_script_sha256 and script_sha != expected_script_sha256:
        raise SystemExit(f"script SHA mismatch: current={script_sha} expected={expected_script_sha256}")
    if locked_sha and script_sha != locked_sha:
        raise SystemExit(f"script SHA mismatch against source lock: current={script_sha} locked={locked_sha}")
    return {
        "source_lock_path": str(Path(source_lock_path)),
        "source_lock_sha256": lock.get("source_lock_sha256", source_lock_hash(lock)),
        "script_sha256": script_sha,
        "script_path": str(script_path),
    }


def git_head_and_status(repo_root: str | Path) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    def run_git(args: Sequence[str]) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=str(repo_root), stderr=subprocess.STDOUT, text=True).strip()
        except Exception as exc:
            return f"UNAVAILABLE:{exc}"
    return {
        "git_head": run_git(["rev-parse", "HEAD"]),
        "git_status_short": run_git(["status", "--short"]),
    }


def provenance(
    *,
    script_path: str | Path,
    argv: Sequence[str],
    source_info: Mapping[str, Any],
    repo_root: str | Path,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    out = {
        "script_path": str(script_path),
        "script_sha256": source_info.get("script_sha256"),
        "source_lock_path": source_info.get("source_lock_path"),
        "source_lock_sha256": source_info.get("source_lock_sha256"),
        "argv": list(argv),
        "python": sys.version,
        "cwd": os.getcwd(),
        **git_head_and_status(repo_root),
    }
    if extra:
        out.update(dict(extra))
    return out


def reject_legacy_result_path(path: str | Path) -> None:
    normalized = str(path).replace("\\", "/")
    if "gate2_v2" in normalized or "gate2_v3" in normalized:
        raise ValueError(f"legacy result path rejected for v4: {path}")
    if RESULTS_ROOT_REQUIRED_FRAGMENT not in normalized:
        raise ValueError(f"v4 result path must include {RESULTS_ROOT_REQUIRED_FRAGMENT}: {path}")


def core_cell_id(model_tag: str, algo: str, seed: str | int, quantizer: str) -> str:
    return f"{model_tag}_{algo}_seed{seed}/{quantizer}"


def iter_core_cells(seeds: Sequence[str] = SEEDS) -> Iterable[Dict[str, str]]:
    for label, cfg in CORE_SETTINGS.items():
        for algo in ALGOS:
            for seed in seeds:
                for quantizer in QUANTIZERS:
                    yield {
                        "label": label,
                        "model_tag": cfg["model_tag"],
                        "dataset": cfg["dataset"],
                        "algo": algo,
                        "seed": str(seed),
                        "quantizer": quantizer,
                        "cell_id": core_cell_id(cfg["model_tag"], algo, str(seed), quantizer),
                    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Gate 2 v4 contract constants")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = {
        "manifest_schema": MANIFEST_SCHEMA,
        "gt_schema": GT_SCHEMA,
        "upqi_schema": UPQI_SCHEMA,
        "loss_contract_id": LOSS_CONTRACT_ID,
        "mask_contract_id": MASK_CONTRACT_ID,
        "truth_ratio_contract_id": TRUTH_RATIO_CONTRACT_ID,
        "core_cells": len(list(iter_core_cells())),
    }
    if args.json:
        print(canonical_json(payload))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
