#!/usr/bin/env python3
"""Audit Gate 2 v4 forget manifests for hard truth-ratio support."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gate2_v4_contract import (  # noqa: E402
    MANIFEST_SCHEMA,
    MANIFEST_SELECTION_CONTRACT_ID,
    assert_manifest_sha,
    load_json,
    perturbation_hash,
)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def audit_manifest(path: Path, expected_n_samples: int) -> Dict[str, Any]:
    errors: List[str] = []
    normalized_path = str(path).replace("\\", "/")
    if "gate2_v2" in normalized_path or "gate2_v3" in normalized_path:
        errors.append(f"legacy_manifest_path_rejected:{path}")
    try:
        obj = load_json(path)
    except Exception as exc:
        return {
            "path": str(path),
            "status": "FAIL",
            "dataset": "",
            "split": "",
            "purpose": "",
            "n_samples": 0,
            "n_errors": 1,
            "errors": f"unreadable_json:{exc}",
        }
    if obj.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(f"wrong_schema:{obj.get('schema_version')}")
    if obj.get("selection_contract_id") != MANIFEST_SELECTION_CONTRACT_ID:
        errors.append("wrong_or_missing_selection_contract_id")
    if obj.get("split") != "forget":
        errors.append(f"not_forget_split:{obj.get('split')}")
    try:
        assert_manifest_sha(obj)
    except Exception as exc:
        errors.append(f"manifest_sha_mismatch:{exc}")
    samples = obj.get("samples")
    if not isinstance(samples, list):
        samples = []
        errors.append("samples_not_list")
    if len(samples) != expected_n_samples:
        errors.append(f"sample_count_mismatch:{len(samples)}!={expected_n_samples}")
    seen_sample_hashes = set()
    for idx, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            errors.append(f"sample_{idx}_not_mapping")
            continue
        sample_hash = sample.get("sample_hash")
        if not sample_hash:
            errors.append(f"sample_{idx}_missing_sample_hash")
        elif sample_hash in seen_sample_hashes:
            errors.append(f"sample_{idx}_duplicate_sample_hash:{sample_hash}")
        else:
            seen_sample_hashes.add(sample_hash)
        question = sample.get("question", sample.get("prompt"))
        answer = sample.get("answer", sample.get("completion"))
        if not str(question or "").strip():
            errors.append(f"sample_{idx}_missing_question")
        if not str(answer or "").strip():
            errors.append(f"sample_{idx}_missing_answer")
        perturbations = sample.get("perturbations")
        if not isinstance(perturbations, list) or not perturbations:
            errors.append(f"sample_{idx}_missing_perturbations")
            continue
        declared_n = sample.get("n_perturbations", len(perturbations))
        if int(declared_n) != len(perturbations):
            errors.append(f"sample_{idx}_n_perturbations_mismatch:{declared_n}!={len(perturbations)}")
        normalized_answer = normalize_text(answer)
        seen_perturbation_hashes = set()
        for pidx, perturbation in enumerate(perturbations):
            if not isinstance(perturbation, Mapping):
                errors.append(f"sample_{idx}_perturbation_{pidx}_not_mapping")
                continue
            p_answer = perturbation.get("answer")
            if not str(p_answer or "").strip():
                errors.append(f"sample_{idx}_perturbation_{pidx}_missing_answer")
                continue
            if normalize_text(p_answer) == normalized_answer:
                errors.append(f"sample_{idx}_perturbation_{pidx}_identical_to_answer")
            p_hash = perturbation.get("perturbation_hash")
            if not p_hash:
                errors.append(f"sample_{idx}_perturbation_{pidx}_missing_hash")
            elif p_hash != perturbation_hash(str(p_answer)):
                errors.append(f"sample_{idx}_perturbation_{pidx}_hash_mismatch")
            elif p_hash in seen_perturbation_hashes:
                errors.append(f"sample_{idx}_perturbation_{pidx}_duplicate_hash")
            else:
                seen_perturbation_hashes.add(p_hash)
    return {
        "path": str(path),
        "status": "PASS" if not errors else "FAIL",
        "dataset": str(obj.get("dataset", "")),
        "split": str(obj.get("split", "")),
        "purpose": str(obj.get("purpose", "")),
        "n_samples": len(samples),
        "n_errors": len(errors),
        "errors": ";".join(errors[:80]),
    }


def discover_manifests(args: argparse.Namespace) -> List[Path]:
    if args.manifest:
        return [Path(p) for p in args.manifest]
    root = Path(args.manifest_dir)
    return sorted(p for p in root.rglob("*forget*.json") if p.is_file())


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "status", "dataset", "split", "purpose", "n_samples", "n_errors", "errors"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(str(row["status"]) for row in rows)
    lines = [
        "# Gate 2 v4 Manifest Truth-Ratio Audit",
        "",
        f"Manifests audited: {len(rows)}",
        f"Status counts: {dict(counts)}",
        "",
        "## Failures",
        "",
    ]
    failures = [row for row in rows if row["status"] != "PASS"]
    if not failures:
        lines.append("None.")
    else:
        for row in failures:
            lines.append(f"- `{row['path']}`: {row['errors']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v4 forget manifests for hard truth-ratio readiness")
    parser.add_argument("--manifest", nargs="*", help="Specific forget manifest paths to audit")
    parser.add_argument("--manifest_dir", default="manifests/gate2_v4")
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--output_tsv", default="manifest_truth_ratio_audit_v4.tsv")
    parser.add_argument("--output_md", default="manifest_truth_ratio_audit_v4.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = discover_manifests(args)
    if not paths:
        raise SystemExit("No v4 forget manifests found to audit")
    rows = [audit_manifest(path, args.n_samples) for path in paths]
    write_tsv(Path(args.output_tsv), rows)
    write_md(Path(args.output_md), rows)
    print(f"audited {len(rows)} manifests")
    print(f"status: {dict(Counter(str(row['status']) for row in rows))}")
    print(f"wrote: {args.output_tsv}")
    print(f"wrote: {args.output_md}")
    if any(row["status"] != "PASS" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
