#!/usr/bin/env python3
"""Create neutral Gate 2 v4 manifests with hard perturbation requirements."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gate2_v4_contract import (  # noqa: E402
    MANIFEST_SCHEMA,
    MANIFEST_SELECTION_CONTRACT_ID,
    canonical_json,
    load_json,
    manifest_fingerprint,
    perturbation_hash,
    sample_hash,
    save_json,
    sha256_text,
)


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def load_data_file(data_dir: Path, dataset: str, split: str) -> List[Mapping[str, Any]]:
    candidates = [
        data_dir / dataset / f"{split}.json",
        data_dir / dataset / f"{split}10.json",
        data_dir / f"{dataset}_{split}.json",
        data_dir / f"{split}.json",
        data_dir / f"{split}10.json",
    ]
    path = first_existing(candidates)
    if path is None:
        raise SystemExit(f"No data file found for dataset={dataset} split={split}; checked: {candidates}")
    data = load_json(path)
    if isinstance(data, Mapping):
        data = list(data.values())
    if not isinstance(data, list):
        raise SystemExit(f"Expected list-like data in {path}")
    return [row for row in data if isinstance(row, Mapping)]


def extract_perturbation_answers(row: Mapping[str, Any]) -> List[str]:
    keys = (
        "perturbations",
        "perturbed_answer",
        "perturbed_answers",
        "paraphrased_answer",
        "paraphrased_answers",
        "wrong_answers",
        "false_answers",
        "negative_answers",
    )
    answers: List[str] = []
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            answers.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    answers.append(item)
                elif isinstance(item, Mapping):
                    nested = item.get("answer", item.get("text", item.get("completion")))
                    if nested:
                        answers.append(str(nested))
        elif isinstance(value, Mapping):
            nested = value.get("answer", value.get("text", value.get("completion")))
            if nested:
                answers.append(str(nested))
    cleaned: List[str] = []
    seen = set()
    for answer in answers:
        answer = str(answer).strip()
        if not answer:
            continue
        key = answer.lower()
        if key not in seen:
            cleaned.append(answer)
            seen.add(key)
    return cleaned


def key_for(row: Mapping[str, Any]) -> str:
    for key in ("id", "sample_id", "idx"):
        if key in row:
            return f"id:{row[key]}"
    return f"sample_hash:{sample_hash(row)}"


def load_perturbation_map(path: Optional[Path]) -> Dict[str, List[str]]:
    if path is None:
        return {}
    data = load_json(path)
    if isinstance(data, Mapping):
        data = list(data.values())
    if not isinstance(data, list):
        raise SystemExit(f"Perturbation file must be list/dict: {path}")
    out: Dict[str, List[str]] = {}
    for row in data:
        if not isinstance(row, Mapping):
            continue
        answers = extract_perturbation_answers(row)
        if answers:
            out[key_for(row)] = answers
            out[f"sample_hash:{sample_hash(row)}"] = answers
            if row.get("question"):
                out[f"question:{str(row['question']).strip().lower()}"] = answers
    return out


def infer_perturbation_path(data_dir: Path, dataset: str, split: str) -> Optional[Path]:
    candidates = [
        data_dir / dataset / f"{split}_perturbed.json",
        data_dir / dataset / f"{split}10_perturbed.json",
        data_dir / f"{dataset}_{split}_perturbed.json",
        data_dir / f"{split}_perturbed.json",
        data_dir / f"{split}10_perturbed.json",
    ]
    return first_existing(candidates)


def perturbations_for(row: Mapping[str, Any], pmap: Mapping[str, List[str]]) -> List[str]:
    answers = extract_perturbation_answers(row)
    for key in (key_for(row), f"sample_hash:{sample_hash(row)}", f"question:{str(row.get('question', row.get('prompt', ''))).strip().lower()}"):
        for answer in pmap.get(key, []):
            if answer not in answers:
                answers.append(answer)
    return [x for x in answers if str(x).strip()]


def deterministic_select(rows: List[Mapping[str, Any]], n_samples: int, seed: int, dataset: str, split: str) -> List[Mapping[str, Any]]:
    scored = []
    for idx, row in enumerate(rows):
        key = sha256_text(canonical_json({
            "seed": seed,
            "dataset": dataset,
            "split": split,
            "idx": idx,
            "sample_hash": row["sample_hash"],
        }))
        scored.append((key, row))
    scored.sort(key=lambda x: x[0])
    return [row for _, row in scored[:n_samples]]


def build_manifest(args: argparse.Namespace) -> Mapping[str, Any]:
    data_dir = Path(args.data_dir)
    raw = load_data_file(data_dir, args.dataset, args.split)
    perturbation_path = Path(args.perturbation_path) if args.perturbation_path else infer_perturbation_path(data_dir, args.dataset, args.split)
    pmap = load_perturbation_map(perturbation_path)
    eligible: List[Mapping[str, Any]] = []
    for idx, row in enumerate(raw):
        question = str(row.get("question", row.get("prompt", ""))).strip()
        answer = str(row.get("answer", row.get("completion", ""))).strip()
        if not question or not answer:
            continue
        perturbations = perturbations_for(row, pmap)
        if not perturbations:
            continue
        item: Dict[str, Any] = {
            "source_index": idx,
            "id": row.get("id", row.get("sample_id", row.get("idx", idx))),
            "question": question,
            "answer": answer,
        }
        item["sample_hash"] = sample_hash(item)
        item["perturbations"] = [
            {"answer": text, "perturbation_hash": perturbation_hash(text)}
            for text in perturbations
        ]
        item["n_perturbations"] = len(item["perturbations"])
        eligible.append(item)
    if len(eligible) < args.n_samples:
        raise SystemExit(
            f"Only {len(eligible)} eligible samples with perturbations for {args.dataset}/{args.split}; need {args.n_samples}. "
            "Stop: do not invent perturbations."
        )
    samples = deterministic_select(eligible, args.n_samples, args.seed, args.dataset, args.split)
    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset": args.dataset,
        "split": args.split,
        "purpose": args.purpose,
        "selection": "deterministic_hash_without_replacement",
        "selection_contract_id": MANIFEST_SELECTION_CONTRACT_ID,
        "seed": args.seed,
        "n_total": len(raw),
        "n_eligible_with_perturbations": len(eligible),
        "n_samples": len(samples),
        "perturbation_source": str(perturbation_path) if perturbation_path else "inline_only",
        "sample_hash_contract": "sha256(id, question, answer)",
        "perturbation_hash_contract": "sha256(stripped perturbation answer)",
        "samples": samples,
    }
    manifest["manifest_sha256"] = manifest_fingerprint(manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Gate 2 v4 hard-TR manifest")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--dataset", required=True, choices=["tofu", "muse_news"])
    parser.add_argument("--split", default="forget")
    parser.add_argument("--purpose", default="core128")
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--perturbation_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    if out.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing manifest without --force: {out}")
    manifest = build_manifest(args)
    save_json(out, manifest)
    print(f"Wrote {out} with {manifest['n_samples']} samples sha={manifest['manifest_sha256']}")


if __name__ == "__main__":
    main()

