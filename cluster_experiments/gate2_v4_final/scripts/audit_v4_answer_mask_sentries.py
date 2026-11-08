#!/usr/bin/env python3
"""Print/fail token-level answer-mask sentries before v4 production."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gate2_v4_contract import load_json  # noqa: E402
from gate2_v4_dataset import mask_sentry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v4 answer-only mask sentries")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--n", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    manifest = load_json(args.manifest_path)
    samples = manifest.get("samples", [])[: args.n]
    failures = 0
    for idx, sample in enumerate(samples):
        sentry = mask_sentry(tokenizer, sample, max_length=args.max_length)
        print(f"==== sample {idx} ====")
        print(f"passed={sentry.passed}")
        print(f"answer_start={sentry.answer_start} n_answer_tokens={sentry.n_answer_tokens}")
        print(f"prefix_tail={sentry.prefix_tail_text!r}")
        print(f"first_unmasked={sentry.first_unmasked_text!r} id={sentry.first_unmasked_id}")
        print(f"answer={sentry.answer[:160]!r}")
        if not sentry.passed:
            failures += 1
    if failures:
        raise SystemExit(f"Mask sentry failed for {failures}/{len(samples)} samples")
    print(f"Mask sentry passed for {len(samples)} samples")


if __name__ == "__main__":
    main()

