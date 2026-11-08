#!/usr/bin/env python3
"""Gate 2 v4 tokenizer dataset and answer-mask sentries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import torch
from torch.utils.data import Dataset


def normalize_question(sample: Mapping[str, Any]) -> str:
    value = sample.get("question", sample.get("prompt", ""))
    if not str(value).strip():
        raise ValueError("sample missing question/prompt")
    return str(value).strip()


def normalize_answer(sample: Mapping[str, Any]) -> str:
    value = sample.get("answer", sample.get("completion", ""))
    if not str(value).strip():
        raise ValueError("sample missing answer/completion")
    return str(value).strip()


def qa_prefix(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def longest_common_prefix(lhs: Sequence[int], rhs: Sequence[int]) -> int:
    common = 0
    for a, b in zip(lhs, rhs):
        if a != b:
            break
        common += 1
    return common


def encode_qa_preserve_answer(tokenizer: Any, question: str, answer: str, max_length: int) -> Dict[str, Any]:
    """Encode QA text without ever truncating all answer tokens away.

    The v4 contract is answer-side preserving: if a prompt is too long, trim the
    prompt prefix and keep the answer tokens. This helper is shared by loss,
    truth-ratio, KL/reference-logit, and GPTQ calibration paths so those paths
    cannot drift into different truncation behavior.
    """
    prefix_text = qa_prefix(question)
    prefix = tokenizer(
        prefix_text,
        truncation=False,
        padding=False,
        return_tensors=None,
    )
    answer_encoded = tokenizer(
        " " + answer.strip(),
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_tensors=None,
    )
    prefix_ids = list(prefix["input_ids"])
    answer_ids = list(answer_encoded["input_ids"])
    if not answer_ids:
        raise ValueError("answer tokenization produced zero tokens")
    if max_length < 2:
        raise ValueError("max_length must allow at least one prefix token and one answer token")
    if len(answer_ids) >= max_length:
        answer_ids = answer_ids[: max_length - 1]
    prefix_budget = max_length - len(answer_ids)
    if prefix_budget <= 0:
        raise ValueError("answer token budget exhausted max_length")
    prefix_truncated = len(prefix_ids) > prefix_budget
    if len(prefix_ids) > prefix_budget:
        # Keep the end of the prompt because it contains the "Answer:" boundary.
        prefix_ids = prefix_ids[-prefix_budget:]
    input_ids = prefix_ids + answer_ids
    valid_len = len(input_ids)
    answer_start = len(prefix_ids)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.ones(valid_len, dtype=torch.long),
        "answer_start": torch.tensor(answer_start, dtype=torch.long),
        "prefix_truncated": prefix_truncated,
    }


def encode_answer_only(tokenizer: Any, question: str, answer: str, max_length: int) -> Dict[str, torch.Tensor]:
    """Encode QA text with labels masked through the complete prompt prefix.

    This intentionally differs from v3. The first unmasked label is the first
    answer token. In Hugging Face causal-LM loss, logits are shifted internally,
    so leaving label position `answer_start` unmasked means token
    `answer_start - 1` predicts the first answer token. We do not unmask
    `answer_start - 1` itself.

    Long MUSE prompts can otherwise consume the whole `max_length` window and
    truncate away every answer token. v4 preserves answer tokens by tokenizing
    the prompt and answer separately, then trimming the prompt side if needed.
    """
    encoded = encode_qa_preserve_answer(tokenizer, question, answer, max_length)
    input_ids = encoded["input_ids"].tolist()
    answer_start = int(encoded["answer_start"].item())
    labels = list(input_ids)
    for i in range(answer_start):
        labels[i] = -100
    if not any(x != -100 for x in labels):
        raise ValueError("answer-only mask produced zero answer tokens")
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
        "answer_start": encoded["answer_start"],
    }


def valid_label_tokens(labels: torch.Tensor) -> int:
    return int((labels != -100).sum().item())


@dataclass
class MaskSentry:
    question: str
    answer: str
    answer_start: int
    first_unmasked_id: int
    first_unmasked_text: str
    prefix_tail_text: str
    n_answer_tokens: int
    passed: bool


def mask_sentry(tokenizer: Any, sample: Mapping[str, Any], max_length: int = 512) -> MaskSentry:
    question = normalize_question(sample)
    answer = normalize_answer(sample)
    encoded = encode_answer_only(tokenizer, question, answer, max_length)
    ids = encoded["input_ids"].tolist()
    labels = encoded["labels"].tolist()
    answer_start = int(encoded["answer_start"].item())
    unmasked = [i for i, x in enumerate(labels) if x != -100]
    first = unmasked[0]
    prefix_tail = tokenizer.decode(ids[max(0, answer_start - 4):answer_start])
    first_text = tokenizer.decode([ids[first]])
    answer_probe = answer.strip()[: max(1, min(12, len(answer.strip())))]
    decoded_answer_start = tokenizer.decode(ids[first:min(len(ids), first + 8)])
    passed = first == answer_start and answer_probe.lower()[:4] in decoded_answer_start.lower()
    return MaskSentry(
        question=question,
        answer=answer,
        answer_start=answer_start,
        first_unmasked_id=int(ids[first]),
        first_unmasked_text=first_text,
        prefix_tail_text=prefix_tail,
        n_answer_tokens=len(unmasked),
        passed=passed,
    )


class AnswerOnlyQADatasetV4(Dataset):
    def __init__(self, samples: Sequence[Mapping[str, Any]], tokenizer: Any, max_length: int = 512):
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return encode_answer_only(
            self.tokenizer,
            normalize_question(sample),
            normalize_answer(sample),
            self.max_length,
        )


def collate_answer_only(batch: Sequence[Mapping[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    max_len = max(int(row["input_ids"].shape[0]) for row in batch)
    input_ids = []
    attention = []
    labels = []
    answer_start = []
    for row in batch:
        n = int(row["input_ids"].shape[0])
        pad = max_len - n
        input_ids.append(torch.cat([row["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)]))
        attention.append(torch.cat([row["attention_mask"], torch.zeros((pad,), dtype=torch.long)]))
        labels.append(torch.cat([row["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        answer_start.append(row["answer_start"])
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention),
        "labels": torch.stack(labels),
        "answer_start": torch.stack(answer_start),
    }
