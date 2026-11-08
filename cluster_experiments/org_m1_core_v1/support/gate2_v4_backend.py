#!/usr/bin/env python3
"""Gate 2 v4 quantization backend.

This file vendors only the quantization and GPTQ-calibration helpers needed by
the v4 GT/UPQI runners. It deliberately avoids importing the old v2 pipeline so
cluster jobs do not depend on a missing ``scripts/pipeline`` package.
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

from gate2_v4_dataset import encode_qa_preserve_answer


LOGGER = logging.getLogger("gate2_v4_backend")

NF4_BLOCK_SIZE = 64
GPTQ_GROUP_SIZE = 128
GPTQ_PERCDAMP = 0.01
GPTQ_BLOCKSIZE = 128


def get_weight_layers(model: Any) -> List[tuple[str, torch.nn.Parameter]]:
    layers: List[tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        if param.ndim == 2 and ("layers." in name or "h." in name) and "weight" in name:
            lower = name.lower()
            if "norm" in lower or "embed" in lower or "lm_head" in name:
                continue
            layers.append((name, param))
    return layers


def extract_block_idx(name: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def classify_submodule(name: str) -> str:
    lower = name.lower()
    for key in ("q_proj", "k_proj", "v_proj", "dense", "o_proj", "wqkv", "out_proj", "self_attn", "attention"):
        if key in lower:
            return "attention"
    for key in ("fc1", "fc2", "gate_proj", "up_proj", "down_proj", "mlp"):
        if key in lower:
            return "mlp"
    return "unknown"


def get_module_by_name(model: Any, dotted_name: str) -> Any:
    module = model
    for part in dotted_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def nf4_quantize_dequantize(weight: torch.Tensor, block_size: int = NF4_BLOCK_SIZE) -> torch.Tensor:
    import bitsandbytes.functional as F_bnb

    device = weight.device
    w = weight.detach().to(device).contiguous().to(torch.float32)
    quantized, quant_state = F_bnb.quantize_4bit(w, quant_type="nf4", blocksize=block_size)
    dequantized = F_bnb.dequantize_4bit(quantized, quant_state, quant_type="nf4")
    return dequantized.to(weight.dtype)


def rtn_int8_quantize_dequantize(weight: torch.Tensor) -> torch.Tensor:
    w = weight.float()
    scale = w.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = scale.clamp(min=1e-10)
    quantized = (w / scale).round().clamp(-128, 127)
    return (quantized * scale).to(weight.dtype)


def _int4_quantize_column(w_col: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    q = (w_col / scale + zero).round().clamp(0, 15)
    return (q - zero) * scale


def gptq_int4_quantize_dequantize(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    group_size: int = GPTQ_GROUP_SIZE,
    percdamp: float = GPTQ_PERCDAMP,
    blocksize: int = GPTQ_BLOCKSIZE,
) -> torch.Tensor:
    orig_dtype = weight.dtype
    w_work = weight.float().clone()
    _, d_in = w_work.shape

    h_work = hessian.float().clone().to(w_work.device)
    dead = h_work.diagonal() == 0
    h_work[dead, dead] = 1.0
    damp = percdamp * h_work.diagonal().mean()
    h_work.diagonal().add_(damp)

    try:
        h_inv = torch.cholesky_inverse(torch.linalg.cholesky(h_work))
    except RuntimeError:
        h_work.diagonal().add_(0.1 * h_work.diagonal().mean())
        h_inv = torch.cholesky_inverse(torch.linalg.cholesky(h_work))
    h_inv_cho = torch.linalg.cholesky(h_inv, upper=True)

    q_weight = torch.zeros_like(w_work)
    for i1 in range(0, d_in, blocksize):
        i2 = min(i1 + blocksize, d_in)
        count = i2 - i1
        w_block = w_work[:, i1:i2].clone()
        q_block = torch.zeros_like(w_block)
        err_block = torch.zeros_like(w_block)
        h_inv_block = h_inv_cho[i1:i2, i1:i2]

        for j in range(count):
            col_global = i1 + j
            col = w_block[:, j]
            denom = h_inv_block[j, j]
            g_start = (col_global // group_size) * group_size - i1
            g_end = g_start + group_size
            g_s = max(0, g_start)
            g_e = min(count, g_end)
            if g_e <= g_s:
                g_s, g_e = max(0, j - group_size // 2), min(count, j + group_size // 2)
            group = w_block[:, g_s:g_e]
            w_min = group.min(dim=1).values
            w_max = group.max(dim=1).values
            scale = ((w_max - w_min) / 15.0).clamp(min=1e-10)
            zero = (-w_min / scale).round().clamp(0, 15)
            q_deq = _int4_quantize_column(col, scale, zero)
            q_block[:, j] = q_deq
            err = (col - q_deq) / denom
            err_block[:, j] = err
            if j < count - 1:
                w_block[:, j + 1:] -= err.unsqueeze(1) * h_inv_block[j, j + 1:].unsqueeze(0)

        q_weight[:, i1:i2] = q_block
        if i2 < d_in:
            w_work[:, i2:] -= err_block @ h_inv_cho[i1:i2, i2:]

    return q_weight.to(orig_dtype)


def quantize_dequantize(weight: torch.Tensor, quantizer: str, hessian: torch.Tensor | None = None) -> torch.Tensor:
    if quantizer == "nf4":
        return nf4_quantize_dequantize(weight)
    if quantizer == "gptq_int4":
        if hessian is None:
            raise ValueError("GPTQ requires a Hessian")
        return gptq_int4_quantize_dequantize(weight, hessian)
    if quantizer == "rtn_int8":
        return rtn_int8_quantize_dequantize(weight)
    raise ValueError(f"Unknown quantizer: {quantizer}")


def collect_layer_hessians(
    model: Any,
    tokenizer: Any,
    calibration_data: Sequence[Mapping[str, Any]],
    target_layer_names: Sequence[str],
    device: Any = "cuda",
    max_samples: int = 128,
) -> Dict[str, torch.Tensor]:
    LOGGER.info(
        "Collecting GPTQ Hessians from %d calibration samples for %d layers...",
        min(max_samples, len(calibration_data)),
        len(target_layer_names),
    )
    module_map: OrderedDict[str, Any] = OrderedDict()
    for name in target_layer_names:
        module_map[name] = get_module_by_name(model, name.replace(".weight", ""))

    hessians: Dict[str, torch.Tensor] = {}
    n_tokens: Dict[str, int] = {}
    for name, module in module_map.items():
        hessians[name] = torch.zeros(module.in_features, module.in_features, dtype=torch.float32)
        n_tokens[name] = 0

    hooks = []
    for name, module in module_map.items():
        def make_hook(layer_name: str):
            def hook_fn(_module: Any, args: tuple[Any, ...]) -> None:
                x = args[0].detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                hessians[layer_name] += (x.T @ x).cpu()
                n_tokens[layer_name] += x.shape[0]
            return hook_fn
        hooks.append(module.register_forward_pre_hook(make_hook(name)))

    try:
        model.eval()
        for idx, sample in enumerate(list(calibration_data)[:max_samples]):
            question = sample.get("question", sample.get("prompt", ""))
            answer = sample.get("answer", sample.get("response", sample.get("completion", "")))
            encoded = encode_qa_preserve_answer(tokenizer, str(question), str(answer), max_length=512)
            inputs = {
                "input_ids": encoded["input_ids"].unsqueeze(0).to(device),
                "attention_mask": encoded["attention_mask"].unsqueeze(0).to(device),
            }
            with torch.no_grad():
                model(**inputs)
            if (idx + 1) % 32 == 0:
                LOGGER.info("  Calibration: %d/%d samples", idx + 1, min(max_samples, len(calibration_data)))
    finally:
        for hook in hooks:
            hook.remove()

    for name in target_layer_names:
        if n_tokens[name] > 0:
            hessians[name] /= n_tokens[name]
        LOGGER.info("  %s: H shape %s, %d tokens", name, tuple(hessians[name].shape), n_tokens[name])
    return hessians


def load_json_data(data_dir: str, dataset: str, split: str) -> List[Mapping[str, Any]]:
    candidates = [
        Path(data_dir) / dataset / f"{split}.json",
        Path(data_dir) / dataset / f"{split}10.json",
        Path(data_dir) / dataset / f"{split}90.json",
        Path(data_dir) / f"{dataset}_{split}.json",
        Path(data_dir) / f"{dataset}_{split}10.json",
        Path(data_dir) / f"{dataset}_{split}90.json",
        Path(data_dir) / f"{split}.json",
        Path(data_dir) / f"{split}10.json",
        Path(data_dir) / f"{split}90.json",
        Path(data_dir) / f"{dataset}_{split}_gptq_calib128_seed20260625.json",
    ]
    for path in candidates:
        if path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, Mapping) and "samples" in obj:
                obj = obj["samples"]
            elif isinstance(obj, Mapping):
                obj = list(obj.values())
            if not isinstance(obj, list):
                raise ValueError(f"Expected list-like JSON in {path}")
            return [row for row in obj if isinstance(row, Mapping)]
    raise FileNotFoundError(f"No data file found for dataset={dataset} split={split}; checked: {candidates}")


def select_manifest_samples(manifest: Mapping[str, Any], data: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        return []
    by_source = {int(row.get("source_index")): row for row in data if isinstance(row, Mapping) and row.get("source_index") is not None}
    by_id = {str(row.get("id", row.get("sample_id", row.get("idx")))): row for row in data if isinstance(row, Mapping)}
    out: List[Mapping[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        selected = None
        if sample.get("source_index") is not None:
            selected = by_source.get(int(sample["source_index"]))
        if selected is None and sample.get("id") is not None:
            selected = by_id.get(str(sample["id"]))
        out.append(selected if selected is not None else sample)
    return out
