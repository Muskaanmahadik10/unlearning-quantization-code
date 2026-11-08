#!/usr/bin/env python3
"""Fast shared-input GPTQ collector with legacy-exact CPU finalization."""

from __future__ import annotations

import gc
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
V4_DIR = REPO_ROOT / "gate2_v4_final" / "scripts"
if str(V4_DIR) not in sys.path:
    sys.path.insert(0, str(V4_DIR))

import torch  # noqa: E402

from gate2_v4_backend import get_module_by_name  # noqa: E402
from gate2_v4_dataset import encode_qa_preserve_answer  # noqa: E402
from test_org_gptq_collector_parity_v1 import canonical_input_layer  # noqa: E402

LOGGER = logging.getLogger("org_gptq_shared_cpu_finalize_collector_v3")


def collect_shared_hessians_gpu_cpu_finalize(
    model: Any,
    tokenizer: Any,
    calibration_data: Sequence[Mapping[str, Any]],
    target_layer_names: Sequence[str],
    *,
    device: torch.device,
    max_samples: int,
) -> tuple[dict[str, torch.Tensor], dict[str, str], dict[str, int]]:
    """Accumulate unique input Hessians on GPU and normalize on CPU.

    The CPU finalization is mandatory: boundary test v4 demonstrated exact
    identity with the frozen legacy collector for all 224 Hessians and all 21
    representative downstream GPTQ tensors.
    """
    layer_to_group = {name: canonical_input_layer(name) for name in target_layer_names}
    groups = list(OrderedDict.fromkeys(layer_to_group.values()))
    module_map = {
        name: get_module_by_name(model, name.replace(".weight", "")) for name in groups
    }
    accumulators = {
        name: torch.zeros(module.in_features, module.in_features, dtype=torch.float32, device=device)
        for name, module in module_map.items()
    }
    n_tokens = {name: 0 for name in groups}
    hooks = []
    for name, module in module_map.items():
        def make_hook(group_name: str):
            def hook_fn(_module: Any, args: tuple[Any, ...]) -> None:
                x = args[0].detach().to(device=device, dtype=torch.float32)
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                # Keep GEMM and accumulation separate; parity evidence covers
                # this exact operation sequence.
                gram = x.T @ x
                accumulators[group_name].add_(gram)
                n_tokens[group_name] += int(x.shape[0])
            return hook_fn
        hooks.append(module.register_forward_pre_hook(make_hook(name)))
    try:
        model.eval()
        total = min(max_samples, len(calibration_data))
        for index, sample in enumerate(list(calibration_data)[:max_samples], start=1):
            question = sample.get("question", sample.get("prompt", ""))
            answer = sample.get("answer", sample.get("response", sample.get("completion", "")))
            encoded = encode_qa_preserve_answer(tokenizer, str(question), str(answer), max_length=512)
            inputs = {
                "input_ids": encoded["input_ids"].unsqueeze(0).to(device),
                "attention_mask": encoded["attention_mask"].unsqueeze(0).to(device),
            }
            with torch.no_grad():
                model(**inputs)
            LOGGER.info("shared GPU calibration %d/%d", index, total)
    finally:
        for hook in hooks:
            hook.remove()

    cpu_groups: dict[str, torch.Tensor] = {}
    for name in groups:
        if n_tokens[name] <= 0:
            raise SystemExit(f"No tokens collected for {name}")
        # Copy before division. GPU-side division changes low bits and can
        # produce different discrete GPTQ codes.
        cpu_groups[name] = accumulators[name].cpu()
        cpu_groups[name].div_(n_tokens[name])
    del accumulators
    gc.collect()
    torch.cuda.empty_cache()
    return (
        {name: cpu_groups[layer_to_group[name]] for name in target_layer_names},
        layer_to_group,
        n_tokens,
    )

