#!/usr/bin/env python3
"""Gate 2 v4 UPQI with shared answer-only mask and source-lock contract."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gate2_v4_contract import (  # noqa: E402
    UPQI_ABS_CONTRACT_ID,
    UPQI_SCHEMA,
    assert_forget_manifest_ready,
    finite_number,
    load_json,
    loss_contract,
    provenance,
    reject_legacy_result_path,
    require_source_lock,
    save_json,
)
from gate2_v4_dataset import AnswerOnlyQADatasetV4, collate_answer_only  # noqa: E402
from validate_upqi_v4 import validate_obj  # noqa: E402


def load_v2_backend() -> Any:
    for path in (
        SCRIPT_DIR / "gate2_v4_backend.py",
        SCRIPT_DIR / "run_gate2_ground_truth_v2.py",
        REPO_ROOT / "scripts" / "pipeline" / "run_gate2_ground_truth_v2.py",
    ):
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("gate2_v4_quant_backend", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ModuleNotFoundError("No Gate 2 quantization backend found in v4 or legacy locations")


gt_v1 = load_v2_backend()


LOGGER = logging.getLogger("gate2_upqi_v4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gate 2 UPQI v4 for one cell")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--dataset", required=True, choices=["tofu", "muse_news"])
    parser.add_argument("--quantizer", required=True, choices=["nf4", "gptq_int4", "rtn_int8"])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--calibration_manifest_path")
    parser.add_argument("--source_lock_path", required=True)
    parser.add_argument("--expected_script_sha256", required=True)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--n_calibration", type=int, default=128)
    parser.add_argument("--n_quadrature", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument("--max_layers", type=int, default=None)
    parser.add_argument("--stop_after_layers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def model_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def mean_loss_grad_for_layer(
    model: Any,
    tokenizer: Any,
    samples: Sequence[Mapping[str, Any]],
    layer_name: str,
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Tuple[torch.Tensor, float, int]:
    param = dict(model.named_parameters())[layer_name]
    dataset = AnswerOnlyQADatasetV4(samples, tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_answer_only(rows, pad_token_id=tokenizer.pad_token_id or 0),
    )
    model.eval()
    model.zero_grad(set_to_none=True)
    loss_sum_total = 0.0
    token_total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        mask = shift_labels != -100
        tokens = int(mask.sum().item())
        if tokens <= 0:
            raise RuntimeError("UPQI v4 batch has zero answer tokens")
        loss_sum = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100, reduction="sum")
        loss_sum.backward()
        loss_sum_total += float(loss_sum.detach().item())
        token_total += tokens
    if token_total <= 0:
        raise RuntimeError("UPQI v4 sample contract produced zero answer tokens")
    if param.grad is None:
        raise RuntimeError(f"No gradient for layer {layer_name}")
    grad = (param.grad.detach().double().cpu() / float(token_total)).clone()
    model.zero_grad(set_to_none=True)
    return grad, loss_sum_total / float(token_total), token_total


def displacement_for_layer(param: torch.nn.Parameter, quantizer: str, hessian: Optional[torch.Tensor]) -> torch.Tensor:
    original = param.data
    dequantized = gt_v1.quantize_dequantize(original, quantizer, hessian=hessian)
    return (dequantized.float() - original.float()).detach().cpu()


def compute_upqi_layers(
    model: Any,
    tokenizer: Any,
    samples: Sequence[Mapping[str, Any]],
    target_layers: Sequence[str],
    *,
    quantizer: str,
    hessians: Optional[Mapping[str, torch.Tensor]],
    resume: Sequence[Mapping[str, Any]],
    n_quadrature: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
    checkpoint_cb: Any,
    stop_after_layers: Optional[int],
) -> List[Dict[str, Any]]:
    param_dict = dict(model.named_parameters())
    completed = {str(row["layer"]): dict(row) for row in resume}
    out: List[Dict[str, Any]] = [completed[name] for name in target_layers if name in completed]
    computed = 0
    nodes = torch.linspace(0.0, 1.0, steps=n_quadrature, dtype=torch.float64).tolist()
    for layer_index, layer_name in enumerate(target_layers):
        if layer_name in completed:
            LOGGER.info("UPQI v4 %d/%d %s already checkpointed; skipping", layer_index + 1, len(target_layers), layer_name)
            continue
        param = param_dict[layer_name]
        original = param.data.clone()
        hessian = hessians[layer_name].to(device) if hessians and layer_name in hessians else None
        eps = displacement_for_layer(param, quantizer, hessian)
        signed_values: List[float] = []
        abs_values: List[float] = []
        grad_norms: List[float] = []
        losses: List[float] = []
        token_counts: List[int] = []
        try:
            for t in nodes:
                interpolated = original.float() + float(t) * eps.to(device)
                param.data.copy_(interpolated.to(param.data.dtype))
                grad, loss, tokens = mean_loss_grad_for_layer(
                    model,
                    tokenizer,
                    samples,
                    layer_name,
                    device=device,
                    batch_size=batch_size,
                    max_length=max_length,
                )
                if not torch.isfinite(grad).all():
                    raise ValueError(f"Non-finite gradient for {layer_name}")
                dot = torch.dot(grad.reshape(-1), eps.double().reshape(-1)).item()
                abs_val = torch.sum(torch.abs(grad.reshape(-1)) * torch.abs(eps.double().reshape(-1))).item()
                signed_values.append(float(dot))
                abs_values.append(float(abs_val))
                grad_norms.append(float(torch.linalg.vector_norm(grad).item()))
                losses.append(float(loss))
                token_counts.append(int(tokens))
        finally:
            param.data.copy_(original)
        entry = {
            "layer": layer_name,
            "layer_index": layer_index,
            "upqi_signed": float(sum(signed_values) / len(signed_values)),
            "upqi_abs": float(sum(abs_values) / len(abs_values)),
            "epsilon_l2": float(torch.linalg.vector_norm(eps.double()).item()),
            "epsilon_max": float(torch.max(torch.abs(eps.double())).item()),
            "mean_grad_l2": float(sum(grad_norms) / len(grad_norms)),
            "mean_loss": float(sum(losses) / len(losses)),
            "n_answer_tokens": int(min(token_counts)),
            "n_quadrature": int(n_quadrature),
            "block_idx": gt_v1.extract_block_idx(layer_name),
            "submodule": gt_v1.classify_submodule(layer_name),
        }
        if entry["upqi_abs"] < abs(entry["upqi_signed"]) - 1e-6:
            raise ValueError(f"UPQI abs contract failed for {layer_name}")
        out.append(entry)
        computed += 1
        LOGGER.info("UPQI v4 %d/%d %s signed=%g abs=%g eps_l2=%g", layer_index + 1, len(target_layers), layer_name, entry["upqi_signed"], entry["upqi_abs"], entry["epsilon_l2"])
        checkpoint_cb(out)
        if stop_after_layers and computed >= stop_after_layers:
            raise SystemExit(f"Stopped after {computed} layers for resume proof")
    by_layer = {row["layer"]: row for row in out}
    return [by_layer[name] for name in target_layers if name in by_layer]


def build_result(
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    calibration_manifest: Optional[Mapping[str, Any]],
    source_info: Mapping[str, Any],
    per_layer: Sequence[Mapping[str, Any]],
    n_layers: int,
    total_seconds: float,
) -> Dict[str, Any]:
    obj: Dict[str, Any] = {
        "schema_version": UPQI_SCHEMA,
        "stage": "upqi_v4",
        "result_root_policy": "v4_only_no_v2_v3_reuse",
        "method": "layerwise_one_at_a_time_integrated_gradients_along_quantization_displacement",
        "upqi_signed_contract_id": "path_integral_signed_grad_dot_epsilon_v4_0",
        "upqi_abs_contract_id": UPQI_ABS_CONTRACT_ID,
        "model_dir": args.model_dir,
        "model_tag": args.model_tag,
        "algo": args.algo,
        "seed": str(args.seed),
        "dataset": args.dataset,
        "quantizer": args.quantizer,
        "dtype": args.dtype,
        "n_samples": int(args.n_samples),
        "n_calibration": int(args.n_calibration) if args.quantizer == "gptq_int4" else None,
        "n_quadrature": int(args.n_quadrature),
        "n_layers": int(n_layers),
        "n_layers_completed": len(per_layer),
        "loss_contract": loss_contract(args.n_samples, manifest),
        "sample_manifest": {
            "path": str(Path(args.manifest_path)),
            "schema_version": manifest.get("schema_version"),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "dataset": manifest.get("dataset"),
            "split": manifest.get("split"),
            "purpose": manifest.get("purpose"),
            "selection_contract_id": manifest.get("selection_contract_id"),
        },
        "calibration_manifest": None if calibration_manifest is None else {
            "path": str(Path(args.calibration_manifest_path)),
            "schema_version": calibration_manifest.get("schema_version"),
            "manifest_sha256": calibration_manifest.get("manifest_sha256"),
            "dataset": calibration_manifest.get("dataset"),
            "split": calibration_manifest.get("split"),
            "purpose": calibration_manifest.get("purpose"),
            "selection_contract_id": calibration_manifest.get("selection_contract_id"),
        },
        "per_layer": [dict(row) for row in per_layer],
        "timing": {"total_seconds": float(total_seconds)},
        "provenance": provenance(
            script_path=__file__,
            argv=sys.argv,
            source_info=source_info,
            repo_root=REPO_ROOT,
            extra={"upqi_schema": UPQI_SCHEMA},
        ),
    }
    obj["validation"] = validate_obj(obj, Path(args.output_dir) / "upqi_v4.json")
    return obj


def existing_complete_ok(path: Path, args: argparse.Namespace, manifest: Mapping[str, Any], source_info: Mapping[str, Any]) -> bool:
    if not path.exists() or args.force:
        return False
    try:
        obj = load_json(path)
        report = validate_obj(obj, path)
    except Exception:
        return False
    return (
        report["passed"]
        and obj.get("sample_manifest", {}).get("manifest_sha256") == manifest.get("manifest_sha256")
        and obj.get("provenance", {}).get("source_lock_sha256") == source_info.get("source_lock_sha256")
        and obj.get("provenance", {}).get("script_sha256") == source_info.get("script_sha256")
        and obj.get("model_tag") == args.model_tag
        and obj.get("algo") == args.algo
        and str(obj.get("seed")) == str(args.seed)
        and obj.get("quantizer") == args.quantizer
    )


def load_calibration_data(args: argparse.Namespace, calibration_manifest: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    try:
        retain_data = gt_v1.load_json_data(args.data_dir, args.dataset, "retain")
        calibration_data = gt_v1.select_manifest_samples(calibration_manifest, retain_data)
        if len(calibration_data) == args.n_calibration:
            return calibration_data
        LOGGER.warning("Raw retain selection returned %d samples, expected %d", len(calibration_data), args.n_calibration)
    except (Exception, SystemExit) as exc:
        LOGGER.warning("Falling back to calibration manifest samples because raw retain load failed: %s", exc)
    samples = calibration_manifest.get("samples")
    if not isinstance(samples, list):
        raise SystemExit("Calibration manifest has no samples")
    calibration_data = [row for row in samples if isinstance(row, Mapping)]
    if len(calibration_data) != args.n_calibration:
        raise SystemExit(f"Calibration manifest contains {len(calibration_data)} usable samples, expected {args.n_calibration}")
    return calibration_data


def partial_rows_ok(path: Path, args: argparse.Namespace, manifest: Mapping[str, Any], source_info: Mapping[str, Any], target_layers: Sequence[str]) -> List[Dict[str, Any]]:
    if not path.exists() or args.force:
        return []
    try:
        obj = load_json(path)
    except Exception:
        return []
    if obj.get("schema_version") != UPQI_SCHEMA or obj.get("stage") != "upqi_v4":
        return []
    if obj.get("sample_manifest", {}).get("manifest_sha256") != manifest.get("manifest_sha256"):
        return []
    if obj.get("provenance", {}).get("source_lock_sha256") != source_info.get("source_lock_sha256"):
        return []
    if obj.get("provenance", {}).get("script_sha256") != source_info.get("script_sha256"):
        return []
    target_index = {name: idx for idx, name in enumerate(target_layers)}
    out: Dict[str, Dict[str, Any]] = {}
    for row in obj.get("per_layer", []) if isinstance(obj.get("per_layer"), list) else []:
        if not isinstance(row, Mapping):
            return []
        layer = str(row.get("layer"))
        if layer not in target_index or int(row.get("layer_index", -1)) != target_index[layer]:
            return []
        if not finite_number(row.get("upqi_signed")) or not finite_number(row.get("upqi_abs")):
            return []
        out[layer] = dict(row)
    return [out[name] for name in target_layers if name in out]


def main() -> None:
    setup_logging()
    t0 = time.time()
    args = parse_args()
    out_path = Path(args.output_dir) / "upqi_v4.json"
    partial_path = out_path.with_suffix(".json.partial")
    reject_legacy_result_path(out_path)
    source_info = require_source_lock(script_path=__file__, source_lock_path=args.source_lock_path, expected_script_sha256=args.expected_script_sha256)
    manifest = load_json(args.manifest_path)
    assert_forget_manifest_ready(manifest, n_samples=args.n_samples)
    calibration_manifest = None
    if args.quantizer == "gptq_int4":
        if not args.calibration_manifest_path:
            raise SystemExit("gptq_int4 requires --calibration_manifest_path")
        calibration_manifest = load_json(args.calibration_manifest_path)
    if existing_complete_ok(out_path, args, manifest, source_info):
        LOGGER.info("Existing UPQI v4 satisfies locked request: %s", out_path)
        return
    if args.dry_run:
        LOGGER.info("dry-run ok before model load: %s", out_path)
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=model_dtype(args.dtype), device_map="auto", trust_remote_code=True)
    device = next(model.parameters()).device
    target_layers = [name for name, _ in gt_v1.get_weight_layers(model)]
    if args.max_layers:
        target_layers = target_layers[: args.max_layers]
    resume = partial_rows_ok(partial_path, args, manifest, source_info, target_layers)
    if resume:
        LOGGER.info("Resuming UPQI v4 partial with %d completed layers", len(resume))
    remaining_layers = [name for name in target_layers if name not in {row["layer"] for row in resume}]
    LOGGER.info("Remaining UPQI v4 layers needing compute: %d/%d", len(remaining_layers), len(target_layers))
    if not remaining_layers:
        obj = build_result(args=args, manifest=manifest, calibration_manifest=calibration_manifest, source_info=source_info, per_layer=resume, n_layers=len(target_layers), total_seconds=time.time() - t0)
        if not obj["validation"]["passed"]:
            raise SystemExit(f"UPQI v4 validation failed: {obj['validation']}")
        save_json(out_path, obj)
        partial_path.unlink(missing_ok=True)
        LOGGER.info("UPQI v4 done from complete partial: %s", out_path)
        return
    hessians = None
    if args.quantizer == "gptq_int4":
        calibration_data = load_calibration_data(args, calibration_manifest)
        LOGGER.info("Collecting GPTQ Hessians for %d remaining layers from locked retain manifest", len(remaining_layers))
        hessians = gt_v1.collect_layer_hessians(model, tokenizer, calibration_data, remaining_layers, device=device, max_samples=args.n_calibration)

    def checkpoint(per_layer: Sequence[Mapping[str, Any]]) -> None:
        obj = build_result(args=args, manifest=manifest, calibration_manifest=calibration_manifest, source_info=source_info, per_layer=per_layer, n_layers=len(target_layers), total_seconds=time.time() - t0)
        obj["validation"] = {"passed": False, "errors": ["partial_incomplete"], "warnings": []}
        save_json(partial_path, obj)
        LOGGER.info("checkpoint saved: %d/%d layers", len(per_layer), len(target_layers))

    per_layer = compute_upqi_layers(
        model,
        tokenizer,
        manifest["samples"],
        target_layers,
        quantizer=args.quantizer,
        hessians=hessians,
        resume=resume,
        n_quadrature=args.n_quadrature,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        checkpoint_cb=checkpoint,
        stop_after_layers=args.stop_after_layers,
    )
    obj = build_result(args=args, manifest=manifest, calibration_manifest=calibration_manifest, source_info=source_info, per_layer=per_layer, n_layers=len(target_layers), total_seconds=time.time() - t0)
    if not obj["validation"]["passed"]:
        save_json(out_path.with_suffix(".json.failed"), obj)
        raise SystemExit(f"UPQI v4 validation failed: {obj['validation']}")
    save_json(out_path, obj)
    partial_path.unlink(missing_ok=True)
    LOGGER.info("UPQI v4 done: %s", out_path)


if __name__ == "__main__":
    main()
