#!/usr/bin/env python3
"""Outcome-blind dual-QDP feature pass for one ORG E2B B0.5 cell."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import uuid


SCHEMA = "org_e2b_b05_feature_cell_v1"
TARGET_COUNT = 224
REAL_QDP_ATOL = 1.0e-10
REFERENCE_ATOL = 1.0e-10


class GateError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise GateError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def qdp_independent(left: Any, right: Any) -> tuple[float, float]:
    import torch

    require(left.shape == right.shape and left.numel() > 0, "QDP tensor shape")
    left64 = left.detach().cpu().to(torch.float64).reshape(-1)
    right64 = right.detach().cpu().to(torch.float64).reshape(-1)
    direct = float(torch.dot(left64, right64).item())
    product = left64 * right64
    blocks = [float(chunk.sum(dtype=torch.float64).item()) for chunk in product.split(1 << 18)]
    independent = float(math.fsum(blocks))
    require(math.isfinite(direct) and math.isfinite(independent), "nonfinite QDP")
    return direct, independent


def select_sets(rows: Sequence[Mapping[str, Any]], total_parameters: int) -> dict[str, Any]:
    require(len(rows) == TARGET_COUNT, "selector row count")
    safe = [row for row in rows if row["protection_forget_benefit"] > 0.0 and row["protection_retain_cost"] <= 0.0]
    beneficial = [row for row in rows if row["protection_forget_benefit"] > 0.0]
    safe.sort(key=lambda row: (-row["protection_forget_benefit"], row["protection_retain_cost"], row["layer"]))
    beneficial.sort(key=lambda row: (-row["protection_forget_benefit"], row["protection_retain_cost"], row["layer"]))
    risky = sorted(rows, key=lambda row: (-row["protection_retain_cost"], -row["protection_forget_benefit"], row["layer"]))
    safe_names = [] if not safe else [str(safe[0]["layer"])]
    risky_names = [str(risky[0]["layer"])]
    pair_pool = safe if len(safe) >= 2 else beneficial
    pair_names = [str(row["layer"]) for row in pair_pool[:2]]
    require(len(set(risky_names + pair_names)) >= 1, "empty selector")
    by_name = {str(row["layer"]): row for row in rows}
    for label, names in (("safe", safe_names), ("risky", risky_names), ("pair", pair_names)):
        require(len(names) <= 2 and len(names) == len(set(names)), f"{label} cardinality")
        mass = sum(int(by_name[name]["n_parameters"]) for name in names)
        require(mass / total_parameters <= 0.02 + 1e-15, f"{label} parameter budget")
    return {
        "bf16": {"protected_layers": "ALL_TARGETS", "alias_of": None},
        "full_packed": {"protected_layers": [], "alias_of": None},
        "safe_looking_singleton": {
            "protected_layers": safe_names,
            "alias_of": "full_packed" if not safe_names else None,
            "abstained": not bool(safe_names),
        },
        "risky_looking_singleton": {"protected_layers": risky_names, "alias_of": None},
        "pair": {"protected_layers": pair_names, "alias_of": None},
        "abstention": {"protected_layers": [], "alias_of": "full_packed"},
    }


def parser() -> argparse.ArgumentParser:
    item = argparse.ArgumentParser(description=__doc__)
    item.add_argument("--cell-id", required=True)
    item.add_argument("--checkpoint-dir", type=Path, required=True)
    item.add_argument("--checkpoint-relative", required=True)
    item.add_argument("--checkpoint-ledger", type=Path, required=True)
    item.add_argument("--checkpoint-certification", type=Path, required=True)
    item.add_argument("--b04-final", type=Path, required=True)
    item.add_argument("--nf4-b04-final", type=Path, required=True)
    item.add_argument("--environment-lock", type=Path, required=True)
    item.add_argument("--forget-manifest", type=Path, required=True)
    item.add_argument("--retain-manifest", type=Path, required=True)
    item.add_argument("--calibration-manifest", type=Path, required=True)
    item.add_argument("--metric-reference", type=Path, required=True)
    item.add_argument("--legacy-tensor-oracle", type=Path, required=True)
    item.add_argument("--quantizer", choices=("nf4", "gptq_int4", "rtn_int8"), required=True)
    item.add_argument("--mode", choices=("smoke", "production"), required=True)
    item.add_argument("--n-samples", type=int, required=True)
    item.add_argument("--n-retain", type=int, required=True)
    item.add_argument("--n-calibration", type=int, required=True)
    item.add_argument("--max-length", type=int, default=512)
    item.add_argument("--expected-script-sha256", required=True)
    item.add_argument("--output", type=Path, required=True)
    return item


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    started = datetime.now(timezone.utc)
    require(sha256_file(Path(__file__)) == args.expected_script_sha256, "script SHA drift")
    require(args.checkpoint_dir.is_dir(), "checkpoint directory missing")
    for path in (args.checkpoint_ledger, args.checkpoint_certification, args.b04_final, args.nf4_b04_final, args.environment_lock,
                 args.forget_manifest, args.retain_manifest, args.calibration_manifest, args.metric_reference,
                 args.legacy_tensor_oracle):
        require(path.is_file(), f"missing prerequisite: {path}")
    require(args.checkpoint_relative in args.checkpoint_ledger.read_text(encoding="utf-8"), "checkpoint absent from certified ledger")
    certification = read_json(args.checkpoint_certification)
    require(str(certification.get("status", "")).upper() == "PASS", "checkpoint certification failed")
    b04 = read_json(args.b04_final)
    require(str(b04.get("status", "")).upper() == "PASS", "B0.4 dual certification failed")
    require(b04.get("b05_authorized") is True or b04.get("B05_AUTHORIZED") is True or b04.get("checks", {}).get("b05_authorized") is True,
            "B0.4 did not authorize B0.5")
    nf4_b04 = read_json(args.nf4_b04_final)
    require(str(nf4_b04.get("status", "")).upper() == "PASS" and nf4_b04.get("b0_5_authorized") is True,
            "NF4 B0.4 post-job certification did not authorize B0.5")
    production = args.mode == "production"
    require((args.n_samples, args.n_retain, args.n_calibration) == ((128, 128, 128) if production else (2, 2, 2)), "sample-count contract")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gate2_v4_final" / "scripts"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "org_gb200_execution_v1" / "scripts"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import gate2_v4_backend as backend
    import run_gate2_metric_suite_v42 as metric
    from run_org_bridge_bundle_v2 import collect_shared_hessians_gpu_cpu_finalize

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one CUDA GPU required")
    require(torch.cuda.get_device_name(0) == "NVIDIA H200", "H200 required")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager",
    )
    model.eval()
    targets = [name for name, _ in backend.get_weight_layers(model)]
    require(len(targets) == TARGET_COUNT and len(targets) == len(set(targets)), "target universe")
    parameters = dict(model.named_parameters())

    forget_obj, forget_samples_all = metric.manifest_samples(str(args.forget_manifest), 128, forget_contract=True)
    retain_obj, retain_samples_all = metric.manifest_samples(str(args.retain_manifest), 128, forget_contract=False)
    calibration_obj, calibration_all = metric.manifest_samples(str(args.calibration_manifest), 128, forget_contract=False)
    forget_samples = forget_samples_all[: args.n_samples]
    retain_samples = retain_samples_all[: args.n_retain]
    calibration_samples = calibration_all[: args.n_calibration]
    forget_gradients, forget_tokens = metric.mean_gradients(
        model, tokenizer, forget_samples, targets, batch_size=1, max_length=args.max_length,
    )
    retain_gradients, retain_tokens = metric.mean_gradients(
        model, tokenizer, retain_samples, targets, batch_size=1, max_length=args.max_length,
    )
    hessians: Mapping[str, Any] = {}
    if args.quantizer == "gptq_int4":
        hessians, _, _ = collect_shared_hessians_gpu_cpu_finalize(
            model, tokenizer, calibration_samples, targets, device=torch.device("cuda:0"), max_samples=args.n_calibration,
        )
    reference = read_json(args.metric_reference)
    require(reference.get("quantizer") == args.quantizer and len(reference.get("per_layer", [])) == TARGET_COUNT, "metric reference identity")
    reference_by_layer = {str(row["layer"]): row for row in reference["per_layer"]}
    oracle = read_json(args.legacy_tensor_oracle)
    require(oracle.get("schema_version") == "org_e2b_b05_legacy_tensor_oracle_v1" and oracle.get("status") == "PASS", "legacy oracle status")
    require(oracle.get("cell_id") == args.cell_id and oracle.get("quantizer") == args.quantizer, "legacy oracle identity")
    require(oracle.get("target_count") == TARGET_COUNT and oracle.get("outcome_bearing") is False, "legacy oracle scope")
    oracle_by_layer = {str(row["layer"]): row for row in oracle["per_layer"]}
    rows = []
    max_retain_delta = 0.0
    max_reference_delta = 0.0
    for index, name in enumerate(targets, start=1):
        weight = parameters[name].detach()
        hessian = hessians.get(name)
        quantized = backend.quantize_dequantize(weight, args.quantizer, hessian=hessian)
        epsilon = (quantized.float() - weight.float()).detach().cpu().contiguous()
        require(tensor_sha256(epsilon) == oracle_by_layer[name]["displacement_sha256"], f"legacy displacement tensor drift: {name}")
        forget_a, forget_b = qdp_independent(forget_gradients[name], epsilon)
        retain_a, retain_b = qdp_independent(retain_gradients[name], epsilon)
        retain_delta = abs(retain_a - retain_b)
        max_retain_delta = max(max_retain_delta, retain_delta)
        require(retain_delta <= REAL_QDP_ATOL, f"retain QDP recomputation drift: {name}")
        reference_delta = abs(forget_a - float(reference_by_layer[name]["gqdp_signed"])) if production else None
        if reference_delta is not None:
            max_reference_delta = max(max_reference_delta, reference_delta)
            require(reference_delta <= REFERENCE_ATOL, f"forget QDP historical reference drift: {name}")
        rows.append({
            "layer": name,
            "n_parameters": int(weight.numel()),
            "forget_qdp_signed": forget_a,
            "retain_qdp_signed": retain_a,
            "protection_forget_benefit": -forget_a,
            "protection_retain_cost": -retain_a,
            "forget_gradient_sha256": tensor_sha256(forget_gradients[name]),
            "retain_gradient_sha256": tensor_sha256(retain_gradients[name]),
            "displacement_sha256": tensor_sha256(epsilon),
            "retain_independent_abs_delta": retain_delta,
            "historical_forget_reference_abs_delta": reference_delta,
        })
        del quantized, epsilon
        if index % 16 == 0:
            print(f"B05_FEATURE_MATRIX={index}/{len(targets)}", flush=True)
    total_parameters = sum(int(row["n_parameters"]) for row in rows)
    selectors = select_sets(rows, total_parameters)
    prerequisite_paths = {
        "checkpoint_ledger": args.checkpoint_ledger,
        "checkpoint_certification": args.checkpoint_certification,
        "b04_final": args.b04_final,
        "nf4_b04_final": args.nf4_b04_final,
        "environment_lock": args.environment_lock,
        "forget_manifest": args.forget_manifest,
        "retain_manifest": args.retain_manifest,
        "calibration_manifest": args.calibration_manifest,
        "metric_reference": args.metric_reference,
        "legacy_tensor_oracle": args.legacy_tensor_oracle,
    }
    payload = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "mode": args.mode,
        "production_eligible": production,
        "outcome_bearing": False,
        "protected_outcomes_inspected": False,
        "cell_id": args.cell_id,
        "quantizer": args.quantizer,
        "checkpoint": {"path": str(args.checkpoint_dir.resolve()), "certified_relative_path": args.checkpoint_relative},
        "runtime": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                    "hardware": torch.cuda.get_device_name(0), "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
        "sample_counts": {"forget": args.n_samples, "retain": args.n_retain, "calibration": args.n_calibration},
        "answer_tokens": {"forget": forget_tokens, "retain": retain_tokens},
        "target_count": len(rows), "target_parameters": total_parameters,
        "qdp_contract": "answer_token_weighted_mean_gradient_dot_Q_of_U_minus_U",
        "max_retain_independent_abs_delta": max_retain_delta,
        "max_historical_forget_reference_abs_delta": max_reference_delta if production else None,
        "conditions": selectors,
        "per_layer": rows,
        "per_layer_sha256": canonical_sha256(rows),
        "prerequisites": {key: {"path": str(path.resolve()), "sha256": sha256_file(path)} for key, path in prerequisite_paths.items()},
        "script_sha256": sha256_file(Path(__file__)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        "b05_behavior_authorized": False,
        "atlas_production_authorized": False,
    }
    atomic_json(args.output, payload)
    print("ORG_E2B_B05_FEATURE_CELL=PASS")
    print(f"MODE={args.mode}")
    print(f"CELL={args.cell_id}/{args.quantizer}")
    print(f"OUTPUT={args.output}")
    print("PROTECTED_OUTCOMES_INSPECTED=false")
    print("B05_BEHAVIOR_AUTHORIZED=false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ORG_E2B_B05_FEATURE_CELL=FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
