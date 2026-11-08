#!/usr/bin/env python3
"""Freeze historical-runtime dense Q/DQ tensor identities for one B0.5 cell."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import uuid


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


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
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    item = argparse.ArgumentParser(description=__doc__)
    item.add_argument("--cell-id", required=True)
    item.add_argument("--checkpoint-dir", type=Path, required=True)
    item.add_argument("--calibration-manifest", type=Path, required=True)
    item.add_argument("--quantizer", choices=("nf4", "gptq_int4", "rtn_int8"), required=True)
    item.add_argument("--n-calibration", type=int, required=True)
    item.add_argument("--expected-script-sha256", required=True)
    item.add_argument("--output", type=Path, required=True)
    return item


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    require(sha256_file(Path(__file__)) == args.expected_script_sha256, "script SHA drift")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gate2_v4_final" / "scripts"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "org_gb200_execution_v1" / "scripts"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import gate2_v4_backend as backend
    import run_gate2_metric_suite_v42 as metric
    from run_org_bridge_bundle_v2 import collect_shared_hessians_gpu_cpu_finalize

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one CUDA GPU required")
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
    require(len(targets) == 224 and len(targets) == len(set(targets)), "target universe")
    hessians = {}
    calibration_sha = sha256_file(args.calibration_manifest)
    if args.quantizer == "gptq_int4":
        _, all_samples = metric.manifest_samples(str(args.calibration_manifest), 128, forget_contract=False)
        samples = all_samples[: args.n_calibration]
        hessians, _, _ = collect_shared_hessians_gpu_cpu_finalize(
            model, tokenizer, samples, targets, device=torch.device("cuda:0"), max_samples=args.n_calibration,
        )
    parameters = dict(model.named_parameters())
    rows = []
    for index, name in enumerate(targets, start=1):
        weight = parameters[name].detach()
        quantized = backend.quantize_dequantize(weight, args.quantizer, hessian=hessians.get(name))
        displacement = (quantized.float() - weight.float()).detach().cpu().contiguous()
        rows.append({
            "layer": name, "n_parameters": int(weight.numel()),
            "source_weight_fp32_sha256": tensor_sha256(weight.float()),
            "quantized_weight_sha256": tensor_sha256(quantized.float()),
            "displacement_sha256": tensor_sha256(displacement),
        })
        del quantized, displacement
        if index % 16 == 0:
            print(f"B05_LEGACY_ORACLE_MATRIX={index}/{len(targets)}", flush=True)
    payload = {
        "schema_version": "org_e2b_b05_legacy_tensor_oracle_v1", "status": "PASS",
        "cell_id": args.cell_id, "quantizer": args.quantizer,
        "runtime": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                    "hardware": torch.cuda.get_device_name(0), "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "calibration_manifest": {"path": str(args.calibration_manifest.resolve()), "sha256": calibration_sha,
                                 "used_samples": args.n_calibration},
        "target_count": len(rows), "per_layer": rows,
        "script_sha256": sha256_file(Path(__file__)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcome_bearing": False, "protected_outcomes_inspected": False,
    }
    atomic_json(args.output, payload)
    print("ORG_E2B_B05_LEGACY_TENSOR_ORACLE=PASS")
    print(f"OUTPUT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
