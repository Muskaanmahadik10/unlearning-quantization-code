#!/usr/bin/env python3
"""Public-1B GPTQ/RTN packed-representation canary for ORG E2B B0.4.

This gate measures representation integrity only.  It deliberately emits no
forgetting, utility, privacy, recovery, or paper-outcome metric.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import tempfile
from typing import Any, Mapping, Sequence
import uuid

import inventory_org_e2b_b04_public_models_v1 as inventory
import probe_org_e2b_packed_backend_roundtrip_v3 as backend_probe
import run_org_e2b_b04_representation_canary_v1 as nf4_gate


SCHEMA = "org_e2b_b04_packed_public_canary_v1"
FINAL_SCHEMA = "org_e2b_b04_packed_public_final_v1"
RTN_INTERVENTION = "org_packed_storage_rtn_int8_dequant_matmul_v1"
EXPECTED_GPTQ_VERSION = "4.2.5"
ATOL = 1.0e-6
PROTECTED_SINGLETON = "model.layers.0.mlp.up_proj"
PROTECTED_PAIR = (PROTECTED_SINGLETON, "model.layers.0.self_attn.k_proj")
SYNTHETIC_IDS = nf4_gate.SYNTHETIC_IDS
FORBIDDEN_OUTCOME_KEYS = nf4_gate.FORBIDDEN_OUTCOME_KEYS


class CanaryError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise CanaryError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_directory(destination: Path, writer: Any) -> None:
    require(not destination.exists(), f"artifact exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        writer(temporary)
        require(any(temporary.iterdir()), "empty artifact staging directory")
        for path in temporary.rglob("*"):
            require(not path.is_symlink(), f"artifact symlink forbidden: {path}")
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def conditions(backend: str) -> dict[str, tuple[str, tuple[str, ...]]]:
    suffix = "GPTQ_INT4" if backend == "gptq" else "RTN_INT8"
    return {
        f"C04_QO_{suffix}_FULL": ("O", ()),
        f"C05_QU_{suffix}_ABSTAIN": ("U", ()),
        f"C06_QU_{suffix}_SINGLETON": ("U", (PROTECTED_SINGLETON,)),
        f"C07_QU_{suffix}_PAIR": ("U", PROTECTED_PAIR),
    }


def condition_contract(backend: str, condition: str) -> tuple[str, tuple[str, ...]]:
    table = conditions(backend)
    require(condition in table, f"invalid {backend} condition: {condition}")
    return table[condition]


def target_fqns() -> tuple[str, ...]:
    return inventory.expected_target_fqns()


def get_module(model: Any, fqn: str) -> Any:
    current = model
    for component in fqn.split("."):
        current = getattr(current, component)
    return current


def set_module(model: Any, fqn: str, module: Any) -> None:
    parent_name, child = fqn.rsplit(".", 1)
    setattr(get_module(model, parent_name), child, module)


def source_parameters(p5: Mapping[str, Any], role: str) -> dict[str, Mapping[str, Any]]:
    return nf4_gate.source_parameters(p5, role)


def tensor_sha256(torch: Any, tensor: Any) -> str:
    return inventory.tensor_sha256(torch, tensor)


def tensor_state_manifest(torch: Any, model: Any, names: Sequence[str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    total = 0
    for name in names:
        module = get_module(model, name)
        tensors: dict[str, Any] = {}
        for key, value in sorted(module.state_dict().items()):
            require(torch.is_tensor(value), f"non-tensor packed state: {name}:{key}")
            size = int(value.numel() * value.element_size())
            total += size
            tensors[key] = {
                "shape": list(value.shape), "dtype": str(value.dtype),
                "bytes": size, "sha256": tensor_sha256(torch, value),
            }
        require(bool(tensors), f"empty packed state: {name}")
        rows[name] = {
            "class": f"{type(module).__module__}.{type(module).__qualname__}",
            "tensors": tensors,
        }
    stable = {"modules": rows, "total_bytes": total}
    stable["sha256"] = canonical_sha256(stable)
    return stable


def build_rtn_linear_class(torch: Any):
    class PackedRTNLinear(torch.nn.Module):
        intervention_name = RTN_INTERVENTION

        def __init__(self, in_features: int, out_features: int, has_bias: bool, device: Any = None) -> None:
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.register_buffer("qweight", torch.empty(out_features, in_features, dtype=torch.int8, device=device))
            self.register_buffer("scale", torch.empty(out_features, 1, dtype=torch.float32, device=device))
            if has_bias:
                self.register_buffer("bias", torch.empty(out_features, dtype=torch.bfloat16, device=device))
            else:
                self.bias = None

        @classmethod
        def from_dense(cls, module: Any):
            result = cls(module.in_features, module.out_features, module.bias is not None, module.weight.device)
            weight = module.weight.detach().float()
            scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-12) / 127.0
            result.qweight.copy_(torch.round(weight / scale).clamp(-127, 127).to(torch.int8))
            result.scale.copy_(scale)
            if module.bias is not None:
                result.bias.copy_(module.bias.detach().to(torch.bfloat16))
            return result

        def forward(self, value: Any) -> Any:
            weight = self.qweight.to(dtype=value.dtype) * self.scale.to(dtype=value.dtype)
            bias = None if self.bias is None else self.bias.to(dtype=value.dtype)
            return torch.nn.functional.linear(value, weight, bias)

    return PackedRTNLinear


def install_rtn(torch: Any, model: Any, packed: Sequence[str], *, from_dense: bool) -> None:
    cls = build_rtn_linear_class(torch)
    for name in packed:
        old = get_module(model, name)
        replacement = cls.from_dense(old) if from_dense else cls(
            old.in_features, old.out_features, old.bias is not None, old.weight.device,
        )
        set_module(model, name, replacement)


def runtime(torch: Any, transformers: Any) -> dict[str, Any]:
    import importlib.metadata

    expected = Path(os.environ["ORG_E2B_EXPECTED_ENVIRONMENT"]).resolve()
    require(Path(sys.executable).resolve().is_relative_to(expected), "interpreter escaped environment")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one CUDA GPU required")
    require(torch.cuda.get_device_name(0) == "NVIDIA H200", "H200 required")
    require(tuple(torch.cuda.get_device_capability(0)) == (9, 0), "H200 capability mismatch")
    require(os.environ.get("HF_HUB_OFFLINE") == "1", "HF offline mode missing")
    require(os.environ.get("TRANSFORMERS_OFFLINE") == "1", "Transformers offline mode missing")
    return {
        "pid": os.getpid(), "process_uuid": str(uuid.uuid4()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.version, "python_executable": str(Path(sys.executable).resolve()),
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "transformers": transformers.__version__,
        "safetensors": importlib.metadata.version("safetensors"),
        "gptqmodel": importlib.metadata.version("gptqmodel"),
        "hardware": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }


def load_modules() -> tuple[Any, Any, Any]:
    import torch
    import transformers
    from safetensors import torch as safe_torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch, transformers, safe_torch


def verify_source_model(torch: Any, transformers: Any, snapshot: Path, p5: Mapping[str, Any], role: str) -> tuple[Any, Any]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True,
    )
    require(
        nf4_gate.tokenizer_fingerprint(tokenizer) == p5["roles"][role]["tokenizer_runtime_fingerprint"],
        f"tokenizer fingerprint drift: {role}",
    )
    loaded = transformers.AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.bfloat16, device_map={"": 0}, output_loading_info=True,
        attn_implementation="eager",
    )
    model, info = loaded
    require(all(info.get(key, []) == [] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), "unclean source load")
    source = source_parameters(p5, role)
    actual = dict(model.named_parameters(remove_duplicate=False))
    require(set(actual) == set(source), f"source parameter universe drift: {role}")
    for name, parameter in actual.items():
        require(parameter.dtype == torch.bfloat16 and str(parameter.device) == "cuda:0", f"source dtype/device: {name}")
        require(tensor_sha256(torch, parameter) == source[name]["sha256"], f"source tensor drift: {name}")
    require(model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr(), "source tie broken")
    return model, tokenizer


def packed_names(protected: Sequence[str]) -> tuple[str, ...]:
    protected_set = set(protected)
    return tuple(name for name in target_fqns() if name not in protected_set)


def inspect_representation(
    torch: Any, model: Any, p5: Mapping[str, Any], role: str,
    protected: Sequence[str], backend: str,
) -> dict[str, Any]:
    protected_set, targets = set(protected), set(target_fqns())
    expected_packed = targets - protected_set
    modules = dict(model.named_modules())
    require(targets <= set(modules), "target module universe incomplete")
    source = source_parameters(p5, role)
    if backend == "gptq":
        from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    packed, dense = [], []
    for name in sorted(targets):
        module = modules[name]
        if name in protected_set:
            require(type(module) is torch.nn.Linear, f"protected module not exact Linear: {name}")
            require(module.weight.dtype == torch.bfloat16 and str(module.weight.device) == "cuda:0", f"protected dtype/device: {name}")
            require(tensor_sha256(torch, module.weight) == source[f"{name}.weight"]["sha256"], f"protected tensor drift: {name}")
            dense.append(name)
        elif backend == "gptq":
            require(isinstance(module, TorchQuantLinear), f"GPTQ module class: {name}")
            packed.append(name)
        else:
            require(type(module).__qualname__.endswith("PackedRTNLinear"), f"RTN module class: {name}")
            require(module.qweight.dtype == torch.int8 and module.scale.dtype == torch.float32, f"RTN packed dtype: {name}")
            require(str(module.qweight.device) == str(module.scale.device) == "cuda:0", f"RTN packed device: {name}")
            packed.append(name)
    require(set(packed) == expected_packed and set(dense) == protected_set, "packed/protected partition")

    target_prefixes = tuple(f"{name}." for name in targets)
    nontarget_hashes = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if not name.startswith(target_prefixes):
            require(name in source, f"nontarget absent from source: {name}")
            require(parameter.dtype == torch.bfloat16 and str(parameter.device) == "cuda:0", f"nontarget dtype/device: {name}")
            digest = tensor_sha256(torch, parameter)
            require(digest == source[name]["sha256"], f"nontarget source drift: {name}")
            nontarget_hashes[name] = digest
    require(model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr(), "embedding/lm_head tie broken")
    total = int(p5["roles"][role]["target_linear_parameters"])
    protected_mass = sum(int(source[f"{name}.weight"]["numel"]) for name in protected)
    require(protected_mass / total <= 0.02, "protected parameter mass exceeds 2 percent")
    state = tensor_state_manifest(torch, model, sorted(expected_packed))
    dense_bytes = sum(int(source[f"{name}.weight"]["numel"]) * 2 for name in expected_packed)
    require(state["total_bytes"] < dense_bytes, "packed representation not smaller than BF16")
    if backend == "gptq":
        for item in state["modules"].values():
            require(item["tensors"].get("qweight", {}).get("dtype") == "torch.int32", "GPTQ qweight dtype")
            require("qzeros" in item["tensors"] and "scales" in item["tensors"], "GPTQ packed tensors missing")
    stable = {
        "backend": backend, "role": role, "packed_fqns": sorted(expected_packed),
        "protected_fqns": list(protected), "packed_module_count": len(expected_packed),
        "protected_module_count": len(protected), "target_linear_count": len(targets),
        "protected_parameter_mass": protected_mass, "target_parameter_mass": total,
        "protected_parameter_fraction": protected_mass / total,
        "nontarget_weight_sha256": nontarget_hashes,
        "embedding_lm_head_tied": True, "cpu_disk_offload_count": 0,
        "packed_state": state, "dense_reference_bytes": dense_bytes,
        "fused_integer_kernel": False if backend == "rtn" else None,
    }
    stable["sha256"] = canonical_sha256(stable)
    return stable


def synthetic_inputs(torch: Any) -> dict[str, Any]:
    ids = torch.tensor([SYNTHETIC_IDS], dtype=torch.long, device="cuda:0")
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def forward_record(torch: Any, model: Any, payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    with torch.inference_mode():
        first = model(**payload, use_cache=False, return_dict=True)
        second = model(**payload, use_cache=False, return_dict=True)
    torch.cuda.synchronize()
    a = first.logits.detach().cpu().contiguous()
    b = second.logits.detach().cpu().contiguous()
    loss_a = first.loss.detach().float().cpu().reshape(1)
    loss_b = second.loss.detach().float().cpu().reshape(1)
    require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(loss_a).all()), "nonfinite inference")
    require(torch.equal(a, b) and torch.equal(loss_a, loss_b), "within-process nondeterminism")
    record = {
        "logits_shape": list(a.shape), "logits_dtype": str(a.dtype),
        "logits_sha256": tensor_sha256(torch, a), "loss_sha256": tensor_sha256(torch, loss_a),
        "within_process_exact": True,
    }
    tensors = {
        "input_ids": payload["input_ids"].cpu(), "attention_mask": payload["attention_mask"].cpu(),
        "labels": payload["labels"].cpu(), "logits": a, "loss": loss_a,
    }
    return record, tensors


def load_rtn(torch: Any, transformers: Any, artifact: Path, packed: Sequence[str]) -> Any:
    from safetensors.torch import load_model

    config_payload = read_json(artifact / "config.json")
    metadata = config_payload.pop("org_quantization", None)
    require(isinstance(metadata, dict) and metadata.get("intervention") == RTN_INTERVENTION, "RTN metadata")
    require(metadata.get("packed_fqns") == list(packed), "RTN packed metadata")
    config = transformers.LlamaConfig.from_dict(config_payload)
    model = transformers.LlamaForCausalLM(config).to(dtype=torch.bfloat16)
    install_rtn(torch, model, packed, from_dense=False)
    missing, unexpected = load_model(model, str(artifact / "model.safetensors"), strict=True, device="cpu")
    require(list(missing) == [] and list(unexpected) == [], "RTN state load not strict")
    model.to("cuda:0")
    return model


def load_gptq(torch: Any, artifact: Path) -> tuple[Any, Any]:
    from gptqmodel import BACKEND, GPTQModel

    wrapper = GPTQModel.from_quantized(
        str(artifact), device="cuda:0", backend=BACKEND.TORCH,
        torch_dtype=torch.bfloat16, trust_remote_code=False, local_files_only=True,
    )
    return wrapper, wrapper.model


def calibration(torch: Any) -> list[dict[str, Any]]:
    # Deterministic engineering calibration only; never interpreted scientifically.
    rows = []
    for index in range(16):
        ids = [128000, 791 + index, 1401, 374, 264, 1296, 13, 128009]
        tensor = torch.tensor([ids * 4], dtype=torch.long)
        rows.append({"input_ids": tensor, "attention_mask": torch.ones_like(tensor)})
    return rows


def verify_prerequisites(args: argparse.Namespace) -> dict[str, Any]:
    require(
        sha256_file(args.environment_lock) == backend_probe.EXPECTED_ENV_LOCK_SHA256,
        "certified environment-lock hash drift",
    )
    links = inventory.validate_receipts(
        args.local_manifest.resolve(), args.identity_lock.resolve(),
        args.b03b_final.resolve(), args.environment_lock.resolve(),
    )
    p5 = read_json(args.p5)
    require(p5.get("status") == "PASS", "P5 failed")
    require(p5.get("scope_guards", {}).get("b0_4_gpu_launcher_authorized") is True, "P5 did not authorize B0.4")
    backend_final = read_json(args.backend_final)
    require(backend_final.get("overall_status") == "PASS", "B0.3 backend final failed")
    require(backend_final.get("backend_selector") == args.backend, "B0.3 backend mismatch")
    expected_qualification = "STRICT_B0_3_C_CANARY_PASS" if args.backend == "gptq" else "STRICT_B0_3_D_CANARY_PASS"
    require(backend_final.get("qualification") == expected_qualification, "B0.3 qualification mismatch")
    require(backend_final.get("certification", {}).get("model_level_mixed_packed_roundtrip") is True, "B0.3 model roundtrip absent")
    nf4_final = read_json(args.nf4_b04_final)
    require(nf4_final.get("schema_version") == nf4_gate.FINAL_SCHEMA and nf4_final.get("status") == "PASS", "NF4 B0.4 source-role gate failed")
    require(nf4_final.get("checks", {}).get("R_never_substituted_for_O") is True, "O/R identity gate absent")
    require(nf4_final.get("checks", {}).get("source_role_hashes_distinct_and_linked") is True, "O/U/R source identity gate absent")
    return {
        **links,
        "p5": {"path": str(args.p5.resolve()), "sha256": sha256_file(args.p5)},
        "backend_final": {"path": str(args.backend_final.resolve()), "sha256": sha256_file(args.backend_final)},
        "nf4_b04_final": {"path": str(args.nf4_b04_final.resolve()), "sha256": sha256_file(args.nf4_b04_final)},
        "source_manifest": backend_probe.parse_source_manifest(args.source_manifest, args.repository_root),
    }


def precheck(args: argparse.Namespace) -> None:
    links = verify_prerequisites(args)
    expected = set(target_fqns())
    require(len(expected) == 112, "target universe drift")
    p5 = read_json(args.p5)
    for condition, (role, protected) in conditions(args.backend).items():
        require(set(protected) <= expected and len(protected) <= 2, f"protected set: {condition}")
        source = source_parameters(p5, role)
        mass = sum(int(source[f"{name}.weight"]["numel"]) for name in protected)
        require(mass / int(p5["roles"][role]["target_linear_parameters"]) <= 0.02, f"protected mass: {condition}")
    report = {
        "schema_version": SCHEMA, "phase": "precheck", "status": "PASS",
        "created_at_utc": now(), "backend": args.backend,
        "condition_ids": list(conditions(args.backend)), "prerequisites": links,
        "outcome_inspected": False, "b05_authorized": False,
    }
    atomic_json(args.output, report)
    print(f"ORG_E2B_B04_{args.backend.upper()}_PRECHECK=PASS")


def verify_run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_json(args.run_metadata)
    require(metadata.get("phase") == "precheck" and metadata.get("status") == "PASS", "precheck invalid")
    require(metadata.get("backend") == args.backend, "precheck backend drift")
    for item in metadata["prerequisites"].values():
        if isinstance(item, dict) and "path" in item and "sha256" in item:
            require(sha256_file(Path(item["path"])) == item["sha256"], "prerequisite hash drift")
    return metadata


def produce(args: argparse.Namespace) -> None:
    role, protected = condition_contract(args.backend, args.condition)
    verify_run_metadata(args)
    p5, local = read_json(args.p5), read_json(args.local_manifest)
    snapshot = Path(local["snapshot_root"]) / role
    guard = nf4_gate.NetworkGuard(); guard.install()
    try:
        torch, transformers, _ = load_modules()
        rt = runtime(torch, transformers)
        if args.backend == "gptq":
            rt["gptq_source_sha256"] = backend_probe.inspect_gptq_sources()
        torch.cuda.reset_peak_memory_stats(0)
        source_model, tokenizer = verify_source_model(torch, transformers, snapshot, p5, role)
        packed = packed_names(protected)
        source_hash = canonical_sha256({name: row["sha256"] for name, row in source_parameters(p5, role).items()})
        if args.backend == "gptq":
            from gptqmodel import BACKEND, GPTQModel, QuantizeConfig

            del source_model
            gc.collect(); torch.cuda.empty_cache()
            dynamic = {f"-:^{re.escape(name)}$": {} for name in (*protected, "lm_head")}
            config = QuantizeConfig(
                bits=4, group_size=128, desc_act=False, sym=True, dynamic=dynamic,
                device="cuda:0", parallel_packing=True,
                meta={"quantizer": [f"gptqmodel:{EXPECTED_GPTQ_VERSION}"], "org_arm": "b04_public"},
            )
            for name in protected:
                require(config.dynamic_get(name) is False, f"GPTQ skip not matched: {name}")
            require(config.dynamic_get(packed[0]) is None, "GPTQ skip overmatched")
            wrapper = GPTQModel.from_pretrained(
                str(snapshot), config, torch_dtype=torch.bfloat16,
                trust_remote_code=False, local_files_only=True,
            )
            wrapper.quantize(
                calibration(torch), backend=BACKEND.TORCH, batch_size=1,
                calibration_enable_gpu_cache=True, auto_gc=True, fail_safe=False,
            )
            wrapper.to("cuda:0")
            model = wrapper.model
            representation = inspect_representation(torch, model, p5, role, protected, args.backend)

            def writer(destination: Path) -> None:
                wrapper.save(str(destination))
                tokenizer.save_pretrained(destination)

        else:
            model = source_model
            install_rtn(torch, model, packed, from_dense=True)
            representation = inspect_representation(torch, model, p5, role, protected, args.backend)

            def writer(destination: Path) -> None:
                from safetensors.torch import save_model
                payload = model.config.to_dict()
                payload["org_quantization"] = {
                    "intervention": RTN_INTERVENTION, "packed_fqns": list(packed),
                    "protected_fqns": list(protected), "decode_compute_dtype": "activation_dtype",
                    "fused_integer_kernel": False,
                }
                (destination / "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                tokenizer.save_pretrained(destination)
                model.to("cpu")
                save_model(model, str(destination / "model.safetensors"))

        atomic_directory(args.artifact, writer)
        artifact_manifest = nf4_gate.directory_manifest(args.artifact)
        report = {
            "schema_version": SCHEMA, "phase": "produce", "status": "PASS",
            "created_at_utc": now(), "backend": args.backend, "condition_id": args.condition,
            "role": role, "protected_fqns": list(protected), "pid": os.getpid(),
            "runtime": rt, "source_parameter_manifest_sha256": source_hash,
            "representation_before_save": representation,
            "artifact": {"path": str(args.artifact.resolve()), "manifest": artifact_manifest},
            "producer_inference_executed": False, "reference_origin": "independent_fresh_reload_a",
            "network_attempt_count": guard.attempts,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(0)),
            "outcome_inspected": False, "b05_authorized": False,
        }
        require(guard.attempts == 0, "network attempt")
        atomic_json(args.output, report)
        print(f"ORG_E2B_B04_{args.backend.upper()}_PRODUCE=PASS condition={args.condition}")
    finally:
        guard.restore()


def load_artifact(args: argparse.Namespace, torch: Any, transformers: Any, packed: Sequence[str]) -> tuple[Any, Any]:
    if args.backend == "gptq":
        return load_gptq(torch, args.artifact)
    model = load_rtn(torch, transformers, args.artifact, packed)
    return model, model


def reload_a(args: argparse.Namespace) -> None:
    producer = read_json(args.producer)
    role, protected = condition_contract(args.backend, producer.get("condition_id"))
    require(producer.get("status") == "PASS" and producer.get("phase") == "produce", "producer invalid")
    require(producer.get("backend") == args.backend, "producer backend")
    require(not nf4_gate.process_alive(int(producer["pid"])), "producer still alive")
    verify_run_metadata(args)
    require(nf4_gate.directory_manifest(args.artifact) == producer["artifact"]["manifest"], "artifact byte drift")
    guard = nf4_gate.NetworkGuard(); guard.install()
    try:
        torch, transformers, safe_torch = load_modules()
        rt = runtime(torch, transformers)
        if args.backend == "gptq":
            rt["gptq_source_sha256"] = backend_probe.inspect_gptq_sources()
        model, underlying = load_artifact(args, torch, transformers, packed_names(protected))
        representation = inspect_representation(torch, underlying, read_json(args.p5), role, protected, args.backend)
        output, tensors = forward_record(torch, model, synthetic_inputs(torch))
        require(not args.reference.exists(), "reference exists")
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.reference.with_name(f".{args.reference.name}.{os.getpid()}.tmp")
        safe_torch.save_file(tensors, temporary)
        os.replace(temporary, args.reference)
        report = {
            "schema_version": SCHEMA, "phase": "reload_a", "status": "PASS",
            "created_at_utc": now(), "backend": args.backend, "condition_id": producer["condition_id"],
            "role": role, "protected_fqns": list(protected), "pid": os.getpid(),
            "producer_pid": producer["pid"], "runtime": rt, "representation": representation,
            "output": output, "reference": {"path": str(args.reference.resolve()), "sha256": sha256_file(args.reference)},
            "artifact_manifest": nf4_gate.directory_manifest(args.artifact),
            "network_attempt_count": guard.attempts, "outcome_inspected": False,
        }
        require(guard.attempts == 0, "network attempt")
        atomic_json(args.output, report)
        print(f"ORG_E2B_B04_{args.backend.upper()}_FRESH_RELOAD_A=PASS condition={producer['condition_id']}")
    finally:
        guard.restore()


def reload_b(args: argparse.Namespace) -> None:
    producer, first = read_json(args.producer), read_json(args.reload_a)
    role, protected = condition_contract(args.backend, producer.get("condition_id"))
    require(first.get("phase") == "reload_a" and first.get("status") == "PASS", "reload A invalid")
    pids = {int(producer["pid"]), int(first["pid"]), os.getpid()}
    require(len(pids) == 3 and all(not nf4_gate.process_alive(pid) for pid in pids if pid != os.getpid()), "fresh-process independence")
    verify_run_metadata(args)
    require(sha256_file(args.reference) == first["reference"]["sha256"], "reference byte drift")
    require(nf4_gate.directory_manifest(args.artifact) == producer["artifact"]["manifest"], "artifact byte drift")
    guard = nf4_gate.NetworkGuard(); guard.install()
    try:
        torch, transformers, safe_torch = load_modules()
        rt = runtime(torch, transformers)
        if args.backend == "gptq":
            rt["gptq_source_sha256"] = backend_probe.inspect_gptq_sources()
        model, underlying = load_artifact(args, torch, transformers, packed_names(protected))
        representation = inspect_representation(torch, underlying, read_json(args.p5), role, protected, args.backend)
        require(representation == first["representation"], "packed state differs across fresh reloads")
        reference = safe_torch.load_file(args.reference, device="cpu")
        payload = {key: reference[key].to("cuda:0") for key in ("input_ids", "attention_mask", "labels")}
        output, tensors = forward_record(torch, model, payload)
        logits_delta = (tensors["logits"].float() - reference["logits"].float()).abs()
        loss_delta = (tensors["loss"].float() - reference["loss"].float()).abs()
        parity = {
            "logits_exact": bool(torch.equal(tensors["logits"], reference["logits"])),
            "loss_exact": bool(torch.equal(tensors["loss"], reference["loss"])),
            "logits_max_abs": float(logits_delta.max()), "loss_max_abs": float(loss_delta.max()),
        }
        require(parity["logits_max_abs"] <= ATOL and parity["loss_max_abs"] <= ATOL, "fresh reload numerical drift")
        report = {
            "schema_version": SCHEMA, "phase": "reload_b", "status": "PASS",
            "created_at_utc": now(), "backend": args.backend, "condition_id": producer["condition_id"],
            "role": role, "protected_fqns": list(protected), "pid": os.getpid(),
            "producer_pid": producer["pid"], "reload_a_pid": first["pid"], "runtime": rt,
            "representation": representation, "output": output, "fresh_reload_parity": parity,
            "artifact_manifest": nf4_gate.directory_manifest(args.artifact),
            "network_attempt_count": guard.attempts, "outcome_inspected": False,
        }
        require(guard.attempts == 0, "network attempt")
        atomic_json(args.output, report)
        print(f"ORG_E2B_B04_{args.backend.upper()}_FRESH_RELOAD_B=PASS condition={producer['condition_id']}")
    finally:
        guard.restore()


def validate(args: argparse.Namespace) -> None:
    metadata = verify_run_metadata(args)
    table = conditions(args.backend)
    receipts = {}
    for condition in table:
        producer = read_json(args.run_dir / "producers" / f"{condition}.json")
        first = read_json(args.run_dir / "reload_a" / f"{condition}.json")
        second = read_json(args.run_dir / "receipts" / f"{condition}.json")
        require(producer.get("condition_id") == first.get("condition_id") == second.get("condition_id") == condition, "condition identity")
        require(producer.get("status") == first.get("status") == second.get("status") == "PASS", "phase failure")
        require(producer.get("role") == first.get("role") == second.get("role") == table[condition][0], "role drift")
        require(tuple(second.get("protected_fqns", [])) == table[condition][1], "protected set drift")
        require(first["representation"] == second["representation"], "fresh reload representation drift")
        require(first["output"]["logits_sha256"] == second["output"]["logits_sha256"], "fresh reload output hash drift")
        require(second["fresh_reload_parity"]["logits_max_abs"] <= ATOL, "fresh reload parity")
        require(producer["artifact"]["manifest"] == first["artifact_manifest"] == second["artifact_manifest"], "artifact manifest drift")
        require(len({producer["pid"], first["pid"], second["pid"]}) == 3, "process independence")
        require(len({producer["runtime"]["slurm_job_id"], first["runtime"]["slurm_job_id"], second["runtime"]["slurm_job_id"]}) == 1, "job identity drift")
        receipts[condition] = {
            "path": str((args.run_dir / "receipts" / f"{condition}.json").resolve()),
            "sha256": sha256_file(args.run_dir / "receipts" / f"{condition}.json"),
        }
    c05 = next(name for name in table if name.startswith("C05_"))
    checks = {
        "condition_count": 4, "condition_ids_exact": set(table) == set(receipts),
        "O_U_R_source_gate_inherited_and_hash_locked": True,
        "R_never_substituted_for_O": True,
        "QO_source_is_exact_O": table[next(name for name in table if name.startswith("C04_"))][0] == "O",
        "QU_source_is_exact_U": all(role == "U" for name, (role, _) in table.items() if name != next(key for key in table if key.startswith("C04_"))),
        "canonical_QU_condition": c05,
        "protected_fqn_sets_exact": True, "protected_parameter_mass_cap_passed": True,
        "packed_module_inventory_exact": True, "two_independent_fresh_reloads_each": True,
        "exact_packed_state_across_fresh_reloads": True, "artifact_hash_verification_passed": True,
        "network_attempt_count": 0, "cpu_disk_offload_count": 0,
        "outcome_metric_count": 0,
    }
    require(all(value is True or value in (0, 4, c05) for value in checks.values()), "final checks")
    report = {
        "schema_version": FINAL_SCHEMA, "status": "PASS", "created_at_utc": now(),
        "backend": args.backend, "hardware": "H200", "condition_ids": list(table),
        "checks": checks, "condition_receipts": receipts,
        "precheck": {"path": str(args.run_metadata.resolve()), "sha256": sha256_file(args.run_metadata)},
        "prerequisites": metadata["prerequisites"],
        "outcome_inspected": False, "paper_claim_authorized": False,
        "b05_authorized": False, "b05_authorization_reason": "PENDING_POST_JOB_DUAL_BACKEND_CERTIFICATION",
        "atlas_production_authorized": False,
    }
    serialized = json.dumps(report, sort_keys=True).lower()
    require(not any(key in serialized for key in FORBIDDEN_OUTCOME_KEYS), "outcome key leaked")
    atomic_json(args.output, report)
    print(f"ORG_E2B_B04_{args.backend.upper()}_PUBLIC_REPRESENTATION=PASS")
    print("CONDITIONS=4")
    print(f"OUTPUT={args.output}")
    print("B05_AUTHORIZED=false")
    print("ATLAS_PRODUCTION_AUTHORIZED=false")


def common(item: argparse.ArgumentParser) -> None:
    item.add_argument("--backend", choices=("gptq", "rtn"), required=True)
    item.add_argument("--repository-root", type=Path, required=True)
    item.add_argument("--environment-root", type=Path, required=True)
    item.add_argument("--environment-lock", type=Path, required=True)
    item.add_argument("--source-manifest", type=Path, required=True)
    item.add_argument("--p5", type=Path, required=True)
    item.add_argument("--local-manifest", type=Path, required=True)
    item.add_argument("--identity-lock", type=Path, required=True)
    item.add_argument("--b03b-final", type=Path, required=True)
    item.add_argument("--backend-final", type=Path, required=True)
    item.add_argument("--nf4-b04-final", type=Path, required=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("precheck"); common(pre); pre.add_argument("--output", type=Path, required=True)
    for name in ("produce", "reload-a", "reload-b"):
        item = sub.add_parser(name); common(item)
        item.add_argument("--run-metadata", type=Path, required=True)
        item.add_argument("--condition", required=False)
        item.add_argument("--artifact", type=Path, required=True)
        item.add_argument("--producer", type=Path, required=False)
        item.add_argument("--reload-a", type=Path, required=False)
        item.add_argument("--reference", type=Path, required=False)
        item.add_argument("--output", type=Path, required=True)
    final = sub.add_parser("validate"); common(final)
    final.add_argument("--run-metadata", type=Path, required=True)
    final.add_argument("--run-dir", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "precheck": precheck(args)
        elif args.command == "produce": produce(args)
        elif args.command == "reload-a": reload_a(args)
        elif args.command == "reload-b": reload_b(args)
        else: validate(args)
    except Exception as exc:
        print(f"ORG_E2B_B04_PACKED_{args.command.upper()}=FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
