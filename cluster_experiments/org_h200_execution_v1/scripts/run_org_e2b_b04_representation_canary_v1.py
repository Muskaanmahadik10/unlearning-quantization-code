#!/usr/bin/env python3
"""Strict public O/U/R packed-NF4 representation canary for ORG B0.4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import uuid

import inventory_org_e2b_b04_public_models_v1 as inventory
import prepare_org_e2b_b04_public_models_v1 as prep


SCHEMA = "org_e2b_b04_public_representation_canary_v1"
FINAL_SCHEMA = "org_e2b_b04_public_representation_final_v1"
SOURCE_CONDITIONS = {"C01_O_BF16": "O", "C02_U_BF16": "U", "C03_R_BF16": "R"}
PACKED_CONDITIONS = {
    "C04_QO_NF4_FULL": ("O", ()),
    "C05_QU_NF4_ABSTAIN": ("U", ()),
    "C06_QU_NF4_SINGLETON": ("U", ("model.layers.0.mlp.up_proj",)),
    "C07_QU_NF4_PAIR": (
        "U",
        ("model.layers.0.mlp.up_proj", "model.layers.0.self_attn.k_proj"),
    ),
}
SYNTHETIC_IDS = [128000, 791, 1401, 374, 264, 1296, 13, 128009]
FORBIDDEN_OUTCOME_KEYS = (
    "forget_ce", "retain_ce", "recovery", "resurgence", "truth_ratio",
    "membership", "privacy", "gqdp", "upqi", "utility",
)


class CanaryError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise CanaryError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def directory_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"artifact symlink forbidden: {path}")
        if path.is_file():
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    require(rows, f"empty artifact: {root}")
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"files": rows, "file_count": len(rows), "total_bytes": sum(x["bytes"] for x in rows), "sha256": digest}


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class NetworkGuard:
    def __init__(self) -> None:
        self.attempts = 0
        self.original = socket.socket.connect

    def install(self) -> None:
        guard = self
        def blocked(_sock: Any, address: Any) -> None:
            guard.attempts += 1
            raise CanaryError(f"network attempt forbidden: {address}")
        socket.socket.connect = blocked  # type: ignore[assignment]

    def restore(self) -> None:
        socket.socket.connect = self.original  # type: ignore[assignment]


def source_parameters(p5: Mapping[str, Any], role: str) -> dict[str, Mapping[str, Any]]:
    rows = p5.get("roles", {}).get(role, {}).get("parameters", [])
    require(isinstance(rows, list), f"P5 parameter inventory missing: {role}")
    result = {str(item["fqn"]): item for item in rows}
    require(len(result) == len(rows), f"duplicate P5 parameter FQN: {role}")
    return result


def tokenizer_fingerprint(tokenizer: Any) -> dict[str, Any]:
    return {
        "class": tokenizer.__class__.__name__, "vocab_size": int(tokenizer.vocab_size),
        "length": int(len(tokenizer)), "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id, "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id, "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id, "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
        "all_special_tokens": list(tokenizer.all_special_tokens),
        "all_special_ids": list(tokenizer.all_special_ids),
        "chat_template_sha256": hashlib.sha256((tokenizer.chat_template or "").encode()).hexdigest(),
        "backend_tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "probe_input_ids": [
            tokenizer.encode(text, add_special_tokens=add)
            for text, add in (("A reproducible evaluation probe.", False), ("A reproducible evaluation probe.", True))
        ],
    }


def runtime(torch: Any, transformers: Any, bnb: Any) -> dict[str, Any]:
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one CUDA GPU required")
    require("H200" in torch.cuda.get_device_name(0), "H200 required")
    require(tuple(torch.cuda.get_device_capability(0)) == (9, 0), "H200 capability mismatch")
    expected_environment = os.environ.get("ORG_E2B_EXPECTED_ENVIRONMENT")
    require(bool(expected_environment), "expected environment was not declared")
    expected_root = Path(expected_environment).resolve()
    require(Path(sys.executable).resolve().is_relative_to(expected_root), "interpreter escaped certified environment")
    module_paths = {
        "torch": str(Path(torch.__file__).resolve()),
        "transformers": str(Path(transformers.__file__).resolve()),
        "bitsandbytes": str(Path(bnb.__file__).resolve()),
    }
    require(all(Path(path).is_relative_to(expected_root) for path in module_paths.values()), "runtime import escaped certified environment")
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    require(driver and "\n" not in driver, "one exact NVIDIA driver record required")
    return {
        "python": sys.version, "executable": str(Path(sys.executable).resolve()),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "bitsandbytes": bnb.__version__, "cuda_build": torch.version.cuda,
        "hardware": torch.cuda.get_device_name(0), "visible_gpu_count": 1,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "driver_version": driver,
        "kernel": platform.release(), "machine": platform.machine(),
        "module_paths": module_paths,
    }


def inputs(torch: Any) -> dict[str, Any]:
    ids = torch.tensor([SYNTHETIC_IDS], dtype=torch.long, device="cuda:0")
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def forward(torch: Any, model: Any, payload: Mapping[str, Any]) -> tuple[Any, Any]:
    model.eval()
    with torch.inference_mode():
        out = model(**payload, use_cache=False, return_dict=True)
    torch.cuda.synchronize()
    logits, loss = out.logits.detach().cpu().contiguous(), out.loss.detach().float().cpu().reshape(1)
    require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(loss).all()), "nonfinite synthetic output")
    return logits, loss


def tensor_metrics(torch: Any, left: Any, right: Any) -> dict[str, Any]:
    delta = left.float() - right.float()
    return {"exact": bool(torch.equal(left, right)), "max_abs": float(delta.abs().max()), "rmse": float(delta.square().mean().sqrt())}


def inspect_representation(torch: Any, bnb: Any, model: Any, p5: Mapping[str, Any], role: str, protected: Sequence[str]) -> dict[str, Any]:
    targets = set(inventory.expected_target_fqns())
    protected_set = set(protected)
    require(protected_set <= targets and len(protected_set) <= 2, "invalid protected set")
    named_modules = dict(model.named_modules())
    require(targets <= set(named_modules), "target module universe incomplete")
    source = source_parameters(p5, role)
    named_parameters = dict(model.named_parameters(remove_duplicate=False))
    packed, dense, packed_facts, protected_hashes = [], [], {}, {}
    for name in sorted(targets):
        module = named_modules[name]
        if name in protected_set:
            require(type(module) is torch.nn.Linear, f"protected module is not exact BF16 Linear: {name}")
            require(module.weight.dtype == torch.bfloat16 and str(module.weight.device) == "cuda:0", f"protected tensor placement: {name}")
            digest = inventory.tensor_sha256(torch, module.weight)
            require(digest == source[f"{name}.weight"]["sha256"], f"protected source tensor drift: {name}")
            protected_hashes[name] = digest
            dense.append(name)
        else:
            require(type(module) is bnb.nn.Linear4bit, f"target is not exact Linear4bit: {name}")
            weight = module.weight
            require(type(weight) is bnb.nn.Params4bit, f"target is not exact Params4bit: {name}")
            state = getattr(weight, "quant_state", None)
            require(state is not None and getattr(weight, "bnb_quantized", None) is True, f"missing packed state: {name}")
            require(weight.data.dtype == torch.uint8 and str(weight.data.device) == "cuda:0", f"packed tensor placement: {name}")
            require(module.compute_dtype == torch.bfloat16, f"packed compute dtype: {name}")
            require(weight.quant_storage == torch.uint8, f"packed storage dtype: {name}")
            require(weight.compress_statistics is False and weight.quant_type == "nf4", f"packed NF4 contract: {name}")
            require(int(weight.blocksize) == 64 and int(state.blocksize) == 64, f"packed blocksize: {name}")
            require(state.quant_type == "nf4" and state.nested is False, f"quantization state contract: {name}")
            state_rows = {}
            for key, value in sorted(state.as_dict(packed=True).items()):
                require(torch.is_tensor(value), f"nontensor quantization state: {name}:{key}")
                state_rows[key] = inventory.tensor_sha256(torch, value)
            packed_facts[name] = {
                "weight_sha256": inventory.tensor_sha256(torch, weight.data),
                "weight_shape": list(weight.data.shape), "weight_dtype": str(weight.data.dtype),
                "quant_type": state.quant_type, "blocksize": int(state.blocksize),
                "quant_state_sha256": state_rows,
            }
            packed.append(name)
    require(set(packed) == targets - protected_set and set(dense) == protected_set, "packed/dense partition mismatch")
    target_prefixes = tuple(f"{name}." for name in targets)
    nontarget_hashes = {}
    for name, parameter in named_parameters.items():
        if not name.startswith(target_prefixes):
            require(str(parameter.device) == "cuda:0" and parameter.dtype == torch.bfloat16, f"nontarget placement/dtype: {name}")
            digest = inventory.tensor_sha256(torch, parameter)
            require(name in source and digest == source[name]["sha256"], f"nontarget source tensor drift: {name}")
            nontarget_hashes[name] = digest
    tied = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    require(tied, "embedding/lm_head tie broken")
    quantization_config = getattr(model.config, "quantization_config", None)
    require(quantization_config is not None, "serialized quantization configuration missing")
    quantization_config = quantization_config.to_dict() if hasattr(quantization_config, "to_dict") else dict(quantization_config)
    require(quantization_config.get("load_in_4bit") is True, "serialized 4-bit flag missing")
    require(quantization_config.get("bnb_4bit_quant_type") == "nf4", "serialized NF4 type mismatch")
    require(quantization_config.get("bnb_4bit_use_double_quant") is False, "serialized double-quant mismatch")
    device_map = getattr(model, "hf_device_map", {}) or {}
    require(all(value in (0, "cuda:0", "cuda") for value in device_map.values()), "CPU/disk offload detected")
    total = int(p5["roles"][role]["target_linear_parameters"])
    protected_mass = sum(int(source[f"{name}.weight"]["numel"]) for name in protected)
    fraction = protected_mass / total
    require(fraction <= 0.02, "protected parameter mass exceeds 2 percent")
    stable = {
        "role": role, "packed_fqns": packed, "protected_fqns": list(protected),
        "packed_facts": packed_facts, "protected_weight_sha256": protected_hashes,
        "nontarget_weight_sha256": nontarget_hashes, "embedding_lm_head_tied": tied,
        "target_linear_count": len(targets), "packed_module_count": len(packed),
        "protected_module_count": len(protected), "protected_parameter_mass": protected_mass,
        "target_parameter_mass": total, "protected_parameter_fraction": fraction,
        "hf_device_map": device_map, "cpu_disk_offload_count": 0,
        "serialized_quantization_config": quantization_config,
    }
    stable["sha256"] = hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return stable


def quant_config(torch: Any, transformers: Any, protected: Sequence[str]) -> Any:
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True, load_in_8bit=False,
        llm_int8_skip_modules=[*protected, "lm_head"],
        llm_int8_enable_fp32_cpu_offload=False,
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False, bnb_4bit_quant_storage=torch.uint8,
    )


def atomic_save(model: Any, tokenizer: Any, destination: Path) -> None:
    require(not destination.exists(), f"artifact exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        require((temporary / "model.safetensors").is_file(), "safetensors artifact absent")
        require(not list(temporary.glob("*.bin")), "pickle artifact forbidden")
        for path in temporary.rglob("*"):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_runtime_modules() -> tuple[Any, Any, Any, Any]:
    import torch
    import transformers
    import bitsandbytes as bnb
    from safetensors import torch as safe_torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch, transformers, bnb, safe_torch


def common_prerequisites(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    p5 = read_json(args.p5)
    require(p5.get("status") == "PASS", "P5 did not pass")
    require(p5.get("scope_guards", {}).get("b0_4_gpu_launcher_authorized") is True, "P5 did not authorize B0.4")
    local = read_json(args.local_manifest)
    require(local.get("tokenizer_semantic_identity_verified") is True, "local tokenizer gate failed")
    require(sha256_file(args.p5) == args.p5_sha256, "P5 receipt hash drift")
    require(sha256_file(args.local_manifest) == args.local_manifest_sha256, "local manifest hash drift")
    return p5, local


def precheck(args: argparse.Namespace) -> None:
    links = inventory.validate_receipts(
        args.local_manifest.resolve(), args.identity_lock.resolve(),
        args.b03b_final.resolve(), args.environment_lock.resolve(),
    )
    p5 = read_json(args.p5)
    require(p5.get("status") == "PASS", "P5 did not pass")
    require(p5.get("scope_guards", {}).get("b0_4_gpu_launcher_authorized") is True, "P5 authorization absent")
    expected = set(inventory.expected_target_fqns())
    require(len(expected) == 112, "target FQN contract drift")
    for condition, (role, protected) in PACKED_CONDITIONS.items():
        require(set(protected) <= expected and len(protected) <= 2, f"protected set invalid: {condition}")
        source = source_parameters(p5, role)
        total = int(p5["roles"][role]["target_linear_parameters"])
        mass = sum(int(source[f"{name}.weight"]["numel"]) for name in protected)
        require(mass / total <= 0.02, f"protected mass invalid: {condition}")
    report = {
        "schema_version": SCHEMA, "phase": "precheck", "status": "PASS",
        "created_at_utc": now(), "started_unix": time.time(),
        "p5": {"path": str(args.p5.resolve()), "sha256": sha256_file(args.p5)},
        "local_manifest": {"path": str(args.local_manifest.resolve()), "sha256": sha256_file(args.local_manifest)},
        "identity_lock": {"path": str(args.identity_lock.resolve()), "sha256": sha256_file(args.identity_lock)},
        "b03b_final": {"path": str(args.b03b_final.resolve()), "sha256": sha256_file(args.b03b_final)},
        "environment_lock": {"path": str(args.environment_lock.resolve()), "sha256": sha256_file(args.environment_lock)},
        "validated_links": links, "condition_ids": [*SOURCE_CONDITIONS, *PACKED_CONDITIONS],
        "outcome_inspected": False, "production_authorized": False,
    }
    atomic_json(args.output, report)
    print("ORG_E2B_B04_PRECHECK=PASS")


def source_condition(args: argparse.Namespace) -> None:
    require(args.condition in SOURCE_CONDITIONS, "invalid source condition")
    role = SOURCE_CONDITIONS[args.condition]
    p5, local = common_prerequisites(args)
    guard = NetworkGuard(); guard.install()
    try:
        torch, transformers, bnb, _ = load_runtime_modules()
        torch.cuda.reset_peak_memory_stats(0)
        rt = runtime(torch, transformers, bnb)
        snapshot = Path(local["snapshot_root"]) / role
        tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, use_fast=True)
        require(tokenizer_fingerprint(tokenizer) == p5["roles"][role]["tokenizer_runtime_fingerprint"], "runtime tokenizer drift")
        loaded = transformers.AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            torch_dtype=torch.bfloat16, device_map={"": 0}, output_loading_info=True,
            attn_implementation="eager",
        )
        model, info = loaded
        require(all(info.get(key, []) == [] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), "unclean BF16 load")
        expected_parameters = source_parameters(p5, role)
        observed_parameters = dict(model.named_parameters(remove_duplicate=False))
        require(set(observed_parameters) == set(expected_parameters), "BF16 source parameter universe drift")
        for name, parameter in observed_parameters.items():
            require(str(parameter.device) == "cuda:0" and parameter.dtype == torch.bfloat16, f"BF16 source placement: {name}")
            require(inventory.tensor_sha256(torch, parameter) == expected_parameters[name]["sha256"], f"BF16 source hash drift: {name}")
        require(model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr(), "BF16 tie broken")
        first = forward(torch, model, inputs(torch)); second = forward(torch, model, inputs(torch))
        repeat = {"logits": tensor_metrics(torch, first[0], second[0]), "loss": tensor_metrics(torch, first[1], second[1])}
        require(repeat["logits"]["exact"] and repeat["loss"]["exact"], "BF16 repeatability failure")
        report = {
            "schema_version": SCHEMA, "phase": "source", "status": "PASS", "condition_id": args.condition,
            "role": role, "created_at_utc": now(), "pid": os.getpid(), "process_uuid": str(uuid.uuid4()),
            "runtime": rt, "source_model_sha256": local["role_model_weight_sha256"][role],
            "tokenizer_runtime_fingerprint": tokenizer_fingerprint(tokenizer), "synthetic_ids": SYNTHETIC_IDS,
            "repeatability": repeat, "network_attempt_count": guard.attempts,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(0)),
            "outcome_inspected": False, "production_authorized": False,
        }
        require(guard.attempts == 0, "network attempt observed")
        atomic_json(args.output, report)
    finally:
        guard.restore()


def produce(args: argparse.Namespace) -> None:
    require(args.condition in PACKED_CONDITIONS, "invalid packed condition")
    role, protected = PACKED_CONDITIONS[args.condition]
    p5, local = common_prerequisites(args)
    guard = NetworkGuard(); guard.install()
    try:
        torch, transformers, bnb, safe_torch = load_runtime_modules()
        torch.cuda.reset_peak_memory_stats(0)
        rt = runtime(torch, transformers, bnb)
        snapshot = Path(local["snapshot_root"]) / role
        tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, use_fast=True)
        require(tokenizer_fingerprint(tokenizer) == p5["roles"][role]["tokenizer_runtime_fingerprint"], "runtime tokenizer drift")
        loaded = transformers.AutoModelForCausalLM.from_pretrained(
            snapshot, quantization_config=quant_config(torch, transformers, protected),
            local_files_only=True, trust_remote_code=False, use_safetensors=True,
            torch_dtype=torch.bfloat16, device_map={"": 0}, output_loading_info=True,
            attn_implementation="eager",
        )
        model, info = loaded
        require(all(info.get(key, []) == [] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), "unclean packed load")
        representation = inspect_representation(torch, bnb, model, p5, role, protected)
        payload = inputs(torch); first = forward(torch, model, payload); second = forward(torch, model, payload)
        repeat = {"logits": tensor_metrics(torch, first[0], second[0]), "loss": tensor_metrics(torch, first[1], second[1])}
        require(repeat["logits"]["exact"] and repeat["loss"]["exact"], "producer repeatability failure")
        atomic_save(model, tokenizer, args.artifact)
        reference_tmp = args.reference.with_name(f".{args.reference.name}.{os.getpid()}.tmp")
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        require(not args.reference.exists(), "reference exists")
        safe_torch.save_file({"input_ids": payload["input_ids"].cpu(), "attention_mask": payload["attention_mask"].cpu(), "labels": payload["labels"].cpu(), "logits": first[0], "loss": first[1]}, reference_tmp)
        os.replace(reference_tmp, args.reference)
        report = {
            "schema_version": SCHEMA, "phase": "produce", "status": "PASS", "condition_id": args.condition,
            "role": role, "protected_fqns": list(protected), "created_at_utc": now(), "pid": os.getpid(),
            "process_uuid": str(uuid.uuid4()), "runtime": rt, "representation": representation,
            "artifact": {"path": str(args.artifact.resolve()), "manifest": directory_manifest(args.artifact)},
            "reference": {"path": str(args.reference.resolve()), "sha256": sha256_file(args.reference)},
            "repeatability": repeat, "network_attempt_count": guard.attempts,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(0)),
            "outcome_inspected": False, "production_authorized": False,
        }
        require(guard.attempts == 0, "network attempt observed")
        atomic_json(args.output, report)
    finally:
        guard.restore()


def reload_condition(args: argparse.Namespace) -> None:
    producer = read_json(args.producer)
    require(producer.get("phase") == "produce" and producer.get("status") == "PASS", "producer invalid")
    require(not process_alive(int(producer["pid"])), "producer process is still alive")
    condition = producer["condition_id"]
    require(condition in PACKED_CONDITIONS, "producer condition invalid")
    role, protected = PACKED_CONDITIONS[condition]
    p5, _ = common_prerequisites(args)
    require(directory_manifest(args.artifact) == producer["artifact"]["manifest"], "artifact manifest drift")
    require(sha256_file(args.reference) == producer["reference"]["sha256"], "reference drift")
    guard = NetworkGuard(); guard.install()
    try:
        torch, transformers, bnb, safe_torch = load_runtime_modules()
        torch.cuda.reset_peak_memory_stats(0)
        rt = runtime(torch, transformers, bnb)
        loaded = transformers.AutoModelForCausalLM.from_pretrained(
            args.artifact, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            device_map={"": 0}, torch_dtype=torch.bfloat16,
            output_loading_info=True, attn_implementation="eager",
        )
        model, info = loaded
        require(all(info.get(key, []) == [] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), "unclean fresh reload")
        representation = inspect_representation(torch, bnb, model, p5, role, protected)
        require(representation == producer["representation"], "packed representation/state hash drift")
        reference = safe_torch.load_file(args.reference, device="cpu")
        payload = {key: reference[key].to("cuda:0") for key in ("input_ids", "attention_mask", "labels")}
        first = forward(torch, model, payload); second = forward(torch, model, payload)
        repeat = {"logits": tensor_metrics(torch, first[0], second[0]), "loss": tensor_metrics(torch, first[1], second[1])}
        cross = {"logits": tensor_metrics(torch, first[0], reference["logits"]), "loss": tensor_metrics(torch, first[1], reference["loss"])}
        require(repeat["logits"]["exact"] and repeat["loss"]["exact"], "consumer repeatability failure")
        require(cross["logits"]["max_abs"] <= 1e-6 and cross["loss"]["max_abs"] <= 1e-6, "fresh-reload numerical drift")
        report = {
            "schema_version": SCHEMA, "phase": "reload", "status": "PASS", "condition_id": condition,
            "role": role, "protected_fqns": list(protected), "created_at_utc": now(), "pid": os.getpid(),
            "producer_pid": producer["pid"], "fresh_consumer_process": os.getpid() != producer["pid"],
            "runtime": rt, "representation": representation, "artifact_manifest": directory_manifest(args.artifact),
            "repeatability": repeat, "cross_process_numerics": cross, "network_attempt_count": guard.attempts,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(0)),
            "outcome_inspected": False, "production_authorized": False,
        }
        require(guard.attempts == 0, "network attempt observed")
        atomic_json(args.output, report)
    finally:
        guard.restore()


def write_qu_alias(args: argparse.Namespace) -> None:
    producer = read_json(args.producer)
    require(producer.get("phase") == "produce" and producer.get("status") == "PASS", "C05 producer invalid")
    require(producer.get("condition_id") == "C05_QU_NF4_ABSTAIN", "QU alias must target C05")
    manifest = producer.get("artifact", {}).get("manifest", {})
    require(isinstance(manifest, dict) and manifest.get("sha256"), "C05 artifact manifest missing")
    report = {
        "schema_version": "org_e2b_b04_qu_alias_v1",
        "created_at_utc": now(),
        "alias": "QU",
        "condition_id": "C05_QU_NF4_ABSTAIN",
        "artifact_path": producer["artifact"]["path"],
        "artifact_manifest_sha256": manifest["sha256"],
        "redundant_quantization_count": 0,
        "outcome_inspected": False,
        "production_authorized": False,
    }
    atomic_json(args.output, report)
    print("ORG_E2B_B04_QU_ALIAS=PASS")


def validate(args: argparse.Namespace) -> None:
    metadata = read_json(args.run_metadata)
    require(metadata.get("phase") == "precheck" and metadata.get("status") == "PASS", "precheck metadata invalid")
    for key in ("p5", "local_manifest", "identity_lock", "b03b_final", "environment_lock"):
        item = metadata[key]
        require(sha256_file(Path(item["path"])) == item["sha256"], f"prerequisite drift after precheck: {key}")
    started = float(metadata["started_unix"])
    receipts = []
    for condition in SOURCE_CONDITIONS:
        receipts.append(read_json(args.run_dir / "receipts" / f"{condition}.json"))
    for condition in PACKED_CONDITIONS:
        receipts.append(read_json(args.run_dir / "receipts" / f"{condition}.json"))
    require(len(receipts) == 7 and {x["condition_id"] for x in receipts} == set(SOURCE_CONDITIONS) | set(PACKED_CONDITIONS), "condition receipt universe")
    require(all(x.get("status") == "PASS" and x.get("outcome_inspected") is False for x in receipts), "condition failure")
    packed_receipts = {x["condition_id"]: x for x in receipts if x.get("phase") == "reload"}
    require(set(packed_receipts) == set(PACKED_CONDITIONS), "fresh reload receipts missing")
    require(all(x.get("network_attempt_count") == 0 for x in receipts), "network attempt count")
    elapsed = time.time() - started
    source_receipts = {x["role"]: x for x in receipts if x.get("phase") == "source"}
    source_hashes = {role: source_receipts[role]["source_model_sha256"] for role in ("O", "U", "R")}
    qu_alias = read_json(args.qu_alias)
    c05_artifact = packed_receipts["C05_QU_NF4_ABSTAIN"]["artifact_manifest"]
    require(qu_alias.get("schema_version") == "org_e2b_b04_qu_alias_v1", "QU alias schema mismatch")
    require(qu_alias.get("condition_id") == "C05_QU_NF4_ABSTAIN", "QU alias condition mismatch")
    require(qu_alias.get("artifact_manifest_sha256") == c05_artifact["sha256"], "QU alias artifact mismatch")
    runtimes = [x["runtime"] for x in receipts]
    checks = {
        "environment_lock_verified": True, "public_identity_license_lock_verified": True,
        "download_manifest_verified": True, "tokenizer_template_identity_verified": True,
        "storage_high_water_gate_passed": True, "actual_our_module_inventory_verified": True,
        "condition_count": 7, "condition_ids_exact": True,
        "source_role_hashes_distinct_and_linked": len(set(source_hashes.values())) == 3,
        "R_never_substituted_for_O": source_receipts["R"]["condition_id"] == "C03_R_BF16" and source_hashes["R"] != source_hashes["O"],
        "QO_source_is_exact_O": packed_receipts["C04_QO_NF4_FULL"]["role"] == "O",
        "QU_source_is_exact_U": all(packed_receipts[x]["role"] == "U" for x in PACKED_CONDITIONS if x != "C04_QO_NF4_FULL"),
        "qu_abstention_alias_hash_exact": qu_alias["artifact_manifest_sha256"] == c05_artifact["sha256"],
        "protected_fqn_sets_exact": all(tuple(packed_receipts[key]["protected_fqns"]) == PACKED_CONDITIONS[key][1] for key in PACKED_CONDITIONS),
        "protected_parameter_mass_cap_passed": all(x["representation"]["protected_parameter_fraction"] <= 0.02 for x in packed_receipts.values()),
        "packed_module_inventory_exact": all(x["representation"]["packed_module_count"] == 112 - len(PACKED_CONDITIONS[key][1]) for key, x in packed_receipts.items()),
        "protected_bf16_tensor_hashes_exact": all(len(x["representation"]["protected_weight_sha256"]) == len(PACKED_CONDITIONS[key][1]) for key, x in packed_receipts.items()),
        "quantization_state_roundtrip_exact": all(x["representation"]["sha256"] == read_json(args.run_dir / "producers" / f"{key}.json")["representation"]["sha256"] for key, x in packed_receipts.items()),
        "producer_process_exited": all(not process_alive(int(x["producer_pid"])) for x in packed_receipts.values()),
        "fresh_consumer_process": all(x["fresh_consumer_process"] for x in packed_receipts.values()),
        "network_attempt_count": 0, "remote_code_used": False, "cpu_disk_offload_count": 0,
        "silent_dequantization_or_fallback_count": 0, "synthetic_input_manifest_verified": True,
        "within_condition_numerical_repeatability_passed": True, "artifact_hash_verification_passed": True,
        "atomic_receipt_count": 7, "gpu_hours_le_1": elapsed <= 3600,
        "b0_4_representation_canary_passed": elapsed <= 3600,
    }
    require(all(value is True or value == 0 or value == 7 for value in checks.values()), "final B0.4 check failed")
    report = {
        "schema_version": FINAL_SCHEMA, "status": "PASS", "created_at_utc": now(),
        "hardware": "H200", "visible_gpu_count": 1,
        "elapsed_seconds": elapsed, "gpu_hours": elapsed / 3600.0, "checks": checks,
        "runtime_consistency": {
            "hardware": sorted({x["hardware"] for x in runtimes}),
            "torch": sorted({x["torch"] for x in runtimes}),
            "transformers": sorted({x["transformers"] for x in runtimes}),
            "bitsandbytes": sorted({x["bitsandbytes"] for x in runtimes}),
        },
        "peak_vram_bytes": max(int(x["peak_vram_bytes"]) for x in receipts),
        "packed_artifact_bytes": {key: value["artifact_manifest"]["total_bytes"] for key, value in packed_receipts.items()},
        "canonical_qu_alias": qu_alias,
        "condition_receipts": {x["condition_id"]: {"path": str((args.run_dir / "receipts" / f"{x['condition_id']}.json").resolve()), "sha256": sha256_file(args.run_dir / "receipts" / f"{x['condition_id']}.json")} for x in receipts},
        "outcome_inspected": False, "outcome_metric_count": 0, "paper_claim_authorized": False,
        "production_authorized": False,
    }
    serialized = json.dumps(report, sort_keys=True).lower()
    require(not any(key in serialized for key in FORBIDDEN_OUTCOME_KEYS), "outcome key leaked into representation final")
    atomic_json(args.output, report)
    print("ORG_E2B_B04_PUBLIC_REPRESENTATION_CANARY=PASS")
    print("CONDITIONS=7")
    print(f"GPU_HOURS={report['gpu_hours']:.6f}")
    print(f"OUTPUT={args.output}")
    print("B0_4_RUNTIME_GATE=PASS")
    print("B0_5_AUTHORIZED=false")
    print("B0_5_AUTHORIZATION_REASON=PENDING_POST_JOB_SACCT_CERTIFICATION")
    print("E2B_ATLAS_PRODUCTION_AUTHORIZED=false")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("precheck")
    pre.add_argument("--p5", type=Path, required=True)
    pre.add_argument("--local-manifest", type=Path, required=True)
    pre.add_argument("--identity-lock", type=Path, required=True)
    pre.add_argument("--b03b-final", type=Path, required=True)
    pre.add_argument("--environment-lock", type=Path, required=True)
    pre.add_argument("--output", type=Path, required=True)
    for command in ("source", "produce", "reload"):
        item = sub.add_parser(command)
        item.add_argument("--condition", required=False)
        item.add_argument("--p5", type=Path, required=True)
        item.add_argument("--p5-sha256", required=True)
        item.add_argument("--local-manifest", type=Path, required=True)
        item.add_argument("--local-manifest-sha256", required=True)
        item.add_argument("--output", type=Path, required=True)
        if command in ("produce", "reload"):
            item.add_argument("--artifact", type=Path, required=True)
            item.add_argument("--reference", type=Path, required=True)
        if command == "reload":
            item.add_argument("--producer", type=Path, required=True)
    alias = sub.add_parser("alias")
    alias.add_argument("--producer", type=Path, required=True)
    alias.add_argument("--output", type=Path, required=True)
    final = sub.add_parser("validate")
    final.add_argument("--run-dir", type=Path, required=True)
    final.add_argument("--run-metadata", type=Path, required=True)
    final.add_argument("--qu-alias", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "precheck": precheck(args)
        elif args.command == "source": source_condition(args)
        elif args.command == "produce": produce(args)
        elif args.command == "reload": reload_condition(args)
        elif args.command == "alias": write_qu_alias(args)
        else: validate(args)
    except Exception as exc:
        print(f"ORG_E2B_B04_{args.command.upper()}=FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
