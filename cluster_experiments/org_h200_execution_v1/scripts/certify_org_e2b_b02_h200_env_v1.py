#!/usr/bin/env python3
"""Install offline and certify the isolated ORG B0.2 H200 environment.

The script has three deliberately separate processes/modes:

* ``install`` verifies Stage A, creates a fresh venv with copied interpreter,
  installs only from the hash-locked wheelhouse, and freezes every env file;
* ``probe`` runs under that new interpreter on one H200 and exercises live CUDA,
  NF4, GPTQModel import/API/kernel discovery, and custom packed-RTN primitives;
* ``finalize`` re-hashes the unchanged environment and creates the immutable
  receipt/source lock.

Even a PASS authorizes only B0.3 backend representation canaries.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping


SCHEMA = "org_e2b_b02_h200_environment_certification_v1"
CONSUMER_LOCK_SCHEMA = "org_e2b_packed_h200_environment_lock_v1"
LOCK_LINE = re.compile(
    r"^(?P<name>[a-z0-9][a-z0-9.-]*)==(?P<version>\S+) "
    r"--hash=sha256:(?P<sha>[0-9a-f]{64})$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def refuse_environment_overlap(target: Path, protected: Path) -> None:
    target = target.resolve()
    protected = protected.resolve()
    if target == protected or path_is_within(target, protected) or path_is_within(protected, target):
        raise ValueError(f"target/protected environment overlap: {target} vs {protected}")


def parse_complete_lock(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = LOCK_LINE.fullmatch(line)
        if not match:
            raise ValueError(f"unlocked or malformed requirement at {path}:{number}: {line!r}")
        name = normalize_name(match.group("name"))
        if name in result:
            raise ValueError(f"duplicate requirement in complete lock: {name}")
        result[name] = {
            "version": match.group("version"),
            "sha256": match.group("sha"),
        }
    if not result:
        raise ValueError("complete requirements lock is empty")
    return result


def verify_stage_a(stage_a: Path, candidate_path: Path) -> dict[str, Any]:
    stage_a = stage_a.resolve()
    candidate_path = candidate_path.resolve()
    required = [
        "stage_a_receipt.json",
        "stage_a_source_lock.json",
        "stage_a_source_lock.json.sha256",
        "requirements.complete.lock",
        "wheel_manifest.json",
        "pip_resolution.json",
        "cuda_module_inventory.txt",
    ]
    for name in required:
        if not (stage_a / name).is_file():
            raise FileNotFoundError(f"Stage A artifact missing: {stage_a / name}")
    receipt = json.loads((stage_a / "stage_a_receipt.json").read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    required_cuda_family = str(
        candidate["platform_contract"]["required_cuda_toolkit_family"]
    )
    if receipt.get("status") != "RESOLVED_NOT_INSTALLED_NOT_CERTIFIED":
        raise ValueError("Stage A status is not the required fail-closed resolved state")
    if receipt.get("target_environment_created_or_modified") is not False:
        raise ValueError("Stage A claims it created or modified the target environment")
    cuda_inventory = receipt.get("cuda_module_inventory", {})
    selected_cuda_module = cuda_inventory.get("selected")
    if (
        receipt.get("authorization", {}).get("stage_b_h200_environment_smoke_allowed")
        is not True
        or cuda_inventory.get("status") != "PASS"
        or not isinstance(selected_cuda_module, str)
        or not re.fullmatch(
            rf"cuda/{re.escape(required_cuda_family)}(?:\.\d+)?",
            selected_cuda_module,
        )
    ):
        raise ValueError(
            "Stage A did not authorize Stage B with a frozen CUDA "
            f"{required_cuda_family} module"
        )
    if receipt.get("candidate", {}).get("sha256") != sha256_file(candidate_path):
        raise ValueError("candidate hash differs from Stage A receipt")
    source_python = Path(candidate["source_python"]).resolve()
    recorded_source = receipt.get("source_python", {})
    if (
        Path(str(recorded_source.get("path", ""))).resolve() != source_python
        or not source_python.is_file()
        or recorded_source.get("sha256") != sha256_file(source_python)
    ):
        raise ValueError("source interpreter drifted after Stage A")
    sidecar_fields = (stage_a / "stage_a_source_lock.json.sha256").read_text(
        encoding="utf-8"
    ).strip().split()
    if len(sidecar_fields) != 2 or sidecar_fields[0] != sha256_file(stage_a / "stage_a_source_lock.json"):
        raise ValueError("Stage A source-lock sidecar mismatch")
    source_lock = json.loads((stage_a / "stage_a_source_lock.json").read_text(encoding="utf-8"))
    relative = {
        "requirements.complete.lock",
        "wheel_manifest.json",
        "pip_resolution.json",
        "stage_a_receipt.json",
        "cuda_module_inventory.txt",
    }
    for name, expected in source_lock.get("files", {}).items():
        file_path = stage_a / name if name in relative else Path(name)
        if not file_path.is_file() or sha256_file(file_path) != expected:
            raise ValueError(f"Stage A source-lock verification failed: {name}")

    locked = parse_complete_lock(stage_a / "requirements.complete.lock")
    manifest = json.loads((stage_a / "wheel_manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", [])
    if len(artifacts) != len(locked):
        raise ValueError("wheel manifest and complete lock distribution counts differ")
    seen: set[str] = set()
    for artifact in artifacts:
        name = normalize_name(str(artifact["distribution"]))
        wheel = stage_a / "wheelhouse" / str(artifact["filename"])
        if name in seen or name not in locked:
            raise ValueError(f"duplicate/unlocked wheel manifest distribution: {name}")
        seen.add(name)
        if not wheel.is_file() or wheel.suffix != ".whl":
            raise ValueError(f"runtime wheel missing or non-wheel: {wheel}")
        actual = sha256_file(wheel)
        if actual != artifact["sha256"] or actual != locked[name]["sha256"]:
            raise ValueError(f"runtime wheel SHA-256 mismatch: {wheel.name}")
        if str(artifact["version"]) != locked[name]["version"]:
            raise ValueError(f"runtime wheel version mismatch: {name}")
    if seen != set(locked):
        raise ValueError(f"wheel manifest missing distributions: {sorted(set(locked) - seen)}")
    return {"receipt": receipt, "source_lock": source_lock, "locked": locked}


def isolated_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base or os.environ)
    for key in (
        "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONHOME", "PYTHONPATH",
        "BNB_CUDA_VERSION", "PIP_REQUIRE_VIRTUALENV",
    ):
        env.pop(key, None)
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
    )
    return env


def run_logged(command: list[str], *, cwd: Path, log: Path, env: Mapping[str, str]) -> None:
    with log.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("COMMAND_JSON=" + json.dumps(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        handle.write(f"\nRETURN_CODE={completed.returncode}\n")
    if completed.returncode:
        raise RuntimeError(f"command failed rc={completed.returncode}; inspect {log}")


def inventory_tree(root: Path) -> dict[str, Any]:
    root = root.resolve()
    files: list[dict[str, Any]] = []
    symlinks: list[dict[str, str]] = []
    total = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            resolved = path.resolve()
            if not path_is_within(resolved, root):
                raise ValueError(f"environment symlink escapes target root: {relative} -> {resolved}")
            symlinks.append({"path": relative, "target": os.readlink(path)})
        elif path.is_file():
            size = path.stat().st_size
            total += size
            files.append({"path": relative, "bytes": size, "sha256": sha256_file(path)})
    digest = hashlib.sha256(
        (json.dumps({"files": files, "symlinks": symlinks}, sort_keys=True) + "\n").encode()
    ).hexdigest()
    return {
        "root": str(root),
        "files": files,
        "symlinks": symlinks,
        "file_count": len(files),
        "symlink_count": len(symlinks),
        "total_bytes": total,
        "content_manifest_sha256": digest,
    }


def installed_versions(python: Path, env: Mapping[str, str]) -> dict[str, str]:
    snippet = (
        "import importlib.metadata,json,re\n"
        "out={}\n"
        "for d in importlib.metadata.distributions():\n"
        " n=re.sub(r'[-_.]+','-',d.metadata['Name']).lower()\n"
        " if n in out: raise RuntimeError('duplicate:'+n)\n"
        " out[n]=d.version\n"
        "print(json.dumps(out,sort_keys=True))\n"
    )
    completed = subprocess.run(
        [str(python), "-c", snippet],
        env=dict(env),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"installed-version inventory failed: {completed.stderr[:1000]}")
    return json.loads(completed.stdout)


def audit_environment_entrypoints(root: Path) -> dict[str, Any]:
    """Reject relocated or vanished Python shebangs/activation prefixes."""

    root = root.resolve()
    bin_dir = root / "bin"
    if not bin_dir.is_dir():
        raise FileNotFoundError(f"environment bin directory missing: {bin_dir}")
    python_shebangs: list[dict[str, str]] = []
    activation_files: list[str] = []
    forbidden_markers = (f".{root.name}.building-", f".{root.name}.tmp-")
    for path in sorted(item for item in bin_dir.iterdir() if item.is_file()):
        raw = path.read_bytes()
        if any(marker.encode() in raw for marker in forbidden_markers):
            raise ValueError(f"vanished staging prefix remains in environment entrypoint: {path}")
        first = raw.splitlines()[0] if raw else b""
        if first.startswith(b"#!"):
            shebang = first[2:].decode("utf-8", errors="strict").strip()
            executable = Path(shebang.split()[0]).resolve()
            if "python" in executable.name.lower():
                if not executable.is_file() or not path_is_within(executable, root):
                    raise ValueError(
                        f"Python console-script shebang escapes or is unavailable: {path}: {shebang}"
                    )
                python_shebangs.append(
                    {"path": str(path), "shebang": shebang, "interpreter": str(executable)}
                )
        if path.name in {"activate", "activate.csh", "activate.fish", "Activate.ps1"}:
            text = raw.decode("utf-8", errors="strict")
            embedded_roots = re.findall(r"/work/anon_user/\.runtime/envs/[A-Za-z0-9_.-]+", text)
            if any(Path(value).resolve() != root for value in embedded_roots):
                raise ValueError(f"activation script names a non-final environment prefix: {path}")
            activation_files.append(str(path))
    if not python_shebangs:
        raise ValueError("no contained Python console-script shebang was found")
    return {
        "status": "PASS",
        "python_shebang_count": len(python_shebangs),
        "python_shebangs": python_shebangs,
        "activation_files": activation_files,
        "vanished_staging_references": 0,
    }


def install_stage(args: argparse.Namespace) -> int:
    candidate_path = args.candidate.resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    stage_a = args.stage_a.resolve()
    verified = verify_stage_a(stage_a, candidate_path)
    target = Path(candidate["environment_root"]).resolve()
    protected = Path(candidate["protected_environment_root"]).resolve()
    refuse_environment_overlap(target, protected)
    if args.environment_root.resolve() != target:
        raise ValueError("--environment-root differs from the frozen candidate")
    if target.exists():
        raise FileExistsError(f"target environment already exists; refusing mutation: {target}")
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"B0.2 result directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, mode=0o700)
    logs = run_dir / "logs"
    logs.mkdir()
    source_python = Path(candidate["source_python"]).resolve()
    if not source_python.is_file():
        raise FileNotFoundError(f"source Python missing: {source_python}")
    env = isolated_env()
    target_claimed = False
    quarantined_target: Path | None = None
    quarantine_error: str | None = None
    try:
        # mkdir is the atomic claim. Build at the final prefix so pip-generated
        # shebangs and activation scripts can never retain a renamed temp path.
        target.mkdir(mode=0o700)
        target_claimed = True
        write_json(
            run_dir / "target_claim.json",
            {
                "target": str(target),
                "pid": os.getpid(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "claimed_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        run_logged(
            [str(source_python), "-m", "venv", "--copies", str(target)],
            cwd=run_dir,
            log=logs / "01_create_target_venv.log",
            env=env,
        )
        target_python = target / "bin" / "python"
        run_logged(
            [
                str(target_python), "-m", "pip", "install", "--no-index",
                "--find-links", str(stage_a / "wheelhouse"), "--require-hashes",
                "-r", str(stage_a / "requirements.complete.lock"),
            ],
            cwd=run_dir,
            log=logs / "02_offline_hash_locked_install.log",
            env={**env, "PIP_NO_INDEX": "1"},
        )
        run_logged(
            [str(target_python), "-m", "pip", "check"],
            cwd=run_dir,
            log=logs / "03_pip_check.log",
            env=env,
        )
        versions = installed_versions(target_python, env)
        expected_versions = {
            name: value["version"] for name, value in verified["locked"].items()
        }
        if versions != expected_versions:
            raise ValueError(
                "installed distribution set/version drift: "
                f"missing={sorted(set(expected_versions) - set(versions))} "
                f"extra={sorted(set(versions) - set(expected_versions))}"
            )
        prefix_probe = subprocess.run(
            [
                str(target_python), "-c",
                "import json,sys; print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix}))",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        prefix = json.loads(prefix_probe.stdout)
        if Path(prefix["executable"]).resolve() != target_python.resolve():
            raise ValueError("target interpreter executable escapes isolated root")
        if Path(prefix["prefix"]).resolve() != target:
            raise ValueError("target interpreter sys.prefix is not the isolated root")
        entrypoint_audit = audit_environment_entrypoints(target)
        tree = inventory_tree(target)
        environment_manifest = run_dir / "environment_file_manifest.json"
        write_json(environment_manifest, tree)
        install_receipt = {
            "schema_version": SCHEMA,
            "mode": "install",
            "status": "OFFLINE_INSTALL_COMPLETE_PENDING_H200_PROBE",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "environment_root": str(target),
            "protected_environment_root": str(protected),
            "protected_environment_mutated": False,
            "network_allowed_during_target_install": False,
            "complete_hash_lock_enforced": True,
            "source_python": str(source_python),
            "interpreter": prefix,
            "entrypoint_audit": entrypoint_audit,
            "locked_versions": expected_versions,
            "installed_versions": versions,
            "stage_a_source_lock_sha256": sha256_file(stage_a / "stage_a_source_lock.json"),
            "environment_file_manifest_sha256": sha256_file(environment_manifest),
            "environment_content_manifest_sha256": tree["content_manifest_sha256"],
            "authorization": {
                "h200_probe_required": True,
                "b02_environment_certified": False,
                "b03_backend_canaries_authorized": False,
                "production_authorized": False,
            },
        }
        write_json(run_dir / "install_receipt.json", install_receipt)
    except BaseException:
        if target_claimed and target.exists():
            marker = os.environ.get("SLURM_JOB_ID") or "noslurm"
            failed = target.with_name(f"{target.name}.FAILED-{marker}-{os.getpid()}")
            if failed.exists():
                quarantine_error = f"refused to overwrite prior failed environment: {failed}"
            else:
                try:
                    os.replace(target, failed)
                    quarantined_target = failed
                except OSError as exc:
                    quarantine_error = f"{type(exc).__name__}: {exc}"
        failure = {
            "schema_version": SCHEMA,
            "mode": "install",
            "status": "FAIL",
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "exception_type": sys.exc_info()[0].__name__ if sys.exc_info()[0] else None,
            "exception": str(sys.exc_info()[1])[:4000],
            "quarantined_target": str(quarantined_target) if quarantined_target else None,
            "quarantine_error": quarantine_error,
            "production_authorized": False,
        }
        write_json(run_dir / "INSTALL_FAILURE.json", failure)
        raise
    print("ORG_E2B_B02_OFFLINE_INSTALL=COMPLETE_PENDING_H200_PROBE")
    print(f"ENVIRONMENT_ROOT={target}")
    print(f"RUN_DIR={run_dir}")
    return 0


def module_path(module: Any) -> Path:
    raw = getattr(module, "__file__", None)
    if not raw:
        raise ValueError(f"module lacks __file__: {module!r}")
    return Path(raw).resolve()


def loaded_shared_objects() -> list[dict[str, Any]]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    paths: set[Path] = set()
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[-1].startswith("/") and ".so" in fields[-1]:
            path = Path(fields[-1].replace(" (deleted)", ""))
            if path.is_file():
                paths.add(path.resolve())
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(paths)
    ]


def executable_record(name: str) -> dict[str, Any]:
    raw = shutil.which(name)
    if not raw:
        return {"available": False, "path": None, "sha256": None, "version": None}
    path = Path(raw).resolve()
    completed = subprocess.run(
        [str(path), "--version"], capture_output=True, text=True, check=False
    )
    return {
        "available": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "version": " ".join((completed.stdout + " " + completed.stderr).split())[:1000],
    }


def nvidia_driver_version() -> str | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    completed = subprocess.run(
        [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    versions = sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
    return versions[0] if completed.returncode == 0 and len(versions) == 1 else None


def kernel_cache_inventory() -> dict[str, Any]:
    variables = (
        "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
        "GPTQMODEL_TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH",
    )
    roots: dict[str, str | None] = {}
    files: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for variable in variables:
        raw = os.environ.get(variable)
        roots[variable] = str(Path(raw).resolve()) if raw else None
        if not raw:
            continue
        root = Path(raw).resolve()
        if not root.is_dir():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(
                {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}
            )
    return {
        "roots": roots,
        "files": files,
        "file_count": len(files),
        "fingerprint": sha256_json(files),
        "all_entries_verified": True,
    }


def probe_stage(args: argparse.Namespace) -> int:
    candidate_path = args.candidate.resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    target = Path(candidate["environment_root"]).resolve()
    run_dir = args.run_dir.resolve()
    install_receipt_path = run_dir / "install_receipt.json"
    install_receipt = json.loads(install_receipt_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    checks: dict[str, Any] = {}

    def check(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}
        if not passed:
            failures.append(name)

    check("linux_x86_64", platform.system() == "Linux" and platform.machine() == "x86_64", platform.platform())
    check("slurm_job_present", bool(os.environ.get("SLURM_JOB_ID")), os.environ.get("SLURM_JOB_ID"))
    check("interpreter_executable_contained", path_is_within(Path(sys.executable), target), sys.executable)
    check("sys_prefix_exact", Path(sys.prefix).resolve() == target, sys.prefix)
    check("python_version_exact", platform.python_version() == candidate["python_version"], platform.python_version())
    check("python_no_user_site", os.environ.get("PYTHONNOUSERSITE") == "1", os.environ.get("PYTHONNOUSERSITE"))
    check("pythonpath_absent", not os.environ.get("PYTHONPATH"), os.environ.get("PYTHONPATH"))
    check("pythonhome_absent", not os.environ.get("PYTHONHOME"), os.environ.get("PYTHONHOME"))
    check(
        "cublas_workspace_deterministic",
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )

    locked = parse_complete_lock(args.stage_a.resolve() / "requirements.complete.lock")
    actual_versions: dict[str, str] = {}
    for name in locked:
        try:
            actual_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"missing_distribution:{name}")
    check(
        "installed_versions_exact",
        actual_versions == {name: row["version"] for name, row in locked.items()},
        actual_versions,
    )

    imported: dict[str, Any] = {}
    module_files: dict[str, str] = {}
    for name in (
        "torch", "transformers", "bitsandbytes", "accelerate", "safetensors",
        "gptqmodel", "triton",
    ):
        try:
            module = importlib.import_module(name)
            imported[name] = module
            path = module_path(module)
            module_files[name] = str(path)
            check(f"module_contained:{name}", path_is_within(path, target), str(path))
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            check(f"module_import:{name}", False, f"{type(exc).__name__}: {exc}"[:1000])

    torch = imported.get("torch")
    bnb = imported.get("bitsandbytes")
    gptq = imported.get("gptqmodel")
    gpu: dict[str, Any] = {}
    nf4: dict[str, Any] = {"executed": False}
    gptq_record: dict[str, Any] = {
        "package_imported": gptq is not None,
        "api_constructed": False,
        "kernel_class_imported": False,
        "kernel_executed": False,
        "note": "Actual GPTQ packing/inference is deliberately deferred to B0.3.",
    }
    rtn: dict[str, Any] = {"executed": False}
    torch_compile: dict[str, Any] = {"executed": False}

    stage_a_receipt = json.loads(
        (args.stage_a.resolve() / "stage_a_receipt.json").read_text(encoding="utf-8")
    )
    selected_cuda_module = stage_a_receipt["cuda_module_inventory"]["selected"]
    loaded_modules = [x for x in os.environ.get("LOADEDMODULES", "").split(":") if x]
    check(
        "frozen_cuda_module_loaded",
        selected_cuda_module in loaded_modules,
        {"selected": selected_cuda_module, "loaded": loaded_modules},
    )

    if torch is not None:
        cuda_available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if cuda_available else 0
        check("exactly_one_cuda_device", cuda_available and count == 1, count)
        if cuda_available and count == 1:
            properties = torch.cuda.get_device_properties(0)
            capability = tuple(int(x) for x in torch.cuda.get_device_capability(0))
            gpu = {
                "name": str(properties.name),
                "compute_capability": list(capability),
                "total_memory": int(properties.total_memory),
                "torch_compiled_cuda": str(torch.version.cuda),
                "driver_version": nvidia_driver_version(),
            }
            check("gpu_driver_version_present", bool(gpu["driver_version"]), gpu["driver_version"])
            required_cuda_family = str(
                candidate["platform_contract"]["required_cuda_toolkit_family"]
            )
            check(
                "torch_cuda_expected",
                str(torch.version.cuda).startswith(required_cuda_family),
                {
                    "expected": required_cuda_family,
                    "observed": str(torch.version.cuda),
                },
            )
            check("gpu_name_h200", re.search(candidate["platform_contract"]["gpu_name_regex"], gpu["name"]) is not None, gpu["name"])
            check("compute_capability_9_0", capability == (9, 0), capability)
            torch.manual_seed(20260906)
            torch.cuda.manual_seed_all(20260906)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            try:
                fn = torch.compile(lambda x: torch.sin(x.float()).square() + 1.0, fullgraph=True)
                value = fn(torch.linspace(-1, 1, 4096, device="cuda", dtype=torch.float32))
                torch.cuda.synchronize()
                torch_compile = {
                    "executed": True,
                    "finite": bool(torch.isfinite(value).all().item()),
                    "sum": float(value.sum().item()),
                }
                check("torch_compile_cuda_kernel", torch_compile["finite"], torch_compile)
            except Exception as exc:
                check("torch_compile_cuda_kernel", False, f"{type(exc).__name__}: {exc}"[:1000])
            try:
                weight = torch.linspace(-2.0, 2.0, 128 * 64, device="cuda", dtype=torch.float32).reshape(128, 64)
                scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
                packed = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
                restored = packed.float() * scale
                inputs = torch.linspace(-1.0, 1.0, 3 * 64, device="cuda", dtype=torch.float32).reshape(3, 64)
                output = inputs @ restored.t()
                torch.cuda.synchronize()
                rtn = {
                    "executed": True,
                    "packed_dtype": str(packed.dtype),
                    "packed_bytes": int(packed.numel() * packed.element_size()),
                    "bf16_bytes": int(weight.numel() * 2),
                    "finite": bool(torch.isfinite(output).all().item()),
                    "nondegenerate": float(output.abs().max().item()) > 1e-8,
                }
                check(
                    "rtn_int8_cuda_primitives",
                    rtn["packed_dtype"] == "torch.int8" and rtn["packed_bytes"] < rtn["bf16_bytes"] and rtn["finite"] and rtn["nondegenerate"],
                    rtn,
                )
            except Exception as exc:
                check("rtn_int8_cuda_primitives", False, f"{type(exc).__name__}: {exc}"[:1000])

    if torch is not None and bnb is not None and bool(torch.cuda.is_available()):
        try:
            source = torch.linspace(-1.5, 1.5, 4096, device="cuda", dtype=torch.float32).reshape(64, 64)
            packed, state = bnb.functional.quantize_4bit(
                source, blocksize=64, quant_type="nf4", compress_statistics=True,
                quant_storage=torch.uint8,
            )
            restored = bnb.functional.dequantize_4bit(packed, quant_state=state)
            linear = bnb.nn.Linear4bit(
                64, 32, bias=False, compute_dtype=torch.float32,
                compress_statistics=True, quant_type="nf4", quant_storage=torch.uint8,
            )
            with torch.no_grad():
                linear.weight.copy_(source[:32].cpu())
            linear = linear.cuda()
            sample = torch.linspace(-1, 1, 128, device="cuda", dtype=torch.float32).reshape(2, 64)
            output = linear(sample)
            torch.cuda.synchronize()
            native = importlib.import_module("bitsandbytes.cextension").lib
            native_candidates = (
                getattr(native, "_name", None),
                getattr(getattr(native, "_lib", None), "_name", None),
            )
            native_path = next(
                (Path(value).resolve() for value in native_candidates if value and Path(value).is_file()),
                None,
            )
            if native_path is None:
                raise ValueError("bitsandbytes native library path is unavailable")
            compiled_with_cuda = bool(getattr(native, "compiled_with_cuda", False))
            nf4 = {
                "executed": True,
                "packed_dtype": str(packed.dtype),
                "packed_bytes": int(packed.numel() * packed.element_size()),
                "source_bytes": int(source.numel() * source.element_size()),
                "finite_dequant": bool(torch.isfinite(restored).all().item()),
                "finite_linear": bool(torch.isfinite(output).all().item()),
                "linear_weight_dtype": str(linear.weight.data.dtype),
                "native_library": str(native_path),
                "native_library_contained": path_is_within(native_path, target),
                "native_library_sha256": sha256_file(native_path),
                "compiled_with_cuda": compiled_with_cuda,
            }
            check(
                "nf4_live_cuda_kernel",
                nf4["packed_dtype"] == "torch.uint8" and nf4["packed_bytes"] < nf4["source_bytes"] and nf4["finite_dequant"] and nf4["finite_linear"] and nf4["linear_weight_dtype"] == "torch.uint8" and nf4["native_library_contained"] and nf4["compiled_with_cuda"],
                nf4,
            )
        except Exception as exc:
            check("nf4_live_cuda_kernel", False, f"{type(exc).__name__}: {exc}"[:1000])

    if gptq is not None:
        try:
            GPTQModel = getattr(gptq, "GPTQModel")
            QuantizeConfig = getattr(gptq, "QuantizeConfig")
            signature = inspect.signature(QuantizeConfig)
            if "dynamic" not in signature.parameters:
                raise ValueError("QuantizeConfig lacks required dynamic exclusion parameter")
            config = QuantizeConfig(
                bits=4,
                group_size=128,
                desc_act=False,
                sym=True,
                dynamic={"-:model.layers.0.self_attn.q_proj": {}},
            )
            kernel_module = importlib.import_module("gptqmodel.nn_modules.qlinear.torch")
            kernel_class = getattr(kernel_module, "TorchQuantLinear")
            kernel_path = module_path(kernel_module)
            gptq_record.update(
                {
                    "api_constructed": config is not None and GPTQModel is not None,
                    "dynamic_parameter_present": True,
                    "kernel_class_imported": inspect.isclass(kernel_class),
                    "kernel_class": f"{kernel_class.__module__}.{kernel_class.__qualname__}",
                    "kernel_module_path": str(kernel_path),
                    "kernel_module_sha256": sha256_file(kernel_path),
                }
            )
            check(
                "gptqmodel_api_and_kernel_surface",
                gptq_record["api_constructed"] and gptq_record["kernel_class_imported"] and path_is_within(kernel_path, target),
                gptq_record,
            )
        except Exception as exc:
            check("gptqmodel_api_and_kernel_surface", False, f"{type(exc).__name__}: {exc}"[:1000])

    kernel_artifacts = kernel_cache_inventory()
    check(
        "compiled_kernel_artifacts_present",
        kernel_artifacts["file_count"] > 0,
        {"file_count": kernel_artifacts["file_count"], "fingerprint": kernel_artifacts["fingerprint"]},
    )
    critical = (
        "linux_x86_64", "slurm_job_present", "interpreter_executable_contained",
        "sys_prefix_exact", "python_version_exact", "python_no_user_site",
        "pythonpath_absent", "pythonhome_absent", "cublas_workspace_deterministic",
        "installed_versions_exact",
        "frozen_cuda_module_loaded", "exactly_one_cuda_device", "gpu_name_h200",
        "compute_capability_9_0", "torch_cuda_expected", "gpu_driver_version_present",
        "torch_compile_cuda_kernel", "nf4_live_cuda_kernel",
        "gptqmodel_api_and_kernel_surface", "rtn_int8_cuda_primitives",
        "compiled_kernel_artifacts_present",
    )
    toolchain = {name: executable_record(name) for name in ("gcc", "g++", "ld", "nvcc", "ninja", "nvidia-smi")}
    nvcc_version = str(toolchain["nvcc"].get("version") or "")
    required_cuda_family = str(
        candidate["platform_contract"]["required_cuda_toolkit_family"]
    )
    check(
        "nvcc_cuda_expected",
        toolchain["nvcc"]["available"]
        and f"release {required_cuda_family}" in nvcc_version,
        {"expected": required_cuda_family, "observed": nvcc_version},
    )
    check(
        "compiler_toolchain_complete",
        all(toolchain[name]["available"] and toolchain[name]["sha256"] for name in ("gcc", "g++", "ld", "ninja")),
        {name: toolchain[name] for name in ("gcc", "g++", "ld", "ninja")},
    )
    passed = (
        all(checks.get(name, {}).get("passed") is True for name in critical)
        and checks["nvcc_cuda_expected"]["passed"]
        and checks["compiler_toolchain_complete"]["passed"]
        and not failures
    )
    compiler_payload = {name: toolchain[name] for name in ("gcc", "g++", "ld", "ninja")}
    cuda_payload = {
        "module": selected_cuda_module,
        "nvcc": toolchain["nvcc"],
        "nvidia_smi": toolchain["nvidia-smi"],
        "gpu": gpu,
    }
    report = {
        "schema_version": SCHEMA,
        "mode": "probe",
        "status": "PASS" if passed else "FAIL",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "environment_root": str(target),
        "checks": checks,
        "failures": failures,
        "gpu": gpu,
        "module_files": module_files,
        "nf4": nf4,
        "gptq_int4": gptq_record,
        "rtn_int8": rtn,
        "torch_compile": torch_compile,
        "loaded_shared_objects": loaded_shared_objects(),
        "toolchain": toolchain,
        "compiler_fingerprint": sha256_json(compiler_payload),
        "cuda_fingerprint": sha256_json(cuda_payload),
        "kernel_artifacts": kernel_artifacts,
        "scope": {
            "environment_runtime_certification_only": True,
            "gptq_actual_packing_or_inference_executed": False,
            "model_checkpoint_tested": False,
            "mixed_protected_modules_tested": False,
            "fresh_process_model_reload_tested": False,
        },
        "authorization": {
            "b03_nf4_canary_allowed": passed,
            "b03_gptq_canary_allowed": passed,
            "b03_rtn_canary_allowed": passed,
            "backend_representation_certified": False,
            "production_authorized": False,
            "scientific_claim_authorized": False,
        },
    }
    write_json(args.report.resolve(), report)
    print(f"ORG_E2B_B02_H200_PROBE={'PASS' if passed else 'FAIL'}")
    print(f"REPORT={args.report.resolve()}")
    return 0 if passed else 1


def finalize_stage(args: argparse.Namespace) -> int:
    candidate_path = args.candidate.resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    stage_a = args.stage_a.resolve()
    verified_stage_a = verify_stage_a(stage_a, candidate_path)
    run_dir = args.run_dir.resolve()
    target = Path(candidate["environment_root"]).resolve()
    install_receipt_path = run_dir / "install_receipt.json"
    probe_receipt_path = run_dir / "h200_probe.json"
    install_receipt = json.loads(install_receipt_path.read_text(encoding="utf-8"))
    probe_receipt = json.loads(probe_receipt_path.read_text(encoding="utf-8"))
    if install_receipt.get("status") != "OFFLINE_INSTALL_COMPLETE_PENDING_H200_PROBE":
        raise ValueError("install receipt is not eligible for finalization")
    if probe_receipt.get("status") != "PASS":
        raise ValueError("H200 probe did not pass")
    for artifact in probe_receipt.get("kernel_artifacts", {}).get("files", []):
        artifact_path = Path(str(artifact["path"])).resolve()
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError(f"compiled kernel artifact drifted after probe: {artifact_path}")
    before = json.loads((run_dir / "environment_file_manifest.json").read_text(encoding="utf-8"))
    after = inventory_tree(target)
    if before["content_manifest_sha256"] != after["content_manifest_sha256"]:
        raise ValueError("isolated environment content changed between install and finalization")
    write_json(run_dir / "environment_file_manifest_postprobe.json", after)
    launcher = args.launcher.resolve()
    script = Path(__file__).resolve()
    source_lock = {
        "schema_version": "org_e2b_b02_h200_environment_source_lock_v1",
        "status": "PASS",
        "environment_root": str(target),
        "candidate_sha256": sha256_file(candidate_path),
        "stage_a_source_lock_sha256": sha256_file(stage_a / "stage_a_source_lock.json"),
        "complete_requirements_lock_sha256": sha256_file(stage_a / "requirements.complete.lock"),
        "wheel_manifest_sha256": sha256_file(stage_a / "wheel_manifest.json"),
        "environment_content_manifest_sha256": after["content_manifest_sha256"],
        "files": {
            str(candidate_path): sha256_file(candidate_path),
            str(script): sha256_file(script),
            str(launcher): sha256_file(launcher),
            str(run_dir / "target_claim.json"): sha256_file(run_dir / "target_claim.json"),
            str(install_receipt_path): sha256_file(install_receipt_path),
            str(probe_receipt_path): sha256_file(probe_receipt_path),
            str(run_dir / "environment_file_manifest.json"): sha256_file(run_dir / "environment_file_manifest.json"),
            str(run_dir / "environment_file_manifest_postprobe.json"): sha256_file(run_dir / "environment_file_manifest_postprobe.json"),
        },
    }
    source_lock_path = run_dir / "B02_H200_ENVIRONMENT_SOURCE_LOCK_V1.json"
    locked = parse_complete_lock(stage_a / "requirements.complete.lock")
    required_package_names = (
        "torch", "bitsandbytes", "transformers", "accelerate", "safetensors",
        "gptqmodel", "triton",
    )
    package_records: dict[str, dict[str, Any]] = {}
    for name in required_package_names:
        module_file = Path(probe_receipt["module_files"][name]).resolve()
        if not module_file.is_file() or not path_is_within(module_file, target):
            raise ValueError(f"certified package module escaped environment: {name}")
        package_records[name] = {
            "version": locked[name]["version"],
            "module_path": str(module_file),
            "module_sha256": sha256_file(module_file),
        }
    python_executable = target / "bin" / "python"
    if not python_executable.is_file() or not path_is_within(python_executable, target):
        raise ValueError("isolated interpreter is missing or outside environment root")
    nf4 = probe_receipt["nf4"]
    if not nf4.get("compiled_with_cuda") or not nf4.get("native_library_contained"):
        raise ValueError("bitsandbytes CUDA native library was not certified")
    gptq_kernel_path = Path(probe_receipt["gptq_int4"]["kernel_module_path"]).resolve()
    if (
        not gptq_kernel_path.is_file()
        or not path_is_within(gptq_kernel_path, target)
        or sha256_file(gptq_kernel_path)
        != probe_receipt["gptq_int4"]["kernel_module_sha256"]
    ):
        raise ValueError("GPTQModel kernel module drifted after probe")
    write_json(source_lock_path, source_lock)
    environment_lock = {
        "schema": CONSUMER_LOCK_SCHEMA,
        "schema_version": CONSUMER_LOCK_SCHEMA,
        "overall_status": "PASS",
        "failures": [],
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "environment_root": str(target),
        "interpreter_identity": {
            "overall_status": "PASS",
            "executable": str(python_executable.resolve()),
            "executable_sha256": sha256_file(python_executable.resolve()),
            "sys_prefix": str(target),
            "base_prefix": install_receipt["interpreter"]["base_prefix"],
            "version": candidate["python_version"],
            "contained_in_environment": True,
            "source_interpreter": verified_stage_a["receipt"]["source_python"],
            "user_site_disabled": True,
            "pythonpath_absent": True,
            "pythonhome_absent": True,
        },
        "environment_tree_manifest": {
            "path": str((run_dir / "environment_file_manifest_postprobe.json").resolve()),
            "sha256": sha256_file(run_dir / "environment_file_manifest_postprobe.json"),
            "content_manifest_sha256": after["content_manifest_sha256"],
            "all_entries_verified": True,
        },
        "packages": package_records,
        "bitsandbytes_native": {
            "compiled_with_cuda": True,
            "library_path": nf4["native_library"],
            "library_sha256": nf4["native_library_sha256"],
        },
        "wheel_artifacts": {
            "manifest_path": str((stage_a / "wheel_manifest.json").resolve()),
            "manifest_sha256": sha256_file(stage_a / "wheel_manifest.json"),
            "complete_lock_path": str((stage_a / "requirements.complete.lock").resolve()),
            "complete_lock_sha256": sha256_file(stage_a / "requirements.complete.lock"),
            "all_entries_verified": True,
        },
        "kernel_artifacts": probe_receipt["kernel_artifacts"],
        "gptqmodel_kernel": {
            "class": probe_receipt["gptq_int4"]["kernel_class"],
            "module_path": probe_receipt["gptq_int4"]["kernel_module_path"],
            "module_sha256": probe_receipt["gptq_int4"]["kernel_module_sha256"],
            "actual_model_pack_deferred_to_b03": True,
        },
        "toolchain": {
            "cuda_version": probe_receipt["gpu"]["torch_compiled_cuda"],
            "driver_version": probe_receipt["gpu"]["driver_version"],
            "cuda_module": probe_receipt["checks"]["frozen_cuda_module_loaded"]["detail"]["selected"],
            "compiler_fingerprint": probe_receipt["compiler_fingerprint"],
            "cuda_fingerprint": probe_receipt["cuda_fingerprint"],
            "executables": probe_receipt["toolchain"],
        },
        "source_manifest": {
            "path": str(source_lock_path.resolve()),
            "sha256": sha256_file(source_lock_path),
            "all_entries_verified": True,
        },
        "builder": {
            "path": str(script),
            "sha256": sha256_file(script),
        },
        "quantizer_roles": {
            "nf4": "coprimary",
            "gptq_int4": "coprimary",
            "rtn_int8": "separate_custom_packed_control",
        },
        "scope_guards": {
            "b03_canary_authorized": True,
            "backend_representation_certified": False,
            "model_roundtrip_certified": False,
            "atlas_production_authorized": False,
            "production_authorized": False,
            "scientific_claim_authorized": False,
        },
    }
    environment_lock_path = run_dir / "ORG_E2B_PACKED_H200_ENVIRONMENT_LOCK_V1.json"
    write_json(environment_lock_path, environment_lock)
    final = {
        "schema_version": SCHEMA,
        "mode": "final",
        "status": "PASS",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "environment_root": str(target),
        "environment_content_manifest_sha256": after["content_manifest_sha256"],
        "environment_source_lock_sha256": sha256_file(source_lock_path),
        "consumer_environment_lock": str(environment_lock_path),
        "consumer_environment_lock_sha256": sha256_file(environment_lock_path),
        "stage_a_source_lock_sha256": sha256_file(stage_a / "stage_a_source_lock.json"),
        "quantizer_scope": {
            "nf4": "coprimary_candidate_retained",
            "gptq_int4": "coprimary_new_packed_intervention_candidate_retained",
            "rtn_int8": "custom_packed_negative_control_candidate_retained",
        },
        "certification": {
            "isolated_environment_hash_locked": True,
            "protected_historical_environment_untouched": True,
            "offline_install_verified": True,
            "interpreter_and_packages_contained": True,
            "h200_live_import_and_kernel_smoke": True,
            "b02_environment_certified_for_b03": True,
            "backend_representation_certified": False,
            "model_level_roundtrip_certified": False,
            "production_authorized": False,
            "scientific_claim_authorized": False,
        },
    }
    final_path = run_dir / "final.json"
    write_json(final_path, final)
    evidence_paths = [
        candidate_path, script, launcher,
        stage_a / "stage_a_source_lock.json",
        stage_a / "requirements.complete.lock",
        stage_a / "wheel_manifest.json",
        run_dir / "target_claim.json",
        install_receipt_path,
        probe_receipt_path,
        run_dir / "environment_file_manifest.json",
        run_dir / "environment_file_manifest_postprobe.json",
        source_lock_path,
        environment_lock_path,
        final_path,
    ]
    write_text(
        run_dir / "evidence_sha256.txt",
        "\n".join(f"{sha256_file(path)}  {path}" for path in evidence_paths) + "\n",
    )
    print("ORG_E2B_B02_H200_ENVIRONMENT=PASS")
    print(f"FINAL={final_path}")
    print("B03_BACKEND_CANARIES_AUTHORIZED=true")
    print("E2B_ATLAS_PRODUCTION_AUTHORIZED=false")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--candidate", type=Path, required=True)
    common.add_argument("--stage-a", type=Path, required=True)
    common.add_argument("--run-dir", type=Path, required=True)
    install = sub.add_parser("install", parents=[common])
    install.add_argument("--environment-root", type=Path, required=True)
    probe = sub.add_parser("probe", parents=[common])
    probe.add_argument("--report", type=Path, required=True)
    finalize = sub.add_parser("finalize", parents=[common])
    finalize.add_argument("--launcher", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.mode == "install":
            return install_stage(args)
        if args.mode == "probe":
            return probe_stage(args)
        if args.mode == "finalize":
            return finalize_stage(args)
        raise ValueError(f"unsupported mode: {args.mode}")
    except Exception as exc:
        print(f"ORG_E2B_B02_{args.mode.upper()}=FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
