#!/usr/bin/env python3
"""Symlink-aware live one-GPU kernel smoke for the ORG E2B B0 gate.

V2 preserves the V1 CUDA, package-origin, NF4-functional, and Linear4bit
checks.  It changes only interpreter identity: a virtual-environment launcher
may legitimately be a symlink to a base interpreter outside the environment.
The executable's lexical absolute path and ``sys.prefix`` must both be under
the expected environment root.  The resolved executable target and
``sys.base_prefix`` are recorded for audit but are not themselves required to
be inside that root.  Imported Torch and bitsandbytes files retain V1's
resolved-path containment checks.

As in V1, PASS is only a live runtime/kernel smoke.  It is not save/reload or
packed-artifact certification and cannot authorize E2B/E3 production.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import site
import socket
import sys
from typing import Any, Callable, Mapping

import probe_org_e2b_live_gpu_runtime_v1 as probe_v1


SCHEMA_VERSION = "org_e2b_live_gpu_runtime_probe_v2"
INTERPRETER_IDENTITY_POLICY = (
    "require lexical absolute sys.executable under expected environment root "
    "and canonical sys.prefix/sys.exec_prefix equality with that root; when "
    "the executable resolves outside, "
    "require a local pyvenv.cfg with system site packages disabled and an exact "
    "base-executable/proc-self-exe chain; retain resolved imported-package and "
    "bitsandbytes native-library containment"
)


def _lexical_absolute(path: str | os.PathLike[str] | None) -> str | None:
    """Return an absolute normalized path without following symlinks."""

    if not path:
        return None
    try:
        return os.path.abspath(os.path.normpath(os.fspath(path)))
    except (OSError, TypeError, ValueError):
        return None


def _lexical_path_is_within(
    path: str | os.PathLike[str] | None,
    root: str | os.PathLike[str] | None,
) -> bool:
    """Test lexical containment after normalization, without dereferencing."""

    absolute_path = _lexical_absolute(path)
    absolute_root = _lexical_absolute(root)
    if absolute_path is None or absolute_root is None:
        return False
    try:
        Path(absolute_path).relative_to(Path(absolute_root))
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _default_path_resolver(path: str) -> str:
    return str(Path(path).resolve())


def _default_proc_self_exe_provider() -> str | None:
    try:
        return os.readlink("/proc/self/exe")
    except (OSError, ValueError):
        return None


def _default_pyvenv_reader(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolved_path_record(
    path: str | os.PathLike[str] | None,
    resolver: Callable[[str], str],
) -> dict[str, Any]:
    lexical = _lexical_absolute(path)
    record: dict[str, Any] = {
        "reported": None if path is None else os.fspath(path),
        "lexical_absolute": lexical,
        "resolved": None,
        "resolve_error": None,
    }
    if lexical is None:
        return record
    try:
        record["resolved"] = str(resolver(lexical))
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        record["resolve_error"] = {
            "type": type(exc).__name__,
            "message": " ".join(str(exc).split())[:1200],
        }
    return record


def _resolved_records_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_value = left.get("resolved")
    right_value = right.get("resolved")
    return bool(
        left_value
        and right_value
        and os.path.normcase(str(left_value)) == os.path.normcase(str(right_value))
    )


def collect_python_path_hygiene(
    *,
    environ: Mapping[str, str],
    expected_environment_root: str,
    sys_path: list[str],
    enable_user_site: bool | None,
    user_site_paths: str | list[str] | tuple[str, ...] | None,
    user_base: str | None,
    path_resolver: Callable[[str], str] = _default_path_resolver,
) -> dict[str, Any]:
    """Reject explicit Python-path and user-site contamination vectors."""

    if isinstance(user_site_paths, str):
        normalized_user_sites = [user_site_paths]
    else:
        normalized_user_sites = list(user_site_paths or [])
    forbidden_roots = [
        value for value in [*normalized_user_sites, user_base] if value
    ]
    path_records = [
        _resolved_path_record(value, path_resolver)
        for value in sys_path
        if value
    ]
    contaminated: list[dict[str, str]] = []
    external_package_dirs: list[str] = []
    for entry in path_records:
        entry_resolved = entry.get("resolved")
        if not entry_resolved:
            continue
        path_parts = {part.lower() for part in Path(str(entry_resolved)).parts}
        if (
            path_parts.intersection({"site-packages", "dist-packages"})
            and not probe_v1._path_is_within(
                str(entry_resolved), expected_environment_root
            )
        ):
            external_package_dirs.append(str(entry_resolved))
        for forbidden in forbidden_roots:
            if probe_v1._path_is_within(str(entry_resolved), forbidden):
                contaminated.append(
                    {"entry": str(entry_resolved), "forbidden_root": forbidden}
                )
    checks = [
        {
            "name": "python_no_user_site_environment",
            "status": "PASS"
            if environ.get("PYTHONNOUSERSITE") == "1"
            else "FAIL",
            "detail": environ.get("PYTHONNOUSERSITE"),
        },
        {
            "name": "python_user_site_disabled",
            "status": "PASS" if enable_user_site is False else "FAIL",
            "detail": enable_user_site,
        },
        {
            "name": "pythonpath_unset",
            "status": "PASS" if not environ.get("PYTHONPATH") else "FAIL",
            "detail": environ.get("PYTHONPATH"),
        },
        {
            "name": "pythonhome_unset",
            "status": "PASS" if not environ.get("PYTHONHOME") else "FAIL",
            "detail": environ.get("PYTHONHOME"),
        },
        {
            "name": "python_sys_path_no_empty_entry",
            "status": "PASS" if all(value for value in sys_path) else "FAIL",
            "detail": {"entry_count": len(sys_path)},
        },
        {
            "name": "python_sys_path_user_site_absent",
            "status": "PASS" if not contaminated else "FAIL",
            "detail": {
                "contaminated_entries": contaminated,
                "user_site_paths": normalized_user_sites,
                "user_base": user_base,
                "scope": (
                    "Reject user-site roots; base-interpreter standard-library "
                    "and explicit probe-script paths are recorded but allowed."
                ),
            },
        },
        {
            "name": "python_sys_path_external_package_dirs_absent",
            "status": "PASS" if not external_package_dirs else "FAIL",
            "detail": {
                "external_package_dirs": external_package_dirs,
                "expected_environment_root": expected_environment_root,
                "policy": (
                    "Every site-packages or dist-packages sys.path entry must "
                    "resolve under the expected environment root."
                ),
            },
        },
    ]
    return {
        "sys_path": list(sys_path),
        "sys_path_records": path_records,
        "enable_user_site": enable_user_site,
        "user_site_paths": normalized_user_sites,
        "user_base": user_base,
        "checks": checks,
        "overall_status": (
            "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
        ),
    }


def collect_interpreter_identity(
    *,
    expected_environment_root: str,
    executable: str | None,
    prefix: str | None,
    exec_prefix: str | None,
    base_prefix: str | None,
    base_executable: str | None,
    proc_self_exe: str | None,
    path_resolver: Callable[[str], str] = _default_path_resolver,
    pyvenv_reader: Callable[[Path], str] = _default_pyvenv_reader,
) -> dict[str, Any]:
    """Collect and validate a venv/Conda interpreter identity chain."""

    root = _resolved_path_record(expected_environment_root, path_resolver)
    executable_record = _resolved_path_record(executable, path_resolver)
    prefix_record = _resolved_path_record(prefix, path_resolver)
    exec_prefix_record = _resolved_path_record(exec_prefix, path_resolver)
    base_prefix_record = _resolved_path_record(base_prefix, path_resolver)
    base_executable_record = _resolved_path_record(base_executable, path_resolver)
    proc_self_exe_record = _resolved_path_record(proc_self_exe, path_resolver)

    root_resolved = root.get("resolved")
    executable_resolved = executable_record.get("resolved")
    resolved_paths_available = bool(root_resolved and executable_resolved)
    executable_resolved_external = bool(
        resolved_paths_available
        and not _lexical_path_is_within(
            str(executable_resolved), str(root_resolved)
        )
    )

    pyvenv_path = Path(_lexical_absolute(expected_environment_root) or "") / "pyvenv.cfg"
    pyvenv: dict[str, Any] = {
        "required": executable_resolved_external,
        "path": str(pyvenv_path),
        "read": False,
        "sha256": None,
        "include_system_site_packages": None,
        "values": {},
        "error": None,
    }
    if executable_resolved_external:
        try:
            content = pyvenv_reader(pyvenv_path)
            values: dict[str, str] = {}
            for raw_line in content.splitlines():
                if "=" not in raw_line:
                    continue
                key, value = raw_line.split("=", 1)
                values[key.strip().lower()] = value.strip().lower()
            pyvenv.update(
                {
                    "read": True,
                    "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "include_system_site_packages": values.get(
                        "include-system-site-packages"
                    ),
                    "values": values,
                }
            )
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            pyvenv["error"] = {
                "type": type(exc).__name__,
                "message": " ".join(str(exc).split())[:1200],
            }

    base_prefix_external = bool(
        base_prefix_record.get("resolved")
        and root_resolved
        and not _lexical_path_is_within(
            str(base_prefix_record["resolved"]), str(root_resolved)
        )
    )
    checks = [
        {
            "name": "python_executable_environment_root",
            "status": "PASS"
            if _lexical_path_is_within(executable, expected_environment_root)
            else "FAIL",
            "detail": {
                "comparison": "lexical_absolute_without_symlink_dereference",
                "executable": executable_record,
                "expected_root": root,
                "resolved_target_is_informational_for_this_check": True,
            },
        },
        {
            "name": "python_prefix_environment_root",
            "status": "PASS"
            if _resolved_records_equal(prefix_record, root)
            else "FAIL",
            "detail": {
                "comparison": "canonical_resolved_equality",
                "sys_prefix": prefix_record,
                "expected_root": root,
            },
        },
        {
            "name": "python_exec_prefix_environment_root",
            "status": "PASS"
            if _resolved_records_equal(exec_prefix_record, root)
            else "FAIL",
            "detail": {
                "comparison": "canonical_resolved_equality",
                "sys_exec_prefix": exec_prefix_record,
                "expected_root": root,
            },
        },
        {
            "name": "python_executable_resolved_target_available",
            "status": "PASS" if resolved_paths_available else "FAIL",
            "detail": {
                "executable": executable_record,
                "expected_root": root,
            },
        },
        {
            "name": "python_external_pyvenv_cfg",
            "status": "PASS"
            if (not executable_resolved_external or pyvenv["read"])
            else "FAIL",
            "detail": pyvenv,
        },
        {
            "name": "python_external_system_site_packages_disabled",
            "status": "PASS"
            if (
                not executable_resolved_external
                or pyvenv["include_system_site_packages"] == "false"
            )
            else "FAIL",
            "detail": {
                "required": executable_resolved_external,
                "include_system_site_packages": pyvenv[
                    "include_system_site_packages"
                ],
            },
        },
        {
            "name": "python_external_base_prefix",
            "status": "PASS"
            if (not executable_resolved_external or base_prefix_external)
            else "FAIL",
            "detail": {
                "required": executable_resolved_external,
                "sys_base_prefix": base_prefix_record,
                "external_to_expected_root": base_prefix_external,
            },
        },
        {
            "name": "python_external_base_executable_match",
            "status": "PASS"
            if (
                not executable_resolved_external
                or _resolved_records_equal(
                    executable_record, base_executable_record
                )
            )
            else "FAIL",
            "detail": {
                "required": executable_resolved_external,
                "executable": executable_record,
                "sys_base_executable": base_executable_record,
            },
        },
        {
            "name": "python_external_proc_self_exe_match",
            "status": "PASS"
            if (
                not executable_resolved_external
                or _resolved_records_equal(executable_record, proc_self_exe_record)
            )
            else "FAIL",
            "detail": {
                "required": executable_resolved_external,
                "executable": executable_record,
                "proc_self_exe": proc_self_exe_record,
            },
        },
    ]
    return {
        "policy": INTERPRETER_IDENTITY_POLICY,
        "resolved_executable_external_to_environment": executable_resolved_external,
        "expected_environment_root": root,
        "sys_executable": executable_record,
        "sys_prefix": prefix_record,
        "sys_exec_prefix": exec_prefix_record,
        "sys_base_prefix": base_prefix_record,
        "sys_base_executable": base_executable_record,
        "proc_self_exe": proc_self_exe_record,
        "pyvenv_cfg": pyvenv,
        "checks": checks,
        "overall_status": (
            "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
        ),
    }


def _replace_interpreter_checks(
    report: dict[str, Any], checks: list[dict[str, Any]]
) -> None:
    """Replace V1's symlink-resolving check with the complete V2 chain."""

    replacement: list[dict[str, Any]] = []
    replaced = False
    for check in report["checks"]:
        if check.get("name") == "python_executable_environment_root":
            replacement.extend(checks)
            replaced = True
        else:
            replacement.append(check)
    if not replaced:
        replacement[1:1] = checks
    report["checks"] = replacement
    report["failures"] = [
        str(check["name"])
        for check in replacement
        if check.get("status") != "PASS"
    ]
    report["overall_status"] = "PASS" if not report["failures"] else "FAIL"


def collect_live_gpu_report(
    *,
    expected_machine: str,
    expected_gpu_name_regex: str,
    expected_compute_capability: tuple[int, int],
    expected_environment_root: str,
    require_slurm: bool,
    importer: Callable[[str], Any] = importlib.import_module,
    version_provider: Callable[[str], str] = importlib.metadata.version,
    environ: Mapping[str, str] | None = None,
    machine_provider: Callable[[], str] = platform.machine,
    hostname_provider: Callable[[], str] = socket.gethostname,
    executable_provider: Callable[[], str] = lambda: sys.executable,
    prefix_provider: Callable[[], str] = lambda: sys.prefix,
    exec_prefix_provider: Callable[[], str] = lambda: sys.exec_prefix,
    base_prefix_provider: Callable[[], str] = lambda: sys.base_prefix,
    base_executable_provider: Callable[[], str | None] = lambda: getattr(
        sys, "_base_executable", None
    ),
    proc_self_exe_provider: Callable[[], str | None] = _default_proc_self_exe_provider,
    sys_path_provider: Callable[[], list[str]] = lambda: list(sys.path),
    enable_user_site_provider: Callable[[], bool | None] = lambda: getattr(
        site, "ENABLE_USER_SITE", None
    ),
    user_site_provider: Callable[[], str | list[str] | None] = site.getusersitepackages,
    user_base_provider: Callable[[], str | None] = site.getuserbase,
    path_resolver: Callable[[str], str] = _default_path_resolver,
    pyvenv_reader: Callable[[Path], str] = _default_pyvenv_reader,
    now: Callable[[], datetime] | None = None,
    functional_runner: Callable[[Any, Any], dict[str, Any]] = probe_v1.run_nf4_functional_smoke,
    linear_runner: Callable[[Any, Any], dict[str, Any]] = probe_v1.run_linear4bit_forward_smoke,
) -> dict[str, Any]:
    """Run V1 kernel checks with a fail-closed symlink-aware identity gate."""

    interpreter_identity = collect_interpreter_identity(
        expected_environment_root=expected_environment_root,
        executable=executable_provider(),
        prefix=prefix_provider(),
        exec_prefix=exec_prefix_provider(),
        base_prefix=base_prefix_provider(),
        base_executable=base_executable_provider(),
        proc_self_exe=proc_self_exe_provider(),
        path_resolver=path_resolver,
        pyvenv_reader=pyvenv_reader,
    )
    env = dict(os.environ if environ is None else environ)
    path_hygiene = collect_python_path_hygiene(
        environ=env,
        expected_environment_root=expected_environment_root,
        sys_path=sys_path_provider(),
        enable_user_site=enable_user_site_provider(),
        user_site_paths=user_site_provider(),
        user_base=user_base_provider(),
        path_resolver=path_resolver,
    )
    executable = interpreter_identity["sys_executable"]["reported"]

    native_path: str | None = None
    native_preflight_error: dict[str, str] | None = None
    try:
        cextension = importer("bitsandbytes.cextension")
        native = getattr(cextension, "lib", None)
        candidates = (
            getattr(native, "_name", None),
            getattr(getattr(native, "_lib", None), "_name", None),
        )
        native_path = next((str(value) for value in candidates if value), None)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        native_preflight_error = {
            "type": type(exc).__name__,
            "message": " ".join(str(exc).split())[:1200],
        }
    native_path_ok = bool(
        native_path
        and probe_v1._path_is_within(native_path, expected_environment_root)
    )
    native_check = {
        "name": "bitsandbytes_native_library_environment_root",
        "status": "PASS" if native_path_ok else "FAIL",
        "detail": {
            "comparison": "resolved_path_containment",
            "library_path": native_path,
            "expected_root": expected_environment_root,
            "preflight_error": native_preflight_error,
        },
    }
    preflight_ok = (
        interpreter_identity["overall_status"] == "PASS"
        and path_hygiene["overall_status"] == "PASS"
        and native_path_ok
    )

    # V1 resolves its executable before checking containment, which rejects a
    # normal venv launcher symlink whose target is the base Python.  V2 owns the
    # stronger identity chain above.  A verified in-root sentinel lets V1 run
    # its unchanged downstream checks; an invalid preflight passes None so V1
    # fails before executing either kernel smoke.
    delegated_executable = (
        str(Path(expected_environment_root) / ".org_e2b_v2_identity_verified")
        if preflight_ok
        else None
    )
    report = probe_v1.collect_live_gpu_report(
        expected_machine=expected_machine,
        expected_gpu_name_regex=expected_gpu_name_regex,
        expected_compute_capability=expected_compute_capability,
        expected_environment_root=expected_environment_root,
        require_slurm=require_slurm,
        importer=importer,
        version_provider=version_provider,
        environ=environ,
        machine_provider=machine_provider,
        hostname_provider=hostname_provider,
        executable_provider=lambda: delegated_executable,
        now=now,
        functional_runner=functional_runner,
        linear_runner=linear_runner,
    )

    report["schema_version"] = SCHEMA_VERSION
    report["interpreter_identity_policy"] = INTERPRETER_IDENTITY_POLICY
    report["host"]["executable"] = executable
    report["host"]["interpreter_identity"] = interpreter_identity
    report["host"]["python_path_hygiene"] = path_hygiene
    report["expected"]["environment_root_identity"] = interpreter_identity[
        "expected_environment_root"
    ]
    _replace_interpreter_checks(
        report, interpreter_identity["checks"] + path_hygiene["checks"]
    )
    native_inserted = False
    expanded_checks: list[dict[str, Any]] = []
    for check in report["checks"]:
        expanded_checks.append(check)
        if check.get("name") == "bitsandbytes_compiled_with_cuda":
            expanded_checks.append(native_check)
            native_inserted = True
    if not native_inserted:
        expanded_checks.append(native_check)
    report["checks"] = expanded_checks
    report["failures"] = [
        str(check["name"])
        for check in expanded_checks
        if check.get("status") != "PASS"
    ]
    report["overall_status"] = "PASS" if not report["failures"] else "FAIL"
    report["limitations"].append(
        "Interpreter identity permits an in-environment launcher symlink to an "
        "external base interpreter only when sys.prefix is also in the expected "
        "environment; imported package paths remain resolved-path checked."
    )
    report["limitations"].append(
        "Python-path hygiene rejects PYTHONPATH, PYTHONHOME, enabled user-site, "
        "empty path entries, and user-site roots. Base standard-library and "
        "the explicit probe-script directory remain allowed and are recorded."
    )
    return report


write_json_atomic = probe_v1.write_json_atomic
parse_compute_capability = probe_v1.parse_compute_capability


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-machine", required=True)
    parser.add_argument("--expected-gpu-name-regex", required=True)
    parser.add_argument(
        "--expected-compute-capability",
        required=True,
        type=parse_compute_capability,
    )
    parser.add_argument("--expected-environment-root", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = collect_live_gpu_report(
        expected_machine=args.expected_machine,
        expected_gpu_name_regex=args.expected_gpu_name_regex,
        expected_compute_capability=args.expected_compute_capability,
        expected_environment_root=args.expected_environment_root,
        require_slurm=args.require_slurm,
    )
    script_path = Path(__file__).resolve()
    engine_path = Path(probe_v1.__file__).resolve()
    source_manifest_path = args.source_manifest.resolve()
    report["probe_script"] = {
        "path": str(script_path),
        "sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
    }
    report["probe_engine_v1"] = {
        "path": str(engine_path),
        "sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
        "role": "unchanged downstream runtime and kernel checks",
    }
    report["source_manifest"] = {
        "path": str(source_manifest_path),
        "sha256": hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
        "contents_verified_by_launcher": True,
    }
    try:
        write_json_atomic(report, args.output, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"ORG_E2B_LIVE_GPU_RUNTIME_PROBE_V2={report['overall_status']} "
        f"output={args.output.resolve()} save_reload_certified=false"
    )
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
