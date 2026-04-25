"""Validate local QuickVina-GPU docking setup without running docking."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from pmcts.config import DOCKING_TARGETS, SUPPORTED_TARGET_PAIRS, get_docking_path, load_env_file


def validate_docking_setup(
    docking_path: Path | None = None,
    target_pair: str | None = None,
) -> dict[str, list[str] | str]:
    """Validate expected docking files and external command availability."""

    load_env_file()
    root = (docking_path or get_docking_path()).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if not root.exists():
        errors.append(f"Docking path does not exist: {root}")
    elif not root.is_dir():
        errors.append(f"Docking path is not a directory: {root}")

    target_pairs = [target_pair] if target_pair else list(SUPPORTED_TARGET_PAIRS)
    for pair in target_pairs:
        if pair not in DOCKING_TARGETS:
            errors.append(f"Unknown target pair: {pair}")
            continue
        for target in DOCKING_TARGETS[pair]:
            receptor_path = root / target.receptor_file
            if not receptor_path.exists():
                errors.append(f"Missing receptor for {pair}/{target.task_id}: {receptor_path}")

    quickvina_dir = root / "Vina-GPU-2.1" / "QuickVina2-GPU-2.1"
    quickvina_binary = quickvina_dir / "QuickVina2-GPU-2-1"
    if not quickvina_dir.exists():
        errors.append(f"Missing QuickVina-GPU directory: {quickvina_dir}")
    elif not quickvina_binary.exists():
        errors.append(f"Missing QuickVina-GPU executable: {quickvina_binary}")
    elif not os.access(quickvina_binary, os.X_OK):
        warnings.append(f"QuickVina-GPU executable is not executable: {quickvina_binary}")

    makefile = quickvina_dir / "Makefile"
    if makefile.exists():
        text = makefile.read_text(errors="ignore")
        if "/home/" in text:
            warnings.append("QuickVina Makefile still appears to contain user-specific /home paths")

    for kernel_file in ("Kernel1_Opt.bin", "Kernel2_Opt.bin"):
        kernel_path = quickvina_dir / kernel_file
        if not kernel_path.exists():
            warnings.append(f"Missing precompiled QuickVina kernel file: {kernel_path}")

    if shutil.which("obabel") is None:
        errors.append("Open Babel executable 'obabel' was not found on PATH")

    clinfo = shutil.which("clinfo")
    if clinfo is None:
        warnings.append("OpenCL diagnostic executable 'clinfo' was not found on PATH")
    else:
        opencl_env = _opencl_env()
        result = subprocess.run([clinfo, "-l"], env=opencl_env, capture_output=True, text=True, check=False)
        opencl_output = (result.stdout + result.stderr).strip()
        if result.returncode != 0 or "Platform #" not in opencl_output:
            warnings.append(
                "No OpenCL platform detected by `clinfo -l`; QuickVina may fail with CL_PLATFORM_NOT_FOUND_KHR"
            )

    return {
        "docking_path": str(root),
        "errors": errors,
        "warnings": warnings,
    }


def _opencl_env() -> dict[str, str]:
    env = os.environ.copy()
    if "OCL_ICD_VENDORS" in env:
        return env

    vendor_dirs = [Path("/etc/OpenCL/vendors")]
    if env.get("CONDA_PREFIX"):
        vendor_dirs.append(Path(env["CONDA_PREFIX"]) / "etc" / "OpenCL" / "vendors")
    for vendor_dir in vendor_dirs:
        if vendor_dir.exists():
            env["OCL_ICD_VENDORS"] = str(vendor_dir)
            break
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CombiMOTS QuickVina-GPU setup")
    parser.add_argument("--docking-path", type=Path, default=None, help="Override COMBIMOTS_DOCKING_PATH")
    parser.add_argument("--target-pair", choices=SUPPORTED_TARGET_PAIRS, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    report = validate_docking_setup(docking_path=args.docking_path, target_pair=args.target_pair)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Docking path: {report['docking_path']}")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        if not report["errors"]:
            print("Docking setup validation passed")

    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
