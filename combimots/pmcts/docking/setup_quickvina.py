"""Clone/configure the QuickVina-GPU dependency for CombiMOTS docking."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

QUICKVINA_REPO_URL = "https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1.git"
QUICKVINA_DIRNAME = "Vina-GPU-2.1"
QUICKVINA_METHOD_DIR = Path(QUICKVINA_DIRNAME) / "QuickVina2-GPU-2.1"


def setup_quickvina(
    docking_path: Path | None = None,
    env_path: Path | None = None,
    repo_url: str = QUICKVINA_REPO_URL,
    clone_depth: int = 1,
    update_existing: bool = False,
    write_env: bool = True,
    patch_makefile: bool = True,
    boost_lib_path: Path | None = None,
    opencl_lib_path: Path | None = None,
    compile_source: bool = False,
) -> None:
    """Clone QuickVina-GPU and configure local paths."""

    root = _repo_root()
    docking_path = (docking_path or root / "combimots" / "pmcts" / "docking").expanduser().resolve()
    env_path = (env_path or root / ".env").expanduser().resolve()
    quickvina_root = docking_path / QUICKVINA_DIRNAME
    quickvina_method = docking_path / QUICKVINA_METHOD_DIR

    docking_path.mkdir(parents=True, exist_ok=True)
    _clone_or_update_quickvina(
        destination=quickvina_root,
        repo_url=repo_url,
        clone_depth=clone_depth,
        update_existing=update_existing,
    )

    if write_env:
        _write_env(env_path, docking_path)

    if patch_makefile:
        _install_makefile_template(
            template_path=docking_path / "Makefile",
            makefile_path=quickvina_method / "Makefile",
        )
        _patch_makefile(
            makefile_path=quickvina_method / "Makefile",
            boost_lib_path=boost_lib_path,
            opencl_lib_path=opencl_lib_path,
        )

    if compile_source:
        _compile_quickvina(quickvina_method)

    _print_summary(docking_path=docking_path, env_path=env_path, quickvina_method=quickvina_method)


def _clone_or_update_quickvina(
    destination: Path,
    repo_url: str,
    clone_depth: int,
    update_existing: bool,
) -> None:
    if destination.exists():
        if not (destination / ".git").exists():
            raise RuntimeError(f"QuickVina destination exists but is not a git clone: {destination}")
        if update_existing:
            print(f"Updating existing QuickVina clone: {destination}")
            subprocess.run(["git", "-C", str(destination), "pull", "--ff-only"], check=True)
        else:
            print(f"QuickVina clone already exists, leaving unchanged: {destination}")
        return

    print(f"Cloning QuickVina-GPU into {destination}")
    subprocess.run(
        ["git", "clone", "--depth", str(clone_depth), repo_url, str(destination)],
        check=True,
    )


def _write_env(env_path: Path, docking_path: Path) -> None:
    existing_lines = []
    if env_path.exists():
        existing_lines = env_path.read_text().splitlines()

    key = "COMBIMOTS_DOCKING_PATH"
    value = f"{key}={docking_path}"
    replaced = False
    new_lines = []
    for line in existing_lines:
        if line.startswith(f"{key}="):
            new_lines.append(value)
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append(value)

    env_path.write_text("\n".join(new_lines) + "\n")
    print(f"Wrote {key} to {env_path}")


def _install_makefile_template(template_path: Path, makefile_path: Path) -> None:
    """Replace QuickVina's native Makefile with the CombiMOTS template when available."""

    if not template_path.exists():
        print(f"CombiMOTS QuickVina Makefile template not found, keeping native Makefile: {template_path}")
        return

    makefile_path.write_text(template_path.read_text())
    print(f"Installed CombiMOTS QuickVina Makefile template: {makefile_path}")


def _patch_makefile(
    makefile_path: Path,
    boost_lib_path: Path | None,
    opencl_lib_path: Path | None,
) -> None:
    if not makefile_path.exists():
        print(f"QuickVina Makefile not found, skipping patch: {makefile_path}")
        return

    replacements = {}
    if boost_lib_path is not None:
        replacements["BOOST_LIB_PATH"] = str(boost_lib_path.expanduser().resolve())
    if opencl_lib_path is not None:
        replacements["OPENCL_LIB_PATH"] = str(opencl_lib_path.expanduser().resolve())

    if not replacements:
        print("QuickVina Makefile uses WORK_DIR=$(CURDIR) and CONDA_PREFIX defaults; no path patch needed")
        return

    lines = []
    for line in makefile_path.read_text().splitlines():
        key = _makefile_assignment_key(line)
        if key in replacements:
            lines.append(f"{key}={replacements[key]}")
        else:
            lines.append(line)

    makefile_path.write_text("\n".join(lines) + "\n")
    patched = ", ".join(replacements)
    print(f"Patched QuickVina Makefile fields: {patched}")


def _makefile_assignment_key(line: str) -> str | None:
    if "=" not in line:
        return None
    key = line.split("=", 1)[0].strip()
    return key.removesuffix("?").strip()


def _compile_quickvina(quickvina_method: Path) -> None:
    print("Compiling QuickVina-GPU with `make source`...")
    subprocess.run(["make", "source"], cwd=quickvina_method, check=True)


def _print_summary(docking_path: Path, env_path: Path, quickvina_method: Path) -> None:
    binary_path = quickvina_method / "QuickVina2-GPU-2-1"
    print("\nDocking setup summary")
    print(f"Docking path: {docking_path}")
    print(f"Environment file: {env_path}")
    print(f"QuickVina method directory: {quickvina_method}")
    print(f"Expected executable: {binary_path}")
    if binary_path.exists():
        print("QuickVina executable exists")
    else:
        print("QuickVina executable is missing; compile QuickVina-GPU before docking")
    print("Run `pmcts-validate-docking --target-pair gsk3b_jnk3` after installing the package.")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone/configure QuickVina-GPU for CombiMOTS docking")
    parser.add_argument("--docking-path", type=Path, default=None, help="Path containing receptors and QuickVina-GPU")
    parser.add_argument("--env-path", type=Path, default=None, help="Path to write .env")
    parser.add_argument("--repo-url", default=QUICKVINA_REPO_URL)
    parser.add_argument("--clone-depth", type=int, default=1)
    parser.add_argument("--update-existing", action="store_true", help="Run git pull if QuickVina already exists")
    parser.add_argument("--no-write-env", action="store_true", help="Do not write COMBIMOTS_DOCKING_PATH to .env")
    parser.add_argument("--no-patch-makefile", action="store_true", help="Do not install or patch the CombiMOTS QuickVina Makefile template")
    parser.add_argument("--boost-lib-path", type=Path, default=None, help="Optional Boost path for QuickVina Makefile")
    parser.add_argument("--opencl-lib-path", type=Path, default=None, help="Optional OpenCL/CUDA path for QuickVina Makefile")
    parser.add_argument("--compile-source", action="store_true", help="Run `make source` after cloning/configuring")
    args = parser.parse_args()

    setup_quickvina(
        docking_path=args.docking_path,
        env_path=args.env_path,
        repo_url=args.repo_url,
        clone_depth=args.clone_depth,
        update_existing=args.update_existing,
        write_env=not args.no_write_env,
        patch_makefile=not args.no_patch_makefile,
        boost_lib_path=args.boost_lib_path,
        opencl_lib_path=args.opencl_lib_path,
        compile_source=args.compile_source,
    )


if __name__ == "__main__":
    main()
