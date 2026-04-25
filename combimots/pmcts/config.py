"""Central configuration for CombiMOTS paths and target metadata."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DockingTarget:
    """Docking metadata for one target in a target pair."""

    task_id: str
    receptor_file: str
    center: tuple[float, float, float]


DOCKING_TARGETS: dict[str, tuple[DockingTarget, DockingTarget]] = {
    "gsk3b_jnk3": (
        DockingTarget("gsk3b", "6Y9S.pdbqt", (24.503, 9.183, 9.226)),
        DockingTarget("jnk3", "4WHZ.pdbqt", (4.327, 101.902, 141.338)),
    ),
    "dhodh_rorgt": (
        DockingTarget("dhodh", "6QU7.pdbqt", (33.359, -11.558, -22.820)),
        DockingTarget("rorgt", "5NTP.pdbqt", (18.003, 11.762, 20.391)),
    ),
    "egfr_met": (
        DockingTarget("egfr", "1M17.pdbqt", (22.014, 0.253, 52.79)),
        DockingTarget("met", "4MXC.pdbqt", (-9.384, 17.423, -28.886)),
    ),
    "pik3ca_mtor": (
        DockingTarget("pik3ca", "8V8I.pdbqt", (-19.947, -23.175, 10.569)),
        DockingTarget("mtor", "3FAP.pdbqt", (-8.630, 26.528, 36.52)),
    ),
}

SUPPORTED_TARGET_PAIRS = tuple(DOCKING_TARGETS)


def load_env_file(path: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from a .env file without overriding env vars."""

    env_path = path or _find_env_file()
    if env_path is None or not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_docking_path() -> Path:
    """Return docking root from COMBIMOTS_DOCKING_PATH or the package default."""

    load_env_file()
    configured = os.environ.get("COMBIMOTS_DOCKING_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parent / "docking").resolve()


def get_docking_tasks(
    target_pair: str,
    smiles_to_pdbqt: dict[str, Path],
    tmp_path: Path,
) -> list[dict[str, Any]]:
    """Build docking task dictionaries for a target pair."""

    if target_pair not in DOCKING_TARGETS:
        supported = ", ".join(SUPPORTED_TARGET_PAIRS)
        raise ValueError(f"Invalid target pair '{target_pair}'. Supported target pairs: {supported}")

    docking_path = get_docking_path()
    return [
        {
            "smiles_to_pdbqt": smiles_to_pdbqt,
            "receptor_path": str(docking_path / target.receptor_file),
            "task_id": target.task_id,
            "center": target.center,
            "tmp_path": tmp_path,
        }
        for target in DOCKING_TARGETS[target_pair]
    ]


def get_real_resource_path(filename: str) -> Path:
    """Return a filesystem path to a file in pmcts/resources/real."""

    return Path(resources.files("pmcts").joinpath("resources", "real", filename))


def get_target_pair_mapping_path(target_pair: str) -> Path:
    """Return the package resource path for a target-pair reaction mapping."""

    if target_pair not in SUPPORTED_TARGET_PAIRS:
        supported = ", ".join(SUPPORTED_TARGET_PAIRS)
        raise ValueError(f"Invalid target pair '{target_pair}'. Supported target pairs: {supported}")
    return get_real_resource_path(f"{target_pair}.pkl")


def _find_env_file() -> Path | None:
    """Find .env from current directory upward, then near this package."""

    for directory in (Path.cwd(), *Path.cwd().parents):
        candidate = directory / ".env"
        if candidate.exists():
            return candidate

    package_candidate = Path(__file__).resolve().parents[2] / ".env"
    if package_candidate.exists():
        return package_candidate
    return None
