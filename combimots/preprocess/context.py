"""Shared preprocessing context and path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreprocessContext:
    """Common paths and options for a target-pair preprocessing run."""

    repo_root: Path
    target_pair: str
    input_csv: Path
    model_name: str
    data_dir: Path
    models_dir: Path
    ckpt_dir: Path
    gpu_id: int = -1
    fgib_epochs: int = 10

    @property
    def target_activities(self) -> tuple[str, str]:
        target_names = self.target_pair.split("_")
        if len(target_names) != 2:
            raise ValueError(f"Cannot infer target activities from target pair: {self.target_pair}")
        return f"{target_names[0]}_activity", f"{target_names[1]}_activity"

    @property
    def model_dir(self) -> Path:
        return self.models_dir / self.model_name

    @property
    def resources_dir(self) -> Path:
        return self.repo_root / "combimots" / "pmcts" / "resources" / "real"

    @classmethod
    def from_paths(
        cls,
        repo_root: Path,
        target_pair: str,
        input_csv: Path,
        model_name: str | None = None,
        data_dir: Path = Path("data"),
        models_dir: Path = Path("models"),
        ckpt_dir: Path = Path("ckpt"),
        gpu_id: int = -1,
        fgib_epochs: int = 10,
    ) -> "PreprocessContext":
        repo_root = repo_root.expanduser().resolve()
        return cls(
            repo_root=repo_root,
            target_pair=target_pair,
            input_csv=_resolve_under(repo_root, input_csv),
            model_name=model_name or target_pair,
            data_dir=_resolve_under(repo_root, data_dir),
            models_dir=_resolve_under(repo_root, models_dir),
            ckpt_dir=_resolve_under(repo_root, ckpt_dir),
            gpu_id=gpu_id,
            fgib_epochs=fgib_epochs,
        )


def _resolve_under(repo_root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else repo_root / path
