"""Reporting primitives for CombiMOTS preprocessing steps."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepResult:
    """Structured result for one preprocessing step."""

    step: str
    status: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    """Structured result for a preprocessing runner invocation."""

    status: str
    steps: list[StepResult]
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "metadata": self.metadata,
            "steps": [step.to_dict() for step in self.steps],
            "warnings": self.warnings,
        }


def write_step_report(result: StepResult, report_path: Path | None) -> None:
    """Write a step report to JSON when a path is provided."""

    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.to_dict(), indent=2, default=str) + "\n")


def write_run_report(result: RunResult, report_path: Path | None) -> None:
    """Write a runner-level report to JSON when a path is provided."""

    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.to_dict(), indent=2, default=str) + "\n")
