"""Chemprop prediction orchestration for preprocessing."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from preprocess.reports import StepResult, write_step_report


def run_chemprop_predict(
    test_path: Path,
    preds_path: Path,
    checkpoint_dir: Path,
    report_path: Path | None = None,
    command: str = "chemprop_predict",
) -> StepResult:
    """Run `chemprop_predict` with input/output validation and reporting."""

    test_path = Path(test_path)
    preds_path = Path(preds_path)
    checkpoint_dir = Path(checkpoint_dir)

    if not test_path.exists():
        raise FileNotFoundError(f"Chemprop input CSV does not exist: {test_path}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Chemprop checkpoint directory does not exist: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Chemprop checkpoint path is not a directory: {checkpoint_dir}")

    input_df = pd.read_csv(test_path)
    if "smiles" not in input_df.columns:
        raise ValueError(f"Chemprop input CSV must contain a 'smiles' column: {test_path}")

    preds_path.parent.mkdir(parents=True, exist_ok=True)
    cli_command = [
        command,
        "--test_path", str(test_path),
        "--preds_path", str(preds_path),
        "--checkpoint_dir", str(checkpoint_dir),
    ]

    print("Running Chemprop predictions...")
    subprocess.run(cli_command, check=True)

    if not preds_path.exists():
        raise FileNotFoundError(f"Chemprop did not produce expected predictions file: {preds_path}")

    output_df = pd.read_csv(preds_path)
    warnings = []
    if len(output_df) != len(input_df):
        warnings.append(
            f"Chemprop output row count ({len(output_df)}) differs from input row count ({len(input_df)})."
        )

    result = StepResult(
        step="precompute-chemprop",
        status="success",
        inputs=[str(test_path), str(checkpoint_dir)],
        outputs=[str(preds_path)],
        metrics={
            "input_rows": len(input_df),
            "output_rows": len(output_df),
            "input_columns": list(input_df.columns),
            "output_columns": list(output_df.columns),
            "checkpoint_dir": str(checkpoint_dir),
            "command": command,
        },
        warnings=warnings,
    )
    write_step_report(result, report_path)
    return result
