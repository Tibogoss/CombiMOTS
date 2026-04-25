#!/usr/bin/env python3
"""Compatibility wrapper for QuickVina-GPU docking-score precomputation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "combimots"))

from pmcts.config import SUPPORTED_TARGET_PAIRS
from preprocess.docking_scores import batch_dock_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch dock SMILES from CSV using QuickVina-GPU")
    parser.add_argument("input_csv", type=Path, help="Input CSV file with SMILES")
    parser.add_argument("output_csv", type=Path, help="Output CSV file")
    parser.add_argument(
        "--target_pair",
        "--target-pair",
        type=str,
        default="gsk3b_jnk3",
        choices=SUPPORTED_TARGET_PAIRS,
        help="Target pair for docking",
    )
    parser.add_argument("--sequential", action="store_true", help="Run docking tasks sequentially")
    parser.add_argument("--report_path", "--report-path", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()

    batch_dock_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        target_pair=args.target_pair,
        sequential=args.sequential,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    main()
