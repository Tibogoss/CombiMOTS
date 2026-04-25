"""Compatibility wrapper for QuickVina element filtering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "combimots"))

from preprocess.filters import filter_for_quickvina_elements, has_forbidden_quickvina_element


has_B_Si_or_Li = has_forbidden_quickvina_element


def filter_molecules(input_file, output_file, report_path=None):
    """Read a CSV, remove rows containing B, Si, or Li atoms, and save it."""

    return filter_for_quickvina_elements(
        input_file=Path(input_file),
        output_file=Path(output_file),
        report_path=Path(report_path) if report_path else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter out molecules containing B, Si, or Li atoms from a CSV file.")
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("output", help="Output CSV file path")
    parser.add_argument("--report-path", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()

    filter_molecules(args.input, args.output, report_path=args.report_path)


if __name__ == "__main__":
    main()
