from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocess.fragments import clean_fragment_smiles, merge_fragment_files


process_smiles = clean_fragment_smiles


def clean_and_concat_smiles(input_files, output_file, report_path=None):
    return merge_fragment_files(
        input_files=[Path(input_file) for input_file in input_files],
        output_file=Path(output_file),
        report_path=Path(report_path) if report_path else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and concatenate SMILES from multiple input files")
    parser.add_argument("inputs", nargs="+", help="Input text file paths (can specify multiple files)")
    parser.add_argument("output", type=str, help="Output CSV file path")
    parser.add_argument("--report-path", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()

    clean_and_concat_smiles(args.inputs, args.output, report_path=args.report_path)


if __name__ == "__main__":
    main()
