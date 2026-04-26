from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def filter_reaction_mapping(
    input: Path,
    real_path: Path,
    save_path: Path,
    smiles_column: str = "smiles",
    report_path: Path | None = None,
) -> None:
    from preprocess.search_space import reduce_mapping_to_csv_blocks

    report = reduce_mapping_to_csv_blocks(
        input_csv=input,
        real_path=real_path,
        save_path=save_path,
        smiles_column=smiles_column,
        report_path=None,
    )
    if report_path is not None:
        from preprocess.reports import StepResult, write_step_report

        fallback_count = len(report.get("fallback_positions", []))
        warnings = []
        if fallback_count:
            warnings.append(f"{fallback_count} reaction positions fell back to original REAL blocks.")
        write_step_report(
            StepResult(
                step="map-search-space",
                status="success",
                inputs=[str(input), str(real_path)],
                outputs=[str(save_path)],
                metrics=report,
                warnings=warnings,
            ),
            report_path,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reduce REAL reaction mapping to CSV building blocks")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV containing building-block SMILES")
    parser.add_argument("--real_path", "--real-path", type=Path, required=True, help="Original REAL reaction mapping pickle")
    parser.add_argument("--save_path", "--save-path", type=Path, required=True, help="Output mapping pickle")
    parser.add_argument("--smiles_column", "--smiles-column", default="smiles")
    parser.add_argument("--report_path", "--report-path", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filter_reaction_mapping(
        input=args.input,
        real_path=args.real_path,
        save_path=args.save_path,
        smiles_column=args.smiles_column,
        report_path=args.report_path,
    )
