from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main(
    custom_path: Path,
    real_path: Path,
    output_path: Path,
    threshold: float = 0.4,
    batch_size: int = 1000,
    report_path: Path | None = None,
) -> None:
    from preprocess.similarity import filter_similar_molecules

    filter_similar_molecules(
        custom_path=custom_path,
        real_path=real_path,
        output_path=output_path,
        threshold=threshold,
        batch_size=batch_size,
        report_path=report_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter REAL building blocks by Tanimoto similarity")
    parser.add_argument("--custom_path", "--custom-path", type=Path, required=True, help="CSV with custom building blocks")
    parser.add_argument("--real_path", "--real-path", type=Path, required=True, help="CSV with REAL building blocks")
    parser.add_argument("--output_path", "--output-path", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=0.4, help="Minimum Tanimoto similarity")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=1000, help="REAL rows to process per batch")
    parser.add_argument("--report_path", "--report-path", type=Path, default=None, help="Optional JSON report path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        custom_path=args.custom_path,
        real_path=args.real_path,
        output_path=args.output_path,
        threshold=args.threshold,
        batch_size=args.batch_size,
        report_path=args.report_path,
    )
