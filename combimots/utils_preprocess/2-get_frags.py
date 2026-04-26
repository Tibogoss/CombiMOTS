from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocess.fgib import extract_fgib_fragments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_id", type=int, default=-1)
    parser.add_argument("-t", "--target", type=str, required=True, help="Target_activity column name in the CSV file e.g. 'GSK3B_activity'")
    parser.add_argument("-m", "--gib_path", type=Path, required=True, help="Path to the trained model")
    parser.add_argument("-v", "--vocab_path", type=Path, required=True, help="Path to save the fragment vocabulary")
    parser.add_argument("-s", "--vocab_size", type=int, default=300, help="Maximum size of the fragment vocabulary")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args()

    extract_fgib_fragments(
        target=args.target,
        gib_path=args.gib_path,
        vocab_path=args.vocab_path,
        gpu_id=args.gpu_id,
        vocab_size=args.vocab_size,
        data_dir=args.data_dir,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    main()
