from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocess.fgib import train_fgib_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_id", type=int, default=-1)
    parser.add_argument("-t", "--target", type=str, required=True, help="Target_activity column name in the CSV file e.g. 'GSK3B_activity'")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--save_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1e-5)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("ckpt"))
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args()

    train_fgib_model(
        target=args.target,
        gpu_id=args.gpu_id,
        epochs=args.epochs,
        output_checkpoint=args.ckpt_dir / f"{args.target}_{args.epochs}.pt",
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        batch_size=args.batch_size,
        save_epoch=args.save_epoch,
        seed=args.seed,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        beta=args.beta,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    main()
