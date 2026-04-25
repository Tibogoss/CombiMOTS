"""FGIB preprocessing, training, and fragment extraction helpers."""

from __future__ import annotations

import argparse
import datetime
from collections import defaultdict
from pathlib import Path
from time import time

import pandas as pd

from preprocess.reports import StepResult, write_step_report


def prepare_fgib_dataset(
    csv_path: Path,
    target: str,
    output_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    report_path: Path | None = None,
) -> StepResult:
    """Convert one target activity column into an FGIB train/test `.pt` dataset."""

    csv_path = Path(csv_path)
    output_path = Path(output_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"FGIB input CSV does not exist: {csv_path}")

    input_df = pd.read_csv(csv_path)
    missing_columns = [column for column in ("smiles", target) if column not in input_df.columns]
    if missing_columns:
        raise ValueError(f"FGIB input CSV is missing required columns: {missing_columns}")

    from utils_fgib.data import process_data

    train, test = process_data(
        csv_path=csv_path,
        target=target,
        test_size=test_size,
        random_state=random_state,
        save_path=output_path,
    )

    valid_target_rows = pd.to_numeric(input_df[target], errors="coerce").notna().sum()
    result = StepResult(
        step="fgib-data",
        status="success",
        inputs=[str(csv_path)],
        outputs=[str(output_path)],
        metrics={
            "target": target,
            "input_rows": len(input_df),
            "valid_target_rows": int(valid_target_rows),
            "train_rows": len(train),
            "test_rows": len(test),
            "dropped_graph_rows": int(valid_target_rows) - len(train) - len(test),
            "test_size": test_size,
            "random_state": random_state,
        },
    )
    write_step_report(result, report_path)
    return result


def train_fgib_model(
    target: str,
    gpu_id: int,
    epochs: int,
    output_checkpoint: Path,
    data_dir: Path = Path("data"),
    ckpt_dir: Path = Path("ckpt"),
    batch_size: int = 1024,
    save_epoch: int | None = None,
    seed: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 20,
    beta: float = 1e-5,
    report_path: Path | None = None,
) -> StepResult:
    """Train an FGIB model for one target and report produced checkpoints."""

    if epochs <= 0:
        raise ValueError(f"Epoch count must be positive, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}")

    import torch
    from torch import optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from tqdm import tqdm

    from fgib_model.fgib import FGIB
    from utils_fgib.utils import get_custom_loader as get_loader
    from utils_fgib.utils import set_seed

    data_dir = Path(data_dir)
    ckpt_dir = Path(ckpt_dir)
    output_checkpoint = Path(output_checkpoint)
    save_epoch = save_epoch or epochs
    expected_data_path = data_dir / f'{target.replace("_activity", "")}.pt'
    if not expected_data_path.exists():
        raise FileNotFoundError(f"Processed FGIB data file does not exist: {expected_data_path}")

    args = argparse.Namespace(
        gpu_id=gpu_id,
        target=target,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        save_epoch=save_epoch,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        beta=beta,
        data_dir=str(data_dir),
        ckpt_dir=str(ckpt_dir),
    )

    start_time = time()
    train_loader, test_loader = get_loader(target, batch_size, data_dir=data_dir)
    if len(train_loader) == 0:
        raise ValueError(f"FGIB training loader is empty for target {target}")

    expname = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{target}"
    print(f"\033[92m{expname}\033[0m")
    exp_dir = ckpt_dir / expname
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "log.log", "a") as f:
        f.write(f"{args}\n")

    set_seed(seed)
    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")
    model = FGIB(device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience, mode="min", verbose=True)
    loss_fn = torch.nn.MSELoss()

    checkpoint_paths: list[Path] = []
    final_metrics: dict[str, float | int | None] = {}

    for epoch in range(1, epochs + 1):
        train_loss, train_mse, test_mse, preserve = 0.0, 0.0, 0.0, 0.0

        model.train()
        for graph, value in tqdm(train_loader, desc=f"Epoch {epoch}"):
            graph, value = graph.to(device), value.to(device)
            optimizer.zero_grad()

            pred, kl_loss, preserve_rate = model(graph)
            mse_loss = loss_fn(pred, value.reshape(-1, 1).float())
            loss = mse_loss + beta * kl_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mse += mse_loss.item()
            preserve += preserve_rate

        has_test_loader = len(test_loader) > 0
        if has_test_loader:
            model.eval()
            for graph, value in test_loader:
                graph, value = graph.to(device), value.to(device)
                pred, _, _ = model(graph)
                mse_loss = loss_fn(pred, value.reshape(-1, 1).float())
                test_mse += mse_loss.item()

        scheduler.step(loss)

        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        preserve /= len(train_loader)
        test_mse_value: float | None = None
        if has_test_loader:
            test_mse /= len(test_loader)
            test_mse_value = test_mse
        preserve_value = float(preserve.item()) if hasattr(preserve, "item") else float(preserve)

        with open(exp_dir / "log.log", "a") as f:
            log = (
                f"[Epoch {epoch:03d} | {time() - start_time:.1f} sec] "
                f"Train Loss: {train_loss:.4e} | Train MSE: {train_mse:.4e} | "
            )
            if has_test_loader:
                log += f"Test MSE: {test_mse:.4e} | Preserve: {preserve_value:.4f}\n"
            else:
                log += f"Preserve: {preserve_value:.4f}\n"
            f.write(log)

        if epoch % save_epoch == 0:
            checkpoint_path = ckpt_dir / f"{target}_{epoch}.pt"
            torch.save({"state_dict": model.state_dict(), "args": args}, checkpoint_path)
            checkpoint_paths.append(checkpoint_path)

        final_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mse": train_mse,
            "test_mse": test_mse_value,
            "preserve": preserve_value,
        }

    if not output_checkpoint.exists():
        raise FileNotFoundError(f"Expected FGIB checkpoint was not produced: {output_checkpoint}")

    result = StepResult(
        step="train-fgib",
        status="success",
        inputs=[str(expected_data_path)],
        outputs=[str(output_checkpoint), str(exp_dir / "log.log")],
        metrics={
            "target": target,
            "epochs": epochs,
            "batch_size": batch_size,
            "save_epoch": save_epoch,
            "gpu_id": gpu_id,
            "device": str(device),
            "checkpoint_paths": [str(path) for path in checkpoint_paths],
            "experiment_dir": str(exp_dir),
            "final_metrics": final_metrics,
        },
    )
    write_step_report(result, report_path)
    return result


def extract_fgib_fragments(
    target: str,
    gib_path: Path,
    vocab_path: Path,
    gpu_id: int = -1,
    vocab_size: int = 300,
    data_dir: Path = Path("data"),
    report_path: Path | None = None,
) -> StepResult:
    """Score and write the top FGIB fragments for one target."""

    if vocab_size <= 0:
        raise ValueError(f"Vocabulary size must be positive, got {vocab_size}")

    import numpy as np
    import torch
    from rdkit import Chem, RDLogger
    from tqdm import tqdm

    from utils_fgib.utils import get_custom_loader as get_loader
    from utils_fgib.utils import get_load_model, get_sanitize_error_frags

    RDLogger.DisableLog("rdApp.*")

    gib_path = Path(gib_path)
    vocab_path = Path(vocab_path)
    data_dir = Path(data_dir)
    expected_data_path = data_dir / f'{target.replace("_activity", "")}.pt'
    if not gib_path.exists():
        raise FileNotFoundError(f"FGIB checkpoint does not exist: {gib_path}")
    if not expected_data_path.exists():
        raise FileNotFoundError(f"Processed FGIB data file does not exist: {expected_data_path}")

    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")
    model = get_load_model(gib_path, device=device)
    train_loader, _ = get_loader(target=target, batch_size=1, data_dir=data_dir)

    frag_w_dict: dict[str, list[float]] = defaultdict(list)
    frag_prop_dict: dict[str, list[float]] = defaultdict(list)
    graph_count = 0
    for graph, value in tqdm(train_loader, desc="Scoring FGIB fragments"):
        graph_count += 1
        with torch.no_grad():
            weights = model(graph.to(device), get_w=True)

        for frag, frag_w in zip(graph[0].frags, weights):
            frag_w_dict[frag].append(frag_w.item())
            frag_prop_dict[frag].append(value.item())

    error_frags = get_sanitize_error_frags(frag_w_dict)
    for frag in error_frags:
        del frag_w_dict[frag]
        del frag_prop_dict[frag]

    frag_num_dict = {frag: Chem.MolFromSmiles(frag).GetNumAtoms() for frag in frag_w_dict}
    scores = [
        (np.array(frag_prop_dict[frag]) * np.array(frag_w_dict[frag])).mean() / np.sqrt(frag_num_dict[frag])
        for frag in frag_w_dict
    ]
    frag_tuples = sorted(zip(frag_w_dict, scores), key=lambda item: item[1], reverse=True)[:vocab_size]

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w") as f:
        f.writelines([f"{frag},{score}\n" for frag, score in frag_tuples])

    result = StepResult(
        step="fragments",
        status="success",
        inputs=[str(gib_path), str(expected_data_path)],
        outputs=[str(vocab_path)],
        metrics={
            "target": target,
            "graphs_scored": graph_count,
            "unique_fragments_before_sanitize_filter": len(frag_w_dict) + len(error_frags),
            "sanitize_error_fragments": len(error_frags),
            "unique_fragments_after_sanitize_filter": len(frag_w_dict),
            "output_fragments": len(frag_tuples),
            "vocab_size": vocab_size,
            "gpu_id": gpu_id,
            "device": str(device),
        },
    )
    write_step_report(result, report_path)
    return result
