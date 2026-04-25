"""Docking-score precomputation for building-block CSV files."""

from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from pmcts.config import get_docking_tasks
from preprocess.reports import StepResult, write_step_report


def batch_dock_csv(
    input_csv: Path,
    output_csv: Path,
    target_pair: str = "gsk3b_jnk3",
    sequential: bool = False,
    report_path: Path | None = None,
    tmp_parent: Path = Path("tmp"),
    ligand_processes: int = 48,
) -> StepResult:
    """Batch dock unique SMILES from a CSV and append `ds_1`/`ds_2` columns."""

    if ligand_processes <= 0:
        raise ValueError(f"Ligand preparation process count must be positive, got {ligand_processes}")

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    df = pd.read_csv(input_csv)
    if "smiles" not in df.columns:
        raise ValueError(f"Input CSV must contain a 'smiles' column: {input_csv}")

    original_rows = len(df)
    df = df.drop_duplicates(subset=["smiles"])
    df["smiles"] = df["smiles"].astype(str)
    smiles_list = df["smiles"].tolist()
    duplicate_rows = original_rows - len(df)

    warnings = []
    if duplicate_rows:
        warnings.append(f"Dropped {duplicate_rows} duplicate SMILES rows before docking.")

    if not smiles_list:
        df["ds_1"] = []
        df["ds_2"] = []
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print("No input SMILES rows were available for docking")
        print(f"Results saved to {output_csv}")
        result = StepResult(
            step="precompute-docking",
            status="success",
            inputs=[str(input_csv)],
            outputs=[str(output_csv)],
            metrics={
                "input_rows": original_rows,
                "unique_smiles_rows": len(df),
                "duplicate_rows": duplicate_rows,
                "prepared_ligands": 0,
                "failed_ligand_preparations": 0,
                "missing_ds_1_outputs": 0,
                "missing_ds_2_outputs": 0,
                "target_pair": target_pair,
                "sequential": sequential,
                "ligand_processes": ligand_processes,
            },
            warnings=warnings + ["No input SMILES rows were available for docking."],
        )
        write_step_report(result, report_path)
        return result

    tmp_parent = Path(tmp_parent)
    tmp_parent.mkdir(parents=True, exist_ok=True)

    # Keep docking imports lazy so CLI help and dry-runs do not need docking dependencies.
    from pmcts.docking.docking_utils import _prepare_ligands, _run_docking

    with tempfile.TemporaryDirectory(dir=tmp_parent) as tmp_dir:
        tmp_path = Path(tmp_dir)

        print("Preparing ligands...")
        smiles_to_pdbqt = _prepare_ligands(smiles_list, tmp_path, n_proc=ligand_processes)
        failed_ligand_count = len(smiles_list) - len(smiles_to_pdbqt)
        print(f"Successfully prepared {len(smiles_to_pdbqt)} ligands")
        if failed_ligand_count:
            print(f"Ligand preparation failed for {failed_ligand_count} SMILES; their docking scores will be 0.0")

        docking_tasks = get_docking_tasks(target_pair, smiles_to_pdbqt, tmp_path)

        print("Running docking...")
        results: list[dict[str, float] | None] = [None for _ in docking_tasks]
        task_failures: list[str] = []

        if sequential:
            for task_idx, task in enumerate(docking_tasks):
                print(f"Running docking task {task_idx + 1}/{len(docking_tasks)}...")
                try:
                    results[task_idx] = _run_docking(
                        task["smiles_to_pdbqt"],
                        task["receptor_path"],
                        task["task_id"],
                        task["center"],
                        task["tmp_path"],
                    )
                except Exception as e:
                    message = f"Docking task {task_idx + 1} failed: {str(e)}"
                    print(message)
                    task_failures.append(message)
                    results[task_idx] = {smiles: 0.0 for smiles in smiles_to_pdbqt}
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_task = {
                    executor.submit(
                        _run_docking,
                        task["smiles_to_pdbqt"],
                        task["receptor_path"],
                        task["task_id"],
                        task["center"],
                        task["tmp_path"],
                    ): task_idx
                    for task_idx, task in enumerate(docking_tasks)
                }

                for future in as_completed(future_to_task):
                    task_idx = future_to_task[future]
                    try:
                        results[task_idx] = future.result()
                    except Exception as e:
                        message = f"Docking task {task_idx + 1} failed: {str(e)}"
                        print(message)
                        task_failures.append(message)
                        results[task_idx] = {smiles: 0.0 for smiles in smiles_to_pdbqt}

    print("Processing results...")
    ds1_scores = results[0] or {}
    ds2_scores = results[1] or {}

    df["ds_1"] = df["smiles"].map(ds1_scores).fillna(0.0)
    df["ds_2"] = df["smiles"].map(ds2_scores).fillna(0.0)

    missing_ds1 = sum(smiles not in ds1_scores for smiles in smiles_list)
    missing_ds2 = sum(smiles not in ds2_scores for smiles in smiles_list)
    if missing_ds1 or missing_ds2:
        print("Assigned 0.0 for missing docking outputs: " f"ds_1={missing_ds1}, ds_2={missing_ds2}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    if failed_ligand_count:
        warnings.append(f"Ligand preparation failed for {failed_ligand_count} SMILES; scores were set to 0.0.")
    if missing_ds1 or missing_ds2:
        warnings.append(f"Missing docking outputs were assigned 0.0: ds_1={missing_ds1}, ds_2={missing_ds2}.")
    warnings.extend(task_failures)

    result = StepResult(
        step="precompute-docking",
        status="success",
        inputs=[str(input_csv)],
        outputs=[str(output_csv)],
        metrics={
            "input_rows": original_rows,
            "unique_smiles_rows": len(df),
            "duplicate_rows": duplicate_rows,
            "prepared_ligands": len(smiles_to_pdbqt),
            "failed_ligand_preparations": failed_ligand_count,
            "missing_ds_1_outputs": missing_ds1,
            "missing_ds_2_outputs": missing_ds2,
            "target_pair": target_pair,
            "sequential": sequential,
            "ligand_processes": ligand_processes,
        },
        warnings=warnings,
    )
    write_step_report(result, report_path)
    return result
