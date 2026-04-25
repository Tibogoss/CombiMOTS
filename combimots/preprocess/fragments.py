"""Fragment cleaning and merge utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from rdkit import Chem, RDLogger

from preprocess.reports import StepResult, write_step_report


RDLogger.DisableLog("rdApp.*")


def clean_fragment_smiles(smiles: str) -> str | None:
    """Remove dummy atoms and normalize a fragment SMILES string."""

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    clean_mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles("*"))
    if clean_mol is None:
        return None

    for atom in clean_mol.GetAtoms():
        radical_electrons = atom.GetNumRadicalElectrons()
        if radical_electrons > 0:
            atom.SetNumExplicitHs(radical_electrons)
            atom.SetNumRadicalElectrons(0)

    try:
        Chem.Kekulize(clean_mol)
        return Chem.MolToSmiles(clean_mol)
    except Exception:
        fallback_mol = Chem.MolFromSmiles(str(smiles).replace("[*:1]", "[H]"))
        if fallback_mol is None:
            return None
        return Chem.MolToSmiles(fallback_mol)


def merge_fragment_files(
    input_files: Iterable[Path],
    output_file: Path,
    report_path: Path | None = None,
) -> StepResult:
    """Clean, concatenate, deduplicate, and save fragment SMILES files."""

    input_paths = [Path(input_file) for input_file in input_files]
    if not input_paths:
        raise ValueError("At least one fragment input file is required")

    cleaned_dfs = []
    input_reports = []
    total_rows = 0
    total_clean_rows = 0

    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Fragment input file does not exist: {input_path}")

        df = pd.read_csv(input_path, header=None, names=["SMILES", "Score"], sep=",")
        df["Clean_SMILES"] = df["SMILES"].apply(clean_fragment_smiles)
        clean_df = df[["Clean_SMILES", "Score"]].dropna()

        input_row_count = len(df)
        clean_row_count = len(clean_df)
        total_rows += input_row_count
        total_clean_rows += clean_row_count
        input_reports.append({
            "path": str(input_path),
            "input_rows": input_row_count,
            "clean_rows": clean_row_count,
            "failed_rows": input_row_count - clean_row_count,
        })

        print(f"\nProcessing {input_path}:")
        print(f"Original number of entries: {input_row_count}")
        print(f"Number of entries after cleaning: {clean_row_count}")
        print(f"Number of failed entries: {input_row_count - clean_row_count}")
        cleaned_dfs.append(clean_df)

    combined_df = pd.concat(cleaned_dfs, ignore_index=True)
    final_df = combined_df.drop_duplicates(subset="Clean_SMILES").reset_index(drop=True)
    duplicate_rows = len(combined_df) - len(final_df)

    final_df["ID"] = range(1, len(final_df) + 1)
    final_df = final_df[["ID", "Clean_SMILES", "Score"]]
    final_df = final_df.rename(columns={"Clean_SMILES": "smiles"})

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file, index=False)

    print("\nFinal Results:")
    print(f"Total input entries: {total_rows}")
    print(f"Total cleaned entries before deduplication: {len(combined_df)}")
    print(f"Total unique SMILES after deduplication: {len(final_df)}")
    print(f"Combined and cleaned SMILES saved to: {output_file}")

    result = StepResult(
        step="merge-fragments",
        status="success",
        inputs=[str(path) for path in input_paths],
        outputs=[str(output_file)],
        metrics={
            "input_files": input_reports,
            "input_rows": total_rows,
            "clean_rows_before_deduplication": total_clean_rows,
            "failed_rows": total_rows - total_clean_rows,
            "duplicate_rows": duplicate_rows,
            "output_rows": len(final_df),
        },
    )
    write_step_report(result, report_path)
    return result
