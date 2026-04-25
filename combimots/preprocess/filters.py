"""Building-block filtering utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from preprocess.reports import StepResult, write_step_report


FORBIDDEN_QUICKVINA_ATOMIC_NUMBERS = frozenset({3, 5, 14})

RDLogger.DisableLog("rdApp.*")


def has_forbidden_quickvina_element(smiles: str) -> bool:
    """Return whether a SMILES contains Li, B, or Si atoms."""

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False
    return any(atom.GetAtomicNum() in FORBIDDEN_QUICKVINA_ATOMIC_NUMBERS for atom in mol.GetAtoms())


def is_valid_smiles(smiles: str) -> bool:
    """Return whether RDKit can parse the SMILES string."""

    return Chem.MolFromSmiles(str(smiles)) is not None


def filter_for_quickvina_elements(
    input_file: Path,
    output_file: Path,
    smiles_column: str = "smiles",
    report_path: Path | None = None,
) -> StepResult:
    """Remove building blocks containing Li, B, or Si atoms and save a CSV."""

    input_file = Path(input_file)
    output_file = Path(output_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_file}")

    df = pd.read_csv(input_file)
    if smiles_column not in df.columns:
        raise ValueError(f"Input CSV must contain column '{smiles_column}': {input_file}")

    invalid_mask = ~df[smiles_column].apply(is_valid_smiles)
    forbidden_mask = df[smiles_column].apply(has_forbidden_quickvina_element)
    filtered_df = df[~forbidden_mask]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_file, index=False)

    print(f"Total molecules: {len(df)}")
    print(f"Molecules without B/Si/Li: {len(filtered_df)}")
    print(f"Molecules removed: {int(forbidden_mask.sum())}")
    if invalid_mask.any():
        print(f"Invalid SMILES kept for legacy compatibility: {int(invalid_mask.sum())}")

    warnings = []
    if invalid_mask.any():
        warnings.append("Invalid SMILES were retained because the legacy filter retained unparsable rows.")

    result = StepResult(
        step="filter-elements",
        status="success",
        inputs=[str(input_file)],
        outputs=[str(output_file)],
        metrics={
            "input_rows": len(df),
            "output_rows": len(filtered_df),
            "removed_rows": int(forbidden_mask.sum()),
            "invalid_smiles_rows_retained": int(invalid_mask.sum()),
            "smiles_column": smiles_column,
        },
        warnings=warnings,
    )
    write_step_report(result, report_path)
    return result
