"""Building-block filtering utilities."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from preprocess.reports import StepResult, write_step_report


FORBIDDEN_QUICKVINA_ATOMIC_NUMBERS = frozenset({3, 5, 14})

RDLogger.DisableLog("rdApp.*")


def canonicalize_smiles_value(smiles: object) -> tuple[str | None, str | None]:
    if smiles is None or pd.isna(smiles):
        return None, "empty"

    smiles_text = str(smiles).strip()
    if not smiles_text:
        return None, "empty"

    mol = Chem.MolFromSmiles(smiles_text)
    if mol is None or mol.GetNumAtoms() == 0:
        return None, "invalid"
    if len(Chem.GetMolFrags(mol)) > 1:
        return None, "disconnected"

    try:
        return Chem.MolToSmiles(mol, canonical=True), None
    except Exception:
        return None, "invalid"


def has_forbidden_quickvina_element(smiles: str) -> bool:
    """Return whether a SMILES contains Li, B, or Si atoms."""

    canonical_smiles, _ = canonicalize_smiles_value(smiles)
    if canonical_smiles is None:
        return False
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return False
    return any(atom.GetAtomicNum() in FORBIDDEN_QUICKVINA_ATOMIC_NUMBERS for atom in mol.GetAtoms())


def is_valid_smiles(smiles: str) -> bool:
    """Return whether RDKit can parse the SMILES string."""

    canonical_smiles, _ = canonicalize_smiles_value(smiles)
    return canonical_smiles is not None


def clean_building_block_dataframe(
    df: pd.DataFrame,
    smiles_column: str = "smiles",
    remove_forbidden_quickvina_elements: bool = False,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    if smiles_column not in df.columns:
        raise ValueError(f"Input CSV must contain column '{smiles_column}'")

    keep_indices = []
    canonical_smiles = []
    empty_rows = 0
    invalid_rows = 0
    disconnected_rows = 0
    forbidden_rows = 0

    for row_index, smiles in df[smiles_column].items():
        canonical, reason = canonicalize_smiles_value(smiles)
        if canonical is None:
            if reason == "empty":
                empty_rows += 1
            elif reason == "disconnected":
                disconnected_rows += 1
            else:
                invalid_rows += 1
            continue

        if remove_forbidden_quickvina_elements and has_forbidden_quickvina_element(canonical):
            forbidden_rows += 1
            continue

        keep_indices.append(row_index)
        canonical_smiles.append(canonical)

    cleaned_df = df.loc[keep_indices].copy()
    cleaned_df.loc[:, smiles_column] = canonical_smiles
    duplicate_rows = int(cleaned_df.duplicated(subset=[smiles_column]).sum())
    cleaned_df = cleaned_df.drop_duplicates(subset=[smiles_column]).reset_index(drop=True)

    metrics = {
        "input_rows": len(df),
        "output_rows": len(cleaned_df),
        "empty_smiles_rows": empty_rows,
        "invalid_smiles_rows": invalid_rows,
        "disconnected_smiles_rows": disconnected_rows,
        "forbidden_element_rows": forbidden_rows,
        "duplicate_canonical_smiles_rows": duplicate_rows,
        "smiles_column": smiles_column,
    }
    return cleaned_df, metrics


def filter_quickvina_compatible_smiles(smiles_values: Iterable[object]) -> tuple[set[str], dict[str, int]]:
    df = pd.DataFrame({"smiles": list(smiles_values)})
    cleaned_df, metrics = clean_building_block_dataframe(
        df,
        smiles_column="smiles",
        remove_forbidden_quickvina_elements=True,
    )
    return set(cleaned_df["smiles"]), {key: int(value) for key, value in metrics.items() if isinstance(value, int)}


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

    filtered_df, cleanup_metrics = clean_building_block_dataframe(
        df,
        smiles_column=smiles_column,
        remove_forbidden_quickvina_elements=True,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_file, index=False)

    print(f"Total molecules: {len(df)}")
    print(f"Molecules retained after cleanup: {len(filtered_df)}")
    print(f"Molecules removed: {len(df) - len(filtered_df)}")

    warnings = []
    if cleanup_metrics["empty_smiles_rows"] or cleanup_metrics["invalid_smiles_rows"]:
        warnings.append("Empty or invalid SMILES were removed before docking/generation.")
    if cleanup_metrics["disconnected_smiles_rows"]:
        warnings.append("Disconnected SMILES were removed before docking/generation.")
    if cleanup_metrics["duplicate_canonical_smiles_rows"]:
        warnings.append("Duplicate canonical SMILES rows were removed.")

    result = StepResult(
        step="filter-elements",
        status="success",
        inputs=[str(input_file)],
        outputs=[str(output_file)],
        metrics={
            **cleanup_metrics,
            "removed_rows": len(df) - len(filtered_df),
        },
        warnings=warnings,
    )
    write_step_report(result, report_path)
    return result
