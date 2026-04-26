"""Similarity filtering for REAL building blocks."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

from preprocess.reports import StepResult, write_step_report


RDLogger.DisableLog("rdApp.*")


def compute_morgan_fingerprint(smiles: str, n_bits: int = 2048) -> DataStructs.ExplicitBitVect | None:
    """Compute an ECFP4/Morgan bit vector for a SMILES string."""

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)


def compute_tanimoto_similarity(
    fp1: DataStructs.ExplicitBitVect,
    fp2: DataStructs.ExplicitBitVect,
) -> float:
    """Compute Tanimoto similarity between two RDKit fingerprints."""

    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def filter_similar_molecules(
    custom_path: Path,
    real_path: Path,
    output_path: Path,
    threshold: float = 0.4,
    batch_size: int = 1000,
    smiles_column: str = "smiles",
    report_path: Path | None = None,
) -> StepResult:
    """Filter REAL blocks by maximum Tanimoto similarity to custom blocks."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Similarity threshold must be between 0 and 1, got {threshold}")
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}")

    custom_path = Path(custom_path)
    real_path = Path(real_path)
    output_path = Path(output_path)

    if not custom_path.exists():
        raise FileNotFoundError(f"Custom block CSV does not exist: {custom_path}")
    if not real_path.exists():
        raise FileNotFoundError(f"REAL block CSV does not exist: {real_path}")

    print("Loading custom building blocks...")
    custom_df = pd.read_csv(custom_path)
    _require_columns(custom_df, custom_path, (smiles_column,))

    custom_fps = []
    for smiles in tqdm(custom_df[smiles_column], total=len(custom_df), desc="Fingerprinting custom blocks", file=sys.stdout):
        fp = compute_morgan_fingerprint(str(smiles))
        if fp is not None:
            custom_fps.append(fp)

    invalid_custom_rows = len(custom_df) - len(custom_fps)
    if not custom_fps:
        raise ValueError(f"No valid custom building block SMILES found in {custom_path}")

    print("Processing REAL building blocks...")
    real_df = pd.read_csv(real_path)
    _require_columns(real_df, real_path, (smiles_column, "reagent_id"))

    similar_batches = []
    valid_real_rows = 0
    invalid_real_rows = 0

    for start in tqdm(range(0, len(real_df), batch_size), desc="Filtering REAL blocks", file=sys.stdout):
        batch = real_df.iloc[start:start + batch_size]
        batch_fps = []
        valid_indices = []

        for row_index, smiles in enumerate(batch[smiles_column]):
            fp = compute_morgan_fingerprint(str(smiles))
            if fp is None:
                invalid_real_rows += 1
                continue
            valid_real_rows += 1
            batch_fps.append(fp)
            valid_indices.append(row_index)

        if not batch_fps:
            continue

        max_similarities = [max(DataStructs.BulkTanimotoSimilarity(real_fp, custom_fps)) for real_fp in batch_fps]
        similar_indices = [index for index, similarity in enumerate(max_similarities) if similarity >= threshold]
        if similar_indices:
            valid_batch = batch.iloc[valid_indices]
            similar_batches.append(valid_batch.iloc[similar_indices])

    if similar_batches:
        result_df = pd.concat(similar_batches, ignore_index=True)[[smiles_column, "reagent_id"]]
        if smiles_column != "smiles":
            result_df = result_df.rename(columns={smiles_column: "smiles"})
        print(f"Found {len(result_df)} similar molecules")
    else:
        result_df = pd.DataFrame(columns=["smiles", "reagent_id"])
        print("No similar molecules found")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    warnings = []
    if invalid_custom_rows:
        warnings.append(f"Skipped {invalid_custom_rows} invalid custom SMILES rows.")
    if invalid_real_rows:
        warnings.append(f"Skipped {invalid_real_rows} invalid REAL SMILES rows.")
    if result_df.empty:
        warnings.append("No REAL building blocks met the similarity threshold.")

    result = StepResult(
        step="similar-blocks",
        status="success",
        inputs=[str(custom_path), str(real_path)],
        outputs=[str(output_path)],
        metrics={
            "custom_rows": len(custom_df),
            "valid_custom_rows": len(custom_fps),
            "invalid_custom_rows": invalid_custom_rows,
            "real_rows": len(real_df),
            "valid_real_rows": valid_real_rows,
            "invalid_real_rows": invalid_real_rows,
            "output_rows": len(result_df),
            "threshold": threshold,
            "batch_size": batch_size,
        },
        warnings=warnings,
    )
    write_step_report(result, report_path)
    return result


def _require_columns(df: pd.DataFrame, path: Path, columns: tuple[str, ...]) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV {path} is missing required columns: {missing_columns}")
