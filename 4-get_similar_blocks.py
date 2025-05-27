"""Filter REAL building blocks based on Tanimoto similarity to custom building blocks."""
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
from tap import tapify

def compute_morgan_fingerprint(smiles: str, n_bits: int = 2048) -> np.ndarray:
    """Compute Morgan fingerprint for a molecule.
    
    :param smiles: SMILES string of the molecule
    :param n_bits: Number of bits in fingerprint
    :return: Binary array representing Morgan fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))    # ECFP4

def compute_tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Tanimoto similarity between two fingerprints.
    
    :param fp1: First fingerprint
    :param fp2: Second fingerprint
    :return: Tanimoto similarity score
    """
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    return intersection / union if union != 0 else 0.0

def filter_similar_molecules(
    custom_path: Path,
    real_path: Path,
    output_path: Path,
    threshold: float = 0.7,
    batch_size: int = 1000
) -> None:
    """Filter REAL building blocks based on Tanimoto similarity to custom building blocks.
    
    :param custom_path: Path to CSV with custom building blocks
    :param real_path: Path to CSV with REAL building blocks
    :param output_path: Path to save filtered building blocks
    :param threshold: Minimum Tanimoto similarity threshold
    :param batch_size: Number of molecules to process at once
    """
    print("Loading custom building blocks...")
    custom_df = pd.read_csv(custom_path)
    custom_fps = []
    
    for _, row in tqdm(custom_df.iterrows(), total=len(custom_df)):
        fp = compute_morgan_fingerprint(row['smiles'])
        if fp is not None:
            custom_fps.append(fp)
    
    custom_fps = np.array(custom_fps)
    
    print("Processing REAL building blocks...")
    real_df = pd.read_csv(real_path)
    similar_molecules = []
    
    for i in tqdm(range(0, len(real_df), batch_size)):
        batch = real_df.iloc[i:i + batch_size]
        batch_fps = []
        
        for _, row in batch.iterrows():
            fp = compute_morgan_fingerprint(row['smiles'])
            batch_fps.append(fp)
        
        valid_indices = [i for i, fp in enumerate(batch_fps) if fp is not None]
        if not valid_indices:
            continue
            
        batch = batch.iloc[valid_indices]
        batch_fps = np.array([batch_fps[i] for i in valid_indices])
        
        max_similarities = np.zeros(len(batch_fps))
        for custom_fp in custom_fps:
            similarities = np.array([
                compute_tanimoto_similarity(custom_fp, batch_fp)
                for batch_fp in batch_fps
            ])
            max_similarities = np.maximum(max_similarities, similarities)
        
        similar_indices = np.where(max_similarities >= threshold)[0]
        similar_molecules.append(batch.iloc[similar_indices])
    
    if similar_molecules:
        result_df = pd.concat(similar_molecules)
        result_df = result_df[['smiles', 'reagent_id']]
        print(f"Found {len(result_df)} similar molecules")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
    else:
        print("No similar molecules found")

def main(
    custom_path: Path,
    real_path: Path, 
    output_path: Path,
    threshold: float = 0.7,
    batch_size: int = 1000
) -> None:
    """Main function to run the similarity filtering.
    
    :param custom_path: Path to CSV with custom building blocks
    :param real_path: Path to CSV with REAL building blocks 
    :param output_path: Path to save filtered building blocks
    :param threshold: Minimum Tanimoto similarity threshold
    :param batch_size: Number of molecules to process at once
    """
    filter_similar_molecules(
        custom_path=custom_path,
        real_path=real_path,
        output_path=output_path,
        threshold=threshold,
        batch_size=batch_size
    )

if __name__ == '__main__':
    tapify(main)