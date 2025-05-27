"""Filter REAL Space reaction mapping to only keep building blocks from a custom set."""
import pickle
from pathlib import Path

import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from tap import tapify


def canonicalize_smiles(smiles: str) -> str | None:
    """Canonicalize a SMILES string using RDKit.
    
    :param smiles: Input SMILES string
    :return: Canonicalized SMILES string or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def filter_reaction_mapping(
    input: Path,
    real_path: Path,
    save_path: Path,
    smiles_column: str = 'smiles'
) -> None:
    """Filter reaction mapping to only keep building blocks from input set.
    
    :param input: Path to CSV with custom building blocks (must have smiles_column)
    :param real_path: Path to original reaction mapping pickle file
    :param save_path: Path to save filtered mapping pickle file
    :param smiles_column: Name of column containing SMILES in input CSV
    """
    print("Loading and processing custom building blocks...")
    custom_df = pd.read_csv(input)
    
    # Set of canonicalized SMILES
    custom_smiles = set()
    for smiles in tqdm(custom_df[smiles_column]):
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            custom_smiles.add(canonical)
    
    if not custom_smiles:
        raise ValueError("No valid building blocks found in input file")
    
    print(f"Found {len(custom_smiles):,} valid building blocks")
    
    print(f"\nLoading reaction mapping from {real_path}")
    with open(real_path, 'rb') as f:
        original_mapping = pickle.load(f)
    
    # initialize
    filtered_mapping = {}
    
    print("\nFiltering building blocks...")
    total_original = 0
    total_filtered = 0
    
    # each reaction
    for reaction_id, positions in tqdm(original_mapping.items()):
        filtered_positions = {}
        
        # each position in reaction
        for position, building_blocks in positions.items():
            filtered_building_blocks = []
            
            # each building block
            for bb_smiles in building_blocks:
                canonical = canonicalize_smiles(bb_smiles)
                if canonical in custom_smiles:
                    filtered_building_blocks.append(bb_smiles)
            
            total_original += len(building_blocks)
            total_filtered += len(filtered_building_blocks)
            
            # Only keep position if it has any building blocks left
            if filtered_building_blocks:
                filtered_positions[position] = filtered_building_blocks
        
        # Only keep reaction if it has any positions left
        if filtered_positions:
            filtered_mapping[reaction_id] = filtered_positions
    
    ### TODO: correct the filtration statistics!! 
    print(f"\nStatistics:")
    print(f"Original building blocks: {total_original:,}")
    print(f"Filtered building blocks: {total_filtered:,}")
    print(f"Retention rate: {100 * total_filtered / total_original:.1f}%")
    print(f"Original reactions: {len(original_mapping):,}")
    print(f"Filtered reactions: {len(filtered_mapping):,}")
    
    # save filtered mapping
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(filtered_mapping, f)
    
    print(f"\nSaved filtered mapping to {save_path}")


def main(
    input: Path,
    real_path: Path,
    save_path: Path,
    smiles_column: str = 'smiles'
) -> None:
    """Main function to run the reaction mapping filtering.
    
    :param input: Path to CSV with custom building blocks (must have smiles_column)
    :param real_path: Path to original reaction mapping pickle file
    :param save_path: Path to save filtered mapping pickle file
    :param smiles_column: Name of column containing SMILES in input CSV
    """
    filter_reaction_mapping(
        input=input,
        real_path=real_path,
        save_path=save_path,
        smiles_column=smiles_column
    )


if __name__ == '__main__':
    tapify(main)