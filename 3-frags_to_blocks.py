from rdkit import Chem, RDLogger
import pandas as pd
import argparse

RDLogger.DisableLog('rdApp.*')

def process_smiles(smiles):
    """Clean individual SMILES string by removing dummy atoms and handling radicals."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    # Remove dummy atoms
    clean_mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles('*'))
    if clean_mol is None:
        return None
        
    # Convert radical electrons to hydrogens
    for atom in clean_mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            atom.SetNumExplicitHs(atom.GetNumRadicalElectrons())
            atom.SetNumRadicalElectrons(0)
    
    # SMILES from the cleaned molecule
    try:
        Chem.Kekulize(clean_mol)
        return Chem.MolToSmiles(clean_mol)
    except:
        # If kekulization fails, try replacing dummy atoms with [H]
        mol = Chem.MolFromSmiles(smiles.replace('[*:1]', '[H]'))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)

def clean_and_concat_smiles(input_files, output_file):
    """Clean and concatenate SMILES from multiple input files."""
    cleaned_dfs = []
    
    for input_file in input_files:
        df = pd.read_csv(input_file, header=None, names=['SMILES', 'Score'], sep=',')
        print(f"\nProcessing {input_file}:")
        print(f"Original number of entries: {len(df)}")
        
        df['Clean_SMILES'] = df['SMILES'].apply(process_smiles)
        
        clean_df = df[['Clean_SMILES', 'Score']].dropna()
        print(f"Number of entries after cleaning: {len(clean_df)}")
        print(f"Number of failed entries: {len(df) - len(clean_df)}")
        
        cleaned_dfs.append(clean_df)
    
    combined_df = pd.concat(cleaned_dfs, ignore_index=True)
    
    final_df = combined_df.drop_duplicates(subset='Clean_SMILES').reset_index(drop=True)
    
    final_df['ID'] = range(1, len(final_df) + 1)
    
    final_df = final_df[['ID', 'Clean_SMILES', 'Score']]
    final_df = final_df.rename(columns={'Clean_SMILES': 'smiles'})
    
    final_df.to_csv(output_file, index=False)
    
    print(f"\nFinal Results:")
    print(f"Total input entries: {len(combined_df)}")
    print(f"Total unique SMILES after deduplication: {len(final_df)}")
    print(f"Combined and cleaned SMILES saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Clean and concatenate SMILES from multiple input files'
    )
    parser.add_argument(
        'inputs', 
        nargs='+', 
        help='Input text file paths (can specify multiple files)'
    )
    parser.add_argument(
        'output', 
        type=str, 
        help='Output CSV file path'
    )
    
    args = parser.parse_args()
    clean_and_concat_smiles(args.inputs, args.output)

if __name__ == "__main__":
    main()
