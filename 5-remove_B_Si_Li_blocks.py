import pandas as pd
from rdkit import Chem
import argparse
import sys

def has_B_Si_or_Li(smiles):
    """
    Check if a molecule contains Boron, Silicon, or Lithium atoms.
    Returns True if B, Si, or Li is present, False otherwise.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # Get all atomic numbers in the molecule
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        
        # Lithium's atomic number is 3, Boron's is 5, Silicon's is 14
        return 3 in atoms or 5 in atoms or 14 in atoms
    
    except:
        return False

def filter_molecules(input_file, output_file):
    """
    Read CSV file, remove entries with B, Si, or Li atoms, and save to new CSV.
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to save filtered CSV file
    """
    try:
        df = pd.read_csv(input_file)
        
        # mask for molecules without B, Si, or Li
        mask = ~df['smiles'].apply(has_B_Si_or_Li)
        
        filtered_df = df[mask]
        
        filtered_df.to_csv(output_file, index=False)
        
        print(f"Total molecules: {len(df)}")
        print(f"Molecules without B/Si/Li: {len(filtered_df)}")
        print(f"Molecules removed: {len(df) - len(filtered_df)}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Filter out molecules containing B, Si, or Li atoms from a CSV file.')
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('output', help='Output CSV file path')
    
    args = parser.parse_args()
    filter_molecules(args.input, args.output)

if __name__ == "__main__":
    main()