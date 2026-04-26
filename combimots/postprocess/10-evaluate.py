import pandas as pd
import argparse
import os
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tdc import Oracle



def validity(df, smiles_col):
    valid = 0
    for smiles in df[smiles_col]:
        if Chem.MolFromSmiles(smiles):
            valid += 1
    return valid / len(df)

def calculate_novelty(generated_df, training_df, smiles_col):
    gen_mols = [Chem.MolFromSmiles(s) for s in generated_df[smiles_col]]
    gen_mols = [m for m in gen_mols if m is not None]
    
    train_mols = [Chem.MolFromSmiles(s) for s in training_df[smiles_col]]
    train_mols = [m for m in train_mols if m is not None]
    
    if len(gen_mols) == 0 or len(train_mols) == 0:
        return 0.0
        
    gen_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 3, 2048) for m in gen_mols]
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 3, 2048) for m in train_mols]
    
    novel_count = 0
    for i in range(len(gen_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(gen_fps[i], train_fps)
        if max(sims) < 0.4:
            novel_count += 1
            
    novelty = novel_count / len(gen_fps)
    return novelty

def calculate_diversity(df, smiles_col):
    mols = [Chem.MolFromSmiles(s) for s in df[smiles_col]]
    mols = [m for m in mols if m is not None]
    
    if len(mols) < 2:
        return 0.0
        
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 3, 2048) for m in mols]
    
    similarity = 0
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        similarity += sum(sims)
    
    n = len(fps)
    n_pairs = n * (n - 1) / 2
    diversity = 1 - similarity / n_pairs
    
    return diversity

def calculate_uniqueness(df, smiles_col):
    return df[smiles_col].nunique() / len(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--generation', required=True)
    parser.add_argument('--training', required=True)
    parser.add_argument('--smiles_col', default='smiles', help='Name of SMILES column')
    args = parser.parse_args()

    model = args.model
    generation = pd.read_csv(args.generation)
    training = pd.read_csv(args.training)


    validity_score = validity(generation, args.smiles_col)
    novelty_score = calculate_novelty(generation, training, args.smiles_col)
    diversity_score = calculate_diversity(generation, args.smiles_col)
    uniqueness_score = calculate_uniqueness(generation, args.smiles_col)

    qed_oracle = Oracle(name='qed')
    sa_oracle = Oracle(name='sa')

    qed_scores = [qed_oracle(smiles) for smiles in generation[args.smiles_col]]
    sa_scores = [sa_oracle(smiles) for smiles in generation[args.smiles_col]]
    
    qed_mean = np.mean([s for s in qed_scores if s is not None])
    sa_mean = np.mean([s for s in sa_scores if s is not None])

    print(f'Validity: {validity_score:.2f}')
    print(f'Novelty: {novelty_score:.2f}')
    print(f'Diversity: {diversity_score:.2f}')
    print(f'Uniqueness: {uniqueness_score:.2f}')
    print(f'QED: {qed_mean:.2f}')
    print(f'SA: {sa_mean:.2f}')

    scores = pd.DataFrame({
        'Validity': [validity_score],
        'Novelty': [novelty_score],
        'Diversity': [diversity_score],
        'Uniqueness': [uniqueness_score],
        'QED': [qed_mean],
        'SA': [sa_mean]
    })
    
    if not os.path.exists('evaluation'):
        os.makedirs('evaluation')
    scores.to_csv(f'evaluation/scores_{model}.csv', index=False)
