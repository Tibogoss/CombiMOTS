from time import time
from rdkit import Chem

import torch
from torch_geometric.loader import DataLoader

from utils_fgib.data import CustomDataClass
from fgib_model.fgib import FGIB

import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np

def get_custom_loader(target, batch_size):
    """
    Get data loaders for custom dataset with target-specific data
    
    Args:
        target: Target column name (e.g., 'GSK3B_activity')
        batch_size: Batch size for DataLoader
    """
    start_time = time()
    # Load target-specific processed data
    save_path = f'data/{target.replace("_activity", "")}.pt'
    try:
        train, test = torch.load(save_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Processed data file not found at {save_path}. " 
                              f"Please run process_data first.")
    
    print(f'{time() - start_time:.2f} sec for data loading')

    train, test = CustomDataClass(train, target), CustomDataClass(test, target)
    print(f'Train: {len(train)} | Test: {len(test)}')

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
    
    
def get_load_model(ckpt, device):
    state_dict = torch.load(ckpt, map_location=device)['state_dict']
    model = FGIB(device).to(device)
    model.load_state_dict(state_dict)
    return model


def get_sanitize_error_frags(frags):
    benzene = Chem.MolFromSmiles('c1ccccc1')
    att = Chem.MolFromSmarts('[#0]')

    error_frags = []
    for frag in frags:
        mols = Chem.ReplaceSubstructs(Chem.MolFromSmiles(frag), att, benzene)
        sanitize_error = False
        for mol in mols:
            mol = Chem.DeleteSubstructs(mol, att)
            try:
                Chem.SanitizeMol(mol)
            except:
                sanitize_error = True
        if sanitize_error:
            error_frags.append(frag)
    return set(error_frags)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
