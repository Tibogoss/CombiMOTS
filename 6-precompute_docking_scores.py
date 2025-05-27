#!/usr/bin/env python3
"""Script to batch dock SMILES from a CSV file using QuickVina-GPU."""

import argparse
from pathlib import Path
import tempfile
import os
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

from combimots.pmcts.docking.docking_utils import _prepare_ligands, _run_docking 

####### SET YOUR DOCKING PATH PREFIX #######
DOCKING_PATH_PREFIX = Path("/workspace/CombiMOTS/combimots/pmcts/docking")
############################################

def batch_dock_csv(input_csv: str,
                  output_csv: str,
                  target: str = 'gsk3b_jnk3',
                  sequential: bool = False):
    """Batch dock SMILES from a CSV file and add docking scores as new columns.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        target: Target pair for docking. Options: 'gsk3b_jnk3', 'dhodh_rorgt', 'egfr_met', 'pik3ca_mtor'
        sequential: If True, run docking tasks sequentially instead of in parallel
    """
    # Load data
    df = pd.read_csv(input_csv)
    df = df.drop_duplicates(subset=['smiles'])

    smiles_list = df['smiles'].tolist()
    
    # Create temporary directory
    os.makedirs("./tmp", exist_ok=True)
    with tempfile.TemporaryDirectory(dir="./tmp") as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        print("Preparing ligands...")
        smiles_to_pdbqt = _prepare_ligands(smiles_list, tmp_path)
        print(f"Successfully prepared {len(smiles_to_pdbqt)} ligands")
        
        # Define docking tasks
        if target == 'gsk3b_jnk3':
            docking_tasks = [
                # Task 1: GSK3B
                {
                    'smiles_to_pdbqt': smiles_to_pdbqt,
                    'receptor_path': f"{DOCKING_PATH_PREFIX}/6Y9S.pdbqt",
                    'task_id': 'gsk3b',
                    'center': (24.503, 9.183, 9.226),
                    'tmp_path': tmp_path
                },
                # Task 2: JNK3
                {
                    'smiles_to_pdbqt': smiles_to_pdbqt,
                    'receptor_path': f"{DOCKING_PATH_PREFIX}/4WHZ.pdbqt",
                    'task_id': 'jnk3',
                    'center': (4.327, 101.902, 141.338),
                    'tmp_path': tmp_path
                }
            ]
        elif target == 'dhodh_rorgt':
            docking_tasks = [
                # Task 1: DHODH
                {
                    'smiles_to_pdbqt': smiles_to_pdbqt,
                    'receptor_path': f"{DOCKING_PATH_PREFIX}/6QU7.pdbqt",
                    'task_id': 'dhodh',
                    'center': (33.359, -11.558, -22.820),
                    'tmp_path': tmp_path
                },
                # Task 2: RORGT
                {
                    'smiles_to_pdbqt': smiles_to_pdbqt,
                    'receptor_path': f"{DOCKING_PATH_PREFIX}/5NTP.pdbqt",
                    'task_id': 'rorgt',
                    'center': (18.003, 11.762, 20.391),
                    'tmp_path': tmp_path
                }
            ]
        elif target == 'egfr_met':
            docking_tasks = [
                # Task 1: EGFR
                {
                    'smiles_to_pdbqt': smiles_to_pdbqt,
                    'receptor_path': f"{DOCKING_PATH_PREFIX}/1M17.pdbqt",
                    'task_id': 'egfr',
                    'center': (22.014,0.253,52.79),
                    'tmp_path': tmp_path
                },
                # Task 2: MET
                {
                    'smiles_to_pdbqt': smiles_to_pdbqt,
                    'receptor_path': f"{DOCKING_PATH_PREFIX}/4MXC.pdbqt",
                    'task_id': 'met',
                    'center': (-9.384,17.423,-28.886),
                    'tmp_path': tmp_path
                }
            ]
        elif target == 'pik3ca_mtor':
            docking_tasks = [
                # Task 1: PIK3CA
                {
                    'smiles_to_pdbqt': smiles_to_pdbqt,
                    'receptor_path': f"{DOCKING_PATH_PREFIX}/8V8I.pdbqt",
                    'task_id': 'pik3ca',
                    'center': (-19.947,-23.175,10.569),
                    'tmp_path': tmp_path
                },
                # Task 2: MTOR
                {
                    'smiles_to_pdbqt': smiles_to_pdbqt,
                    'receptor_path': f"{DOCKING_PATH_PREFIX}/3FAP.pdbqt",
                    'task_id': 'mtor',
                    'center': (-8.630,26.528,36.52),
                    'tmp_path': tmp_path
                }
            ]
        else:
            raise ValueError(f"Invalid target: {target}")
        
        print("Running docking...")
        results = [None, None]
        
        if sequential:
            for task_idx, task in enumerate(docking_tasks):
                print(f"Running docking task {task_idx + 1}/{len(docking_tasks)}...")
                try:
                    results[task_idx] = _run_docking(
                        task['smiles_to_pdbqt'],
                        task['receptor_path'],
                        task['task_id'],
                        task['center'],
                        task['tmp_path']
                    )
                except Exception as e:
                    print(f"Docking task {task_idx + 1} failed: {str(e)}")
                    results[task_idx] = {smiles: 0.0 for smiles in smiles_to_pdbqt}
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_task = {
                    executor.submit(
                        _run_docking,
                        task['smiles_to_pdbqt'],
                        task['receptor_path'],
                        task['task_id'],
                        task['center'],
                        task['tmp_path']
                    ): i for i, task in enumerate(docking_tasks)
                }

                for future in as_completed(future_to_task):
                    task_idx = future_to_task[future]
                    try:
                        results[task_idx] = future.result()
                    except Exception as e:
                        print(f"Docking task {task_idx + 1} failed: {str(e)}")
                        results[task_idx] = {smiles: 0.0 for smiles in smiles_to_pdbqt}
        
        print("Processing results...")
        ds1_scores = results[0]
        ds2_scores = results[1]

        df['ds_1'] = df['smiles'].map(ds1_scores)
        df['ds_2'] = df['smiles'].map(ds2_scores)
        
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='Batch dock SMILES from CSV using QuickVina-GPU')
    parser.add_argument('input_csv', type=str, help='Input CSV file with SMILES')
    parser.add_argument('output_csv', type=str, help='Output CSV file')
    parser.add_argument('--target_pair', type=str, default='gsk3b_jnk3', help='Target pair for docking')
    parser.add_argument('--sequential', action='store_true', help='Run docking tasks sequentially')
    
    args = parser.parse_args()
    
    batch_dock_csv(args.input_csv,
                  args.output_csv,
                  args.target_pair,
                  args.sequential)

if __name__ == '__main__':
    main()
