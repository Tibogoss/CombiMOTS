from functools import cache
from pathlib import Path
from typing import Callable, List, Dict, Tuple
import os
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

from pmcts.generate.node import Node

from rdkit import Chem
from rdkit.Chem import AllChem

####### SET YOUR DOCKING PATH PREFIX #######
DOCKING_PATH_PREFIX = Path("/workspace/CombiMOTS/combimots/pmcts/docking")
############################################


def _prepare_single_ligand(args: Tuple[str, Path, Path]) -> Tuple[str, Path | None]:
    """Prepare a single ligand for docking.
    
    Args:
        args: Tuple of (smiles, ligand_dir, tmp_path)
        
    Returns:
        Tuple of (smiles, pdbqt_path) if successful, (smiles, None) if failed
    """
    smiles, ligand_dir, tmp_path = args
    name = hash(smiles)
    smiles_path = ligand_dir / f"{name}.smiles"
    mol2_path = ligand_dir / f"{name}.mol2"
    pdb_path = ligand_dir / f"{name}.pdb"
    pdbqt_path = ligand_dir / f"{name}.pdbqt"
    
    # Write SMILES file
    with open(smiles_path, 'w') as f:
        f.write(smiles)
    
    try:
        # SMILES to mol2 with 3D coordinates
        result = subprocess.run(['obabel', 
                               str(smiles_path), 
                               '-O', str(mol2_path),
                               '--gen3d', 'best',
                               '-p', '7.4'],
                              capture_output=True,
                              text=True)
        
        if result.returncode != 0:
            print(f"Failed to convert SMILES to mol2 for {smiles}: {result.stderr}")
            return smiles, None
            
        # mol2 to PDB
        result = subprocess.run(['obabel',
                               str(mol2_path),
                               '-O', str(pdb_path),
                               '-h',
                               '--gen3d', 'best',
                               '-p', '7.4'],  # Add hydrogens
                              capture_output=True,
                              text=True)
                              
        if result.returncode != 0:
            print(f"Failed to convert mol2 to PDB for {smiles}: {result.stderr}")
            return smiles, None
            
        # PDB to PDBQT
        result = subprocess.run(['obabel',
                               str(pdb_path),
                               '-O', str(pdbqt_path),
                               '--gen3d', 'best',
                               '-p', '7.4',
                               '--partialcharge', 'gasteiger'],
                              capture_output=True,
                              text=True)
                              
        if result.returncode != 0:
            print(f"Failed to convert PDB to PDBQT for {smiles}: {result.stderr}")
            return smiles, None
            
        # clean intermediate files
        smiles_path.unlink(missing_ok=True)
        mol2_path.unlink(missing_ok=True)
        pdb_path.unlink(missing_ok=True)
        
        return smiles, pdbqt_path
        
    except Exception as e:
        print(f"Conversion failed for {smiles}: {str(e)}")
        smiles_path.unlink(missing_ok=True)
        mol2_path.unlink(missing_ok=True)
        pdb_path.unlink(missing_ok=True)
        pdbqt_path.unlink(missing_ok=True)
        return smiles, None

def _prepare_ligands(smiles_list: List[str], tmp_path: Path, n_proc: int = 48) -> Dict[str, Path]:
    """Prepare ligands for docking by converting SMILES to PDBQT files in parallel.
    
    Args:
        smiles_list: List of SMILES strings to prepare
        tmp_path: Path to temporary directory for files
        
    Returns:
        Dictionary mapping SMILES to their PDBQT file paths
    """
    ligand_dir = tmp_path / "ligands"
    ligand_dir.mkdir(exist_ok=True)
    
    args = [(smiles, ligand_dir, tmp_path) for smiles in smiles_list]
    
    smiles_to_pdbqt = {}
    with Pool(processes=n_proc) as pool:
        for smiles, pdbqt_path in pool.imap(_prepare_single_ligand, args):
            if pdbqt_path is not None:
                smiles_to_pdbqt[smiles] = pdbqt_path
    
    for smiles_path in ligand_dir.glob('*.smiles'):
        smiles_path.unlink()
        
    return smiles_to_pdbqt

def _run_docking(smiles_to_pdbqt: Dict[str, Path], 
                receptor_path: str,
                task_id: str,
                center: Tuple[float, float, float],
                tmp_path: Path) -> Dict[str, float]:
    """Run docking for prepared ligands against a receptor.
    
    Args:
        smiles_to_pdbqt: Dictionary mapping SMILES to their PDBQT file paths
        receptor_path: Path to receptor PDBQT file
        task_id: for temporary directories in tmp_path
        center: (x, y, z) coordinates of binding site center
        tmp_path: Path to temporary directory
        
    Returns:
        Dictionary mapping SMILES to their docking scores
    """
    # task-specific directories and config files
    task_dir = tmp_path / f"task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    output_dir = task_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = task_dir / "config.txt"
    with open(config_path, 'w') as f:
        f.write(f"""receptor = {receptor_path}
center_x = {center[0]}
center_y = {center[1]}
center_z = {center[2]}
size_x = 20
size_y = 20
size_z = 20
thread = 5000
num_modes = 5
rilc_bfgs = 1
ligand_directory = {(tmp_path / "ligands").absolute()}
output_directory = {output_dir.absolute()}
opencl_binary_path = {DOCKING_PATH_PREFIX}/Vina-GPU-2.1/QuickVina2-GPU-2.1""")
    
    """ # Debug: Print ligand directory contents
    print("\nLigand directory contents:")
    for file in (tmp_path / "ligands").glob('*'):
        print(f"  {file.name}") """
    
    
    # Run QuickVina2-GPU
    vina_dir = DOCKING_PATH_PREFIX / "Vina-GPU-2.1" / "QuickVina2-GPU-2.1"
    result = subprocess.run(['./QuickVina2-GPU-2-1', '--config', str(config_path.absolute())], 
                          cwd=vina_dir,
                          capture_output=True,
                          text=True)
    
    #if result.returncode != 0:
    #    print(f"\nQuickVina2-GPU failed:")
    #    print(f"stdout: {result.stdout}")
    #    print(f"stderr: {result.stderr}")
        
    scores = {}
    for smiles in smiles_to_pdbqt:
        output_path = output_dir / f"{hash(smiles)}_out.pdbqt"
        try:
            with open(output_path) as f:
                for line in f:
                    if "REMARK VINA RESULT" in line:
                        score = float(line.split()[3])
                        scores[smiles] = score
                        break
        except:
            scores[smiles] = 0.0 
            
    return scores


def batch_dock(child_nodes_mol: Dict[Node, Tuple[str]], target: str, n_proc: int, sequential: bool = False) -> Tuple[Dict[Node, float], Dict[Node, float]]:
    """Batch dock molecules against both targets using QuickVina2-GPU.
    
    Args:
        child_nodes_mol: Dictionary mapping Nodes to their molecule SMILES
        target: Target protein name
        n_proc: Number of processors to use for ligand preparation
        sequential: If True, run docking tasks sequentially instead of in parallel -> tradeoff gpu memory
        
    Returns:
        Tuple of (ds1_scores, ds2_scores) where each is a dictionary mapping Nodes to their docking scores
        ds1_scores: Docking scores against target 1
        ds2_scores: Docking scores against target 2
    """
    os.makedirs("./tmp", exist_ok=True)
    with tempfile.TemporaryDirectory(dir="./tmp") as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # SMILES from each node
        node_to_smiles = {node: molecules[0] for node, molecules in child_nodes_mol.items()}
        unique_smiles = list(set(node_to_smiles.values()))
        
        smiles_to_pdbqt = _prepare_ligands(unique_smiles, tmp_path, n_proc=n_proc)
        
        # docking tasks
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
        
        results = [None, None]
        
        if sequential:
            # sequential docking -> tradeoff gpu memory
            for task_idx, task in enumerate(docking_tasks):
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
            # parallel docking
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
        
        # Map scores back to nodes
        ds1_scores = {node: results[0][smiles] for node, smiles in node_to_smiles.items()}
        ds2_scores = {node: results[1][smiles] for node, smiles in node_to_smiles.items()}
        
        return ds1_scores, ds2_scores