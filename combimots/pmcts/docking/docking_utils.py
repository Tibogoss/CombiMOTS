import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import os
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

from pmcts.generate.node import Node
from pmcts.config import get_docking_path, get_docking_tasks


DOCKING_PATH_PREFIX = get_docking_path()


def _ligand_id(smiles: str) -> str:
    """Return a deterministic filesystem-safe ligand ID for a SMILES string."""

    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:24]


def _quickvina_env() -> dict[str, str]:
    """Return an environment that can find the system OpenCL ICD when available."""

    env = os.environ.copy()
    if "OCL_ICD_VENDORS" not in env:
        vendor_dirs = [Path("/etc/OpenCL/vendors")]
        if env.get("CONDA_PREFIX"):
            vendor_dirs.append(Path(env["CONDA_PREFIX"]) / "etc" / "OpenCL" / "vendors")
        for vendor_dir in vendor_dirs:
            if vendor_dir.exists():
                env["OCL_ICD_VENDORS"] = str(vendor_dir)
                break
    return env


def _prepare_single_ligand(args: Tuple[str, Path, Path]) -> Tuple[str, Path | None]:
    """Prepare a single ligand for docking.
    
    Args:
        args: Tuple of (smiles, ligand_dir, tmp_path)
        
    Returns:
        Tuple of (smiles, pdbqt_path) if successful, (smiles, None) if failed
    """
    smiles, ligand_dir, tmp_path = args
    name = _ligand_id(smiles)
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
    if not smiles_list:
        return {}

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
    
    docking_path = get_docking_path()
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
opencl_binary_path = {docking_path}/Vina-GPU-2.1/QuickVina2-GPU-2.1""")
    
    """ # Debug: Print ligand directory contents
    print("\nLigand directory contents:")
    for file in (tmp_path / "ligands").glob('*'):
        print(f"  {file.name}") """
    
    
    # Run QuickVina2-GPU
    vina_dir = docking_path / "Vina-GPU-2.1" / "QuickVina2-GPU-2.1"
    result = subprocess.run(['./QuickVina2-GPU-2-1', '--config', str(config_path.absolute())], 
                          cwd=vina_dir,
                          env=_quickvina_env(),
                          capture_output=True,
                          text=True)
    
    #if result.returncode != 0:
    #    print(f"\nQuickVina2-GPU failed:")
    #    print(f"stdout: {result.stdout}")
    #    print(f"stderr: {result.stderr}")
        
    scores = {}
    for smiles, pdbqt_path in smiles_to_pdbqt.items():
        output_path = output_dir / f"{pdbqt_path.stem}_out.pdbqt"
        try:
            with open(output_path) as f:
                for line in f:
                    if "REMARK VINA RESULT" in line:
                        score = float(line.split()[3])
                        scores[smiles] = score
                        break
        except:
            scores[smiles] = 0.0 
        scores.setdefault(smiles, 0.0)
            
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
    if not child_nodes_mol:
        return {}, {}

    os.makedirs("./tmp", exist_ok=True)
    with tempfile.TemporaryDirectory(dir="./tmp") as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # SMILES from each node
        node_to_smiles = {node: molecules[0] for node, molecules in child_nodes_mol.items()}
        unique_smiles = list(set(node_to_smiles.values()))
        
        smiles_to_pdbqt = _prepare_ligands(unique_smiles, tmp_path, n_proc=n_proc)
        
        docking_tasks = get_docking_tasks(target, smiles_to_pdbqt, tmp_path)
        
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
        ds1_scores = {node: results[0].get(smiles, 0.0) for node, smiles in node_to_smiles.items()}
        ds2_scores = {node: results[1].get(smiles, 0.0) for node, smiles in node_to_smiles.items()}
        
        return ds1_scores, ds2_scores
