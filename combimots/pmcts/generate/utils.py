"""Utility functions for generating molecules."""
from functools import cache
from pathlib import Path
from typing import Callable, List
import numpy as np
import pandas as pd
import torch

from pmcts.constants import MODEL_TYPES
from pmcts.generate.node import Node
from pmcts.models import (
    chemprop_load,
    chemprop_load_scaler,
    chemprop_predict_on_molecule_ensemble,
)

def create_model_scoring_fn(
        model_path: Path,  # To chemprop model
        model_type: MODEL_TYPES,
        smiles_to_scores: dict[int, dict[str, float]] | None = None  # Precomputed scores for each objective
) -> Callable[[str], List[float]]:
    """Creates a function that scores a molecule using a single model or ensemble.

    :param model_path: A path to a model or directory of models.
    :param model_type: The type of model.
    :param smiles_to_score: An optional dictionary mapping SMILES to precomputed scores.
    :return: A function that scores molecules using a model or ensemble of models.
    """
    # Get model paths
    if model_path.is_dir():
        model_paths = list(model_path.glob('**/*.pt' if model_type == 'chemprop' else '**/*.pkl'))

        if len(model_paths) == 0:
            raise ValueError(f'Could not find any models in directory {model_path}.')
    else:
        model_paths = [model_path]

    if model_type == 'chemprop':
        # Ensure reproducibility
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)

        models = [chemprop_load(model_path=model_path) for model_path in model_paths]
        scalers = [chemprop_load_scaler(model_path=model_path) for model_path in model_paths]

    # Create combined scoring function that returns list of scores
    @cache
    def multi_objective_scoring_fn(smiles: str | None = None) -> List[float]:
        """Score molecule on all objectives."""
        
        # If precomputed scores
        if smiles_to_scores is not None and smiles in smiles_to_scores[0]:
            act1 = smiles_to_scores[0][smiles]
            act2 = smiles_to_scores[1][smiles]
            return [act1, act2]
        
        act1, act2 = chemprop_predict_on_molecule_ensemble(models,
                                                        smiles = smiles,
                                                        scalers = scalers)
        
        return [act1, act2]

    return multi_objective_scoring_fn


def is_dominated(scores1: np.ndarray, scores2: np.ndarray) -> bool:
    """Check if scores1 is dominated by scores2.
    
    A solution is dominated if another solution is better or equal in all objectives
    and strictly better in at least one objective.
    
    Args:
        scores1: First solution's scores
        scores2: Second solution's scores
        
    Returns:
        True if scores1 is dominated by scores2, False otherwise
    """
    return np.all(scores2 >= scores1) and np.any(scores2 > scores1)


def get_pareto_front(scores: np.ndarray) -> np.ndarray:
    """Get indices of Pareto optimal solutions.
    
    Args:
        scores: Array of shape (n_solutions, n_objectives) containing scores for each solution
        
    Returns:
        Boolean array indicating which solutions are Pareto optimal
    """
    n_solutions = scores.shape[0]
    is_pareto = np.ones(n_solutions, dtype=bool)
    
    for i in range(n_solutions):
        if is_pareto[i]:
            for j in range(n_solutions):
                if i != j and is_pareto[j]:
                    if is_dominated(scores[i], scores[j]):
                        is_pareto[i] = False
                        break
                    elif is_dominated(scores[j], scores[i]):
                        is_pareto[j] = False
    return is_pareto


def get_pareto_fronts(scores: np.ndarray, k: int = None) -> List[np.ndarray]:
    """Get indices of solutions in each Pareto front, ranked from best to worst.
    
    Args:
        scores: Array of shape (n_solutions, n_objectives) containing scores for each solution
        k: Number of fronts to return (None for all fronts)
        
    Returns:
        List of arrays containing indices of solutions in each front
    """
    remaining_indices = np.arange(len(scores))
    remaining_scores = scores.copy()
    fronts = []
    
    while len(remaining_indices) > 0:
        is_pareto = get_pareto_front(remaining_scores)
        front_indices = remaining_indices[is_pareto]
        fronts.append(front_indices)
        
        # Remove solutions in current front
        remaining_indices = remaining_indices[~is_pareto]
        remaining_scores = remaining_scores[~is_pareto]
        
        if k is not None and len(fronts) == k:
            break
            
    return fronts


def save_generated_molecules(
        nodes: list[Node],
        building_block_id_to_smiles: dict[int, str],
        save_path: Path,
        qed_sa: bool = False,
        scalarize: bool = False,
        all_objectives: bool = False
) -> None:
    """Save generated molecules to a CSV file.

    :param nodes: A list of Nodes containing molecules. Only nodes with a single molecule are saved.
    :param building_block_id_to_smiles: A dictionary mapping building block IDs to SMILES.
    :param save_path: A path to a CSV file where the molecules will be saved.
    """
    # Only keep nodes with one molecule
    nodes = [node for node in nodes if node.num_molecules == 1]

    # Convert construction logs from lists to dictionaries
    construction_dicts = []
    max_reaction_num = 0
    reaction_num_to_max_reactant_num = {}

    for node in nodes:
        construction_dict = {'num_reactions': len(node.construction_log)}
        max_reaction_num = max(max_reaction_num, len(node.construction_log))

        for reaction_index, reaction_log in enumerate(node.construction_log):
            reaction_num = reaction_index + 1
            construction_dict[f'reaction_{reaction_num}_id'] = reaction_log['reaction_id']

            reaction_num_to_max_reactant_num[reaction_num] = max(
                reaction_num_to_max_reactant_num.get(reaction_num, 0),
                len(reaction_log['building_block_ids'])
            )

            for reactant_index, building_block_id in enumerate(reaction_log['building_block_ids']):
                reactant_num = reactant_index + 1
                construction_dict[f'building_block_{reaction_num}_{reactant_num}_id'] = building_block_id
                construction_dict[f'building_block_{reaction_num}_{reactant_num}_smiles'] = building_block_id_to_smiles.get(building_block_id, '')

        construction_dicts.append(construction_dict)

    # Specify column order for CSV file
    if scalarize:
        columns = ['smiles', 'node_id', 'num_expansions', 'rollout_num', 'score', 'activity1', 'activity2', 'dockingscore1', 'dockingscore2', 'num_reactions']
    elif qed_sa:
        columns = ['smiles', 'node_id', 'num_expansions', 'rollout_num', 'activity1', 'activity2', 'qed', 'sa', 'is_pareto_optimal', 'num_reactions']
    elif all_objectives:
        columns = ['smiles', 'node_id', 'num_expansions', 'rollout_num', 'activity1', 'activity2', 'dockingscore1', 'dockingscore2', 'qed', 'sa', 'is_pareto_optimal', 'num_reactions']
    else:
        columns = ['smiles', 'node_id', 'num_expansions', 'rollout_num', 'activity1', 'activity2', 'dockingscore1', 'dockingscore2', 'is_pareto_optimal', 'num_reactions']

    for reaction_num in range(1, max_reaction_num + 1):
        columns.append(f'reaction_{reaction_num}_id')

        for reactant_num in range(1, reaction_num_to_max_reactant_num[reaction_num] + 1):
            columns.append(f'building_block_{reaction_num}_{reactant_num}_id')
            columns.append(f'building_block_{reaction_num}_{reactant_num}_smiles')

    # Save data
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if scalarize:
        data = pd.DataFrame(
            data=[
                {
                    'smiles': node.molecules[0],
                    'node_id': node.node_id,
                    'num_expansions': node.N,
                    'rollout_num': node.rollout_num,
                    'score': node.P[0],
                    'activity1': node.save_scores[0],
                    'activity2': node.save_scores[1],
                    'dockingscore1': node.save_scores[2],
                    'dockingscore2': node.save_scores[3],
                    'num_reactions': construction_dict['num_reactions'],
                    **construction_dict
                }
                for node, construction_dict in zip(nodes, construction_dicts)
            ],
            columns=columns
        )
    elif qed_sa:
        data = pd.DataFrame(
            data=[
                {
                    'smiles': node.molecules[0],
                    'node_id': node.node_id,
                    'num_expansions': node.N,
                    'rollout_num': node.rollout_num,
                    'activity1': node.P[0],
                    'activity2': node.P[1],
                    'qed': node.P[2],
                    'sa': node.P[3],
                    'is_pareto_optimal': node.is_pareto_optimal,
                    **construction_dict
                }
                for node, construction_dict in zip(nodes, construction_dicts)
            ],
            columns=columns
        )
    elif all_objectives:
        data = pd.DataFrame(
            data=[
                {
                    'smiles': node.molecules[0],
                    'node_id': node.node_id,
                    'num_expansions': node.N,
                    'rollout_num': node.rollout_num,
                    'activity1': node.P[0],
                    'activity2': node.P[1],
                    'dockingscore1': node.P[2],
                    'dockingscore2': node.P[3],
                    'qed': node.P[4],
                    'sa': node.P[5],
                    'is_pareto_optimal': node.is_pareto_optimal,
                    **construction_dict
                }
                for node, construction_dict in zip(nodes, construction_dicts)
            ],
            columns=columns
        )

    else:
        data = pd.DataFrame(
            data=[
                {
                    'smiles': node.molecules[0],
                    'node_id': node.node_id,
                    'num_expansions': node.N,
                    'rollout_num': node.rollout_num,
                    'activity1': node.P[0],
                    'activity2': node.P[1],
                    'dockingscore1': node.P[2],
                    'dockingscore2': node.P[3],
                    'is_pareto_optimal': node.is_pareto_optimal,
                    **construction_dict
                }
                for node, construction_dict in zip(nodes, construction_dicts)
            ],
            columns=columns
        )

    # Get scores for Pareto analysis
    if scalarize:
        scores = np.array([[node.P[0]] for node in nodes])
    elif not all_objectives:
        scores = np.array([[node.P[0], node.P[1], node.P[2], node.P[3]] for node in nodes])
    else:
        scores = np.array([[node.P[0], node.P[1], node.P[2], node.P[3], node.P[4], node.P[5]] for node in nodes])
    
    # Get all Pareto fronts
    fronts = get_pareto_fronts(scores)
    
    # Save Pareto optimal solutions (first front)
    pareto_indices = fronts[0]
    pareto_data = data.iloc[pareto_indices].copy()
    pareto_data['pareto_rank'] = 1
    
    # Add solutions from subsequent fronts
    for rank, front_indices in enumerate(fronts[1:], start=2):
        front_data = data.iloc[front_indices].copy()
        front_data['pareto_rank'] = rank
        pareto_data = pd.concat([pareto_data, front_data])
    
    pareto_data.to_csv(save_path, index=False)
