# Our code was adapted from the implementation of the following paper:
# Swanson, K. et al. (2024). Generative AI for designing and validating easily synthesizable and structurally novel antibiotics. Nature Machine Intelligence, 6(3), 338-353.
# The original implementation can be found at: https://github.com/swansonk14/SyntheMol/ 

"""Generate molecules combinatorially using a multi-objective Monte Carlo tree search."""
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from tap import tapify

from pmcts.reactions import (
    Reaction,
    REACTIONS,
    QueryMol,
    load_and_set_allowed_reaction_building_blocks,
    set_all_building_blocks
)
from pmcts.config import get_target_pair_mapping_path
from pmcts.generate.generator import Generator
from pmcts.generate.utils import create_model_scoring_fn, save_generated_molecules


def _require_columns(data: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {', '.join(missing)}")


def _clone_reactions(reactions: tuple[Reaction]) -> tuple[Reaction, ...]:
    """Clone reaction templates so per-run allowed building blocks do not mutate globals."""

    return tuple(
        Reaction(
            reactants=[QueryMol(reactant.smarts_with_atom_mapping) for reactant in reaction.reactants],
            product=QueryMol(reaction.product.smarts_with_atom_mapping),
            reaction_id=reaction.id,
        )
        for reaction in reactions
    )


def generate(
        model_path: Path,
        save_dir: Path,
        building_blocks_path: Path,
        reaction_to_building_blocks_path: Path | None = None,
        building_blocks_id_column: str = "reagent_id",
        target_activities: List[str] = ["gsk3b_activity", "jnk3_activity"],
        building_blocks_smiles_column: str = "smiles",
        reactions: tuple[Reaction] = REACTIONS,
        max_reactions: int = 1, # Equivalent to explore REAL Space with 1 reaction
        n_rollout: int = 10,
        save_freq: int = 1000,
        target_pair: str = 'gsk3b_jnk3',
        explore_weight: float = 10.0,
        pareto_function: str | None = "pmcts",
        num_expand_nodes: int | None = None,
        rng_seed: int = 0,
        no_building_block_diversity: bool = False,  
        store_nodes: bool = False,
        verbose: bool = False,
        replicate: bool = False, 
        qed_sa: bool = False,
        scalarize: bool = False,
        all_objectives: bool = False,
        n_proc: int = 48,
        sequential: bool = False
) -> None:
    """Generate molecules combinatorially using a multi-objective Monte Carlo tree search.

    :param model_path: Path to a directory of model checkpoints or to a specific PKL or PT file containing a trained model.
    :param building_blocks_path: Path to CSV file containing molecular building blocks.
    :param save_dir: Path to directory where the generated molecules will be saved.
    :param reaction_to_building_blocks_path: Path to PKL file containing mapping from REAL reactions to allowed building blocks.
    :param building_blocks_id_column: Name of the column containing IDs for each building block.
    :param building_blocks_score_columns: List of column names containing scores for each objective.
    :param building_blocks_smiles_column: Name of the column containing SMILES for each building block.
    :param reactions: A tuple of reactions that combine molecular building blocks.
    :param max_reactions: Maximum number of reactions that can be performed to expand building blocks into molecules.
    :param n_rollout: The number of times to run the generation process.
    :param save_freq: The frequency with which to save generated molecules.
    :param target_pair: The target pair for docking.
    :param explore_weight: The hyperparameter that encourages exploration.
    :param pareto_function: The function to use to determine Pareto optimality.
    :param num_expand_nodes: The number of child nodes to include when expanding a given node.
    :param rng_seed: Seed for random number generators.
    :param no_building_block_diversity: Whether to turn off the score modification that encourages diverse building blocks.
    :param store_nodes: Whether to store in memory all the nodes of the search tree.
    :param verbose: Whether to print out additional information during generation.
    :param replicate: This is necessary to replicate the results from the paper, but otherwise should not be used.
    :param qed_sa: Whether to use QED and SA scores as objectives.
    :param scalarize: Whether to use scalarization to combine objectives.
    :param all_objectives: Whether to use all objectives (activity1, activity2, dockingscore1, dockingscore2, qed, sa)
    :param n_proc: Number of processes to use for ligand preparation.
    :param sequential: Whether to run docking tasks sequentially.
    """
    if sum([bool(qed_sa), bool(scalarize), bool(all_objectives)]) > 1:
        raise ValueError("Use only one of qed_sa, scalarize, or all_objectives at a time.")

    reactions = _clone_reactions(reactions)

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load building blocks
    print('Loading building blocks...')
    building_block_data = pd.read_csv(building_blocks_path)

    print(f'Loaded {len(building_block_data):,} building blocks')

    base_columns = [building_blocks_id_column, building_blocks_smiles_column, *target_activities]
    docking_columns = ["ds_1", "ds_2"]
    if qed_sa:
        _require_columns(building_block_data, base_columns, "QED/SA generation")
    else:
        _require_columns(building_block_data, [*base_columns, *docking_columns], "activity/docking generation")

    # Ensure unique building block IDs
    if building_block_data[building_blocks_id_column].nunique() != len(building_block_data):
        raise ValueError('Building block IDs are not unique.')

    # Map building blocks SMILES to IDs, IDs to SMILES, and SMILES to scores
    building_block_smiles_to_id = dict(zip(
        building_block_data[building_blocks_smiles_column],
        building_block_data[building_blocks_id_column]
    ))
    building_block_id_to_smiles = dict(zip(
        building_block_data[building_blocks_id_column],
        building_block_data[building_blocks_smiles_column]
    ))

    building_block_smiles_to_activities = {
        0: dict(zip(
            building_block_data[building_blocks_smiles_column],
            building_block_data[target_activities[0]])), # activity1
        1: dict(zip(
            building_block_data[building_blocks_smiles_column],
            building_block_data[target_activities[1]])) # activity2
    }

    if qed_sa:
        building_block_smiles_to_docking_scores = [{}, {}]
    else:
        building_block_smiles_to_docking_scores = {
            0: dict(zip(
                building_block_data[building_blocks_smiles_column],
                building_block_data["ds_1"])), # dockingscore1
            1: dict(zip(
                building_block_data[building_blocks_smiles_column],
                building_block_data["ds_2"])) # dockingscore2
        }

    print(f'Found {len(building_block_smiles_to_id):,} unique building blocks')

    # Set all building blocks for each reaction
    set_all_building_blocks(
        reactions=reactions,
        building_blocks=set(building_block_smiles_to_id)
    )

    # Optionally, set allowed building blocks for each reaction
    if reaction_to_building_blocks_path is None:
        reaction_to_building_blocks_path = get_target_pair_mapping_path(target_pair)
    else:
        reaction_to_building_blocks_path = Path(reaction_to_building_blocks_path)
        if not reaction_to_building_blocks_path.exists():
            raise FileNotFoundError(f"Reaction mapping does not exist: {reaction_to_building_blocks_path}")

    if reaction_to_building_blocks_path.exists():
        print('Loading and setting allowed building blocks for each reaction...')
        load_and_set_allowed_reaction_building_blocks(
            reactions=reactions,
            reaction_to_reactant_to_building_blocks_path=reaction_to_building_blocks_path
        )

    # Define model scoring function
    print('Loading models and creating model scoring function...')
    model_scoring_fn = create_model_scoring_fn(
        model_path=model_path,
        model_type="chemprop",
        smiles_to_scores=building_block_smiles_to_activities # dict[dict[smiles:act1], dict[smiles:act2]]
    )
    if qed_sa or all_objectives:
        if qed_sa:
            print('Using QED and SA as objectives - No docking scores will be used')
        else:
            print('Using all objectives: activity1, activity2, dockingscore1, dockingscore2, qed, sa')
        if "qed" in building_block_data.columns and "sa" in building_block_data.columns:
            building_block_smiles_to_qed_sa = {
                0: dict(zip(
                    building_block_data[building_blocks_smiles_column],
                    building_block_data["qed"])),
                1: dict(zip(
                    building_block_data[building_blocks_smiles_column],
                    building_block_data["sa"]))
            }
        else:
            building_block_smiles_to_qed_sa = None
    else:
        building_block_smiles_to_qed_sa = None

    # Set up Generator
    print('Setting up generator...')
    generator = Generator(
        building_block_smiles_to_id=building_block_smiles_to_id,
        building_block_id_to_smiles=building_block_id_to_smiles,
        max_reactions=max_reactions,
        scoring_fn=model_scoring_fn,
        precomp_ds=building_block_smiles_to_docking_scores,
        target_pair=target_pair,
        explore_weight=explore_weight,
        pareto_function=pareto_function,
        num_expand_nodes=num_expand_nodes,
        reactions=reactions,
        rng_seed=rng_seed,
        no_building_block_diversity=no_building_block_diversity,
        store_nodes=store_nodes,
        verbose=verbose,
        replicate=replicate,
        qed_sa=qed_sa,
        scalarize=scalarize,
        all_objectives=all_objectives,
        building_block_smiles_to_qed_sa=building_block_smiles_to_qed_sa,
        n_proc= n_proc,
        sequential=sequential
    )

    # Search for molecules
    print('Generating molecules...')
    start_time = datetime.now()
    nodes = generator.generate(n_rollout=n_rollout, save_dir=save_dir ,save_freq=save_freq)

    # Save generated molecules
    print('Saving molecules...')
    save_generated_molecules(
        nodes=nodes,
        building_block_id_to_smiles=building_block_id_to_smiles,
        save_path=save_dir / 'pareto_molecules.csv',
        qed_sa=qed_sa,
        scalarize=scalarize,
        all_objectives=all_objectives
    )

    # Compute, print, and save stats
    stats = {
        'mcts_time': datetime.now() - start_time,
        'num_nonzero_reaction_molecules': len(nodes),
        'approx_num_nodes_searched': generator.approx_num_nodes_searched,
        'num_pareto_optimal': sum(1 for node in nodes if node.is_pareto_optimal)
    }

    print(f'MCTS time = {stats["mcts_time"]}')
    print(f'Number of full molecule, nonzero reaction nodes = {stats["num_nonzero_reaction_molecules"]:,}')
    print(f'Number of Pareto optimal molecules = {stats["num_pareto_optimal"]:,}')
    print(f'Approximate total number of nodes searched = {stats["approx_num_nodes_searched"]:,}')

    if store_nodes:
        stats['num_nodes_searched'] = generator.num_nodes_searched
        print(f'Total number of nodes searched = {stats["num_nodes_searched"]:,}')

    pd.DataFrame(data=[stats]).to_csv(save_dir / 'mcts_stats.csv', index=False)


def generate_command_line() -> None:
    """Run generate function from command line."""
    tapify(generate)
