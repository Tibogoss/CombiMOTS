# Helper functions are adapted from https://github.com/swansonk14/SyntheMol/

"""Generator class for multi-objective molecular optimization."""
import itertools
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Callable, List, Dict, Tuple
from pathlib import Path

import numpy as np
from rdkit import Chem
from tqdm import trange

from pmcts.generate.node import Node
from pmcts.docking.docking_utils import batch_dock
from pmcts.reactions import Reaction
from pmcts.utils import random_choice
from pmcts.generate.utils import save_generated_molecules

# from tdc import Oracle


class Generator:
    """Dual-target molecule generator using multi-objective Pareto MCTS."""

    def __init__(
            self,
            building_block_smiles_to_id: dict[str, int],
            building_block_id_to_smiles: dict[int, str],
            max_reactions: int,
            scoring_fn: Callable[[str], List[float]],  # activity prediction
            precomp_ds: List[dict[str, float]],
            target_pair: str,
            explore_weight: float,
            pareto_function: str | None,
            num_expand_nodes: int | None,
            reactions: tuple[Reaction],
            rng_seed: int,
            no_building_block_diversity: bool,
            store_nodes: bool,
            verbose: bool,
            replicate: bool = False,
            qed_sa: bool = False,
            scalarize: bool = False,
            all_objectives: bool = False,
            building_block_smiles_to_qed_sa: List[dict[str, float]] | None = None,
            n_proc: int = 48,
            sequential: bool = False
    ) -> None:
        """Creates the Generator.

        :param building_block_smiles_to_id: A dictionary mapping building block SMILES to their IDs.
        :param building_block_id_to_smiles: A dictionary mapping building block IDs to their SMILES.
        :param max_reactions: The maximum number of reactions to use to construct a molecule.
        :param scoring_fn: A function that takes as input a SMILES and returns a list of scores.
        :param precomp_ds: Dictionaries of precomputed docking scores for both targets.
        :param target_pair: Target pair for docking. Options: 'gsk3b_jnk3', 'dhodh_rorgt'
        :param explore_weight: The hyperparameter that encourages exploration.
        :param pareto_function: The function to use to determine Pareto optimality.
        :param num_expand_nodes: The number of tree nodes to expand when extending the child nodes in the search tree.
                                  If None, then all nodes are expanded.
        :param reactions: A tuple of reactions that combine molecular building blocks.
        :param rng_seed: Seed for the random number generator.
        :param no_building_block_diversity: Whether to turn off the score modification that encourages diverse building blocks.
        :param store_nodes: Whether to store the child nodes of each node in the search tree.
        :param verbose: Whether to print out additional statements during generation.
        :param replicate: This is necessary to replicate the results from the paper, but otherwise should not be used.
        :param qed_sa: Whether to use QED and SA scores as objectives.
        :param scalarize: Whether to use scalarization to combine objectives.
        :param all_objectives: Whether to use all objectives for optimization.
        :param building_block_smiles_to_qed_sa: A list of dictionaries mapping building block SMILES to their QED and SA scores.
        :param n_proc: Number of processes to use for ligand preparation.
        :param sequential: Whether to run docking tasks sequentially.
        """
        self.building_block_smiles_to_id = building_block_smiles_to_id
        self.building_block_id_to_smiles = building_block_id_to_smiles
        self.max_reactions = max_reactions
        self.scoring_fn = scoring_fn
        self.precomp_ds = precomp_ds
        self.target_pair = target_pair
        self.explore_weight = explore_weight
        self.pareto_function = pareto_function
        self.num_expand_nodes = num_expand_nodes
        self.reactions = reactions
        self.rng = np.random.default_rng(seed=rng_seed)
        self.building_block_diversity = not no_building_block_diversity
        self.store_nodes = store_nodes
        self.verbose = verbose
        self.qed_sa = qed_sa
        self.scalarize = scalarize
        self.all_objectives = all_objectives
        self.building_block_smiles_to_qed_sa = building_block_smiles_to_qed_sa
        self.n_proc = n_proc
        self.sequential = sequential
        
        # Cache for QED/SA scores to avoid repeated calculations
        self.qed_cache = {}
        self.sa_cache = {}

        if qed_sa or all_objectives:
            self.oracle_qed = Oracle(name='QED')
            self.oracle_sa = Oracle(name='SA')
            
            # Precompute QED and SA for building blocks
            if building_block_smiles_to_qed_sa:
                self.qed_cache = building_block_smiles_to_qed_sa[0].copy()
                self.sa_cache = building_block_smiles_to_qed_sa[1].copy()

        # Get all building blocks that are used in at least one reaction
        if replicate:
            self.all_building_blocks = list(dict.fromkeys(
                building_block
                for reaction in self.reactions
                for reactant in reaction.reactants
                for building_block in reactant.allowed_building_blocks
            ))
        else:
            self.all_building_blocks = sorted(
                building_block
                for building_block in self.building_block_smiles_to_id
                if any(reactant.has_match(building_block) for reaction in reactions for reactant in reaction.reactants)
            )

        # Initialize the root node
        self.rollout_num = 0
        self.root = Node(
            explore_weight=explore_weight,
            pareto_function=pareto_function,
            scoring_fn=scoring_fn,
            node_id=0,
            rollout_num=self.rollout_num,
            scalarize=self.scalarize,
            all_objectives=all_objectives
        )

        # Initialize the rollout num, node map, building block counts, and node to children
        self.node_map: dict[Node, Node] = {self.root: self.root}
        self.building_block_counts = Counter()
        self.node_to_children: dict[Node, list[Node]] = {}
        
        # Track visited states to prevent cycles
        self.visited_states = set()
        
        # lookup tables
        self.reaction_to_reactants = defaultdict(list)
        self.reactant_to_building_blocks = defaultdict(list)
        
        for reaction in self.reactions:
            for i, reactant in enumerate(reaction.reactants):
                self.reaction_to_reactants[reaction.id].append((i, reactant))
                for bb in reactant.allowed_building_blocks:
                    self.reactant_to_building_blocks[(reaction.id, i)].append(bb)

    def calculate_qed_sa_scores(self, molecules: tuple[str]) -> tuple[float, float]:
        """Calculate QED and SA scores for a set of molecules efficiently with caching.
        
        :param molecules: Tuple of SMILES strings
        :return: Tuple of (avg_qed, avg_sa)
        """
        qed_values = []
        sa_values = []
        uncached_molecules = []
        
        # check cache
        for mol in molecules:
            if mol in self.qed_cache:
                qed_values.append(self.qed_cache[mol])
                sa_values.append(self.sa_cache[mol])
            else:
                uncached_molecules.append(mol)
        
        # uncached molecules
        if uncached_molecules:
            new_qed_values = self.oracle_qed(uncached_molecules)
            new_sa_values = self.oracle_sa(uncached_molecules)
            
            # update cache
            for mol, qed_val, sa_val in zip(uncached_molecules, new_qed_values, new_sa_values):
                self.qed_cache[mol] = qed_val
                self.sa_cache[mol] = sa_val
                qed_values.append(qed_val)
                sa_values.append(sa_val)
        
        avg_qed = sum(qed_values) / len(qed_values) if qed_values else 0
        avg_sa = sum(sa_values) / len(sa_values) if sa_values else 0
        
        return avg_qed, avg_sa

    def get_next_building_blocks(self, molecules: tuple[str]) -> list[str]:
        """Get the next building blocks that can be added to the given molecules.

        :param molecules: A tuple of SMILES strings representing the molecules to get the next building blocks for.
        :return: A list of SMILES strings representing the next building blocks that can be added to the given molecules.
        """
        # Initialize list of allowed building blocks
        available_building_blocks = []

        # Loop through each reaction
        for reaction in self.reactions:
            # Get indices of the reactants in this reaction
            reactant_indices = set(range(reaction.num_reactants))

            # Skip reaction if there's no room to add more reactants
            if len(molecules) >= reaction.num_reactants:
                continue

            # For each molecule, get a list of indices of reactants it matches
            reactant_matches_per_molecule = [
                reaction.get_reactant_matches(smiles=molecule)
                for molecule in molecules
            ]

            # Loop through products of reactant indices that the molecules match to
            # and for each product, if it matches to all separate reactants,
            # then include the missing reactants in the set of unfilled reactants
            for matched_reactant_indices in itertools.product(*reactant_matches_per_molecule):
                matched_reactant_indices = set(matched_reactant_indices)

                if len(matched_reactant_indices) == len(molecules):
                    for index in sorted(reactant_indices - matched_reactant_indices):
                        available_building_blocks += reaction.reactants[index].allowed_building_blocks

        # Remove duplicates but maintain order for reproducibility
        available_building_blocks = list(dict.fromkeys(available_building_blocks))

        return available_building_blocks

    def get_reactions_for_molecules(self, molecules: tuple[str]) -> list[tuple[Reaction, dict[str, int]]]:
        """Get all reactions that can be run on the given molecules.

        :param molecules: A tuple of SMILES strings representing the molecules to run reactions on.
        :return: A list of tuples, where each tuple contains a reaction and a dictionary mapping
                 the molecules to the indices of the reactants they match.
        """
        matching_reactions = []

        # Check each reaction to see if it can be run on the given molecules
        for reaction in self.reactions:
            # Skip reaction if the number of molecules doesn't match the number of reactants
            if len(molecules) != reaction.num_reactants:
                continue

            # For each molecule, get a list of indices of reactants it matches
            reactant_matches_per_molecule = [
                reaction.get_reactant_matches(smiles=molecule)
                for molecule in molecules
            ]

            # Include every assignment of molecules to reactants that fills all the reactants
            for matched_reactant_indices in itertools.product(*reactant_matches_per_molecule):
                if len(set(matched_reactant_indices)) == reaction.num_reactants:
                    molecule_to_reactant_index = dict(zip(molecules, matched_reactant_indices))
                    matching_reactions.append((reaction, molecule_to_reactant_index))

        return matching_reactions

    def run_all_reactions(self, node: Node, all_objectives: bool) -> list[Node]:
        """Run all possible reactions for the molecules in the Node and return the resulting product Nodes.

        :param node: A Node to run reactions for.
        :return: A list of Nodes for the products of the reactions.
        """
        # Get all reactions that are possible for the molecules in the Node
        matching_reactions = self.get_reactions_for_molecules(molecules=node.molecules)

        # Run all possible reactions and create Nodes for the products
        product_nodes = []
        product_set = set()
        for reaction, molecule_to_reactant_index in matching_reactions:
            # Put molecules in the right order for the reaction
            molecules = sorted(node.molecules, key=lambda frag: molecule_to_reactant_index[frag])

            # Run reaction
            products = reaction.run_reactants(molecules)

            if len(products) == 0:
                raise ValueError('Reaction failed to produce products.')

            assert all(len(product) == 1 for product in products)

            # Convert product mols to SMILES (and remove Hs)
            products = [Chem.MolToSmiles(Chem.RemoveHs(product[0])) for product in products]

            # Filter out products that have already been created and deduplicate
            products = list(dict.fromkeys(product for product in products if product not in product_set))
            product_set |= set(products)

            # Create reaction log
            reaction_log = {
                'reaction_id': reaction.id,
                'building_block_ids': tuple(
                    self.building_block_smiles_to_id.get(molecule, -1)
                    for molecule in molecules
                ),
            }

            product_nodes += [
                Node(
                    explore_weight=self.explore_weight,
                    pareto_function=self.pareto_function,
                    scoring_fn=self.scoring_fn,
                    molecules=(product,),
                    unique_building_block_ids=node.unique_building_block_ids,
                    construction_log=node.construction_log + (reaction_log,),
                    rollout_num=self.rollout_num,
                    scalarize=self.scalarize,
                    all_objectives=all_objectives
                )
                for product in products
            ]

        return product_nodes

    def get_child_nodes(self, node: Node, all_objectives: bool) -> list[Node]:
        """Get the child Nodes of a given Node.

        Child Nodes are created in two ways:
        1. By running all possible reactions on the molecules in the Node and creating Nodes with the products.
        2. By adding all possible next building blocks to the molecules in the Node and creating a new Node for each.

        :param node: The Node to get the child Nodes of.
        :return: A list of child Nodes of the given Node.
        """
        # Run all valid reactions on the current molecules to combine them into new molecules
        new_nodes = self.run_all_reactions(node=node, all_objectives=all_objectives)

        # Add all possible next building blocks to the current molecules in the Node
        if node.num_molecules == 0:
            next_building_blocks = self.all_building_blocks
        else:
            next_building_blocks = self.get_next_building_blocks(molecules=node.molecules)

        # Optionally, limit the number of next nodes
        if self.num_expand_nodes is not None and len(next_building_blocks) > self.num_expand_nodes:
            next_building_blocks = random_choice(
                rng=self.rng,
                array=next_building_blocks,
                size=self.num_expand_nodes,
                replace=False
            )

        # Convert next node molecule tuples into Node objects
        new_nodes += [
            Node(
                explore_weight=self.explore_weight,
                pareto_function=self.pareto_function,
                scoring_fn=self.scoring_fn,
                molecules=(next_building_block,) + node.molecules,
                unique_building_block_ids=node.unique_building_block_ids | {next_building_block},
                construction_log=node.construction_log,
                rollout_num=self.rollout_num,
                scalarize=self.scalarize,
                all_objectives=all_objectives
            )
            for next_building_block in next_building_blocks
        ]

        # Remove duplicates but maintain order for reproducibility
        new_nodes = list(dict.fromkeys(new_nodes))

        return new_nodes

    def select_from_pareto_front(self, nodes: list[Node], total_visit_count: int) -> Node:
        """Select a node from the Pareto front of nodes.
        
        :param nodes: List of nodes to select from
        :param total_visit_count: Total visit count for UCB calculation
        :return: Selected node from the Pareto front
        """
        # Get UCB scores for all nodes
        scores = []
        for node in nodes:
            ucb_score = node.get_ucb_score(n=total_visit_count)
            
            # Apply diversity penalty if enabled
            if self.building_block_diversity and node.num_molecules > 0:
                max_bb_count = max(
                    self.building_block_counts[bb_id]
                    for bb_id in node.unique_building_block_ids
                )
                diversity_factor = np.exp(-(max_bb_count - 1) / 100)
                ucb_score *= diversity_factor
                
            scores.append(ucb_score)
            
        scores = np.array(scores)
        
        # Find Pareto optimal nodes based on UCB scores
        pareto_mask = Node.is_pareto_efficient(scores)
        pareto_nodes = [node for node, is_pareto in zip(nodes, pareto_mask) if is_pareto]
        
        # If no Pareto optimal nodes found (shouldn't happen), return random node
        if not pareto_nodes:
            return self.rng.choice(nodes)
            
        # Select randomly from Pareto front
        return self.rng.choice(pareto_nodes)

    def rollout(self, node: Node) -> np.ndarray:
        """Performs a generation rollout.

        :param node: A Node representing the root of the generation.
        :return: The value (reward) vector of the rollout.
        """
        if self.verbose:
            print(f'Node {node.node_id} (rollout {self.rollout_num})')
            print(f'Molecules = {node.molecules}')
            print(f'Num molecules = {node.num_molecules}')
            print(f'Num unique building blocks = {len(node.unique_building_block_ids)}')
            print(f'Num reactions = {node.num_reactions}')
            print(f'Scores = {node.P}')
            print()

        # Stop the search if we've reached the maximum number of reactions
        if node.num_reactions >= self.max_reactions:
            if not np.array_equal(node.P, np.zeros(node.P.size)): # Properties are computed
                return node.P
            ############################################
            else: # Need to compute properties
                # QED/SA optimization mode
                if self.qed_sa:
                    activities = node.compute_score(molecules=node.molecules, scoring_fn=node.scoring_fn)
                    avg_qed, avg_sa = self.calculate_qed_sa_scores(node.molecules)
                    node.P = np.array([
                        activities[0], 
                        activities[1],
                        avg_qed, 
                        (10-avg_sa)/9  # -> [0, 1] maximization
                    ])
                    return node.P
                # Other modes
                elif self.scalarize:
                    node_mol = {}
                    node_mol[node] = node.molecules
                    activities = node.compute_score(molecules=node.molecules, scoring_fn=node.scoring_fn)
                    node_ds1, node_ds2 = batch_dock(node_mol, target=self.target_pair, n_proc=1, sequential=self.sequential)
                    node.P = np.array([(activities[0]+activities[1]-node_ds1[node]/20.0 -node_ds2[node]/20.0)/4]) # -> [0, 1] maximization
                    node.save_scores = np.array([activities[0],activities[1],node_ds1[node], node_ds2[node]])
                    return node.P
                elif self.all_objectives:
                    node_mol = {}
                    node_mol[node] = node.molecules
                    activities = node.compute_score(molecules=node.molecules, scoring_fn=node.scoring_fn)
                    avg_qed, avg_sa = self.calculate_qed_sa_scores(node.molecules)
                    node_ds1, node_ds2 = batch_dock(node_mol, target=self.target_pair, n_proc=1, sequential=self.sequential)
                    node.P = np.array([activities[0], 
                                activities[1],
                                -node_ds1[node]/20.0, 
                                -node_ds2[node]/20.0,
                                avg_qed,
                                (10-avg_sa)/9])
                    return node.P
                else:
                    node_mol = {}
                    node_mol[node] = node.molecules
                    node_ds1, node_ds2 = batch_dock(node_mol, target=self.target_pair, n_proc=1, sequential=self.sequential)
                    activities = node.compute_score(molecules=node.molecules, scoring_fn=node.scoring_fn)
                    node.P = np.array([activities[0], 
                                    activities[1],
                                    -node_ds1[node]/20.0, 
                                    -node_ds2[node]/20.0])
                    return node.P

        # If this node has already been visited and the children have been stored, get its children from the dictionary
        if node in self.node_to_children:
            child_nodes = self.node_to_children[node]

        # Otherwise, expand the node to get its children
        else:
            # Expand the node both by running reactions with the current molecules and adding new building blocks
            child_nodes = self.get_child_nodes(node=node, all_objectives=self.all_objectives)

            # Check the node map and merge with an existing node if available
            child_nodes = [self.node_map.get(new_node, new_node) for new_node in child_nodes]

            # Process nodes differently based on mode
            if self.qed_sa:
                # all nodes at once
                for child_node in child_nodes:
                    # Skip if already computed
                    if not np.array_equal(child_node.P, np.zeros(child_node.P.shape)):
                        continue
                        
                    activities = child_node.compute_score(molecules=child_node.molecules, scoring_fn=child_node.scoring_fn)
                    avg_qed, avg_sa = self.calculate_qed_sa_scores(child_node.molecules)
                    
                    # set properties
                    child_node.P = np.array([
                        activities[0],
                        activities[1],
                        avg_qed,
                        (10-avg_sa)/9
                    ])
            else:
                not_blocks_to_mol = {}    # dict[Node, tuple[str]] = node -> [molecule1, molecule2...]
                precomp_blocks = []     # List[Node]
                
                for child_node in child_nodes:
                    if child_node.molecules[0] not in self.precomp_ds[0]:
                        not_blocks_to_mol[child_node] = child_node.molecules
                    else:
                        precomp_blocks.append(child_node)

                # docking scores based on mode
                if self.all_objectives:
                    # compute for non-building blocks
                    children_ds1, children_ds2 = batch_dock(not_blocks_to_mol, target=self.target_pair, n_proc=self.n_proc, sequential=self.sequential)
                    
                    # add precomputed blocks' scores (lookup)
                    for block in precomp_blocks:
                        children_ds1[block] = self.precomp_ds[0][block.molecules[0]]
                        children_ds2[block] = self.precomp_ds[1][block.molecules[0]]
                    
                    # activities, qed, sa
                    for child_node in child_nodes:
                        if not np.array_equal(child_node.P, np.zeros(child_node.P.shape)):
                            continue
                            
                        activities = child_node.compute_score(molecules=child_node.molecules, scoring_fn=child_node.scoring_fn)
                        avg_qed, avg_sa = self.calculate_qed_sa_scores(child_node.molecules)
                        
                        child_node_ds1 = children_ds1[child_node]
                        child_node_ds2 = children_ds2[child_node]
                        
                        child_node.P = np.array([
                            activities[0],
                            activities[1],
                            -child_node_ds1/20.0,
                            -child_node_ds2/20.0,
                            avg_qed,
                            (10-avg_sa)/9
                        ])
                else:
                    # compute docking scores
                    children_ds1, children_ds2 = batch_dock(not_blocks_to_mol, target=self.target_pair, n_proc=self.n_proc, sequential=self.sequential)
                    
                    # precomputed blocks' scores (lookup)
                    for block in precomp_blocks:
                        children_ds1[block] = self.precomp_ds[0][block.molecules[0]]
                        children_ds2[block] = self.precomp_ds[1][block.molecules[0]]
                    
                    # assign scores
                    for child_node in child_nodes:
                        if not np.array_equal(child_node.P, np.zeros(child_node.P.shape)):
                            continue
                            
                        activities = child_node.compute_score(molecules=child_node.molecules, scoring_fn=child_node.scoring_fn)
                        
                        if self.scalarize:
                            ds1 = children_ds1[child_node]
                            ds2 = children_ds2[child_node]
                            child_node.P = np.array([(activities[0] + activities[1] - ds1/20.0 - ds2/20.0)/4])
                            child_node.save_scores = np.array([activities[0], activities[1], ds1, ds2])
                        else:
                            child_node.P = np.array([
                                activities[0],
                                activities[1],
                                -children_ds1[child_node]/20.0,
                                -children_ds2[child_node]/20.0
                            ])

            # Add complete molecules to the node map
            for child_node in child_nodes:
                if child_node.num_molecules == 1 and child_node not in self.node_map:
                    child_node.node_id = len(self.node_map)
                    self.node_map[child_node] = child_node
                    self.building_block_counts.update(child_node.unique_building_block_ids)
            
            # Save the number of children in order to maintain a total node count
            node.num_children = len(child_nodes)

            # If storing nodes, store the children
            if self.store_nodes:
                self.node_to_children[node] = child_nodes

        # If no new nodes were generated, return the current node's value
        if len(child_nodes) == 0:
            if node.num_molecules == 1:
                return node.P
            else:
                raise ValueError('Failed to expand a partially expanded node.')

        # Select a node from the Pareto front
        total_visit_count = sum(child_node.N for child_node in child_nodes)
        selected_node = self.select_from_pareto_front(child_nodes, total_visit_count)

        # Check the node map and merge with an existing node if available
        if selected_node in self.node_map:
            selected_node = self.node_map[selected_node]
        # Otherwise, assign node ID and add to node map
        else:
            selected_node.node_id = len(self.node_map)
            self.node_map[selected_node] = selected_node

        # Unroll the selected node
        v = self.rollout(node=selected_node)

        # Get max whole molecule (non-building block) score across rollouts as feedback
        if selected_node.num_molecules == 1 and node.num_reactions > 0:
            v = np.maximum(v, selected_node.P)

        # Update exploit score and visit count
        selected_node.W += v
        selected_node.N += 1

        return v

    def get_pareto_fronts(self, nodes: list[Node]) -> list[list[Node]]:
        """Get nodes sorted into Pareto fronts using dictionary-based approach.
        
        :param nodes: List of nodes to sort into fronts
        :return: List of lists, where each inner list contains nodes in a Pareto front
        """
        remaining_nodes = nodes.copy()
        fronts = []
        
        while remaining_nodes:
            # Get scores for remaining nodes
            scores = np.array([node.P for node in remaining_nodes])
            
            # Find Pareto optimal nodes in remaining set using paretoset
            pareto_mask = Node.is_pareto_efficient(scores)
            
            # Add nodes in current front
            current_front = [node for node, is_pareto in zip(remaining_nodes, pareto_mask) if is_pareto]
            fronts.append(current_front)
            
            # Remove nodes in current front from remaining nodes
            remaining_nodes = [node for node, is_pareto in zip(remaining_nodes, pareto_mask) if not is_pareto]
            
        return fronts

    def generate(self, n_rollout: int, save_dir: Path, save_freq: int = 1000) -> list[Node]:
        """Generate molecules for the specified number of rollouts.

        NOTE: Only returns Nodes with exactly one molecule and at least one reaction.
        Saves all results every 1000 rollouts.

        :param n_rollout: The number of rollouts to perform.
        :param save_freq: The frequency at which to save intermediate results.
        :param save_dir: The directory to save the results to.
        :return: A list of Node objects sorted by Pareto dominance.
        """
        # Set up rollout bounds
        rollout_start = self.rollout_num + 1
        rollout_end = rollout_start + n_rollout

        print(f'Running rollouts {rollout_start} through {rollout_end - 1}...')
        
        # (create) save directory
        save_dir.mkdir(parents=True, exist_ok=True)

        # Run the generation algorithm for the specified number of rollouts
        for rollout_num in trange(rollout_start, rollout_end):
            self.rollout_num = rollout_num
            self.rollout(node=self.root)

            # save intermediate results
            if (rollout_num - rollout_start + 1) % save_freq == 0:
                current_nodes = [
                    node
                    for _, node in self.node_map.items()
                    if node.num_molecules == 1 and node.num_reactions > 0 and (rollout_start <= node.rollout_num <= rollout_num)
                ]

                current_nodes.sort(key=lambda x: x.node_id)
                
                fronts = self.get_pareto_fronts(current_nodes)
            
                for node in current_nodes:
                    node.is_pareto_optimal = node in fronts[0] if fronts else False

                intermediate_save_path = save_dir / f'intermediate_rollout_{rollout_num}.csv'
                
                # save results up to this point
                save_generated_molecules(
                    nodes=current_nodes,
                    building_block_id_to_smiles=self.building_block_id_to_smiles,
                    save_path=intermediate_save_path,
                    qed_sa=self.qed_sa,
                    scalarize=self.scalarize,
                    all_objectives=self.all_objectives
                )

                print(f'\nSaved results at rollout {rollout_num} to {intermediate_save_path}')
                print(f'Current number of valid molecules: {len(current_nodes)}')

        # Nodes representing fully constructed molecules that are not building blocks within these rollouts
        nodes = [
            node
            for _, node in self.node_map.items()
            if node.num_molecules == 1 and node.num_reactions > 0 and (rollout_start <= node.rollout_num < rollout_end)
        ]

        # Sort nodes by node ID for reproducibility
        nodes.sort(key=lambda x: x.node_id)
        
        # Get all Pareto fronts
        fronts = self.get_pareto_fronts(nodes)
        
        # Return all nodes, but mark only first front as Pareto optimal
        for node in nodes:
            node.is_pareto_optimal = node in fronts[0] if fronts else False
            
        return nodes

    @property
    def approx_num_nodes_searched(self) -> int:
        """Gets the approximate number of nodes seen during the search.

        Note: This will over count any node that appears as a child node of multiple parent nodes.
        """
        return 1 + sum(node.num_children for node in self.node_map)

    @property
    def num_nodes_searched(self) -> int:
        """Gets the precise number of nodes seen during the search. Only possible if store_nodes is True."""
        if not self.store_nodes:
            raise ValueError('Cannot get the precise number of nodes searched if store_nodes is False.'
                                'Use approx_num_nodes_searched instead.')

        # Get a set of all nodes and child nodes that have been visited
        visited_nodes = set()
        for node, children in self.node_to_children.items():
            visited_nodes.add(node)
            visited_nodes.update(children)

        return len(visited_nodes)