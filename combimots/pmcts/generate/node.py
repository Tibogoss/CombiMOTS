"""Contains the Node class, which represents a step in the combinatorial molecule construction process."""
import math
from functools import cached_property
from typing import Any, Callable, List
import numpy as np
from paretoset import paretoset


class Node:
    """A Node represents a step in the combinatorial molecule construction process."""

    def __init__(
            self,
            explore_weight: float,
            pareto_function: str,
            scoring_fn: Callable[[str], List[float]],
            node_id: int | None = None,
            molecules: tuple[str] | None = None,
            unique_building_block_ids: set[int] | None = None,
            construction_log: tuple[dict[str, Any]] | None = None,
            rollout_num: int | None = None,
            scalarize: bool = False,
            all_objectives: bool = False
    ) -> None:
        """Initializes the Node.

        :param explore_weight: The hyperparameter that encourages exploration.
        :param scoring_fn: A function that takes a list of SMILES and returns a list of score lists.
                         Each score list contains [QED, SA, activity1, activity2, dockingscore1, dockingscore2].
        :param pareto_function: Pareto UCB formula to use ["aizynthfinder" by default or "pmcts"].
        :param node_id: The ID of the Node, which should correspond to the order in which the Mode was visited.
        :param molecules: A tuple of SMILES. The first element is the currently constructed molecule
                          while the remaining elements are the building blocks that are about to be added.
        :param unique_building_block_ids: A set of building block IDS used in this Node.
        :param construction_log: A tuple of dictionaries containing information about each reaction
                                 used to construct the molecules in this Node.
        :param rollout_num: The number of the rollout on which this Node was created.
        :param scalarize: Whether to use scalarization for multi-objective optimization.
        :param all_objectives: Whether to use the six-objective settings (activities, docking scores, qed, sa).
        """
        self.explore_weight = explore_weight
        self.pareto_function = pareto_function
        self.scoring_fn = scoring_fn # Returns [act1, act2]
        self.node_id = node_id
        self.molecules = molecules if molecules is not None else tuple()
        self.unique_building_block_ids = unique_building_block_ids if unique_building_block_ids is not None else set()
        self.construction_log = construction_log if construction_log is not None else tuple()
        self.all_objectives = all_objectives
        self.scalarize = scalarize
        if scalarize:
            self.P = np.zeros(1) # Scalarized setting, 1 objective and we save the docking scores for stats
            self.W = np.zeros(1)
            self.save_scores = np.zeros(4)
        elif all_objectives:
            self.P = np.zeros(6) # 6 objective setting
            self.W = np.zeros(6)
        else:
            self.P = np.zeros(4) # 4 objective setting
            self.W = np.zeros(4)  
        self.N = 0  
        self.rollout_num = rollout_num
        self.num_children = 0
        self.is_pareto_optimal = False

    @staticmethod
    def is_pareto_efficient(scores: np.ndarray) -> np.ndarray:
        """Find the pareto-efficient points among a set of points using the paretoset library.
        
        :param scores: An (n_points, n_objectives) array where higher is better for all objectives
                    (activities and negated docking scores)
        :return: A boolean array of the same length as scores indicating whether each point is Pareto efficient
        """
        return paretoset(scores, sense=["max"] * scores.shape[1])

    @classmethod
    def compute_score(cls, molecules: tuple[str], scoring_fn: Callable[[str], List[float]]) -> np.ndarray:
        """Computes the scores of the molecules.

        :param molecules: A tuple of SMILES. The first element is the currently constructed molecule
                          while the remaining elements are the building blocks that are about to be added.
        :param scoring_fn: A function that takes a SMILES and returns a list of scores.
        :return: The average scores of the molecules as a numpy array.
        """
        if len(molecules) == 0:
            if self.scalarize:
                return np.zeros(1)
            elif self.all_objectives:
                return np.zeros(6)
            return np.zeros(4)
        
        scores = [scoring_fn(molecule) for molecule in molecules]
        avg_act1 = np.mean([score[0] for score in scores])
        avg_act2 = np.mean([score[1] for score in scores])

        return np.array([avg_act1, avg_act2])


    def Q(self) -> np.ndarray:
        """Value that encourages exploitation of Nodes with high reward."""
        if self.scalarize:
            return self.W / self.N if self.N > 0 else np.zeros(1)
        elif self.all_objectives:
            return self.W / self.N if self.N > 0 else np.zeros(6)
        return self.W / self.N if self.N > 0 else np.zeros(4)

    def U(self, n: int) -> np.ndarray:
        """Value that encourages exploration of Nodes with few visits.
        Modified to include ln(nb_objectives=4) in the exploration term for multi-objective optimization.
        """

        if self.pareto_function == "aizynthfinder":
            if self.scalarize:
                return self.explore_weight * self.P * (math.sqrt(math.log(1 + n)) / (1 + self.N))
            elif self.all_objectives:
                return self.explore_weight * self.P * (math.sqrt(math.log(6) + math.log(1 + n)) / (1 + self.N))
            return self.explore_weight * self.P * (math.sqrt(math.log(4) + math.log(1 + n)) / (1 + self.N)) #aizynthfinder-style
        
        elif self.pareto_function == "pmcts":
            if self.scalarize:
                return self.explore_weight * self.P * (math.sqrt(math.log(1 + n)) / (1 + self.N))
            elif self.all_objectives:
                return self.explore_weight * self.P * (math.sqrt(math.log(6) + 4 * math.log(1 + n) / (2 * (1 + self.N))))
            return self.explore_weight * self.P * (math.sqrt(math.log(4) + 4 * math.log(1 + n) / (2 * (1 + self.N)))) #pmcts-style
        

    def get_ucb_score(self, n: int) -> np.ndarray:
        """Get the UCB score vector for this node."""
        q_score = self.Q()
        u_score = self.U(n)
        return q_score + u_score

    @property
    def num_molecules(self) -> int:
        """Gets the number of building blocks in the Node."""
        return len(self.molecules)

    @property
    def num_reactions(self) -> int:
        """Gets the number of reactions used so far to generate the molecule in the Node."""
        return len(self.construction_log)

    def __hash__(self) -> int:
        """Hashes the Node based on the building blocks."""
        return hash(self.molecules)

    def __eq__(self, other: Any) -> bool:
        """Checks if the Node is equal to another Node based on the building blocks."""
        if not isinstance(other, Node):
            return False

        return self.molecules == other.molecules
