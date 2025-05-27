""" Adapted from the implementation of:
https://github.com/swansonk14/SyntheMol/ """

"""SMARTS representations custom reactions."""
from pmcts.reactions.query_mol import QueryMol
from pmcts.reactions.reaction import Reaction

# To use custom chemical reactions instead of Enamine REAL reactions, replace None with a list of Reaction objects.
# If CUSTOM_REACTIONS is None, pmcts will default to the reactions in real.py.
CUSTOM_REACTIONS: tuple[Reaction] | None = None
