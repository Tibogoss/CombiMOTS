"""pmcts.reactions package."""
from pmcts.reactions.custom import CUSTOM_REACTIONS
from pmcts.reactions.query_mol import QueryMol
from pmcts.reactions.reaction import Reaction
from pmcts.reactions.real import REAL_REACTIONS
from pmcts.reactions.utils import (
    load_and_set_allowed_reaction_building_blocks,
    set_all_building_blocks
)

if CUSTOM_REACTIONS is None:
    REACTIONS: tuple[Reaction] = REAL_REACTIONS
else:
    REACTIONS: tuple[Reaction] = CUSTOM_REACTIONS
