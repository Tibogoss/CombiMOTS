"""Compatibility wrapper for filtering REAL building blocks by templates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "combimots"))


def filter_real_reactions_to_building_blocks(
    reaction_to_building_blocks_path: Path,
    save_path: Path,
    original_reaction_to_building_blocks_path: Path | None = None,
    report_path: Path | None = None,
) -> None:
    """Filter a SMILES-set reaction mapping by reaction template compatibility."""

    from pmcts.constants import REACTION_TO_BUILDING_BLOCKS_PATH
    from pmcts.reactions import REACTIONS
    from preprocess.search_space import filter_mapping_to_reaction_templates

    filter_mapping_to_reaction_templates(
        reaction_to_building_blocks_path=reaction_to_building_blocks_path,
        save_path=save_path,
        reactions=REACTIONS,
        original_reaction_to_building_blocks_path=original_reaction_to_building_blocks_path or REACTION_TO_BUILDING_BLOCKS_PATH,
        report_path=report_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter REAL mapping by reaction templates")
    parser.add_argument("--reaction_to_building_blocks_path", "--reaction-to-building-blocks-path", type=Path, required=True)
    parser.add_argument("--save_path", "--save-path", type=Path, required=True)
    parser.add_argument("--original_reaction_to_building_blocks_path", "--original-reaction-to-building-blocks-path", type=Path, default=None)
    parser.add_argument("--report_path", "--report-path", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filter_real_reactions_to_building_blocks(
        reaction_to_building_blocks_path=args.reaction_to_building_blocks_path,
        save_path=args.save_path,
        original_reaction_to_building_blocks_path=args.original_reaction_to_building_blocks_path,
        report_path=args.report_path,
    )
