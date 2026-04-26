"""Search-space mapping utilities for REAL building-block pickles."""

from __future__ import annotations

import json
import pickle
from math import prod
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from pmcts.reactions import Reaction
from preprocess.filters import canonicalize_smiles_value, filter_quickvina_compatible_smiles


ReactionMapping = dict[int, dict[int, set[str]]]


def canonicalize_smiles(smiles: str, cache: dict[str, str | None] | None = None) -> str | None:
    """Canonicalize a SMILES string with optional memoization."""

    if cache is not None and smiles in cache:
        return cache[smiles]

    canonical, _ = canonicalize_smiles_value(smiles)
    if cache is not None:
        cache[smiles] = canonical
    return canonical


def load_reaction_mapping(path: Path) -> ReactionMapping:
    """Load a REAL reaction mapping pickle and normalize values to sets."""

    with open(path, "rb") as f:
        raw_mapping = pickle.load(f)

    if not isinstance(raw_mapping, dict):
        raise ValueError(f"Reaction mapping must be a dictionary: {path}")

    mapping: ReactionMapping = {}
    for reaction_id, positions in raw_mapping.items():
        if not isinstance(positions, dict):
            raise ValueError(f"Reaction {reaction_id} must map reactant positions to SMILES sets")
        mapping[int(reaction_id)] = {
            int(position): set(building_blocks)
            for position, building_blocks in positions.items()
        }
    return mapping


def save_reaction_mapping(mapping: ReactionMapping, path: Path) -> None:
    """Save a reaction mapping pickle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(mapping, f)


def theoretical_product_count(mapping: ReactionMapping) -> int:
    """Compute the upper-bound number of reaction products represented by a mapping."""

    total = 0
    for positions in mapping.values():
        if not positions or any(len(blocks) == 0 for blocks in positions.values()):
            continue
        total += prod(len(blocks) for blocks in positions.values())
    return total


def reduce_mapping_to_csv_blocks(
    input_csv: Path,
    real_path: Path,
    save_path: Path,
    smiles_column: str = "smiles",
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Reduce a REAL SMILES-set mapping to blocks found in an input CSV.

    Any reaction position that would become empty falls back to the original
    position set, because generation requires at least one block per position.
    """

    print("Loading and processing custom building blocks...")
    custom_df = pd.read_csv(input_csv)
    if smiles_column not in custom_df.columns:
        raise ValueError(f"Input CSV must contain column '{smiles_column}': {input_csv}")

    canonical_cache: dict[str, str | None] = {}
    custom_smiles = {
        canonical
        for smiles in tqdm(custom_df[smiles_column], desc="Canonicalizing input blocks")
        if (canonical := canonicalize_smiles(str(smiles), canonical_cache)) is not None
    }

    if not custom_smiles:
        raise ValueError("No valid building blocks found in input file")

    print(f"Found {len(custom_smiles):,} valid canonical input building blocks")
    print(f"Loading reaction mapping from {real_path}")
    original_mapping = load_reaction_mapping(real_path)

    filtered_mapping: ReactionMapping = {}
    fallback_positions: list[dict[str, Any]] = []
    fallback_cleanup_metrics = _empty_cleanup_totals()
    matched_block_total = 0

    print("Filtering building blocks by CSV membership...")
    for reaction_id, positions in tqdm(original_mapping.items()):
        filtered_mapping[reaction_id] = {}
        for position, building_blocks in positions.items():
            filtered_blocks = set()
            for bb_smiles in building_blocks:
                canonical = canonicalize_smiles(bb_smiles, canonical_cache)
                if canonical in custom_smiles:
                    filtered_blocks.add(canonical)
            matched_block_total += len(filtered_blocks)

            if filtered_blocks:
                filtered_mapping[reaction_id][position] = filtered_blocks
            else:
                fallback_blocks, cleanup_metrics = filter_quickvina_compatible_smiles(building_blocks)
                _add_cleanup_totals(fallback_cleanup_metrics, cleanup_metrics)
                if not fallback_blocks:
                    raise ValueError(
                        f"Reaction {reaction_id} position {position} has no compatible fallback building blocks"
                    )
                filtered_mapping[reaction_id][position] = fallback_blocks
                fallback_positions.append({
                    "reaction_id": reaction_id,
                    "position": position,
                    "original_count": len(building_blocks),
                    "kept_compatible_count": len(fallback_blocks),
                    "reason": "no_csv_match",
                })

    save_reaction_mapping(filtered_mapping, save_path)

    report = {
        "input_csv": str(input_csv),
        "real_path": str(real_path),
        "save_path": str(save_path),
        "input_rows": len(custom_df),
        "valid_canonical_input_smiles": len(custom_smiles),
        "original_reactions": len(original_mapping),
        "filtered_reactions": len(filtered_mapping),
        "original_position_blocks": _position_block_total(original_mapping),
        "csv_matched_position_blocks": matched_block_total,
        "final_position_blocks_after_fallback": _position_block_total(filtered_mapping),
        "fallback_positions": fallback_positions,
        "fallback_cleanup": fallback_cleanup_metrics,
        "original_theoretical_products": theoretical_product_count(original_mapping),
        "final_theoretical_products": theoretical_product_count(filtered_mapping),
    }

    _print_reduction_report(report)
    _write_report(report, report_path)
    return report


def filter_mapping_to_reaction_templates(
    reaction_to_building_blocks_path: Path,
    save_path: Path,
    reactions: tuple[Reaction],
    original_reaction_to_building_blocks_path: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Filter a SMILES-set reaction mapping by reaction template compatibility."""

    reduced_mapping = load_reaction_mapping(reaction_to_building_blocks_path)
    original_mapping = load_reaction_mapping(original_reaction_to_building_blocks_path)
    reaction_by_id = {reaction.id: reaction for reaction in reactions}

    filtered_mapping: ReactionMapping = {}
    fallback_positions: list[dict[str, Any]] = []
    input_cleanup_metrics = _empty_cleanup_totals()
    fallback_cleanup_metrics = _empty_cleanup_totals()

    print("Filtering building blocks by reaction templates...")
    for reaction_id, reaction in tqdm(reaction_by_id.items()):
        if reaction_id not in original_mapping:
            raise ValueError(f"Original mapping is missing reaction {reaction_id}")

        source_positions = reduced_mapping.get(reaction_id, {})
        original_positions = original_mapping[reaction_id]
        filtered_mapping[reaction_id] = {}

        for reactant_index, reactant in enumerate(reaction.reactants):
            if reactant_index not in original_positions:
                raise ValueError(f"Original mapping is missing reaction {reaction_id} position {reactant_index}")

            source_blocks, cleanup_metrics = filter_quickvina_compatible_smiles(
                source_positions.get(reactant_index, set())
            )
            _add_cleanup_totals(input_cleanup_metrics, cleanup_metrics)
            if not source_blocks:
                source_blocks, cleanup_metrics = filter_quickvina_compatible_smiles(original_positions[reactant_index])
                _add_cleanup_totals(fallback_cleanup_metrics, cleanup_metrics)
                if not source_blocks:
                    raise ValueError(
                        f"Reaction {reaction_id} position {reactant_index} has no compatible fallback building blocks"
                    )
                fallback_positions.append({
                    "reaction_id": reaction_id,
                    "position": reactant_index,
                    "original_count": len(original_positions[reactant_index]),
                    "kept_compatible_count": len(source_blocks),
                    "reason": "missing_or_empty_input_position",
                })

            template_filtered = _filter_template_matches(source_blocks, reactant)
            if template_filtered:
                filtered_mapping[reaction_id][reactant_index] = template_filtered
                continue

            fallback_source_blocks, cleanup_metrics = filter_quickvina_compatible_smiles(original_positions[reactant_index])
            _add_cleanup_totals(fallback_cleanup_metrics, cleanup_metrics)
            fallback_blocks = _filter_template_matches(fallback_source_blocks, reactant)
            if not fallback_blocks:
                raise ValueError(
                    f"Reaction {reaction_id} position {reactant_index} has no template-compatible "
                    "building blocks after fallback cleanup"
                )

            filtered_mapping[reaction_id][reactant_index] = fallback_blocks
            fallback_positions.append({
                "reaction_id": reaction_id,
                "position": reactant_index,
                "original_count": len(original_positions[reactant_index]),
                "kept_compatible_count": len(fallback_blocks),
                "reason": "template_filter_empty",
            })

    save_reaction_mapping(filtered_mapping, save_path)

    report = {
        "input_path": str(reaction_to_building_blocks_path),
        "original_path": str(original_reaction_to_building_blocks_path),
        "save_path": str(save_path),
        "input_reactions": len(reduced_mapping),
        "filtered_reactions": len(filtered_mapping),
        "input_position_blocks": _position_block_total(reduced_mapping),
        "final_position_blocks_after_fallback": _position_block_total(filtered_mapping),
        "fallback_positions": fallback_positions,
        "input_cleanup": input_cleanup_metrics,
        "fallback_cleanup": fallback_cleanup_metrics,
        "input_theoretical_products": theoretical_product_count(reduced_mapping),
        "final_theoretical_products": theoretical_product_count(filtered_mapping),
    }

    _print_filter_report(report)
    _write_report(report, report_path)
    return report


def _filter_template_matches(building_blocks: set[str], reactant: Any) -> set[str]:
    """Return building blocks matching a reaction template, skipping invalid entries."""

    matches = set()
    for smiles in building_blocks:
        try:
            if reactant.has_substruct_match(smiles):
                matches.add(smiles)
        except Exception:
            continue
    return matches


def _position_block_total(mapping: ReactionMapping) -> int:
    return sum(len(blocks) for positions in mapping.values() for blocks in positions.values())


def _empty_cleanup_totals() -> dict[str, int]:
    return {
        "input_rows": 0,
        "output_rows": 0,
        "empty_smiles_rows": 0,
        "invalid_smiles_rows": 0,
        "disconnected_smiles_rows": 0,
        "forbidden_element_rows": 0,
        "duplicate_canonical_smiles_rows": 0,
    }


def _add_cleanup_totals(totals: dict[str, int], metrics: dict[str, int]) -> None:
    for key in totals:
        totals[key] += metrics.get(key, 0)


def _print_reduction_report(report: dict[str, Any]) -> None:
    print("\nSearch-space reduction statistics:")
    print(f"Original reactions: {report['original_reactions']:,}")
    print(f"Filtered reactions: {report['filtered_reactions']:,}")
    print(f"Original position blocks: {report['original_position_blocks']:,}")
    print(f"CSV-matched position blocks before fallback: {report['csv_matched_position_blocks']:,}")
    print(f"Final position blocks after fallback: {report['final_position_blocks_after_fallback']:,}")
    print(f"Fallback positions: {len(report['fallback_positions']):,}")
    print(f"Original theoretical products: {report['original_theoretical_products']:,}")
    print(f"Final theoretical products: {report['final_theoretical_products']:,}")
    print(f"Saved filtered mapping to {report['save_path']}")


def _print_filter_report(report: dict[str, Any]) -> None:
    print("\nReaction-template filter statistics:")
    print(f"Input reactions: {report['input_reactions']:,}")
    print(f"Filtered reactions: {report['filtered_reactions']:,}")
    print(f"Input position blocks: {report['input_position_blocks']:,}")
    print(f"Final position blocks after fallback: {report['final_position_blocks_after_fallback']:,}")
    print(f"Fallback positions: {len(report['fallback_positions']):,}")
    print(f"Input theoretical products: {report['input_theoretical_products']:,}")
    print(f"Final theoretical products: {report['final_theoretical_products']:,}")
    print(f"Saved filtered mapping to {report['save_path']}")


def _write_report(report: dict[str, Any], report_path: Path | None) -> None:
    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
