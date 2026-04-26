#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "combimots"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight CombiMOTS refactor smoke checks")
    parser.add_argument("--work-dir", type=Path, default=None, help="Keep smoke artifacts in this directory")
    args = parser.parse_args()

    if args.work_dir is not None:
        work_dir = args.work_dir.expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        _run_smokes(work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="combimots-smoke-") as tmp_dir:
            _run_smokes(Path(tmp_dir))

    print("Smoke checks passed")


def _run_smokes(work_dir: Path) -> None:
    _smoke_fragments(work_dir)
    _smoke_filters(work_dir)
    _smoke_similarity(work_dir)
    _smoke_search_space(work_dir)
    _smoke_empty_docking_precompute(work_dir)
    _smoke_runner_dry_run(work_dir)
    _smoke_generation_setup_helpers(work_dir)
    _smoke_generation_empty_output(work_dir)
    _smoke_generation_scalarized_output(work_dir)
    _smoke_generation_with_stub_docking(work_dir)


def _smoke_fragments(work_dir: Path) -> None:
    from preprocess.fragments import merge_fragment_files

    data_dir = work_dir / "fragments"
    data_dir.mkdir(parents=True, exist_ok=True)
    frag_1 = data_dir / "target1.txt"
    frag_2 = data_dir / "target2.txt"
    output = data_dir / "fgib_frags.csv"
    report = data_dir / "merge-fragments.json"

    frag_1.write_text("[*:1]CC,0.9\nINVALID,0.1\n")
    frag_2.write_text("[*:1]CC,0.8\n[*:1]CO,0.7\n")

    result = merge_fragment_files((frag_1, frag_2), output, report_path=report)
    output_df = pd.read_csv(output)
    _assert(result.metrics["output_rows"] == 2, "merge-fragments should deduplicate to two rows")
    _assert(set(output_df["smiles"]) == {"CC", "CO"}, "merge-fragments output SMILES mismatch")
    _assert(report.exists(), "merge-fragments report was not written")


def _smoke_filters(work_dir: Path) -> None:
    from preprocess.filters import filter_for_quickvina_elements

    filter_dir = work_dir / "filters"
    filter_dir.mkdir(parents=True, exist_ok=True)
    input_csv = filter_dir / "similar.csv"
    output_csv = filter_dir / "filtered.csv"
    report = filter_dir / "filter-elements.json"

    pd.DataFrame({
        "smiles": ["CCO", "B(O)O", "C[SiH3]", "[Li]C", "not_a_smiles"],
        "reagent_id": [1, 2, 3, 4, 5],
    }).to_csv(input_csv, index=False)

    result = filter_for_quickvina_elements(input_csv, output_csv, report_path=report)
    output_df = pd.read_csv(output_csv)
    _assert(result.metrics["removed_rows"] == 4, "filter-elements should remove invalid and B/Si/Li rows")
    _assert(set(output_df["smiles"]) == {"CCO"}, "filter-elements output mismatch")
    _assert(result.metrics["invalid_smiles_rows"] == 1, "invalid SMILES metric mismatch")


def _smoke_similarity(work_dir: Path) -> None:
    from preprocess.similarity import filter_similar_molecules

    sim_dir = work_dir / "similarity"
    sim_dir.mkdir(parents=True, exist_ok=True)
    custom_csv = sim_dir / "fgib_frags.csv"
    real_csv = sim_dir / "building_blocks.csv"
    output_csv = sim_dir / "similar.csv"
    report = sim_dir / "similar-blocks.json"

    pd.DataFrame({"smiles": ["CCO"]}).to_csv(custom_csv, index=False)
    pd.DataFrame({
        "smiles": ["CCO", "CCC", "not_a_smiles"],
        "reagent_id": ["r1", "r2", "r3"],
    }).to_csv(real_csv, index=False)

    result = filter_similar_molecules(custom_csv, real_csv, output_csv, threshold=1.0, batch_size=2, report_path=report)
    output_df = pd.read_csv(output_csv)
    _assert(result.metrics["output_rows"] == 1, "similar-blocks should retain one exact match")
    _assert(output_df.iloc[0]["smiles"] == "CCO", "similar-blocks exact-match output mismatch")
    _assert(result.metrics["invalid_real_rows"] == 1, "similar-blocks invalid REAL count mismatch")


def _smoke_search_space(work_dir: Path) -> None:
    from pmcts.reactions import QueryMol, Reaction
    from preprocess.search_space import (
        filter_mapping_to_reaction_templates,
        load_reaction_mapping,
        reduce_mapping_to_csv_blocks,
        save_reaction_mapping,
    )

    search_dir = work_dir / "search_space"
    search_dir.mkdir(parents=True, exist_ok=True)
    blocks_csv = search_dir / "final_blocks.csv"
    original_pkl = search_dir / "original.pkl"
    reduced_pkl = search_dir / "reduced.pkl"
    filtered_pkl = search_dir / "filtered.pkl"
    reduction_report = search_dir / "map-search-space.json"
    filter_report = search_dir / "filter-reactions.json"

    pd.DataFrame({"smiles": ["CCO"]}).to_csv(blocks_csv, index=False)
    original_mapping = {1: {0: {"CCO", "CCN"}, 1: {"CCC", "B(O)O", "not_a_smiles"}}}
    save_reaction_mapping(original_mapping, original_pkl)

    reduction = reduce_mapping_to_csv_blocks(blocks_csv, original_pkl, reduced_pkl, report_path=reduction_report)
    reduced_mapping = load_reaction_mapping(reduced_pkl)
    _assert(reduced_mapping[1][0] == {"CCO"}, "search-space reduction should keep matching block")
    _assert(reduced_mapping[1][1] == {"CCC"}, "search-space reduction should fallback empty position")
    _assert(len(reduction["fallback_positions"]) == 1, "search-space fallback count mismatch")
    _assert(reduction["fallback_cleanup"]["forbidden_element_rows"] == 1, "fallback forbidden metric mismatch")
    _assert(reduction["fallback_cleanup"]["invalid_smiles_rows"] == 1, "fallback invalid metric mismatch")

    reaction = Reaction(
        reactants=[QueryMol("[C:1]"), QueryMol("[O:2]")],
        product=QueryMol("[C:1][O:2]"),
        reaction_id=7,
    )
    save_reaction_mapping({7: {0: {"CC"}, 1: {"O"}}}, original_pkl)
    save_reaction_mapping({7: {0: {"N"}, 1: {"O"}}}, reduced_pkl)

    filtered = filter_mapping_to_reaction_templates(
        reaction_to_building_blocks_path=reduced_pkl,
        save_path=filtered_pkl,
        reactions=(reaction,),
        original_reaction_to_building_blocks_path=original_pkl,
        report_path=filter_report,
    )
    filtered_mapping = load_reaction_mapping(filtered_pkl)
    _assert(filtered_mapping[7][0] == {"CC"}, "template filter should fallback to compatible original block")
    _assert(filtered_mapping[7][1] == {"O"}, "template filter should keep compatible source block")
    _assert(len(filtered["fallback_positions"]) == 1, "template filter fallback count mismatch")


def _smoke_empty_docking_precompute(work_dir: Path) -> None:
    from preprocess.docking_scores import batch_dock_csv

    docking_dir = work_dir / "docking"
    docking_dir.mkdir(parents=True, exist_ok=True)
    input_csv = docking_dir / "precompute.csv"
    output_csv = docking_dir / "final_blocks.csv"
    report = docking_dir / "precompute-docking.json"

    pd.DataFrame(columns=["smiles", "prediction"]).to_csv(input_csv, index=False)
    result = batch_dock_csv(input_csv, output_csv, report_path=report, tmp_parent=docking_dir / "tmp")
    output_df = pd.read_csv(output_csv)
    _assert(list(output_df.columns) == ["smiles", "prediction", "ds_1", "ds_2"], "empty docking output columns mismatch")
    _assert(result.metrics["unique_smiles_rows"] == 0, "empty docking metric mismatch")


def _smoke_runner_dry_run(work_dir: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PACKAGE_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "preprocess.runner",
            "--target-pair",
            "gsk3b_jnk3",
            "--input-csv",
            "data/GSK3B_JNK3.csv",
            "--repo-root",
            str(work_dir),
            "--to-step",
            "merge-fragments",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    _assert("[run] fgib-data" in result.stdout, "runner dry-run did not include fgib-data")
    _assert("[run] merge-fragments" in result.stdout, "runner dry-run did not include merge-fragments")


def _smoke_generation_empty_output(work_dir: Path) -> None:
    from pmcts.generate.utils import save_generated_molecules

    output_csv = work_dir / "generation" / "pareto_molecules.csv"
    save_generated_molecules([], {}, output_csv)
    output_df = pd.read_csv(output_csv)
    _assert(output_df.empty, "empty generation output should have no rows")
    _assert("pareto_rank" in output_df.columns, "empty generation output missing pareto_rank")
    _assert("dockingscore1_raw" in output_df.columns, "empty generation output missing raw docking column")


def _smoke_generation_scalarized_output(work_dir: Path) -> None:
    import numpy as np

    from pmcts.generate.node import Node
    from pmcts.generate.utils import save_generated_molecules

    output_csv = work_dir / "generation" / "scalarized_molecules.csv"
    node = Node(
        explore_weight=1.0,
        pareto_function="pmcts",
        scoring_fn=lambda smiles: [0.7, 0.8],
        node_id=1,
        molecules=("CO",),
        construction_log=(),
        rollout_num=1,
        scalarize=True,
    )
    node.P = np.array([0.4875])
    node.save_scores = np.array([0.7, 0.8, -4.0, -5.0])

    save_generated_molecules([node], {}, output_csv, scalarize=True)
    output_df = pd.read_csv(output_csv)
    row = output_df.iloc[0]
    _assert(abs(row["dockingscore1"] - 0.2) < 1e-12, "scalarized docking objective 1 mismatch")
    _assert(abs(row["dockingscore2"] - 0.25) < 1e-12, "scalarized docking objective 2 mismatch")
    _assert(abs(row["dockingscore1_raw"] + 4.0) < 1e-12, "scalarized raw docking score 1 mismatch")
    _assert(abs(row["dockingscore2_raw"] + 5.0) < 1e-12, "scalarized raw docking score 2 mismatch")


def _smoke_generation_setup_helpers(work_dir: Path) -> None:
    from pmcts.generate.generate import _load_building_block_data, _prepare_reactions, _resolve_reaction_mapping_path
    from pmcts.reactions import QueryMol, Reaction

    generation_dir = work_dir / "generation-setup"
    generation_dir.mkdir(parents=True, exist_ok=True)
    blocks_csv = generation_dir / "qed_sa_blocks.csv"
    pd.DataFrame({
        "reagent_id": [1],
        "smiles": ["CCO"],
        "gsk3b_activity": [0.1],
        "jnk3_activity": [0.2],
    }).to_csv(blocks_csv, index=False)

    _load_building_block_data(
        building_blocks_path=blocks_csv,
        building_blocks_id_column="reagent_id",
        building_blocks_smiles_column="smiles",
        target_activities=["gsk3b_activity", "jnk3_activity"],
        qed_sa=True,
    )

    try:
        _load_building_block_data(
            building_blocks_path=blocks_csv,
            building_blocks_id_column="reagent_id",
            building_blocks_smiles_column="smiles",
            target_activities=["gsk3b_activity", "jnk3_activity"],
            qed_sa=False,
        )
    except ValueError as error:
        _assert("ds_1" in str(error) and "ds_2" in str(error), "docking column validation mismatch")
    else:
        raise AssertionError("default generation should require docking columns")

    mapping_path = _resolve_reaction_mapping_path("custom_pair", None)
    _assert(mapping_path.name == "custom_pair.pkl", "custom target-pair mapping path mismatch")
    _assert(not mapping_path.exists(), "unexpected custom target-pair mapping resource exists")

    reaction = Reaction(
        reactants=[QueryMol("[C:1]"), QueryMol("[O:2]")],
        product=QueryMol("[C:1][O:2]"),
        reaction_id=100,
    )
    prepared_reactions = _prepare_reactions(
        reactions=(reaction,),
        building_block_smiles={"C", "O"},
        target_pair="custom_pair",
        reaction_to_building_blocks_path=None,
    )
    _assert(prepared_reactions[0].reactants[0].allowed_building_blocks == ["C"], "fallback carbon reactants mismatch")
    _assert(prepared_reactions[0].reactants[1].allowed_building_blocks == ["O"], "fallback oxygen reactants mismatch")


def _smoke_generation_with_stub_docking(work_dir: Path) -> None:
    from pmcts.generate.generator import Generator
    from pmcts.generate.utils import save_generated_molecules
    from pmcts.reactions import QueryMol, Reaction, set_all_building_blocks

    generation_dir = work_dir / "generation-run"
    docked_smiles = []

    reaction = Reaction(
        reactants=[QueryMol("[C:1]"), QueryMol("[O:2]")],
        product=QueryMol("[C:1][O:2]"),
        reaction_id=99,
    )
    set_all_building_blocks((reaction,), {"C", "O"})
    reaction.reactants[0].allowed_building_blocks = {"C"}
    reaction.reactants[1].allowed_building_blocks = {"O"}

    def scoring_fn(smiles: str) -> list[float]:
        scores = {
            "C": [0.1, 0.2],
            "O": [0.2, 0.1],
            "CO": [0.7, 0.8],
        }
        return scores.get(smiles, [0.0, 0.0])

    def stub_dock(child_nodes_mol, target: str, n_proc: int, sequential: bool):
        _assert(target == "gsk3b_jnk3", "stub docking target mismatch")
        _assert(n_proc in {1, 2}, "stub docking n_proc mismatch")
        _assert(sequential is True, "stub docking sequential flag mismatch")

        ds1_scores = {}
        ds2_scores = {}
        for node, molecules in child_nodes_mol.items():
            docked_smiles.append(molecules[0])
            ds1_scores[node] = -4.0
            ds2_scores[node] = -5.0
        return ds1_scores, ds2_scores

    generator = Generator(
        building_block_smiles_to_id={"C": 1, "O": 2},
        building_block_id_to_smiles={1: "C", 2: "O"},
        max_reactions=1,
        scoring_fn=scoring_fn,
        precomp_ds={0: {"C": -1.0, "O": -2.0}, 1: {"C": -1.5, "O": -2.5}},
        target_pair="gsk3b_jnk3",
        explore_weight=1.0,
        pareto_function="pmcts",
        num_expand_nodes=None,
        reactions=(reaction,),
        rng_seed=0,
        no_building_block_diversity=False,
        store_nodes=True,
        verbose=False,
        n_proc=2,
        sequential=True,
        docking_fn=stub_dock,
    )

    nodes = generator.generate(n_rollout=2, save_dir=generation_dir, save_freq=1000)
    generated_smiles = {node.molecules[0] for node in nodes}
    _assert("CO" in generated_smiles, "stub generation did not create expected product")
    _assert(docked_smiles == ["CO"], "generated-molecule docking cache or precomputed lookup was not used")

    product_node = next(node for node in nodes if node.molecules[0] == "CO")
    _assert(abs(product_node.P[2] - 0.2) < 1e-12, "generated docking objective 1 mismatch")
    _assert(abs(product_node.P[3] - 0.25) < 1e-12, "generated docking objective 2 mismatch")

    output_csv = generation_dir / "stub_pareto_molecules.csv"
    save_generated_molecules(nodes, {1: "C", 2: "O"}, output_csv)
    output_df = pd.read_csv(output_csv)
    product_row = output_df[output_df["smiles"] == "CO"].iloc[0]
    _assert(abs(product_row["dockingscore1"] - 0.2) < 1e-12, "saved docking objective 1 mismatch")
    _assert(abs(product_row["dockingscore2"] - 0.25) < 1e-12, "saved docking objective 2 mismatch")
    _assert(abs(product_row["dockingscore1_raw"] + 4.0) < 1e-12, "saved raw docking score 1 mismatch")
    _assert(abs(product_row["dockingscore2_raw"] + 5.0) < 1e-12, "saved raw docking score 2 mismatch")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
