"""Sequential preprocessing runner for the CombiMOTS pipeline."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pmcts.config import SUPPORTED_TARGET_PAIRS
from preprocess.reports import RunResult, StepResult, write_run_report, write_step_report


STEP_ORDER = (
    "fgib-data",
    "train-fgib",
    "fragments",
    "merge-fragments",
    "similar-blocks",
    "canonicalize-smiles",
    "filter-elements",
    "precompute-chemprop",
    "precompute-docking",
    "map-search-space",
    "filter-reactions",
)

STEP_ALIASES = {
    "chemprop-predict": "precompute-chemprop",
    "remove-salts": "canonicalize-smiles",
}
STEP_CHOICES = (*STEP_ORDER, *STEP_ALIASES)


@dataclass(frozen=True)
class StepPlan:
    name: str
    commands: tuple[tuple[str, ...], ...]
    outputs: tuple[Path, ...]
    description: str
    action: Callable[[], object] | None = None
    report_path: Path | None = None
    resume_outputs: tuple[Path, ...] | None = None


def build_plan(args: argparse.Namespace) -> list[StepPlan]:
    """Build the command plan for a target-pair preprocessing run."""

    repo_root = args.repo_root.expanduser().resolve()
    model_name = args.model_name or args.target_pair
    target1, target2 = _target_activities(args.target_pair)
    target1_name = target1.removesuffix("_activity")
    target2_name = target2.removesuffix("_activity")
    input_csv = _resolve_under(repo_root, args.input_csv)

    data_dir = _resolve_under(repo_root, args.data_dir)
    models_dir = _resolve_under(repo_root, args.models_dir)
    ckpt_dir = _resolve_under(repo_root, args.ckpt_dir)
    model_dir = models_dir / model_name
    script_dir = repo_root / "combimots" / "utils_preprocess"
    resources_dir = repo_root / "combimots" / "pmcts" / "resources" / "real"
    report_dir = _resolve_under(repo_root, args.report_dir) if args.report_dir else model_dir / "preprocess_reports"

    fragments_1 = data_dir / f"{target1_name}.txt"
    fragments_2 = data_dir / f"{target2_name}.txt"
    fgib_data_1 = data_dir / f"{target1_name}.pt"
    fgib_data_2 = data_dir / f"{target2_name}.pt"
    fgib_ckpt_1 = ckpt_dir / f"{target1}_{args.fgib_epochs}.pt"
    fgib_ckpt_2 = ckpt_dir / f"{target2}_{args.fgib_epochs}.pt"
    fgib_frags = data_dir / "fgib_frags.csv"
    similar_blocks = model_dir / "similar.csv"
    precompute_blocks = model_dir / "precompute.csv"
    final_blocks = model_dir / "final_blocks.csv"
    target_mapping = resources_dir / f"{args.target_pair}.pkl"
    python = args.python_executable

    plans = [
        StepPlan(
            name="fgib-data",
            description="Convert input CSV to per-target FGIB .pt datasets",
            commands=(
                (
                    python,
                    str(repo_root / "utils_fgib" / "data.py"),
                    "--csv_path", str(input_csv),
                    "--target", target1,
                    "--test_size", str(args.fgib_test_size),
                    "--random_state", str(args.fgib_random_state),
                    "--save_path", str(fgib_data_1),
                ),
                (
                    python,
                    str(repo_root / "utils_fgib" / "data.py"),
                    "--csv_path", str(input_csv),
                    "--target", target2,
                    "--test_size", str(args.fgib_test_size),
                    "--random_state", str(args.fgib_random_state),
                    "--save_path", str(fgib_data_2),
                ),
            ),
            outputs=(fgib_data_1, fgib_data_2),
            action=lambda: _run_fgib_data(
                repo_root=repo_root,
                input_csv=input_csv,
                targets=((target1, fgib_data_1), (target2, fgib_data_2)),
                test_size=args.fgib_test_size,
                random_state=args.fgib_random_state,
                report_path=_step_report_path(report_dir, "fgib-data"),
            ),
            report_path=_step_report_path(report_dir, "fgib-data"),
        ),
        StepPlan(
            name="train-fgib",
            description="Train one FGIB model per target",
            commands=(
                (
                    python,
                    str(script_dir / "1-train_fgib.py"),
                    "-g", str(args.gpu_id),
                    "--target", target1,
                    "--epochs", str(args.fgib_epochs),
                    "--save_epoch", str(args.fgib_save_epoch or args.fgib_epochs),
                    "--batch_size", str(args.fgib_batch_size),
                    "--data-dir", str(data_dir),
                    "--ckpt-dir", str(ckpt_dir),
                    "--report-path", str(_step_report_path(report_dir, f"train-fgib-{target1_name}")),
                ),
                (
                    python,
                    str(script_dir / "1-train_fgib.py"),
                    "-g", str(args.gpu_id),
                    "--target", target2,
                    "--epochs", str(args.fgib_epochs),
                    "--save_epoch", str(args.fgib_save_epoch or args.fgib_epochs),
                    "--batch_size", str(args.fgib_batch_size),
                    "--data-dir", str(data_dir),
                    "--ckpt-dir", str(ckpt_dir),
                    "--report-path", str(_step_report_path(report_dir, f"train-fgib-{target2_name}")),
                ),
            ),
            outputs=(fgib_ckpt_1, fgib_ckpt_2),
            action=lambda: _run_train_fgib(
                repo_root=repo_root,
                targets=((target1, fgib_ckpt_1), (target2, fgib_ckpt_2)),
                gpu_id=args.gpu_id,
                epochs=args.fgib_epochs,
                data_dir=data_dir,
                ckpt_dir=ckpt_dir,
                batch_size=args.fgib_batch_size,
                save_epoch=args.fgib_save_epoch or args.fgib_epochs,
                report_path=_step_report_path(report_dir, "train-fgib"),
            ),
            report_path=_step_report_path(report_dir, "train-fgib"),
        ),
        StepPlan(
            name="fragments",
            description="Extract FGIB fragments for both targets",
            commands=(
                (
                    python,
                    str(script_dir / "2-get_frags.py"),
                    "-g", str(args.gpu_id),
                    "-t", target1,
                    "-m", str(fgib_ckpt_1),
                    "-v", str(fragments_1),
                    "--vocab_size", str(args.vocab_size),
                    "--data-dir", str(data_dir),
                    "--report-path", str(_step_report_path(report_dir, f"fragments-{target1_name}")),
                ),
                (
                    python,
                    str(script_dir / "2-get_frags.py"),
                    "-g", str(args.gpu_id),
                    "-t", target2,
                    "-m", str(fgib_ckpt_2),
                    "-v", str(fragments_2),
                    "--vocab_size", str(args.vocab_size),
                    "--data-dir", str(data_dir),
                    "--report-path", str(_step_report_path(report_dir, f"fragments-{target2_name}")),
                ),
            ),
            outputs=(fragments_1, fragments_2),
            action=lambda: _run_extract_fragments(
                repo_root=repo_root,
                targets=((target1, fgib_ckpt_1, fragments_1), (target2, fgib_ckpt_2, fragments_2)),
                gpu_id=args.gpu_id,
                vocab_size=args.vocab_size,
                data_dir=data_dir,
                report_path=_step_report_path(report_dir, "fragments"),
            ),
            report_path=_step_report_path(report_dir, "fragments"),
        ),
        StepPlan(
            name="merge-fragments",
            description="Clean and merge per-target fragments",
            commands=(
                (
                    python,
                    str(script_dir / "3-frags_to_blocks.py"),
                    str(fragments_1),
                    str(fragments_2),
                    str(fgib_frags),
                    "--report-path",
                    str(_step_report_path(report_dir, "merge-fragments")),
                ),
            ),
            outputs=(fgib_frags,),
            action=lambda: _run_merge_fragments(
                input_files=(fragments_1, fragments_2),
                output_file=fgib_frags,
                report_path=_step_report_path(report_dir, "merge-fragments"),
            ),
            report_path=_step_report_path(report_dir, "merge-fragments"),
        ),
        StepPlan(
            name="similar-blocks",
            description="Map FGIB fragments to similar Enamine REAL building blocks",
            commands=(
                (
                    python,
                    str(script_dir / "4-get_similar_blocks.py"),
                    "--custom_path", str(fgib_frags),
                    "--real_path", str(resources_dir / "building_blocks.csv"),
                    "--output_path", str(similar_blocks),
                    "--threshold", str(args.similarity_threshold),
                    "--batch_size", str(args.similarity_batch_size),
                    "--report_path", str(_step_report_path(report_dir, "similar-blocks")),
                ),
            ),
            outputs=(similar_blocks,),
            action=lambda: _run_similar_blocks(
                custom_path=fgib_frags,
                real_path=resources_dir / "building_blocks.csv",
                output_path=similar_blocks,
                threshold=args.similarity_threshold,
                batch_size=args.similarity_batch_size,
                report_path=_step_report_path(report_dir, "similar-blocks"),
            ),
            report_path=_step_report_path(report_dir, "similar-blocks"),
        ),
        StepPlan(
            name="canonicalize-smiles",
            description="Canonicalize candidate blocks, remove salts, and drop invalid/disconnected SMILES",
            commands=(
                (
                    args.chemfunc_command,
                    "canonicalize_smiles",
                    "--data_path", str(similar_blocks),
                    "--save_path", str(similar_blocks),
                    "--remove_salts",
                    "--delete_disconnected_mols",
                ),
            ),
            outputs=(similar_blocks,),
            action=lambda: _run_canonicalize_smiles(
                input_file=similar_blocks,
                output_file=similar_blocks,
                report_path=_step_report_path(report_dir, "canonicalize-smiles"),
                command=args.chemfunc_command,
            ),
            report_path=_step_report_path(report_dir, "canonicalize-smiles"),
            resume_outputs=(similar_blocks, _step_report_path(report_dir, "canonicalize-smiles")),
        ),
        StepPlan(
            name="filter-elements",
            description="Remove B/Si/Li blocks for QuickVina compatibility",
            commands=(
                (
                    python,
                    str(script_dir / "5-remove_B_Si_Li_blocks.py"),
                    str(similar_blocks),
                    str(similar_blocks),
                    "--report-path",
                    str(_step_report_path(report_dir, "filter-elements")),
                ),
            ),
            outputs=(similar_blocks,),
            action=lambda: _run_filter_elements(
                input_file=similar_blocks,
                output_file=similar_blocks,
                report_path=_step_report_path(report_dir, "filter-elements"),
            ),
            report_path=_step_report_path(report_dir, "filter-elements"),
            resume_outputs=(similar_blocks, _step_report_path(report_dir, "filter-elements")),
        ),
        StepPlan(
            name="precompute-chemprop",
            description="Predict target activities for candidate building blocks",
            commands=(
                (
                    args.chemprop_command,
                    "--test_path", str(similar_blocks),
                    "--preds_path", str(precompute_blocks),
                    "--checkpoint_dir", str(model_dir),
                ),
            ),
            outputs=(precompute_blocks,),
            action=lambda: _run_chemprop_predict(
                test_path=similar_blocks,
                preds_path=precompute_blocks,
                checkpoint_dir=model_dir,
                report_path=_step_report_path(report_dir, "precompute-chemprop"),
                command=args.chemprop_command,
            ),
            report_path=_step_report_path(report_dir, "precompute-chemprop"),
        ),
        StepPlan(
            name="precompute-docking",
            description="Precompute QuickVina docking scores for candidate building blocks",
            commands=(
                _append_if(
                    (
                        python,
                        str(script_dir / "6-precompute_docking_scores.py"),
                        str(precompute_blocks),
                        str(final_blocks),
                        "--target_pair", args.target_pair,
                        "--report_path", str(_step_report_path(report_dir, "precompute-docking")),
                    ),
                    args.sequential_docking,
                    "--sequential",
                ),
            ),
            outputs=(final_blocks,),
            action=lambda: _run_precompute_docking(
                input_csv=precompute_blocks,
                output_csv=final_blocks,
                target_pair=args.target_pair,
                sequential=args.sequential_docking,
                report_path=_step_report_path(report_dir, "precompute-docking"),
                tmp_parent=repo_root / "tmp",
            ),
            report_path=_step_report_path(report_dir, "precompute-docking"),
        ),
        StepPlan(
            name="map-search-space",
            description="Reduce REAL reaction-position mapping to candidate blocks",
            commands=(
                (
                    python,
                    str(script_dir / "7-map_bbs_to_search_space.py"),
                    "--input", str(final_blocks),
                    "--real_path", str(resources_dir / "reaction_to_building_blocks.pkl"),
                    "--save_path", str(target_mapping),
                    "--smiles_column", "smiles",
                    "--report_path", str(_step_report_path(report_dir, "map-search-space")),
                ),
            ),
            outputs=(target_mapping,),
            action=lambda: _run_map_search_space(
                input_csv=final_blocks,
                real_path=resources_dir / "reaction_to_building_blocks.pkl",
                save_path=target_mapping,
                smiles_column="smiles",
                report_path=_step_report_path(report_dir, "map-search-space"),
            ),
            report_path=_step_report_path(report_dir, "map-search-space"),
        ),
        StepPlan(
            name="filter-reactions",
            description="Filter reduced mapping by REAL reaction templates",
            commands=(
                (
                    python,
                    str(script_dir / "8-filter_reactions_to_blocks.py"),
                    "--reaction_to_building_blocks_path", str(target_mapping),
                    "--save_path", str(target_mapping),
                    "--original_reaction_to_building_blocks_path", str(resources_dir / "reaction_to_building_blocks.pkl"),
                    "--report_path", str(_step_report_path(report_dir, "filter-reactions")),
                ),
            ),
            outputs=(target_mapping,),
            action=lambda: _run_filter_reactions(
                reaction_to_building_blocks_path=target_mapping,
                save_path=target_mapping,
                original_reaction_to_building_blocks_path=resources_dir / "reaction_to_building_blocks.pkl",
                report_path=_step_report_path(report_dir, "filter-reactions"),
            ),
            report_path=_step_report_path(report_dir, "filter-reactions"),
            resume_outputs=(target_mapping, _step_report_path(report_dir, "filter-reactions")),
        ),
    ]

    selected_names = _select_step_names(args.step, args.from_step, args.to_step)
    return [plan for plan in plans if plan.name in selected_names]


def run_plan(
    plans: list[StepPlan],
    dry_run: bool,
    resume: bool,
    force: bool,
    cwd: Path,
    run_report_path: Path | None,
    metadata: dict[str, object],
) -> list[StepResult]:
    """Print or execute a preprocessing plan."""

    produced_this_run: set[Path] = set()
    step_results: list[StepResult] = []
    for plan in plans:
        resume_outputs = plan.resume_outputs or plan.outputs
        outputs_exist = bool(resume_outputs) and all(output.exists() for output in resume_outputs)
        output_was_produced_this_run = any(output in produced_this_run for output in plan.outputs)
        status = "skip" if resume and outputs_exist and not force and not output_was_produced_this_run else "run"
        print(f"\n[{status}] {plan.name}: {plan.description}")
        for output in plan.outputs:
            print(f"  output: {output}")
        for command in plan.commands:
            print(f"  $ {_format_command(command)}")

        if dry_run:
            continue

        if status == "skip":
            step_results.append(StepResult(
                step=plan.name,
                status="skipped",
                outputs=[str(output) for output in plan.outputs],
            ))
            continue

        sys.stdout.flush()
        try:
            for output in plan.outputs:
                output.parent.mkdir(parents=True, exist_ok=True)
            if plan.action is not None:
                action_result = plan.action()
            else:
                for command in plan.commands:
                    subprocess.run(command, check=True, cwd=cwd)
                action_result = None

            missing_outputs = [output for output in plan.outputs if not output.exists()]
            if missing_outputs:
                raise FileNotFoundError(f"Step {plan.name} did not produce expected outputs: {missing_outputs}")
            step_results.append(_coerce_step_result(plan, action_result))
        except Exception as e:
            failure = StepResult(
                step=plan.name,
                status="failed",
                outputs=[str(output) for output in plan.outputs],
                warnings=[str(e)],
            )
            step_results.append(failure)
            _write_run_summary("failed", step_results, metadata, run_report_path, warnings=[str(e)])
            raise
        produced_this_run.update(plan.outputs)

    if not dry_run:
        _write_run_summary("success", step_results, metadata, run_report_path)
    return step_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or plan CombiMOTS preprocessing steps")
    parser.add_argument("--target-pair", required=True, choices=SUPPORTED_TARGET_PAIRS)
    parser.add_argument("--input-csv", type=Path, required=True, help="Training CSV with smiles and target activity columns")
    parser.add_argument("--model-name", default=None, help="Defaults to --target-pair")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("ckpt"))
    parser.add_argument("--report-dir", type=Path, default=None, help="Defaults to models/{model_name}/preprocess_reports")
    parser.add_argument("--run-report-path", type=Path, default=None, help="Defaults to {report_dir}/run-summary.json")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--step", choices=STEP_CHOICES, default=None, help="Run exactly one step")
    parser.add_argument("--from-step", choices=STEP_CHOICES, default=None)
    parser.add_argument("--to-step", choices=STEP_CHOICES, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing")
    parser.add_argument("--resume", action="store_true", help="Skip steps whose outputs already exist")
    parser.add_argument("--force", action="store_true", help="Run selected steps even if --resume outputs exist")
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--fgib-epochs", type=int, default=10)
    parser.add_argument("--fgib-batch-size", type=int, default=1024)
    parser.add_argument("--fgib-save-epoch", type=int, default=None, help="Defaults to --fgib-epochs")
    parser.add_argument("--fgib-test-size", type=float, default=0.2)
    parser.add_argument("--fgib-random-state", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=300)
    parser.add_argument("--similarity-threshold", type=float, default=0.4)
    parser.add_argument("--similarity-batch-size", type=int, default=2500)
    parser.add_argument("--chemfunc-command", default="chemfunc")
    parser.add_argument("--chemprop-command", default="chemprop_predict")
    parser.add_argument("--sequential-docking", action="store_true")
    args = parser.parse_args()

    plans = build_plan(args)
    run_plan(
        plans,
        dry_run=args.dry_run,
        resume=args.resume,
        force=args.force,
        cwd=args.repo_root.expanduser().resolve(),
        run_report_path=_run_report_path(args),
        metadata=_run_metadata(args, plans),
    )


def _target_activities(target_pair: str) -> tuple[str, str]:
    target_names = target_pair.split("_")
    if len(target_names) != 2:
        raise ValueError(f"Cannot infer target activities from target pair: {target_pair}")
    return f"{target_names[0]}_activity", f"{target_names[1]}_activity"


def _resolve_under(repo_root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else repo_root / path


def _report_dir(args: argparse.Namespace) -> Path:
    repo_root = args.repo_root.expanduser().resolve()
    model_name = args.model_name or args.target_pair
    models_dir = _resolve_under(repo_root, args.models_dir)
    return _resolve_under(repo_root, args.report_dir) if args.report_dir else models_dir / model_name / "preprocess_reports"


def _run_report_path(args: argparse.Namespace) -> Path | None:
    if args.dry_run:
        return None
    repo_root = args.repo_root.expanduser().resolve()
    return _resolve_under(repo_root, args.run_report_path) if args.run_report_path else _report_dir(args) / "run-summary.json"


def _run_metadata(args: argparse.Namespace, plans: list[StepPlan]) -> dict[str, object]:
    repo_root = args.repo_root.expanduser().resolve()
    return {
        "target_pair": args.target_pair,
        "model_name": args.model_name or args.target_pair,
        "repo_root": str(repo_root),
        "input_csv": str(_resolve_under(repo_root, args.input_csv)),
        "data_dir": str(_resolve_under(repo_root, args.data_dir)),
        "models_dir": str(_resolve_under(repo_root, args.models_dir)),
        "ckpt_dir": str(_resolve_under(repo_root, args.ckpt_dir)),
        "report_dir": str(_report_dir(args)),
        "selected_steps": [plan.name for plan in plans],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def _select_step_names(step: str | None, from_step: str | None, to_step: str | None) -> set[str]:
    step = _canonical_step_name(step)
    from_step = _canonical_step_name(from_step)
    to_step = _canonical_step_name(to_step)
    if step and (from_step or to_step):
        raise ValueError("Use either --step or --from-step/--to-step, not both")
    if step:
        return {step}
    start = STEP_ORDER.index(from_step) if from_step else 0
    end = STEP_ORDER.index(to_step) if to_step else len(STEP_ORDER) - 1
    if start > end:
        raise ValueError("--from-step must be before --to-step")
    return set(STEP_ORDER[start:end + 1])


def _canonical_step_name(step_name: str | None) -> str | None:
    return STEP_ALIASES.get(step_name, step_name) if step_name is not None else None


def _append_if(command: tuple[str, ...], condition: bool, *extra: str) -> tuple[str, ...]:
    return command + extra if condition else command


def _step_report_path(report_dir: Path, step_name: str) -> Path:
    return report_dir / f"{_step_report_stem(step_name)}.json"


def _step_report_stem(step_name: str) -> str:
    step_name = _canonical_step_name(step_name) or step_name
    for index, base_name in enumerate(STEP_ORDER):
        if step_name == base_name or step_name.startswith(f"{base_name}-"):
            return f"{index:02d}-{step_name}"
    return step_name


def _coerce_step_result(plan: StepPlan, action_result: object) -> StepResult:
    if isinstance(action_result, StepResult):
        return action_result
    metrics = action_result if isinstance(action_result, dict) else {}
    result = StepResult(
        step=plan.name,
        status="success",
        outputs=[str(output) for output in plan.outputs],
        metrics=metrics,
    )
    write_step_report(result, plan.report_path)
    return result


def _write_run_summary(
    status: str,
    step_results: list[StepResult],
    metadata: dict[str, object],
    report_path: Path | None,
    warnings: list[str] | None = None,
) -> None:
    metadata = dict(metadata)
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_run_report(
        RunResult(
            status=status,
            steps=step_results,
            metadata=metadata,
            warnings=warnings or [],
        ),
        report_path,
    )


def _ensure_repo_root_on_path(repo_root: Path) -> None:
    repo_root_string = str(repo_root)
    if repo_root_string not in sys.path:
        sys.path.insert(0, repo_root_string)


def _combine_step_results(
    step_name: str,
    child_results: list[StepResult],
    outputs: list[Path],
    report_path: Path,
) -> StepResult:
    result = StepResult(
        step=step_name,
        status="success",
        inputs=[input_path for child in child_results for input_path in child.inputs],
        outputs=[str(output) for output in outputs],
        metrics={"targets": [child.to_dict() for child in child_results]},
        warnings=[warning for child in child_results for warning in child.warnings],
    )
    write_step_report(result, report_path)
    return result


def _run_fgib_data(
    repo_root: Path,
    input_csv: Path,
    targets: tuple[tuple[str, Path], ...],
    test_size: float,
    random_state: int,
    report_path: Path,
) -> StepResult:
    _ensure_repo_root_on_path(repo_root)
    from preprocess.fgib import prepare_fgib_dataset

    child_results = [
        prepare_fgib_dataset(
            csv_path=input_csv,
            target=target,
            output_path=output_path,
            test_size=test_size,
            random_state=random_state,
            report_path=None,
        )
        for target, output_path in targets
    ]
    return _combine_step_results("fgib-data", child_results, [output for _, output in targets], report_path)


def _run_train_fgib(
    repo_root: Path,
    targets: tuple[tuple[str, Path], ...],
    gpu_id: int,
    epochs: int,
    data_dir: Path,
    ckpt_dir: Path,
    batch_size: int,
    save_epoch: int,
    report_path: Path,
) -> StepResult:
    _ensure_repo_root_on_path(repo_root)
    from preprocess.fgib import train_fgib_model

    child_results = [
        train_fgib_model(
            target=target,
            gpu_id=gpu_id,
            epochs=epochs,
            output_checkpoint=output_checkpoint,
            data_dir=data_dir,
            ckpt_dir=ckpt_dir,
            batch_size=batch_size,
            save_epoch=save_epoch,
            report_path=None,
        )
        for target, output_checkpoint in targets
    ]
    return _combine_step_results("train-fgib", child_results, [output for _, output in targets], report_path)


def _run_extract_fragments(
    repo_root: Path,
    targets: tuple[tuple[str, Path, Path], ...],
    gpu_id: int,
    vocab_size: int,
    data_dir: Path,
    report_path: Path,
) -> StepResult:
    _ensure_repo_root_on_path(repo_root)
    from preprocess.fgib import extract_fgib_fragments

    child_results = [
        extract_fgib_fragments(
            target=target,
            gib_path=checkpoint_path,
            vocab_path=vocab_path,
            gpu_id=gpu_id,
            vocab_size=vocab_size,
            data_dir=data_dir,
            report_path=None,
        )
        for target, checkpoint_path, vocab_path in targets
    ]
    return _combine_step_results("fragments", child_results, [vocab_path for _, _, vocab_path in targets], report_path)


def _run_merge_fragments(input_files: tuple[Path, ...], output_file: Path, report_path: Path) -> object:
    from preprocess.fragments import merge_fragment_files

    return merge_fragment_files(input_files=input_files, output_file=output_file, report_path=report_path)


def _run_similar_blocks(
    custom_path: Path,
    real_path: Path,
    output_path: Path,
    threshold: float,
    batch_size: int,
    report_path: Path,
) -> object:
    from preprocess.similarity import filter_similar_molecules

    return filter_similar_molecules(
        custom_path=custom_path,
        real_path=real_path,
        output_path=output_path,
        threshold=threshold,
        batch_size=batch_size,
        report_path=report_path,
    )


def _run_filter_elements(input_file: Path, output_file: Path, report_path: Path) -> object:
    from preprocess.filters import filter_for_quickvina_elements

    return filter_for_quickvina_elements(input_file=input_file, output_file=output_file, report_path=report_path)


def _run_canonicalize_smiles(input_file: Path, output_file: Path, report_path: Path, command: str) -> StepResult:
    import pandas as pd
    from preprocess.filters import clean_building_block_dataframe

    input_rows = len(pd.read_csv(input_file)) if input_file.exists() else 0
    try:
        subprocess.run(
            [
                command,
                "canonicalize_smiles",
                "--data_path", str(input_file),
                "--save_path", str(output_file),
                "--remove_salts",
                "--delete_disconnected_mols",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "").strip()
        hint = (
            "If this mentions GLIBCXX or libstdc++, reactivate the conda environment created by "
            "setup_env.sh or export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}."
        )
        raise RuntimeError(f"{command} canonicalize_smiles failed. {hint}\n{details}") from error
    output_df = pd.read_csv(output_file) if output_file.exists() else pd.DataFrame(columns=["smiles"])
    chemfunc_rows = len(output_df)
    cleaned_df, cleanup_metrics = clean_building_block_dataframe(output_df)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_file, index=False)
    result = StepResult(
        step="canonicalize-smiles",
        status="success",
        inputs=[str(input_file)],
        outputs=[str(output_file)],
        metrics={
            "input_rows": input_rows,
            "chemfunc_output_rows": chemfunc_rows,
            "output_rows": len(cleaned_df),
            "removed_rows": input_rows - len(cleaned_df),
            "post_chemfunc_removed_rows": chemfunc_rows - len(cleaned_df),
            "command": command,
            "cleanup": cleanup_metrics,
        },
    )
    write_step_report(result, report_path)
    return result


def _run_chemprop_predict(
    test_path: Path,
    preds_path: Path,
    checkpoint_dir: Path,
    report_path: Path,
    command: str,
) -> object:
    from preprocess.chemprop import run_chemprop_predict

    return run_chemprop_predict(
        test_path=test_path,
        preds_path=preds_path,
        checkpoint_dir=checkpoint_dir,
        report_path=report_path,
        command=command,
    )


def _run_precompute_docking(
    input_csv: Path,
    output_csv: Path,
    target_pair: str,
    sequential: bool,
    report_path: Path,
    tmp_parent: Path,
) -> object:
    from preprocess.docking_scores import batch_dock_csv

    return batch_dock_csv(
        input_csv=input_csv,
        output_csv=output_csv,
        target_pair=target_pair,
        sequential=sequential,
        report_path=report_path,
        tmp_parent=tmp_parent,
    )


def _run_map_search_space(
    input_csv: Path,
    real_path: Path,
    save_path: Path,
    smiles_column: str,
    report_path: Path,
) -> StepResult:
    from preprocess.search_space import reduce_mapping_to_csv_blocks

    report = reduce_mapping_to_csv_blocks(
        input_csv=input_csv,
        real_path=real_path,
        save_path=save_path,
        smiles_column=smiles_column,
        report_path=None,
    )
    warnings = []
    fallback_count = len(report.get("fallback_positions", []))
    if fallback_count:
        warnings.append(f"{fallback_count} reaction positions fell back to original REAL blocks.")
    result = StepResult(
        step="map-search-space",
        status="success",
        inputs=[str(input_csv), str(real_path)],
        outputs=[str(save_path)],
        metrics=report,
        warnings=warnings,
    )
    write_step_report(result, report_path)
    return result


def _run_filter_reactions(
    reaction_to_building_blocks_path: Path,
    save_path: Path,
    original_reaction_to_building_blocks_path: Path,
    report_path: Path,
) -> StepResult:
    from pmcts.reactions import REACTIONS
    from preprocess.search_space import filter_mapping_to_reaction_templates

    report = filter_mapping_to_reaction_templates(
        reaction_to_building_blocks_path=reaction_to_building_blocks_path,
        save_path=save_path,
        reactions=REACTIONS,
        original_reaction_to_building_blocks_path=original_reaction_to_building_blocks_path,
        report_path=None,
    )
    warnings = []
    fallback_count = len(report.get("fallback_positions", []))
    if fallback_count:
        warnings.append(f"{fallback_count} reaction positions fell back to original/template-compatible blocks.")
    result = StepResult(
        step="filter-reactions",
        status="success",
        inputs=[str(reaction_to_building_blocks_path), str(original_reaction_to_building_blocks_path)],
        outputs=[str(save_path)],
        metrics=report,
        warnings=warnings,
    )
    write_step_report(result, report_path)
    return result


def _format_command(command: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


if __name__ == "__main__":
    main()
