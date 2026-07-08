from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OPEN_RUNNER = ROOT / "scripts" / "open_cpmp_benchmarks.py"
PAPER_RUNNER = ROOT / "scripts" / "benchmark_compact_paper_set.py"

V4_BASE_VARIANT = "qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish"
OPEN120_FAMILIES = (
    "5x5_cv1_test",
    "5x5_cv2_test",
    "5x5_cv3_test",
    "7x5_cv1_test",
    "7x5_cv2_test",
    "7x5_cv3_test",
)

V4_BASELINE_OPTUNA_PARAMS = {
    "beam_width": 1200,
    "core_ratio": 0.90,
    "parent_core_cap": 4,
    "union_diversity_qubo_supplement_width": 4,
    "union_diversity_qubo_max_vars": 48,
    "union_diversity_qubo_select": 4,
    "union_diversity_qubo_rank_weight": 0.02,
    "union_diversity_qubo_same_parent_penalty": 0.55,
    "union_diversity_qubo_same_source_penalty": 0.25,
    "union_diversity_qubo_signature_penalty": 0.20,
    "union_diversity_qubo_reduce_signature_penalty": 0.15,
    "union_diversity_qubo_n_band_penalty": 0.08,
    "union_diversity_qubo_macro_ancestry_penalty_weight": 40.0,
    "union_diversity_qubo_macro_ancestry_penalty_scale": 4.0,
    "union_diversity_qubo_unified_macro_future_credit_scale": 1.0,
    "union_diversity_qubo_unified_macro_pair_penalty": 0.0,
    "unified_macro_supply_parent_width": 4,
    "unified_macro_supply_parent_offset": 4,
    "unified_macro_supply_max_generated": 768,
    "macro_chain_witness_parent_width": 4,
    "macro_chain_witness_max_generated": 0,
    "frg_frontier_child_certify_time_limit": 0.004,
    "post_polish_successor_cache": False,
    "post_polish_max_attempts": 0,
}

CONSERVATIVE_TUNABLE_PARAMS = frozenset(
    {
        "beam_width",
        "core_width",
        "parent_core_cap",
        "union_diversity_qubo_supplement_width",
        "union_diversity_qubo_max_vars",
        "union_diversity_qubo_select",
        "union_diversity_qubo_rank_weight",
        "union_diversity_qubo_same_parent_penalty",
        "union_diversity_qubo_same_source_penalty",
        "union_diversity_qubo_signature_penalty",
        "union_diversity_qubo_reduce_signature_penalty",
        "union_diversity_qubo_n_band_penalty",
        "union_diversity_qubo_macro_ancestry_penalty_weight",
        "union_diversity_qubo_macro_ancestry_penalty_scale",
        "union_diversity_qubo_unified_macro_future_credit_scale",
        "union_diversity_qubo_unified_macro_pair_penalty",
        "unified_macro_supply_parent_width",
        "unified_macro_supply_parent_offset",
        "unified_macro_supply_max_generated",
        "macro_chain_witness_parent_width",
        "macro_chain_witness_max_generated",
        "frg_frontier_child_certify_time_limit",
        "post_polish_successor_cache",
        "post_polish_max_attempts",
    }
)


@dataclass(frozen=True)
class BenchmarkSummary:
    label: str
    cases: int
    solved: int
    moves: int
    full_quality_penalty: float
    runtime_sec: float
    failures: int = 0


@dataclass(frozen=True)
class TrialOutcome:
    trial_name: str
    score: float
    params: dict[str, object]
    summaries: list[BenchmarkSummary]
    open_returncode: int
    paper_returncode: int
    open_output: str
    paper_output: str


def _is_true(value: object) -> bool:
    return value in (True, 1, "1", "True", "true", "TRUE", "yes", "YES")


def _numeric(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_solved(row: Mapping[str, object]) -> bool:
    primary = row.get("primary_ok")
    if primary not in (None, ""):
        return _is_true(primary)
    return _is_true(row.get("business_ok")) and _is_true(row.get("goal_validation_ok"))


def summarize_result_csv(
    path: Path,
    *,
    label: str,
    row_filter: callable | None = None,
) -> BenchmarkSummary:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if row_filter is not None:
        rows = [row for row in rows if row_filter(row)]
    solved_rows = [row for row in rows if _row_solved(row)]
    return BenchmarkSummary(
        label=label,
        cases=len(rows),
        solved=len(solved_rows),
        moves=int(sum(_numeric(row.get("moves")) for row in solved_rows)),
        full_quality_penalty=sum(_numeric(row.get("full_quality_penalty")) for row in rows),
        runtime_sec=sum(_numeric(row.get("runtime_sec")) for row in rows),
        failures=len(rows) - len(solved_rows),
    )


def summarize_paper_csv_by_dataset(path: Path) -> list[BenchmarkSummary]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    labels = sorted({str(row.get("dataset", "") or "paper") for row in rows})
    summaries: list[BenchmarkSummary] = []
    for label in labels:
        selected = [row for row in rows if str(row.get("dataset", "") or "paper") == label]
        solved_rows = [row for row in selected if _row_solved(row)]
        summaries.append(
            BenchmarkSummary(
                label=label,
                cases=len(selected),
                solved=len(solved_rows),
                moves=int(sum(_numeric(row.get("moves")) for row in solved_rows)),
                full_quality_penalty=sum(_numeric(row.get("full_quality_penalty")) for row in selected),
                runtime_sec=sum(_numeric(row.get("runtime_sec")) for row in selected),
                failures=len(selected) - len(solved_rows),
            )
        )
    return summaries


def summarize_csv_by_family(path: Path, *, default_dataset: str) -> dict[str, BenchmarkSummary]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        dataset = str(row.get("dataset", "") or default_dataset)
        family = str(row.get("family", "") or "unknown")
        grouped.setdefault(f"{dataset}/{family}", []).append(row)

    summaries: dict[str, BenchmarkSummary] = {}
    for label, selected in grouped.items():
        solved_rows = [row for row in selected if _row_solved(row)]
        summaries[label] = BenchmarkSummary(
            label=label,
            cases=len(selected),
            solved=len(solved_rows),
            moves=int(sum(_numeric(row.get("moves")) for row in solved_rows)),
            full_quality_penalty=sum(_numeric(row.get("full_quality_penalty")) for row in selected),
            runtime_sec=sum(_numeric(row.get("runtime_sec")) for row in selected),
            failures=len(selected) - len(solved_rows),
        )
    return summaries


def constraint_objective_score(
    summaries: Iterable[BenchmarkSummary],
    expected_cases: Mapping[str, int] | None = None,
    *,
    missing_solution_penalty: float = 1_000_000_000.0,
    full_penalty_weight: float = 1_000.0,
    runtime_weight: float = 0.10,
) -> float:
    expected_cases = expected_cases or {}
    score = 0.0
    for summary in summaries:
        expected = int(expected_cases.get(summary.label, summary.cases))
        missing = max(0, expected - summary.solved)
        score += missing_solution_penalty * missing
        score += full_penalty_weight * float(summary.full_quality_penalty)
        score += float(summary.moves)
        score += runtime_weight * float(summary.runtime_sec)
    return score


def family_safe_objective_score(
    summaries: Iterable[BenchmarkSummary],
    expected_cases: Mapping[str, int] | None,
    *,
    current_family: Mapping[str, BenchmarkSummary],
    baseline_family: Mapping[str, BenchmarkSummary],
    move_tolerance: float = 20.0,
    move_regression_weight: float = 5_000.0,
    solved_loss_penalty: float = 1_000_000_000.0,
    missing_solution_penalty: float = 1_000_000_000.0,
    full_penalty_weight: float = 1_000.0,
    runtime_weight: float = 0.10,
) -> float:
    score = constraint_objective_score(
        summaries,
        expected_cases,
        missing_solution_penalty=missing_solution_penalty,
        full_penalty_weight=full_penalty_weight,
        runtime_weight=runtime_weight,
    )
    for label, baseline in baseline_family.items():
        current = current_family.get(label)
        if current is None:
            score += solved_loss_penalty * max(1, baseline.solved)
            continue
        solved_loss = max(0, baseline.solved - current.solved)
        score += solved_loss_penalty * solved_loss
        move_regression = max(0.0, float(current.moves - baseline.moves) - float(move_tolerance))
        score += move_regression_weight * move_regression
    return score


def _flag_name(param_name: str) -> str:
    return "--" + param_name.replace("_", "-")


def params_to_cli_args(
    params: Mapping[str, object],
    *,
    allowed_params: frozenset[str] | None = None,
) -> list[str]:
    args: list[str] = []
    for name in sorted(params):
        if allowed_params is not None and name not in allowed_params:
            continue
        value = params[name]
        if value is None:
            continue
        flag = _flag_name(name)
        if isinstance(value, bool):
            args.append(flag if value else "--no-" + flag[2:])
        else:
            args.extend([flag, str(value)])
    return args


def suggest_conservative_v4_params(trial) -> dict[str, object]:
    beam_width = int(trial.suggest_categorical("beam_width", [1000, 1120, 1200, 1320, 1440]))
    core_ratio = float(trial.suggest_float("core_ratio", 0.78, 0.92, step=0.02))
    core_width = int(round((beam_width * core_ratio) / 8.0) * 8)
    core_width = max(1, min(beam_width, core_width))

    return {
        "beam_width": beam_width,
        "core_width": core_width,
        "parent_core_cap": int(trial.suggest_categorical("parent_core_cap", [3, 4, 5, 6])),
        "union_diversity_qubo_supplement_width": int(
            trial.suggest_categorical("union_diversity_qubo_supplement_width", [4, 8, 16, 24, 32, 48, 64, 80, 96, 128])
        ),
        "union_diversity_qubo_max_vars": int(
            trial.suggest_categorical("union_diversity_qubo_max_vars", [48, 64, 96, 128, 160, 192])
        ),
        "union_diversity_qubo_select": int(
            trial.suggest_categorical("union_diversity_qubo_select", [4, 8, 16, 24, 32, 48])
        ),
        "union_diversity_qubo_rank_weight": float(
            trial.suggest_float("union_diversity_qubo_rank_weight", 0.01, 0.05, step=0.005)
        ),
        "union_diversity_qubo_same_parent_penalty": float(
            trial.suggest_float("union_diversity_qubo_same_parent_penalty", 0.25, 0.70, step=0.05)
        ),
        "union_diversity_qubo_same_source_penalty": float(
            trial.suggest_float("union_diversity_qubo_same_source_penalty", 0.05, 0.35, step=0.05)
        ),
        "union_diversity_qubo_signature_penalty": float(
            trial.suggest_float("union_diversity_qubo_signature_penalty", 0.08, 0.32, step=0.04)
        ),
        "union_diversity_qubo_reduce_signature_penalty": float(
            trial.suggest_float("union_diversity_qubo_reduce_signature_penalty", 0.03, 0.24, step=0.03)
        ),
        "union_diversity_qubo_n_band_penalty": float(
            trial.suggest_float("union_diversity_qubo_n_band_penalty", 0.02, 0.18, step=0.02)
        ),
        "union_diversity_qubo_macro_ancestry_penalty_weight": float(
            trial.suggest_float("union_diversity_qubo_macro_ancestry_penalty_weight", 24.0, 56.0, step=4.0)
        ),
        "union_diversity_qubo_macro_ancestry_penalty_scale": float(
            trial.suggest_float("union_diversity_qubo_macro_ancestry_penalty_scale", 3.0, 6.0, step=0.5)
        ),
        "union_diversity_qubo_unified_macro_future_credit_scale": float(
            trial.suggest_float("union_diversity_qubo_unified_macro_future_credit_scale", 0.55, 1.0, step=0.05)
        ),
        "union_diversity_qubo_unified_macro_pair_penalty": float(
            trial.suggest_float("union_diversity_qubo_unified_macro_pair_penalty", 0.0, 0.30, step=0.02)
        ),
        "unified_macro_supply_parent_width": int(
            trial.suggest_categorical("unified_macro_supply_parent_width", [3, 4, 5, 6])
        ),
        "unified_macro_supply_parent_offset": int(
            trial.suggest_categorical("unified_macro_supply_parent_offset", [3, 4, 5, 6])
        ),
        "unified_macro_supply_max_generated": int(
            trial.suggest_categorical("unified_macro_supply_max_generated", [512, 768, 1024])
        ),
        "macro_chain_witness_parent_width": int(
            trial.suggest_categorical("macro_chain_witness_parent_width", [3, 4, 5, 6])
        ),
        "macro_chain_witness_max_generated": int(
            trial.suggest_categorical("macro_chain_witness_max_generated", [0, 512, 768, 1024])
        ),
        "frg_frontier_child_certify_time_limit": float(
            trial.suggest_categorical("frg_frontier_child_certify_time_limit", [0.002, 0.004, 0.006, 0.008])
        ),
        "post_polish_successor_cache": bool(trial.suggest_categorical("post_polish_successor_cache", [False, True])),
        "post_polish_max_attempts": int(trial.suggest_categorical("post_polish_max_attempts", [0, 640, 960, 1280])),
    }


def suggest_safe_v4_params(trial) -> dict[str, object]:
    beam_width = int(trial.suggest_categorical("beam_width", [1120, 1200, 1320, 1440]))
    core_ratio = float(trial.suggest_float("core_ratio", 0.84, 0.92, step=0.02))
    core_width = int(round((beam_width * core_ratio) / 8.0) * 8)
    core_width = max(1, min(beam_width, core_width))

    return {
        "beam_width": beam_width,
        "core_width": core_width,
        "parent_core_cap": int(trial.suggest_categorical("parent_core_cap", [3, 4, 5])),
        "union_diversity_qubo_supplement_width": int(
            trial.suggest_categorical("union_diversity_qubo_supplement_width", [4, 16, 24, 32, 48])
        ),
        "union_diversity_qubo_max_vars": int(trial.suggest_categorical("union_diversity_qubo_max_vars", [48])),
        "union_diversity_qubo_select": int(trial.suggest_categorical("union_diversity_qubo_select", [4, 8])),
        "union_diversity_qubo_rank_weight": float(
            trial.suggest_float("union_diversity_qubo_rank_weight", 0.01, 0.04, step=0.005)
        ),
        "union_diversity_qubo_same_parent_penalty": float(
            trial.suggest_float("union_diversity_qubo_same_parent_penalty", 0.35, 0.65, step=0.05)
        ),
        "union_diversity_qubo_same_source_penalty": float(
            trial.suggest_float("union_diversity_qubo_same_source_penalty", 0.15, 0.35, step=0.05)
        ),
        "union_diversity_qubo_signature_penalty": float(
            trial.suggest_float("union_diversity_qubo_signature_penalty", 0.16, 0.32, step=0.04)
        ),
        "union_diversity_qubo_reduce_signature_penalty": float(
            trial.suggest_float("union_diversity_qubo_reduce_signature_penalty", 0.09, 0.24, step=0.03)
        ),
        "union_diversity_qubo_n_band_penalty": float(
            trial.suggest_float("union_diversity_qubo_n_band_penalty", 0.06, 0.14, step=0.02)
        ),
        "union_diversity_qubo_macro_ancestry_penalty_weight": float(
            trial.suggest_float("union_diversity_qubo_macro_ancestry_penalty_weight", 36.0, 52.0, step=4.0)
        ),
        "union_diversity_qubo_macro_ancestry_penalty_scale": float(
            trial.suggest_float("union_diversity_qubo_macro_ancestry_penalty_scale", 3.0, 5.0, step=0.5)
        ),
        "union_diversity_qubo_unified_macro_future_credit_scale": float(
            trial.suggest_float("union_diversity_qubo_unified_macro_future_credit_scale", 0.70, 1.0, step=0.05)
        ),
        "union_diversity_qubo_unified_macro_pair_penalty": float(
            trial.suggest_float("union_diversity_qubo_unified_macro_pair_penalty", 0.0, 0.22, step=0.02)
        ),
        "unified_macro_supply_parent_width": int(
            trial.suggest_categorical("unified_macro_supply_parent_width", [3, 4, 5])
        ),
        "unified_macro_supply_parent_offset": int(
            trial.suggest_categorical("unified_macro_supply_parent_offset", [4, 5])
        ),
        "unified_macro_supply_max_generated": int(
            trial.suggest_categorical("unified_macro_supply_max_generated", [512, 768])
        ),
        "macro_chain_witness_parent_width": int(
            trial.suggest_categorical("macro_chain_witness_parent_width", [3, 4, 5])
        ),
        "macro_chain_witness_max_generated": int(
            trial.suggest_categorical("macro_chain_witness_max_generated", [0, 256, 512])
        ),
        "frg_frontier_child_certify_time_limit": float(
            trial.suggest_categorical("frg_frontier_child_certify_time_limit", [0.002, 0.004, 0.006])
        ),
        "post_polish_successor_cache": bool(trial.suggest_categorical("post_polish_successor_cache", [False, True])),
        "post_polish_max_attempts": int(trial.suggest_categorical("post_polish_max_attempts", [0, 640, 960])),
    }


def build_open_command(
    *,
    trial_name: str,
    params: Mapping[str, object],
    output_dir: Path,
    time_limit: float,
    jobs: int | str,
    source: Path | None = None,
    families: Sequence[str] = OPEN120_FAMILIES,
    limit_per_family: int = 20,
    case_ids: str = "",
    max_total: int | None = None,
    alpha: float = 10.0,
    variant: str = V4_BASE_VARIANT,
    no_business_cap: bool = True,
) -> list[str]:
    source = source or (ROOT / "external" / "DLTS")
    command = [
        sys.executable,
        str(OPEN_RUNNER),
        "--source",
        str(source),
        "--no-auto-clone",
        "--families",
        *families,
        "--limit-per-family",
        str(limit_per_family),
        "--variant",
        variant,
        "--time-limit",
        str(time_limit),
        "--alpha",
        str(alpha),
        "--jobs",
        str(jobs),
    ]
    if case_ids:
        command.extend(["--case-ids", case_ids])
    if max_total is not None:
        command.extend(["--max-total", str(max_total)])
    if no_business_cap:
        command.append("--no-business-cap")
    command.extend(["--output", str(output_dir / f"{trial_name}_open.csv")])
    command.extend(params_to_cli_args(params, allowed_params=CONSERVATIVE_TUNABLE_PARAMS))
    return command


def build_paper_command(
    *,
    trial_name: str,
    params: Mapping[str, object],
    output_dir: Path,
    time_limit: float,
    jobs: int,
    source: Path | None = None,
    datasets: Sequence[str] = ("BF", "CV"),
    families: Sequence[str] | None = None,
    sample_per_family: int = 10,
    max_total: int | None = None,
    alpha: float = 10.0,
    variant: str = V4_BASE_VARIANT,
    no_business_cap: bool = True,
) -> list[str]:
    source = source or (ROOT / "external" / "paper_cpmp")
    command = [
        sys.executable,
        str(PAPER_RUNNER),
        "--source",
        str(source),
        "--datasets",
        *datasets,
        "--sample-per-family",
        str(sample_per_family),
        "--variant",
        variant,
        "--time-limit",
        str(time_limit),
        "--alpha",
        str(alpha),
        "--jobs",
        str(jobs),
    ]
    if families:
        command.extend(["--families", *families])
    if max_total is not None:
        command.extend(["--max-total", str(max_total)])
    if no_business_cap:
        command.append("--no-business-cap")
    command.extend(["--output", str(output_dir / f"{trial_name}_paper.csv")])
    command.extend(params_to_cli_args(params, allowed_params=CONSERVATIVE_TUNABLE_PARAMS))
    return command


def _failure_summary(label: str, expected_cases: int) -> BenchmarkSummary:
    cases = max(1, int(expected_cases or 1))
    return BenchmarkSummary(
        label=label,
        cases=cases,
        solved=0,
        moves=0,
        full_quality_penalty=1_000_000.0 * cases,
        runtime_sec=0.0,
        failures=cases,
    )


def _expected_cases_from_args(args: argparse.Namespace) -> dict[str, int]:
    expected: dict[str, int] = {}
    if args.expected_open_cases > 0:
        expected["open"] = args.expected_open_cases
    if args.expected_bf_cases > 0:
        expected["BF"] = args.expected_bf_cases
    if args.expected_cv_cases > 0:
        expected["CV"] = args.expected_cv_cases
    return expected


def _family_metrics_from_outputs(open_output: Path, paper_output: Path) -> dict[str, BenchmarkSummary]:
    metrics: dict[str, BenchmarkSummary] = {}
    if open_output.exists():
        metrics.update(summarize_csv_by_family(open_output, default_dataset="open"))
    if paper_output.exists():
        metrics.update(summarize_csv_by_family(paper_output, default_dataset="paper"))
    return metrics


def _baseline_family_metrics(args: argparse.Namespace, output_dir: Path) -> dict[str, BenchmarkSummary]:
    open_path = args.baseline_open_csv or (output_dir / "trial_0000_open.csv")
    paper_path = args.baseline_paper_csv or (output_dir / "trial_0000_paper.csv")
    return _family_metrics_from_outputs(open_path, paper_path)


def run_trial_once(
    *,
    trial_name: str,
    params: Mapping[str, object],
    args: argparse.Namespace,
) -> TrialOutcome:
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_cases = _expected_cases_from_args(args)

    open_cmd = build_open_command(
        trial_name=trial_name,
        params=params,
        output_dir=output_dir,
        time_limit=args.time_limit,
        jobs=args.jobs,
        source=args.open_source,
        families=tuple(args.open_families),
        limit_per_family=args.open_limit_per_family,
        case_ids=args.open_case_ids,
        max_total=args.open_max_total,
        alpha=args.alpha,
        variant=args.base_variant,
        no_business_cap=args.no_business_cap,
    )
    paper_cmd = build_paper_command(
        trial_name=trial_name,
        params=params,
        output_dir=output_dir,
        time_limit=args.time_limit,
        jobs=int(args.jobs) if str(args.jobs).isdigit() else 1,
        source=args.paper_source,
        datasets=tuple(args.paper_datasets),
        families=tuple(args.paper_families) if args.paper_families else None,
        sample_per_family=args.paper_sample_per_family,
        max_total=args.paper_max_total,
        alpha=args.alpha,
        variant=args.base_variant,
        no_business_cap=args.no_business_cap,
    )

    if args.dry_run:
        print("OPEN", " ".join(open_cmd), flush=True)
        print("PAPER", " ".join(paper_cmd), flush=True)
        summaries = [
            _failure_summary("open", expected_cases.get("open", 1)),
            _failure_summary("BF", expected_cases.get("BF", 1)),
            _failure_summary("CV", expected_cases.get("CV", 1)),
        ]
        return TrialOutcome(
            trial_name=trial_name,
            score=constraint_objective_score(summaries, expected_cases),
            params=dict(params),
            summaries=summaries,
            open_returncode=0,
            paper_returncode=0,
            open_output=str(output_dir / f"{trial_name}_open.csv"),
            paper_output=str(output_dir / f"{trial_name}_paper.csv"),
        )

    open_proc = subprocess.run(open_cmd, cwd=ROOT, check=False)
    paper_proc = subprocess.run(paper_cmd, cwd=ROOT, check=False)

    open_output = output_dir / f"{trial_name}_open.csv"
    paper_output = output_dir / f"{trial_name}_paper.csv"
    summaries: list[BenchmarkSummary] = []
    if open_proc.returncode == 0 and open_output.exists():
        summaries.append(summarize_result_csv(open_output, label="open"))
    else:
        summaries.append(_failure_summary("open", expected_cases.get("open", 0)))

    if paper_proc.returncode == 0 and paper_output.exists():
        summaries.extend(summarize_paper_csv_by_dataset(paper_output))
    else:
        for label in args.paper_datasets:
            summaries.append(_failure_summary(label, expected_cases.get(label, 0)))

    if args.objective == "family-safe":
        score = family_safe_objective_score(
            summaries,
            expected_cases,
            current_family=_family_metrics_from_outputs(open_output, paper_output),
            baseline_family=_baseline_family_metrics(args, output_dir),
            move_tolerance=args.family_regression_move_tolerance,
            move_regression_weight=args.family_regression_weight,
            solved_loss_penalty=args.family_solved_loss_penalty,
        )
    else:
        score = constraint_objective_score(summaries, expected_cases)
    outcome = TrialOutcome(
        trial_name=trial_name,
        score=score,
        params=dict(params),
        summaries=summaries,
        open_returncode=int(open_proc.returncode),
        paper_returncode=int(paper_proc.returncode),
        open_output=str(open_output),
        paper_output=str(paper_output),
    )
    write_trial_json(output_dir / f"{trial_name}_outcome.json", outcome)
    return outcome


def write_trial_json(path: Path, outcome: TrialOutcome) -> None:
    payload = asdict(outcome)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_trial_csv(path: Path, outcome: TrialOutcome) -> None:
    fields = [
        "trial_name",
        "score",
        "params_json",
        "open_solved",
        "open_cases",
        "bf_solved",
        "bf_cases",
        "cv_solved",
        "cv_cases",
        "moves_total",
        "full_quality_penalty_total",
        "runtime_sec_total",
        "open_returncode",
        "paper_returncode",
        "open_output",
        "paper_output",
    ]
    by_label = {summary.label: summary for summary in outcome.summaries}
    row = {
        "trial_name": outcome.trial_name,
        "score": outcome.score,
        "params_json": json.dumps(outcome.params, sort_keys=True),
        "open_solved": by_label.get("open", _failure_summary("open", 0)).solved,
        "open_cases": by_label.get("open", _failure_summary("open", 0)).cases,
        "bf_solved": by_label.get("BF", _failure_summary("BF", 0)).solved,
        "bf_cases": by_label.get("BF", _failure_summary("BF", 0)).cases,
        "cv_solved": by_label.get("CV", _failure_summary("CV", 0)).solved,
        "cv_cases": by_label.get("CV", _failure_summary("CV", 0)).cases,
        "moves_total": sum(summary.moves for summary in outcome.summaries),
        "full_quality_penalty_total": sum(summary.full_quality_penalty for summary in outcome.summaries),
        "runtime_sec_total": sum(summary.runtime_sec for summary in outcome.summaries),
        "open_returncode": outcome.open_returncode,
        "paper_returncode": outcome.paper_returncode,
        "open_output": outcome.open_output,
        "paper_output": outcome.paper_output,
    }
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_best_cli_params(study, output_dir: Path, study_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{study_name}_best_optuna_params.json"
    raw_path.write_text(json.dumps(study.best_params, indent=2, sort_keys=True), encoding="utf-8")

    outcome_path_value = getattr(study.best_trial, "user_attrs", {}).get("outcome_json")
    params = dict(study.best_params)
    if outcome_path_value:
        outcome_path = Path(outcome_path_value)
        if outcome_path.exists():
            payload = json.loads(outcome_path.read_text(encoding="utf-8"))
            params = dict(payload.get("params", params))

    best_path = output_dir / f"{study_name}_best_cli_params.json"
    best_path.write_text(json.dumps(params, indent=2, sort_keys=True), encoding="utf-8")
    return best_path


def enqueue_baseline_anchor_if_needed(study, *, enabled: bool) -> bool:
    if not enabled:
        return False
    if getattr(study, "trials", ()):
        return False
    study.enqueue_trial(dict(V4_BASELINE_OPTUNA_PARAMS), user_attrs={"anchor": "v4.0_default"})
    return True


def normalize_storage_url(storage: str | None, output_dir: Path) -> str | None:
    if not storage:
        return None
    if storage.startswith("sqlite:///"):
        return storage
    storage_path = Path(storage)
    storage_path = storage_path if storage_path.is_absolute() else output_dir / storage_path
    return f"sqlite:///{storage_path.as_posix()}"


def run_optuna(args: argparse.Namespace) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("Optuna is not installed. Install with: python -m pip install optuna") from exc

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_csv = output_dir / f"{args.study_name}_trials.csv"

    def objective(trial) -> float:
        if args.search_space == "safe":
            params = suggest_safe_v4_params(trial)
        else:
            params = suggest_conservative_v4_params(trial)
        trial_name = f"trial_{trial.number:04d}"
        outcome = run_trial_once(trial_name=trial_name, params=params, args=args)
        append_trial_csv(trials_csv, outcome)
        trial.set_user_attr("params_json", json.dumps(params, sort_keys=True))
        trial.set_user_attr("outcome_json", str(output_dir / f"{trial_name}_outcome.json"))
        return outcome.score

    storage = normalize_storage_url(args.storage, output_dir)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )
    enqueue_baseline_anchor_if_needed(study, enabled=args.enqueue_baseline_anchor)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    best_path = write_best_cli_params(study, output_dir, args.study_name)
    print(f"BEST_SCORE {study.best_value}", flush=True)
    print(f"BEST_PARAMS {best_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optuna tuner for the v4.0 chainaware CPMP mainline.")
    parser.add_argument("--base-variant", default=V4_BASE_VARIANT)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--study-name", default="v4_optuna_conservative")
    parser.add_argument("--search-space", choices=["conservative", "safe"], default="conservative")
    parser.add_argument("--objective", choices=["aggregate", "family-safe"], default="aggregate")
    parser.add_argument("--storage", default="")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "experiments" / "optuna_v4")
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--jobs", default="4")
    parser.add_argument("--open-source", type=Path, default=ROOT / "external" / "DLTS")
    parser.add_argument("--open-families", nargs="+", default=list(OPEN120_FAMILIES))
    parser.add_argument("--open-limit-per-family", type=int, default=20)
    parser.add_argument("--open-case-ids", default="")
    parser.add_argument("--open-max-total", type=int)
    parser.add_argument("--paper-source", type=Path, default=ROOT / "external" / "paper_cpmp")
    parser.add_argument("--paper-datasets", nargs="+", choices=["BF", "CV"], default=["BF", "CV"])
    parser.add_argument("--paper-families", nargs="+")
    parser.add_argument("--paper-sample-per-family", type=int, default=10)
    parser.add_argument("--paper-max-total", type=int)
    parser.add_argument("--expected-open-cases", type=int, default=0)
    parser.add_argument("--expected-bf-cases", type=int, default=0)
    parser.add_argument("--expected-cv-cases", type=int, default=0)
    parser.add_argument("--baseline-open-csv", type=Path)
    parser.add_argument("--baseline-paper-csv", type=Path)
    parser.add_argument("--family-regression-move-tolerance", type=float, default=20.0)
    parser.add_argument("--family-regression-weight", type=float, default=5_000.0)
    parser.add_argument("--family-solved-loss-penalty", type=float, default=1_000_000_000.0)
    parser.add_argument("--no-business-cap", dest="no_business_cap", action="store_true", default=True)
    parser.add_argument("--with-business-cap", dest="no_business_cap", action="store_false")
    parser.add_argument("--enqueue-baseline-anchor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    args.open_source = args.open_source if args.open_source.is_absolute() else ROOT / args.open_source
    args.paper_source = args.paper_source if args.paper_source.is_absolute() else ROOT / args.paper_source
    if args.baseline_open_csv is not None:
        args.baseline_open_csv = (
            args.baseline_open_csv if args.baseline_open_csv.is_absolute() else ROOT / args.baseline_open_csv
        )
    if args.baseline_paper_csv is not None:
        args.baseline_paper_csv = (
            args.baseline_paper_csv if args.baseline_paper_csv.is_absolute() else ROOT / args.baseline_paper_csv
        )
    run_optuna(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
