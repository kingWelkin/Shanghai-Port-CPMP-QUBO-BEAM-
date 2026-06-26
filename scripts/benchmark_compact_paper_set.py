"""Run compact beam/QUBO profiles on sampled BF/CV paper benchmark cases."""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_bs_frg_paper import PaperCase, discover_paper_cases, load_bf_bay, load_cv_dat  # noqa: E402
from benchmark_compact_beam_qubo_n0 import run_compact_beam_instance  # noqa: E402
from open_cpmp_benchmarks import (  # noqa: E402
    OPEN_RESULT_FIELDS,
    _TERMINAL_ESCAPE_LAZY_PORTFOLIO_VARIANTS,
    _TERMINAL_ESCAPE_PORTFOLIO_VARIANTS,
    _args_for_variant,
    _open_full_quality_penalty,
    _open_row_dict_from_instance_result,
    _open_row_valid,
    _select_open_no_regret_row,
    _open_validate_instance_result,
)


@dataclass(frozen=True)
class PaperRunTask:
    task_index: int
    case: PaperCase
    variant: str


@dataclass(frozen=True)
class PaperRunTaskResult:
    task_index: int
    case_id: str
    variant: str
    row: dict[str, object]


def discover_sampled_paper_cases(
    source: Path,
    *,
    datasets: Iterable[str] = ("BF", "CV"),
    sample_per_family: int = 10,
    max_total: int | None = None,
) -> list[PaperCase]:
    cases: list[PaperCase] = []
    for dataset in datasets:
        dataset_cases = discover_paper_cases(
            source,
            dataset=dataset,
            limit_per_family=sample_per_family,
        )
        for case in dataset_cases:
            cases.append(case)
            if max_total is not None and len(cases) >= max_total:
                return cases
    return cases


def _load_paper_case(case: PaperCase):
    if case.dataset == "BF":
        return load_bf_bay(case.instance_path)
    return load_cv_dat(case.instance_path)


def _is_true(value: object) -> bool:
    return value in (True, "True", "true", "1", 1)


def _paper_has_move_path(row: dict[str, object]) -> bool:
    return row.get("moves") not in (None, "")


def _paper_time_limit_ok(row: dict[str, object], *, time_limit: float) -> bool:
    try:
        runtime = float(row.get("runtime_sec"))
    except (TypeError, ValueError):
        return False
    return runtime <= float(time_limit)


def _apply_paper_scorecard(row: dict[str, object], *, time_limit: float) -> dict[str, object]:
    feasibility_ok = (
        _is_true(row.get("goal_validation_ok"))
        and row.get("final_n") == 0
        and _paper_has_move_path(row)
    )
    time_limit_ok = feasibility_ok and _paper_time_limit_ok(row, time_limit=time_limit)
    row.update(
        {
            "feasibility_ok": feasibility_ok,
            "time_limit_ok": time_limit_ok,
            "primary_ok": feasibility_ok and _is_true(row.get("business_ok")) and time_limit_ok,
        }
    )
    return row


def _is_solved_row(row: dict[str, object]) -> bool:
    return _is_true(row.get("goal_validation_ok")) and row.get("final_n") == 0 and row.get("moves") not in ("", None)


def row_from_paper_exception(
    case: PaperCase,
    args: argparse.Namespace,
    variant: str,
    exc: Exception,
) -> dict[str, object]:
    return {
        "dataset": case.dataset,
        "family": case.family,
        "case": case.case_id,
        "variant": variant,
        "source_file": str(case.instance_path),
        "reference_file": "",
        "reference_moves": "",
        "move_gap": "",
        "full_quality_penalty": 1_000_000,
        "reference_tiers": "",
        "business_alpha": args.alpha,
        "business_ok": False,
        "goal_validation_ok": False,
        "feasibility_ok": False,
        "time_limit_ok": False,
        "primary_ok": False,
        "validation_msg": f"{type(exc).__name__}: {exc}",
        "status": "ERROR",
        "moves": "",
        "runtime_sec": 0.0,
        "final_n": "",
        "n0": "",
    }


def _numeric(row: dict[str, object], field: str, default: float = 0.0) -> float:
    value = row.get(field)
    if value in (None, ""):
        return default
    return float(value)


def _run_paper_variant_row(
    case: PaperCase,
    args: argparse.Namespace,
    variant: str,
    instance=None,
) -> dict[str, object]:
    if instance is None:
        instance = _load_paper_case(case)
    compact_args = _args_for_variant(args, variant, instance)
    result = run_compact_beam_instance(case.case_id, instance, compact_args)
    row = _open_row_dict_from_instance_result(result, instance, compact_args)
    business_ok, goal_ok, msg = _open_validate_instance_result(result, instance, business_alpha=args.alpha)
    row.update(
        {
            "dataset": case.dataset,
            "family": case.family,
            "variant": variant,
            "source_file": str(case.instance_path),
            "reference_file": "",
            "reference_moves": "",
            "move_gap": "",
            "reference_tiers": "",
            "business_alpha": args.alpha,
            "business_ok": business_ok,
            "goal_validation_ok": goal_ok,
            "validation_msg": msg,
        }
    )
    row = _apply_paper_scorecard(row, time_limit=args.time_limit)
    row["full_quality_penalty"] = _open_full_quality_penalty(row)
    return row


def _run_paper_portfolio_row(
    case: PaperCase,
    args: argparse.Namespace,
    variant: str,
    child_variants: tuple[str, ...],
    *,
    stop_on_first_valid: bool = False,
) -> dict[str, object]:
    instance = _load_paper_case(case)
    candidates: list[dict[str, object]] = []
    portfolio_runtime_sec = 0.0
    for child_variant in child_variants:
        child_row = _run_paper_variant_row(case, args, child_variant, instance=instance)
        portfolio_runtime_sec += _numeric(child_row, "runtime_sec", 0.0)
        candidates.append(child_row)
        if stop_on_first_valid and _open_row_valid(child_row):
            break
    selected = dict(_select_open_no_regret_row(candidates))
    selected["variant"] = variant
    selected["portfolio_profiles"] = ",".join(child_variants)
    selected["portfolio_selected_profile"] = selected.get("solver_profile", "")
    selected["portfolio_runtime_sec"] = f"{portfolio_runtime_sec:.6f}"
    selected["full_quality_penalty"] = _open_full_quality_penalty(selected)
    return selected


def _run_task(task: PaperRunTask, args_payload: dict[str, object]) -> PaperRunTaskResult:
    args = argparse.Namespace(**args_payload)
    try:
        if task.variant == "qubo_scheduler_state_v5_qubo_diversity_union_terminal_escape_portfolio":
            row = _run_paper_portfolio_row(task.case, args, task.variant, _TERMINAL_ESCAPE_PORTFOLIO_VARIANTS)
        elif task.variant == "qubo_scheduler_state_v5_qubo_diversity_union_terminal_escape_lazy_portfolio":
            row = _run_paper_portfolio_row(
                task.case,
                args,
                task.variant,
                _TERMINAL_ESCAPE_LAZY_PORTFOLIO_VARIANTS,
                stop_on_first_valid=True,
            )
        else:
            row = _run_paper_variant_row(task.case, args, task.variant)
    except Exception as exc:  # Keep long benchmark batches from losing completed rows.
        row = row_from_paper_exception(task.case, args, task.variant, exc)
    return PaperRunTaskResult(task.task_index, task.case.case_id, task.variant, row)


def _collect_rows(tasks: list[PaperRunTask], args: argparse.Namespace) -> list[dict[str, object]]:
    payload = dict(vars(args))
    total = len(tasks)
    results: list[PaperRunTaskResult] = []
    if args.jobs <= 1:
        for idx, task in enumerate(tasks, start=1):
            print(f"CASE_START {idx}/{total} {task.variant} {task.case.dataset}/{task.case.family}/{task.case.case_id}", flush=True)
            result = _run_task(task, payload)
            results.append(result)
            _print_case_done(result)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
            futures = []
            for idx, task in enumerate(tasks, start=1):
                print(
                    f"CASE_START {idx}/{total} {task.variant} {task.case.dataset}/{task.case.family}/{task.case.case_id}",
                    flush=True,
                )
                futures.append(executor.submit(_run_task, task, payload))
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                _print_case_done(result)
    return [result.row for result in sorted(results, key=lambda item: item.task_index)]


def _print_case_done(result: PaperRunTaskResult) -> None:
    row = result.row
    print(
        f"CASE_DONE {result.variant} {row['dataset']}/{row['family']}/{result.case_id} "
        f"final_n={row['final_n']} moves={row['moves']} runtime={float(row['runtime_sec']):.3f}s "
        f"goal_ok={row['goal_validation_ok']}",
        flush=True,
    )


def _write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OPEN_RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_groups(rows: list[dict[str, object]], key_fields: tuple[str, ...]) -> list[tuple[tuple[str, ...], dict[str, float | int]]]:
    groups: dict[tuple[str, ...], dict[str, float | int]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in key_fields)
        item = groups.setdefault(
            key,
            {
                "cases": 0,
                "solved": 0,
                "moves": 0,
                "penalty": 0,
                "runtime": 0.0,
            },
        )
        solved = _is_solved_row(row)
        item["cases"] = int(item["cases"]) + 1
        item["solved"] = int(item["solved"]) + (1 if solved else 0)
        item["moves"] = int(item["moves"]) + (int(_numeric(row, "moves", 0)) if solved else 0)
        item["penalty"] = int(item["penalty"]) + int(_numeric(row, "full_quality_penalty", 0))
        item["runtime"] = float(item["runtime"]) + _numeric(row, "runtime_sec", 0.0)
    return sorted(groups.items(), key=lambda item: item[0])


def _write_summary(rows: list[dict[str, object]], output: Path, source: Path, args: argparse.Namespace) -> Path:
    summary = output.with_name(output.stem + "_summary.md")
    solved = [row for row in rows if _is_solved_row(row)]
    moves = sum(int(_numeric(row, "moves", 0)) for row in solved)
    penalty = sum(int(_numeric(row, "full_quality_penalty", 0)) for row in rows)
    avg_runtime = sum(_numeric(row, "runtime_sec", 0.0) for row in rows) / max(1, len(rows))
    lines = [
        "# Compact Paper Set Benchmark Summary",
        "",
        f"source: `{source}`",
        f"result_csv: `{output}`",
        f"variant: `{args.variant}`",
        f"sample_per_family: `{args.sample_per_family}`",
        f"time_limit: `{args.time_limit}`",
        "",
        "| scope | cases | solved | moves | full penalty | avg runtime sec |",
        "|---|---:|---:|---:|---:|---:|",
        f"| ALL | {len(rows)} | {len(solved)} | {moves} | {penalty} | {avg_runtime:.3f} |",
    ]
    for (dataset,), item in _summary_groups(rows, ("dataset",)):
        avg = float(item["runtime"]) / max(1, int(item["cases"]))
        lines.append(
            f"| {dataset} | {int(item['cases'])} | {int(item['solved'])} | {int(item['moves'])} | "
            f"{int(item['penalty'])} | {avg:.3f} |"
        )
    lines.extend(["", "## Family Breakdown", "", "| dataset | family | cases | solved | moves | full penalty | avg runtime sec |"])
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for (dataset, family), item in _summary_groups(rows, ("dataset", "family")):
        avg = float(item["runtime"]) / max(1, int(item["cases"]))
        lines.append(
            f"| {dataset} | {family} | {int(item['cases'])} | {int(item['solved'])} | {int(item['moves'])} | "
            f"{int(item['penalty'])} | {avg:.3f} |"
        )
    failures = [row for row in rows if not _is_solved_row(row)]
    lines.extend(["", "## Failures", ""])
    if failures:
        for row in failures:
            lines.append(
                f"- `{row['dataset']}/{row['family']}/{Path(str(row['source_file'])).stem}`: "
                f"status={row.get('status', '')} final_n={row.get('final_n', '')} "
                f"moves={row.get('moves', '')} runtime={float(_numeric(row, 'runtime_sec', 0.0)):.3f}s"
            )
    else:
        lines.append("- none")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _strict_positive_time_limit(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("strict paper scorecard requires --time-limit > 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "external" / "paper_cpmp")
    parser.add_argument("--datasets", nargs="+", choices=["BF", "CV", "CVS"], default=["BF", "CV"])
    parser.add_argument("--sample-per-family", type=int, default=10)
    parser.add_argument("--max-total", type=int)
    parser.add_argument("--variant", default="qubo_scheduler_state_v5_qubo_diversity_union_polish")
    parser.add_argument("--solver-profile", default="speed_v2")
    parser.add_argument("--time-limit", type=_strict_positive_time_limit, default=20.0)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--beam-width", type=int)
    parser.add_argument("--core-width", type=int)
    parser.add_argument("--parent-core-cap", type=int)
    parser.add_argument("--union-diversity-qubo-origin-similarity-penalty", type=float)
    parser.add_argument("--union-diversity-qubo-same-origin-parent-n-band-penalty", type=float)
    parser.add_argument("--union-rollout-probe-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--union-rollout-probe-max-candidates", type=int)
    parser.add_argument("--union-rollout-probe-time-limit", type=float)
    parser.add_argument("--union-rollout-probe-max-steps", type=int)
    parser.add_argument("--union-rollout-probe-feasible-bonus", type=float)
    parser.add_argument("--union-rollout-probe-final-n-weight", type=float)
    parser.add_argument("--union-rollout-probe-moves-weight", type=float)
    parser.add_argument("--union-rollout-probe-dead-end-penalty", type=float)
    parser.add_argument("--union-rollout-terminal-graft", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--union-rollout-terminal-graft-respect-cap",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--micro-rollout-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--micro-rollout-trace-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--receiver-safety-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--micro-rollout-tiebreak-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--micro-rollout-tiebreak-epsilon", type=float)
    parser.add_argument("--micro-rollout-window", type=int)
    parser.add_argument("--micro-first-child-limit", type=int)
    parser.add_argument("--micro-second-child-limit", type=int)
    parser.add_argument("--micro-cache-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--w-micro-delta-n", type=float)
    parser.add_argument("--w-micro-safe-continuation", type=float)
    parser.add_argument("--w-receiver-safe-delta", type=float)
    parser.add_argument("--w-last-buffer-consumed", type=float)
    parser.add_argument("--w-height-pressure", type=float)
    parser.add_argument("--frg-frontier-child-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-frontier-child-parent-width", type=int)
    parser.add_argument("--frg-frontier-child-limit", type=int)
    parser.add_argument("--frg-frontier-child-max-steps", type=int)
    parser.add_argument("--frg-frontier-child-append-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-frontier-child-adaptive-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-frontier-child-adaptive-parent-width", type=int)
    parser.add_argument("--frg-frontier-child-adaptive-limit", type=int)
    parser.add_argument("--frg-frontier-child-adaptive-max-steps", type=int)
    parser.add_argument("--frg-frontier-child-adaptive-plateau-window", type=int)
    parser.add_argument("--frg-frontier-child-adaptive-plateau-delta", type=int)
    parser.add_argument("--frg-frontier-child-adaptive-min-n", type=int)
    parser.add_argument("--frg-frontier-child-adaptive-small-stack-max", type=int)
    parser.add_argument("--frg-frontier-child-adaptive-min-container-count", type=int)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "compact_paper_sample.csv",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    cases = discover_sampled_paper_cases(
        source,
        datasets=args.datasets,
        sample_per_family=args.sample_per_family,
        max_total=args.max_total,
    )
    tasks = [PaperRunTask(task_index=idx, case=case, variant=args.variant) for idx, case in enumerate(cases)]
    rows = _collect_rows(tasks, args)
    _write_csv(rows, output)
    summary = _write_summary(rows, output, source, args)
    print(f"WROTE {output}", flush=True)
    print(f"WROTE {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
