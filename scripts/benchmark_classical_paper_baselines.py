"""Benchmark classical CPMP baselines on BF/CV paper benchmark instances."""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_bs_frg_paper import _load_case, discover_paper_cases  # noqa: E402
from benchmark_classical_cpmp_baselines import FIELDS, build_solver, row_from_solution  # noqa: E402
from container_reshuffle.core import validate_instance  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402


@dataclass(frozen=True)
class PaperClassicalTask:
    source: str
    dataset: str
    family: str
    case_id: str
    instance_path: str
    solver_name: str
    max_depth: int
    beam_width: int
    max_candidates_per_layer: int
    time_limit: float
    entropy_alpha: float
    seed: int


def _task_from_case(case, args: argparse.Namespace, solver_name: str) -> PaperClassicalTask:
    return PaperClassicalTask(
        source=str(args.source),
        dataset=case.dataset,
        family=case.family,
        case_id=case.case_id,
        instance_path=str(case.instance_path),
        solver_name=solver_name,
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        max_candidates_per_layer=args.max_candidates_per_layer,
        time_limit=args.time_limit,
        entropy_alpha=args.entropy_alpha,
        seed=args.seed,
    )


def _run_task(task: PaperClassicalTask) -> dict[str, object]:
    from benchmark_bs_frg_paper import PaperCase

    case = PaperCase(
        dataset=task.dataset,
        family=task.family,
        case_id=task.case_id,
        instance_path=Path(task.instance_path),
    )
    instance = _load_case(case)
    try:
        validate_instance(instance)
        allow_priority_groups = False
    except ValueError:
        allow_priority_groups = True
    solver = build_solver(task.solver_name)
    config = SolverConfig(
        max_depth=task.max_depth,
        beam_width=task.beam_width,
        max_candidates_per_layer=task.max_candidates_per_layer,
        time_limit_sec=task.time_limit,
        entropy_alpha=task.entropy_alpha,
        random_seed=task.seed,
    )
    solution = solver.solve(instance, config)
    return row_from_solution(
        instance,
        task.family,
        task.case_id,
        solution,
        reference_moves=None,
        allow_priority_groups=allow_priority_groups,
    )


def _summary_rows(rows: list[dict[str, object]]) -> list[str]:
    by_solver: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_solver.setdefault(str(row["solver"]), []).append(row)
    lines = [
        "| solver | solved | moves_sum | avg_runtime_sec |",
        "|---|---:|---:|---:|",
    ]
    for solver, solver_rows in sorted(by_solver.items()):
        solved = [
            row
            for row in solver_rows
            if str(row.get("goal_validation_ok")).lower() in {"true", "1", "yes", "ok"}
            and row.get("final_n") == 0
            and row.get("moves") != ""
        ]
        moves = sum(int(row["moves"]) for row in solved)
        runtime = sum(float(row["runtime_sec"]) for row in solver_rows) / max(1, len(solver_rows))
        lines.append(f"| {solver} | {len(solved)}/{len(solver_rows)} | {moves if solved else ''} | {runtime:.3f} |")
    return lines


def _write_summary(output: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    invalid = [
        row
        for row in rows
        if not (
            str(row.get("goal_validation_ok")).lower() in {"true", "1", "yes", "ok"}
            and row.get("final_n") == 0
            and row.get("moves") != ""
        )
    ]
    lines = [
        "# Classical Paper Baseline Summary",
        "",
        f"Result CSV: `{output}`",
        "",
        f"source: `{args.source}`",
        f"datasets: `{','.join(args.datasets)}`",
        f"solvers: `{','.join(args.solvers)}`",
        f"time_limit: `{args.time_limit}`",
        f"max_depth: `{args.max_depth}`",
        f"beam_width: `{args.beam_width}`",
        f"max_candidates_per_layer: `{args.max_candidates_per_layer}`",
        f"jobs: `{args.jobs}`",
        "",
        *_summary_rows(rows),
        "",
        "## Unsolved",
        "",
    ]
    if invalid:
        lines.extend(
            f"- `{row['family']}/{row['case']}` `{row['solver']}`: status={row['status']} final_n={row['final_n']} moves={row['moves']}"
            for row in invalid[:200]
        )
        if len(invalid) > 200:
            lines.append(f"- ... {len(invalid) - 200} more")
    else:
        lines.append("- none")
    output.with_name(output.stem + "_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "external" / "paper_cpmp")
    parser.add_argument("--datasets", nargs="+", choices=["BF", "CV", "CVS"], default=["BF", "CV"])
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--limit-per-family", type=int)
    parser.add_argument("--sample-per-family", type=int)
    parser.add_argument("--solvers", nargs="+", default=["wang_bs_b"])
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--max-depth", type=int, default=1000)
    parser.add_argument("--beam-width", type=int, default=100)
    parser.add_argument("--max-candidates-per-layer", type=int, default=64)
    parser.add_argument("--entropy-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "classical_paper_baselines.csv")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    limit_per_family = args.sample_per_family if args.sample_per_family is not None else args.limit_per_family

    cases = []
    for dataset in args.datasets:
        cases.extend(
            discover_paper_cases(
                args.source,
                dataset=dataset,
                families=args.families,
                limit_per_family=limit_per_family,
            )
        )
    tasks = [_task_from_case(case, args, solver_name) for case in cases for solver_name in args.solvers]
    rows: list[dict[str, object]] = []

    if args.jobs <= 1:
        for task in tasks:
            print(f"CASE_START {task.dataset} {task.family}/{task.case_id} {task.solver_name}", flush=True)
            row = _run_task(task)
            rows.append(row)
            print(
                f"CASE_DONE {task.family}/{task.case_id} {task.solver_name} final_n={row['final_n']} "
                f"moves={row['moves']} runtime={float(row['runtime_sec']):.3f}s status={row['status']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            future_to_task = {pool.submit(_run_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                row = future.result()
                rows.append(row)
                print(
                    f"CASE_DONE {task.family}/{task.case_id} {task.solver_name} final_n={row['final_n']} "
                    f"moves={row['moves']} runtime={float(row['runtime_sec']):.3f}s status={row['status']}",
                    flush=True,
                )

    rows.sort(key=lambda row: (str(row["solver"]), str(row["family"]), str(row["case"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(output, rows, args)
    print(f"WROTE {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
