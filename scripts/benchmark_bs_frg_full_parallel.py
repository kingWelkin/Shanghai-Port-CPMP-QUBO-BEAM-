"""Run standalone BS-FRG on open120 and paper BF/CV with process parallelism."""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_bs_frg_paper import discover_paper_cases, load_bf_bay, load_cv_dat  # noqa: E402
from benchmark_classical_cpmp_baselines import FIELDS, row_from_solution  # noqa: E402
from container_reshuffle.core import validate_instance  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.bs_frg import BsFrgSolver  # noqa: E402
from open_cpmp_benchmarks import discover_dlts_cases, load_dlts_dat  # noqa: E402


OUT_FIELDS = ["dataset", *FIELDS, "paper_pure", "beam_width", "k", "max_depth", "time_limit"]


@dataclass(frozen=True)
class Task:
    index: int
    dataset: str
    family: str
    case_id: str
    instance_path: str
    max_height: int | None
    reference_moves: int | None


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _is_solved(row: dict[str, object]) -> bool:
    return _truthy(row.get("goal_validation_ok")) and str(row.get("final_n")) == "0" and row.get("moves") not in ("", None)


def _load_task_instance(task: Task):
    path = Path(task.instance_path)
    if task.dataset == "DLTS":
        if task.max_height is None:
            raise ValueError(f"DLTS task {task.case_id} missing max_height")
        return load_dlts_dat(path, max_height=task.max_height)
    if task.dataset == "BF":
        return load_bf_bay(path)
    if task.dataset == "CV":
        return load_cv_dat(path)
    raise ValueError(f"unknown dataset: {task.dataset}")


def _run_task(payload: tuple[Task, dict[str, object]]) -> tuple[int, dict[str, object]]:
    task, settings = payload
    instance = _load_task_instance(task)
    try:
        validate_instance(instance)
        allow_priority_groups = False
    except ValueError:
        allow_priority_groups = True

    solver = BsFrgSolver(paper_pure=bool(settings["paper_pure"]))
    config = SolverConfig(
        max_depth=int(settings["max_depth"]),
        beam_width=int(settings["beam_width"]),
        candidate_limit_per_state=int(settings["k"]),
        time_limit_sec=float(settings["time_limit"]),
        entropy_alpha=10.0,
    )
    solution = solver.solve(instance, config)
    row = row_from_solution(
        instance,
        task.family,
        task.case_id,
        solution,
        reference_moves=task.reference_moves,
        allow_priority_groups=allow_priority_groups,
    )
    row.update(
        {
            "dataset": task.dataset,
            "paper_pure": bool(settings["paper_pure"]),
            "beam_width": int(settings["beam_width"]),
            "k": int(settings["k"]),
            "max_depth": int(settings["max_depth"]),
            "time_limit": float(settings["time_limit"]),
        }
    )
    return task.index, row


def _open_tasks(source: Path, families: list[str], limit_per_family: int | None, start_index: int) -> list[Task]:
    tasks: list[Task] = []
    for offset, case in enumerate(discover_dlts_cases(source, families=families, limit_per_family=limit_per_family)):
        tasks.append(
            Task(
                index=start_index + offset,
                dataset="DLTS",
                family=case.family,
                case_id=case.case_id,
                instance_path=str(case.instance_path),
                max_height=case.reference.max_height,
                reference_moves=case.reference.optimal_moves,
            )
        )
    return tasks


def _paper_tasks(
    source: Path,
    datasets: Iterable[str],
    families: list[str] | None,
    limit_per_family: int | None,
    start_index: int,
) -> list[Task]:
    tasks: list[Task] = []
    index = start_index
    for dataset in datasets:
        for case in discover_paper_cases(
            source,
            dataset=dataset,
            families=families,
            limit_per_family=limit_per_family,
        ):
            tasks.append(
                Task(
                    index=index,
                    dataset=case.dataset,
                    family=case.family,
                    case_id=case.case_id,
                    instance_path=str(case.instance_path),
                    max_height=None,
                    reference_moves=None,
                )
            )
            index += 1
    return tasks


def _write_summary(output: Path, rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), str(row["family"])), []).append(row)

    solved = [row for row in rows if _is_solved(row)]
    total_moves = sum(int(row["moves"]) for row in solved)
    avg_runtime = sum(float(row["runtime_sec"]) for row in rows) / max(1, len(rows))
    lines = [
        "# BS-FRG Full Parallel Benchmark Summary",
        "",
        f"Result CSV: `{output}`",
        "",
        "| scope | cases | solved | moves | avg runtime sec |",
        "|---|---:|---:|---:|---:|",
        f"| ALL | {len(rows)} | {len(solved)} | {total_moves if solved else ''} | {avg_runtime:.3f} |",
    ]
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        dataset_rows = [row for row in rows if str(row["dataset"]) == dataset]
        dataset_solved = [row for row in dataset_rows if _is_solved(row)]
        dataset_moves = sum(int(row["moves"]) for row in dataset_solved)
        dataset_runtime = sum(float(row["runtime_sec"]) for row in dataset_rows) / max(1, len(dataset_rows))
        lines.append(
            f"| {dataset} | {len(dataset_rows)} | {len(dataset_solved)} | "
            f"{dataset_moves if dataset_solved else ''} | {dataset_runtime:.3f} |"
        )
    lines.extend(["", "## Family Breakdown", "", "| dataset | family | cases | solved | moves | avg runtime sec |", "|---|---|---:|---:|---:|---:|"])
    for (dataset, family), family_rows in sorted(groups.items()):
        family_solved = [row for row in family_rows if _is_solved(row)]
        family_moves = sum(int(row["moves"]) for row in family_solved)
        family_runtime = sum(float(row["runtime_sec"]) for row in family_rows) / max(1, len(family_rows))
        lines.append(
            f"| {dataset} | {family} | {len(family_rows)} | {len(family_solved)} | "
            f"{family_moves if family_solved else ''} | {family_runtime:.3f} |"
        )

    failures = [row for row in rows if not _is_solved(row)]
    lines.extend(["", "## Unsolved", ""])
    if failures:
        lines.extend(
            f"- `{row['dataset']}/{row['family']}/{row['case']}`: final_n={row['final_n']} status={row['status']}"
            for row in failures
        )
    else:
        lines.append("- none")
    output.with_name(output.stem + "_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", choices=["open", "paper", "all"], default="all")
    parser.add_argument("--source-dlts", type=Path, default=ROOT / "external" / "DLTS")
    parser.add_argument("--source-paper", type=Path, default=ROOT / "external" / "paper_cpmp")
    parser.add_argument(
        "--open-families",
        nargs="+",
        default=["5x5_cv1_test", "5x5_cv2_test", "5x5_cv3_test", "7x5_cv1_test", "7x5_cv2_test", "7x5_cv3_test"],
    )
    parser.add_argument("--paper-datasets", nargs="+", default=["BF", "CV"])
    parser.add_argument("--paper-families", nargs="+")
    parser.add_argument("--limit-per-family", type=int)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--max-depth", type=int, default=1000)
    parser.add_argument("--beam-width", type=int, default=1000)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    parser.add_argument("--engineered", action="store_true", help="use engineered BS-FRG instead of paper-pure mode")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "bs_frg_full_parallel.csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_dlts = args.source_dlts if args.source_dlts.is_absolute() else ROOT / args.source_dlts
    source_paper = args.source_paper if args.source_paper.is_absolute() else ROOT / args.source_paper
    output = args.output if args.output.is_absolute() else ROOT / args.output

    tasks: list[Task] = []
    if args.surface in {"open", "all"}:
        tasks.extend(_open_tasks(source_dlts, args.open_families, args.limit_per_family, len(tasks)))
    if args.surface in {"paper", "all"}:
        tasks.extend(_paper_tasks(source_paper, args.paper_datasets, args.paper_families, args.limit_per_family, len(tasks)))
    if not tasks:
        raise SystemExit("no benchmark tasks discovered")

    settings = {
        "paper_pure": not args.engineered,
        "max_depth": args.max_depth,
        "beam_width": args.beam_width,
        "k": args.k,
        "time_limit": args.time_limit,
    }
    print(
        f"RUN_START tasks={len(tasks)} jobs={args.jobs} paper_pure={settings['paper_pure']} "
        f"beam_width={args.beam_width} k={args.k} max_depth={args.max_depth} time_limit={args.time_limit}",
        flush=True,
    )
    rows_by_index: dict[int, dict[str, object]] = {}
    with mp.Pool(processes=max(1, args.jobs)) as pool:
        for index, row in pool.imap_unordered(_run_task, ((task, settings) for task in tasks), chunksize=1):
            rows_by_index[index] = row
            print(
                f"CASE_DONE {row['dataset']} {row['family']}/{row['case']} "
                f"final_n={row['final_n']} moves={row['moves']} runtime={float(row['runtime_sec']):.3f}s "
                f"status={row['status']}",
                flush=True,
            )

    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(output, rows)
    print(f"WROTE {output}", flush=True)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
