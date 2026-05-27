"""Benchmark current greedy planner against exact MIP models on small instances."""

from __future__ import annotations

import csv
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from greedy.greedy_plan import plan_greedy_until_terminal, terminal_sort_reached
from greedy.multi_bay_loader import load_combined_zone_bays
from greedy.premarshalling_mip import (
    Move,
    PremarshallingInstance,
    PremarshallingSolution,
    Solve3DIntegerPositionMip,
    Solve4DSlotMip,
    validate_move_sequence,
)
from greedy.problem_types import BayLike, CompatProblem


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    instance: PremarshallingInstance
    max_steps: int


@dataclass
class BenchmarkResult:
    case: str
    method: str
    m: int
    H: int
    n: int
    max_steps: int
    feasible: bool
    optimal: Optional[bool]
    final_sorted: bool
    moves: Optional[int]
    mip_objective: Optional[float]
    gap_vs_best: Optional[float]
    wall_sec: float
    solver_runtime_sec: Optional[float]
    status: str
    sequence: str


CASES: List[BenchmarkCase] = [
    BenchmarkCase(
        "sorted_2x2",
        PremarshallingInstance(2, 2, 2, [[2, 1], []]),
        2,
    ),
    BenchmarkCase(
        "one_move_2x2",
        PremarshallingInstance(2, 2, 2, [[1, 2], []]),
        3,
    ),
    BenchmarkCase(
        "two_bad_stacks_3x2",
        PremarshallingInstance(3, 2, 4, [[1, 4], [2, 3], []]),
        4,
    ),
    BenchmarkCase(
        "buffer_needed_3x3",
        PremarshallingInstance(3, 3, 5, [[1, 5], [2, 4], [3]]),
        5,
    ),
    BenchmarkCase(
        "mixed_4x3",
        PremarshallingInstance(4, 3, 6, [[1, 6], [2, 5], [3, 4], []]),
        6,
    ),
]

REAL_DATA_CASE_SPECS = [
    ("real_A1_stacks_1_2_3", "A1", [1, 2, 3], 5),
    ("real_A1_stacks_1_2_4", "A1", [1, 2, 4], 6),
    ("real_A2_stacks_1_2_8", "A2", [1, 2, 8], 6),
    ("real_B1_stacks_1_2_3", "B1", [1, 2, 3], 5),
    ("real_C1_stacks_1_2_3", "C1", [1, 2, 3], 5),
]


def _derive_real_data_case(
    name: str,
    zone: str,
    stack_indices: List[int],
    max_steps: int,
) -> BenchmarkCase:
    problem = load_combined_zone_bays(ROOT / "data" / "input", zone, ["01", "26"])
    seq_stacks: List[List[int]] = []
    seq_ids: List[int] = []
    for global_i in stack_indices:
        stack: List[int] = []
        for level in range(problem.H):
            seq = problem.bay.seq[global_i - 1][level]
            if seq is not None:
                seq_int = int(seq)
                stack.append(seq_int)
                seq_ids.append(seq_int)
        seq_stacks.append(stack)

    rank = {seq: idx + 1 for idx, seq in enumerate(sorted(set(seq_ids)))}
    stacks = [[rank[seq] for seq in stack] for stack in seq_stacks]
    return BenchmarkCase(
        name,
        PremarshallingInstance(
            StackCount=len(stacks),
            MaxHeight=problem.H,
            ContainerCount=len(rank),
            Stacks=stacks,
        ),
        max_steps,
    )


def _all_cases() -> List[BenchmarkCase]:
    real_cases = [
        _derive_real_data_case(name, zone, stacks, max_steps)
        for name, zone, stacks, max_steps in REAL_DATA_CASE_SPECS
    ]
    return CASES + real_cases


def _to_compat_problem(instance: PremarshallingInstance) -> CompatProblem:
    pri = [
        list(stack) + [None] * (instance.MaxHeight - len(stack))
        for stack in instance.Stacks
    ]
    return CompatProblem(
        W=instance.StackCount,
        H=instance.MaxHeight,
        N=instance.ContainerCount,
        T=instance.ContainerCount * 3,
        bay=BayLike(pri=pri, seq=[list(row) for row in pri]),
    )


def _format_moves(moves: Iterable[Move]) -> str:
    parts = [
        f"{mv.Step}:{mv.Box} {mv.FromStack}->{mv.ToStack}"
        for mv in sorted(moves, key=lambda x: x.Step)
    ]
    return "; ".join(parts) if parts else "-"


def _run_greedy(case: BenchmarkCase) -> BenchmarkResult:
    inst = case.instance
    problem = _to_compat_problem(inst)
    t0 = time.perf_counter()
    raw_moves = plan_greedy_until_terminal(
        problem,
        seed=42,
        max_moves=case.max_steps,
        verbose=False,
        use_bay_bias=False,
    )
    wall = time.perf_counter() - t0
    moves = [
        Move(
            Step=idx,
            Box=int(raw[5]),
            FromStack=int(raw[1]),
            ToStack=int(raw[3]),
            FromLevel=int(raw[2]),
            ToLevel=int(raw[4]),
        )
        for idx, raw in enumerate(raw_moves, start=1)
    ]
    final_sorted = terminal_sort_reached(problem, raw_moves)
    feasible = False
    status = "NOT_SORTED"
    if final_sorted:
        validate_move_sequence(inst, moves, expected_objective=len(moves))
        feasible = True
        status = "SORTED"

    return BenchmarkResult(
        case=case.name,
        method="Greedy",
        m=inst.StackCount,
        H=inst.MaxHeight,
        n=inst.ContainerCount,
        max_steps=case.max_steps,
        feasible=feasible,
        optimal=None,
        final_sorted=final_sorted,
        moves=len(moves),
        mip_objective=None,
        gap_vs_best=None,
        wall_sec=wall,
        solver_runtime_sec=None,
        status=status,
        sequence=_format_moves(moves),
    )


def _run_mip(
    case: BenchmarkCase,
    method: str,
    solver: Callable[..., PremarshallingSolution],
    *,
    time_limit_sec: float,
    diagnostics_dir: Path,
) -> BenchmarkResult:
    inst = case.instance
    t0 = time.perf_counter()
    sol = solver(
        inst,
        case.max_steps,
        timeLimitSec=time_limit_sec,
        diagnosticsDir=str(diagnostics_dir / method.lower()),
    )
    wall = time.perf_counter() - t0
    return BenchmarkResult(
        case=case.name,
        method=method,
        m=inst.StackCount,
        H=inst.MaxHeight,
        n=inst.ContainerCount,
        max_steps=case.max_steps,
        feasible=sol.Feasible,
        optimal=sol.Optimal,
        final_sorted=sol.Feasible,
        moves=len(sol.Moves) if sol.Feasible else None,
        mip_objective=sol.Objective if sol.Feasible else None,
        gap_vs_best=None,
        wall_sec=wall,
        solver_runtime_sec=sol.RuntimeSec,
        status=sol.Status,
        sequence=_format_moves(sol.Moves),
    )


def _apply_gaps(results: List[BenchmarkResult]) -> None:
    best_by_case = {}
    for result in results:
        if result.method == "Greedy":
            continue
        if result.optimal and result.mip_objective is not None:
            best_by_case[result.case] = min(
                best_by_case.get(result.case, result.mip_objective),
                result.mip_objective,
            )
    for result in results:
        best = best_by_case.get(result.case)
        if best is None:
            continue
        if result.method == "Greedy":
            result.gap_vs_best = (
                float(result.moves) - best if result.feasible and result.moves is not None else None
            )
        elif result.mip_objective is not None:
            result.gap_vs_best = result.mip_objective - best


def _write_csv(results: List[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(BenchmarkResult.__annotations__.keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)


def _fmt_num(value: Optional[float], ndigits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{ndigits}f}"


def _write_markdown(results: List[BenchmarkResult], path: Path, csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Greedy vs Exact MIP Benchmark",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Python: {platform.python_version()}",
        "Gurobi time limit per MIP solve: 20 seconds",
        "",
        "This benchmark uses small pre-marshalling instances plus small cases derived from `data/input` by selecting a few real stacks and re-ranking their original seq IDs to `1..n`.",
        "The exact MIP models are not intended for the full 60-container production cases.",
        "",
        f"CSV: `{csv_path.as_posix()}`",
        "",
        "## Summary Table",
        "",
        "| Case | Method | m | H | n | maxSteps | Feasible | Optimal | Moves/Obj | Gap vs best | Wall sec | Solver sec | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        move_obj = r.moves if r.method == "Greedy" else r.mip_objective
        lines.append(
            "| "
            + " | ".join(
                [
                    r.case,
                    r.method,
                    str(r.m),
                    str(r.H),
                    str(r.n),
                    str(r.max_steps),
                    str(r.feasible),
                    "-" if r.optimal is None else str(r.optimal),
                    "-" if move_obj is None else _fmt_num(float(move_obj), 0),
                    _fmt_num(r.gap_vs_best, 0),
                    _fmt_num(r.wall_sec, 4),
                    _fmt_num(r.solver_runtime_sec, 4),
                    r.status,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Move Sequences",
            "",
        ]
    )
    for r in results:
        lines.append(f"- `{r.case}` / `{r.method}`: {r.sequence}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Wall sec` includes Python model construction, Gurobi solve, extraction, and validation.",
            "- `Solver sec` is Gurobi's own runtime and is blank for the greedy planner.",
            "- `Gap vs best` uses the best proven optimal MIP objective for each case.",
            "- Greedy rows marked `NOT_SORTED` are partial heuristic trajectories, not feasible final pre-marshalling solutions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    diagnostics = ROOT / "tmp" / "benchmark_mip_diagnostics"
    results: List[BenchmarkResult] = []
    for case in _all_cases():
        results.append(_run_greedy(case))
        results.append(
            _run_mip(
                case,
                "MIP-3D",
                Solve3DIntegerPositionMip,
                time_limit_sec=20.0,
                diagnostics_dir=diagnostics,
            )
        )
        results.append(
            _run_mip(
                case,
                "MIP-4D",
                Solve4DSlotMip,
                time_limit_sec=20.0,
                diagnostics_dir=diagnostics,
            )
        )
    _apply_gaps(results)
    csv_path = ROOT / "docs" / "experiments" / "greedy_vs_mip.csv"
    md_path = ROOT / "docs" / "experiments" / "greedy_vs_mip.md"
    _write_csv(results, csv_path)
    _write_markdown(results, md_path, csv_path.relative_to(ROOT))
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
