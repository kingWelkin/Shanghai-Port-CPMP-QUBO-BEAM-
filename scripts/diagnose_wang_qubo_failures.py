"""Focused diagnostics for Wang-vs-QUBO open CPMP failure cases."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_compact_beam_qubo_n0 import (
    _candidate_wang_probe_score,
    make_compact_args,
    run_compact_beam_instance,
)
from container_reshuffle.core import Instance, Move, State, apply_move
from container_reshuffle.features import future_disorder_count_stacks
from container_reshuffle.solvers.base import SolverConfig
from container_reshuffle.solvers.classical_cpmp import WangBsBabyMovesSolver
from open_cpmp_benchmarks import discover_dlts_cases, ensure_dlts_source, load_dlts_dat
from run_business_exact_target_search_sqlite import _encode

DEFAULT_CASES = ("5x5_cv1_test_4", "5x5_cv2_test_10")
FIELDS = [
    "case_id",
    "family",
    "compact_status",
    "compact_final_n",
    "compact_moves",
    "compact_plateau_tail",
    "fixed_prefix",
    "fix_pair",
    "parent_top_share",
    "wang_moves",
    "wang_first_moves",
]


def _case_selected(case_id: str, exact_cases: set[str]) -> bool:
    return not exact_cases or case_id in exact_cases


def _plateau_tail(history: Iterable[int], width: int = 3) -> str:
    values = list(history)[-max(1, width):]
    return ",".join(str(value) for value in values)


def _format_first_moves(moves: list[Move], limit: int = 8) -> list[str]:
    return [
        f"{move.step}:{move.box}:{move.from_stack}->{move.to_stack}"
        for move in moves[: max(0, limit)]
    ]


def _stacks_after_moves(instance: Instance, moves: list[Move]) -> list[list[int]]:
    state = State(stacks=[list(stack) for stack in instance.stacks])
    for move in moves:
        state = apply_move(instance, state, move)
    return state.stacks


def _wang_signal_fields(
    stacks: list[list[int]],
    stack_count: int,
    max_height: int,
    compact_args: argparse.Namespace,
) -> tuple[int, int]:
    compact_args._stack_count = int(getattr(compact_args, "_stack_count", 0) or stack_count)
    compact_args._max_height = int(getattr(compact_args, "_max_height", 0) or max_height)
    code, _n = _encode(stacks, stack_count, max_height)
    _score, fix_pair, fixed_prefix, _target_gap = _candidate_wang_probe_score(code, compact_args)
    return fixed_prefix, fix_pair


def _diagnostic_row(
    *,
    case_id: str,
    family: str,
    compact_status: str,
    compact_final_n: int | None,
    compact_moves: int | None,
    compact_best_n_history: list[int],
    fixed_prefix: int,
    fix_pair: int,
    parent_top_share: float,
    wang_moves: int | None,
    wang_first_moves: list[str],
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "family": family,
        "compact_status": compact_status,
        "compact_final_n": "" if compact_final_n is None else compact_final_n,
        "compact_moves": "" if compact_moves is None else compact_moves,
        "compact_plateau_tail": _plateau_tail(compact_best_n_history),
        "fixed_prefix": fixed_prefix,
        "fix_pair": fix_pair,
        "parent_top_share": f"{parent_top_share:.6f}",
        "wang_moves": "" if wang_moves is None else wang_moves,
        "wang_first_moves": ";".join(wang_first_moves),
    }


def _write_rows(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "external" / "DLTS")
    parser.add_argument("--families", nargs="+", default=["5x5_cv1_test"])
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--compact-solver-profile", default="scheduler_state_v5_wang_rescue")
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--wang-move-limit", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/wang_qubo_failure_autopsy.csv"),
    )
    args = parser.parse_args()

    source = ensure_dlts_source(args.source, auto_clone=False, refresh=False)
    cases = discover_dlts_cases(source, families=args.families, limit_per_family=None, max_total=None)
    exact = set(args.cases)
    rows: list[dict[str, object]] = []
    for case in cases:
        if not _case_selected(case.case_id, exact):
            continue
        instance = load_dlts_dat(case.instance_path, max_height=case.reference.max_height)
        compact_args = make_compact_args(
            solver_profile=args.compact_solver_profile,
            alpha=10.0,
            time_limit=args.time_limit,
            max_moves=None,
        )
        compact_result = run_compact_beam_instance(case.case_id, instance, compact_args)
        compact_stacks = _stacks_after_moves(instance, compact_result.moves)
        compact_final_n = future_disorder_count_stacks(compact_stacks)
        fixed_prefix, fix_pair = _wang_signal_fields(
            compact_stacks,
            instance.stack_count,
            instance.max_height,
            compact_args,
        )
        wang_solution = WangBsBabyMovesSolver().solve(
            instance,
            SolverConfig(
                max_depth=200,
                beam_width=100,
                max_candidates_per_layer=64,
                time_limit_sec=args.time_limit,
            ),
        )
        rows.append(
            _diagnostic_row(
                case_id=case.case_id,
                family=case.family,
                compact_status=compact_result.status,
                compact_final_n=compact_final_n,
                compact_moves=(len(compact_result.moves) if compact_result.moves else None),
                compact_best_n_history=list(getattr(compact_args, "_best_n_history", [])),
                fixed_prefix=fixed_prefix,
                fix_pair=fix_pair,
                parent_top_share=float(getattr(compact_args, "_layer_parent_top_share", 0.0)),
                wang_moves=(len(wang_solution.moves) if wang_solution.moves else None),
                wang_first_moves=_format_first_moves(
                    wang_solution.moves,
                    limit=args.wang_move_limit,
                ),
            )
        )
    _write_rows(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
