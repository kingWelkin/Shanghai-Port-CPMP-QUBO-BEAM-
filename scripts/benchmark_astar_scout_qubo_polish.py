"""A* no-proof scout followed by QUBO path polish.

The scout phase is the fast global-open target search from
``benchmark_target_astar_memory_n0.py`` with QUBO disabled.  After it reaches
N=0, this script re-encodes the found trajectory and applies the compact QUBO
post-polish routine to try to shorten the suffix/window path.  This keeps the
A* property the experiments care about here: many visited states, fast target
reachability, no optimality proof.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import benchmark_compact_beam_qubo_n0 as compact  # noqa: E402
import benchmark_target_astar_memory_n0 as astar  # noqa: E402
from benchmark_explainable_rule_macro_beam import _write_explain_csv  # noqa: E402
from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from benchmark_fast_qubo_integrated_v1 import _exact_moves_by_case, _move_gap, _parse_case, _rel  # noqa: E402
from benchmark_full_real_business_algorithms import TrajectoryResult  # noqa: E402
from container_reshuffle.core import Move, apply_move, create_initial_state  # noqa: E402
from container_reshuffle.features import disorder_box_count, entropy_limit, max_entropy_along_moves  # noqa: E402
from run_business_exact_target_search_sqlite import _encode  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


SOLVER_NAME = "astar_scout_qubo_polish_v1"
RESULT_FIELDS = list(TrajectoryResult.__annotations__.keys())
ACCEPTANCE_FIELDS = astar.ACCEPTANCE_FIELDS


def _case_filter(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return [case if case.endswith("_single") else f"{case}_single" for case in args.cases]
    return [f"{zone}_{bay}_single" for zone, bay in discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))]


def _codes_from_moves(zone: str, bay: str, moves: list[Move]) -> list[bytes]:
    instance = _instance_from_single_bay(zone, bay)
    state = create_initial_state(instance)
    code, _n_value = _encode(state.stacks, instance.stack_count, instance.max_height)
    codes = [code]
    for move in moves:
        state = apply_move(instance, state, move)
        code, _n_value = _encode(state.stacks, instance.stack_count, instance.max_height)
        codes.append(code)
    return codes


def _polish_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        post_polish_qubo=True,
        post_polish_passes=args.post_polish_passes,
        post_polish_focus_n=args.post_polish_focus_n,
        post_polish_tail_span=args.post_polish_tail_span,
        post_polish_max_window=args.post_polish_max_window,
        post_polish_max_windows=args.post_polish_max_windows,
        post_polish_select_windows=args.post_polish_select_windows,
        post_polish_max_qubo_vars=args.post_polish_max_qubo_vars,
        post_polish_max_bridge_depth=args.post_polish_max_bridge_depth,
        post_polish_bridge_expanded=args.post_polish_bridge_expanded,
        post_polish_suffix_focus_n=args.post_polish_suffix_focus_n,
        post_polish_suffix_tail_span=args.post_polish_suffix_tail_span,
        post_polish_suffix_max_candidates=args.post_polish_suffix_max_candidates,
        post_polish_suffix_select=args.post_polish_suffix_select,
        post_polish_suffix_max_qubo_vars=args.post_polish_suffix_max_qubo_vars,
        post_polish_suffix_max_depth=args.post_polish_suffix_max_depth,
        post_polish_suffix_expanded=args.post_polish_suffix_expanded,
        post_polish_suffix_near_gap=args.post_polish_suffix_near_gap,
        post_polish_suffix_near_penalty=args.post_polish_suffix_near_penalty,
        post_polish_overlap_penalty=args.post_polish_overlap_penalty,
        post_polish_near_window_gap=args.post_polish_near_window_gap,
        post_polish_penalty_a=args.post_polish_penalty_a,
        post_polish_qubo_time_limit=args.post_polish_qubo_time_limit,
        post_polish_qubo_reads=args.post_polish_qubo_reads,
        post_polish_qubo_iterations=args.post_polish_qubo_iterations,
        seed=args.seed,
    )


def _astar_args(args: argparse.Namespace) -> argparse.Namespace:
    base = astar.argparse.Namespace if False else argparse.Namespace
    return base(
        cases=None,
        bays=args.bays,
        output=args.output,
        priority=args.priority,
        g_weight=args.g_weight,
        n_weight=args.n_weight,
        target_final_n=0,
        alpha=args.alpha,
        max_moves=args.max_moves,
        time_limit=args.time_limit,
        max_visited=args.max_visited,
        phase_schedule=args.phase_schedule,
        phase1_time_limit=args.phase1_time_limit,
        phase2_time_limit=args.phase2_time_limit,
        push_mode="batch",
        pop_batch_size=args.pop_batch_size,
        insert_batch_size=args.insert_batch_size,
        use_qubo=False,
        qubo_profile="frontier_cohort",
        max_qubo_calls=0,
        max_qubo_vars=48,
        qubo_core_width=96,
        qubo_supplement_width=10,
        qubo_min_candidates=128,
        qubo_priority_bonus=0.0001,
        qubo_diversity_weight=0.08,
        qubo_frontier_n_window=1,
        qubo_frontier_depth_window=2,
        qubo_rank_weight=1.0,
        qubo_n_weight=2.0,
        qubo_depth_weight=1.0,
        qubo_parent_weight=0.15,
        qubo_penalty_a=30.0,
        qubo_time_limit=0.003,
        qubo_reads=2,
        qubo_iterations=40,
        layout_qubo_target_weight=0.0,
        layout_qubo_time_limit=0.04,
        layout_qubo_reads=16,
        layout_qubo_iterations=240,
        layout_target_samples=6,
        action_qubo_max_calls=0,
        action_qubo_tail_n=8,
        action_qubo_max_vars=40,
        action_qubo_top_k=4,
        action_qubo_min_options=6,
        action_qubo_diversity_weight=0.12,
        action_qubo_n_weight=2.0,
        action_qubo_delta_weight=1.0,
        action_qubo_height_weight=0.20,
        action_qubo_dirty_weight=0.40,
        action_qubo_penalty_a=30.0,
        action_qubo_time_limit=0.001,
        action_qubo_reads=2,
        action_qubo_iterations=20,
        seed=args.seed,
    )


def _row_from_run(
    case: str,
    zone: str,
    bay: str,
    result: astar.MemorySearchResult,
    moves: list[Move],
    runtime_sec: float,
    status: str,
    qcalls: int,
    qvars: int,
    qtime: float,
) -> TrajectoryResult:
    instance = _instance_from_single_bay(zone, bay)
    initial = create_initial_state(instance)
    n0 = disorder_box_count(instance, initial)
    state = initial
    for move in moves:
        state = apply_move(instance, state, move)
    final_n = disorder_box_count(instance, state)
    max_g = max_entropy_along_moves(instance, moves) if moves else n0
    cap13 = entropy_limit(instance, 1.3)
    cap15 = entropy_limit(instance, 1.5)
    return TrajectoryResult(
        case=case,
        solver=SOLVER_NAME,
        n0=n0,
        cap_13=cap13,
        cap_15=cap15,
        moves=len(moves) if moves else None,
        final_n=final_n,
        reduction=n0 - final_n,
        g_end=(len(moves) + final_n) if moves else None,
        max_g=max_g,
        g_ok_13=max_g <= cap13,
        g_ok_15=max_g <= cap15,
        runtime_sec=runtime_sec,
        expanded_states=result.expanded,
        generated_states=result.generated,
        visited_states=result.visited,
        qubo_calls=result.qubo_calls + qcalls,
        qubo_total_vars=result.qubo_vars + qvars,
        qubo_time_sec=result.qubo_time_sec + qtime,
        status=status,
    )


def _write_rows(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_acceptance(
    rows: list[TrajectoryResult],
    validation: dict[str, tuple[bool, bool, str]],
    moves_paths: dict[str, Path],
    exact_refs: dict[str, int],
    path: Path,
) -> None:
    out = []
    for row in rows:
        business_ok, goal_ok, msg = validation[row.case]
        out.append(
            {
                "case": row.case,
                "selected_solver": row.solver,
                "final_n": row.final_n,
                "moves": row.moves,
                "runtime_sec": f"{row.runtime_sec:.6f}",
                "exact_moves": exact_refs.get(row.case, ""),
                "move_gap": _move_gap(row, exact_refs),
                "qubo_used": str(row.qubo_calls > 0),
                "selected_reason": row.status,
                "business_ok": str(business_ok),
                "goal_validation_ok": str(goal_ok),
                "validation_msg": msg,
                "moves_path": _rel(moves_paths[row.case]) if row.case in moves_paths else "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ACCEPTANCE_FIELDS)
        writer.writeheader()
        writer.writerows(out)


def _write_report(rows: list[TrajectoryResult], output: Path, acceptance: Path, report: Path) -> None:
    solved = sum(1 for row in rows if row.final_n == 0)
    avg_runtime = sum(row.runtime_sec for row in rows) / len(rows) if rows else 0.0
    avg_moves = sum(row.moves or 0 for row in rows if row.final_n == 0) / solved if solved else 0.0
    lines = [
        f"# {SOLVER_NAME}",
        "",
        "A* no-proof scout plus QUBO path polish.",
        "",
        f"Result CSV: `{_rel(output)}`",
        f"Acceptance CSV: `{_rel(acceptance)}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| cases | {len(rows)} |",
        f"| N=0 solved | {solved}/{len(rows)} |",
        f"| avg moves solved | {avg_moves:.3f} |",
        f"| avg runtime sec | {avg_runtime:.3f} |",
        f"| max runtime sec | {max((row.runtime_sec for row in rows), default=0.0):.3f} |",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "astar_scout_qubo_polish.csv")
    parser.add_argument("--priority", choices=["n_then_depth", "astar", "weighted_astar"], default="n_then_depth")
    parser.add_argument("--g-weight", type=float, default=1.0)
    parser.add_argument("--n-weight", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--time-limit", type=float, default=0.0)
    parser.add_argument("--max-visited", type=int, default=0)
    parser.add_argument("--phase-schedule", choices=["small_medium_large", "single"], default="small_medium_large")
    parser.add_argument("--phase1-time-limit", type=float, default=8.0)
    parser.add_argument("--phase2-time-limit", type=float, default=35.0)
    parser.add_argument("--pop-batch-size", type=int, default=5000)
    parser.add_argument("--insert-batch-size", type=int, default=50000)
    parser.add_argument("--post-polish-passes", type=int, default=2)
    parser.add_argument("--post-polish-focus-n", type=int, default=8)
    parser.add_argument("--post-polish-tail-span", type=int, default=14)
    parser.add_argument("--post-polish-max-window", type=int, default=8)
    parser.add_argument("--post-polish-max-windows", type=int, default=96)
    parser.add_argument("--post-polish-select-windows", type=int, default=0)
    parser.add_argument("--post-polish-max-qubo-vars", type=int, default=72)
    parser.add_argument("--post-polish-max-bridge-depth", type=int, default=7)
    parser.add_argument("--post-polish-bridge-expanded", type=int, default=20000)
    parser.add_argument("--post-polish-suffix-focus-n", type=int, default=9)
    parser.add_argument("--post-polish-suffix-tail-span", type=int, default=18)
    parser.add_argument("--post-polish-suffix-max-candidates", type=int, default=48)
    parser.add_argument("--post-polish-suffix-select", type=int, default=5)
    parser.add_argument("--post-polish-suffix-max-qubo-vars", type=int, default=48)
    parser.add_argument("--post-polish-suffix-max-depth", type=int, default=18)
    parser.add_argument("--post-polish-suffix-expanded", type=int, default=60000)
    parser.add_argument("--post-polish-suffix-near-gap", type=int, default=2)
    parser.add_argument("--post-polish-suffix-near-penalty", type=float, default=0.30)
    parser.add_argument("--post-polish-overlap-penalty", type=float, default=8.0)
    parser.add_argument("--post-polish-near-window-gap", type=int, default=2)
    parser.add_argument("--post-polish-penalty-a", type=float, default=18.0)
    parser.add_argument("--post-polish-qubo-time-limit", type=float, default=0.001)
    parser.add_argument("--post-polish-qubo-reads", type=int, default=2)
    parser.add_argument("--post-polish-qubo-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    acceptance = output.with_name(output.stem + "_acceptance.csv")
    moves_dir = output.with_name(output.stem + "_moves")
    report = output.with_suffix(".md")
    moves_dir.mkdir(parents=True, exist_ok=True)

    exact_refs = _exact_moves_by_case()
    rows: list[TrajectoryResult] = []
    validation: dict[str, tuple[bool, bool, str]] = {}
    moves_paths: dict[str, Path] = {}

    for idx, case in enumerate(_case_filter(args), start=1):
        zone, bay = _parse_case(case)
        instance = _instance_from_single_bay(zone, bay)
        print(f"CASE_START {idx} {case} solver={SOLVER_NAME}", flush=True)
        start = time.perf_counter()
        scout = astar._search_case(case, zone, bay, _astar_args(args))
        qcalls = qvars = 0
        qtime = 0.0
        moves = list(scout.moves)
        status = (
            f"{scout.status}; solver={SOLVER_NAME}; scout_solver=target_astar_memory_n0_v1; "
            f"proof=not_optimality_proof; scout_visited={scout.visited}; scout_peak_open={scout.peak_open}"
        )
        if scout.moves:
            codes = _codes_from_moves(zone, bay, scout.moves)
            deadline = start + args.time_limit if args.time_limit > 0 else float("inf")
            polished, polish_stats = compact._post_polish_codes_with_qubo(
                codes,
                instance.stack_count,
                instance.max_height,
                entropy_limit(instance, args.alpha),
                _polish_args(args),
                deadline,
            )
            qcalls = polish_stats.qcalls
            qvars = polish_stats.qvars
            qtime = polish_stats.qtime
            if len(polished) < len(codes):
                try:
                    moves = compact._codes_to_moves(zone, bay, polished)
                    status += (
                        f"; polish_improved=True; polish_moves_saved={len(codes) - len(polished)}; "
                        f"polish_attempts={polish_stats.attempts}; polish_qcalls={polish_stats.qcalls}"
                    )
                except RuntimeError:
                    moves = list(scout.moves)
                    status += "; polish_improved=False; polish_lift_failed_reverted=True"
            else:
                status += (
                    f"; polish_improved=False; polish_moves_saved=0; "
                    f"polish_attempts={polish_stats.attempts}; polish_qcalls={polish_stats.qcalls}"
                )
        runtime = time.perf_counter() - start
        row = _row_from_run(case, zone, bay, scout, moves, runtime, status, qcalls, qvars, qtime)
        business_ok, goal_ok, msg = astar._validate_result(
            astar.MemorySearchResult(
                case=case,
                status=status,
                moves=moves,
                runtime_sec=runtime,
                expanded=scout.expanded,
                generated=scout.generated,
                visited=scout.visited,
                open_count=scout.open_count,
                pruned_bound=scout.pruned_bound,
                pruned_business=scout.pruned_business,
                dominated=scout.dominated,
                stale_pops=scout.stale_pops,
                peak_open=scout.peak_open,
                flushes=scout.flushes,
                phase=scout.phase,
                phases_tried=scout.phases_tried,
                qubo_calls=scout.qubo_calls + qcalls,
                qubo_vars=scout.qubo_vars + qvars,
                qubo_time_sec=scout.qubo_time_sec + qtime,
            ),
            zone,
            bay,
        )
        rows.append(row)
        validation[case] = (business_ok, goal_ok, msg)
        if moves:
            move_path = moves_dir / f"{case}_moves.csv"
            _write_explain_csv(instance, moves, move_path)
            moves_paths[case] = move_path
        _write_rows(rows, output)
        _write_acceptance(rows, validation, moves_paths, exact_refs, acceptance)
        _write_report(rows, output, acceptance, report)
        print(
            f"CASE_DONE {case} final_n={row.final_n} moves={row.moves} runtime={row.runtime_sec:.3f}s "
            f"visited={row.visited_states} qcalls={row.qubo_calls} valid={business_ok and goal_ok}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
