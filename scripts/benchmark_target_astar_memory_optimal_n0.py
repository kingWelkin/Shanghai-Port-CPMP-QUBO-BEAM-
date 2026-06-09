"""Fresh in-memory optimal A* target search for N=0.

This is a proof-oriented counterpart to ``benchmark_target_astar_memory_n0.py``.
It uses the same compact byte-encoded states, but runs one continuous A* search
with ``f = depth + N(Future)`` and returns only when an N=0 state is popped from
the heap. Under the business transition model, one relocation can reduce
N(Future) by at most one, so N(Future) is an admissible and consistent lower
bound on the remaining moves.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam import _write_explain_csv  # noqa: E402
from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from benchmark_fast_qubo_integrated_v1 import _exact_moves_by_case, _move_gap, _parse_case, _rel  # noqa: E402
from benchmark_full_real_business_algorithms import TrajectoryResult  # noqa: E402
from benchmark_target_astar_memory_n0 import _search_case as _fast_memory_search_case  # noqa: E402
from container_reshuffle.core import Move, Solution, apply_move, create_initial_state, validate_solution  # noqa: E402
from container_reshuffle.features import disorder_box_count, entropy_limit, max_entropy_along_moves  # noqa: E402
from extract_business_path_from_sqlite import _lift_to_physical  # noqa: E402
from run_business_exact_target_search_sqlite import _encode, _remaining_lower_bound, _successors_with_n  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


SOLVER_NAME = "target_astar_memory_optimal_n0_v1"
RESULT_FIELDS = list(TrajectoryResult.__annotations__.keys())
ACCEPTANCE_FIELDS = [
    "case",
    "selected_solver",
    "final_n",
    "moves",
    "runtime_sec",
    "exact_moves",
    "move_gap",
    "qubo_used",
    "selected_reason",
    "business_ok",
    "goal_validation_ok",
    "validation_msg",
    "moves_path",
]


@dataclass
class OptimalSearchResult:
    case: str
    status: str
    moves: list[Move]
    runtime_sec: float
    expanded: int
    generated: int
    visited: int
    open_count: int
    pruned_bound: int
    pruned_business: int
    dominated: int
    stale_pops: int
    peak_open: int
    proof: bool
    incumbent_moves: int | None = None
    min_open_f: int | None = None


def _priority(n_value: int, depth: int) -> int:
    return depth + n_value


def _codes_to_moves(zone: str, bay: str, codes: list[bytes]) -> list[Move]:
    instance = _instance_from_single_bay(zone, bay)
    physical_rows = _lift_to_physical(instance.stacks, codes, instance.stack_count, instance.max_height)
    return [
        Move(
            step=int(row["step"]) - 1,
            box=int(row["box_rank"]),
            from_stack=int(row["from_stack"]),
            to_stack=int(row["to_stack"]),
            from_level=int(row["from_level"]),
            to_level=int(row["to_level"]),
        )
        for row in physical_rows
    ]


def _backtrack(parent: dict[bytes, bytes | None], target: bytes) -> list[bytes]:
    codes = [target]
    current = target
    while parent[current] is not None:
        current = parent[current]  # type: ignore[assignment]
        codes.append(current)
    codes.reverse()
    return codes


def _fast_incumbent(case: str, zone: str, bay: str, args: argparse.Namespace) -> list[Move]:
    if not args.use_fast_incumbent:
        return []
    fast_args = SimpleNamespace(
        priority="n_then_depth",
        g_weight=1.0,
        n_weight=1.0,
        target_final_n=args.target_final_n,
        alpha=args.alpha,
        max_moves=args.max_moves,
        time_limit=args.fast_incumbent_time_limit,
        max_visited=args.fast_incumbent_max_visited,
        phase_schedule="small_medium_large",
        phase1_time_limit=8.0,
        phase2_time_limit=35.0,
        push_mode="batch",
        pop_batch_size=5000,
        insert_batch_size=50000,
        use_qubo=True,
        qubo_profile="frontier_cohort",
        max_qubo_calls=12,
        max_qubo_vars=64,
        qubo_core_width=224,
        qubo_supplement_width=8,
        qubo_min_candidates=64,
        qubo_priority_bonus=0.0001,
        qubo_diversity_weight=0.10,
        qubo_frontier_n_window=0,
        qubo_frontier_depth_window=1,
        qubo_rank_weight=1.0,
        qubo_n_weight=2.0,
        qubo_depth_weight=1.0,
        qubo_parent_weight=0.15,
        qubo_penalty_a=30.0,
        qubo_time_limit=0.001,
        qubo_reads=2,
        qubo_iterations=20,
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
    result = _fast_memory_search_case(case, zone, bay, fast_args)
    if not result.moves:
        return []
    instance = _instance_from_single_bay(zone, bay)
    solution = Solution(
        solver_name="fast_incumbent_frontier_cohort",
        feasible=True,
        optimal=False,
        objective=float(len(result.moves)),
        runtime_sec=result.runtime_sec,
        moves=result.moves,
        status=result.status,
    )
    goal_ok, _goal_msg, _ = validate_solution(instance, solution)
    business_ok = max_entropy_along_moves(instance, result.moves) <= entropy_limit(instance, 1.3)
    return result.moves if goal_ok and business_ok else []


def _search_case(case: str, zone: str, bay: str, args: argparse.Namespace) -> OptimalSearchResult:
    start = time.perf_counter()
    deadline = start + args.time_limit if args.time_limit > 0 else math.inf
    instance = _instance_from_single_bay(zone, bay)
    initial, n0 = _encode(instance.stacks, instance.stack_count, instance.max_height)
    cap = entropy_limit(instance, args.alpha)
    depth_cap = min(cap, args.max_moves) if args.max_moves is not None else cap
    incumbent = _fast_incumbent(case, zone, bay, args)
    incumbent_bound = len(incumbent) if incumbent else math.inf
    proof_depth_cap = min(depth_cap, int(incumbent_bound) - 1) if incumbent_bound < math.inf else depth_cap

    best_depth: dict[bytes, int] = {initial: 0}
    parent: dict[bytes, bytes | None] = {initial: None}
    heap: list[tuple[int, int, int, int, bytes]] = []
    counter = 0
    heapq.heappush(heap, (_priority(n0, 0), n0, 0, counter, initial))

    expanded = generated = pruned_bound = pruned_business = dominated = stale_pops = 0
    peak_open = 1
    target: bytes | None = None
    status = "TIME_LIMIT_OR_MEMORY_LIMIT"
    stop_min_open_f: int | None = None

    while heap and time.perf_counter() < deadline:
        if incumbent_bound < math.inf and heap[0][0] >= incumbent_bound:
            status = "INCUMBENT_PROVEN_OPTIMAL"
            break
        _f, current_n, depth, _idx, code = heapq.heappop(heap)
        if depth != best_depth.get(code):
            stale_pops += 1
            continue

        if current_n <= args.target_final_n:
            target = code
            status = "OPTIMAL_TARGET_POPPED"
            break

        if _remaining_lower_bound(current_n, depth, proof_depth_cap) > args.target_final_n:
            pruned_bound += 1
            continue
        if depth >= proof_depth_cap:
            pruned_bound += 1
            continue

        expanded += 1
        next_depth = depth + 1
        for nxt, nxt_n in _successors_with_n(code, current_n, instance.stack_count, instance.max_height):
            generated += 1
            if args.max_visited > 0 and len(best_depth) >= args.max_visited:
                stop_min_open_f = heap[0][0] if heap else None
                heap.clear()
                status = "MAX_VISITED_REACHED"
                break
            if next_depth + nxt_n > cap:
                pruned_business += 1
                continue
            if _remaining_lower_bound(nxt_n, next_depth, proof_depth_cap) > args.target_final_n:
                pruned_bound += 1
                continue
            if next_depth + nxt_n >= incumbent_bound:
                pruned_bound += 1
                continue
            old_depth = best_depth.get(nxt)
            if old_depth is not None and next_depth >= old_depth:
                dominated += 1
                continue
            best_depth[nxt] = next_depth
            parent[nxt] = code
            counter += 1
            heapq.heappush(heap, (_priority(nxt_n, next_depth), nxt_n, next_depth, counter, nxt))
        peak_open = max(peak_open, len(heap))

    runtime = time.perf_counter() - start
    moves: list[Move] = []
    if target is not None:
        moves = _codes_to_moves(zone, bay, _backtrack(parent, target))
    elif incumbent:
        moves = incumbent
        if not heap and status not in {"MAX_VISITED_REACHED", "TIME_LIMIT_OR_MEMORY_LIMIT"}:
            status = "INCUMBENT_PROVEN_OPTIMAL"
    elif not heap and status != "MAX_VISITED_REACHED":
        status = "NO_TARGET_WITHIN_BUSINESS_CAP"
    if incumbent and not heap and status == "TIME_LIMIT_OR_MEMORY_LIMIT":
        status = "INCUMBENT_PROVEN_OPTIMAL"
    proof = (target is not None and status == "OPTIMAL_TARGET_POPPED") or status == "INCUMBENT_PROVEN_OPTIMAL"
    return OptimalSearchResult(
        case=case,
        status=status,
        moves=moves,
        runtime_sec=runtime,
        expanded=expanded,
        generated=generated,
        visited=len(best_depth),
        open_count=len(heap),
        pruned_bound=pruned_bound,
        pruned_business=pruned_business,
        dominated=dominated,
        stale_pops=stale_pops,
        peak_open=peak_open,
        proof=proof,
        incumbent_moves=len(incumbent) if incumbent else None,
        min_open_f=heap[0][0] if heap else stop_min_open_f,
    )


def _case_filter(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return [case if case.endswith("_single") else f"{case}_single" for case in args.cases]
    return [f"{zone}_{bay}_single" for zone, bay in discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))]


def _row_from_result(result: OptimalSearchResult, zone: str, bay: str, args: argparse.Namespace) -> TrajectoryResult:
    instance = _instance_from_single_bay(zone, bay)
    initial = create_initial_state(instance)
    n0 = disorder_box_count(instance, initial)
    if result.moves:
        state = initial
        for move in result.moves:
            state = apply_move(instance, state, move)
        final_n = disorder_box_count(instance, state)
        moves_count = len(result.moves)
        max_g = max_entropy_along_moves(instance, result.moves)
    else:
        final_n = n0
        moves_count = None
        max_g = n0
    cap13 = entropy_limit(instance, 1.3)
    cap15 = entropy_limit(instance, 1.5)
    return TrajectoryResult(
        case=result.case,
        solver=SOLVER_NAME,
        n0=n0,
        cap_13=cap13,
        cap_15=cap15,
        moves=moves_count,
        final_n=final_n,
        reduction=n0 - final_n,
        g_end=(moves_count + final_n) if moves_count is not None else None,
        max_g=max_g,
        g_ok_13=max_g <= cap13,
        g_ok_15=max_g <= cap15,
        runtime_sec=result.runtime_sec,
        expanded_states=result.expanded,
        generated_states=result.generated,
        visited_states=result.visited,
        qubo_calls=0,
        qubo_total_vars=0,
        qubo_time_sec=0.0,
        status=(
            f"{result.status}; solver={SOLVER_NAME}; fresh_run=True; storage=memory; "
            f"priority=depth_plus_n; target_stop=goal_popped; proof={result.proof}; "
            f"heuristic=N_future_admissible_consistent; open={result.open_count}; "
            f"incumbent_moves={result.incumbent_moves}; incumbent_source=frontier_cohort; "
            f"min_open_f={result.min_open_f}; "
            f"peak_open={result.peak_open}; stale_pops={result.stale_pops}; "
            f"pruned_bound={result.pruned_bound}; pruned_business={result.pruned_business}; "
            f"dominated={result.dominated}; max_visited={args.max_visited}; "
            f"depth_cap={min(entropy_limit(instance, args.alpha), args.max_moves) if args.max_moves is not None else entropy_limit(instance, args.alpha)}"
        ),
    )


def _validate_result(result: OptimalSearchResult, zone: str, bay: str) -> tuple[bool, bool, str]:
    if not result.moves:
        return False, False, result.status
    instance = _instance_from_single_bay(zone, bay)
    solution = Solution(
        solver_name=SOLVER_NAME,
        feasible=True,
        optimal=result.proof,
        objective=float(len(result.moves)),
        runtime_sec=result.runtime_sec,
        moves=result.moves,
        status=result.status,
    )
    goal_ok, goal_msg, _ = validate_solution(instance, solution)
    business_ok = max_entropy_along_moves(instance, result.moves) <= entropy_limit(instance, 1.3)
    return business_ok, goal_ok, goal_msg


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
                "qubo_used": "False",
                "selected_reason": row.status,
                "business_ok": str(business_ok),
                "goal_validation_ok": str(goal_ok),
                "validation_msg": msg,
                "moves_path": _rel(moves_paths.get(row.case, Path(""))),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ACCEPTANCE_FIELDS)
        writer.writeheader()
        writer.writerows(out)


def _write_report(rows: list[TrajectoryResult], output: Path, acceptance: Path, report: Path, exact_refs: dict[str, int]) -> None:
    solved = sum(1 for row in rows if row.final_n == 0)
    proved = sum(1 for row in rows if "proof=True" in row.status)
    avg_runtime = sum(row.runtime_sec for row in rows) / len(rows) if rows else 0.0
    gaps = [
        row.moves - exact_refs[row.case]
        for row in rows
        if row.final_n == 0 and row.moves is not None and row.case in exact_refs
    ]
    lines = [
        f"# {SOLVER_NAME}",
        "",
        "Fresh in-memory optimal target A* benchmark. No SQLite and no witness reuse.",
        "",
        f"Result CSV: `{_rel(output)}`",
        f"Acceptance CSV: `{_rel(acceptance)}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| cases | {len(rows)} |",
        f"| N=0 solved | {solved}/{len(rows)} |",
        f"| optimal proof | {proved}/{len(rows)} |",
        f"| avg runtime sec | {avg_runtime:.3f} |",
        f"| max runtime sec | {max((row.runtime_sec for row in rows), default=0.0):.3f} |",
        f"| avg move gap | {sum(gaps) / len(gaps) if gaps else 0.0:.3f} |",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "target_astar_memory_optimal_n0.csv")
    parser.add_argument("--target-final-n", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--time-limit", type=float, default=0.0, help="Per-case limit in seconds; 0 means no limit.")
    parser.add_argument("--max-visited", type=int, default=0, help="Per-case memory guard; 0 means no explicit guard.")
    parser.add_argument("--use-fast-incumbent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-incumbent-time-limit", type=float, default=120.0)
    parser.add_argument("--fast-incumbent-max-visited", type=int, default=8_000_000)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    acceptance = output.with_name(output.stem + "_acceptance.csv")
    moves_dir = output.with_name(output.stem + "_moves")
    report = output.with_suffix(".md")
    moves_dir.mkdir(parents=True, exist_ok=True)

    cases = _case_filter(args)
    exact_refs = _exact_moves_by_case()
    rows: list[TrajectoryResult] = []
    validation: dict[str, tuple[bool, bool, str]] = {}
    moves_paths: dict[str, Path] = {}

    for idx, case in enumerate(cases, start=1):
        zone, bay = _parse_case(case)
        print(f"CASE_START {idx}/{len(cases)} {case} solver={SOLVER_NAME}", flush=True)
        result = _search_case(case, zone, bay, args)
        row = _row_from_result(result, zone, bay, args)
        business_ok, goal_ok, msg = _validate_result(result, zone, bay)
        if result.moves:
            move_path = moves_dir / f"{case}_moves.csv"
            _write_explain_csv(_instance_from_single_bay(zone, bay), result.moves, move_path)
            moves_paths[case] = move_path
        rows.append(row)
        validation[case] = (business_ok, goal_ok, msg)
        _write_rows(rows, output)
        _write_acceptance(rows, validation, moves_paths, exact_refs, acceptance)
        _write_report(rows, output, acceptance, report, exact_refs)
        print(
            "CASE_DONE "
            f"{case} status={result.status} proof={result.proof} final_n={row.final_n} moves={row.moves} "
            f"runtime={row.runtime_sec:.3f}s valid={business_ok and goal_ok} "
            f"visited={result.visited} peak_open={result.peak_open}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
