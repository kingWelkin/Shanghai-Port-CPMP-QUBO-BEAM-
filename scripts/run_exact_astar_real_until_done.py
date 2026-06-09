"""Run exact A* on one real paired case until proof or exhaustive failure."""

from __future__ import annotations

import argparse
import heapq
import sys
import time
from math import inf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_full_real_business_algorithms import _instance_from_zone  # noqa: E402
from greedy.input_loader import load_problem_from_input_txt  # noqa: E402
from container_reshuffle.core import (  # noqa: E402
    Instance,
    ProblemVariant,
    State,
    apply_move,
    create_initial_state,
    generate_legal_moves,
    hash_state,
    is_goal,
    validate_solution,
)
from container_reshuffle.features import disorder_box_count, entropy_limit  # noqa: E402
from container_reshuffle.scoring import cpmp_business_feasible  # noqa: E402
from container_reshuffle.solvers.astar import ExactAStarSolver, ExactIDAStarSolver  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig, solution_from_state  # noqa: E402


def _initial_n(instance) -> int:
    return disorder_box_count(instance, State([list(stack) for stack in instance.stacks]))


def _instance_from_single_bay(zone: str, bay: str) -> Instance:
    path = ROOT / "data" / "input" / f"{zone.upper()}_{bay}_input.txt"
    problem = load_problem_from_input_txt(path)
    placements: list[tuple[int, int, int]] = []
    for stack_idx, row in enumerate(problem.bay.seq):
        for level, seq in enumerate(row):
            if seq is not None:
                placements.append((int(seq), stack_idx, level))
    rank_by_slot = {
        (stack_idx, level): rank
        for rank, (_seq, stack_idx, level) in enumerate(sorted(placements), start=1)
    }
    stacks = []
    for stack_idx, row in enumerate(problem.bay.seq):
        stack = []
        for level, seq in enumerate(row):
            if seq is not None:
                stack.append(rank_by_slot[(stack_idx, level)])
        stacks.append(stack)
    return Instance(
        instance_id=f"{zone.upper()}_{bay}_single",
        variant=ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=problem.W,
        max_height=problem.H,
        container_count=len(placements),
        stacks=stacks,
    )


def _run_solver_api(instance, config: SolverConfig, algorithm: str) -> int:
    solver = ExactIDAStarSolver() if algorithm == "exact_idastar" else ExactAStarSolver()
    sol = solver.solve(instance, config)
    print(
        "FINAL "
        f"algorithm={algorithm} status={sol.status} feasible={sol.feasible} optimal={sol.optimal} "
        f"objective={sol.objective} expanded={sol.expanded_states} generated={sol.generated_states} "
        f"visited={sol.visited_states} runtime={sol.runtime_sec:.3f}s",
        flush=True,
    )
    return 0 if sol.feasible or sol.status in {"NO_FEASIBLE_FOUND_EXHAUSTIVE", "DEPTH_LIMIT_NO_FEASIBLE_FOUND"} else 1


def run(
    zone: str,
    bay: str | None,
    alpha: float,
    progress_interval: float,
    canonical: bool,
    algorithm: str,
    time_limit: float,
) -> int:
    instance = _instance_from_single_bay(zone, bay) if bay else _instance_from_zone(zone)
    cap = entropy_limit(instance, alpha)
    config = SolverConfig(
        max_depth=cap,
        time_limit_sec=time_limit,
        entropy_alpha=alpha,
        enable_canonical_hash=canonical,
    )
    solver = ExactAStarSolver()
    start = time.perf_counter()
    next_progress = start

    initial = create_initial_state(instance)
    print(
        "START "
        f"case={instance.instance_id} stacks={instance.stack_count} H={instance.max_height} "
        f"containers={instance.container_count} initial_N={_initial_n(instance)} cap={cap}",
        flush=True,
    )
    if algorithm != "exact_astar_progress":
        return _run_solver_api(instance, config, algorithm)

    if is_goal(instance, initial):
        sol = solution_from_state(solver.name, instance, initial, start, status="OPTIMAL", optimal=True)
        print(f"FINAL status={sol.status} feasible={sol.feasible} optimal={sol.optimal} objective={sol.objective}", flush=True)
        return 0

    initial_key = hash_state(instance, initial, canonical)
    best_g = {initial_key: 0}
    heap: list[tuple[float, int, int, int, object]] = []
    counter = 0
    h0 = solver._lower_bound(instance, initial)
    heapq.heappush(heap, (h0, h0, 0, counter, initial))

    expanded = 0
    generated = 0
    depth_limited = False
    incumbent = None

    while heap:
        now = time.perf_counter()
        if now >= next_progress:
            best_obj = "None" if incumbent is None else str(incumbent.move_count)
            print(
                "PROGRESS "
                f"elapsed={now - start:.1f}s expanded={expanded} generated={generated} "
                f"visited={len(best_g)} open={len(heap)} incumbent={best_obj}",
                flush=True,
            )
            next_progress = now + progress_interval

        _f, _h, g, _idx, state = heapq.heappop(heap)
        state_key = hash_state(instance, state, canonical)
        if g != best_g.get(state_key, inf):
            continue
        if is_goal(instance, state):
            sol = solution_from_state(
                solver.name,
                instance,
                state,
                start,
                status="OPTIMAL",
                optimal=True,
                expanded_states=expanded,
                generated_states=generated,
                visited_states=len(best_g),
            )
            ok, msg, _ = validate_solution(instance, sol)
            print(
                "FINAL "
                f"status={sol.status} feasible={sol.feasible} optimal={sol.optimal} "
                f"objective={sol.objective} validation={ok}:{msg} "
                f"expanded={expanded} generated={generated} visited={len(best_g)}",
                flush=True,
            )
            return 0
        if state.move_count >= config.max_depth:
            depth_limited = True
            continue

        expanded += 1
        for move in generate_legal_moves(instance, state):
            try:
                nxt = apply_move(instance, state, move)
            except ValueError:
                continue
            if not cpmp_business_feasible(instance, nxt, alpha):
                continue

            generated += 1
            nxt_key = hash_state(instance, nxt, canonical)
            nxt_g = nxt.move_count
            if nxt_g >= best_g.get(nxt_key, inf):
                continue

            best_g[nxt_key] = nxt_g
            if is_goal(instance, nxt) and (incumbent is None or nxt.move_count < incumbent.move_count):
                incumbent = nxt
                print(
                    "INCUMBENT "
                    f"elapsed={time.perf_counter() - start:.1f}s moves={nxt.move_count} "
                    f"expanded={expanded} generated={generated} visited={len(best_g)}",
                    flush=True,
                )

            counter += 1
            nxt_h = solver._lower_bound(instance, nxt)
            heapq.heappush(heap, (nxt_g + nxt_h, nxt_h, nxt_g, counter, nxt))

    status = "DEPTH_LIMIT_NO_FEASIBLE_FOUND" if depth_limited else "NO_FEASIBLE_FOUND_EXHAUSTIVE"
    print(
        "FINAL "
        f"status={status} feasible=False optimal=False objective=inf "
        f"expanded={expanded} generated={generated} visited={len(best_g)}",
        flush=True,
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="C1")
    parser.add_argument("--bay", help="Optional single bay suffix such as 01 or 26. Omit to run combined 01+26.")
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--time-limit", type=float, default=1_000_000_000.0)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument(
        "--algorithm",
        choices=["exact_astar_progress", "exact_astar", "exact_idastar"],
        default="exact_astar_progress",
    )
    args = parser.parse_args()
    return run(args.zone, args.bay, args.alpha, args.progress_interval, args.canonical, args.algorithm, args.time_limit)


if __name__ == "__main__":
    raise SystemExit(main())
