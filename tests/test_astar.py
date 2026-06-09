from __future__ import annotations

from collections import deque

from container_reshuffle.core import (
    Instance,
    Move,
    ProblemVariant,
    apply_move,
    create_initial_state,
    generate_legal_moves,
    hash_state,
    is_goal,
    validate_solution,
)
from container_reshuffle.scoring import cpmp_business_feasible
from container_reshuffle.solvers.astar import ExactAStarSolver, ExactIDAStarSolver
from container_reshuffle.solvers.base import SolverConfig


def _bfs_optimum(instance: Instance, max_depth: int = 8) -> int | None:
    initial = create_initial_state(instance)
    queue = deque([initial])
    seen = {hash_state(instance, initial)}
    while queue:
        state = queue.popleft()
        if is_goal(instance, state):
            return state.move_count
        if state.move_count >= max_depth:
            continue
        for move in generate_legal_moves(instance, state):
            nxt = apply_move(instance, state, move)
            if not cpmp_business_feasible(instance, nxt):
                continue
            key = hash_state(instance, nxt)
            if key in seen:
                continue
            seen.add(key)
            queue.append(nxt)
    return None


def test_exact_astar_cpmp_returns_verified_optimum():
    inst = Instance("cpmp", ProblemVariant.PREMARSHALLING_CPMP, 3, 2, 3, [[1, 3], [2], []])
    expected = _bfs_optimum(inst, max_depth=4)

    sol = ExactAStarSolver().solve(inst, SolverConfig(max_depth=4, time_limit_sec=10.0))

    assert expected == 1
    assert sol.feasible
    assert sol.optimal
    assert sol.status == "OPTIMAL"
    assert sol.objective == float(expected)
    ok, msg, _ = validate_solution(inst, sol)
    assert ok, msg


def test_exact_astar_crp_returns_verified_optimum():
    inst = Instance("crp", ProblemVariant.RETRIEVAL_CRP, 2, 2, 2, [[1, 2], []])
    sol = ExactAStarSolver().solve(inst, SolverConfig(max_depth=4, time_limit_sec=10.0))

    assert sol.feasible
    assert sol.optimal
    assert sol.objective == 1.0
    ok, msg, _ = validate_solution(inst, sol)
    assert ok, msg


def test_exact_astar_reports_depth_limit_without_false_optimality():
    inst = Instance("cpmp", ProblemVariant.PREMARSHALLING_CPMP, 2, 2, 2, [[1, 2], []])
    sol = ExactAStarSolver().solve(inst, SolverConfig(max_depth=0, time_limit_sec=10.0))

    assert not sol.feasible
    assert not sol.optimal
    assert sol.status == "DEPTH_LIMIT_NO_FEASIBLE_FOUND"


def test_exact_astar_reports_state_limit_without_false_optimality():
    inst = Instance("cpmp", ProblemVariant.PREMARSHALLING_CPMP, 3, 2, 3, [[1, 3], [2], []])
    sol = ExactAStarSolver().solve(
        inst,
        SolverConfig(max_depth=4, time_limit_sec=10.0, exact_max_visited_states=1),
    )

    assert not sol.feasible
    assert not sol.optimal
    assert sol.status == "STATE_LIMIT_NO_FEASIBLE_FOUND"


def test_exact_astar_can_prove_warm_start_incumbent_optimal():
    inst = Instance("cpmp", ProblemVariant.PREMARSHALLING_CPMP, 3, 2, 3, [[1, 3], [2], []])
    incumbent = [Move(0, 3, 0, 2)]

    sol = ExactAStarSolver().solve(
        inst,
        SolverConfig(max_depth=4, time_limit_sec=10.0, exact_incumbent_moves=incumbent),
    )

    assert sol.feasible
    assert sol.optimal
    assert sol.status == "OPTIMAL_WARM_START_PROVEN"
    assert sol.objective == 1.0


def test_exact_astar_warm_start_does_not_hide_better_solution():
    inst = Instance("cpmp", ProblemVariant.PREMARSHALLING_CPMP, 3, 2, 3, [[1, 3], [2], []])
    incumbent = [
        Move(0, 2, 1, 2),
        Move(1, 3, 0, 1),
    ]

    sol = ExactAStarSolver().solve(
        inst,
        SolverConfig(max_depth=4, time_limit_sec=10.0, entropy_alpha=2.0, exact_incumbent_moves=incumbent),
    )

    assert sol.feasible
    assert sol.optimal
    assert sol.status == "OPTIMAL"
    assert sol.objective == 1.0


def test_exact_idastar_cpmp_returns_verified_optimum():
    inst = Instance("cpmp", ProblemVariant.PREMARSHALLING_CPMP, 3, 2, 3, [[1, 3], [2], []])
    expected = _bfs_optimum(inst, max_depth=4)

    sol = ExactIDAStarSolver().solve(inst, SolverConfig(max_depth=4, time_limit_sec=10.0))

    assert sol.feasible
    assert sol.optimal
    assert sol.status == "OPTIMAL"
    assert sol.objective == float(expected)
    ok, msg, _ = validate_solution(inst, sol)
    assert ok, msg


def test_exact_idastar_crp_returns_verified_optimum():
    inst = Instance("crp", ProblemVariant.RETRIEVAL_CRP, 2, 2, 2, [[1, 2], []])
    sol = ExactIDAStarSolver().solve(inst, SolverConfig(max_depth=4, time_limit_sec=10.0))

    assert sol.feasible
    assert sol.optimal
    assert sol.objective == 1.0
    ok, msg, _ = validate_solution(inst, sol)
    assert ok, msg
