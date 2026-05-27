from __future__ import annotations

from container_reshuffle.core import Instance, ProblemVariant, validate_solution
from container_reshuffle.solvers.base import SolverConfig
from container_reshuffle.solvers.greedy import GreedyPremarshallingSolver, GreedyRetrievalSolver


def test_cpmp_already_sorted_zero():
    inst = Instance("sorted", ProblemVariant.PREMARSHALLING_CPMP, 2, 2, 2, [[2, 1], []])
    sol = GreedyPremarshallingSolver().solve(inst, SolverConfig(max_depth=5))
    assert sol.feasible
    assert sol.objective == 0


def test_crp_target_on_top_zero():
    inst = Instance("crp", ProblemVariant.RETRIEVAL_CRP, 2, 2, 2, [[2, 1], []])
    sol = GreedyRetrievalSolver().solve(inst, SolverConfig(max_depth=5))
    assert sol.feasible
    assert sol.objective == 0


def test_crp_one_blocker_feasible():
    inst = Instance("crp", ProblemVariant.RETRIEVAL_CRP, 3, 2, 2, [[1, 2], [], []])
    sol = GreedyRetrievalSolver().solve(inst, SolverConfig(max_depth=5))
    assert sol.feasible
    assert sol.objective >= 1
    ok, msg, _ = validate_solution(inst, sol)
    assert ok, msg


def test_cpmp_simple_feasible_validates():
    inst = Instance("cpmp", ProblemVariant.PREMARSHALLING_CPMP, 2, 2, 2, [[1, 2], []])
    sol = GreedyPremarshallingSolver().solve(inst, SolverConfig(max_depth=5))
    assert sol.feasible
    ok, msg, _ = validate_solution(inst, sol)
    assert ok, msg
