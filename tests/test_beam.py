from __future__ import annotations

from container_reshuffle.core import Instance, ProblemVariant, validate_solution
from container_reshuffle.solvers.base import SolverConfig
from container_reshuffle.solvers.beam import BeamSearchSolver
from container_reshuffle.solvers.diverse_beam import DiverseBeamSearchSolver


def test_beam_width_one_runs():
    inst = Instance("cpmp", ProblemVariant.PREMARSHALLING_CPMP, 2, 2, 2, [[1, 2], []])
    sol = BeamSearchSolver().solve(inst, SolverConfig(max_depth=4, beam_width=1))
    assert sol.feasible


def test_beam_tiny_cpmp_validated():
    inst = Instance("cpmp", ProblemVariant.PREMARSHALLING_CPMP, 3, 2, 3, [[1, 3], [2], []])
    sol = BeamSearchSolver().solve(inst, SolverConfig(max_depth=4, beam_width=5))
    assert sol.feasible
    ok, msg, _ = validate_solution(inst, sol)
    assert ok, msg


def test_diverse_beam_runs():
    inst = Instance("cpmp", ProblemVariant.PREMARSHALLING_CPMP, 3, 2, 3, [[1, 3], [2], []])
    sol = DiverseBeamSearchSolver().solve(inst, SolverConfig(max_depth=4, beam_width=3))
    assert sol.status in {"FEASIBLE", "NO_FEASIBLE_FOUND"} or sol.feasible
