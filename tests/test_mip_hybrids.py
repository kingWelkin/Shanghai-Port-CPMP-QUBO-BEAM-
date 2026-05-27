from __future__ import annotations

import pytest

pytest.importorskip("gurobipy")

from container_reshuffle.core import Instance, ProblemVariant, validate_solution
from container_reshuffle.mip.hybrid_solvers import (
    FinalStackAssignmentMipSolver,
    PatternColumnGenerationSolver,
    RollingHorizonMipSolver,
)
from container_reshuffle.solvers.base import SolverConfig


def test_mip_hybrid_solvers_tiny_cpmp_validate():
    inst = Instance("tiny", ProblemVariant.PREMARSHALLING_CPMP, 3, 2, 3, [[1, 3], [2], []])
    cfg = SolverConfig(max_depth=4, rolling_depth=2, prefix_commit_length=1, time_limit_sec=20)
    for solver in (RollingHorizonMipSolver(), FinalStackAssignmentMipSolver(), PatternColumnGenerationSolver()):
        sol = solver.solve(inst, cfg)
        assert sol.feasible, sol.status
        ok, msg, _ = validate_solution(inst, sol)
        assert ok, msg
