from __future__ import annotations

import pytest

gurobipy = pytest.importorskip("gurobipy")

from container_reshuffle.core import Instance, ProblemVariant, validate_solution
from container_reshuffle.mip import solve_cpmp_3d_integer_position, solve_cpmp_4d_slot_occupancy


def test_tiny_cpmp_mips_match_and_validate():
    inst = Instance("tiny", ProblemVariant.PREMARSHALLING_CPMP, 3, 2, 3, [[1, 3], [2], []])
    sol3 = solve_cpmp_3d_integer_position(inst, max_steps=3, time_limit_sec=20)
    sol4 = solve_cpmp_4d_slot_occupancy(inst, max_steps=3, time_limit_sec=20)
    assert sol3.feasible, sol3.status
    assert sol4.feasible, sol4.status
    assert sol3.objective == pytest.approx(sol4.objective)
    for sol in (sol3, sol4):
        ok, msg, _ = validate_solution(inst, sol)
        assert ok, msg
