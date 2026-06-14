from __future__ import annotations

import sys
from pathlib import Path

from container_reshuffle.core import Instance, ProblemVariant
from container_reshuffle.solvers.base import SolverConfig

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_classical_runner_builds_requested_solver():
    from benchmark_classical_cpmp_baselines import build_solver

    assert build_solver("lpfh_multi_heuristic").name == "lpfh_multi_heuristic"
    assert build_solver("target_guided_beam").name == "target_guided_beam"
    assert build_solver("wang_bs_g").name == "wang_bs_g"
    assert build_solver("wang_bs_b").name == "wang_bs_b"
    assert build_solver("brkga_cpmp").name == "brkga_cpmp"


def test_classical_runner_row_reports_valid_tiny_solution():
    from benchmark_classical_cpmp_baselines import row_from_solution
    from container_reshuffle.solvers.classical_cpmp import TargetGuidedBeamSolver

    inst = Instance("tiny", ProblemVariant.PREMARSHALLING_CPMP, 3, 3, 3, [[1, 3], [2], []])
    sol = TargetGuidedBeamSolver().solve(inst, SolverConfig(max_depth=8, beam_width=8, entropy_alpha=10.0))

    row = row_from_solution(inst, "tiny_family", "tiny_case", sol, reference_moves=1)

    assert row["solver"] == "target_guided_beam"
    assert row["moves"] == 1
    assert row["final_n"] == 0
    assert row["move_gap"] == 0
    assert row["goal_validation_ok"] is True


def test_classical_runner_row_accepts_wang_priority_group_solution():
    from benchmark_classical_cpmp_baselines import row_from_solution
    from container_reshuffle.solvers.classical_cpmp import WangBsGiantMovesSolver

    inst = Instance("priority_group", ProblemVariant.PREMARSHALLING_CPMP, 3, 3, 4, [[1, 3], [2, 3], []])
    sol = WangBsGiantMovesSolver().solve(inst, SolverConfig(max_depth=12, beam_width=8, entropy_alpha=10.0))

    row = row_from_solution(inst, "pg_family", "pg_case", sol, reference_moves=None, allow_priority_groups=True)

    assert row["solver"] == "wang_bs_g"
    assert row["final_n"] == 0
    assert row["goal_validation_ok"] is True
