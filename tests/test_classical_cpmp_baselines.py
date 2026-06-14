from __future__ import annotations

from container_reshuffle.core import Instance, ProblemVariant, validate_solution
from container_reshuffle.solvers.base import SolverConfig


def _tiny_cpmp() -> Instance:
    return Instance(
        "tiny_cpmp",
        ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=3,
        max_height=3,
        container_count=3,
        stacks=[[1, 3], [2], []],
    )


def test_lpfh_multi_heuristic_solves_tiny_cpmp():
    from container_reshuffle.solvers.classical_cpmp import LpfhMultiHeuristicSolver

    inst = _tiny_cpmp()
    sol = LpfhMultiHeuristicSolver().solve(inst, SolverConfig(max_depth=8, time_limit_sec=1.0))

    assert sol.feasible
    assert len(sol.moves) == 1
    ok, msg, _ = validate_solution(inst, sol)
    assert ok, msg


def test_target_guided_beam_solves_tiny_cpmp_with_legal_moves():
    from container_reshuffle.solvers.classical_cpmp import TargetGuidedBeamSolver

    inst = _tiny_cpmp()
    sol = TargetGuidedBeamSolver().solve(inst, SolverConfig(max_depth=8, beam_width=8, time_limit_sec=1.0))

    assert sol.feasible
    ok, msg, _ = validate_solution(inst, sol)
    assert ok, msg


def test_wang_mark_fixed_prefix_follows_descending_group_labels():
    from container_reshuffle.solvers.classical_cpmp import _wang_mark_fixed_prefix_for_group

    stacks = [[4, 3, 1], [4, 2], []]
    fixed = [0, 0, 0]

    assert _wang_mark_fixed_prefix_for_group(stacks, fixed, 4)
    assert fixed == [1, 1, 0]

    assert _wang_mark_fixed_prefix_for_group(stacks, fixed, 3)
    assert fixed == [2, 1, 0]


def test_wang_pair_cost_matches_giant_move_cost_definition():
    from container_reshuffle.solvers.classical_cpmp import _wang_pair_cost

    stacks = [[5, 3, 1], [4, 2], []]
    fixed = [1, 1, 0]

    assert _wang_pair_cost(stacks, fixed, source=0, level=1, target=0) == 1
    assert _wang_pair_cost(stacks, fixed, source=0, level=1, target=1) == 2


def test_wang_candidate_pairs_use_small_lp_low_t_tiebreak():
    from container_reshuffle.solvers.classical_cpmp import _wang_candidate_pairs

    stacks = [[5, 3], [4, 2], []]
    fixed = [1, 1, 0]

    pairs = _wang_candidate_pairs(stacks, fixed, group=3, max_height=3)

    assert (pairs[0].cost, pairs[0].target_fixed, pairs[0].target) == (0, 0, 2)
    assert (pairs[1].cost, pairs[1].target_fixed, pairs[1].target) == (0, 1, 0)
    assert (pairs[2].cost, pairs[2].target_fixed, pairs[2].target) == (1, 1, 1)


def test_wang_bs_g_solves_tiny_cpmp_with_legal_moves():
    from container_reshuffle.solvers.classical_cpmp import WangBsGiantMovesSolver

    inst = _tiny_cpmp()
    sol = WangBsGiantMovesSolver().solve(inst, SolverConfig(max_depth=8, beam_width=8, time_limit_sec=1.0))

    assert sol.feasible
    ok, msg, _ = validate_solution(inst, sol)
    assert ok, msg


def test_wang_bs_b_solves_tiny_cpmp_with_legal_moves():
    from container_reshuffle.solvers.classical_cpmp import WangBsBabyMovesSolver

    inst = _tiny_cpmp()
    sol = WangBsBabyMovesSolver().solve(inst, SolverConfig(max_depth=8, beam_width=8, time_limit_sec=1.0))

    assert sol.feasible
    ok, msg, _ = validate_solution(inst, sol)
    assert ok, msg


def test_brkga_decoder_solves_tiny_cpmp_deterministically():
    from container_reshuffle.solvers.classical_cpmp import BrkgaCpmpSolver

    inst = _tiny_cpmp()
    config = SolverConfig(max_depth=8, time_limit_sec=1.0, random_seed=123)

    first = BrkgaCpmpSolver(population_size=18, generations=8).solve(inst, config)
    second = BrkgaCpmpSolver(population_size=18, generations=8).solve(inst, config)

    assert first.feasible
    assert [move_to_tuple(move) for move in first.moves] == [move_to_tuple(move) for move in second.moves]
    ok, msg, _ = validate_solution(inst, first)
    assert ok, msg


def move_to_tuple(move):
    return move.box, move.from_stack, move.to_stack, move.from_level, move.to_level
