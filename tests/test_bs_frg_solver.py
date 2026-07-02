from __future__ import annotations

import sys
from pathlib import Path

from container_reshuffle.core import Instance, Move, ProblemVariant
from container_reshuffle.solvers.base import SolverConfig

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_bs_frg_solver_solves_direct_fill_instance():
    from container_reshuffle.solvers.bs_frg import BsFrgSolver

    instance = Instance(
        "tiny_direct_fill",
        ProblemVariant.PREMARSHALLING_CPMP,
        3,
        3,
        3,
        [[1, 3], [2], []],
    )

    solution = BsFrgSolver().solve(
        instance,
        SolverConfig(
            max_depth=8,
            beam_width=4,
            candidate_limit_per_state=3,
            time_limit_sec=1.0,
            entropy_alpha=10.0,
        ),
    )

    assert solution.solver_name == "bs_frg"
    assert solution.feasible is True
    assert len(solution.moves) == 1
    assert solution.moves[0].box == 3
    assert solution.status.startswith("FEASIBLE")


def test_bs_frg_solver_supports_priority_group_instances():
    from benchmark_classical_cpmp_baselines import row_from_solution
    from container_reshuffle.solvers.bs_frg import BsFrgSolver

    instance = Instance(
        "priority_group",
        ProblemVariant.PREMARSHALLING_CPMP,
        3,
        3,
        4,
        [[1, 3], [2, 3], []],
    )

    solution = BsFrgSolver().solve(
        instance,
        SolverConfig(
            max_depth=12,
            beam_width=8,
            candidate_limit_per_state=4,
            time_limit_sec=1.0,
            entropy_alpha=10.0,
        ),
    )
    row = row_from_solution(
        instance,
        "pg_family",
        "pg_case",
        solution,
        reference_moves=None,
        allow_priority_groups=True,
    )

    assert row["solver"] == "bs_frg"
    assert row["goal_validation_ok"] is True
    assert row["final_n"] == 0
    assert row["moves"] == len(solution.moves)


def test_frg_completion_honors_generated_limit(monkeypatch):
    import math

    import container_reshuffle.solvers.bs_frg as bs_frg

    start = bs_frg._FrgLayout(bs_frg._to_key([[2, 3], [1], []]), None, (0, 0, 0))
    child = bs_frg._FrgLayout(bs_frg._to_key([[2], [1, 3], []]), None, (0, 0, 0))
    move = Move(0, 3, 0, 1, 2, 2)

    def fake_step_children(layout, max_height, step, *, limit=None, paper_pure=False):
        del layout, max_height, step, limit, paper_pure
        return [bs_frg._Child(child, (move,), 5, (0.0,))]

    monkeypatch.setattr(bs_frg, "_frg_step_children", fake_step_children)

    result = bs_frg._frg_completion(
        start,
        3,
        start_step=0,
        max_steps=10,
        deadline=math.inf,
        max_generated=1,
    )

    assert result.status == "FRG_GENERATED_LIMIT"
    assert result.generated == 5
    assert result.moves == (move,)


def test_gen_sequence_uses_top_down_nonincreasing_order():
    from container_reshuffle.solvers.bs_frg import _gen_sequence

    entries = ((3, 1), (2, 7), (1, 5), (0, 8))

    sequence = _gen_sequence(entries, min_size=2)

    assert tuple(value for _idx, value in sequence) == (7, 5)


def test_unblocking_assignment_maps_source_tiers_to_destinations():
    from container_reshuffle.solvers.bs_frg import _unblocking_assignment

    assignments, remaining_destinations = _unblocking_assignment(
        [[8, 5, 7, 1], [], [], []],
        source=0,
        max_height=4,
    )

    assert assignments == (1, 2, 2, 3)
    assert remaining_destinations == ()


def test_paper_pure_unblocking_assignment_uses_reduction_pop_order():
    from container_reshuffle.solvers.bs_frg import _unblocking_assignment

    assignments, remaining_destinations = _unblocking_assignment(
        [[8, 5, 7, 1], [], [], []],
        source=0,
        max_height=4,
        paper_pure=True,
    )

    assert assignments == (1, 2, 2, 3)
    assert remaining_destinations == ()


def test_reduction_choices_follow_unblocking_assignment_for_assigned_top():
    from container_reshuffle.solvers.bs_frg import (
        _FrgLayout,
        _reduction_choices,
        _to_key,
        _unblocking_assignment,
    )

    stacks = [[8, 5, 7, 1], [], [], []]
    assignments, destinations = _unblocking_assignment(stacks, source=0, max_height=4)
    layout = _FrgLayout(
        stacks=_to_key(stacks),
        active_reduce_stack=0,
        selected_counts=(1, 0, 0, 0),
        reduction_assignments=assignments,
        reduction_destinations=destinations,
    )

    choices = _reduction_choices(layout, max_height=4, step=0)

    assert choices
    assert choices[0].source == 0
    assert choices[0].target == 3


def test_select_destination_for_sequence_uses_mth_container_for_tie_break():
    from container_reshuffle.solvers.bs_frg import _select_destination_for_sequence

    target = _select_destination_for_sequence(
        [
            [9, 7, 5],
            [9, 8, 8],
            [7, 6],
        ],
        source=0,
        sequence=((2, 7), (1, 5), (0, 1)),
        destinations={1, 2},
        remaining_count=3,
        max_height=4,
    )

    assert target == 2


def test_unblocked_bad_count_counts_only_top_down_nonincreasing_bad_suffix():
    from container_reshuffle.solvers.bs_frg import _unblocked_bad_count

    assert _unblocked_bad_count([8, 5, 6, 1]) == 1
    assert _unblocked_bad_count([8, 5, 6, 7]) == 2


def test_destination_choices_score_empty_stack_as_max_group():
    from container_reshuffle.solvers.bs_frg import _destination_choices

    choices = _destination_choices([[8], [], [9]], box=9, source=2, max_height=4)

    assert choices[0][1] == 1
    assert choices[0][0][1] == 0.0


def test_destination_choices_do_not_prefer_equal_group_unordered_stack_as_xb1():
    from container_reshuffle.solvers.bs_frg import _destination_choices

    choices = _destination_choices([[5, 6], [4], [6]], box=6, source=2, max_height=3)

    assert choices[0][1] == 1
    assert choices[0][0][0] == 2.0
    equal_group_choice = next(choice for choice in choices if choice[1] == 0)
    assert equal_group_choice[0][0] == 3.0


def test_bg_choices_use_fractional_average_height_for_full_stack_guard():
    from container_reshuffle.solvers.bs_frg import _FrgLayout, _bg_choices, _to_key

    layout = _FrgLayout(
        stacks=_to_key(
            [
                [9, 8, 7],
                [1, 6],
                [9, 9, 9, 9],
                [8, 8, 8, 8],
                [7, 6, 5],
            ]
        ),
        active_reduce_stack=None,
        selected_counts=(0, 0, 0, 0, 0),
    )

    choices = _bg_choices(layout, max_height=4, step=0)

    assert all((choice.source, choice.target) != (1, 0) for choice in choices)


def test_paper_pure_bg_choices_do_not_force_bg_before_gg_moves():
    from container_reshuffle.solvers.bs_frg import _FrgLayout, _bg_choices, _to_key

    layout = _FrgLayout(
        stacks=_to_key(
            [
                [9, 8, 5, 6],
                [9, 7],
                [9, 8, 7, 7],
                [9, 9, 9, 9],
            ]
        ),
        active_reduce_stack=None,
        selected_counts=(0, 0, 0, 0),
    )

    default_choices = _bg_choices(layout, max_height=4, step=0)
    pure_choices = _bg_choices(layout, max_height=4, step=0, paper_pure=True)

    assert (default_choices[0].source, default_choices[0].target) == (0, 1)
    assert (pure_choices[0].source, pure_choices[0].target) == (2, 1)


def test_candidate_rollout_budget_does_not_clip_to_incumbent():
    from container_reshuffle.solvers.bs_frg import _candidate_rollout_budget

    assert _candidate_rollout_budget(max_depth=160, child_move_count=12, incumbent_move_count=30) == 148


def test_paper_pure_keeps_frontier_candidates_when_frg_rollout_is_infeasible(monkeypatch):
    import container_reshuffle.solvers.bs_frg as bs_frg
    from container_reshuffle.core import Instance, Move, ProblemVariant
    from container_reshuffle.solvers.base import SolverConfig

    instance = Instance(
        "frontier_infeasible_rollout",
        ProblemVariant.PREMARSHALLING_CPMP,
        3,
        3,
        4,
        [[1, 4], [2, 3], []],
    )
    child_layout = bs_frg._FrgLayout(
        stacks=bs_frg._to_key([[1], [2, 3, 4], []]),
        active_reduce_stack=None,
        selected_counts=(0, 0, 0),
    )
    child_move = Move(0, 4, 0, 1, 2, 3)

    def fake_generate_candidates(node, max_height, *, k, max_depth, deadline, paper_pure=False):
        if node.moves:
            return []
        return [bs_frg._Child(child_layout, (child_move,), 1, (0.0,))]

    def fake_frg_completion(layout, max_height, *, start_step, max_steps, deadline, paper_pure=False):
        return bs_frg._RolloutResult(
            False,
            layout,
            tuple(),
            bs_frg.future_disorder_count_stacks(bs_frg._to_lists(layout.stacks)),
            0,
            "FRG_FAKE_INFEASIBLE",
        )

    monkeypatch.setattr(bs_frg, "_generate_candidates", fake_generate_candidates)
    monkeypatch.setattr(bs_frg, "_frg_completion", fake_frg_completion)

    solution = bs_frg.BsFrgSolver(paper_pure=True).solve(
        instance,
        SolverConfig(max_depth=2, beam_width=2, candidate_limit_per_state=1, time_limit_sec=1.0),
    )

    assert solution.expanded_states == 2


def test_paper_pure_evaluation_score_preserves_candidate_score_hint():
    from container_reshuffle.solvers.bs_frg import (
        _Child,
        _FrgLayout,
        _RolloutResult,
        _paper_pure_evaluation_score,
        _to_key,
    )

    layout = _FrgLayout(_to_key([[3], [2], []]), None, (0, 0, 0))
    rollout = _RolloutResult(False, layout, tuple(), 4, 0, "FRG_FAKE_INFEASIBLE")
    better_hint = _Child(layout, tuple(), 1, (0.0, 2.0))
    worse_hint = _Child(layout, tuple(), 1, (2.0, 0.0))

    assert _paper_pure_evaluation_score(
        better_hint,
        rollout,
        estimated_total=10,
        child_rank=5,
    ) < _paper_pure_evaluation_score(
        worse_hint,
        rollout,
        estimated_total=10,
        child_rank=0,
    )


def test_paper_pure_frontier_deduplicates_identical_layouts(monkeypatch):
    import container_reshuffle.solvers.bs_frg as bs_frg
    from container_reshuffle.core import Instance, Move, ProblemVariant
    from container_reshuffle.solvers.base import SolverConfig

    instance = Instance(
        "frontier_duplicate_layouts",
        ProblemVariant.PREMARSHALLING_CPMP,
        3,
        3,
        4,
        [[1, 4], [2, 3], []],
    )
    duplicate_layout = bs_frg._FrgLayout(
        stacks=bs_frg._to_key([[1], [2, 3, 4], []]),
        active_reduce_stack=None,
        selected_counts=(0, 0, 0),
    )
    child_a = bs_frg._Child(duplicate_layout, (Move(0, 4, 0, 1, 2, 3),), 2, (0.0,))
    child_b = bs_frg._Child(duplicate_layout, (Move(0, 4, 0, 1, 2, 3),), 2, (1.0,))

    def fake_generate_candidates(node, max_height, *, k, max_depth, deadline, paper_pure=False):
        if node.moves:
            return []
        return [child_a, child_b]

    def fake_frg_completion(layout, max_height, *, start_step, max_steps, deadline, paper_pure=False):
        return bs_frg._RolloutResult(False, layout, tuple(), 2, 0, "FRG_FAKE_INFEASIBLE")

    monkeypatch.setattr(bs_frg, "_generate_candidates", fake_generate_candidates)
    monkeypatch.setattr(bs_frg, "_frg_completion", fake_frg_completion)

    solution = bs_frg.BsFrgSolver(paper_pure=True).solve(
        instance,
        SolverConfig(max_depth=2, beam_width=2, candidate_limit_per_state=1, time_limit_sec=1.0),
    )

    assert solution.expanded_states == 2


def test_paper_pure_candidate_generation_keeps_best_duplicate_layout(monkeypatch):
    import container_reshuffle.solvers.bs_frg as bs_frg
    from container_reshuffle.core import Move

    root_layout = bs_frg._FrgLayout(
        stacks=bs_frg._to_key([[1, 4], [2, 3], []]),
        active_reduce_stack=None,
        selected_counts=(0, 0, 0),
    )
    duplicate_layout = bs_frg._FrgLayout(
        stacks=bs_frg._to_key([[1], [2, 3, 4], []]),
        active_reduce_stack=None,
        selected_counts=(0, 0, 0),
    )
    longer_child = bs_frg._Child(
        duplicate_layout,
        (Move(0, 4, 0, 1, 2, 3), Move(1, 3, 1, 2, 2, 3)),
        2,
        (0.0,),
    )
    shorter_child = bs_frg._Child(
        duplicate_layout,
        (Move(0, 4, 0, 1, 2, 3),),
        2,
        (1.0,),
    )

    monkeypatch.setattr(bs_frg, "_single_move_children", lambda layout, max_height, step, k: [longer_child])
    monkeypatch.setattr(
        bs_frg,
        "_frg_iteration_child",
        lambda layout, max_height, step, *, paper_pure=False: [shorter_child],
    )
    monkeypatch.setattr(
        bs_frg,
        "_compound_reduction_child",
        lambda layout, max_height, step, source, max_moves, deadline, *, paper_pure=False: None,
    )

    candidates = bs_frg._generate_candidates(
        bs_frg._BeamNode(root_layout, tuple()),
        max_height=3,
        k=3,
        max_depth=10,
        deadline=float("inf"),
        paper_pure=True,
    )

    assert candidates == [shorter_child]


def test_paper_pure_reuses_frg_completion_for_duplicate_child_layouts(monkeypatch):
    import container_reshuffle.solvers.bs_frg as bs_frg
    from container_reshuffle.core import Instance, Move, ProblemVariant
    from container_reshuffle.solvers.base import SolverConfig

    instance = Instance(
        "pure_completion_cache",
        ProblemVariant.PREMARSHALLING_CPMP,
        4,
        4,
        5,
        [[1, 5], [2, 4], [3], []],
    )
    root = bs_frg._FrgLayout(bs_frg._to_key(instance.stacks), None, (0, 0, 0, 0))
    layout_a = bs_frg._FrgLayout(bs_frg._to_key([[1], [2, 4, 5], [3], []]), None, (0, 0, 0, 0))
    layout_b = bs_frg._FrgLayout(bs_frg._to_key([[1, 5], [2], [3, 4], []]), None, (0, 0, 0, 0))
    layout_c = bs_frg._FrgLayout(bs_frg._to_key([[1], [2], [3, 4, 5], []]), None, (0, 0, 0, 0))
    root_children = [
        bs_frg._Child(layout_a, (Move(0, 5, 0, 1, 2, 3),), 1, (0.0,)),
        bs_frg._Child(layout_b, (Move(0, 4, 1, 2, 2, 3),), 1, (1.0,)),
    ]
    shared_child = bs_frg._Child(layout_c, (Move(1, 5, 1, 2, 3, 4),), 1, (0.0,))
    call_counts: dict[bs_frg._FrgLayout, int] = {}

    def fake_generate_candidates(node, max_height, *, k, max_depth, deadline, paper_pure=False):
        if node.layout == root:
            return root_children
        if node.layout in {layout_a, layout_b}:
            return [shared_child]
        return []

    def fake_frg_completion(layout, max_height, *, start_step, max_steps, deadline, paper_pure=False):
        call_counts[layout] = call_counts.get(layout, 0) + 1
        return bs_frg._RolloutResult(False, layout, tuple(), 2, 0, "FRG_CYCLE_OR_DEAD_END")

    monkeypatch.setattr(bs_frg, "_initial_layout", lambda instance: root)
    monkeypatch.setattr(bs_frg, "_generate_candidates", fake_generate_candidates)
    monkeypatch.setattr(bs_frg, "_frg_completion", fake_frg_completion)

    bs_frg.BsFrgSolver(paper_pure=True).solve(
        instance,
        SolverConfig(max_depth=3, beam_width=2, candidate_limit_per_state=2, time_limit_sec=1.0),
    )

    assert call_counts[layout_c] == 1


def test_paper_pure_visited_pruning_skips_dominated_layout_revisit(monkeypatch):
    import container_reshuffle.solvers.bs_frg as bs_frg
    from container_reshuffle.core import Instance, Move, ProblemVariant
    from container_reshuffle.solvers.base import SolverConfig

    instance = Instance(
        "pure_visited_pruning",
        ProblemVariant.PREMARSHALLING_CPMP,
        3,
        3,
        4,
        [[1, 4], [2, 3], []],
    )
    root = bs_frg._FrgLayout(bs_frg._to_key(instance.stacks), None, (0, 0, 0))
    layout_a = bs_frg._FrgLayout(bs_frg._to_key([[1], [2, 3, 4], []]), None, (0, 0, 0))
    to_a = bs_frg._Child(layout_a, (Move(0, 4, 0, 1, 2, 3),), 1, (0.0,))
    back_to_root = bs_frg._Child(root, (Move(1, 4, 1, 0, 3, 2),), 1, (0.0,))

    def fake_generate_candidates(node, max_height, *, k, max_depth, deadline, paper_pure=False):
        if node.layout == root and not node.moves:
            return [to_a]
        if node.layout == layout_a:
            return [back_to_root]
        return []

    def fake_frg_completion(layout, max_height, *, start_step, max_steps, deadline, paper_pure=False):
        return bs_frg._RolloutResult(False, layout, tuple(), 2, 0, "FRG_CYCLE_OR_DEAD_END")

    monkeypatch.setattr(bs_frg, "_initial_layout", lambda instance: root)
    monkeypatch.setattr(bs_frg, "_generate_candidates", fake_generate_candidates)
    monkeypatch.setattr(bs_frg, "_frg_completion", fake_frg_completion)

    solution = bs_frg.BsFrgSolver(paper_pure=True).solve(
        instance,
        SolverConfig(max_depth=4, beam_width=1, candidate_limit_per_state=1, time_limit_sec=1.0),
    )

    assert solution.expanded_states == 2


def test_destination_rank_follows_strict_unordered_xb_preference():
    from container_reshuffle.solvers.bs_frg import _destination_rank

    # Algorithm 1 uses g(sd) < g(c) for unordered XB1 destinations. Equal
    # group values fall through to the final minimum-height XB class.
    rank = _destination_rank([5, 4, 5], box=5, max_height=4, empty_group=9)

    assert rank is not None
    assert rank[0] == 3.0


def test_rebase_moves_restarts_steps_without_changing_relocation():
    from container_reshuffle.core import Move
    from container_reshuffle.solvers.bs_frg import _rebase_moves

    moves = (Move(0, 7, 1, 2, 3, 4), Move(1, 5, 2, 0, 4, 1))

    rebased = _rebase_moves(moves, start_step=10)

    assert [move.step for move in rebased] == [10, 11]
    assert [(move.box, move.from_stack, move.to_stack, move.from_level, move.to_level) for move in rebased] == [
        (7, 1, 2, 3, 4),
        (5, 2, 0, 4, 1),
    ]


def test_frontier_dedup_key_preserves_frg_internal_state():
    from container_reshuffle.solvers.bs_frg import _FrgLayout, _frontier_dedup_key, _to_key

    stacks = _to_key([[3], [2], []])
    reducing = _FrgLayout(stacks, active_reduce_stack=0, selected_counts=(1, 0, 0))
    plain = _FrgLayout(stacks, active_reduce_stack=None, selected_counts=(1, 0, 0))

    assert _frontier_dedup_key(reducing) != _frontier_dedup_key(plain)


def test_visit_depth_dominance_keeps_shorter_duplicate_layout():
    from container_reshuffle.solvers.bs_frg import _FrgLayout, _is_visit_dominated, _to_key

    layout = _FrgLayout(_to_key([[3], [2], []]), None, (0, 0, 0))
    visited_depths = {layout: 5}

    assert _is_visit_dominated(visited_depths, layout, move_count=6) is True
    assert _is_visit_dominated(visited_depths, layout, move_count=4) is False


def test_bs_frg_solves_dlts_case_zero_with_frg_evaluated_candidates():
    from open_cpmp_benchmarks import load_dlts_dat
    from benchmark_classical_cpmp_baselines import row_from_solution
    from container_reshuffle.solvers.bs_frg import BsFrgSolver

    instance_path = ROOT / "external" / "DLTS" / "instances" / "5x5_cv1_test" / "0.dat"
    reference_path = ROOT / "external" / "DLTS" / "reference_solutions" / "5x5_cv1_test" / "0.out"
    if not instance_path.exists() or not reference_path.exists():
        import pytest

        pytest.skip("DLTS fixture checkout is unavailable")
    instance = load_dlts_dat(instance_path, max_height=7)

    solution = BsFrgSolver().solve(
        instance,
        SolverConfig(
            max_depth=160,
            beam_width=128,
            candidate_limit_per_state=3,
            time_limit_sec=1.0,
        ),
    )
    row = row_from_solution(
        instance,
        "5x5_cv1_test",
        "0",
        solution,
        reference_moves=None,
        allow_priority_groups=True,
    )

    assert solution.feasible is True
    assert row["goal_validation_ok"] is True
    assert row["final_n"] == 0
