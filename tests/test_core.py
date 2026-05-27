from __future__ import annotations

import pytest

from container_reshuffle.core import (
    Instance,
    Move,
    ProblemVariant,
    Solution,
    apply_move,
    create_initial_state,
    generate_legal_moves,
    is_goal,
    validate_instance,
    validate_solution,
)


def test_validate_instance_rejects_duplicate_and_missing():
    inst = Instance("bad", ProblemVariant.PREMARSHALLING_CPMP, 2, 2, 3, [[1, 1], [3]])
    with pytest.raises(ValueError):
        validate_instance(inst)


def test_generate_legal_moves_only_moves_top():
    inst = Instance("x", ProblemVariant.PREMARSHALLING_CPMP, 2, 3, 3, [[3, 2, 1], []])
    state = create_initial_state(inst)
    moves = generate_legal_moves(inst, state)
    assert len(moves) == 1
    assert moves[0].box == 1
    assert moves[0].from_level == 3


def test_apply_move_updates_stacks():
    inst = Instance("x", ProblemVariant.PREMARSHALLING_CPMP, 2, 2, 2, [[1, 2], []])
    state = create_initial_state(inst)
    new_state = apply_move(inst, state, Move(0, 2, 0, 1))
    assert new_state.stacks == [[1], [2]]
    assert new_state.move_count == 1


def test_crp_normalize_retrieves_top_targets():
    inst = Instance("x", ProblemVariant.RETRIEVAL_CRP, 2, 2, 2, [[2, 1], []])
    state = create_initial_state(inst)
    assert state.next_retrieval_id == 3
    assert state.stacks == [[], []]


def test_cpmp_goal_and_solution_validation():
    inst = Instance("x", ProblemVariant.PREMARSHALLING_CPMP, 2, 2, 2, [[2, 1], []])
    state = create_initial_state(inst)
    assert is_goal(inst, state)
    sol = Solution("test", True, False, 0.0, 0.0, [], "OK")
    ok, msg, _ = validate_solution(inst, sol)
    assert ok, msg
