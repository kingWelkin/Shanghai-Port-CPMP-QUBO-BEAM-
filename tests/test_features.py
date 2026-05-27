from __future__ import annotations

from container_reshuffle.core import Instance, ProblemVariant, create_initial_state
from container_reshuffle.features import (
    bad_count,
    current_target_blockers,
    disorder_box_count,
    entropy_limit,
    entropy_value,
    inversion_count,
    well_prefix_len,
)


def test_well_prefix_len():
    assert well_prefix_len([10, 7, 8, 3]) == 2


def test_bad_count():
    inst = Instance("y", ProblemVariant.PREMARSHALLING_CPMP, 2, 4, 4, [[4, 2, 3, 1], []])
    state = create_initial_state(inst)
    assert bad_count(inst, state) == 2
    assert disorder_box_count(inst, state) == 2
    assert entropy_value(inst, state) == 2
    assert entropy_limit(inst, 1.3) == 2


def test_inversion_count():
    inst = Instance("x", ProblemVariant.PREMARSHALLING_CPMP, 1, 4, 4, [[4, 2, 3, 1]])
    state = create_initial_state(inst)
    assert inversion_count(inst, state) == 1


def test_current_target_blockers():
    inst = Instance("x", ProblemVariant.RETRIEVAL_CRP, 2, 3, 3, [[1, 3, 2], []])
    state = create_initial_state(inst)
    assert current_target_blockers(inst, state) == 2
