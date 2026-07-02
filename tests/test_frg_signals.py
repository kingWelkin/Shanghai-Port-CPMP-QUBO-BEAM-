from __future__ import annotations

import pytest

from container_reshuffle.solvers.frg_signals import (
    FrgTransitionSignals,
    compute_frg_layout_signals,
    compute_frg_transition_signals,
    frg_transition_score,
)


def test_frg_layout_signals_detect_unblocked_bad_and_fillable_ordered_stack():
    signals = compute_frg_layout_signals(
        stacks=[[6, 5], [4, 7, 2], [3], []],
        max_height=3,
    )

    assert signals.stable_prefix_total == 4
    assert signals.unblocked_bad_containers >= 1
    assert signals.fillable_ordered_stacks >= 2
    assert signals.empty_slots == 6


def test_frg_unblocked_bad_uses_top_down_non_increasing_sequence():
    blocked_suffix = compute_frg_layout_signals(
        stacks=[[9, 1, 8, 7, 6]],
        max_height=5,
    )
    unblocked_suffix = compute_frg_layout_signals(
        stacks=[[9, 1, 6, 7, 8]],
        max_height=5,
    )

    assert blocked_suffix.unblocked_bad_containers == 1
    assert unblocked_suffix.unblocked_bad_containers == 3


def test_frg_layout_signals_expose_scarcity_and_reduction_assignment_pressure():
    signals = compute_frg_layout_signals(
        stacks=[[9, 1, 6, 7, 8], [10, 5], [], [4]],
        max_height=5,
    )
    full_ordered = compute_frg_layout_signals(
        stacks=[[9, 8, 7, 6, 5], [10, 5], [], [4]],
        max_height=5,
    )

    assert signals.available_stack_count == 3
    assert signals.avg_height_pressure == pytest.approx(0.4)
    assert signals.reduction_pop_order_assignment_score == 3
    assert full_ordered.max_height_ordered_risk == 1


def test_frg_layout_signals_count_safe_receivers_for_top_unblocked_bad_containers():
    scarce = compute_frg_layout_signals(
        stacks=[[9, 1, 8], [7, 6, 5], [10, 9, 8], [10], []],
        max_height=3,
    )
    roomy = compute_frg_layout_signals(
        stacks=[[9, 1, 8], [9], [10], [], []],
        max_height=3,
    )

    assert scarce.avg_height_pressure == pytest.approx(10 / 15)
    assert scarce.reduction_receiver_stack_count == 2
    assert scarce.reduction_receiver_pressure > 0.0
    assert roomy.reduction_receiver_stack_count == 4
    assert roomy.reduction_receiver_pressure < scarce.reduction_receiver_pressure


def test_frg_transition_prefers_fill_over_destabilizing_reduce():
    fill = compute_frg_transition_signals(
        before=[[6, 5], [4, 7, 2], [3], []],
        after=[[6, 5, 2], [4, 7], [3], []],
        max_height=3,
    )
    reduce = compute_frg_transition_signals(
        before=[[6, 5], [4, 7, 2], [3], []],
        after=[[6], [4, 7, 2], [3, 5], []],
        max_height=3,
    )

    assert fill.fill_gain > reduce.fill_gain
    assert frg_transition_score(fill, n_value=3, depth=10) < frg_transition_score(
        reduce, n_value=3, depth=10
    )


def test_frg_transition_signal_defaults_are_value_objects():
    signals = FrgTransitionSignals(
        fill_gain=1,
        reduce_gain=0,
        unblocked_bad_delta=1,
        slack_delta=0,
        stable_prefix_delta=1,
        max_height_ordered_penalty=0,
        low_tier_damage=0,
    )

    assert frg_transition_score(signals, n_value=2, depth=5) < 2100.0
