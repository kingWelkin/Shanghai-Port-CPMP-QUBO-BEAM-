from __future__ import annotations

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
