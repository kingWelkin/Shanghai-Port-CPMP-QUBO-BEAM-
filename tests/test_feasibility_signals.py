from __future__ import annotations

from container_reshuffle.solvers.feasibility_signals import (
    DltsSignalAdapter,
    DltsStateFeatures,
    compute_feasibility_signals,
    feasibility_escape_score,
)


def test_feasibility_signals_capture_stability_slack_and_dead_end_risk():
    signals = compute_feasibility_signals(
        stacks=[[5, 4, 1], [3, 6], [2], []],
        max_height=3,
    )

    assert signals.stable_prefix_total == 5
    assert signals.empty_slots == 6
    assert signals.blocked_fixed_candidates == 1
    assert signals.temporary_slack_moves > 0
    assert signals.low_tier_risk > 0
    assert 0.0 <= signals.dead_end_risk <= 1.0


def test_feasibility_escape_score_prefers_slack_without_low_tier_damage():
    risky = compute_feasibility_signals(
        stacks=[[5, 4, 3], [2, 6], []],
        max_height=3,
    )
    flexible = compute_feasibility_signals(
        stacks=[[5, 4], [3, 1], [2]],
        max_height=3,
    )

    assert feasibility_escape_score(flexible, n_value=2, depth=10) < feasibility_escape_score(
        risky,
        n_value=2,
        depth=10,
    )


def test_dlts_signal_adapter_policy_and_value_are_deterministic():
    signals = compute_feasibility_signals(
        stacks=[[5, 4], [3, 1], [2]],
        max_height=3,
    )
    features = DltsStateFeatures(
        n_value=2,
        depth=7,
        parent_rank=3,
        selected=False,
        signals=signals,
    )
    adapter = DltsSignalAdapter()

    assert adapter.policy_score(features) == adapter.policy_score(features)
    assert adapter.value_estimate(features) >= 0.0
