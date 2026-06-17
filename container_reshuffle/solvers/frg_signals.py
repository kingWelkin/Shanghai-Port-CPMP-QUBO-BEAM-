"""FRG-style fill-and-reduce signals for CPMP layouts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrgLayoutSignals:
    stable_prefix_total: int
    unblocked_bad_containers: int
    fillable_ordered_stacks: int
    empty_slots: int
    max_height_ordered_stacks: int
    low_tier_risk: int


@dataclass(frozen=True)
class FrgTransitionSignals:
    fill_gain: int
    reduce_gain: int
    unblocked_bad_delta: int
    slack_delta: int
    stable_prefix_delta: int
    max_height_ordered_penalty: int
    low_tier_damage: int


def _stable_prefix_len(stack: list[int]) -> int:
    if not stack:
        return 0
    prefix = 1
    for idx in range(1, len(stack)):
        if stack[idx] <= stack[idx - 1]:
            prefix += 1
        else:
            break
    return prefix


def _is_ordered(stack: list[int]) -> bool:
    return _stable_prefix_len(stack) == len(stack)


def _unblocked_bad_count(stack: list[int]) -> int:
    stable = _stable_prefix_len(stack)
    if stable >= len(stack):
        return 0
    return len(stack) - stable


def _low_tier_risk(stacks: list[list[int]]) -> int:
    risk = 0
    for stack in stacks:
        stable = _stable_prefix_len(stack)
        if stable == len(stack) and len(stack) > 1:
            risk += 1
    return risk


def compute_frg_layout_signals(
    stacks: list[list[int]], max_height: int
) -> FrgLayoutSignals:
    stable_prefix_total = sum(_stable_prefix_len(stack) for stack in stacks)
    unblocked_bad_containers = sum(_unblocked_bad_count(stack) for stack in stacks)
    fillable_ordered_stacks = sum(
        1 for stack in stacks if len(stack) < max_height and _is_ordered(stack)
    )
    empty_slots = sum(max(0, max_height - len(stack)) for stack in stacks)
    max_height_ordered_stacks = sum(
        1 for stack in stacks if len(stack) >= max_height and _is_ordered(stack)
    )
    return FrgLayoutSignals(
        stable_prefix_total=stable_prefix_total,
        unblocked_bad_containers=unblocked_bad_containers,
        fillable_ordered_stacks=fillable_ordered_stacks,
        empty_slots=empty_slots,
        max_height_ordered_stacks=max_height_ordered_stacks,
        low_tier_risk=_low_tier_risk(stacks),
    )


def compute_frg_transition_signals(
    before: list[list[int]],
    after: list[list[int]],
    max_height: int,
) -> FrgTransitionSignals:
    b = compute_frg_layout_signals(before, max_height)
    a = compute_frg_layout_signals(after, max_height)
    return FrgTransitionSignals(
        fill_gain=max(0, a.stable_prefix_total - b.stable_prefix_total),
        reduce_gain=max(0, b.unblocked_bad_containers - a.unblocked_bad_containers),
        unblocked_bad_delta=b.unblocked_bad_containers - a.unblocked_bad_containers,
        slack_delta=a.empty_slots - b.empty_slots,
        stable_prefix_delta=a.stable_prefix_total - b.stable_prefix_total,
        max_height_ordered_penalty=max(
            0, a.max_height_ordered_stacks - b.max_height_ordered_stacks
        ),
        low_tier_damage=max(0, a.low_tier_risk - b.low_tier_risk),
    )


def frg_transition_score(
    signals: FrgTransitionSignals, n_value: int, depth: int
) -> float:
    return (
        float(n_value) * 1000.0
        + float(depth)
        - 26.0 * float(signals.fill_gain)
        - 18.0 * float(signals.reduce_gain)
        - 10.0 * float(signals.unblocked_bad_delta)
        - 6.0 * float(signals.stable_prefix_delta)
        - 3.0 * float(signals.slack_delta)
        + 18.0 * float(signals.max_height_ordered_penalty)
        + 20.0 * float(signals.low_tier_damage)
    )
