"""Feasibility and DLTS-ready signals for CPMP layouts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeasibilitySignals:
    fixed_prefix_total: int
    stable_prefix_total: int
    target_gap: int
    empty_slots: int
    blocked_fixed_candidates: int
    temporary_slack_moves: int
    low_tier_risk: int
    dead_end_risk: float


@dataclass(frozen=True)
class DltsStateFeatures:
    n_value: int
    depth: int
    parent_rank: int
    selected: bool
    signals: FeasibilitySignals


def _stable_prefix_len(stack: list[int]) -> int:
    if not stack:
        return 0
    prefix = 1
    for idx in range(1, len(stack)):
        if stack[idx - 1] > stack[idx]:
            prefix += 1
        else:
            break
    return prefix


def _future_disorder_count(stacks: list[list[int]]) -> int:
    total = 0
    for stack in stacks:
        for idx in range(1, len(stack)):
            if stack[idx] > stack[idx - 1]:
                total += len(stack) - idx
                break
    return total


def _move_top(stacks: list[list[int]], source: int, target: int) -> list[list[int]]:
    copied = [list(stack) for stack in stacks]
    copied[target].append(copied[source].pop())
    return copied


def compute_feasibility_signals(
    stacks: list[list[int]],
    max_height: int,
    target_gap: int = 0,
) -> FeasibilitySignals:
    stable_lengths = [_stable_prefix_len(stack) for stack in stacks]
    stable_prefix_total = sum(stable_lengths)
    empty_slots = sum(max(0, max_height - len(stack)) for stack in stacks)
    current_n = _future_disorder_count(stacks)
    blocked_fixed_candidates = 0
    temporary_slack_moves = 0
    low_tier_risk = 0

    for source, stack in enumerate(stacks):
        if not stack:
            continue
        if stable_lengths[source] < len(stack):
            blocked_fixed_candidates += len(stack) - stable_lengths[source]
        source_top_is_stable = stable_lengths[source] == len(stack) and len(stack) > 1
        for target, dst in enumerate(stacks):
            if source == target or len(dst) >= max_height:
                continue
            moved = _move_top(stacks, source, target)
            if _future_disorder_count(moved) <= current_n:
                temporary_slack_moves += 1
            if source_top_is_stable:
                low_tier_risk += 1

    capacity = max(1, len(stacks) * max_height)
    n_pressure = min(1.0, float(current_n) / float(capacity))
    slack_pressure = 1.0 - min(1.0, float(empty_slots) / float(max(1, len(stacks))))
    escape_pressure = 1.0 if temporary_slack_moves == 0 and current_n > 0 else 0.0
    dead_end_risk = max(0.0, min(1.0, 0.45 * n_pressure + 0.35 * slack_pressure + 0.20 * escape_pressure))

    return FeasibilitySignals(
        fixed_prefix_total=stable_prefix_total,
        stable_prefix_total=stable_prefix_total,
        target_gap=max(0, int(target_gap)),
        empty_slots=empty_slots,
        blocked_fixed_candidates=blocked_fixed_candidates,
        temporary_slack_moves=temporary_slack_moves,
        low_tier_risk=low_tier_risk,
        dead_end_risk=dead_end_risk,
    )


def feasibility_escape_score(signals: FeasibilitySignals, n_value: int, depth: int) -> float:
    return (
        float(n_value) * 1000.0
        + float(depth)
        + signals.dead_end_risk * 120.0
        + float(signals.low_tier_risk) * 5.0
        + float(signals.target_gap) * 2.0
        - float(signals.temporary_slack_moves) * 30.0
        - float(signals.stable_prefix_total) * 3.0
        - float(signals.blocked_fixed_candidates) * 2.0
    )


class DltsSignalAdapter:
    def policy_score(self, features: DltsStateFeatures) -> float:
        return feasibility_escape_score(features.signals, features.n_value, features.depth) + float(features.parent_rank)

    def value_estimate(self, features: DltsStateFeatures) -> float:
        signals = features.signals
        value = (
            float(features.n_value)
            + signals.dead_end_risk * 3.0
            + float(signals.target_gap) * 0.10
            - float(signals.temporary_slack_moves) * 0.25
            - float(signals.stable_prefix_total) * 0.05
        )
        return max(0.0, value)
