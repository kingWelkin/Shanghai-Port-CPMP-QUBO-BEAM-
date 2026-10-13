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
    available_stack_count: int
    avg_height_pressure: float
    max_height_ordered_risk: int
    reduction_pop_order_assignment_score: int
    unblocked_bad_source_count: int
    reduction_receiver_stack_count: int
    reduction_receiver_pressure: float


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
    return len(_unblocked_bad_sequence(stack))


def _unblocked_bad_sequence(stack: list[int]) -> list[int]:
    stable = _stable_prefix_len(stack)
    if stable >= len(stack):
        return []
    bad_suffix = stack[stable:]
    sequence = [bad_suffix[-1]]
    previous = bad_suffix[-1]
    for box in reversed(bad_suffix[:-1]):
        if box > previous:
            break
        sequence.append(box)
        previous = box
    return sequence


def _low_tier_risk(stacks: list[list[int]]) -> int:
    risk = 0
    for stack in stacks:
        stable = _stable_prefix_len(stack)
        if stable == len(stack) and len(stack) > 1:
            risk += 1
    return risk


def _reduction_pop_order_assignment_score(stacks: list[list[int]], max_height: int) -> int:
    score = 0
    for source_idx, source in enumerate(stacks):
        sequence = _unblocked_bad_sequence(source)
        if not sequence:
            continue
        destinations: list[tuple[int, int, int]] = []
        for dest_idx, dest in enumerate(stacks):
            if dest_idx == source_idx or len(dest) >= max_height or not _is_ordered(dest):
                continue
            top = dest[-1] if dest else 10**9
            destinations.append((top, len(dest), dest_idx))
        for box in sequence:
            feasible = [
                (top, height, idx)
                for top, height, idx in destinations
                if height < max_height and box <= top
            ]
            if not feasible:
                continue
            top, height, idx = min(feasible)
            destinations = [
                (box if dest_idx == idx else old_top, height + 1 if dest_idx == idx else old_height, dest_idx)
                for old_top, old_height, dest_idx in destinations
            ]
            score += 1
    return score


def _top_unblocked_bad_containers(stacks: list[list[int]]) -> list[tuple[int, int]]:
    top_bad: list[tuple[int, int]] = []
    for source_idx, source in enumerate(stacks):
        sequence = _unblocked_bad_sequence(source)
        if sequence:
            top_bad.append((source_idx, sequence[0]))
    return top_bad


def _reduction_receiver_stack_count(stacks: list[list[int]], max_height: int) -> int:
    receivers: set[int] = set()
    for source_idx, box in _top_unblocked_bad_containers(stacks):
        for dest_idx, dest in enumerate(stacks):
            if dest_idx == source_idx or len(dest) >= max_height or not _is_ordered(dest):
                continue
            top = dest[-1] if dest else 10**9
            if box <= top:
                receivers.add(dest_idx)
    return len(receivers)


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
    available_stack_count = sum(1 for stack in stacks if len(stack) < max_height)
    capacity = max(1, len(stacks) * max(1, max_height))
    avg_height_pressure = sum(len(stack) for stack in stacks) / float(capacity)
    top_unblocked_bad = _top_unblocked_bad_containers(stacks)
    receiver_stack_count = _reduction_receiver_stack_count(stacks, max_height)
    bad_source_count = len(top_unblocked_bad)
    receiver_pressure = (
        float(bad_source_count) / float(max(1, bad_source_count + receiver_stack_count))
        if bad_source_count > 0
        else 0.0
    )
    return FrgLayoutSignals(
        stable_prefix_total=stable_prefix_total,
        unblocked_bad_containers=unblocked_bad_containers,
        fillable_ordered_stacks=fillable_ordered_stacks,
        empty_slots=empty_slots,
        max_height_ordered_stacks=max_height_ordered_stacks,
        low_tier_risk=_low_tier_risk(stacks),
        available_stack_count=available_stack_count,
        avg_height_pressure=avg_height_pressure,
        max_height_ordered_risk=max_height_ordered_stacks,
        reduction_pop_order_assignment_score=_reduction_pop_order_assignment_score(
            stacks, max_height
        ),
        unblocked_bad_source_count=bad_source_count,
        reduction_receiver_stack_count=receiver_stack_count,
        reduction_receiver_pressure=receiver_pressure,
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
