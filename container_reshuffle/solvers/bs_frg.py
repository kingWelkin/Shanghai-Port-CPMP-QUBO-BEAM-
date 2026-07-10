"""Standalone BS-FRG solver for the container pre-marshalling problem.

This module follows the 2023 fill-and-reduce greedy paper closely enough to make
the algorithmic boundary explicit:

* FRG alternates BG filling moves and sustained reduction of one stack ``s_r``.
* Reduction stacks are rotated by selection frequency to avoid the cycles noted
  in the paper.
* BS-FRG generates single-move candidates and compound reduction candidates, then
  evaluates each candidate by running FRG to completion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from math import inf

from ..core import Instance, Move, ProblemVariant, Solution, future_disorder_count_stacks
from .base import Solver, SolverConfig, failure_solution

_INF_SCORE = 10**9


@dataclass(frozen=True)
class _FrgLayout:
    stacks: tuple[tuple[int, ...], ...]
    active_reduce_stack: int | None
    selected_counts: tuple[int, ...]
    reduction_assignments: tuple[int | None, ...] = ()
    reduction_destinations: tuple[int, ...] = ()


@dataclass(frozen=True)
class _BeamNode:
    layout: _FrgLayout
    moves: tuple[Move, ...]


@dataclass(frozen=True)
class _MoveChoice:
    score: tuple[float, ...]
    source: int
    target: int
    keep_reduction: bool = False
    selected_counts: tuple[int, ...] | None = None
    reduction_assignments: tuple[int | None, ...] | None = None
    reduction_destinations: tuple[int, ...] | None = None


@dataclass(frozen=True)
class _Child:
    layout: _FrgLayout
    moves: tuple[Move, ...]
    generated: int
    score_hint: tuple[float, ...]


@dataclass(frozen=True)
class _RolloutResult:
    feasible: bool
    layout: _FrgLayout
    moves: tuple[Move, ...]
    final_n: int
    generated: int
    status: str


@dataclass(frozen=True)
class _EvaluatedChild:
    score: tuple[float, ...]
    node: _BeamNode
    rollout: _RolloutResult


def _to_key(stacks: list[list[int]] | tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(box) for box in stack) for stack in stacks)


def _to_lists(stacks: tuple[tuple[int, ...], ...]) -> list[list[int]]:
    return [list(stack) for stack in stacks]


def _initial_layout(instance: Instance) -> _FrgLayout:
    return _FrgLayout(
        stacks=_to_key(instance.stacks),
        active_reduce_stack=None,
        selected_counts=tuple(0 for _ in range(instance.stack_count)),
    )


@lru_cache(maxsize=262144)
def _is_ordered_stack_tuple(stack: tuple[int, ...]) -> bool:
    return all(stack[idx] >= stack[idx + 1] for idx in range(len(stack) - 1))


def _is_ordered_stack(stack: list[int] | tuple[int, ...]) -> bool:
    return _is_ordered_stack_tuple(stack if isinstance(stack, tuple) else tuple(stack))


def _is_goal(stacks: list[list[int]] | tuple[tuple[int, ...], ...]) -> bool:
    return all(_is_ordered_stack(stack) for stack in stacks)


@lru_cache(maxsize=262144)
def _stable_prefix_len_tuple(stack: tuple[int, ...]) -> int:
    if not stack:
        return 0
    length = 1
    for idx in range(1, len(stack)):
        if stack[idx - 1] >= stack[idx]:
            length += 1
        else:
            break
    return length


def _stable_prefix_len(stack: list[int] | tuple[int, ...]) -> int:
    return _stable_prefix_len_tuple(stack if isinstance(stack, tuple) else tuple(stack))


def _bad_suffix_len(stack: list[int] | tuple[int, ...]) -> int:
    return len(stack) - _stable_prefix_len(stack)


def _dirty_stack_count(stacks: list[list[int]] | tuple[tuple[int, ...], ...]) -> int:
    return sum(1 for stack in stacks if not _is_ordered_stack(stack))


def _stable_prefix_total(stacks: list[list[int]] | tuple[tuple[int, ...], ...]) -> int:
    return sum(_stable_prefix_len(stack) for stack in stacks)


def _layout_max_group(stacks: list[list[int]] | tuple[tuple[int, ...], ...]) -> int:
    values = [int(box) for stack in stacks for box in stack]
    return max(values) if values else 0


def _stack_group(stack: list[int] | tuple[int, ...], empty_group: int = _INF_SCORE) -> int:
    return int(stack[-1]) if stack else int(empty_group)


def _empty_slots(stack: list[int] | tuple[int, ...], max_height: int) -> int:
    return max_height - len(stack)


def _average_group(stack: list[int] | tuple[int, ...]) -> float:
    return sum(stack) / len(stack) if stack else -inf


def _can_accept_well_placed(stack: list[int], box: int, max_height: int, empty_group: int) -> bool:
    return len(stack) < max_height and _is_ordered_stack(stack) and _stack_group(stack, empty_group) >= box


def _make_move(stacks: list[list[int]], source: int, target: int, step: int) -> Move:
    src = stacks[source]
    dst = stacks[target]
    return Move(
        step=step,
        box=int(src[-1]),
        from_stack=source,
        to_stack=target,
        from_level=len(src),
        to_level=len(dst) + 1,
    )


def _apply_move(
    layout: _FrgLayout,
    choice: _MoveChoice,
    step: int,
    *,
    reset_reduction: bool,
) -> tuple[_FrgLayout, Move]:
    stacks = layout.stacks
    src_stack = stacks[choice.source]
    dst_stack = stacks[choice.target]
    box = int(src_stack[-1])
    move = Move(
        step=step,
        box=box,
        from_stack=choice.source,
        to_stack=choice.target,
        from_level=len(src_stack),
        to_level=len(dst_stack) + 1,
    )
    next_stacks = list(stacks)
    next_stacks[choice.source] = src_stack[:-1]
    next_stacks[choice.target] = dst_stack + (box,)
    selected_counts = choice.selected_counts if choice.selected_counts is not None else layout.selected_counts
    if reset_reduction or not choice.keep_reduction:
        next_active = None
        reduction_assignments: tuple[int | None, ...] = ()
        reduction_destinations: tuple[int, ...] = ()
    else:
        next_active = choice.source
        reduction_assignments = (
            choice.reduction_assignments
            if choice.reduction_assignments is not None
            else layout.reduction_assignments
        )
        reduction_destinations = (
            choice.reduction_destinations
            if choice.reduction_destinations is not None
            else layout.reduction_destinations
        )
    return (
        _FrgLayout(
            stacks=tuple(next_stacks),
            active_reduce_stack=next_active,
            selected_counts=tuple(selected_counts),
            reduction_assignments=reduction_assignments,
            reduction_destinations=reduction_destinations,
        ),
        move,
    )


def _destination_rank(stack: list[int], box: int, max_height: int, empty_group: int) -> tuple[float, ...] | None:
    if len(stack) >= max_height:
        return None
    g_stack = _stack_group(stack, empty_group)
    if _is_ordered_stack(stack) and g_stack >= box:
        return (0.0, float(g_stack - box), float(len(stack)))
    if not _is_ordered_stack(stack) and g_stack < box:
        return (1.0, float(box - g_stack), float(len(stack)))
    if _is_ordered_stack(stack):
        return (2.0, float(len(stack)), float(g_stack))
    return (3.0, float(len(stack)), float(g_stack))


def _destination_choices(
    stacks: list[list[int]],
    box: int,
    source: int,
    max_height: int,
    *,
    allowed: set[int] | None = None,
) -> list[tuple[tuple[float, ...], int]]:
    choices: list[tuple[tuple[float, ...], int]] = []
    empty_group = _layout_max_group(stacks)
    for target, stack in enumerate(stacks):
        if source == target:
            continue
        if allowed is not None and target not in allowed:
            continue
        rank = _destination_rank(stack, box, max_height, empty_group)
        if rank is None:
            continue
        choices.append((rank + (float(target),), target))
    choices.sort(key=lambda item: item[0])
    return choices


def _gen_sequence(entries: tuple[tuple[int, int], ...], min_size: int) -> tuple[tuple[int, int], ...]:
    """Return a lexicographically large top-down non-increasing subsequence."""

    if not entries:
        return ()
    required = max(1, int(min_size))
    lds = [1 for _ in entries]
    for idx in range(len(entries) - 1, -1, -1):
        value = entries[idx][1]
        for next_idx in range(idx + 1, len(entries)):
            if entries[next_idx][1] <= value:
                lds[idx] = max(lds[idx], 1 + lds[next_idx])
    if max(lds) < required:
        return ()

    sequence: list[tuple[int, int]] = []
    last_position = -1
    for position, entry in sorted(enumerate(entries), key=lambda item: (-item[1][1], item[0])):
        if position <= last_position:
            continue
        if sequence and entry[1] > sequence[-1][1]:
            continue
        if len(sequence) + lds[position] < required:
            continue
        sequence.append(entry)
        last_position = position
    return tuple(sequence)


def _select_destination_for_sequence(
    stacks: list[list[int]],
    source: int,
    sequence: tuple[tuple[int, int], ...],
    destinations: set[int],
    remaining_count: int,
    max_height: int,
) -> int | None:
    total_empty = sum(_empty_slots(stacks[idx], max_height) for idx in destinations)
    empty_group = _layout_max_group(stacks)
    candidates: list[tuple[tuple[float, ...], int]] = []
    for target in destinations:
        empty = _empty_slots(stacks[target], max_height)
        if empty <= 0:
            continue
        move_count = min(empty, len(sequence))
        if remaining_count - move_count > total_empty - empty:
            continue
        rank = _destination_rank(stacks[target], sequence[move_count - 1][1], max_height, empty_group)
        if rank is None:
            continue
        exact_rank = 0.0 if empty == len(sequence) else 1.0
        candidates.append(((exact_rank, *rank, float(source), float(target)), target))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _unblocking_assignment(
    stacks: list[list[int]],
    source: int,
    max_height: int,
    *,
    paper_pure: bool = False,
) -> tuple[tuple[int | None, ...], tuple[int, ...]]:
    """Assign source-stack tiers to destinations before reducing a dense stack."""

    source_stack = stacks[source]
    assignments: list[int | None] = [None for _ in source_stack]
    destinations = {idx for idx, stack in enumerate(stacks) if idx != source and len(stack) < max_height}
    # Stacks are stored bottom-to-top, while reduction moves pop top-to-bottom.
    # Generate assignments in pop order so the sequence constraints match the
    # order in which containers are actually moved out of the reduced stack.
    ordered_entries = reversed(list(enumerate(source_stack)))
    remaining = tuple((idx, int(value)) for idx, value in ordered_entries)

    while remaining and destinations:
        total_empty = sum(_empty_slots(stacks[idx], max_height) for idx in destinations)
        min_empty = min(_empty_slots(stacks[idx], max_height) for idx in destinations)
        min_size = max(1, len(remaining) - total_empty + min_empty)
        sequence = _gen_sequence(remaining, min_size)
        if not sequence:
            break
        target = _select_destination_for_sequence(
            stacks,
            source,
            sequence,
            destinations,
            len(remaining),
            max_height,
        )
        if target is None:
            break
        move_count = min(_empty_slots(stacks[target], max_height), len(sequence))
        assigned = set(sequence[:move_count])
        for original_index, _value in sequence[:move_count]:
            assignments[original_index] = target
        remaining = tuple(entry for entry in remaining if entry not in assigned)
        destinations.remove(target)
    return tuple(assignments), tuple(sorted(destinations))


def _available_stack_count(stacks: list[list[int]], max_height: int) -> int:
    return sum(1 for stack in stacks if len(stack) < max_height)


def _bg_choices(layout: _FrgLayout, max_height: int, step: int, *, paper_pure: bool = False) -> list[_MoveChoice]:
    stacks = _to_lists(layout.stacks)
    total_containers = sum(len(stack) for stack in stacks)
    avg_height = total_containers / len(stacks) if stacks else 0.0
    available_count = _available_stack_count(stacks, max_height)
    prevent_full_ordered = available_count < avg_height
    include_gg = available_count < (avg_height / 2.0)
    current_n = future_disorder_count_stacks(stacks)
    choices: list[_MoveChoice] = []
    empty_group = _layout_max_group(stacks)

    for source, src in enumerate(stacks):
        if not src:
            continue
        box = int(src[-1])
        source_is_unordered = not _is_ordered_stack(src)
        source_is_full_ordered = include_gg and _is_ordered_stack(src) and len(src) == max_height
        if not source_is_unordered and not source_is_full_ordered:
            continue
        for target, dst in enumerate(stacks):
            if source == target or not _can_accept_well_placed(dst, box, max_height, empty_group):
                continue
            if prevent_full_ordered and len(dst) + 1 >= max_height and source_is_unordered:
                continue
            next_layout, _ = _apply_move(
                layout,
                _MoveChoice((0.0,), source, target),
                step,
                reset_reduction=True,
            )
            next_stacks = _to_lists(next_layout.stacks)
            child_n = future_disorder_count_stacks(next_stacks)
            move_type = 0.0 if source_is_unordered else 1.0
            if paper_pure:
                score = (
                    float(_stack_group(dst, empty_group) - box),
                    float(source),
                    float(target),
                )
            else:
                score = (
                    move_type,
                    float(_stack_group(dst, empty_group) - box),
                    float(child_n),
                    float(max(0, child_n - current_n)),
                    -float(_stable_prefix_total(next_stacks)),
                    float(source),
                    float(target),
                )
            choices.append(_MoveChoice(score, source, target))
    choices.sort(key=lambda choice: choice.score)
    return choices


def _select_reduction_stack(stacks: list[list[int]], selected_counts: tuple[int, ...]) -> int | None:
    candidates = [idx for idx, stack in enumerate(stacks) if stack]
    if not candidates:
        return None
    min_count = min(selected_counts[idx] for idx in candidates)
    candidates = [idx for idx in candidates if selected_counts[idx] == min_count]
    min_height = min(len(stacks[idx]) for idx in candidates)
    candidates = [idx for idx in candidates if len(stacks[idx]) == min_height]
    unordered = [idx for idx in candidates if not _is_ordered_stack(stacks[idx])]
    if unordered:
        return max(unordered, key=lambda idx: (_average_group(stacks[idx]), -idx))
    return min(candidates, key=lambda idx: (_average_group(stacks[idx]), idx))


def _unblocked_bad_count(stack: list[int] | tuple[int, ...]) -> int:
    bad_start = _stable_prefix_len(stack)
    if bad_start >= len(stack):
        return 0
    count = 1
    previous = int(stack[-1])
    for value in reversed(stack[bad_start:-1]):
        current = int(value)
        if previous < current:
            break
        count += 1
        previous = current
    return count


def _stopping_reduction_criterion(layout: _FrgLayout, max_height: int, *, r: int = 1) -> bool:
    sr = layout.active_reduce_stack
    if sr is None:
        return False
    stacks = _to_lists(layout.stacks)
    if not stacks[sr]:
        return True

    unordered = [idx for idx, stack in enumerate(stacks) if idx != sr and not _is_ordered_stack(stack)]
    if not unordered:
        return True

    g_sr = _stack_group(stacks[sr])
    empty_sr = _empty_slots(stacks[sr], max_height)
    all_unblocked = sum(_unblocked_bad_count(stacks[idx]) for idx in unordered)
    max_unordered_group = max(_stack_group(stacks[idx]) for idx in unordered)
    if all_unblocked <= empty_sr and g_sr >= max_unordered_group:
        return True

    fillable = sum(
        _unblocked_bad_count(stacks[idx])
        for idx in unordered
        if _stack_group(stacks[idx]) <= g_sr
    )
    return fillable >= max(0, empty_sr - r)


def _start_reduction_layout(
    layout: _FrgLayout,
    source: int,
    max_height: int,
    *,
    paper_pure: bool = False,
) -> _FrgLayout:
    stacks = _to_lists(layout.stacks)
    counts = list(layout.selected_counts)
    counts[source] += 1
    nonfull_destinations = [
        idx for idx, stack in enumerate(stacks)
        if idx != source and len(stack) < max_height
    ]
    if len(nonfull_destinations) < len(stacks[source]):
        assignments, destinations = _unblocking_assignment(stacks, source, max_height, paper_pure=paper_pure)
    else:
        assignments = ()
        destinations = ()
    return _FrgLayout(
        stacks=layout.stacks,
        active_reduce_stack=source,
        selected_counts=tuple(counts),
        reduction_assignments=assignments,
        reduction_destinations=destinations,
    )


def _reduction_choices(
    layout: _FrgLayout,
    max_height: int,
    step: int,
    *,
    force_source: int | None = None,
    paper_pure: bool = False,
) -> list[_MoveChoice]:
    stacks = _to_lists(layout.stacks)
    source = force_source if force_source is not None else layout.active_reduce_stack
    current_layout = layout
    if source is None:
        source = _select_reduction_stack(stacks, layout.selected_counts)
        if source is None:
            return []
        current_layout = _start_reduction_layout(layout, source, max_height, paper_pure=paper_pure)
    elif layout.active_reduce_stack != source:
        current_layout = _start_reduction_layout(layout, source, max_height, paper_pure=paper_pure)
    if source < 0 or source >= len(stacks) or not stacks[source]:
        return []

    box = int(stacks[source][-1])
    current_n = future_disorder_count_stacks(stacks)
    assigned_target: int | None = None
    allowed: set[int] | None = None
    top_index = len(stacks[source]) - 1
    if current_layout.reduction_assignments:
        if 0 <= top_index < len(current_layout.reduction_assignments):
            assigned_target = current_layout.reduction_assignments[top_index]
        if assigned_target is not None:
            allowed = {assigned_target}
        else:
            allowed = set(current_layout.reduction_destinations)
    choices: list[_MoveChoice] = []
    destinations = _destination_choices(stacks, box, source, max_height, allowed=allowed)
    if not destinations and assigned_target is not None:
        fallback_allowed = set(current_layout.reduction_destinations) or None
        destinations = _destination_choices(stacks, box, source, max_height, allowed=fallback_allowed)
    if not destinations:
        destinations = _destination_choices(stacks, box, source, max_height)
    for dest_rank, target in destinations:
        next_layout, _ = _apply_move(
            current_layout,
            _MoveChoice(
                dest_rank,
                source,
                target,
                keep_reduction=True,
                selected_counts=current_layout.selected_counts,
                reduction_assignments=current_layout.reduction_assignments,
                reduction_destinations=current_layout.reduction_destinations,
            ),
            step,
            reset_reduction=False,
        )
        next_stacks = _to_lists(next_layout.stacks)
        child_n = future_disorder_count_stacks(next_stacks)
        score = (
            *dest_rank,
            float(child_n),
            float(max(0, child_n - current_n)),
            float(_dirty_stack_count(next_stacks)),
            -float(_stable_prefix_total(next_stacks)),
        )
        choices.append(
            _MoveChoice(
                score,
                source,
                target,
                keep_reduction=True,
                selected_counts=current_layout.selected_counts,
                reduction_assignments=current_layout.reduction_assignments,
                reduction_destinations=current_layout.reduction_destinations,
            )
        )
    choices.sort(key=lambda choice: choice.score)
    return choices


def _frg_step_children(
    layout: _FrgLayout,
    max_height: int,
    step: int,
    *,
    limit: int | None = None,
    paper_pure: bool = False,
) -> list[_Child]:
    if layout.active_reduce_stack is None:
        choices = _bg_choices(layout, max_height, step, paper_pure=paper_pure)
        if choices:
            children: list[_Child] = []
            for choice in choices[:limit]:
                next_layout, move = _apply_move(layout, choice, step, reset_reduction=True)
                children.append(_Child(next_layout, (move,), len(choices), choice.score))
            return children
    choices = _reduction_choices(layout, max_height, step, paper_pure=paper_pure)
    children = []
    for choice in choices[:limit]:
        next_layout, move = _apply_move(layout, choice, step, reset_reduction=False)
        if _stopping_reduction_criterion(next_layout, max_height):
            next_layout = _FrgLayout(next_layout.stacks, None, next_layout.selected_counts)
        children.append(_Child(next_layout, (move,), len(choices), choice.score))
    return children


def _frg_completion(
    layout: _FrgLayout,
    max_height: int,
    *,
    start_step: int,
    max_steps: int,
    deadline: float,
    max_generated: int | None = None,
    paper_pure: bool = False,
) -> _RolloutResult:
    current = layout
    moves: list[Move] = []
    generated = 0
    seen = {current}

    while not _is_goal(current.stacks):
        if len(moves) >= max_steps:
            return _RolloutResult(False, current, tuple(moves), future_disorder_count_stacks(_to_lists(current.stacks)), generated, "FRG_DEPTH_LIMIT")
        if time.perf_counter() >= deadline:
            return _RolloutResult(False, current, tuple(moves), future_disorder_count_stacks(_to_lists(current.stacks)), generated, "TIME_LIMIT")
        if max_generated is not None and generated >= max_generated:
            return _RolloutResult(False, current, tuple(moves), future_disorder_count_stacks(_to_lists(current.stacks)), generated, "FRG_GENERATED_LIMIT")

        children = _frg_step_children(
            current,
            max_height,
            start_step + len(moves),
            limit=None,
            paper_pure=paper_pure,
        )
        generated += children[0].generated if children else 0
        selected: _Child | None = None
        for child in children:
            if child.layout not in seen:
                selected = child
                break
        if selected is None:
            return _RolloutResult(False, current, tuple(moves), future_disorder_count_stacks(_to_lists(current.stacks)), generated, "FRG_CYCLE_OR_DEAD_END")
        current = selected.layout
        moves.extend(selected.moves)
        seen.add(current)

    return _RolloutResult(True, current, tuple(moves), 0, generated, "FRG_FEASIBLE")


def _rebase_moves(moves: tuple[Move, ...], start_step: int) -> tuple[Move, ...]:
    return tuple(
        Move(
            step=start_step + offset,
            box=move.box,
            from_stack=move.from_stack,
            to_stack=move.to_stack,
            from_level=move.from_level,
            to_level=move.to_level,
        )
        for offset, move in enumerate(moves)
    )


def _frg_completion_cached(
    layout: _FrgLayout,
    max_height: int,
    *,
    start_step: int,
    max_steps: int,
    deadline: float,
    cache: dict[_FrgLayout, _RolloutResult],
    max_generated: int | None = None,
    paper_pure: bool = False,
) -> _RolloutResult:
    cached = cache.get(layout)
    if cached is not None:
        if cached.feasible and len(cached.moves) <= max_steps:
            return _RolloutResult(
                True,
                cached.layout,
                _rebase_moves(cached.moves, start_step),
                cached.final_n,
                0,
                cached.status,
            )
        if not cached.feasible and cached.status == "FRG_CYCLE_OR_DEAD_END":
            return _RolloutResult(
                False,
                cached.layout,
                _rebase_moves(cached.moves, start_step),
                cached.final_n,
                0,
                cached.status,
            )

    completion_kwargs = {
        "start_step": start_step,
        "max_steps": max_steps,
        "deadline": deadline,
        "paper_pure": paper_pure,
    }
    if max_generated is not None:
        completion_kwargs["max_generated"] = max_generated
    result = _frg_completion(layout, max_height, **completion_kwargs)
    if result.status not in {"TIME_LIMIT", "FRG_DEPTH_LIMIT", "FRG_GENERATED_LIMIT"}:
        cache[layout] = _RolloutResult(
            result.feasible,
            result.layout,
            _rebase_moves(result.moves, 0),
            result.final_n,
            result.generated,
            result.status,
        )
    return result


def _candidate_rollout_budget(max_depth: int, child_move_count: int, incumbent_move_count: int | None) -> int:
    """Return the FRG completion budget for paper-style candidate evaluation.

    BS-FRG evaluates each child by completing it with FRG and ranking the full
    path length. The completion budget therefore should be the remaining search
    depth, even when an incumbent is already known; clipping it to incumbent - 1
    turns worse-but-promising candidates into artificial dead ends.
    """

    del incumbent_move_count
    return max(0, int(max_depth) - int(child_move_count))


def _frontier_dedup_key(layout: _FrgLayout) -> _FrgLayout:
    return layout


def _is_visit_dominated(visited_depths: dict[_FrgLayout, int], layout: _FrgLayout, move_count: int) -> bool:
    previous_depth = visited_depths.get(layout)
    return previous_depth is not None and previous_depth <= move_count


def _paper_pure_evaluation_score(
    child: _Child,
    rollout: _RolloutResult,
    *,
    estimated_total: int,
    child_rank: int,
) -> tuple[float, ...]:
    rollout_penalty = 0.0 if rollout.feasible else 1.0
    return (
        rollout_penalty,
        float(estimated_total + 1000 * rollout.final_n),
        *child.score_hint,
        float(child_rank),
    )


def _single_move_children(layout: _FrgLayout, max_height: int, step: int, k: int) -> list[_Child]:
    stacks = _to_lists(layout.stacks)
    children: list[_Child] = []
    for source, src in enumerate(stacks):
        if not src:
            continue
        box = int(src[-1])
        destinations = _destination_choices(stacks, box, source, max_height)[:k]
        for rank, target in destinations:
            next_layout, move = _apply_move(
                layout,
                _MoveChoice(rank, source, target),
                step,
                reset_reduction=True,
            )
            children.append(_Child(next_layout, (move,), len(destinations), (0.0, *rank, float(source), float(target))))
    children.sort(key=lambda child: child.score_hint)
    return children


def _frg_iteration_child(layout: _FrgLayout, max_height: int, step: int, *, paper_pure: bool = False) -> list[_Child]:
    children = _frg_step_children(layout, max_height, step, limit=1, paper_pure=paper_pure)
    return [
        _Child(child.layout, child.moves, child.generated, (1.0, *child.score_hint))
        for child in children
    ]


def _compound_reduction_child(
    layout: _FrgLayout,
    max_height: int,
    step: int,
    source: int,
    max_moves: int,
    deadline: float,
    *,
    paper_pure: bool = False,
) -> _Child | None:
    stacks = _to_lists(layout.stacks)
    if source < 0 or source >= len(stacks) or not stacks[source]:
        return None
    current = _start_reduction_layout(layout, source, max_height, paper_pure=paper_pure)
    moves: list[Move] = []
    generated = 0
    seen = {current}

    while current.active_reduce_stack is not None:
        if len(moves) >= max_moves or time.perf_counter() >= deadline:
            return None
        choices = _reduction_choices(
            current,
            max_height,
            step + len(moves),
            force_source=current.active_reduce_stack,
            paper_pure=paper_pure,
        )
        generated += len(choices)
        selected: tuple[_FrgLayout, Move, _MoveChoice] | None = None
        for choice in choices:
            next_layout, move = _apply_move(current, choice, step + len(moves), reset_reduction=False)
            if _stopping_reduction_criterion(next_layout, max_height):
                next_layout = _FrgLayout(next_layout.stacks, None, next_layout.selected_counts)
            if next_layout not in seen:
                selected = (next_layout, move, choice)
                break
        if selected is None:
            return None
        current, move, choice = selected
        moves.append(move)
        seen.add(current)

    if not moves:
        return None
    score_hint = (
        2.0,
        float(len(moves)),
        float(future_disorder_count_stacks(_to_lists(current.stacks))),
        float(source),
    )
    return _Child(current, tuple(moves), generated, score_hint)


def _generate_candidates(
    node: _BeamNode,
    max_height: int,
    *,
    k: int,
    max_depth: int,
    deadline: float,
    paper_pure: bool = False,
) -> list[_Child]:
    step = len(node.moves)
    remaining = max_depth - step
    if remaining <= 0:
        return []

    children: list[_Child] = []
    children.extend(_single_move_children(node.layout, max_height, step, k))
    children.extend(_frg_iteration_child(node.layout, max_height, step, paper_pure=paper_pure))
    compound_limit = max(1, remaining)
    for source in range(len(node.layout.stacks)):
        child = _compound_reduction_child(
            node.layout,
            max_height,
            step,
            source,
            compound_limit,
            deadline,
            paper_pure=paper_pure,
        )
        if child is not None:
            children.append(child)

    dedup: dict[_FrgLayout, _Child] = {}
    for child in children:
        existing = dedup.get(child.layout)
        if existing is None or (len(child.moves), child.score_hint) < (len(existing.moves), existing.score_hint):
            dedup[child.layout] = child
    return sorted(dedup.values(), key=lambda child: (child.score_hint, child.layout.stacks))


class BsFrgSolver(Solver):
    """Beam search using FRG completion as the candidate evaluator."""

    name = "bs_frg"

    def __init__(self, *, paper_pure: bool = False):
        self.paper_pure = bool(paper_pure)
        if self.paper_pure:
            self.name = "bs_frg_pure"

    def solve(self, instance: Instance, config: SolverConfig) -> Solution:
        start = time.perf_counter()
        deadline = start + config.time_limit_sec if config.time_limit_sec > 0 else inf
        if instance.variant is not ProblemVariant.PREMARSHALLING_CPMP:
            return failure_solution(self.name, start, "UNSUPPORTED_VARIANT")

        initial = _initial_layout(instance)
        if _is_goal(initial.stacks):
            return Solution(self.name, True, False, 0.0, time.perf_counter() - start, [], "FEASIBLE_ALREADY_SORTED")

        max_depth = max(0, int(config.max_depth))
        beam_width = max(1, int(config.beam_width))
        k = config.candidate_limit_per_state
        destination_limit = max(1, int(k if k is not None else 3))

        expanded = 0
        generated = 0
        visited_depths: dict[_FrgLayout, int] = {initial: 0}
        completion_cache: dict[_FrgLayout, _RolloutResult] = {}
        best_moves: tuple[Move, ...] | None = None
        best_status = "NO_SOLUTION"
        incumbent_move_count = (
            len(config.exact_incumbent_moves)
            if config.exact_incumbent_moves is not None and len(config.exact_incumbent_moves) > 0
            else None
        )
        stop_on_incumbent_improvement = bool(getattr(config, "exact_stop_on_incumbent_improvement", False))

        def incumbent_improved() -> bool:
            return (
                incumbent_move_count is not None
                and best_moves is not None
                and len(best_moves) < int(incumbent_move_count)
            )

        def build_feasible_solution() -> Solution:
            return Solution(
                solver_name=self.name,
                feasible=True,
                optimal=False,
                objective=float(len(best_moves or ())),
                runtime_sec=time.perf_counter() - start,
                moves=list(best_moves or ()),
                status=best_status,
                expanded_states=expanded,
                generated_states=generated,
                visited_states=len(visited_depths),
            )

        if self.paper_pure:
            initial_rollout = _frg_completion_cached(
                initial,
                instance.max_height,
                start_step=0,
                max_steps=max_depth,
                deadline=deadline,
                cache=completion_cache,
                paper_pure=True,
            )
        else:
            initial_rollout = _frg_completion_cached(
                initial,
                instance.max_height,
                start_step=0,
                max_steps=max_depth,
                deadline=deadline,
                cache=completion_cache,
            )
        generated += initial_rollout.generated
        if initial_rollout.feasible:
            best_moves = initial_rollout.moves
            best_status = "FEASIBLE_FRG_INITIAL"
            if stop_on_incumbent_improvement and incumbent_improved():
                return build_feasible_solution()

        frontier = [_BeamNode(initial, tuple())]
        while frontier and time.perf_counter() < deadline:
            evaluated: list[_EvaluatedChild] = []
            for node in frontier:
                if len(node.moves) >= max_depth or time.perf_counter() >= deadline:
                    continue
                if not self.paper_pure and best_moves is not None and len(node.moves) >= len(best_moves):
                    continue
                if incumbent_move_count is not None and len(node.moves) >= int(incumbent_move_count):
                    continue
                expanded += 1
                children = _generate_candidates(
                    node,
                    instance.max_height,
                    k=destination_limit,
                    max_depth=max_depth,
                    deadline=deadline,
                    paper_pure=self.paper_pure,
                )
                generated += sum(child.generated for child in children)
                for child_rank, child in enumerate(children):
                    child_moves = node.moves + child.moves
                    if len(child_moves) > max_depth:
                        continue
                    if (
                        config.enable_visited_pruning
                        and not _is_goal(child.layout.stacks)
                        and _is_visit_dominated(visited_depths, child.layout, len(child_moves))
                    ):
                        continue
                    previous_depth = visited_depths.get(child.layout)
                    if previous_depth is None or len(child_moves) < previous_depth:
                        visited_depths[child.layout] = len(child_moves)
                    if _is_goal(child.layout.stacks):
                        if best_moves is None or len(child_moves) < len(best_moves):
                            best_moves = child_moves
                            best_status = "FEASIBLE_BEAM_GOAL"
                            if stop_on_incumbent_improvement and incumbent_improved():
                                return build_feasible_solution()
                    remaining_depth = _candidate_rollout_budget(
                        max_depth,
                        len(child_moves),
                        len(best_moves) if best_moves is not None else None,
                    )
                    if self.paper_pure:
                        rollout = _frg_completion_cached(
                            child.layout,
                            instance.max_height,
                            start_step=len(child_moves),
                            max_steps=remaining_depth,
                            deadline=deadline,
                            cache=completion_cache,
                            paper_pure=True,
                        )
                    else:
                        rollout = _frg_completion_cached(
                            child.layout,
                            instance.max_height,
                            start_step=len(child_moves),
                            max_steps=remaining_depth,
                            deadline=deadline,
                            cache=completion_cache,
                        )
                    generated += rollout.generated
                    if rollout.feasible:
                        full_moves = child_moves + rollout.moves
                        if best_moves is None or len(full_moves) < len(best_moves):
                            best_moves = full_moves
                            best_status = "FEASIBLE_BS_FRG_ROLLOUT"
                            if stop_on_incumbent_improvement and incumbent_improved():
                                return build_feasible_solution()
                    estimated_total = len(child_moves) + len(rollout.moves)
                    if self.paper_pure:
                        score = _paper_pure_evaluation_score(
                            child,
                            rollout,
                            estimated_total=estimated_total,
                            child_rank=child_rank,
                        )
                    else:
                        rollout_penalty = 0.0 if rollout.feasible else 1.0
                        score = (
                            rollout_penalty,
                            float(estimated_total + 1000 * rollout.final_n),
                            *child.score_hint,
                            float(child_rank),
                        )
                    evaluated.append(_EvaluatedChild(score, _BeamNode(child.layout, child_moves), rollout))

            if not evaluated:
                break
            evaluated.sort(key=lambda item: (item.score, item.node.layout.stacks))
            next_frontier: list[_BeamNode] = []
            layer_seen: set[_FrgLayout] = set()
            for item in evaluated:
                frontier_key = _frontier_dedup_key(item.node.layout)
                if frontier_key in layer_seen:
                    continue
                layer_seen.add(frontier_key)
                next_frontier.append(item.node)
                if len(next_frontier) >= beam_width:
                    break
            frontier = next_frontier

        runtime = time.perf_counter() - start
        if best_moves is not None:
            return build_feasible_solution()
        status = "TIME_LIMIT" if time.perf_counter() >= deadline else best_status
        return Solution(
            solver_name=self.name,
            feasible=False,
            optimal=False,
            objective=float("inf"),
            runtime_sec=runtime,
            moves=[],
            status=status,
            expanded_states=expanded,
            generated_states=generated,
            visited_states=len(visited_depths),
        )
