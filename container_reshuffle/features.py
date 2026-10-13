"""Feature functions used by heuristic, beam, and QUBO-assisted solvers."""

from __future__ import annotations

from math import inf
from typing import List, Optional, Tuple

from .core import Instance, Move, ProblemVariant, State, apply_move, create_initial_state, future_disorder_count_stacks


def find_box(stacks: List[List[int]], box: int) -> Optional[Tuple[int, int]]:
    """Return ``(stack_index, zero_based_level)`` for a box, or None."""

    for i, stack in enumerate(stacks):
        for level, value in enumerate(stack):
            if value == box:
                return i, level
    return None


def current_target_blockers(instance: Instance, state: State) -> int:
    """Count blockers above the next retrieval target."""

    if state.next_retrieval_id > instance.container_count:
        return 0
    loc = find_box(state.stacks, state.next_retrieval_id)
    if loc is None:
        return 0
    i, level = loc
    return len(state.stacks[i]) - 1 - level


def blocking_count(instance: Instance, state: State) -> int:
    """Count boxes that block a smaller not-yet-retrieved box below them."""

    threshold = state.next_retrieval_id if instance.variant is ProblemVariant.RETRIEVAL_CRP else 1
    count = 0
    for stack in state.stacks:
        for idx, box in enumerate(stack):
            if box < threshold:
                continue
            below = [b for b in stack[:idx] if b >= threshold]
            if below and min(below) < box:
                count += 1
    return count


def min_priority(stack: List[int]) -> float:
    """Return the smallest ID in a stack; empty stacks return +infinity."""

    return inf if not stack else float(min(stack))


def bad_relocation_count(instance: Instance, state: State) -> int:
    """Simple bad-relocation proxy for CRP."""

    return blocking_count(instance, state)


def safe_destination_count(instance: Instance, state: State, box: int) -> int:
    """Count destinations where placing ``box`` creates no immediate CRP blocker."""

    count = 0
    for stack in state.stacks:
        if len(stack) >= instance.max_height:
            continue
        if not stack or box < min_priority(stack):
            count += 1
    return count


def retrieval_progress(instance: Instance, state: State) -> int:
    """Return number of retrieved target containers."""

    return min(instance.container_count, state.next_retrieval_id - 1)


def well_prefix_len(stack: List[int]) -> int:
    """Longest strictly decreasing prefix from the bottom."""

    if not stack:
        return 0
    length = 1
    for idx in range(len(stack) - 1):
        if stack[idx] > stack[idx + 1]:
            length += 1
        else:
            break
    return length


def bad_count(instance: Instance, state: State) -> int:
    """Count boxes outside the well-ordered bottom prefix."""

    return sum(len(stack) - well_prefix_len(stack) for stack in state.stacks)


def disorder_box_count(instance: Instance, state: State) -> int:
    """Business definition of future disorder boxes ``N(Future)``.

    For each stack, scan upward from the second level. At the first box whose
    ID is larger than the box immediately below it, count that box and every
    box above it.
    """

    return future_disorder_count_stacks(state.stacks)


def entropy_value(instance: Instance, state: State) -> int:
    """Return ``G(x) = M(Now) + N(Future)`` for a state."""

    return state.move_count + disorder_box_count(instance, state)


def entropy_limit(instance: Instance, alpha: float = 1.3) -> int:
    """Return the integer business limit ``floor(alpha * N(0))``.

    The multiplier is capped at 2.0 according to the stated business rule.
    """

    from math import floor

    initial = State(stacks=[list(stack) for stack in instance.stacks])
    capped_alpha = min(float(alpha), 2.0)
    return int(floor(capped_alpha * disorder_box_count(instance, initial) + 1e-9))


def entropy_trace(instance: Instance, moves: List[Move]) -> List[int]:
    """Return ``G`` values along a move sequence, including the initial state."""

    state = create_initial_state(instance)
    values = [entropy_value(instance, state)]
    for move in moves:
        state = apply_move(instance, state, move)
        values.append(entropy_value(instance, state))
    return values


def max_entropy_along_moves(instance: Instance, moves: List[Move]) -> int:
    """Return max ``G`` over the initial and intermediate states."""

    return max(entropy_trace(instance, moves))


def disorder_trace(instance: Instance, moves: List[Move]) -> List[int]:
    """Return ``N(Future)`` values along a move sequence."""

    state = create_initial_state(instance)
    values = [disorder_box_count(instance, state)]
    for move in moves:
        state = apply_move(instance, state, move)
        values.append(disorder_box_count(instance, state))
    return values


def total_disorder_along_moves(instance: Instance, moves: List[Move]) -> int:
    """Return cumulative ``N(Future)`` over the initial and intermediate states."""

    return sum(disorder_trace(instance, moves))


def validate_entropy_constraint(instance: Instance, moves: List[Move], alpha: float = 1.3) -> Tuple[bool, str, List[int]]:
    """Validate the business entropy cap for a move sequence."""

    values = entropy_trace(instance, moves)
    limit = entropy_limit(instance, alpha)
    max_value = max(values)
    if max_value > limit:
        return False, f"entropy cap violated: max G={max_value}, limit={limit}", values
    return True, "ok", values


def inversion_count(instance: Instance, state: State) -> int:
    """Count within-stack pairs violating bottom -> top decreasing order."""

    total = 0
    for stack in state.stacks:
        for i in range(len(stack)):
            for j in range(i + 1, len(stack)):
                if stack[i] < stack[j]:
                    total += 1
    return total


def dirty_stack_count(instance: Instance, state: State) -> int:
    """Count stacks that are not fully decreasing."""

    return sum(1 for stack in state.stacks if well_prefix_len(stack) < len(stack))


def settled_box_count(instance: Instance, state: State) -> int:
    """Count boxes in well-ordered bottom prefixes."""

    return sum(well_prefix_len(stack) for stack in state.stacks)


def well_prefix_gain(instance: Instance, prev_state: State, new_state: State) -> int:
    """Return increase in settled boxes between two states."""

    return settled_box_count(instance, new_state) - settled_box_count(instance, prev_state)
