"""State similarity functions."""

from __future__ import annotations

from typing import Dict

from .core import State


def container_to_stack_map(state: State) -> Dict[int, int]:
    """Map each currently present container to its stack index."""

    return {box: i for i, stack in enumerate(state.stacks) for box in stack}


def stack_assignment_similarity(a: State, b: State) -> float:
    """Fraction of common containers assigned to the same stack."""

    ma = container_to_stack_map(a)
    mb = container_to_stack_map(b)
    common = sorted(set(ma) & set(mb))
    if not common:
        return 1.0
    return sum(1 for box in common if ma[box] == mb[box]) / len(common)


def combined_similarity(a: State, b: State) -> float:
    """Combined state similarity. First version uses stack assignment only."""

    return stack_assignment_similarity(a, b)
