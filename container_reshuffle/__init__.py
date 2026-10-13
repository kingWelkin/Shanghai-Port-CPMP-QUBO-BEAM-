"""Experimental framework for CRP/BRP retrieval and CPMP pre-marshalling."""

from .core import (
    Instance,
    Move,
    ProblemVariant,
    Solution,
    State,
    apply_move,
    create_initial_state,
    generate_legal_moves,
    hash_state,
    is_goal,
    validate_instance,
    validate_solution,
)

__all__ = [
    "ProblemVariant",
    "Instance",
    "Move",
    "State",
    "Solution",
    "validate_instance",
    "create_initial_state",
    "generate_legal_moves",
    "apply_move",
    "is_goal",
    "hash_state",
    "validate_solution",
]
