"""Scoring functions for classical search."""

from __future__ import annotations

from dataclasses import dataclass

from .core import Instance, ProblemVariant, State
from .features import (
    bad_count,
    bad_relocation_count,
    blocking_count,
    current_target_blockers,
    dirty_stack_count,
    disorder_box_count,
    entropy_limit,
    inversion_count,
    retrieval_progress,
    settled_box_count,
    total_disorder_along_moves,
)


@dataclass
class ScoreWeights:
    """Tunable state scoring weights."""

    w_blocking: float = 1.0
    w_target: float = 2.0
    w_bad: float = 0.5
    w_progress: float = 0.1

    w_badcount: float = 1.0
    w_inv: float = 0.2
    w_dirty: float = 0.5
    w_settled: float = 0.1
    w_gain: float = 0.5


def score_retrieval(instance: Instance, state: State, weights: ScoreWeights) -> float:
    """Lower-is-better CRP state score."""

    return (
        state.move_count
        + weights.w_blocking * blocking_count(instance, state)
        + weights.w_target * current_target_blockers(instance, state)
        + weights.w_bad * bad_relocation_count(instance, state)
        - weights.w_progress * retrieval_progress(instance, state)
    )


def score_cpmp(instance: Instance, state: State, weights: ScoreWeights) -> float:
    """Lower-is-better CPMP state score.

    Business objective priority:
    1. current ``N(Future)``,
    2. use the available ``G = moves + N(Future)`` budget to keep improving,
    3. move count and secondary layout diagnostics.
    """

    disorder_total = state.disorder_total or total_disorder_along_moves(instance, state.moves)
    disorder_current = state.disorder_current or disorder_box_count(instance, state)
    g_value = state.move_count + disorder_current
    return (
        1_000_000.0 * disorder_current
        - 1_000.0 * g_value
        + 10.0 * state.move_count
        + 0.01 * disorder_total
        + weights.w_badcount * bad_count(instance, state)
        + weights.w_inv * inversion_count(instance, state)
        + weights.w_dirty * dirty_stack_count(instance, state)
        - weights.w_settled * settled_box_count(instance, state)
    )


def cpmp_business_feasible(instance: Instance, state: State, alpha: float = 1.3) -> bool:
    """Return whether a CPMP state satisfies the business ``G`` cap."""

    if instance.variant is not ProblemVariant.PREMARSHALLING_CPMP:
        return True
    return state.move_count + disorder_box_count(instance, state) <= entropy_limit(instance, alpha)


def dominance_value(instance: Instance, state: State) -> float:
    """Value used to prune revisits to the same state."""

    if instance.variant is ProblemVariant.PREMARSHALLING_CPMP:
        return 1_000_000.0 * state.move_count + float(state.entropy_max)
    return float(state.move_count)


def score_state(instance: Instance, state: State, weights: ScoreWeights) -> float:
    """Dispatch to variant-specific state scoring."""

    if instance.variant is ProblemVariant.RETRIEVAL_CRP:
        return score_retrieval(instance, state, weights)
    return score_cpmp(instance, state, weights)
