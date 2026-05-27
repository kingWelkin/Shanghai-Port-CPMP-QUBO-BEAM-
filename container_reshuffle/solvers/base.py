"""Solver base classes and helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..core import Instance, Solution, State, validate_solution
from ..scoring import ScoreWeights


@dataclass
class SolverConfig:
    """Common configuration shared by solvers."""

    max_depth: int = 200
    beam_width: int = 100
    max_candidates_per_layer: int = 1000
    candidate_limit_per_state: Optional[int] = None
    enable_visited_pruning: bool = True
    enable_canonical_hash: bool = False
    random_seed: int = 1
    time_limit_sec: float = 3600.0

    max_qubo_vars: int = 1000
    qubo_penalty_a: float = 10.0
    diversity_weight: float = 1.0
    rolling_depth: int = 4
    prefix_commit_length: int = 1
    qubo_num_reads: int = 50
    qubo_time_limit_sec: Optional[float] = None
    qubo_sa_iterations: int = 5000
    qubo_call_interval: int = 1
    qubo_min_candidates_for_call: int = 0
    qubo_repair_hamming_slack: int = 3
    entropy_alpha: float = 1.3
    mip_warm_start: bool = True
    mip_warm_start_time_limit_sec: float = 5.0

    score_weights: ScoreWeights = field(default_factory=ScoreWeights)


class Solver:
    """Base solver API."""

    name = "solver"

    def solve(self, instance: Instance, config: SolverConfig) -> Solution:
        """Solve an instance and return a structured result."""

        raise NotImplementedError


def not_applicable(name: str, message: str = "NOT_APPLICABLE") -> Solution:
    """Return a structured not-applicable result."""

    return Solution(name, False, False, float("inf"), 0.0, [], message)


def solution_from_state(
    name: str,
    instance: Instance,
    state: State,
    start_time: float,
    status: str = "FEASIBLE",
    optimal: bool = False,
    **diagnostics,
) -> Solution:
    """Build and validate a solution from a terminal state."""

    sol = Solution(
        solver_name=name,
        feasible=True,
        optimal=optimal,
        objective=float(len(state.moves)),
        runtime_sec=time.perf_counter() - start_time,
        moves=state.moves,
        status=status,
        **diagnostics,
    )
    ok, msg, _ = validate_solution(instance, sol)
    if not ok:
        sol.feasible = False
        sol.status = f"VALIDATION_FAILED: {msg}"
        sol.objective = float("inf")
    return sol


def failure_solution(
    name: str,
    start_time: float,
    status: str,
    **diagnostics,
) -> Solution:
    """Build a structured failure result."""

    return Solution(
        solver_name=name,
        feasible=False,
        optimal=False,
        objective=float("inf"),
        runtime_sec=time.perf_counter() - start_time,
        moves=[],
        status=status,
        **diagnostics,
    )
