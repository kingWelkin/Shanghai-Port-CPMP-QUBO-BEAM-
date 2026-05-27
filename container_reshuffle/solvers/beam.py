"""Beam search solver."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from ..core import Instance, Move, State, apply_move, create_initial_state, generate_legal_moves, hash_state, is_goal
from ..scoring import cpmp_business_feasible, dominance_value, score_state
from .base import Solver, SolverConfig, failure_solution, solution_from_state


@dataclass
class Candidate:
    """A scored beam candidate."""

    state: State
    score: float
    last_move: Optional[Move]
    hash_value: Tuple


class BeamSearchSolver(Solver):
    """Plain beam search over legal classical moves."""

    name = "beam"

    def solve(self, instance: Instance, config: SolverConfig):
        """Run beam search and validate any returned solution."""

        start = time.perf_counter()
        initial = create_initial_state(instance)
        if is_goal(instance, initial):
            return solution_from_state(self.name, instance, initial, start, optimal=False)
        h0 = hash_state(instance, initial, config.enable_canonical_hash)
        beam = [Candidate(initial, score_state(instance, initial, config.score_weights), None, h0)]
        visited = {h0: dominance_value(instance, initial)}
        expanded = generated = 0
        best_goal: Optional[State] = None

        for _depth in range(config.max_depth):
            if time.perf_counter() - start > config.time_limit_sec:
                break
            candidates = []
            for cand in beam:
                expanded += 1
                moves = generate_legal_moves(instance, cand.state)
                if config.candidate_limit_per_state is not None:
                    moves = self._ordered_moves(instance, cand.state, moves, config)[: config.candidate_limit_per_state]
                for mv in moves:
                    try:
                        ns = apply_move(instance, cand.state, mv)
                    except ValueError:
                        continue
                    if not cpmp_business_feasible(instance, ns, config.entropy_alpha):
                        continue
                    h = hash_state(instance, ns, config.enable_canonical_hash)
                    dom = dominance_value(instance, ns)
                    if config.enable_visited_pruning and visited.get(h, float("inf")) <= dom:
                        continue
                    visited[h] = dom
                    sc = score_state(instance, ns, config.score_weights)
                    nc = Candidate(ns, sc, mv, h)
                    if is_goal(instance, ns):
                        if best_goal is None or len(ns.moves) < len(best_goal.moves):
                            best_goal = ns
                    candidates.append(nc)
            generated += len(candidates)
            if best_goal is not None:
                return solution_from_state(
                    self.name,
                    instance,
                    best_goal,
                    start,
                    expanded_states=expanded,
                    generated_states=generated,
                    visited_states=len(visited),
                )
            if not candidates:
                break
            candidates.sort(key=lambda c: (c.score, c.state.move_count))
            if config.max_candidates_per_layer:
                candidates = candidates[: config.max_candidates_per_layer]
            beam = candidates[: config.beam_width]

        return failure_solution(
            self.name,
            start,
            "NO_FEASIBLE_FOUND",
            expanded_states=expanded,
            generated_states=generated,
            visited_states=len(visited),
        )

    def _ordered_moves(self, instance: Instance, state: State, moves, config: SolverConfig):
        scored = []
        for mv in moves:
            try:
                ns = apply_move(instance, state, mv)
                if not cpmp_business_feasible(instance, ns, config.entropy_alpha):
                    continue
                scored.append((score_state(instance, ns, config.score_weights), mv))
            except ValueError:
                continue
        scored.sort(key=lambda x: x[0])
        return [mv for _, mv in scored]
