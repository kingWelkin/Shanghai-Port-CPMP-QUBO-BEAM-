"""Rolling-horizon lookahead solver."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from ..core import Move, State, apply_move, create_initial_state, generate_legal_moves, hash_state, is_goal
from ..scoring import cpmp_business_feasible, score_state
from .base import Solver, SolverConfig, failure_solution, solution_from_state


class RollingHorizonSolver(Solver):
    """Repeatedly plan a shallow tree and commit a prefix."""

    name = "rolling_horizon"

    def solve(self, instance, config: SolverConfig):
        """Run rolling-horizon search."""

        start = time.perf_counter()
        state = create_initial_state(instance)
        expanded = generated = 0
        seen = {hash_state(instance, state)}
        while not is_goal(instance, state) and state.move_count < config.max_depth:
            path, e, g = self._best_path(instance, state, config)
            expanded += e
            generated += g
            if not path:
                break
            for mv in path[: max(1, config.prefix_commit_length)]:
                if state.move_count >= config.max_depth:
                    break
                state = apply_move(instance, state, mv)
                h = hash_state(instance, state)
                if h in seen and config.enable_visited_pruning:
                    break
                seen.add(h)
                if is_goal(instance, state):
                    return solution_from_state(
                        self.name,
                        instance,
                        state,
                        start,
                        expanded_states=expanded,
                        generated_states=generated,
                        visited_states=len(seen),
                    )
        if is_goal(instance, state):
            return solution_from_state(self.name, instance, state, start)
        return failure_solution(
            self.name,
            start,
            "NO_FEASIBLE_FOUND",
            expanded_states=expanded,
            generated_states=generated,
            visited_states=len(seen),
        )

    def _best_path(self, instance, root: State, config: SolverConfig) -> Tuple[List[Move], int, int]:
        frontier: List[Tuple[State, List[Move]]] = [(root, [])]
        best_path: List[Move] = []
        best_score = float("inf")
        expanded = generated = 0
        for _ in range(config.rolling_depth):
            next_frontier: List[Tuple[State, List[Move]]] = []
            for state, path in frontier:
                expanded += 1
                moves = generate_legal_moves(instance, state)
                scored = []
                for mv in moves:
                    ns = apply_move(instance, state, mv)
                    if not cpmp_business_feasible(instance, ns, config.entropy_alpha):
                        continue
                    if is_goal(instance, ns):
                        return path + [mv], expanded, generated + 1
                    scored.append((score_state(instance, ns, config.score_weights), ns, path + [mv]))
                scored.sort(key=lambda x: x[0])
                if config.candidate_limit_per_state is not None:
                    scored = scored[: config.candidate_limit_per_state]
                for sc, ns, pth in scored:
                    generated += 1
                    if sc < best_score or is_goal(instance, ns):
                        best_score = sc
                        best_path = pth
                    next_frontier.append((ns, pth))
            next_frontier.sort(key=lambda item: score_state(instance, item[0], config.score_weights))
            frontier = next_frontier[: config.beam_width]
            if not frontier:
                break
        return best_path, expanded, generated
