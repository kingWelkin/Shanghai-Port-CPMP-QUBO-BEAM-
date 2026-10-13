"""Exact A* search solver."""

from __future__ import annotations

import heapq
import time
from math import inf

from ..core import (
    Instance,
    ProblemVariant,
    State,
    apply_move,
    create_initial_state,
    generate_legal_moves,
    hash_state,
    is_goal,
)
from ..features import current_target_blockers, disorder_box_count
from ..scoring import cpmp_business_feasible
from .base import Solver, SolverConfig, failure_solution, solution_from_state


class ExactAStarSolver(Solver):
    """Optimal A* over the complete legal relocation graph.

    This solver deliberately does not apply beam-width, per-state candidate,
    weighted-heuristic, or incumbent-quality pruning.  It can mark a solution
    optimal only when the goal state is popped from the open set.
    """

    name = "exact_astar"

    def solve(self, instance: Instance, config: SolverConfig):
        """Run exact A* and validate any returned solution."""

        start = time.perf_counter()
        initial = create_initial_state(instance)
        if is_goal(instance, initial):
            return solution_from_state(self.name, instance, initial, start, status="OPTIMAL", optimal=True)

        initial_key = hash_state(instance, initial, config.enable_canonical_hash)
        best_g = {initial_key: 0}
        heap: list[tuple[float, int, int, int, State]] = []
        counter = 0
        h0 = self._lower_bound(instance, initial)
        heapq.heappush(heap, (h0, h0, 0, counter, initial))

        expanded = 0
        generated = 0
        depth_limited = False
        incumbent = self._initial_incumbent(instance, initial, config)
        incumbent_from_config = incumbent is not None
        max_visited = getattr(config, "exact_max_visited_states", None)

        while heap:
            if time.perf_counter() - start > config.time_limit_sec:
                if incumbent is not None:
                    return solution_from_state(
                        self.name,
                        instance,
                        incumbent,
                        start,
                        status="TIME_LIMIT_FEASIBLE_NOT_PROVEN",
                        optimal=False,
                        expanded_states=expanded,
                        generated_states=generated,
                        visited_states=len(best_g),
                    )
                return failure_solution(
                    self.name,
                    start,
                    "TIME_LIMIT_NO_FEASIBLE_FOUND",
                    expanded_states=expanded,
                    generated_states=generated,
                    visited_states=len(best_g),
                )
            if max_visited is not None and len(best_g) >= max_visited:
                if incumbent is not None:
                    return solution_from_state(
                        self.name,
                        instance,
                        incumbent,
                        start,
                        status="STATE_LIMIT_FEASIBLE_NOT_PROVEN",
                        optimal=False,
                        expanded_states=expanded,
                        generated_states=generated,
                        visited_states=len(best_g),
                    )
                return failure_solution(
                    self.name,
                    start,
                    "STATE_LIMIT_NO_FEASIBLE_FOUND",
                    expanded_states=expanded,
                    generated_states=generated,
                    visited_states=len(best_g),
                )

            _f, _h, g, _idx, state = heapq.heappop(heap)
            state_key = hash_state(instance, state, config.enable_canonical_hash)
            if g != best_g.get(state_key, inf):
                continue
            if incumbent is not None and _f >= incumbent.move_count:
                status = "OPTIMAL_WARM_START_PROVEN" if incumbent_from_config else "OPTIMAL"
                return solution_from_state(
                    self.name,
                    instance,
                    incumbent,
                    start,
                    status=status,
                    optimal=True,
                    expanded_states=expanded,
                    generated_states=generated,
                    visited_states=len(best_g),
                )
            if is_goal(instance, state):
                return solution_from_state(
                    self.name,
                    instance,
                    state,
                    start,
                    status="OPTIMAL",
                    optimal=True,
                    expanded_states=expanded,
                    generated_states=generated,
                    visited_states=len(best_g),
                )
            if state.move_count >= config.max_depth:
                depth_limited = True
                continue

            expanded += 1
            for move in generate_legal_moves(instance, state):
                try:
                    nxt = apply_move(instance, state, move)
                except ValueError:
                    continue
                if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                    continue

                generated += 1
                nxt_key = hash_state(instance, nxt, config.enable_canonical_hash)
                nxt_g = nxt.move_count
                if nxt_g >= best_g.get(nxt_key, inf):
                    continue

                if max_visited is not None and len(best_g) >= max_visited:
                    if incumbent is not None:
                        return solution_from_state(
                            self.name,
                            instance,
                            incumbent,
                            start,
                            status="STATE_LIMIT_FEASIBLE_NOT_PROVEN",
                            optimal=False,
                            expanded_states=expanded,
                            generated_states=generated,
                            visited_states=len(best_g),
                        )
                    return failure_solution(
                        self.name,
                        start,
                        "STATE_LIMIT_NO_FEASIBLE_FOUND",
                        expanded_states=expanded,
                        generated_states=generated,
                        visited_states=len(best_g),
                    )

                best_g[nxt_key] = nxt_g
                if is_goal(instance, nxt) and (incumbent is None or nxt.move_count < incumbent.move_count):
                    incumbent = nxt
                    incumbent_from_config = False

                counter += 1
                nxt_h = self._lower_bound(instance, nxt)
                if incumbent is not None and nxt_g + nxt_h >= incumbent.move_count:
                    continue
                heapq.heappush(heap, (nxt_g + nxt_h, nxt_h, nxt_g, counter, nxt))

        if incumbent is not None:
            status = "OPTIMAL_WARM_START_PROVEN" if incumbent_from_config else "OPTIMAL"
            return solution_from_state(
                self.name,
                instance,
                incumbent,
                start,
                status=status,
                optimal=True,
                expanded_states=expanded,
                generated_states=generated,
                visited_states=len(best_g),
            )

        status = "DEPTH_LIMIT_NO_FEASIBLE_FOUND" if depth_limited else "NO_FEASIBLE_FOUND_EXHAUSTIVE"
        return failure_solution(
            self.name,
            start,
            status,
            expanded_states=expanded,
            generated_states=generated,
            visited_states=len(best_g),
        )

    @staticmethod
    def _lower_bound(instance: Instance, state: State) -> int:
        """Return an admissible lower bound on remaining relocations."""

        if instance.variant is ProblemVariant.RETRIEVAL_CRP:
            return current_target_blockers(instance, state)
        return disorder_box_count(instance, state)

    @staticmethod
    def _initial_incumbent(instance: Instance, initial: State, config: SolverConfig) -> State | None:
        """Validate configured incumbent moves and return their terminal state."""

        moves = getattr(config, "exact_incumbent_moves", None)
        if not moves:
            return None
        state = initial
        for move in moves:
            try:
                state = apply_move(instance, state, move)
            except ValueError:
                return None
            if not cpmp_business_feasible(instance, state, config.entropy_alpha):
                return None
        if state.move_count > config.max_depth or not is_goal(instance, state):
            return None
        return state


class _SearchTimedOut(Exception):
    """Internal control-flow exception for time-limited exact search."""


class ExactIDAStarSolver(Solver):
    """Memory-bounded exact IDA* search.

    IDA* keeps only the current DFS path and repeats bounded ``f = g + h``
    passes.  It preserves optimality when it finishes, but it may do much more
    repeated work than open-list A*.
    """

    name = "exact_idastar"

    def solve(self, instance: Instance, config: SolverConfig):
        """Run exact iterative-deepening A*."""

        start = time.perf_counter()
        initial = create_initial_state(instance)
        if is_goal(instance, initial):
            return solution_from_state(self.name, instance, initial, start, status="OPTIMAL", optimal=True)

        bound = ExactAStarSolver._lower_bound(instance, initial)
        path_keys = {hash_state(instance, initial, config.enable_canonical_hash)}
        stats = {"expanded": 0, "generated": 0, "visited": 1, "depth_limited": False}

        try:
            while bound <= config.max_depth:
                result = self._dfs(instance, initial, bound, path_keys, start, config, stats)
                if isinstance(result, State):
                    return solution_from_state(
                        self.name,
                        instance,
                        result,
                        start,
                        status="OPTIMAL",
                        optimal=True,
                        expanded_states=stats["expanded"],
                        generated_states=stats["generated"],
                        visited_states=stats["visited"],
                    )
                if result == inf:
                    status = "DEPTH_LIMIT_NO_FEASIBLE_FOUND" if stats["depth_limited"] else "NO_FEASIBLE_FOUND_EXHAUSTIVE"
                    return failure_solution(
                        self.name,
                        start,
                        status,
                        expanded_states=stats["expanded"],
                        generated_states=stats["generated"],
                        visited_states=stats["visited"],
                    )
                bound = int(result)
        except _SearchTimedOut:
            return failure_solution(
                self.name,
                start,
                "TIME_LIMIT_NO_PROOF",
                expanded_states=stats["expanded"],
                generated_states=stats["generated"],
                visited_states=stats["visited"],
            )

        return failure_solution(
            self.name,
            start,
            "DEPTH_LIMIT_NO_FEASIBLE_FOUND",
            expanded_states=stats["expanded"],
            generated_states=stats["generated"],
            visited_states=stats["visited"],
        )

    def _dfs(
        self,
        instance: Instance,
        state: State,
        bound: int,
        path_keys: set[tuple],
        start: float,
        config: SolverConfig,
        stats: dict[str, int | bool],
    ):
        if time.perf_counter() - start > config.time_limit_sec:
            raise _SearchTimedOut

        g_value = state.move_count
        h_value = ExactAStarSolver._lower_bound(instance, state)
        f_value = g_value + h_value
        if f_value > bound:
            return f_value
        if is_goal(instance, state):
            return state
        if state.move_count >= config.max_depth:
            stats["depth_limited"] = True
            return inf

        stats["expanded"] += 1
        min_next = inf
        successors = []
        for move in generate_legal_moves(instance, state):
            try:
                nxt = apply_move(instance, state, move)
            except ValueError:
                continue
            if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                continue
            key = hash_state(instance, nxt, config.enable_canonical_hash)
            if key in path_keys:
                continue
            stats["generated"] += 1
            nxt_h = ExactAStarSolver._lower_bound(instance, nxt)
            successors.append((nxt.move_count + nxt_h, nxt_h, nxt.move_count, key, nxt))

        successors.sort(key=lambda item: item[:3])
        for _f, _h, _g, key, nxt in successors:
            path_keys.add(key)
            stats["visited"] += 1
            result = self._dfs(instance, nxt, bound, path_keys, start, config, stats)
            path_keys.remove(key)
            if isinstance(result, State):
                return result
            if result < min_next:
                min_next = result
        return min_next
