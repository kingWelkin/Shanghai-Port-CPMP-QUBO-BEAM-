"""Classical random and greedy baseline solvers."""

from __future__ import annotations

import random
import time
from typing import Callable, List, Optional

from ..core import (
    Instance,
    Move,
    ProblemVariant,
    State,
    apply_move,
    create_initial_state,
    generate_legal_moves,
    hash_state,
    is_goal,
    normalize_retrieval_state,
)
from ..features import bad_count, find_box, inversion_count, min_priority, well_prefix_len
from ..scoring import cpmp_business_feasible, dominance_value, score_cpmp
from .base import Solver, SolverConfig, failure_solution, not_applicable, solution_from_state


class RandomSolver(Solver):
    """Random legal-move baseline."""

    name = "random"

    def solve(self, instance: Instance, config: SolverConfig):
        """Run random legal moves until goal or depth limit."""

        start = time.perf_counter()
        rng = random.Random(config.random_seed)
        state = create_initial_state(instance)
        seen = {hash_state(instance, state)}
        for _ in range(config.max_depth):
            if is_goal(instance, state):
                return solution_from_state(self.name, instance, state, start)
            moves = generate_legal_moves(instance, state)
            if instance.variant is ProblemVariant.PREMARSHALLING_CPMP:
                filtered = []
                for mv in moves:
                    ns = apply_move(instance, state, mv)
                    if cpmp_business_feasible(instance, ns, config.entropy_alpha):
                        filtered.append(mv)
                moves = filtered
            if not moves:
                break
            rng.shuffle(moves)
            state = apply_move(instance, state, moves[0])
            h = hash_state(instance, state)
            if h in seen and config.enable_visited_pruning:
                break
            seen.add(h)
        if is_goal(instance, state):
            return solution_from_state(self.name, instance, state, start)
        return failure_solution(self.name, start, "NO_FEASIBLE_FOUND")


def _greedy_retrieval_destination(instance: Instance, state: State, source: int, box: int) -> Optional[int]:
    candidates = [
        q
        for q, stack in enumerate(state.stacks)
        if q != source and len(stack) < instance.max_height
    ]
    if not candidates:
        return None
    safe = [q for q in candidates if not state.stacks[q] or box < min_priority(state.stacks[q])]
    if safe:
        non_empty = [q for q in safe if state.stacks[q]]
        pool = non_empty if non_empty else safe
        return min(pool, key=lambda q: min_priority(state.stacks[q]))
    return max(candidates, key=lambda q: min_priority(state.stacks[q]))


class GreedyRetrievalSolver(Solver):
    """Classical CRP greedy relocation rule."""

    name = "greedy_retrieval"

    def solve(self, instance: Instance, config: SolverConfig):
        """Solve a retrieval instance with a safe-destination greedy rule."""

        if instance.variant is not ProblemVariant.RETRIEVAL_CRP:
            return not_applicable(self.name)
        start = time.perf_counter()
        state = create_initial_state(instance)
        while not is_goal(instance, state) and state.move_count < config.max_depth:
            normalize_retrieval_state(instance, state)
            if is_goal(instance, state):
                break
            loc = find_box(state.stacks, state.next_retrieval_id)
            if loc is None:
                break
            source, level = loc
            if level == len(state.stacks[source]) - 1:
                normalize_retrieval_state(instance, state)
                continue
            box = state.stacks[source][-1]
            dest = _greedy_retrieval_destination(instance, state, source, box)
            if dest is None:
                break
            state = apply_move(
                instance,
                state,
                Move(state.move_count, box, source, dest, len(state.stacks[source]), len(state.stacks[dest]) + 1),
            )
        if is_goal(instance, state):
            return solution_from_state(self.name, instance, state, start)
        return failure_solution(self.name, start, "NO_FEASIBLE_FOUND")


def _dirty_stacks(state: State) -> List[int]:
    return [i for i, stack in enumerate(state.stacks) if well_prefix_len(stack) < len(stack)]


def _best_cpmp_destination(instance: Instance, state: State, source: int, config: SolverConfig) -> Optional[int]:
    box = state.stacks[source][-1]
    candidates = [
        q
        for q, stack in enumerate(state.stacks)
        if q != source and len(stack) < instance.max_height
    ]
    if not candidates:
        return None
    clean = []
    for q in candidates:
        if well_prefix_len(state.stacks[q]) != len(state.stacks[q]) or (state.stacks[q] and state.stacks[q][-1] <= box):
            continue
        mv = Move(state.move_count, box, source, q, len(state.stacks[source]), len(state.stacks[q]) + 1)
        if cpmp_business_feasible(instance, apply_move(instance, state, mv), config.entropy_alpha):
            clean.append(q)
    if clean:
        return min(clean, key=lambda q: state.stacks[q][-1] if state.stacks[q] else float("inf"))
    best_q = None
    best_score = float("inf")
    for q in candidates:
        mv = Move(state.move_count, box, source, q, len(state.stacks[source]), len(state.stacks[q]) + 1)
        new_state = apply_move(instance, state, mv)
        if not cpmp_business_feasible(instance, new_state, config.entropy_alpha):
            continue
        sc = score_cpmp(instance, new_state, config.score_weights)
        if sc < best_score:
            best_score = sc
            best_q = q
    return best_q


class GreedyPremarshallingSolver(Solver):
    """Classical CPMP greedy relocation rule."""

    name = "greedy_cpmp"

    def solve(self, instance: Instance, config: SolverConfig):
        """Solve a CPMP instance with a dirty-stack greedy rule."""

        if instance.variant is not ProblemVariant.PREMARSHALLING_CPMP:
            return not_applicable(self.name)
        start = time.perf_counter()
        state = create_initial_state(instance)
        seen = {hash_state(instance, state)}
        while not is_goal(instance, state) and state.move_count < config.max_depth:
            dirty = _dirty_stacks(state)
            if not dirty:
                break
            source = max(dirty, key=lambda i: len(state.stacks[i]) - well_prefix_len(state.stacks[i]))
            dest = _best_cpmp_destination(instance, state, source, config)
            if dest is None:
                break
            box = state.stacks[source][-1]
            state = apply_move(
                instance,
                state,
                Move(state.move_count, box, source, dest, len(state.stacks[source]), len(state.stacks[dest]) + 1),
            )
            if not cpmp_business_feasible(instance, state, config.entropy_alpha):
                break
            h = hash_state(instance, state)
            if h in seen:
                break
            seen.add(h)
        if is_goal(instance, state):
            return solution_from_state(self.name, instance, state, start)
        return failure_solution(self.name, start, "NO_FEASIBLE_FOUND")


class MultiRuleGreedySolver(Solver):
    """Try several greedy rule variants and keep the best feasible result."""

    name = "multi_rule_greedy"

    def solve(self, instance: Instance, config: SolverConfig):
        """Run multiple deterministic baselines and return the best feasible one."""

        if instance.variant is ProblemVariant.RETRIEVAL_CRP:
            return GreedyRetrievalSolver().solve(instance, config)
        rules: List[Callable[[Instance, State, SolverConfig], Optional[Move]]] = [
            self._safe_best_fit,
            self._safe_worst_fit,
            self._min_bad_count,
            self._min_inversion,
            self._protect_clean,
        ]
        best = None
        start_all = time.perf_counter()
        for idx, rule in enumerate(rules):
            sol = self._run_rule(instance, config, rule, f"{self.name}_{idx}")
            if sol.feasible and (best is None or _solution_rank(instance, sol) < _solution_rank(instance, best)):
                best = sol
        if best is not None:
            best.solver_name = self.name
            return best
        return failure_solution(self.name, start_all, "NO_FEASIBLE_FOUND")

    def _run_rule(self, instance: Instance, config: SolverConfig, rule, name: str):
        start = time.perf_counter()
        state = create_initial_state(instance)
        seen = {hash_state(instance, state)}
        while not is_goal(instance, state) and state.move_count < config.max_depth:
            mv = rule(instance, state, config)
            if mv is None:
                break
            state = apply_move(instance, state, mv)
            if not cpmp_business_feasible(instance, state, config.entropy_alpha):
                break
            h = hash_state(instance, state)
            if h in seen:
                break
            seen.add(h)
        if is_goal(instance, state):
            return solution_from_state(name, instance, state, start)
        return failure_solution(name, start, "NO_FEASIBLE_FOUND")

    @staticmethod
    def _source(state: State) -> Optional[int]:
        dirty = _dirty_stacks(state)
        return None if not dirty else max(dirty, key=lambda i: len(state.stacks[i]) - well_prefix_len(state.stacks[i]))

    def _destinations(self, instance: Instance, state: State, source: int) -> List[int]:
        return [q for q, st in enumerate(state.stacks) if q != source and len(st) < instance.max_height]

    def _move(self, state: State, source: int, dest: int) -> Move:
        return Move(state.move_count, state.stacks[source][-1], source, dest, len(state.stacks[source]), len(state.stacks[dest]) + 1)

    def _safe_best_fit(self, instance: Instance, state: State, config: SolverConfig) -> Optional[Move]:
        source = self._source(state)
        if source is None:
            return None
        box = state.stacks[source][-1]
        safe = [
            q
            for q in self._destinations(instance, state, source)
            if (not state.stacks[q] or state.stacks[q][-1] > box)
            and cpmp_business_feasible(instance, apply_move(instance, state, self._move(state, source, q)), config.entropy_alpha)
        ]
        if not safe:
            return self._min_bad_count(instance, state, config)
        return self._move(state, source, min(safe, key=lambda q: state.stacks[q][-1] if state.stacks[q] else float("inf")))

    def _safe_worst_fit(self, instance: Instance, state: State, config: SolverConfig) -> Optional[Move]:
        source = self._source(state)
        if source is None:
            return None
        box = state.stacks[source][-1]
        safe = [
            q
            for q in self._destinations(instance, state, source)
            if (not state.stacks[q] or state.stacks[q][-1] > box)
            and cpmp_business_feasible(instance, apply_move(instance, state, self._move(state, source, q)), config.entropy_alpha)
        ]
        if not safe:
            return self._min_bad_count(instance, state, config)
        return self._move(state, source, max(safe, key=lambda q: state.stacks[q][-1] if state.stacks[q] else float("inf")))

    def _min_bad_count(self, instance: Instance, state: State, config: SolverConfig) -> Optional[Move]:
        source = self._source(state)
        if source is None:
            return None
        best = None
        best_val = float("inf")
        for q in self._destinations(instance, state, source):
            mv = self._move(state, source, q)
            ns = apply_move(instance, state, mv)
            if not cpmp_business_feasible(instance, ns, config.entropy_alpha):
                continue
            val = bad_count(instance, ns)
            if val < best_val:
                best = mv
                best_val = val
        return best

    def _min_inversion(self, instance: Instance, state: State, config: SolverConfig) -> Optional[Move]:
        source = self._source(state)
        if source is None:
            return None
        best = None
        best_val = float("inf")
        for q in self._destinations(instance, state, source):
            mv = self._move(state, source, q)
            ns = apply_move(instance, state, mv)
            if not cpmp_business_feasible(instance, ns, config.entropy_alpha):
                continue
            val = inversion_count(instance, ns)
            if val < best_val:
                best = mv
                best_val = val
        return best

    def _protect_clean(self, instance: Instance, state: State, config: SolverConfig) -> Optional[Move]:
        dirty = _dirty_stacks(state)
        if not dirty:
            return None
        source = max(dirty, key=lambda i: len(state.stacks[i]) - well_prefix_len(state.stacks[i]))
        return self._safe_best_fit(instance, state, config)


def _solution_rank(instance: Instance, solution) -> tuple[float, float]:
    if not solution.moves:
        return (float("inf"), float("inf"))
    state = create_initial_state(instance)
    for mv in solution.moves:
        state = apply_move(instance, state, mv)
    return (dominance_value(instance, state), float(len(solution.moves)))
