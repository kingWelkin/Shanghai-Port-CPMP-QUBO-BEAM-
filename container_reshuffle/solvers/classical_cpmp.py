"""Classical non-neural CPMP baseline solvers.

The implementations in this module are designed as fair, same-model baselines:
they only use legal top-to-top CPMP moves from the shared state engine.  They
are inspired by the traditional CPMP literature families discussed in the
project docs: LPFH/multi-rule constructive heuristics, target-guided beam
search, and BRKGA-style decoders.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from math import inf
from typing import Callable, Iterable

from ..core import (
    Instance,
    Move,
    ProblemVariant,
    Solution,
    State,
    apply_move,
    create_initial_state,
    future_disorder_count_stacks,
    generate_legal_moves,
    hash_state,
    is_goal,
    validate_solution,
)
from ..features import bad_count, dirty_stack_count, disorder_box_count, inversion_count, settled_box_count, well_prefix_len
from ..scoring import cpmp_business_feasible, score_cpmp
from .base import Solver, SolverConfig, failure_solution, not_applicable, solution_from_state


def _require_cpmp(instance: Instance, name: str):
    if instance.variant is not ProblemVariant.PREMARSHALLING_CPMP:
        return not_applicable(name)
    return None


def _dirty_stacks(state: State) -> list[int]:
    return [idx for idx, stack in enumerate(state.stacks) if well_prefix_len(stack) < len(stack)]


def _legal_next_states(instance: Instance, state: State, config: SolverConfig) -> list[tuple[Move, State]]:
    next_states: list[tuple[Move, State]] = []
    for move in generate_legal_moves(instance, state):
        try:
            new_state = apply_move(instance, state, move)
        except ValueError:
            continue
        if cpmp_business_feasible(instance, new_state, config.entropy_alpha):
            next_states.append((move, new_state))
    return next_states


def _state_rank(instance: Instance, state: State, targets: list[list[list[int]]] | None = None) -> tuple:
    target_gap = min((_target_gap(state.stacks, target) for target in targets), default=0) if targets else 0
    n_value = disorder_box_count(instance, state)
    g_value = state.move_count + n_value
    return (
        n_value,
        -g_value,
        target_gap,
        inversion_count(instance, state),
        dirty_stack_count(instance, state),
        state.move_count,
        -settled_box_count(instance, state),
        state.stacks,
    )


def _target_guided_rank(instance: Instance, state: State, config: SolverConfig, targets: list[list[list[int]]]) -> tuple:
    target_gap = min((_target_gap(state.stacks, target) for target in targets), default=0)
    return (
        score_cpmp(instance, state, config.score_weights) + 50.0 * target_gap,
        target_gap,
        len(state.moves),
        state.stacks,
    )


def _move_to_empty(move: Move, state: State) -> int:
    return 1 if len(state.stacks[move.to_stack]) == 0 else 0


def _is_safe_destination(box: int, dst: list[int]) -> bool:
    return not dst or dst[-1] > box


def _lpfh_source_score(state: State, source: int) -> tuple:
    stack = state.stacks[source]
    dirty_depth = len(stack) - well_prefix_len(stack)
    top = stack[-1]
    return (-top, -dirty_depth, -len(stack), source)


def _target_layouts(instance: Instance) -> list[list[list[int]]]:
    boxes = sorted((box for stack in instance.stacks for box in stack), reverse=True)
    heights_seen: set[tuple[int, ...]] = set()
    height_patterns: list[list[int]] = []

    current = sorted((len(stack) for stack in instance.stacks), reverse=True)
    height_patterns.append(current)

    compact: list[int] = []
    remaining = instance.container_count
    for _ in range(instance.stack_count):
        take = min(instance.max_height, remaining)
        compact.append(take)
        remaining -= take
    height_patterns.append(compact)

    balanced = [instance.container_count // instance.stack_count] * instance.stack_count
    for idx in range(instance.container_count % instance.stack_count):
        balanced[idx] += 1
    height_patterns.append(sorted(balanced, reverse=True))

    layouts: list[list[list[int]]] = []
    for heights in height_patterns:
        normalized = tuple(sorted((height for height in heights if height > 0), reverse=True))
        padded = normalized + (0,) * (instance.stack_count - len(normalized))
        if padded in heights_seen:
            continue
        heights_seen.add(padded)
        cursor = 0
        layout: list[list[int]] = []
        for height in padded:
            layout.append(boxes[cursor : cursor + height])
            cursor += height
        layouts.append(layout)
    return layouts


def _target_gap(stacks: list[list[int]], target: list[list[int]]) -> int:
    used: set[int] = set()
    matched = 0
    stack_scores: list[tuple[int, int, int]] = []
    for src_idx, stack in enumerate(stacks):
        for tgt_idx, target_stack in enumerate(target):
            limit = min(len(stack), len(target_stack))
            prefix = 0
            while prefix < limit and stack[prefix] == target_stack[prefix]:
                prefix += 1
            stack_scores.append((-prefix, src_idx, tgt_idx))
    for neg_prefix, src_idx, tgt_idx in sorted(stack_scores):
        if src_idx in used or tgt_idx in used:
            continue
        used.add(src_idx)
        matched += -neg_prefix
    return sum(len(stack) for stack in stacks) - matched


def _wang_clean_stack(stack: list[int]) -> bool:
    return all(stack[idx] >= stack[idx + 1] for idx in range(len(stack) - 1))


def _wang_goal_stacks(stacks: list[list[int]]) -> bool:
    return all(_wang_clean_stack(stack) for stack in stacks)


def _wang_validate_shape(instance: Instance) -> None:
    if instance.stack_count <= 0:
        raise ValueError("stack_count must be positive")
    if instance.max_height <= 0:
        raise ValueError("max_height must be positive")
    if len(instance.stacks) != instance.stack_count:
        raise ValueError("len(stacks) must equal stack_count")
    count = 0
    for idx, stack in enumerate(instance.stacks):
        if len(stack) > instance.max_height:
            raise ValueError(f"stack {idx} exceeds max_height")
        for box in stack:
            if int(box) <= 0:
                raise ValueError("group labels must be positive")
        count += len(stack)
    if count != instance.container_count:
        raise ValueError(f"expected {instance.container_count} containers, got {count}")


def _wang_initial_state(instance: Instance) -> State:
    stacks = [list(stack) for stack in instance.stacks]
    disorder = future_disorder_count_stacks(stacks)
    return State(
        stacks=stacks,
        move_count=0,
        moves=[],
        disorder_current=disorder,
        disorder_total=disorder,
        entropy_max=disorder,
    )


def _wang_mark_fixed_prefix_for_group(stacks: list[list[int]], fixed: list[int], group: int) -> bool:
    changed = False
    for idx, stack in enumerate(stacks):
        while fixed[idx] < len(stack) and stack[fixed[idx]] == group:
            fixed[idx] += 1
            changed = True
    return changed


def _wang_pair_cost(stacks: list[list[int]], fixed: list[int], source: int, level: int, target: int) -> int:
    blockers = len(stacks[source]) - level - 1
    if source == target:
        return blockers
    target_unfixed = max(0, len(stacks[target]) - fixed[target])
    return target_unfixed + blockers


@dataclass(frozen=True)
class _WangPair:
    source: int
    level: int
    target: int
    box: int
    cost: int
    target_fixed: int
    target_unfixed: int
    blockers: int


@dataclass(frozen=True)
class _WangNode:
    state: State
    fixed: tuple[int, ...]
    group: int


def _wang_partial_path_key(node: _WangNode) -> tuple:
    return (
        future_disorder_count_stacks(node.state.stacks),
        node.state.move_count,
        tuple(tuple(stack) for stack in node.state.stacks),
    )


def _wang_max_group(stacks: list[list[int]]) -> int:
    return max((int(box) for stack in stacks for box in stack), default=0)


def _wang_target_can_accept(stacks: list[list[int]], fixed: list[int], target: int, group: int, max_height: int) -> bool:
    if fixed[target] >= max_height:
        return False
    if fixed[target] == 0:
        return True
    return stacks[target][fixed[target] - 1] >= group


def _wang_candidate_pairs(stacks: list[list[int]], fixed: list[int], group: int, max_height: int) -> list[_WangPair]:
    pairs: list[_WangPair] = []
    for source, stack in enumerate(stacks):
        for level in range(fixed[source], len(stack)):
            box = int(stack[level])
            if box != group:
                continue
            blockers = len(stack) - level - 1
            for target in range(len(stacks)):
                if not _wang_target_can_accept(stacks, fixed, target, group, max_height):
                    continue
                target_unfixed = max(0, len(stacks[target]) - fixed[target])
                cost = _wang_pair_cost(stacks, fixed, source, level, target)
                pairs.append(
                    _WangPair(
                        source=source,
                        level=level,
                        target=target,
                        box=box,
                        cost=cost,
                        target_fixed=fixed[target],
                        target_unfixed=target_unfixed,
                        blockers=blockers,
                    )
                )
    pairs.sort(key=lambda pair: (pair.cost, pair.target_fixed, pair.target_unfixed, pair.blockers, pair.source, pair.level, pair.target))
    return pairs


def _wang_node_key(node: _WangNode) -> tuple:
    return tuple(tuple(stack) for stack in node.state.stacks), node.fixed, node.group


def _wang_choose_relocation_destination(
    stacks: list[list[int]],
    fixed: list[int],
    box: int,
    forbidden: set[int],
    max_height: int,
) -> int | None:
    candidates = [idx for idx, stack in enumerate(stacks) if idx not in forbidden and len(stack) < max_height]
    if not candidates:
        return None

    clean_safe = [
        idx
        for idx in candidates
        if len(stacks[idx]) == fixed[idx] and (not stacks[idx] or stacks[idx][-1] >= box)
    ]
    if clean_safe:
        return min(clean_safe, key=lambda idx: (stacks[idx][-1] if stacks[idx] else inf, idx))

    dirty_safe: list[int] = []
    dirty_unsafe: list[int] = []
    for idx in candidates:
        dirty_part = stacks[idx][fixed[idx] :]
        if not dirty_part:
            continue
        largest_dirty = max(dirty_part)
        if largest_dirty <= box:
            dirty_safe.append(idx)
        else:
            dirty_unsafe.append(idx)
    if dirty_safe:
        return max(dirty_safe, key=lambda idx: (max(stacks[idx][fixed[idx] :]), -idx))
    if dirty_unsafe:
        return min(dirty_unsafe, key=lambda idx: (max(stacks[idx][fixed[idx] :]), idx))

    return max(candidates, key=lambda idx: (stacks[idx][-1] if stacks[idx] else -inf, -idx))


def _wang_move_top(instance: Instance, state: State, source: int, target: int) -> State:
    stack = state.stacks[source]
    move = Move(
        step=state.move_count,
        box=int(stack[-1]),
        from_stack=source,
        to_stack=target,
        from_level=len(stack),
        to_level=len(state.stacks[target]) + 1,
    )
    return apply_move(instance, state, move)


def _wang_clear_stack_to_level(
    instance: Instance,
    state: State,
    fixed: list[int],
    stack_idx: int,
    target_len: int,
    forbidden: set[int],
) -> State | None:
    while len(state.stacks[stack_idx]) > target_len:
        if len(state.stacks[stack_idx]) <= fixed[stack_idx]:
            return None
        box = int(state.stacks[stack_idx][-1])
        dst = _wang_choose_relocation_destination(state.stacks, fixed, box, forbidden | {stack_idx}, instance.max_height)
        if dst is None:
            return None
        state = _wang_move_top(instance, state, stack_idx, dst)
    return state


def _wang_prepare_node(instance: Instance, node: _WangNode) -> _WangNode:
    fixed = list(node.fixed)
    group = int(node.group)
    while group > 0:
        while _wang_mark_fixed_prefix_for_group(node.state.stacks, fixed, group):
            pass
        if all(fixed[idx] == len(stack) for idx, stack in enumerate(node.state.stacks)):
            return _WangNode(node.state, tuple(fixed), group)
        if _wang_candidate_pairs(node.state.stacks, fixed, group, instance.max_height):
            return _WangNode(node.state, tuple(fixed), group)
        group -= 1
    return _WangNode(node.state, tuple(fixed), group)


def _wang_apply_giant_move(instance: Instance, node: _WangNode, pair: _WangPair) -> _WangNode | None:
    state = node.state
    fixed = list(node.fixed)
    if pair.source >= len(state.stacks) or pair.target >= len(state.stacks):
        return None
    if pair.level >= len(state.stacks[pair.source]) or state.stacks[pair.source][pair.level] != pair.box:
        return None

    if pair.source == pair.target:
        state = _wang_clear_stack_to_level(instance, state, fixed, pair.source, pair.level + 1, {pair.source})
        if state is None:
            return None
        if pair.level > fixed[pair.source]:
            temp = _wang_choose_relocation_destination(state.stacks, fixed, pair.box, {pair.source}, instance.max_height)
            if temp is None:
                return None
            state = _wang_move_top(instance, state, pair.source, temp)
            state = _wang_clear_stack_to_level(instance, state, fixed, pair.source, fixed[pair.source], {pair.source, temp})
            if state is None or not state.stacks[temp] or state.stacks[temp][-1] != pair.box:
                return None
            state = _wang_move_top(instance, state, temp, pair.source)
        if len(state.stacks[pair.source]) != fixed[pair.source] + 1 or state.stacks[pair.source][-1] != pair.box:
            return None
        fixed[pair.source] += 1
        return _wang_prepare_node(instance, _WangNode(state, tuple(fixed), node.group))

    state = _wang_clear_stack_to_level(instance, state, fixed, pair.target, fixed[pair.target], {pair.target, pair.source})
    if state is None:
        state = node.state
        state = _wang_clear_stack_to_level(instance, state, fixed, pair.target, fixed[pair.target], {pair.target})
    if state is None:
        return None

    state = _wang_clear_stack_to_level(instance, state, fixed, pair.source, pair.level + 1, {pair.source, pair.target})
    if state is None:
        return None
    if not state.stacks[pair.source] or state.stacks[pair.source][-1] != pair.box:
        return None
    if len(state.stacks[pair.target]) != fixed[pair.target]:
        return None
    state = _wang_move_top(instance, state, pair.source, pair.target)
    fixed[pair.target] += 1
    return _wang_prepare_node(instance, _WangNode(state, tuple(fixed), node.group))


def _wang_node_rank(instance: Instance, node: _WangNode) -> tuple:
    fixed_total = sum(node.fixed)
    n_value = future_disorder_count_stacks(node.state.stacks)
    return (
        0 if _wang_goal_stacks(node.state.stacks) else 1,
        -fixed_total,
        n_value,
        node.state.move_count,
        node.group,
        node.state.stacks,
    )


def _wang_first_interim_from_giant_child(instance: Instance, base: _WangNode, child: _WangNode) -> _WangNode:
    base_len = len(base.state.moves)
    if len(child.state.moves) <= base_len + 1:
        return child
    first_move = child.state.moves[base_len]
    interim_state = apply_move(instance, base.state, first_move)
    return _WangNode(interim_state, base.fixed, base.group)


def _wang_solution_from_node(
    name: str,
    instance: Instance,
    node: _WangNode,
    start: float,
    expanded: int,
    generated: int,
    visited: int,
    status: str,
) -> Solution:
    sol = Solution(
        solver_name=name,
        feasible=_wang_goal_stacks(node.state.stacks),
        optimal=False,
        objective=float(len(node.state.moves)) if _wang_goal_stacks(node.state.stacks) else inf,
        runtime_sec=time.perf_counter() - start,
        moves=node.state.moves,
        status=status,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=visited,
    )
    values = [box for stack in instance.stacks for box in stack]
    if sorted(values) == list(range(1, instance.container_count + 1)):
        ok, msg, _ = validate_solution(instance, sol)
        if not ok:
            sol.feasible = False
            sol.objective = inf
            sol.status = f"VALIDATION_FAILED: {msg}"
    elif not sol.feasible:
        sol.status = f"VALIDATION_FAILED: final state is not a priority-group goal"
    return sol


class LpfhMultiHeuristicSolver(Solver):
    """Deterministic LPFH-style multi-rule constructive CPMP baseline."""

    name = "lpfh_multi_heuristic"

    def solve(self, instance: Instance, config: SolverConfig):
        not_cpmp = _require_cpmp(instance, self.name)
        if not_cpmp is not None:
            return not_cpmp
        start_all = time.perf_counter()
        rules: list[Callable[[Instance, State, SolverConfig], Move | None]] = [
            self._safe_best_fit,
            self._safe_worst_fit,
            self._min_disorder,
            self._min_inversion,
            self._lookahead,
        ]
        best = None
        for idx, rule in enumerate(rules):
            if time.perf_counter() - start_all > config.time_limit_sec:
                break
            sol = self._run_rule(instance, config, rule, f"{self.name}_{idx}")
            if sol.feasible and (best is None or len(sol.moves) < len(best.moves)):
                best = sol
        if best is not None:
            best.solver_name = self.name
            best.runtime_sec = time.perf_counter() - start_all
            return best
        return failure_solution(self.name, start_all, "NO_FEASIBLE_FOUND")

    def _run_rule(self, instance: Instance, config: SolverConfig, rule, name: str):
        start = time.perf_counter()
        state = create_initial_state(instance)
        seen = {hash_state(instance, state)}
        expanded = generated = 0
        while not is_goal(instance, state) and state.move_count < config.max_depth:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expanded += 1
            move = rule(instance, state, config)
            if move is None:
                break
            state = apply_move(instance, state, move)
            generated += 1
            key = hash_state(instance, state)
            if key in seen:
                break
            seen.add(key)
        if is_goal(instance, state):
            return solution_from_state(
                name,
                instance,
                state,
                start,
                expanded_states=expanded,
                generated_states=generated,
                visited_states=len(seen),
            )
        return failure_solution(name, start, "NO_FEASIBLE_FOUND", expanded_states=expanded, generated_states=generated, visited_states=len(seen))

    def _candidate_sources(self, state: State) -> list[int]:
        dirty = _dirty_stacks(state)
        pool = dirty if dirty else [idx for idx, stack in enumerate(state.stacks) if stack]
        return sorted(pool, key=lambda source: _lpfh_source_score(state, source))

    def _candidate_moves(self, instance: Instance, state: State, config: SolverConfig) -> list[tuple[Move, State]]:
        source_order = {source: rank for rank, source in enumerate(self._candidate_sources(state))}
        pairs = _legal_next_states(instance, state, config)
        pairs.sort(key=lambda item: (source_order.get(item[0].from_stack, 10**6), item[0].to_stack))
        return pairs

    def _safe_best_fit(self, instance: Instance, state: State, config: SolverConfig) -> Move | None:
        pairs = self._candidate_moves(instance, state, config)
        safe = [
            (move, new_state)
            for move, new_state in pairs
            if _is_safe_destination(move.box, state.stacks[move.to_stack])
        ]
        pool = safe or pairs
        if not pool:
            return None
        return min(
            pool,
            key=lambda item: (
                _state_rank(instance, item[1]),
                state.stacks[item[0].to_stack][-1] if state.stacks[item[0].to_stack] else inf,
                -item[0].box,
            ),
        )[0]

    def _safe_worst_fit(self, instance: Instance, state: State, config: SolverConfig) -> Move | None:
        pairs = self._candidate_moves(instance, state, config)
        safe = [
            (move, new_state)
            for move, new_state in pairs
            if _is_safe_destination(move.box, state.stacks[move.to_stack])
        ]
        pool = safe or pairs
        if not pool:
            return None
        return min(
            pool,
            key=lambda item: (
                _state_rank(instance, item[1]),
                -(state.stacks[item[0].to_stack][-1] if state.stacks[item[0].to_stack] else inf),
                -item[0].box,
            ),
        )[0]

    def _min_disorder(self, instance: Instance, state: State, config: SolverConfig) -> Move | None:
        pairs = self._candidate_moves(instance, state, config)
        if not pairs:
            return None
        return min(pairs, key=lambda item: (_state_rank(instance, item[1]), -item[0].box))[0]

    def _min_inversion(self, instance: Instance, state: State, config: SolverConfig) -> Move | None:
        pairs = self._candidate_moves(instance, state, config)
        if not pairs:
            return None
        return min(
            pairs,
            key=lambda item: (
                inversion_count(instance, item[1]),
                disorder_box_count(instance, item[1]),
                dirty_stack_count(instance, item[1]),
                -item[0].box,
            ),
        )[0]

    def _lookahead(self, instance: Instance, state: State, config: SolverConfig) -> Move | None:
        pairs = self._candidate_moves(instance, state, config)
        if not pairs:
            return None
        scored = []
        for move, new_state in pairs:
            children = _legal_next_states(instance, new_state, config)
            child_rank = min((_state_rank(instance, child) for _child_move, child in children), default=_state_rank(instance, new_state))
            scored.append((child_rank, _state_rank(instance, new_state), -move.box, move))
        return min(scored)[-1]


class TargetGuidedBeamSolver(Solver):
    """Beam search guided by deterministic target-layout distance."""

    name = "target_guided_beam"

    def solve(self, instance: Instance, config: SolverConfig):
        not_cpmp = _require_cpmp(instance, self.name)
        if not_cpmp is not None:
            return not_cpmp
        start = time.perf_counter()
        initial = create_initial_state(instance)
        if is_goal(instance, initial):
            return solution_from_state(self.name, instance, initial, start)

        targets = _target_layouts(instance)
        beam = [initial]
        visited = {hash_state(instance, initial)}
        expanded = generated = 0
        best_goal: State | None = None

        for _depth in range(config.max_depth):
            if time.perf_counter() - start > config.time_limit_sec:
                break
            candidates: list[State] = []
            for state in beam:
                expanded += 1
                for _move, new_state in _legal_next_states(instance, state, config):
                    generated += 1
                    key = hash_state(instance, new_state)
                    if config.enable_visited_pruning and key in visited:
                        continue
                    visited.add(key)
                    if is_goal(instance, new_state):
                        if best_goal is None or len(new_state.moves) < len(best_goal.moves):
                            best_goal = new_state
                    candidates.append(new_state)
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
            candidates.sort(key=lambda state: _target_guided_rank(instance, state, config, targets))
            if config.max_candidates_per_layer:
                candidates = candidates[: config.max_candidates_per_layer]
            beam = candidates[: max(1, config.beam_width)]

        return failure_solution(
            self.name,
            start,
            "NO_FEASIBLE_FOUND",
            expanded_states=expanded,
            generated_states=generated,
            visited_states=len(visited),
        )


class WangBsGiantMovesSolver(Solver):
    """Wang-Jin-Lim BS-G reproduction using target-guided giant moves.

    This solver keeps the paper's fixed-prefix view: group labels are processed
    from large to small, a giant move fixes one target container into the lowest
    unfixed slot of a target stack, and beam search keeps the best intermediate
    layouts.  It intentionally stays separate from the project's QUBO beam.
    """

    name = "wang_bs_g"

    def solve(self, instance: Instance, config: SolverConfig):
        not_cpmp = _require_cpmp(instance, self.name)
        if not_cpmp is not None:
            return not_cpmp
        start = time.perf_counter()
        try:
            _wang_validate_shape(instance)
        except ValueError as exc:
            return failure_solution(self.name, start, f"INVALID_INSTANCE: {exc}")

        initial_state = _wang_initial_state(instance)
        initial = _wang_prepare_node(
            instance,
            _WangNode(initial_state, tuple(0 for _ in range(instance.stack_count)), _wang_max_group(initial_state.stacks)),
        )
        expanded = 0
        generated = 0
        best_depth: dict[tuple, int] = {_wang_node_key(initial): 0}
        beam = [initial]
        incumbent: _WangNode | None = initial if _wang_goal_stacks(initial.state.stacks) else None

        if incumbent is not None:
            return _wang_solution_from_node(
                self.name,
                instance,
                incumbent,
                start,
                expanded,
                generated,
                len(best_depth),
                "FEASIBLE; paper=Wang2015; algorithm=BS-G; giant_moves",
            )

        while beam and time.perf_counter() - start <= config.time_limit_sec:
            candidates: list[_WangNode] = []
            for node in beam:
                if node.state.move_count >= config.max_depth:
                    continue
                prepared = _wang_prepare_node(instance, node)
                if _wang_goal_stacks(prepared.state.stacks):
                    if incumbent is None or prepared.state.move_count < incumbent.state.move_count:
                        incumbent = prepared
                    continue
                pairs = _wang_candidate_pairs(prepared.state.stacks, list(prepared.fixed), prepared.group, instance.max_height)
                pair_limit = config.candidate_limit_per_state or config.max_candidates_per_layer or len(pairs)
                expanded += 1
                for pair in pairs[:pair_limit]:
                    child = _wang_apply_giant_move(instance, prepared, pair)
                    generated += 1
                    if child is None or child.state.move_count > config.max_depth:
                        continue
                    key = _wang_node_key(child)
                    previous_depth = best_depth.get(key)
                    if config.enable_visited_pruning and previous_depth is not None and previous_depth <= child.state.move_count:
                        continue
                    best_depth[key] = child.state.move_count
                    if _wang_goal_stacks(child.state.stacks):
                        if incumbent is None or child.state.move_count < incumbent.state.move_count:
                            incumbent = child
                        continue
                    candidates.append(child)
            if incumbent is not None:
                active_lower_bound = min((node.state.move_count for node in candidates), default=inf)
                if active_lower_bound >= incumbent.state.move_count:
                    return _wang_solution_from_node(
                        self.name,
                        instance,
                        incumbent,
                        start,
                        expanded,
                        generated,
                        len(best_depth),
                        "FEASIBLE; paper=Wang2015; algorithm=BS-G; giant_moves",
                    )
            if not candidates:
                break
            candidates.sort(key=lambda node: _wang_node_rank(instance, node))
            beam = candidates[: max(1, config.beam_width)]

        if incumbent is not None:
            return _wang_solution_from_node(
                self.name,
                instance,
                incumbent,
                start,
                expanded,
                generated,
                len(best_depth),
                "FEASIBLE; paper=Wang2015; algorithm=BS-G; giant_moves",
            )

        return failure_solution(
            self.name,
            start,
            "NO_FEASIBLE_FOUND; paper=Wang2015; algorithm=BS-G; giant_moves",
            expanded_states=expanded,
            generated_states=generated,
            visited_states=len(best_depth),
        )


class WangBsBabyMovesSolver(Solver):
    """Wang-Jin-Lim BS-B reproduction using baby-move beam successors."""

    name = "wang_bs_b"

    def solve(self, instance: Instance, config: SolverConfig):
        not_cpmp = _require_cpmp(instance, self.name)
        if not_cpmp is not None:
            return not_cpmp
        start = time.perf_counter()
        try:
            _wang_validate_shape(instance)
        except ValueError as exc:
            return failure_solution(self.name, start, f"INVALID_INSTANCE: {exc}")

        initial_state = _wang_initial_state(instance)
        initial = _wang_prepare_node(
            instance,
            _WangNode(initial_state, tuple(0 for _ in range(instance.stack_count)), _wang_max_group(initial_state.stacks)),
        )
        expanded = 0
        generated = 0
        best_depth: dict[tuple, int] = {_wang_node_key(initial): 0}
        beam = [initial]
        incumbent: _WangNode | None = initial if _wang_goal_stacks(initial.state.stacks) else None
        best_partial = initial

        def consider_partial(node: _WangNode) -> None:
            nonlocal best_partial
            if _wang_partial_path_key(node) < _wang_partial_path_key(best_partial):
                best_partial = node

        while beam and time.perf_counter() - start <= config.time_limit_sec:
            scored_candidates: list[tuple[tuple, _WangNode]] = []
            for node in beam:
                if node.state.move_count >= config.max_depth:
                    continue
                prepared = _wang_prepare_node(instance, node)
                consider_partial(prepared)
                if _wang_goal_stacks(prepared.state.stacks):
                    if incumbent is None or prepared.state.move_count < incumbent.state.move_count:
                        incumbent = prepared
                    continue
                pairs = _wang_candidate_pairs(prepared.state.stacks, list(prepared.fixed), prepared.group, instance.max_height)
                pair_limit = config.candidate_limit_per_state or config.max_candidates_per_layer or len(pairs)
                expanded += 1
                for pair in pairs[:pair_limit]:
                    giant_child = _wang_apply_giant_move(instance, prepared, pair)
                    generated += 1
                    if giant_child is None or giant_child.state.move_count > config.max_depth:
                        continue
                    consider_partial(giant_child)
                    if _wang_goal_stacks(giant_child.state.stacks):
                        if incumbent is None or giant_child.state.move_count < incumbent.state.move_count:
                            incumbent = giant_child
                    interim = _wang_first_interim_from_giant_child(instance, prepared, giant_child)
                    if interim.state.move_count > config.max_depth:
                        continue
                    consider_partial(interim)
                    key = _wang_node_key(interim)
                    previous_depth = best_depth.get(key)
                    if config.enable_visited_pruning and previous_depth is not None and previous_depth <= interim.state.move_count:
                        continue
                    best_depth[key] = interim.state.move_count
                    score = (_wang_node_rank(instance, giant_child), _wang_node_rank(instance, interim))
                    scored_candidates.append((score, interim))
            if incumbent is not None:
                active_lower_bound = min((node.state.move_count for _score, node in scored_candidates), default=inf)
                if active_lower_bound >= incumbent.state.move_count:
                    return _wang_solution_from_node(
                        self.name,
                        instance,
                        incumbent,
                        start,
                        expanded,
                        generated,
                        len(best_depth),
                        "FEASIBLE; paper=Wang2015; algorithm=BS-B; baby_moves",
                    )
            if not scored_candidates:
                break
            scored_candidates.sort(key=lambda item: item[0])
            beam = [node for _score, node in scored_candidates[: max(1, config.beam_width)]]

        if incumbent is not None:
            return _wang_solution_from_node(
                self.name,
                instance,
                incumbent,
                start,
                expanded,
                generated,
                len(best_depth),
                "FEASIBLE; paper=Wang2015; algorithm=BS-B; baby_moves",
            )

        return failure_solution(
            self.name,
            start,
            "NO_FEASIBLE_FOUND; paper=Wang2015; algorithm=BS-B; baby_moves",
            expanded_states=expanded,
            generated_states=generated,
            visited_states=len(best_depth),
            best_partial_moves=list(best_partial.state.moves),
        )


@dataclass(frozen=True)
class _Decoded:
    state: State
    expanded: int
    generated: int
    visited: int


class BrkgaCpmpSolver(Solver):
    """Biased random-key genetic algorithm with a legal CPMP decoder."""

    name = "brkga_cpmp"

    def __init__(
        self,
        population_size: int = 48,
        generations: int = 24,
        elite_fraction: float = 0.25,
        mutant_fraction: float = 0.15,
        inheritance_bias: float = 0.70,
    ) -> None:
        self.population_size = max(6, population_size)
        self.generations = max(1, generations)
        self.elite_fraction = min(0.8, max(0.1, elite_fraction))
        self.mutant_fraction = min(0.8, max(0.0, mutant_fraction))
        self.inheritance_bias = min(0.95, max(0.5, inheritance_bias))

    def solve(self, instance: Instance, config: SolverConfig):
        not_cpmp = _require_cpmp(instance, self.name)
        if not_cpmp is not None:
            return not_cpmp
        start = time.perf_counter()
        rng = random.Random(config.random_seed)
        population = self._initial_population(rng)
        best_decoded: _Decoded | None = None
        expanded = generated = visited = 0

        for _generation in range(self.generations):
            if time.perf_counter() - start > config.time_limit_sec:
                break
            evaluated = []
            for chromosome in population:
                decoded = self._decode(instance, config, chromosome, rng)
                expanded += decoded.expanded
                generated += decoded.generated
                visited += decoded.visited
                evaluated.append((self._fitness(instance, decoded.state), chromosome, decoded))
                if best_decoded is None or self._fitness(instance, decoded.state) < self._fitness(instance, best_decoded.state):
                    best_decoded = decoded
            evaluated.sort(key=lambda item: item[0])
            elite_count = max(1, int(round(self.population_size * self.elite_fraction)))
            mutant_count = int(round(self.population_size * self.mutant_fraction))
            elites = [chromosome for _fit, chromosome, _decoded in evaluated[:elite_count]]
            non_elites = [chromosome for _fit, chromosome, _decoded in evaluated[elite_count:]] or elites
            next_population = list(elites)
            while len(next_population) + mutant_count < self.population_size:
                elite = rng.choice(elites)
                other = rng.choice(non_elites)
                next_population.append(self._crossover(elite, other, rng))
            while len(next_population) < self.population_size:
                next_population.append(self._random_chromosome(rng))
            population = next_population

        if best_decoded is not None and is_goal(instance, best_decoded.state):
            return solution_from_state(
                self.name,
                instance,
                best_decoded.state,
                start,
                expanded_states=expanded,
                generated_states=generated,
                visited_states=visited,
            )
        return failure_solution(
            self.name,
            start,
            "NO_FEASIBLE_FOUND",
            expanded_states=expanded,
            generated_states=generated,
            visited_states=visited,
        )

    def _initial_population(self, rng: random.Random) -> list[tuple[float, ...]]:
        seeds = [
            (0.95, 0.70, 0.45, 0.35, 0.20, 0.10, 0.55, 0.05),
            (0.90, 0.30, 0.70, 0.50, 0.10, 0.15, 0.20, 0.05),
            (0.80, 0.50, 0.35, 0.85, 0.40, 0.10, 0.35, 0.05),
        ]
        population = list(seeds)
        while len(population) < self.population_size:
            population.append(self._random_chromosome(rng))
        return population

    @staticmethod
    def _random_chromosome(rng: random.Random) -> tuple[float, ...]:
        return tuple(rng.random() for _ in range(8))

    def _crossover(self, elite: tuple[float, ...], other: tuple[float, ...], rng: random.Random) -> tuple[float, ...]:
        return tuple(e if rng.random() < self.inheritance_bias else o for e, o in zip(elite, other))

    def _decode(self, instance: Instance, config: SolverConfig, chromosome: tuple[float, ...], rng: random.Random) -> _Decoded:
        state = create_initial_state(instance)
        targets = _target_layouts(instance)
        seen = {hash_state(instance, state)}
        expanded = generated = 0
        for _depth in range(config.max_depth):
            if is_goal(instance, state):
                break
            expanded += 1
            pairs = _legal_next_states(instance, state, config)
            generated += len(pairs)
            ranked = []
            for idx, (move, new_state) in enumerate(pairs):
                key = hash_state(instance, new_state)
                if config.enable_visited_pruning and key in seen:
                    continue
                ranked.append((self._move_score(instance, state, move, new_state, targets, chromosome, idx), move, new_state, key))
            if not ranked:
                break
            _score, _move, state, key = min(ranked)
            seen.add(key)
        return _Decoded(state=state, expanded=expanded, generated=generated, visited=len(seen))

    @staticmethod
    def _fitness(instance: Instance, state: State) -> tuple:
        return (
            0 if is_goal(instance, state) else 1,
            disorder_box_count(instance, state),
            inversion_count(instance, state),
            dirty_stack_count(instance, state),
            len(state.moves),
            state.stacks,
        )

    @staticmethod
    def _move_score(
        instance: Instance,
        prev_state: State,
        move: Move,
        new_state: State,
        targets: list[list[list[int]]],
        chromosome: tuple[float, ...],
        move_index: int,
    ) -> tuple[float, int]:
        box = move.box
        safe = 0 if _is_safe_destination(box, prev_state.stacks[move.to_stack]) else 1
        from_dirty = 0 if move.from_stack in _dirty_stacks(prev_state) else 1
        target_gap = min(_target_gap(new_state.stacks, target) for target in targets)
        values = (
            disorder_box_count(instance, new_state),
            inversion_count(instance, new_state),
            dirty_stack_count(instance, new_state),
            bad_count(instance, new_state),
            target_gap,
            _move_to_empty(move, prev_state),
            safe + from_dirty,
            move_index / max(1, instance.stack_count * (instance.stack_count - 1)),
        )
        weighted = sum((0.05 + weight) * value for weight, value in zip(chromosome, values))
        return weighted, move_index
