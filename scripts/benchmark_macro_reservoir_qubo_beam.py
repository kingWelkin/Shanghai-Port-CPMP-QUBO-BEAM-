"""Experimental macro-reservoir QUBO beam search.

This script tests a QUBO integration point that happens before beam truncation:

1. keep a source/destination/plateau action reservoir for each beam state,
2. expand each retained action into short macro paths,
3. let QUBO select a diverse set of macro paths,
4. commit only the first move of each selected macro into the next beam layer.

The algorithm is intentionally kept as an experiment script. It does not change
the production solver registry.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_full_real_business_algorithms import (  # noqa: E402
    TrajectoryResult,
    _instance_from_zone,
    _result_from_state,
)
from container_reshuffle.core import (  # noqa: E402
    Instance,
    Move,
    State,
    apply_move,
    create_initial_state,
    generate_legal_moves,
    hash_state,
)
from container_reshuffle.features import (  # noqa: E402
    bad_count,
    dirty_stack_count,
    disorder_box_count,
    entropy_limit,
    entropy_value,
    inversion_count,
    settled_box_count,
    well_prefix_gain,
)
from container_reshuffle.qubo.model import QuboModel  # noqa: E402
from container_reshuffle.qubo.sampler import QuboSamplerConfig, SimulatedAnnealingQuboSampler  # noqa: E402
from container_reshuffle.scoring import cpmp_business_feasible, dominance_value, score_state  # noqa: E402
from container_reshuffle.similarity import combined_similarity  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.beam import Candidate  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


@dataclass(frozen=True)
class ReservoirMove:
    """One legal first-step option retained before macro expansion."""

    move: Move
    state: State
    score: float
    tags: tuple[str, ...]


@dataclass(frozen=True)
class MacroCandidate:
    """A short legal path used as a QUBO variable."""

    moves: tuple[Move, ...]
    next_state: State
    commit_state: State
    leaf_state: State
    root_n: int
    first_n: int
    leaf_n: int
    first_hash: tuple
    commit_hash: tuple
    leaf_hash: tuple
    score: float

    @property
    def first_move(self) -> Move:
        return self.moves[0]


def _state_rank(instance: Instance, state: State) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        disorder_box_count(instance, state),
        state.entropy_max,
        state.disorder_total,
        state.move_count,
        inversion_count(instance, state),
        bad_count(instance, state),
        dirty_stack_count(instance, state),
        -settled_box_count(instance, state),
    )


def _macro_score(instance: Instance, root: State, macro: State, first: State, depth: int) -> float:
    """Lower-is-better macro-path score."""

    cap = entropy_limit(instance, 1.3)
    root_n = disorder_box_count(instance, root)
    first_n = disorder_box_count(instance, first)
    leaf_n = disorder_box_count(instance, macro)
    gain = root_n - leaf_n
    immediate_gain = root_n - first_n
    plateau_bonus = 1.0 if first_n == root_n and gain > 0 else 0.0
    low_slack_risk = max(0, macro.entropy_max - (cap - 2))
    return (
        1_000_000.0 * leaf_n
        + 35_000.0 * macro.entropy_max
        + 180_000.0 * low_slack_risk * low_slack_risk
        + 10.0 * macro.disorder_total
        + 40.0 * macro.move_count
        + 1.0 * inversion_count(instance, macro)
        + 2.0 * bad_count(instance, macro)
        + 8.0 * dirty_stack_count(instance, macro)
        - 50.0 * settled_box_count(instance, macro)
        - 5_000.0 * gain
        - 800.0 * max(0, immediate_gain)
        - 1_200.0 * plateau_bonus
        + 20.0 * depth
    )


def _move_key(move: Move) -> tuple[int, int, int, int | None, int | None]:
    return (move.box, move.from_stack, move.to_stack, move.from_level, move.to_level)


def _is_immediate_undo(previous: Move | None, move: Move) -> bool:
    if previous is None:
        return False
    return (
        previous.box == move.box
        and previous.from_stack == move.to_stack
        and previous.to_stack == move.from_stack
    )


def _move_sort_key(instance: Instance, current: State, nxt: State, score: float) -> tuple:
    return (
        disorder_box_count(instance, nxt),
        nxt.entropy_max,
        nxt.disorder_total,
        -well_prefix_gain(instance, current, nxt),
        inversion_count(instance, nxt),
        bad_count(instance, nxt),
        dirty_stack_count(instance, nxt),
        -settled_box_count(instance, nxt),
        score,
    )


def build_reservoir_moves(
    instance: Instance,
    state: State,
    config: SolverConfig,
    *,
    global_keep: int,
    source_keep: int,
    dest_keep: int,
    plateau_keep: int,
    max_reservoir: int,
    previous_move: Move | None = None,
) -> list[ReservoirMove]:
    """Keep globally strong moves plus structured representatives."""

    evaluated: list[tuple[Move, State, float]] = []
    n_current = disorder_box_count(instance, state)
    for move in generate_legal_moves(instance, state):
        if _is_immediate_undo(previous_move, move):
            continue
        try:
            nxt = apply_move(instance, state, move)
        except ValueError:
            continue
        if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
            continue
        evaluated.append((move, nxt, score_state(instance, nxt, config.score_weights)))

    selected: dict[tuple, ReservoirMove] = {}

    def add(rows: Iterable[tuple[Move, State, float]], tag: str) -> None:
        for move, nxt, score in rows:
            key = _move_key(move)
            old = selected.get(key)
            tags = (tag,) if old is None else tuple(sorted(set(old.tags + (tag,))))
            selected[key] = ReservoirMove(move, nxt, score, tags)

    ordered = sorted(evaluated, key=lambda row: _move_sort_key(instance, state, row[1], row[2]))
    add(ordered[:global_keep], "global")

    for src in sorted({move.from_stack for move, _nxt, _score in evaluated}):
        bucket = [row for row in evaluated if row[0].from_stack == src]
        bucket.sort(key=lambda row: _move_sort_key(instance, state, row[1], row[2]))
        add(bucket[:source_keep], "source")

    for dst in sorted({move.to_stack for move, _nxt, _score in evaluated}):
        bucket = [row for row in evaluated if row[0].to_stack == dst]
        bucket.sort(key=lambda row: _move_sort_key(instance, state, row[1], row[2]))
        add(bucket[:dest_keep], "dest")

    plateau = [
        row
        for row in evaluated
        if disorder_box_count(instance, row[1]) <= n_current
    ]
    plateau.sort(
        key=lambda row: (
            row[1].entropy_max,
            row[1].disorder_total,
            row[2],
        )
    )
    add(plateau[:plateau_keep], "plateau")

    # Releasing a short stack can create a reusable staging buffer. Keeping
    # multiple destination variants here is important because the downstream
    # route may depend on where the released top box is parked.
    buffer_release = [
        row
        for row in evaluated
        if len(state.stacks[row[0].from_stack]) <= 2
        and disorder_box_count(instance, row[1]) <= n_current
    ]
    buffer_release.sort(
        key=lambda row: (
            len(state.stacks[row[0].from_stack]),
            -len(state.stacks[row[0].to_stack]),
            row[1].entropy_max,
            row[1].disorder_total,
            _move_sort_key(instance, state, row[1], row[2]),
        )
    )
    add(buffer_release, "buffer_release")

    protected: list[ReservoirMove] = []
    protected_keys: set[tuple] = set()
    for move, _nxt, _score in buffer_release[: max(4, max_reservoir // 3)]:
        key = _move_key(move)
        item = selected.get(key)
        if item is not None:
            protected.append(item)
            protected_keys.add(key)

    ranked = list(selected.values())
    ranked.sort(key=lambda item: _move_sort_key(instance, state, item.state, item.score))
    out_by_key: dict[tuple, ReservoirMove] = {}
    for item in protected + ranked:
        out_by_key.setdefault(_move_key(item.move), item)
        if len(out_by_key) >= max_reservoir:
            break
    out = list(out_by_key.values())
    out.sort(key=lambda item: _move_sort_key(instance, state, item.state, item.score))
    return out


def build_macro_candidates(
    instance: Instance,
    root: State,
    config: SolverConfig,
    *,
    macro_depth: int,
    commit_length: int,
    global_keep: int,
    source_keep: int,
    dest_keep: int,
    plateau_keep: int,
    max_reservoir: int,
    max_macros_per_state: int,
) -> list[MacroCandidate]:
    """Expand a structured reservoir into short legal macro paths."""

    frontier: list[tuple[tuple[Move, ...], State, State, State]] = []
    first_moves = build_reservoir_moves(
        instance,
        root,
        config,
        global_keep=global_keep,
        source_keep=source_keep,
        dest_keep=dest_keep,
        plateau_keep=plateau_keep,
        max_reservoir=max_reservoir,
    )
    for item in first_moves:
        frontier.append(((item.move,), item.state, item.state, item.state))

    completed: list[MacroCandidate] = []
    for depth in range(1, max(1, macro_depth) + 1):
        next_frontier: list[tuple[tuple[Move, ...], State, State, State]] = []
        for moves, first_state, commit_state, current in frontier:
            if len(moves) <= max(1, commit_length):
                commit_state = current
            completed.append(
                MacroCandidate(
                    moves=moves,
                    next_state=first_state,
                    commit_state=commit_state,
                    leaf_state=current,
                    root_n=disorder_box_count(instance, root),
                    first_n=disorder_box_count(instance, first_state),
                    leaf_n=disorder_box_count(instance, current),
                    first_hash=hash_state(instance, first_state, config.enable_canonical_hash),
                    commit_hash=hash_state(instance, commit_state, config.enable_canonical_hash),
                    leaf_hash=hash_state(instance, current, config.enable_canonical_hash),
                    score=_macro_score(instance, root, current, first_state, len(moves)),
                )
            )
            if depth >= macro_depth:
                continue
            child_moves = build_reservoir_moves(
                instance,
                current,
                config,
                global_keep=max(1, global_keep - 1),
                source_keep=source_keep,
                dest_keep=max(0, dest_keep - 1),
                plateau_keep=plateau_keep,
                max_reservoir=max(4, max_reservoir // 2),
                previous_move=moves[-1],
            )
            for item in child_moves:
                next_commit = item.state if len(moves) + 1 <= max(1, commit_length) else commit_state
                next_frontier.append((moves + (item.move,), first_state, next_commit, item.state))

        next_frontier.sort(
            key=lambda row: (
                _macro_score(instance, root, row[3], row[1], len(row[0])),
                _state_rank(instance, row[3]),
            )
        )
        frontier = next_frontier[: max(1, max_macros_per_state * 2)]

    best_by_leaf: dict[tuple, MacroCandidate] = {}
    for macro in completed:
        old = best_by_leaf.get(macro.leaf_hash)
        if old is None or macro.score < old.score:
            best_by_leaf[macro.leaf_hash] = macro
    macros = sorted(best_by_leaf.values(), key=lambda m: (m.score, _state_rank(instance, m.leaf_state)))
    return macros[:max_macros_per_state]


def _select_usable_macros(
    macros: list[MacroCandidate],
    *,
    max_qubo_vars: int,
    elite_keep: int,
    source_keep: int,
    dest_keep: int,
    plateau_keep: int,
    instance: Instance,
) -> list[MacroCandidate]:
    """Build a QUBO variable set without losing structured representatives."""

    selected: dict[tuple[tuple, tuple], MacroCandidate] = {}

    def add(rows: Iterable[MacroCandidate]) -> None:
        for macro in rows:
            key = (macro.first_hash, macro.leaf_hash)
            old = selected.get(key)
            if old is None or macro.score < old.score:
                selected[key] = macro

    ranked = sorted(macros, key=lambda m: (m.score, _state_rank(instance, m.leaf_state)))
    add(ranked[:elite_keep])

    for src in sorted({macro.first_move.from_stack for macro in macros}):
        bucket = [macro for macro in macros if macro.first_move.from_stack == src]
        bucket.sort(key=lambda m: (m.score, _state_rank(instance, m.leaf_state)))
        add(bucket[:source_keep])

    for dst in sorted({macro.first_move.to_stack for macro in macros}):
        bucket = [macro for macro in macros if macro.first_move.to_stack == dst]
        bucket.sort(key=lambda m: (m.score, _state_rank(instance, m.leaf_state)))
        add(bucket[:dest_keep])

    plateau = [macro for macro in macros if macro.first_n >= macro.root_n and macro.leaf_n < macro.root_n]
    add(sorted(plateau, key=lambda m: (m.score, _state_rank(instance, m.leaf_state)))[:plateau_keep])

    usable = sorted(selected.values(), key=lambda m: (m.score, _state_rank(instance, m.leaf_state)))
    return usable[:max_qubo_vars]


def _selection_value(
    macros: list[MacroCandidate],
    selected: Iterable[int],
    *,
    score_norm: list[float],
    state_similarity_weight: float,
    same_first_weight: float,
    same_source_weight: float,
    same_dest_weight: float,
) -> float:
    idxs = list(selected)
    total = sum(score_norm[i] for i in idxs)
    for pos, i in enumerate(idxs):
        mi = macros[i]
        for j in idxs[pos + 1 :]:
            mj = macros[j]
            total += state_similarity_weight * combined_similarity(mi.leaf_state, mj.leaf_state)
            if mi.first_hash == mj.first_hash:
                total += same_first_weight
            if mi.first_move.from_stack == mj.first_move.from_stack:
                total += same_source_weight
            if mi.first_move.to_stack == mj.first_move.to_stack:
                total += same_dest_weight
    return float(total)


def _repair_selection(
    macros: list[MacroCandidate],
    selected: Iterable[int],
    target: int,
    *,
    score_norm: list[float],
    state_similarity_weight: float,
    same_first_weight: float,
    same_source_weight: float,
    same_dest_weight: float,
) -> list[int]:
    selected_set = set(i for i in selected if 0 <= i < len(macros))
    target = min(target, len(macros))

    def value(items: Iterable[int]) -> float:
        return _selection_value(
            macros,
            items,
            score_norm=score_norm,
            state_similarity_weight=state_similarity_weight,
            same_first_weight=same_first_weight,
            same_source_weight=same_source_weight,
            same_dest_weight=same_dest_weight,
        )

    while len(selected_set) > target:
        remove = min(selected_set, key=lambda idx: (value(selected_set - {idx}), -score_norm[idx], idx))
        selected_set.remove(remove)

    while len(selected_set) < target:
        missing = [idx for idx in range(len(macros)) if idx not in selected_set]
        if not missing:
            break
        add = min(missing, key=lambda idx: (value(selected_set | {idx}), score_norm[idx], idx))
        selected_set.add(add)

    current = value(selected_set)
    improved = True
    swaps = 0
    while improved and swaps < 4:
        improved = False
        best_pair = None
        best_value = current
        for remove in list(selected_set):
            base = selected_set - {remove}
            for add in range(len(macros)):
                if add in selected_set:
                    continue
                trial = base | {add}
                trial_value = value(trial)
                if trial_value + 1e-12 < best_value:
                    best_value = trial_value
                    best_pair = (remove, add)
        if best_pair is not None:
            selected_set.remove(best_pair[0])
            selected_set.add(best_pair[1])
            current = best_value
            improved = True
            swaps += 1

    return sorted(selected_set, key=lambda idx: (score_norm[idx], idx))


def select_macros_with_qubo(
    instance: Instance,
    macros: list[MacroCandidate],
    config: SolverConfig,
    *,
    target_count: int,
    state_similarity_weight: float,
    same_first_weight: float,
    same_source_weight: float,
    same_dest_weight: float,
) -> tuple[list[MacroCandidate], int, float]:
    """Select macro paths with a QUBO and repair to exact cardinality."""

    if not macros:
        return [], 0, 0.0
    target = min(target_count, len(macros))
    usable = _select_usable_macros(
        macros,
        max_qubo_vars=config.max_qubo_vars,
        elite_keep=max(target * 2, target + 4),
        source_keep=max(2, target // 3),
        dest_keep=max(1, target // 4),
        plateau_keep=max(4, target // 2),
        instance=instance,
    )
    if len(usable) <= target:
        return usable, 0, 0.0

    scores = [macro.score for macro in usable]
    lo, hi = min(scores), max(scores)
    score_norm = [0.0 if hi == lo else (score - lo) / (hi - lo) for score in scores]

    if config.qubo_num_reads <= 0:
        repaired = _repair_selection(
            usable,
            [],
            target,
            score_norm=score_norm,
            state_similarity_weight=state_similarity_weight,
            same_first_weight=same_first_weight,
            same_source_weight=same_source_weight,
            same_dest_weight=same_dest_weight,
        )
        return [usable[idx] for idx in repaired], len(usable), 0.0

    model = QuboModel()
    vars_ = [model.add_binary_var(f"m_{idx}") for idx in range(len(usable))]
    for idx, var in enumerate(vars_):
        model.add_linear(var, score_norm[idx])
    for i, mi in enumerate(usable):
        for j in range(i + 1, len(usable)):
            mj = usable[j]
            coeff = state_similarity_weight * combined_similarity(mi.leaf_state, mj.leaf_state)
            if mi.first_hash == mj.first_hash:
                coeff += same_first_weight
            if mi.first_move.from_stack == mj.first_move.from_stack:
                coeff += same_source_weight
            if mi.first_move.to_stack == mj.first_move.to_stack:
                coeff += same_dest_weight
            if coeff:
                model.add_quadratic(vars_[i], vars_[j], coeff)
    model.add_exactly_k_penalty(vars_, target, config.qubo_penalty_a)

    sampler = SimulatedAnnealingQuboSampler()
    sample_set = sampler.sample(
        model,
        QuboSamplerConfig(
            num_reads=config.qubo_num_reads,
            random_seed=config.random_seed,
            time_limit_sec=config.qubo_time_limit_sec or 0.05,
            sa_iterations=config.qubo_sa_iterations,
        ),
    )

    best: list[int] | None = None
    best_value = float("inf")
    for sample in sample_set.samples or []:
        selected = [idx for idx in range(len(usable)) if sample.binary_values.get(idx, 0) == 1]
        repaired = _repair_selection(
            usable,
            selected,
            target,
            score_norm=score_norm,
            state_similarity_weight=state_similarity_weight,
            same_first_weight=same_first_weight,
            same_source_weight=same_source_weight,
            same_dest_weight=same_dest_weight,
        )
        value = _selection_value(
            usable,
            repaired,
            score_norm=score_norm,
            state_similarity_weight=state_similarity_weight,
            same_first_weight=same_first_weight,
            same_source_weight=same_source_weight,
            same_dest_weight=same_dest_weight,
        )
        if value < best_value:
            best_value = value
            best = repaired
    if best is None:
        best = _repair_selection(
            usable,
            [],
            target,
            score_norm=score_norm,
            state_similarity_weight=state_similarity_weight,
            same_first_weight=same_first_weight,
            same_source_weight=same_source_weight,
            same_dest_weight=same_dest_weight,
        )
    return [usable[idx] for idx in best], len(model.var_index), sample_set.runtime_sec


def _best_state_key(instance: Instance, state: State) -> tuple:
    return (
        disorder_box_count(instance, state),
        state.entropy_max,
        state.disorder_total,
        state.move_count,
    )


def tail_repair_to_n_zero(
    instance: Instance,
    start_state: State,
    config: SolverConfig,
    *,
    deadline: float,
    max_expanded: int,
) -> tuple[State | None, int, int]:
    """Best-first exact tail repair from a low-N state to N=0."""

    if disorder_box_count(instance, start_state) == 0:
        return start_state, 0, 0
    cap = entropy_limit(instance, config.entropy_alpha)
    if start_state.move_count >= cap:
        return None, 0, 0

    start_hash = hash_state(instance, start_state, config.enable_canonical_hash)
    visited = {start_hash: start_state.move_count}
    heap: list[tuple[tuple, int, State]] = []
    counter = 0
    heapq.heappush(heap, (_state_rank(instance, start_state), counter, start_state))
    expanded = generated = 0

    while heap and expanded < max_expanded and time.perf_counter() < deadline:
        _rank, _idx, state = heapq.heappop(heap)
        if disorder_box_count(instance, state) == 0:
            return state, expanded, generated
        if state.move_count >= cap:
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
            h = hash_state(instance, nxt, config.enable_canonical_hash)
            if visited.get(h, math.inf) <= nxt.move_count:
                continue
            visited[h] = nxt.move_count
            counter += 1
            heapq.heappush(heap, (_state_rank(instance, nxt), counter, nxt))
    return None, expanded, generated


def run_macro_reservoir_qubo_beam(
    case: str,
    instance: Instance,
    config: SolverConfig,
    *,
    macro_depth: int,
    commit_length: int,
    global_keep: int,
    source_keep: int,
    dest_keep: int,
    plateau_keep: int,
    max_reservoir: int,
    max_macros_per_state: int,
    state_similarity_weight: float,
    same_first_weight: float,
    same_source_weight: float,
    same_dest_weight: float,
    tail_repair_n: int,
    tail_repair_expanded: int,
) -> tuple[TrajectoryResult, State]:
    """Run the experimental macro-reservoir QUBO beam."""

    start = time.perf_counter()
    initial = create_initial_state(instance)
    best = initial
    h0 = hash_state(instance, initial, config.enable_canonical_hash)
    beam = [Candidate(initial, score_state(instance, initial, config.score_weights), None, h0)]
    visited = {h0: dominance_value(instance, initial)}
    expanded = generated = qcalls = qvars = 0
    tail_expanded = tail_generated = 0
    qtime = 0.0
    found_tail = False

    for depth in range(config.max_depth):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        all_macros: list[MacroCandidate] = []
        for cand in beam:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expanded += 1
            macros = build_macro_candidates(
                instance,
                cand.state,
                config,
                macro_depth=macro_depth,
                commit_length=commit_length,
                global_keep=global_keep,
                source_keep=source_keep,
                dest_keep=dest_keep,
                plateau_keep=plateau_keep,
                max_reservoir=max_reservoir,
                max_macros_per_state=max_macros_per_state,
            )
            all_macros.extend(macros)
        generated += len(all_macros)
        if not all_macros:
            break

        selected, vars_count, elapsed = select_macros_with_qubo(
            instance,
            all_macros,
            config,
            target_count=config.beam_width,
            state_similarity_weight=state_similarity_weight,
            same_first_weight=same_first_weight,
            same_source_weight=same_source_weight,
            same_dest_weight=same_dest_weight,
        )
        qcalls += 1 if vars_count else 0
        qvars += vars_count
        qtime += elapsed

        next_by_state: dict[tuple, Candidate] = {}
        filler = sorted(all_macros, key=lambda macro: (macro.score, _state_rank(instance, macro.leaf_state)))
        for macro in selected + filler:
            h = macro.commit_hash
            dom = dominance_value(instance, macro.commit_state)
            if config.enable_visited_pruning and visited.get(h, float("inf")) <= dom:
                continue
            old = next_by_state.get(h)
            commit_move = macro.moves[min(len(macro.moves), max(1, commit_length)) - 1]
            candidate = Candidate(macro.commit_state, macro.score, commit_move, h)
            if old is None or candidate.score < old.score:
                next_by_state[h] = candidate
            if len(next_by_state) >= config.beam_width:
                break

        if not next_by_state:
            break
        for candidate in next_by_state.values():
            visited[candidate.hash_value] = dominance_value(instance, candidate.state)
            if _best_state_key(instance, candidate.state) < _best_state_key(instance, best):
                best = candidate.state

        beam = sorted(
            next_by_state.values(),
            key=lambda cand: (
                _best_state_key(instance, cand.state),
                cand.score,
            ),
        )[: config.beam_width]

        if tail_repair_n > 0:
            low_n_states = [
                cand.state
                for cand in beam
                if disorder_box_count(instance, cand.state) <= tail_repair_n
            ]
            low_n_states.sort(key=lambda state: _best_state_key(instance, state))
            for state in low_n_states[: min(6, len(low_n_states))]:
                repaired, exp_count, gen_count = tail_repair_to_n_zero(
                    instance,
                    state,
                    config,
                    deadline=start + config.time_limit_sec,
                    max_expanded=tail_repair_expanded,
                )
                tail_expanded += exp_count
                tail_generated += gen_count
                if repaired is not None:
                    best = repaired
                    found_tail = True
                    break

        print(
            "LAYER "
            f"case={case} depth={depth + 1} beam={len(beam)} macros={len(all_macros)} "
            f"bestN={disorder_box_count(instance, best)} beamBestN={disorder_box_count(instance, beam[0].state)} "
            f"elapsed={time.perf_counter() - start:.2f}s",
            flush=True,
        )

        if disorder_box_count(instance, best) == 0:
            break

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += (
        f"; macro_depth={macro_depth}; reservoir=g{global_keep}/s{source_keep}/d{dest_keep}/p{plateau_keep}; "
        f"commit_length={commit_length}; max_reservoir={max_reservoir}; max_macros_per_state={max_macros_per_state}; "
        f"tail_repair_n={tail_repair_n}; tail_expanded={tail_expanded}; tail_generated={tail_generated}; "
        f"tail_found={found_tail}"
    )
    row = _result_from_state(
        case,
        "macro_reservoir_qubo_beam",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=len(visited),
        qubo_calls=qcalls,
        qubo_total_vars=qvars,
        qubo_time_sec=qtime,
        status=status,
    )
    return row, best


def _write_single_csv(row: TrajectoryResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrajectoryResult.__annotations__.keys()))
        writer.writeheader()
        writer.writerow(row.__dict__)


def _write_moves_csv(instance: Instance, moves: list[Move], path: Path) -> None:
    def label(idx: int) -> str:
        return str(idx + 1) if idx < 9 else "A"

    path.parent.mkdir(parents=True, exist_ok=True)
    state = create_initial_state(instance)
    rows = []
    for step, move in enumerate(moves, start=1):
        state = apply_move(instance, state, move)
        rows.append(
            {
                "step": step,
                "box_rank": move.box,
                "from_stack_label": label(move.from_stack),
                "to_stack_label": label(move.to_stack),
                "from_level": move.from_level,
                "to_level": move.to_level,
                "n_after": disorder_box_count(instance, state),
                "g_after": entropy_value(instance, state),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "step",
            "box_rank",
            "from_stack_label",
            "to_stack_label",
            "from_level",
            "to_level",
            "n_after",
            "g_after",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_trace(instance: Instance, moves: list[Move]) -> None:
    def label(idx: int) -> str:
        return str(idx + 1) if idx < 9 else "A"

    state = create_initial_state(instance)
    print(f"TRACE 0 N={disorder_box_count(instance, state)} G={entropy_value(instance, state)}")
    for step, move in enumerate(moves, start=1):
        state = apply_move(instance, state, move)
        print(
            "TRACE "
            f"{step} {move.box}:{label(move.from_stack)}->{label(move.to_stack)} "
            f"N={disorder_box_count(instance, state)} G={entropy_value(instance, state)}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="C1")
    parser.add_argument("--bay", default="26", help="Single bay suffix. Use empty string for paired 01+26.")
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--macro-depth", type=int, default=3)
    parser.add_argument("--commit-length", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=60)
    parser.add_argument("--max-qubo-vars", type=int, default=160)
    parser.add_argument("--qubo-time-limit", type=float, default=0.05)
    parser.add_argument("--qubo-reads", type=int, default=24)
    parser.add_argument("--qubo-iterations", type=int, default=220)
    parser.add_argument("--global-keep", type=int, default=4)
    parser.add_argument("--source-keep", type=int, default=2)
    parser.add_argument("--dest-keep", type=int, default=1)
    parser.add_argument("--plateau-keep", type=int, default=8)
    parser.add_argument("--max-reservoir", type=int, default=24)
    parser.add_argument("--max-macros-per-state", type=int, default=80)
    parser.add_argument("--tail-repair-n", type=int, default=0)
    parser.add_argument("--tail-repair-expanded", type=int, default=200000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    instance = _instance_from_single_bay(args.zone, args.bay) if args.bay else _instance_from_zone(args.zone)
    cap = entropy_limit(instance, 1.3)
    config = SolverConfig(
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        max_candidates_per_layer=500,
        time_limit_sec=args.time_limit,
        max_qubo_vars=args.max_qubo_vars,
        qubo_penalty_a=40.0,
        qubo_num_reads=args.qubo_reads,
        qubo_time_limit_sec=args.qubo_time_limit,
        qubo_sa_iterations=args.qubo_iterations,
        entropy_alpha=1.3,
        random_seed=37,
    )
    print(
        "START "
        f"case={instance.instance_id} stacks={instance.stack_count} H={instance.max_height} "
        f"containers={instance.container_count} initial_N={disorder_box_count(instance, create_initial_state(instance))} "
        f"cap={cap}",
        flush=True,
    )
    row, best_state = run_macro_reservoir_qubo_beam(
        instance.instance_id,
        instance,
        config,
        macro_depth=args.macro_depth,
        commit_length=args.commit_length,
        global_keep=args.global_keep,
        source_keep=args.source_keep,
        dest_keep=args.dest_keep,
        plateau_keep=args.plateau_keep,
        max_reservoir=args.max_reservoir,
        max_macros_per_state=args.max_macros_per_state,
        state_similarity_weight=0.08,
        same_first_weight=0.60,
        same_source_weight=0.20,
        same_dest_weight=0.06,
        tail_repair_n=args.tail_repair_n,
        tail_repair_expanded=args.tail_repair_expanded,
    )
    print(f"FINAL {row}", flush=True)
    _print_trace(instance, best_state.moves)
    output = args.output or ROOT / "docs" / "experiments" / f"macro_reservoir_qubo_beam_{instance.instance_id}.csv"
    _write_single_csv(row, output)
    moves_output = output.with_name(output.stem + "_moves.csv")
    _write_moves_csv(instance, best_state.moves, moves_output)
    print(f"wrote {output}", flush=True)
    print(f"wrote {moves_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
