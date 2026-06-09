"""QUBO-centered frontier selection primitives.

This module is the shared place for beam-frontier decisions.  Search solvers
generate legal candidates; this selector decides which candidates survive by
combining deterministic protected cores with QUBO-selected supplements or
dynamic QUBO-controlled frontier sizes.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal, Sequence

from ..core import State
from ..similarity import combined_similarity
from ..solvers.beam import Candidate
from .beam_selection import QuboBeamSelectionBuilder
from .model import QuboModel
from .sampler import IQuboSampler, QuboSamplerConfig, SimulatedAnnealingQuboSampler

QuboFrontierPolicy = Literal["classic", "enhanced", "dynamic_qubo"]


@dataclass
class QuboFrontierConfig:
    """Configuration for QUBO-centered frontier selection."""

    policy: QuboFrontierPolicy = "enhanced"
    use_qubo: bool = True
    beam_width: int = 100
    core_width: int = 0
    parent_core_cap: int = 0
    max_candidates: int = 0

    qubo_min_candidates: int = 64
    qubo_max_vars: int = 96
    qubo_supplement_width: int = 24
    qubo_penalty_a: float = 30.0
    qubo_diversity_weight: float = 0.10
    qubo_num_reads: int = 2
    qubo_time_limit_sec: float = 0.001
    qubo_sa_iterations: int = 20
    qubo_repair_hamming_slack: int = 3

    group_qubo_enabled: bool = False
    group_qubo_max_vars: int = 96
    group_qubo_select: int = 24
    group_qubo_take: int = 4
    group_same_n_penalty: float = 0.04

    dynamic_min_frontier: int = 128
    dynamic_core_frontier: int = 0
    dynamic_max_frontier: int = 0
    dynamic_candidate_pool: int = 4096
    dynamic_keep_threshold: float = 0.55

    dynamic_adaptive_frontier: bool = True
    dynamic_stagnation_window: int = 3
    dynamic_stagnation_epsilon: float = 1e-6
    dynamic_similarity_window: int = 64
    dynamic_similarity_expand_threshold: float = 0.58
    dynamic_similarity_contract_threshold: float = 0.35
    dynamic_parent_share_expand_threshold: float = 0.55
    dynamic_score_gap_expand_ratio: float = 0.03
    dynamic_frontier_pressure_step: float = 0.10
    dynamic_frontier_pressure_decay: float = 0.07

    random_seed: int = 1


@dataclass
class QuboFrontierSelection:
    """Selected frontier and QUBO accounting for one search layer."""

    selected: list[Candidate]
    selected_by_qubo: list[Candidate]
    qubo_calls: int = 0
    qubo_vars: int = 0
    qubo_time_sec: float = 0.0
    repair_count: int = 0
    policy: str = "enhanced"
    frontier_width: int | None = None
    frontier_pressure: float | None = None
    reason: str = ""


class QuboFrontierSelector:
    """Select beam frontiers with QUBO as the central policy component."""

    def __init__(self, sampler: IQuboSampler | None = None) -> None:
        self.sampler = sampler or SimulatedAnnealingQuboSampler()
        self.builder = QuboBeamSelectionBuilder()
        self._best_score_history: deque[float] = deque(maxlen=32)
        self._frontier_pressure: float = 1.0

    def select(
        self,
        candidates: Sequence[Candidate],
        config: QuboFrontierConfig,
        *,
        call_index: int = 0,
    ) -> QuboFrontierSelection:
        ranked = _ranked(candidates, config.max_candidates)
        if not ranked:
            return QuboFrontierSelection([], [], policy=config.policy)
        if config.policy == "classic" or not config.use_qubo:
            selected = ranked[: max(1, min(config.beam_width, len(ranked)))]
            return QuboFrontierSelection(selected, [], policy="classic")
        if config.policy == "dynamic_qubo":
            return self._select_dynamic(ranked, config, call_index=call_index)
        return self._select_enhanced(ranked, config, call_index=call_index)

    def _select_enhanced(
        self,
        ranked: list[Candidate],
        config: QuboFrontierConfig,
        *,
        call_index: int,
    ) -> QuboFrontierSelection:
        target = max(1, min(config.beam_width, len(ranked)))
        core_width = max(0, min(config.core_width, target))
        selected: list[Candidate] = []
        selected_keys: set[tuple] = set()
        qselected: list[Candidate] = []
        qcalls = qvars = repairs = 0
        qtime = 0.0

        self._append_ranked_core(ranked, selected, selected_keys, core_width, config.parent_core_cap)

        remaining = [cand for cand in ranked if cand.hash_value not in selected_keys]
        if len(remaining) >= config.qubo_min_candidates and len(selected) < target:
            take = min(config.qubo_supplement_width, target - len(selected), len(remaining))
            picked, vars_, time_, repair = self._select_candidate_qubo(
                remaining,
                take,
                config,
                seed_offset=1009 + call_index,
            )
            qcalls += int(vars_ > 0)
            qvars += vars_
            qtime += time_
            repairs += repair
            for cand in picked:
                if cand.hash_value in selected_keys:
                    continue
                selected.append(cand)
                selected_keys.add(cand.hash_value)
                qselected.append(cand)
                if len(selected) >= target:
                    break

        if config.group_qubo_enabled and len(selected) < target:
            remaining = [cand for cand in ranked if cand.hash_value not in selected_keys]
            picked, vars_, time_ = self._select_group_qubo(
                remaining,
                config,
                capacity=target - len(selected),
                call_index=call_index,
            )
            qcalls += int(vars_ > 0)
            qvars += vars_
            qtime += time_
            for cand in picked:
                if cand.hash_value in selected_keys:
                    continue
                selected.append(cand)
                selected_keys.add(cand.hash_value)
                qselected.append(cand)
                if len(selected) >= target:
                    break

        self._append_ranked_core(ranked, selected, selected_keys, target, 0)
        return QuboFrontierSelection(
            selected[:target],
            qselected,
            qubo_calls=qcalls,
            qubo_vars=qvars,
            qubo_time_sec=qtime,
            repair_count=repairs,
            policy="enhanced",
        )

    def _select_dynamic(
        self,
        ranked: list[Candidate],
        config: QuboFrontierConfig,
        *,
        call_index: int,
    ) -> QuboFrontierSelection:
        max_frontier = config.dynamic_max_frontier or config.beam_width
        max_frontier = max(1, min(max_frontier, len(ranked)))
        min_frontier = max(1, min(config.dynamic_min_frontier, max_frontier))
        core_frontier = max(0, min(config.dynamic_core_frontier, max_frontier))
        if config.dynamic_adaptive_frontier:
            max_frontier, pressure, should_expand = self._select_frontier_width(
                ranked,
                config,
                best_score=ranked[0].score,
            )
        else:
            pressure = 1.0
            should_expand = True

        selected: list[Candidate] = []
        selected_keys: set[tuple] = set()
        self._append_ranked_core(ranked, selected, selected_keys, core_frontier, config.parent_core_cap)
        fallback_target = max(min_frontier, core_frontier)
        fallback = ranked[:fallback_target]

        if len(ranked) < config.qubo_min_candidates or max_frontier <= fallback_target:
            return QuboFrontierSelection(fallback, [], policy="dynamic_qubo")

        if pressure < 0.40 and not should_expand:
            return QuboFrontierSelection(fallback, [], policy="dynamic_qubo")

        pool = ranked[: max(1, min(self._adaptive_pool_size(config, pressure), len(ranked)))]
        picked, qvars, qtime = self._select_group_qubo(
            [cand for cand in pool if cand.hash_value not in selected_keys],
            config,
            capacity=max_frontier - len(selected),
            call_index=call_index,
            dynamic=True,
        )
        if not qvars:
            return QuboFrontierSelection(fallback, [], policy="dynamic_qubo")

        qselected: list[Candidate] = []
        for cand in picked:
            if cand.hash_value in selected_keys:
                continue
            selected.append(cand)
            selected_keys.add(cand.hash_value)
            qselected.append(cand)
            if len(selected) >= max_frontier:
                break
        self._append_ranked_core(ranked, selected, selected_keys, min_frontier, 0)
        return QuboFrontierSelection(
            selected[:max_frontier],
            qselected,
            qubo_calls=1,
            qubo_vars=qvars,
            qubo_time_sec=qtime,
            policy="dynamic_qubo",
            frontier_width=max_frontier,
            frontier_pressure=pressure,
            reason=f"pressure={pressure:.3f}; expand={should_expand}",
        )

    def _select_frontier_width(
        self,
        ranked: list[Candidate],
        config: QuboFrontierConfig,
        *,
        best_score: float,
    ) -> tuple[int, float, bool]:
        min_frontier = max(1, min(config.dynamic_min_frontier, len(ranked)))
        max_frontier = max(min_frontier, max(1, min(config.dynamic_max_frontier or config.beam_width, len(ranked))))

        similarity = self._recent_similarity(ranked, config.dynamic_similarity_window)
        parent_share = self._top_parent_share(ranked, config.dynamic_similarity_window)
        score_gap_ratio = self._score_gap_ratio(ranked, config.dynamic_similarity_window)

        stalled = False
        if len(self._best_score_history) >= config.dynamic_stagnation_window:
            oldest = self._best_score_history[0]
            if 0 <= oldest - best_score <= config.dynamic_stagnation_epsilon:
                stalled = True

        self._best_score_history.append(best_score)

        expand_pressure = (
            stalled
            or similarity >= config.dynamic_similarity_expand_threshold
            or parent_share >= config.dynamic_parent_share_expand_threshold
            or score_gap_ratio <= config.dynamic_score_gap_expand_ratio
        )

        pressure = self._frontier_pressure
        if expand_pressure:
            pressure = min(1.0, pressure + config.dynamic_frontier_pressure_step)
        elif (
            similarity <= config.dynamic_similarity_contract_threshold
            and parent_share <= min(1.0, config.dynamic_parent_share_expand_threshold * 0.75)
            and score_gap_ratio >= 0.20
        ):
            pressure = max(0.0, pressure - config.dynamic_frontier_pressure_step)
        else:
            pressure = max(0.0, pressure - config.dynamic_frontier_pressure_decay)

        self._frontier_pressure = pressure
        frontier_width = max(
            min_frontier,
            min(max_frontier, int(round(min_frontier + (max_frontier - min_frontier) * pressure))),
        )
        return frontier_width, pressure, expand_pressure

    @staticmethod
    def _adaptive_pool_size(config: QuboFrontierConfig, pressure: float) -> int:
        scale = 0.55 + 0.45 * max(0.0, min(1.0, pressure))
        return max(config.qubo_min_candidates, int(math.ceil(config.dynamic_candidate_pool * scale)))

    @staticmethod
    def _recent_similarity(ranked: list[Candidate], window: int) -> float:
        if window <= 1 or len(ranked) < 2:
            return 0.0
        sample = ranked[: min(len(ranked), max(2, window))]
        return _avg_similarity(sample)

    @staticmethod
    def _top_parent_share(ranked: list[Candidate], window: int) -> float:
        if not ranked:
            return 0.0
        sample = ranked[: min(len(ranked), max(1, window))]
        counts: dict[tuple, int] = {}
        for cand in sample:
            parent = _parent_key(cand)
            counts[parent] = counts.get(parent, 0) + 1
        best = max(counts.values())
        return best / len(sample)

    @staticmethod
    def _score_gap_ratio(ranked: list[Candidate], window: int) -> float:
        if len(ranked) < 2:
            return 0.0
        sample = ranked[: min(len(ranked), max(2, window))]
        best = sample[0].score
        worst = sample[-1].score
        span = abs(worst - best)
        base = abs(best) + 1.0
        return span / base

    def _select_candidate_qubo(
        self,
        candidates: list[Candidate],
        target: int,
        config: QuboFrontierConfig,
        *,
        seed_offset: int,
    ) -> tuple[list[Candidate], int, float, int]:
        if target <= 0 or not candidates:
            return [], 0, 0.0, 0
        model, usable = self.builder.build(
            candidates,
            target,
            config.qubo_diversity_weight,
            config.qubo_penalty_a,
            config.qubo_max_vars,
        )
        if not usable or not model.var_index:
            return [], 0, 0.0, 0
        sample_set = self.sampler.sample(
            model,
            QuboSamplerConfig(
                num_reads=config.qubo_num_reads,
                random_seed=config.random_seed + seed_offset,
                time_limit_sec=max(0.001, config.qubo_time_limit_sec),
                sa_iterations=max(1, config.qubo_sa_iterations),
            ),
        )
        picked, repairs = self.builder.decode_samples_and_repair(
            sample_set.samples,
            usable,
            target,
            elite_keep_count=0,
            diversity_weight=config.qubo_diversity_weight,
            repair_hamming_slack=config.qubo_repair_hamming_slack,
        )
        return picked, len(model.var_index), sample_set.runtime_sec, repairs

    def _select_group_qubo(
        self,
        candidates: list[Candidate],
        config: QuboFrontierConfig,
        *,
        capacity: int,
        call_index: int,
        dynamic: bool = False,
    ) -> tuple[list[Candidate], int, float]:
        if capacity <= 0 or not candidates:
            return [], 0, 0.0
        groups = _group_by_parent(candidates)
        if not groups:
            return [], 0, 0.0
        max_vars = max(1, config.group_qubo_max_vars)
        group_rows = sorted(groups.values(), key=lambda rows: _candidate_key(rows[0]))[:max_vars]
        raw = [_group_score(rows) for rows in group_rows]
        lo, hi = min(raw), max(raw)
        quality = [1.0 if hi == lo else 1.0 - ((value - lo) / (hi - lo)) for value in raw]

        model = QuboModel()
        vars_ = [model.add_binary_var(f"g_{idx}") for idx in range(len(group_rows))]
        if dynamic:
            for idx, var in enumerate(vars_):
                model.add_linear(var, config.dynamic_keep_threshold - quality[idx])
        else:
            target_groups = _target_group_count(config, capacity, len(group_rows))
            for idx, var in enumerate(vars_):
                model.add_linear(var, -quality[idx])
            model.add_exactly_k_penalty(vars_, target_groups, config.qubo_penalty_a)

        for i in range(len(group_rows)):
            ni = _state_n(group_rows[i][0].state)
            for j in range(i + 1, len(group_rows)):
                if ni == _state_n(group_rows[j][0].state):
                    model.add_quadratic(vars_[i], vars_[j], config.group_same_n_penalty)

        sample_set = self.sampler.sample(
            model,
            QuboSamplerConfig(
                num_reads=config.qubo_num_reads,
                random_seed=config.random_seed + 7001 + call_index,
                time_limit_sec=max(0.001, config.qubo_time_limit_sec),
                sa_iterations=max(1, config.qubo_sa_iterations),
            ),
        )
        picked = _picked_indices(sample_set.samples[0].binary_values if sample_set.samples else {}, len(group_rows))
        if dynamic:
            min_groups = max(1, math.ceil(min(config.dynamic_min_frontier, capacity) / max(1, config.group_qubo_take)))
            picked = _repair_at_least_groups(picked, quality, min_groups)
        else:
            picked = _repair_exact_groups(picked, quality, _target_group_count(config, capacity, len(group_rows)))

        result: list[Candidate] = []
        for idx in sorted(picked, key=lambda i: (-quality[i], i)):
            for cand in group_rows[idx][: max(1, config.group_qubo_take)]:
                result.append(cand)
                if len(result) >= capacity:
                    return result, len(model.var_index), sample_set.runtime_sec
        return result, len(model.var_index), sample_set.runtime_sec

    @staticmethod
    def _append_ranked_core(
        ranked: list[Candidate],
        selected: list[Candidate],
        selected_keys: set[tuple],
        target_len: int,
        parent_core_cap: int,
    ) -> None:
        if len(selected) >= target_len:
            return
        if parent_core_cap > 0:
            parent_counts: dict[tuple, int] = {}
            for cand in ranked:
                if cand.hash_value in selected_keys:
                    continue
                parent = _parent_key(cand)
                if parent_counts.get(parent, 0) >= parent_core_cap:
                    continue
                selected.append(cand)
                selected_keys.add(cand.hash_value)
                parent_counts[parent] = parent_counts.get(parent, 0) + 1
                if len(selected) >= target_len:
                    return
        for cand in ranked:
            if cand.hash_value in selected_keys:
                continue
            selected.append(cand)
            selected_keys.add(cand.hash_value)
            if len(selected) >= target_len:
                return


def _ranked(candidates: Sequence[Candidate], max_candidates: int) -> list[Candidate]:
    ranked = sorted(candidates, key=_candidate_key)
    if max_candidates > 0:
        ranked = ranked[:max_candidates]
    return ranked


def _candidate_key(candidate: Candidate) -> tuple[float, int, tuple]:
    return (float(candidate.score), candidate.state.move_count, candidate.hash_value)


def _parent_key(candidate: Candidate) -> tuple:
    return candidate.parent_hash if candidate.parent_hash is not None else candidate.hash_value


def _group_by_parent(candidates: list[Candidate]) -> dict[tuple, list[Candidate]]:
    groups: dict[tuple, list[Candidate]] = {}
    for cand in candidates:
        groups.setdefault(_parent_key(cand), []).append(cand)
    for rows in groups.values():
        rows.sort(key=_candidate_key)
    return groups


def _group_score(rows: list[Candidate]) -> float:
    best = rows[0]
    return float(best.score) - 0.001 * min(16, len(rows))


def _state_n(state: State) -> int:
    value = int(getattr(state, "disorder_current", 0) or 0)
    return value


def _target_group_count(config: QuboFrontierConfig, capacity: int, group_count: int) -> int:
    by_capacity = math.ceil(max(1, capacity) / max(1, config.group_qubo_take))
    configured = max(1, config.group_qubo_select)
    return max(1, min(group_count, configured, by_capacity))



def _avg_similarity(candidates: list[Candidate]) -> float:
    if len(candidates) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            total += combined_similarity(candidates[i].state, candidates[j].state)
            count += 1
    return total / count if count else 0.0


def _picked_indices(binary_values: dict[int, int], size: int) -> set[int]:
    return {idx for idx in range(size) if binary_values.get(idx, 0) == 1}


def _repair_exact_groups(picked: set[int], quality: list[float], target: int) -> set[int]:
    picked = set(picked)
    while len(picked) > target:
        picked.remove(min(picked, key=lambda idx: (quality[idx], -idx)))
    while len(picked) < target:
        missing = [idx for idx in range(len(quality)) if idx not in picked]
        if not missing:
            break
        picked.add(max(missing, key=lambda idx: (quality[idx], -idx)))
    return picked


def _repair_at_least_groups(picked: set[int], quality: list[float], target: int) -> set[int]:
    picked = set(picked)
    while len(picked) < target:
        missing = [idx for idx in range(len(quality)) if idx not in picked]
        if not missing:
            break
        picked.add(max(missing, key=lambda idx: (quality[idx], -idx)))
    return picked

