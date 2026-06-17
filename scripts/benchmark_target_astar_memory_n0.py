"""Fresh in-memory target A*/best-first benchmark for N=0.

This is the no-SQL counterpart to ``benchmark_target_astar_fresh_n0.py``.  It
uses compact byte-encoded states, Python heap/dicts for open/best/parent, stops
as soon as N=0 is reached, reconstructs the route from the in-memory parent map,
and writes the same result/acceptance/moves artifacts.
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam import _write_explain_csv  # noqa: E402
from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from benchmark_fast_qubo_integrated_v1 import _exact_moves_by_case, _move_gap, _parse_case, _rel  # noqa: E402
from benchmark_target_layout_qubo_upper_bound_v1 import TargetLayoutQuboBuilder, _decode_target  # noqa: E402
from benchmark_full_real_business_algorithms import TrajectoryResult  # noqa: E402
from container_reshuffle.core import Move, Solution, apply_move, create_initial_state, validate_solution  # noqa: E402
from container_reshuffle.features import disorder_box_count, entropy_limit, max_entropy_along_moves  # noqa: E402
from container_reshuffle.qubo.model import QuboModel  # noqa: E402
from container_reshuffle.qubo.sampler import QuboSamplerConfig, SimulatedAnnealingQuboSampler  # noqa: E402
from extract_business_path_from_sqlite import _lift_to_physical  # noqa: E402
from run_business_exact_target_search_sqlite import (  # noqa: E402
    _canonical,
    _encode,
    _height,
    _remaining_lower_bound,
    _split,
    _stack_n,
    _successors_with_n,
)
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


BASE_SOLVER_NAME = "target_astar_memory_n0_v1"
QUBO_SOLVER_NAME = "target_astar_memory_qubo_n0_v1"
BROAD_QUBO_SOLVER_NAME = "target_astar_memory_qubo_broad_n0_v1"
QUALITY_GUARD_QUBO_SOLVER_NAME = "target_astar_memory_qubo_quality_guard_n0_v1"
FRONTIER_COHORT_QUBO_SOLVER_NAME = "target_astar_memory_qubo_frontier_cohort_n0_v1"
ACTION_ORDER_QUBO_SOLVER_NAME = "target_astar_memory_qubo_action_order_n0_v1"
TAIL_PLAN_QUBO_SOLVER_NAME = "target_astar_memory_qubo_tail_plan_n0_v1"
HYBRID_DEEP_QUBO_SOLVER_NAME = "target_astar_memory_qubo_hybrid_deep_n0_v1"
LAYOUT_GUIDED_QUBO_SOLVER_NAME = "target_astar_memory_qubo_layout_guided_n0_v1"
RESULT_FIELDS = list(TrajectoryResult.__annotations__.keys())
ACCEPTANCE_FIELDS = [
    "case",
    "selected_solver",
    "final_n",
    "moves",
    "runtime_sec",
    "exact_moves",
    "move_gap",
    "qubo_used",
    "selected_reason",
    "business_ok",
    "goal_validation_ok",
    "validation_msg",
    "moves_path",
]


@dataclass
class MemorySearchResult:
    case: str
    status: str
    moves: list[Move]
    runtime_sec: float
    expanded: int
    generated: int
    visited: int
    open_count: int
    pruned_bound: int
    pruned_business: int
    dominated: int
    stale_pops: int
    peak_open: int
    flushes: int
    phase: str
    phases_tried: int
    qubo_calls: int = 0
    qubo_vars: int = 0
    qubo_time_sec: float = 0.0


@dataclass(frozen=True)
class ActionSuccessor:
    code: bytes
    n_value: int
    src: int
    dst: int
    box: int
    src_height: int
    dst_height: int
    delta_n: int
    src_after_n: int
    dst_after_n: int


def _priority(n_value: int, depth: int, args: argparse.Namespace) -> float:
    bonus = float(getattr(args, "priority_bonus", 0.0))
    if args.priority == "astar":
        return float(depth + n_value) - bonus
    if args.priority == "weighted_astar":
        return float(args.g_weight * depth + args.n_weight * n_value) - bonus
    return float(n_value * 1_000_000 + depth) - bonus


def _solver_name(args: argparse.Namespace) -> str:
    if not getattr(args, "use_qubo", False):
        return BASE_SOLVER_NAME
    profile = getattr(args, "qubo_profile", "frontier_cohort")
    if profile == "quality_guard":
        return QUALITY_GUARD_QUBO_SOLVER_NAME
    if profile == "frontier_cohort":
        return FRONTIER_COHORT_QUBO_SOLVER_NAME
    if profile == "action_order":
        return ACTION_ORDER_QUBO_SOLVER_NAME
    if profile == "tail_plan":
        return TAIL_PLAN_QUBO_SOLVER_NAME
    if profile == "hybrid_deep":
        return HYBRID_DEEP_QUBO_SOLVER_NAME
    if profile == "layout_guided":
        return LAYOUT_GUIDED_QUBO_SOLVER_NAME
    if profile == "broad_quality":
        return BROAD_QUBO_SOLVER_NAME
    return QUBO_SOLVER_NAME


def _uses_quality_qubo(args: argparse.Namespace) -> bool:
    return getattr(args, "qubo_profile", "frontier_cohort") in {
        "broad_quality",
        "quality_guard",
        "frontier_cohort",
        "tail_plan",
        "hybrid_deep",
        "layout_guided",
    }


def _uses_action_qubo(args: argparse.Namespace, current_n: int) -> bool:
    profile = getattr(args, "qubo_profile", "quality_guard")
    if profile == "action_order":
        return True
    if profile in {"tail_plan", "hybrid_deep"}:
        return current_n <= args.action_qubo_tail_n
    return False


def _candidate_sort_key(item: tuple[bytes, tuple[int, int, bytes]], args: argparse.Namespace) -> tuple:
    code, (n_value, depth, parent_code) = item
    return (_priority(n_value, depth, args), n_value, depth, parent_code, code)


def _candidate_similarity(a: bytes, b: bytes) -> float:
    if not a or not b:
        return 0.0
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / min(len(a), len(b))


def _uses_layout_qubo(args: argparse.Namespace) -> bool:
    return getattr(args, "use_qubo", False) and getattr(args, "qubo_profile", "") == "layout_guided"


def _target_groups_from_assignment(instance, assignment: dict[int, int]) -> list[frozenset[int]]:
    groups: list[frozenset[int]] = []
    for stack_idx in range(instance.stack_count):
        groups.append(frozenset(box for box, target_stack in assignment.items() if target_stack == stack_idx))
    return groups


def _chunk_boxes(chunk: bytes, max_height: int) -> tuple[int, ...]:
    height = _height(chunk, max_height)
    return tuple(int(box) for box in chunk[:height] if box)


def _chunk_layout_cost(boxes: tuple[int, ...], target_groups: list[frozenset[int]]) -> float:
    if not boxes or not target_groups:
        return 0.0
    group_cost = min(sum(1 for box in boxes if box not in group) for group in target_groups)
    inversion_cost = sum(1 for idx in range(1, len(boxes)) if boxes[idx] > boxes[idx - 1])
    return float(group_cost + 0.35 * inversion_cost)


def _code_layout_cost(code: bytes, args: argparse.Namespace) -> float:
    target_groups = getattr(args, "_layout_target_groups", None)
    stack_count = getattr(args, "_stack_count", None)
    max_height = getattr(args, "_max_height", None)
    if not target_groups or stack_count is None or max_height is None:
        return 0.0
    return sum(
        _chunk_layout_cost(_chunk_boxes(chunk, max_height), target_groups)
        for chunk in _split(code, stack_count, max_height)
    )


def _layout_qubo_namespace(args: argparse.Namespace) -> argparse.Namespace:
    data = vars(args).copy()
    defaults = {
        "assignment_penalty": 500.0,
        "capacity_penalty": 450.0,
        "move_weight": 0.35,
        "keep_weight": 0.60,
        "settled_break_weight": 3.0,
        "stack_order_weight": 0.08,
        "empty_slot_weight": 0.04,
        "conflict_weight": 4.0,
        "qubo_reads": getattr(args, "layout_qubo_reads", 16),
        "qubo_time_limit": getattr(args, "layout_qubo_time_limit", 0.04),
        "qubo_iterations": getattr(args, "layout_qubo_iterations", 240),
    }
    for key, value in defaults.items():
        data[key] = value
    return SimpleNamespace(**data)


def _init_layout_qubo_target(instance, args: argparse.Namespace) -> None:
    if not _uses_layout_qubo(args) or getattr(args, "_layout_qubo_initialized", False):
        return
    args._layout_qubo_initialized = True
    args._layout_target_groups = []
    args._layout_qubo_calls = 0
    args._layout_qubo_vars = 0
    args._layout_qubo_time_sec = 0.0
    args._layout_qubo_repairs = 0
    args._layout_target_cost = 0.0
    args._stack_count = instance.stack_count
    args._max_height = instance.max_height

    layout_args = _layout_qubo_namespace(args)
    model, var_for = TargetLayoutQuboBuilder().build(instance, layout_args)
    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        QuboSamplerConfig(
            num_reads=max(1, args.layout_qubo_reads),
            random_seed=args.seed,
            time_limit_sec=max(0.001, args.layout_qubo_time_limit),
            sa_iterations=max(1, args.layout_qubo_iterations),
        ),
    )
    decoded = [
        _decode_target(instance, sample, var_for, idx)
        for idx, sample in enumerate(sample_set.samples[: max(1, args.layout_target_samples)])
    ]
    decoded.sort(key=lambda target: (target.repairs, target.target_cost, target.energy, target.sample_index))
    if not decoded:
        return

    target = decoded[0]
    args._layout_target_groups = _target_groups_from_assignment(instance, target.assignment)
    args._layout_qubo_calls = 1
    args._layout_qubo_vars = len(model.var_index)
    args._layout_qubo_time_sec = sample_set.runtime_sec
    args._layout_qubo_repairs = target.repairs
    args._layout_target_cost = target.target_cost


def _legacy_qubo_pair_penalty(
    a: tuple[int, int, bytes, bytes],
    b: tuple[int, int, bytes, bytes],
    diversity_weight: float,
) -> float:
    a_n, a_depth, a_parent, a_code = a
    b_n, b_depth, b_parent, b_code = b
    penalty = 0.0
    if a_parent == b_parent:
        penalty += 0.65
    if a_n == b_n:
        penalty += 0.10
    if a_depth == b_depth:
        penalty += 0.05
    similarity = _candidate_similarity(a_code, b_code)
    if similarity > 0.80:
        penalty += similarity - 0.80
    return diversity_weight * penalty


def _stack_signature_similarity(a_code: bytes, b_code: bytes, args: argparse.Namespace | None) -> float:
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return _candidate_similarity(a_code, b_code)
    chunks_a = _split(a_code, stack_count, max_height)
    chunks_b = _split(b_code, stack_count, max_height)
    if not chunks_a or not chunks_b:
        return 0.0
    total = 0.0
    for chunk_a, chunk_b in zip(chunks_a, chunks_b):
        height_a = _height(chunk_a, max_height)
        height_b = _height(chunk_b, max_height)
        top_a = chunk_a[height_a - 1] if height_a else 0
        top_b = chunk_b[height_b - 1] if height_b else 0
        score = 0.0
        if height_a == height_b:
            score += 0.25
        if top_a == top_b:
            score += 0.30
        if _stack_n(chunk_a, max_height) == _stack_n(chunk_b, max_height):
            score += 0.25
        if max_height - height_a == max_height - height_b:
            score += 0.20
        total += score
    structural = total / min(len(chunks_a), len(chunks_b))
    return max(structural, _candidate_similarity(a_code, b_code))


def _qubo_pair_penalty(
    a: tuple[int, int, bytes, bytes],
    b: tuple[int, int, bytes, bytes],
    diversity_weight: float,
    args: argparse.Namespace | None = None,
) -> float:
    policy = getattr(args, "qubo_pair_policy", "baseline") if args is not None else "baseline"
    if policy == "baseline":
        return _legacy_qubo_pair_penalty(a, b, diversity_weight)
    if policy == "none":
        return 0.0

    a_n, a_depth, a_parent, a_code = a
    b_n, b_depth, b_parent, b_code = b
    parent_weight = float(getattr(args, "qubo_parent_sim_weight", 0.65)) if args is not None else 0.65
    layout_weight = float(getattr(args, "qubo_layout_sim_weight", 1.0)) if args is not None else 1.0
    stack_weight = float(getattr(args, "qubo_stack_sim_weight", 0.35)) if args is not None else 0.35
    threshold = float(getattr(args, "qubo_sim_threshold", 0.80)) if args is not None else 0.80

    penalty = 0.0
    if policy in {"parent_only", "hybrid"} and a_parent == b_parent:
        penalty += parent_weight
    if policy in {"byte_similarity", "hybrid"}:
        similarity = _candidate_similarity(a_code, b_code)
        if similarity > threshold:
            penalty += layout_weight * (similarity - threshold)
    if policy in {"stack_signature", "hybrid"}:
        similarity = _stack_signature_similarity(a_code, b_code, args)
        if similarity > threshold:
            penalty += stack_weight * (similarity - threshold)
    if policy == "hybrid":
        if a_n == b_n:
            penalty += 0.10
        if a_depth == b_depth:
            penalty += 0.05
    return diversity_weight * penalty


def _select_qubo_boosted(
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
) -> tuple[set[bytes], int, float]:
    if not getattr(args, "use_qubo", False) or len(candidates) < args.qubo_min_candidates:
        return set(), 0, 0.0
    target = max(0, min(args.qubo_supplement_width, len(candidates)))
    if target <= 0:
        return set(), 0, 0.0

    ranked_items = sorted(candidates.items(), key=lambda item: _candidate_sort_key(item, args))
    core_codes = {code for code, _data in ranked_items[: max(0, args.qubo_core_width)]}
    supplement = [(code, data) for code, data in ranked_items if code not in core_codes]
    if not supplement:
        return set(), 0, 0.0

    if _uses_quality_qubo(args):
        best_n = ranked_items[0][1][0]
        best_depth = ranked_items[0][1][1]
        filtered = [
            (code, data)
            for code, data in supplement
            if data[0] <= best_n + args.qubo_frontier_n_window
            and data[1] <= best_depth + args.qubo_frontier_depth_window
        ]
        if len(filtered) >= target:
            supplement = filtered

    usable_items = supplement[: args.max_qubo_vars]
    target = min(target, len(usable_items))
    if len(usable_items) <= target:
        return {code for code, _data in usable_items}, 0, 0.0

    score_norm = _qubo_linear_scores(usable_items, ranked_items[0][1], args)
    payload = [
        (data[0], data[1], data[2], code)
        for code, data in usable_items
    ]

    model = QuboModel()
    vars_ = [model.add_binary_var(f"m_{idx}") for idx in range(len(usable_items))]
    for idx, var in enumerate(vars_):
        model.add_linear(var, score_norm[idx])
    pair_penalties = _pair_penalty_matrix(payload, args.qubo_diversity_weight, args)
    for i in range(len(usable_items)):
        for j in range(i + 1, len(usable_items)):
            coeff = pair_penalties[i][j]
            if coeff:
                model.add_quadratic(vars_[i], vars_[j], coeff)
    model.add_exactly_k_penalty(vars_, target, args.qubo_penalty_a)

    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        QuboSamplerConfig(
            num_reads=args.qubo_reads,
            random_seed=args.seed,
            time_limit_sec=max(0.001, args.qubo_time_limit),
            sa_iterations=max(1, args.qubo_iterations),
            respect_time_limit=not bool(getattr(args, "deterministic_qubo_sampler", False)),
        ),
    )

    best: set[int] | None = None
    best_value = float("inf")
    for sample in sample_set.samples:
        selected = {idx for idx in range(len(usable_items)) if sample.binary_values.get(idx, 0) == 1}
        selected = _repair_qubo_selection_with_pairs(selected, target, score_norm, pair_penalties)
        value = _qubo_selection_value_with_pairs(selected, score_norm, pair_penalties)
        if value < best_value:
            best = selected
            best_value = value
    if best is None:
        best = _repair_qubo_selection_with_pairs(set(), target, score_norm, pair_penalties)
    return {usable_items[idx][0] for idx in best}, len(model.var_index), sample_set.runtime_sec


def _qubo_linear_scores(
    usable_items: list[tuple[bytes, tuple[int, int, bytes]]],
    best_data: tuple[int, int, bytes],
    args: argparse.Namespace,
) -> list[float]:
    if not _uses_quality_qubo(args):
        raw_scores = [idx for idx, _item in enumerate(usable_items)]
        hi = max(raw_scores) if raw_scores else 0
        return [0.0 if hi == 0 else score / hi for score in raw_scores]

    best_n, best_depth, _best_parent = best_data
    denom_rank = max(1, len(usable_items) - 1)
    denom_n = max(1, args.qubo_frontier_n_window)
    denom_depth = max(1, args.qubo_frontier_depth_window)
    parent_counts: dict[bytes, int] = {}
    for _code, data in usable_items:
        parent_counts[data[2]] = parent_counts.get(data[2], 0) + 1
    max_parent_count = max(parent_counts.values()) if parent_counts else 1
    layout_costs = [_code_layout_cost(code, args) for code, _data in usable_items]
    min_layout = min(layout_costs) if layout_costs else 0.0
    max_layout = max(layout_costs) if layout_costs else 0.0
    layout_span = max(1.0, max_layout - min_layout)
    layout_weight = float(getattr(args, "layout_qubo_target_weight", 0.0)) if _uses_layout_qubo(args) else 0.0
    next_n_weight = float(getattr(args, "qubo_next_n_weight", 0.0))
    if next_n_weight > 0.0:
        next_best_n = [
            _one_step_best_n(code, data[0], args)
            for code, data in usable_items
        ]
        min_next_n = min(next_best_n) if next_best_n else best_n
        max_next_n = max(next_best_n) if next_best_n else best_n
        next_n_span = max(1.0, float(max_next_n - min_next_n))
    else:
        next_best_n = []
        min_next_n = best_n
        next_n_span = 1.0

    scores: list[float] = []
    for idx, (_code, (n_value, depth, parent_code)) in enumerate(usable_items):
        rank_term = idx / denom_rank
        n_term = max(0.0, (n_value - best_n) / denom_n)
        depth_term = max(0.0, (depth - best_depth) / denom_depth)
        parent_term = parent_counts[parent_code] / max_parent_count
        layout_term = (layout_costs[idx] - min_layout) / layout_span
        next_n_term = 0.0
        if next_n_weight > 0.0:
            next_n_term = max(0.0, (next_best_n[idx] - min_next_n) / next_n_span)
        scores.append(
            args.qubo_rank_weight * rank_term
            + args.qubo_n_weight * n_term
            + args.qubo_depth_weight * depth_term
            + args.qubo_parent_weight * parent_term
            + layout_weight * layout_term
            + next_n_weight * next_n_term
        )
    hi = max(scores) if scores else 0.0
    return [0.0 if hi <= 0.0 else score / hi for score in scores]


def _one_step_best_n(code: bytes, current_n: int, args: argparse.Namespace) -> int:
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return current_n
    cache = getattr(args, "_qubo_next_n_cache", None)
    if cache is None:
        cache = {}
        setattr(args, "_qubo_next_n_cache", cache)
    key = (code, current_n)
    if key in cache:
        return cache[key]
    best = current_n
    for _next_code, next_n in _successors_with_n(code, current_n, stack_count, max_height):
        if next_n < best:
            best = next_n
    cache[key] = best
    return best


def _qubo_selection_value(
    selected: set[int],
    score_norm: list[float],
    payload: list[tuple[int, int, bytes, bytes]],
    diversity_weight: float,
    args: argparse.Namespace | None = None,
) -> float:
    value = sum(score_norm[idx] for idx in selected)
    rows = sorted(selected)
    for pos, i in enumerate(rows):
        for j in rows[pos + 1 :]:
            value += _qubo_pair_penalty(payload[i], payload[j], diversity_weight, args)
    return value


def _pair_penalty_matrix(
    payload: list[tuple[int, int, bytes, bytes]],
    diversity_weight: float,
    args: argparse.Namespace | None = None,
) -> list[list[float]]:
    pair_penalties = [[0.0 for _ in payload] for _ in payload]
    for i in range(len(payload)):
        for j in range(i + 1, len(payload)):
            coeff = _qubo_pair_penalty(payload[i], payload[j], diversity_weight, args)
            pair_penalties[i][j] = coeff
            pair_penalties[j][i] = coeff
    return pair_penalties


def _qubo_selection_value_with_pairs(
    selected: set[int],
    score_norm: list[float],
    pair_penalties: list[list[float]],
) -> float:
    value = sum(score_norm[idx] for idx in selected)
    rows = sorted(selected)
    for pos, i in enumerate(rows):
        row = pair_penalties[i]
        for j in rows[pos + 1 :]:
            value += row[j]
    return value


def _repair_qubo_selection_with_pairs(
    selected: set[int],
    target: int,
    score_norm: list[float],
    pair_penalties: list[list[float]],
) -> set[int]:
    selected = set(selected)
    while len(selected) > target:
        selected.remove(
            min(
                selected,
                key=lambda idx: _qubo_selection_value_with_pairs(selected - {idx}, score_norm, pair_penalties),
            )
        )
    while len(selected) < target:
        missing = [idx for idx in range(len(score_norm)) if idx not in selected]
        if not missing:
            break
        selected.add(
            min(
                missing,
                key=lambda idx: _qubo_selection_value_with_pairs(selected | {idx}, score_norm, pair_penalties),
            )
        )
    return selected


def _repair_qubo_selection(
    selected: set[int],
    target: int,
    score_norm: list[float],
    payload: list[tuple[int, int, bytes, bytes]],
    diversity_weight: float,
) -> set[int]:
    selected = set(selected)
    while len(selected) > target:
        selected.remove(
            min(
                selected,
                key=lambda idx: _qubo_selection_value(selected - {idx}, score_norm, payload, diversity_weight),
            )
        )
    while len(selected) < target:
        missing = [idx for idx in range(len(score_norm)) if idx not in selected]
        if not missing:
            break
        selected.add(
            min(
                missing,
                key=lambda idx: _qubo_selection_value(selected | {idx}, score_norm, payload, diversity_weight),
            )
        )
    return selected


def _successors_with_n_and_action(
    code: bytes,
    current_n: int,
    stack_count: int,
    max_height: int,
) -> list[ActionSuccessor]:
    chunks = _split(code, stack_count, max_height)
    heights = [_height(chunk, max_height) for chunk in chunks]
    contrib = [_stack_n(chunk, max_height) for chunk in chunks]
    seen: set[bytes] = set()
    out: list[ActionSuccessor] = []
    for i, src_h in enumerate(heights):
        if src_h == 0:
            continue
        src_top = chunks[i][src_h - 1]
        for j, dst_h in enumerate(heights):
            if i == j or dst_h >= max_height:
                continue
            src = bytearray(chunks[i])
            dst = bytearray(chunks[j])
            src[src_h - 1] = 0
            dst[dst_h] = src_top
            src_b = bytes(src)
            dst_b = bytes(dst)
            src_after_n = _stack_n(src_b, max_height)
            dst_after_n = _stack_n(dst_b, max_height)
            nxt_n = current_n - contrib[i] - contrib[j] + src_after_n + dst_after_n
            nxt_chunks = list(chunks)
            nxt_chunks[i] = src_b
            nxt_chunks[j] = dst_b
            nxt = _canonical(nxt_chunks)
            if nxt in seen:
                continue
            seen.add(nxt)
            out.append(
                ActionSuccessor(
                    code=nxt,
                    n_value=nxt_n,
                    src=i,
                    dst=j,
                    box=src_top,
                    src_height=src_h,
                    dst_height=dst_h,
                    delta_n=nxt_n - current_n,
                    src_after_n=src_after_n,
                    dst_after_n=dst_after_n,
                )
            )
    return out


def _action_pair_penalty(a: ActionSuccessor, b: ActionSuccessor, args: argparse.Namespace) -> float:
    penalty = 0.0
    if a.src == b.src:
        penalty += 0.20
    if a.dst == b.dst:
        penalty += 0.65
    if a.src == b.dst or a.dst == b.src:
        penalty += 0.25
    if a.delta_n == b.delta_n:
        penalty += 0.10
    if _candidate_similarity(a.code, b.code) > 0.80:
        penalty += 0.15
    return args.action_qubo_diversity_weight * penalty


def _action_linear_scores(options: list[ActionSuccessor], current_n: int, args: argparse.Namespace) -> list[float]:
    if not options:
        return []
    best_n = min(option.n_value for option in options)
    best_delta = min(option.delta_n for option in options)
    max_dst_height = max(1, max(option.dst_height for option in options))
    scores: list[float] = []
    for option in options:
        n_term = max(0.0, option.n_value - best_n)
        delta_term = max(0.0, option.delta_n - best_delta)
        dst_height_term = option.dst_height / max_dst_height
        dirty_term = 1.0 if option.dst_after_n > 0 and option.delta_n >= 0 else 0.0
        release_term = -0.25 if option.delta_n < 0 else 0.0
        scores.append(
            args.action_qubo_n_weight * n_term
            + args.action_qubo_delta_weight * delta_term
            + args.action_qubo_height_weight * dst_height_term
            + args.action_qubo_dirty_weight * dirty_term
            + release_term
        )
    lo = min(scores)
    shifted = [score - lo for score in scores]
    hi = max(shifted)
    return [0.0 if hi <= 0.0 else score / hi for score in shifted]


def _select_qubo_actions(
    options: list[ActionSuccessor],
    current_n: int,
    args: argparse.Namespace,
) -> tuple[set[bytes], int, float]:
    if len(options) < args.action_qubo_min_options:
        return set(), 0, 0.0
    usable = sorted(options, key=lambda option: (option.n_value, option.delta_n, option.dst_height, option.src, option.dst))
    usable = usable[: args.action_qubo_max_vars]
    target = min(args.action_qubo_top_k, len(usable))
    if target <= 0:
        return set(), 0, 0.0
    if len(usable) <= target:
        return {option.code for option in usable}, 0, 0.0

    score_norm = _action_linear_scores(usable, current_n, args)
    model = QuboModel()
    vars_ = [model.add_binary_var(f"a_{idx}") for idx in range(len(usable))]
    for idx, var in enumerate(vars_):
        model.add_linear(var, score_norm[idx])
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            coeff = _action_pair_penalty(usable[i], usable[j], args)
            if coeff:
                model.add_quadratic(vars_[i], vars_[j], coeff)
    model.add_exactly_k_penalty(vars_, target, args.action_qubo_penalty_a)
    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        QuboSamplerConfig(
            num_reads=args.action_qubo_reads,
            random_seed=args.seed,
            time_limit_sec=max(0.001, args.action_qubo_time_limit),
            sa_iterations=max(1, args.action_qubo_iterations),
        ),
    )

    best: set[int] | None = None
    best_value = float("inf")
    payload = [(option.n_value, option.dst_height, bytes([option.src, option.dst]), option.code) for option in usable]
    for sample in sample_set.samples:
        selected = {idx for idx in range(len(usable)) if sample.binary_values.get(idx, 0) == 1}
        selected = _repair_qubo_selection(
            selected,
            target,
            score_norm,
            payload,
            args.action_qubo_diversity_weight,
        )
        value = _qubo_selection_value(selected, score_norm, payload, args.action_qubo_diversity_weight)
        if value < best_value:
            best = selected
            best_value = value
    if best is None:
        best = _repair_qubo_selection(set(), target, score_norm, payload, args.action_qubo_diversity_weight)
    return {usable[idx].code for idx in best}, len(model.var_index), sample_set.runtime_sec


def _ordered_successors(
    code: bytes,
    current_n: int,
    stack_count: int,
    max_height: int,
    args: argparse.Namespace,
) -> tuple[list[tuple[bytes, int]], int, int, float]:
    if (
        not _uses_action_qubo(args, current_n)
        or args._action_qubo_calls_done >= args.action_qubo_max_calls
    ):
        return list(_successors_with_n(code, current_n, stack_count, max_height)), 0, 0, 0.0

    options = _successors_with_n_and_action(code, current_n, stack_count, max_height)
    selected, qvars, qtime = _select_qubo_actions(options, current_n, args)
    if not qvars:
        return [(option.code, option.n_value) for option in options], 0, 0, 0.0
    args._action_qubo_calls_done += 1
    ordered = sorted(
        options,
        key=lambda option: (
            0 if option.code in selected else 1,
            option.n_value,
            option.delta_n,
            option.dst_height,
            option.src,
            option.dst,
        ),
    )
    return [(option.code, option.n_value) for option in ordered], 1, qvars, qtime


def _codes_to_moves(zone: str, bay: str, codes: list[bytes]) -> list[Move]:
    instance = _instance_from_single_bay(zone, bay)
    physical_rows = _lift_to_physical(instance.stacks, codes, instance.stack_count, instance.max_height)
    return [
        Move(
            step=int(row["step"]) - 1,
            box=int(row["box_rank"]),
            from_stack=int(row["from_stack"]),
            to_stack=int(row["to_stack"]),
            from_level=int(row["from_level"]),
            to_level=int(row["to_level"]),
        )
        for row in physical_rows
    ]


def _backtrack(parent: dict[bytes, bytes | None], target: bytes) -> list[bytes]:
    codes = [target]
    current = target
    while parent[current] is not None:
        current = parent[current]  # type: ignore[assignment]
        codes.append(current)
    codes.reverse()
    return codes


def _push_candidate(
    heap: list[tuple[float, int, int, int, bytes]],
    code: bytes,
    n_value: int,
    depth: int,
    counter: int,
    args: argparse.Namespace,
) -> None:
    heapq.heappush(heap, (_priority(n_value, depth, args), n_value, depth, counter, code))


def _flush_candidates(
    candidates: dict[bytes, tuple[int, int, bytes]],
    best_depth: dict[bytes, int],
    parent: dict[bytes, bytes | None],
    heap: list[tuple[float, int, int, int, bytes]],
    counter: int,
    args: argparse.Namespace,
) -> tuple[int, int, int, int, int, float]:
    pushed = dominated = 0
    qcalls = qvars = 0
    qtime = 0.0
    boosted: set[bytes] = set()
    if getattr(args, "use_qubo", False) and args._qubo_calls_done < args.max_qubo_calls:
        boosted, qvars, qtime = _select_qubo_boosted(candidates, args)
        if qvars:
            qcalls = 1
            args._qubo_calls_done += 1
    for code, (n_value, depth, parent_code) in candidates.items():
        old_depth = best_depth.get(code)
        if old_depth is not None and depth >= old_depth:
            dominated += 1
            continue
        best_depth[code] = depth
        parent[code] = parent_code
        counter += 1
        if code in boosted:
            boosted_args = _copy_args(args, priority_bonus=args.qubo_priority_bonus)
            _push_candidate(heap, code, n_value, depth, counter, boosted_args)
        else:
            _push_candidate(heap, code, n_value, depth, counter, args)
        pushed += 1
    candidates.clear()
    return pushed, dominated, counter, qcalls, qvars, qtime


def _search_case_once(case: str, zone: str, bay: str, args: argparse.Namespace) -> MemorySearchResult:
    start = time.perf_counter()
    deadline = start + args.time_limit if args.time_limit > 0 else math.inf
    instance = _instance_from_single_bay(zone, bay)
    _init_layout_qubo_target(instance, args)
    initial, n0 = _encode(instance.stacks, instance.stack_count, instance.max_height)
    cap = entropy_limit(instance, args.alpha)
    depth_cap = min(cap, args.max_moves) if args.max_moves is not None else cap

    best_depth: dict[bytes, int] = {initial: 0}
    parent: dict[bytes, bytes | None] = {initial: None}
    heap: list[tuple[float, int, int, int, bytes]] = []
    counter = 0
    _push_candidate(heap, initial, n0, 0, counter, args)
    args._qubo_calls_done = 0
    args._action_qubo_calls_done = 0

    expanded = generated = pruned_bound = pruned_business = dominated = stale_pops = 0
    qcalls = int(getattr(args, "_layout_qubo_calls", 0))
    qvars = int(getattr(args, "_layout_qubo_vars", 0))
    qtime = float(getattr(args, "_layout_qubo_time_sec", 0.0))
    peak_open = 1
    flushes = 0
    target: bytes | None = initial if n0 <= args.target_final_n else None

    while target is None and heap and time.perf_counter() < deadline:
        batch: list[tuple[bytes, int, int]] = []
        batch_limit = 1 if args.push_mode == "immediate" else args.pop_batch_size
        while heap and len(batch) < batch_limit:
            _prio, current_n, depth, _idx, code = heapq.heappop(heap)
            if depth != best_depth.get(code):
                stale_pops += 1
                continue
            batch.append((code, current_n, depth))
        if not batch:
            continue

        candidates: dict[bytes, tuple[int, int, bytes]] = {}
        for code, current_n, depth in batch:
            if time.perf_counter() >= deadline:
                break
            if current_n <= args.target_final_n:
                target = code
                break
            if _remaining_lower_bound(current_n, depth, depth_cap) > args.target_final_n:
                pruned_bound += 1
                continue
            if depth >= depth_cap:
                pruned_bound += 1
                continue

            expanded += 1
            next_depth = depth + 1
            successors, add_qcalls, add_qvars, add_qtime = _ordered_successors(
                code,
                current_n,
                instance.stack_count,
                instance.max_height,
                args,
            )
            qcalls += add_qcalls
            qvars += add_qvars
            qtime += add_qtime
            for nxt, nxt_n in successors:
                generated += 1
                if args.max_visited > 0 and len(best_depth) + len(candidates) >= args.max_visited:
                    heap.clear()
                    candidates.clear()
                    break
                if next_depth + nxt_n > cap:
                    pruned_business += 1
                    continue
                old_depth = best_depth.get(nxt)
                if old_depth is not None and next_depth >= old_depth:
                    dominated += 1
                    continue
                old_candidate = candidates.get(nxt)
                if old_candidate is not None and next_depth >= old_candidate[1]:
                    dominated += 1
                    continue
                if nxt_n <= args.target_final_n:
                    parent[nxt] = code
                    best_depth[nxt] = next_depth
                    target = nxt
                    candidates.clear()
                    break
                if _remaining_lower_bound(nxt_n, next_depth, depth_cap) > args.target_final_n:
                    pruned_bound += 1
                    continue
                if args.push_mode == "immediate":
                    best_depth[nxt] = next_depth
                    parent[nxt] = code
                    counter += 1
                    _push_candidate(heap, nxt, nxt_n, next_depth, counter, args)
                else:
                    candidates[nxt] = (nxt_n, next_depth, code)
                    if len(candidates) >= args.insert_batch_size:
                        _pushed, extra_dominated, counter, add_qcalls, add_qvars, add_qtime = _flush_candidates(
                            candidates,
                            best_depth,
                            parent,
                            heap,
                            counter,
                            args,
                        )
                        flushes += 1
                        dominated += extra_dominated
                        qcalls += add_qcalls
                        qvars += add_qvars
                        qtime += add_qtime
                        peak_open = max(peak_open, len(heap))
            if target is not None:
                break
        if candidates:
            _pushed, extra_dominated, counter, add_qcalls, add_qvars, add_qtime = _flush_candidates(
                candidates,
                best_depth,
                parent,
                heap,
                counter,
                args,
            )
            flushes += 1
            dominated += extra_dominated
            qcalls += add_qcalls
            qvars += add_qvars
            qtime += add_qtime
        peak_open = max(peak_open, len(heap))

    moves: list[Move] = []
    status = "TARGET_REACHABLE" if target is not None else "TIME_LIMIT_OR_MEMORY_LIMIT"
    if target is not None:
        moves = _codes_to_moves(zone, bay, _backtrack(parent, target))
    return MemorySearchResult(
        case=case,
        status=status,
        moves=moves,
        runtime_sec=time.perf_counter() - start,
        expanded=expanded,
        generated=generated,
        visited=len(best_depth),
        open_count=len(heap),
        pruned_bound=pruned_bound,
        pruned_business=pruned_business,
        dominated=dominated,
        stale_pops=stale_pops,
        peak_open=peak_open,
        flushes=flushes,
        phase=getattr(args, "phase_name", "single"),
        phases_tried=1,
        qubo_calls=qcalls,
        qubo_vars=qvars,
        qubo_time_sec=qtime,
    )


def _copy_args(args: argparse.Namespace, **updates) -> argparse.Namespace:
    data = vars(args).copy()
    data.update(updates)
    return SimpleNamespace(**data)  # type: ignore[return-value]


def _search_case(case: str, zone: str, bay: str, args: argparse.Namespace) -> MemorySearchResult:
    if _uses_layout_qubo(args):
        base_args = _copy_args(args)
        _init_layout_qubo_target(_instance_from_single_bay(zone, bay), base_args)
        for attr in (
            "_layout_target_groups",
            "_layout_qubo_calls",
            "_layout_qubo_vars",
            "_layout_qubo_time_sec",
            "_layout_qubo_repairs",
            "_layout_target_cost",
            "_stack_count",
            "_max_height",
        ):
            if hasattr(base_args, attr):
                setattr(args, attr, getattr(base_args, attr))
    else:
        base_args = args

    if args.phase_schedule == "single":
        return _search_case_once(case, zone, bay, _copy_args(base_args, phase_name="single"))

    total_start = time.perf_counter()
    total_limit = args.time_limit if args.time_limit > 0 else math.inf
    phases = [
        ("small_batch", args.phase1_time_limit, 1000, 20_000),
        ("medium_batch", args.phase2_time_limit, 2000, 30_000),
        ("large_batch", math.inf, args.pop_batch_size, args.insert_batch_size),
    ]
    aggregate = {
        "expanded": 0,
        "generated": 0,
        "visited": 0,
        "open_count": 0,
        "pruned_bound": 0,
        "pruned_business": 0,
        "dominated": 0,
        "stale_pops": 0,
        "peak_open": 0,
        "flushes": 0,
        "qubo_calls": 0,
        "qubo_vars": 0,
        "qubo_time_sec": 0.0,
    }
    last: MemorySearchResult | None = None
    tried = 0
    layout_qubo_accounted = False

    for phase_name, phase_limit, pop_batch_size, insert_batch_size in phases:
        elapsed = time.perf_counter() - total_start
        remaining = total_limit - elapsed
        if remaining <= 0:
            break
        tried += 1
        phase_args = _copy_args(
            base_args,
            phase_schedule="single",
            phase_name=phase_name,
            push_mode="batch",
            pop_batch_size=pop_batch_size,
            insert_batch_size=insert_batch_size,
            time_limit=min(float(phase_limit), remaining),
        )
        if _uses_layout_qubo(args) and layout_qubo_accounted:
            phase_args._layout_qubo_calls = 0
            phase_args._layout_qubo_vars = 0
            phase_args._layout_qubo_time_sec = 0.0
        result = _search_case_once(case, zone, bay, phase_args)
        if _uses_layout_qubo(args):
            layout_qubo_accounted = True
        last = result
        aggregate["expanded"] += result.expanded
        aggregate["generated"] += result.generated
        aggregate["visited"] += result.visited
        aggregate["open_count"] = result.open_count
        aggregate["pruned_bound"] += result.pruned_bound
        aggregate["pruned_business"] += result.pruned_business
        aggregate["dominated"] += result.dominated
        aggregate["stale_pops"] += result.stale_pops
        aggregate["peak_open"] = max(aggregate["peak_open"], result.peak_open)
        aggregate["flushes"] += result.flushes
        aggregate["qubo_calls"] += result.qubo_calls
        aggregate["qubo_vars"] += result.qubo_vars
        aggregate["qubo_time_sec"] += result.qubo_time_sec
        if result.status == "TARGET_REACHABLE":
            return MemorySearchResult(
                case=result.case,
                status=result.status,
                moves=result.moves,
                runtime_sec=time.perf_counter() - total_start,
                expanded=aggregate["expanded"],
                generated=aggregate["generated"],
                visited=aggregate["visited"],
                open_count=aggregate["open_count"],
                pruned_bound=aggregate["pruned_bound"],
                pruned_business=aggregate["pruned_business"],
                dominated=aggregate["dominated"],
                stale_pops=aggregate["stale_pops"],
                peak_open=aggregate["peak_open"],
                flushes=aggregate["flushes"],
                phase=phase_name,
                phases_tried=tried,
                qubo_calls=aggregate["qubo_calls"],
                qubo_vars=aggregate["qubo_vars"],
                qubo_time_sec=aggregate["qubo_time_sec"],
            )

    runtime = time.perf_counter() - total_start
    if last is None:
        return MemorySearchResult(
            case=case,
            status="TIME_LIMIT_OR_MEMORY_LIMIT",
            moves=[],
            runtime_sec=runtime,
            expanded=0,
            generated=0,
            visited=0,
            open_count=0,
            pruned_bound=0,
            pruned_business=0,
            dominated=0,
            stale_pops=0,
            peak_open=0,
            flushes=0,
            phase="none",
            phases_tried=tried,
            qubo_calls=0,
            qubo_vars=0,
            qubo_time_sec=0.0,
        )
    return MemorySearchResult(
        case=last.case,
        status=last.status,
        moves=last.moves,
        runtime_sec=runtime,
        expanded=aggregate["expanded"],
        generated=aggregate["generated"],
        visited=aggregate["visited"],
        open_count=aggregate["open_count"],
        pruned_bound=aggregate["pruned_bound"],
        pruned_business=aggregate["pruned_business"],
        dominated=aggregate["dominated"],
        stale_pops=aggregate["stale_pops"],
        peak_open=aggregate["peak_open"],
        flushes=aggregate["flushes"],
        phase=last.phase,
        phases_tried=tried,
        qubo_calls=aggregate["qubo_calls"],
        qubo_vars=aggregate["qubo_vars"],
        qubo_time_sec=aggregate["qubo_time_sec"],
    )


def _case_filter(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return [case if case.endswith("_single") else f"{case}_single" for case in args.cases]
    return [f"{zone}_{bay}_single" for zone, bay in discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))]


def _row_from_result(result: MemorySearchResult, zone: str, bay: str, args: argparse.Namespace) -> TrajectoryResult:
    instance = _instance_from_single_bay(zone, bay)
    initial = create_initial_state(instance)
    n0 = disorder_box_count(instance, initial)
    if result.moves:
        state = initial
        for move in result.moves:
            state = apply_move(instance, state, move)
        final_n = disorder_box_count(instance, state)
        moves_count = len(result.moves)
        max_g = max_entropy_along_moves(instance, result.moves)
    else:
        final_n = n0
        moves_count = None
        max_g = n0
    cap13 = entropy_limit(instance, 1.3)
    cap15 = entropy_limit(instance, 1.5)
    return TrajectoryResult(
        case=result.case,
        solver=_solver_name(args),
        n0=n0,
        cap_13=cap13,
        cap_15=cap15,
        moves=moves_count,
        final_n=final_n,
        reduction=n0 - final_n,
        g_end=(moves_count + final_n) if moves_count is not None else None,
        max_g=max_g,
        g_ok_13=max_g <= cap13,
        g_ok_15=max_g <= cap15,
        runtime_sec=result.runtime_sec,
        expanded_states=result.expanded,
        generated_states=result.generated,
        visited_states=result.visited,
        qubo_calls=result.qubo_calls,
        qubo_total_vars=result.qubo_vars,
        qubo_time_sec=result.qubo_time_sec,
        status=(
            f"{result.status}; solver={_solver_name(args)}; fresh_run=True; storage=memory; "
            f"priority={args.priority}; target_stop=N0; proof=not_optimality_proof; "
            f"open={result.open_count}; peak_open={result.peak_open}; stale_pops={result.stale_pops}; "
            f"phase_schedule={args.phase_schedule}; phase={result.phase}; phases_tried={result.phases_tried}; "
            f"qubo_used={args.use_qubo}; qubo_profile={args.qubo_profile}; "
            f"qubo_calls={result.qubo_calls}; qubo_vars={result.qubo_vars}; "
            f"qubo_time_sec={result.qubo_time_sec:.6f}; "
            f"qubo_core_width={args.qubo_core_width}; qubo_supplement_width={args.qubo_supplement_width}; "
            f"qubo_frontier_n_window={args.qubo_frontier_n_window}; "
            f"qubo_frontier_depth_window={args.qubo_frontier_depth_window}; "
            f"layout_qubo_target_weight={getattr(args, 'layout_qubo_target_weight', 0.0)}; "
            f"layout_target_cost={getattr(args, '_layout_target_cost', '')}; "
            f"layout_qubo_repairs={getattr(args, '_layout_qubo_repairs', '')}; "
            f"action_qubo_max_calls={args.action_qubo_max_calls}; "
            f"action_qubo_tail_n={args.action_qubo_tail_n}; "
            f"action_qubo_top_k={args.action_qubo_top_k}; "
            f"push_mode={args.push_mode}; pop_batch_size={args.pop_batch_size}; flushes={result.flushes}; "
            f"pruned_bound={result.pruned_bound}; pruned_business={result.pruned_business}; "
            f"dominated={result.dominated}; max_visited={args.max_visited}"
        ),
    )


def _validate_result(result: MemorySearchResult, zone: str, bay: str) -> tuple[bool, bool, str]:
    if not result.moves:
        return False, False, result.status
    instance = _instance_from_single_bay(zone, bay)
    solution = Solution(
        solver_name=_solver_name(argparse.Namespace(use_qubo=False)),
        feasible=True,
        optimal=False,
        objective=float(len(result.moves)),
        runtime_sec=result.runtime_sec,
        moves=result.moves,
        status=result.status,
    )
    goal_ok, goal_msg, _ = validate_solution(instance, solution)
    business_ok = max_entropy_along_moves(instance, result.moves) <= entropy_limit(instance, 1.3)
    return business_ok, goal_ok, goal_msg


def _write_rows(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_acceptance(
    rows: list[TrajectoryResult],
    validation: dict[str, tuple[bool, bool, str]],
    moves_paths: dict[str, Path],
    exact_refs: dict[str, int],
    path: Path,
) -> None:
    out = []
    for row in rows:
        business_ok, goal_ok, msg = validation[row.case]
        out.append(
            {
                "case": row.case,
                "selected_solver": row.solver,
                "final_n": row.final_n,
                "moves": row.moves,
                "runtime_sec": f"{row.runtime_sec:.6f}",
                "exact_moves": exact_refs.get(row.case, ""),
                "move_gap": _move_gap(row, exact_refs),
                "qubo_used": str(row.qubo_calls > 0),
                "selected_reason": row.status,
                "business_ok": str(business_ok),
                "goal_validation_ok": str(goal_ok),
                "validation_msg": msg,
                "moves_path": _rel(moves_paths.get(row.case, Path(""))),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ACCEPTANCE_FIELDS)
        writer.writeheader()
        writer.writerows(out)


def _write_report(rows: list[TrajectoryResult], output: Path, acceptance: Path, report: Path, exact_refs: dict[str, int]) -> None:
    solved = sum(1 for row in rows if row.final_n == 0)
    avg_runtime = sum(row.runtime_sec for row in rows) / len(rows) if rows else 0.0
    gaps = [
        row.moves - exact_refs[row.case]
        for row in rows
        if row.final_n == 0 and row.moves is not None and row.case in exact_refs
    ]
    lines = [
        f"# {rows[0].solver if rows else BASE_SOLVER_NAME}",
        "",
        "Fresh in-memory target A*/best-first benchmark. No SQLite and no witness reuse.",
        "",
        f"Result CSV: `{_rel(output)}`",
        f"Acceptance CSV: `{_rel(acceptance)}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| cases | {len(rows)} |",
        f"| N=0 solved | {solved}/{len(rows)} |",
        f"| avg runtime sec | {avg_runtime:.3f} |",
        f"| max runtime sec | {max((row.runtime_sec for row in rows), default=0.0):.3f} |",
        f"| avg move gap | {sum(gaps) / len(gaps) if gaps else 0.0:.3f} |",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def _cli_flag_supplied(flag: str) -> bool:
    prefix = f"{flag}="
    return any(arg == flag or arg.startswith(prefix) for arg in sys.argv[1:])


def _apply_qubo_profile_defaults(args: argparse.Namespace) -> None:
    profile_defaults = {
        "quality_guard": [
            ("max_qubo_calls", "--max-qubo-calls", 9),
            ("max_qubo_vars", "--max-qubo-vars", 48),
            ("qubo_core_width", "--qubo-core-width", 224),
            ("qubo_supplement_width", "--qubo-supplement-width", 6),
            ("qubo_min_candidates", "--qubo-min-candidates", 64),
            ("qubo_priority_bonus", "--qubo-priority-bonus", 0.0001),
            ("qubo_diversity_weight", "--qubo-diversity-weight", 0.08),
            ("qubo_frontier_n_window", "--qubo-frontier-n-window", 0),
            ("qubo_frontier_depth_window", "--qubo-frontier-depth-window", 1),
            ("qubo_rank_weight", "--qubo-rank-weight", 1.0),
            ("qubo_n_weight", "--qubo-n-weight", 2.0),
            ("qubo_depth_weight", "--qubo-depth-weight", 1.0),
            ("qubo_parent_weight", "--qubo-parent-weight", 0.15),
            ("qubo_time_limit", "--qubo-time-limit", 0.001),
            ("qubo_reads", "--qubo-reads", 2),
            ("qubo_iterations", "--qubo-iterations", 20),
            ("action_qubo_max_calls", "--action-qubo-max-calls", 0),
        ],
        "frontier_cohort": [
            ("max_qubo_calls", "--max-qubo-calls", 12),
            ("max_qubo_vars", "--max-qubo-vars", 64),
            ("qubo_core_width", "--qubo-core-width", 224),
            ("qubo_supplement_width", "--qubo-supplement-width", 8),
            ("qubo_min_candidates", "--qubo-min-candidates", 64),
            ("qubo_priority_bonus", "--qubo-priority-bonus", 0.0001),
            ("qubo_diversity_weight", "--qubo-diversity-weight", 0.10),
            ("qubo_frontier_n_window", "--qubo-frontier-n-window", 0),
            ("qubo_frontier_depth_window", "--qubo-frontier-depth-window", 1),
            ("qubo_time_limit", "--qubo-time-limit", 0.001),
            ("qubo_reads", "--qubo-reads", 2),
            ("qubo_iterations", "--qubo-iterations", 20),
            ("action_qubo_max_calls", "--action-qubo-max-calls", 0),
        ],
        "action_order": [
            ("max_qubo_calls", "--max-qubo-calls", 0),
            ("qubo_supplement_width", "--qubo-supplement-width", 0),
            ("action_qubo_max_calls", "--action-qubo-max-calls", 24),
            ("action_qubo_tail_n", "--action-qubo-tail-n", 99),
            ("action_qubo_max_vars", "--action-qubo-max-vars", 36),
            ("action_qubo_top_k", "--action-qubo-top-k", 3),
            ("action_qubo_min_options", "--action-qubo-min-options", 6),
            ("action_qubo_time_limit", "--action-qubo-time-limit", 0.001),
            ("action_qubo_reads", "--action-qubo-reads", 2),
            ("action_qubo_iterations", "--action-qubo-iterations", 20),
        ],
        "tail_plan": [
            ("max_qubo_calls", "--max-qubo-calls", 0),
            ("qubo_supplement_width", "--qubo-supplement-width", 0),
            ("action_qubo_max_calls", "--action-qubo-max-calls", 18),
            ("action_qubo_tail_n", "--action-qubo-tail-n", 8),
            ("action_qubo_max_vars", "--action-qubo-max-vars", 40),
            ("action_qubo_top_k", "--action-qubo-top-k", 4),
            ("action_qubo_min_options", "--action-qubo-min-options", 6),
            ("action_qubo_time_limit", "--action-qubo-time-limit", 0.001),
            ("action_qubo_reads", "--action-qubo-reads", 2),
            ("action_qubo_iterations", "--action-qubo-iterations", 20),
        ],
        "hybrid_deep": [
            ("max_qubo_calls", "--max-qubo-calls", 9),
            ("max_qubo_vars", "--max-qubo-vars", 48),
            ("qubo_core_width", "--qubo-core-width", 224),
            ("qubo_supplement_width", "--qubo-supplement-width", 6),
            ("qubo_min_candidates", "--qubo-min-candidates", 64),
            ("qubo_priority_bonus", "--qubo-priority-bonus", 0.0001),
            ("qubo_diversity_weight", "--qubo-diversity-weight", 0.08),
            ("qubo_frontier_n_window", "--qubo-frontier-n-window", 0),
            ("qubo_frontier_depth_window", "--qubo-frontier-depth-window", 1),
            ("qubo_time_limit", "--qubo-time-limit", 0.001),
            ("qubo_reads", "--qubo-reads", 2),
            ("qubo_iterations", "--qubo-iterations", 20),
            ("action_qubo_max_calls", "--action-qubo-max-calls", 12),
            ("action_qubo_tail_n", "--action-qubo-tail-n", 8),
            ("action_qubo_max_vars", "--action-qubo-max-vars", 40),
            ("action_qubo_top_k", "--action-qubo-top-k", 4),
            ("action_qubo_min_options", "--action-qubo-min-options", 6),
            ("action_qubo_time_limit", "--action-qubo-time-limit", 0.001),
            ("action_qubo_reads", "--action-qubo-reads", 2),
            ("action_qubo_iterations", "--action-qubo-iterations", 20),
        ],
        "layout_guided": [
            ("max_qubo_calls", "--max-qubo-calls", 6),
            ("max_qubo_vars", "--max-qubo-vars", 64),
            ("qubo_core_width", "--qubo-core-width", 224),
            ("qubo_supplement_width", "--qubo-supplement-width", 8),
            ("qubo_min_candidates", "--qubo-min-candidates", 64),
            ("qubo_priority_bonus", "--qubo-priority-bonus", 0.0001),
            ("qubo_diversity_weight", "--qubo-diversity-weight", 0.10),
            ("qubo_frontier_n_window", "--qubo-frontier-n-window", 0),
            ("qubo_frontier_depth_window", "--qubo-frontier-depth-window", 1),
            ("qubo_rank_weight", "--qubo-rank-weight", 0.80),
            ("qubo_n_weight", "--qubo-n-weight", 2.0),
            ("qubo_depth_weight", "--qubo-depth-weight", 0.80),
            ("qubo_parent_weight", "--qubo-parent-weight", 0.15),
            ("layout_qubo_target_weight", "--layout-qubo-target-weight", 0.10),
            ("layout_qubo_time_limit", "--layout-qubo-time-limit", 0.04),
            ("layout_qubo_reads", "--layout-qubo-reads", 16),
            ("layout_qubo_iterations", "--layout-qubo-iterations", 240),
            ("layout_target_samples", "--layout-target-samples", 6),
            ("qubo_time_limit", "--qubo-time-limit", 0.001),
            ("qubo_reads", "--qubo-reads", 2),
            ("qubo_iterations", "--qubo-iterations", 20),
            ("action_qubo_max_calls", "--action-qubo-max-calls", 0),
        ],
    }
    tuned_defaults = profile_defaults.get(args.qubo_profile, [])
    for attr, flag, value in tuned_defaults:
        if not _cli_flag_supplied(flag):
            setattr(args, attr, value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "target_astar_memory_n0.csv")
    parser.add_argument("--priority", choices=["n_then_depth", "astar", "weighted_astar"], default="n_then_depth")
    parser.add_argument("--g-weight", type=float, default=1.0)
    parser.add_argument("--n-weight", type=float, default=1.0)
    parser.add_argument("--target-final-n", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--time-limit", type=float, default=0.0, help="Per-case limit in seconds; 0 means no limit.")
    parser.add_argument("--max-visited", type=int, default=0, help="Per-case memory guard; 0 means no explicit guard.")
    parser.add_argument("--phase-schedule", choices=["small_medium_large", "single"], default="small_medium_large")
    parser.add_argument("--phase1-time-limit", type=float, default=8.0)
    parser.add_argument("--phase2-time-limit", type=float, default=35.0)
    parser.add_argument("--push-mode", choices=["batch", "immediate"], default="batch")
    parser.add_argument("--pop-batch-size", type=int, default=5000)
    parser.add_argument("--insert-batch-size", type=int, default=50000)
    parser.add_argument("--use-qubo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--qubo-profile",
        choices=[
            "protected",
            "broad_quality",
            "quality_guard",
            "frontier_cohort",
            "action_order",
            "tail_plan",
            "hybrid_deep",
            "layout_guided",
        ],
        default="frontier_cohort",
    )
    parser.add_argument("--max-qubo-calls", type=int, default=6)
    parser.add_argument("--max-qubo-vars", type=int, default=48)
    parser.add_argument("--qubo-core-width", type=int, default=96)
    parser.add_argument("--qubo-supplement-width", type=int, default=10)
    parser.add_argument("--qubo-min-candidates", type=int, default=128)
    parser.add_argument("--qubo-priority-bonus", type=float, default=0.0001)
    parser.add_argument("--qubo-diversity-weight", type=float, default=0.08)
    parser.add_argument(
        "--qubo-pair-policy",
        choices=["baseline", "none", "parent_only", "byte_similarity", "stack_signature", "hybrid"],
        default="baseline",
    )
    parser.add_argument("--qubo-parent-sim-weight", type=float, default=0.65)
    parser.add_argument("--qubo-layout-sim-weight", type=float, default=1.0)
    parser.add_argument("--qubo-action-sim-weight", type=float, default=0.0)
    parser.add_argument("--qubo-stack-sim-weight", type=float, default=0.35)
    parser.add_argument("--qubo-sim-threshold", type=float, default=0.80)
    parser.add_argument("--qubo-frontier-n-window", type=int, default=1)
    parser.add_argument("--qubo-frontier-depth-window", type=int, default=2)
    parser.add_argument("--qubo-rank-weight", type=float, default=1.0)
    parser.add_argument("--qubo-n-weight", type=float, default=2.0)
    parser.add_argument("--qubo-depth-weight", type=float, default=1.0)
    parser.add_argument("--qubo-parent-weight", type=float, default=0.15)
    parser.add_argument("--qubo-penalty-a", type=float, default=30.0)
    parser.add_argument("--qubo-time-limit", type=float, default=0.003)
    parser.add_argument("--qubo-reads", type=int, default=2)
    parser.add_argument("--qubo-iterations", type=int, default=40)
    parser.add_argument("--layout-qubo-target-weight", type=float, default=0.0)
    parser.add_argument("--layout-qubo-time-limit", type=float, default=0.04)
    parser.add_argument("--layout-qubo-reads", type=int, default=16)
    parser.add_argument("--layout-qubo-iterations", type=int, default=240)
    parser.add_argument("--layout-target-samples", type=int, default=6)
    parser.add_argument("--action-qubo-max-calls", type=int, default=0)
    parser.add_argument("--action-qubo-tail-n", type=int, default=8)
    parser.add_argument("--action-qubo-max-vars", type=int, default=40)
    parser.add_argument("--action-qubo-top-k", type=int, default=4)
    parser.add_argument("--action-qubo-min-options", type=int, default=6)
    parser.add_argument("--action-qubo-diversity-weight", type=float, default=0.12)
    parser.add_argument("--action-qubo-n-weight", type=float, default=2.0)
    parser.add_argument("--action-qubo-delta-weight", type=float, default=1.0)
    parser.add_argument("--action-qubo-height-weight", type=float, default=0.20)
    parser.add_argument("--action-qubo-dirty-weight", type=float, default=0.40)
    parser.add_argument("--action-qubo-penalty-a", type=float, default=30.0)
    parser.add_argument("--action-qubo-time-limit", type=float, default=0.001)
    parser.add_argument("--action-qubo-reads", type=int, default=2)
    parser.add_argument("--action-qubo-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()
    _apply_qubo_profile_defaults(args)

    output = args.output if args.output.is_absolute() else ROOT / args.output
    acceptance = output.with_name(output.stem + "_acceptance.csv")
    moves_dir = output.with_name(output.stem + "_moves")
    report = output.with_suffix(".md")
    moves_dir.mkdir(parents=True, exist_ok=True)

    cases = _case_filter(args)
    exact_refs = _exact_moves_by_case()
    rows: list[TrajectoryResult] = []
    validation: dict[str, tuple[bool, bool, str]] = {}
    moves_paths: dict[str, Path] = {}

    for idx, case in enumerate(cases, start=1):
        zone, bay = _parse_case(case)
        print(f"CASE_START {idx}/{len(cases)} {case} solver={_solver_name(args)} priority={args.priority}", flush=True)
        result = _search_case(case, zone, bay, args)
        row = _row_from_result(result, zone, bay, args)
        business_ok, goal_ok, msg = _validate_result(result, zone, bay)
        if result.moves:
            move_path = moves_dir / f"{case}_moves.csv"
            _write_explain_csv(_instance_from_single_bay(zone, bay), result.moves, move_path)
            moves_paths[case] = move_path
        rows.append(row)
        validation[case] = (business_ok, goal_ok, msg)
        _write_rows(rows, output)
        _write_acceptance(rows, validation, moves_paths, exact_refs, acceptance)
        _write_report(rows, output, acceptance, report, exact_refs)
        print(
            "CASE_DONE "
            f"{case} status={result.status} final_n={row.final_n} moves={row.moves} "
            f"runtime={row.runtime_sec:.3f}s valid={business_ok and goal_ok} "
            f"visited={result.visited} peak_open={result.peak_open} qcalls={result.qubo_calls}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
