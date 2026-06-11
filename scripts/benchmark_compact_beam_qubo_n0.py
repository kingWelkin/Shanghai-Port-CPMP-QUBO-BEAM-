"""Compact byte-state beam + QUBO benchmark for N=0 target search.

This is a fairer beam-speed comparison against the memory A* line: it uses the
same compact bytes state representation, incremental N successor generation,
and late route reconstruction.  The search policy remains beam-like: each
depth keeps only a bounded frontier, with QUBO allowed to force a small diverse
supplement into that next frontier without replacing the protected core.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam import _write_explain_csv  # noqa: E402
from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from benchmark_fast_qubo_integrated_v1 import _exact_moves_by_case, _move_gap, _parse_case, _rel  # noqa: E402
from benchmark_full_real_business_algorithms import TrajectoryResult  # noqa: E402
from benchmark_target_astar_memory_n0 import _select_qubo_boosted  # noqa: E402
from container_reshuffle.core import Instance, Move, Solution, apply_move, create_initial_state, validate_solution  # noqa: E402
from container_reshuffle.features import disorder_box_count, entropy_limit, max_entropy_along_moves  # noqa: E402
from container_reshuffle.qubo.model import QuboModel  # noqa: E402
from container_reshuffle.qubo.sampler import QuboSamplerConfig, SimulatedAnnealingQuboSampler  # noqa: E402
from run_business_exact_target_search_sqlite import (  # noqa: E402
    _encode,
    _remaining_lower_bound,
    _split,
    _stack_n,
    _successors_with_n,
)
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


SOLVER_NAME = "compact_beam_qubo_n0_v1"
RESULT_FIELDS = list(TrajectoryResult.__annotations__.keys())
COMPACT_EXTRA_FIELDS = [
    "profile",
    "solver_profile",
    "qubo_influence_profile",
    "selection_policy",
    "priority",
    "beam_width",
    "core_width",
    "parent_core_cap",
    "peak_frontier",
    "peak_candidates",
    "depth_reached",
    "open_count",
    "pruned_bound",
    "pruned_business",
    "dominated",
    "use_qubo",
    "qubo_mode",
    "qubo_focus_n",
    "qubo_first",
    "max_qubo_vars",
    "qubo_core_width",
    "qubo_supplement_width",
    "qubo_cohorts_per_layer",
    "qubo_selected_total",
    "qubo_selected_on_search_path",
    "qubo_selected_on_path",
    "qubo_ancestor_on_search_path",
    "qubo_ancestor_on_path",
    "group_qubo_supplement",
    "group_qubo_calls",
    "group_qubo_vars",
    "group_qubo_time_sec",
    "group_qubo_selected_total",
    "qubo_group_max_vars",
    "qubo_group_select",
    "qubo_group_take",
    "post_polish_attempts",
    "post_polish_improvements",
    "post_polish_moves_saved",
    "global_reservoir",
    "reservoir_size",
    "peak_reservoir",
    "reservoir_added_total",
    "reservoir_pulled_total",
    "stream_candidate_limit",
    "stream_candidate_trim_factor",
    "stream_pruned_candidates",
    "rank_pool_size",
    "rank_pool_min_candidates",
    "rank_pool_truncated_candidates",
]
COMPACT_RESULT_FIELDS = RESULT_FIELDS + [field for field in COMPACT_EXTRA_FIELDS if field not in RESULT_FIELDS]
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


_DYNAMIC_QUBO_PRESETS: dict[str, dict[str, int | float | bool]] = {
    "none": {},
    "legacy": {
        "dynamic_min_frontier": 1200,
        "dynamic_core_frontier": 1200,
        "dynamic_max_frontier": 4800,
        "dynamic_candidate_pool": 12000,
        "dynamic_qubo_keep_threshold": 0.15,
        "max_qubo_calls": 96,
        "max_group_qubo_calls": 96,
        "qubo_group_max_vars": 160,
        "qubo_group_select": 24,
        "qubo_group_take": 32,
    },
    "mid3200": {
        "dynamic_min_frontier": 1200,
        "dynamic_core_frontier": 1200,
        "dynamic_max_frontier": 3200,
        "dynamic_candidate_pool": 12000,
        "dynamic_qubo_keep_threshold": 0.15,
        "max_qubo_calls": 96,
        "max_group_qubo_calls": 96,
        "qubo_group_max_vars": 96,
        "qubo_group_select": 24,
        "qubo_group_take": 4,
    },
    "speed2000": {
        "dynamic_min_frontier": 1200,
        "dynamic_core_frontier": 1200,
        "dynamic_max_frontier": 3200,
        "dynamic_candidate_pool": 12000,
        "dynamic_qubo_keep_threshold": 0.15,
        "max_qubo_calls": 96,
        "max_group_qubo_calls": 96,
        "qubo_group_max_vars": 96,
        "qubo_group_select": 24,
        "qubo_group_take": 4,
        "stream_candidate_limit": 2000,
        "stream_candidate_trim_factor": 2.0,
    },
    "balanced2200": {
        "dynamic_min_frontier": 1200,
        "dynamic_core_frontier": 1200,
        "dynamic_max_frontier": 3200,
        "dynamic_candidate_pool": 12000,
        "dynamic_qubo_keep_threshold": 0.15,
        "max_qubo_calls": 96,
        "max_group_qubo_calls": 96,
        "qubo_group_max_vars": 96,
        "qubo_group_select": 24,
        "qubo_group_take": 4,
        "stream_candidate_limit": 2200,
        "stream_candidate_trim_factor": 2.0,
    },
    "mid2400": {
        "dynamic_min_frontier": 1200,
        "dynamic_core_frontier": 1200,
        "dynamic_max_frontier": 2400,
        "dynamic_candidate_pool": 12000,
        "dynamic_qubo_keep_threshold": 0.15,
        "max_qubo_calls": 96,
        "max_group_qubo_calls": 96,
        "qubo_group_max_vars": 96,
        "qubo_group_select": 24,
        "qubo_group_take": 4,
    },
}


_SOLVER_PROFILES: dict[str, dict[str, int | float | bool | str]] = {
    "speed_v2": {
        "priority": "n_then_depth",
        "g_weight": 1.0,
        "n_weight": 1.0,
        "beam_width": 1200,
        "core_width": 1080,
        "parent_core_cap": 4,
        "selection_policy": "enhanced",
        "use_qubo": True,
        "qubo_mode": "focused_multi",
        "qubo_influence_profile": "stable",
        "max_qubo_calls": 32,
        "max_qubo_vars": 64,
        "qubo_core_width": 224,
        "qubo_supplement_width": 24,
        "qubo_min_candidates": 64,
        "qubo_focus_n": 8,
        "qubo_first": False,
        "qubo_target_share": 0.50,
        "qubo_cohorts_per_layer": 2,
        "qubo_cohort_stride": 64,
        "group_qubo_supplement": False,
        "max_group_qubo_calls": 32,
        "qubo_group_max_vars": 96,
        "qubo_group_select": 24,
        "qubo_group_take": 4,
        "qubo_group_lookahead_weight": 0.0,
        "qubo_group_branch_weight": 0.0,
        "group_qubo_risk_gate": False,
        "dynamic_min_frontier": 128,
        "dynamic_core_frontier": 0,
        "dynamic_max_frontier": 0,
        "dynamic_candidate_pool": 4096,
        "dynamic_qubo_keep_threshold": 0.55,
        "dynamic_profile": "legacy",
        "global_reservoir": False,
        "stream_candidate_limit": 0,
        "stream_candidate_trim_factor": 2.0,
        "rank_pool_size": 0,
        "rank_pool_min_candidates": 0,
        "post_polish_qubo": True,
        "post_polish_passes": 2,
        "post_polish_focus_n": 8,
        "post_polish_tail_span": 14,
        "post_polish_max_window": 8,
        "post_polish_max_windows": 96,
        "post_polish_select_windows": 0,
        "post_polish_max_qubo_vars": 72,
        "post_polish_max_bridge_depth": 7,
        "post_polish_bridge_expanded": 20000,
        "post_polish_suffix_focus_n": 9,
        "post_polish_suffix_tail_span": 18,
        "post_polish_suffix_max_candidates": 48,
        "post_polish_suffix_select": 5,
        "post_polish_suffix_max_qubo_vars": 48,
        "post_polish_suffix_max_depth": 18,
        "post_polish_suffix_expanded": 60000,
        "post_polish_suffix_near_gap": 2,
        "post_polish_suffix_near_penalty": 0.30,
        "post_polish_overlap_penalty": 8.0,
        "post_polish_near_window_gap": 2,
        "post_polish_penalty_a": 18.0,
        "post_polish_qubo_time_limit": 0.001,
        "post_polish_qubo_reads": 2,
        "post_polish_qubo_iterations": 20,
        "tail_order_steps": 6,
        "seed": 37,
    },
}


def make_compact_args(**overrides: object) -> argparse.Namespace:
    """Build a programmatic args namespace matching the CLI defaults.

    Programmatic overrides are marked explicit so solver profiles cannot
    silently reset them when the shared search entrypoint is reused by other
    benchmark drivers.
    """

    explicit = set(overrides.pop("_profile_explicit_keys", set()) or set())
    explicit.update(overrides.keys())
    defaults: dict[str, object] = {
        **_SOLVER_PROFILES["speed_v2"],
        "solver_profile": "speed_v2",
        "cases": None,
        "bays": ["01", "26"],
        "output": ROOT / "docs" / "experiments" / "compact_beam_qubo_n0_v1.csv",
        "target_final_n": 0,
        "alpha": 1.3,
        "max_moves": None,
        "time_limit": 0.0,
        "qubo_profile": "frontier_cohort",
        "qubo_priority_bonus": 0.0001,
        "qubo_diversity_weight": 0.10,
        "qubo_frontier_n_window": 0,
        "qubo_frontier_depth_window": 1,
        "qubo_group_same_n_penalty": 0.04,
        "group_qubo_risk_min_candidates": 50000,
        "group_qubo_risk_parent_share": 0.12,
        "group_qubo_risk_plateau_window": 3,
        "group_qubo_risk_plateau_delta": 0,
        "reservoir_max_states": 50000,
        "reservoir_pull": 800,
        "reservoir_add_per_layer": 4000,
        "qubo_rank_weight": 1.0,
        "qubo_n_weight": 2.0,
        "qubo_depth_weight": 1.0,
        "qubo_parent_weight": 0.15,
        "qubo_penalty_a": 30.0,
        "qubo_time_limit": 0.001,
        "qubo_reads": 2,
        "qubo_iterations": 20,
    }
    defaults.update(overrides)
    defaults["_profile_explicit_keys"] = explicit
    return argparse.Namespace(**defaults)


_QUBO_INFLUENCE_PRESETS: dict[str, dict[str, int | float | bool]] = {
    "stable": {},
    "light": {
        "beam_width": 1200,
        "core_width": 960,
        "max_qubo_vars": 96,
        "qubo_supplement_width": 48,
        "qubo_cohorts_per_layer": 4,
        "qubo_focus_n": 12,
        "max_qubo_calls": 64,
        "qubo_time_limit": 0.002,
        "qubo_iterations": 30,
    },
    "medium": {
        "beam_width": 1200,
        "core_width": 840,
        "max_qubo_vars": 128,
        "qubo_supplement_width": 64,
        "qubo_cohorts_per_layer": 4,
        "qubo_focus_n": 12,
        "max_qubo_calls": 96,
        "qubo_time_limit": 0.002,
        "qubo_iterations": 30,
    },
    "strong": {
        "beam_width": 1200,
        "core_width": 720,
        "max_qubo_vars": 128,
        "qubo_supplement_width": 72,
        "qubo_cohorts_per_layer": 6,
        "qubo_focus_n": 16,
        "max_qubo_calls": 128,
        "qubo_time_limit": 0.003,
        "qubo_iterations": 40,
    },
    "early": {
        "beam_width": 1200,
        "core_width": 840,
        "max_qubo_vars": 128,
        "qubo_supplement_width": 64,
        "qubo_cohorts_per_layer": 4,
        "qubo_focus_n": -1,
        "max_qubo_calls": 96,
        "qubo_time_limit": 0.002,
        "qubo_iterations": 30,
    },
}


@dataclass
class CompactBeamResult:
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
    peak_frontier: int
    peak_candidates: int
    depth_reached: int
    qubo_calls: int = 0
    qubo_vars: int = 0
    qubo_time_sec: float = 0.0
    qubo_selected_total: int = 0
    qubo_selected_on_search_path: int = 0
    qubo_selected_on_path: int = 0
    qubo_ancestor_on_search_path: int = 0
    qubo_ancestor_on_path: int = 0
    group_qubo_calls: int = 0
    group_qubo_vars: int = 0
    group_qubo_time_sec: float = 0.0
    group_qubo_selected_total: int = 0
    post_polish_attempts: int = 0
    post_polish_improvements: int = 0
    post_polish_moves_saved: int = 0
    tail_n_trace: str = ""
    tail_ordered_trace: str = ""
    tail_ordered_gain: int = 0
    reservoir_size: int = 0
    peak_reservoir: int = 0
    reservoir_added_total: int = 0
    reservoir_pulled_total: int = 0
    stream_pruned_candidates: int = 0
    rank_pool_truncated_candidates: int = 0


@dataclass(frozen=True)
class PostPolishWindow:
    start: int
    end: int
    old_len: int
    min_n: int
    max_n: int
    score: float


@dataclass(frozen=True)
class PostPolishSuffix:
    start: int
    old_remaining: int
    n_value: int
    score: float


@dataclass
class PostPolishStats:
    qcalls: int = 0
    qvars: int = 0
    qtime: float = 0.0
    attempts: int = 0
    improvements: int = 0
    moves_saved: int = 0
    bridge_expanded: int = 0
    suffix_attempts: int = 0
    suffix_improvements: int = 0
    suffix_expanded: int = 0


def _case_filter(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return [case if case.endswith("_single") else f"{case}_single" for case in args.cases]
    return [f"{zone}_{bay}_single" for zone, bay in discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))]


def _flag_set(flag: str) -> bool:
    if not flag.startswith("--"):
        flag = f"--{flag}"
    needle = flag.lower()
    for token in sys.argv[1:]:
        token_low = token.lower()
        if token_low == needle or token_low.startswith(f"{needle}="):
            return True
        if token_low.startswith("--") and "=" in token_low:
            left, _right = token_low.split("=", 1)
            if left == needle:
                return True
    return False


def _flag_set_any(flags: str | tuple[str, ...]) -> bool:
    if isinstance(flags, str):
        flags = (flags,)
    return any(_flag_set(flag) for flag in flags)


def _profile_key_explicit(args: argparse.Namespace, key: str) -> bool:
    return key in set(getattr(args, "_profile_explicit_keys", set()))


def _apply_solver_profile(args: argparse.Namespace) -> None:
    profile = getattr(args, "solver_profile", "speed_v2")
    preset = _SOLVER_PROFILES.get(profile)
    if preset is None:
        raise ValueError(f"unknown solver profile: {profile}")
    explicit_flags: dict[str, str | tuple[str, ...]] = {
        "priority": "--priority",
        "g_weight": "--g-weight",
        "n_weight": "--n-weight",
        "beam_width": "--beam-width",
        "core_width": "--core-width",
        "parent_core_cap": "--parent-core-cap",
        "selection_policy": "--selection-policy",
        "use_qubo": ("--use-qubo", "--no-use-qubo"),
        "qubo_mode": "--qubo-mode",
        "qubo_influence_profile": "--qubo-influence-profile",
        "max_qubo_calls": "--max-qubo-calls",
        "max_qubo_vars": "--max-qubo-vars",
        "qubo_core_width": "--qubo-core-width",
        "qubo_supplement_width": "--qubo-supplement-width",
        "qubo_min_candidates": "--qubo-min-candidates",
        "qubo_focus_n": "--qubo-focus-n",
        "qubo_first": ("--qubo-first", "--no-qubo-first"),
        "qubo_target_share": "--qubo-target-share",
        "qubo_cohorts_per_layer": "--qubo-cohorts-per-layer",
        "qubo_cohort_stride": "--qubo-cohort-stride",
        "group_qubo_supplement": ("--group-qubo-supplement", "--no-group-qubo-supplement"),
        "max_group_qubo_calls": "--max-group-qubo-calls",
        "qubo_group_max_vars": "--qubo-group-max-vars",
        "qubo_group_select": "--qubo-group-select",
        "qubo_group_take": "--qubo-group-take",
        "qubo_group_lookahead_weight": "--qubo-group-lookahead-weight",
        "qubo_group_branch_weight": "--qubo-group-branch-weight",
        "group_qubo_risk_gate": ("--group-qubo-risk-gate", "--no-group-qubo-risk-gate"),
        "dynamic_min_frontier": "--dynamic-min-frontier",
        "dynamic_core_frontier": "--dynamic-core-frontier",
        "dynamic_max_frontier": "--dynamic-max-frontier",
        "dynamic_candidate_pool": "--dynamic-candidate-pool",
        "dynamic_qubo_keep_threshold": "--dynamic-qubo-keep-threshold",
        "dynamic_profile": "--dynamic-profile",
        "global_reservoir": ("--global-reservoir", "--no-global-reservoir"),
        "stream_candidate_limit": "--stream-candidate-limit",
        "stream_candidate_trim_factor": "--stream-candidate-trim-factor",
        "rank_pool_size": "--rank-pool-size",
        "rank_pool_min_candidates": "--rank-pool-min-candidates",
        "post_polish_qubo": ("--post-polish-qubo", "--no-post-polish-qubo"),
        "post_polish_passes": "--post-polish-passes",
        "post_polish_focus_n": "--post-polish-focus-n",
        "post_polish_tail_span": "--post-polish-tail-span",
        "post_polish_max_window": "--post-polish-max-window",
        "post_polish_max_windows": "--post-polish-max-windows",
        "post_polish_select_windows": "--post-polish-select-windows",
        "post_polish_max_qubo_vars": "--post-polish-max-qubo-vars",
        "post_polish_max_bridge_depth": "--post-polish-max-bridge-depth",
        "post_polish_bridge_expanded": "--post-polish-bridge-expanded",
        "post_polish_suffix_focus_n": "--post-polish-suffix-focus-n",
        "post_polish_suffix_tail_span": "--post-polish-suffix-tail-span",
        "post_polish_suffix_max_candidates": "--post-polish-suffix-max-candidates",
        "post_polish_suffix_select": "--post-polish-suffix-select",
        "post_polish_suffix_max_qubo_vars": "--post-polish-suffix-max-qubo-vars",
        "post_polish_suffix_max_depth": "--post-polish-suffix-max-depth",
        "post_polish_suffix_expanded": "--post-polish-suffix-expanded",
        "post_polish_suffix_near_gap": "--post-polish-suffix-near-gap",
        "post_polish_suffix_near_penalty": "--post-polish-suffix-near-penalty",
        "post_polish_overlap_penalty": "--post-polish-overlap-penalty",
        "post_polish_near_window_gap": "--post-polish-near-window-gap",
        "post_polish_penalty_a": "--post-polish-penalty-a",
        "post_polish_qubo_time_limit": "--post-polish-qubo-time-limit",
        "post_polish_qubo_reads": "--post-polish-qubo-reads",
        "post_polish_qubo_iterations": "--post-polish-qubo-iterations",
        "tail_order_steps": "--tail-order-steps",
        "seed": "--seed",
    }
    for key, value in preset.items():
        if _profile_key_explicit(args, key):
            continue
        if _flag_set_any(explicit_flags.get(key, f"--{key.replace('_', '-')}")):
            continue
        setattr(args, key, value)


def _apply_dynamic_profile(args: argparse.Namespace) -> None:
    profile = getattr(args, "dynamic_profile", "none")
    if profile == "none":
        return
    if getattr(args, "selection_policy", "enhanced") != "dynamic_qubo":
        return
    preset = _DYNAMIC_QUBO_PRESETS.get(profile)
    if not preset:
        return
    explicit_flags = {
        "dynamic_min_frontier": "--dynamic-min-frontier",
        "dynamic_core_frontier": "--dynamic-core-frontier",
        "dynamic_max_frontier": "--dynamic-max-frontier",
        "dynamic_candidate_pool": "--dynamic-candidate-pool",
        "dynamic_qubo_keep_threshold": "--dynamic-qubo-keep-threshold",
        "max_qubo_calls": "--max-qubo-calls",
        "max_group_qubo_calls": "--max-group-qubo-calls",
        "qubo_group_max_vars": "--qubo-group-max-vars",
        "qubo_group_select": "--qubo-group-select",
        "qubo_group_take": "--qubo-group-take",
        "stream_candidate_limit": "--stream-candidate-limit",
        "stream_candidate_trim_factor": "--stream-candidate-trim-factor",
    }
    for key, value in preset.items():
        if _flag_set(explicit_flags.get(key, f"--{key.replace('_', '-')}")):
            continue
        setattr(args, key, value)


def _apply_qubo_influence_profile(args: argparse.Namespace) -> None:
    profile = getattr(args, "qubo_influence_profile", "stable")
    preset = _QUBO_INFLUENCE_PRESETS.get(profile)
    if preset is None:
        raise ValueError(f"unknown QUBO influence profile: {profile}")
    explicit_flags = {
        "beam_width": "--beam-width",
        "core_width": "--core-width",
        "max_qubo_vars": "--max-qubo-vars",
        "qubo_supplement_width": "--qubo-supplement-width",
        "qubo_cohorts_per_layer": "--qubo-cohorts-per-layer",
        "qubo_focus_n": "--qubo-focus-n",
        "max_qubo_calls": "--max-qubo-calls",
        "qubo_time_limit": "--qubo-time-limit",
        "qubo_iterations": "--qubo-iterations",
    }
    for key, value in preset.items():
        if _flag_set(explicit_flags.get(key, f"--{key.replace('_', '-')}")):
            continue
        setattr(args, key, value)


def _candidate_rank(item: tuple[bytes, tuple[int, int, bytes]], args: argparse.Namespace) -> tuple:
    code, (n_value, depth, parent_code) = item
    if args.priority == "astar":
        primary = float(depth + n_value)
    elif args.priority == "weighted_astar":
        primary = float(args.g_weight * depth + args.n_weight * n_value)
    else:
        primary = float(n_value * 1_000_000 + depth)
    return (primary, n_value, depth, parent_code, code)


def _candidate_sort_key(item: tuple[bytes, tuple[int, int, bytes]], args: argparse.Namespace) -> tuple:
    if args.priority != "n_then_depth":
        return _candidate_rank(item, args)
    code, (n_value, depth, parent_code) = item
    return (n_value, depth, parent_code, code)


def _candidate_key_func(args: argparse.Namespace):
    if args.priority == "n_then_depth":
        return lambda item: (item[1][0], item[1][1], item[1][2], item[0])
    return lambda item: _candidate_rank(item, args)


def _rank_candidates(
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
) -> list[tuple[bytes, tuple[int, int, bytes]]]:
    rank_pool_size = max(0, int(getattr(args, "rank_pool_size", 0) or 0))
    min_candidates = max(0, int(getattr(args, "rank_pool_min_candidates", 0) or 0))
    if not hasattr(args, "_rank_pool_truncated_candidates"):
        args._rank_pool_truncated_candidates = 0
    key_func = _candidate_key_func(args)
    if rank_pool_size <= 0 or len(candidates) <= max(rank_pool_size, min_candidates):
        return sorted(candidates.items(), key=key_func)
    pool = heapq.nsmallest(rank_pool_size, candidates.items(), key=key_func)
    args._rank_pool_truncated_candidates += len(candidates) - len(pool)
    return sorted(pool, key=key_func)


def _repair_group_qubo_selection(linear: list[float], selected: set[int], target: int) -> set[int]:
    selected = set(selected)
    target = min(target, len(linear))
    while len(selected) > target:
        selected.remove(max(selected, key=lambda idx: (linear[idx], idx)))
    while len(selected) < target:
        missing = [idx for idx in range(len(linear)) if idx not in selected]
        if not missing:
            break
        selected.add(min(missing, key=lambda idx: (linear[idx], idx)))
    return selected


def _one_step_best_n(code: bytes, n_value: int, args: argparse.Namespace) -> int:
    if getattr(args, "qubo_group_lookahead_weight", 0.0) <= 0:
        return n_value
    cache: dict[bytes, int] = getattr(args, "_group_lookahead_cache", {})
    cached = cache.get(code)
    if cached is not None:
        return cached
    best = n_value
    for _nxt, nxt_n in _successors_with_n_cached(code, n_value, args._stack_count, args._max_height):
        if nxt_n < best:
            best = nxt_n
            if best == 0:
                break
    cache[code] = best
    args._group_lookahead_cache = cache
    return best


def _group_qubo_linear_score(rows: list[tuple[bytes, tuple[int, int, bytes]]], args: argparse.Namespace) -> float:
    best_code, best_data = rows[0]
    best_n, depth, _parent = best_data
    score = float(best_n * 1000 + depth)
    lookahead_weight = getattr(args, "qubo_group_lookahead_weight", 0.0)
    if lookahead_weight > 0:
        next_best_n = _one_step_best_n(best_code, best_n, args)
        score -= lookahead_weight * float(max(0, best_n - next_best_n)) * 1000.0
    branch_weight = getattr(args, "qubo_group_branch_weight", 0.0)
    if branch_weight > 0 and len(rows) > 1:
        score -= branch_weight * math.log1p(len(rows))
    return score


def _group_qubo_risk_active(args: argparse.Namespace) -> bool:
    if not getattr(args, "group_qubo_risk_gate", False):
        return True
    if getattr(args, "_layer_candidate_count", 0) >= getattr(args, "group_qubo_risk_min_candidates", 10**9):
        return True
    if getattr(args, "_layer_parent_top_share", 0.0) >= getattr(args, "group_qubo_risk_parent_share", 1.1):
        return True
    window = max(2, getattr(args, "group_qubo_risk_plateau_window", 3))
    history = getattr(args, "_best_n_history", [])
    if len(history) >= window:
        recent = history[-window:]
        if min(recent) == recent[0] and max(recent) - min(recent) <= getattr(args, "group_qubo_risk_plateau_delta", 0):
            return True
    return False


def _select_group_qubo_supplement(
    ranked_remaining: list[tuple[bytes, tuple[int, int, bytes]]],
    selected: dict[bytes, None],
    args: argparse.Namespace,
    remaining_capacity: int,
) -> tuple[list[bytes], int, float]:
    if remaining_capacity <= 0 or not ranked_remaining:
        return [], 0, 0.0
    groups: dict[bytes, list[tuple[bytes, tuple[int, int, bytes]]]] = {}
    for code, data in ranked_remaining:
        groups.setdefault(data[2], []).append((code, data))
    group_rows = sorted(
        groups.items(),
        key=lambda item: _candidate_sort_key(item[1][0], args),
    )[: max(1, args.qubo_group_max_vars)]
    if not group_rows:
        return [], 0, 0.0

    take_per_group = max(1, args.qubo_group_take)
    max_groups_by_capacity = max(1, math.ceil(remaining_capacity / take_per_group))
    target_groups = max(1, min(args.qubo_group_select, len(group_rows), max_groups_by_capacity))
    raw = []
    for _parent, rows in group_rows:
        raw.append(_group_qubo_linear_score(rows, args))
    lo, hi = min(raw), max(raw)
    linear = [0.0 if hi == lo else (value - lo) / (hi - lo) for value in raw]

    model = QuboModel()
    vars_ = [model.add_binary_var(f"g_{idx}") for idx in range(len(group_rows))]
    for idx, var in enumerate(vars_):
        model.add_linear(var, linear[idx])
    for i in range(len(group_rows)):
        ni = group_rows[i][1][0][1][0]
        for j in range(i + 1, len(group_rows)):
            nj = group_rows[j][1][0][1][0]
            if ni == nj:
                model.add_quadratic(vars_[i], vars_[j], args.qubo_group_same_n_penalty)
    model.add_exactly_k_penalty(vars_, target_groups, args.qubo_penalty_a)
    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        QuboSamplerConfig(
            num_reads=args.qubo_reads,
            random_seed=args.seed + 5003 + args._group_qubo_calls_done,
            time_limit_sec=max(0.001, args.qubo_time_limit),
            sa_iterations=max(1, args.qubo_iterations),
        ),
    )
    picked: set[int] = set()
    if sample_set.samples:
        picked = {idx for idx in range(len(group_rows)) if sample_set.samples[0].binary_values.get(idx, 0) == 1}
    picked = _repair_group_qubo_selection(linear, picked, target_groups)

    boosted: list[bytes] = []
    for idx in sorted(picked, key=lambda i: linear[i]):
        for code, _data in group_rows[idx][1][:take_per_group]:
            if code in selected or code in boosted:
                continue
            boosted.append(code)
            if len(boosted) >= remaining_capacity:
                break
        if len(boosted) >= remaining_capacity:
            break
    return boosted, len(model.var_index), sample_set.runtime_sec


def _select_dynamic_qubo_frontier(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
) -> tuple[list[bytes], set[bytes], int, int, float]:
    max_frontier = getattr(args, "dynamic_max_frontier", 0)
    if max_frontier <= 0:
        max_frontier = args.beam_width
    max_frontier = max(1, min(max_frontier, len(ranked)))
    min_frontier = max(1, min(getattr(args, "dynamic_min_frontier", 1), max_frontier))
    core_frontier = max(0, min(getattr(args, "dynamic_core_frontier", 0), max_frontier))
    selected: dict[bytes, None] = {}
    if core_frontier > 0 and args.parent_core_cap > 0:
        parent_counts: dict[bytes, int] = {}
        for code, data in ranked:
            parent_code = data[2]
            if parent_counts.get(parent_code, 0) >= args.parent_core_cap:
                continue
            selected.setdefault(code, None)
            parent_counts[parent_code] = parent_counts.get(parent_code, 0) + 1
            if len(selected) >= core_frontier:
                break
    for code, _data in ranked:
        selected.setdefault(code, None)
        if len(selected) >= core_frontier:
            break

    fallback_target = max(min_frontier, core_frontier)
    fallback = [code for code, _data in ranked[:fallback_target]]
    if (
        not args.use_qubo
        or len(ranked) < args.qubo_min_candidates
        or args._group_qubo_calls_done >= args.max_group_qubo_calls
    ):
        return fallback, set(), 0, 0, 0.0

    groups: dict[bytes, list[tuple[bytes, tuple[int, int, bytes]]]] = {}
    candidate_pool = max(1, min(getattr(args, "dynamic_candidate_pool", len(ranked)), len(ranked)))
    for code, data in ranked[:candidate_pool]:
        if code in selected:
            continue
        groups.setdefault(data[2], []).append((code, data))
    group_rows = sorted(
        groups.items(),
        key=lambda item: _candidate_sort_key(item[1][0], args),
    )[: max(1, args.qubo_group_max_vars)]
    if not group_rows:
        return fallback, set(), 0, 0, 0.0

    raw = [_group_qubo_linear_score(rows, args) for _parent, rows in group_rows]
    lo, hi = min(raw), max(raw)
    quality = [1.0 if hi == lo else 1.0 - ((value - lo) / (hi - lo)) for value in raw]
    keep_threshold = getattr(args, "dynamic_qubo_keep_threshold", 0.55)

    model = QuboModel()
    vars_ = [model.add_binary_var(f"dg_{idx}") for idx in range(len(group_rows))]
    for idx, var in enumerate(vars_):
        model.add_linear(var, keep_threshold - quality[idx])
    for i in range(len(group_rows)):
        ni = group_rows[i][1][0][1][0]
        for j in range(i + 1, len(group_rows)):
            nj = group_rows[j][1][0][1][0]
            if ni == nj:
                model.add_quadratic(vars_[i], vars_[j], args.qubo_group_same_n_penalty)

    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        QuboSamplerConfig(
            num_reads=args.qubo_reads,
            random_seed=args.seed + 7001 + args._group_qubo_calls_done,
            time_limit_sec=max(0.001, args.qubo_time_limit),
            sa_iterations=max(1, args.qubo_iterations),
        ),
    )
    picked: set[int] = set()
    if sample_set.samples:
        picked = {idx for idx in range(len(group_rows)) if sample_set.samples[0].binary_values.get(idx, 0) == 1}

    take_per_group = max(1, args.qubo_group_take)
    min_groups = max(1, math.ceil(min_frontier / take_per_group))
    while len(picked) < min_groups:
        missing = [idx for idx in range(len(group_rows)) if idx not in picked]
        if not missing:
            break
        picked.add(max(missing, key=lambda idx: (quality[idx], -idx)))

    selected_by_qubo: set[bytes] = set()
    for idx in sorted(picked, key=lambda i: (-quality[i], i)):
        for code, _data in group_rows[idx][1][:take_per_group]:
            if code in selected:
                continue
            selected[code] = None
            selected_by_qubo.add(code)
            if len(selected) >= max_frontier:
                break
        if len(selected) >= max_frontier:
            break

    for code, _data in ranked:
        if len(selected) >= min_frontier:
            break
        selected.setdefault(code, None)

    selected_codes = list(selected.keys())[:max_frontier]
    selected_set = set(selected_codes)
    qvars = len(model.var_index)
    qtime = sample_set.runtime_sec
    args._group_qubo_calls_done += 1
    args._group_qubo_vars += qvars
    args._group_qubo_time += qtime
    args._group_qubo_selected_total += len(selected_by_qubo & selected_set)
    return selected_codes, selected_by_qubo & selected_set, 1, qvars, qtime


def _select_next_frontier(
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
) -> tuple[list[bytes], set[bytes], int, int, float]:
    if max(0, int(getattr(args, "rank_pool_size", 0) or 0)) <= 0:
        ranked = sorted(candidates.items(), key=_candidate_key_func(args))
    else:
        ranked = _rank_candidates(candidates, args)
    if getattr(args, "selection_policy", "enhanced") == "classic":
        return [code for code, _data in ranked[: args.beam_width]], set(), 0, 0, 0.0
    if getattr(args, "selection_policy", "enhanced") == "dynamic_qubo":
        return _select_dynamic_qubo_frontier(ranked, args)

    selected: dict[bytes, None] = {}
    selected_by_qubo: set[bytes] = set()

    core_target = max(0, min(args.core_width, args.beam_width))

    def add_deterministic_until(target_len: int) -> None:
        target_len = max(0, min(target_len, args.beam_width))
        if args.parent_core_cap > 0:
            parent_counts: dict[bytes, int] = {}
            for code, data in ranked:
                if code in selected:
                    continue
                parent_code = data[2]
                if parent_counts.get(parent_code, 0) >= args.parent_core_cap:
                    continue
                selected.setdefault(code, None)
                parent_counts[parent_code] = parent_counts.get(parent_code, 0) + 1
                if len(selected) >= target_len:
                    break
        for code, _data in ranked:
            selected.setdefault(code, None)
            if len(selected) >= target_len:
                break

    if not args.qubo_first:
        add_deterministic_until(core_target)

    qcalls = qvars = 0
    qtime = 0.0
    best_n = ranked[0][1][0] if ranked else math.inf
    qubo_enabled = (
        args.use_qubo
        and len(candidates) >= args.qubo_min_candidates
        and args._qubo_calls_done < args.max_qubo_calls
        and (args.qubo_focus_n < 0 or best_n <= args.qubo_focus_n)
    )
    if qubo_enabled and args.qubo_mode == "supplement":
        boosted, qvars, qtime = _select_qubo_boosted(candidates, args)
        if qvars:
            qcalls = 1
            args._qubo_calls_done += 1
            for code in boosted:
                if code not in selected:
                    selected_by_qubo.add(code)
                selected.setdefault(code, None)
                if len(selected) >= args.beam_width:
                    break
    elif qubo_enabled and args.qubo_mode in {"focused_multi", "majority_multi"}:
        ranked_remaining = [(code, data) for code, data in ranked if code not in selected]
        cursor = 0
        cohorts_left = max(1, args.qubo_cohorts_per_layer)
        stride = max(1, args.qubo_cohort_stride or args.max_qubo_vars)
        target_qubo = args.beam_width
        if args.qubo_mode == "majority_multi":
            target_qubo = max(1, min(args.beam_width - len(selected), int(round(args.beam_width * args.qubo_target_share))))
        while (
            cohorts_left > 0
            and cursor < len(ranked_remaining)
            and len(selected) < args.beam_width
            and (args.qubo_mode != "majority_multi" or len(selected_by_qubo) < target_qubo)
            and args._qubo_calls_done < args.max_qubo_calls
        ):
            cohort = ranked_remaining[cursor : cursor + max(1, args.max_qubo_vars)]
            cursor += stride
            if len(cohort) < args.qubo_min_candidates:
                continue
            remaining = args.beam_width - len(selected)
            q_remaining = max(1, target_qubo - len(selected_by_qubo)) if args.qubo_mode == "majority_multi" else remaining
            qargs = SimpleNamespace(**vars(args))
            qargs.qubo_core_width = 0
            qargs.qubo_supplement_width = max(1, min(args.qubo_supplement_width, remaining, q_remaining, len(cohort) - 1))
            boosted, add_qvars, add_qtime = _select_qubo_boosted(dict(cohort), qargs)
            if not add_qvars:
                continue
            cohorts_left -= 1
            qcalls += 1
            qvars += add_qvars
            qtime += add_qtime
            args._qubo_calls_done += 1
            for code in boosted:
                if code not in selected:
                    selected_by_qubo.add(code)
                selected.setdefault(code, None)
                if len(selected) >= args.beam_width:
                    break

    best_n = ranked[0][1][0] if ranked else math.inf
    group_qubo_enabled = (
        args.use_qubo
        and getattr(args, "group_qubo_supplement", False)
        and len(candidates) >= args.qubo_min_candidates
        and args._group_qubo_calls_done < args.max_group_qubo_calls
        and (args.qubo_focus_n < 0 or best_n <= args.qubo_focus_n)
        and _group_qubo_risk_active(args)
        and len(selected) < args.beam_width
    )
    if group_qubo_enabled:
        ranked_remaining = [(code, data) for code, data in ranked if code not in selected]
        remaining = args.beam_width - len(selected)
        boosted, add_qvars, add_qtime = _select_group_qubo_supplement(
            ranked_remaining,
            selected,
            args,
            remaining,
        )
        if add_qvars:
            qcalls += 1
            qvars += add_qvars
            qtime += add_qtime
            args._group_qubo_calls_done += 1
            args._group_qubo_vars += add_qvars
            args._group_qubo_time += add_qtime
            for code in boosted:
                selected_by_qubo.add(code)
                args._group_qubo_selected_total += 1
                selected.setdefault(code, None)
                if len(selected) >= args.beam_width:
                    break

    if args.qubo_first:
        add_deterministic_until(len(selected) + core_target)

    for code, _data in ranked:
        selected.setdefault(code, None)
        if len(selected) >= args.beam_width:
            break
    selected_codes = list(selected.keys())[: args.beam_width]
    selected_set = set(selected_codes)
    return selected_codes, selected_by_qubo & selected_set, qcalls, qvars, qtime


def _backtrack(parent: dict[bytes, bytes | None], target: bytes) -> list[bytes]:
    codes = [target]
    current = target
    while parent[current] is not None:
        current = parent[current]  # type: ignore[assignment]
        codes.append(current)
    codes.reverse()
    return codes


def _code_n(code: bytes, stack_count: int, max_height: int) -> int:
    return sum(_stack_n(chunk, max_height) for chunk in _split(code, stack_count, max_height))


def _path_n_values(codes: list[bytes], stack_count: int, max_height: int) -> list[int]:
    return [_code_n(code, stack_count, max_height) for code in codes]


def _path_business_ok(codes: list[bytes], stack_count: int, max_height: int, cap: int) -> bool:
    for depth, code in enumerate(codes):
        if depth + _code_n(code, stack_count, max_height) > cap:
            return False
    return True


@lru_cache(maxsize=262144)
def _cached_height(chunk: bytes, max_height: int) -> int:
    zero = chunk.find(0)
    return max_height if zero < 0 else zero


@lru_cache(maxsize=262144)
def _cached_stack_n(chunk: bytes, max_height: int) -> int:
    height = _cached_height(chunk, max_height)
    for h in range(1, height):
        if chunk[h] > chunk[h - 1]:
            return height - h
    return 0


@lru_cache(maxsize=262144)
def _cached_chunk_meta(chunk: bytes, max_height: int) -> tuple[int, int]:
    return _cached_height(chunk, max_height), _cached_stack_n(chunk, max_height)


@lru_cache(maxsize=128)
def _zero_bytes(length: int) -> bytes:
    return bytes(length)


@lru_cache(maxsize=262144)
def _cached_pop_chunk(chunk: bytes, src_h: int, max_height: int) -> bytes:
    return chunk[: src_h - 1] + _zero_bytes(max_height - src_h + 1)


@lru_cache(maxsize=524288)
def _cached_push_chunk(chunk: bytes, dst_h: int, src_top: int, max_height: int) -> bytes:
    return chunk[:dst_h] + bytes((src_top,)) + _zero_bytes(max_height - dst_h - 1)


def _successors_with_n_cached(code: bytes, current_n: int, stack_count: int, max_height: int):
    chunks = tuple(code[i * max_height : (i + 1) * max_height] for i in range(stack_count))
    chunk_meta = [_cached_chunk_meta(chunk, max_height) for chunk in chunks]
    seen: set[bytes] = set()
    for i, (src_h, src_contrib) in enumerate(chunk_meta):
        if src_h == 0:
            continue
        src_chunk = chunks[i]
        src_top = chunks[i][src_h - 1]
        src_b = _cached_pop_chunk(src_chunk, src_h, max_height)
        src_after_n = _cached_stack_n(src_b, max_height)
        for j, (dst_h, dst_contrib) in enumerate(chunk_meta):
            if i == j or dst_h >= max_height:
                continue
            dst_chunk = chunks[j]
            dst_b = _cached_push_chunk(dst_chunk, dst_h, src_top, max_height)
            nxt_n = current_n - src_contrib - dst_contrib + src_after_n + _cached_stack_n(dst_b, max_height)
            nxt_chunks = list(chunks)
            nxt_chunks[i] = src_b
            nxt_chunks[j] = dst_b
            nxt = b"".join(sorted(nxt_chunks))
            if nxt in seen:
                continue
            seen.add(nxt)
            yield nxt, nxt_n


class _ReverseRank:
    __slots__ = ("rank",)

    def __init__(self, rank: tuple) -> None:
        self.rank = rank

    def __lt__(self, other: "_ReverseRank") -> bool:
        return self.rank > other.rank

    def __le__(self, other: "_ReverseRank") -> bool:
        return self.rank >= other.rank

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ReverseRank) and self.rank == other.rank


def _stream_worst_key(
    code: bytes,
    data: tuple[int, int, bytes],
    args: argparse.Namespace,
    seq: int,
) -> _ReverseRank:
    return _ReverseRank(_candidate_rank((code, data), args))


def _trim_stream_candidates(
    candidates: dict[bytes, tuple[int, int, bytes]],
    worst_heap: list[tuple[_ReverseRank, int, bytes, tuple[int, int, bytes]]],
    limit: int,
    args: argparse.Namespace,
) -> int:
    if limit <= 0:
        return 0
    pruned = 0
    while len(candidates) > limit:
        while worst_heap:
            _key, _seq, code, data = heapq.heappop(worst_heap)
            if candidates.get(code) == data:
                del candidates[code]
                pruned += 1
                break
        else:
            code, _data = max(candidates.items(), key=_candidate_key_func(args))
            del candidates[code]
            pruned += 1
    return pruned


def _peek_stream_worst(
    candidates: dict[bytes, tuple[int, int, bytes]],
    worst_heap: list[tuple[_ReverseRank, int, bytes, tuple[int, int, bytes]]],
) -> _ReverseRank | None:
    while worst_heap:
        key, _seq, code, data = worst_heap[0]
        if candidates.get(code) == data:
            return key
        heapq.heappop(worst_heap)
    return None


def _offer_stream_candidate(
    candidates: dict[bytes, tuple[int, int, bytes]],
    worst_heap: list[tuple[_ReverseRank, int, bytes, tuple[int, int, bytes]]],
    code: bytes,
    data: tuple[int, int, bytes],
    args: argparse.Namespace,
    seq: int,
) -> int:
    limit = max(0, getattr(args, "stream_candidate_limit", 0))
    candidate_key = _stream_worst_key(code, data, args, seq)
    if limit <= 0:
        candidates[code] = data
        return 0
    factor = max(1.0, float(getattr(args, "stream_candidate_trim_factor", 2.0)))
    trim_threshold = max(limit, int(limit * factor))
    if code not in candidates and len(candidates) >= trim_threshold:
        worst_key = _peek_stream_worst(candidates, worst_heap)
        if worst_key is not None and candidate_key <= worst_key:
            return 1
    candidates[code] = data
    heapq.heappush(worst_heap, (candidate_key, seq, code, data))
    if len(candidates) <= trim_threshold:
        return 0
    return _trim_stream_candidates(candidates, worst_heap, limit, args)


def _reservoir_trim(
    reservoir: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
) -> None:
    cap = max(0, getattr(args, "reservoir_max_states", 0))
    if cap <= 0 or len(reservoir) <= cap:
        return
    kept = sorted(reservoir.items(), key=_candidate_key_func(args))[:cap]
    reservoir.clear()
    reservoir.update(kept)


def _reservoir_pull(
    reservoir: dict[bytes, tuple[int, int, bytes]],
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
) -> int:
    if not getattr(args, "global_reservoir", False) or not reservoir:
        return 0
    take = max(0, getattr(args, "reservoir_pull", 0))
    if take <= 0:
        return 0
    pulled = 0
    for code, data in sorted(reservoir.items(), key=_candidate_key_func(args)):
        if code in candidates:
            continue
        candidates[code] = data
        del reservoir[code]
        pulled += 1
        if pulled >= take:
            break
    return pulled


def _post_polish_suffixes(
    codes: list[bytes],
    n_values: list[int],
    args: argparse.Namespace,
) -> list[PostPolishSuffix]:
    if len(codes) <= 3:
        return []
    last_move = len(codes) - 1
    tail_begin = max(0, last_move - max(1, args.post_polish_suffix_tail_span))
    suffixes: list[PostPolishSuffix] = []
    for start in range(tail_begin, last_move):
        old_remaining = last_move - start
        n_value = n_values[start]
        if old_remaining < 2 or n_value > args.post_polish_suffix_focus_n:
            continue
        if n_value > old_remaining - 1:
            continue
        low_n_bonus = max(0, args.post_polish_suffix_focus_n - n_value)
        score = float(old_remaining + 0.75 * low_n_bonus + 0.02 * start)
        suffixes.append(PostPolishSuffix(start, old_remaining, n_value, score))
    suffixes.sort(key=lambda s: (-s.score, s.start))
    return suffixes[: max(1, args.post_polish_suffix_max_candidates)]


def _select_post_polish_suffixes_qubo(
    suffixes: list[PostPolishSuffix],
    args: argparse.Namespace,
) -> tuple[list[PostPolishSuffix], int, float]:
    if not suffixes:
        return [], 0, 0.0
    target = max(1, min(args.post_polish_suffix_select, len(suffixes)))
    usable = suffixes[: max(1, min(args.post_polish_suffix_max_qubo_vars, len(suffixes)))]
    if len(usable) <= target:
        return usable, 0, 0.0

    max_score = max(item.score for item in usable) or 1.0
    model = QuboModel()
    vars_ = [model.add_binary_var(f"s_{idx}") for idx in range(len(usable))]
    for idx, item in enumerate(usable):
        model.add_linear(vars_[idx], -item.score / max_score)
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            distance = abs(usable[i].start - usable[j].start)
            if distance <= args.post_polish_suffix_near_gap:
                model.add_quadratic(vars_[i], vars_[j], args.post_polish_suffix_near_penalty)
    model.add_exactly_k_penalty(vars_, target, args.post_polish_penalty_a)
    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        QuboSamplerConfig(
            num_reads=args.post_polish_qubo_reads,
            random_seed=args.seed + 2003,
            time_limit_sec=max(0.001, args.post_polish_qubo_time_limit),
            sa_iterations=max(1, args.post_polish_qubo_iterations),
        ),
    )
    selected: set[int] = set()
    if sample_set.samples:
        selected = {idx for idx in range(len(usable)) if sample_set.samples[0].binary_values.get(idx, 0) == 1}

    repaired: list[int] = []
    for idx in sorted(selected, key=lambda i: -usable[i].score):
        if all(abs(usable[idx].start - usable[old].start) > args.post_polish_suffix_near_gap for old in repaired):
            repaired.append(idx)
        if len(repaired) >= target:
            break
    for idx in sorted(range(len(usable)), key=lambda i: -usable[i].score):
        if len(repaired) >= target:
            break
        if idx in repaired:
            continue
        if all(abs(usable[idx].start - usable[old].start) > args.post_polish_suffix_near_gap for old in repaired):
            repaired.append(idx)
    repaired.sort(key=lambda idx: usable[idx].start)
    return [usable[idx] for idx in repaired], len(model.var_index), sample_set.runtime_sec


def _suffix_shorter_path_to_n0(
    start: bytes,
    start_n: int,
    offset_depth: int,
    cap: int,
    stack_count: int,
    max_height: int,
    max_depth: int,
    deadline: float,
    args: argparse.Namespace,
) -> tuple[list[bytes] | None, int]:
    if max_depth <= 0 or start_n <= 0:
        return None, 0
    heap: list[tuple[float, int, int, bytes]] = [(float(start_n), start_n, 0, start)]
    parent: dict[bytes, bytes | None] = {start: None}
    depth_by_code: dict[bytes, int] = {start: 0}
    n_by_code: dict[bytes, int] = {start: start_n}
    expanded = 0

    while heap and time.perf_counter() < deadline and expanded < args.post_polish_suffix_expanded:
        _priority, current_n, depth, code = heapq.heappop(heap)
        if depth != depth_by_code.get(code):
            continue
        if depth >= max_depth:
            continue
        if current_n > max_depth - depth:
            continue
        expanded += 1
        next_depth = depth + 1
        for nxt, nxt_n in _successors_with_n_cached(code, current_n, stack_count, max_height):
            if offset_depth + next_depth + nxt_n > cap:
                continue
            if nxt_n > max_depth - next_depth:
                continue
            old_depth = depth_by_code.get(nxt)
            if old_depth is not None and next_depth >= old_depth:
                continue
            parent[nxt] = code
            depth_by_code[nxt] = next_depth
            n_by_code[nxt] = nxt_n
            if nxt_n == 0:
                return _backtrack(parent, nxt), expanded
            score = next_depth + 1_000_000 * nxt_n
            heapq.heappush(heap, (float(score), nxt_n, next_depth, nxt))
    return None, expanded


def _post_polish_windows(
    codes: list[bytes],
    n_values: list[int],
    args: argparse.Namespace,
) -> list[PostPolishWindow]:
    if len(codes) <= 3:
        return []
    last_move = len(codes) - 1
    tail_begin = max(0, last_move - max(0, args.post_polish_tail_span))
    windows: list[PostPolishWindow] = []
    max_len = max(3, args.post_polish_max_window)
    for old_len in range(3, max_len + 1):
        for start in range(0, last_move - old_len + 1):
            end = start + old_len
            segment_n = n_values[start : end + 1]
            min_n = min(segment_n)
            max_n = max(segment_n)
            if end < tail_begin and min_n > args.post_polish_focus_n:
                continue
            low_n_bonus = max(0, args.post_polish_focus_n - min_n)
            tail_bonus = 2.0 if start >= tail_begin else 0.0
            score = float(old_len + 0.40 * low_n_bonus + 0.10 * max_n + tail_bonus)
            windows.append(PostPolishWindow(start, end, old_len, min_n, max_n, score))
    windows.sort(key=lambda w: (-w.score, w.start, w.end))
    return windows[: max(1, args.post_polish_max_windows)]


def _windows_overlap(a: PostPolishWindow, b: PostPolishWindow) -> bool:
    return not (a.end <= b.start or b.end <= a.start)


def _select_post_polish_windows_qubo(
    windows: list[PostPolishWindow],
    args: argparse.Namespace,
) -> tuple[list[PostPolishWindow], int, float]:
    if not windows or args.post_polish_select_windows <= 0:
        return [], 0, 0.0
    target = max(1, min(args.post_polish_select_windows, len(windows)))
    usable = windows[: max(1, min(args.post_polish_max_qubo_vars, len(windows)))]
    if len(usable) <= target:
        return usable, 0, 0.0

    max_score = max(window.score for window in usable) or 1.0
    model = QuboModel()
    vars_ = [model.add_binary_var(f"w_{idx}") for idx in range(len(usable))]
    for idx, window in enumerate(usable):
        model.add_linear(vars_[idx], -window.score / max_score)
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            if _windows_overlap(usable[i], usable[j]):
                model.add_quadratic(vars_[i], vars_[j], args.post_polish_overlap_penalty)
            else:
                distance = min(abs(usable[i].start - usable[j].start), abs(usable[i].end - usable[j].end))
                if distance <= args.post_polish_near_window_gap:
                    model.add_quadratic(vars_[i], vars_[j], 0.15)
    model.add_exactly_k_penalty(vars_, target, args.post_polish_penalty_a)
    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        QuboSamplerConfig(
            num_reads=args.post_polish_qubo_reads,
            random_seed=args.seed + 1009,
            time_limit_sec=max(0.001, args.post_polish_qubo_time_limit),
            sa_iterations=max(1, args.post_polish_qubo_iterations),
        ),
    )

    selected: set[int] = set()
    if sample_set.samples:
        selected = {idx for idx in range(len(usable)) if sample_set.samples[0].binary_values.get(idx, 0) == 1}

    repaired: list[int] = []
    for idx in sorted(selected, key=lambda i: -usable[i].score):
        if all(not _windows_overlap(usable[idx], usable[old]) for old in repaired):
            repaired.append(idx)
        if len(repaired) >= target:
            break
    for idx in sorted(range(len(usable)), key=lambda i: -usable[i].score):
        if len(repaired) >= target:
            break
        if idx in repaired:
            continue
        if all(not _windows_overlap(usable[idx], usable[old]) for old in repaired):
            repaired.append(idx)
    repaired.sort(key=lambda idx: usable[idx].start, reverse=True)
    return [usable[idx] for idx in repaired], len(model.var_index), sample_set.runtime_sec


def _bridge_shorter_path(
    start: bytes,
    goal: bytes,
    offset_depth: int,
    cap: int,
    stack_count: int,
    max_height: int,
    max_depth: int,
    deadline: float,
    args: argparse.Namespace,
) -> tuple[list[bytes] | None, int]:
    if max_depth <= 0 or start == goal:
        return None, 0
    start_n = _code_n(start, stack_count, max_height)
    goal_n = _code_n(goal, stack_count, max_height)
    heap: list[tuple[float, int, bytes]] = [(abs(start_n - goal_n), 0, start)]
    parent: dict[bytes, bytes | None] = {start: None}
    depth_by_code: dict[bytes, int] = {start: 0}
    expanded = 0

    while heap and time.perf_counter() < deadline and expanded < args.post_polish_bridge_expanded:
        _priority, depth, code = heapq.heappop(heap)
        if depth != depth_by_code.get(code):
            continue
        if depth >= max_depth:
            continue
        current_n = _code_n(code, stack_count, max_height)
        expanded += 1
        next_depth = depth + 1
        for nxt, nxt_n in _successors_with_n_cached(code, current_n, stack_count, max_height):
            if offset_depth + next_depth + nxt_n > cap:
                continue
            old_depth = depth_by_code.get(nxt)
            if old_depth is not None and next_depth >= old_depth:
                continue
            parent[nxt] = code
            depth_by_code[nxt] = next_depth
            if nxt == goal:
                return _backtrack(parent, nxt), expanded
            score = next_depth + 0.35 * abs(nxt_n - goal_n) + 0.05 * nxt_n
            heapq.heappush(heap, (score, next_depth, nxt))
    return None, expanded


def _post_polish_codes_with_qubo(
    codes: list[bytes],
    stack_count: int,
    max_height: int,
    cap: int,
    args: argparse.Namespace,
    deadline: float,
) -> tuple[list[bytes], PostPolishStats]:
    stats = PostPolishStats()
    if not args.post_polish_qubo or len(codes) <= 3:
        return codes, stats

    current = list(codes)
    for _pass_idx in range(max(1, args.post_polish_passes)):
        if time.perf_counter() >= deadline:
            break
        n_values = _path_n_values(current, stack_count, max_height)
        suffixes = _post_polish_suffixes(current, n_values, args)
        selected_suffixes, suffix_qvars, suffix_qtime = _select_post_polish_suffixes_qubo(suffixes, args)
        if suffix_qvars:
            stats.qcalls += 1
            stats.qvars += suffix_qvars
            stats.qtime += suffix_qtime
        improved_this_pass = False
        for suffix in selected_suffixes:
            if time.perf_counter() >= deadline:
                break
            if suffix.start >= len(current) - 1:
                continue
            old_remaining = len(current) - 1 - suffix.start
            max_depth = min(old_remaining - 1, args.post_polish_suffix_max_depth)
            stats.attempts += 1
            stats.suffix_attempts += 1
            bridge, expanded = _suffix_shorter_path_to_n0(
                current[suffix.start],
                _code_n(current[suffix.start], stack_count, max_height),
                suffix.start,
                cap,
                stack_count,
                max_height,
                max_depth,
                deadline,
                args,
            )
            stats.suffix_expanded += expanded
            if bridge is None:
                continue
            saved = old_remaining - (len(bridge) - 1)
            if saved <= 0:
                continue
            candidate = current[: suffix.start] + bridge
            if not _path_business_ok(candidate, stack_count, max_height, cap):
                continue
            current = candidate
            stats.improvements += 1
            stats.suffix_improvements += 1
            stats.moves_saved += saved
            improved_this_pass = True
            break
        if improved_this_pass:
            continue

        n_values = _path_n_values(current, stack_count, max_height)
        windows = _post_polish_windows(current, n_values, args)
        selected, qvars, qtime = _select_post_polish_windows_qubo(windows, args)
        if qvars:
            stats.qcalls += 1
            stats.qvars += qvars
            stats.qtime += qtime
        if not selected:
            break
        for window in selected:
            if time.perf_counter() >= deadline:
                break
            if window.end >= len(current):
                continue
            old_len = window.end - window.start
            max_depth = min(old_len - 1, args.post_polish_max_bridge_depth)
            stats.attempts += 1
            bridge, expanded = _bridge_shorter_path(
                current[window.start],
                current[window.end],
                window.start,
                cap,
                stack_count,
                max_height,
                max_depth,
                deadline,
                args,
            )
            stats.bridge_expanded += expanded
            if bridge is None:
                continue
            saved = old_len - (len(bridge) - 1)
            if saved <= 0:
                continue
            candidate = current[: window.start] + bridge + current[window.end + 1 :]
            if not _path_business_ok(candidate, stack_count, max_height, cap):
                continue
            current = candidate
            stats.improvements += 1
            stats.moves_saved += saved
            improved_this_pass = True
        if not improved_this_pass:
            break
    return current, stats


def _codes_to_moves_instance(instance: Instance, codes: list[bytes]) -> list[Move]:
    stacks = [list(stack) for stack in instance.stacks]
    current_code, _ = _encode(stacks, instance.stack_count, instance.max_height)
    if current_code != codes[0]:
        raise RuntimeError("initial physical state does not match compact path start")

    moves: list[Move] = []
    for step, next_code in enumerate(codes[1:], start=1):
        chosen: Move | None = None
        for src_idx, src in enumerate(stacks):
            if not src:
                continue
            for dst_idx, dst in enumerate(stacks):
                if src_idx == dst_idx or len(dst) >= instance.max_height:
                    continue
                trial = [list(stack) for stack in stacks]
                box = trial[src_idx].pop()
                from_level = len(src)
                to_level = len(dst) + 1
                trial[dst_idx].append(box)
                code, _n_after = _encode(trial, instance.stack_count, instance.max_height)
                if code == next_code:
                    chosen = Move(
                        step=step - 1,
                        box=box,
                        from_stack=src_idx,
                        to_stack=dst_idx,
                        from_level=from_level,
                        to_level=to_level,
                    )
                    stacks = trial
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError(f"could not lift compact transition at step {step}")
        moves.append(chosen)
    return moves


def run_compact_beam_instance(case: str, instance: Instance, args: argparse.Namespace) -> CompactBeamResult:
    _apply_solver_profile(args)
    _apply_qubo_influence_profile(args)
    _apply_dynamic_profile(args)
    start = time.perf_counter()
    deadline = start + args.time_limit if args.time_limit > 0 else math.inf
    initial, n0 = _encode(instance.stacks, instance.stack_count, instance.max_height)
    cap = entropy_limit(instance, args.alpha)
    depth_cap = min(cap, args.max_moves) if args.max_moves is not None else cap

    parent: dict[bytes, bytes | None] = {initial: None}
    best_depth: dict[bytes, int] = {initial: 0}
    source_by_code: dict[bytes, str] = {initial: "initial"}
    qubo_ancestor_by_code: dict[bytes, bool] = {initial: False}
    reservoir: dict[bytes, tuple[int, int, bytes]] = {}
    frontier: list[tuple[bytes, int, int]] = [(initial, n0, 0)]
    target: bytes | None = initial if n0 <= args.target_final_n else None
    args._qubo_calls_done = 0
    args._group_qubo_calls_done = 0
    args._group_qubo_vars = 0
    args._group_qubo_time = 0.0
    args._group_qubo_selected_total = 0
    args._rank_pool_truncated_candidates = 0
    args._group_lookahead_cache = {}
    args._best_n_history = []
    args._layout_target_groups = []
    args._stack_count = instance.stack_count
    args._max_height = instance.max_height
    stream_candidate_limit = max(0, getattr(args, "stream_candidate_limit", 0))
    stream_trim_enabled = stream_candidate_limit > 0

    expanded = generated = pruned_bound = pruned_business = dominated = 0
    qcalls = qvars = 0
    qtime = 0.0
    peak_frontier = len(frontier)
    peak_candidates = 0
    peak_reservoir = 0
    reservoir_added_total = 0
    reservoir_pulled_total = 0
    stream_pruned_candidates = 0
    stream_seq = 0
    depth_reached = 0

    while target is None and frontier and time.perf_counter() < deadline:
        candidates: dict[bytes, tuple[int, int, bytes]] = {}
        candidate_worst_heap: list[tuple[_ReverseRank, int, bytes, tuple[int, int, bytes]]] = []
        goals: dict[bytes, tuple[int, int, bytes]] = {}
        for code, current_n, depth in frontier:
            if time.perf_counter() >= deadline:
                break
            depth_reached = max(depth_reached, depth)
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
            for nxt, nxt_n in _successors_with_n_cached(code, current_n, instance.stack_count, instance.max_height):
                generated += 1
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
                data = (nxt_n, next_depth, code)
                if nxt_n <= args.target_final_n:
                    goals[nxt] = data
                if stream_trim_enabled:
                    stream_seq += 1
                    stream_pruned_candidates += _offer_stream_candidate(
                        candidates,
                        candidate_worst_heap,
                        nxt,
                        data,
                        args,
                        stream_seq,
                    )
                else:
                    candidates[nxt] = data
        if target is not None:
            break
        if goals:
            target, data = min(goals.items(), key=_candidate_key_func(args))
            parent[target] = data[2]
            best_depth[target] = data[1]
            source_by_code[target] = "goal"
            qubo_ancestor_by_code[target] = qubo_ancestor_by_code.get(data[2], False)
            depth_reached = max(depth_reached, data[1])
            break
        if not candidates:
            break
        pulled = _reservoir_pull(reservoir, candidates, args)
        reservoir_pulled_total += pulled
        if pulled and stream_trim_enabled:
            candidate_worst_heap = []
            for code, data in candidates.items():
                stream_seq += 1
                heapq.heappush(candidate_worst_heap, (_stream_worst_key(code, data, args, stream_seq), stream_seq, code, data))
            stream_pruned_candidates += _trim_stream_candidates(
                candidates,
                candidate_worst_heap,
                stream_candidate_limit,
                args,
            )
        if stream_trim_enabled:
            stream_pruned_candidates += _trim_stream_candidates(
                candidates,
                candidate_worst_heap,
                stream_candidate_limit,
                args,
            )
        peak_candidates = max(peak_candidates, len(candidates))
        layer_best_n = min(data[0] for data in candidates.values())
        parent_counts: dict[bytes, int] = {}
        for _code, data in candidates.items():
            parent_counts[data[2]] = parent_counts.get(data[2], 0) + 1
        args._layer_candidate_count = len(candidates)
        args._layer_parent_count = len(parent_counts)
        args._layer_parent_top_share = (max(parent_counts.values()) / len(candidates)) if candidates else 0.0
        args._best_n_history.append(layer_best_n)
        selected_codes, selected_by_qubo, add_qcalls, add_qvars, add_qtime = _select_next_frontier(candidates, args)
        qcalls += add_qcalls
        qvars += add_qvars
        qtime += add_qtime
        selected_set = set(selected_codes)
        if getattr(args, "global_reservoir", False):
            max_add = max(0, getattr(args, "reservoir_add_per_layer", 0))
            added_this_layer = 0
            for code, data in sorted(candidates.items(), key=_candidate_key_func(args)):
                if code in selected_set:
                    continue
                n_value, depth, parent_code = data
                old_depth = best_depth.get(code)
                if old_depth is not None and depth >= old_depth:
                    continue
                parent[code] = parent_code
                best_depth[code] = depth
                source_by_code.setdefault(code, "reservoir")
                qubo_ancestor_by_code.setdefault(code, qubo_ancestor_by_code.get(parent_code, False))
                reservoir[code] = data
                reservoir_added_total += 1
                added_this_layer += 1
                if max_add > 0 and added_this_layer >= max_add:
                    break
            _reservoir_trim(reservoir, args)
            peak_reservoir = max(peak_reservoir, len(reservoir))
        next_frontier: list[tuple[bytes, int, int]] = []
        for code in selected_codes:
            n_value, depth, parent_code = candidates[code]
            parent[code] = parent_code
            best_depth[code] = depth
            source_by_code[code] = "qubo" if code in selected_by_qubo else "deterministic"
            qubo_ancestor_by_code[code] = code in selected_by_qubo or qubo_ancestor_by_code.get(parent_code, False)
            next_frontier.append((code, n_value, depth))
        frontier = next_frontier
        peak_frontier = max(peak_frontier, len(frontier))

    moves: list[Move] = []
    status = "TARGET_REACHABLE" if target is not None else "TIME_LIMIT_OR_BEAM_EXHAUSTED"
    polish_stats = PostPolishStats()
    qubo_selected_on_search_path = 0
    qubo_selected_on_path = 0
    qubo_ancestor_on_search_path = 0
    qubo_ancestor_on_path = 0
    tail_n_trace = ""
    tail_ordered_trace = ""
    tail_ordered_gain = 0
    if target is not None:
        codes = _backtrack(parent, target)
        original_codes = list(codes)
        qubo_selected_on_search_path = sum(1 for code in original_codes[1:] if source_by_code.get(code) == "qubo")
        qubo_ancestor_on_search_path = sum(1 for code in original_codes[1:] if qubo_ancestor_by_code.get(code, False))
        codes, polish_stats = _post_polish_codes_with_qubo(
            codes,
            instance.stack_count,
            instance.max_height,
            cap,
            args,
            deadline,
        )
        if polish_stats.improvements:
            try:
                moves = _codes_to_moves_instance(instance, codes)
            except RuntimeError:
                codes = original_codes
                moves = _codes_to_moves_instance(instance, codes)
                status += "; post_polish_lift_failed_reverted"
        else:
            moves = _codes_to_moves_instance(instance, codes)
        qubo_selected_on_path = sum(1 for code in codes[1:] if source_by_code.get(code) == "qubo")
        qubo_ancestor_on_path = sum(1 for code in codes[1:] if qubo_ancestor_by_code.get(code, False))
        final_n_values = _path_n_values(codes, instance.stack_count, instance.max_height)
        tail_steps = max(1, getattr(args, "tail_order_steps", 6))
        tail_n_values = final_n_values[-tail_steps:]
        tail_ordered_values = [instance.container_count - n_value for n_value in tail_n_values]
        tail_n_trace = ">".join(str(value) for value in tail_n_values)
        tail_ordered_trace = ">".join(str(value) for value in tail_ordered_values)
        tail_ordered_gain = tail_ordered_values[-1] - tail_ordered_values[0] if tail_ordered_values else 0
        if polish_stats.qcalls or polish_stats.attempts:
            status += (
                f"; post_polish_qubo=True; post_polish_qcalls={polish_stats.qcalls}; "
                f"post_polish_attempts={polish_stats.attempts}; "
                f"post_polish_improvements={polish_stats.improvements}; "
                f"post_polish_moves_saved={polish_stats.moves_saved}; "
                f"post_polish_suffix_attempts={polish_stats.suffix_attempts}; "
                f"post_polish_suffix_improvements={polish_stats.suffix_improvements}; "
                f"post_polish_suffix_expanded={polish_stats.suffix_expanded}; "
                f"post_polish_bridge_expanded={polish_stats.bridge_expanded}"
            )
    return CompactBeamResult(
        case=case,
        status=status,
        moves=moves,
        runtime_sec=time.perf_counter() - start,
        expanded=expanded,
        generated=generated,
        visited=len(best_depth),
        open_count=len(frontier),
        pruned_bound=pruned_bound,
        pruned_business=pruned_business,
        dominated=dominated,
        peak_frontier=peak_frontier,
        peak_candidates=peak_candidates,
        depth_reached=depth_reached,
        qubo_calls=qcalls + polish_stats.qcalls,
        qubo_vars=qvars + polish_stats.qvars,
        qubo_time_sec=qtime + polish_stats.qtime,
        qubo_selected_total=sum(1 for source in source_by_code.values() if source == "qubo"),
        qubo_selected_on_search_path=qubo_selected_on_search_path,
        qubo_selected_on_path=qubo_selected_on_path,
        qubo_ancestor_on_search_path=qubo_ancestor_on_search_path,
        qubo_ancestor_on_path=qubo_ancestor_on_path,
        group_qubo_calls=args._group_qubo_calls_done,
        group_qubo_vars=args._group_qubo_vars,
        group_qubo_time_sec=args._group_qubo_time,
        group_qubo_selected_total=args._group_qubo_selected_total,
        post_polish_attempts=polish_stats.attempts,
        post_polish_improvements=polish_stats.improvements,
        post_polish_moves_saved=polish_stats.moves_saved,
        tail_n_trace=tail_n_trace,
        tail_ordered_trace=tail_ordered_trace,
        tail_ordered_gain=tail_ordered_gain,
        reservoir_size=len(reservoir),
        peak_reservoir=peak_reservoir,
        reservoir_added_total=reservoir_added_total,
        reservoir_pulled_total=reservoir_pulled_total,
        stream_pruned_candidates=stream_pruned_candidates,
        rank_pool_truncated_candidates=args._rank_pool_truncated_candidates,
    )


def run_compact_beam(case: str, zone: str, bay: str, args: argparse.Namespace) -> CompactBeamResult:
    return run_compact_beam_instance(case, _instance_from_single_bay(zone, bay), args)


def _row_from_instance_result(result: CompactBeamResult, instance: Instance, args: argparse.Namespace) -> TrajectoryResult:
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
        solver=SOLVER_NAME,
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
            f"{result.status}; solver={SOLVER_NAME}; storage=memory; representation=compact_bytes; "
            f"policy=layered_beam; selection_policy={getattr(args, 'selection_policy', 'enhanced')}; "
            f"target_stop=N0; proof=not_optimality_proof; "
            f"solver_profile={getattr(args, 'solver_profile', 'speed_v2')}; "
            f"qubo_influence_profile={getattr(args, 'qubo_influence_profile', 'stable')}; "
            f"priority={args.priority}; beam_width={args.beam_width}; core_width={args.core_width}; "
            f"peak_frontier={result.peak_frontier}; peak_candidates={result.peak_candidates}; "
            f"depth_reached={result.depth_reached}; open={result.open_count}; "
            f"qubo_used={args.use_qubo}; qubo_mode={args.qubo_mode}; qubo_focus_n={args.qubo_focus_n}; "
            f"qubo_first={args.qubo_first}; "
            f"group_qubo_supplement={getattr(args, 'group_qubo_supplement', False)}; "
            f"max_group_qubo_calls={getattr(args, 'max_group_qubo_calls', 0)}; "
            f"qubo_cohorts_per_layer={args.qubo_cohorts_per_layer}; "
            f"qubo_calls={result.qubo_calls}; qubo_vars={result.qubo_vars}; "
            f"qubo_time_sec={result.qubo_time_sec:.6f}; qubo_core_width={args.qubo_core_width}; "
            f"qubo_supplement_width={args.qubo_supplement_width}; "
            f"qubo_target_share={args.qubo_target_share}; "
            f"qubo_group_max_vars={getattr(args, 'qubo_group_max_vars', 0)}; "
            f"qubo_group_select={getattr(args, 'qubo_group_select', 0)}; "
            f"qubo_group_take={getattr(args, 'qubo_group_take', 0)}; "
            f"qubo_group_lookahead_weight={getattr(args, 'qubo_group_lookahead_weight', 0.0)}; "
            f"qubo_group_branch_weight={getattr(args, 'qubo_group_branch_weight', 0.0)}; "
            f"group_qubo_risk_gate={getattr(args, 'group_qubo_risk_gate', False)}; "
            f"dynamic_min_frontier={getattr(args, 'dynamic_min_frontier', 0)}; "
            f"dynamic_core_frontier={getattr(args, 'dynamic_core_frontier', 0)}; "
            f"dynamic_max_frontier={getattr(args, 'dynamic_max_frontier', 0)}; "
            f"dynamic_candidate_pool={getattr(args, 'dynamic_candidate_pool', 0)}; "
            f"dynamic_qubo_keep_threshold={getattr(args, 'dynamic_qubo_keep_threshold', 0.0)}; "
            f"dynamic_profile={getattr(args, 'dynamic_profile', 'none')}; "
            f"group_qubo_calls={result.group_qubo_calls}; "
            f"group_qubo_vars={result.group_qubo_vars}; "
            f"group_qubo_time_sec={result.group_qubo_time_sec:.6f}; "
            f"group_qubo_selected_total={result.group_qubo_selected_total}; "
            f"qubo_selected_total={result.qubo_selected_total}; "
            f"qubo_selected_on_search_path={result.qubo_selected_on_search_path}; "
            f"qubo_selected_on_path={result.qubo_selected_on_path}; "
            f"qubo_ancestor_on_search_path={result.qubo_ancestor_on_search_path}; "
            f"qubo_ancestor_on_path={result.qubo_ancestor_on_path}; "
            f"post_polish_qubo={args.post_polish_qubo}; "
            f"post_polish_attempts={result.post_polish_attempts}; "
            f"post_polish_improvements={result.post_polish_improvements}; "
            f"post_polish_moves_saved={result.post_polish_moves_saved}; "
            f"tail_n_trace={result.tail_n_trace}; "
            f"tail_ordered_trace={result.tail_ordered_trace}; "
            f"tail_ordered_gain={result.tail_ordered_gain}; "
            f"global_reservoir={getattr(args, 'global_reservoir', False)}; "
            f"reservoir_size={result.reservoir_size}; "
            f"peak_reservoir={result.peak_reservoir}; "
            f"reservoir_added_total={result.reservoir_added_total}; "
            f"reservoir_pulled_total={result.reservoir_pulled_total}; "
            f"stream_candidate_limit={getattr(args, 'stream_candidate_limit', 0)}; "
            f"stream_candidate_trim_factor={getattr(args, 'stream_candidate_trim_factor', 2.0)}; "
            f"stream_pruned_candidates={result.stream_pruned_candidates}; "
            f"rank_pool_size={getattr(args, 'rank_pool_size', 0)}; "
            f"rank_pool_min_candidates={getattr(args, 'rank_pool_min_candidates', 0)}; "
            f"rank_pool_truncated_candidates={result.rank_pool_truncated_candidates}; "
            f"pruned_bound={result.pruned_bound}; pruned_business={result.pruned_business}; "
            f"dominated={result.dominated}"
        ),
    )


def _row_from_result(result: CompactBeamResult, zone: str, bay: str, args: argparse.Namespace) -> TrajectoryResult:
    return _row_from_instance_result(result, _instance_from_single_bay(zone, bay), args)


def _row_dict_from_instance_result(result: CompactBeamResult, instance: Instance, args: argparse.Namespace) -> dict[str, object]:
    base = _row_from_instance_result(result, instance, args)
    row = base.__dict__.copy()
    row.update(
        {
            "profile": getattr(args, "solver_profile", "speed_v2"),
            "solver_profile": getattr(args, "solver_profile", "speed_v2"),
            "qubo_influence_profile": getattr(args, "qubo_influence_profile", "stable"),
            "selection_policy": getattr(args, "selection_policy", "enhanced"),
            "priority": args.priority,
            "beam_width": args.beam_width,
            "core_width": args.core_width,
            "parent_core_cap": getattr(args, "parent_core_cap", 0),
            "peak_frontier": result.peak_frontier,
            "peak_candidates": result.peak_candidates,
            "depth_reached": result.depth_reached,
            "open_count": result.open_count,
            "pruned_bound": result.pruned_bound,
            "pruned_business": result.pruned_business,
            "dominated": result.dominated,
            "use_qubo": args.use_qubo,
            "qubo_mode": args.qubo_mode,
            "qubo_focus_n": args.qubo_focus_n,
            "qubo_first": args.qubo_first,
            "max_qubo_vars": args.max_qubo_vars,
            "qubo_core_width": args.qubo_core_width,
            "qubo_supplement_width": args.qubo_supplement_width,
            "qubo_cohorts_per_layer": args.qubo_cohorts_per_layer,
            "qubo_selected_total": result.qubo_selected_total,
            "qubo_selected_on_search_path": result.qubo_selected_on_search_path,
            "qubo_selected_on_path": result.qubo_selected_on_path,
            "qubo_ancestor_on_search_path": result.qubo_ancestor_on_search_path,
            "qubo_ancestor_on_path": result.qubo_ancestor_on_path,
            "group_qubo_supplement": getattr(args, "group_qubo_supplement", False),
            "group_qubo_calls": result.group_qubo_calls,
            "group_qubo_vars": result.group_qubo_vars,
            "group_qubo_time_sec": result.group_qubo_time_sec,
            "group_qubo_selected_total": result.group_qubo_selected_total,
            "qubo_group_max_vars": getattr(args, "qubo_group_max_vars", 0),
            "qubo_group_select": getattr(args, "qubo_group_select", 0),
            "qubo_group_take": getattr(args, "qubo_group_take", 0),
            "post_polish_attempts": result.post_polish_attempts,
            "post_polish_improvements": result.post_polish_improvements,
            "post_polish_moves_saved": result.post_polish_moves_saved,
            "global_reservoir": getattr(args, "global_reservoir", False),
            "reservoir_size": result.reservoir_size,
            "peak_reservoir": result.peak_reservoir,
            "reservoir_added_total": result.reservoir_added_total,
            "reservoir_pulled_total": result.reservoir_pulled_total,
            "stream_candidate_limit": getattr(args, "stream_candidate_limit", 0),
            "stream_candidate_trim_factor": getattr(args, "stream_candidate_trim_factor", 2.0),
            "stream_pruned_candidates": result.stream_pruned_candidates,
            "rank_pool_size": getattr(args, "rank_pool_size", 0),
            "rank_pool_min_candidates": getattr(args, "rank_pool_min_candidates", 0),
            "rank_pool_truncated_candidates": result.rank_pool_truncated_candidates,
        }
    )
    return row


def _row_dict_from_result(result: CompactBeamResult, zone: str, bay: str, args: argparse.Namespace) -> dict[str, object]:
    return _row_dict_from_instance_result(result, _instance_from_single_bay(zone, bay), args)


def _validate_instance_result(
    result: CompactBeamResult,
    instance: Instance,
    business_alpha: float = 1.3,
) -> tuple[bool, bool, str]:
    if not result.moves:
        return False, False, result.status
    solution = Solution(
        solver_name=SOLVER_NAME,
        feasible=True,
        optimal=False,
        objective=float(len(result.moves)),
        runtime_sec=result.runtime_sec,
        moves=result.moves,
        status=result.status,
    )
    goal_ok, goal_msg, _ = validate_solution(instance, solution)
    business_ok = max_entropy_along_moves(instance, result.moves) <= entropy_limit(instance, business_alpha)
    return business_ok, goal_ok, goal_msg


def _validate_result(result: CompactBeamResult, zone: str, bay: str) -> tuple[bool, bool, str]:
    return _validate_instance_result(result, _instance_from_single_bay(zone, bay))


def _row_mapping(row: TrajectoryResult | dict[str, object]) -> dict[str, object]:
    if hasattr(row, "__dict__"):
        return row.__dict__
    return row


def _write_rows(rows: list[TrajectoryResult] | list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPACT_RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_mapping(row))


def _numeric(row: dict[str, object], field: str, default: float = 0.0) -> float:
    value = row.get(field)
    if value in (None, ""):
        return default
    return float(value)


def _write_summary(rows: list[dict[str, object]], path: Path) -> None:
    def subset_summary(selected: list[dict[str, object]]) -> dict[str, float | int]:
        solved_rows = [row for row in selected if int(_numeric(row, "final_n", 10**9)) == 0]
        runtimes = [_numeric(row, "runtime_sec") for row in selected]
        return {
            "cases": len(selected),
            "solved": len(solved_rows),
            "moves_total": int(sum(_numeric(row, "moves") for row in solved_rows)),
            "avg_runtime": sum(runtimes) / len(runtimes) if runtimes else 0.0,
            "max_runtime": max(runtimes) if runtimes else 0.0,
            "qubo_calls": int(sum(_numeric(row, "qubo_calls") for row in selected)),
            "qubo_vars": int(sum(_numeric(row, "qubo_total_vars") for row in selected)),
            "qubo_time": sum(_numeric(row, "qubo_time_sec") for row in selected),
            "qubo_selected": int(sum(_numeric(row, "qubo_selected_total") for row in selected)),
            "qubo_on_path": int(sum(_numeric(row, "qubo_selected_on_path") for row in selected)),
            "qubo_ancestor": int(sum(_numeric(row, "qubo_ancestor_on_path") for row in selected)),
            "group_qubo_calls": int(sum(_numeric(row, "group_qubo_calls") for row in selected)),
            "group_qubo_time": sum(_numeric(row, "group_qubo_time_sec") for row in selected),
            "post_polish_saved": int(sum(_numeric(row, "post_polish_moves_saved") for row in selected)),
            "peak_candidates_max": int(max((_numeric(row, "peak_candidates") for row in selected), default=0)),
            "peak_candidates_avg": (
                sum(_numeric(row, "peak_candidates") for row in selected) / len(selected) if selected else 0.0
            ),
            "stream_pruned": int(sum(_numeric(row, "stream_pruned_candidates") for row in selected)),
            "rank_pool_truncated": int(sum(_numeric(row, "rank_pool_truncated_candidates") for row in selected)),
        }

    all_summary = subset_summary(rows)
    bay26 = [row for row in rows if str(row.get("case", "")).split("_")[1:2] == ["26"]]
    bay26_summary = subset_summary(bay26)
    profile = rows[0].get("profile", "") if rows else ""
    selection_policy = rows[0].get("selection_policy", "") if rows else ""
    lines = [
        "# Compact QUBO Beam Summary",
        "",
        f"profile: `{profile}`",
        f"selection_policy: `{selection_policy}`",
        "",
        "| scope | cases | solved | moves total | avg runtime sec | max runtime sec | qubo calls | qubo vars | qubo time sec | qubo path | qubo ancestor | group qcalls | group qtime sec | polish saved | peak cand max | peak cand avg | stream pruned | rank-pool truncated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in [("all", all_summary), ("bay26", bay26_summary)]:
        lines.append(
            f"| {label} | {summary['cases']} | {summary['solved']} | {summary['moves_total']} | "
            f"{summary['avg_runtime']:.3f} | {summary['max_runtime']:.3f} | "
            f"{summary['qubo_calls']} | {summary['qubo_vars']} | {summary['qubo_time']:.3f} | "
            f"{summary['qubo_on_path']} | {summary['qubo_ancestor']} | "
            f"{summary['group_qubo_calls']} | {summary['group_qubo_time']:.3f} | "
            f"{summary['post_polish_saved']} | {summary['peak_candidates_max']} | "
            f"{summary['peak_candidates_avg']:.1f} | {summary['stream_pruned']} | "
            f"{summary['rank_pool_truncated']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        f"# {rows[0].solver if rows else SOLVER_NAME}",
        "",
        "Compact byte-state layered beam with protected QUBO supplement.",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "compact_beam_qubo_n0_v1.csv")
    parser.add_argument(
        "--solver-profile",
        "--profile",
        dest="solver_profile",
        choices=sorted(_SOLVER_PROFILES.keys()),
        default="speed_v2",
        help="Reusable solver baseline. speed_v2 freezes the historical 30/30, 509-move, ~2.2s QUBO Beam profile.",
    )
    parser.add_argument("--priority", choices=["n_then_depth", "astar", "weighted_astar"], default="n_then_depth")
    parser.add_argument("--g-weight", type=float, default=1.0)
    parser.add_argument("--n-weight", type=float, default=1.0)
    parser.add_argument("--target-final-n", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--time-limit", type=float, default=0.0, help="Per-case limit in seconds; 0 means no limit.")
    parser.add_argument("--stream-candidate-limit", type=int, default=0, help="0 keeps the full per-layer candidate set.")
    parser.add_argument("--stream-candidate-trim-factor", type=float, default=2.0)
    parser.add_argument("--rank-pool-size", type=int, default=0, help="0 uses the old full candidate sort.")
    parser.add_argument("--rank-pool-min-candidates", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=1200)
    parser.add_argument("--core-width", type=int, default=1080)
    parser.add_argument("--parent-core-cap", type=int, default=4)
    parser.add_argument("--selection-policy", choices=["enhanced", "classic", "dynamic_qubo"], default="enhanced")
    parser.add_argument("--use-qubo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qubo-influence-profile", choices=sorted(_QUBO_INFLUENCE_PRESETS.keys()), default="stable")
    parser.add_argument("--qubo-mode", choices=["supplement", "focused_multi", "majority_multi"], default="focused_multi")
    parser.add_argument("--qubo-profile", default="frontier_cohort")
    parser.add_argument("--max-qubo-calls", type=int, default=32)
    parser.add_argument("--max-qubo-vars", type=int, default=64)
    parser.add_argument("--qubo-core-width", type=int, default=224)
    parser.add_argument("--qubo-supplement-width", type=int, default=24)
    parser.add_argument("--qubo-min-candidates", type=int, default=64)
    parser.add_argument("--qubo-priority-bonus", type=float, default=0.0001)
    parser.add_argument("--qubo-diversity-weight", type=float, default=0.10)
    parser.add_argument("--qubo-frontier-n-window", type=int, default=0)
    parser.add_argument("--qubo-frontier-depth-window", type=int, default=1)
    parser.add_argument("--qubo-focus-n", type=int, default=8, help="Only call QUBO when the layer best N is <= this value; -1 disables the gate.")
    parser.add_argument("--qubo-first", action=argparse.BooleanOptionalAction, default=False, help="Let QUBO fill its share before deterministic core fill.")
    parser.add_argument("--qubo-target-share", type=float, default=0.50, help="Target frontier share selected by QUBO in majority_multi mode.")
    parser.add_argument("--qubo-cohorts-per-layer", type=int, default=2)
    parser.add_argument("--qubo-cohort-stride", type=int, default=64, help="Candidate stride between QUBO cohorts; 0 uses max_qubo_vars.")
    parser.add_argument("--group-qubo-supplement", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-group-qubo-calls", type=int, default=32)
    parser.add_argument("--qubo-group-max-vars", type=int, default=96)
    parser.add_argument("--qubo-group-select", type=int, default=24)
    parser.add_argument("--qubo-group-take", type=int, default=4)
    parser.add_argument("--qubo-group-same-n-penalty", type=float, default=0.04)
    parser.add_argument("--qubo-group-lookahead-weight", type=float, default=0.0)
    parser.add_argument("--qubo-group-branch-weight", type=float, default=0.0)
    parser.add_argument("--group-qubo-risk-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--group-qubo-risk-min-candidates", type=int, default=50000)
    parser.add_argument("--group-qubo-risk-parent-share", type=float, default=0.12)
    parser.add_argument("--group-qubo-risk-plateau-window", type=int, default=3)
    parser.add_argument("--group-qubo-risk-plateau-delta", type=int, default=0)
    parser.add_argument("--dynamic-min-frontier", type=int, default=128)
    parser.add_argument("--dynamic-core-frontier", type=int, default=0)
    parser.add_argument("--dynamic-max-frontier", type=int, default=0, help="0 uses --beam-width as a safety cap.")
    parser.add_argument("--dynamic-candidate-pool", type=int, default=4096)
    parser.add_argument("--dynamic-qubo-keep-threshold", type=float, default=0.55)
    parser.add_argument(
        "--dynamic-profile",
        choices=sorted(_DYNAMIC_QUBO_PRESETS.keys()),
        default="legacy",
        help=(
            "Reusable preset for dynamic frontier. Use 'legacy' to force the 30/30 profile, "
            "'speed2000' for the A*-speed-alignment profile, 'balanced2200' for the tuned "
            "speed-quality profile, or 'mid3200'/'mid2400' for frontier-only options. "
            "Explicit --dynamic-* flags override preset values."
        ),
    )
    parser.add_argument("--global-reservoir", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reservoir-max-states", type=int, default=50000)
    parser.add_argument("--reservoir-pull", type=int, default=800)
    parser.add_argument("--reservoir-add-per-layer", type=int, default=4000)
    parser.add_argument("--qubo-rank-weight", type=float, default=1.0)
    parser.add_argument("--qubo-n-weight", type=float, default=2.0)
    parser.add_argument("--qubo-depth-weight", type=float, default=1.0)
    parser.add_argument("--qubo-parent-weight", type=float, default=0.15)
    parser.add_argument("--qubo-penalty-a", type=float, default=30.0)
    parser.add_argument("--qubo-time-limit", type=float, default=0.001)
    parser.add_argument("--qubo-reads", type=int, default=2)
    parser.add_argument("--qubo-iterations", type=int, default=20)
    parser.add_argument("--post-polish-qubo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-polish-passes", type=int, default=2)
    parser.add_argument("--post-polish-focus-n", type=int, default=8)
    parser.add_argument("--post-polish-tail-span", type=int, default=14)
    parser.add_argument("--post-polish-max-window", type=int, default=8)
    parser.add_argument("--post-polish-max-windows", type=int, default=96)
    parser.add_argument("--post-polish-select-windows", type=int, default=0)
    parser.add_argument("--post-polish-max-qubo-vars", type=int, default=72)
    parser.add_argument("--post-polish-max-bridge-depth", type=int, default=7)
    parser.add_argument("--post-polish-bridge-expanded", type=int, default=20000)
    parser.add_argument("--post-polish-suffix-focus-n", type=int, default=9)
    parser.add_argument("--post-polish-suffix-tail-span", type=int, default=18)
    parser.add_argument("--post-polish-suffix-max-candidates", type=int, default=48)
    parser.add_argument("--post-polish-suffix-select", type=int, default=5)
    parser.add_argument("--post-polish-suffix-max-qubo-vars", type=int, default=48)
    parser.add_argument("--post-polish-suffix-max-depth", type=int, default=18)
    parser.add_argument("--post-polish-suffix-expanded", type=int, default=60000)
    parser.add_argument("--post-polish-suffix-near-gap", type=int, default=2)
    parser.add_argument("--post-polish-suffix-near-penalty", type=float, default=0.30)
    parser.add_argument("--post-polish-overlap-penalty", type=float, default=8.0)
    parser.add_argument("--post-polish-near-window-gap", type=int, default=2)
    parser.add_argument("--post-polish-penalty-a", type=float, default=18.0)
    parser.add_argument("--post-polish-qubo-time-limit", type=float, default=0.001)
    parser.add_argument("--post-polish-qubo-reads", type=int, default=2)
    parser.add_argument("--post-polish-qubo-iterations", type=int, default=20)
    parser.add_argument("--tail-order-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    acceptance = output.with_name(output.stem + "_acceptance.csv")
    moves_dir = output.with_name(output.stem + "_moves")
    report = output.with_suffix(".md")
    summary = output.with_name(output.stem + "_summary.md")
    moves_dir.mkdir(parents=True, exist_ok=True)

    exact_refs = _exact_moves_by_case()
    rows: list[TrajectoryResult] = []
    row_dicts: list[dict[str, object]] = []
    validation: dict[str, tuple[bool, bool, str]] = {}
    moves_paths: dict[str, Path] = {}

    cases = _case_filter(args)
    for idx, case in enumerate(cases, start=1):
        zone, bay = _parse_case(case)
        print(f"CASE_START {idx} {case}", flush=True)
        result = run_compact_beam(case, zone, bay, args)
        row = _row_from_result(result, zone, bay, args)
        rows.append(row)
        row_dicts.append(_row_dict_from_result(result, zone, bay, args))
        validation[case] = _validate_result(result, zone, bay)
        if result.moves:
            move_path = moves_dir / f"{case}_moves.csv"
            _write_explain_csv(_instance_from_single_bay(zone, bay), result.moves, move_path)
            moves_paths[case] = move_path
        print(
            f"CASE_DONE {case} final_n={row.final_n} moves={row.moves} "
            f"runtime={row.runtime_sec:.3f}s expanded={row.expanded_states} "
            f"generated={row.generated_states} qcalls={row.qubo_calls} "
            f"business_ok={validation[case][0]} goal_ok={validation[case][1]}",
            flush=True,
        )

    _write_rows(row_dicts, output)
    _write_acceptance(rows, validation, moves_paths, exact_refs, acceptance)
    _write_report(rows, output, acceptance, report, exact_refs)
    _write_summary(row_dicts, summary)
    print(f"WROTE {_rel(output)}", flush=True)
    print(f"WROTE {_rel(acceptance)}", flush=True)
    print(f"WROTE {_rel(report)}", flush=True)
    print(f"WROTE {_rel(summary)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
