"""Run the current compact QUBO beam on open CPMP benchmark instances.

The first supported source is the DLTS CPMP repository:
https://github.com/ahottung/DLTS
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_compact_beam_qubo_n0 import (  # noqa: E402
    COMPACT_RESULT_FIELDS,
    SOLVER_NAME,
    _SOLVER_PROFILES,
    _initial_scarcity_metrics,
    _light_route_score,
    _reset_beam_survivor_output,
    _reset_frontier_trace_output,
    _reset_terminal_trace_output,
    _row_dict_from_instance_result,
    _validate_instance_result,
    make_compact_args,
    run_compact_beam_instance,
)
from container_reshuffle.core import Instance, Move, ProblemVariant, validate_instance  # noqa: E402
from container_reshuffle.features import future_disorder_count_stacks  # noqa: E402

DLTS_URL = "https://github.com/ahottung/DLTS.git"
OPEN_RUNNER_RECOMMENDED_MAX_JOBS = 16
OPEN_EXTRA_FIELDS = [
    "dataset",
    "family",
    "variant",
    "source_file",
    "reference_file",
    "reference_moves",
    "move_gap",
    "full_quality_penalty",
    "reference_tiers",
    "business_alpha",
    "business_ok",
    "goal_validation_ok",
    "feasibility_ok",
    "time_limit_ok",
    "primary_ok",
    "validation_msg",
    "portfolio_profiles",
    "portfolio_selected_profile",
    "portfolio_runtime_sec",
]
OPEN_RESULT_FIELDS = [field for field in OPEN_EXTRA_FIELDS if field not in COMPACT_RESULT_FIELDS] + COMPACT_RESULT_FIELDS

_WANG2017_TASK_GUARDED_PORTFOLIO_VARIANTS = (
    "qubo_scheduler_state_v5_wang_rescue_polish",
    "qubo_scheduler_state_v5_wang2017_task_guarded_polish",
)
_TERMINAL_ESCAPE_PORTFOLIO_VARIANTS = (
    "qubo_scheduler_state_v5_qubo_diversity_union_terminal_bridge_polish",
    "qubo_scheduler_state_v5_qubo_diversity_union_tail_lookahead_wide_polish",
)
_TERMINAL_ESCAPE_LAZY_PORTFOLIO_VARIANTS = _TERMINAL_ESCAPE_PORTFOLIO_VARIANTS


@dataclass(frozen=True)
class DltsReference:
    optimal_moves: int | None = None
    max_height: int | None = None


@dataclass(frozen=True)
class OpenCpmpCase:
    family: str
    case_id: str
    instance_path: Path
    reference_path: Path | None
    reference: DltsReference


@dataclass(frozen=True)
class OpenRunTask:
    task_index: int
    case: OpenCpmpCase
    variant: str


@dataclass(frozen=True)
class OpenRunTaskResult:
    task_index: int
    case_id: str
    variant: str
    rows: list[dict[str, object]]
    skip_reason: str = ""


def _build_open_run_tasks(cases: list[OpenCpmpCase], variants: list[str]) -> list[OpenRunTask]:
    tasks: list[OpenRunTask] = []
    for case in cases:
        for variant in variants:
            tasks.append(OpenRunTask(task_index=len(tasks), case=case, variant=variant))
    return tasks


def _parse_open_jobs(value: str) -> int | str:
    if value.lower() == "auto":
        return "auto"
    try:
        jobs = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer or 'auto'") from exc
    if jobs < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer or 'auto'")
    return jobs


def _resolve_open_jobs(jobs: int | str, cpu_count: int | None = None, warn: bool = True) -> int:
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1
    if isinstance(jobs, str) and jobs.lower() == "auto":
        return max(1, min(int(cpu_count), OPEN_RUNNER_RECOMMENDED_MAX_JOBS))
    resolved = int(jobs)
    if resolved == 0:
        return max(1, min(int(cpu_count), OPEN_RUNNER_RECOMMENDED_MAX_JOBS))
    if warn and resolved > OPEN_RUNNER_RECOMMENDED_MAX_JOBS:
        print(
            f"WARNING --jobs={resolved} exceeds recommended safe cap "
            f"{OPEN_RUNNER_RECOMMENDED_MAX_JOBS}; open120 timing evidence preserved 118/120 up to "
            f"{OPEN_RUNNER_RECOMMENDED_MAX_JOBS} jobs, while 20 jobs dropped coverage.",
            file=sys.stderr,
            flush=True,
        )
    return resolved


def _case_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.stem
    except ValueError:
        return 10**9, path.stem


def _priority_values(instance: Instance) -> list[int]:
    return [int(box) for stack in instance.stacks for box in stack]


def requires_priority_group_semantics(instance: Instance) -> bool:
    """Return True when an instance stores repeated priority labels, not unique box IDs."""

    values = _priority_values(instance)
    return sorted(values) != list(range(1, instance.container_count + 1))


def _validate_open_shape(instance: Instance, path: Path) -> None:
    if len(instance.stacks) != instance.stack_count:
        raise ValueError(f"expected {instance.stack_count} stacks in {path}, got {len(instance.stacks)}")
    box_count = 0
    for idx, stack in enumerate(instance.stacks):
        if len(stack) > instance.max_height:
            raise ValueError(f"stack {idx} exceeds max_height in {path}")
        for box in stack:
            if box <= 0:
                raise ValueError(f"priority values must be positive in {path}")
            if box > 255:
                raise ValueError(f"priority value {box} exceeds compact byte limit in {path}")
        box_count += len(stack)
    if box_count != instance.container_count:
        raise ValueError(f"expected {instance.container_count} boxes in {path}, got {box_count}")


def load_dlts_dat(path: Path, max_height: int | None = None, allow_priority_groups: bool = True) -> Instance:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty DLTS instance file: {path}")
    header = [int(part) for part in lines[0].split()]
    if len(header) != 2:
        raise ValueError(f"bad DLTS header in {path}: {lines[0]!r}")
    stack_count, container_count = header
    stacks: list[list[int]] = []
    max_seen_height = 0
    for line in lines[1:]:
        values = [int(part) for part in line.split()]
        if not values:
            continue
        height, boxes = values[0], values[1:]
        if height != len(boxes):
            raise ValueError(f"declared height {height} does not match row in {path}: {line!r}")
        stacks.append(boxes)
        max_seen_height = max(max_seen_height, height)
    if len(stacks) != stack_count:
        raise ValueError(f"expected {stack_count} stacks in {path}, got {len(stacks)}")
    inferred_height = max_seen_height + 2
    instance = Instance(
        instance_id=f"{path.parent.name}_{path.stem}",
        variant=ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=stack_count,
        max_height=max_height if max_height is not None else inferred_height,
        container_count=container_count,
        stacks=stacks,
    )
    _validate_open_shape(instance, path)
    try:
        validate_instance(instance)
    except ValueError:
        if not allow_priority_groups:
            raise
    return instance


def _priority_group_sorted_stack(stack: list[int]) -> bool:
    return all(stack[idx] >= stack[idx + 1] for idx in range(len(stack) - 1))


def _simulate_priority_group_moves(
    instance: Instance,
    moves: list[Move],
) -> tuple[bool, str, list[list[int]], list[int], int]:
    stacks = [list(stack) for stack in instance.stacks]
    n_trace = [future_disorder_count_stacks(stacks)]
    max_g = n_trace[0]
    last_step = -1

    for depth, move in enumerate(moves, start=1):
        if move.step < last_step:
            return False, "move steps are not nondecreasing", stacks, n_trace, max_g
        last_step = move.step
        if move.from_stack == move.to_stack:
            return False, "source and destination must differ", stacks, n_trace, max_g
        if not (0 <= move.from_stack < instance.stack_count):
            return False, "source stack out of range", stacks, n_trace, max_g
        if not (0 <= move.to_stack < instance.stack_count):
            return False, "destination stack out of range", stacks, n_trace, max_g
        src = stacks[move.from_stack]
        dst = stacks[move.to_stack]
        if not src:
            return False, "source stack is empty", stacks, n_trace, max_g
        if len(dst) >= instance.max_height:
            return False, "destination stack is full", stacks, n_trace, max_g
        if src[-1] != move.box:
            return False, f"move box {move.box} is not source top {src[-1]}", stacks, n_trace, max_g
        if move.from_level is not None and move.from_level != len(src):
            return False, "from_level does not match state", stacks, n_trace, max_g
        if move.to_level is not None and move.to_level != len(dst) + 1:
            return False, "to_level does not match state", stacks, n_trace, max_g

        dst.append(src.pop())
        n_value = future_disorder_count_stacks(stacks)
        n_trace.append(n_value)
        max_g = max(max_g, depth + n_value)

    if not all(_priority_group_sorted_stack(stack) for stack in stacks):
        return False, "final state is not a priority-group goal", stacks, n_trace, max_g
    return True, "ok", stacks, n_trace, max_g


def _unified_wang_row_fields(result, args: argparse.Namespace) -> dict[str, object]:
    return {
        "unified_wang_qubo_enabled": getattr(args, "unified_wang_qubo_enabled", False),
        "unified_wang_qubo_supplement_width": getattr(args, "unified_wang_qubo_supplement_width", 0),
        "unified_wang_qubo_focus_n": getattr(args, "unified_wang_qubo_focus_n", 8),
        "unified_wang_qubo_n_window": getattr(args, "unified_wang_qubo_n_window", 1),
        "unified_wang_qubo_max_vars": getattr(args, "unified_wang_qubo_max_vars", 64),
        "unified_wang_qubo_select": getattr(args, "unified_wang_qubo_select", 12),
        "unified_wang_n_weight": getattr(args, "unified_wang_n_weight", 1000.0),
        "unified_wang_depth_weight": getattr(args, "unified_wang_depth_weight", 1.0),
        "unified_wang_fixed_prefix_weight": getattr(args, "unified_wang_fixed_prefix_weight", 0.0),
        "unified_wang_fix_pair_weight": getattr(args, "unified_wang_fix_pair_weight", 0.0),
        "unified_wang_target_gap_weight": getattr(args, "unified_wang_target_gap_weight", 0.0),
        "unified_wang_same_parent_penalty": getattr(args, "unified_wang_same_parent_penalty", 0.0),
        "unified_wang_pair_same_parent_penalty": getattr(args, "unified_wang_pair_same_parent_penalty", 0.0),
        "unified_wang_signature_penalty": getattr(args, "unified_wang_signature_penalty", 0.0),
        "unified_wang_complement_bonus": getattr(args, "unified_wang_complement_bonus", 0.0),
        "unified_wang_qubo_calls": result.unified_wang_qubo_calls,
        "unified_wang_qubo_vars": result.unified_wang_qubo_vars,
        "unified_wang_qubo_time_sec": result.unified_wang_qubo_time_sec,
        "unified_wang_fixed_prefix_selected": result.unified_wang_fixed_prefix_selected,
        "unified_wang_fix_pair_selected": result.unified_wang_fix_pair_selected,
        "target_layout_count": len(getattr(args, "_layout_target_groups", ()) or ()),
    }


def _union_diversity_qubo_row_fields(result, args: argparse.Namespace) -> dict[str, object]:
    return {
        "union_diversity_qubo_enabled": getattr(args, "union_diversity_qubo_enabled", False),
        "union_diversity_qubo_supplement_width": getattr(args, "union_diversity_qubo_supplement_width", 0),
        "union_diversity_qubo_focus_n": getattr(args, "union_diversity_qubo_focus_n", -1),
        "union_diversity_qubo_n_window": getattr(args, "union_diversity_qubo_n_window", 2),
        "union_diversity_qubo_min_candidates": getattr(args, "union_diversity_qubo_min_candidates", 64),
        "union_diversity_qubo_max_vars": getattr(args, "union_diversity_qubo_max_vars", 64),
        "union_diversity_qubo_select": getattr(args, "union_diversity_qubo_select", 8),
        "union_diversity_qubo_rank_weight": getattr(args, "union_diversity_qubo_rank_weight", 0.02),
        "union_diversity_qubo_wang_fixed_weight": getattr(args, "union_diversity_qubo_wang_fixed_weight", 0.0),
        "union_diversity_qubo_wang_pair_weight": getattr(args, "union_diversity_qubo_wang_pair_weight", 0.0),
        "union_diversity_qubo_frg_weight": getattr(args, "union_diversity_qubo_frg_weight", 0.0),
        "union_diversity_qubo_rollout_bonus": getattr(args, "union_diversity_qubo_rollout_bonus", 0.0),
        "union_diversity_qubo_same_parent_penalty": getattr(args, "union_diversity_qubo_same_parent_penalty", 0.45),
        "union_diversity_qubo_same_source_penalty": getattr(args, "union_diversity_qubo_same_source_penalty", 0.20),
        "union_diversity_qubo_signature_penalty": getattr(args, "union_diversity_qubo_signature_penalty", 0.18),
        "union_diversity_qubo_reduce_signature_penalty": getattr(args, "union_diversity_qubo_reduce_signature_penalty", 0.12),
        "union_diversity_qubo_n_band_penalty": getattr(args, "union_diversity_qubo_n_band_penalty", 0.08),
        "union_diversity_qubo_n_band_size": getattr(args, "union_diversity_qubo_n_band_size", 1),
        "union_diversity_qubo_origin_similarity_penalty": getattr(
            args, "union_diversity_qubo_origin_similarity_penalty", 0.0
        ),
        "union_diversity_qubo_same_origin_parent_n_band_penalty": getattr(
            args, "union_diversity_qubo_same_origin_parent_n_band_penalty", 0.0
        ),
        "union_diversity_qubo_frg_witness_signature_penalty": getattr(
            args, "union_diversity_qubo_frg_witness_signature_penalty", 0.0
        ),
        "union_diversity_qubo_adaptive_macro_quota_enabled": getattr(
            args, "union_diversity_qubo_adaptive_macro_quota_enabled", False
        ),
        "union_diversity_qubo_macro_quota_penalty": getattr(
            args, "union_diversity_qubo_macro_quota_penalty", 0.0
        ),
        "union_diversity_qubo_macro_quota_base_share": getattr(
            args, "union_diversity_qubo_macro_quota_base_share", 0.0
        ),
        "union_diversity_qubo_macro_quota_min_share": getattr(
            args, "union_diversity_qubo_macro_quota_min_share", 0.0
        ),
        "union_diversity_qubo_macro_quota_max_share": getattr(
            args, "union_diversity_qubo_macro_quota_max_share", 0.5
        ),
        "union_diversity_qubo_macro_quota_debt_weight": getattr(
            args, "union_diversity_qubo_macro_quota_debt_weight", 0.35
        ),
        "union_diversity_qubo_macro_quota_scarcity_weight": getattr(
            args, "union_diversity_qubo_macro_quota_scarcity_weight", 0.35
        ),
        "union_diversity_qubo_macro_quota_witness_weight": getattr(
            args, "union_diversity_qubo_macro_quota_witness_weight", 0.30
        ),
        "union_diversity_qubo_macro_quality_bonus": getattr(
            args, "union_diversity_qubo_macro_quality_bonus", 0.0
        ),
        "union_diversity_qubo_macro_edge_chain_cost_weight": getattr(
            args,
            "union_diversity_qubo_macro_edge_chain_cost_weight",
            0.0,
        ),
        "union_diversity_qubo_macro_future_credit_enabled": getattr(
            args, "union_diversity_qubo_macro_future_credit_enabled", False
        ),
        "union_diversity_qubo_completion_witness_credit_weight": getattr(
            args,
            "union_diversity_qubo_completion_witness_credit_weight",
            0.0,
        ),
        "union_diversity_qubo_receiver_debt_credit_weight": getattr(
            args,
            "union_diversity_qubo_receiver_debt_credit_weight",
            0.0,
        ),
        "union_diversity_qubo_source_scarcity_credit_weight": getattr(
            args,
            "union_diversity_qubo_source_scarcity_credit_weight",
            0.0,
        ),
        "union_diversity_qubo_macro_path_credit_weight": getattr(
            args,
            "union_diversity_qubo_macro_path_credit_weight",
            0.0,
        ),
        "union_diversity_qubo_macro_witness_move_penalty_weight": getattr(
            args,
            "union_diversity_qubo_macro_witness_move_penalty_weight",
            0.0,
        ),
        "union_diversity_qubo_macro_witness_move_penalty_scale": getattr(
            args,
            "union_diversity_qubo_macro_witness_move_penalty_scale",
            256,
        ),
        "union_diversity_qubo_macro_prefix_penalty_weight": getattr(
            args,
            "union_diversity_qubo_macro_prefix_penalty_weight",
            0.0,
        ),
        "union_diversity_qubo_macro_witness_move_penalty_min_stack_count": getattr(
            args,
            "union_diversity_qubo_macro_witness_move_penalty_min_stack_count",
            0,
        ),
        "union_diversity_qubo_macro_witness_move_penalty_max_height": getattr(
            args,
            "union_diversity_qubo_macro_witness_move_penalty_max_height",
            0,
        ),
        "union_diversity_qubo_macro_witness_move_penalty_max_stack_count": getattr(
            args,
            "union_diversity_qubo_macro_witness_move_penalty_max_stack_count",
            0,
        ),
        "union_diversity_qubo_macro_future_credit_min_witness_quality": getattr(
            args,
            "union_diversity_qubo_macro_future_credit_min_witness_quality",
            0.0,
        ),
        "union_diversity_qubo_receiver_debt_penalty_weight": getattr(
            args,
            "union_diversity_qubo_receiver_debt_penalty_weight",
            0.0,
        ),
        "union_diversity_qubo_macro_future_credit_cap": getattr(
            args,
            "union_diversity_qubo_macro_future_credit_cap",
            0.0,
        ),
        "union_diversity_qubo_unified_macro_future_credit_scale": getattr(
            args,
            "union_diversity_qubo_unified_macro_future_credit_scale",
            1.0,
        ),
        "union_diversity_qubo_unified_macro_pair_penalty": getattr(
            args,
            "union_diversity_qubo_unified_macro_pair_penalty",
            0.0,
        ),
        "union_diversity_qubo_calls": getattr(result, "union_diversity_qubo_calls", 0),
        "union_diversity_qubo_vars": getattr(result, "union_diversity_qubo_vars", 0),
        "union_diversity_qubo_time_sec": getattr(result, "union_diversity_qubo_time_sec", 0.0),
        "union_diversity_qubo_selected_total": getattr(result, "union_diversity_qubo_selected_total", 0),
        "union_rollout_probe_enabled": getattr(args, "union_rollout_probe_enabled", False),
        "union_rollout_probe_max_candidates": getattr(args, "union_rollout_probe_max_candidates", 0),
        "union_rollout_probe_time_limit": getattr(args, "union_rollout_probe_time_limit", 0.0),
        "union_rollout_probe_max_steps": getattr(args, "union_rollout_probe_max_steps", 0),
        "union_rollout_probe_max_generated": getattr(args, "union_rollout_probe_max_generated", 0),
        "union_rollout_probe_feasible_bonus": getattr(args, "union_rollout_probe_feasible_bonus", 0.0),
        "union_rollout_probe_final_n_weight": getattr(args, "union_rollout_probe_final_n_weight", 0.0),
        "union_rollout_probe_moves_weight": getattr(args, "union_rollout_probe_moves_weight", 0.0),
        "union_rollout_probe_dead_end_penalty": getattr(args, "union_rollout_probe_dead_end_penalty", 0.0),
        "union_rollout_probe_evaluator_enabled": getattr(args, "union_rollout_probe_evaluator_enabled", False),
        "union_rollout_probe_evaluator_weight": getattr(args, "union_rollout_probe_evaluator_weight", 0.0),
        "union_rollout_probe_evaluator_final_n_penalty": getattr(
            args,
            "union_rollout_probe_evaluator_final_n_penalty",
            1000.0,
        ),
        "union_rollout_probe_evaluator_require_feasible": getattr(
            args,
            "union_rollout_probe_evaluator_require_feasible",
            True,
        ),
        "union_rollout_probe_evaluator_min_stack_count": getattr(
            args,
            "union_rollout_probe_evaluator_min_stack_count",
            0,
        ),
        "union_rollout_probe_calls": getattr(result, "union_rollout_probe_calls", 0),
        "union_rollout_probe_candidates": getattr(result, "union_rollout_probe_candidates", 0),
        "union_rollout_probe_feasible": getattr(result, "union_rollout_probe_feasible", 0),
        "union_rollout_probe_time_sec": getattr(result, "union_rollout_probe_time_sec", 0.0),
        "union_rollout_probe_cache_hits": getattr(result, "union_rollout_probe_cache_hits", 0),
        "union_rollout_probe_selected_total": getattr(result, "union_rollout_probe_selected_total", 0),
        "union_rollout_terminal_graft": getattr(args, "union_rollout_terminal_graft", False),
        "union_rollout_terminal_graft_respect_cap": getattr(args, "union_rollout_terminal_graft_respect_cap", True),
        "union_rollout_terminal_grafted_total": getattr(result, "union_rollout_terminal_grafted_total", 0),
        "union_rollout_terminal_graft_used": getattr(result, "union_rollout_terminal_graft_used", False),
        "micro_rollout_enabled": getattr(args, "micro_rollout_enabled", False),
        "micro_rollout_trace_only": getattr(args, "micro_rollout_trace_only", False),
        "receiver_safety_enabled": getattr(args, "receiver_safety_enabled", False),
        "micro_rollout_tiebreak_only": getattr(args, "micro_rollout_tiebreak_only", False),
        "micro_rollout_tiebreak_epsilon": getattr(args, "micro_rollout_tiebreak_epsilon", 0.0),
        "micro_rollout_window": getattr(args, "micro_rollout_window", 0),
        "micro_first_child_limit": getattr(args, "micro_first_child_limit", 0),
        "micro_second_child_limit": getattr(args, "micro_second_child_limit", 0),
        "micro_cache_enabled": getattr(args, "micro_cache_enabled", True),
        "w_micro_delta_n": getattr(args, "w_micro_delta_n", 0.0),
        "w_micro_safe_continuation": getattr(args, "w_micro_safe_continuation", 0.0),
        "w_receiver_safe_delta": getattr(args, "w_receiver_safe_delta", 0.0),
        "w_last_buffer_consumed": getattr(args, "w_last_buffer_consumed", 0.0),
        "w_height_pressure": getattr(args, "w_height_pressure", 0.0),
        "micro_rollout_calls": getattr(result, "micro_rollout_calls", 0),
        "micro_rollout_cache_hits": getattr(result, "micro_rollout_cache_hits", 0),
        "micro_rollout_candidates": getattr(result, "micro_rollout_candidates", 0),
        "micro_rollout_selected_total": getattr(result, "micro_rollout_selected_total", 0),
        "receiver_safety_scored_total": getattr(result, "receiver_safety_scored_total", 0),
    }


def _quality_reservoir_row_fields(result, args: argparse.Namespace) -> dict[str, object]:
    return {
        "quality_reservoir_enabled": getattr(args, "quality_reservoir_enabled", False),
        "quality_reservoir_max_candidates": getattr(args, "quality_reservoir_max_candidates", 0),
        "quality_reservoir_top_quality": getattr(args, "quality_reservoir_top_quality", 0),
        "quality_reservoir_origin_reserve": getattr(args, "quality_reservoir_origin_reserve", 0),
        "quality_reservoir_witness_reserve": getattr(args, "quality_reservoir_witness_reserve", 0),
        "quality_reservoir_progress_reserve": getattr(args, "quality_reservoir_progress_reserve", 0),
        "quality_reservoir_witness_weight": getattr(args, "quality_reservoir_witness_weight", 0.0),
        "quality_reservoir_progress_weight": getattr(args, "quality_reservoir_progress_weight", 0.0),
        "quality_reservoir_source_scarcity_weight": getattr(
            args,
            "quality_reservoir_source_scarcity_weight",
            0.0,
        ),
        "quality_reservoir_receiver_debt_weight": getattr(args, "quality_reservoir_receiver_debt_weight", 0.0),
        "quality_reservoir_height_risk_weight": getattr(args, "quality_reservoir_height_risk_weight", 0.0),
        "quality_reservoir_chain_cost_weight": getattr(args, "quality_reservoir_chain_cost_weight", 0.0),
        "quality_reservoir_pair_receiver_debt_penalty": getattr(
            args,
            "quality_reservoir_pair_receiver_debt_penalty",
            0.0,
        ),
        "quality_reservoir_pair_macro_signature_penalty": getattr(
            args,
            "quality_reservoir_pair_macro_signature_penalty",
            0.0,
        ),
        "quality_reservoir_generated_total": getattr(result, "quality_reservoir_generated_total", 0),
        "quality_reservoir_kept_total": getattr(result, "quality_reservoir_kept_total", 0),
        "quality_reservoir_selected_total": getattr(result, "quality_reservoir_selected_total", 0),
        "quality_reservoir_witness_found_total": getattr(result, "quality_reservoir_witness_found_total", 0),
        "quality_reservoir_avg_witness_quality": getattr(result, "quality_reservoir_avg_witness_quality", 0.0),
        "quality_reservoir_avg_receiver_debt": getattr(result, "quality_reservoir_avg_receiver_debt", 0.0),
        "quality_reservoir_avg_height_risk": getattr(result, "quality_reservoir_avg_height_risk", 0.0),
        "quality_reservoir_pair_penalty_total": getattr(result, "quality_reservoir_pair_penalty_total", 0.0),
    }


def _micro_rollout_row_fields(result, args: argparse.Namespace) -> dict[str, object]:
    return {
        "debt_basin_enabled": getattr(args, "debt_basin_enabled", False),
        "w_receiver_debt": getattr(args, "w_receiver_debt", 0.0),
        "w_escape_potential": getattr(args, "w_escape_potential", 0.0),
        "w_debt_basin_chain_cost": getattr(args, "w_debt_basin_chain_cost", 0.0),
        "union_diversity_qubo_basin_similarity_penalty": getattr(
            args, "union_diversity_qubo_basin_similarity_penalty", 0.0
        ),
        "debt_basin_trace_window": getattr(args, "debt_basin_trace_window", 0),
        "debt_basin_reduction_chain_enabled": getattr(args, "debt_basin_reduction_chain_enabled", False),
        "receiver_debt_selected_total": getattr(result, "receiver_debt_selected_total", 0.0),
        "receiver_debt_avg_selected": getattr(result, "receiver_debt_avg_selected", 0.0),
        "escape_potential_selected_total": getattr(result, "escape_potential_selected_total", 0.0),
        "escape_potential_avg_selected": getattr(result, "escape_potential_avg_selected", 0.0),
        "basin_pair_penalty_total": getattr(result, "basin_pair_penalty_total", 0.0),
        "basin_unique_selected_total": getattr(result, "basin_unique_selected_total", 0),
        "reduction_chain_generated_total": getattr(result, "reduction_chain_generated_total", 0),
        "reduction_chain_selected_total": getattr(result, "reduction_chain_selected_total", 0),
        "debt_basin_selected_on_path": getattr(result, "debt_basin_selected_on_path", 0),
    }


def _wang_pair_qubo_row_fields(result, args: argparse.Namespace) -> dict[str, object]:
    return {
        "wang_pair_qubo_enabled": getattr(args, "wang_pair_qubo_enabled", False),
        "wang_pair_qubo_start_layer": getattr(args, "wang_pair_qubo_start_layer", 0),
        "wang_pair_qubo_calls": getattr(result, "wang_pair_qubo_calls", 0),
        "wang_pair_qubo_vars": getattr(result, "wang_pair_qubo_vars", 0),
        "wang_pair_qubo_time_sec": getattr(result, "wang_pair_qubo_time_sec", 0.0),
        "wang_feature_fixed_prefix_weight": getattr(args, "wang_feature_fixed_prefix_weight", 0.0),
        "wang_feature_fix_pair_weight": getattr(args, "wang_feature_fix_pair_weight", 0.0),
        "wang_feature_risk_weight": getattr(args, "wang_feature_risk_weight", 0.0),
        "wang_feature_same_parent_weight": getattr(args, "wang_feature_same_parent_weight", 0.0),
    }


def _feasibility_escape_row_fields(result, args: argparse.Namespace) -> dict[str, object]:
    return {
        "feasibility_escape_enabled": getattr(args, "feasibility_escape_enabled", False),
        "feasibility_escape_supplement_width": getattr(args, "feasibility_escape_supplement_width", 0),
        "feasibility_escape_focus_n": getattr(args, "feasibility_escape_focus_n", 8),
        "feasibility_escape_n_window": getattr(args, "feasibility_escape_n_window", 1),
        "feasibility_escape_dead_end_threshold": getattr(args, "feasibility_escape_dead_end_threshold", 0.45),
        "feasibility_escape_min_temporary_slack": getattr(args, "feasibility_escape_min_temporary_slack", 1),
        "feasibility_escape_calls": result.feasibility_escape_calls,
        "feasibility_escape_added_total": result.feasibility_escape_added_total,
        "feasibility_escape_active_layers": getattr(result, "feasibility_escape_active_layers", 0),
        "feasibility_escape_gate_reasons": getattr(result, "feasibility_escape_gate_reasons", ""),
    }


def _frg_dlts_row_fields(result, args: argparse.Namespace) -> dict[str, object]:
    return {
        "frg_candidate_supplement_enabled": getattr(args, "frg_candidate_supplement_enabled", False),
        "frg_candidate_supplement_width": getattr(args, "frg_candidate_supplement_width", 0),
        "frg_candidate_focus_n": getattr(args, "frg_candidate_focus_n", -1),
        "frg_candidate_n_window": getattr(args, "frg_candidate_n_window", 2),
        "frg_candidate_max_ranked": getattr(args, "frg_candidate_max_ranked", 96),
        "frg_candidate_scarcity_gate_enabled": getattr(args, "frg_candidate_scarcity_gate_enabled", False),
        "frg_candidate_selected_total": getattr(result, "frg_candidate_selected_total", 0),
        "frg_linear_score_enabled": getattr(args, "frg_linear_score_enabled", False),
        "frg_linear_fill_weight": getattr(args, "frg_linear_fill_weight", 0.0),
        "frg_linear_reduce_weight": getattr(args, "frg_linear_reduce_weight", 0.0),
        "frg_linear_unblocked_weight": getattr(args, "frg_linear_unblocked_weight", 0.0),
        "frg_linear_assignment_weight": getattr(args, "frg_linear_assignment_weight", 0.0),
        "frg_linear_max_height_ordered_weight": getattr(args, "frg_linear_max_height_ordered_weight", 0.0),
        "frg_linear_height_pressure_weight": getattr(args, "frg_linear_height_pressure_weight", 0.0),
        "frg_rank_tiebreak_enabled": getattr(args, "frg_rank_tiebreak_enabled", False),
        "frg_rank_tiebreak_weight": getattr(args, "frg_rank_tiebreak_weight", 0.0),
        "frg_rank_tiebreak_focus_n": getattr(args, "frg_rank_tiebreak_focus_n", -1),
        "frg_scarcity_gate_enabled": getattr(args, "frg_scarcity_gate_enabled", False),
        "frg_scarcity_gate_max_available_stacks": getattr(args, "frg_scarcity_gate_max_available_stacks", 1),
        "frg_scarcity_gate_min_avg_height_pressure": getattr(args, "frg_scarcity_gate_min_avg_height_pressure", 0.85),
        "frg_scarcity_gate_max_reduction_receivers": getattr(args, "frg_scarcity_gate_max_reduction_receivers", -1),
        "frg_scarcity_gate_min_unblocked_bad": getattr(args, "frg_scarcity_gate_min_unblocked_bad", 0),
        "frg_scarcity_gate_min_candidates": getattr(args, "frg_scarcity_gate_min_candidates", 1000),
        "frg_scarcity_gate_cap_slack": getattr(args, "frg_scarcity_gate_cap_slack", 24),
        "frg_layout_available_stack_count": getattr(args, "_frg_layout_available_stack_count", 0),
        "frg_layout_avg_height_pressure": getattr(args, "_frg_layout_avg_height_pressure", 0.0),
        "frg_layout_max_height_ordered_risk": getattr(args, "_frg_layout_max_height_ordered_risk", 0),
        "frg_layout_unblocked_bad_containers": getattr(args, "_frg_layout_unblocked_bad_containers", 0),
        "frg_layout_reduction_receiver_stack_count": getattr(args, "_frg_layout_reduction_receiver_stack_count", 0),
        "frg_layout_reduction_receiver_pressure": getattr(args, "_frg_layout_reduction_receiver_pressure", 0.0),
        "frg_compound_escape_enabled": getattr(args, "frg_compound_escape_enabled", False),
        "frg_compound_escape_width": getattr(args, "frg_compound_escape_width", 0),
        "frg_compound_escape_focus_n": getattr(args, "frg_compound_escape_focus_n", -1),
        "frg_compound_escape_plateau_window": getattr(args, "frg_compound_escape_plateau_window", 5),
        "frg_compound_escape_plateau_delta": getattr(args, "frg_compound_escape_plateau_delta", 0),
        "frg_compound_escape_cap_slack": getattr(args, "frg_compound_escape_cap_slack", 24),
        "frg_compound_escape_min_candidates": getattr(args, "frg_compound_escape_min_candidates", 7000),
        "frg_compound_appended_total": getattr(result, "frg_compound_appended_total", 0),
        "dlts_value_supplement_enabled": getattr(args, "dlts_value_supplement_enabled", False),
        "dlts_value_supplement_width": getattr(args, "dlts_value_supplement_width", 0),
        "dlts_value_selected_total": getattr(result, "dlts_value_selected_total", 0),
    }


def _frg_frontier_child_row_fields(result, args: argparse.Namespace) -> dict[str, object]:
    return {
        "frg_frontier_child_enabled": getattr(args, "frg_frontier_child_enabled", False),
        "frg_frontier_child_mode": getattr(args, "frg_frontier_child_mode", "rollout"),
        "frg_frontier_child_min_stack_count": getattr(args, "frg_frontier_child_min_stack_count", 0),
        "frg_frontier_child_parent_width": getattr(args, "frg_frontier_child_parent_width", 0),
        "frg_frontier_child_limit": getattr(args, "frg_frontier_child_limit", 0),
        "frg_frontier_child_max_steps": getattr(args, "frg_frontier_child_max_steps", 1),
        "frg_frontier_child_append_only": getattr(args, "frg_frontier_child_append_only", False),
        "frg_frontier_child_certify_enabled": getattr(args, "frg_frontier_child_certify_enabled", False),
        "frg_frontier_child_certify_max_steps": getattr(args, "frg_frontier_child_certify_max_steps", 512),
        "frg_frontier_child_certify_time_limit": getattr(args, "frg_frontier_child_certify_time_limit", 0.003),
        "frg_frontier_child_certify_max_generated": getattr(args, "frg_frontier_child_certify_max_generated", 0),
        "frg_frontier_child_completion_max_generated": getattr(
            args,
            "frg_frontier_child_completion_max_generated",
            0,
        ),
        "frg_frontier_child_adaptive_enabled": getattr(args, "frg_frontier_child_adaptive_enabled", False),
        "frg_frontier_child_adaptive_parent_width": getattr(
            args,
            "frg_frontier_child_adaptive_parent_width",
            0,
        ),
        "frg_frontier_child_adaptive_limit": getattr(args, "frg_frontier_child_adaptive_limit", 0),
        "frg_frontier_child_adaptive_max_steps": getattr(args, "frg_frontier_child_adaptive_max_steps", 1),
        "frg_frontier_child_adaptive_plateau_window": getattr(
            args,
            "frg_frontier_child_adaptive_plateau_window",
            0,
        ),
        "frg_frontier_child_adaptive_plateau_delta": getattr(
            args,
            "frg_frontier_child_adaptive_plateau_delta",
            0,
        ),
        "frg_frontier_child_adaptive_min_n": getattr(args, "frg_frontier_child_adaptive_min_n", 0),
        "frg_frontier_child_adaptive_small_stack_max": getattr(
            args,
            "frg_frontier_child_adaptive_small_stack_max",
            0,
        ),
        "frg_frontier_child_adaptive_min_container_count": getattr(
            args,
            "frg_frontier_child_adaptive_min_container_count",
            0,
        ),
        "frg_frontier_child_soft_supply_enabled": getattr(
            args,
            "frg_frontier_child_soft_supply_enabled",
            False,
        ),
        "frg_frontier_child_soft_supply_parent_width": getattr(
            args,
            "frg_frontier_child_soft_supply_parent_width",
            0,
        ),
        "frg_frontier_child_soft_supply_limit": getattr(args, "frg_frontier_child_soft_supply_limit", 0),
        "frg_frontier_child_soft_supply_max_steps": getattr(
            args,
            "frg_frontier_child_soft_supply_max_steps",
            1,
        ),
        "frg_frontier_child_soft_supply_pressure_threshold": getattr(
            args,
            "frg_frontier_child_soft_supply_pressure_threshold",
            0.65,
        ),
        "frg_frontier_child_soft_supply_min_n": getattr(args, "frg_frontier_child_soft_supply_min_n", 0),
        "frg_frontier_child_soft_supply_plateau_window": getattr(
            args,
            "frg_frontier_child_soft_supply_plateau_window",
            2,
        ),
        "frg_frontier_child_soft_supply_plateau_delta": getattr(
            args,
            "frg_frontier_child_soft_supply_plateau_delta",
            0,
        ),
        "frg_frontier_child_soft_supply_last_pressure": getattr(
            args,
            "_frg_frontier_child_soft_supply_last_pressure",
            0.0,
        ),
        "frg_frontier_child_soft_supply_active_layers": getattr(
            args,
            "_frg_frontier_child_soft_supply_active_layers",
            0,
        ),
        "frg_frontier_child_calls": getattr(result, "frg_frontier_child_calls", 0),
        "frg_frontier_child_generated": getattr(result, "frg_frontier_child_generated", 0),
        "frg_frontier_child_added_total": getattr(result, "frg_frontier_child_added_total", 0),
        "frg_frontier_child_macro_edge_enabled": getattr(args, "frg_frontier_child_macro_edge_enabled", False),
        "frg_macro_edge_recorded_total": getattr(result, "frg_macro_edge_recorded_total", 0),
        "frg_macro_edge_selected_total": getattr(result, "frg_macro_edge_selected_total", 0),
        "frg_macro_edge_expanded_steps_total": getattr(result, "frg_macro_edge_expanded_steps_total", 0),
        "macro_chain_witness_enabled": getattr(args, "macro_chain_witness_enabled", False),
        "macro_chain_witness_respect_time_limit": getattr(args, "macro_chain_witness_respect_time_limit", True),
        "macro_chain_witness_parent_width": getattr(args, "macro_chain_witness_parent_width", 0),
        "macro_chain_witness_prefix_steps": getattr(args, "macro_chain_witness_prefix_steps", 0),
        "macro_chain_witness_max_steps": getattr(args, "macro_chain_witness_max_steps", 0),
        "macro_chain_witness_time_limit": getattr(args, "macro_chain_witness_time_limit", 0.0),
        "macro_chain_witness_max_generated": getattr(args, "macro_chain_witness_max_generated", 0),
        "macro_chain_witness_min_stack_count": getattr(args, "macro_chain_witness_min_stack_count", 0),
        "macro_chain_witness_min_height": getattr(args, "macro_chain_witness_min_height", 0),
        "macro_chain_witness_min_n": getattr(args, "macro_chain_witness_min_n", 0),
        "macro_chain_witness_min_candidates": getattr(args, "macro_chain_witness_min_candidates", 0),
        "macro_chain_witness_calls": getattr(result, "macro_chain_witness_calls", 0),
        "macro_chain_witness_generated": getattr(result, "macro_chain_witness_generated", 0),
        "macro_chain_witness_added_total": getattr(result, "macro_chain_witness_added_total", 0),
        "macro_chain_witness_selected_total": getattr(result, "macro_chain_witness_selected_total", 0),
        "frg_frontier_child_terminal_witness_only_enabled": getattr(
            args,
            "frg_frontier_child_terminal_witness_only_enabled",
            False,
        ),
        "frg_frontier_child_terminal_witness_respect_cap": getattr(
            args,
            "frg_frontier_child_terminal_witness_respect_cap",
            True,
        ),
        "frg_terminal_witness_recorded_total": getattr(result, "frg_terminal_witness_recorded_total", 0),
        "frg_terminal_witness_grafted_total": getattr(result, "frg_terminal_witness_grafted_total", 0),
        "frg_terminal_witness_used": getattr(result, "frg_terminal_witness_used", False),
        "frg_terminal_witness_moves": getattr(result, "frg_terminal_witness_moves", 0),
    }


def _bs_frg_qubo_row_fields(result, args: argparse.Namespace) -> dict[str, object]:
    return {
        "bs_frg_qubo_enabled": getattr(args, "bs_frg_qubo_enabled", False),
        "bs_frg_qubo_supplement_width": getattr(args, "bs_frg_qubo_supplement_width", 0),
        "bs_frg_qubo_focus_n": getattr(args, "bs_frg_qubo_focus_n", -1),
        "bs_frg_qubo_n_window": getattr(args, "bs_frg_qubo_n_window", 2),
        "bs_frg_qubo_min_candidates": getattr(args, "bs_frg_qubo_min_candidates", 64),
        "bs_frg_qubo_max_vars": getattr(args, "bs_frg_qubo_max_vars", 64),
        "bs_frg_qubo_rank_weight": getattr(args, "bs_frg_qubo_rank_weight", 0.05),
        "bs_frg_qubo_same_parent_penalty": getattr(args, "bs_frg_qubo_same_parent_penalty", 0.35),
        "bs_frg_qubo_same_source_penalty": getattr(args, "bs_frg_qubo_same_source_penalty", 0.0),
        "bs_frg_qubo_signature_penalty": getattr(args, "bs_frg_qubo_signature_penalty", 0.15),
        "bs_frg_qubo_reduce_signature_penalty": getattr(args, "bs_frg_qubo_reduce_signature_penalty", 0.0),
        "bs_frg_qubo_n_band_penalty": getattr(args, "bs_frg_qubo_n_band_penalty", 0.05),
        "bs_frg_qubo_n_band_size": getattr(args, "bs_frg_qubo_n_band_size", 1),
        "bs_frg_qubo_calls": getattr(result, "bs_frg_qubo_calls", 0),
        "bs_frg_qubo_vars": getattr(result, "bs_frg_qubo_vars", 0),
        "bs_frg_qubo_time_sec": getattr(result, "bs_frg_qubo_time_sec", 0.0),
        "bs_frg_qubo_selected_total": getattr(result, "bs_frg_qubo_selected_total", 0),
    }


def _wang2017_task_row_fields(result, args: argparse.Namespace) -> dict[str, object]:
    return {
        "wang2017_task_enabled": getattr(args, "wang2017_task_enabled", False),
        "wang2017_task_supplement_width": getattr(args, "wang2017_task_supplement_width", 0),
        "wang2017_task_focus_n": getattr(args, "wang2017_task_focus_n", 8),
        "wang2017_task_n_window": getattr(args, "wang2017_task_n_window", 1),
        "wang2017_task_plateau_window": getattr(args, "wang2017_task_plateau_window", 0),
        "wang2017_task_plateau_delta": getattr(args, "wang2017_task_plateau_delta", 0),
        "wang2017_task_calls": getattr(result, "wang2017_task_calls", 0),
        "wang2017_task_added_total": getattr(result, "wang2017_task_added_total", 0),
        "wang2017_task_active_layers": getattr(result, "wang2017_task_active_layers", 0),
        "wang2017_task_selected_tier_protection_total": getattr(
            result,
            "wang2017_task_selected_tier_protection_total",
            0,
        ),
        "wang2017_task_selected_stable_height_total": getattr(
            result,
            "wang2017_task_selected_stable_height_total",
            0,
        ),
    }


def _priority_group_row_dict_from_instance_result(
    result,
    instance: Instance,
    args: argparse.Namespace,
) -> dict[str, object]:
    n0 = future_disorder_count_stacks(instance.stacks)
    cap13 = int(math.floor(min(1.3, 2.0) * n0 + 1e-9))
    cap15 = int(math.floor(min(1.5, 2.0) * n0 + 1e-9))
    if result.moves:
        _ok, _msg, _stacks, n_trace, max_g = _simulate_priority_group_moves(instance, result.moves)
        final_n = n_trace[-1]
        moves_count = len(result.moves)
    else:
        initial_ok, _msg, _stacks, n_trace, max_g = _simulate_priority_group_moves(instance, [])
        final_n = n_trace[-1]
        moves_count = 0 if initial_ok and final_n == 0 else None
    row = {
        "case": result.case,
        "solver": SOLVER_NAME,
        "n0": n0,
        "cap_13": cap13,
        "cap_15": cap15,
        "moves": moves_count,
        "final_n": final_n,
        "reduction": n0 - final_n,
        "g_end": (moves_count + final_n) if moves_count is not None else None,
        "max_g": max_g,
        "g_ok_13": max_g <= cap13,
        "g_ok_15": max_g <= cap15,
        "runtime_sec": result.runtime_sec,
        "expanded_states": result.expanded,
        "generated_states": result.generated,
        "visited_states": result.visited,
        "qubo_calls": result.qubo_calls,
        "qubo_total_vars": result.qubo_vars,
        "qubo_time_sec": result.qubo_time_sec,
        "status": (
            f"{result.status}; solver={SOLVER_NAME}; representation=compact_bytes; "
            f"open_cpmp_semantics=priority_groups; target_stop=N0; proof=not_optimality_proof"
        ),
        "profile": getattr(args, "solver_profile", "speed_v2"),
        "solver_profile": getattr(args, "solver_profile", "speed_v2"),
        "qubo_influence_profile": getattr(args, "qubo_influence_profile", "stable"),
        "business_cap_enabled": getattr(args, "business_cap_enabled", True),
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
        "qubo_start_layer": getattr(args, "qubo_start_layer", 0),
        "qubo_focus_n": args.qubo_focus_n,
        "qubo_first": args.qubo_first,
        "max_qubo_vars": args.max_qubo_vars,
        "qubo_core_width": args.qubo_core_width,
        "qubo_supplement_width": args.qubo_supplement_width,
        "qubo_cohorts_per_layer": args.qubo_cohorts_per_layer,
        "qubo_pair_policy": getattr(args, "qubo_pair_policy", "baseline"),
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
        "post_polish_mode": getattr(args, "post_polish_mode", "legacy"),
        "post_polish_accept": getattr(args, "post_polish_accept", "better"),
        "post_polish_respect_time_limit": getattr(args, "post_polish_respect_time_limit", True),
        "post_polish_max_attempts": getattr(args, "post_polish_max_attempts", 0),
        "post_polish_time_budget": getattr(args, "post_polish_time_budget", 0.0),
        "inner_jobs": getattr(args, "inner_jobs", 1),
        "post_polish_window_min_stacks": getattr(args, "post_polish_window_min_stacks", 0),
        "post_polish_attempts": result.post_polish_attempts,
        "post_polish_improvements": result.post_polish_improvements,
        "post_polish_moves_saved": result.post_polish_moves_saved,
        "post_polish_window_min_size": getattr(args, "post_polish_window_min_size", 0),
        "post_polish_window_max_size": getattr(args, "post_polish_window_max_size", 0),
        "post_polish_window_stride": getattr(args, "post_polish_window_stride", 0),
        "post_polish_window_budget": getattr(args, "post_polish_window_budget", 0),
        "post_polish_window_qubo_vars": getattr(args, "post_polish_window_qubo_vars", 0),
        "post_polish_window_select": getattr(args, "post_polish_window_select", 0),
        "post_polish_window_repair_expanded": getattr(args, "post_polish_window_repair_expanded", 0),
        "post_polish_window_repair_depth_extra": getattr(args, "post_polish_window_repair_depth_extra", 0),
        "post_polish_shortcut_replay_enabled": getattr(args, "post_polish_shortcut_replay_enabled", False),
        "post_polish_shortcut_min_delete": getattr(args, "post_polish_shortcut_min_delete", 0),
        "post_polish_shortcut_max_delete": getattr(args, "post_polish_shortcut_max_delete", 0),
        "post_polish_shortcut_window_budget": getattr(args, "post_polish_shortcut_window_budget", 0),
        "post_polish_bridge_scan_enabled": getattr(args, "post_polish_bridge_scan_enabled", False),
        "post_polish_bridge_scan_window_min_size": getattr(args, "post_polish_bridge_scan_window_min_size", 0),
        "post_polish_bridge_scan_window_max_size": getattr(args, "post_polish_bridge_scan_window_max_size", 0),
        "post_polish_bridge_scan_min_moves": getattr(args, "post_polish_bridge_scan_min_moves", 0),
        "post_polish_bridge_scan_budget": getattr(args, "post_polish_bridge_scan_budget", 0),
        "post_polish_bridge_scan_expanded": getattr(args, "post_polish_bridge_scan_expanded", 0),
        "post_polish_window_candidates": result.post_polish_window_candidates,
        "post_polish_window_qubo_selected": result.post_polish_window_qubo_selected,
        "post_polish_window_attempts": result.post_polish_window_attempts,
        "post_polish_window_successes": result.post_polish_window_successes,
        "post_polish_window_moves_saved": result.post_polish_window_moves_saved,
        "post_polish_window_time_sec": result.post_polish_window_time_sec,
        "post_polish_successor_cache": getattr(args, "post_polish_successor_cache", False),
        "global_reservoir": getattr(args, "global_reservoir", False),
        "reservoir_size": result.reservoir_size,
        "peak_reservoir": result.peak_reservoir,
        "reservoir_added_total": result.reservoir_added_total,
        "reservoir_pulled_total": result.reservoir_pulled_total,
        "stream_candidate_limit": getattr(args, "stream_candidate_limit", 0),
        "stream_candidate_trim_factor": getattr(args, "stream_candidate_trim_factor", 2.0),
        "stream_bucket_topk": getattr(args, "stream_bucket_topk", 0),
        "stream_pruned_candidates": result.stream_pruned_candidates,
        "rank_pool_size": getattr(args, "rank_pool_size", 0),
        "rank_pool_min_candidates": getattr(args, "rank_pool_min_candidates", 0),
        "rank_bucket_topk": getattr(args, "rank_bucket_topk", 0),
        "rank_bucket_key": getattr(args, "rank_bucket_key", "n_depth_parent"),
        "rank_bucket_n_band_size": getattr(args, "rank_bucket_n_band_size", 0),
        "rank_bucket_topk_high_n": getattr(args, "rank_bucket_topk_high_n", 0),
        "rank_bucket_topk_min_candidates": getattr(args, "rank_bucket_topk_min_candidates", 0),
        "rank_pool_truncated_candidates": result.rank_pool_truncated_candidates,
        "scheduler_adaptive_min_stacks": getattr(args, "scheduler_adaptive_min_stacks", 0),
        "scheduler_adaptive_high_n": getattr(args, "scheduler_adaptive_high_n", 0),
        "scheduler_adaptive_cv1_only": getattr(args, "scheduler_adaptive_cv1_only", False),
        "scheduler_adaptive_max_duplication_ratio": getattr(args, "scheduler_adaptive_max_duplication_ratio", 1.0),
        "scheduler_adaptive_min_best_n_ratio": getattr(args, "scheduler_adaptive_min_best_n_ratio", 0.0),
        "scheduler_adaptive_max_candidate_pressure": getattr(args, "scheduler_adaptive_max_candidate_pressure", 0.0),
        "scheduler_adaptive_max_parent_top_share": getattr(args, "scheduler_adaptive_max_parent_top_share", 0.0),
        "scheduler_adaptive_plateau_window": getattr(args, "scheduler_adaptive_plateau_window", 0),
        "scheduler_adaptive_cap_slack": getattr(args, "scheduler_adaptive_cap_slack", 0),
        "scheduler_state_controller": getattr(args, "scheduler_state_controller", "off"),
        "scheduler_state_contract_pressure": getattr(args, "scheduler_state_contract_pressure", 0.0),
        "scheduler_state_max_contract_pressure": getattr(args, "scheduler_state_max_contract_pressure", 0.0),
        "scheduler_state_min_best_n_ratio": getattr(args, "scheduler_state_min_best_n_ratio", 0.0),
        "scheduler_state_progress_window": getattr(args, "scheduler_state_progress_window", 0),
        "scheduler_state_progress_delta": getattr(args, "scheduler_state_progress_delta", 0),
        "scheduler_state_failed_contract_cooldown": getattr(args, "scheduler_state_failed_contract_cooldown", 0),
        "scheduler_state_post_rescue_plateau_window": getattr(args, "scheduler_state_post_rescue_plateau_window", 0),
        "scheduler_state_post_rescue_plateau_delta": getattr(args, "scheduler_state_post_rescue_plateau_delta", 0),
        "scheduler_state_post_rescue_min_ratio": getattr(args, "scheduler_state_post_rescue_min_ratio", 0.0),
        "scheduler_state_post_rescue_action": getattr(args, "scheduler_state_post_rescue_action", ""),
        "scheduler_state_initial_plateau_window": getattr(args, "scheduler_state_initial_plateau_window", 0),
        "scheduler_state_initial_plateau_min_ratio": getattr(args, "scheduler_state_initial_plateau_min_ratio", 0.0),
        "scheduler_state_initial_plateau_action": getattr(args, "scheduler_state_initial_plateau_action", ""),
        "scheduler_state_initial_plateau_lock": getattr(args, "scheduler_state_initial_plateau_lock", False),
        "scheduler_state_initial_plateau_min_duplication": getattr(args, "scheduler_state_initial_plateau_min_duplication", 0.0),
        "scheduler_state_initial_plateau_unique_min_n": getattr(args, "scheduler_state_initial_plateau_unique_min_n", 0),
        "scheduler_state_protect_parent_top_share": getattr(args, "scheduler_state_protect_parent_top_share", 0.0),
        "scheduler_state_protect_n_band": getattr(args, "scheduler_state_protect_n_band", 0),
        "scheduler_state_top_window": getattr(args, "scheduler_state_top_window", 0),
        "scheduler_state_rescue_plateau_window": getattr(args, "scheduler_state_rescue_plateau_window", 0),
        "scheduler_state_rescue_plateau_delta": getattr(args, "scheduler_state_rescue_plateau_delta", 0),
        "scheduler_state_rescue_cap_slack": getattr(args, "scheduler_state_rescue_cap_slack", 0),
        "scheduler_state_last_action": getattr(args, "_scheduler_state_last_action", ""),
        "scheduler_state_last_effective_action": getattr(args, "_scheduler_state_last_effective_action", ""),
        "scheduler_state_locked_action": getattr(args, "_scheduler_state_locked_action", ""),
        "scheduler_state_contract_layers": getattr(args, "_scheduler_state_contract_layers", 0),
        "scheduler_state_blend_layers": getattr(args, "_scheduler_state_blend_layers", 0),
        "scheduler_state_protect_layers": getattr(args, "_scheduler_state_protect_layers", 0),
        "scheduler_state_rescue_layers": getattr(args, "_scheduler_state_rescue_layers", 0),
        "adaptive_high_n": getattr(args, "adaptive_high_n", 0),
        "adaptive_low_n": getattr(args, "adaptive_low_n", 0),
        "adaptive_high_frontier": getattr(args, "adaptive_high_frontier", 0),
        "adaptive_mid_frontier": getattr(args, "adaptive_mid_frontier", 0),
        "adaptive_low_frontier": getattr(args, "adaptive_low_frontier", 0),
        "adaptive_core_ratio": getattr(args, "adaptive_core_ratio", 0.0),
        "adaptive_candidate_pool": getattr(args, "adaptive_candidate_pool", 0),
        "adaptive_controller": getattr(args, "adaptive_controller", "tiered"),
        "adaptive_expand_frontier": getattr(args, "adaptive_expand_frontier", 0),
        "adaptive_contract_pressure": getattr(args, "adaptive_contract_pressure", 0.0),
        "adaptive_mid_pressure": getattr(args, "adaptive_mid_pressure", 0.0),
        "adaptive_min_best_n_ratio": getattr(args, "adaptive_min_best_n_ratio", 0.0),
        "adaptive_max_duplication_ratio": getattr(args, "adaptive_max_duplication_ratio", 1.0),
        "adaptive_parent_top_share": getattr(args, "adaptive_parent_top_share", 0.0),
        "adaptive_plateau_window": getattr(args, "adaptive_plateau_window", 0),
        "adaptive_plateau_delta": getattr(args, "adaptive_plateau_delta", 0),
        "adaptive_cap_slack": getattr(args, "adaptive_cap_slack", 0),
        "adaptive_large_stack_min_stacks": getattr(args, "adaptive_large_stack_min_stacks", 0),
        "adaptive_large_stack_expand_frontier": getattr(args, "adaptive_large_stack_expand_frontier", 0),
        "adaptive_large_stack_plateau_window": getattr(args, "adaptive_large_stack_plateau_window", 0),
        "adaptive_large_stack_cap_slack": getattr(args, "adaptive_large_stack_cap_slack", 0),
        "adaptive_contract_layers": getattr(args, "_adaptive_contract_layers", 0),
        "adaptive_mid_layers": getattr(args, "_adaptive_mid_layers", 0),
        "adaptive_wide_layers": getattr(args, "_adaptive_wide_layers", 0),
        "adaptive_expand_layers": getattr(args, "_adaptive_expand_layers", 0),
        "adaptive_last_width": getattr(args, "_adaptive_last_width", 0),
        "adaptive_last_reason": getattr(args, "_adaptive_last_reason", ""),
        "bucket_key": getattr(args, "bucket_key", "n_depth"),
        "bucket_n_band_size": getattr(args, "bucket_n_band_size", 0),
        "bucket_core_ratio": getattr(args, "bucket_core_ratio", 0.0),
        "bucket_qubo_share": getattr(args, "bucket_qubo_share", 0.0),
        "bucket_max_qubo_vars": getattr(args, "bucket_max_qubo_vars", 0),
        "bucket_qubo_select": getattr(args, "bucket_qubo_select", 0),
        "bucket_min_candidates": getattr(args, "bucket_min_candidates", 0),
        "rescue_layers_used": result.rescue_layers_used,
        "scarcity_inner_rescue_layers": getattr(result, "scarcity_inner_rescue_layers", 0),
        "scarcity_inner_rescue_gate_reasons": getattr(result, "scarcity_inner_rescue_gate_reasons", ""),
        "scarcity_inner_rescue_enabled": getattr(args, "scarcity_inner_rescue_enabled", False),
        "scarcity_inner_rescue_risk_threshold": getattr(args, "scarcity_inner_rescue_risk_threshold", 0.0),
        "scarcity_inner_rescue_max_l0": getattr(args, "scarcity_inner_rescue_max_l0", 0),
        "scarcity_inner_rescue_min_one_safe_ratio": getattr(
            args, "scarcity_inner_rescue_min_one_safe_ratio", 0.0
        ),
        "scarcity_inner_rescue_min_n": getattr(args, "scarcity_inner_rescue_min_n", 0),
        "scarcity_inner_rescue_min_candidates": getattr(args, "scarcity_inner_rescue_min_candidates", 0),
        "tail_lookahead_enabled": getattr(args, "tail_lookahead_enabled", False),
        "tail_lookahead_core_path_only": getattr(args, "tail_lookahead_core_path_only", False),
        "tail_lookahead_max_stack_count": getattr(args, "tail_lookahead_max_stack_count", 0),
        "tail_lookahead_focus_n": getattr(args, "tail_lookahead_focus_n", 2),
        "tail_lookahead_depth": getattr(args, "tail_lookahead_depth", 3),
        "tail_lookahead_max_seed_candidates": getattr(args, "tail_lookahead_max_seed_candidates", 0),
        "tail_lookahead_supplement_width": getattr(args, "tail_lookahead_supplement_width", 0),
        "tail_lookahead_max_expanded": getattr(args, "tail_lookahead_max_expanded", 0),
        "tail_lookahead_calls": getattr(result, "tail_lookahead_calls", 0),
        "tail_lookahead_selected_total": getattr(result, "tail_lookahead_selected_total", 0),
        "tail_lookahead_goal_reachable_total": getattr(result, "tail_lookahead_goal_reachable_total", 0),
        "tail_lookahead_best_delta_total": getattr(result, "tail_lookahead_best_delta_total", 0),
        "tail_lookahead_expanded_total": getattr(result, "tail_lookahead_expanded_total", 0),
        "terminal_bridge_enabled": getattr(args, "terminal_bridge_enabled", False),
        "terminal_bridge_focus_n": getattr(args, "terminal_bridge_focus_n", 1),
        "terminal_bridge_depth": getattr(args, "terminal_bridge_depth", 3),
        "terminal_bridge_max_seed_candidates": getattr(args, "terminal_bridge_max_seed_candidates", 0),
        "terminal_bridge_max_expanded": getattr(args, "terminal_bridge_max_expanded", 0),
        "terminal_bridge_calls": getattr(result, "terminal_bridge_calls", 0),
        "terminal_bridge_successes": getattr(result, "terminal_bridge_successes", 0),
        "terminal_bridge_expanded_total": getattr(result, "terminal_bridge_expanded_total", 0),
        "terminal_bridge_moves": getattr(result, "terminal_bridge_moves", 0),
        "tail_risk_reserve_enabled": getattr(args, "tail_risk_reserve_enabled", False),
        "tail_risk_focus_n": getattr(args, "tail_risk_focus_n", 2),
        "tail_risk_n_window": getattr(args, "tail_risk_n_window", 1),
        "tail_risk_reserve_width": getattr(args, "tail_risk_reserve_width", 0),
        "tail_risk_max_scan": getattr(args, "tail_risk_max_scan", 0),
        "tail_risk_min_candidates": getattr(args, "tail_risk_min_candidates", 0),
        "tail_risk_min_depth_slack": getattr(args, "tail_risk_min_depth_slack", 0),
        "tail_risk_max_depth_slack": getattr(args, "tail_risk_max_depth_slack", 0),
        "tail_risk_max_stack_count": getattr(args, "tail_risk_max_stack_count", 0),
        "tail_risk_reserve_calls": getattr(result, "tail_risk_reserve_calls", 0),
        "tail_risk_reserve_selected_total": getattr(result, "tail_risk_reserve_selected_total", 0),
        "tail_risk_reserve_scanned_total": getattr(result, "tail_risk_reserve_scanned_total", 0),
        "tail_risk_reserve_one_step_improved_total": getattr(
            result,
            "tail_risk_reserve_one_step_improved_total",
            0,
        ),
        "frg_seed_enabled": getattr(args, "frg_seed_enabled", False),
        "frg_seed_time_limit": getattr(args, "frg_seed_time_limit", 0.0),
        "frg_seed_min_max_steps": getattr(args, "frg_seed_min_max_steps", 0),
        "frg_seed_max_steps_multiplier": getattr(args, "frg_seed_max_steps_multiplier", 0),
        "frg_seed_max_steps_cap": getattr(args, "frg_seed_max_steps_cap", 0),
        "frg_seed_respect_cap": getattr(args, "frg_seed_respect_cap", True),
        "frg_seed_prune_to_incumbent": getattr(args, "frg_seed_prune_to_incumbent", True),
        "frg_seed_allow_large_over_cap": getattr(args, "frg_seed_allow_large_over_cap", False),
        "frg_seed_paper_pure": getattr(args, "frg_seed_paper_pure", True),
        "frg_seed_beam_enabled": getattr(args, "frg_seed_beam_enabled", False),
        "frg_seed_beam_width": getattr(args, "frg_seed_beam_width", 1000),
        "frg_seed_candidate_limit_per_state": getattr(args, "frg_seed_candidate_limit_per_state", 3),
        "frg_seed_beam_large_only": getattr(args, "frg_seed_beam_large_only", False),
        "frg_seed_beam_min_stack_count": getattr(args, "frg_seed_beam_min_stack_count", 0),
        "frg_seed_beam_min_height": getattr(args, "frg_seed_beam_min_height", 0),
        "frg_seed_small_time_limit": getattr(args, "frg_seed_small_time_limit", 0.0),
        "frg_seed_cache_input": getattr(result, "frg_seed_cache_input", ""),
        "frg_seed_cache_key": getattr(result, "frg_seed_cache_key", ""),
        "frg_seed_cache_hit": getattr(result, "frg_seed_cache_hit", False),
        "frg_seed_cache_source": getattr(result, "frg_seed_cache_source", ""),
        "frg_seed_moves": getattr(result, "frg_seed_moves", 0),
        "frg_seed_runtime_sec": getattr(result, "frg_seed_runtime_sec", 0.0),
        "frg_seed_used": getattr(result, "frg_seed_used", False),
        "frg_seed_status": getattr(result, "frg_seed_status", ""),
        "frg_seed_generated": getattr(result, "frg_seed_generated", 0),
        "frg_seed_cap_ok": getattr(result, "frg_seed_cap_ok", False),
        "frg_seed_large_layout_ok": getattr(result, "frg_seed_large_layout_ok", False),
        "scarcity_group_qubo_enabled": getattr(args, "scarcity_group_qubo_enabled", False),
        "scarcity_group_qubo_threshold": getattr(args, "scarcity_group_qubo_threshold", 0.0),
        "scarcity_group_qubo_plateau_window": getattr(args, "scarcity_group_qubo_plateau_window", 0),
        "scarcity_group_qubo_plateau_delta": getattr(args, "scarcity_group_qubo_plateau_delta", 0),
        "scarcity_group_qubo_min_best_n_ratio": getattr(args, "scarcity_group_qubo_min_best_n_ratio", 0.0),
        "scarcity_group_qubo_min_candidates": getattr(args, "scarcity_group_qubo_min_candidates", 0),
        "scarcity_group_qubo_min_n": getattr(args, "scarcity_group_qubo_min_n", 0),
        "initial_workload_hardness": getattr(args, "_initial_workload_hardness", 0.0),
        "initial_scarcity_risk": getattr(args, "_initial_scarcity_risk", 0.0),
        "initial_scarcity_n0": getattr(args, "_initial_scarcity_n0", 0),
        "initial_scarcity_l0": getattr(args, "_initial_scarcity_l0", 0),
        "initial_scarcity_one_safe_ratio": getattr(args, "_initial_scarcity_one_safe_ratio", 0.0),
        "initial_light_route_score": getattr(args, "_initial_light_route_score", 0.0),
    }
    row.update(_wang_pair_qubo_row_fields(result, args))
    row.update(_unified_wang_row_fields(result, args))
    row.update(_union_diversity_qubo_row_fields(result, args))
    row.update(_quality_reservoir_row_fields(result, args))
    row.update(_micro_rollout_row_fields(result, args))
    row.update(_feasibility_escape_row_fields(result, args))
    row.update(_frg_dlts_row_fields(result, args))
    row.update(_frg_frontier_child_row_fields(result, args))
    row.update(_bs_frg_qubo_row_fields(result, args))
    row.update(_wang2017_task_row_fields(result, args))
    return row


def _open_row_dict_from_instance_result(result, instance: Instance, args: argparse.Namespace) -> dict[str, object]:
    if requires_priority_group_semantics(instance):
        return _priority_group_row_dict_from_instance_result(result, instance, args)
    return _row_dict_from_instance_result(result, instance, args)


def _validate_priority_group_result(
    result,
    instance: Instance,
    business_alpha: float,
    business_cap_enabled: bool = True,
) -> tuple[bool, bool, str]:
    if not result.moves:
        goal_ok, goal_msg, _stacks, _n_trace, max_g = _simulate_priority_group_moves(instance, [])
        cap = int(math.floor(min(float(business_alpha), 2.0) * future_disorder_count_stacks(instance.stacks) + 1e-9))
        return (True if not business_cap_enabled else max_g <= cap), goal_ok, goal_msg if goal_ok else result.status
    goal_ok, goal_msg, _stacks, _n_trace, max_g = _simulate_priority_group_moves(instance, result.moves)
    cap = int(math.floor(min(float(business_alpha), 2.0) * future_disorder_count_stacks(instance.stacks) + 1e-9))
    business_ok = True if not business_cap_enabled else max_g <= cap
    return business_ok, goal_ok, goal_msg


def _open_validate_instance_result(
    result,
    instance: Instance,
    business_alpha: float,
    business_cap_enabled: bool = True,
) -> tuple[bool, bool, str]:
    if requires_priority_group_semantics(instance):
        return _validate_priority_group_result(
            result,
            instance,
            business_alpha,
            business_cap_enabled=business_cap_enabled,
        )
    return _validate_instance_result(
        result,
        instance,
        business_alpha=business_alpha,
        business_cap_enabled=business_cap_enabled,
    )


def read_dlts_reference(path: Path | None) -> DltsReference:
    if path is None or not path.exists():
        return DltsReference()
    optimal_moves: int | None = None
    max_height: int | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("AA tiers-stacks-time"):
            parts = line.split()
            if len(parts) >= 3:
                max_height = int(parts[2])
        elif line.startswith("opt="):
            optimal_moves = int(line.split("=", 1)[1].strip())
    return DltsReference(optimal_moves=optimal_moves, max_height=max_height)


def ensure_dlts_source(source: Path, auto_clone: bool = True, refresh: bool = False) -> Path:
    if refresh and source.exists():
        subprocess.run(["git", "-C", str(source), "pull", "--ff-only"], check=True)
    if (source / "instances").is_dir():
        return source
    if not auto_clone:
        raise FileNotFoundError(f"DLTS source not found at {source}; pass --auto-clone or --source")
    source.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", DLTS_URL, str(source)], check=True)
    return source


def discover_dlts_cases(
    source: Path,
    families: list[str] | None = None,
    limit_per_family: int | None = None,
    max_total: int | None = None,
) -> list[OpenCpmpCase]:
    instances_dir = source / "instances"
    refs_dir = source / "reference_solutions"
    selected_families = set(families or [])
    cases: list[OpenCpmpCase] = []
    for family_dir in sorted(path for path in instances_dir.iterdir() if path.is_dir()):
        if selected_families and family_dir.name not in selected_families:
            continue
        dat_files = sorted(family_dir.glob("*.dat"), key=_case_sort_key)
        if limit_per_family is not None:
            dat_files = dat_files[:limit_per_family]
        for dat_path in dat_files:
            ref_path = refs_dir / family_dir.name / f"{dat_path.stem}.out"
            ref = read_dlts_reference(ref_path if ref_path.exists() else None)
            cases.append(
                OpenCpmpCase(
                    family=family_dir.name,
                    case_id=f"{family_dir.name}_{dat_path.stem}",
                    instance_path=dat_path,
                    reference_path=ref_path if ref_path.exists() else None,
                    reference=ref,
                )
            )
            if max_total is not None and len(cases) >= max_total:
                return cases
    return cases


def _numeric(row: dict[str, object], field: str, default: float = 0.0) -> float:
    value = row.get(field)
    if value in (None, ""):
        return default
    return float(value)


def _is_true(value: object) -> bool:
    return value in (True, "True", "true", "1", 1)


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _open_row_valid(row: dict[str, object]) -> bool:
    primary_ok = row.get("primary_ok")
    if primary_ok not in (None, ""):
        return _is_true(primary_ok)
    return (
        _is_true(row.get("business_ok"))
        and _is_true(row.get("goal_validation_ok"))
        and _optional_int(row.get("final_n")) == 0
        and _optional_int(row.get("moves")) is not None
    )


def _open_full_quality_penalty(row: dict[str, object], penalty_factor: int = 4) -> int:
    moves = _optional_int(row.get("moves"))
    if _open_row_valid(row) and moves is not None:
        return moves
    n0 = int(_numeric(row, "n0", 0))
    reference = _optional_int(row.get("reference_moves"))
    base = reference if reference is not None else int(_numeric(row, "cap_13", n0))
    return max(base, n0) + int(penalty_factor) * max(1, n0)


def _stable_solver_priority(row: dict[str, object]) -> int:
    return 0 if str(row.get("solver_profile")) == "scheduler_state_v5_wang_rescue_polish" else 1


def _select_open_no_regret_row(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("no open benchmark rows to select")
    valid_rows = [row for row in rows if _open_row_valid(row)]
    if valid_rows:
        return min(
            valid_rows,
            key=lambda row: (
                int(_numeric(row, "moves", 10**9)),
                float(_numeric(row, "runtime_sec", 0.0)),
                _stable_solver_priority(row),
            ),
        )
    return min(
        rows,
        key=lambda row: (
            _open_full_quality_penalty(row),
            float(_numeric(row, "runtime_sec", 0.0)),
            _stable_solver_priority(row),
        ),
    )


def _open_full_quality_summary(
    rows: list[dict[str, object]],
    penalty_factor: int = 4,
) -> dict[tuple[str, str], dict[str, float | int]]:
    groups: dict[tuple[str, str], dict[str, float | int]] = {}
    for row in rows:
        key = (str(row["variant"]), str(row["family"]))
        item = groups.setdefault(
            key,
            {
                "cases": 0,
                "valid": 0,
                "invalid_or_unsolved": 0,
                "moves_total": 0,
                "full_quality_penalty_total": 0,
                "runtime_total": 0.0,
            },
        )
        valid = _open_row_valid(row)
        item["cases"] = int(item["cases"]) + 1
        item["valid"] = int(item["valid"]) + (1 if valid else 0)
        item["invalid_or_unsolved"] = int(item["invalid_or_unsolved"]) + (0 if valid else 1)
        item["moves_total"] = int(item["moves_total"]) + (int(_numeric(row, "moves", 0)) if valid else 0)
        item["full_quality_penalty_total"] = int(item["full_quality_penalty_total"]) + _open_full_quality_penalty(
            row,
            penalty_factor=penalty_factor,
        )
        item["runtime_total"] = float(item["runtime_total"]) + _numeric(row, "runtime_sec", 0.0)
    return groups


def _write_open_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OPEN_RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _family_size_sort_key(family: str) -> tuple[int, int, str]:
    match = re.match(r"(\d+)x(\d+)_", family)
    if not match:
        return 10**9, 10**9, family
    return int(match.group(1)), int(match.group(2)), family


def _write_summary(rows: list[dict[str, object]], output: Path, summary: Path, source: Path) -> None:
    lines = [
        "# Open CPMP Benchmark Summary",
        "",
        f"source: `{_display_path(source)}`",
        f"result_csv: `{_display_path(output)}`",
        "",
        "| variant | family | cases | valid | invalid/unsolved | moves total | full quality penalty | avg full penalty | avg runtime sec | avg ref gap | qcalls | qtime sec |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["variant"]), str(row["family"])), []).append(row)
    quality = _open_full_quality_summary(rows)
    for (variant, family), selected in sorted(groups.items(), key=lambda item: (item[0][0], _family_size_sort_key(item[0][1]))):
        solved = [row for row in selected if _open_row_valid(row)]
        gaps = [
            _numeric(row, "move_gap")
            for row in solved
            if row.get("move_gap") not in (None, "")
        ]
        qitem = quality[(variant, family)]
        cases = int(qitem["cases"])
        penalty_total = int(qitem["full_quality_penalty_total"])
        lines.append(
            f"| {variant} | {family} | {len(selected)} | {int(qitem['valid'])} | "
            f"{int(qitem['invalid_or_unsolved'])} | "
            f"{int(qitem['moves_total'])} | {penalty_total} | "
            f"{penalty_total / cases if cases else 0.0:.3f} | "
            f"{float(qitem['runtime_total']) / cases if cases else 0.0:.3f} | "
            f"{sum(gaps) / len(gaps) if gaps else 0.0:.3f} | "
            f"{int(sum(_numeric(row, 'qubo_calls') for row in selected))} | "
            f"{sum(_numeric(row, 'qubo_time_sec') for row in selected):.3f} |"
        )
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _variant_names(name: str) -> list[str]:
    if name == "both":
        return ["qubo", "classic_noqubo"]
    if name == "all":
        return ["qubo", "qubo_adaptive", "classic_noqubo"]
    if name == "wang_all":
        return [
            "qubo",
            "qubo_scheduler_state_v5",
            "qubo_scheduler_state_v5_wang_pair",
            "qubo_scheduler_state_v5_wang_features",
            "qubo_scheduler_state_v5_wang_rescue",
            "qubo_scheduler_state_v5_wang_rescue_suffix_polish",
            "qubo_scheduler_state_v5_wang_rescue_polish",
            "qubo_scheduler_state_v5_wang_rescue_successor_cache_polish",
            "qubo_scheduler_state_v5_wang_rescue_fast_polish",
            "qubo_scheduler_state_v5_wang_rescue_light_polish",
            "qubo_scheduler_state_v5_wang_rescue_tail_risk_polish",
            "qubo_scheduler_state_v5_wang2017_task_polish",
            "qubo_scheduler_state_v5_wang2017_task_guarded_polish",
            "qubo_scheduler_state_v5_wang2017_task_guarded_portfolio",
            "qubo_scheduler_state_v5_wang_hybrid",
            "qubo_scheduler_state_v7_wang_qubo_unified",
        ]
    if name == "wang_research_all":
        return [
            "qubo_scheduler_state_v5_wang_rescue_polish",
            "qubo_scheduler_state_v5_wang_rescue_successor_cache_polish",
            "qubo_scheduler_state_v5_wang_rescue_fast_polish",
            "qubo_scheduler_state_v5_wang_rescue_light_polish",
            "qubo_scheduler_state_v5_wang_rescue_tail_lookahead_polish",
            "qubo_scheduler_state_v5_wang_rescue_tail_risk_polish",
            "qubo_scheduler_state_v5_wang_rescue_scarcity_inner_polish",
            "qubo_scheduler_state_v5_feasibility_escape_polish",
            "qubo_scheduler_state_v5_frg_dlts_polish",
            "qubo_scheduler_state_v5_frg_linear_polish",
            "qubo_scheduler_state_v5_frg_safe_tiebreak_polish",
            "qubo_scheduler_state_v5_frg_diverse_qubo_polish",
            "qubo_scheduler_state_v5_frg_candidate_pool_polish",
            "qubo_scheduler_state_v5_frg_scarcity_gate_polish",
            "qubo_scheduler_state_v5_frg_all_ablation_polish",
            "qubo_scheduler_state_v5_bs_frg_qubo_polish",
            "qubo_scheduler_state_v5_frg_seed_qubo_polish",
            "qubo_scheduler_state_v5_frg_seed_ledger_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_origin_aware_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_rollout_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_rollout_score_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_witness_qubo_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_witness_qubo_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_completion_macro_qubo_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_hybrid_macro_qubo_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_adaptive_macro_qubo_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_soft_supply_qubo_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_det_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_deterministic_frg_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_quality_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_quality_global_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_quality_large_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_deep_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_stable_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_budget_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_bounded_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_softcredit_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_terminal_witness_qubo_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_micro_trace_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_receiver_safety_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_micro_only_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_micro_rollout_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_micro_tiebreak_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_rollout_graft_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_terminal_bridge_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_tail_lookahead_lite_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_tail_lookahead_wide_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_terminal_escape_portfolio",
            "qubo_scheduler_state_v5_qubo_diversity_union_terminal_escape_lazy_portfolio",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_deep_chain_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_compound_sources_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_compound_certified_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_evaluated_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_adaptive_polish",
            "qubo_scheduler_state_v5_pareto_union_polish",
            "qubo_scheduler_state_v5_frg_compound_polish",
            "qubo_scheduler_state_v5_frg_compound_loose_polish",
            "qubo_scheduler_state_v5_frg_compound_guarded_polish",
            "qubo_scheduler_state_v5_wang2017_task_polish",
            "qubo_scheduler_state_v5_wang2017_task_guarded_polish",
            "qubo_scheduler_state_v5_wang2017_task_guarded_portfolio",
            "qubo_scheduler_state_v5_wang_rescue_dynamic_polish",
            "qubo_scheduler_state_v5_wang_rescue_risk_polish",
            "qubo_scheduler_state_v5_wang_rescue_scarcity_polish",
            "qubo_scheduler_state_v5_wang_rescue_hardness_routed_polish",
            "qubo_scheduler_state_v5_wang_rescue_selective_window_polish",
        ]
    return [name]


def _case_id_selected(case_id: str, case_ids_csv: str) -> bool:
    requested = {part.strip() for part in str(case_ids_csv or "").split(",") if part.strip()}
    if not requested:
        return True
    numeric = case_id.rsplit("_", 1)[-1]
    return case_id in requested or numeric in requested


def _args_for_variant(args: argparse.Namespace, variant: str, instance: Instance | None = None) -> argparse.Namespace:
    overrides: dict[str, object] = {
        "alpha": args.alpha,
        "business_cap_enabled": getattr(args, "business_cap_enabled", True),
        "max_moves": args.max_moves,
        "solver_profile": args.solver_profile,
        "target_final_n": 0,
        "time_limit": args.time_limit,
    }
    for field in ("beam_width", "core_width", "parent_core_cap"):
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value
    value = getattr(args, "qubo_start_layer", None)
    if value is not None:
        overrides["qubo_start_layer"] = value
    value = getattr(args, "deterministic_qubo_sampler", None)
    if value is not None:
        overrides["deterministic_qubo_sampler"] = value
    for field in (
        "survivor_parent_cap_enabled",
        "survivor_parent_cap_trigger_ratio",
        "survivor_parent_cap_release_ratio",
        "survivor_parent_cap_value",
        "survivor_parent_cap_window",
        "survivor_parent_cap_min_n",
    ):
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value
    if getattr(args, "beam_survivor_telemetry_output", None) is not None:
        overrides["beam_survivor_telemetry_output"] = args.beam_survivor_telemetry_output
    if getattr(args, "frontier_trace_output", None) is not None:
        overrides["frontier_trace_output"] = args.frontier_trace_output
        overrides["frontier_trace_top_k"] = int(getattr(args, "frontier_trace_top_k", 64) or 64)
    if getattr(args, "terminal_trace_output", None) is not None:
        overrides["terminal_trace_output"] = args.terminal_trace_output
    value = getattr(args, "post_polish_qubo", None)
    if value is not None:
        overrides["post_polish_qubo"] = value
    stream_candidate_limit = int(getattr(args, "stream_candidate_limit", 0) or 0)
    if stream_candidate_limit > 0:
        overrides.update(
            {
                "stream_candidate_limit": stream_candidate_limit,
                "stream_candidate_trim_factor": float(getattr(args, "stream_candidate_trim_factor", 2.0)),
            }
        )
    rank_bucket_topk = int(getattr(args, "rank_bucket_topk", 0) or 0)
    stream_bucket_topk = int(getattr(args, "stream_bucket_topk", 0) or 0)
    if stream_bucket_topk > 0:
        overrides.update(
            {
                "stream_bucket_topk": stream_bucket_topk,
                "rank_bucket_key": getattr(args, "rank_bucket_key", "n_depth_parent"),
                "rank_bucket_n_band_size": getattr(args, "rank_bucket_n_band_size", 2),
            }
        )
    if rank_bucket_topk > 0:
        overrides.update(
            {
                "rank_bucket_topk": rank_bucket_topk,
                "rank_bucket_key": getattr(args, "rank_bucket_key", "n_depth_parent"),
                "rank_bucket_n_band_size": getattr(args, "rank_bucket_n_band_size", 2),
                "rank_bucket_topk_high_n": getattr(args, "rank_bucket_topk_high_n", 0),
                "rank_bucket_topk_min_candidates": getattr(args, "rank_bucket_topk_min_candidates", 0),
            }
        )
    if getattr(args, "scheduler_adaptive_cv1_only", False):
        overrides["scheduler_adaptive_cv1_only"] = True
    for field in (
        "scheduler_adaptive_max_duplication_ratio",
        "scheduler_adaptive_min_best_n_ratio",
        "scheduler_adaptive_max_candidate_pressure",
        "scheduler_adaptive_max_parent_top_share",
        "scheduler_adaptive_plateau_window",
        "scheduler_adaptive_cap_slack",
        "scheduler_state_contract_pressure",
        "scheduler_state_max_contract_pressure",
        "scheduler_state_min_best_n_ratio",
        "scheduler_state_progress_window",
        "scheduler_state_progress_delta",
        "scheduler_state_failed_contract_cooldown",
        "scheduler_state_post_rescue_plateau_window",
        "scheduler_state_post_rescue_plateau_delta",
        "scheduler_state_post_rescue_min_ratio",
        "scheduler_state_post_rescue_action",
        "scheduler_state_initial_plateau_window",
        "scheduler_state_initial_plateau_min_ratio",
        "scheduler_state_initial_plateau_action",
        "scheduler_state_initial_plateau_lock",
        "scheduler_state_initial_plateau_min_duplication",
        "scheduler_state_initial_plateau_unique_min_n",
        "scheduler_state_protect_parent_top_share",
        "scheduler_state_protect_n_band",
        "scheduler_state_top_window",
        "scheduler_state_rescue_plateau_window",
        "scheduler_state_rescue_plateau_delta",
        "scheduler_state_rescue_cap_slack",
        "scarcity_inner_rescue_enabled",
        "scarcity_inner_rescue_risk_threshold",
        "scarcity_inner_rescue_max_l0",
        "scarcity_inner_rescue_min_one_safe_ratio",
        "scarcity_inner_rescue_min_n",
        "scarcity_inner_rescue_min_candidates",
        "tail_lookahead_enabled",
        "tail_lookahead_core_path_only",
        "tail_lookahead_max_stack_count",
        "tail_lookahead_focus_n",
        "tail_lookahead_depth",
        "tail_lookahead_max_seed_candidates",
        "tail_lookahead_supplement_width",
        "tail_lookahead_max_expanded",
        "terminal_bridge_enabled",
        "terminal_bridge_focus_n",
        "terminal_bridge_depth",
        "terminal_bridge_max_seed_candidates",
        "terminal_bridge_max_expanded",
        "tail_risk_reserve_enabled",
        "tail_risk_focus_n",
        "tail_risk_n_window",
        "tail_risk_reserve_width",
        "tail_risk_max_scan",
        "tail_risk_min_candidates",
        "tail_risk_min_depth_slack",
        "tail_risk_max_depth_slack",
        "tail_risk_max_stack_count",
        "rescue_qubo_supplement_width",
        "rescue_max_qubo_vars",
        "qubo_rank_weight",
        "qubo_n_weight",
        "qubo_depth_weight",
        "qubo_parent_weight",
        "qubo_next_n_weight",
        "qubo_diversity_weight",
        "qubo_penalty_a",
    ):
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value
    for field in (
        "adaptive_controller",
        "adaptive_expand_frontier",
        "adaptive_contract_pressure",
        "adaptive_mid_pressure",
        "adaptive_min_best_n_ratio",
        "adaptive_max_duplication_ratio",
        "adaptive_parent_top_share",
        "adaptive_plateau_window",
        "adaptive_plateau_delta",
        "adaptive_cap_slack",
        "adaptive_large_stack_min_stacks",
        "adaptive_large_stack_expand_frontier",
        "adaptive_large_stack_plateau_window",
        "adaptive_large_stack_cap_slack",
    ):
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value
    for field in (
        "wang_incumbent_enabled",
        "wang_incumbent_beam_width",
        "wang_incumbent_candidate_limit",
        "wang_incumbent_max_depth",
        "wang_incumbent_time_limit",
        "wang_cap_enabled",
        "wang_rollout_enabled",
        "wang_rollout_strategy",
        "wang_rollout_inject_candidates",
        "wang_rollout_width",
        "wang_rollout_focus_n",
        "wang_rollout_plateau_window",
        "wang_rollout_plateau_delta",
        "wang_rollout_max_steps",
        "wang_rollout_candidate_limit",
        "wang_pair_qubo_enabled",
        "wang_pair_qubo_start_layer",
        "wang_pair_qubo_supplement_width",
        "wang_pair_qubo_focus_n",
        "wang_pair_qubo_n_window",
        "wang_pair_qubo_max_vars",
        "wang_pair_qubo_select",
        "wang_pair_qubo_same_parent_penalty",
        "wang_probe_supplement_width",
        "unified_wang_qubo_enabled",
        "unified_wang_qubo_supplement_width",
        "unified_wang_qubo_focus_n",
        "unified_wang_qubo_n_window",
        "unified_wang_qubo_max_vars",
        "unified_wang_qubo_select",
        "unified_wang_n_weight",
        "unified_wang_depth_weight",
        "unified_wang_fixed_prefix_weight",
        "unified_wang_fix_pair_weight",
        "unified_wang_target_gap_weight",
        "unified_wang_same_parent_penalty",
        "unified_wang_pair_same_parent_penalty",
        "unified_wang_signature_penalty",
        "unified_wang_complement_bonus",
        "unified_wang_rollout_bonus",
        "feasibility_escape_enabled",
        "feasibility_escape_supplement_width",
        "feasibility_escape_focus_n",
        "feasibility_escape_n_window",
        "feasibility_escape_dead_end_threshold",
        "feasibility_escape_min_temporary_slack",
        "candidate_ledger_score_enabled",
        "candidate_ledger_wang_fixed_weight",
        "candidate_ledger_wang_pair_weight",
        "candidate_ledger_frg_fill_weight",
        "candidate_ledger_frg_reduce_weight",
        "candidate_ledger_frg_unblocked_weight",
        "candidate_ledger_frg_assignment_weight",
        "candidate_ledger_rollout_bonus",
        "candidate_ledger_rank_window",
        "union_diversity_qubo_enabled",
        "union_diversity_qubo_supplement_width",
        "union_diversity_qubo_focus_n",
        "union_diversity_qubo_n_window",
        "union_diversity_qubo_min_candidates",
        "union_diversity_qubo_max_vars",
        "union_diversity_qubo_select",
        "union_diversity_qubo_rank_weight",
        "union_diversity_qubo_wang_fixed_weight",
        "union_diversity_qubo_wang_pair_weight",
        "union_diversity_qubo_frg_weight",
        "union_diversity_qubo_rollout_bonus",
        "union_diversity_qubo_same_parent_penalty",
        "union_diversity_qubo_same_source_penalty",
        "union_diversity_qubo_signature_penalty",
        "union_diversity_qubo_reduce_signature_penalty",
        "union_diversity_qubo_n_band_penalty",
        "union_diversity_qubo_n_band_size",
        "union_diversity_qubo_origin_similarity_penalty",
        "union_diversity_qubo_same_origin_parent_n_band_penalty",
        "union_diversity_qubo_frg_witness_signature_penalty",
        "union_diversity_qubo_adaptive_macro_quota_enabled",
        "union_diversity_qubo_macro_quota_penalty",
        "union_diversity_qubo_macro_quota_base_share",
        "union_diversity_qubo_macro_quota_min_share",
        "union_diversity_qubo_macro_quota_max_share",
        "union_diversity_qubo_macro_quota_debt_weight",
        "union_diversity_qubo_macro_quota_scarcity_weight",
        "union_diversity_qubo_macro_quota_witness_weight",
        "union_diversity_qubo_macro_quality_bonus",
        "union_diversity_qubo_macro_edge_chain_cost_weight",
        "union_diversity_qubo_macro_future_credit_enabled",
        "union_diversity_qubo_completion_witness_credit_weight",
        "union_diversity_qubo_receiver_debt_credit_weight",
        "union_diversity_qubo_source_scarcity_credit_weight",
        "union_diversity_qubo_macro_path_credit_weight",
        "union_diversity_qubo_macro_future_credit_min_witness_quality",
        "union_diversity_qubo_receiver_debt_penalty_weight",
        "union_diversity_qubo_macro_future_credit_cap",
        "union_diversity_qubo_unified_macro_future_credit_scale",
        "union_diversity_qubo_unified_macro_pair_penalty",
        "union_diversity_qubo_macro_witness_move_penalty_weight",
        "union_diversity_qubo_macro_witness_move_penalty_scale",
        "union_diversity_qubo_macro_prefix_penalty_weight",
        "union_diversity_qubo_macro_witness_move_penalty_min_stack_count",
        "union_diversity_qubo_macro_witness_move_penalty_max_height",
        "union_diversity_qubo_macro_witness_move_penalty_max_stack_count",
        "post_polish_respect_time_limit",
        "post_polish_max_attempts",
        "post_polish_time_budget",
        "inner_jobs",
        "post_polish_window_repair_expanded",
        "post_polish_shortcut_replay_enabled",
        "post_polish_shortcut_min_delete",
        "post_polish_shortcut_max_delete",
        "post_polish_shortcut_window_budget",
        "post_polish_bridge_scan_enabled",
        "post_polish_bridge_scan_window_min_size",
        "post_polish_bridge_scan_window_max_size",
        "post_polish_bridge_scan_min_moves",
        "post_polish_bridge_scan_budget",
        "post_polish_bridge_scan_expanded",
        "debt_basin_enabled",
        "w_receiver_debt",
        "w_escape_potential",
        "w_debt_basin_chain_cost",
        "union_diversity_qubo_basin_similarity_penalty",
        "debt_basin_trace_window",
        "debt_basin_reduction_chain_enabled",
        "frg_rollout_respect_time_limit",
        "frg_rollout_max_generated",
        "union_rollout_probe_enabled",
        "union_rollout_probe_max_candidates",
        "union_rollout_probe_time_limit",
        "union_rollout_probe_max_steps",
        "union_rollout_probe_max_generated",
        "union_rollout_probe_feasible_bonus",
        "union_rollout_probe_final_n_weight",
        "union_rollout_probe_moves_weight",
        "union_rollout_probe_dead_end_penalty",
        "union_rollout_probe_evaluator_enabled",
        "union_rollout_probe_evaluator_weight",
        "union_rollout_probe_evaluator_final_n_penalty",
        "union_rollout_probe_evaluator_require_feasible",
        "union_rollout_probe_evaluator_min_stack_count",
        "union_rollout_terminal_graft",
        "union_rollout_terminal_graft_respect_cap",
        "micro_rollout_enabled",
        "micro_rollout_trace_only",
        "receiver_safety_enabled",
        "micro_rollout_tiebreak_only",
        "micro_rollout_tiebreak_epsilon",
        "micro_rollout_window",
        "micro_first_child_limit",
        "micro_second_child_limit",
        "micro_cache_enabled",
        "w_micro_delta_n",
        "w_micro_safe_continuation",
        "w_receiver_safe_delta",
        "w_last_buffer_consumed",
        "w_height_pressure",
        "frg_frontier_child_enabled",
        "frg_frontier_child_mode",
        "frg_frontier_child_min_stack_count",
        "frg_frontier_child_parent_width",
        "frg_frontier_child_limit",
        "frg_frontier_child_max_steps",
        "frg_frontier_child_append_only",
        "frg_frontier_child_macro_edge_enabled",
        "frg_frontier_child_terminal_witness_only_enabled",
        "frg_frontier_child_terminal_witness_respect_cap",
        "frg_frontier_child_certify_enabled",
        "frg_frontier_child_certify_max_steps",
        "frg_frontier_child_certify_time_limit",
        "frg_frontier_child_certify_max_generated",
        "frg_frontier_child_completion_prefix_enabled",
        "frg_frontier_child_completion_max_stack_count",
        "frg_frontier_child_completion_time_limit",
        "frg_frontier_child_completion_max_generated",
        "frg_frontier_child_adaptive_enabled",
        "frg_frontier_child_adaptive_parent_width",
        "frg_frontier_child_adaptive_limit",
        "frg_frontier_child_adaptive_max_steps",
        "frg_frontier_child_adaptive_plateau_window",
        "frg_frontier_child_adaptive_plateau_delta",
        "frg_frontier_child_adaptive_min_n",
        "frg_frontier_child_adaptive_small_stack_max",
        "frg_frontier_child_adaptive_min_container_count",
        "macro_chain_witness_enabled",
        "macro_chain_witness_respect_time_limit",
        "macro_chain_witness_parent_width",
        "macro_chain_witness_prefix_steps",
        "macro_chain_witness_max_steps",
        "macro_chain_witness_time_limit",
        "macro_chain_witness_max_generated",
        "macro_chain_witness_min_stack_count",
        "macro_chain_witness_min_height",
        "macro_chain_witness_min_n",
        "macro_chain_witness_min_candidates",
        "frg_seed_enabled",
        "frg_seed_time_limit",
        "frg_seed_min_max_steps",
        "frg_seed_max_steps_multiplier",
        "frg_seed_max_steps_cap",
        "frg_seed_respect_cap",
        "frg_seed_prune_to_incumbent",
        "frg_seed_allow_large_over_cap",
        "frg_seed_paper_pure",
        "frg_seed_beam_enabled",
        "frg_seed_beam_width",
        "frg_seed_candidate_limit_per_state",
        "frg_seed_beam_large_only",
        "frg_seed_beam_min_stack_count",
        "frg_seed_beam_min_height",
        "frg_seed_small_time_limit",
        "frg_seed_cache_input",
        "frg_candidate_supplement_enabled",
        "frg_candidate_supplement_width",
        "frg_candidate_focus_n",
        "frg_candidate_n_window",
        "frg_candidate_max_ranked",
        "frg_candidate_scarcity_gate_enabled",
        "frg_linear_score_enabled",
        "frg_linear_fill_weight",
        "frg_linear_reduce_weight",
        "frg_linear_unblocked_weight",
        "frg_linear_assignment_weight",
        "frg_linear_max_height_ordered_weight",
        "frg_linear_height_pressure_weight",
        "frg_rank_tiebreak_enabled",
        "frg_rank_tiebreak_weight",
        "frg_rank_tiebreak_focus_n",
        "frg_scarcity_gate_enabled",
        "frg_scarcity_gate_max_available_stacks",
        "frg_scarcity_gate_min_avg_height_pressure",
        "frg_scarcity_gate_max_reduction_receivers",
        "frg_scarcity_gate_min_unblocked_bad",
        "frg_scarcity_gate_min_candidates",
        "frg_scarcity_gate_cap_slack",
        "frg_layout_unblocked_bad_containers",
        "frg_layout_reduction_receiver_stack_count",
        "frg_layout_reduction_receiver_pressure",
        "frg_compound_escape_enabled",
        "frg_compound_escape_width",
        "frg_compound_escape_focus_n",
        "frg_compound_escape_plateau_window",
        "frg_compound_escape_plateau_delta",
        "frg_compound_escape_cap_slack",
        "frg_compound_escape_min_candidates",
        "frg_compound_guard_enabled",
        "frg_compound_guard_max_stack_count",
        "frg_compound_guard_max_initial_n",
        "frg_compound_guard_min_best_n",
        "frg_compound_guard_max_best_n",
        "frg_compound_guard_min_candidates",
        "frg_compound_guard_min_parent_count",
        "frg_compound_guard_max_parent_share",
        "dlts_value_supplement_enabled",
        "dlts_value_supplement_width",
        "bs_frg_qubo_enabled",
        "bs_frg_qubo_supplement_width",
        "bs_frg_qubo_focus_n",
        "bs_frg_qubo_n_window",
        "bs_frg_qubo_min_candidates",
        "bs_frg_qubo_max_vars",
        "bs_frg_qubo_rank_weight",
        "bs_frg_qubo_same_parent_penalty",
        "bs_frg_qubo_same_source_penalty",
        "bs_frg_qubo_signature_penalty",
        "bs_frg_qubo_reduce_signature_penalty",
        "bs_frg_qubo_n_band_penalty",
        "bs_frg_qubo_n_band_size",
        "post_polish_successor_cache",
        "wang2017_task_enabled",
        "wang2017_task_supplement_width",
        "wang2017_task_focus_n",
        "wang2017_task_n_window",
        "wang2017_task_plateau_window",
        "wang2017_task_plateau_delta",
    ):
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value
    if variant in {"qubo_adaptive", "qubo_adaptive_rescue", "qubo_adaptive_polish", "qubo_adaptive_controller"}:
        adaptive_high_frontier = args.adaptive_high_frontier
        adaptive_mid_frontier = args.adaptive_mid_frontier
        adaptive_low_frontier = args.adaptive_low_frontier
        if instance is not None and instance.stack_count <= 5:
            adaptive_high_frontier = adaptive_mid_frontier = adaptive_low_frontier = max(
                adaptive_high_frontier,
                adaptive_mid_frontier,
                adaptive_low_frontier,
            )
        overrides.update(
            {
                "adaptive_candidate_pool": args.adaptive_candidate_pool,
                "adaptive_high_frontier": adaptive_high_frontier,
                "adaptive_high_n": args.adaptive_high_n,
                "adaptive_low_frontier": adaptive_low_frontier,
                "adaptive_low_n": args.adaptive_low_n,
                "adaptive_mid_frontier": adaptive_mid_frontier,
                "selection_policy": "adaptive_qubo",
            }
        )
    if variant == "qubo_adaptive_controller":
        overrides.update({"solver_profile": "adaptive_controller_v1"})
    if variant == "qubo_adaptive_rescue":
        overrides.update({"solver_profile": "adaptive_rescue_v1"})
    if variant == "qubo_adaptive_polish":
        overrides.update({"solver_profile": "adaptive_polish_v1"})
    if variant == "qubo_light_rescue":
        overrides.update({"solver_profile": "light_rescue_qubo_v1"})
    if variant == "qubo_scheduler":
        overrides.update(
            {
                "solver_profile": "scheduler_v1",
                "scheduler_adaptive_min_stacks": getattr(args, "scheduler_adaptive_min_stacks", 7),
                "scheduler_adaptive_high_n": getattr(args, "scheduler_adaptive_high_n", 16),
            }
        )
    if variant == "qubo_scheduler_topk":
        overrides.update(
            {
                "solver_profile": "scheduler_topk_v1",
                "scheduler_adaptive_min_stacks": getattr(args, "scheduler_adaptive_min_stacks", 7),
                "scheduler_adaptive_high_n": getattr(args, "scheduler_adaptive_high_n", 16),
            }
        )
    if variant == "qubo_scheduler_state":
        overrides.update({"solver_profile": "scheduler_state_v2"})
    if variant == "qubo_scheduler_state_v3":
        overrides.update({"solver_profile": "scheduler_state_v3"})
    if variant == "qubo_scheduler_state_v4":
        overrides.update({"solver_profile": "scheduler_state_v4"})
    if variant == "qubo_scheduler_state_v5":
        overrides.update({"solver_profile": "scheduler_state_v5"})
    if variant == "qubo_scheduler_state_v5_wang_pair":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_pair_qubo"})
    if variant == "qubo_scheduler_state_v5_wang_features":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_features"})
    if variant == "qubo_scheduler_state_v5_wang_rescue":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_rescue"})
    if variant == "qubo_scheduler_state_v5_wang_rescue_suffix_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_rescue_suffix_polish"})
    if variant == "qubo_scheduler_state_v5_wang_rescue_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_rescue_polish"})
    if variant == "qubo_scheduler_state_v5_wang_rescue_successor_cache_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_rescue_successor_cache_polish"})
    if variant == "qubo_scheduler_state_v5_wang_rescue_fast_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_rescue_fast_polish"})
    if variant == "qubo_scheduler_state_v5_wang_rescue_light_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_rescue_light_polish"})
    if variant == "qubo_scheduler_state_v5_wang_rescue_tail_lookahead_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_rescue_tail_lookahead_polish"})
    if variant == "qubo_scheduler_state_v5_wang_rescue_tail_risk_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_rescue_tail_risk_polish"})
    if variant == "qubo_scheduler_state_v5_wang_rescue_scarcity_inner_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_rescue_scarcity_inner_polish"})
    if variant == "qubo_scheduler_state_v5_feasibility_escape_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_feasibility_escape_polish"})
    if variant == "qubo_scheduler_state_v5_frg_dlts_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_frg_dlts_polish"})
    if variant == "qubo_scheduler_state_v5_frg_linear_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_frg_linear_polish"})
    if variant == "qubo_scheduler_state_v5_frg_safe_tiebreak_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_frg_safe_tiebreak_polish"})
    if variant == "qubo_scheduler_state_v5_frg_diverse_qubo_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_frg_diverse_qubo_polish"})
    if variant == "qubo_scheduler_state_v5_frg_candidate_pool_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_frg_candidate_pool_polish"})
    if variant == "qubo_scheduler_state_v5_frg_scarcity_gate_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_frg_scarcity_gate_polish"})
    if variant == "qubo_scheduler_state_v5_frg_all_ablation_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_frg_all_ablation_polish"})
    if variant == "qubo_scheduler_state_v5_bs_frg_qubo_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_bs_frg_qubo_polish"})
    if variant == "qubo_scheduler_state_v5_frg_seed_qubo_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_frg_seed_qubo_polish"})
    if variant == "qubo_scheduler_state_v5_frg_seed_ledger_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_frg_seed_ledger_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_origin_aware_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_origin_aware_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_rollout_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_rollout_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_frg_rollout_score_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_frg_rollout_score_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_frg_witness_qubo_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_frg_witness_qubo_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_witness_qubo_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_frg_macro_witness_qubo_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_frg_completion_macro_qubo_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_frg_completion_macro_qubo_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_frg_hybrid_macro_qubo_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_frg_hybrid_macro_qubo_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_frg_adaptive_macro_qubo_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_frg_adaptive_macro_qubo_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_frg_soft_supply_qubo_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_frg_soft_supply_qubo_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_macro_credit_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_det_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_macro_credit_v2_det_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_deterministic_frg_polish":
        overrides.update(
            {"solver_profile": "scheduler_state_v5_qubo_diversity_union_macro_chain_deterministic_frg_polish"}
        )
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_quality_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_macro_chain_quality_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_quality_global_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_macro_chain_quality_global_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_quality_large_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_macro_chain_quality_large_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_deep_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_macro_chain_deep_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_stable_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_macro_chain_stable_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_budget_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_macro_chain_budget_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_bounded_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_macro_chain_bounded_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_unified_macro_supply_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_softcredit_polish":
        overrides.update(
            {"solver_profile": "scheduler_state_v5_qubo_diversity_union_unified_macro_supply_softcredit_polish"}
        )
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish":
        overrides.update(
            {"solver_profile": "scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish"}
        )
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_frg_terminal_witness_qubo_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_frg_terminal_witness_qubo_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_micro_trace_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_micro_trace_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_receiver_safety_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_receiver_safety_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_micro_only_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_micro_only_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_micro_rollout_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_micro_rollout_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_micro_tiebreak_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_micro_tiebreak_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_rollout_graft_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_rollout_graft_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_terminal_bridge_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_terminal_bridge_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_tail_lookahead_lite_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_tail_lookahead_lite_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_tail_lookahead_wide_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_tail_lookahead_wide_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_frg_deep_chain_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_frg_deep_chain_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_frg_compound_sources_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_frg_compound_sources_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_frg_compound_certified_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_frg_compound_certified_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_frg_evaluated_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_frg_evaluated_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_debt_basin_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_frg_frontier_polish"})
    if variant == "qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_adaptive_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_qubo_diversity_union_frg_frontier_adaptive_polish"})
    if variant == "qubo_scheduler_state_v5_pareto_union_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_pareto_union_polish"})
    if variant == "qubo_scheduler_state_v5_frg_compound_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_frg_compound_polish"})
    if variant == "qubo_scheduler_state_v5_frg_compound_loose_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_frg_compound_loose_polish"})
    if variant == "qubo_scheduler_state_v5_frg_compound_guarded_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_frg_compound_guarded_polish"})
    if variant == "qubo_scheduler_state_v5_wang2017_task_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_wang2017_task_polish"})
    if variant == "qubo_scheduler_state_v5_wang2017_task_guarded_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_wang2017_task_guarded_polish"})
    if variant == "qubo_scheduler_state_v5_wang_rescue_dynamic_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_rescue_dynamic_polish"})
    if variant == "qubo_scheduler_state_v5_wang_rescue_risk_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_rescue_risk_polish"})
    if variant == "qubo_scheduler_state_v5_wang_rescue_scarcity_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_rescue_scarcity_polish"})
    if variant == "qubo_scheduler_state_v5_wang_rescue_hardness_routed_polish":
        selected_profile = "scheduler_state_v5_wang_rescue_risk_polish"
        if instance is not None:
            metrics = _initial_scarcity_metrics(instance.stacks, instance.max_height)
            threshold = float(getattr(args, "hardness_route_light_threshold", 0.65) or 0.65)
            if _light_route_score(metrics, instance.container_count) >= threshold:
                selected_profile = "scheduler_state_v5_wang_rescue_light_polish"
        overrides.update({"solver_profile": selected_profile})
    if variant == "qubo_scheduler_state_v5_wang_rescue_selective_window_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_rescue_selective_window_polish"})
    if variant == "qubo_scheduler_state_v5_wang_hybrid":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_hybrid"})
    if variant == "qubo_scheduler_state_v7_wang_qubo_unified":
        overrides.update({"solver_profile": "scheduler_state_v7_wang_qubo_unified"})
    if variant == "qubo_scheduler_state_v7_wang_qubo_pool":
        overrides.update({"solver_profile": "scheduler_state_v7_wang_qubo_pool"})
    if variant == "qubo_scheduler_state_v5_window":
        overrides.update({"solver_profile": "scheduler_state_v5_window_polish"})
    if variant == "qubo_scheduler_state_v6":
        overrides.update({"solver_profile": "scheduler_state_v6"})
    if variant == "qubo_bucket":
        overrides.update(
            {
                "selection_policy": "qubo_bucket",
                "bucket_key": args.bucket_key,
                "bucket_n_band_size": args.bucket_n_band_size,
                "bucket_core_ratio": args.bucket_core_ratio,
                "bucket_qubo_share": args.bucket_qubo_share,
                "bucket_max_qubo_vars": args.bucket_max_qubo_vars,
                "bucket_qubo_select": args.bucket_qubo_select,
                "bucket_min_candidates": args.bucket_min_candidates,
            }
        )
    if variant == "classic_noqubo":
        overrides.update(
            {
                "beam_width": args.no_qubo_beam_width,
                "core_width": args.no_qubo_beam_width,
                "max_qubo_calls": 0,
                "post_polish_qubo": False,
                "selection_policy": "classic",
                "use_qubo": False,
            }
        )
    return make_compact_args(**overrides)


def _run_open_variant_row(
    case: OpenCpmpCase,
    instance: Instance,
    args: argparse.Namespace,
    variant: str,
) -> dict[str, object]:
    compact_args = _args_for_variant(args, variant, instance)
    result = run_compact_beam_instance(case.case_id, instance, compact_args)
    row = _open_row_dict_from_instance_result(result, instance, compact_args)
    business_ok, goal_ok, msg = _open_validate_instance_result(
        result,
        instance,
        business_alpha=args.alpha,
        business_cap_enabled=getattr(compact_args, "business_cap_enabled", True),
    )
    reference_moves = case.reference.optimal_moves
    moves = row.get("moves")
    row.update(
        {
            "dataset": "DLTS",
            "family": case.family,
            "variant": variant,
            "source_file": str(case.instance_path),
            "reference_file": str(case.reference_path) if case.reference_path else "",
            "reference_moves": reference_moves if reference_moves is not None else "",
            "move_gap": (int(moves) - reference_moves) if moves not in (None, "") and reference_moves is not None else "",
            "reference_tiers": case.reference.max_height if case.reference.max_height is not None else "",
            "business_alpha": args.alpha,
            "business_ok": business_ok,
            "goal_validation_ok": goal_ok,
            "validation_msg": msg,
        }
    )
    row["full_quality_penalty"] = _open_full_quality_penalty(row)
    return row


def _print_open_case_done(label: str, case_id: str, row: dict[str, object]) -> None:
    print(
        f"CASE_DONE {label} {case_id} final_n={row['final_n']} moves={row['moves']} "
        f"ref={row['reference_moves']} runtime={float(row['runtime_sec']):.3f}s "
        f"goal_ok={row['goal_validation_ok']} business_ok={row['business_ok']}",
        flush=True,
    )


def _print_open_task_start(run_idx: int, total_runs: int, task: OpenRunTask) -> None:
    print(f"CASE_START {run_idx}/{total_runs} {task.variant} {task.case.case_id}", flush=True)


def _print_open_task_result(result: OpenRunTaskResult) -> None:
    if result.skip_reason:
        print(f"CASE_SKIP {result.case_id} reason={result.skip_reason}", flush=True)
        return
    for row in result.rows:
        _print_open_case_done(result.variant, result.case_id, row)


def _run_open_portfolio_task(
    task: OpenRunTask,
    instance: Instance,
    args: argparse.Namespace,
    child_variants: tuple[str, ...],
    stop_on_first_valid: bool = False,
) -> OpenRunTaskResult:
    candidates: list[dict[str, object]] = []
    portfolio_runtime_sec = 0.0
    for child_variant in child_variants:
        child_row = _run_open_variant_row(task.case, instance, args, child_variant)
        portfolio_runtime_sec += _numeric(child_row, "runtime_sec", 0.0)
        candidates.append(child_row)
        if stop_on_first_valid and _open_row_valid(child_row):
            break
    selected = dict(_select_open_no_regret_row(candidates))
    selected["variant"] = task.variant
    selected["portfolio_profiles"] = ",".join(child_variants)
    selected["portfolio_selected_profile"] = selected.get("solver_profile", "")
    selected["portfolio_runtime_sec"] = f"{portfolio_runtime_sec:.6f}"
    selected["runtime_sec"] = f"{portfolio_runtime_sec:.6f}"
    selected["full_quality_penalty"] = _open_full_quality_penalty(selected)
    return OpenRunTaskResult(task.task_index, task.case.case_id, task.variant, [selected])


def _run_open_task(task: OpenRunTask, args_payload: dict[str, object]) -> OpenRunTaskResult:
    args = argparse.Namespace(**args_payload)
    try:
        instance = load_dlts_dat(task.case.instance_path, max_height=task.case.reference.max_height)
    except ValueError as exc:
        if args.skip_invalid:
            return OpenRunTaskResult(task.task_index, task.case.case_id, task.variant, [], str(exc))
        raise

    if task.variant == "qubo_scheduler_state_v5_wang2017_task_guarded_portfolio":
        return _run_open_portfolio_task(task, instance, args, _WANG2017_TASK_GUARDED_PORTFOLIO_VARIANTS)
    if task.variant == "qubo_scheduler_state_v5_qubo_diversity_union_terminal_escape_portfolio":
        return _run_open_portfolio_task(task, instance, args, _TERMINAL_ESCAPE_PORTFOLIO_VARIANTS)
    if task.variant == "qubo_scheduler_state_v5_qubo_diversity_union_terminal_escape_lazy_portfolio":
        return _run_open_portfolio_task(
            task,
            instance,
            args,
            _TERMINAL_ESCAPE_LAZY_PORTFOLIO_VARIANTS,
            stop_on_first_valid=True,
        )

    return OpenRunTaskResult(
        task.task_index,
        task.case.case_id,
        task.variant,
        [_run_open_variant_row(task.case, instance, args, task.variant)],
    )


def _args_payload(args: argparse.Namespace) -> dict[str, object]:
    return dict(vars(args))


def _flatten_task_results(results: list[OpenRunTaskResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in sorted(results, key=lambda item: item.task_index):
        rows.extend(result.rows)
    return rows


def _collect_serial_open_rows(tasks: list[OpenRunTask], args: argparse.Namespace) -> list[dict[str, object]]:
    payload = _args_payload(args)
    total_runs = len(tasks)
    results: list[OpenRunTaskResult] = []
    for run_idx, task in enumerate(tasks, start=1):
        _print_open_task_start(run_idx, total_runs, task)
        result = _run_open_task(task, payload)
        results.append(result)
        _print_open_task_result(result)
    return _flatten_task_results(results)


def _collect_parallel_open_rows(tasks: list[OpenRunTask], args: argparse.Namespace) -> list[dict[str, object]]:
    payload = _args_payload(args)
    results: list[OpenRunTaskResult] = []
    total_runs = len(tasks)
    with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
        futures = []
        for run_idx, task in enumerate(tasks, start=1):
            _print_open_task_start(run_idx, total_runs, task)
            futures.append(executor.submit(_run_open_task, task, payload))
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            _print_open_task_result(result)
    return _flatten_task_results(results)


def run_open_benchmark(args: argparse.Namespace) -> list[dict[str, object]]:
    source = ensure_dlts_source(args.source, auto_clone=args.auto_clone, refresh=args.refresh_source)
    families = None if args.families == ["all"] else args.families
    cases = discover_dlts_cases(
        source,
        families=families,
        limit_per_family=args.limit_per_family,
        max_total=args.max_total,
    )
    cases = [case for case in cases if _case_id_selected(case.case_id, args.case_ids)]
    variants = _variant_names(args.variant)
    tasks = _build_open_run_tasks(cases, variants)
    args.jobs = _resolve_open_jobs(getattr(args, "jobs", 1))
    if args.jobs <= 1:
        return _collect_serial_open_rows(tasks, args)
    return _collect_parallel_open_rows(tasks, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "external" / "DLTS")
    parser.add_argument("--auto-clone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refresh-source", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--families", nargs="+", default=["5x5_cv1_test"])
    parser.add_argument("--limit-per-family", type=int, default=20)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--max-total", type=int)
    parser.add_argument(
        "--jobs",
        type=_parse_open_jobs,
        default=1,
        help=(
            "Number of worker processes for open benchmark case/variant runs. "
            "Use 1 for serial execution, 0 or auto for the recommended safe cap."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=[
            "qubo",
            "qubo_adaptive",
            "qubo_adaptive_controller",
            "qubo_adaptive_rescue",
            "qubo_adaptive_polish",
            "qubo_light_rescue",
            "qubo_scheduler",
            "qubo_scheduler_topk",
            "qubo_scheduler_state",
            "qubo_scheduler_state_v3",
            "qubo_scheduler_state_v4",
            "qubo_scheduler_state_v5",
            "qubo_scheduler_state_v5_wang_pair",
            "qubo_scheduler_state_v5_wang_features",
            "qubo_scheduler_state_v5_wang_rescue",
            "qubo_scheduler_state_v5_wang_rescue_suffix_polish",
            "qubo_scheduler_state_v5_wang_rescue_polish",
            "qubo_scheduler_state_v5_wang_rescue_successor_cache_polish",
            "qubo_scheduler_state_v5_wang_rescue_fast_polish",
            "qubo_scheduler_state_v5_wang_rescue_light_polish",
            "qubo_scheduler_state_v5_wang_rescue_tail_lookahead_polish",
            "qubo_scheduler_state_v5_wang_rescue_tail_risk_polish",
            "qubo_scheduler_state_v5_wang_rescue_scarcity_inner_polish",
            "qubo_scheduler_state_v5_feasibility_escape_polish",
            "qubo_scheduler_state_v5_frg_dlts_polish",
            "qubo_scheduler_state_v5_frg_linear_polish",
            "qubo_scheduler_state_v5_frg_safe_tiebreak_polish",
            "qubo_scheduler_state_v5_frg_diverse_qubo_polish",
            "qubo_scheduler_state_v5_frg_candidate_pool_polish",
            "qubo_scheduler_state_v5_frg_scarcity_gate_polish",
            "qubo_scheduler_state_v5_frg_all_ablation_polish",
            "qubo_scheduler_state_v5_bs_frg_qubo_polish",
            "qubo_scheduler_state_v5_frg_seed_qubo_polish",
            "qubo_scheduler_state_v5_frg_seed_ledger_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_origin_aware_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_rollout_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_rollout_score_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_witness_qubo_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_witness_qubo_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_completion_macro_qubo_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_hybrid_macro_qubo_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_adaptive_macro_qubo_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_soft_supply_qubo_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_det_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_deterministic_frg_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_quality_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_quality_global_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_quality_large_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_deep_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_stable_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_budget_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_bounded_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_softcredit_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_terminal_witness_qubo_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_micro_trace_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_receiver_safety_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_micro_only_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_micro_rollout_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_micro_tiebreak_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_rollout_graft_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_terminal_bridge_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_tail_lookahead_lite_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_tail_lookahead_wide_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_terminal_escape_portfolio",
            "qubo_scheduler_state_v5_qubo_diversity_union_terminal_escape_lazy_portfolio",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_deep_chain_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_compound_sources_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_compound_certified_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_evaluated_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_polish",
            "qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_adaptive_polish",
            "qubo_scheduler_state_v5_pareto_union_polish",
            "qubo_scheduler_state_v5_frg_compound_polish",
            "qubo_scheduler_state_v5_frg_compound_loose_polish",
            "qubo_scheduler_state_v5_frg_compound_guarded_polish",
            "qubo_scheduler_state_v5_wang2017_task_polish",
            "qubo_scheduler_state_v5_wang2017_task_guarded_polish",
            "qubo_scheduler_state_v5_wang2017_task_guarded_portfolio",
            "qubo_scheduler_state_v5_wang_rescue_dynamic_polish",
            "qubo_scheduler_state_v5_wang_rescue_risk_polish",
            "qubo_scheduler_state_v5_wang_rescue_scarcity_polish",
            "qubo_scheduler_state_v5_wang_rescue_hardness_routed_polish",
            "qubo_scheduler_state_v5_wang_rescue_selective_window_polish",
            "qubo_scheduler_state_v5_wang_hybrid",
            "qubo_scheduler_state_v7_wang_qubo_unified",
            "qubo_scheduler_state_v7_wang_qubo_pool",
            "qubo_scheduler_state_v5_window",
            "qubo_scheduler_state_v6",
            "qubo_bucket",
            "classic_noqubo",
            "both",
            "all",
            "wang_all",
            "wang_research_all",
        ],
        default="qubo_scheduler_state_v5_qubo_diversity_union_polish",
    )
    parser.add_argument("--skip-invalid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--solver-profile", choices=sorted(_SOLVER_PROFILES.keys()), default="speed_v2")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument(
        "--business-cap",
        dest="business_cap_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the historical entropy business cap; pass --no-business-cap for uncapped benchmark alignment.",
    )
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--beam-width", type=int)
    parser.add_argument("--core-width", type=int)
    parser.add_argument("--parent-core-cap", type=int)
    parser.add_argument("--hardness-route-light-threshold", type=float, default=0.65)
    parser.add_argument("--post-polish-qubo", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--post-polish-shortcut-replay-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--post-polish-shortcut-min-delete", type=int)
    parser.add_argument("--post-polish-shortcut-max-delete", type=int)
    parser.add_argument("--post-polish-shortcut-window-budget", type=int)
    parser.add_argument("--post-polish-bridge-scan-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--post-polish-bridge-scan-window-min-size", type=int)
    parser.add_argument("--post-polish-bridge-scan-window-max-size", type=int)
    parser.add_argument("--post-polish-bridge-scan-min-moves", type=int)
    parser.add_argument("--post-polish-bridge-scan-budget", type=int)
    parser.add_argument("--post-polish-bridge-scan-expanded", type=int)
    parser.add_argument("--survivor-parent-cap-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--survivor-parent-cap-trigger-ratio", type=float)
    parser.add_argument("--survivor-parent-cap-release-ratio", type=float)
    parser.add_argument("--survivor-parent-cap-value", type=int)
    parser.add_argument("--survivor-parent-cap-window", type=int)
    parser.add_argument("--survivor-parent-cap-min-n", type=int)
    parser.add_argument("--no-qubo-beam-width", type=int, default=1200)
    parser.add_argument("--adaptive-high-n", type=int, default=16)
    parser.add_argument("--adaptive-low-n", type=int, default=8)
    parser.add_argument("--adaptive-high-frontier", type=int, default=800)
    parser.add_argument("--adaptive-mid-frontier", type=int, default=1000)
    parser.add_argument("--adaptive-low-frontier", type=int, default=1200)
    parser.add_argument("--adaptive-candidate-pool", type=int, default=0)
    parser.add_argument("--adaptive-controller", choices=["tiered", "signal_v1"])
    parser.add_argument("--adaptive-expand-frontier", type=int)
    parser.add_argument("--adaptive-contract-pressure", type=float)
    parser.add_argument("--adaptive-mid-pressure", type=float)
    parser.add_argument("--adaptive-min-best-n-ratio", type=float)
    parser.add_argument("--adaptive-max-duplication-ratio", type=float)
    parser.add_argument("--adaptive-parent-top-share", type=float)
    parser.add_argument("--adaptive-plateau-window", type=int)
    parser.add_argument("--adaptive-plateau-delta", type=int)
    parser.add_argument("--adaptive-cap-slack", type=int)
    parser.add_argument("--adaptive-large-stack-min-stacks", type=int)
    parser.add_argument("--adaptive-large-stack-expand-frontier", type=int)
    parser.add_argument("--adaptive-large-stack-plateau-window", type=int)
    parser.add_argument("--adaptive-large-stack-cap-slack", type=int)
    parser.add_argument("--scheduler-adaptive-min-stacks", type=int, default=7)
    parser.add_argument("--scheduler-adaptive-high-n", type=int, default=16)
    parser.add_argument("--scheduler-adaptive-cv1-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scheduler-adaptive-max-duplication-ratio", type=float)
    parser.add_argument("--scheduler-adaptive-min-best-n-ratio", type=float)
    parser.add_argument("--scheduler-adaptive-max-candidate-pressure", type=float)
    parser.add_argument("--scheduler-adaptive-max-parent-top-share", type=float)
    parser.add_argument("--scheduler-adaptive-plateau-window", type=int)
    parser.add_argument("--scheduler-adaptive-cap-slack", type=int)
    parser.add_argument("--scheduler-state-contract-pressure", type=float)
    parser.add_argument("--scheduler-state-max-contract-pressure", type=float)
    parser.add_argument("--scheduler-state-min-best-n-ratio", type=float)
    parser.add_argument("--scheduler-state-progress-window", type=int)
    parser.add_argument("--scheduler-state-progress-delta", type=int)
    parser.add_argument("--scheduler-state-failed-contract-cooldown", type=int)
    parser.add_argument("--scheduler-state-post-rescue-plateau-window", type=int)
    parser.add_argument("--scheduler-state-post-rescue-plateau-delta", type=int)
    parser.add_argument("--scheduler-state-post-rescue-min-ratio", type=float)
    parser.add_argument("--scheduler-state-post-rescue-action", choices=["protect", "blend", "rescue"])
    parser.add_argument("--scheduler-state-initial-plateau-window", type=int)
    parser.add_argument("--scheduler-state-initial-plateau-min-ratio", type=float)
    parser.add_argument("--scheduler-state-initial-plateau-action", choices=["protect", "blend", "rescue"])
    parser.add_argument("--scheduler-state-initial-plateau-lock", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--scheduler-state-initial-plateau-min-duplication", type=float)
    parser.add_argument("--scheduler-state-initial-plateau-unique-min-n", type=int)
    parser.add_argument("--scheduler-state-protect-parent-top-share", type=float)
    parser.add_argument("--scheduler-state-protect-n-band", type=int)
    parser.add_argument("--scheduler-state-top-window", type=int)
    parser.add_argument("--scheduler-state-rescue-plateau-window", type=int)
    parser.add_argument("--scheduler-state-rescue-plateau-delta", type=int)
    parser.add_argument("--scheduler-state-rescue-cap-slack", type=int)
    parser.add_argument("--scarcity-inner-rescue-enabled", action=argparse.BooleanOptionalAction)
    parser.add_argument("--scarcity-inner-rescue-risk-threshold", type=float)
    parser.add_argument("--scarcity-inner-rescue-max-l0", type=int)
    parser.add_argument("--scarcity-inner-rescue-min-one-safe-ratio", type=float)
    parser.add_argument("--scarcity-inner-rescue-min-n", type=int)
    parser.add_argument("--scarcity-inner-rescue-min-candidates", type=int)
    parser.add_argument("--tail-lookahead-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tail-lookahead-core-path-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tail-lookahead-max-stack-count", type=int)
    parser.add_argument("--tail-lookahead-focus-n", type=int)
    parser.add_argument("--tail-lookahead-depth", type=int)
    parser.add_argument("--tail-lookahead-max-seed-candidates", type=int)
    parser.add_argument("--tail-lookahead-supplement-width", type=int)
    parser.add_argument("--tail-lookahead-max-expanded", type=int)
    parser.add_argument("--terminal-bridge-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--terminal-bridge-focus-n", type=int)
    parser.add_argument("--terminal-bridge-depth", type=int)
    parser.add_argument("--terminal-bridge-max-seed-candidates", type=int)
    parser.add_argument("--terminal-bridge-max-expanded", type=int)
    parser.add_argument("--tail-risk-reserve-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tail-risk-focus-n", type=int)
    parser.add_argument("--tail-risk-n-window", type=int)
    parser.add_argument("--tail-risk-reserve-width", type=int)
    parser.add_argument("--tail-risk-max-scan", type=int)
    parser.add_argument("--tail-risk-min-candidates", type=int)
    parser.add_argument("--tail-risk-min-depth-slack", type=int)
    parser.add_argument("--tail-risk-max-depth-slack", type=int)
    parser.add_argument("--tail-risk-max-stack-count", type=int)
    parser.add_argument("--rescue-qubo-supplement-width", type=int)
    parser.add_argument("--rescue-max-qubo-vars", type=int)
    parser.add_argument("--qubo-rank-weight", type=float)
    parser.add_argument("--qubo-n-weight", type=float)
    parser.add_argument("--qubo-depth-weight", type=float)
    parser.add_argument("--qubo-parent-weight", type=float)
    parser.add_argument("--qubo-next-n-weight", type=float)
    parser.add_argument("--qubo-diversity-weight", type=float)
    parser.add_argument("--qubo-penalty-a", type=float)
    parser.add_argument("--rank-bucket-topk", type=int, default=0)
    parser.add_argument(
        "--rank-bucket-key",
        choices=["n", "n_depth_parent", "n_depth", "n_band_parent", "n_band", "parent"],
        default="n_depth_parent",
    )
    parser.add_argument("--rank-bucket-n-band-size", type=int, default=2)
    parser.add_argument("--rank-bucket-topk-high-n", type=int, default=0)
    parser.add_argument("--rank-bucket-topk-min-candidates", type=int, default=0)
    parser.add_argument("--stream-candidate-limit", type=int, default=0)
    parser.add_argument("--stream-candidate-trim-factor", type=float, default=2.0)
    parser.add_argument("--stream-bucket-topk", type=int, default=0)
    parser.add_argument("--bucket-key", choices=["n_depth", "f_n", "n_band"], default="n_depth")
    parser.add_argument("--bucket-n-band-size", type=int, default=2)
    parser.add_argument("--bucket-core-ratio", type=float, default=0.50)
    parser.add_argument("--bucket-qubo-share", type=float, default=0.35)
    parser.add_argument("--bucket-max-qubo-vars", type=int, default=96)
    parser.add_argument("--bucket-qubo-select", type=int, default=24)
    parser.add_argument("--bucket-min-candidates", type=int, default=64)
    parser.add_argument("--wang-incumbent-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wang-incumbent-beam-width", type=int)
    parser.add_argument("--wang-incumbent-candidate-limit", type=int)
    parser.add_argument("--wang-incumbent-max-depth", type=int)
    parser.add_argument("--wang-incumbent-time-limit", type=float)
    parser.add_argument("--wang-cap-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wang-rollout-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wang-rollout-strategy", choices=["local", "baby_lookahead"])
    parser.add_argument("--wang-rollout-inject-candidates", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wang-rollout-width", type=int)
    parser.add_argument("--wang-rollout-focus-n", type=int)
    parser.add_argument("--wang-rollout-plateau-window", type=int)
    parser.add_argument("--wang-rollout-plateau-delta", type=int)
    parser.add_argument("--wang-rollout-max-steps", type=int)
    parser.add_argument("--wang-rollout-candidate-limit", type=int)
    parser.add_argument("--wang-pair-qubo-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wang-pair-qubo-start-layer", type=int)
    parser.add_argument("--wang-pair-qubo-supplement-width", type=int)
    parser.add_argument("--wang-pair-qubo-focus-n", type=int)
    parser.add_argument("--wang-pair-qubo-n-window", type=int)
    parser.add_argument("--wang-pair-qubo-max-vars", type=int)
    parser.add_argument("--wang-pair-qubo-select", type=int)
    parser.add_argument("--wang-pair-qubo-same-parent-penalty", type=float)
    parser.add_argument("--wang-probe-supplement-width", type=int)
    parser.add_argument("--deterministic-qubo-sampler", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--unified-wang-qubo-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--unified-wang-qubo-supplement-width", type=int)
    parser.add_argument("--unified-wang-qubo-focus-n", type=int)
    parser.add_argument("--unified-wang-qubo-n-window", type=int)
    parser.add_argument("--unified-wang-qubo-max-vars", type=int)
    parser.add_argument("--unified-wang-qubo-select", type=int)
    parser.add_argument("--unified-wang-n-weight", type=float)
    parser.add_argument("--unified-wang-depth-weight", type=float)
    parser.add_argument("--unified-wang-fixed-prefix-weight", type=float)
    parser.add_argument("--unified-wang-fix-pair-weight", type=float)
    parser.add_argument("--unified-wang-target-gap-weight", type=float)
    parser.add_argument("--unified-wang-same-parent-penalty", type=float)
    parser.add_argument("--unified-wang-pair-same-parent-penalty", type=float)
    parser.add_argument("--unified-wang-signature-penalty", type=float)
    parser.add_argument("--unified-wang-complement-bonus", type=float)
    parser.add_argument("--unified-wang-rollout-bonus", type=float)
    parser.add_argument("--feasibility-escape-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--feasibility-escape-supplement-width", type=int)
    parser.add_argument("--feasibility-escape-focus-n", type=int)
    parser.add_argument("--feasibility-escape-n-window", type=int)
    parser.add_argument("--feasibility-escape-dead-end-threshold", type=float)
    parser.add_argument("--feasibility-escape-min-temporary-slack", type=int)
    parser.add_argument("--candidate-ledger-score-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--candidate-ledger-wang-fixed-weight", type=float)
    parser.add_argument("--candidate-ledger-wang-pair-weight", type=float)
    parser.add_argument("--candidate-ledger-frg-fill-weight", type=float)
    parser.add_argument("--candidate-ledger-frg-reduce-weight", type=float)
    parser.add_argument("--candidate-ledger-frg-unblocked-weight", type=float)
    parser.add_argument("--candidate-ledger-frg-assignment-weight", type=float)
    parser.add_argument("--candidate-ledger-rollout-bonus", type=float)
    parser.add_argument("--candidate-ledger-rank-window", type=int)
    parser.add_argument("--union-diversity-qubo-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--union-diversity-qubo-supplement-width", type=int)
    parser.add_argument("--union-diversity-qubo-focus-n", type=int)
    parser.add_argument("--union-diversity-qubo-n-window", type=int)
    parser.add_argument("--union-diversity-qubo-min-candidates", type=int)
    parser.add_argument("--union-diversity-qubo-max-vars", type=int)
    parser.add_argument("--union-diversity-qubo-select", type=int)
    parser.add_argument("--union-diversity-qubo-rank-weight", type=float)
    parser.add_argument("--union-diversity-qubo-wang-fixed-weight", type=float)
    parser.add_argument("--union-diversity-qubo-wang-pair-weight", type=float)
    parser.add_argument("--union-diversity-qubo-frg-weight", type=float)
    parser.add_argument("--union-diversity-qubo-rollout-bonus", type=float)
    parser.add_argument("--union-diversity-qubo-same-parent-penalty", type=float)
    parser.add_argument("--union-diversity-qubo-same-source-penalty", type=float)
    parser.add_argument("--union-diversity-qubo-signature-penalty", type=float)
    parser.add_argument("--union-diversity-qubo-reduce-signature-penalty", type=float)
    parser.add_argument("--union-diversity-qubo-n-band-penalty", type=float)
    parser.add_argument("--union-diversity-qubo-n-band-size", type=int)
    parser.add_argument("--union-diversity-qubo-origin-similarity-penalty", type=float)
    parser.add_argument("--union-diversity-qubo-same-origin-parent-n-band-penalty", type=float)
    parser.add_argument("--union-diversity-qubo-frg-witness-signature-penalty", type=float)
    parser.add_argument(
        "--union-diversity-qubo-adaptive-macro-quota-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--union-diversity-qubo-macro-quota-penalty", type=float)
    parser.add_argument("--union-diversity-qubo-macro-quota-base-share", type=float)
    parser.add_argument("--union-diversity-qubo-macro-quota-min-share", type=float)
    parser.add_argument("--union-diversity-qubo-macro-quota-max-share", type=float)
    parser.add_argument("--union-diversity-qubo-macro-quota-debt-weight", type=float)
    parser.add_argument("--union-diversity-qubo-macro-quota-scarcity-weight", type=float)
    parser.add_argument("--union-diversity-qubo-macro-quota-witness-weight", type=float)
    parser.add_argument("--union-diversity-qubo-macro-quality-bonus", type=float)
    parser.add_argument("--union-diversity-qubo-macro-edge-chain-cost-weight", type=float)
    parser.add_argument("--union-diversity-qubo-macro-ancestry-penalty-weight", type=float)
    parser.add_argument("--union-diversity-qubo-macro-ancestry-penalty-scale", type=float)
    parser.add_argument(
        "--union-diversity-qubo-macro-future-credit-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--union-diversity-qubo-completion-witness-credit-weight", type=float)
    parser.add_argument("--union-diversity-qubo-receiver-debt-credit-weight", type=float)
    parser.add_argument("--union-diversity-qubo-source-scarcity-credit-weight", type=float)
    parser.add_argument("--union-diversity-qubo-macro-path-credit-weight", type=float)
    parser.add_argument("--union-diversity-qubo-macro-future-credit-min-witness-quality", type=float)
    parser.add_argument("--union-diversity-qubo-receiver-debt-penalty-weight", type=float)
    parser.add_argument("--union-diversity-qubo-macro-future-credit-cap", type=float)
    parser.add_argument("--union-diversity-qubo-unified-macro-future-credit-scale", type=float)
    parser.add_argument("--union-diversity-qubo-unified-macro-pair-penalty", type=float)
    parser.add_argument("--union-diversity-qubo-macro-witness-move-penalty-weight", type=float)
    parser.add_argument("--union-diversity-qubo-macro-witness-move-penalty-scale", type=float)
    parser.add_argument("--union-diversity-qubo-macro-prefix-penalty-weight", type=float)
    parser.add_argument("--union-diversity-qubo-macro-witness-move-penalty-min-stack-count", type=int)
    parser.add_argument("--union-diversity-qubo-macro-witness-move-penalty-max-height", type=int)
    parser.add_argument("--union-diversity-qubo-macro-witness-move-penalty-max-stack-count", type=int)
    parser.add_argument("--post-polish-respect-time-limit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--post-polish-max-attempts", type=int)
    parser.add_argument("--post-polish-time-budget", type=float)
    parser.add_argument("--inner-jobs", type=int)
    parser.add_argument("--post-polish-window-repair-expanded", type=int)
    parser.add_argument("--post-polish-successor-cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--debt-basin-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--w-receiver-debt", type=float, default=None)
    parser.add_argument("--w-escape-potential", type=float, default=None)
    parser.add_argument("--w-debt-basin-chain-cost", type=float, default=None)
    parser.add_argument("--union-diversity-qubo-basin-similarity-penalty", type=float, default=None)
    parser.add_argument("--debt-basin-trace-window", type=int, default=None)
    parser.add_argument("--debt-basin-reduction-chain-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-rollout-respect-time-limit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-rollout-max-generated", type=int)
    parser.add_argument("--union-rollout-probe-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--union-rollout-probe-max-candidates", type=int)
    parser.add_argument("--union-rollout-probe-time-limit", type=float)
    parser.add_argument("--union-rollout-probe-max-steps", type=int)
    parser.add_argument("--union-rollout-probe-max-generated", type=int)
    parser.add_argument("--union-rollout-probe-feasible-bonus", type=float)
    parser.add_argument("--union-rollout-probe-final-n-weight", type=float)
    parser.add_argument("--union-rollout-probe-moves-weight", type=float)
    parser.add_argument("--union-rollout-probe-dead-end-penalty", type=float)
    parser.add_argument("--union-rollout-probe-evaluator-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--union-rollout-probe-evaluator-weight", type=float)
    parser.add_argument("--union-rollout-probe-evaluator-final-n-penalty", type=float)
    parser.add_argument(
        "--union-rollout-probe-evaluator-require-feasible",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--union-rollout-terminal-graft", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--union-rollout-terminal-graft-respect-cap",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--micro-rollout-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--micro-rollout-trace-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--receiver-safety-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--micro-rollout-tiebreak-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--micro-rollout-tiebreak-epsilon", type=float)
    parser.add_argument("--micro-rollout-window", type=int)
    parser.add_argument("--micro-first-child-limit", type=int)
    parser.add_argument("--micro-second-child-limit", type=int)
    parser.add_argument("--micro-cache-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--w-micro-delta-n", type=float)
    parser.add_argument("--w-micro-safe-continuation", type=float)
    parser.add_argument("--w-receiver-safe-delta", type=float)
    parser.add_argument("--w-last-buffer-consumed", type=float)
    parser.add_argument("--w-height-pressure", type=float)
    parser.add_argument("--frg-frontier-child-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-frontier-child-mode", choices=["rollout", "compound_sources", "hybrid_macro"])
    parser.add_argument("--frg-frontier-child-min-stack-count", type=int)
    parser.add_argument("--frg-frontier-child-parent-width", type=int)
    parser.add_argument("--frg-frontier-child-limit", type=int)
    parser.add_argument("--frg-frontier-child-max-steps", type=int)
    parser.add_argument("--frg-frontier-child-append-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-frontier-child-macro-edge-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--frg-frontier-child-terminal-witness-only-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--frg-frontier-child-terminal-witness-respect-cap",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--frg-frontier-child-certify-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-frontier-child-certify-max-steps", type=int)
    parser.add_argument("--frg-frontier-child-certify-time-limit", type=float)
    parser.add_argument("--frg-frontier-child-certify-max-generated", type=int)
    parser.add_argument("--frg-frontier-child-completion-prefix-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-frontier-child-completion-max-stack-count", type=int)
    parser.add_argument("--frg-frontier-child-completion-time-limit", type=float)
    parser.add_argument("--frg-frontier-child-completion-max-generated", type=int)
    parser.add_argument("--frg-frontier-child-adaptive-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-frontier-child-adaptive-parent-width", type=int)
    parser.add_argument("--frg-frontier-child-adaptive-limit", type=int)
    parser.add_argument("--frg-frontier-child-adaptive-max-steps", type=int)
    parser.add_argument("--frg-frontier-child-adaptive-plateau-window", type=int)
    parser.add_argument("--frg-frontier-child-adaptive-plateau-delta", type=int)
    parser.add_argument("--frg-frontier-child-adaptive-min-n", type=int)
    parser.add_argument("--frg-frontier-child-adaptive-small-stack-max", type=int)
    parser.add_argument("--frg-frontier-child-adaptive-min-container-count", type=int)
    parser.add_argument("--macro-chain-witness-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--macro-chain-witness-respect-time-limit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--macro-chain-witness-parent-width", type=int)
    parser.add_argument("--macro-chain-witness-prefix-steps", type=int)
    parser.add_argument("--macro-chain-witness-max-steps", type=int)
    parser.add_argument("--macro-chain-witness-time-limit", type=float)
    parser.add_argument("--macro-chain-witness-max-generated", type=int)
    parser.add_argument("--macro-chain-witness-min-stack-count", type=int)
    parser.add_argument("--macro-chain-witness-min-height", type=int)
    parser.add_argument("--macro-chain-witness-min-n", type=int)
    parser.add_argument("--macro-chain-witness-min-candidates", type=int)
    parser.add_argument("--frg-seed-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-seed-time-limit", type=float)
    parser.add_argument("--frg-seed-min-max-steps", type=int)
    parser.add_argument("--frg-seed-max-steps-multiplier", type=int)
    parser.add_argument("--frg-seed-max-steps-cap", type=int)
    parser.add_argument("--frg-seed-respect-cap", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-seed-prune-to-incumbent", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-seed-allow-large-over-cap", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-seed-paper-pure", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-seed-beam-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-seed-beam-width", type=int)
    parser.add_argument("--frg-seed-candidate-limit-per-state", type=int)
    parser.add_argument("--frg-seed-beam-large-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-seed-beam-min-stack-count", type=int)
    parser.add_argument("--frg-seed-beam-min-height", type=int)
    parser.add_argument("--frg-seed-small-time-limit", type=float)
    parser.add_argument("--frg-candidate-supplement-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-candidate-supplement-width", type=int)
    parser.add_argument("--frg-candidate-focus-n", type=int)
    parser.add_argument("--frg-candidate-n-window", type=int)
    parser.add_argument("--frg-candidate-max-ranked", type=int)
    parser.add_argument("--frg-candidate-scarcity-gate-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-linear-score-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-linear-fill-weight", type=float)
    parser.add_argument("--frg-linear-reduce-weight", type=float)
    parser.add_argument("--frg-linear-unblocked-weight", type=float)
    parser.add_argument("--frg-linear-assignment-weight", type=float)
    parser.add_argument("--frg-linear-max-height-ordered-weight", type=float)
    parser.add_argument("--frg-linear-height-pressure-weight", type=float)
    parser.add_argument("--frg-rank-tiebreak-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-rank-tiebreak-weight", type=float)
    parser.add_argument("--frg-rank-tiebreak-focus-n", type=int)
    parser.add_argument("--frg-scarcity-gate-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-scarcity-gate-max-available-stacks", type=int)
    parser.add_argument("--frg-scarcity-gate-min-avg-height-pressure", type=float)
    parser.add_argument("--frg-scarcity-gate-max-reduction-receivers", type=int)
    parser.add_argument("--frg-scarcity-gate-min-unblocked-bad", type=int)
    parser.add_argument("--frg-scarcity-gate-min-candidates", type=int)
    parser.add_argument("--frg-scarcity-gate-cap-slack", type=int)
    parser.add_argument("--frg-compound-escape-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-compound-escape-width", type=int)
    parser.add_argument("--frg-compound-escape-focus-n", type=int)
    parser.add_argument("--frg-compound-escape-plateau-window", type=int)
    parser.add_argument("--frg-compound-escape-plateau-delta", type=int)
    parser.add_argument("--frg-compound-escape-cap-slack", type=int)
    parser.add_argument("--frg-compound-escape-min-candidates", type=int)
    parser.add_argument("--dlts-value-supplement-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dlts-value-supplement-width", type=int)
    parser.add_argument("--bs-frg-qubo-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bs-frg-qubo-supplement-width", type=int)
    parser.add_argument("--bs-frg-qubo-focus-n", type=int)
    parser.add_argument("--bs-frg-qubo-n-window", type=int)
    parser.add_argument("--bs-frg-qubo-min-candidates", type=int)
    parser.add_argument("--bs-frg-qubo-max-vars", type=int)
    parser.add_argument("--bs-frg-qubo-rank-weight", type=float)
    parser.add_argument("--bs-frg-qubo-same-parent-penalty", type=float)
    parser.add_argument("--bs-frg-qubo-same-source-penalty", type=float)
    parser.add_argument("--bs-frg-qubo-signature-penalty", type=float)
    parser.add_argument("--bs-frg-qubo-reduce-signature-penalty", type=float)
    parser.add_argument("--bs-frg-qubo-n-band-penalty", type=float)
    parser.add_argument("--bs-frg-qubo-n-band-size", type=int)
    parser.add_argument("--wang2017-task-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wang2017-task-supplement-width", type=int)
    parser.add_argument("--wang2017-task-focus-n", type=int)
    parser.add_argument("--wang2017-task-n-window", type=int)
    parser.add_argument("--wang2017-task-plateau-window", type=int)
    parser.add_argument("--wang2017-task-plateau-delta", type=int)
    parser.add_argument("--qubo-start-layer", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "open_cpmp_dlts_speed_v2.csv",
    )
    parser.add_argument("--beam-survivor-telemetry-output", type=Path)
    parser.add_argument("--frontier-trace-output", type=Path)
    parser.add_argument("--frontier-trace-top-k", type=int, default=64)
    parser.add_argument("--terminal-trace-output", type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.source = args.source if args.source.is_absolute() else ROOT / args.source
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.beam_survivor_telemetry_output is not None and not args.beam_survivor_telemetry_output.is_absolute():
        args.beam_survivor_telemetry_output = ROOT / args.beam_survivor_telemetry_output
    if args.frontier_trace_output is not None and not args.frontier_trace_output.is_absolute():
        args.frontier_trace_output = ROOT / args.frontier_trace_output
    if args.terminal_trace_output is not None and not args.terminal_trace_output.is_absolute():
        args.terminal_trace_output = ROOT / args.terminal_trace_output
    _reset_beam_survivor_output(args.beam_survivor_telemetry_output)
    _reset_frontier_trace_output(args.frontier_trace_output)
    _reset_terminal_trace_output(args.terminal_trace_output)
    rows = run_open_benchmark(args)
    _write_open_csv(rows, args.output)
    summary = args.output.with_name(args.output.stem + "_summary.md")
    _write_summary(rows, args.output, summary, args.source)
    print(f"WROTE {args.output}", flush=True)
    print(f"WROTE {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
