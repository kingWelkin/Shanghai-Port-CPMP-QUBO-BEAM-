"""Run the current compact QUBO beam on open CPMP benchmark instances.

The first supported source is the DLTS CPMP repository:
https://github.com/ahottung/DLTS
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
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
    _reset_beam_survivor_output,
    _row_dict_from_instance_result,
    _validate_instance_result,
    make_compact_args,
    run_compact_beam_instance,
)
from container_reshuffle.core import Instance, Move, ProblemVariant, validate_instance  # noqa: E402
from container_reshuffle.features import future_disorder_count_stacks  # noqa: E402

DLTS_URL = "https://github.com/ahottung/DLTS.git"
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


def _wang_pair_qubo_row_fields(result, args: argparse.Namespace) -> dict[str, object]:
    return {
        "wang_pair_qubo_enabled": getattr(args, "wang_pair_qubo_enabled", False),
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
        "frg_candidate_selected_total": getattr(result, "frg_candidate_selected_total", 0),
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
        final_n = n0
        moves_count = None
        max_g = n0
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
        "post_polish_window_candidates": result.post_polish_window_candidates,
        "post_polish_window_qubo_selected": result.post_polish_window_qubo_selected,
        "post_polish_window_attempts": result.post_polish_window_attempts,
        "post_polish_window_successes": result.post_polish_window_successes,
        "post_polish_window_moves_saved": result.post_polish_window_moves_saved,
        "post_polish_window_time_sec": result.post_polish_window_time_sec,
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
    }
    row.update(_wang_pair_qubo_row_fields(result, args))
    row.update(_unified_wang_row_fields(result, args))
    row.update(_feasibility_escape_row_fields(result, args))
    row.update(_frg_dlts_row_fields(result, args))
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
) -> tuple[bool, bool, str]:
    if not result.moves:
        return False, False, result.status
    goal_ok, goal_msg, _stacks, _n_trace, max_g = _simulate_priority_group_moves(instance, result.moves)
    cap = int(math.floor(min(float(business_alpha), 2.0) * future_disorder_count_stacks(instance.stacks) + 1e-9))
    business_ok = max_g <= cap
    return business_ok, goal_ok, goal_msg


def _open_validate_instance_result(result, instance: Instance, business_alpha: float) -> tuple[bool, bool, str]:
    if requires_priority_group_semantics(instance):
        return _validate_priority_group_result(result, instance, business_alpha)
    return _validate_instance_result(result, instance, business_alpha=business_alpha)


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
            "qubo_scheduler_state_v5_wang_rescue_light_polish",
            "qubo_scheduler_state_v5_wang2017_task_polish",
            "qubo_scheduler_state_v5_wang2017_task_guarded_polish",
            "qubo_scheduler_state_v5_wang2017_task_guarded_portfolio",
            "qubo_scheduler_state_v5_wang_hybrid",
            "qubo_scheduler_state_v7_wang_qubo_unified",
        ]
    if name == "wang_research_all":
        return [
            "qubo_scheduler_state_v5_wang_rescue_polish",
            "qubo_scheduler_state_v5_wang_rescue_light_polish",
            "qubo_scheduler_state_v5_feasibility_escape_polish",
            "qubo_scheduler_state_v5_frg_dlts_polish",
            "qubo_scheduler_state_v5_bs_frg_qubo_polish",
            "qubo_scheduler_state_v5_frg_compound_polish",
            "qubo_scheduler_state_v5_frg_compound_loose_polish",
            "qubo_scheduler_state_v5_frg_compound_guarded_polish",
            "qubo_scheduler_state_v5_wang2017_task_polish",
            "qubo_scheduler_state_v5_wang2017_task_guarded_polish",
            "qubo_scheduler_state_v5_wang2017_task_guarded_portfolio",
            "qubo_scheduler_state_v5_wang_rescue_dynamic_polish",
            "qubo_scheduler_state_v5_wang_rescue_risk_polish",
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
        "max_moves": args.max_moves,
        "solver_profile": args.solver_profile,
        "target_final_n": 0,
        "time_limit": args.time_limit,
    }
    for field in ("beam_width", "core_width", "parent_core_cap"):
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value
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
        "frg_candidate_supplement_enabled",
        "frg_candidate_supplement_width",
        "frg_candidate_focus_n",
        "frg_candidate_n_window",
        "frg_candidate_max_ranked",
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
        "bs_frg_qubo_signature_penalty",
        "bs_frg_qubo_n_band_penalty",
        "bs_frg_qubo_n_band_size",
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
    if variant == "qubo_scheduler_state_v5_wang_rescue_light_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_wang_rescue_light_polish"})
    if variant == "qubo_scheduler_state_v5_feasibility_escape_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_feasibility_escape_polish"})
    if variant == "qubo_scheduler_state_v5_frg_dlts_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_frg_dlts_polish"})
    if variant == "qubo_scheduler_state_v5_bs_frg_qubo_polish":
        overrides.update({"solver_profile": "scheduler_state_v5_bs_frg_qubo_polish"})
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
    business_ok, goal_ok, msg = _open_validate_instance_result(result, instance, business_alpha=args.alpha)
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


def _open_variant_run_count(variant: str) -> int:
    if variant == "qubo_scheduler_state_v5_wang2017_task_guarded_portfolio":
        return len(_WANG2017_TASK_GUARDED_PORTFOLIO_VARIANTS)
    return 1


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
    rows: list[dict[str, object]] = []
    variants = _variant_names(args.variant)
    total_runs = len(cases) * sum(_open_variant_run_count(variant) for variant in variants)
    run_idx = 0
    for case in cases:
        try:
            instance = load_dlts_dat(case.instance_path, max_height=case.reference.max_height)
        except ValueError as exc:
            if args.skip_invalid:
                print(f"CASE_SKIP {case.case_id} reason={exc}", flush=True)
                continue
            raise
        for variant in variants:
            if variant == "qubo_scheduler_state_v5_wang2017_task_guarded_portfolio":
                candidates: list[dict[str, object]] = []
                portfolio_runtime_sec = 0.0
                for child_variant in _WANG2017_TASK_GUARDED_PORTFOLIO_VARIANTS:
                    run_idx += 1
                    label = f"{variant}:{child_variant}"
                    print(f"CASE_START {run_idx}/{total_runs} {label} {case.case_id}", flush=True)
                    child_row = _run_open_variant_row(case, instance, args, child_variant)
                    portfolio_runtime_sec += _numeric(child_row, "runtime_sec", 0.0)
                    candidates.append(child_row)
                    _print_open_case_done(label, case.case_id, child_row)
                selected = dict(_select_open_no_regret_row(candidates))
                selected["variant"] = variant
                selected["portfolio_profiles"] = ",".join(_WANG2017_TASK_GUARDED_PORTFOLIO_VARIANTS)
                selected["portfolio_selected_profile"] = selected.get("solver_profile", "")
                selected["portfolio_runtime_sec"] = f"{portfolio_runtime_sec:.6f}"
                selected["full_quality_penalty"] = _open_full_quality_penalty(selected)
                rows.append(selected)
                print(
                    f"CASE_DONE {variant} {case.case_id} selected_profile={selected['portfolio_selected_profile']} "
                    f"final_n={selected['final_n']} moves={selected['moves']} "
                    f"portfolio_runtime={portfolio_runtime_sec:.3f}s",
                    flush=True,
                )
                continue
            run_idx += 1
            print(f"CASE_START {run_idx}/{total_runs} {variant} {case.case_id}", flush=True)
            row = _run_open_variant_row(case, instance, args, variant)
            rows.append(row)
            _print_open_case_done(variant, case.case_id, row)
    return rows


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
            "qubo_scheduler_state_v5_wang_rescue_light_polish",
            "qubo_scheduler_state_v5_feasibility_escape_polish",
            "qubo_scheduler_state_v5_frg_dlts_polish",
            "qubo_scheduler_state_v5_bs_frg_qubo_polish",
            "qubo_scheduler_state_v5_frg_compound_polish",
            "qubo_scheduler_state_v5_frg_compound_loose_polish",
            "qubo_scheduler_state_v5_frg_compound_guarded_polish",
            "qubo_scheduler_state_v5_wang2017_task_polish",
            "qubo_scheduler_state_v5_wang2017_task_guarded_polish",
            "qubo_scheduler_state_v5_wang2017_task_guarded_portfolio",
            "qubo_scheduler_state_v5_wang_rescue_dynamic_polish",
            "qubo_scheduler_state_v5_wang_rescue_risk_polish",
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
        default="qubo",
    )
    parser.add_argument("--skip-invalid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--solver-profile", choices=sorted(_SOLVER_PROFILES.keys()), default="speed_v2")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--beam-width", type=int)
    parser.add_argument("--core-width", type=int)
    parser.add_argument("--parent-core-cap", type=int)
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
    parser.add_argument("--frg-candidate-supplement-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-candidate-supplement-width", type=int)
    parser.add_argument("--frg-candidate-focus-n", type=int)
    parser.add_argument("--frg-candidate-n-window", type=int)
    parser.add_argument("--frg-candidate-max-ranked", type=int)
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
    parser.add_argument("--bs-frg-qubo-signature-penalty", type=float)
    parser.add_argument("--bs-frg-qubo-n-band-penalty", type=float)
    parser.add_argument("--bs-frg-qubo-n-band-size", type=int)
    parser.add_argument("--wang2017-task-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wang2017-task-supplement-width", type=int)
    parser.add_argument("--wang2017-task-focus-n", type=int)
    parser.add_argument("--wang2017-task-n-window", type=int)
    parser.add_argument("--wang2017-task-plateau-window", type=int)
    parser.add_argument("--wang2017-task-plateau-delta", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "open_cpmp_dlts_speed_v2.csv",
    )
    parser.add_argument("--beam-survivor-telemetry-output", type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.source = args.source if args.source.is_absolute() else ROOT / args.source
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.beam_survivor_telemetry_output is not None and not args.beam_survivor_telemetry_output.is_absolute():
        args.beam_survivor_telemetry_output = ROOT / args.beam_survivor_telemetry_output
    _reset_beam_survivor_output(args.beam_survivor_telemetry_output)
    rows = run_open_benchmark(args)
    _write_open_csv(rows, args.output)
    summary = args.output.with_name(args.output.stem + "_summary.md")
    _write_summary(rows, args.output, summary, args.source)
    print(f"WROTE {args.output}", flush=True)
    print(f"WROTE {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
