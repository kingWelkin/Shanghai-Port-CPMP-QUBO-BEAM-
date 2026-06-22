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
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam import _write_explain_csv  # noqa: E402
from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from benchmark_fast_qubo_integrated_v1 import _exact_moves_by_case, _move_gap, _parse_case, _rel  # noqa: E402
from benchmark_full_real_business_algorithms import TrajectoryResult  # noqa: E402
from benchmark_target_astar_memory_n0 import _select_qubo_boosted  # noqa: E402
from container_reshuffle.core import (  # noqa: E402
    Instance,
    Move,
    ProblemVariant,
    Solution,
    State,
    apply_move,
    create_initial_state,
    future_disorder_count_stacks,
    validate_solution,
)
from container_reshuffle.features import disorder_box_count, entropy_limit, max_entropy_along_moves  # noqa: E402
from container_reshuffle.qubo.model import QuboModel  # noqa: E402
from container_reshuffle.qubo.sampler import QuboSamplerConfig, SimulatedAnnealingQuboSampler  # noqa: E402
from container_reshuffle.solvers.feasibility_signals import (  # noqa: E402
    FeasibilitySignals,
    compute_feasibility_signals,
    feasibility_escape_score,
)
from container_reshuffle.solvers.bs_frg import (  # noqa: E402
    _frg_completion_cached as _bs_frg_completion_cached,
    _frg_step_children as _bs_frg_step_children,
    _initial_layout as _bs_frg_initial_layout,
)
from container_reshuffle.solvers.frg_signals import (  # noqa: E402
    compute_frg_layout_signals,
    compute_frg_transition_signals,
    frg_transition_score,
)
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.classical_cpmp import (  # noqa: E402
    WangBsBabyMovesSolver,
    _WangNode,
    _wang_apply_giant_move,
    _wang_candidate_pairs,
    _wang_first_interim_from_giant_child,
    _wang_goal_stacks,
    _wang_max_group,
    _wang_node_rank,
    _wang_prepare_node,
)
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
    "deterministic_qubo_sampler",
    "target_guided_weight",
    "target_guided_focus_n",
    "target_guided_supplement_width",
    "target_guided_n_window",
    "wang_probe_supplement_width",
    "wang_probe_focus_n",
    "wang_probe_n_window",
    "wang_probe_fixed_weight",
    "wang_probe_pair_weight",
    "wang_probe_target_weight",
    "wang_probe_plateau_window",
    "wang_probe_plateau_delta",
    "wang_incumbent_enabled",
    "wang_incumbent_moves",
    "wang_incumbent_runtime_sec",
    "wang_incumbent_used",
    "wang_cap_enabled",
    "wang_rollout_enabled",
    "wang_rollout_strategy",
    "wang_rollout_inject_candidates",
    "wang_rollout_width",
    "wang_rollout_focus_n",
    "wang_rollout_candidate_limit",
    "wang_rollout_added_total",
    "wang_pair_qubo_enabled",
    "wang_pair_qubo_start_layer",
    "wang_pair_qubo_calls",
    "wang_pair_qubo_vars",
    "wang_pair_qubo_time_sec",
    "wang_feature_fixed_prefix_weight",
    "wang_feature_fix_pair_weight",
    "wang_feature_risk_weight",
    "wang_feature_same_parent_weight",
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
    "unified_wang_qubo_calls",
    "unified_wang_qubo_vars",
    "unified_wang_qubo_time_sec",
    "unified_wang_fixed_prefix_selected",
    "unified_wang_fix_pair_selected",
    "feasibility_escape_enabled",
    "feasibility_escape_supplement_width",
    "feasibility_escape_focus_n",
    "feasibility_escape_n_window",
    "feasibility_escape_dead_end_threshold",
    "feasibility_escape_min_temporary_slack",
    "feasibility_escape_calls",
    "feasibility_escape_added_total",
    "feasibility_escape_active_layers",
    "feasibility_escape_gate_reasons",
    "pareto_union_enabled",
    "pareto_union_candidate_window",
    "pareto_union_seed_width",
    "pareto_union_qubo_width",
    "pareto_union_max_vars",
    "pareto_union_n_band_size",
    "pareto_union_quality_weight",
    "pareto_union_repair_weight",
    "pareto_union_space_weight",
    "pareto_union_risk_weight",
    "pareto_union_same_parent_penalty",
    "pareto_union_same_source_penalty",
    "pareto_union_same_dest_penalty",
    "pareto_union_same_source_dest_penalty",
    "pareto_union_signature_penalty",
    "pareto_union_reduce_signature_penalty",
    "pareto_union_n_band_penalty",
    "pareto_union_risk_bucket_penalty",
    "pareto_union_layers",
    "pareto_union_seed_selected_total",
    "pareto_union_qubo_calls",
    "pareto_union_qubo_vars",
    "pareto_union_qubo_time_sec",
    "pareto_union_qubo_selected_total",
    "pareto_union_quality_selected",
    "pareto_union_space_selected",
    "pareto_union_repair_selected",
    "pareto_union_risk_selected",
    "pareto_union_distinct_parent_count_last",
    "pareto_union_distinct_source_count_last",
    "pareto_union_distinct_dest_count_last",
    "pareto_union_distinct_signature_count_last",
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
    "union_diversity_qubo_calls",
    "union_diversity_qubo_vars",
    "union_diversity_qubo_time_sec",
    "union_diversity_qubo_selected_total",
    "union_rollout_probe_enabled",
    "union_rollout_probe_max_candidates",
    "union_rollout_probe_time_limit",
    "union_rollout_probe_max_steps",
    "union_rollout_probe_feasible_bonus",
    "union_rollout_probe_final_n_weight",
    "union_rollout_probe_moves_weight",
    "union_rollout_probe_dead_end_penalty",
    "union_rollout_probe_calls",
    "union_rollout_probe_candidates",
    "union_rollout_probe_feasible",
    "union_rollout_probe_time_sec",
    "union_rollout_probe_cache_hits",
    "union_rollout_probe_selected_total",
    "union_rollout_terminal_graft",
    "union_rollout_terminal_graft_respect_cap",
    "union_rollout_terminal_grafted_total",
    "union_rollout_terminal_graft_used",
    "micro_rollout_enabled",
    "micro_rollout_trace_only",
    "receiver_safety_enabled",
    "micro_rollout_window",
    "micro_first_child_limit",
    "micro_second_child_limit",
    "micro_cache_enabled",
    "w_micro_delta_n",
    "w_micro_safe_continuation",
    "w_receiver_safe_delta",
    "w_last_buffer_consumed",
    "w_height_pressure",
    "micro_rollout_calls",
    "micro_rollout_cache_hits",
    "micro_rollout_candidates",
    "micro_rollout_selected_total",
    "receiver_safety_scored_total",
    "frg_frontier_child_enabled",
    "frg_frontier_child_parent_width",
    "frg_frontier_child_limit",
    "frg_frontier_child_max_steps",
    "frg_frontier_child_append_only",
    "frg_frontier_child_adaptive_enabled",
    "frg_frontier_child_adaptive_parent_width",
    "frg_frontier_child_adaptive_limit",
    "frg_frontier_child_adaptive_max_steps",
    "frg_frontier_child_adaptive_plateau_window",
    "frg_frontier_child_adaptive_plateau_delta",
    "frg_frontier_child_adaptive_min_n",
    "frg_frontier_child_adaptive_small_stack_max",
    "frg_frontier_child_adaptive_min_container_count",
    "frg_frontier_child_calls",
    "frg_frontier_child_generated",
    "frg_frontier_child_added_total",
    "frg_seed_enabled",
    "frg_seed_time_limit",
    "frg_seed_min_max_steps",
    "frg_seed_max_steps_multiplier",
    "frg_seed_max_steps_cap",
    "frg_seed_respect_cap",
    "frg_seed_prune_to_incumbent",
    "frg_seed_allow_large_over_cap",
    "frg_seed_paper_pure",
    "frg_seed_moves",
    "frg_seed_runtime_sec",
    "frg_seed_used",
    "frg_seed_status",
    "frg_seed_generated",
    "frg_seed_cap_ok",
    "frg_seed_large_layout_ok",
    "frg_candidate_supplement_enabled",
    "frg_candidate_supplement_width",
    "frg_candidate_focus_n",
    "frg_candidate_n_window",
    "frg_candidate_max_ranked",
    "frg_candidate_scarcity_gate_enabled",
    "frg_candidate_selected_total",
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
    "frg_layout_available_stack_count",
    "frg_layout_avg_height_pressure",
    "frg_layout_max_height_ordered_risk",
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
    "frg_compound_appended_total",
    "dlts_value_supplement_enabled",
    "dlts_value_supplement_width",
    "dlts_value_selected_total",
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
    "bs_frg_qubo_calls",
    "bs_frg_qubo_vars",
    "bs_frg_qubo_time_sec",
    "bs_frg_qubo_selected_total",
    "wang2017_task_enabled",
    "wang2017_task_supplement_width",
    "wang2017_task_focus_n",
    "wang2017_task_n_window",
    "wang2017_task_plateau_window",
    "wang2017_task_plateau_delta",
    "wang2017_task_calls",
    "wang2017_task_added_total",
    "wang2017_task_active_layers",
    "wang2017_task_selected_tier_protection_total",
    "wang2017_task_selected_stable_height_total",
    "target_layout_count",
    "beam_width",
    "core_width",
    "parent_core_cap",
    "survivor_parent_cap_enabled",
    "survivor_parent_cap_trigger_ratio",
    "survivor_parent_cap_release_ratio",
    "survivor_parent_cap_value",
    "survivor_parent_cap_window",
    "survivor_parent_cap_min_n",
    "survivor_parent_cap_layers",
    "survivor_parent_cap_last",
    "survivor_parent_cap_last_reason",
    "peak_frontier",
    "peak_candidates",
    "depth_reached",
    "open_count",
    "pruned_bound",
    "pruned_business",
    "dominated",
    "use_qubo",
    "qubo_mode",
    "qubo_start_layer",
    "qubo_focus_n",
    "qubo_first",
    "max_qubo_vars",
    "qubo_core_width",
    "qubo_supplement_width",
    "qubo_cohorts_per_layer",
    "qubo_pair_policy",
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
    "post_polish_mode",
    "post_polish_accept",
    "post_polish_window_min_stacks",
    "post_polish_attempts",
    "post_polish_improvements",
    "post_polish_moves_saved",
    "post_polish_window_min_size",
    "post_polish_window_max_size",
    "post_polish_window_stride",
    "post_polish_window_budget",
    "post_polish_window_qubo_vars",
    "post_polish_window_select",
    "post_polish_window_repair_expanded",
    "post_polish_window_repair_depth_extra",
    "post_polish_window_candidates",
    "post_polish_window_qubo_selected",
    "post_polish_window_attempts",
    "post_polish_window_successes",
    "post_polish_window_moves_saved",
    "post_polish_window_time_sec",
    "post_polish_successor_cache",
    "global_reservoir",
    "reservoir_size",
    "peak_reservoir",
    "reservoir_added_total",
    "reservoir_pulled_total",
    "stream_candidate_limit",
    "stream_candidate_trim_factor",
    "stream_bucket_topk",
    "stream_pruned_candidates",
    "rank_pool_size",
    "rank_pool_min_candidates",
    "rank_bucket_topk",
    "rank_bucket_key",
    "rank_bucket_n_band_size",
    "rank_bucket_topk_high_n",
    "rank_bucket_topk_min_candidates",
    "rank_pool_truncated_candidates",
    "scheduler_adaptive_min_stacks",
    "scheduler_adaptive_high_n",
    "scheduler_adaptive_cv1_only",
    "scheduler_adaptive_max_duplication_ratio",
    "scheduler_adaptive_min_best_n_ratio",
    "scheduler_adaptive_max_candidate_pressure",
    "scheduler_adaptive_max_parent_top_share",
    "scheduler_adaptive_plateau_window",
    "scheduler_adaptive_cap_slack",
    "scheduler_state_controller",
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
    "scarcity_inner_rescue_layers",
    "scarcity_inner_rescue_gate_reasons",
    "tail_lookahead_enabled",
    "tail_lookahead_core_path_only",
    "tail_lookahead_max_stack_count",
    "tail_lookahead_focus_n",
    "tail_lookahead_depth",
    "tail_lookahead_max_seed_candidates",
    "tail_lookahead_supplement_width",
    "tail_lookahead_max_expanded",
    "tail_lookahead_calls",
    "tail_lookahead_selected_total",
    "tail_lookahead_goal_reachable_total",
    "tail_lookahead_best_delta_total",
    "tail_lookahead_expanded_total",
    "tail_risk_reserve_enabled",
    "tail_risk_focus_n",
    "tail_risk_n_window",
    "tail_risk_reserve_width",
    "tail_risk_max_scan",
    "tail_risk_min_candidates",
    "tail_risk_min_depth_slack",
    "tail_risk_max_depth_slack",
    "tail_risk_max_stack_count",
    "tail_risk_reserve_calls",
    "tail_risk_reserve_selected_total",
    "tail_risk_reserve_scanned_total",
    "tail_risk_reserve_one_step_improved_total",
    "scarcity_group_qubo_enabled",
    "scarcity_group_qubo_threshold",
    "scarcity_group_qubo_plateau_window",
    "scarcity_group_qubo_plateau_delta",
    "scarcity_group_qubo_min_best_n_ratio",
    "scarcity_group_qubo_min_candidates",
    "scarcity_group_qubo_min_n",
    "initial_workload_hardness",
    "initial_scarcity_risk",
    "initial_scarcity_n0",
    "initial_scarcity_l0",
    "initial_scarcity_one_safe_ratio",
    "initial_light_route_score",
    "scheduler_state_last_action",
    "scheduler_state_last_effective_action",
    "scheduler_state_locked_action",
    "scheduler_state_contract_layers",
    "scheduler_state_blend_layers",
    "scheduler_state_protect_layers",
    "scheduler_state_rescue_layers",
    "adaptive_high_n",
    "adaptive_low_n",
    "adaptive_high_frontier",
    "adaptive_mid_frontier",
    "adaptive_low_frontier",
    "adaptive_core_ratio",
    "adaptive_candidate_pool",
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
    "adaptive_contract_layers",
    "adaptive_mid_layers",
    "adaptive_wide_layers",
    "adaptive_expand_layers",
    "adaptive_last_width",
    "adaptive_last_reason",
    "bucket_key",
    "bucket_n_band_size",
    "bucket_core_ratio",
    "bucket_qubo_share",
    "bucket_max_qubo_vars",
    "bucket_qubo_select",
    "bucket_min_candidates",
    "rescue_layers_used",
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
BEAM_SURVIVOR_TOP_LIMITS = (10, 64, 128)
BEAM_SURVIVOR_FIELDS = [
    "case",
    "solver_profile",
    "selection_policy",
    "layer",
    "layer_best_n",
    "frontier_size",
    "candidate_count",
    "candidate_count_before_rollout",
    "candidate_parent_count",
    "selected_count",
    "selected_parent_count",
    "selected_frontier_parent_count",
    "dead_parent_count",
    "dead_parent_ratio",
    "selected_children_top10_parent",
    "selected_children_top64_parent",
    "selected_children_top128_parent",
    "selected_children_tail_half_parent",
    "selected_parents_top10",
    "selected_parents_top64",
    "selected_parents_top128",
    "selected_parents_tail_half",
    "qubo_selected_count",
    "qubo_selected_parent_count",
    "rollout_candidate_count",
    "rollout_appended_count",
    "qcalls_total",
    "qvars_total",
    "qtime_total",
    "parent_core_cap_effective",
    "parent_core_cap_reason",
    "survivor_dead_parent_ratio_signal",
    "on_final_search_path",
    "path_child_selected_rank",
    "path_parent_frontier_rank",
]
DLTS_FEATURE_FIELDS = [
    "case",
    "solver_profile",
    "layer",
    "candidate_code",
    "n_value",
    "depth",
    "parent_rank",
    "selected",
    "on_final_path",
    "final_status",
    "fixed_prefix_total",
    "stable_prefix_total",
    "target_gap",
    "empty_slots",
    "blocked_fixed_candidates",
    "temporary_slack_moves",
    "low_tier_risk",
    "dead_end_risk",
]
FRONTIER_TRACE_FIELDS = [
    "case",
    "solver_profile",
    "selection_policy",
    "layer",
    "selection_context",
    "scheduler_action",
    "scheduler_effective_action",
    "rescue_layers_used",
    "scarcity_inner_rescue_layers",
    "wang_pair_qubo_calls",
    "qubo_calls_done",
    "frontier_best_n",
    "layer_best_n",
    "layer_has_improving_candidate",
    "candidate_count",
    "candidate_parent_count",
    "selected_count",
    "selected_qubo_count",
    "candidate_rank",
    "candidate_code",
    "n_value",
    "depth",
    "n_delta_from_layer_best",
    "candidate_improves_frontier_n",
    "parent_code",
    "parent_rank",
    "parent_candidate_count",
    "parent_selected_count",
    "parent_qubo_selected_count",
    "selected",
    "selected_rank",
    "selected_by_qubo",
    "selected_source",
    "candidate_origin",
    "micro_first_children_count",
    "micro_second_children_count",
    "micro_best_n",
    "micro_best_delta_n",
    "micro_has_reduce",
    "micro_has_safe_continuation",
    "receiver_safe_receivers_before",
    "receiver_safe_receivers_after",
    "receiver_safe_delta",
    "receiver_free_slots_before",
    "receiver_free_slots_after",
    "receiver_height_pressure_after",
    "receiver_last_buffer_consumed",
    "micro_linear_adjustment",
]


def _beam_survivor_parent_metrics(
    frontier: list[tuple[bytes, int, int]],
    candidates: dict[bytes, tuple[int, int, bytes]],
    selected_codes: list[bytes],
    selected_by_qubo: set[bytes],
    top_limits: tuple[int, ...] = BEAM_SURVIVOR_TOP_LIMITS,
) -> dict[str, int | float]:
    parent_rank_by_code = {code: idx + 1 for idx, (code, _n_value, _depth) in enumerate(frontier)}
    selected_parent_sequence = [candidates[code][2] for code in selected_codes if code in candidates]
    selected_parent_set = set(selected_parent_sequence)
    selected_frontier_parent_set = {parent for parent in selected_parent_set if parent in parent_rank_by_code}
    candidate_parent_set = {data[2] for data in candidates.values()}
    qubo_selected_codes = [code for code in selected_codes if code in selected_by_qubo and code in candidates]
    qubo_parent_set = {candidates[code][2] for code in qubo_selected_codes}
    qubo_frontier_parent_set = {parent for parent in qubo_parent_set if parent in parent_rank_by_code}
    frontier_size = len(frontier)
    dead_parent_count = max(0, frontier_size - len(selected_frontier_parent_set))
    metrics: dict[str, int | float] = {
        "frontier_size": frontier_size,
        "candidate_count": len(candidates),
        "candidate_parent_count": len(candidate_parent_set),
        "selected_count": len(selected_codes),
        "selected_parent_count": len(selected_parent_set),
        "selected_frontier_parent_count": len(selected_frontier_parent_set),
        "dead_parent_count": dead_parent_count,
        "dead_parent_ratio": dead_parent_count / frontier_size if frontier_size else 0.0,
        "qubo_selected_count": len(qubo_selected_codes),
        "qubo_selected_parent_count": len(qubo_frontier_parent_set),
    }
    for limit in top_limits:
        metrics[f"selected_children_top{limit}_parent"] = sum(
            1 for parent in selected_parent_sequence if parent_rank_by_code.get(parent, frontier_size + 1) <= limit
        )
        metrics[f"selected_parents_top{limit}"] = sum(
            1 for parent in selected_frontier_parent_set if parent_rank_by_code.get(parent, frontier_size + 1) <= limit
        )
    tail_start_rank = frontier_size // 2 + 1
    metrics["selected_children_tail_half_parent"] = sum(
        1 for parent in selected_parent_sequence if parent_rank_by_code.get(parent, 0) >= tail_start_rank
    )
    metrics["selected_parents_tail_half"] = sum(
        1 for parent in selected_frontier_parent_set if parent_rank_by_code.get(parent, 0) >= tail_start_rank
    )
    return metrics


def _finalize_beam_survivor_rows(
    records: list[dict[str, object]],
    search_path_codes: list[bytes],
) -> list[dict[str, object]]:
    path_edges = set(zip(search_path_codes, search_path_codes[1:]))
    rows: list[dict[str, object]] = []
    for record in records:
        row = dict(record["row"])
        selected_rank_by_code = record["selected_rank_by_code"]
        parent_by_code = record["parent_by_code"]
        parent_rank_by_code = record["parent_rank_by_code"]
        row["on_final_search_path"] = 0
        row["path_child_selected_rank"] = 0
        row["path_parent_frontier_rank"] = 0
        for code, selected_rank in selected_rank_by_code.items():
            parent_code = parent_by_code.get(code)
            if (parent_code, code) in path_edges:
                row["on_final_search_path"] = 1
                row["path_child_selected_rank"] = selected_rank
                row["path_parent_frontier_rank"] = parent_rank_by_code.get(parent_code, 0)
                break
        rows.append(row)
    return rows


def _write_beam_survivor_rows(rows: list[dict[str, object]], path: Path) -> None:
    output = path if path.is_absolute() else ROOT / path
    output.parent.mkdir(parents=True, exist_ok=True)
    append = output.exists() and output.stat().st_size > 0
    with output.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BEAM_SURVIVOR_FIELDS, extrasaction="ignore")
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def _reset_beam_survivor_output(path: Path | None) -> None:
    if path is None:
        return
    output = path if path.is_absolute() else ROOT / path
    if output.exists():
        output.unlink()


def _dlts_feature_row(
    *,
    case: str,
    layer: int,
    code: bytes,
    data: tuple[int, int, bytes],
    parent_rank: int,
    selected: bool,
    args: argparse.Namespace,
) -> dict[str, object]:
    signals = _feasibility_signals_for_code(code, args)
    return {
        "case": case,
        "solver_profile": getattr(args, "solver_profile", ""),
        "layer": layer,
        "candidate_code": code.hex(),
        "n_value": int(data[0]),
        "depth": int(data[1]),
        "parent_rank": int(parent_rank),
        "selected": 1 if selected else 0,
        "on_final_path": 0,
        "final_status": "",
        "fixed_prefix_total": signals.fixed_prefix_total,
        "stable_prefix_total": signals.stable_prefix_total,
        "target_gap": signals.target_gap,
        "empty_slots": signals.empty_slots,
        "blocked_fixed_candidates": signals.blocked_fixed_candidates,
        "temporary_slack_moves": signals.temporary_slack_moves,
        "low_tier_risk": signals.low_tier_risk,
        "dead_end_risk": f"{signals.dead_end_risk:.6f}",
    }


def _finalize_dlts_feature_rows(rows: list[dict[str, object]], search_path_codes: list[bytes], status: str) -> None:
    path_hex = {code.hex() for code in search_path_codes}
    for row in rows:
        row["on_final_path"] = 1 if row["candidate_code"] in path_hex else 0
        row["final_status"] = status


def _write_dlts_feature_rows(rows: list[dict[str, object]], path: Path) -> None:
    output = path if path.is_absolute() else ROOT / path
    output.parent.mkdir(parents=True, exist_ok=True)
    append = output.exists() and output.stat().st_size > 0
    with output.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DLTS_FEATURE_FIELDS, extrasaction="ignore")
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def _reset_dlts_feature_output(path: Path | None) -> None:
    if path is None:
        return
    output = path if path.is_absolute() else ROOT / path
    if output.exists():
        output.unlink()


def _write_frontier_trace_rows(rows: list[dict[str, object]], path: Path) -> None:
    output = path if path.is_absolute() else ROOT / path
    output.parent.mkdir(parents=True, exist_ok=True)
    append = output.exists() and output.stat().st_size > 0
    with output.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FRONTIER_TRACE_FIELDS, extrasaction="ignore")
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def _reset_frontier_trace_output(path: Path | None) -> None:
    if path is None:
        return
    output = path if path.is_absolute() else ROOT / path
    if output.exists():
        output.unlink()


def _frontier_trace_candidate_rows(
    *,
    case: str,
    layer: int,
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    frontier: list[tuple[bytes, int, int]],
    selected_codes: list[bytes],
    selected_by_qubo: set[bytes],
    selected_source_by_code: dict[bytes, str],
    args: argparse.Namespace,
    top_k: int,
    candidate_origin_by_code: dict[bytes, str] | None = None,
) -> list[dict[str, object]]:
    if not ranked or top_k <= 0:
        return []
    parent_rank_by_code = {code: idx + 1 for idx, (code, _n_value, _depth) in enumerate(frontier)}
    selected_rank_by_code = {code: idx + 1 for idx, code in enumerate(selected_codes)}
    selected_set = set(selected_codes)
    selected_qubo_set = selected_by_qubo & selected_set
    origin_by_code = candidate_origin_by_code or {}
    micro_feature_by_code = getattr(args, "_micro_rollout_feature_trace_by_code", {}) or {}
    parent_candidate_counts: dict[bytes, int] = {}
    parent_selected_counts: dict[bytes, int] = {}
    parent_qubo_selected_counts: dict[bytes, int] = {}
    for _code, data in ranked:
        parent_code = data[2]
        parent_candidate_counts[parent_code] = parent_candidate_counts.get(parent_code, 0) + 1
    data_by_code = {code: data for code, data in ranked}
    for code in selected_set:
        data = data_by_code.get(code)
        if data is None:
            continue
        parent_code = data[2]
        parent_selected_counts[parent_code] = parent_selected_counts.get(parent_code, 0) + 1
        if code in selected_qubo_set:
            parent_qubo_selected_counts[parent_code] = parent_qubo_selected_counts.get(parent_code, 0) + 1

    frontier_best_n = min((n_value for _code, n_value, _depth in frontier), default=0)
    layer_best_n = int(ranked[0][1][0])
    layer_has_improving_candidate = any(int(data[0]) < frontier_best_n for _code, data in ranked)
    top = ranked[: max(0, min(int(top_k), len(ranked)))]
    rows: list[dict[str, object]] = []
    for rank, (code, data) in enumerate(top, start=1):
        n_value, depth, parent_code = data
        micro_feature = micro_feature_by_code.get(code)
        selected = code in selected_set
        source = selected_source_by_code.get(code)
        if source is None:
            source = "qubo" if code in selected_qubo_set else ("selected" if selected else "not_selected")
        rows.append(
            {
                "case": case,
                "solver_profile": getattr(args, "solver_profile", ""),
                "selection_policy": getattr(args, "selection_policy", ""),
                "layer": int(layer),
                "selection_context": getattr(args, "_frontier_trace_last_context", ""),
                "scheduler_action": getattr(args, "_scheduler_state_last_action", ""),
                "scheduler_effective_action": getattr(args, "_scheduler_state_last_effective_action", ""),
                "rescue_layers_used": int(getattr(args, "_rescue_layers_used", 0)),
                "scarcity_inner_rescue_layers": int(getattr(args, "_scarcity_inner_rescue_layers", 0)),
                "wang_pair_qubo_calls": int(getattr(args, "_wang_pair_qubo_calls_done", 0)),
                "qubo_calls_done": int(getattr(args, "_qubo_calls_done", 0)),
                "frontier_best_n": int(frontier_best_n),
                "layer_best_n": int(layer_best_n),
                "layer_has_improving_candidate": int(layer_has_improving_candidate),
                "candidate_count": len(ranked),
                "candidate_parent_count": len(parent_candidate_counts),
                "selected_count": len(selected_set),
                "selected_qubo_count": len(selected_qubo_set),
                "candidate_rank": rank,
                "candidate_code": code.hex(),
                "n_value": int(n_value),
                "depth": int(depth),
                "n_delta_from_layer_best": int(n_value) - int(layer_best_n),
                "candidate_improves_frontier_n": int(int(n_value) < frontier_best_n),
                "parent_code": parent_code.hex(),
                "parent_rank": int(parent_rank_by_code.get(parent_code, 0)),
                "parent_candidate_count": int(parent_candidate_counts.get(parent_code, 0)),
                "parent_selected_count": int(parent_selected_counts.get(parent_code, 0)),
                "parent_qubo_selected_count": int(parent_qubo_selected_counts.get(parent_code, 0)),
                "selected": int(selected),
                "selected_rank": int(selected_rank_by_code.get(code, 0)),
                "selected_by_qubo": int(code in selected_qubo_set),
                "selected_source": source,
                "candidate_origin": origin_by_code.get(code, "beam"),
                "micro_first_children_count": int(getattr(micro_feature, "micro_first_children_count", 0)),
                "micro_second_children_count": int(getattr(micro_feature, "micro_second_children_count", 0)),
                "micro_best_n": int(getattr(micro_feature, "micro_best_n", -1)),
                "micro_best_delta_n": int(getattr(micro_feature, "micro_best_delta_n", 0)),
                "micro_has_reduce": int(bool(getattr(micro_feature, "micro_has_reduce", False))),
                "micro_has_safe_continuation": int(
                    bool(getattr(micro_feature, "micro_has_safe_continuation", False))
                ),
                "receiver_safe_receivers_before": int(
                    getattr(micro_feature, "receiver_safe_receivers_before", 0)
                ),
                "receiver_safe_receivers_after": int(
                    getattr(micro_feature, "receiver_safe_receivers_after", 0)
                ),
                "receiver_safe_delta": int(getattr(micro_feature, "receiver_safe_delta", 0)),
                "receiver_free_slots_before": int(getattr(micro_feature, "receiver_free_slots_before", 0)),
                "receiver_free_slots_after": int(getattr(micro_feature, "receiver_free_slots_after", 0)),
                "receiver_height_pressure_after": float(
                    getattr(micro_feature, "receiver_height_pressure_after", 0.0)
                ),
                "receiver_last_buffer_consumed": int(
                    bool(getattr(micro_feature, "receiver_last_buffer_consumed", False))
                ),
                "micro_linear_adjustment": float(getattr(micro_feature, "micro_linear_adjustment", 0.0)),
            }
        )
    return rows


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
        "target_guided_weight": 0.0,
        "target_guided_focus_n": 8,
        "target_guided_supplement_width": 0,
        "target_guided_n_window": 0,
        "wang_probe_supplement_width": 0,
        "wang_probe_focus_n": 8,
        "wang_probe_n_window": 0,
        "wang_probe_fixed_weight": 1.0,
        "wang_probe_pair_weight": 0.0,
        "wang_probe_target_weight": 0.0,
        "wang_probe_plateau_window": 0,
        "wang_probe_plateau_delta": 0,
        "wang_incumbent_enabled": False,
        "wang_incumbent_beam_width": 100,
        "wang_incumbent_candidate_limit": 64,
        "wang_incumbent_max_depth": 0,
        "wang_incumbent_time_limit": 0.0,
        "wang_cap_enabled": False,
        "wang_rollout_enabled": False,
        "wang_rollout_strategy": "local",
        "wang_rollout_inject_candidates": False,
        "wang_rollout_width": 0,
        "wang_rollout_focus_n": 8,
        "wang_rollout_plateau_window": 3,
        "wang_rollout_plateau_delta": 0,
        "wang_rollout_max_steps": 1,
        "wang_rollout_candidate_limit": 64,
        "wang_pair_qubo_enabled": False,
        "wang_pair_qubo_start_layer": 0,
        "wang_pair_qubo_supplement_width": 0,
        "wang_pair_qubo_focus_n": 8,
        "wang_pair_qubo_n_window": 1,
        "wang_pair_qubo_max_vars": 32,
        "wang_pair_qubo_select": 8,
        "wang_pair_qubo_same_parent_penalty": 0.25,
        "wang_feature_fixed_prefix_weight": 0.0,
        "wang_feature_fix_pair_weight": 0.0,
        "wang_feature_risk_weight": 0.0,
        "wang_feature_same_parent_weight": 0.0,
        "unified_wang_qubo_enabled": False,
        "unified_wang_qubo_supplement_width": 0,
        "unified_wang_qubo_focus_n": 8,
        "unified_wang_qubo_n_window": 1,
        "unified_wang_qubo_max_vars": 64,
        "unified_wang_qubo_select": 12,
        "unified_wang_n_weight": 1000.0,
        "unified_wang_depth_weight": 1.0,
        "unified_wang_fixed_prefix_weight": 0.0,
        "unified_wang_fix_pair_weight": 0.0,
        "unified_wang_target_gap_weight": 0.0,
        "unified_wang_same_parent_penalty": 0.0,
        "unified_wang_pair_same_parent_penalty": 0.0,
        "unified_wang_signature_penalty": 0.0,
        "unified_wang_complement_bonus": 0.0,
        "unified_wang_rollout_bonus": 0.0,
        "feasibility_escape_enabled": False,
        "feasibility_escape_supplement_width": 0,
        "feasibility_escape_focus_n": 8,
        "feasibility_escape_n_window": 1,
        "feasibility_escape_dead_end_threshold": 0.45,
        "feasibility_escape_min_temporary_slack": 1,
        "pareto_union_enabled": False,
        "pareto_union_candidate_window": 128,
        "pareto_union_seed_width": 32,
        "pareto_union_qubo_width": 32,
        "pareto_union_max_vars": 96,
        "pareto_union_n_band_size": 2,
        "pareto_union_quality_weight": 1.00,
        "pareto_union_repair_weight": 0.20,
        "pareto_union_space_weight": 0.12,
        "pareto_union_risk_weight": 0.16,
        "pareto_union_same_parent_penalty": 0.55,
        "pareto_union_same_source_penalty": 0.25,
        "pareto_union_same_dest_penalty": 0.20,
        "pareto_union_same_source_dest_penalty": 0.35,
        "pareto_union_signature_penalty": 0.20,
        "pareto_union_reduce_signature_penalty": 0.15,
        "pareto_union_n_band_penalty": 0.08,
        "pareto_union_risk_bucket_penalty": 0.08,
        "candidate_ledger_score_enabled": False,
        "candidate_ledger_wang_fixed_weight": 0.0,
        "candidate_ledger_wang_pair_weight": 0.0,
        "candidate_ledger_frg_fill_weight": 0.0,
        "candidate_ledger_frg_reduce_weight": 0.0,
        "candidate_ledger_frg_unblocked_weight": 0.0,
        "candidate_ledger_frg_assignment_weight": 0.0,
        "candidate_ledger_rollout_bonus": 0.0,
        "candidate_ledger_rank_window": 512,
        "union_diversity_qubo_enabled": False,
        "union_diversity_qubo_supplement_width": 0,
        "union_diversity_qubo_focus_n": -1,
        "union_diversity_qubo_n_window": 2,
        "union_diversity_qubo_min_candidates": 64,
        "union_diversity_qubo_max_vars": 64,
        "union_diversity_qubo_select": 8,
        "union_diversity_qubo_rank_weight": 0.02,
        "union_diversity_qubo_wang_fixed_weight": 0.02,
        "union_diversity_qubo_wang_pair_weight": 0.02,
        "union_diversity_qubo_frg_weight": 0.05,
        "union_diversity_qubo_rollout_bonus": 0.03,
        "union_diversity_qubo_same_parent_penalty": 0.45,
        "union_diversity_qubo_same_source_penalty": 0.20,
        "union_diversity_qubo_signature_penalty": 0.18,
        "union_diversity_qubo_reduce_signature_penalty": 0.12,
        "union_diversity_qubo_n_band_penalty": 0.08,
        "union_diversity_qubo_n_band_size": 1,
        "union_diversity_qubo_origin_similarity_penalty": 0.0,
        "union_diversity_qubo_same_origin_parent_n_band_penalty": 0.0,
        "union_rollout_probe_enabled": False,
        "union_rollout_probe_max_candidates": 8,
        "union_rollout_probe_time_limit": 0.001,
        "union_rollout_probe_max_steps": 512,
        "union_rollout_probe_feasible_bonus": 0.0,
        "union_rollout_probe_final_n_weight": 0.08,
        "union_rollout_probe_moves_weight": 0.0,
        "union_rollout_probe_dead_end_penalty": 0.12,
        "union_rollout_terminal_graft": False,
        "union_rollout_terminal_graft_respect_cap": True,
        "micro_rollout_enabled": False,
        "micro_rollout_trace_only": False,
        "receiver_safety_enabled": False,
        "micro_rollout_window": 32,
        "micro_first_child_limit": 8,
        "micro_second_child_limit": 8,
        "micro_cache_enabled": True,
        "w_micro_delta_n": 0.05,
        "w_micro_safe_continuation": 0.03,
        "w_receiver_safe_delta": 0.03,
        "w_last_buffer_consumed": 0.05,
        "w_height_pressure": 0.02,
        "frg_frontier_child_enabled": False,
        "frg_frontier_child_parent_width": 0,
        "frg_frontier_child_limit": 0,
        "frg_frontier_child_max_steps": 1,
        "frg_frontier_child_append_only": False,
        "frg_frontier_child_adaptive_enabled": False,
        "frg_frontier_child_adaptive_parent_width": 0,
        "frg_frontier_child_adaptive_limit": 0,
        "frg_frontier_child_adaptive_max_steps": 1,
        "frg_frontier_child_adaptive_plateau_window": 0,
        "frg_frontier_child_adaptive_plateau_delta": 0,
        "frg_frontier_child_adaptive_min_n": 0,
        "frg_frontier_child_adaptive_small_stack_max": 0,
        "frg_frontier_child_adaptive_min_container_count": 0,
        "frg_seed_enabled": False,
        "frg_seed_time_limit": 0.0,
        "frg_seed_min_max_steps": 256,
        "frg_seed_max_steps_multiplier": 10,
        "frg_seed_max_steps_cap": 0,
        "frg_seed_respect_cap": True,
        "frg_seed_prune_to_incumbent": True,
        "frg_seed_allow_large_over_cap": False,
        "frg_seed_paper_pure": True,
        "frg_candidate_supplement_enabled": False,
        "frg_candidate_supplement_width": 0,
        "frg_candidate_focus_n": -1,
        "frg_candidate_n_window": 2,
        "frg_candidate_max_ranked": 96,
        "frg_candidate_scarcity_gate_enabled": False,
        "frg_linear_score_enabled": False,
        "frg_linear_fill_weight": 0.0,
        "frg_linear_reduce_weight": 0.0,
        "frg_linear_unblocked_weight": 0.0,
        "frg_linear_assignment_weight": 0.0,
        "frg_linear_max_height_ordered_weight": 0.0,
        "frg_linear_height_pressure_weight": 0.0,
        "frg_rank_tiebreak_enabled": False,
        "frg_rank_tiebreak_weight": 0.0,
        "frg_rank_tiebreak_focus_n": -1,
        "frg_scarcity_gate_enabled": False,
        "frg_scarcity_gate_max_available_stacks": 1,
        "frg_scarcity_gate_min_avg_height_pressure": 0.85,
        "frg_scarcity_gate_max_reduction_receivers": -1,
        "frg_scarcity_gate_min_unblocked_bad": 0,
        "frg_scarcity_gate_min_candidates": 1000,
        "frg_scarcity_gate_cap_slack": 24,
        "frg_compound_escape_enabled": False,
        "frg_compound_escape_width": 0,
        "frg_compound_escape_focus_n": -1,
        "frg_compound_escape_plateau_window": 5,
        "frg_compound_escape_plateau_delta": 0,
        "frg_compound_escape_cap_slack": 24,
        "frg_compound_escape_min_candidates": 7000,
        "frg_compound_guard_enabled": False,
        "frg_compound_guard_max_stack_count": 0,
        "frg_compound_guard_max_initial_n": -1,
        "frg_compound_guard_min_best_n": -1,
        "frg_compound_guard_max_best_n": -1,
        "frg_compound_guard_min_candidates": 0,
        "frg_compound_guard_min_parent_count": 0,
        "frg_compound_guard_max_parent_share": 1.0,
        "dlts_value_supplement_enabled": False,
        "dlts_value_supplement_width": 0,
        "bs_frg_qubo_enabled": False,
        "bs_frg_qubo_supplement_width": 0,
        "bs_frg_qubo_focus_n": -1,
        "bs_frg_qubo_n_window": 2,
        "bs_frg_qubo_min_candidates": 64,
        "bs_frg_qubo_max_vars": 64,
        "bs_frg_qubo_rank_weight": 0.05,
        "bs_frg_qubo_same_parent_penalty": 0.35,
        "bs_frg_qubo_same_source_penalty": 0.0,
        "bs_frg_qubo_signature_penalty": 0.15,
        "bs_frg_qubo_reduce_signature_penalty": 0.0,
        "bs_frg_qubo_n_band_penalty": 0.05,
        "bs_frg_qubo_n_band_size": 1,
        "wang2017_task_enabled": False,
        "wang2017_task_supplement_width": 0,
        "wang2017_task_focus_n": 8,
        "wang2017_task_n_window": 1,
        "wang2017_task_plateau_window": 0,
        "wang2017_task_plateau_delta": 0,
        "beam_width": 1200,
        "core_width": 1080,
        "parent_core_cap": 4,
        "survivor_parent_cap_enabled": False,
        "survivor_parent_cap_trigger_ratio": 0.70,
        "survivor_parent_cap_release_ratio": 0.55,
        "survivor_parent_cap_value": 3,
        "survivor_parent_cap_window": 2,
        "survivor_parent_cap_min_n": 8,
        "selection_policy": "enhanced",
        "use_qubo": True,
        "qubo_mode": "focused_multi",
        "qubo_influence_profile": "stable",
        "max_qubo_calls": 32,
        "max_qubo_vars": 64,
        "qubo_core_width": 224,
        "qubo_supplement_width": 24,
        "qubo_min_candidates": 64,
        "qubo_start_layer": 0,
        "qubo_focus_n": 8,
        "qubo_first": False,
        "qubo_target_share": 0.50,
        "qubo_cohorts_per_layer": 2,
        "qubo_cohort_stride": 64,
        "qubo_pair_policy": "baseline",
        "qubo_parent_sim_weight": 0.65,
        "qubo_layout_sim_weight": 1.0,
        "qubo_action_sim_weight": 0.0,
        "qubo_stack_sim_weight": 0.35,
        "qubo_sim_threshold": 0.80,
        "deterministic_qubo_sampler": False,
        "group_qubo_supplement": False,
        "max_group_qubo_calls": 32,
        "qubo_group_max_vars": 96,
        "qubo_group_select": 24,
        "qubo_group_take": 4,
        "qubo_group_lookahead_weight": 0.0,
        "qubo_group_branch_weight": 0.0,
        "qubo_next_n_weight": 0.0,
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
        "stream_bucket_topk": 0,
        "rank_pool_size": 0,
        "rank_pool_min_candidates": 0,
        "rank_bucket_topk": 0,
        "rank_bucket_key": "n_depth_parent",
        "rank_bucket_n_band_size": 2,
        "rank_bucket_topk_high_n": 0,
        "rank_bucket_topk_min_candidates": 0,
        "scheduler_adaptive_min_stacks": 7,
        "scheduler_adaptive_high_n": 16,
        "scheduler_adaptive_cv1_only": False,
        "scheduler_adaptive_max_duplication_ratio": 1.0,
        "scheduler_adaptive_min_best_n_ratio": 0.0,
        "scheduler_adaptive_max_candidate_pressure": 0.0,
        "scheduler_adaptive_max_parent_top_share": 0.0,
        "scheduler_adaptive_plateau_window": 0,
        "scheduler_adaptive_cap_slack": 0,
        "scheduler_state_controller": "off",
        "scheduler_state_contract_pressure": 8.0,
        "scheduler_state_max_contract_pressure": 40.0,
        "scheduler_state_min_best_n_ratio": 0.0,
        "scheduler_state_progress_window": 3,
        "scheduler_state_progress_delta": 2,
        "scheduler_state_failed_contract_cooldown": 0,
        "scheduler_state_post_rescue_plateau_window": 0,
        "scheduler_state_post_rescue_plateau_delta": 0,
        "scheduler_state_post_rescue_min_ratio": 0.90,
        "scheduler_state_post_rescue_action": "protect",
        "scheduler_state_initial_plateau_window": 0,
        "scheduler_state_initial_plateau_min_ratio": 0.98,
        "scheduler_state_initial_plateau_action": "protect",
        "scheduler_state_initial_plateau_lock": False,
        "scheduler_state_initial_plateau_min_duplication": 0.0,
        "scheduler_state_initial_plateau_unique_min_n": 0,
        "scheduler_state_protect_parent_top_share": 0.65,
        "scheduler_state_protect_n_band": 2,
        "scheduler_state_top_window": 64,
        "scheduler_state_rescue_plateau_window": 3,
        "scheduler_state_rescue_plateau_delta": 0,
        "scheduler_state_rescue_cap_slack": 8,
        "scarcity_inner_rescue_enabled": False,
        "scarcity_inner_rescue_risk_threshold": 0.95,
        "scarcity_inner_rescue_max_l0": 5,
        "scarcity_inner_rescue_min_one_safe_ratio": 0.90,
        "scarcity_inner_rescue_min_n": 12,
        "scarcity_inner_rescue_min_candidates": 64,
        "tail_lookahead_enabled": False,
        "tail_lookahead_core_path_only": False,
        "tail_lookahead_max_stack_count": 0,
        "tail_lookahead_focus_n": 2,
        "tail_lookahead_depth": 3,
        "tail_lookahead_max_seed_candidates": 128,
        "tail_lookahead_supplement_width": 0,
        "tail_lookahead_max_expanded": 4096,
        "tail_risk_reserve_enabled": False,
        "tail_risk_focus_n": 2,
        "tail_risk_n_window": 1,
        "tail_risk_reserve_width": 0,
        "tail_risk_max_scan": 4096,
        "tail_risk_min_candidates": 1000,
        "tail_risk_min_depth_slack": 2,
        "tail_risk_max_depth_slack": 10,
        "tail_risk_max_stack_count": 0,
        "scarcity_group_qubo_enabled": False,
        "scarcity_group_qubo_threshold": 0.95,
        "scarcity_group_qubo_plateau_window": 2,
        "scarcity_group_qubo_plateau_delta": 0,
        "scarcity_group_qubo_min_best_n_ratio": 0.70,
        "scarcity_group_qubo_min_candidates": 1000,
        "scarcity_group_qubo_min_n": 12,
        "post_polish_qubo": True,
        "post_polish_mode": "legacy",
        "post_polish_accept": "better",
        "post_polish_window_min_stacks": 0,
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
        "post_polish_window_min_size": 3,
        "post_polish_window_max_size": 10,
        "post_polish_window_stride": 1,
        "post_polish_window_budget": 96,
        "post_polish_window_qubo_vars": 72,
        "post_polish_window_select": 8,
        "post_polish_window_min_score_ratio": 0.0,
        "post_polish_window_time_limit": 0.001,
        "post_polish_window_repair_expanded": 20000,
        "post_polish_window_repair_depth_extra": 0,
        "post_polish_successor_cache": False,
        "tail_order_steps": 6,
        "seed": 37,
        "adaptive_high_n": 16,
        "adaptive_low_n": 8,
        "adaptive_high_frontier": 800,
        "adaptive_mid_frontier": 1000,
        "adaptive_low_frontier": 1200,
        "adaptive_core_ratio": 0.90,
        "adaptive_candidate_pool": 0,
        "adaptive_controller": "tiered",
        "adaptive_expand_frontier": 1200,
        "adaptive_contract_pressure": 0.0,
        "adaptive_mid_pressure": 0.0,
        "adaptive_min_best_n_ratio": 0.0,
        "adaptive_max_duplication_ratio": 1.0,
        "adaptive_parent_top_share": 0.0,
        "adaptive_plateau_window": 0,
        "adaptive_plateau_delta": 0,
        "adaptive_cap_slack": 0,
        "adaptive_large_stack_min_stacks": 0,
        "adaptive_large_stack_expand_frontier": 0,
        "adaptive_large_stack_plateau_window": 0,
        "adaptive_large_stack_cap_slack": 0,
        "bucket_key": "n_depth",
        "bucket_n_band_size": 2,
        "bucket_core_ratio": 0.50,
        "bucket_qubo_share": 0.35,
        "bucket_max_qubo_vars": 96,
        "bucket_qubo_select": 24,
        "bucket_min_candidates": 64,
        "rescue_enabled": False,
        "rescue_plateau_window": 2,
        "rescue_plateau_delta": 0,
        "rescue_cap_slack": 26,
        "rescue_min_n": 12,
        "rescue_min_candidates": 5000,
        "rescue_max_layers": 16,
        "rescue_beam_width": 1350,
        "rescue_core_width": 1215,
        "rescue_escalate_after": 2,
        "rescue_escalate_beam_width": 1600,
        "rescue_escalate_core_width": 1440,
        "rescue_max_qubo_vars": 96,
        "rescue_qubo_supplement_width": 48,
        "rescue_qubo_cohorts_per_layer": 4,
        "rescue_qubo_time_limit": 0.002,
        "rescue_qubo_iterations": 30,
        "rescue_next_n_weight": 0.75,
    },
}
_SOLVER_PROFILES["target_guided_v1"] = {
    **_SOLVER_PROFILES["speed_v2"],
    "target_guided_weight": 0.35,
    "target_guided_focus_n": 8,
    "target_guided_supplement_width": 24,
    "target_guided_n_window": 0,
}
_SOLVER_PROFILES["wang_probe_v1"] = {
    **_SOLVER_PROFILES["speed_v2"],
    "wang_probe_supplement_width": 12,
    "wang_probe_focus_n": 8,
    "wang_probe_n_window": 1,
    "wang_probe_fixed_weight": 0.0,
    "wang_probe_pair_weight": 1.0,
    "wang_probe_target_weight": 0.0,
    "wang_probe_plateau_window": 3,
    "wang_probe_plateau_delta": 0,
}
_SOLVER_PROFILES["post_polish_v1"] = {
    **_SOLVER_PROFILES["speed_v2"],
    "post_polish_passes": 3,
    "post_polish_tail_span": 18,
    "post_polish_max_windows": 128,
    "post_polish_max_qubo_vars": 96,
    "post_polish_max_bridge_depth": 8,
    "post_polish_bridge_expanded": 30000,
    "post_polish_suffix_focus_n": 10,
    "post_polish_suffix_tail_span": 22,
    "post_polish_suffix_max_candidates": 72,
    "post_polish_suffix_select": 7,
    "post_polish_suffix_max_qubo_vars": 64,
    "post_polish_suffix_max_depth": 20,
    "post_polish_suffix_expanded": 90000,
}
_SOLVER_PROFILES["window_polish_v2"] = {
    **_SOLVER_PROFILES["speed_v2"],
    "post_polish_mode": "qubo_windows_v2",
    "post_polish_accept": "better",
    "post_polish_passes": 2,
    "post_polish_focus_n": 8,
    "post_polish_tail_span": 16,
    "post_polish_window_min_size": 3,
    "post_polish_window_max_size": 8,
    "post_polish_window_stride": 1,
    "post_polish_window_budget": 96,
    "post_polish_window_qubo_vars": 72,
    "post_polish_window_select": 1,
    "post_polish_window_time_limit": 0.001,
    "post_polish_window_repair_expanded": 3000,
    "post_polish_window_repair_depth_extra": 0,
}
_SOLVER_PROFILES["frontier_quality_v1"] = {
    **_SOLVER_PROFILES["speed_v2"],
    "max_qubo_calls": 64,
    "max_qubo_vars": 96,
    "qubo_supplement_width": 32,
    "qubo_focus_n": 10,
    "qubo_cohorts_per_layer": 3,
    "qubo_time_limit": 0.002,
    "qubo_iterations": 30,
    "qubo_parent_weight": 0.30,
    "qubo_next_n_weight": 0.75,
}
_SOLVER_PROFILES["rescue_qubo_v1"] = {
    **_SOLVER_PROFILES["speed_v2"],
    "selection_policy": "rescue_qubo",
    "rescue_enabled": True,
    "rescue_plateau_window": 2,
    "rescue_plateau_delta": 0,
    "rescue_cap_slack": 26,
    "rescue_min_n": 12,
    "rescue_min_candidates": 5000,
    "rescue_max_layers": 16,
    "rescue_beam_width": 1350,
    "rescue_core_width": 1215,
    "rescue_escalate_after": 2,
    "rescue_escalate_beam_width": 1600,
    "rescue_escalate_core_width": 1440,
    "rescue_max_qubo_vars": 96,
    "rescue_qubo_supplement_width": 48,
    "rescue_qubo_cohorts_per_layer": 4,
    "rescue_qubo_time_limit": 0.002,
    "rescue_qubo_iterations": 30,
    "rescue_next_n_weight": 0.75,
}
_SOLVER_PROFILES["light_rescue_qubo_v1"] = {
    **_SOLVER_PROFILES["speed_v2"],
    "selection_policy": "rescue_qubo",
    "rescue_enabled": True,
    "rescue_plateau_window": 2,
    "rescue_plateau_delta": 0,
    "rescue_cap_slack": 24,
    "rescue_min_n": 12,
    "rescue_min_candidates": 7000,
    "rescue_max_layers": 8,
    "rescue_beam_width": 1250,
    "rescue_core_width": 1125,
    "rescue_escalate_after": 4,
    "rescue_escalate_beam_width": 1450,
    "rescue_escalate_core_width": 1305,
    "rescue_max_qubo_vars": 80,
    "rescue_qubo_supplement_width": 32,
    "rescue_qubo_cohorts_per_layer": 3,
    "rescue_qubo_time_limit": 0.0015,
    "rescue_qubo_iterations": 24,
    "rescue_next_n_weight": 0.65,
}
_SOLVER_PROFILES["adaptive_rescue_v1"] = {
    **_SOLVER_PROFILES["light_rescue_qubo_v1"],
    "selection_policy": "adaptive_qubo",
    "rescue_plateau_window": 3,
    "rescue_cap_slack": 12,
    "rescue_min_candidates": 20000,
    "rescue_max_layers": 3,
    "rescue_beam_width": 1200,
    "rescue_core_width": 1080,
    "rescue_escalate_after": 0,
    "rescue_escalate_beam_width": 1200,
    "rescue_escalate_core_width": 1080,
    "adaptive_high_n": 16,
    "adaptive_low_n": 8,
    "adaptive_high_frontier": 800,
    "adaptive_mid_frontier": 1000,
    "adaptive_low_frontier": 1200,
    "adaptive_core_ratio": 0.90,
    "adaptive_candidate_pool": 0,
}
_SOLVER_PROFILES["adaptive_polish_v1"] = {
    **_SOLVER_PROFILES["post_polish_v1"],
    "selection_policy": "adaptive_qubo",
    "adaptive_high_n": 16,
    "adaptive_low_n": 8,
    "adaptive_high_frontier": 800,
    "adaptive_mid_frontier": 1000,
    "adaptive_low_frontier": 1200,
    "adaptive_core_ratio": 0.90,
    "adaptive_candidate_pool": 0,
}
_SOLVER_PROFILES["adaptive_controller_v1"] = {
    **_SOLVER_PROFILES["speed_v2"],
    "selection_policy": "adaptive_qubo",
    "adaptive_controller": "signal_v1",
    "adaptive_high_n": 16,
    "adaptive_low_n": 8,
    "adaptive_high_frontier": 800,
    "adaptive_mid_frontier": 1000,
    "adaptive_low_frontier": 1200,
    "adaptive_expand_frontier": 1200,
    "adaptive_core_ratio": 0.90,
    "adaptive_candidate_pool": 0,
    "adaptive_contract_pressure": 0.5,
    "adaptive_mid_pressure": 0.25,
    "adaptive_min_best_n_ratio": 0.45,
    "adaptive_max_duplication_ratio": 1.0,
    "adaptive_parent_top_share": 0.0,
    "adaptive_plateau_window": 4,
    "adaptive_plateau_delta": 0,
    "adaptive_cap_slack": 4,
    "adaptive_large_stack_min_stacks": 8,
    "adaptive_large_stack_expand_frontier": 1400,
    "adaptive_large_stack_plateau_window": 3,
    "adaptive_large_stack_cap_slack": 12,
}
_SOLVER_PROFILES["scheduler_v1"] = {
    **_SOLVER_PROFILES["light_rescue_qubo_v1"],
    "selection_policy": "scheduler_qubo",
    "scheduler_adaptive_min_stacks": 7,
    "scheduler_adaptive_high_n": 16,
    "scheduler_adaptive_cv1_only": False,
    "scheduler_adaptive_max_duplication_ratio": 0.0,
    "scheduler_adaptive_min_best_n_ratio": 0.45,
    "scheduler_adaptive_max_candidate_pressure": 0.0,
    "scheduler_adaptive_max_parent_top_share": 0.0,
    "scheduler_adaptive_plateau_window": 3,
    "scheduler_adaptive_cap_slack": 12,
    "adaptive_high_frontier": 800,
    "adaptive_mid_frontier": 1000,
    "adaptive_low_frontier": 1200,
    "adaptive_core_ratio": 0.90,
    "adaptive_candidate_pool": 0,
}
_SOLVER_PROFILES["scheduler_topk_v1"] = {
    **_SOLVER_PROFILES["scheduler_v1"],
    "rank_bucket_topk": 64,
    "rank_bucket_key": "n_depth_parent",
    "rank_bucket_n_band_size": 2,
    "rank_bucket_topk_high_n": 18,
    "rank_bucket_topk_min_candidates": 10000,
}
_SOLVER_PROFILES["scheduler_state_v2"] = {
    **_SOLVER_PROFILES["light_rescue_qubo_v1"],
    "selection_policy": "scheduler_state_qubo",
    "scheduler_state_controller": "state_v2",
    "scheduler_state_contract_pressure": 8.0,
    "scheduler_state_max_contract_pressure": 40.0,
    "scheduler_state_min_best_n_ratio": 0.0,
    "scheduler_state_progress_window": 3,
    "scheduler_state_progress_delta": 0,
    "scheduler_state_failed_contract_cooldown": 0,
    "scheduler_state_post_rescue_plateau_window": 0,
    "scheduler_state_post_rescue_plateau_delta": 0,
    "scheduler_state_post_rescue_min_ratio": 0.90,
    "scheduler_state_post_rescue_action": "protect",
    "scheduler_state_initial_plateau_window": 0,
    "scheduler_state_initial_plateau_min_ratio": 0.98,
    "scheduler_state_initial_plateau_action": "protect",
    "scheduler_state_initial_plateau_lock": False,
    "scheduler_state_initial_plateau_min_duplication": 0.0,
    "scheduler_state_initial_plateau_unique_min_n": 0,
    "scheduler_state_protect_parent_top_share": 0.65,
    "scheduler_state_protect_n_band": 2,
    "scheduler_state_top_window": 64,
    "scheduler_state_rescue_plateau_window": 3,
    "scheduler_state_rescue_plateau_delta": 0,
    "scheduler_state_rescue_cap_slack": 8,
    "adaptive_controller": "signal_v1",
    "adaptive_contract_pressure": 0.5,
    "adaptive_mid_pressure": 0.25,
    "adaptive_min_best_n_ratio": 0.45,
    "adaptive_high_frontier": 800,
    "adaptive_mid_frontier": 1000,
    "adaptive_low_frontier": 1200,
}

_SOLVER_PROFILES["scheduler_state_v3"] = {
    **_SOLVER_PROFILES["scheduler_state_v2"],
    "scheduler_state_failed_contract_cooldown": 0,
    "scheduler_state_post_rescue_plateau_window": 0,
    "scheduler_state_post_rescue_plateau_delta": 0,
    "scheduler_state_post_rescue_min_ratio": 0.90,
    "scheduler_state_post_rescue_action": "protect",
    "scheduler_state_initial_plateau_window": 5,
    "scheduler_state_initial_plateau_min_ratio": 0.98,
    "scheduler_state_initial_plateau_action": "rescue",
    "scheduler_state_initial_plateau_lock": True,
    "scheduler_state_initial_plateau_min_duplication": 0.10,
}

_SOLVER_PROFILES["scheduler_state_v4"] = {
    **_SOLVER_PROFILES["scheduler_state_v3"],
    "scheduler_state_initial_plateau_min_duplication": 0.0,
}

_SOLVER_PROFILES["scheduler_state_v5"] = {
    **_SOLVER_PROFILES["scheduler_state_v4"],
    "scheduler_state_initial_plateau_unique_min_n": 24,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_probe"] = {
    **_SOLVER_PROFILES["scheduler_state_v5"],
    "wang_probe_supplement_width": 12,
    "wang_probe_focus_n": 8,
    "wang_probe_n_window": 1,
    "wang_probe_fixed_weight": 0.0,
    "wang_probe_pair_weight": 1.0,
    "wang_probe_target_weight": 0.0,
    "wang_probe_plateau_window": 3,
    "wang_probe_plateau_delta": 0,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_hybrid"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_probe"],
    "wang_incumbent_enabled": True,
    "wang_incumbent_beam_width": 100,
    "wang_incumbent_candidate_limit": 64,
    "wang_incumbent_max_depth": 0,
    "wang_incumbent_time_limit": 0.25,
    "wang_cap_enabled": True,
    "wang_rollout_enabled": True,
    "wang_rollout_width": 12,
    "wang_rollout_focus_n": 8,
    "wang_rollout_plateau_window": 3,
    "wang_rollout_plateau_delta": 0,
    "wang_rollout_max_steps": 1,
    "wang_pair_qubo_enabled": True,
    "wang_pair_qubo_supplement_width": 8,
    "wang_pair_qubo_focus_n": 8,
    "wang_pair_qubo_n_window": 1,
    "wang_pair_qubo_max_vars": 32,
    "wang_pair_qubo_select": 8,
    "wang_pair_qubo_same_parent_penalty": 0.25,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_pair_qubo"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_probe"],
    "wang_pair_qubo_enabled": True,
    "wang_pair_qubo_supplement_width": 8,
    "wang_pair_qubo_focus_n": 8,
    "wang_pair_qubo_n_window": 1,
    "wang_pair_qubo_max_vars": 32,
    "wang_pair_qubo_select": 8,
    "wang_pair_qubo_same_parent_penalty": 0.25,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_rescue"] = {
    **_SOLVER_PROFILES["scheduler_state_v5"],
    "wang_pair_qubo_enabled": True,
    "wang_pair_qubo_supplement_width": 12,
    "wang_pair_qubo_focus_n": -1,
    "wang_pair_qubo_n_window": 1,
    "wang_pair_qubo_max_vars": 48,
    "wang_pair_qubo_select": 12,
    "wang_pair_qubo_same_parent_penalty": 0.35,
    "rescue_qubo_supplement_width": 56,
    "rescue_max_qubo_vars": 112,
    "rescue_next_n_weight": 0.75,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_light"] = {
    **_SOLVER_PROFILES["scheduler_state_v5"],
    "wang_pair_qubo_enabled": True,
    "wang_pair_qubo_supplement_width": 8,
    "wang_pair_qubo_focus_n": -1,
    "wang_pair_qubo_n_window": 1,
    "wang_pair_qubo_max_vars": 32,
    "wang_pair_qubo_select": 8,
    "wang_pair_qubo_same_parent_penalty": 0.25,
    "rescue_qubo_supplement_width": 32,
    "rescue_max_qubo_vars": 64,
    "rescue_next_n_weight": 0.70,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_features"] = {
    **_SOLVER_PROFILES["scheduler_state_v5"],
    "group_qubo_supplement": True,
    "max_group_qubo_calls": 64,
    "qubo_group_max_vars": 96,
    "qubo_group_select": 18,
    "qubo_group_take": 3,
    "qubo_group_lookahead_weight": 0.25,
    "qubo_group_branch_weight": 0.05,
    "group_qubo_risk_gate": True,
    "group_qubo_risk_min_candidates": 12000,
    "group_qubo_risk_parent_share": 0.18,
    "group_qubo_risk_plateau_window": 3,
    "group_qubo_risk_plateau_delta": 0,
    "wang_feature_fixed_prefix_weight": 18.0,
    "wang_feature_fix_pair_weight": 8.0,
    "wang_feature_risk_weight": 1.0,
    "wang_feature_same_parent_weight": 0.35,
    "wang_pair_qubo_enabled": False,
}
_SOLVER_PROFILES["scheduler_state_v7_wang_qubo_unified"] = {
    **_SOLVER_PROFILES["scheduler_state_v5"],
    "group_qubo_supplement": True,
    "max_group_qubo_calls": 64,
    "qubo_group_max_vars": 96,
    "qubo_group_select": 18,
    "qubo_group_take": 3,
    "group_qubo_risk_gate": True,
    "group_qubo_risk_min_candidates": 12000,
    "group_qubo_risk_parent_share": 0.18,
    "group_qubo_risk_plateau_window": 3,
    "group_qubo_risk_plateau_delta": 0,
    "unified_wang_qubo_enabled": True,
    "unified_wang_qubo_supplement_width": 1,
    "unified_wang_qubo_focus_n": -1,
    "unified_wang_qubo_n_window": 1,
    "unified_wang_qubo_max_vars": 64,
    "unified_wang_qubo_select": 1,
    "unified_wang_n_weight": 1000.0,
    "unified_wang_depth_weight": 1.0,
    "unified_wang_fixed_prefix_weight": 20.0,
    "unified_wang_fix_pair_weight": 8.0,
    "unified_wang_target_gap_weight": 1.0,
    "unified_wang_same_parent_penalty": 0.10,
    "unified_wang_pair_same_parent_penalty": 0.25,
    "unified_wang_signature_penalty": 0.10,
    "unified_wang_complement_bonus": 0.05,
    "unified_wang_rollout_bonus": 0.0,
    "wang_incumbent_enabled": False,
    "wang_cap_enabled": False,
    "wang_rollout_enabled": True,
    "wang_rollout_strategy": "baby_lookahead",
    "wang_rollout_inject_candidates": False,
    "wang_rollout_width": 2,
    "wang_rollout_focus_n": -1,
    "wang_rollout_plateau_window": 0,
    "wang_rollout_candidate_limit": 64,
    "wang_pair_qubo_enabled": False,
    "rescue_enabled": False,
    "rescue_max_layers": 0,
    "rescue_cap_slack": 0,
    "rescue_qubo_supplement_width": 0,
    "scheduler_state_rescue_cap_slack": 0,
    "scheduler_state_initial_plateau_window": 0,
    "scheduler_state_initial_plateau_action": "protect",
    "scheduler_state_initial_plateau_lock": False,
    "post_polish_qubo": False,
}
_SOLVER_PROFILES["scheduler_state_v7_wang_qubo_pool"] = {
    **_SOLVER_PROFILES["scheduler_state_v7_wang_qubo_unified"],
    "wang_rollout_inject_candidates": True,
    "unified_wang_rollout_bonus": 25.0,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue"],
    "post_polish_qubo": True,
    "post_polish_mode": "qubo_windows_v2",
    "post_polish_accept": "better",
    "post_polish_passes": 3,
    "post_polish_suffix_max_candidates": 72,
    "post_polish_suffix_select": 7,
    "post_polish_suffix_max_qubo_vars": 64,
    "post_polish_window_select": 2,
    "post_polish_window_max_size": 8,
    "post_polish_window_budget": 96,
    "post_polish_window_qubo_vars": 72,
    "post_polish_window_repair_expanded": 6000,
    "post_polish_window_repair_depth_extra": 0,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_fast_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "rescue_max_qubo_vars": 64,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_scarcity_inner_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "scarcity_inner_rescue_enabled": True,
    "scarcity_inner_rescue_risk_threshold": 0.95,
    "scarcity_inner_rescue_max_l0": 5,
    "scarcity_inner_rescue_min_one_safe_ratio": 0.90,
    "scarcity_inner_rescue_min_n": 12,
    "scarcity_inner_rescue_min_candidates": 64,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_tail_lookahead_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "tail_lookahead_enabled": True,
    "tail_lookahead_focus_n": 2,
    "tail_lookahead_depth": 3,
    "tail_lookahead_max_seed_candidates": 128,
    "tail_lookahead_supplement_width": 24,
    "tail_lookahead_max_expanded": 4096,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_tail_risk_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "tail_risk_reserve_enabled": True,
    "tail_risk_focus_n": 2,
    "tail_risk_n_window": 1,
    "tail_risk_reserve_width": 2,
    "tail_risk_max_scan": 12000,
    "tail_risk_min_candidates": 1000,
    "tail_risk_min_depth_slack": 2,
    "tail_risk_max_depth_slack": 8,
    "tail_risk_max_stack_count": 5,
    "tail_lookahead_enabled": False,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_successor_cache_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "post_polish_successor_cache": True,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_light_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_light"],
    "post_polish_qubo": True,
    "post_polish_mode": "qubo_windows_v2",
    "post_polish_accept": "better",
    "post_polish_passes": 3,
    "post_polish_suffix_max_candidates": 72,
    "post_polish_suffix_select": 7,
    "post_polish_suffix_max_qubo_vars": 64,
    "post_polish_window_select": 2,
    "post_polish_window_max_size": 8,
    "post_polish_window_budget": 96,
    "post_polish_window_qubo_vars": 72,
    "post_polish_window_repair_expanded": 6000,
    "post_polish_window_repair_depth_extra": 0,
}
_SOLVER_PROFILES["scheduler_state_v5_feasibility_escape_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "feasibility_escape_enabled": True,
    "feasibility_escape_supplement_width": 4,
    "feasibility_escape_focus_n": -1,
    "feasibility_escape_n_window": 2,
    "feasibility_escape_dead_end_threshold": 0.35,
    "feasibility_escape_min_temporary_slack": 1,
}
_SOLVER_PROFILES["scheduler_state_v5_frg_dlts_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "frg_candidate_supplement_enabled": True,
    "frg_candidate_supplement_width": 8,
    "frg_candidate_focus_n": -1,
    "frg_candidate_n_window": 2,
    "frg_candidate_max_ranked": 96,
    "frg_compound_escape_enabled": False,
    "frg_compound_escape_width": 0,
    "dlts_value_supplement_enabled": True,
    "dlts_value_supplement_width": 4,
}
_SOLVER_PROFILES["scheduler_state_v5_frg_linear_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "frg_linear_score_enabled": True,
    "frg_linear_fill_weight": 0.18,
    "frg_linear_reduce_weight": 0.12,
    "frg_linear_unblocked_weight": 0.08,
    "frg_linear_assignment_weight": 0.04,
    "frg_linear_max_height_ordered_weight": 0.18,
    "frg_linear_height_pressure_weight": 0.05,
    "frg_candidate_supplement_enabled": False,
    "frg_candidate_supplement_width": 0,
    "bs_frg_qubo_enabled": False,
    "bs_frg_qubo_supplement_width": 0,
}
_SOLVER_PROFILES["scheduler_state_v5_frg_safe_tiebreak_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "frg_rank_tiebreak_enabled": True,
    "frg_rank_tiebreak_weight": 1.0,
    "frg_rank_tiebreak_focus_n": 0,
    "frg_linear_score_enabled": False,
    "frg_candidate_supplement_enabled": False,
    "frg_candidate_supplement_width": 0,
    "bs_frg_qubo_enabled": False,
    "bs_frg_qubo_supplement_width": 0,
}
_SOLVER_PROFILES["scheduler_state_v5_frg_diverse_qubo_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "bs_frg_qubo_enabled": True,
    "bs_frg_qubo_supplement_width": 4,
    "bs_frg_qubo_focus_n": -1,
    "bs_frg_qubo_n_window": 2,
    "bs_frg_qubo_min_candidates": 64,
    "bs_frg_qubo_max_vars": 64,
    "bs_frg_qubo_rank_weight": 0.05,
    "bs_frg_qubo_same_parent_penalty": 0.55,
    "bs_frg_qubo_same_source_penalty": 0.35,
    "bs_frg_qubo_signature_penalty": 0.25,
    "bs_frg_qubo_reduce_signature_penalty": 0.20,
    "bs_frg_qubo_n_band_penalty": 0.12,
    "bs_frg_qubo_n_band_size": 1,
    "frg_candidate_supplement_enabled": False,
    "frg_candidate_supplement_width": 0,
}
_SOLVER_PROFILES["scheduler_state_v5_frg_candidate_pool_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "frg_candidate_supplement_enabled": True,
    "frg_candidate_supplement_width": 8,
    "frg_candidate_focus_n": -1,
    "frg_candidate_n_window": 2,
    "frg_candidate_max_ranked": 96,
    "frg_candidate_scarcity_gate_enabled": False,
    "dlts_value_supplement_enabled": False,
    "dlts_value_supplement_width": 0,
    "bs_frg_qubo_enabled": False,
    "bs_frg_qubo_supplement_width": 0,
}
_SOLVER_PROFILES["scheduler_state_v5_frg_scarcity_gate_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "frg_candidate_supplement_enabled": True,
    "frg_candidate_supplement_width": 4,
    "frg_candidate_focus_n": -1,
    "frg_candidate_n_window": 2,
    "frg_candidate_max_ranked": 96,
    "frg_candidate_scarcity_gate_enabled": True,
    "frg_scarcity_gate_enabled": True,
    "frg_scarcity_gate_max_available_stacks": 99,
    "frg_scarcity_gate_min_avg_height_pressure": 0.0,
    "frg_scarcity_gate_max_reduction_receivers": 2,
    "frg_scarcity_gate_min_unblocked_bad": 1,
    "frg_scarcity_gate_min_candidates": 1000,
    "frg_scarcity_gate_cap_slack": 24,
    "dlts_value_supplement_enabled": False,
    "dlts_value_supplement_width": 0,
    "bs_frg_qubo_enabled": False,
    "bs_frg_qubo_supplement_width": 0,
}
_SOLVER_PROFILES["scheduler_state_v5_frg_all_ablation_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_frg_scarcity_gate_polish"],
    "frg_linear_score_enabled": True,
    "frg_linear_fill_weight": 0.18,
    "frg_linear_reduce_weight": 0.12,
    "frg_linear_unblocked_weight": 0.08,
    "frg_linear_assignment_weight": 0.04,
    "frg_linear_max_height_ordered_weight": 0.18,
    "frg_linear_height_pressure_weight": 0.05,
    "bs_frg_qubo_enabled": True,
    "bs_frg_qubo_supplement_width": 4,
    "bs_frg_qubo_focus_n": -1,
    "bs_frg_qubo_n_window": 2,
    "bs_frg_qubo_min_candidates": 64,
    "bs_frg_qubo_max_vars": 64,
    "bs_frg_qubo_same_parent_penalty": 0.55,
    "bs_frg_qubo_same_source_penalty": 0.35,
    "bs_frg_qubo_signature_penalty": 0.25,
    "bs_frg_qubo_reduce_signature_penalty": 0.20,
    "bs_frg_qubo_n_band_penalty": 0.12,
}
_SOLVER_PROFILES["scheduler_state_v5_bs_frg_qubo_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "bs_frg_qubo_enabled": True,
    "bs_frg_qubo_supplement_width": 1,
    "bs_frg_qubo_focus_n": 8,
    "bs_frg_qubo_n_window": 2,
    "bs_frg_qubo_min_candidates": 256,
    "bs_frg_qubo_max_vars": 64,
    "bs_frg_qubo_rank_weight": 0.05,
    "bs_frg_qubo_same_parent_penalty": 0.35,
    "bs_frg_qubo_signature_penalty": 0.15,
    "bs_frg_qubo_n_band_penalty": 0.05,
    "bs_frg_qubo_n_band_size": 1,
}
_SOLVER_PROFILES["scheduler_state_v5_frg_seed_qubo_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "frg_seed_enabled": True,
    "frg_seed_time_limit": 0.75,
    "frg_seed_min_max_steps": 2048,
    "frg_seed_max_steps_multiplier": 12,
    "frg_seed_max_steps_cap": 0,
    "frg_seed_respect_cap": False,
    "frg_seed_prune_to_incumbent": True,
    "frg_seed_allow_large_over_cap": True,
    "frg_seed_paper_pure": True,
}
_SOLVER_PROFILES["scheduler_state_v5_frg_seed_ledger_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_frg_seed_qubo_polish"],
    "candidate_ledger_score_enabled": True,
    "candidate_ledger_wang_fixed_weight": 0.04,
    "candidate_ledger_wang_pair_weight": 0.03,
    "candidate_ledger_frg_fill_weight": 0.08,
    "candidate_ledger_frg_reduce_weight": 0.04,
    "candidate_ledger_frg_unblocked_weight": 0.03,
    "candidate_ledger_frg_assignment_weight": 0.02,
    "candidate_ledger_rollout_bonus": 0.06,
    "candidate_ledger_rank_window": 512,
}
_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_frg_seed_qubo_polish"],
    "candidate_ledger_score_enabled": False,
    "union_diversity_qubo_enabled": True,
    "union_diversity_qubo_supplement_width": 4,
    "union_diversity_qubo_focus_n": -1,
    "union_diversity_qubo_n_window": 2,
    "union_diversity_qubo_min_candidates": 64,
    "union_diversity_qubo_max_vars": 48,
    "union_diversity_qubo_select": 4,
    "union_diversity_qubo_rank_weight": 0.02,
    "union_diversity_qubo_wang_fixed_weight": 0.02,
    "union_diversity_qubo_wang_pair_weight": 0.02,
    "union_diversity_qubo_frg_weight": 0.04,
    "union_diversity_qubo_rollout_bonus": 0.03,
    "union_diversity_qubo_same_parent_penalty": 0.55,
    "union_diversity_qubo_same_source_penalty": 0.25,
    "union_diversity_qubo_signature_penalty": 0.20,
    "union_diversity_qubo_reduce_signature_penalty": 0.15,
    "union_diversity_qubo_n_band_penalty": 0.08,
    "union_diversity_qubo_n_band_size": 1,
    "union_diversity_qubo_origin_similarity_penalty": 0.0,
    "union_diversity_qubo_same_origin_parent_n_band_penalty": 0.0,
}
_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_origin_aware_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_polish"],
    "union_diversity_qubo_origin_similarity_penalty": 0.10,
    "union_diversity_qubo_same_origin_parent_n_band_penalty": 0.35,
}
_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_rollout_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_polish"],
    "union_rollout_probe_enabled": True,
    "union_rollout_probe_max_candidates": 8,
    "union_rollout_probe_time_limit": 0.001,
    "union_rollout_probe_max_steps": 512,
    "union_rollout_probe_feasible_bonus": 0.0,
    "union_rollout_probe_final_n_weight": 0.08,
    "union_rollout_probe_moves_weight": 0.0,
    "union_rollout_probe_dead_end_penalty": 0.12,
}

_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_micro_trace_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_polish"],
    "micro_rollout_enabled": True,
    "micro_rollout_trace_only": True,
    "receiver_safety_enabled": True,
}

_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_receiver_safety_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_polish"],
    "receiver_safety_enabled": True,
}

_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_micro_only_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_polish"],
    "micro_rollout_enabled": True,
}

_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_micro_rollout_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_polish"],
    "micro_rollout_enabled": True,
    "receiver_safety_enabled": True,
}

_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_rollout_graft_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_polish"],
    "union_rollout_probe_enabled": True,
    "union_rollout_probe_max_candidates": 16,
    "union_rollout_probe_time_limit": 0.003,
    "union_rollout_probe_max_steps": 512,
    "union_rollout_probe_feasible_bonus": 0.0,
    "union_rollout_probe_final_n_weight": 0.0,
    "union_rollout_probe_moves_weight": 0.0,
    "union_rollout_probe_dead_end_penalty": 0.0,
    "union_rollout_terminal_graft": True,
}

_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_frg_frontier_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_rollout_graft_polish"],
    "frg_frontier_child_enabled": True,
    "frg_frontier_child_parent_width": 16,
    "frg_frontier_child_limit": 2,
    "frg_frontier_child_max_steps": 3,
    "frg_frontier_child_append_only": True,
}

_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_rollout_graft_polish"],
    "frg_frontier_child_enabled": True,
    "frg_frontier_child_parent_width": 16,
    "frg_frontier_child_limit": 2,
    "frg_frontier_child_max_steps": 3,
    "frg_frontier_child_append_only": False,
}

_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_frg_frontier_adaptive_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_qubo_diversity_union_rollout_graft_polish"],
    "frg_frontier_child_enabled": True,
    "frg_frontier_child_parent_width": 4,
    "frg_frontier_child_limit": 1,
    "frg_frontier_child_max_steps": 2,
    "frg_frontier_child_append_only": True,
    "frg_frontier_child_adaptive_enabled": True,
    "frg_frontier_child_adaptive_parent_width": 12,
    "frg_frontier_child_adaptive_limit": 2,
    "frg_frontier_child_adaptive_max_steps": 3,
    "frg_frontier_child_adaptive_plateau_window": 2,
    "frg_frontier_child_adaptive_plateau_delta": 0,
    "frg_frontier_child_adaptive_min_n": 8,
    "frg_frontier_child_adaptive_small_stack_max": 4,
    "frg_frontier_child_adaptive_min_container_count": 0,
}
_SOLVER_PROFILES["scheduler_state_v5_pareto_union_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "frg_seed_enabled": False,
    "frg_seed_time_limit": 0.75,
    "frg_seed_min_max_steps": 2048,
    "frg_seed_max_steps_multiplier": 12,
    "frg_seed_max_steps_cap": 0,
    "frg_seed_respect_cap": False,
    "frg_seed_prune_to_incumbent": True,
    "frg_seed_allow_large_over_cap": True,
    "frg_seed_paper_pure": True,
    "candidate_ledger_score_enabled": False,
    "union_diversity_qubo_enabled": False,
    "union_diversity_qubo_supplement_width": 4,
    "union_diversity_qubo_focus_n": -1,
    "union_diversity_qubo_n_window": 2,
    "union_diversity_qubo_min_candidates": 64,
    "union_diversity_qubo_max_vars": 48,
    "union_diversity_qubo_select": 4,
    "union_diversity_qubo_rank_weight": 0.02,
    "union_diversity_qubo_wang_fixed_weight": 0.02,
    "union_diversity_qubo_wang_pair_weight": 0.02,
    "union_diversity_qubo_frg_weight": 0.04,
    "union_diversity_qubo_rollout_bonus": 0.03,
    "union_diversity_qubo_same_parent_penalty": 0.55,
    "union_diversity_qubo_same_source_penalty": 0.25,
    "union_diversity_qubo_signature_penalty": 0.20,
    "union_diversity_qubo_reduce_signature_penalty": 0.15,
    "union_diversity_qubo_n_band_penalty": 0.08,
    "union_diversity_qubo_n_band_size": 1,
    "union_diversity_qubo_origin_similarity_penalty": 0.0,
    "union_diversity_qubo_same_origin_parent_n_band_penalty": 0.0,
    "pareto_union_enabled": True,
    "pareto_union_candidate_window": 16,
    "pareto_union_seed_width": 2,
    "pareto_union_qubo_width": 2,
    "pareto_union_max_vars": 16,
    "pareto_union_n_band_size": 2,
    "pareto_union_quality_weight": 1.00,
    "pareto_union_repair_weight": 0.20,
    "pareto_union_space_weight": 0.12,
    "pareto_union_risk_weight": 0.16,
    "pareto_union_same_parent_penalty": 0.55,
    "pareto_union_same_source_penalty": 0.25,
    "pareto_union_same_dest_penalty": 0.20,
    "pareto_union_same_source_dest_penalty": 0.35,
    "pareto_union_signature_penalty": 0.20,
    "pareto_union_reduce_signature_penalty": 0.15,
    "pareto_union_n_band_penalty": 0.08,
    "pareto_union_risk_bucket_penalty": 0.08,
}
_SOLVER_PROFILES["scheduler_state_v5_frg_compound_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_frg_dlts_polish"],
    "frg_compound_escape_enabled": True,
    "frg_compound_escape_width": 2,
    "frg_compound_escape_focus_n": -1,
    "frg_compound_escape_plateau_window": 5,
    "frg_compound_escape_plateau_delta": 0,
    "frg_compound_escape_cap_slack": 24,
    "frg_compound_escape_min_candidates": 7000,
}
_SOLVER_PROFILES["scheduler_state_v5_frg_compound_loose_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_frg_compound_polish"],
    "frg_compound_escape_width": 1,
    "frg_compound_escape_plateau_window": 2,
    "frg_compound_escape_cap_slack": 99,
    "frg_compound_escape_min_candidates": 1,
}
_SOLVER_PROFILES["scheduler_state_v5_frg_compound_guarded_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "frg_candidate_supplement_enabled": False,
    "frg_candidate_supplement_width": 0,
    "dlts_value_supplement_enabled": False,
    "dlts_value_supplement_width": 0,
    "frg_compound_escape_enabled": True,
    "frg_compound_escape_width": 1,
    "frg_compound_escape_focus_n": -1,
    "frg_compound_escape_plateau_window": 2,
    "frg_compound_escape_plateau_delta": 0,
    "frg_compound_escape_cap_slack": 99,
    "frg_compound_escape_min_candidates": 1000,
    "frg_compound_guard_enabled": True,
    "frg_compound_guard_max_stack_count": 5,
    "frg_compound_guard_max_initial_n": 18,
    "frg_compound_guard_min_best_n": 12,
    "frg_compound_guard_max_best_n": 18,
    "frg_compound_guard_min_candidates": 1000,
    "frg_compound_guard_min_parent_count": 3,
    "frg_compound_guard_max_parent_share": 0.75,
}
_SOLVER_PROFILES["scheduler_state_v5_wang2017_task_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "wang2017_task_enabled": True,
    "wang2017_task_supplement_width": 4,
    "wang2017_task_focus_n": -1,
    "wang2017_task_n_window": 2,
}
_SOLVER_PROFILES["scheduler_state_v5_wang2017_task_guarded_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "wang2017_task_enabled": True,
    "wang2017_task_supplement_width": 1,
    "wang2017_task_focus_n": -1,
    "wang2017_task_n_window": 0,
    "wang2017_task_plateau_window": 7,
    "wang2017_task_plateau_delta": 0,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_suffix_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "post_polish_window_select": 0,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_dynamic_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "survivor_parent_cap_enabled": True,
    "survivor_parent_cap_trigger_ratio": 0.70,
    "survivor_parent_cap_release_ratio": 0.55,
    "survivor_parent_cap_value": 3,
    "survivor_parent_cap_window": 2,
    "survivor_parent_cap_min_n": 12,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_risk_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "group_qubo_supplement": True,
    "max_group_qubo_calls": 48,
    "qubo_group_max_vars": 96,
    "qubo_group_select": 12,
    "qubo_group_take": 2,
    "qubo_group_lookahead_weight": 0.15,
    "qubo_group_branch_weight": 0.03,
    "group_qubo_risk_gate": True,
    "group_qubo_risk_min_candidates": 12000,
    "group_qubo_risk_parent_share": 0.18,
    "group_qubo_risk_plateau_window": 3,
    "group_qubo_risk_plateau_delta": 0,
    "wang_feature_fixed_prefix_weight": 12.0,
    "wang_feature_fix_pair_weight": 5.0,
    "wang_feature_risk_weight": 0.5,
    "wang_feature_same_parent_weight": 0.60,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_scarcity_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_risk_polish"],
    "scarcity_group_qubo_enabled": True,
    "scarcity_group_qubo_threshold": 0.95,
    "scarcity_group_qubo_plateau_window": 2,
    "scarcity_group_qubo_plateau_delta": 0,
    "scarcity_group_qubo_min_best_n_ratio": 0.70,
    "scarcity_group_qubo_min_candidates": 1000,
    "scarcity_group_qubo_min_n": 12,
}
_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_selective_window_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_wang_rescue_polish"],
    "post_polish_window_select": 3,
    "post_polish_window_min_score_ratio": 0.80,
    "post_polish_window_max_size": 10,
    "post_polish_window_repair_expanded": 8000,
}
_SOLVER_PROFILES["scheduler_state_v5_window_polish"] = {
    **_SOLVER_PROFILES["scheduler_state_v5"],
    "post_polish_mode": "qubo_windows_v2",
    "post_polish_accept": "better",
    "post_polish_passes": 2,
    "post_polish_focus_n": 8,
    "post_polish_tail_span": 16,
    "post_polish_window_min_size": 3,
    "post_polish_window_max_size": 8,
    "post_polish_window_stride": 1,
    "post_polish_window_budget": 96,
    "post_polish_window_qubo_vars": 72,
    "post_polish_window_select": 1,
    "post_polish_window_time_limit": 0.001,
    "post_polish_window_repair_expanded": 3000,
    "post_polish_window_repair_depth_extra": 0,
}
_SOLVER_PROFILES["scheduler_state_v6"] = {
    **_SOLVER_PROFILES["scheduler_state_v5_window_polish"],
    "post_polish_window_min_stacks": 7,
}
_SOLVER_PROFILES["scheduler_state_v7"] = {
    **_SOLVER_PROFILES["scheduler_state_v5"],
    "post_polish_passes": 3,
    "post_polish_tail_span": 18,
    "post_polish_max_windows": 128,
    "post_polish_max_qubo_vars": 96,
    "post_polish_max_bridge_depth": 8,
    "post_polish_bridge_expanded": 30000,
    "post_polish_suffix_focus_n": 10,
    "post_polish_suffix_tail_span": 22,
    "post_polish_suffix_max_candidates": 72,
    "post_polish_suffix_select": 7,
    "post_polish_suffix_max_qubo_vars": 64,
    "post_polish_suffix_max_depth": 20,
    "post_polish_suffix_expanded": 90000,
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
        "beam_survivor_telemetry_output": None,
        "dlts_feature_output": None,
        "frontier_trace_output": None,
        "frontier_trace_top_k": 64,
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
        "qubo_next_n_weight": 0.0,
        "qubo_penalty_a": 30.0,
        "qubo_time_limit": 0.001,
        "qubo_reads": 2,
        "qubo_iterations": 20,
        "scheduler_state_controller": "off",
        "scheduler_state_contract_pressure": 8.0,
        "scheduler_state_max_contract_pressure": 40.0,
        "scheduler_state_min_best_n_ratio": 0.0,
        "scheduler_state_progress_window": 3,
        "scheduler_state_progress_delta": 2,
        "scheduler_state_failed_contract_cooldown": 0,
        "scheduler_state_post_rescue_plateau_window": 0,
        "scheduler_state_post_rescue_plateau_delta": 0,
        "scheduler_state_post_rescue_min_ratio": 0.90,
        "scheduler_state_post_rescue_action": "protect",
        "scheduler_state_initial_plateau_window": 0,
        "scheduler_state_initial_plateau_min_ratio": 0.98,
        "scheduler_state_initial_plateau_action": "protect",
        "scheduler_state_initial_plateau_lock": False,
        "scheduler_state_initial_plateau_min_duplication": 0.0,
        "scheduler_state_initial_plateau_unique_min_n": 0,
        "scheduler_state_protect_parent_top_share": 0.65,
        "scheduler_state_protect_n_band": 2,
        "scheduler_state_top_window": 64,
        "scheduler_state_rescue_plateau_window": 3,
        "scheduler_state_rescue_plateau_delta": 0,
        "scheduler_state_rescue_cap_slack": 8,
        "adaptive_high_n": 16,
        "adaptive_low_n": 8,
        "adaptive_high_frontier": 800,
        "adaptive_mid_frontier": 1000,
        "adaptive_low_frontier": 1200,
        "adaptive_core_ratio": 0.90,
        "adaptive_candidate_pool": 0,
        "adaptive_controller": "tiered",
        "adaptive_expand_frontier": 1200,
        "adaptive_contract_pressure": 0.0,
        "adaptive_mid_pressure": 0.0,
        "adaptive_min_best_n_ratio": 0.0,
        "adaptive_max_duplication_ratio": 1.0,
        "adaptive_parent_top_share": 0.0,
        "adaptive_plateau_window": 0,
        "adaptive_plateau_delta": 0,
        "adaptive_cap_slack": 0,
        "adaptive_large_stack_min_stacks": 0,
        "adaptive_large_stack_expand_frontier": 0,
        "adaptive_large_stack_plateau_window": 0,
        "adaptive_large_stack_cap_slack": 0,
        "bucket_key": "n_depth",
        "bucket_n_band_size": 2,
        "bucket_core_ratio": 0.50,
        "bucket_qubo_share": 0.35,
        "bucket_max_qubo_vars": 96,
        "bucket_qubo_select": 24,
        "bucket_min_candidates": 64,
        "post_polish_mode": "legacy",
        "post_polish_accept": "better",
        "post_polish_window_min_stacks": 0,
        "post_polish_window_min_size": 3,
        "post_polish_window_max_size": 10,
        "post_polish_window_stride": 1,
        "post_polish_window_budget": 96,
        "post_polish_window_qubo_vars": 72,
        "post_polish_window_select": 8,
        "post_polish_window_time_limit": 0.001,
        "post_polish_window_repair_expanded": 20000,
        "post_polish_window_repair_depth_extra": 0,
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
    post_polish_window_candidates: int = 0
    post_polish_window_qubo_selected: int = 0
    post_polish_window_attempts: int = 0
    post_polish_window_successes: int = 0
    post_polish_window_moves_saved: int = 0
    post_polish_window_time_sec: float = 0.0
    tail_n_trace: str = ""
    tail_ordered_trace: str = ""
    tail_ordered_gain: int = 0
    reservoir_size: int = 0
    peak_reservoir: int = 0
    reservoir_added_total: int = 0
    reservoir_pulled_total: int = 0
    stream_pruned_candidates: int = 0
    rank_pool_truncated_candidates: int = 0
    rescue_layers_used: int = 0
    scarcity_inner_rescue_layers: int = 0
    scarcity_inner_rescue_gate_reasons: str = ""
    tail_lookahead_calls: int = 0
    tail_lookahead_selected_total: int = 0
    tail_lookahead_goal_reachable_total: int = 0
    tail_lookahead_best_delta_total: int = 0
    tail_lookahead_expanded_total: int = 0
    tail_risk_reserve_calls: int = 0
    tail_risk_reserve_selected_total: int = 0
    tail_risk_reserve_scanned_total: int = 0
    tail_risk_reserve_one_step_improved_total: int = 0
    wang_incumbent_moves: int = 0
    wang_incumbent_runtime_sec: float = 0.0
    wang_incumbent_used: bool = False
    wang_rollout_added_total: int = 0
    wang_pair_qubo_calls: int = 0
    wang_pair_qubo_vars: int = 0
    wang_pair_qubo_time_sec: float = 0.0
    unified_wang_qubo_calls: int = 0
    unified_wang_qubo_vars: int = 0
    unified_wang_qubo_time_sec: float = 0.0
    unified_wang_fixed_prefix_selected: int = 0
    unified_wang_fix_pair_selected: int = 0
    union_diversity_qubo_calls: int = 0
    union_diversity_qubo_vars: int = 0
    union_diversity_qubo_time_sec: float = 0.0
    union_diversity_qubo_selected_total: int = 0
    pareto_union_layers: int = 0
    pareto_union_seed_selected_total: int = 0
    pareto_union_qubo_calls: int = 0
    pareto_union_qubo_vars: int = 0
    pareto_union_qubo_time_sec: float = 0.0
    pareto_union_qubo_selected_total: int = 0
    pareto_union_quality_selected: int = 0
    pareto_union_space_selected: int = 0
    pareto_union_repair_selected: int = 0
    pareto_union_risk_selected: int = 0
    pareto_union_distinct_parent_count_last: int = 0
    pareto_union_distinct_source_count_last: int = 0
    pareto_union_distinct_dest_count_last: int = 0
    pareto_union_distinct_signature_count_last: int = 0
    union_rollout_probe_calls: int = 0
    union_rollout_probe_candidates: int = 0
    union_rollout_probe_feasible: int = 0
    union_rollout_probe_time_sec: float = 0.0
    union_rollout_probe_cache_hits: int = 0
    union_rollout_probe_selected_total: int = 0
    union_rollout_terminal_grafted_total: int = 0
    union_rollout_terminal_graft_used: bool = False
    micro_rollout_calls: int = 0
    micro_rollout_cache_hits: int = 0
    micro_rollout_candidates: int = 0
    micro_rollout_selected_total: int = 0
    receiver_safety_scored_total: int = 0
    frg_frontier_child_calls: int = 0
    frg_frontier_child_generated: int = 0
    frg_frontier_child_added_total: int = 0
    feasibility_escape_calls: int = 0
    feasibility_escape_added_total: int = 0
    feasibility_escape_active_layers: int = 0
    feasibility_escape_gate_reasons: str = ""
    frg_seed_moves: int = 0
    frg_seed_runtime_sec: float = 0.0
    frg_seed_used: bool = False
    frg_seed_status: str = ""
    frg_seed_generated: int = 0
    frg_seed_cap_ok: bool = False
    frg_seed_large_layout_ok: bool = False
    frg_candidate_selected_total: int = 0
    frg_compound_appended_total: int = 0
    dlts_value_selected_total: int = 0
    bs_frg_qubo_calls: int = 0
    bs_frg_qubo_vars: int = 0
    bs_frg_qubo_time_sec: float = 0.0
    bs_frg_qubo_selected_total: int = 0
    wang2017_task_calls: int = 0
    wang2017_task_added_total: int = 0
    wang2017_task_active_layers: int = 0
    wang2017_task_selected_tier_protection_total: int = 0
    wang2017_task_selected_stable_height_total: int = 0


@dataclass(frozen=True)
class FrgSeedResult:
    moves: tuple[Move, ...]
    final_n: int
    generated: int
    runtime_sec: float
    status: str
    cap_ok: bool


@dataclass(frozen=True)
class UnionRolloutProbeResult:
    feasible: bool
    final_n: int
    moves: int
    status: str
    runtime_sec: float = 0.0
    move_path: tuple[Move, ...] = ()


@dataclass(frozen=True)
class PostPolishWindow:
    start: int
    end: int
    old_len: int
    min_n: int
    max_n: int
    score: float
    plateau_count: int = 0
    increase_count: int = 0
    tail_dirty: int = 0
    oscillation_count: int = 0


@dataclass(frozen=True)
class WangQuboFeatures:
    fixed_prefix: int
    fix_pair: int
    risk: float
    same_parent: float


@dataclass(frozen=True)
class UnifiedWangQuboFeatures:
    fixed_prefix: int
    fix_pair: int
    target_gap: int
    same_parent_count: int
    rollout_preferred: int
    depth: int
    n_value: int


@dataclass(frozen=True)
class Wang2017TaskFeatures:
    source: int
    target: int
    moved_group: int
    tier_protection: int
    task_move_estimate: int
    stable_loss: int
    affected_demand: int
    target_stable_height: int
    child_n: int
    depth: int
    dead_end_risk: float


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
    window_candidates: int = 0
    window_qubo_selected: int = 0
    window_attempts: int = 0
    window_successes: int = 0
    window_moves_saved: int = 0
    window_time_sec: float = 0.0


@dataclass(frozen=True)
class InitialScarcityMetrics:
    n0: int
    d0: int
    b0: int
    l0: int
    one_safe_ratio: float
    buffer_scarcity: float
    top_pressure: float
    dirty_spread: float
    workload_hardness: float
    scarcity_risk: float


def _business_blockers_from_stacks(stacks: list[list[int]]) -> list[tuple[int, int, int]]:
    blockers: list[tuple[int, int, int]] = []
    for stack_idx, stack in enumerate(stacks):
        for level in range(1, len(stack)):
            if stack[level] > stack[level - 1]:
                blockers.extend((stack[idx], stack_idx, idx) for idx in range(level, len(stack)))
                break
    return blockers


def _safe_targets_for_box(stacks: list[list[int]], max_height: int, box: int, source_stack: int) -> list[int]:
    targets: list[int] = []
    for stack_idx, stack in enumerate(stacks):
        if stack_idx == source_stack or len(stack) >= max_height:
            continue
        if not stack or box < min(stack):
            targets.append(stack_idx)
    return targets


def _initial_scarcity_metrics(stacks: list[list[int]], max_height: int) -> InitialScarcityMetrics:
    stack_count = len(stacks)
    container_count = sum(len(stack) for stack in stacks)
    blockers = _business_blockers_from_stacks(stacks)
    n0 = len(blockers)
    if n0 == 0 or stack_count == 0:
        return InitialScarcityMetrics(
            n0=n0,
            d0=0,
            b0=0,
            l0=0,
            one_safe_ratio=0.0,
            buffer_scarcity=0.0,
            top_pressure=0.0,
            dirty_spread=0.0,
            workload_hardness=0.0,
            scarcity_risk=0.0,
        )

    dirty_stacks = {stack_idx for _box, stack_idx, _level in blockers}
    landing_sets = [
        _safe_targets_for_box(stacks, max_height, box, stack_idx)
        for box, stack_idx, _level in blockers
    ]
    l0 = sum(len(targets) for targets in landing_sets)
    one_safe_ratio = sum(1 for targets in landing_sets if len(targets) <= 1) / n0
    available_target_stacks = sum(1 for stack in stacks if len(stack) < max_height)
    buffer_capacity = n0 * max(1, available_target_stacks - 1)
    buffer_scarcity = 1.0 - (l0 / (buffer_capacity + 1e-9))

    top_blockers = 0
    top_safe_total = 0
    for stack_idx, stack in enumerate(stacks):
        if not stack:
            continue
        top_box = stack[-1]
        for box, blocker_stack, level in blockers:
            if box == top_box and blocker_stack == stack_idx and level == len(stack) - 1:
                top_blockers += 1
                top_safe_total += len(_safe_targets_for_box(stacks, max_height, top_box, stack_idx))
                break
    top_pressure = min(1.0, top_blockers / (top_safe_total + 1.0))

    b0 = sum(len(stacks[stack_idx]) - 1 - level for _box, stack_idx, level in blockers)
    d0 = len(dirty_stacks)
    dirty_spread = d0 / stack_count
    workload_hardness = (
        0.40 * (b0 / max(1.0, container_count * max(1, max_height - 1)))
        + 0.35 * (n0 / max(1, container_count))
        + 0.15 * (d0 / stack_count)
        + 0.10 * (container_count / 40.0)
    )
    scarcity_risk = (
        0.45 * one_safe_ratio
        + 0.25 * top_pressure
        + 0.20 * buffer_scarcity
        + 0.10 * dirty_spread
    )
    return InitialScarcityMetrics(
        n0=n0,
        d0=d0,
        b0=b0,
        l0=l0,
        one_safe_ratio=one_safe_ratio,
        buffer_scarcity=buffer_scarcity,
        top_pressure=top_pressure,
        dirty_spread=dirty_spread,
        workload_hardness=workload_hardness,
        scarcity_risk=scarcity_risk,
    )


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _light_route_score(metrics: InitialScarcityMetrics, container_count: int) -> float:
    if metrics.n0 <= 0 or container_count <= 0:
        return 0.0
    workload_fit = _clamp01((0.55 - metrics.workload_hardness) / 0.15)
    scarcity_fit = _clamp01((metrics.scarcity_risk - 0.85) / 0.15)
    one_safe_fit = _clamp01((metrics.one_safe_ratio - 0.75) / 0.25)
    blocker_density_fit = _clamp01(((metrics.n0 / container_count) - 0.35) / 0.25)
    return (
        0.40 * workload_fit
        + 0.30 * scarcity_fit
        + 0.20 * one_safe_fit
        + 0.10 * blocker_density_fit
    )


def _case_filter(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return [case if case.endswith("_single") else f"{case}_single" for case in args.cases]
    return [f"{zone}_{bay}_single" for zone, bay in discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))]


def _priority_duplication_ratio_for_instance(instance: Instance) -> float:
    values = [box for stack in instance.stacks for box in stack]
    if not values:
        return 0.0
    return 1.0 - (len(set(values)) / len(values))


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
        "target_guided_weight": "--target-guided-weight",
        "target_guided_focus_n": "--target-guided-focus-n",
        "target_guided_supplement_width": "--target-guided-supplement-width",
        "target_guided_n_window": "--target-guided-n-window",
        "wang_probe_supplement_width": "--wang-probe-supplement-width",
        "wang_probe_focus_n": "--wang-probe-focus-n",
        "wang_probe_n_window": "--wang-probe-n-window",
        "wang_probe_fixed_weight": "--wang-probe-fixed-weight",
        "wang_probe_pair_weight": "--wang-probe-pair-weight",
        "wang_probe_target_weight": "--wang-probe-target-weight",
        "wang_probe_plateau_window": "--wang-probe-plateau-window",
        "wang_probe_plateau_delta": "--wang-probe-plateau-delta",
        "wang_incumbent_enabled": ("--wang-incumbent-enabled", "--no-wang-incumbent-enabled"),
        "wang_incumbent_beam_width": "--wang-incumbent-beam-width",
        "wang_incumbent_candidate_limit": "--wang-incumbent-candidate-limit",
        "wang_incumbent_max_depth": "--wang-incumbent-max-depth",
        "wang_incumbent_time_limit": "--wang-incumbent-time-limit",
        "wang_cap_enabled": ("--wang-cap-enabled", "--no-wang-cap-enabled"),
        "wang_rollout_enabled": ("--wang-rollout-enabled", "--no-wang-rollout-enabled"),
        "wang_rollout_strategy": "--wang-rollout-strategy",
        "wang_rollout_inject_candidates": ("--wang-rollout-inject-candidates", "--no-wang-rollout-inject-candidates"),
        "wang_rollout_width": "--wang-rollout-width",
        "wang_rollout_focus_n": "--wang-rollout-focus-n",
        "wang_rollout_plateau_window": "--wang-rollout-plateau-window",
        "wang_rollout_plateau_delta": "--wang-rollout-plateau-delta",
        "wang_rollout_max_steps": "--wang-rollout-max-steps",
        "wang_rollout_candidate_limit": "--wang-rollout-candidate-limit",
        "wang_pair_qubo_enabled": ("--wang-pair-qubo-enabled", "--no-wang-pair-qubo-enabled"),
        "wang_pair_qubo_start_layer": "--wang-pair-qubo-start-layer",
        "wang_pair_qubo_supplement_width": "--wang-pair-qubo-supplement-width",
        "wang_pair_qubo_focus_n": "--wang-pair-qubo-focus-n",
        "wang_pair_qubo_n_window": "--wang-pair-qubo-n-window",
        "wang_pair_qubo_max_vars": "--wang-pair-qubo-max-vars",
        "wang_pair_qubo_select": "--wang-pair-qubo-select",
        "wang_pair_qubo_same_parent_penalty": "--wang-pair-qubo-same-parent-penalty",
        "unified_wang_qubo_enabled": ("--unified-wang-qubo-enabled", "--no-unified-wang-qubo-enabled"),
        "unified_wang_qubo_supplement_width": "--unified-wang-qubo-supplement-width",
        "unified_wang_qubo_focus_n": "--unified-wang-qubo-focus-n",
        "unified_wang_qubo_n_window": "--unified-wang-qubo-n-window",
        "unified_wang_qubo_max_vars": "--unified-wang-qubo-max-vars",
        "unified_wang_qubo_select": "--unified-wang-qubo-select",
        "unified_wang_n_weight": "--unified-wang-n-weight",
        "unified_wang_depth_weight": "--unified-wang-depth-weight",
        "unified_wang_fixed_prefix_weight": "--unified-wang-fixed-prefix-weight",
        "unified_wang_fix_pair_weight": "--unified-wang-fix-pair-weight",
        "unified_wang_target_gap_weight": "--unified-wang-target-gap-weight",
        "unified_wang_same_parent_penalty": "--unified-wang-same-parent-penalty",
        "unified_wang_pair_same_parent_penalty": "--unified-wang-pair-same-parent-penalty",
        "unified_wang_signature_penalty": "--unified-wang-signature-penalty",
        "unified_wang_complement_bonus": "--unified-wang-complement-bonus",
        "unified_wang_rollout_bonus": "--unified-wang-rollout-bonus",
        "feasibility_escape_enabled": (
            "--feasibility-escape-enabled",
            "--no-feasibility-escape-enabled",
        ),
        "feasibility_escape_supplement_width": "--feasibility-escape-supplement-width",
        "feasibility_escape_focus_n": "--feasibility-escape-focus-n",
        "feasibility_escape_n_window": "--feasibility-escape-n-window",
        "feasibility_escape_dead_end_threshold": "--feasibility-escape-dead-end-threshold",
        "feasibility_escape_min_temporary_slack": "--feasibility-escape-min-temporary-slack",
        "candidate_ledger_score_enabled": (
            "--candidate-ledger-score-enabled",
            "--no-candidate-ledger-score-enabled",
        ),
        "candidate_ledger_wang_fixed_weight": "--candidate-ledger-wang-fixed-weight",
        "candidate_ledger_wang_pair_weight": "--candidate-ledger-wang-pair-weight",
        "candidate_ledger_frg_fill_weight": "--candidate-ledger-frg-fill-weight",
        "candidate_ledger_frg_reduce_weight": "--candidate-ledger-frg-reduce-weight",
        "candidate_ledger_frg_unblocked_weight": "--candidate-ledger-frg-unblocked-weight",
        "candidate_ledger_frg_assignment_weight": "--candidate-ledger-frg-assignment-weight",
        "candidate_ledger_rollout_bonus": "--candidate-ledger-rollout-bonus",
        "candidate_ledger_rank_window": "--candidate-ledger-rank-window",
        "union_rollout_probe_enabled": (
            "--union-rollout-probe-enabled",
            "--no-union-rollout-probe-enabled",
        ),
        "union_rollout_probe_max_candidates": "--union-rollout-probe-max-candidates",
        "union_rollout_probe_time_limit": "--union-rollout-probe-time-limit",
        "union_rollout_probe_max_steps": "--union-rollout-probe-max-steps",
        "union_rollout_probe_feasible_bonus": "--union-rollout-probe-feasible-bonus",
        "union_rollout_probe_final_n_weight": "--union-rollout-probe-final-n-weight",
        "union_rollout_probe_moves_weight": "--union-rollout-probe-moves-weight",
        "union_rollout_probe_dead_end_penalty": "--union-rollout-probe-dead-end-penalty",
        "union_rollout_terminal_graft": (
            "--union-rollout-terminal-graft",
            "--no-union-rollout-terminal-graft",
        ),
        "union_rollout_terminal_graft_respect_cap": (
            "--union-rollout-terminal-graft-respect-cap",
            "--no-union-rollout-terminal-graft-respect-cap",
        ),
        "micro_rollout_enabled": (
            "--micro-rollout-enabled",
            "--no-micro-rollout-enabled",
        ),
        "micro_rollout_trace_only": (
            "--micro-rollout-trace-only",
            "--no-micro-rollout-trace-only",
        ),
        "receiver_safety_enabled": (
            "--receiver-safety-enabled",
            "--no-receiver-safety-enabled",
        ),
        "micro_rollout_window": "--micro-rollout-window",
        "micro_first_child_limit": "--micro-first-child-limit",
        "micro_second_child_limit": "--micro-second-child-limit",
        "micro_cache_enabled": (
            "--micro-cache-enabled",
            "--no-micro-cache-enabled",
        ),
        "w_micro_delta_n": "--w-micro-delta-n",
        "w_micro_safe_continuation": "--w-micro-safe-continuation",
        "w_receiver_safe_delta": "--w-receiver-safe-delta",
        "w_last_buffer_consumed": "--w-last-buffer-consumed",
        "w_height_pressure": "--w-height-pressure",
        "frg_frontier_child_enabled": (
            "--frg-frontier-child-enabled",
            "--no-frg-frontier-child-enabled",
        ),
        "frg_frontier_child_parent_width": "--frg-frontier-child-parent-width",
        "frg_frontier_child_limit": "--frg-frontier-child-limit",
        "frg_frontier_child_max_steps": "--frg-frontier-child-max-steps",
        "frg_frontier_child_append_only": (
            "--frg-frontier-child-append-only",
            "--no-frg-frontier-child-append-only",
        ),
        "frg_frontier_child_adaptive_enabled": (
            "--frg-frontier-child-adaptive-enabled",
            "--no-frg-frontier-child-adaptive-enabled",
        ),
        "frg_frontier_child_adaptive_parent_width": "--frg-frontier-child-adaptive-parent-width",
        "frg_frontier_child_adaptive_limit": "--frg-frontier-child-adaptive-limit",
        "frg_frontier_child_adaptive_max_steps": "--frg-frontier-child-adaptive-max-steps",
        "frg_frontier_child_adaptive_plateau_window": "--frg-frontier-child-adaptive-plateau-window",
        "frg_frontier_child_adaptive_plateau_delta": "--frg-frontier-child-adaptive-plateau-delta",
        "frg_frontier_child_adaptive_min_n": "--frg-frontier-child-adaptive-min-n",
        "frg_frontier_child_adaptive_small_stack_max": "--frg-frontier-child-adaptive-small-stack-max",
        "frg_frontier_child_adaptive_min_container_count": "--frg-frontier-child-adaptive-min-container-count",
        "frg_seed_enabled": ("--frg-seed-enabled", "--no-frg-seed-enabled"),
        "frg_seed_time_limit": "--frg-seed-time-limit",
        "frg_seed_min_max_steps": "--frg-seed-min-max-steps",
        "frg_seed_max_steps_multiplier": "--frg-seed-max-steps-multiplier",
        "frg_seed_max_steps_cap": "--frg-seed-max-steps-cap",
        "frg_seed_respect_cap": ("--frg-seed-respect-cap", "--no-frg-seed-respect-cap"),
        "frg_seed_prune_to_incumbent": ("--frg-seed-prune-to-incumbent", "--no-frg-seed-prune-to-incumbent"),
        "frg_seed_allow_large_over_cap": (
            "--frg-seed-allow-large-over-cap",
            "--no-frg-seed-allow-large-over-cap",
        ),
        "frg_seed_paper_pure": ("--frg-seed-paper-pure", "--no-frg-seed-paper-pure"),
        "frg_candidate_supplement_enabled": (
            "--frg-candidate-supplement-enabled",
            "--no-frg-candidate-supplement-enabled",
        ),
        "frg_candidate_supplement_width": "--frg-candidate-supplement-width",
        "frg_candidate_focus_n": "--frg-candidate-focus-n",
        "frg_candidate_n_window": "--frg-candidate-n-window",
        "frg_candidate_max_ranked": "--frg-candidate-max-ranked",
        "frg_candidate_scarcity_gate_enabled": (
            "--frg-candidate-scarcity-gate-enabled",
            "--no-frg-candidate-scarcity-gate-enabled",
        ),
        "frg_linear_score_enabled": (
            "--frg-linear-score-enabled",
            "--no-frg-linear-score-enabled",
        ),
        "frg_linear_fill_weight": "--frg-linear-fill-weight",
        "frg_linear_reduce_weight": "--frg-linear-reduce-weight",
        "frg_linear_unblocked_weight": "--frg-linear-unblocked-weight",
        "frg_linear_assignment_weight": "--frg-linear-assignment-weight",
        "frg_linear_max_height_ordered_weight": "--frg-linear-max-height-ordered-weight",
        "frg_linear_height_pressure_weight": "--frg-linear-height-pressure-weight",
        "frg_rank_tiebreak_enabled": (
            "--frg-rank-tiebreak-enabled",
            "--no-frg-rank-tiebreak-enabled",
        ),
        "frg_rank_tiebreak_weight": "--frg-rank-tiebreak-weight",
        "frg_rank_tiebreak_focus_n": "--frg-rank-tiebreak-focus-n",
        "frg_scarcity_gate_enabled": (
            "--frg-scarcity-gate-enabled",
            "--no-frg-scarcity-gate-enabled",
        ),
        "frg_scarcity_gate_max_available_stacks": "--frg-scarcity-gate-max-available-stacks",
        "frg_scarcity_gate_min_avg_height_pressure": "--frg-scarcity-gate-min-avg-height-pressure",
        "frg_scarcity_gate_max_reduction_receivers": "--frg-scarcity-gate-max-reduction-receivers",
        "frg_scarcity_gate_min_unblocked_bad": "--frg-scarcity-gate-min-unblocked-bad",
        "frg_scarcity_gate_min_candidates": "--frg-scarcity-gate-min-candidates",
        "frg_scarcity_gate_cap_slack": "--frg-scarcity-gate-cap-slack",
        "frg_compound_escape_enabled": (
            "--frg-compound-escape-enabled",
            "--no-frg-compound-escape-enabled",
        ),
        "frg_compound_escape_width": "--frg-compound-escape-width",
        "frg_compound_escape_focus_n": "--frg-compound-escape-focus-n",
        "frg_compound_escape_plateau_window": "--frg-compound-escape-plateau-window",
        "frg_compound_escape_plateau_delta": "--frg-compound-escape-plateau-delta",
        "frg_compound_escape_cap_slack": "--frg-compound-escape-cap-slack",
        "frg_compound_escape_min_candidates": "--frg-compound-escape-min-candidates",
        "frg_compound_guard_enabled": (
            "--frg-compound-guard-enabled",
            "--no-frg-compound-guard-enabled",
        ),
        "frg_compound_guard_max_stack_count": "--frg-compound-guard-max-stack-count",
        "frg_compound_guard_max_initial_n": "--frg-compound-guard-max-initial-n",
        "frg_compound_guard_min_best_n": "--frg-compound-guard-min-best-n",
        "frg_compound_guard_max_best_n": "--frg-compound-guard-max-best-n",
        "frg_compound_guard_min_candidates": "--frg-compound-guard-min-candidates",
        "frg_compound_guard_min_parent_count": "--frg-compound-guard-min-parent-count",
        "frg_compound_guard_max_parent_share": "--frg-compound-guard-max-parent-share",
        "dlts_value_supplement_enabled": (
            "--dlts-value-supplement-enabled",
            "--no-dlts-value-supplement-enabled",
        ),
        "dlts_value_supplement_width": "--dlts-value-supplement-width",
        "bs_frg_qubo_enabled": (
            "--bs-frg-qubo-enabled",
            "--no-bs-frg-qubo-enabled",
        ),
        "bs_frg_qubo_supplement_width": "--bs-frg-qubo-supplement-width",
        "bs_frg_qubo_focus_n": "--bs-frg-qubo-focus-n",
        "bs_frg_qubo_n_window": "--bs-frg-qubo-n-window",
        "bs_frg_qubo_min_candidates": "--bs-frg-qubo-min-candidates",
        "bs_frg_qubo_max_vars": "--bs-frg-qubo-max-vars",
        "bs_frg_qubo_rank_weight": "--bs-frg-qubo-rank-weight",
        "bs_frg_qubo_same_parent_penalty": "--bs-frg-qubo-same-parent-penalty",
        "bs_frg_qubo_same_source_penalty": "--bs-frg-qubo-same-source-penalty",
        "bs_frg_qubo_signature_penalty": "--bs-frg-qubo-signature-penalty",
        "bs_frg_qubo_reduce_signature_penalty": "--bs-frg-qubo-reduce-signature-penalty",
        "bs_frg_qubo_n_band_penalty": "--bs-frg-qubo-n-band-penalty",
        "bs_frg_qubo_n_band_size": "--bs-frg-qubo-n-band-size",
        "wang2017_task_enabled": (
            "--wang2017-task-enabled",
            "--no-wang2017-task-enabled",
        ),
        "wang2017_task_supplement_width": "--wang2017-task-supplement-width",
        "wang2017_task_focus_n": "--wang2017-task-focus-n",
        "wang2017_task_n_window": "--wang2017-task-n-window",
        "wang2017_task_plateau_window": "--wang2017-task-plateau-window",
        "wang2017_task_plateau_delta": "--wang2017-task-plateau-delta",
        "beam_width": "--beam-width",
        "core_width": "--core-width",
        "parent_core_cap": "--parent-core-cap",
        "survivor_parent_cap_enabled": (
            "--survivor-parent-cap-enabled",
            "--no-survivor-parent-cap-enabled",
        ),
        "survivor_parent_cap_trigger_ratio": "--survivor-parent-cap-trigger-ratio",
        "survivor_parent_cap_release_ratio": "--survivor-parent-cap-release-ratio",
        "survivor_parent_cap_value": "--survivor-parent-cap-value",
        "survivor_parent_cap_window": "--survivor-parent-cap-window",
        "survivor_parent_cap_min_n": "--survivor-parent-cap-min-n",
        "selection_policy": "--selection-policy",
        "use_qubo": ("--use-qubo", "--no-use-qubo"),
        "qubo_mode": "--qubo-mode",
        "qubo_influence_profile": "--qubo-influence-profile",
        "max_qubo_calls": "--max-qubo-calls",
        "max_qubo_vars": "--max-qubo-vars",
        "qubo_core_width": "--qubo-core-width",
        "qubo_supplement_width": "--qubo-supplement-width",
        "qubo_min_candidates": "--qubo-min-candidates",
        "qubo_start_layer": "--qubo-start-layer",
        "qubo_focus_n": "--qubo-focus-n",
        "qubo_first": ("--qubo-first", "--no-qubo-first"),
        "qubo_target_share": "--qubo-target-share",
        "qubo_cohorts_per_layer": "--qubo-cohorts-per-layer",
        "qubo_cohort_stride": "--qubo-cohort-stride",
        "qubo_pair_policy": "--qubo-pair-policy",
        "qubo_parent_sim_weight": "--qubo-parent-sim-weight",
        "qubo_layout_sim_weight": "--qubo-layout-sim-weight",
        "qubo_action_sim_weight": "--qubo-action-sim-weight",
        "qubo_stack_sim_weight": "--qubo-stack-sim-weight",
        "qubo_sim_threshold": "--qubo-sim-threshold",
        "group_qubo_supplement": ("--group-qubo-supplement", "--no-group-qubo-supplement"),
        "max_group_qubo_calls": "--max-group-qubo-calls",
        "qubo_group_max_vars": "--qubo-group-max-vars",
        "qubo_group_select": "--qubo-group-select",
        "qubo_group_take": "--qubo-group-take",
        "qubo_group_lookahead_weight": "--qubo-group-lookahead-weight",
        "qubo_group_branch_weight": "--qubo-group-branch-weight",
        "qubo_rank_weight": "--qubo-rank-weight",
        "qubo_n_weight": "--qubo-n-weight",
        "qubo_depth_weight": "--qubo-depth-weight",
        "qubo_parent_weight": "--qubo-parent-weight",
        "qubo_next_n_weight": "--qubo-next-n-weight",
        "qubo_penalty_a": "--qubo-penalty-a",
        "qubo_time_limit": "--qubo-time-limit",
        "qubo_reads": "--qubo-reads",
        "qubo_iterations": "--qubo-iterations",
        "deterministic_qubo_sampler": (
            "--deterministic-qubo-sampler",
            "--no-deterministic-qubo-sampler",
        ),
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
        "stream_bucket_topk": "--stream-bucket-topk",
        "rank_pool_size": "--rank-pool-size",
        "rank_pool_min_candidates": "--rank-pool-min-candidates",
        "rank_bucket_topk": "--rank-bucket-topk",
        "rank_bucket_key": "--rank-bucket-key",
        "rank_bucket_n_band_size": "--rank-bucket-n-band-size",
        "rank_bucket_topk_high_n": "--rank-bucket-topk-high-n",
        "rank_bucket_topk_min_candidates": "--rank-bucket-topk-min-candidates",
        "scheduler_adaptive_min_stacks": "--scheduler-adaptive-min-stacks",
        "scheduler_adaptive_high_n": "--scheduler-adaptive-high-n",
        "scheduler_adaptive_cv1_only": ("--scheduler-adaptive-cv1-only", "--no-scheduler-adaptive-cv1-only"),
        "scheduler_adaptive_max_duplication_ratio": "--scheduler-adaptive-max-duplication-ratio",
        "scheduler_adaptive_min_best_n_ratio": "--scheduler-adaptive-min-best-n-ratio",
        "scheduler_adaptive_max_candidate_pressure": "--scheduler-adaptive-max-candidate-pressure",
        "scheduler_adaptive_max_parent_top_share": "--scheduler-adaptive-max-parent-top-share",
        "scheduler_adaptive_plateau_window": "--scheduler-adaptive-plateau-window",
        "scheduler_adaptive_cap_slack": "--scheduler-adaptive-cap-slack",
        "scheduler_state_controller": "--scheduler-state-controller",
        "scheduler_state_contract_pressure": "--scheduler-state-contract-pressure",
        "scheduler_state_max_contract_pressure": "--scheduler-state-max-contract-pressure",
        "scheduler_state_min_best_n_ratio": "--scheduler-state-min-best-n-ratio",
        "scheduler_state_progress_window": "--scheduler-state-progress-window",
        "scheduler_state_progress_delta": "--scheduler-state-progress-delta",
        "scheduler_state_failed_contract_cooldown": "--scheduler-state-failed-contract-cooldown",
        "scheduler_state_post_rescue_plateau_window": "--scheduler-state-post-rescue-plateau-window",
        "scheduler_state_post_rescue_plateau_delta": "--scheduler-state-post-rescue-plateau-delta",
        "scheduler_state_post_rescue_min_ratio": "--scheduler-state-post-rescue-min-ratio",
        "scheduler_state_post_rescue_action": "--scheduler-state-post-rescue-action",
        "scheduler_state_initial_plateau_window": "--scheduler-state-initial-plateau-window",
        "scheduler_state_initial_plateau_min_ratio": "--scheduler-state-initial-plateau-min-ratio",
        "scheduler_state_initial_plateau_action": "--scheduler-state-initial-plateau-action",
        "scheduler_state_initial_plateau_lock": ("--scheduler-state-initial-plateau-lock", "--no-scheduler-state-initial-plateau-lock"),
        "scheduler_state_initial_plateau_min_duplication": "--scheduler-state-initial-plateau-min-duplication",
        "scheduler_state_initial_plateau_unique_min_n": "--scheduler-state-initial-plateau-unique-min-n",
        "scheduler_state_protect_parent_top_share": "--scheduler-state-protect-parent-top-share",
        "scheduler_state_protect_n_band": "--scheduler-state-protect-n-band",
        "scheduler_state_top_window": "--scheduler-state-top-window",
        "scheduler_state_rescue_plateau_window": "--scheduler-state-rescue-plateau-window",
        "scheduler_state_rescue_plateau_delta": "--scheduler-state-rescue-plateau-delta",
        "scheduler_state_rescue_cap_slack": "--scheduler-state-rescue-cap-slack",
        "scarcity_inner_rescue_enabled": (
            "--scarcity-inner-rescue-enabled",
            "--no-scarcity-inner-rescue-enabled",
        ),
        "scarcity_inner_rescue_risk_threshold": "--scarcity-inner-rescue-risk-threshold",
        "scarcity_inner_rescue_max_l0": "--scarcity-inner-rescue-max-l0",
        "scarcity_inner_rescue_min_one_safe_ratio": "--scarcity-inner-rescue-min-one-safe-ratio",
        "scarcity_inner_rescue_min_n": "--scarcity-inner-rescue-min-n",
        "scarcity_inner_rescue_min_candidates": "--scarcity-inner-rescue-min-candidates",
        "tail_lookahead_enabled": (
            "--tail-lookahead-enabled",
            "--no-tail-lookahead-enabled",
        ),
        "tail_lookahead_core_path_only": (
            "--tail-lookahead-core-path-only",
            "--no-tail-lookahead-core-path-only",
        ),
        "tail_lookahead_max_stack_count": "--tail-lookahead-max-stack-count",
        "tail_lookahead_focus_n": "--tail-lookahead-focus-n",
        "tail_lookahead_depth": "--tail-lookahead-depth",
        "tail_lookahead_max_seed_candidates": "--tail-lookahead-max-seed-candidates",
        "tail_lookahead_supplement_width": "--tail-lookahead-supplement-width",
        "tail_lookahead_max_expanded": "--tail-lookahead-max-expanded",
        "tail_risk_reserve_enabled": (
            "--tail-risk-reserve-enabled",
            "--no-tail-risk-reserve-enabled",
        ),
        "tail_risk_focus_n": "--tail-risk-focus-n",
        "tail_risk_n_window": "--tail-risk-n-window",
        "tail_risk_reserve_width": "--tail-risk-reserve-width",
        "tail_risk_max_scan": "--tail-risk-max-scan",
        "tail_risk_min_candidates": "--tail-risk-min-candidates",
        "tail_risk_min_depth_slack": "--tail-risk-min-depth-slack",
        "tail_risk_max_depth_slack": "--tail-risk-max-depth-slack",
        "tail_risk_max_stack_count": "--tail-risk-max-stack-count",
        "scarcity_group_qubo_enabled": ("--scarcity-group-qubo-enabled", "--no-scarcity-group-qubo-enabled"),
        "scarcity_group_qubo_threshold": "--scarcity-group-qubo-threshold",
        "scarcity_group_qubo_plateau_window": "--scarcity-group-qubo-plateau-window",
        "scarcity_group_qubo_plateau_delta": "--scarcity-group-qubo-plateau-delta",
        "scarcity_group_qubo_min_best_n_ratio": "--scarcity-group-qubo-min-best-n-ratio",
        "scarcity_group_qubo_min_candidates": "--scarcity-group-qubo-min-candidates",
        "scarcity_group_qubo_min_n": "--scarcity-group-qubo-min-n",
        "post_polish_qubo": ("--post-polish-qubo", "--no-post-polish-qubo"),
        "post_polish_mode": "--post-polish-mode",
        "post_polish_accept": "--post-polish-accept",
        "post_polish_window_min_stacks": "--post-polish-window-min-stacks",
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
        "post_polish_window_min_size": "--post-polish-window-min-size",
        "post_polish_window_max_size": "--post-polish-window-max-size",
        "post_polish_window_stride": "--post-polish-window-stride",
        "post_polish_window_budget": "--post-polish-window-budget",
        "post_polish_window_qubo_vars": "--post-polish-window-qubo-vars",
        "post_polish_window_select": "--post-polish-window-select",
        "post_polish_window_time_limit": "--post-polish-window-time-limit",
        "post_polish_window_repair_expanded": "--post-polish-window-repair-expanded",
        "post_polish_window_repair_depth_extra": "--post-polish-window-repair-depth-extra",
        "post_polish_successor_cache": (
            "--post-polish-successor-cache",
            "--no-post-polish-successor-cache",
        ),
        "adaptive_controller": "--adaptive-controller",
        "adaptive_expand_frontier": "--adaptive-expand-frontier",
        "adaptive_contract_pressure": "--adaptive-contract-pressure",
        "adaptive_mid_pressure": "--adaptive-mid-pressure",
        "adaptive_min_best_n_ratio": "--adaptive-min-best-n-ratio",
        "adaptive_max_duplication_ratio": "--adaptive-max-duplication-ratio",
        "adaptive_parent_top_share": "--adaptive-parent-top-share",
        "adaptive_plateau_window": "--adaptive-plateau-window",
        "adaptive_plateau_delta": "--adaptive-plateau-delta",
        "adaptive_cap_slack": "--adaptive-cap-slack",
        "adaptive_large_stack_min_stacks": "--adaptive-large-stack-min-stacks",
        "adaptive_large_stack_expand_frontier": "--adaptive-large-stack-expand-frontier",
        "adaptive_large_stack_plateau_window": "--adaptive-large-stack-plateau-window",
        "adaptive_large_stack_cap_slack": "--adaptive-large-stack-cap-slack",
        "bucket_key": "--bucket-key",
        "bucket_n_band_size": "--bucket-n-band-size",
        "bucket_core_ratio": "--bucket-core-ratio",
        "bucket_qubo_share": "--bucket-qubo-share",
        "bucket_max_qubo_vars": "--bucket-max-qubo-vars",
        "bucket_qubo_select": "--bucket-qubo-select",
        "bucket_min_candidates": "--bucket-min-candidates",
        "rescue_enabled": ("--rescue-enabled", "--no-rescue-enabled"),
        "rescue_plateau_window": "--rescue-plateau-window",
        "rescue_plateau_delta": "--rescue-plateau-delta",
        "rescue_cap_slack": "--rescue-cap-slack",
        "rescue_min_n": "--rescue-min-n",
        "rescue_min_candidates": "--rescue-min-candidates",
        "rescue_max_layers": "--rescue-max-layers",
        "rescue_beam_width": "--rescue-beam-width",
        "rescue_core_width": "--rescue-core-width",
        "rescue_escalate_after": "--rescue-escalate-after",
        "rescue_escalate_beam_width": "--rescue-escalate-beam-width",
        "rescue_escalate_core_width": "--rescue-escalate-core-width",
        "rescue_max_qubo_vars": "--rescue-max-qubo-vars",
        "rescue_qubo_supplement_width": "--rescue-qubo-supplement-width",
        "rescue_qubo_cohorts_per_layer": "--rescue-qubo-cohorts-per-layer",
        "rescue_qubo_time_limit": "--rescue-qubo-time-limit",
        "rescue_qubo_iterations": "--rescue-qubo-iterations",
        "rescue_next_n_weight": "--rescue-next-n-weight",
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
        "stream_bucket_topk": "--stream-bucket-topk",
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


def _compact_target_layouts(instance: Instance) -> tuple[tuple[bytes, ...], ...]:
    boxes = sorted((box for stack in instance.stacks for box in stack), reverse=True)
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

    layouts: list[tuple[bytes, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for heights in height_patterns:
        normalized = tuple(sorted((height for height in heights if height > 0), reverse=True))
        padded = normalized + (0,) * (instance.stack_count - len(normalized))
        if padded in seen:
            continue
        seen.add(padded)
        cursor = 0
        chunks: list[bytes] = []
        for height in padded:
            stack_values = boxes[cursor : cursor + height]
            cursor += height
            chunks.append(bytes(stack_values + [0] * (instance.max_height - height)))
        layouts.append(tuple(chunks))
    return tuple(layouts)


def _chunk_height_raw(chunk: bytes) -> int:
    zero = chunk.find(0)
    return len(chunk) if zero < 0 else zero


def _target_prefix_len(chunk: bytes, target_chunk: bytes) -> int:
    limit = min(_chunk_height_raw(chunk), _chunk_height_raw(target_chunk))
    prefix = 0
    while prefix < limit and chunk[prefix] == target_chunk[prefix]:
        prefix += 1
    return prefix


@lru_cache(maxsize=262144)
def _compact_target_gap(code: bytes, target: tuple[bytes, ...], stack_count: int, max_height: int) -> int:
    chunks = _split(code, stack_count, max_height)
    stack_scores: list[tuple[int, int, int]] = []
    for src_idx, chunk in enumerate(chunks):
        for tgt_idx, target_chunk in enumerate(target):
            stack_scores.append((-_target_prefix_len(chunk, target_chunk), src_idx, tgt_idx))

    used_src: set[int] = set()
    used_tgt: set[int] = set()
    matched = 0
    for neg_prefix, src_idx, tgt_idx in sorted(stack_scores):
        if src_idx in used_src or tgt_idx in used_tgt:
            continue
        used_src.add(src_idx)
        used_tgt.add(tgt_idx)
        matched += -neg_prefix
    return len([value for value in code if value]) - matched


def _candidate_target_gap(code: bytes, args: argparse.Namespace) -> int:
    targets = tuple(getattr(args, "_layout_target_groups", ()) or ())
    if not targets:
        return 0
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return 0
    return min(_compact_target_gap(code, target, stack_count, max_height) for target in targets)


def _qubo_sampler_config(
    args: argparse.Namespace,
    *,
    num_reads: int,
    random_seed: int,
    time_limit_sec: float,
    sa_iterations: int,
) -> QuboSamplerConfig:
    return QuboSamplerConfig(
        num_reads=max(1, int(num_reads)),
        random_seed=int(random_seed),
        time_limit_sec=max(0.001, float(time_limit_sec)),
        sa_iterations=max(1, int(sa_iterations)),
        respect_time_limit=not bool(getattr(args, "deterministic_qubo_sampler", False)),
    )


class _WangStackFeatures(NamedTuple):
    fixed_prefix: int
    acceptor_top: int
    dirty_boxes: tuple[int, ...]


def _wang_fixed_prefix_len(chunk: bytes) -> int:
    height = _chunk_height_raw(chunk)
    if height <= 0:
        return 0
    prefix = 1
    while prefix < height and chunk[prefix] <= chunk[prefix - 1]:
        prefix += 1
    return prefix


@lru_cache(maxsize=1048576)
def _compact_wang_stack_features(chunk: bytes, max_height: int) -> _WangStackFeatures:
    height = _chunk_height_raw(chunk)
    dirty_count = _stack_n(chunk, max_height)
    fixed_prefix = _wang_fixed_prefix_len(chunk)
    acceptor_top = -1
    if dirty_count == 0 and height < max_height:
        acceptor_top = int(chunk[height - 1]) if height > 0 else 255
    dirty_boxes: tuple[int, ...] = ()
    if dirty_count > 0:
        start = height - dirty_count
        dirty_boxes = tuple(int(box) for box in chunk[start:height])
    return _WangStackFeatures(
        fixed_prefix=fixed_prefix,
        acceptor_top=acceptor_top,
        dirty_boxes=dirty_boxes,
    )


@lru_cache(maxsize=262144)
def _compact_wang_fixed_prefix_score(code: bytes, stack_count: int, max_height: int) -> int:
    if stack_count <= 0 or max_height <= 0:
        return 0
    return sum(
        _compact_wang_stack_features(chunk, max_height).fixed_prefix
        for chunk in _split(code, stack_count, max_height)
    )


@lru_cache(maxsize=262144)
def _compact_wang_fix_pair_score(code: bytes, stack_count: int, max_height: int) -> int:
    if stack_count <= 0 or max_height <= 0:
        return 0
    features = [
        _compact_wang_stack_features(chunk, max_height)
        for chunk in _split(code, stack_count, max_height)
    ]
    acceptors = [
        (stack_idx, feature.acceptor_top)
        for stack_idx, feature in enumerate(features)
        if feature.acceptor_top >= 0
    ]
    score = 0
    for origin, feature in enumerate(features):
        for box in feature.dirty_boxes:
            score += sum(1 for stack_idx, top in acceptors if stack_idx != origin and box <= top)
    return score


def _candidate_wang_probe_score(code: bytes, args: argparse.Namespace) -> tuple[float, int, int, int]:
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    fixed_prefix = _compact_wang_fixed_prefix_score(code, stack_count, max_height)
    fix_pairs = _compact_wang_fix_pair_score(code, stack_count, max_height)
    target_gap = _candidate_target_gap(code, args)
    score = (
        float(getattr(args, "wang_probe_fixed_weight", 1.0) or 0.0) * fixed_prefix
        + float(getattr(args, "wang_probe_pair_weight", 0.0) or 0.0) * fix_pairs
        - float(getattr(args, "wang_probe_target_weight", 0.0) or 0.0) * target_gap
    )
    return score, fix_pairs, fixed_prefix, target_gap


def _wang_qubo_feature_values(
    rows: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
) -> WangQuboFeatures:
    if not rows:
        return WangQuboFeatures(0, 0, 0.0, 0.0)
    best_code, _best_data = rows[0]
    _score, fix_pairs, fixed_prefix, target_gap = _candidate_wang_probe_score(best_code, args)
    parent_counts: dict[bytes, int] = {}
    for _code, data in rows:
        parent_counts[data[2]] = parent_counts.get(data[2], 0) + 1
    same_parent = float(max(parent_counts.values(), default=1) - 1)
    risk = float(target_gap)
    return WangQuboFeatures(
        fixed_prefix=int(fixed_prefix),
        fix_pair=int(fix_pairs),
        risk=risk,
        same_parent=same_parent,
    )


def _unified_wang_qubo_features(
    code: bytes,
    rec: tuple[int, int, bytes],
    entries: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
) -> UnifiedWangQuboFeatures:
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    n_value, depth, parent = rec
    fix_pair = _compact_wang_fix_pair_score(code, stack_count, max_height)
    fixed_prefix = _compact_wang_fixed_prefix_score(code, stack_count, max_height)
    target_gap = _candidate_target_gap(code, args)
    same_parent_count = sum(1 for other_code, data in entries if other_code != code and data[2] == parent)
    rollout_codes = getattr(args, "_wang_rollout_candidate_codes", set()) or set()
    return UnifiedWangQuboFeatures(
        fixed_prefix=int(fixed_prefix),
        fix_pair=int(fix_pair),
        target_gap=int(target_gap),
        same_parent_count=int(same_parent_count),
        rollout_preferred=1 if code in rollout_codes else 0,
        depth=int(depth),
        n_value=int(n_value),
    )


def _unified_wang_qubo_linear_energy(
    code: bytes,
    rec: tuple[int, int, bytes],
    entries: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
) -> float:
    features = _unified_wang_qubo_features(code, rec, entries, args)
    def _weight(name: str, default: float) -> float:
        value = getattr(args, name, None)
        return float(default if value is None else value)

    n_weight = _weight("unified_wang_n_weight", 1000.0)
    depth_weight = _weight("unified_wang_depth_weight", 1.0)
    fixed_prefix_weight = _weight("unified_wang_fixed_prefix_weight", 0.0)
    fix_pair_weight = _weight("unified_wang_fix_pair_weight", 0.0)
    target_gap_weight = _weight("unified_wang_target_gap_weight", 0.0)
    same_parent_penalty = _weight("unified_wang_same_parent_penalty", 0.0)
    rollout_bonus = _weight("unified_wang_rollout_bonus", 0.0)
    return (
        n_weight * float(features.n_value)
        + depth_weight * float(features.depth)
        - fixed_prefix_weight * float(features.fixed_prefix)
        - fix_pair_weight * float(features.fix_pair)
        + target_gap_weight * float(features.target_gap)
        + same_parent_penalty * float(features.same_parent_count)
        - rollout_bonus * float(features.rollout_preferred)
    )


def _wang_feature_adjusted_group_score(
    base_score: float,
    rows: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
) -> float:
    fixed_weight = float(getattr(args, "wang_feature_fixed_prefix_weight", 0.0) or 0.0)
    pair_weight = float(getattr(args, "wang_feature_fix_pair_weight", 0.0) or 0.0)
    risk_weight = float(getattr(args, "wang_feature_risk_weight", 0.0) or 0.0)
    parent_weight = float(getattr(args, "wang_feature_same_parent_weight", 0.0) or 0.0)
    if fixed_weight == 0.0 and pair_weight == 0.0 and risk_weight == 0.0 and parent_weight == 0.0:
        return base_score
    features = _wang_qubo_feature_values(rows, args)
    return (
        base_score
        - fixed_weight * float(features.fixed_prefix)
        - pair_weight * float(features.fix_pair)
        + risk_weight * float(features.risk)
        + parent_weight * float(features.same_parent)
    )


def _wang_probe_layer_active(args: argparse.Namespace) -> bool:
    window = max(0, int(getattr(args, "wang_probe_plateau_window", 0) or 0))
    if window <= 1:
        return True
    history = list(getattr(args, "_best_n_history", []) or [])
    if len(history) < window:
        return False
    recent = history[-window:]
    delta = max(0, int(getattr(args, "wang_probe_plateau_delta", 0) or 0))
    return max(recent) - min(recent) <= delta


def _wang_search_depth_cap(base_depth_cap: int, incumbent_moves: int | None, args: argparse.Namespace) -> int:
    if not getattr(args, "wang_cap_enabled", False):
        return base_depth_cap
    if incumbent_moves is None or incumbent_moves <= 0:
        return base_depth_cap
    return min(base_depth_cap, max(0, incumbent_moves - 1))


def _code_to_stacks(code: bytes, stack_count: int, max_height: int) -> list[list[int]]:
    return [[int(box) for box in chunk if box] for chunk in _split(code, stack_count, max_height)]


def _select_frg_candidate_supplement(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    target: int,
) -> list[bytes]:
    if not getattr(args, "frg_candidate_supplement_enabled", False):
        return []
    if target <= 0 or not ranked:
        return []
    best_n = int(ranked[0][1][0])
    focus_n = int(getattr(args, "frg_candidate_focus_n", -1))
    if focus_n >= 0 and best_n > focus_n:
        return []
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return []
    n_window = max(0, int(getattr(args, "frg_candidate_n_window", 2) or 0))
    max_ranked = max(1, int(getattr(args, "frg_candidate_max_ranked", 96) or 96))
    scored: list[tuple[float, int, int, bytes]] = []
    for code, data in ranked[:max_ranked]:
        n_value, depth, parent_code = data
        if int(n_value) > best_n + n_window:
            continue
        before = _code_to_stacks(parent_code, stack_count, max_height)
        after = _code_to_stacks(code, stack_count, max_height)
        signals = compute_frg_transition_signals(before, after, max_height)
        score = frg_transition_score(signals, n_value=int(n_value), depth=int(depth))
        scored.append((score, int(n_value), int(depth), code))
    scored.sort()
    return [code for _score, _n, _depth, code in scored[:target]]


def _select_dlts_value_supplement(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    target: int,
) -> list[bytes]:
    if not getattr(args, "dlts_value_supplement_enabled", False):
        return []
    if target <= 0 or not ranked:
        return []
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return []
    scored: list[tuple[float, int, int, bytes]] = []
    for code, data in ranked[:64]:
        n_value, depth, parent_code = data
        before = _code_to_stacks(parent_code, stack_count, max_height)
        after = _code_to_stacks(code, stack_count, max_height)
        signals = compute_frg_transition_signals(before, after, max_height)
        value_score = (
            1000.0 * float(n_value)
            + float(depth)
            - 12.0 * float(signals.fill_gain)
            - 8.0 * float(signals.reduce_gain)
            + 16.0 * float(signals.low_tier_damage)
        )
        scored.append((value_score, int(n_value), int(depth), code))
    scored.sort()
    return [code for _score, _n, _depth, code in scored[:target]]


class BsFrgQuboFeatures(NamedTuple):
    energy: float
    n_value: int
    depth: int
    parent_code: bytes
    source_stack: int
    signature: tuple[int, int, int, int, int]
    reduce_signature: tuple[int, int, int]


class UnionDiversityQuboFeatures(NamedTuple):
    energy: float
    n_value: int
    depth: int
    parent_code: bytes
    source_stack: int
    signature: tuple[int, int, int, int, int]
    reduce_signature: tuple[int, int, int]
    origin: str = "core"
    rollout_feasible: bool = False
    rollout_final_n: int = -1
    rollout_moves: int = 0
    rollout_status: str = ""
    micro_first_children_count: int = 0
    micro_second_children_count: int = 0
    micro_best_delta_n: int = 0
    micro_best_n: int = -1
    micro_has_reduce: bool = False
    micro_has_safe_continuation: bool = False
    receiver_safe_receivers_before: int = 0
    receiver_safe_receivers_after: int = 0
    receiver_safe_delta: int = 0
    receiver_free_slots_before: int = 0
    receiver_free_slots_after: int = 0
    receiver_height_pressure_after: float = 0.0
    receiver_last_buffer_consumed: bool = False
    micro_linear_adjustment: float = 0.0


@dataclass(frozen=True)
class MicroRolloutProbeResult:
    first_children_count: int
    second_children_count: int
    best_n: int
    best_delta_n: int
    has_reduce: bool
    has_safe_continuation: bool


@dataclass(frozen=True)
class ReceiverSafetyScore:
    safe_receivers_before: int
    safe_receivers_after: int
    safe_receiver_delta: int
    free_slots_before: int
    free_slots_after: int
    height_pressure_after: float
    last_buffer_consumed: bool


def _micro_rollout_successors(
    code: bytes,
    current_n: int,
    args: argparse.Namespace,
) -> tuple[tuple[bytes, int], ...]:
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return ()
    if bool(getattr(args, "micro_cache_enabled", True)):
        return tuple(_successors_with_n_cached(code, int(current_n), stack_count, max_height))
    return tuple(_successors_with_n_uncached(code, int(current_n), stack_count, max_height))


def _micro_rollout_probe(
    code: bytes,
    current_n: int,
    args: argparse.Namespace,
    *,
    successor_fn=None,
) -> MicroRolloutProbeResult:
    current_n = int(current_n)
    cache_key = (code, current_n)
    if bool(getattr(args, "micro_cache_enabled", True)):
        cache = getattr(args, "_micro_rollout_cache", None)
        if cache is None:
            cache = {}
            args._micro_rollout_cache = cache
        cached = cache.get(cache_key)
        if cached is not None:
            if hasattr(args, "_micro_rollout_cache_hits"):
                args._micro_rollout_cache_hits += 1
            return cached
    else:
        cache = None

    if successor_fn is None:
        successor_fn = _micro_rollout_successors
    if hasattr(args, "_micro_rollout_calls"):
        args._micro_rollout_calls += 1

    first_limit = max(0, int(getattr(args, "micro_first_child_limit", 8) or 0))
    second_limit = max(0, int(getattr(args, "micro_second_child_limit", 8) or 0))
    best_n = current_n
    first_children = list(successor_fn(code, current_n, args))[:first_limit]
    second_children_count = 0
    has_safe_continuation = False
    for child_code, child_n in first_children:
        child_n = int(child_n)
        best_n = min(best_n, child_n)
        if child_n < current_n:
            has_safe_continuation = True
        if second_limit <= 0:
            continue
        second_children = list(successor_fn(child_code, child_n, args))[:second_limit]
        second_children_count += len(second_children)
        for _grandchild_code, grandchild_n in second_children:
            grandchild_n = int(grandchild_n)
            best_n = min(best_n, grandchild_n)
            if grandchild_n < child_n or grandchild_n < current_n:
                has_safe_continuation = True

    result = MicroRolloutProbeResult(
        first_children_count=len(first_children),
        second_children_count=second_children_count,
        best_n=int(best_n),
        best_delta_n=int(best_n) - current_n,
        has_reduce=int(best_n) < current_n,
        has_safe_continuation=bool(has_safe_continuation),
    )
    if cache is not None:
        cache[cache_key] = result
    return result


def _stack_is_ordered_receiver(chunk: bytes, max_height: int) -> bool:
    height = _cached_height(chunk, max_height)
    if height >= max_height:
        return False
    if _cached_stack_n(chunk, max_height) > 0:
        return False
    return True


def _safe_receiver_count(code: bytes, stack_count: int, max_height: int) -> int:
    chunks = _split(code, stack_count, max_height)
    receiver_tops: list[int] = []
    empty_receivers = 0
    for chunk in chunks:
        if not _stack_is_ordered_receiver(chunk, max_height):
            continue
        height = _cached_height(chunk, max_height)
        if height == 0:
            empty_receivers += 1
        else:
            receiver_tops.append(int(chunk[height - 1]))
    if not receiver_tops:
        return empty_receivers
    top_threshold = max(receiver_tops) - 1
    flexible_receivers = sum(1 for top in receiver_tops if top >= top_threshold)
    return int(empty_receivers + flexible_receivers)


def _free_slot_count(code: bytes, stack_count: int, max_height: int) -> int:
    return int(sum(max_height - _cached_height(chunk, max_height) for chunk in _split(code, stack_count, max_height)))


def _height_pressure(code: bytes, stack_count: int, max_height: int) -> float:
    if stack_count <= 0 or max_height <= 0:
        return 0.0
    occupied = sum(_cached_height(chunk, max_height) for chunk in _split(code, stack_count, max_height))
    return float(occupied) / float(stack_count * max_height)


def _empty_buffer_low_receiver_move(
    parent_code: bytes,
    child_code: bytes,
    stack_count: int,
    max_height: int,
) -> bool:
    parent_chunks = tuple(_split(parent_code, stack_count, max_height))
    child_chunks = tuple(_split(child_code, stack_count, max_height))
    child_tops = [
        int(chunk[_cached_height(chunk, max_height) - 1])
        for chunk in child_chunks
        if _stack_is_ordered_receiver(chunk, max_height) and _cached_height(chunk, max_height) > 0
    ]
    if not child_tops:
        return False
    low_threshold = max(child_tops) - 1
    for src_idx, src_chunk in enumerate(parent_chunks):
        src_h = _cached_height(src_chunk, max_height)
        if src_h <= 0:
            continue
        moved = int(src_chunk[src_h - 1])
        src_after = _cached_pop_chunk(src_chunk, src_h, max_height)
        for dst_idx, dst_chunk in enumerate(parent_chunks):
            if dst_idx == src_idx:
                continue
            dst_h = _cached_height(dst_chunk, max_height)
            if dst_h != 0:
                continue
            dst_after = _cached_push_chunk(dst_chunk, dst_h, moved, max_height)
            next_chunks = list(parent_chunks)
            next_chunks[src_idx] = src_after
            next_chunks[dst_idx] = dst_after
            if tuple(sorted(next_chunks)) == child_chunks and moved < low_threshold:
                return True
    return False


def _receiver_safety_score(parent_code: bytes, child_code: bytes, args: argparse.Namespace) -> ReceiverSafetyScore:
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return ReceiverSafetyScore(0, 0, 0, 0, 0, 0.0, False)
    before = _safe_receiver_count(parent_code, stack_count, max_height)
    after = _safe_receiver_count(child_code, stack_count, max_height)
    empty_buffer_consumed_by_low_receiver = _empty_buffer_low_receiver_move(
        parent_code,
        child_code,
        stack_count,
        max_height,
    )
    if empty_buffer_consumed_by_low_receiver:
        after = max(0, after - 1)
    free_before = _free_slot_count(parent_code, stack_count, max_height)
    free_after = _free_slot_count(child_code, stack_count, max_height)
    delta = int(after) - int(before)
    return ReceiverSafetyScore(
        safe_receivers_before=int(before),
        safe_receivers_after=int(after),
        safe_receiver_delta=delta,
        free_slots_before=free_before,
        free_slots_after=free_after,
        height_pressure_after=_height_pressure(child_code, stack_count, max_height),
        last_buffer_consumed=bool(empty_buffer_consumed_by_low_receiver or (before > 1 and after <= 1 and delta < 0)),
    )


def _micro_rollout_linear_adjustment(
    micro: MicroRolloutProbeResult | None,
    receiver: ReceiverSafetyScore | None,
    args: argparse.Namespace,
) -> float:
    if bool(getattr(args, "micro_rollout_trace_only", False)):
        return 0.0
    adjustment = 0.0
    if micro is not None and bool(getattr(args, "micro_rollout_enabled", False)):
        improvement = max(0, -int(micro.best_delta_n))
        adjustment -= float(getattr(args, "w_micro_delta_n", 0.05) or 0.0) * float(improvement)
        if bool(micro.has_safe_continuation):
            adjustment -= float(getattr(args, "w_micro_safe_continuation", 0.03) or 0.0)
    if receiver is not None and bool(getattr(args, "receiver_safety_enabled", False)):
        adjustment -= float(getattr(args, "w_receiver_safe_delta", 0.03) or 0.0) * float(receiver.safe_receiver_delta)
        if bool(receiver.last_buffer_consumed):
            adjustment += float(getattr(args, "w_last_buffer_consumed", 0.05) or 0.0)
        adjustment += float(getattr(args, "w_height_pressure", 0.02) or 0.0) * float(receiver.height_pressure_after)
    return float(adjustment)


def _union_rollout_probe_sample(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
) -> list[tuple[bytes, tuple[int, int, bytes]]]:
    limit = max(0, int(getattr(args, "union_rollout_probe_max_candidates", 16) or 0))
    if limit <= 0 or not ranked:
        return []
    if len(ranked) <= limit:
        return list(ranked)
    top_count = max(1, min(len(ranked), limit // 2))
    sampled = list(ranked[:top_count])
    remaining = ranked[top_count:]
    slots = limit - len(sampled)
    if slots <= 0 or not remaining:
        return sampled[:limit]
    step = max(1, math.ceil(len(remaining) / slots))
    for row in remaining[::step]:
        sampled.append(row)
        if len(sampled) >= limit:
            break
    return sampled[:limit]


def _union_rollout_probe_energy_adjustment(
    result: UnionRolloutProbeResult | None,
    args: argparse.Namespace,
    *,
    n_value: int,
) -> float:
    if result is None:
        return 0.0
    adjustment = 0.0
    if result.feasible or int(result.final_n) <= 0:
        adjustment -= float(getattr(args, "union_rollout_probe_feasible_bonus", 0.30) or 0.0)
    else:
        denom = max(1.0, float(max(1, int(n_value))))
        adjustment += float(getattr(args, "union_rollout_probe_final_n_weight", 0.08) or 0.0) * (
            float(max(0, int(result.final_n))) / denom
        )
    max_steps = max(1, int(getattr(args, "union_rollout_probe_max_steps", 512) or 512))
    adjustment += float(getattr(args, "union_rollout_probe_moves_weight", 0.002) or 0.0) * (
        float(max(0, int(result.moves))) / float(max_steps)
    )
    if str(result.status) in {"FRG_CYCLE_OR_DEAD_END", "NO_SOLUTION", "ERROR"}:
        adjustment += float(getattr(args, "union_rollout_probe_dead_end_penalty", 0.10) or 0.0)
    return float(adjustment)


def _union_rollout_probe_completion(
    code: bytes,
    data: tuple[int, int, bytes],
    args: argparse.Namespace,
) -> UnionRolloutProbeResult:
    start = time.perf_counter()
    n_value, depth, _parent_code = data
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return UnionRolloutProbeResult(False, int(n_value), 0, "NO_LAYOUT", 0.0)

    stacks = _code_to_stacks(code, stack_count, max_height)
    group_count = max((int(box) for stack in stacks for box in stack), default=0)
    instance = Instance(
        "union_rollout_probe",
        ProblemVariant.PREMARSHALLING_CPMP,
        stack_count,
        max_height,
        group_count,
        stacks,
    )
    layout = _bs_frg_initial_layout(instance)
    max_steps = max(1, int(getattr(args, "union_rollout_probe_max_steps", 512) or 512))
    time_limit = max(0.0, float(getattr(args, "union_rollout_probe_time_limit", 0.003) or 0.0))
    deadline = start + time_limit if time_limit > 0.0 else start
    cache = getattr(args, "_union_rollout_probe_frg_cache", None)
    if cache is None:
        cache = {}
        args._union_rollout_probe_frg_cache = cache
    rollout = _bs_frg_completion_cached(
        layout,
        max_height,
        start_step=int(depth),
        max_steps=max_steps,
        deadline=deadline,
        cache=cache,
        paper_pure=bool(getattr(args, "frg_seed_paper_pure", True)),
    )
    runtime_sec = time.perf_counter() - start
    return UnionRolloutProbeResult(
        feasible=bool(rollout.feasible),
        final_n=int(rollout.final_n),
        moves=len(rollout.moves),
        status=str(rollout.status),
        runtime_sec=runtime_sec,
        move_path=tuple(rollout.moves),
    )


def _canonical_to_physical_stack_map(
    logical_stacks: list[list[int]],
    physical_stacks: list[list[int]],
) -> dict[int, int] | None:
    unused = set(range(len(physical_stacks)))
    stack_map: dict[int, int] = {}
    for logical_idx, logical_stack in enumerate(logical_stacks):
        matched: int | None = None
        for physical_idx in sorted(unused):
            if physical_stacks[physical_idx] == logical_stack:
                matched = physical_idx
                break
        if matched is None:
            return None
        stack_map[logical_idx] = matched
        unused.remove(matched)
    return stack_map


def _translate_moves_by_stack_map(moves: tuple[Move, ...], stack_map: dict[int, int]) -> list[Move] | None:
    translated: list[Move] = []
    for move in moves:
        if move.from_stack not in stack_map or move.to_stack not in stack_map:
            return None
        translated.append(
            Move(
                step=move.step,
                box=move.box,
                from_stack=stack_map[move.from_stack],
                to_stack=stack_map[move.to_stack],
                from_level=move.from_level,
                to_level=move.to_level,
            )
        )
    return translated


def _union_rollout_probe_grafted_moves(
    code: bytes,
    data: tuple[int, int, bytes],
    parent: dict[bytes, bytes | None],
    instance: Instance,
    args: argparse.Namespace,
    rollout: UnionRolloutProbeResult,
    *,
    cap: int,
) -> list[Move] | None:
    if not getattr(args, "union_rollout_terminal_graft", False):
        return None
    if not rollout.feasible or int(rollout.final_n) != 0 or not rollout.move_path:
        return None
    _n_value, _depth, parent_code = data
    try:
        prefix_codes = _backtrack(parent, parent_code)
    except (KeyError, RuntimeError):
        return None
    prefix_codes.append(code)
    try:
        prefix_moves = _codes_to_moves_instance(instance, prefix_codes)
    except RuntimeError:
        return None
    state = State(stacks=[list(stack) for stack in instance.stacks])
    try:
        for move in prefix_moves:
            state = apply_move(instance, state, move)
    except Exception:
        return None
    stack_map = _canonical_to_physical_stack_map(
        _code_to_stacks(code, instance.stack_count, instance.max_height),
        state.stacks,
    )
    if stack_map is None:
        return None
    tail_moves = _translate_moves_by_stack_map(rollout.move_path, stack_map)
    if tail_moves is None:
        return None
    grafted = prefix_moves + tail_moves
    validation_cap = int(cap)
    if not getattr(args, "union_rollout_terminal_graft_respect_cap", True):
        validation_cap = 1_000_000_000
    if not _moves_reach_n0_with_cap(instance, grafted, validation_cap):
        return None
    return grafted


def _union_rollout_probe_results(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    *,
    completion_fn=None,
) -> dict[bytes, UnionRolloutProbeResult]:
    if not getattr(args, "union_rollout_probe_enabled", False):
        args._union_rollout_probe_current_results = {}
        return {}
    sampled = _union_rollout_probe_sample(ranked, args)
    if not sampled:
        args._union_rollout_probe_current_results = {}
        return {}
    if completion_fn is None:
        completion_fn = _union_rollout_probe_completion
    cache = getattr(args, "_union_rollout_probe_cache", None)
    if cache is None:
        cache = {}
        args._union_rollout_probe_cache = cache
    if not hasattr(args, "_union_rollout_probe_calls"):
        args._union_rollout_probe_calls = 0
    if not hasattr(args, "_union_rollout_probe_candidates"):
        args._union_rollout_probe_candidates = 0
    if not hasattr(args, "_union_rollout_probe_feasible"):
        args._union_rollout_probe_feasible = 0
    if not hasattr(args, "_union_rollout_probe_time"):
        args._union_rollout_probe_time = 0.0
    if not hasattr(args, "_union_rollout_probe_cache_hits"):
        args._union_rollout_probe_cache_hits = 0

    results: dict[bytes, UnionRolloutProbeResult] = {}
    for code, data in sampled:
        args._union_rollout_probe_candidates += 1
        cached = cache.get(code)
        if cached is not None:
            args._union_rollout_probe_cache_hits += 1
            results[code] = cached
            continue
        result = completion_fn(code, data, args)
        cache[code] = result
        results[code] = result
        args._union_rollout_probe_calls += 1
        args._union_rollout_probe_time += float(getattr(result, "runtime_sec", 0.0) or 0.0)
        if bool(getattr(result, "feasible", False)):
            args._union_rollout_probe_feasible += 1
    args._union_rollout_probe_current_results = results
    return results


def _ensure_frg_frontier_child_counters(args: argparse.Namespace) -> None:
    if not hasattr(args, "_frg_frontier_child_calls"):
        args._frg_frontier_child_calls = 0
    if not hasattr(args, "_frg_frontier_child_generated"):
        args._frg_frontier_child_generated = 0
    if not hasattr(args, "_frg_frontier_child_added_total"):
        args._frg_frontier_child_added_total = 0


def _frg_frontier_child_effective_params(args: argparse.Namespace) -> tuple[int, int, int]:
    parent_width = max(0, int(getattr(args, "frg_frontier_child_parent_width", 0) or 0))
    child_limit = max(0, int(getattr(args, "frg_frontier_child_limit", 0) or 0))
    max_steps = max(1, int(getattr(args, "frg_frontier_child_max_steps", 1) or 1))
    if not getattr(args, "frg_frontier_child_adaptive_enabled", False):
        return parent_width, child_limit, max_steps

    adaptive_parent_width = max(
        parent_width,
        int(getattr(args, "frg_frontier_child_adaptive_parent_width", 0) or 0),
    )
    adaptive_child_limit = max(
        child_limit,
        int(getattr(args, "frg_frontier_child_adaptive_limit", 0) or 0),
    )
    adaptive_max_steps = max(
        max_steps,
        int(getattr(args, "frg_frontier_child_adaptive_max_steps", 1) or 1),
    )
    small_stack_max = max(0, int(getattr(args, "frg_frontier_child_adaptive_small_stack_max", 0) or 0))
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    if small_stack_max > 0 and 0 < stack_count <= small_stack_max:
        return adaptive_parent_width, adaptive_child_limit, adaptive_max_steps

    min_container_count = max(
        0,
        int(getattr(args, "frg_frontier_child_adaptive_min_container_count", 0) or 0),
    )
    container_count = int(getattr(args, "_container_count", 0) or 0)
    if min_container_count > 0 and container_count < min_container_count:
        return parent_width, child_limit, max_steps

    history = [int(value) for value in getattr(args, "_best_n_history", []) if value is not None]
    window = max(1, int(getattr(args, "frg_frontier_child_adaptive_plateau_window", 0) or 0))
    min_n = max(0, int(getattr(args, "frg_frontier_child_adaptive_min_n", 0) or 0))
    plateau_delta = max(0, int(getattr(args, "frg_frontier_child_adaptive_plateau_delta", 0) or 0))
    if len(history) < window:
        return parent_width, child_limit, max_steps
    recent = history[-window:]
    current_n = int(recent[-1])
    if current_n < min_n:
        return parent_width, child_limit, max_steps
    progress = int(recent[0]) - current_n
    if progress > plateau_delta:
        return parent_width, child_limit, max_steps
    return adaptive_parent_width, adaptive_child_limit, adaptive_max_steps


def _frg_frontier_child_rollout_children(
    layout,
    max_height: int,
    step: int,
    args: argparse.Namespace,
    *,
    child_limit: int,
    max_steps: int,
) -> list[tuple[object, tuple[Move, ...], tuple[object, ...]]]:
    branch_limit = max(1, int(child_limit))
    max_steps = max(1, int(max_steps))
    paper_pure = bool(getattr(args, "frg_seed_paper_pure", True))
    nodes: list[tuple[object, tuple[Move, ...], tuple[object, ...]]] = [(layout, (), ())]
    outputs: list[tuple[object, tuple[Move, ...], tuple[object, ...]]] = []
    seen_layouts = {getattr(layout, "stacks", ())}
    while nodes:
        next_nodes: list[tuple[object, tuple[Move, ...], tuple[object, ...]]] = []
        for node_layout, prefix_moves, prefix_layouts in nodes:
            if len(prefix_moves) >= max_steps:
                continue
            args._frg_frontier_child_calls += 1
            children = _bs_frg_step_children(
                node_layout,
                max_height,
                int(step) + len(prefix_moves),
                limit=branch_limit,
                paper_pure=paper_pure,
            )
            args._frg_frontier_child_generated += len(children)
            for child in children:
                child_moves = tuple(getattr(child, "moves", ()) or ())
                if not child_moves:
                    continue
                combined_moves = prefix_moves + child_moves
                child_layout = child.layout
                combined_layouts = prefix_layouts + (child_layout,)
                outputs.append((child_layout, combined_moves, combined_layouts))
                layout_key = getattr(child_layout, "stacks", ())
                if len(combined_moves) < max_steps and layout_key not in seen_layouts:
                    seen_layouts.add(layout_key)
                    next_nodes.append((child_layout, combined_moves, combined_layouts))
        nodes = next_nodes
    return outputs


def _parent_chain_would_cycle(
    parent_chain: dict[bytes, bytes | None] | None,
    child_code: bytes,
    parent_code: bytes,
) -> bool:
    if child_code == parent_code:
        return True
    if parent_chain is None:
        return False
    seen: set[bytes] = set()
    current: bytes | None = parent_code
    while current is not None:
        if current == child_code:
            return True
        if current in seen:
            return True
        seen.add(current)
        current = parent_chain.get(current)
    return False


def _inject_frg_frontier_child_candidates(
    frontier: list[tuple[bytes, int, int]],
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
    depth_cap: int,
    best_depth: dict[bytes, int] | None = None,
    parent_chain: dict[bytes, bytes | None] | None = None,
) -> set[bytes]:
    if not getattr(args, "frg_frontier_child_enabled", False):
        return set()
    if not frontier:
        return set()
    parent_width, child_limit, max_steps = _frg_frontier_child_effective_params(args)
    if parent_width <= 0 or child_limit <= 0:
        return set()
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return set()

    _ensure_frg_frontier_child_counters(args)
    added: set[bytes] = set()
    origins = sorted(frontier, key=lambda item: (item[1], item[2], item[0]))[:parent_width]
    for code, _n_value, depth in origins:
        if depth >= depth_cap:
            continue
        stacks = _code_to_stacks(code, stack_count, max_height)
        group_count = max((int(box) for stack in stacks for box in stack), default=0)
        instance = Instance(
            "frg_frontier_child",
            ProblemVariant.PREMARSHALLING_CPMP,
            stack_count,
            max_height,
            group_count,
            stacks,
        )
        layout = _bs_frg_initial_layout(instance)
        children = _frg_frontier_child_rollout_children(
            layout,
            max_height,
            int(depth),
            args,
            child_limit=child_limit,
            max_steps=max_steps,
        )
        for child_layout, moves, layout_path in children:
            if not moves:
                continue
            child_depth = int(depth) + len(moves)
            previous_code = code
            previous_depth = int(depth)
            for prefix_layout in layout_path[:-1]:
                prefix_stacks = [list(stack) for stack in getattr(prefix_layout, "stacks", ())]
                if len(prefix_stacks) != stack_count:
                    previous_code = b""
                    break
                prefix_code, _prefix_n = _encode(prefix_stacks, stack_count, max_height)
                previous_depth += 1
                if parent_chain is not None and prefix_code not in parent_chain:
                    if _parent_chain_would_cycle(parent_chain, prefix_code, previous_code):
                        previous_code = b""
                        break
                    parent_chain[prefix_code] = previous_code
                previous_code = prefix_code
            if not previous_code:
                continue
            child_stacks = [list(stack) for stack in getattr(child_layout, "stacks", ())]
            if len(child_stacks) != stack_count:
                continue
            child_code, child_n = _encode(child_stacks, stack_count, max_height)
            if child_code == code:
                continue
            if _parent_chain_would_cycle(parent_chain, child_code, previous_code):
                continue
            if child_depth + int(child_n) > depth_cap:
                continue
            old_best_depth = best_depth.get(child_code) if best_depth is not None else None
            if old_best_depth is not None and child_depth >= old_best_depth:
                continue
            old = candidates.get(child_code)
            if old is not None and old[1] <= child_depth:
                continue
            candidates[child_code] = (int(child_n), child_depth, previous_code)
            added.add(child_code)
    args._frg_frontier_child_added_total += len(added)
    return added


def _append_frg_frontier_child_frontier(
    selected_codes: list[bytes],
    frontier: list[tuple[bytes, int, int]],
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
    depth_cap: int,
    best_depth: dict[bytes, int] | None = None,
    parent_chain: dict[bytes, bytes | None] | None = None,
) -> set[bytes]:
    if not getattr(args, "frg_frontier_child_append_only", False):
        return set()
    before_selected = set(selected_codes)
    added = _inject_frg_frontier_child_candidates(
        frontier,
        candidates,
        args,
        depth_cap,
        best_depth,
        parent_chain=parent_chain,
    )
    appended: set[bytes] = set()
    for code in sorted(added, key=lambda item: _candidate_sort_key((item, candidates[item]), args)):
        if code in before_selected or code not in candidates:
            continue
        selected_codes.append(code)
        appended.add(code)
    return appended


def _sign_bucket(value: int) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def _transition_source_stack(parent_code: bytes, child_code: bytes, stack_count: int, max_height: int) -> int:
    parent_chunks = _split(parent_code, stack_count, max_height)
    child_chunks = _split(child_code, stack_count, max_height)
    for idx, (before, after) in enumerate(zip(parent_chunks, child_chunks)):
        if _chunk_height_raw(after) < _chunk_height_raw(before):
            return idx
    return -1


def _bs_frg_qubo_features(
    code: bytes,
    data: tuple[int, int, bytes],
    rank: int,
    args: argparse.Namespace,
) -> BsFrgQuboFeatures:
    n_value, depth, parent_code = data
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        frg_energy = float(n_value) * 1000.0 + float(depth)
        source_stack = -1
        signature = (int(n_value), 0, 0, 0, 0)
        reduce_signature = (0, 0, 0)
    else:
        before = _code_to_stacks(parent_code, stack_count, max_height)
        after = _code_to_stacks(code, stack_count, max_height)
        signals = compute_frg_transition_signals(before, after, max_height)
        frg_energy = frg_transition_score(signals, n_value=int(n_value), depth=int(depth))
        source_stack = _transition_source_stack(parent_code, code, stack_count, max_height)
        signature = (
            min(3, max(0, int(signals.fill_gain))),
            min(3, max(0, int(signals.reduce_gain))),
            _sign_bucket(int(signals.unblocked_bad_delta)),
            _sign_bucket(int(signals.stable_prefix_delta)),
            min(3, max(0, int(signals.low_tier_damage))),
        )
        reduce_signature = (
            source_stack,
            min(3, max(0, int(signals.reduce_gain))),
            _sign_bucket(int(signals.unblocked_bad_delta)),
        )
    rank_weight = float(getattr(args, "bs_frg_qubo_rank_weight", 0.05) or 0.0)
    energy = frg_energy + rank_weight * float(rank)
    return BsFrgQuboFeatures(
        energy=float(energy),
        n_value=int(n_value),
        depth=int(depth),
        parent_code=parent_code,
        source_stack=source_stack,
        signature=signature,
        reduce_signature=reduce_signature,
    )


def _union_diversity_qubo_features(
    code: bytes,
    data: tuple[int, int, bytes],
    rank: int,
    args: argparse.Namespace,
    rollout: UnionRolloutProbeResult | None = None,
    *,
    origin: str = "core",
) -> UnionDiversityQuboFeatures:
    n_value, depth, parent_code = data
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    energy = float(int(n_value) * 1000 + int(depth))
    energy += float(getattr(args, "union_diversity_qubo_rank_weight", 0.02) or 0.0) * float(rank)
    source_stack = -1
    signature = (int(n_value), 0, 0, 0, 0)
    reduce_signature = (-1, 0, 0)
    if stack_count > 0 and max_height > 0:
        fixed_weight = float(getattr(args, "union_diversity_qubo_wang_fixed_weight", 0.0) or 0.0)
        pair_weight = float(getattr(args, "union_diversity_qubo_wang_pair_weight", 0.0) or 0.0)
        if fixed_weight:
            energy -= fixed_weight * float(_compact_wang_fixed_prefix_score(code, stack_count, max_height))
        if pair_weight:
            energy -= pair_weight * float(_compact_wang_fix_pair_score(code, stack_count, max_height))

        before = _code_to_stacks(parent_code, stack_count, max_height)
        after = _code_to_stacks(code, stack_count, max_height)
        signals = compute_frg_transition_signals(before, after, max_height)
        frg_weight = float(getattr(args, "union_diversity_qubo_frg_weight", 0.0) or 0.0)
        if frg_weight:
            base = float(int(n_value) * 1000 + int(depth))
            energy += frg_weight * float(frg_transition_score(signals, n_value=int(n_value), depth=int(depth)) - base)
        source_stack = _transition_source_stack(parent_code, code, stack_count, max_height)
        signature = (
            min(3, max(0, int(signals.fill_gain))),
            min(3, max(0, int(signals.reduce_gain))),
            _sign_bucket(int(signals.unblocked_bad_delta)),
            _sign_bucket(int(signals.stable_prefix_delta)),
            min(3, max(0, int(signals.low_tier_damage))),
        )
        reduce_signature = (
            source_stack,
            min(3, max(0, int(signals.reduce_gain))),
            _sign_bucket(int(signals.unblocked_bad_delta)),
        )
    rollout_bonus = float(getattr(args, "union_diversity_qubo_rollout_bonus", 0.0) or 0.0)
    if rollout_bonus and code in (getattr(args, "_wang_rollout_candidate_codes", set()) or set()):
        energy -= rollout_bonus
    if rollout is not None:
        energy += _union_rollout_probe_energy_adjustment(rollout, args, n_value=int(n_value))
    micro: MicroRolloutProbeResult | None = None
    receiver: ReceiverSafetyScore | None = None
    micro_window = max(0, int(getattr(args, "micro_rollout_window", 32) or 0))
    micro_active = bool(getattr(args, "micro_rollout_enabled", False))
    receiver_active = bool(getattr(args, "receiver_safety_enabled", False))
    trace_only = bool(getattr(args, "micro_rollout_trace_only", False))
    if stack_count > 0 and max_height > 0 and rank < micro_window and (micro_active or trace_only):
        if hasattr(args, "_micro_rollout_candidates"):
            args._micro_rollout_candidates += 1
        micro = _micro_rollout_probe(code, int(n_value), args)
    if stack_count > 0 and max_height > 0 and rank < micro_window and (receiver_active or trace_only):
        if hasattr(args, "_receiver_safety_scored_total"):
            args._receiver_safety_scored_total += 1
        receiver = _receiver_safety_score(parent_code, code, args)
    micro_linear_adjustment = 0.0
    if micro is not None or receiver is not None:
        micro_linear_adjustment = _micro_rollout_linear_adjustment(micro, receiver, args)
        energy += micro_linear_adjustment
    features = UnionDiversityQuboFeatures(
        energy=float(energy),
        n_value=int(n_value),
        depth=int(depth),
        parent_code=parent_code,
        source_stack=source_stack,
        signature=signature,
        reduce_signature=reduce_signature,
        origin=str(origin or "core"),
        rollout_feasible=bool(rollout.feasible) if rollout is not None else False,
        rollout_final_n=int(rollout.final_n) if rollout is not None else -1,
        rollout_moves=int(rollout.moves) if rollout is not None else 0,
        rollout_status=str(rollout.status) if rollout is not None else "",
        micro_first_children_count=int(micro.first_children_count) if micro is not None else 0,
        micro_second_children_count=int(micro.second_children_count) if micro is not None else 0,
        micro_best_delta_n=int(micro.best_delta_n) if micro is not None else 0,
        micro_best_n=int(micro.best_n) if micro is not None else -1,
        micro_has_reduce=bool(micro.has_reduce) if micro is not None else False,
        micro_has_safe_continuation=bool(micro.has_safe_continuation) if micro is not None else False,
        receiver_safe_receivers_before=int(receiver.safe_receivers_before) if receiver is not None else 0,
        receiver_safe_receivers_after=int(receiver.safe_receivers_after) if receiver is not None else 0,
        receiver_safe_delta=int(receiver.safe_receiver_delta) if receiver is not None else 0,
        receiver_free_slots_before=int(receiver.free_slots_before) if receiver is not None else 0,
        receiver_free_slots_after=int(receiver.free_slots_after) if receiver is not None else 0,
        receiver_height_pressure_after=float(receiver.height_pressure_after) if receiver is not None else 0.0,
        receiver_last_buffer_consumed=bool(receiver.last_buffer_consumed) if receiver is not None else False,
        micro_linear_adjustment=float(micro_linear_adjustment),
    )
    if micro is not None or receiver is not None:
        trace_features = getattr(args, "_micro_rollout_feature_trace_by_code", None)
        if trace_features is None:
            trace_features = {}
            args._micro_rollout_feature_trace_by_code = trace_features
        trace_features[code] = features
    return features


def _union_diversity_candidate_origin(code: bytes, args: argparse.Namespace) -> str:
    origin_by_code = getattr(args, "_candidate_origin_by_code", None) or {}
    origin = origin_by_code.get(code)
    if origin:
        return str(origin)
    if code in (getattr(args, "_wang_rollout_candidate_codes", set()) or set()):
        return "wang_rollout"
    if code in (getattr(args, "_frg_frontier_child_codes", set()) or set()):
        return "frg_frontier_child"
    return "core"


def _union_diversity_objective(
    selected: set[int],
    linear: list[float],
    features: list[UnionDiversityQuboFeatures],
    args: argparse.Namespace,
) -> float:
    value = sum(linear[idx] for idx in selected)
    same_parent_penalty = float(getattr(args, "union_diversity_qubo_same_parent_penalty", 0.45) or 0.0)
    same_source_penalty = float(getattr(args, "union_diversity_qubo_same_source_penalty", 0.20) or 0.0)
    signature_penalty = float(getattr(args, "union_diversity_qubo_signature_penalty", 0.18) or 0.0)
    reduce_signature_penalty = float(getattr(args, "union_diversity_qubo_reduce_signature_penalty", 0.12) or 0.0)
    n_band_penalty = float(getattr(args, "union_diversity_qubo_n_band_penalty", 0.08) or 0.0)
    n_band_size = max(1, int(getattr(args, "union_diversity_qubo_n_band_size", 1) or 1))
    origin_similarity_penalty = float(getattr(args, "union_diversity_qubo_origin_similarity_penalty", 0.0) or 0.0)
    same_origin_parent_n_band_penalty = float(
        getattr(args, "union_diversity_qubo_same_origin_parent_n_band_penalty", 0.0) or 0.0
    )
    ordered = sorted(selected)
    for pos, i in enumerate(ordered):
        for j in ordered[pos + 1 :]:
            same_origin = bool(features[i].origin) and features[i].origin == features[j].origin
            same_n_band = features[i].n_value // n_band_size == features[j].n_value // n_band_size
            if same_parent_penalty > 0.0 and features[i].parent_code == features[j].parent_code:
                value += same_parent_penalty
            if same_source_penalty > 0.0 and features[i].source_stack >= 0 and features[i].source_stack == features[j].source_stack:
                value += same_source_penalty
            if signature_penalty > 0.0 and features[i].signature == features[j].signature:
                value += signature_penalty
            if reduce_signature_penalty > 0.0 and features[i].reduce_signature == features[j].reduce_signature:
                value += reduce_signature_penalty
            if n_band_penalty > 0.0 and same_n_band:
                value += n_band_penalty
            if origin_similarity_penalty > 0.0 and same_origin:
                value += origin_similarity_penalty
            if (
                same_origin_parent_n_band_penalty > 0.0
                and same_origin
                and same_n_band
                and features[i].parent_code == features[j].parent_code
            ):
                value += same_origin_parent_n_band_penalty
    return float(value)


def _repair_union_diversity_selection(
    picked: set[int],
    target: int,
    linear: list[float],
    features: list[UnionDiversityQuboFeatures],
    args: argparse.Namespace,
) -> set[int]:
    picked = {idx for idx in picked if 0 <= idx < len(linear)}
    target = max(0, min(target, len(linear)))
    while len(picked) > target:
        picked.remove(
            min(
                picked,
                key=lambda idx: (
                    _union_diversity_objective(picked - {idx}, linear, features, args),
                    -linear[idx],
                    idx,
                ),
            )
        )
    while len(picked) < target:
        missing = [idx for idx in range(len(linear)) if idx not in picked]
        if not missing:
            break
        picked.add(
            min(
                missing,
                key=lambda idx: (
                    _union_diversity_objective(picked | {idx}, linear, features, args),
                    linear[idx],
                    idx,
                ),
            )
        )
    improved = True
    while improved and len(picked) == target:
        improved = False
        current = _union_diversity_objective(picked, linear, features, args)
        best_swap: tuple[int, int, float] | None = None
        for out_idx in sorted(picked):
            base = picked - {out_idx}
            for in_idx in range(len(linear)):
                if in_idx in picked:
                    continue
                trial = base | {in_idx}
                value = _union_diversity_objective(trial, linear, features, args)
                if value + 1e-12 < current and (
                    best_swap is None or value < best_swap[2] - 1e-12
                ):
                    best_swap = (out_idx, in_idx, value)
        if best_swap is not None:
            picked.remove(best_swap[0])
            picked.add(best_swap[1])
            improved = True
    return picked


def _select_union_diversity_qubo_supplement(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    target: int,
) -> tuple[list[bytes], int, float]:
    target = max(0, int(target))
    if target <= 0 or not ranked:
        return [], 0, 0.0
    max_vars = max(1, int(getattr(args, "union_diversity_qubo_max_vars", 64) or 64))
    usable = list(ranked[:max_vars])
    target = min(target, len(usable))
    if target <= 0:
        return [], 0, 0.0
    rollout_by_code = _union_rollout_probe_results(usable, args)
    features = [
        _union_diversity_qubo_features(
            code,
            data,
            rank,
            args,
            rollout_by_code.get(code),
            origin=_union_diversity_candidate_origin(code, args),
        )
        for rank, (code, data) in enumerate(usable)
    ]
    linear = _normalize_values([feature.energy for feature in features])
    model = QuboModel()
    vars_ = [model.add_binary_var(f"ud_{idx}") for idx in range(len(usable))]
    for idx, value in enumerate(linear):
        model.add_linear(vars_[idx], value)

    same_parent_penalty = float(getattr(args, "union_diversity_qubo_same_parent_penalty", 0.45) or 0.0)
    same_source_penalty = float(getattr(args, "union_diversity_qubo_same_source_penalty", 0.20) or 0.0)
    signature_penalty = float(getattr(args, "union_diversity_qubo_signature_penalty", 0.18) or 0.0)
    reduce_signature_penalty = float(getattr(args, "union_diversity_qubo_reduce_signature_penalty", 0.12) or 0.0)
    n_band_penalty = float(getattr(args, "union_diversity_qubo_n_band_penalty", 0.08) or 0.0)
    n_band_size = max(1, int(getattr(args, "union_diversity_qubo_n_band_size", 1) or 1))
    origin_similarity_penalty = float(getattr(args, "union_diversity_qubo_origin_similarity_penalty", 0.0) or 0.0)
    same_origin_parent_n_band_penalty = float(
        getattr(args, "union_diversity_qubo_same_origin_parent_n_band_penalty", 0.0) or 0.0
    )
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            same_origin = bool(features[i].origin) and features[i].origin == features[j].origin
            same_n_band = features[i].n_value // n_band_size == features[j].n_value // n_band_size
            if same_parent_penalty > 0.0 and features[i].parent_code == features[j].parent_code:
                model.add_quadratic(vars_[i], vars_[j], same_parent_penalty)
            if same_source_penalty > 0.0 and features[i].source_stack >= 0 and features[i].source_stack == features[j].source_stack:
                model.add_quadratic(vars_[i], vars_[j], same_source_penalty)
            if signature_penalty > 0.0 and features[i].signature == features[j].signature:
                model.add_quadratic(vars_[i], vars_[j], signature_penalty)
            if reduce_signature_penalty > 0.0 and features[i].reduce_signature == features[j].reduce_signature:
                model.add_quadratic(vars_[i], vars_[j], reduce_signature_penalty)
            if n_band_penalty > 0.0 and same_n_band:
                model.add_quadratic(vars_[i], vars_[j], n_band_penalty)
            if origin_similarity_penalty > 0.0 and same_origin:
                model.add_quadratic(vars_[i], vars_[j], origin_similarity_penalty)
            if (
                same_origin_parent_n_band_penalty > 0.0
                and same_origin
                and same_n_band
                and features[i].parent_code == features[j].parent_code
            ):
                model.add_quadratic(vars_[i], vars_[j], same_origin_parent_n_band_penalty)
    model.add_exactly_k_penalty(vars_, target, float(getattr(args, "qubo_penalty_a", 30.0) or 30.0))
    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        _qubo_sampler_config(
            args,
            num_reads=int(getattr(args, "qubo_reads", 2) or 2),
            random_seed=int(getattr(args, "seed", 37)) + 13007 + int(getattr(args, "_union_diversity_qubo_calls_done", 0)),
            time_limit_sec=float(getattr(args, "qubo_time_limit", 0.001) or 0.001),
            sa_iterations=int(getattr(args, "qubo_iterations", 20) or 20),
        ),
    )
    picked: set[int] = set()
    if sample_set.samples:
        picked = {
            idx
            for idx in range(len(usable))
            if sample_set.samples[0].binary_values.get(idx, 0) == 1
        }
    picked = _repair_union_diversity_selection(picked, target, linear, features, args)
    ordered = sorted(
        picked,
        key=lambda idx: (
            _union_diversity_objective({idx}, linear, features, args),
            features[idx].n_value,
            features[idx].depth,
            usable[idx][0],
        ),
    )
    if hasattr(args, "_micro_rollout_selected_total"):
        args._micro_rollout_selected_total += sum(
            1
            for idx in picked
            if features[idx].micro_best_n >= 0 or features[idx].receiver_safe_delta != 0
        )
    return [usable[idx][0] for idx in ordered], len(model.var_index), sample_set.runtime_sec


class ParetoUnionFeatures(NamedTuple):
    energy: float
    n_value: int
    depth: int
    parent_code: bytes
    source_stack: int
    dest_stack: int
    source_dest: tuple[int, int]
    signature: tuple[int, ...]
    reduce_signature: tuple[int, ...]
    n_band: int
    space_score: float
    repair_score: float
    risk_score: float
    quality_rank: int


def _pareto_transition_source_stack(parent_code: bytes, code: bytes, stack_count: int, max_height: int) -> int:
    if not parent_code or stack_count <= 0 or max_height <= 0:
        return -1
    before = _split(parent_code, stack_count, max_height)
    after = _split(code, stack_count, max_height)
    for idx, (before_chunk, after_chunk) in enumerate(zip(before, after)):
        if _chunk_height_raw(after_chunk) < _chunk_height_raw(before_chunk):
            return idx
    return -1


def _transition_dest_stack(parent_code: bytes, code: bytes, stack_count: int, max_height: int) -> int:
    if not parent_code or stack_count <= 0 or max_height <= 0:
        return -1
    before = _split(parent_code, stack_count, max_height)
    after = _split(code, stack_count, max_height)
    for idx, (before_chunk, after_chunk) in enumerate(zip(before, after)):
        if _chunk_height_raw(after_chunk) > _chunk_height_raw(before_chunk):
            return idx
    return -1


def _pareto_union_space_score(code: bytes, stack_count: int, max_height: int) -> float:
    if stack_count <= 0 or max_height <= 0:
        return 0.0
    chunks = _split(code, stack_count, max_height)
    heights = [_chunk_height_raw(chunk) for chunk in chunks]
    empty_slots = sum(max_height - height for height in heights)
    total_slots = max(1.0, float(stack_count * max_height))
    avg_pressure = sum(heights) / total_slots
    max_pressure = max(heights, default=0) / max(1.0, float(max_height))
    return max(0.0, float(empty_slots) / total_slots - 0.25 * avg_pressure - 0.25 * max_pressure)


def _pareto_union_risk_score(code: bytes, stack_count: int, max_height: int) -> float:
    if stack_count <= 0 or max_height <= 0:
        return 0.0
    stacks = _code_to_stacks(code, stack_count, max_height)
    disorder = future_disorder_count_stacks(stacks)
    total = sum(len(stack) for stack in stacks)
    return float(disorder) / max(1.0, float(total))


def _pareto_union_features(
    code: bytes,
    data: tuple[int, int, bytes],
    rank: int,
    args: argparse.Namespace,
) -> ParetoUnionFeatures:
    n_value, depth, parent_code = data
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    source_stack = _pareto_transition_source_stack(parent_code, code, stack_count, max_height)
    dest_stack = _transition_dest_stack(parent_code, code, stack_count, max_height)
    signature: tuple[int, ...] = (int(n_value), 0, 0, 0, 0)
    reduce_signature: tuple[int, ...] = (source_stack, 0, 0)
    if stack_count > 0 and max_height > 0:
        before = _code_to_stacks(parent_code, stack_count, max_height)
        after = _code_to_stacks(code, stack_count, max_height)
        signals = compute_frg_transition_signals(before, after, max_height)
        signature = (
            min(3, max(0, int(signals.fill_gain))),
            min(3, max(0, int(signals.reduce_gain))),
            _sign_bucket(int(signals.unblocked_bad_delta)),
            _sign_bucket(int(signals.stable_prefix_delta)),
            min(3, max(0, int(signals.low_tier_damage))),
        )
        reduce_signature = (
            source_stack,
            min(3, max(0, int(signals.reduce_gain))),
            _sign_bucket(int(signals.unblocked_bad_delta)),
        )
    fixed = _compact_wang_fixed_prefix_score(code, stack_count, max_height) if stack_count > 0 and max_height > 0 else 0
    pair = _compact_wang_fix_pair_score(code, stack_count, max_height) if stack_count > 0 and max_height > 0 else 0
    repair_score = float(fixed + pair)
    space_score = _pareto_union_space_score(code, stack_count, max_height)
    risk_score = _pareto_union_risk_score(code, stack_count, max_height)
    n_band_size = max(1, int(getattr(args, "pareto_union_n_band_size", 2) or 2))
    quality_weight = float(getattr(args, "pareto_union_quality_weight", 1.0) or 1.0)
    repair_weight = float(getattr(args, "pareto_union_repair_weight", 0.20) or 0.0)
    space_weight = float(getattr(args, "pareto_union_space_weight", 0.12) or 0.0)
    risk_weight = float(getattr(args, "pareto_union_risk_weight", 0.16) or 0.0)
    quality_energy = float(int(n_value) * 1000 + int(depth)) + float(rank) * 0.02
    energy = (
        quality_weight * quality_energy
        - repair_weight * repair_score
        - space_weight * space_score
        + risk_weight * risk_score
    )
    return ParetoUnionFeatures(
        energy=float(energy),
        n_value=int(n_value),
        depth=int(depth),
        parent_code=parent_code,
        source_stack=int(source_stack),
        dest_stack=int(dest_stack),
        source_dest=(int(source_stack), int(dest_stack)),
        signature=signature,
        reduce_signature=reduce_signature,
        n_band=int(n_value) // n_band_size,
        space_score=float(space_score),
        repair_score=float(repair_score),
        risk_score=float(risk_score),
        quality_rank=int(rank),
    )


def _select_pareto_union_seed_indices(
    features: list[ParetoUnionFeatures],
    args: argparse.Namespace,
    target: int,
) -> list[int]:
    target = max(0, min(int(target), len(features)))
    if target <= 0:
        return []
    seed_width = max(0, int(getattr(args, "pareto_union_seed_width", target) or 0))
    if seed_width > 0:
        target = min(target, seed_width)
    picked: list[int] = []
    seen: set[int] = set()

    def add(idx: int) -> None:
        if idx not in seen and len(picked) < target:
            seen.add(idx)
            picked.append(idx)

    objective_orders = [
        sorted(range(len(features)), key=lambda idx: (features[idx].energy, features[idx].quality_rank, idx)),
        sorted(range(len(features)), key=lambda idx: (-features[idx].space_score, features[idx].quality_rank, idx)),
        sorted(range(len(features)), key=lambda idx: (-features[idx].repair_score, features[idx].quality_rank, idx)),
        sorted(range(len(features)), key=lambda idx: (features[idx].risk_score, features[idx].quality_rank, idx)),
    ]
    for order in objective_orders:
        if order:
            add(order[0])

    for attr in ("source_dest", "parent_code", "signature"):
        representatives: dict[object, int] = {}
        for idx, feature in enumerate(features):
            key = getattr(feature, attr)
            if key not in representatives:
                representatives[key] = idx
        for idx in sorted(representatives.values(), key=lambda value: (features[value].quality_rank, value)):
            add(idx)
            if len(picked) >= target:
                break
        if len(picked) >= target:
            break

    for idx in sorted(range(len(features)), key=lambda value: (features[value].quality_rank, value)):
        add(idx)
        if len(picked) >= target:
            break
    return picked


def _risk_bucket(value: float) -> int:
    return max(0, min(4, int(float(value) * 5.0)))


def _pareto_union_objective(
    selected: set[int],
    linear: list[float],
    features: list[ParetoUnionFeatures],
    args: argparse.Namespace,
) -> float:
    value = sum(linear[idx] for idx in selected)
    penalties = {
        "parent": float(getattr(args, "pareto_union_same_parent_penalty", 0.55) or 0.0),
        "source": float(getattr(args, "pareto_union_same_source_penalty", 0.25) or 0.0),
        "dest": float(getattr(args, "pareto_union_same_dest_penalty", 0.20) or 0.0),
        "source_dest": float(getattr(args, "pareto_union_same_source_dest_penalty", 0.35) or 0.0),
        "signature": float(getattr(args, "pareto_union_signature_penalty", 0.20) or 0.0),
        "reduce": float(getattr(args, "pareto_union_reduce_signature_penalty", 0.15) or 0.0),
        "n_band": float(getattr(args, "pareto_union_n_band_penalty", 0.08) or 0.0),
        "risk": float(getattr(args, "pareto_union_risk_bucket_penalty", 0.08) or 0.0),
    }
    ordered = sorted(selected)
    for pos, i in enumerate(ordered):
        for j in ordered[pos + 1 :]:
            if penalties["parent"] > 0.0 and features[i].parent_code == features[j].parent_code:
                value += penalties["parent"]
            if penalties["source"] > 0.0 and features[i].source_stack >= 0 and features[i].source_stack == features[j].source_stack:
                value += penalties["source"]
            if penalties["dest"] > 0.0 and features[i].dest_stack >= 0 and features[i].dest_stack == features[j].dest_stack:
                value += penalties["dest"]
            if penalties["source_dest"] > 0.0 and features[i].source_dest == features[j].source_dest:
                value += penalties["source_dest"]
            if penalties["signature"] > 0.0 and features[i].signature == features[j].signature:
                value += penalties["signature"]
            if penalties["reduce"] > 0.0 and features[i].reduce_signature == features[j].reduce_signature:
                value += penalties["reduce"]
            if penalties["n_band"] > 0.0 and features[i].n_band == features[j].n_band:
                value += penalties["n_band"]
            if penalties["risk"] > 0.0 and _risk_bucket(features[i].risk_score) == _risk_bucket(features[j].risk_score):
                value += penalties["risk"]
    return float(value)


def _repair_pareto_union_selection(
    picked: set[int],
    target: int,
    linear: list[float],
    features: list[ParetoUnionFeatures],
    args: argparse.Namespace,
) -> set[int]:
    picked = {idx for idx in picked if 0 <= idx < len(linear)}
    target = max(0, min(target, len(linear)))
    while len(picked) > target:
        picked.remove(
            min(
                picked,
                key=lambda idx: (
                    _pareto_union_objective(picked - {idx}, linear, features, args),
                    -features[idx].quality_rank,
                    -linear[idx],
                    idx,
                ),
            )
        )
    while len(picked) < target:
        missing = [idx for idx in range(len(linear)) if idx not in picked]
        if not missing:
            break
        picked.add(
            min(
                missing,
                key=lambda idx: (
                    _pareto_union_objective(picked | {idx}, linear, features, args),
                    linear[idx],
                    features[idx].quality_rank,
                    idx,
                ),
            )
        )
    improved = True
    while improved and len(picked) == target:
        improved = False
        current = _pareto_union_objective(picked, linear, features, args)
        best_swap: tuple[int, int, float, tuple[float, int, int, int, int]] | None = None
        for out_idx in sorted(picked):
            base = picked - {out_idx}
            for in_idx in range(len(linear)):
                if in_idx in picked:
                    continue
                trial = base | {in_idx}
                value = _pareto_union_objective(trial, linear, features, args)
                key = (
                    value,
                    features[in_idx].quality_rank,
                    -features[out_idx].quality_rank,
                    in_idx,
                    out_idx,
                )
                if value + 1e-12 < current and (best_swap is None or key < best_swap[3]):
                    best_swap = (out_idx, in_idx, value, key)
        if best_swap is not None:
            picked.remove(best_swap[0])
            picked.add(best_swap[1])
            improved = True
    return picked


def _select_pareto_union_qubo_supplement(
    usable: list[tuple[bytes, tuple[int, int, bytes]]],
    features: list[ParetoUnionFeatures],
    args: argparse.Namespace,
    target: int,
) -> tuple[list[bytes], int, float]:
    target = max(0, int(target))
    if target <= 0 or not usable:
        return [], 0, 0.0
    max_vars = max(1, int(getattr(args, "pareto_union_max_vars", 96) or 96))
    usable = list(usable[:max_vars])
    features = list(features[: len(usable)])
    target = min(target, len(usable), len(features))
    if target <= 0:
        return [], 0, 0.0
    linear = _normalize_values([feature.energy for feature in features])
    model = QuboModel()
    vars_ = [model.add_binary_var(f"pu_{idx}") for idx in range(len(usable))]
    for idx, value in enumerate(linear):
        model.add_linear(vars_[idx], value)

    penalties = {
        "parent": float(getattr(args, "pareto_union_same_parent_penalty", 0.55) or 0.0),
        "source": float(getattr(args, "pareto_union_same_source_penalty", 0.25) or 0.0),
        "dest": float(getattr(args, "pareto_union_same_dest_penalty", 0.20) or 0.0),
        "source_dest": float(getattr(args, "pareto_union_same_source_dest_penalty", 0.35) or 0.0),
        "signature": float(getattr(args, "pareto_union_signature_penalty", 0.20) or 0.0),
        "reduce": float(getattr(args, "pareto_union_reduce_signature_penalty", 0.15) or 0.0),
        "n_band": float(getattr(args, "pareto_union_n_band_penalty", 0.08) or 0.0),
        "risk": float(getattr(args, "pareto_union_risk_bucket_penalty", 0.08) or 0.0),
    }
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            if penalties["parent"] > 0.0 and features[i].parent_code == features[j].parent_code:
                model.add_quadratic(vars_[i], vars_[j], penalties["parent"])
            if penalties["source"] > 0.0 and features[i].source_stack >= 0 and features[i].source_stack == features[j].source_stack:
                model.add_quadratic(vars_[i], vars_[j], penalties["source"])
            if penalties["dest"] > 0.0 and features[i].dest_stack >= 0 and features[i].dest_stack == features[j].dest_stack:
                model.add_quadratic(vars_[i], vars_[j], penalties["dest"])
            if penalties["source_dest"] > 0.0 and features[i].source_dest == features[j].source_dest:
                model.add_quadratic(vars_[i], vars_[j], penalties["source_dest"])
            if penalties["signature"] > 0.0 and features[i].signature == features[j].signature:
                model.add_quadratic(vars_[i], vars_[j], penalties["signature"])
            if penalties["reduce"] > 0.0 and features[i].reduce_signature == features[j].reduce_signature:
                model.add_quadratic(vars_[i], vars_[j], penalties["reduce"])
            if penalties["n_band"] > 0.0 and features[i].n_band == features[j].n_band:
                model.add_quadratic(vars_[i], vars_[j], penalties["n_band"])
            if penalties["risk"] > 0.0 and _risk_bucket(features[i].risk_score) == _risk_bucket(features[j].risk_score):
                model.add_quadratic(vars_[i], vars_[j], penalties["risk"])
    model.add_exactly_k_penalty(vars_, target, float(getattr(args, "qubo_penalty_a", 30.0) or 30.0))
    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        _qubo_sampler_config(
            args,
            num_reads=int(getattr(args, "qubo_reads", 2) or 2),
            random_seed=int(getattr(args, "seed", 37)) + 17041,
            time_limit_sec=float(getattr(args, "qubo_time_limit", 0.001) or 0.001),
            sa_iterations=int(getattr(args, "qubo_iterations", 20) or 20),
        ),
    )
    picked: set[int] = set()
    if sample_set.samples:
        picked = {
            idx
            for idx in range(len(usable))
            if sample_set.samples[0].binary_values.get(idx, 0) == 1
        }
    picked = _repair_pareto_union_selection(picked, target, linear, features, args)
    ordered = sorted(
        picked,
        key=lambda idx: (
            _pareto_union_objective({idx}, linear, features, args),
            features[idx].n_value,
            features[idx].depth,
            features[idx].quality_rank,
            usable[idx][0],
        ),
    )
    return [usable[idx][0] for idx in ordered], len(model.var_index), sample_set.runtime_sec


def _ensure_pareto_union_telemetry(args: argparse.Namespace) -> None:
    defaults = {
        "_pareto_union_layers": 0,
        "_pareto_union_seed_selected_total": 0,
        "_pareto_union_qubo_calls_done": 0,
        "_pareto_union_qubo_vars": 0,
        "_pareto_union_qubo_time": 0.0,
        "_pareto_union_qubo_selected_total": 0,
        "_pareto_union_quality_selected": 0,
        "_pareto_union_space_selected": 0,
        "_pareto_union_repair_selected": 0,
        "_pareto_union_risk_selected": 0,
        "_pareto_union_distinct_parent_count_last": 0,
        "_pareto_union_distinct_source_count_last": 0,
        "_pareto_union_distinct_dest_count_last": 0,
        "_pareto_union_distinct_signature_count_last": 0,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)


def _select_pareto_union_supplement(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    target: int,
) -> tuple[list[bytes], set[bytes], int, int, float]:
    target = max(0, int(target))
    if target <= 0 or not ranked:
        return [], set(), 0, 0, 0.0

    _ensure_pareto_union_telemetry(args)
    window = max(1, int(getattr(args, "pareto_union_candidate_window", 128) or 128))
    usable = list(ranked[:window])
    features = [_pareto_union_features(code, data, rank, args) for rank, (code, data) in enumerate(usable)]
    seed_target = min(
        target,
        max(0, int(getattr(args, "pareto_union_seed_width", 0) or 0)),
        len(usable),
    )
    seed_indices = _select_pareto_union_seed_indices(features, args, seed_target)
    seed_index_set = set(seed_indices)
    selected_codes: list[bytes] = []
    selected_set: set[bytes] = set()
    for idx in seed_indices:
        code = usable[idx][0]
        if code not in selected_set:
            selected_set.add(code)
            selected_codes.append(code)

    qcodes: list[bytes] = []
    qcalls = qvars = 0
    qtime = 0.0
    qubo_target = min(
        target - len(selected_codes),
        max(0, int(getattr(args, "pareto_union_qubo_width", 0) or 0)),
    )
    if qubo_target > 0:
        q_usable = [(code, data) for idx, (code, data) in enumerate(usable) if idx not in seed_index_set]
        q_features = [feature for idx, feature in enumerate(features) if idx not in seed_index_set]
        qcodes, qvars, qtime = _select_pareto_union_qubo_supplement(q_usable, q_features, args, qubo_target)
        if qvars:
            qcalls = 1
        for code in qcodes:
            if code not in selected_set:
                selected_set.add(code)
                selected_codes.append(code)

    args._pareto_union_seed_selected_total += len(seed_indices)
    args._pareto_union_qubo_selected_total += sum(1 for code in qcodes if code in selected_set)
    args._pareto_union_quality_selected += int(0 in seed_index_set)
    args._pareto_union_space_selected += sum(1 for idx in seed_indices if features[idx].space_score > 0.0)
    args._pareto_union_repair_selected += sum(1 for idx in seed_indices if features[idx].repair_score > 0.0)
    args._pareto_union_risk_selected += sum(1 for idx in seed_indices if features[idx].risk_score <= 0.25)
    selected_features = [features[idx] for idx, (code, _data) in enumerate(usable) if code in selected_set]
    args._pareto_union_distinct_parent_count_last = len({feature.parent_code for feature in selected_features})
    args._pareto_union_distinct_source_count_last = len({feature.source_stack for feature in selected_features})
    args._pareto_union_distinct_dest_count_last = len({feature.dest_stack for feature in selected_features})
    args._pareto_union_distinct_signature_count_last = len({feature.signature for feature in selected_features})
    return selected_codes[:target], set(qcodes), qcalls, qvars, qtime


def _normalize_values(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 1e-12:
        return [0.0 for _value in values]
    return [(value - lo) / span for value in values]


def _bs_frg_selection_objective(
    selected: set[int],
    linear: list[float],
    features: list[BsFrgQuboFeatures],
    args: argparse.Namespace,
) -> float:
    value = sum(linear[idx] for idx in selected)
    same_parent_penalty = float(getattr(args, "bs_frg_qubo_same_parent_penalty", 0.35) or 0.0)
    same_source_penalty = float(getattr(args, "bs_frg_qubo_same_source_penalty", 0.0) or 0.0)
    signature_penalty = float(getattr(args, "bs_frg_qubo_signature_penalty", 0.15) or 0.0)
    reduce_signature_penalty = float(getattr(args, "bs_frg_qubo_reduce_signature_penalty", 0.0) or 0.0)
    n_band_penalty = float(getattr(args, "bs_frg_qubo_n_band_penalty", 0.05) or 0.0)
    n_band_size = max(1, int(getattr(args, "bs_frg_qubo_n_band_size", 1) or 1))
    ordered = sorted(selected)
    for pos, i in enumerate(ordered):
        for j in ordered[pos + 1 :]:
            if same_parent_penalty > 0.0 and features[i].parent_code == features[j].parent_code:
                value += same_parent_penalty
            if same_source_penalty > 0.0 and features[i].source_stack >= 0 and features[i].source_stack == features[j].source_stack:
                value += same_source_penalty
            if signature_penalty > 0.0 and features[i].signature == features[j].signature:
                value += signature_penalty
            if reduce_signature_penalty > 0.0 and features[i].reduce_signature == features[j].reduce_signature:
                value += reduce_signature_penalty
            if n_band_penalty > 0.0 and features[i].n_value // n_band_size == features[j].n_value // n_band_size:
                value += n_band_penalty
    return float(value)


def _repair_bs_frg_selection(
    picked: set[int],
    target: int,
    linear: list[float],
    features: list[BsFrgQuboFeatures],
    args: argparse.Namespace,
) -> set[int]:
    picked = {idx for idx in picked if 0 <= idx < len(linear)}
    target = max(0, min(target, len(linear)))
    while len(picked) > target:
        picked.remove(
            min(
                picked,
                key=lambda idx: (
                    _bs_frg_selection_objective(picked - {idx}, linear, features, args),
                    -linear[idx],
                    idx,
                ),
            )
        )
    while len(picked) < target:
        missing = [idx for idx in range(len(linear)) if idx not in picked]
        if not missing:
            break
        picked.add(
            min(
                missing,
                key=lambda idx: (
                    _bs_frg_selection_objective(picked | {idx}, linear, features, args),
                    linear[idx],
                    idx,
                ),
            )
        )
    improved = True
    while improved and len(picked) == target:
        improved = False
        current = _bs_frg_selection_objective(picked, linear, features, args)
        best_swap: tuple[int, int, float] | None = None
        for out_idx in sorted(picked):
            base = picked - {out_idx}
            for in_idx in range(len(linear)):
                if in_idx in picked:
                    continue
                trial = base | {in_idx}
                value = _bs_frg_selection_objective(trial, linear, features, args)
                if value + 1e-12 < current and (
                    best_swap is None or value < best_swap[2] - 1e-12
                ):
                    best_swap = (out_idx, in_idx, value)
        if best_swap is not None:
            picked.remove(best_swap[0])
            picked.add(best_swap[1])
            improved = True
    return picked


def _select_bs_frg_qubo_supplement(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    target: int,
) -> tuple[list[bytes], int, float]:
    target = max(0, int(target))
    if target <= 0 or not ranked:
        return [], 0, 0.0
    max_vars = max(1, int(getattr(args, "bs_frg_qubo_max_vars", 64) or 64))
    usable = list(ranked[:max_vars])
    target = min(target, len(usable))
    if target <= 0:
        return [], 0, 0.0
    features = [
        _bs_frg_qubo_features(code, data, rank, args)
        for rank, (code, data) in enumerate(usable)
    ]
    linear = _normalize_values([feature.energy for feature in features])
    model = QuboModel()
    vars_ = [model.add_binary_var(f"bf_{idx}") for idx in range(len(usable))]
    for idx, value in enumerate(linear):
        model.add_linear(vars_[idx], value)

    same_parent_penalty = float(getattr(args, "bs_frg_qubo_same_parent_penalty", 0.35) or 0.0)
    same_source_penalty = float(getattr(args, "bs_frg_qubo_same_source_penalty", 0.0) or 0.0)
    signature_penalty = float(getattr(args, "bs_frg_qubo_signature_penalty", 0.15) or 0.0)
    reduce_signature_penalty = float(getattr(args, "bs_frg_qubo_reduce_signature_penalty", 0.0) or 0.0)
    n_band_penalty = float(getattr(args, "bs_frg_qubo_n_band_penalty", 0.05) or 0.0)
    n_band_size = max(1, int(getattr(args, "bs_frg_qubo_n_band_size", 1) or 1))
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            if same_parent_penalty > 0.0 and features[i].parent_code == features[j].parent_code:
                model.add_quadratic(vars_[i], vars_[j], same_parent_penalty)
            if same_source_penalty > 0.0 and features[i].source_stack >= 0 and features[i].source_stack == features[j].source_stack:
                model.add_quadratic(vars_[i], vars_[j], same_source_penalty)
            if signature_penalty > 0.0 and features[i].signature == features[j].signature:
                model.add_quadratic(vars_[i], vars_[j], signature_penalty)
            if reduce_signature_penalty > 0.0 and features[i].reduce_signature == features[j].reduce_signature:
                model.add_quadratic(vars_[i], vars_[j], reduce_signature_penalty)
            if n_band_penalty > 0.0 and features[i].n_value // n_band_size == features[j].n_value // n_band_size:
                model.add_quadratic(vars_[i], vars_[j], n_band_penalty)
    model.add_exactly_k_penalty(vars_, target, float(getattr(args, "qubo_penalty_a", 30.0) or 30.0))
    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        _qubo_sampler_config(
            args,
            num_reads=int(getattr(args, "qubo_reads", 2) or 2),
            random_seed=int(getattr(args, "seed", 37)) + 11003 + int(getattr(args, "_bs_frg_qubo_calls_done", 0)),
            time_limit_sec=float(getattr(args, "qubo_time_limit", 0.001) or 0.001),
            sa_iterations=int(getattr(args, "qubo_iterations", 20) or 20),
        ),
    )
    picked: set[int] = set()
    if sample_set.samples:
        picked = {
            idx
            for idx in range(len(usable))
            if sample_set.samples[0].binary_values.get(idx, 0) == 1
        }
    picked = _repair_bs_frg_selection(picked, target, linear, features, args)
    ordered = sorted(
        picked,
        key=lambda idx: (
            _bs_frg_selection_objective({idx}, linear, features, args),
            features[idx].n_value,
            features[idx].depth,
            usable[idx][0],
        ),
    )
    return [usable[idx][0] for idx in ordered], len(model.var_index), sample_set.runtime_sec


def _frg_scarcity_gate_active(args: argparse.Namespace, best_n: int) -> bool:
    if not getattr(args, "frg_scarcity_gate_enabled", False):
        return False
    focus_n = int(getattr(args, "frg_scarcity_gate_focus_n", -1))
    if focus_n >= 0 and int(best_n) > focus_n:
        return False
    max_available = int(getattr(args, "frg_scarcity_gate_max_available_stacks", 1) or 1)
    available = int(getattr(args, "_frg_layout_available_stack_count", 10**9) or 10**9)
    if available > max_available:
        return False
    min_pressure = float(getattr(args, "frg_scarcity_gate_min_avg_height_pressure", 0.85) or 0.0)
    pressure = float(getattr(args, "_frg_layout_avg_height_pressure", 0.0) or 0.0)
    if pressure < min_pressure:
        return False
    max_receivers = int(getattr(args, "frg_scarcity_gate_max_reduction_receivers", -1) or -1)
    if max_receivers >= 0:
        receivers = int(getattr(args, "_frg_layout_reduction_receiver_stack_count", 10**9) or 10**9)
        if receivers > max_receivers:
            return False
    min_unblocked_bad = int(getattr(args, "frg_scarcity_gate_min_unblocked_bad", 0) or 0)
    if min_unblocked_bad > 0:
        unblocked_bad = int(getattr(args, "_frg_layout_unblocked_bad_containers", 0) or 0)
        if unblocked_bad < min_unblocked_bad:
            return False
    min_candidates = int(getattr(args, "frg_scarcity_gate_min_candidates", 1000) or 0)
    if int(getattr(args, "_layer_candidate_count", 0) or 0) < min_candidates:
        return False
    cap_slack = int(getattr(args, "frg_scarcity_gate_cap_slack", 24) or 0)
    if int(getattr(args, "_layer_depth_slack", 10**9) or 10**9) > cap_slack:
        return False
    return True


def _frg_compound_escape_active(args: argparse.Namespace, best_n: int) -> bool:
    if not getattr(args, "frg_compound_escape_enabled", False):
        return False
    focus_n = int(getattr(args, "frg_compound_escape_focus_n", -1))
    if focus_n >= 0 and int(best_n) > focus_n:
        return False
    min_candidates = int(getattr(args, "frg_compound_escape_min_candidates", 7000) or 0)
    if int(getattr(args, "_layer_candidate_count", 0)) < min_candidates:
        return False
    cap_slack = int(getattr(args, "frg_compound_escape_cap_slack", 24) or 0)
    if int(getattr(args, "_layer_depth_slack", 10**9)) > cap_slack:
        return False
    window = max(2, int(getattr(args, "frg_compound_escape_plateau_window", 5) or 5))
    history = list(getattr(args, "_best_n_history", []) or [])
    if len(history) < window:
        return False
    recent = history[-window:]
    delta = max(0, int(getattr(args, "frg_compound_escape_plateau_delta", 0) or 0))
    return max(recent) - min(recent) <= delta


def _frg_compound_guard_accepts(
    args: argparse.Namespace,
    best_n: int,
    origins: list[bytes],
    candidates: dict[bytes, tuple[int, int, bytes]],
) -> bool:
    if not getattr(args, "frg_compound_guard_enabled", False):
        return True
    max_stack_count = int(getattr(args, "frg_compound_guard_max_stack_count", 0) or 0)
    if max_stack_count > 0 and int(getattr(args, "_stack_count", 0) or 0) > max_stack_count:
        return False
    max_initial_n = int(getattr(args, "frg_compound_guard_max_initial_n", -1))
    if max_initial_n >= 0 and int(getattr(args, "_initial_n", 0) or 0) > max_initial_n:
        return False
    min_best_n = int(getattr(args, "frg_compound_guard_min_best_n", -1))
    if min_best_n >= 0 and int(best_n) < min_best_n:
        return False
    max_best_n = int(getattr(args, "frg_compound_guard_max_best_n", -1))
    if max_best_n >= 0 and int(best_n) > max_best_n:
        return False
    min_candidates = int(getattr(args, "frg_compound_guard_min_candidates", 0) or 0)
    if min_candidates > 0 and int(getattr(args, "_layer_candidate_count", 0)) < min_candidates:
        return False
    parent_counts: dict[bytes, int] = {}
    for code in origins:
        data = candidates.get(code)
        if data is None:
            continue
        parent_counts[data[2]] = parent_counts.get(data[2], 0) + 1
    if not parent_counts:
        return False
    min_parent_count = int(getattr(args, "frg_compound_guard_min_parent_count", 0) or 0)
    if min_parent_count > 0 and len(parent_counts) < min_parent_count:
        return False
    max_parent_share = float(getattr(args, "frg_compound_guard_max_parent_share", 1.0) or 1.0)
    if max_parent_share < 1.0 and max(parent_counts.values()) / float(sum(parent_counts.values())) > max_parent_share:
        return False
    return True


def _frg_best_successor(
    code: bytes,
    n_value: int,
    depth: int,
    args: argparse.Namespace,
) -> tuple[bytes, tuple[int, int, bytes]] | None:
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return None
    before = _code_to_stacks(code, stack_count, max_height)
    best: tuple[float, int, int, bytes] | None = None
    for child_code, child_n in _successors_with_n_uncached(code, int(n_value), stack_count, max_height):
        child_depth = int(depth) + 1
        after = _code_to_stacks(child_code, stack_count, max_height)
        signals = compute_frg_transition_signals(before, after, max_height)
        score = frg_transition_score(signals, n_value=int(child_n), depth=child_depth)
        item = (score, int(child_n), child_depth, child_code)
        if best is None or item < best:
            best = item
    if best is None:
        return None
    _score, child_n, child_depth, child_code = best
    return child_code, (child_n, child_depth, code)


def _moves_reach_n0_with_cap(instance: Instance, moves: list[Move], cap: int) -> bool:
    state = State(stacks=[list(stack) for stack in instance.stacks])
    _initial_code, initial_n = _encode(state.stacks, instance.stack_count, instance.max_height)
    max_g = initial_n
    try:
        for move in moves:
            state = apply_move(instance, state, move)
            _code, n_value = _encode(state.stacks, instance.stack_count, instance.max_height)
            max_g = max(max_g, state.move_count + n_value)
    except Exception:
        return False
    _final_code, final_n = _encode(state.stacks, instance.stack_count, instance.max_height)
    return final_n == 0 and max_g <= cap


def _frg_seed_large_layout_allowed(instance: Instance, args: argparse.Namespace) -> bool:
    if not bool(getattr(args, "frg_seed_allow_large_over_cap", False)):
        return False
    return instance.stack_count >= 8 or instance.max_height >= 8


def _frg_seed_incumbent(instance: Instance, args: argparse.Namespace, deadline: float) -> FrgSeedResult | None:
    if not getattr(args, "frg_seed_enabled", False):
        return None
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        return None
    configured_limit = float(getattr(args, "frg_seed_time_limit", 0.0) or 0.0)
    if configured_limit <= 0:
        return None
    respect_cap = bool(getattr(args, "frg_seed_respect_cap", True))
    if not respect_cap and not _frg_seed_large_layout_allowed(instance, args):
        return None
    seed_start = time.perf_counter()
    seed_deadline = seed_start + min(remaining, configured_limit)
    initial_n = future_disorder_count_stacks(instance.stacks)
    multiplier = max(1, int(getattr(args, "frg_seed_max_steps_multiplier", 10) or 10))
    min_steps = max(1, int(getattr(args, "frg_seed_min_max_steps", 256) or 256))
    max_steps = max(min_steps, initial_n * multiplier)
    cap = entropy_limit(instance, float(getattr(args, "alpha", 1.3) or 1.3))
    if respect_cap:
        max_steps = min(max_steps, max(1, int(cap)))
    max_steps_cap = max(0, int(getattr(args, "frg_seed_max_steps_cap", 0) or 0))
    if max_steps_cap > 0:
        max_steps = min(max_steps, max_steps_cap)

    rollout = _bs_frg_completion_cached(
        _bs_frg_initial_layout(instance),
        instance.max_height,
        start_step=0,
        max_steps=max_steps,
        deadline=seed_deadline,
        cache={},
        paper_pure=bool(getattr(args, "frg_seed_paper_pure", True)),
    )
    runtime_sec = time.perf_counter() - seed_start
    if not rollout.feasible or not rollout.moves:
        return None
    cap_ok = _moves_reach_n0_with_cap(instance, list(rollout.moves), int(cap))
    if respect_cap and not cap_ok:
        return None
    return FrgSeedResult(
        moves=tuple(rollout.moves),
        final_n=int(rollout.final_n),
        generated=int(rollout.generated),
        runtime_sec=runtime_sec,
        status=rollout.status,
        cap_ok=cap_ok,
    )


def _validated_wang_incumbent(instance: Instance, args: argparse.Namespace, cap: int, deadline: float) -> Solution | None:
    if not getattr(args, "wang_incumbent_enabled", False):
        return None
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        return None
    configured_limit = float(getattr(args, "wang_incumbent_time_limit", 0.0) or 0.0)
    time_limit = min(remaining, configured_limit if configured_limit > 0 else remaining)
    max_depth = int(getattr(args, "wang_incumbent_max_depth", 0) or 0)
    if max_depth <= 0:
        max_depth = max(1, cap)
    config = SolverConfig(
        max_depth=max_depth,
        beam_width=max(1, int(getattr(args, "wang_incumbent_beam_width", 100) or 100)),
        max_candidates_per_layer=max(1, int(getattr(args, "wang_incumbent_candidate_limit", 64) or 64)),
        candidate_limit_per_state=max(1, int(getattr(args, "wang_incumbent_candidate_limit", 64) or 64)),
        time_limit_sec=max(0.001, time_limit),
    )
    solution = WangBsBabyMovesSolver().solve(instance, config)
    if not solution.feasible or not solution.moves:
        return None
    if not _moves_reach_n0_with_cap(instance, solution.moves, cap):
        return None
    return solution


def _wang_rollout_layer_active(args: argparse.Namespace, best_n: int) -> bool:
    if not getattr(args, "wang_rollout_enabled", False):
        return False
    focus_n = int(getattr(args, "wang_rollout_focus_n", 8))
    if focus_n >= 0 and best_n > focus_n:
        return False
    window = max(0, int(getattr(args, "wang_rollout_plateau_window", 0) or 0))
    if window <= 1:
        return True
    history = list(getattr(args, "_best_n_history", []) or [])
    if len(history) < window:
        return False
    recent = history[-window:]
    delta = max(0, int(getattr(args, "wang_rollout_plateau_delta", 0) or 0))
    return max(recent) - min(recent) <= delta


def _wang_baby_lookahead_successor(
    code: bytes,
    n_value: int,
    depth: int,
    args: argparse.Namespace,
) -> tuple[bytes, tuple[int, int, bytes]] | None:
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return None

    stacks = _code_to_stacks(code, stack_count, max_height)
    instance = Instance(
        instance_id=str(getattr(args, "_case_id", "wang_rollout")),
        variant=ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=stack_count,
        max_height=max_height,
        container_count=sum(len(stack) for stack in stacks),
        stacks=[list(stack) for stack in stacks],
    )
    disorder = future_disorder_count_stacks(stacks)
    state = State(
        stacks=[list(stack) for stack in stacks],
        move_count=depth,
        moves=[],
        disorder_current=disorder,
        disorder_total=disorder,
        entropy_max=depth + disorder,
    )
    base = _wang_prepare_node(
        instance,
        _WangNode(
            state,
            tuple(0 for _ in range(stack_count)),
            _wang_max_group(stacks),
        ),
    )
    if _wang_goal_stacks(base.state.stacks):
        return None

    pairs = _wang_candidate_pairs(base.state.stacks, list(base.fixed), base.group, max_height)
    limit = max(0, int(getattr(args, "wang_rollout_candidate_limit", 64) or 0))
    if limit > 0:
        pairs = pairs[:limit]

    best: tuple[tuple, bytes, int] | None = None
    for pair in pairs:
        giant_child = _wang_apply_giant_move(instance, base, pair)
        if giant_child is None:
            continue
        interim = _wang_first_interim_from_giant_child(instance, base, giant_child)
        if interim.state.move_count != depth + 1:
            continue
        child_code, child_n = _encode(interim.state.stacks, stack_count, max_height)
        if child_code == code:
            continue
        score = (
            _wang_node_rank(instance, giant_child),
            _wang_node_rank(instance, interim),
            child_code,
        )
        if best is None or score < best[0]:
            best = (score, child_code, child_n)

    if best is None:
        return None
    return best[1], (best[2], depth + 1, code)


def _wang_rollout_successor(
    code: bytes,
    n_value: int,
    depth: int,
    args: argparse.Namespace,
) -> tuple[bytes, tuple[int, int, bytes]] | None:
    if getattr(args, "wang_rollout_strategy", "local") == "baby_lookahead":
        return _wang_baby_lookahead_successor(code, n_value, depth, args)

    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return None
    successors = list(_successors_with_n_uncached(code, n_value, stack_count, max_height))
    if not successors:
        return None
    child_code, child_n = min(
        successors,
        key=lambda item: (
            item[1],
            -_compact_wang_fix_pair_score(item[0], stack_count, max_height),
            -_compact_wang_fixed_prefix_score(item[0], stack_count, max_height),
            item[0],
        ),
    )
    return child_code, (child_n, depth + 1, code)


def _select_wang_pair_qubo_supplement(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    target: int,
) -> tuple[list[bytes], int, float]:
    target = max(0, int(target))
    if target <= 0 or not ranked:
        return [], 0, 0.0
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return [], 0, 0.0
    max_vars = max(1, int(getattr(args, "wang_pair_qubo_max_vars", 32) or 32))
    scored = [
        (
            _compact_wang_fix_pair_score(code, stack_count, max_height),
            _compact_wang_fixed_prefix_score(code, stack_count, max_height),
            code,
            data,
        )
        for code, data in ranked
    ]
    scored = [item for item in scored if item[0] > 0 or item[1] > 0]
    if not scored:
        return [], 0, 0.0
    scored.sort(key=lambda item: (-item[0], -item[1], item[3][0], item[3][1], item[3][2], item[2]))
    usable = scored[:max_vars]
    target = min(target, len(usable))
    hi = max((pair_score for pair_score, _fixed, _code, _data in usable), default=0)
    if hi <= 0:
        hi = max((fixed for _pair_score, fixed, _code, _data in usable), default=1)

    model = QuboModel()
    vars_ = [model.add_binary_var(f"wp_{idx}") for idx in range(len(usable))]
    linear: list[float] = []
    for idx, (pair_score, fixed, _code, _data) in enumerate(usable):
        score = pair_score if pair_score > 0 else fixed
        value = -float(score) / max(1.0, float(hi))
        linear.append(value)
        model.add_linear(vars_[idx], value)
    same_parent_penalty = float(getattr(args, "wang_pair_qubo_same_parent_penalty", 0.25) or 0.0)
    if same_parent_penalty > 0:
        for i in range(len(usable)):
            parent_i = usable[i][3][2]
            for j in range(i + 1, len(usable)):
                if parent_i == usable[j][3][2]:
                    model.add_quadratic(vars_[i], vars_[j], same_parent_penalty)
    model.add_exactly_k_penalty(vars_, target, float(getattr(args, "qubo_penalty_a", 30.0)))
    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        _qubo_sampler_config(
            args,
            num_reads=int(getattr(args, "qubo_reads", 2) or 2),
            random_seed=int(getattr(args, "seed", 37)) + 7001 + int(getattr(args, "_wang_pair_qubo_calls_done", 0)),
            time_limit_sec=float(getattr(args, "qubo_time_limit", 0.001) or 0.001),
            sa_iterations=int(getattr(args, "qubo_iterations", 20) or 20),
        ),
    )
    picked: set[int] = set()
    if sample_set.samples:
        picked = {idx for idx in range(len(usable)) if sample_set.samples[0].binary_values.get(idx, 0) == 1}
    while len(picked) > target:
        picked.remove(max(picked, key=lambda idx: (linear[idx], idx)))
    while len(picked) < target:
        missing = [idx for idx in range(len(usable)) if idx not in picked]
        if not missing:
            break
        picked.add(min(missing, key=lambda idx: (linear[idx], idx)))
    ordered = sorted(picked, key=lambda idx: (linear[idx], usable[idx][3][0], usable[idx][3][1], usable[idx][2]))
    return [usable[idx][2] for idx in ordered], len(model.var_index), sample_set.runtime_sec


def _select_unified_wang_qubo_supplement(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    target: int,
) -> tuple[list[bytes], int, float, dict[str, int]]:
    empty_metrics = {"fixed_prefix_selected": 0, "fix_pair_selected": 0}
    target = max(0, int(target))
    if target <= 0 or not ranked:
        return [], 0, 0.0, empty_metrics

    def _int_attr(name: str, default: int) -> int:
        value = getattr(args, name, None)
        if value is None:
            value = default
        return int(value)

    def _float_attr(name: str, default: float) -> float:
        value = getattr(args, name, None)
        if value is None:
            value = default
        return float(value)

    max_vars = max(1, _int_attr("unified_wang_qubo_max_vars", 64))
    usable = list(ranked[:max_vars])
    target = min(target, len(usable))
    if target <= 0:
        return [], 0, 0.0, empty_metrics

    energies = [_unified_wang_qubo_linear_energy(code, data, usable, args) for code, data in usable]
    features = [_unified_wang_qubo_features(code, data, usable, args) for code, data in usable]
    lo = min(energies)
    hi = max(energies)
    span = hi - lo
    linear = [(energy - lo) / span if span > 1e-12 else 0.0 for energy in energies]

    model = QuboModel()
    vars_ = [model.add_binary_var(f"uw_{idx}") for idx in range(len(usable))]
    for idx, value in enumerate(linear):
        model.add_linear(vars_[idx], value)

    pair_same_parent_penalty = _float_attr("unified_wang_pair_same_parent_penalty", 0.0)
    signature_penalty = _float_attr("unified_wang_signature_penalty", 0.0)
    complement_bonus = _float_attr("unified_wang_complement_bonus", 0.0)
    for i in range(len(usable)):
        parent_i = usable[i][1][2]
        signature_i = (features[i].n_value, features[i].target_gap)
        for j in range(i + 1, len(usable)):
            if pair_same_parent_penalty > 0.0 and parent_i == usable[j][1][2]:
                model.add_quadratic(vars_[i], vars_[j], pair_same_parent_penalty)
            if signature_penalty > 0.0 and signature_i == (features[j].n_value, features[j].target_gap):
                model.add_quadratic(vars_[i], vars_[j], signature_penalty)
            if complement_bonus > 0.0 and (
                (features[i].fix_pair > 0 and features[j].fixed_prefix > 0)
                or (features[j].fix_pair > 0 and features[i].fixed_prefix > 0)
            ):
                model.add_quadratic(vars_[i], vars_[j], -complement_bonus)

    model.add_exactly_k_penalty(vars_, target, _float_attr("qubo_penalty_a", 30.0))
    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        _qubo_sampler_config(
            args,
            num_reads=_int_attr("qubo_reads", 2),
            random_seed=_int_attr("seed", 37) + 9001 + _int_attr("_unified_wang_qubo_calls_done", 0),
            time_limit_sec=_float_attr("qubo_time_limit", 0.001),
            sa_iterations=_int_attr("qubo_iterations", 20),
        ),
    )
    picked: set[int] = set()
    if sample_set.samples:
        picked = {idx for idx in range(len(usable)) if sample_set.samples[0].binary_values.get(idx, 0) == 1}

    def _energy_key(idx: int) -> tuple[float, int, int, bytes]:
        data = usable[idx][1]
        return energies[idx], data[0], data[1], usable[idx][0]

    while len(picked) > target:
        picked.remove(max(picked, key=_energy_key))
    while len(picked) < target:
        missing = [idx for idx in range(len(usable)) if idx not in picked]
        if not missing:
            break
        picked.add(min(missing, key=_energy_key))

    def _assignment(indices: set[int]) -> dict[int, int]:
        return {idx: 1 for idx in indices}

    improved = True
    while improved and len(picked) == target:
        improved = False
        current_energy = model.evaluate(_assignment(picked))
        best_swap: tuple[int, int, float] | None = None
        missing = sorted((idx for idx in range(len(usable)) if idx not in picked), key=_energy_key)
        for out_idx in sorted(picked, key=_energy_key):
            for in_idx in missing:
                trial = set(picked)
                trial.remove(out_idx)
                trial.add(in_idx)
                trial_energy = model.evaluate(_assignment(trial))
                if trial_energy < current_energy - 1e-12 and (
                    best_swap is None or trial_energy < best_swap[2] - 1e-12
                ):
                    best_swap = (out_idx, in_idx, trial_energy)
        if best_swap is not None:
            picked.remove(best_swap[0])
            picked.add(best_swap[1])
            improved = True

    ordered = sorted(picked, key=_energy_key)
    metrics = {
        "fixed_prefix_selected": sum(features[idx].fixed_prefix for idx in ordered),
        "fix_pair_selected": sum(features[idx].fix_pair for idx in ordered),
    }
    return [usable[idx][0] for idx in ordered], len(model.var_index), sample_set.runtime_sec, metrics


def _feasibility_signals_for_code(code: bytes, args: argparse.Namespace) -> FeasibilitySignals:
    stacks = _split(code, int(args._stack_count), int(args._max_height))
    target_gap = _candidate_target_gap(code, args) if getattr(args, "_layout_target_groups", ()) else 0
    return compute_feasibility_signals(stacks, int(args._max_height), target_gap=target_gap)


def _format_feasibility_escape_gate_reasons(counts: dict[str, int] | None) -> str:
    if not counts:
        return ""
    return ";".join(f"{key}:{int(counts[key])}" for key in sorted(counts) if int(counts[key]) > 0)


def _format_scarcity_inner_rescue_gate_reasons(counts: dict[str, int] | None) -> str:
    if not counts:
        return ""
    return ";".join(f"{key}:{int(counts[key])}" for key in sorted(counts) if int(counts[key]) > 0)


def _record_scarcity_inner_rescue_gate_reason(args: argparse.Namespace, reason: str) -> None:
    if not getattr(args, "_scheduler_state_requested_rescue", False):
        return
    if not hasattr(args, "_scarcity_inner_rescue_gate_reasons"):
        args._scarcity_inner_rescue_gate_reasons = {}
    args._scarcity_inner_rescue_gate_reasons[reason] = (
        int(args._scarcity_inner_rescue_gate_reasons.get(reason, 0)) + 1
    )


def _feasibility_escape_gate_reason(args: argparse.Namespace, best_n: int) -> str:
    if not getattr(args, "feasibility_escape_enabled", False):
        return "disabled"
    focus_n = int(getattr(args, "feasibility_escape_focus_n", 8))
    if focus_n >= 0 and int(best_n) > focus_n:
        return "focus_n"
    history = list(getattr(args, "_best_n_history", []) or [])
    if len(history) < 3:
        return "history_warmup"
    if min(history[-3:]) != history[-3]:
        return "not_plateau"
    if int(getattr(args, "_layer_depth_slack", 10**9)) > int(getattr(args, "rescue_cap_slack", 26)):
        return "slack_too_loose"
    return "active"


def _feasibility_escape_layer_active(args: argparse.Namespace, best_n: int) -> bool:
    return _feasibility_escape_gate_reason(args, best_n) == "active"


def _select_feasibility_escape_supplement(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    target: int,
) -> list[bytes]:
    target = max(0, min(int(target), len(ranked)))
    if target <= 0:
        return []
    threshold = float(getattr(args, "feasibility_escape_dead_end_threshold", 0.45) or 0.45)
    min_slack = int(getattr(args, "feasibility_escape_min_temporary_slack", 1) or 1)
    scored: list[tuple[float, int, int, bytes]] = []
    for code, data in ranked:
        signals = _feasibility_signals_for_code(code, args)
        if signals.dead_end_risk < threshold and signals.temporary_slack_moves < min_slack:
            continue
        score = feasibility_escape_score(signals, int(data[0]), int(data[1]))
        scored.append((score, int(data[0]), int(data[1]), code))
    return [item[3] for item in sorted(scored)[:target]]


def _wang2017_task_gate_reason(args: argparse.Namespace, best_n: int) -> str:
    if not getattr(args, "wang2017_task_enabled", False):
        return "disabled"
    if int(getattr(args, "wang2017_task_supplement_width", 0) or 0) <= 0:
        return "disabled"
    focus_n = int(getattr(args, "wang2017_task_focus_n", 8))
    if focus_n >= 0 and int(best_n) > focus_n:
        return "focus_n"
    window = max(0, int(getattr(args, "wang2017_task_plateau_window", 0) or 0))
    if window > 1:
        history = list(getattr(args, "_best_n_history", []) or [])
        if len(history) < window:
            return "history_warmup"
        recent = history[-window:]
        delta = max(0, int(getattr(args, "wang2017_task_plateau_delta", 0) or 0))
        if max(recent) - min(recent) > delta:
            return "not_plateau"
        if int(getattr(args, "_layer_depth_slack", 10**9)) > int(getattr(args, "rescue_cap_slack", 26)):
            return "slack_too_loose"
    return "active"


def _wang2017_task_layer_active(args: argparse.Namespace, best_n: int) -> bool:
    return _wang2017_task_gate_reason(args, best_n) == "active"


def _wang2017_stable_prefix_len(stack: list[int]) -> int:
    if not stack:
        return 0
    prefix = 1
    for idx in range(1, len(stack)):
        if stack[idx - 1] > stack[idx]:
            prefix += 1
        else:
            break
    return prefix


def _wang2017_infer_move(
    parent_stacks: list[list[int]],
    child_stacks: list[list[int]],
) -> tuple[int, int, int] | None:
    child_signature = sorted(tuple(stack) for stack in child_stacks)
    for source, source_stack in enumerate(parent_stacks):
        if not source_stack:
            continue
        moved = source_stack[-1]
        for parent_target, target_stack in enumerate(parent_stacks):
            if source == parent_target:
                continue
            expected = [list(stack) for stack in parent_stacks]
            expected[source] = expected[source][:-1]
            expected[parent_target] = expected[parent_target] + [moved]
            if sorted(tuple(stack) for stack in expected) != child_signature:
                continue
            child_targets = [
                idx
                for idx, child_stack in enumerate(child_stacks)
                if child_stack == expected[parent_target]
            ]
            if not child_targets:
                continue
            return source, child_targets[0], moved
    return None


def _wang2017_task_features_for_code(
    parent_code: bytes,
    child_code: bytes,
    data: tuple[int, int, bytes],
    args: argparse.Namespace,
) -> Wang2017TaskFeatures:
    stack_count = int(args._stack_count)
    max_height = int(args._max_height)
    parent_stacks = [[int(box) for box in stack if int(box) > 0] for stack in _split(parent_code, stack_count, max_height)]
    child_stacks = [[int(box) for box in stack if int(box) > 0] for stack in _split(child_code, stack_count, max_height)]
    child_n = int(data[0])
    depth = int(data[1])
    move = _wang2017_infer_move(parent_stacks, child_stacks)
    if move is None:
        return Wang2017TaskFeatures(
            source=-1,
            target=-1,
            moved_group=0,
            tier_protection=1,
            task_move_estimate=10**6,
            stable_loss=10**6,
            affected_demand=10**6,
            target_stable_height=0,
            child_n=child_n,
            depth=depth,
            dead_end_risk=1.0,
        )
    source, target_stack, moved_group = move
    parent_stable = [_wang2017_stable_prefix_len(stack) for stack in parent_stacks]
    child_stable = [_wang2017_stable_prefix_len(stack) for stack in child_stacks]
    source_top_index = len(parent_stacks[source]) - 1
    tier_protection = 1 if source_top_index < parent_stable[source] else 0
    stable_loss = max(0, parent_stable[source] - child_stable[source])
    target_gap = _candidate_target_gap(child_code, args) if getattr(args, "_layout_target_groups", ()) else 0
    signals = compute_feasibility_signals(child_stacks, max_height, target_gap=target_gap)
    affected_demand = (
        int(signals.blocked_fixed_candidates)
        + int(signals.target_gap)
        + int(signals.low_tier_risk)
    )
    task_move_estimate = (
        1
        + stable_loss
        + affected_demand
        + int(round(float(signals.dead_end_risk) * 4.0))
    )
    return Wang2017TaskFeatures(
        source=source,
        target=target_stack,
        moved_group=int(moved_group),
        tier_protection=tier_protection,
        task_move_estimate=task_move_estimate,
        stable_loss=stable_loss,
        affected_demand=affected_demand,
        target_stable_height=child_stable[target_stack],
        child_n=child_n,
        depth=depth,
        dead_end_risk=float(signals.dead_end_risk),
    )


def _wang2017_task_score(features: Wang2017TaskFeatures) -> tuple[float, int, int, int, int, int, int, int]:
    return (
        float(features.tier_protection),
        int(features.task_move_estimate),
        int(features.stable_loss),
        int(features.affected_demand),
        -int(features.target_stable_height),
        -int(features.moved_group),
        int(features.child_n),
        int(features.depth),
    )


def _select_wang2017_task_supplement(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    target: int,
) -> tuple[list[bytes], dict[str, int]]:
    target = max(0, min(int(target), len(ranked)))
    metrics = {
        "selected_tier_protection_total": 0,
        "selected_stable_height_total": 0,
    }
    if target <= 0:
        return [], metrics
    scored: list[tuple[tuple[float, int, int, int, int, int, int, int], int, bytes, Wang2017TaskFeatures]] = []
    for rank, (code, data) in enumerate(ranked):
        parent_code = data[2]
        features = _wang2017_task_features_for_code(parent_code, code, data, args)
        scored.append((_wang2017_task_score(features), rank, code, features))
    chosen = sorted(scored)[:target]
    metrics["selected_tier_protection_total"] = sum(item[3].tier_protection for item in chosen)
    metrics["selected_stable_height_total"] = sum(item[3].target_stable_height for item in chosen)
    return [item[2] for item in chosen], metrics


def _append_wang_rollout_frontier(
    selected_codes: list[bytes],
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
    depth_cap: int,
    best_depth: dict[bytes, int] | None = None,
) -> set[bytes]:
    if getattr(args, "wang_rollout_inject_candidates", False):
        return set()
    if not selected_codes:
        return set()
    best_n = min(candidates[code][0] for code in selected_codes if code in candidates)
    if not _wang_rollout_layer_active(args, best_n):
        return set()
    width = max(0, int(getattr(args, "wang_rollout_width", 0) or 0))
    if width <= 0:
        return set()
    added: set[bytes] = set()
    selected_set = set(selected_codes)
    origins = sorted(
        [code for code in selected_codes if code in candidates],
        key=lambda code: _candidate_sort_key((code, candidates[code]), args),
    )
    for code in origins:
        if len(added) >= width:
            break
        n_value, depth, _parent = candidates[code]
        child = _wang_rollout_successor(code, n_value, depth, args)
        if child is None:
            continue
        child_code, child_data = child
        child_n, child_depth, _child_parent = child_data
        if child_code in selected_set or child_code in added:
            continue
        if child_depth + child_n > depth_cap:
            continue
        old_best_depth = best_depth.get(child_code) if best_depth is not None else None
        if old_best_depth is not None and child_depth >= old_best_depth:
            continue
        old = candidates.get(child_code)
        if old is not None and old[1] <= child_depth:
            continue
        candidates[child_code] = child_data
        selected_codes.append(child_code)
        added.add(child_code)
    return added


def _append_frg_compound_frontier(
    selected_codes: list[bytes],
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
    depth_cap: int,
    best_depth: dict[bytes, int] | None = None,
) -> set[bytes]:
    origins = [code for code in selected_codes if code in candidates]
    if not origins:
        return set()
    best_n = min(candidates[code][0] for code in origins)
    if not _frg_compound_escape_active(args, best_n):
        return set()
    if not _frg_compound_guard_accepts(args, best_n, origins, candidates):
        return set()
    width = max(0, int(getattr(args, "frg_compound_escape_width", 0) or 0))
    if width <= 0:
        return set()
    if not hasattr(args, "_frg_compound_appended_total"):
        args._frg_compound_appended_total = 0
    added: set[bytes] = set()
    selected_set = set(selected_codes)
    for code in sorted(origins, key=lambda item: _candidate_sort_key((item, candidates[item]), args)):
        if len(added) >= width:
            break
        n_value, depth, _parent_code = candidates[code]
        child = _frg_best_successor(code, n_value, depth, args)
        if child is None:
            continue
        child_code, child_data = child
        child_n, child_depth, _child_parent = child_data
        if child_code in selected_set or child_code in added:
            continue
        if child_depth + child_n > depth_cap:
            continue
        old_best_depth = best_depth.get(child_code) if best_depth is not None else None
        if old_best_depth is not None and child_depth >= old_best_depth:
            continue
        old = candidates.get(child_code)
        if old is not None and old[1] <= child_depth:
            continue
        candidates[child_code] = child_data
        selected_codes.append(child_code)
        added.add(child_code)
        args._frg_compound_appended_total += 1
    return added


def _inject_wang_rollout_candidates(
    frontier: list[tuple[bytes, int, int]],
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
    depth_cap: int,
    best_depth: dict[bytes, int] | None = None,
) -> set[bytes]:
    if not getattr(args, "wang_rollout_inject_candidates", False):
        return set()
    if not frontier:
        return set()
    best_n = min((data[0] for data in candidates.values()), default=min(item[1] for item in frontier))
    if not _wang_rollout_layer_active(args, best_n):
        return set()
    width = max(0, int(getattr(args, "wang_rollout_width", 0) or 0))
    if width <= 0:
        return set()
    marked: set[bytes] = set()
    origins = sorted(frontier, key=lambda item: (item[1], item[2], item[0]))
    for code, n_value, depth in origins:
        if len(marked) >= width:
            break
        child = _wang_rollout_successor(code, n_value, depth, args)
        if child is None:
            continue
        child_code, child_data = child
        child_n, child_depth, _child_parent = child_data
        if child_code in marked:
            continue
        if child_depth + child_n > depth_cap:
            continue
        old_best_depth = best_depth.get(child_code) if best_depth is not None else None
        if old_best_depth is not None and child_depth >= old_best_depth:
            continue
        old = candidates.get(child_code)
        if old is not None and old[1] <= child_depth:
            marked.add(child_code)
            continue
        candidates[child_code] = child_data
        marked.add(child_code)
    return marked


def _update_layer_candidate_metrics(
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
    cap: int,
) -> int:
    layer_best_n = min(data[0] for data in candidates.values())
    args._layer_best_n = layer_best_n
    best_n_depth = min(data[1] for data in candidates.values() if data[0] == layer_best_n)
    parent_counts: dict[bytes, int] = {}
    for _code, data in candidates.items():
        parent_counts[data[2]] = parent_counts.get(data[2], 0) + 1
    args._layer_candidate_count = len(candidates)
    args._layer_parent_count = len(parent_counts)
    args._layer_parent_top_share = (max(parent_counts.values()) / len(candidates)) if candidates else 0.0
    args._layer_depth_slack = cap - best_n_depth - layer_best_n
    args._frg_layout_available_stack_count = 0
    args._frg_layout_avg_height_pressure = 0.0
    args._frg_layout_max_height_ordered_risk = 0
    args._frg_layout_unblocked_bad_containers = 0
    args._frg_layout_reduction_receiver_stack_count = 0
    args._frg_layout_reduction_receiver_pressure = 0.0
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count > 0 and max_height > 0:
        best_code, _best_data = min(
            candidates.items(),
            key=lambda item: (item[1][0], item[1][1], item[0]),
        )
        layout = compute_frg_layout_signals(_code_to_stacks(best_code, stack_count, max_height), max_height)
        args._frg_layout_available_stack_count = layout.available_stack_count
        args._frg_layout_avg_height_pressure = layout.avg_height_pressure
        args._frg_layout_max_height_ordered_risk = layout.max_height_ordered_risk
        args._frg_layout_unblocked_bad_containers = layout.unblocked_bad_containers
        args._frg_layout_reduction_receiver_stack_count = layout.reduction_receiver_stack_count
        args._frg_layout_reduction_receiver_pressure = layout.reduction_receiver_pressure
    return layer_best_n


def _candidate_rank(item: tuple[bytes, tuple[int, int, bytes]], args: argparse.Namespace) -> tuple:
    code, (n_value, depth, parent_code) = item
    if args.priority == "astar":
        primary = float(depth + n_value)
    elif args.priority == "weighted_astar":
        primary = float(args.g_weight * depth + args.n_weight * n_value)
    elif args.priority == "target_guided":
        focus_n = int(getattr(args, "target_guided_focus_n", 8))
        layer_best_n = int(getattr(args, "_layer_best_n", n_value))
        if focus_n >= 0 and layer_best_n > focus_n:
            ledger_score = _candidate_feature_ledger_score(code, (n_value, depth, parent_code), args)
            return (n_value, depth, ledger_score, parent_code, code)
        target_gap = _candidate_target_gap(code, args)
        weighted_gap = float(getattr(args, "target_guided_weight", 0.0)) * target_gap
        return (n_value, depth, weighted_gap, target_gap, parent_code, code)
    else:
        primary = float(n_value * 1_000_000 + depth)
    return (primary, n_value, depth, parent_code, code)


def _frg_rank_tiebreak_active(item: tuple[bytes, tuple[int, int, bytes]], args: argparse.Namespace) -> bool:
    if not getattr(args, "frg_rank_tiebreak_enabled", False):
        return False
    focus_raw = getattr(args, "frg_rank_tiebreak_focus_n", -1)
    focus_n = -1 if focus_raw is None else int(focus_raw)
    layer_raw = getattr(args, "_layer_best_n", item[1][0])
    layer_best_n = int(item[1][0] if layer_raw is None else layer_raw)
    return focus_n < 0 or layer_best_n <= focus_n


def _frg_rank_tiebreak_value(item: tuple[bytes, tuple[int, int, bytes]], args: argparse.Namespace) -> float:
    if not _frg_rank_tiebreak_active(item, args):
        return 0.0
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return 0.0
    code, (n_value, depth, parent_code) = item
    before = _code_to_stacks(parent_code, stack_count, max_height)
    after = _code_to_stacks(code, stack_count, max_height)
    signals = compute_frg_transition_signals(before, after, max_height)
    base_score = frg_transition_score(signals, n_value=int(n_value), depth=int(depth))
    transition_only = base_score - float(int(n_value) * 1000 + int(depth))
    return float(getattr(args, "frg_rank_tiebreak_weight", 0.0) or 0.0) * transition_only


def _candidate_feature_ledger_score(
    code: bytes,
    data: tuple[int, int, bytes],
    args: argparse.Namespace,
) -> float:
    if not bool(getattr(args, "candidate_ledger_score_enabled", False)):
        return 0.0
    active_codes = getattr(args, "_candidate_ledger_active_codes", None)
    if active_codes is not None and code not in active_codes:
        return 0.0
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return 0.0

    _n_value, _depth, parent_code = data
    fixed_weight = float(getattr(args, "candidate_ledger_wang_fixed_weight", 0.0) or 0.0)
    pair_weight = float(getattr(args, "candidate_ledger_wang_pair_weight", 0.0) or 0.0)
    fill_weight = float(getattr(args, "candidate_ledger_frg_fill_weight", 0.0) or 0.0)
    reduce_weight = float(getattr(args, "candidate_ledger_frg_reduce_weight", 0.0) or 0.0)
    unblocked_weight = float(getattr(args, "candidate_ledger_frg_unblocked_weight", 0.0) or 0.0)
    assignment_weight = float(getattr(args, "candidate_ledger_frg_assignment_weight", 0.0) or 0.0)
    rollout_bonus = float(getattr(args, "candidate_ledger_rollout_bonus", 0.0) or 0.0)

    score = 0.0
    if fixed_weight:
        score -= fixed_weight * float(_compact_wang_fixed_prefix_score(code, stack_count, max_height))
    if pair_weight:
        score -= pair_weight * float(_compact_wang_fix_pair_score(code, stack_count, max_height))
    if fill_weight or reduce_weight or unblocked_weight or assignment_weight:
        before = _code_to_stacks(parent_code, stack_count, max_height)
        after = _code_to_stacks(code, stack_count, max_height)
        transition = compute_frg_transition_signals(before, after, max_height)
        score -= fill_weight * float(transition.fill_gain)
        score -= reduce_weight * float(transition.reduce_gain)
        score -= unblocked_weight * float(max(0, transition.unblocked_bad_delta))
        if assignment_weight:
            layout = compute_frg_layout_signals(after, max_height)
            score -= assignment_weight * float(layout.reduction_pop_order_assignment_score)
    if rollout_bonus and code in (getattr(args, "_wang_rollout_candidate_codes", set()) or set()):
        score -= rollout_bonus
    return score


def _candidate_feature_ledger_window_codes(
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
) -> set[bytes]:
    window = int(getattr(args, "candidate_ledger_rank_window", 512) or 0)
    if window <= 0 or len(candidates) <= window:
        return set(candidates)
    ranked = heapq.nsmallest(window, candidates.items(), key=_candidate_primary_key_func(args))
    return {code for code, _data in ranked}


def _candidate_primary_key_func(args: argparse.Namespace):
    if args.priority == "target_guided":
        return lambda item: _candidate_rank(item, args)
    if args.priority == "n_then_depth":
        if getattr(args, "frg_rank_tiebreak_enabled", False):
            return lambda item: (
                item[1][0],
                item[1][1],
                _frg_rank_tiebreak_value(item, args),
                item[1][2],
                item[0],
            ) if _frg_rank_tiebreak_active(item, args) else (item[1][0], item[1][1], item[1][2], item[0])
        return lambda item: (item[1][0], item[1][1], item[1][2], item[0])
    return lambda item: _candidate_rank(item, args)


def _candidate_sort_key(item: tuple[bytes, tuple[int, int, bytes]], args: argparse.Namespace) -> tuple:
    if args.priority == "target_guided":
        return _candidate_rank(item, args)
    if args.priority != "n_then_depth":
        return _candidate_rank(item, args)
    code, (n_value, depth, parent_code) = item
    ledger_score = _candidate_feature_ledger_score(code, (n_value, depth, parent_code), args)
    if _frg_rank_tiebreak_active(item, args):
        return (n_value, depth, ledger_score, _frg_rank_tiebreak_value(item, args), parent_code, code)
    return (n_value, depth, ledger_score, parent_code, code)


def _candidate_key_func(args: argparse.Namespace):
    if args.priority == "target_guided":
        return lambda item: _candidate_rank(item, args)
    if args.priority == "n_then_depth":
        if getattr(args, "frg_rank_tiebreak_enabled", False):
            return lambda item: (
                item[1][0],
                item[1][1],
                _candidate_feature_ledger_score(item[0], item[1], args),
                _frg_rank_tiebreak_value(item, args),
                item[1][2],
                item[0],
            ) if _frg_rank_tiebreak_active(item, args) else (
                item[1][0],
                item[1][1],
                _candidate_feature_ledger_score(item[0], item[1], args),
                item[1][2],
                item[0],
            )
        return lambda item: (
            item[1][0],
            item[1][1],
            _candidate_feature_ledger_score(item[0], item[1], args),
            item[1][2],
            item[0],
        )
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


def _rank_bucket_key_for_candidate(data: tuple[int, int, bytes], args: argparse.Namespace) -> tuple:
    n_value, depth, parent_code = data
    key = getattr(args, "rank_bucket_key", "n_depth_parent")
    if key == "n":
        return (n_value,)
    if key == "n_depth_parent":
        return (n_value, depth, parent_code)
    if key == "n_depth":
        return (n_value, depth)
    if key == "n_band_parent":
        band_size = max(1, int(getattr(args, "rank_bucket_n_band_size", 2)))
        return (n_value // band_size, parent_code)
    if key == "n_band":
        band_size = max(1, int(getattr(args, "rank_bucket_n_band_size", 2)))
        return (n_value // band_size, depth)
    if key == "parent":
        return (parent_code,)
    raise ValueError(f"unknown rank bucket key: {key}")


def _rank_bucket_topk_candidates(
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
) -> list[tuple[bytes, tuple[int, int, bytes]]]:
    topk = max(0, int(getattr(args, "rank_bucket_topk", 0) or 0))
    if topk <= 0:
        return sorted(candidates.items(), key=_candidate_key_func(args))
    if not hasattr(args, "_rank_pool_truncated_candidates"):
        args._rank_pool_truncated_candidates = 0

    buckets: dict[tuple, list[tuple[bytes, tuple[int, int, bytes]]]] = {}
    for item in candidates.items():
        buckets.setdefault(_rank_bucket_key_for_candidate(item[1], args), []).append(item)

    kept: list[tuple[bytes, tuple[int, int, bytes]]] = []
    key_func = _candidate_key_func(args)
    for bucket_items in buckets.values():
        if len(bucket_items) > topk:
            kept.extend(heapq.nsmallest(topk, bucket_items, key=key_func))
            args._rank_pool_truncated_candidates += len(bucket_items) - topk
        else:
            kept.extend(bucket_items)
    return sorted(kept, key=key_func)


def _add_target_guided_supplement(
    selected: dict[bytes, None],
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    effective_beam: int,
    best_n: int,
) -> None:
    width = max(0, int(getattr(args, "target_guided_supplement_width", 0) or 0))
    if width <= 0 or len(selected) >= effective_beam:
        return
    focus_n = int(getattr(args, "target_guided_focus_n", 8))
    if focus_n >= 0 and best_n > focus_n:
        return
    if not getattr(args, "_layout_target_groups", ()):
        return
    n_window = max(0, int(getattr(args, "target_guided_n_window", 0) or 0))
    remaining = [
        (code, data)
        for code, data in ranked
        if code not in selected and data[0] <= best_n + n_window
    ]
    if not remaining:
        return
    target = min(effective_beam, len(selected) + width)
    for code, _data in sorted(
        remaining,
        key=lambda item: (_candidate_target_gap(item[0], args), item[1][0], item[1][1], item[1][2], item[0]),
    ):
        selected.setdefault(code, None)
        if len(selected) >= target:
            break


def _add_wang_probe_supplement(
    selected: dict[bytes, None],
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    effective_beam: int,
    best_n: int,
) -> None:
    width = max(0, int(getattr(args, "wang_probe_supplement_width", 0) or 0))
    if width <= 0 or len(selected) >= effective_beam:
        return
    focus_n = int(getattr(args, "wang_probe_focus_n", 8))
    if focus_n >= 0 and best_n > focus_n:
        return
    if not _wang_probe_layer_active(args):
        return
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return
    n_window = max(0, int(getattr(args, "wang_probe_n_window", 0) or 0))
    remaining = [
        (code, data)
        for code, data in ranked
        if code not in selected and data[0] <= best_n + n_window
    ]
    if not remaining:
        return
    target = min(effective_beam, len(selected) + width)
    def sort_key(item: tuple[bytes, tuple[int, int, bytes]]) -> tuple:
        score, fix_pairs, fixed_prefix, target_gap = _candidate_wang_probe_score(item[0], args)
        return (-score, -fix_pairs, -fixed_prefix, target_gap, item[1][0], item[1][1], item[1][2], item[0])

    take = max(1, target - len(selected))
    for code, _data in heapq.nsmallest(take, remaining, key=sort_key):
        selected.setdefault(code, None)
        if len(selected) >= target:
            break


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
    for _nxt, nxt_n in _successors_with_n_uncached(code, n_value, args._stack_count, args._max_height):
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
    return _wang_feature_adjusted_group_score(score, rows, args)


def _scarcity_group_qubo_active(args: argparse.Namespace) -> bool:
    if not getattr(args, "scarcity_group_qubo_enabled", False):
        return False
    if float(getattr(args, "_initial_scarcity_risk", 0.0) or 0.0) < float(
        getattr(args, "scarcity_group_qubo_threshold", 0.95) or 0.95
    ):
        return False
    history = list(getattr(args, "_best_n_history", []) or [])
    best_n = int(getattr(args, "_layer_best_n", history[-1] if history else 0) or 0)
    if best_n < int(getattr(args, "scarcity_group_qubo_min_n", 0) or 0):
        return False
    initial_n = max(1, int(getattr(args, "_initial_n", best_n if best_n > 0 else 1)))
    min_ratio = float(getattr(args, "scarcity_group_qubo_min_best_n_ratio", 0.0) or 0.0)
    if min_ratio > 0.0 and (best_n / initial_n) < min_ratio:
        return False
    if int(getattr(args, "_layer_candidate_count", 0)) < int(
        getattr(args, "scarcity_group_qubo_min_candidates", 0) or 0
    ):
        return False
    window = max(2, int(getattr(args, "scarcity_group_qubo_plateau_window", 2) or 2))
    if len(history) < window:
        return False
    delta = max(0, int(getattr(args, "scarcity_group_qubo_plateau_delta", 0) or 0))
    return _plateau_active(history, window, delta)


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
    return _scarcity_group_qubo_active(args)


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
        _qubo_sampler_config(
            args,
            num_reads=args.qubo_reads,
            random_seed=args.seed + 5003 + args._group_qubo_calls_done,
            time_limit_sec=args.qubo_time_limit,
            sa_iterations=args.qubo_iterations,
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
        _qubo_sampler_config(
            args,
            num_reads=args.qubo_reads,
            random_seed=args.seed + 7001 + args._group_qubo_calls_done,
            time_limit_sec=args.qubo_time_limit,
            sa_iterations=args.qubo_iterations,
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


def _record_adaptive_width(args: argparse.Namespace, width: int, reason: str) -> int:
    width = max(1, int(width))
    setattr(args, "_adaptive_last_width", width)
    setattr(args, "_adaptive_last_reason", reason)
    if reason.startswith("contract"):
        args._adaptive_contract_layers = int(getattr(args, "_adaptive_contract_layers", 0)) + 1
    elif reason.startswith("mid"):
        args._adaptive_mid_layers = int(getattr(args, "_adaptive_mid_layers", 0)) + 1
    elif reason.startswith("expand"):
        args._adaptive_expand_layers = int(getattr(args, "_adaptive_expand_layers", 0)) + 1
    else:
        args._adaptive_wide_layers = int(getattr(args, "_adaptive_wide_layers", 0)) + 1
    return width


def _tiered_adaptive_frontier_width(args: argparse.Namespace, best_n: int) -> tuple[int, str]:
    high_n = int(getattr(args, "adaptive_high_n", 16))
    low_n = int(getattr(args, "adaptive_low_n", 8))
    high_frontier = max(1, int(getattr(args, "adaptive_high_frontier", args.beam_width)))
    mid_frontier = max(1, int(getattr(args, "adaptive_mid_frontier", args.beam_width)))
    low_frontier = max(1, int(getattr(args, "adaptive_low_frontier", args.beam_width)))
    if best_n > high_n:
        return high_frontier, "contract_high_n"
    if best_n <= low_n:
        return low_frontier, "wide_low_n"
    return mid_frontier, "mid_n"


def _plateau_active(history: list[int], window: int, delta: int) -> bool:
    if window <= 1 or len(history) < window:
        return False
    recent = history[-window:]
    return max(recent) - min(recent) <= delta


def _survivor_parent_core_cap(args: argparse.Namespace, best_n: int) -> tuple[int, str]:
    base_cap = max(0, int(getattr(args, "parent_core_cap", 0) or 0))
    if not getattr(args, "survivor_parent_cap_enabled", False) or base_cap <= 0:
        setattr(args, "_survivor_parent_cap_last", base_cap)
        setattr(args, "_survivor_parent_cap_last_reason", "survivor_disabled")
        setattr(args, "_survivor_parent_cap_last_signal", "")
        return base_cap, "survivor_disabled"

    min_n = max(0, int(getattr(args, "survivor_parent_cap_min_n", 0) or 0))
    if min_n > 0 and best_n < min_n:
        args._survivor_parent_cap_active = False
        setattr(args, "_survivor_parent_cap_last", base_cap)
        setattr(args, "_survivor_parent_cap_last_reason", "survivor_tail")
        setattr(args, "_survivor_parent_cap_last_signal", "")
        return base_cap, "survivor_tail"

    window = max(1, int(getattr(args, "survivor_parent_cap_window", 1) or 1))
    history = list(getattr(args, "_survivor_dead_parent_ratio_history", []) or [])
    if len(history) < window:
        setattr(args, "_survivor_parent_cap_last", base_cap)
        setattr(args, "_survivor_parent_cap_last_reason", "survivor_warmup")
        setattr(args, "_survivor_parent_cap_last_signal", "")
        return base_cap, "survivor_warmup"

    signal = sum(float(value) for value in history[-window:]) / window
    setattr(args, "_survivor_parent_cap_last_signal", signal)
    trigger = float(getattr(args, "survivor_parent_cap_trigger_ratio", 0.70) or 0.70)
    release = float(getattr(args, "survivor_parent_cap_release_ratio", 0.55) or 0.55)
    target_cap = int(getattr(args, "survivor_parent_cap_value", max(1, base_cap - 1)) or max(1, base_cap - 1))
    target_cap = max(1, min(base_cap, target_cap))
    active = bool(getattr(args, "_survivor_parent_cap_active", False))

    if active and signal <= release:
        args._survivor_parent_cap_active = False
        cap, reason = base_cap, "survivor_release"
    elif active:
        cap, reason = target_cap, "survivor_hold"
    elif signal >= trigger:
        args._survivor_parent_cap_active = True
        cap, reason = target_cap, "survivor_contract"
    else:
        cap, reason = base_cap, "survivor_stable"

    setattr(args, "_survivor_parent_cap_last", cap)
    setattr(args, "_survivor_parent_cap_last_reason", reason)
    if cap < base_cap:
        args._survivor_parent_cap_layers = int(getattr(args, "_survivor_parent_cap_layers", 0)) + 1
    return cap, reason


def _record_survivor_parent_cap_signal(args: argparse.Namespace, metrics: dict[str, int | float]) -> None:
    if not getattr(args, "survivor_parent_cap_enabled", False):
        return
    history = list(getattr(args, "_survivor_dead_parent_ratio_history", []) or [])
    history.append(float(metrics.get("dead_parent_ratio", 0.0) or 0.0))
    window = max(1, int(getattr(args, "survivor_parent_cap_window", 1) or 1))
    max_history = max(8, window * 4)
    args._survivor_dead_parent_ratio_history = history[-max_history:]


def _signal_adaptive_frontier_width(args: argparse.Namespace, best_n: int) -> tuple[int, str]:
    high_n = int(getattr(args, "adaptive_high_n", 16))
    low_n = int(getattr(args, "adaptive_low_n", 8))
    high_frontier = max(1, int(getattr(args, "adaptive_high_frontier", args.beam_width)))
    mid_frontier = max(1, int(getattr(args, "adaptive_mid_frontier", args.beam_width)))
    low_frontier = max(1, int(getattr(args, "adaptive_low_frontier", args.beam_width)))
    expand_frontier = max(low_frontier, int(getattr(args, "adaptive_expand_frontier", low_frontier)))
    base_width = max(1, int(getattr(args, "beam_width", low_frontier)))
    candidate_count = max(0, int(getattr(args, "_layer_candidate_count", 0)))
    candidate_pressure = candidate_count / base_width
    contract_pressure = float(getattr(args, "adaptive_contract_pressure", 0.0) or 0.0)
    mid_pressure = float(getattr(args, "adaptive_mid_pressure", 0.0) or 0.0)

    history = list(getattr(args, "_best_n_history", []) or [])
    plateau_window = max(0, int(getattr(args, "adaptive_plateau_window", 0) or 0))
    plateau_delta = max(0, int(getattr(args, "adaptive_plateau_delta", 0) or 0))
    cap_slack = int(getattr(args, "adaptive_cap_slack", 0) or 0)
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    large_stack_min = int(getattr(args, "adaptive_large_stack_min_stacks", 0) or 0)
    if large_stack_min > 0 and stack_count >= large_stack_min:
        large_expand = int(getattr(args, "adaptive_large_stack_expand_frontier", 0) or 0)
        if large_expand > 0:
            expand_frontier = max(expand_frontier, large_expand)
        large_window = int(getattr(args, "adaptive_large_stack_plateau_window", 0) or 0)
        if large_window > 0:
            plateau_window = large_window if plateau_window <= 1 else min(plateau_window, large_window)
        large_slack = int(getattr(args, "adaptive_large_stack_cap_slack", 0) or 0)
        if large_slack > 0:
            cap_slack = max(cap_slack, large_slack)
    depth_slack = int(getattr(args, "_layer_depth_slack", 10**9))
    if cap_slack > 0 and depth_slack <= cap_slack and _plateau_active(history, plateau_window, plateau_delta):
        return expand_frontier, "expand_plateau_tight_slack"

    parent_limit = float(getattr(args, "adaptive_parent_top_share", 0.0) or 0.0)
    if parent_limit > 0.0 and float(getattr(args, "_layer_parent_top_share", 0.0)) >= parent_limit:
        return expand_frontier, "expand_parent_collapse"

    max_duplication = float(getattr(args, "adaptive_max_duplication_ratio", 1.0))
    if max_duplication >= 0.0:
        duplication = float(getattr(args, "_priority_duplication_ratio", 0.0))
        if duplication > max_duplication + 1e-12:
            return low_frontier, "wide_priority_duplication"

    if best_n <= low_n:
        return low_frontier, "wide_low_n"

    initial_n = max(1, int(getattr(args, "_initial_n", best_n if best_n > 0 else 1)))
    min_ratio = float(getattr(args, "adaptive_min_best_n_ratio", 0.0) or 0.0)
    if min_ratio > 0.0 and (best_n / initial_n) < min_ratio:
        return low_frontier, "wide_late_search"

    if best_n > high_n and contract_pressure > 0.0 and candidate_pressure >= contract_pressure:
        return high_frontier, "contract_high_pressure"
    if best_n > low_n and mid_pressure > 0.0 and candidate_pressure >= mid_pressure:
        return mid_frontier, "mid_candidate_pressure"
    return low_frontier, "wide_low_pressure"


def _adaptive_frontier_width(args: argparse.Namespace, best_n: int) -> int:
    if getattr(args, "adaptive_controller", "tiered") == "signal_v1":
        width, reason = _signal_adaptive_frontier_width(args, best_n)
    else:
        width, reason = _tiered_adaptive_frontier_width(args, best_n)
    return _record_adaptive_width(args, width, reason)


def _layer_start_active(args: argparse.Namespace, field: str) -> bool:
    start_layer = int(getattr(args, field, 0) or 0)
    if start_layer <= 0:
        return True
    return int(getattr(args, "_layer_index", 0) or 0) >= start_layer


def _select_enhanced_qubo_frontier(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    candidate_count: int,
    args: argparse.Namespace,
    beam_width: int | None = None,
    core_width: int | None = None,
    qubo_supplement_width: int | None = None,
) -> tuple[list[bytes], set[bytes], int, int, float]:
    selected: dict[bytes, None] = {}
    selected_source_by_code: dict[bytes, str] = {}
    selected_by_qubo: set[bytes] = set()
    args._frontier_trace_last_context = getattr(args, "_frontier_trace_selection_context", "standard")
    effective_beam = max(1, min(beam_width if beam_width is not None else args.beam_width, len(ranked)))
    effective_core = core_width if core_width is not None else args.core_width
    effective_core = max(0, min(effective_core, effective_beam))
    effective_qubo_supplement = (
        qubo_supplement_width if qubo_supplement_width is not None else args.qubo_supplement_width
    )
    best_n = int(ranked[0][1][0]) if ranked else 0
    effective_parent_core_cap, _parent_cap_reason = _survivor_parent_core_cap(args, best_n)

    core_target = effective_core

    def add_deterministic_until(target_len: int, source: str = "core") -> None:
        target_len = max(0, min(target_len, effective_beam))
        if len(selected) >= target_len:
            return
        if effective_parent_core_cap > 0:
            parent_counts: dict[bytes, int] = {}
            for code, data in ranked:
                if code in selected:
                    continue
                parent_code = data[2]
                if parent_counts.get(parent_code, 0) >= effective_parent_core_cap:
                    continue
                selected[code] = None
                selected_source_by_code.setdefault(code, source)
                parent_counts[parent_code] = parent_counts.get(parent_code, 0) + 1
                if len(selected) >= target_len:
                    break
        for code, _data in ranked:
            if code not in selected:
                selected[code] = None
                selected_source_by_code.setdefault(code, source)
            if len(selected) >= target_len:
                break

    if not args.qubo_first:
        add_deterministic_until(core_target)
    _add_tail_lookahead_path_supplement(
        selected,
        selected_source_by_code,
        ranked,
        args,
        effective_beam,
    )
    _add_tail_risk_path_supplement(
        selected,
        selected_source_by_code,
        ranked,
        args,
        effective_beam,
    )
    _add_tail_risk_reserve(
        selected,
        selected_source_by_code,
        ranked,
        args,
        effective_beam,
        int(best_n),
    )

    qcalls = qvars = 0
    qtime = 0.0
    best_n = ranked[0][1][0] if ranked else math.inf
    _add_target_guided_supplement(selected, ranked, args, effective_beam, int(best_n))
    _add_wang_probe_supplement(selected, ranked, args, effective_beam, int(best_n))
    pair_width = max(0, int(getattr(args, "wang_pair_qubo_supplement_width", 0) or 0))
    pair_focus_n = int(getattr(args, "wang_pair_qubo_focus_n", 8))
    if (
        getattr(args, "wang_pair_qubo_enabled", False)
        and args.use_qubo
        and pair_width > 0
        and len(selected) < effective_beam
        and _layer_start_active(args, "wang_pair_qubo_start_layer")
        and (pair_focus_n < 0 or int(best_n) <= pair_focus_n)
    ):
        n_window = max(0, int(getattr(args, "wang_pair_qubo_n_window", 1) or 0))
        remaining = [
            (code, data)
            for code, data in ranked
            if code not in selected and data[0] <= int(best_n) + n_window
        ]
        target = min(pair_width, effective_beam - len(selected), int(getattr(args, "wang_pair_qubo_select", pair_width) or pair_width))
        boosted, add_qvars, add_qtime = _select_wang_pair_qubo_supplement(remaining, args, target)
        if add_qvars:
            if not hasattr(args, "_wang_pair_qubo_calls_done"):
                args._wang_pair_qubo_calls_done = 0
            if not hasattr(args, "_wang_pair_qubo_vars"):
                args._wang_pair_qubo_vars = 0
            if not hasattr(args, "_wang_pair_qubo_time"):
                args._wang_pair_qubo_time = 0.0
            qcalls += 1
            qvars += add_qvars
            qtime += add_qtime
            args._wang_pair_qubo_calls_done += 1
            args._wang_pair_qubo_vars += add_qvars
            args._wang_pair_qubo_time += add_qtime
            for code in boosted:
                if code not in selected:
                    selected_by_qubo.add(code)
                    selected_source_by_code.setdefault(code, "wang_pair_qubo")
                    selected[code] = None
                else:
                    selected.setdefault(code, None)
                if len(selected) >= effective_beam:
                    break
    _add_tail_lookahead_supplement(
        selected,
        selected_source_by_code,
        ranked,
        args,
        effective_beam,
        int(best_n),
    )
    frg_width = max(0, int(getattr(args, "frg_candidate_supplement_width", 0) or 0))
    if (
        getattr(args, "frg_candidate_supplement_enabled", False)
        and frg_width > 0
        and len(selected) < effective_beam
        and (
            not getattr(args, "frg_candidate_scarcity_gate_enabled", False)
            or _frg_scarcity_gate_active(args, int(best_n))
        )
    ):
        target = min(frg_width, effective_beam - len(selected))
        remaining = [(code, data) for code, data in ranked if code not in selected]
        boosted = _select_frg_candidate_supplement(remaining, args, target)
        if boosted:
            if not hasattr(args, "_frg_candidate_selected_total"):
                args._frg_candidate_selected_total = 0
            for code in boosted:
                before = len(selected)
                selected.setdefault(code, None)
                if len(selected) > before:
                    selected_source_by_code.setdefault(code, "frg_candidate")
                    args._frg_candidate_selected_total += 1
                if len(selected) >= effective_beam:
                    break
    dlts_width = max(0, int(getattr(args, "dlts_value_supplement_width", 0) or 0))
    if (
        getattr(args, "dlts_value_supplement_enabled", False)
        and dlts_width > 0
        and len(selected) < effective_beam
    ):
        target = min(dlts_width, effective_beam - len(selected))
        remaining = [(code, data) for code, data in ranked if code not in selected]
        boosted = _select_dlts_value_supplement(remaining, args, target)
        if boosted:
            if not hasattr(args, "_dlts_value_selected_total"):
                args._dlts_value_selected_total = 0
            for code in boosted:
                before = len(selected)
                selected.setdefault(code, None)
                if len(selected) > before:
                    selected_source_by_code.setdefault(code, "dlts_value")
                    args._dlts_value_selected_total += 1
                if len(selected) >= effective_beam:
                    break
    bs_frg_width = max(0, int(getattr(args, "bs_frg_qubo_supplement_width", 0) or 0))
    bs_frg_focus_n = int(getattr(args, "bs_frg_qubo_focus_n", -1))
    if (
        getattr(args, "bs_frg_qubo_enabled", False)
        and args.use_qubo
        and bs_frg_width > 0
        and len(selected) < effective_beam
        and candidate_count >= max(1, int(getattr(args, "bs_frg_qubo_min_candidates", 64) or 64))
        and (bs_frg_focus_n < 0 or int(best_n) <= bs_frg_focus_n)
    ):
        n_window = max(0, int(getattr(args, "bs_frg_qubo_n_window", 2) or 0))
        remaining = [
            (code, data)
            for code, data in ranked
            if code not in selected and data[0] <= int(best_n) + n_window
        ]
        target = min(bs_frg_width, effective_beam - len(selected), len(remaining))
        boosted, add_qvars, add_qtime = _select_bs_frg_qubo_supplement(remaining, args, target)
        if add_qvars:
            if not hasattr(args, "_bs_frg_qubo_calls_done"):
                args._bs_frg_qubo_calls_done = 0
            if not hasattr(args, "_bs_frg_qubo_vars"):
                args._bs_frg_qubo_vars = 0
            if not hasattr(args, "_bs_frg_qubo_time"):
                args._bs_frg_qubo_time = 0.0
            if not hasattr(args, "_bs_frg_qubo_selected_total"):
                args._bs_frg_qubo_selected_total = 0
            qcalls += 1
            qvars += add_qvars
            qtime += add_qtime
            args._bs_frg_qubo_calls_done += 1
            args._bs_frg_qubo_vars += add_qvars
            args._bs_frg_qubo_time += add_qtime
            for code in boosted:
                if code not in selected:
                    selected_by_qubo.add(code)
                    selected_source_by_code.setdefault(code, "bs_frg_qubo")
                before = len(selected)
                selected.setdefault(code, None)
                if len(selected) > before:
                    args._bs_frg_qubo_selected_total += 1
                if len(selected) >= effective_beam:
                    break
    unified_width = max(0, int(getattr(args, "unified_wang_qubo_supplement_width", 0) or 0))
    unified_focus_n_value = getattr(args, "unified_wang_qubo_focus_n", None)
    if unified_focus_n_value is None:
        unified_focus_n_value = 8
    unified_focus_n = int(unified_focus_n_value)
    if (
        getattr(args, "unified_wang_qubo_enabled", False)
        and args.use_qubo
        and unified_width > 0
        and len(selected) < effective_beam
        and (unified_focus_n < 0 or int(best_n) <= unified_focus_n)
    ):
        n_window = max(0, int(getattr(args, "unified_wang_qubo_n_window", 1) or 0))
        remaining = [
            (code, data)
            for code, data in ranked
            if code not in selected and data[0] <= int(best_n) + n_window
        ]
        target = min(
            unified_width,
            effective_beam - len(selected),
            int(getattr(args, "unified_wang_qubo_select", unified_width) or unified_width),
        )
        boosted, add_qvars, add_qtime, metrics = _select_unified_wang_qubo_supplement(remaining, args, target)
        if add_qvars:
            if not hasattr(args, "_unified_wang_qubo_calls_done"):
                args._unified_wang_qubo_calls_done = 0
            if not hasattr(args, "_unified_wang_qubo_vars"):
                args._unified_wang_qubo_vars = 0
            if not hasattr(args, "_unified_wang_qubo_time"):
                args._unified_wang_qubo_time = 0.0
            if not hasattr(args, "_unified_wang_fixed_prefix_selected"):
                args._unified_wang_fixed_prefix_selected = 0
            if not hasattr(args, "_unified_wang_fix_pair_selected"):
                args._unified_wang_fix_pair_selected = 0
            qcalls += 1
            qvars += add_qvars
            qtime += add_qtime
            args._unified_wang_qubo_calls_done += 1
            args._unified_wang_qubo_vars += add_qvars
            args._unified_wang_qubo_time += add_qtime
            args._unified_wang_fixed_prefix_selected += int(metrics.get("fixed_prefix_selected", 0))
            args._unified_wang_fix_pair_selected += int(metrics.get("fix_pair_selected", 0))
            for code in boosted:
                if code not in selected:
                    selected_by_qubo.add(code)
                selected.setdefault(code, None)
                if len(selected) >= effective_beam:
                    break
    pareto_width = max(0, int(getattr(args, "pareto_union_seed_width", 0) or 0)) + max(
        0,
        int(getattr(args, "pareto_union_qubo_width", 0) or 0),
    )
    if (
        getattr(args, "pareto_union_enabled", False)
        and args.use_qubo
        and pareto_width > 0
        and len(selected) < effective_beam
        and candidate_count >= max(1, int(getattr(args, "union_diversity_qubo_min_candidates", 64) or 64))
    ):
        _ensure_pareto_union_telemetry(args)
        n_window = max(0, int(getattr(args, "union_diversity_qubo_n_window", 2) or 0))
        remaining = [
            (code, data)
            for code, data in ranked
            if code not in selected and data[0] <= int(best_n) + n_window
        ]
        target = min(pareto_width, effective_beam - len(selected), len(remaining))
        boosted, pareto_qubo_codes, add_qcalls, add_qvars, add_qtime = _select_pareto_union_supplement(
            remaining,
            args,
            target,
        )
        if boosted:
            args._pareto_union_layers += 1
            if add_qcalls:
                qcalls += add_qcalls
                qvars += add_qvars
                qtime += add_qtime
                args._pareto_union_qubo_calls_done += add_qcalls
                args._pareto_union_qubo_vars += add_qvars
                args._pareto_union_qubo_time += add_qtime
            for code in boosted:
                if code not in selected:
                    if code in pareto_qubo_codes:
                        selected_by_qubo.add(code)
                        selected_source_by_code.setdefault(code, "pareto_union_qubo")
                    else:
                        selected_source_by_code.setdefault(code, "pareto_union_seed")
                selected.setdefault(code, None)
                if len(selected) >= effective_beam:
                    break
    union_width = max(0, int(getattr(args, "union_diversity_qubo_supplement_width", 0) or 0))
    union_focus_n = int(getattr(args, "union_diversity_qubo_focus_n", -1) or -1)
    if (
        getattr(args, "union_diversity_qubo_enabled", False)
        and args.use_qubo
        and union_width > 0
        and len(selected) < effective_beam
        and candidate_count >= max(1, int(getattr(args, "union_diversity_qubo_min_candidates", 64) or 64))
        and (union_focus_n < 0 or int(best_n) <= union_focus_n)
    ):
        n_window = max(0, int(getattr(args, "union_diversity_qubo_n_window", 2) or 0))
        remaining = [
            (code, data)
            for code, data in ranked
            if code not in selected and data[0] <= int(best_n) + n_window
        ]
        target = min(
            union_width,
            effective_beam - len(selected),
            int(getattr(args, "union_diversity_qubo_select", union_width) or union_width),
            len(remaining),
        )
        boosted, add_qvars, add_qtime = _select_union_diversity_qubo_supplement(remaining, args, target)
        if add_qvars:
            if not hasattr(args, "_union_diversity_qubo_calls_done"):
                args._union_diversity_qubo_calls_done = 0
            if not hasattr(args, "_union_diversity_qubo_vars"):
                args._union_diversity_qubo_vars = 0
            if not hasattr(args, "_union_diversity_qubo_time"):
                args._union_diversity_qubo_time = 0.0
            if not hasattr(args, "_union_diversity_qubo_selected_total"):
                args._union_diversity_qubo_selected_total = 0
            if not hasattr(args, "_union_rollout_probe_selected_total"):
                args._union_rollout_probe_selected_total = 0
            qcalls += 1
            qvars += add_qvars
            qtime += add_qtime
            args._union_diversity_qubo_calls_done += 1
            args._union_diversity_qubo_vars += add_qvars
            args._union_diversity_qubo_time += add_qtime
            probe_results = getattr(args, "_union_rollout_probe_current_results", {}) or {}
            for code in boosted:
                if code not in selected:
                    selected_by_qubo.add(code)
                    selected_source_by_code.setdefault(code, "union_diversity_qubo")
                before = len(selected)
                selected.setdefault(code, None)
                if len(selected) > before:
                    args._union_diversity_qubo_selected_total += 1
                    if code in probe_results:
                        args._union_rollout_probe_selected_total += 1
                if len(selected) >= effective_beam:
                    break
    feasibility_width = max(0, int(getattr(args, "feasibility_escape_supplement_width", 0) or 0))
    if (
        getattr(args, "feasibility_escape_enabled", False)
        and feasibility_width > 0
        and len(selected) < effective_beam
    ):
        gate_reason = _feasibility_escape_gate_reason(args, int(best_n))
        if not hasattr(args, "_feasibility_escape_gate_reasons"):
            args._feasibility_escape_gate_reasons = {}
        args._feasibility_escape_gate_reasons[gate_reason] = (
            int(args._feasibility_escape_gate_reasons.get(gate_reason, 0)) + 1
        )
        if gate_reason == "active":
            if not hasattr(args, "_feasibility_escape_active_layers"):
                args._feasibility_escape_active_layers = 0
            args._feasibility_escape_active_layers += 1
            n_window = max(0, int(getattr(args, "feasibility_escape_n_window", 1) or 0))
            remaining = [
                (code, data)
                for code, data in ranked
                if code not in selected and data[0] <= int(best_n) + n_window
            ]
            target = min(feasibility_width, effective_beam - len(selected))
            boosted = _select_feasibility_escape_supplement(remaining, args, target)
            if boosted:
                if not hasattr(args, "_feasibility_escape_calls_done"):
                    args._feasibility_escape_calls_done = 0
                if not hasattr(args, "_feasibility_escape_added_total"):
                    args._feasibility_escape_added_total = 0
                args._feasibility_escape_calls_done += 1
                for code in boosted:
                    before = len(selected)
                    selected.setdefault(code, None)
                    if len(selected) > before:
                        args._feasibility_escape_added_total += 1
                    if len(selected) >= effective_beam:
                        break
    wang2017_width = max(0, int(getattr(args, "wang2017_task_supplement_width", 0) or 0))
    wang2017_focus_n = int(getattr(args, "wang2017_task_focus_n", 8))
    if (
        getattr(args, "wang2017_task_enabled", False)
        and wang2017_width > 0
        and len(selected) < effective_beam
        and (wang2017_focus_n < 0 or int(best_n) <= wang2017_focus_n)
        and _wang2017_task_layer_active(args, int(best_n))
    ):
        if not hasattr(args, "_wang2017_task_active_layers"):
            args._wang2017_task_active_layers = 0
        if not hasattr(args, "_wang2017_task_calls_done"):
            args._wang2017_task_calls_done = 0
        if not hasattr(args, "_wang2017_task_added_total"):
            args._wang2017_task_added_total = 0
        if not hasattr(args, "_wang2017_task_selected_tier_protection_total"):
            args._wang2017_task_selected_tier_protection_total = 0
        if not hasattr(args, "_wang2017_task_selected_stable_height_total"):
            args._wang2017_task_selected_stable_height_total = 0
        args._wang2017_task_active_layers += 1
        n_window = max(0, int(getattr(args, "wang2017_task_n_window", 1) or 0))
        remaining = [
            (code, data)
            for code, data in ranked
            if code not in selected and data[0] <= int(best_n) + n_window
        ]
        target = min(wang2017_width, effective_beam - len(selected))
        boosted, metrics = _select_wang2017_task_supplement(remaining, args, target)
        args._wang2017_task_calls_done += 1
        args._wang2017_task_selected_tier_protection_total += int(
            metrics.get("selected_tier_protection_total", 0)
        )
        args._wang2017_task_selected_stable_height_total += int(
            metrics.get("selected_stable_height_total", 0)
        )
        for code in boosted:
            before = len(selected)
            selected.setdefault(code, None)
            if len(selected) > before:
                args._wang2017_task_added_total += 1
            if len(selected) >= effective_beam:
                break
    qubo_enabled = (
        args.use_qubo
        and candidate_count >= args.qubo_min_candidates
        and getattr(args, "_qubo_calls_done", 0) < args.max_qubo_calls
        and _layer_start_active(args, "qubo_start_layer")
        and (args.qubo_focus_n < 0 or best_n <= args.qubo_focus_n)
    )
    if qubo_enabled and args.qubo_mode == "supplement":
        boosted, add_qvars, add_qtime = _select_qubo_boosted(dict(ranked), args)
        if add_qvars:
            if not hasattr(args, "_qubo_calls_done"):
                args._qubo_calls_done = 0
            qcalls += 1
            qvars += add_qvars
            qtime += add_qtime
            args._qubo_calls_done += 1
            for code in boosted:
                if code not in selected:
                    selected_by_qubo.add(code)
                    selected_source_by_code.setdefault(
                        code,
                        str(getattr(args, "_frontier_trace_qubo_source", "qubo")),
                    )
                    selected[code] = None
                else:
                    selected.setdefault(code, None)
                if len(selected) >= effective_beam:
                    break
    elif qubo_enabled and args.qubo_mode in {"focused_multi", "majority_multi"}:
        ranked_remaining = [(code, data) for code, data in ranked if code not in selected]
        cursor = 0
        cohorts_left = max(1, args.qubo_cohorts_per_layer)
        stride = max(1, args.qubo_cohort_stride or args.max_qubo_vars)
        target_qubo = effective_beam
        if args.qubo_mode == "majority_multi":
            target_qubo = max(1, min(effective_beam - len(selected), int(round(effective_beam * args.qubo_target_share))))
        while (
            cohorts_left > 0
            and cursor < len(ranked_remaining)
            and len(selected) < effective_beam
            and (args.qubo_mode != "majority_multi" or len(selected_by_qubo) < target_qubo)
            and getattr(args, "_qubo_calls_done", 0) < args.max_qubo_calls
        ):
            cohort = ranked_remaining[cursor : cursor + max(1, args.max_qubo_vars)]
            cursor += stride
            if len(cohort) < args.qubo_min_candidates:
                continue
            remaining = effective_beam - len(selected)
            q_remaining = max(1, target_qubo - len(selected_by_qubo)) if args.qubo_mode == "majority_multi" else remaining
            qargs = SimpleNamespace(**vars(args))
            qargs.qubo_core_width = 0
            qargs.qubo_supplement_width = max(1, min(effective_qubo_supplement, remaining, q_remaining, len(cohort) - 1))
            boosted, add_qvars, add_qtime = _select_qubo_boosted(dict(cohort), qargs)
            if not add_qvars:
                continue
            cohorts_left -= 1
            qcalls += 1
            qvars += add_qvars
            qtime += add_qtime
            if not hasattr(args, "_qubo_calls_done"):
                args._qubo_calls_done = 0
            args._qubo_calls_done += 1
            for code in boosted:
                if code not in selected:
                    selected_by_qubo.add(code)
                    selected_source_by_code.setdefault(
                        code,
                        str(getattr(args, "_frontier_trace_qubo_source", "qubo")),
                    )
                    selected[code] = None
                else:
                    selected.setdefault(code, None)
                if len(selected) >= effective_beam:
                    break

    best_n = ranked[0][1][0] if ranked else math.inf
    group_qubo_enabled = (
        args.use_qubo
        and getattr(args, "group_qubo_supplement", False)
        and candidate_count >= args.qubo_min_candidates
        and getattr(args, "_group_qubo_calls_done", 0) < args.max_group_qubo_calls
        and _layer_start_active(args, "qubo_start_layer")
        and (args.qubo_focus_n < 0 or best_n <= args.qubo_focus_n)
        and _group_qubo_risk_active(args)
        and len(selected) < effective_beam
    )
    if group_qubo_enabled:
        ranked_remaining = [(code, data) for code, data in ranked if code not in selected]
        remaining = effective_beam - len(selected)
        boosted, add_qvars, add_qtime = _select_group_qubo_supplement(
            ranked_remaining,
            selected,
            args,
            remaining,
        )
        if add_qvars:
            if not hasattr(args, "_group_qubo_calls_done"):
                args._group_qubo_calls_done = 0
            if not hasattr(args, "_group_qubo_vars"):
                args._group_qubo_vars = 0
            if not hasattr(args, "_group_qubo_time"):
                args._group_qubo_time = 0.0
            if not hasattr(args, "_group_qubo_selected_total"):
                args._group_qubo_selected_total = 0
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
                if len(selected) >= effective_beam:
                    break

    if args.qubo_first:
        add_deterministic_until(len(selected) + core_target, source="qubo_first_core")

    for code, _data in ranked:
        if code not in selected:
            selected[code] = None
            selected_source_by_code.setdefault(code, "tail_fill")
        if len(selected) >= effective_beam:
            break
    selected_codes = list(selected.keys())[:effective_beam]
    selected_set = set(selected_codes)
    args._frontier_trace_last_selected_sources = {
        code: source for code, source in selected_source_by_code.items() if code in selected_set
    }
    return selected_codes, selected_by_qubo & selected_set, qcalls, qvars, qtime


def _rescue_layer_active(args: argparse.Namespace, best_n: int) -> bool:
    return _standard_rescue_layer_active(args, best_n) or _scarcity_inner_rescue_active(args, best_n)


def _standard_rescue_layer_active(args: argparse.Namespace, best_n: int) -> bool:
    if not getattr(args, "rescue_enabled", False):
        return False
    if best_n < int(getattr(args, "rescue_min_n", 0)):
        return False
    if int(getattr(args, "_rescue_layers_used", 0)) >= int(getattr(args, "rescue_max_layers", 0)):
        return False
    if int(getattr(args, "_layer_candidate_count", 0)) < int(getattr(args, "rescue_min_candidates", 0)):
        return False
    if int(getattr(args, "_layer_depth_slack", 10**9)) > int(getattr(args, "rescue_cap_slack", 0)):
        return False

    window = max(2, int(getattr(args, "rescue_plateau_window", 4)))
    history = getattr(args, "_best_n_history", [])
    if len(history) < window:
        return False
    recent = history[-window:]
    return max(recent) - min(recent) <= int(getattr(args, "rescue_plateau_delta", 0))


def _scarcity_inner_rescue_active(args: argparse.Namespace, best_n: int) -> bool:
    if not getattr(args, "scarcity_inner_rescue_enabled", False):
        return False
    if not getattr(args, "rescue_enabled", False):
        _record_scarcity_inner_rescue_gate_reason(args, "rescue_disabled")
        return False
    if not getattr(args, "_scheduler_state_requested_rescue", False):
        return False
    if int(getattr(args, "_rescue_layers_used", 0)) >= int(getattr(args, "rescue_max_layers", 0)):
        _record_scarcity_inner_rescue_gate_reason(args, "max_layers")
        return False
    if int(getattr(args, "_initial_scarcity_n0", 0) or 0) < int(
        getattr(args, "scarcity_inner_rescue_min_n", 0) or 0
    ):
        _record_scarcity_inner_rescue_gate_reason(args, "initial_n")
        return False
    if int(getattr(args, "_layer_candidate_count", 0) or 0) < int(
        getattr(args, "scarcity_inner_rescue_min_candidates", 64) or 0
    ):
        _record_scarcity_inner_rescue_gate_reason(args, "candidates")
        return False
    if float(getattr(args, "_initial_scarcity_risk", 0.0) or 0.0) < float(
        getattr(args, "scarcity_inner_rescue_risk_threshold", 0.95) or 0.95
    ):
        _record_scarcity_inner_rescue_gate_reason(args, "risk")
        return False
    max_l0 = int(getattr(args, "scarcity_inner_rescue_max_l0", -1) or -1)
    if max_l0 >= 0 and int(getattr(args, "_initial_scarcity_l0", 10**9) or 10**9) > max_l0:
        _record_scarcity_inner_rescue_gate_reason(args, "l0")
        return False
    if float(getattr(args, "_initial_scarcity_one_safe_ratio", 0.0) or 0.0) < float(
        getattr(args, "scarcity_inner_rescue_min_one_safe_ratio", 0.90) or 0.90
    ):
        _record_scarcity_inner_rescue_gate_reason(args, "one_safe_ratio")
        return False
    return True


def _with_temporary_args(
    args: argparse.Namespace,
    overrides: dict[str, object],
    fn,
):
    missing = object()
    saved = {key: getattr(args, key, missing) for key in overrides}
    for key, value in overrides.items():
        setattr(args, key, value)
    try:
        return fn()
    finally:
        for key, value in saved.items():
            if value is missing:
                delattr(args, key)
            else:
                setattr(args, key, value)


def _rescue_frontier_widths(args: argparse.Namespace) -> tuple[int, int]:
    beam_width = max(args.beam_width, int(getattr(args, "rescue_beam_width", args.beam_width)))
    core_width = max(args.core_width, int(getattr(args, "rescue_core_width", args.core_width)))
    escalate_after = int(getattr(args, "rescue_escalate_after", 0))
    if escalate_after > 0 and int(getattr(args, "_rescue_layers_used", 0)) >= escalate_after:
        escalate_beam = int(getattr(args, "rescue_escalate_beam_width", 0) or beam_width)
        escalate_core = int(getattr(args, "rescue_escalate_core_width", 0) or core_width)
        beam_width = max(beam_width, escalate_beam)
        core_width = max(core_width, escalate_core)
    core_width = min(core_width, beam_width)
    return beam_width, core_width


def _select_rescue_qubo_frontier(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    candidate_count: int,
    args: argparse.Namespace,
) -> tuple[list[bytes], set[bytes], int, int, float]:
    best_n = int(ranked[0][1][0]) if ranked else math.inf
    standard_rescue_active = _standard_rescue_layer_active(args, best_n)
    scarcity_inner_active = not standard_rescue_active and _scarcity_inner_rescue_active(args, best_n)
    if not standard_rescue_active and not scarcity_inner_active:
        return _select_enhanced_qubo_frontier(ranked, candidate_count, args)

    beam_width, core_width = _rescue_frontier_widths(args)
    supplement_width = max(
        args.qubo_supplement_width,
        int(getattr(args, "rescue_qubo_supplement_width", args.qubo_supplement_width)),
    )
    cohorts_per_layer = max(1, int(getattr(args, "rescue_qubo_cohorts_per_layer", args.qubo_cohorts_per_layer)))
    overrides: dict[str, object] = {
        "max_qubo_calls": max(args.max_qubo_calls, args._qubo_calls_done + cohorts_per_layer),
        "max_qubo_vars": max(args.max_qubo_vars, int(getattr(args, "rescue_max_qubo_vars", args.max_qubo_vars))),
        "qubo_supplement_width": supplement_width,
        "qubo_focus_n": -1,
        "qubo_cohorts_per_layer": cohorts_per_layer,
        "qubo_time_limit": max(args.qubo_time_limit, float(getattr(args, "rescue_qubo_time_limit", args.qubo_time_limit))),
        "qubo_iterations": max(args.qubo_iterations, int(getattr(args, "rescue_qubo_iterations", args.qubo_iterations))),
        "qubo_next_n_weight": max(
            float(getattr(args, "qubo_next_n_weight", 0.0)),
            float(getattr(args, "rescue_next_n_weight", getattr(args, "qubo_next_n_weight", 0.0))),
        ),
        "_frontier_trace_selection_context": "scarcity_inner_rescue" if scarcity_inner_active else "rescue",
        "_frontier_trace_qubo_source": "rescue_qubo",
    }

    before_calls = args._qubo_calls_done

    def select():
        return _select_enhanced_qubo_frontier(
            ranked,
            candidate_count,
            args,
            beam_width=beam_width,
            core_width=core_width,
            qubo_supplement_width=supplement_width,
        )

    selected = _with_temporary_args(args, overrides, select)
    args._rescue_layers_used += 1
    if scarcity_inner_active:
        args._scarcity_inner_rescue_layers = int(getattr(args, "_scarcity_inner_rescue_layers", 0)) + 1
    args._rescue_qubo_calls += args._qubo_calls_done - before_calls
    args._rescue_qubo_vars += selected[3]
    args._rescue_qubo_time += selected[4]
    return selected


def _select_adaptive_qubo_frontier(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    candidate_count: int,
    args: argparse.Namespace,
) -> tuple[list[bytes], set[bytes], int, int, float]:
    best_n = int(ranked[0][1][0]) if ranked else math.inf
    if _rescue_layer_active(args, best_n):
        return _select_rescue_qubo_frontier(ranked, candidate_count, args)
    beam_width = _adaptive_frontier_width(args, int(best_n))
    core_width = int(round(beam_width * float(getattr(args, "adaptive_core_ratio", 0.9))))
    return _select_enhanced_qubo_frontier(
        ranked,
        candidate_count,
        args,
        beam_width=beam_width,
        core_width=core_width,
        qubo_supplement_width=min(args.qubo_supplement_width, max(1, beam_width - core_width)),
    )


def _scheduler_uses_adaptive(args: argparse.Namespace, best_n: int) -> bool:
    min_stacks = max(1, int(getattr(args, "scheduler_adaptive_min_stacks", 7)))
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    if stack_count < min_stacks:
        return False
    case_id = str(getattr(args, "_case_id", "")).lower()
    if getattr(args, "scheduler_adaptive_cv1_only", False) and "cv" in case_id and "cv1" not in case_id:
        return False
    max_duplication = float(getattr(args, "scheduler_adaptive_max_duplication_ratio", 1.0))
    if max_duplication >= 0.0:
        duplication = float(getattr(args, "_priority_duplication_ratio", 0.0))
        if duplication > max_duplication + 1e-12:
            return False
    initial_n = max(1, int(getattr(args, "_initial_n", best_n if best_n > 0 else 1)))
    min_best_n_ratio = float(getattr(args, "scheduler_adaptive_min_best_n_ratio", 0.0) or 0.0)
    if min_best_n_ratio > 0.0 and (best_n / initial_n) < min_best_n_ratio:
        return False
    max_candidate_pressure = float(getattr(args, "scheduler_adaptive_max_candidate_pressure", 0.0) or 0.0)
    if max_candidate_pressure > 0.0:
        pressure = int(getattr(args, "_layer_candidate_count", 0)) / max(1, int(getattr(args, "beam_width", 1)))
        if pressure > max_candidate_pressure:
            return False
    max_parent_top_share = float(getattr(args, "scheduler_adaptive_max_parent_top_share", 0.0) or 0.0)
    if max_parent_top_share > 0.0 and float(getattr(args, "_layer_parent_top_share", 0.0)) > max_parent_top_share:
        return False
    plateau_window = max(0, int(getattr(args, "scheduler_adaptive_plateau_window", 0) or 0))
    cap_slack = int(getattr(args, "scheduler_adaptive_cap_slack", 0) or 0)
    history = getattr(args, "_best_n_history", [])
    if plateau_window > 1 and cap_slack > 0 and len(history) >= plateau_window:
        recent = history[-plateau_window:]
        if max(recent) == min(recent) and int(getattr(args, "_layer_depth_slack", 10**9)) <= cap_slack:
            return False
    return best_n > int(getattr(args, "scheduler_adaptive_high_n", getattr(args, "adaptive_high_n", 16)))


def _select_scheduler_qubo_frontier(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    candidate_count: int,
    args: argparse.Namespace,
) -> tuple[list[bytes], set[bytes], int, int, float]:
    if not ranked:
        return [], set(), 0, 0, 0.0
    best_n = int(ranked[0][1][0])
    if _rescue_layer_active(args, best_n):
        return _select_rescue_qubo_frontier(ranked, candidate_count, args)
    if _scheduler_uses_adaptive(args, best_n):
        return _select_adaptive_qubo_frontier(ranked, candidate_count, args)
    return _select_enhanced_qubo_frontier(ranked, candidate_count, args)


def _top_n_band_for_ranked(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    window: int,
) -> int:
    if not ranked:
        return 0
    take = max(1, min(window, len(ranked)))
    n_values = [data[0] for _code, data in ranked[:take]]
    return max(n_values) - min(n_values)


def _scheduler_state_guard_action(args: argparse.Namespace, name: str) -> str:
    action = str(getattr(args, name, "protect") or "protect")
    return action if action in {"protect", "blend", "rescue"} else "protect"


def _scheduler_state_action(args: argparse.Namespace, best_n: int) -> str:
    if getattr(args, "scheduler_state_controller", "off") != "state_v2":
        return "protect"

    locked_action = str(getattr(args, "_scheduler_state_locked_action", "") or "")
    if locked_action in {"protect", "blend", "rescue"}:
        return locked_action

    history = list(getattr(args, "_best_n_history", []) or [])
    depth_slack = int(getattr(args, "_layer_depth_slack", 10**9))
    rescue_window = max(2, int(getattr(args, "scheduler_state_rescue_plateau_window", 3)))
    rescue_delta = max(0, int(getattr(args, "scheduler_state_rescue_plateau_delta", 0)))
    rescue_slack = int(getattr(args, "scheduler_state_rescue_cap_slack", 0))
    if rescue_slack > 0 and depth_slack <= rescue_slack and _plateau_active(history, rescue_window, rescue_delta):
        return "rescue"

    initial_plateau_window = max(0, int(getattr(args, "scheduler_state_initial_plateau_window", 0) or 0))
    if initial_plateau_window > 1 and len(history) >= initial_plateau_window:
        initial_n = max(1, int(getattr(args, "_initial_n", best_n if best_n > 0 else 1)))
        min_initial_ratio = float(getattr(args, "scheduler_state_initial_plateau_min_ratio", 0.0) or 0.0)
        min_duplication = float(getattr(args, "scheduler_state_initial_plateau_min_duplication", 0.0) or 0.0)
        unique_min_n = max(0, int(getattr(args, "scheduler_state_initial_plateau_unique_min_n", 0) or 0))
        duplication = float(getattr(args, "_priority_duplication_ratio", 0.0) or 0.0)
        recent_initial = history[-initial_plateau_window:]
        unique_low_n_blocked = duplication <= 0.0 and initial_n < unique_min_n
        if (
            not unique_low_n_blocked
            and duplication >= min_duplication
            and max(recent_initial) == min(recent_initial)
            and (best_n / initial_n) >= min_initial_ratio
        ):
            action = _scheduler_state_guard_action(args, "scheduler_state_initial_plateau_action")
            if getattr(args, "scheduler_state_initial_plateau_lock", False):
                args._scheduler_state_locked_action = action
            return action

    parent_limit = float(getattr(args, "scheduler_state_protect_parent_top_share", 0.0) or 0.0)
    if parent_limit > 0.0 and float(getattr(args, "_layer_parent_top_share", 0.0)) >= parent_limit:
        return "protect"

    n_band_limit = int(getattr(args, "scheduler_state_protect_n_band", 0) or 0)
    if n_band_limit > 0 and int(getattr(args, "_layer_top_n_band", 0)) > n_band_limit:
        return "protect"

    min_ratio = float(getattr(args, "scheduler_state_min_best_n_ratio", 0.0) or 0.0)
    if min_ratio > 0.0:
        initial_n = max(1, int(getattr(args, "_initial_n", best_n if best_n > 0 else 1)))
        if best_n / initial_n < min_ratio:
            return "protect"

    contract_pressure = float(getattr(args, "scheduler_state_contract_pressure", 0.0) or 0.0)
    candidate_count = max(0, int(getattr(args, "_layer_candidate_count", 0)))
    pressure = candidate_count / max(1, int(getattr(args, "beam_width", 1)))
    if contract_pressure > 0.0 and pressure < contract_pressure:
        return "protect"
    max_contract_pressure = float(getattr(args, "scheduler_state_max_contract_pressure", 0.0) or 0.0)
    if max_contract_pressure > 0.0 and pressure > max_contract_pressure:
        return "protect"

    progress_window = max(2, int(getattr(args, "scheduler_state_progress_window", 3)))
    progress_delta = max(0, int(getattr(args, "scheduler_state_progress_delta", 1)))
    if len(history) < progress_window:
        return "protect"
    recent = history[-progress_window:]
    if recent[0] - recent[-1] < progress_delta:
        return "protect"

    cooldown = max(0, int(getattr(args, "scheduler_state_failed_contract_cooldown", 0) or 0))
    if cooldown > 0 and getattr(args, "_scheduler_state_last_action", "") == "contract" and len(history) >= 2:
        if history[-2] <= history[-1]:
            return "protect"

    post_rescue_window = max(0, int(getattr(args, "scheduler_state_post_rescue_plateau_window", 0) or 0))
    if (
        post_rescue_window > 1
        and getattr(args, "_scheduler_state_last_effective_action", "") == "rescue"
        and _plateau_active(
            history,
            post_rescue_window,
            max(0, int(getattr(args, "scheduler_state_post_rescue_plateau_delta", 0) or 0)),
        )
    ):
        initial_n = max(1, int(getattr(args, "_initial_n", best_n if best_n > 0 else 1)))
        min_rescue_ratio = float(getattr(args, "scheduler_state_post_rescue_min_ratio", 0.0) or 0.0)
        if min_rescue_ratio <= 0.0 or (best_n / initial_n) >= min_rescue_ratio:
            return _scheduler_state_guard_action(args, "scheduler_state_post_rescue_action")

    high_n = int(getattr(args, "scheduler_adaptive_high_n", getattr(args, "adaptive_high_n", 16)))
    if best_n <= high_n:
        return "protect"
    return "contract"


def _select_scheduler_state_blend_frontier(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    candidate_count: int,
    args: argparse.Namespace,
) -> tuple[list[bytes], set[bytes], int, int, float]:
    selected_codes, selected_by_qubo, qcalls, qvars, qtime = _select_adaptive_qubo_frontier(ranked, candidate_count, args)
    target = max(1, min(int(getattr(args, "beam_width", len(ranked))), len(ranked)))
    if len(selected_codes) >= target:
        return selected_codes, selected_by_qubo, qcalls, qvars, qtime
    selected = dict.fromkeys(selected_codes)
    for code, _data in ranked:
        selected.setdefault(code, None)
        if len(selected) >= target:
            break
    selected_set = set(selected)
    blended_codes = [code for code, _data in ranked if code in selected_set][:target]
    return blended_codes, selected_by_qubo & set(blended_codes), qcalls, qvars, qtime


def _select_scheduler_state_qubo_frontier(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    candidate_count: int,
    args: argparse.Namespace,
) -> tuple[list[bytes], set[bytes], int, int, float]:
    if not ranked:
        return [], set(), 0, 0, 0.0
    top_window = max(1, int(getattr(args, "scheduler_state_top_window", 64) or 64))
    args._layer_top_n_band = _top_n_band_for_ranked(ranked, top_window)
    best_n = int(ranked[0][1][0])
    action = _scheduler_state_action(args, best_n)
    args._scheduler_state_last_action = action
    before_rescue_layers = int(getattr(args, "_rescue_layers_used", 0))
    if action == "rescue":
        args._scheduler_state_rescue_layers = int(getattr(args, "_scheduler_state_rescue_layers", 0)) + 1
        selected = _with_temporary_args(
            args,
            {"_scheduler_state_requested_rescue": True},
            lambda: _select_rescue_qubo_frontier(ranked, candidate_count, args),
        )
        after_rescue_layers = int(getattr(args, "_rescue_layers_used", 0))
        args._scheduler_state_last_effective_action = "rescue" if after_rescue_layers > before_rescue_layers else "protect"
        return selected
    if action == "contract":
        args._scheduler_state_contract_layers = int(getattr(args, "_scheduler_state_contract_layers", 0)) + 1
        selected = _select_adaptive_qubo_frontier(ranked, candidate_count, args)
        after_rescue_layers = int(getattr(args, "_rescue_layers_used", 0))
        args._scheduler_state_last_effective_action = "rescue" if after_rescue_layers > before_rescue_layers else "contract"
        return selected
    if action == "blend":
        args._scheduler_state_blend_layers = int(getattr(args, "_scheduler_state_blend_layers", 0)) + 1
        selected = _select_scheduler_state_blend_frontier(ranked, candidate_count, args)
        after_rescue_layers = int(getattr(args, "_rescue_layers_used", 0))
        args._scheduler_state_last_effective_action = "rescue" if after_rescue_layers > before_rescue_layers else "blend"
        return selected
    args._scheduler_state_protect_layers = int(getattr(args, "_scheduler_state_protect_layers", 0)) + 1
    selected = _select_enhanced_qubo_frontier(ranked, candidate_count, args)
    args._scheduler_state_last_effective_action = "protect"
    return selected


def _bucket_key_for_candidate(data: tuple[int, int, bytes], args: argparse.Namespace) -> tuple[int, int]:
    n_value, depth, _parent = data
    key = getattr(args, "bucket_key", "n_depth")
    if key == "n_depth":
        return (n_value, depth)
    if key == "f_n":
        return (n_value + depth, n_value)
    if key == "n_band":
        band_size = max(1, int(getattr(args, "bucket_n_band_size", 2)))
        return (n_value // band_size, depth)
    raise ValueError(f"unknown bucket key: {key}")


def _select_qubo_bucket_frontier(
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    candidate_count: int,
    args: argparse.Namespace,
) -> tuple[list[bytes], set[bytes], int, int, float]:
    del candidate_count
    if not ranked:
        return [], set(), 0, 0, 0.0

    effective_beam = max(1, min(args.beam_width, len(ranked)))
    selected: dict[bytes, None] = {}
    selected_by_qubo: set[bytes] = set()
    qcalls = qvars = 0
    qtime = 0.0

    bucket_order: list[tuple[int, int]] = []
    buckets: dict[tuple[int, int], list[tuple[bytes, tuple[int, int, bytes]]]] = {}
    for item in ranked:
        key = _bucket_key_for_candidate(item[1], args)
        if key not in buckets:
            buckets[key] = []
            bucket_order.append(key)
        buckets[key].append(item)

    core_ratio = max(0.0, min(1.0, float(getattr(args, "bucket_core_ratio", 0.5))))
    qubo_share = max(0.0, min(1.0, float(getattr(args, "bucket_qubo_share", 0.35))))
    bucket_min = max(1, int(getattr(args, "bucket_min_candidates", args.qubo_min_candidates)))

    for key in bucket_order:
        bucket_items = buckets[key]
        remaining = effective_beam - len(selected)
        if remaining <= 0:
            break

        core_count = int(round(len(bucket_items) * core_ratio))
        if core_ratio > 0.0 and core_count == 0:
            core_count = 1
        core_count = max(0, min(core_count, remaining, len(bucket_items)))
        for code, _data in bucket_items[:core_count]:
            selected.setdefault(code, None)
            if len(selected) >= effective_beam:
                break

        remaining = effective_beam - len(selected)
        if remaining <= 0:
            break

        qubo_candidates = [(code, data) for code, data in bucket_items if code not in selected]
        target_by_share = int(round(len(bucket_items) * qubo_share))
        if qubo_share > 0.0 and target_by_share == 0:
            target_by_share = 1
        qtarget = min(
            max(0, int(getattr(args, "bucket_qubo_select", args.qubo_supplement_width))),
            max(0, target_by_share),
            remaining,
            len(qubo_candidates),
        )
        qubo_enabled = (
            getattr(args, "use_qubo", False)
            and qtarget > 0
            and len(bucket_items) >= bucket_min
            and args._qubo_calls_done < args.max_qubo_calls
        )
        if qubo_enabled:
            qargs = SimpleNamespace(**vars(args))
            qargs.qubo_core_width = 0
            qargs.qubo_supplement_width = qtarget
            qargs.max_qubo_vars = max(1, int(getattr(args, "bucket_max_qubo_vars", args.max_qubo_vars)))
            qargs.qubo_min_candidates = max(1, min(bucket_min, len(qubo_candidates)))
            boosted, add_qvars, add_qtime = _select_qubo_boosted(dict(qubo_candidates), qargs)
            if add_qvars:
                qcalls += 1
                qvars += add_qvars
                qtime += add_qtime
                args._qubo_calls_done += 1
                for code in boosted:
                    if code not in selected:
                        selected_by_qubo.add(code)
                    selected.setdefault(code, None)
                    if len(selected) >= effective_beam:
                        break

        for code, _data in bucket_items:
            selected.setdefault(code, None)
            if len(selected) >= effective_beam:
                break

    for code, _data in ranked:
        selected.setdefault(code, None)
        if len(selected) >= effective_beam:
            break

    selected_codes = list(selected.keys())[:effective_beam]
    selected_set = set(selected_codes)
    return selected_codes, selected_by_qubo & selected_set, qcalls, qvars, qtime


def _rank_next_frontier_candidates(
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
) -> list[tuple[bytes, tuple[int, int, bytes]]]:
    if bool(getattr(args, "candidate_ledger_score_enabled", False)):
        args._candidate_ledger_active_codes = _candidate_feature_ledger_window_codes(candidates, args)
    else:
        args._candidate_ledger_active_codes = None
    bucket_topk = max(0, int(getattr(args, "rank_bucket_topk", 0) or 0))
    if bucket_topk > 0:
        best_n = min(data[0] for data in candidates.values()) if candidates else math.inf
        high_n_gate = max(0, int(getattr(args, "rank_bucket_topk_high_n", 0) or 0))
        min_candidates = max(0, int(getattr(args, "rank_bucket_topk_min_candidates", 0) or 0))
        topk_allowed = len(candidates) >= min_candidates and (high_n_gate <= 0 or best_n >= high_n_gate)
        if topk_allowed:
            return _rank_bucket_topk_candidates(candidates, args)
    adaptive_pool = 0
    if getattr(args, "selection_policy", "enhanced") == "adaptive_qubo":
        adaptive_pool = max(0, int(getattr(args, "adaptive_candidate_pool", 0) or 0))
    if adaptive_pool > 0 and len(candidates) > adaptive_pool:
        if not hasattr(args, "_rank_pool_truncated_candidates"):
            args._rank_pool_truncated_candidates = 0
        pool = heapq.nsmallest(adaptive_pool, candidates.items(), key=_candidate_key_func(args))
        args._rank_pool_truncated_candidates += len(candidates) - len(pool)
        return sorted(pool, key=_candidate_key_func(args))
    if max(0, int(getattr(args, "rank_pool_size", 0) or 0)) <= 0:
        return sorted(candidates.items(), key=_candidate_key_func(args))
    return _rank_candidates(candidates, args)


def _select_next_frontier(
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
) -> tuple[list[bytes], set[bytes], int, int, float]:
    ranked = _rank_next_frontier_candidates(candidates, args)
    args._frontier_trace_last_ranked = ranked
    args._frontier_trace_last_selected_sources = {}
    args._frontier_trace_last_context = ""
    policy = getattr(args, "selection_policy", "enhanced")
    if policy == "classic":
        return [code for code, _data in ranked[: args.beam_width]], set(), 0, 0, 0.0
    if policy == "dynamic_qubo":
        return _select_dynamic_qubo_frontier(ranked, args)
    if policy == "adaptive_qubo":
        return _select_adaptive_qubo_frontier(ranked, len(candidates), args)
    if policy == "rescue_qubo":
        return _select_rescue_qubo_frontier(ranked, len(candidates), args)
    if policy == "scheduler_qubo":
        return _select_scheduler_qubo_frontier(ranked, len(candidates), args)
    if policy == "scheduler_state_qubo":
        return _select_scheduler_state_qubo_frontier(ranked, len(candidates), args)
    if policy == "qubo_bucket":
        return _select_qubo_bucket_frontier(ranked, len(candidates), args)
    return _select_enhanced_qubo_frontier(ranked, len(candidates), args)


def _backtrack(parent: dict[bytes, bytes | None], target: bytes) -> list[bytes]:
    codes = [target]
    seen = {target}
    current = target
    while True:
        next_code = parent[current]
        if next_code is None:
            break
        if next_code in seen:
            raise RuntimeError("parent chain cycle detected")
        current = next_code
        seen.add(current)
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


def _successors_with_n_uncached(
    code: bytes,
    current_n: int,
    stack_count: int,
    max_height: int,
):
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


@lru_cache(maxsize=65536)
def _successor_tuple_cached(
    code: bytes,
    current_n: int,
    stack_count: int,
    max_height: int,
) -> tuple[tuple[bytes, int], ...]:
    return tuple(_successors_with_n_uncached(code, current_n, stack_count, max_height))


def _successors_with_n_cached(code: bytes, current_n: int, stack_count: int, max_height: int):
    return _successor_tuple_cached(code, current_n, stack_count, max_height)


_successors_with_n_cached.cache_clear = _successor_tuple_cached.cache_clear
_successors_with_n_cached.cache_info = _successor_tuple_cached.cache_info


@dataclass(frozen=True)
class TailLookaheadProbe:
    goal_reachable: bool
    best_n: int
    best_depth: int
    expanded: int
    path: tuple[bytes, ...] = ()


@lru_cache(maxsize=131072)
def _tail_lookahead_probe(
    code: bytes,
    current_n: int,
    stack_count: int,
    max_height: int,
    target_final_n: int,
    max_depth: int,
    max_expanded: int,
    base_depth: int = 0,
    business_cap: int = -1,
) -> TailLookaheadProbe:
    max_depth = max(0, int(max_depth))
    max_expanded = max(0, int(max_expanded))
    base_depth = max(0, int(base_depth))
    business_cap = int(business_cap)
    best_n = int(current_n)
    best_depth = 0
    if current_n <= target_final_n:
        return TailLookaheadProbe(True, int(current_n), 0, 0)
    if max_depth <= 0 or max_expanded <= 0:
        return TailLookaheadProbe(False, best_n, best_depth, 0)

    queue: deque[tuple[bytes, int, int]] = deque([(code, int(current_n), 0)])
    seen: dict[bytes, int] = {code: 0}
    predecessor: dict[bytes, bytes] = {}
    expanded = 0

    def build_path(goal_code: bytes) -> tuple[bytes, ...]:
        path: list[bytes] = [goal_code]
        cursor = goal_code
        while cursor != code and cursor in predecessor:
            cursor = predecessor[cursor]
            if cursor != code:
                path.append(cursor)
        path.reverse()
        return tuple(path)

    while queue and expanded < max_expanded:
        current_code, n_value, depth = queue.popleft()
        if depth >= max_depth:
            continue
        expanded += 1
        for nxt, nxt_n in _successors_with_n_uncached(current_code, n_value, stack_count, max_height):
            next_depth = depth + 1
            if business_cap >= 0 and base_depth + next_depth + int(nxt_n) > business_cap:
                continue
            old_depth = seen.get(nxt)
            if old_depth is not None and old_depth <= next_depth:
                continue
            seen[nxt] = next_depth
            predecessor[nxt] = current_code
            if nxt_n < best_n or (nxt_n == best_n and (best_depth == 0 or next_depth < best_depth)):
                best_n = int(nxt_n)
                best_depth = next_depth
            if nxt_n <= target_final_n:
                return TailLookaheadProbe(True, int(nxt_n), next_depth, expanded, build_path(nxt))
            if next_depth < max_depth:
                queue.append((nxt, int(nxt_n), next_depth))
    return TailLookaheadProbe(False, best_n, best_depth, expanded)


def _ensure_tail_lookahead_counters(args: argparse.Namespace) -> None:
    if not hasattr(args, "_tail_lookahead_calls_done"):
        args._tail_lookahead_calls_done = 0
    if not hasattr(args, "_tail_lookahead_selected_total"):
        args._tail_lookahead_selected_total = 0
    if not hasattr(args, "_tail_lookahead_goal_reachable_total"):
        args._tail_lookahead_goal_reachable_total = 0
    if not hasattr(args, "_tail_lookahead_best_delta_total"):
        args._tail_lookahead_best_delta_total = 0
    if not hasattr(args, "_tail_lookahead_expanded_total"):
        args._tail_lookahead_expanded_total = 0


def _ensure_tail_risk_reserve_counters(args: argparse.Namespace) -> None:
    if not hasattr(args, "_tail_risk_reserve_calls_done"):
        args._tail_risk_reserve_calls_done = 0
    if not hasattr(args, "_tail_risk_reserve_selected_total"):
        args._tail_risk_reserve_selected_total = 0
    if not hasattr(args, "_tail_risk_reserve_scanned_total"):
        args._tail_risk_reserve_scanned_total = 0
    if not hasattr(args, "_tail_risk_reserve_one_step_improved_total"):
        args._tail_risk_reserve_one_step_improved_total = 0
    if not hasattr(args, "_tail_risk_priority_codes"):
        args._tail_risk_priority_codes = set()


def _tail_risk_one_step_best_n(code: bytes, n_value: int, args: argparse.Namespace) -> int:
    cache: dict[bytes, int] = getattr(args, "_tail_risk_one_step_cache", {})
    cached = cache.get(code)
    if cached is not None:
        return cached
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return int(n_value)
    best = int(n_value)
    for _nxt, nxt_n in _successors_with_n_cached(code, int(n_value), stack_count, max_height):
        if int(nxt_n) < best:
            best = int(nxt_n)
            if best <= 0:
                break
    cache[code] = best
    args._tail_risk_one_step_cache = cache
    return best


def _tail_risk_one_step_recovery_codes(
    code: bytes,
    n_value: int,
    args: argparse.Namespace,
    best_n: int,
) -> tuple[bytes, ...]:
    cache: dict[tuple[bytes, int], tuple[bytes, ...]] = getattr(args, "_tail_risk_recovery_cache", {})
    key = (code, int(best_n))
    cached = cache.get(key)
    if cached is not None:
        return cached
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return ()
    recovery = [
        (int(nxt_n), nxt)
        for nxt, nxt_n in _successors_with_n_cached(code, int(n_value), stack_count, max_height)
        if int(nxt_n) <= int(best_n)
    ]
    recovery.sort()
    codes = tuple(nxt for _nxt_n, nxt in recovery[:2])
    cache[key] = codes
    args._tail_risk_recovery_cache = cache
    return codes


def _tail_risk_reserve_active(args: argparse.Namespace, best_n: int) -> bool:
    if not getattr(args, "tail_risk_reserve_enabled", False):
        return False
    if int(getattr(args, "tail_risk_reserve_width", 0) or 0) <= 0:
        return False
    focus_n = int(getattr(args, "tail_risk_focus_n", 2))
    if focus_n >= 0 and int(best_n) > focus_n:
        return False
    min_candidates = int(getattr(args, "tail_risk_min_candidates", 0) or 0)
    if int(getattr(args, "_layer_candidate_count", 0) or 0) < min_candidates:
        return False
    slack = int(getattr(args, "_layer_depth_slack", 10**9) or 0)
    min_slack = int(getattr(args, "tail_risk_min_depth_slack", 0) or 0)
    max_slack = int(getattr(args, "tail_risk_max_depth_slack", 0) or 0)
    if slack < min_slack:
        return False
    if max_slack > 0 and slack > max_slack:
        return False
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return False
    max_stack_count = int(getattr(args, "tail_risk_max_stack_count", 0) or 0)
    return max_stack_count <= 0 or stack_count <= max_stack_count


def _add_tail_risk_path_supplement(
    selected: dict[bytes, None],
    selected_source_by_code: dict[bytes, str],
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    effective_beam: int,
) -> None:
    priority_codes = getattr(args, "_tail_risk_priority_codes", None)
    if not priority_codes or len(selected) >= effective_beam:
        return
    advanced: list[bytes] = []
    for code, _data in ranked:
        if code not in priority_codes:
            continue
        if code not in selected and len(selected) < effective_beam:
            selected[code] = None
            selected_source_by_code.setdefault(code, "tail_risk_path")
        advanced.append(code)
        if len(selected) >= effective_beam:
            break
    for code in advanced:
        priority_codes.discard(code)


def _add_tail_risk_reserve(
    selected: dict[bytes, None],
    selected_source_by_code: dict[bytes, str],
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    effective_beam: int,
    best_n: int,
) -> None:
    if len(selected) >= effective_beam:
        return
    if not _tail_risk_reserve_active(args, int(best_n)):
        return
    n_window = max(0, int(getattr(args, "tail_risk_n_window", 1) or 0))
    reserve_width = max(0, int(getattr(args, "tail_risk_reserve_width", 0) or 0))
    max_scan = max(1, int(getattr(args, "tail_risk_max_scan", 4096) or 4096))
    target = min(effective_beam, len(selected) + reserve_width)
    if target <= len(selected):
        return

    _ensure_tail_risk_reserve_counters(args)
    args._tail_risk_reserve_calls_done += 1
    scored: list[tuple[int, int, int, int, bytes]] = []
    scanned = 0
    improved_seen = 0
    for code, data in ranked:
        if scanned >= max_scan:
            break
        n_value, depth, _parent_code = data
        n_value = int(n_value)
        if n_value <= int(best_n):
            continue
        if n_value > int(best_n) + n_window:
            continue
        scanned += 1
        if code in selected:
            continue
        one_step_best = _tail_risk_one_step_best_n(code, n_value, args)
        if one_step_best > int(best_n):
            continue
        improved_seen += 1
        scored.append((one_step_best, n_value, int(depth), scanned, code))
    args._tail_risk_reserve_scanned_total += scanned
    args._tail_risk_reserve_one_step_improved_total += improved_seen
    if not scored:
        return
    for _best_after, _n_value, _depth, _scan_rank, code in sorted(scored):
        before = len(selected)
        selected.setdefault(code, None)
        selected_source_by_code.setdefault(code, "tail_risk")
        if len(selected) > before:
            args._tail_risk_reserve_selected_total += 1
            for recovery_code in _tail_risk_one_step_recovery_codes(code, _n_value, args, int(best_n)):
                args._tail_risk_priority_codes.add(recovery_code)
        if len(selected) >= target:
            break


def _add_tail_lookahead_path_supplement(
    selected: dict[bytes, None],
    selected_source_by_code: dict[bytes, str],
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    effective_beam: int,
) -> None:
    if not getattr(args, "tail_lookahead_enabled", False):
        return
    priority_codes = getattr(args, "_tail_lookahead_priority_codes", None)
    if not priority_codes:
        return
    path_next_by_code = getattr(args, "_tail_lookahead_path_next_by_code", {})
    advanced: list[bytes] = []
    for code, _data in ranked:
        if code not in priority_codes:
            continue
        if code in selected:
            advanced.append(code)
            continue
        if len(selected) >= effective_beam:
            continue
        selected[code] = None
        selected_source_by_code.setdefault(code, "tail_lookahead_path")
        advanced.append(code)
        if len(selected) >= effective_beam:
            break
    for code in advanced:
        priority_codes.discard(code)
        next_code = path_next_by_code.get(code)
        if next_code is not None:
            priority_codes.add(next_code)


def _add_tail_lookahead_supplement(
    selected: dict[bytes, None],
    selected_source_by_code: dict[bytes, str],
    ranked: list[tuple[bytes, tuple[int, int, bytes]]],
    args: argparse.Namespace,
    effective_beam: int,
    best_n: int,
) -> None:
    if not getattr(args, "tail_lookahead_enabled", False):
        return
    width = max(0, int(getattr(args, "tail_lookahead_supplement_width", 0) or 0))
    if width <= 0:
        return
    focus_n = int(getattr(args, "tail_lookahead_focus_n", 2))
    if focus_n >= 0 and int(best_n) > focus_n:
        return
    stack_count = int(getattr(args, "_stack_count", 0) or 0)
    max_height = int(getattr(args, "_max_height", 0) or 0)
    if stack_count <= 0 or max_height <= 0:
        return
    max_stack_count = int(getattr(args, "tail_lookahead_max_stack_count", 0) or 0)
    if max_stack_count > 0 and stack_count > max_stack_count:
        return
    max_seed = max(1, int(getattr(args, "tail_lookahead_max_seed_candidates", 128) or 128))
    max_depth = max(1, int(getattr(args, "tail_lookahead_depth", 3) or 3))
    expanded_budget = max(1, int(getattr(args, "tail_lookahead_max_expanded", 4096) or 4096))
    target_final_n = int(getattr(args, "target_final_n", 0) or 0)
    business_cap = int(getattr(args, "_business_cap", -1) or -1)
    core_path_only = bool(getattr(args, "tail_lookahead_core_path_only", False))

    seed_candidates: list[tuple[int, bytes, tuple[int, int, bytes], bool]] = []
    for rank, (code, data) in enumerate(ranked):
        n_value = int(data[0])
        if focus_n >= 0 and n_value > focus_n:
            continue
        already_selected = code in selected
        if core_path_only and not already_selected:
            continue
        if not already_selected and len(selected) >= effective_beam:
            continue
        seed_candidates.append((rank, code, data, already_selected))
        if len(seed_candidates) >= max_seed:
            break
    if not seed_candidates:
        return

    _ensure_tail_lookahead_counters(args)
    if not hasattr(args, "_tail_lookahead_priority_codes"):
        args._tail_lookahead_priority_codes = set()
    if not hasattr(args, "_tail_lookahead_path_next_by_code"):
        args._tail_lookahead_path_next_by_code = {}
    args._tail_lookahead_calls_done += 1
    scored: list[tuple[int, int, int, int, bytes, TailLookaheadProbe, int, bool]] = []
    expanded_used = 0
    goal_reachable_seen = 0
    for rank, code, data, already_selected in seed_candidates:
        if expanded_used >= expanded_budget:
            break
        n_value = int(data[0])
        probe = _tail_lookahead_probe(
            code,
            n_value,
            stack_count,
            max_height,
            target_final_n,
            max_depth,
            expanded_budget - expanded_used,
            base_depth=int(data[1]),
            business_cap=business_cap,
        )
        expanded_used += probe.expanded
        if probe.goal_reachable:
            goal_reachable_seen += 1
        if not probe.goal_reachable and probe.best_n >= n_value:
            continue
        scored.append(
            (
                0 if probe.goal_reachable else 1,
                int(probe.best_n),
                int(probe.best_depth),
                rank,
                code,
                probe,
                n_value,
                already_selected,
            )
        )
    args._tail_lookahead_goal_reachable_total += goal_reachable_seen
    args._tail_lookahead_expanded_total += expanded_used
    if not scored:
        return

    target = 0 if core_path_only else min(width, max(0, effective_beam - len(selected)), len(scored))
    added = 0
    for _goal_rank, _best_n, _best_depth, _rank, code, probe, n_value, already_selected in sorted(scored):
        if already_selected:
            if probe.goal_reachable and probe.path:
                args._tail_lookahead_priority_codes.add(probe.path[0])
                for idx in range(len(probe.path) - 1):
                    args._tail_lookahead_path_next_by_code[probe.path[idx]] = probe.path[idx + 1]
            continue
        if added >= target:
            continue
        before = len(selected)
        selected.setdefault(code, None)
        if len(selected) > before:
            added += 1
            selected_source_by_code.setdefault(code, "tail_lookahead")
            args._tail_lookahead_selected_total += 1
            args._tail_lookahead_best_delta_total += max(0, n_value - int(probe.best_n))
            if probe.goal_reachable and probe.path:
                args._tail_lookahead_priority_codes.add(probe.path[0])
                for idx in range(len(probe.path) - 1):
                    args._tail_lookahead_path_next_by_code[probe.path[idx]] = probe.path[idx + 1]
        if len(selected) >= effective_beam and added >= target:
            break


def _post_polish_successors(
    code: bytes,
    current_n: int,
    stack_count: int,
    max_height: int,
    args: argparse.Namespace,
):
    if bool(getattr(args, "post_polish_successor_cache", False)):
        return _successors_with_n_cached(code, current_n, stack_count, max_height)
    return _successors_with_n_uncached(code, current_n, stack_count, max_height)


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


def _trim_stream_candidates_batch(
    candidates: dict[bytes, tuple[int, int, bytes]],
    limit: int,
    args: argparse.Namespace,
) -> int:
    if limit <= 0 or len(candidates) <= limit:
        return 0
    kept = heapq.nsmallest(limit, candidates.items(), key=_candidate_key_func(args))
    pruned = len(candidates) - len(kept)
    candidates.clear()
    candidates.update(kept)
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


def _live_bucket_entry(
    candidates: dict[bytes, tuple[int, int, bytes]],
    bucket_members: dict[tuple, set[bytes]],
    bucket: tuple,
    code: bytes,
    data: tuple[int, int, bytes],
    args: argparse.Namespace,
) -> bool:
    return (
        candidates.get(code) == data
        and code in bucket_members.get(bucket, set())
        and _rank_bucket_key_for_candidate(data, args) == bucket
    )


def _peek_stream_bucket_worst(
    candidates: dict[bytes, tuple[int, int, bytes]],
    bucket_heap: list[tuple[_ReverseRank, int, bytes, tuple[int, int, bytes]]],
    bucket_members: dict[tuple, set[bytes]],
    bucket: tuple,
    args: argparse.Namespace,
) -> _ReverseRank | None:
    while bucket_heap:
        key, _seq, code, data = bucket_heap[0]
        if _live_bucket_entry(candidates, bucket_members, bucket, code, data, args):
            return key
        heapq.heappop(bucket_heap)
    return None


def _trim_stream_bucket_candidates(
    candidates: dict[bytes, tuple[int, int, bytes]],
    bucket_heap: list[tuple[_ReverseRank, int, bytes, tuple[int, int, bytes]]],
    bucket_members: dict[tuple, set[bytes]],
    bucket: tuple,
    topk: int,
    args: argparse.Namespace,
) -> int:
    if topk <= 0:
        return 0
    members = bucket_members.get(bucket)
    if not members:
        return 0
    pruned = 0
    while len(members) > topk:
        while bucket_heap:
            _key, _seq, code, data = heapq.heappop(bucket_heap)
            if _live_bucket_entry(candidates, bucket_members, bucket, code, data, args):
                del candidates[code]
                members.discard(code)
                pruned += 1
                break
        else:
            active = [(code, candidates[code]) for code in members if code in candidates]
            if not active:
                members.clear()
                break
            code, _data = max(active, key=_candidate_key_func(args))
            del candidates[code]
            members.discard(code)
            pruned += 1
    if not members:
        bucket_members.pop(bucket, None)
    return pruned


def _rebuild_stream_bucket_state(
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
) -> tuple[
    dict[tuple, list[tuple[_ReverseRank, int, bytes, tuple[int, int, bytes]]]],
    dict[tuple, set[bytes]],
    int,
]:
    bucket_heaps: dict[tuple, list[tuple[_ReverseRank, int, bytes, tuple[int, int, bytes]]]] = {}
    bucket_members: dict[tuple, set[bytes]] = {}
    seq = 0
    for code, data in candidates.items():
        bucket = _rank_bucket_key_for_candidate(data, args)
        bucket_members.setdefault(bucket, set()).add(code)
        seq += 1
        heapq.heappush(bucket_heaps.setdefault(bucket, []), (_stream_worst_key(code, data, args, seq), seq, code, data))
    return bucket_heaps, bucket_members, seq


def _offer_stream_bucket_candidate(
    candidates: dict[bytes, tuple[int, int, bytes]],
    bucket_heaps: dict[tuple, list[tuple[_ReverseRank, int, bytes, tuple[int, int, bytes]]]],
    bucket_members: dict[tuple, set[bytes]],
    code: bytes,
    data: tuple[int, int, bytes],
    args: argparse.Namespace,
    seq: int,
) -> int:
    topk = max(0, int(getattr(args, "stream_bucket_topk", 0) or 0))
    bucket = _rank_bucket_key_for_candidate(data, args)
    if topk <= 0:
        candidates[code] = data
        bucket_members.setdefault(bucket, set()).add(code)
        heapq.heappush(bucket_heaps.setdefault(bucket, []), (_stream_worst_key(code, data, args, seq), seq, code, data))
        return 0

    old_data = candidates.get(code)
    if old_data is not None:
        old_bucket = _rank_bucket_key_for_candidate(old_data, args)
        old_members = bucket_members.get(old_bucket)
        if old_members is not None:
            old_members.discard(code)
            if not old_members:
                bucket_members.pop(old_bucket, None)

    members = bucket_members.setdefault(bucket, set())
    candidate_key = _stream_worst_key(code, data, args, seq)
    if old_data is None and len(members) >= topk:
        worst_key = _peek_stream_bucket_worst(candidates, bucket_heaps.setdefault(bucket, []), bucket_members, bucket, args)
        if worst_key is not None and candidate_key <= worst_key:
            return 1

    candidates[code] = data
    members.add(code)
    bucket_heap = bucket_heaps.setdefault(bucket, [])
    heapq.heappush(bucket_heap, (candidate_key, seq, code, data))
    return _trim_stream_bucket_candidates(candidates, bucket_heap, bucket_members, bucket, topk, args)


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
    tail_span = int(getattr(args, "post_polish_suffix_tail_span", 18))
    focus_n = int(getattr(args, "post_polish_suffix_focus_n", 9))
    max_candidates = int(getattr(args, "post_polish_suffix_max_candidates", 48))
    tail_begin = max(0, last_move - max(1, tail_span))
    suffixes: list[PostPolishSuffix] = []
    for start in range(tail_begin, last_move):
        old_remaining = last_move - start
        n_value = n_values[start]
        if old_remaining < 2 or n_value > focus_n:
            continue
        if n_value > old_remaining - 1:
            continue
        low_n_bonus = max(0, focus_n - n_value)
        score = float(old_remaining + 0.75 * low_n_bonus + 0.02 * start)
        suffixes.append(PostPolishSuffix(start, old_remaining, n_value, score))
    suffixes.sort(key=lambda s: (-s.score, s.start))
    return suffixes[: max(1, max_candidates)]


def _select_post_polish_suffixes_qubo(
    suffixes: list[PostPolishSuffix],
    args: argparse.Namespace,
) -> tuple[list[PostPolishSuffix], int, float]:
    if not suffixes:
        return [], 0, 0.0
    target = max(1, min(int(getattr(args, "post_polish_suffix_select", 5)), len(suffixes)))
    usable = suffixes[: max(1, min(int(getattr(args, "post_polish_suffix_max_qubo_vars", 48)), len(suffixes)))]
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
            if distance <= int(getattr(args, "post_polish_suffix_near_gap", 2)):
                model.add_quadratic(vars_[i], vars_[j], float(getattr(args, "post_polish_suffix_near_penalty", 0.30)))
    model.add_exactly_k_penalty(vars_, target, float(getattr(args, "post_polish_penalty_a", 18.0)))
    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        _qubo_sampler_config(
            args,
            num_reads=int(getattr(args, "post_polish_qubo_reads", 2)),
            random_seed=int(getattr(args, "seed", 37)) + 2003,
            time_limit_sec=float(getattr(args, "post_polish_qubo_time_limit", 0.001)),
            sa_iterations=int(getattr(args, "post_polish_qubo_iterations", 20)),
        ),
    )
    selected: set[int] = set()
    if sample_set.samples:
        selected = {idx for idx in range(len(usable)) if sample_set.samples[0].binary_values.get(idx, 0) == 1}

    repaired: list[int] = []
    for idx in sorted(selected, key=lambda i: -usable[i].score):
        if all(abs(usable[idx].start - usable[old].start) > int(getattr(args, "post_polish_suffix_near_gap", 2)) for old in repaired):
            repaired.append(idx)
        if len(repaired) >= target:
            break
    for idx in sorted(range(len(usable)), key=lambda i: -usable[i].score):
        if len(repaired) >= target:
            break
        if idx in repaired:
            continue
        if all(abs(usable[idx].start - usable[old].start) > int(getattr(args, "post_polish_suffix_near_gap", 2)) for old in repaired):
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

    max_expanded = int(getattr(args, "post_polish_suffix_expanded", 60000))
    while heap and time.perf_counter() < deadline and expanded < max_expanded:
        _priority, current_n, depth, code = heapq.heappop(heap)
        if depth != depth_by_code.get(code):
            continue
        if depth >= max_depth:
            continue
        if current_n > max_depth - depth:
            continue
        expanded += 1
        next_depth = depth + 1
        for nxt, nxt_n in _post_polish_successors(code, current_n, stack_count, max_height, args):
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


def _try_post_polish_suffix_pass(
    current: list[bytes],
    stack_count: int,
    max_height: int,
    cap: int,
    args: argparse.Namespace,
    deadline: float,
    stats: PostPolishStats,
) -> tuple[list[bytes], bool]:
    n_values = _path_n_values(current, stack_count, max_height)
    suffixes = _post_polish_suffixes(current, n_values, args)
    selected_suffixes, suffix_qvars, suffix_qtime = _select_post_polish_suffixes_qubo(suffixes, args)
    if suffix_qvars:
        stats.qcalls += 1
        stats.qvars += suffix_qvars
        stats.qtime += suffix_qtime
    for suffix in selected_suffixes:
        if time.perf_counter() >= deadline:
            break
        if suffix.start >= len(current) - 1:
            continue
        old_remaining = len(current) - 1 - suffix.start
        max_depth = min(old_remaining - 1, int(getattr(args, "post_polish_suffix_max_depth", 18)))
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
        stats.improvements += 1
        stats.suffix_improvements += 1
        stats.moves_saved += saved
        return candidate, True
    return current, False


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
        _qubo_sampler_config(
            args,
            num_reads=args.post_polish_qubo_reads,
            random_seed=args.seed + 1009,
            time_limit_sec=args.post_polish_qubo_time_limit,
            sa_iterations=args.post_polish_qubo_iterations,
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


def _post_polish_windows_v2(
    codes: list[bytes],
    n_values: list[int],
    args: argparse.Namespace,
) -> list[PostPolishWindow]:
    if len(codes) <= 3:
        return []
    last_move = len(codes) - 1
    min_len = max(2, int(getattr(args, "post_polish_window_min_size", 3)))
    max_len_default = int(getattr(args, "post_polish_max_window", 8))
    max_len = max(min_len, int(getattr(args, "post_polish_window_max_size", max_len_default)))
    max_len = min(max_len, last_move)
    stride = max(1, int(getattr(args, "post_polish_window_stride", 1)))
    budget_default = int(getattr(args, "post_polish_max_windows", 96))
    budget = max(1, int(getattr(args, "post_polish_window_budget", budget_default)))
    tail_begin = max(0, last_move - max(0, int(getattr(args, "post_polish_tail_span", 0))))
    focus_n = int(getattr(args, "post_polish_focus_n", 0))
    windows: list[PostPolishWindow] = []

    for old_len in range(min_len, max_len + 1):
        for start in range(0, last_move - old_len + 1, stride):
            end = start + old_len
            segment_n = n_values[start : end + 1]
            deltas = [segment_n[idx + 1] - segment_n[idx] for idx in range(len(segment_n) - 1)]
            plateau_count = sum(1 for delta in deltas if delta == 0)
            increase_count = sum(1 for delta in deltas if delta > 0)
            min_n = min(segment_n)
            max_n = max(segment_n)
            span = max_n - min_n
            tail_dirty = 1 if end >= tail_begin and max(segment_n) > 0 else 0
            repeated_states = len(segment_n) - len(set(codes[start : end + 1]))
            oscillation_count = repeated_states
            for idx in range(len(segment_n) - 2):
                if segment_n[idx] == segment_n[idx + 2] and segment_n[idx] != segment_n[idx + 1]:
                    oscillation_count += 1
            low_n_bonus = max(0, focus_n - min_n)
            score = (
                float(old_len)
                + 3.0 * plateau_count
                + 5.0 * increase_count
                + 0.25 * span
                + 0.10 * tail_dirty
                + 1.25 * oscillation_count
                + 0.35 * low_n_bonus
                - 0.20 * start
            )
            windows.append(
                PostPolishWindow(
                    start=start,
                    end=end,
                    old_len=old_len,
                    min_n=min_n,
                    max_n=max_n,
                    score=score,
                    plateau_count=plateau_count,
                    increase_count=increase_count,
                    tail_dirty=tail_dirty,
                    oscillation_count=oscillation_count,
                )
            )
    windows.sort(key=lambda w: (-w.score, w.start, w.end))
    return windows[:budget]


def _select_post_polish_windows_v2_qubo(
    windows: list[PostPolishWindow],
    args: argparse.Namespace,
) -> tuple[list[PostPolishWindow], int, float]:
    if not windows:
        return [], 0, 0.0
    requested = int(getattr(args, "post_polish_window_select", 8))
    if requested <= 0:
        return [], 0, 0.0
    min_score_ratio = float(getattr(args, "post_polish_window_min_score_ratio", 0.0) or 0.0)
    if min_score_ratio > 0.0:
        best_score = max((window.score for window in windows), default=0.0)
        if best_score > 0.0:
            windows = [window for window in windows if window.score >= best_score * min_score_ratio]
            if not windows:
                return [], 0, 0.0
    target = max(1, min(requested, len(windows)))
    usable = windows[: max(1, min(int(getattr(args, "post_polish_window_qubo_vars", 72)), len(windows)))]
    if len(usable) <= target:
        return usable, 0, 0.0

    max_score = max(window.score for window in usable) or 1.0
    model = QuboModel()
    vars_ = [model.add_binary_var(f"wv2_{idx}") for idx in range(len(usable))]
    for idx, window in enumerate(usable):
        model.add_linear(vars_[idx], -window.score / max_score)
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            if _windows_overlap(usable[i], usable[j]):
                model.add_quadratic(vars_[i], vars_[j], float(getattr(args, "post_polish_overlap_penalty", 8.0)))
                continue
            distance = min(abs(usable[i].start - usable[j].start), abs(usable[i].end - usable[j].end))
            if distance <= int(getattr(args, "post_polish_near_window_gap", 2)):
                model.add_quadratic(vars_[i], vars_[j], 0.20)
            if usable[i].plateau_count and usable[j].plateau_count and usable[i].min_n == usable[j].min_n:
                model.add_quadratic(vars_[i], vars_[j], 0.10)
    model.add_exactly_k_penalty(vars_, target, float(getattr(args, "post_polish_penalty_a", 18.0)))
    time_limit = float(getattr(args, "post_polish_window_time_limit", getattr(args, "post_polish_qubo_time_limit", 0.001)))
    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        _qubo_sampler_config(
            args,
            num_reads=int(getattr(args, "post_polish_qubo_reads", 2)),
            random_seed=int(getattr(args, "seed", 37)) + 3011,
            time_limit_sec=time_limit,
            sa_iterations=int(getattr(args, "post_polish_qubo_iterations", 20)),
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
        for nxt, nxt_n in _post_polish_successors(code, current_n, stack_count, max_height, args):
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


def _post_polish_codes_with_qubo_windows_v2(
    codes: list[bytes],
    stack_count: int,
    max_height: int,
    cap: int,
    args: argparse.Namespace,
    deadline: float,
) -> tuple[list[bytes], PostPolishStats]:
    stats = PostPolishStats()
    current = list(codes)
    polish_start = time.perf_counter()
    accept_mode = getattr(args, "post_polish_accept", "better")

    try:
        for _pass_idx in range(max(1, args.post_polish_passes)):
            if time.perf_counter() >= deadline:
                break
            current, suffix_improved = _try_post_polish_suffix_pass(
                current,
                stack_count,
                max_height,
                cap,
                args,
                deadline,
                stats,
            )
            if suffix_improved:
                continue

            n_values = _path_n_values(current, stack_count, max_height)
            windows = _post_polish_windows_v2(current, n_values, args)
            stats.window_candidates += len(windows)
            selected, qvars, qtime = _select_post_polish_windows_v2_qubo(windows, args)
            stats.window_qubo_selected += len(selected)
            if qvars:
                stats.qcalls += 1
                stats.qvars += qvars
                stats.qtime += qtime
            if not selected:
                break

            improved_this_pass = False
            for window in selected:
                if time.perf_counter() >= deadline:
                    break
                if window.start < 0 or window.end >= len(current) or window.end <= window.start:
                    continue
                old_len = window.end - window.start
                if old_len <= 0:
                    continue
                if accept_mode == "equal_or_better":
                    max_depth = old_len + int(getattr(args, "post_polish_window_repair_depth_extra", 0))
                else:
                    max_depth = old_len - 1 + int(getattr(args, "post_polish_window_repair_depth_extra", 0))
                max_depth = min(max(0, max_depth), int(getattr(args, "post_polish_max_bridge_depth", max_depth)))
                if max_depth <= 0:
                    continue

                stats.attempts += 1
                stats.window_attempts += 1

                def bridge_call() -> tuple[list[bytes] | None, int]:
                    return _bridge_shorter_path(
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

                bridge, expanded = _with_temporary_args(
                    args,
                        {
                            "post_polish_bridge_expanded": int(
                                getattr(
                                    args,
                                    "post_polish_window_repair_expanded",
                                    getattr(args, "post_polish_bridge_expanded", 20000),
                                )
                            )
                        },
                        bridge_call,
                    )
                stats.bridge_expanded += expanded
                if bridge is None:
                    continue

                saved = old_len - (len(bridge) - 1)
                if accept_mode == "better":
                    if saved <= 0:
                        continue
                elif saved < 0:
                    continue

                candidate = current[: window.start] + bridge + current[window.end + 1 :]
                if not _path_business_ok(candidate, stack_count, max_height, cap):
                    continue

                current = candidate
                stats.improvements += 1
                stats.moves_saved += max(0, saved)
                stats.window_successes += 1
                stats.window_moves_saved += max(0, saved)
                improved_this_pass = saved > 0
                break
            if not improved_this_pass:
                break
    finally:
        stats.window_time_sec += time.perf_counter() - polish_start
    return current, stats


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

    post_polish_mode = getattr(args, "post_polish_mode", "legacy")
    window_min_stacks = int(getattr(args, "post_polish_window_min_stacks", 0) or 0)
    if post_polish_mode == "qubo_windows_v2" and stack_count >= window_min_stacks:
        return _post_polish_codes_with_qubo_windows_v2(codes, stack_count, max_height, cap, args, deadline)

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
    args._business_cap = cap
    frg_seed = _frg_seed_incumbent(instance, args, deadline)
    frg_seed_moves = len(frg_seed.moves) if frg_seed is not None else None
    frg_seed_large_layout_ok = _frg_seed_large_layout_allowed(instance, args) if frg_seed is not None else False
    frg_seed_accepted = (
        frg_seed is not None
        and (frg_seed.cap_ok or frg_seed_large_layout_ok)
    )
    if (
        frg_seed_moves is not None
        and frg_seed_moves > 0
        and bool(getattr(args, "frg_seed_prune_to_incumbent", True))
        and frg_seed_accepted
    ):
        depth_cap = min(depth_cap, max(0, frg_seed_moves - 1))
    wang_incumbent = _validated_wang_incumbent(instance, args, cap, deadline)
    wang_incumbent_moves = len(wang_incumbent.moves) if wang_incumbent is not None else None
    depth_cap = _wang_search_depth_cap(depth_cap, wang_incumbent_moves, args)

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
    args._wang_pair_qubo_calls_done = 0
    args._wang_pair_qubo_vars = 0
    args._wang_pair_qubo_time = 0.0
    args._unified_wang_qubo_calls_done = 0
    args._unified_wang_qubo_vars = 0
    args._unified_wang_qubo_time = 0.0
    args._unified_wang_fixed_prefix_selected = 0
    args._unified_wang_fix_pair_selected = 0
    args._union_diversity_qubo_calls_done = 0
    args._union_diversity_qubo_vars = 0
    args._union_diversity_qubo_time = 0.0
    args._union_diversity_qubo_selected_total = 0
    args._pareto_union_layers = 0
    args._pareto_union_seed_selected_total = 0
    args._pareto_union_qubo_calls_done = 0
    args._pareto_union_qubo_vars = 0
    args._pareto_union_qubo_time = 0.0
    args._pareto_union_qubo_selected_total = 0
    args._pareto_union_quality_selected = 0
    args._pareto_union_space_selected = 0
    args._pareto_union_repair_selected = 0
    args._pareto_union_risk_selected = 0
    args._pareto_union_distinct_parent_count_last = 0
    args._pareto_union_distinct_source_count_last = 0
    args._pareto_union_distinct_dest_count_last = 0
    args._pareto_union_distinct_signature_count_last = 0
    args._union_rollout_probe_calls = 0
    args._union_rollout_probe_candidates = 0
    args._union_rollout_probe_feasible = 0
    args._union_rollout_probe_time = 0.0
    args._union_rollout_probe_cache_hits = 0
    args._union_rollout_probe_selected_total = 0
    args._union_rollout_terminal_grafted_total = 0
    args._frg_frontier_child_calls = 0
    args._frg_frontier_child_generated = 0
    args._frg_frontier_child_added_total = 0
    args._union_rollout_probe_cache = {}
    args._union_rollout_probe_frg_cache = {}
    args._union_rollout_probe_current_results = {}
    args._micro_rollout_calls = 0
    args._micro_rollout_cache_hits = 0
    args._micro_rollout_candidates = 0
    args._micro_rollout_selected_total = 0
    args._receiver_safety_scored_total = 0
    args._micro_rollout_cache = {}
    args._micro_rollout_feature_trace_by_code = {}
    args._feasibility_escape_calls_done = 0
    args._feasibility_escape_added_total = 0
    args._feasibility_escape_active_layers = 0
    args._feasibility_escape_gate_reasons = {}
    args._frg_candidate_selected_total = 0
    args._frg_compound_appended_total = 0
    args._dlts_value_selected_total = 0
    args._bs_frg_qubo_calls_done = 0
    args._bs_frg_qubo_vars = 0
    args._bs_frg_qubo_time = 0.0
    args._bs_frg_qubo_selected_total = 0
    args._wang2017_task_calls_done = 0
    args._wang2017_task_added_total = 0
    args._wang2017_task_active_layers = 0
    args._wang2017_task_selected_tier_protection_total = 0
    args._wang2017_task_selected_stable_height_total = 0
    args._wang_rollout_candidate_codes = set()
    args._frg_frontier_child_codes = set()
    args._candidate_origin_by_code = {}
    args._rank_pool_truncated_candidates = 0
    args._group_lookahead_cache = {}
    args._best_n_history = []
    args._rescue_layers_used = 0
    args._scarcity_inner_rescue_layers = 0
    args._scarcity_inner_rescue_gate_reasons = {}
    args._tail_lookahead_calls_done = 0
    args._tail_lookahead_selected_total = 0
    args._tail_lookahead_goal_reachable_total = 0
    args._tail_lookahead_best_delta_total = 0
    args._tail_lookahead_expanded_total = 0
    args._tail_lookahead_priority_codes = set()
    args._tail_lookahead_path_next_by_code = {}
    args._tail_risk_reserve_calls_done = 0
    args._tail_risk_reserve_selected_total = 0
    args._tail_risk_reserve_scanned_total = 0
    args._tail_risk_reserve_one_step_improved_total = 0
    args._tail_risk_priority_codes = set()
    args._tail_risk_one_step_cache = {}
    args._tail_risk_recovery_cache = {}
    args._rescue_qubo_calls = 0
    args._rescue_qubo_vars = 0
    args._rescue_qubo_time = 0.0
    args._adaptive_contract_layers = 0
    args._adaptive_mid_layers = 0
    args._adaptive_wide_layers = 0
    args._adaptive_expand_layers = 0
    args._adaptive_last_width = 0
    args._adaptive_last_reason = ""
    args._survivor_dead_parent_ratio_history = list(getattr(args, "_survivor_dead_parent_ratio_history", []) or [])
    args._survivor_parent_cap_active = bool(getattr(args, "_survivor_parent_cap_active", False))
    args._survivor_parent_cap_layers = 0
    args._survivor_parent_cap_last = int(getattr(args, "parent_core_cap", 0) or 0)
    args._survivor_parent_cap_last_reason = ""
    args._survivor_parent_cap_last_signal = ""
    args._scheduler_state_last_action = ""
    args._scheduler_state_last_effective_action = ""
    args._scheduler_state_locked_action = ""
    args._scheduler_state_contract_layers = 0
    args._scheduler_state_blend_layers = 0
    args._scheduler_state_protect_layers = 0
    args._scheduler_state_rescue_layers = 0
    args._layout_target_groups = (
        _compact_target_layouts(instance)
        if getattr(args, "priority", "n_then_depth") == "target_guided"
        or float(getattr(args, "target_guided_weight", 0.0) or 0.0) > 0.0
        or float(getattr(args, "wang_probe_target_weight", 0.0) or 0.0) > 0.0
        or float(getattr(args, "unified_wang_target_gap_weight", 0.0) or 0.0) > 0.0
        or bool(getattr(args, "wang2017_task_enabled", False))
        else ()
    )
    args._case_id = case
    args._initial_n = n0
    scarcity_metrics = _initial_scarcity_metrics(instance.stacks, instance.max_height)
    args._initial_workload_hardness = scarcity_metrics.workload_hardness
    args._initial_scarcity_risk = scarcity_metrics.scarcity_risk
    args._initial_scarcity_n0 = scarcity_metrics.n0
    args._initial_scarcity_l0 = scarcity_metrics.l0
    args._initial_scarcity_one_safe_ratio = scarcity_metrics.one_safe_ratio
    args._initial_light_route_score = _light_route_score(scarcity_metrics, instance.container_count)
    args._layer_best_n = n0
    args._priority_duplication_ratio = _priority_duplication_ratio_for_instance(instance)
    args._fill_ratio = instance.container_count / max(1, instance.stack_count * instance.max_height)
    args._stack_count = instance.stack_count
    args._max_height = instance.max_height
    args._container_count = instance.container_count
    stream_candidate_limit = max(0, getattr(args, "stream_candidate_limit", 0))
    stream_bucket_topk = max(0, int(getattr(args, "stream_bucket_topk", 0) or 0))
    stream_trim_enabled = stream_candidate_limit > 0
    stream_bucket_enabled = stream_bucket_topk > 0
    stream_trim_factor = max(1.0, float(getattr(args, "stream_candidate_trim_factor", 2.0)))
    stream_trim_threshold = max(stream_candidate_limit, int(stream_candidate_limit * stream_trim_factor))

    expanded = generated = pruned_bound = pruned_business = dominated = 0
    qcalls = qvars = 0
    qtime = 0.0
    peak_frontier = len(frontier)
    peak_candidates = 0
    peak_reservoir = 0
    reservoir_added_total = 0
    reservoir_pulled_total = 0
    stream_pruned_candidates = 0
    wang_rollout_added_total = 0
    rollout_terminal_moves: list[Move] | None = None
    stream_seq = 0
    depth_reached = 0
    beam_survivor_output = getattr(args, "beam_survivor_telemetry_output", None)
    collect_survivor_metrics = beam_survivor_output is not None or bool(
        getattr(args, "survivor_parent_cap_enabled", False)
    )
    beam_survivor_records: list[dict[str, object]] = []
    dlts_feature_rows: list[dict[str, object]] = []
    dlts_feature_output = getattr(args, "dlts_feature_output", None)
    frontier_trace_rows: list[dict[str, object]] = []
    frontier_trace_output = getattr(args, "frontier_trace_output", None)
    frontier_trace_top_k = max(0, int(getattr(args, "frontier_trace_top_k", 64) or 0))
    layer_index = 0

    while target is None and frontier and time.perf_counter() < deadline:
        candidates: dict[bytes, tuple[int, int, bytes]] = {}
        candidate_worst_heap: list[tuple[_ReverseRank, int, bytes, tuple[int, int, bytes]]] = []
        candidate_bucket_heaps: dict[tuple, list[tuple[_ReverseRank, int, bytes, tuple[int, int, bytes]]]] = {}
        candidate_bucket_members: dict[tuple, set[bytes]] = {}
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
            for nxt, nxt_n in _successors_with_n_uncached(code, current_n, instance.stack_count, instance.max_height):
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
                if stream_bucket_enabled:
                    stream_seq += 1
                    stream_pruned_candidates += _offer_stream_bucket_candidate(
                        candidates,
                        candidate_bucket_heaps,
                        candidate_bucket_members,
                        nxt,
                        data,
                        args,
                        stream_seq,
                    )
                elif stream_trim_enabled:
                    candidates[nxt] = data
                    if len(candidates) > stream_trim_threshold:
                        stream_pruned_candidates += _trim_stream_candidates_batch(
                            candidates,
                            stream_candidate_limit,
                            args,
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
        if pulled and stream_bucket_enabled:
            candidate_bucket_heaps, candidate_bucket_members, rebuilt_seq = _rebuild_stream_bucket_state(candidates, args)
            stream_seq += rebuilt_seq
            for bucket, bucket_heap in list(candidate_bucket_heaps.items()):
                stream_pruned_candidates += _trim_stream_bucket_candidates(
                    candidates,
                    bucket_heap,
                    candidate_bucket_members,
                    bucket,
                    stream_bucket_topk,
                    args,
                )
        if pulled and stream_trim_enabled:
            stream_pruned_candidates += _trim_stream_candidates_batch(candidates, stream_candidate_limit, args)
        if stream_trim_enabled:
            stream_pruned_candidates += _trim_stream_candidates_batch(candidates, stream_candidate_limit, args)
        peak_candidates = max(peak_candidates, len(candidates))
        layer_best_n = _update_layer_candidate_metrics(candidates, args, cap)
        args._layer_index = layer_index
        args._best_n_history.append(layer_best_n)
        candidate_count_before_rollout = len(candidates)
        candidate_origin_by_code: dict[bytes, str] = {}
        args._candidate_origin_by_code = candidate_origin_by_code
        rollout_codes = _inject_wang_rollout_candidates(frontier, candidates, args, depth_cap, best_depth)
        args._wang_rollout_candidate_codes = set(rollout_codes)
        for code in rollout_codes:
            candidate_origin_by_code.setdefault(code, "wang_rollout")
        if rollout_codes:
            wang_rollout_added_total += max(0, len(candidates) - candidate_count_before_rollout)
            peak_candidates = max(peak_candidates, len(candidates))
            if len(candidates) != candidate_count_before_rollout:
                _update_layer_candidate_metrics(candidates, args, cap)
        frg_frontier_child_codes: set[bytes] = set()
        if not getattr(args, "frg_frontier_child_append_only", False):
            candidate_count_before_frg_frontier = len(candidates)
            frg_frontier_child_codes = _inject_frg_frontier_child_candidates(
                frontier,
                candidates,
                args,
                depth_cap,
                best_depth,
                parent_chain=parent,
            )
            if frg_frontier_child_codes:
                for code in frg_frontier_child_codes:
                    candidate_origin_by_code.setdefault(code, "frg_frontier_child")
                peak_candidates = max(peak_candidates, len(candidates))
                if len(candidates) != candidate_count_before_frg_frontier:
                    _update_layer_candidate_metrics(candidates, args, cap)
        args._frg_frontier_child_codes = set(frg_frontier_child_codes)
        selected_codes, selected_by_qubo, add_qcalls, add_qvars, add_qtime = _select_next_frontier(candidates, args)
        qcalls += add_qcalls
        qvars += add_qvars
        qtime += add_qtime
        if getattr(args, "union_rollout_terminal_graft", False):
            probe_results = getattr(args, "_union_rollout_probe_current_results", {}) or {}
            for code, rollout in probe_results.items():
                data = candidates.get(code)
                if data is None:
                    continue
                grafted = _union_rollout_probe_grafted_moves(
                    code,
                    data,
                    parent,
                    instance,
                    args,
                    rollout,
                    cap=cap,
                )
                if grafted is None:
                    continue
                if rollout_terminal_moves is None or len(grafted) < len(rollout_terminal_moves):
                    rollout_terminal_moves = grafted
                    depth_cap = min(depth_cap, max(0, len(grafted) - 1))
                args._union_rollout_terminal_grafted_total += 1
        if dlts_feature_output is not None:
            parent_rank_by_code = {code: idx + 1 for idx, (code, _n_value, _depth) in enumerate(frontier)}
            selected_set = set(selected_codes)
            ranked_candidates = _rank_bucket_topk_candidates(candidates, args)
            for rank, (code, data) in enumerate(ranked_candidates[: min(256, len(ranked_candidates))], start=1):
                dlts_feature_rows.append(
                    _dlts_feature_row(
                        case=case,
                        layer=layer_index,
                        code=code,
                        data=data,
                        parent_rank=parent_rank_by_code.get(data[2], rank),
                        selected=code in selected_set,
                        args=args,
                    )
                )
        appended_rollout_codes = _append_wang_rollout_frontier(selected_codes, candidates, args, depth_cap, best_depth)
        if appended_rollout_codes:
            rollout_codes |= appended_rollout_codes
            for code in appended_rollout_codes:
                candidate_origin_by_code.setdefault(code, "wang_rollout")
            wang_rollout_added_total += len(appended_rollout_codes)
        frg_compound_codes = _append_frg_compound_frontier(selected_codes, candidates, args, depth_cap, best_depth)
        if frg_compound_codes:
            peak_candidates = max(peak_candidates, len(candidates))
        appended_frg_frontier_child_codes = _append_frg_frontier_child_frontier(
            selected_codes,
            frontier,
            candidates,
            args,
            depth_cap,
            best_depth,
            parent_chain=parent,
        )
        if appended_frg_frontier_child_codes:
            frg_frontier_child_codes |= appended_frg_frontier_child_codes
            for code in appended_frg_frontier_child_codes:
                candidate_origin_by_code.setdefault(code, "frg_frontier_child")
            args._frg_frontier_child_codes = set(frg_frontier_child_codes)
            peak_candidates = max(peak_candidates, len(candidates))
        survivor_metrics: dict[str, int | float] | None = None
        if collect_survivor_metrics:
            parent_rank_by_code = {code: idx + 1 for idx, (code, _n_value, _depth) in enumerate(frontier)}
            survivor_metrics = _beam_survivor_parent_metrics(frontier, candidates, selected_codes, selected_by_qubo)
            _record_survivor_parent_cap_signal(args, survivor_metrics)
        if beam_survivor_output is not None and survivor_metrics is not None:
            row: dict[str, object] = {
                "case": case,
                "solver_profile": getattr(args, "solver_profile", ""),
                "selection_policy": getattr(args, "selection_policy", ""),
                "layer": layer_index,
                "layer_best_n": layer_best_n,
                "candidate_count_before_rollout": candidate_count_before_rollout,
                "rollout_candidate_count": len(rollout_codes),
                "rollout_appended_count": len(appended_rollout_codes),
                "qcalls_total": qcalls,
                "qvars_total": qvars,
                "qtime_total": qtime,
                "parent_core_cap_effective": getattr(args, "_survivor_parent_cap_last", getattr(args, "parent_core_cap", 0)),
                "parent_core_cap_reason": getattr(args, "_survivor_parent_cap_last_reason", ""),
                "survivor_dead_parent_ratio_signal": getattr(args, "_survivor_parent_cap_last_signal", ""),
            }
            row.update(survivor_metrics)
            beam_survivor_records.append(
                {
                    "row": row,
                    "selected_rank_by_code": {code: idx + 1 for idx, code in enumerate(selected_codes)},
                    "parent_by_code": {code: candidates[code][2] for code in selected_codes if code in candidates},
                    "parent_rank_by_code": parent_rank_by_code,
                }
            )
        if frontier_trace_output is not None and frontier_trace_top_k > 0:
            frontier_trace_rows.extend(
                _frontier_trace_candidate_rows(
                    case=case,
                    layer=layer_index,
                    ranked=list(getattr(args, "_frontier_trace_last_ranked", [])),
                    frontier=frontier,
                    selected_codes=selected_codes,
                    selected_by_qubo=selected_by_qubo,
                    selected_source_by_code=dict(getattr(args, "_frontier_trace_last_selected_sources", {})),
                    args=args,
                    top_k=frontier_trace_top_k,
                    candidate_origin_by_code=dict(candidate_origin_by_code),
                )
            )
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
            source_by_code[code] = (
                "qubo"
                if code in selected_by_qubo
                else (
                    "wang_rollout"
                    if code in rollout_codes
                    else (
                        "frg_frontier_child"
                        if code in frg_frontier_child_codes
                        else ("frg_compound" if code in frg_compound_codes else "deterministic")
                    )
                )
            )
            qubo_ancestor_by_code[code] = code in selected_by_qubo or qubo_ancestor_by_code.get(parent_code, False)
            next_frontier.append((code, n_value, depth))
        frontier = next_frontier
        peak_frontier = max(peak_frontier, len(frontier))
        layer_index += 1

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
    wang_incumbent_used = False
    frg_seed_used = False
    union_rollout_terminal_graft_used = False
    search_path_codes: list[bytes] = []
    if target is not None:
        codes = _backtrack(parent, target)
        original_codes = list(codes)
        search_path_codes = list(original_codes)
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
                f"post_polish_window_candidates={polish_stats.window_candidates}; "
                f"post_polish_window_qubo_selected={polish_stats.window_qubo_selected}; "
                f"post_polish_window_attempts={polish_stats.window_attempts}; "
                f"post_polish_window_successes={polish_stats.window_successes}; "
                f"post_polish_window_moves_saved={polish_stats.window_moves_saved}; "
                f"post_polish_bridge_expanded={polish_stats.bridge_expanded}"
            )
    if rollout_terminal_moves is not None and (not moves or len(rollout_terminal_moves) < len(moves)):
        moves = rollout_terminal_moves
        union_rollout_terminal_graft_used = True
        status += (
            f"; union_rollout_terminal_graft_used=True; "
            f"union_rollout_terminal_graft_moves={len(rollout_terminal_moves)}"
        )
    if frg_seed is not None and frg_seed_accepted and (not moves or len(frg_seed.moves) < len(moves)):
        moves = list(frg_seed.moves)
        frg_seed_used = True
        status += (
            f"; frg_seed_used=True; frg_seed_moves={len(frg_seed.moves)}; "
            f"frg_seed_runtime_sec={frg_seed.runtime_sec:.6f}; frg_seed_status={frg_seed.status}; "
            f"frg_seed_cap_ok={frg_seed.cap_ok}; frg_seed_large_layout_ok={frg_seed_large_layout_ok}"
        )
    if wang_incumbent is not None and (not moves or len(wang_incumbent.moves) < len(moves)):
        moves = wang_incumbent.moves
        wang_incumbent_used = True
        status += (
            f"; wang_incumbent_used=True; wang_incumbent_moves={len(wang_incumbent.moves)}; "
            f"wang_incumbent_runtime_sec={wang_incumbent.runtime_sec:.6f}"
        )
    if beam_survivor_output is not None and beam_survivor_records:
        _write_beam_survivor_rows(
            _finalize_beam_survivor_rows(beam_survivor_records, search_path_codes),
            Path(beam_survivor_output),
        )
    if dlts_feature_output is not None and dlts_feature_rows:
        _finalize_dlts_feature_rows(dlts_feature_rows, search_path_codes, status)
        _write_dlts_feature_rows(dlts_feature_rows, Path(dlts_feature_output))
    if frontier_trace_output is not None and frontier_trace_rows:
        _write_frontier_trace_rows(frontier_trace_rows, Path(frontier_trace_output))
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
        post_polish_window_candidates=polish_stats.window_candidates,
        post_polish_window_qubo_selected=polish_stats.window_qubo_selected,
        post_polish_window_attempts=polish_stats.window_attempts,
        post_polish_window_successes=polish_stats.window_successes,
        post_polish_window_moves_saved=polish_stats.window_moves_saved,
        post_polish_window_time_sec=polish_stats.window_time_sec,
        tail_n_trace=tail_n_trace,
        tail_ordered_trace=tail_ordered_trace,
        tail_ordered_gain=tail_ordered_gain,
        reservoir_size=len(reservoir),
        peak_reservoir=peak_reservoir,
        reservoir_added_total=reservoir_added_total,
        reservoir_pulled_total=reservoir_pulled_total,
        stream_pruned_candidates=stream_pruned_candidates,
        rank_pool_truncated_candidates=args._rank_pool_truncated_candidates,
        rescue_layers_used=args._rescue_layers_used,
        scarcity_inner_rescue_layers=args._scarcity_inner_rescue_layers,
        scarcity_inner_rescue_gate_reasons=_format_scarcity_inner_rescue_gate_reasons(
            args._scarcity_inner_rescue_gate_reasons
        ),
        tail_lookahead_calls=args._tail_lookahead_calls_done,
        tail_lookahead_selected_total=args._tail_lookahead_selected_total,
        tail_lookahead_goal_reachable_total=args._tail_lookahead_goal_reachable_total,
        tail_lookahead_best_delta_total=args._tail_lookahead_best_delta_total,
        tail_lookahead_expanded_total=args._tail_lookahead_expanded_total,
        tail_risk_reserve_calls=args._tail_risk_reserve_calls_done,
        tail_risk_reserve_selected_total=args._tail_risk_reserve_selected_total,
        tail_risk_reserve_scanned_total=args._tail_risk_reserve_scanned_total,
        tail_risk_reserve_one_step_improved_total=args._tail_risk_reserve_one_step_improved_total,
        wang_incumbent_moves=len(wang_incumbent.moves) if wang_incumbent is not None else 0,
        wang_incumbent_runtime_sec=wang_incumbent.runtime_sec if wang_incumbent is not None else 0.0,
        wang_incumbent_used=wang_incumbent_used,
        wang_rollout_added_total=wang_rollout_added_total,
        wang_pair_qubo_calls=args._wang_pair_qubo_calls_done,
        wang_pair_qubo_vars=args._wang_pair_qubo_vars,
        wang_pair_qubo_time_sec=args._wang_pair_qubo_time,
        unified_wang_qubo_calls=args._unified_wang_qubo_calls_done,
        unified_wang_qubo_vars=args._unified_wang_qubo_vars,
        unified_wang_qubo_time_sec=args._unified_wang_qubo_time,
        unified_wang_fixed_prefix_selected=args._unified_wang_fixed_prefix_selected,
        unified_wang_fix_pair_selected=args._unified_wang_fix_pair_selected,
        union_diversity_qubo_calls=args._union_diversity_qubo_calls_done,
        union_diversity_qubo_vars=args._union_diversity_qubo_vars,
        union_diversity_qubo_time_sec=args._union_diversity_qubo_time,
        union_diversity_qubo_selected_total=args._union_diversity_qubo_selected_total,
        pareto_union_layers=args._pareto_union_layers,
        pareto_union_seed_selected_total=args._pareto_union_seed_selected_total,
        pareto_union_qubo_calls=args._pareto_union_qubo_calls_done,
        pareto_union_qubo_vars=args._pareto_union_qubo_vars,
        pareto_union_qubo_time_sec=args._pareto_union_qubo_time,
        pareto_union_qubo_selected_total=args._pareto_union_qubo_selected_total,
        pareto_union_quality_selected=args._pareto_union_quality_selected,
        pareto_union_space_selected=args._pareto_union_space_selected,
        pareto_union_repair_selected=args._pareto_union_repair_selected,
        pareto_union_risk_selected=args._pareto_union_risk_selected,
        pareto_union_distinct_parent_count_last=args._pareto_union_distinct_parent_count_last,
        pareto_union_distinct_source_count_last=args._pareto_union_distinct_source_count_last,
        pareto_union_distinct_dest_count_last=args._pareto_union_distinct_dest_count_last,
        pareto_union_distinct_signature_count_last=args._pareto_union_distinct_signature_count_last,
        union_rollout_probe_calls=args._union_rollout_probe_calls,
        union_rollout_probe_candidates=args._union_rollout_probe_candidates,
        union_rollout_probe_feasible=args._union_rollout_probe_feasible,
        union_rollout_probe_time_sec=args._union_rollout_probe_time,
        union_rollout_probe_cache_hits=args._union_rollout_probe_cache_hits,
        union_rollout_probe_selected_total=args._union_rollout_probe_selected_total,
        union_rollout_terminal_grafted_total=args._union_rollout_terminal_grafted_total,
        union_rollout_terminal_graft_used=union_rollout_terminal_graft_used,
        micro_rollout_calls=args._micro_rollout_calls,
        micro_rollout_cache_hits=args._micro_rollout_cache_hits,
        micro_rollout_candidates=args._micro_rollout_candidates,
        micro_rollout_selected_total=args._micro_rollout_selected_total,
        receiver_safety_scored_total=args._receiver_safety_scored_total,
        frg_frontier_child_calls=args._frg_frontier_child_calls,
        frg_frontier_child_generated=args._frg_frontier_child_generated,
        frg_frontier_child_added_total=args._frg_frontier_child_added_total,
        feasibility_escape_calls=args._feasibility_escape_calls_done,
        feasibility_escape_added_total=args._feasibility_escape_added_total,
        feasibility_escape_active_layers=args._feasibility_escape_active_layers,
        feasibility_escape_gate_reasons=_format_feasibility_escape_gate_reasons(args._feasibility_escape_gate_reasons),
        frg_seed_moves=len(frg_seed.moves) if frg_seed is not None else 0,
        frg_seed_runtime_sec=frg_seed.runtime_sec if frg_seed is not None else 0.0,
        frg_seed_used=frg_seed_used,
        frg_seed_status=frg_seed.status if frg_seed is not None else "",
        frg_seed_generated=frg_seed.generated if frg_seed is not None else 0,
        frg_seed_cap_ok=frg_seed.cap_ok if frg_seed is not None else False,
        frg_seed_large_layout_ok=frg_seed_large_layout_ok,
        frg_candidate_selected_total=args._frg_candidate_selected_total,
        frg_compound_appended_total=args._frg_compound_appended_total,
        dlts_value_selected_total=args._dlts_value_selected_total,
        bs_frg_qubo_calls=args._bs_frg_qubo_calls_done,
        bs_frg_qubo_vars=args._bs_frg_qubo_vars,
        bs_frg_qubo_time_sec=args._bs_frg_qubo_time,
        bs_frg_qubo_selected_total=args._bs_frg_qubo_selected_total,
        wang2017_task_calls=args._wang2017_task_calls_done,
        wang2017_task_added_total=args._wang2017_task_added_total,
        wang2017_task_active_layers=args._wang2017_task_active_layers,
        wang2017_task_selected_tier_protection_total=args._wang2017_task_selected_tier_protection_total,
        wang2017_task_selected_stable_height_total=args._wang2017_task_selected_stable_height_total,
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
            f"wang_incumbent_enabled={getattr(args, 'wang_incumbent_enabled', False)}; "
            f"wang_incumbent_used={result.wang_incumbent_used}; "
            f"wang_incumbent_moves={result.wang_incumbent_moves}; "
            f"frg_seed_enabled={getattr(args, 'frg_seed_enabled', False)}; "
            f"frg_seed_used={result.frg_seed_used}; "
            f"frg_seed_moves={result.frg_seed_moves}; "
            f"frg_seed_status={result.frg_seed_status}; "
            f"frg_seed_cap_ok={result.frg_seed_cap_ok}; "
            f"frg_seed_large_layout_ok={result.frg_seed_large_layout_ok}; "
            f"wang_cap_enabled={getattr(args, 'wang_cap_enabled', False)}; "
            f"wang_rollout_enabled={getattr(args, 'wang_rollout_enabled', False)}; "
            f"wang_rollout_strategy={getattr(args, 'wang_rollout_strategy', 'local')}; "
            f"wang_rollout_inject_candidates={getattr(args, 'wang_rollout_inject_candidates', False)}; "
            f"wang_rollout_width={getattr(args, 'wang_rollout_width', 0)}; "
            f"wang_rollout_focus_n={getattr(args, 'wang_rollout_focus_n', 8)}; "
            f"wang_rollout_candidate_limit={getattr(args, 'wang_rollout_candidate_limit', 64)}; "
            f"wang_rollout_added_total={result.wang_rollout_added_total}; "
            f"wang_pair_qubo_enabled={getattr(args, 'wang_pair_qubo_enabled', False)}; "
            f"wang_pair_qubo_start_layer={getattr(args, 'wang_pair_qubo_start_layer', 0)}; "
            f"wang_pair_qubo_calls={result.wang_pair_qubo_calls}; "
            f"unified_wang_qubo_enabled={getattr(args, 'unified_wang_qubo_enabled', False)}; "
            f"unified_wang_qubo_calls={result.unified_wang_qubo_calls}; "
            f"unified_wang_qubo_vars={result.unified_wang_qubo_vars}; "
            f"unified_wang_qubo_time_sec={result.unified_wang_qubo_time_sec:.6f}; "
            f"unified_wang_fixed_prefix_selected={result.unified_wang_fixed_prefix_selected}; "
            f"unified_wang_fix_pair_selected={result.unified_wang_fix_pair_selected}; "
            f"priority={args.priority}; beam_width={args.beam_width}; core_width={args.core_width}; "
            f"parent_core_cap={getattr(args, 'parent_core_cap', 0)}; "
            f"survivor_parent_cap_enabled={getattr(args, 'survivor_parent_cap_enabled', False)}; "
            f"survivor_parent_cap_layers={getattr(args, '_survivor_parent_cap_layers', 0)}; "
            f"survivor_parent_cap_last={getattr(args, '_survivor_parent_cap_last', getattr(args, 'parent_core_cap', 0))}; "
            f"survivor_parent_cap_last_reason={getattr(args, '_survivor_parent_cap_last_reason', '')}; "
            f"peak_frontier={result.peak_frontier}; peak_candidates={result.peak_candidates}; "
            f"depth_reached={result.depth_reached}; open={result.open_count}; "
            f"qubo_used={args.use_qubo}; qubo_mode={args.qubo_mode}; "
            f"qubo_start_layer={getattr(args, 'qubo_start_layer', 0)}; qubo_focus_n={args.qubo_focus_n}; "
            f"qubo_first={args.qubo_first}; "
            f"group_qubo_supplement={getattr(args, 'group_qubo_supplement', False)}; "
            f"max_group_qubo_calls={getattr(args, 'max_group_qubo_calls', 0)}; "
            f"qubo_cohorts_per_layer={args.qubo_cohorts_per_layer}; "
            f"qubo_calls={result.qubo_calls}; qubo_vars={result.qubo_vars}; "
            f"qubo_time_sec={result.qubo_time_sec:.6f}; qubo_core_width={args.qubo_core_width}; "
            f"qubo_supplement_width={args.qubo_supplement_width}; "
            f"qubo_target_share={args.qubo_target_share}; "
            f"qubo_pair_policy={getattr(args, 'qubo_pair_policy', 'baseline')}; "
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
            f"post_polish_mode={getattr(args, 'post_polish_mode', 'legacy')}; "
            f"post_polish_accept={getattr(args, 'post_polish_accept', 'better')}; "
            f"post_polish_window_min_stacks={getattr(args, 'post_polish_window_min_stacks', 0)}; "
            f"post_polish_attempts={result.post_polish_attempts}; "
            f"post_polish_improvements={result.post_polish_improvements}; "
            f"post_polish_moves_saved={result.post_polish_moves_saved}; "
            f"post_polish_window_candidates={result.post_polish_window_candidates}; "
            f"post_polish_window_qubo_selected={result.post_polish_window_qubo_selected}; "
            f"post_polish_window_attempts={result.post_polish_window_attempts}; "
            f"post_polish_window_successes={result.post_polish_window_successes}; "
            f"post_polish_window_moves_saved={result.post_polish_window_moves_saved}; "
            f"post_polish_window_time_sec={result.post_polish_window_time_sec:.6f}; "
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
            f"stream_bucket_topk={getattr(args, 'stream_bucket_topk', 0)}; "
            f"stream_pruned_candidates={result.stream_pruned_candidates}; "
            f"rank_pool_size={getattr(args, 'rank_pool_size', 0)}; "
            f"rank_pool_min_candidates={getattr(args, 'rank_pool_min_candidates', 0)}; "
            f"rank_bucket_topk={getattr(args, 'rank_bucket_topk', 0)}; "
            f"rank_bucket_key={getattr(args, 'rank_bucket_key', 'n_depth_parent')}; "
            f"rank_bucket_n_band_size={getattr(args, 'rank_bucket_n_band_size', 0)}; "
            f"rank_bucket_topk_high_n={getattr(args, 'rank_bucket_topk_high_n', 0)}; "
            f"rank_bucket_topk_min_candidates={getattr(args, 'rank_bucket_topk_min_candidates', 0)}; "
            f"rank_pool_truncated_candidates={result.rank_pool_truncated_candidates}; "
            f"scheduler_adaptive_min_stacks={getattr(args, 'scheduler_adaptive_min_stacks', 0)}; "
            f"scheduler_adaptive_high_n={getattr(args, 'scheduler_adaptive_high_n', 0)}; "
            f"scheduler_adaptive_cv1_only={getattr(args, 'scheduler_adaptive_cv1_only', False)}; "
            f"scheduler_adaptive_max_duplication_ratio={getattr(args, 'scheduler_adaptive_max_duplication_ratio', 1.0)}; "
            f"scheduler_adaptive_min_best_n_ratio={getattr(args, 'scheduler_adaptive_min_best_n_ratio', 0.0)}; "
            f"scheduler_adaptive_max_candidate_pressure={getattr(args, 'scheduler_adaptive_max_candidate_pressure', 0.0)}; "
            f"scheduler_adaptive_max_parent_top_share={getattr(args, 'scheduler_adaptive_max_parent_top_share', 0.0)}; "
            f"scheduler_adaptive_plateau_window={getattr(args, 'scheduler_adaptive_plateau_window', 0)}; "
            f"scheduler_adaptive_cap_slack={getattr(args, 'scheduler_adaptive_cap_slack', 0)}; "
            f"scarcity_inner_rescue_enabled={getattr(args, 'scarcity_inner_rescue_enabled', False)}; "
            f"scarcity_inner_rescue_layers={getattr(args, '_scarcity_inner_rescue_layers', 0)}; "
            f"scarcity_inner_rescue_gate_reasons={_format_scarcity_inner_rescue_gate_reasons(getattr(args, '_scarcity_inner_rescue_gate_reasons', {}))}; "
            f"scarcity_group_qubo_enabled={getattr(args, 'scarcity_group_qubo_enabled', False)}; "
            f"initial_scarcity_risk={getattr(args, '_initial_scarcity_risk', 0.0):.6f}; "
            f"initial_workload_hardness={getattr(args, '_initial_workload_hardness', 0.0):.6f}; "
            f"initial_light_route_score={getattr(args, '_initial_light_route_score', 0.0):.6f}; "
            f"scheduler_state_controller={getattr(args, 'scheduler_state_controller', 'off')}; "
            f"scheduler_state_contract_pressure={getattr(args, 'scheduler_state_contract_pressure', 0.0)}; "
            f"scheduler_state_max_contract_pressure={getattr(args, 'scheduler_state_max_contract_pressure', 0.0)}; "
            f"scheduler_state_min_best_n_ratio={getattr(args, 'scheduler_state_min_best_n_ratio', 0.0)}; "
            f"scheduler_state_progress_window={getattr(args, 'scheduler_state_progress_window', 0)}; "
            f"scheduler_state_progress_delta={getattr(args, 'scheduler_state_progress_delta', 0)}; "
            f"scheduler_state_failed_contract_cooldown={getattr(args, 'scheduler_state_failed_contract_cooldown', 0)}; "
            f"scheduler_state_post_rescue_plateau_window={getattr(args, 'scheduler_state_post_rescue_plateau_window', 0)}; "
            f"scheduler_state_post_rescue_plateau_delta={getattr(args, 'scheduler_state_post_rescue_plateau_delta', 0)}; "
            f"scheduler_state_post_rescue_min_ratio={getattr(args, 'scheduler_state_post_rescue_min_ratio', 0.0)}; "
            f"scheduler_state_post_rescue_action={getattr(args, 'scheduler_state_post_rescue_action', '')}; "
            f"scheduler_state_initial_plateau_window={getattr(args, 'scheduler_state_initial_plateau_window', 0)}; "
            f"scheduler_state_initial_plateau_min_ratio={getattr(args, 'scheduler_state_initial_plateau_min_ratio', 0.0)}; "
            f"scheduler_state_initial_plateau_action={getattr(args, 'scheduler_state_initial_plateau_action', '')}; "
            f"scheduler_state_initial_plateau_lock={getattr(args, 'scheduler_state_initial_plateau_lock', False)}; "
            f"scheduler_state_initial_plateau_min_duplication={getattr(args, 'scheduler_state_initial_plateau_min_duplication', 0.0)}; "
            f"scheduler_state_initial_plateau_unique_min_n={getattr(args, 'scheduler_state_initial_plateau_unique_min_n', 0)}; "
            f"scheduler_state_protect_parent_top_share={getattr(args, 'scheduler_state_protect_parent_top_share', 0.0)}; "
            f"scheduler_state_protect_n_band={getattr(args, 'scheduler_state_protect_n_band', 0)}; "
            f"scheduler_state_top_window={getattr(args, 'scheduler_state_top_window', 0)}; "
            f"scheduler_state_rescue_plateau_window={getattr(args, 'scheduler_state_rescue_plateau_window', 0)}; "
            f"scheduler_state_rescue_plateau_delta={getattr(args, 'scheduler_state_rescue_plateau_delta', 0)}; "
            f"scheduler_state_rescue_cap_slack={getattr(args, 'scheduler_state_rescue_cap_slack', 0)}; "
            f"scheduler_state_last_action={getattr(args, '_scheduler_state_last_action', '')}; "
            f"scheduler_state_last_effective_action={getattr(args, '_scheduler_state_last_effective_action', '')}; "
            f"scheduler_state_locked_action={getattr(args, '_scheduler_state_locked_action', '')}; "
            f"scheduler_state_contract_layers={getattr(args, '_scheduler_state_contract_layers', 0)}; "
            f"scheduler_state_blend_layers={getattr(args, '_scheduler_state_blend_layers', 0)}; "
            f"scheduler_state_protect_layers={getattr(args, '_scheduler_state_protect_layers', 0)}; "
            f"scheduler_state_rescue_layers={getattr(args, '_scheduler_state_rescue_layers', 0)}; "
            f"adaptive_high_n={getattr(args, 'adaptive_high_n', 0)}; "
            f"adaptive_low_n={getattr(args, 'adaptive_low_n', 0)}; "
            f"adaptive_high_frontier={getattr(args, 'adaptive_high_frontier', 0)}; "
            f"adaptive_mid_frontier={getattr(args, 'adaptive_mid_frontier', 0)}; "
            f"adaptive_low_frontier={getattr(args, 'adaptive_low_frontier', 0)}; "
            f"adaptive_core_ratio={getattr(args, 'adaptive_core_ratio', 0.0)}; "
            f"adaptive_candidate_pool={getattr(args, 'adaptive_candidate_pool', 0)}; "
            f"adaptive_controller={getattr(args, 'adaptive_controller', 'tiered')}; "
            f"adaptive_expand_frontier={getattr(args, 'adaptive_expand_frontier', 0)}; "
            f"adaptive_contract_pressure={getattr(args, 'adaptive_contract_pressure', 0.0)}; "
            f"adaptive_mid_pressure={getattr(args, 'adaptive_mid_pressure', 0.0)}; "
            f"adaptive_min_best_n_ratio={getattr(args, 'adaptive_min_best_n_ratio', 0.0)}; "
            f"adaptive_max_duplication_ratio={getattr(args, 'adaptive_max_duplication_ratio', 1.0)}; "
            f"adaptive_parent_top_share={getattr(args, 'adaptive_parent_top_share', 0.0)}; "
            f"adaptive_plateau_window={getattr(args, 'adaptive_plateau_window', 0)}; "
            f"adaptive_plateau_delta={getattr(args, 'adaptive_plateau_delta', 0)}; "
            f"adaptive_cap_slack={getattr(args, 'adaptive_cap_slack', 0)}; "
            f"adaptive_large_stack_min_stacks={getattr(args, 'adaptive_large_stack_min_stacks', 0)}; "
            f"adaptive_large_stack_expand_frontier={getattr(args, 'adaptive_large_stack_expand_frontier', 0)}; "
            f"adaptive_large_stack_plateau_window={getattr(args, 'adaptive_large_stack_plateau_window', 0)}; "
            f"adaptive_large_stack_cap_slack={getattr(args, 'adaptive_large_stack_cap_slack', 0)}; "
            f"adaptive_contract_layers={getattr(args, '_adaptive_contract_layers', 0)}; "
            f"adaptive_mid_layers={getattr(args, '_adaptive_mid_layers', 0)}; "
            f"adaptive_wide_layers={getattr(args, '_adaptive_wide_layers', 0)}; "
            f"adaptive_expand_layers={getattr(args, '_adaptive_expand_layers', 0)}; "
            f"adaptive_last_width={getattr(args, '_adaptive_last_width', 0)}; "
            f"adaptive_last_reason={getattr(args, '_adaptive_last_reason', '')}; "
            f"bucket_key={getattr(args, 'bucket_key', 'n_depth')}; "
            f"bucket_n_band_size={getattr(args, 'bucket_n_band_size', 0)}; "
            f"bucket_core_ratio={getattr(args, 'bucket_core_ratio', 0.0)}; "
            f"bucket_qubo_share={getattr(args, 'bucket_qubo_share', 0.0)}; "
            f"bucket_max_qubo_vars={getattr(args, 'bucket_max_qubo_vars', 0)}; "
            f"bucket_qubo_select={getattr(args, 'bucket_qubo_select', 0)}; "
            f"bucket_min_candidates={getattr(args, 'bucket_min_candidates', 0)}; "
            f"rescue_layers_used={getattr(args, '_rescue_layers_used', 0)}; "
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
            "deterministic_qubo_sampler": getattr(args, "deterministic_qubo_sampler", False),
            "target_guided_weight": getattr(args, "target_guided_weight", 0.0),
            "target_guided_focus_n": getattr(args, "target_guided_focus_n", 8),
            "target_guided_supplement_width": getattr(args, "target_guided_supplement_width", 0),
            "target_guided_n_window": getattr(args, "target_guided_n_window", 0),
            "wang_probe_supplement_width": getattr(args, "wang_probe_supplement_width", 0),
            "wang_probe_focus_n": getattr(args, "wang_probe_focus_n", 8),
            "wang_probe_n_window": getattr(args, "wang_probe_n_window", 0),
            "wang_probe_fixed_weight": getattr(args, "wang_probe_fixed_weight", 1.0),
            "wang_probe_pair_weight": getattr(args, "wang_probe_pair_weight", 0.0),
            "wang_probe_target_weight": getattr(args, "wang_probe_target_weight", 0.0),
            "wang_probe_plateau_window": getattr(args, "wang_probe_plateau_window", 0),
            "wang_probe_plateau_delta": getattr(args, "wang_probe_plateau_delta", 0),
            "wang_incumbent_enabled": getattr(args, "wang_incumbent_enabled", False),
            "wang_incumbent_moves": result.wang_incumbent_moves,
            "wang_incumbent_runtime_sec": result.wang_incumbent_runtime_sec,
            "wang_incumbent_used": result.wang_incumbent_used,
            "wang_cap_enabled": getattr(args, "wang_cap_enabled", False),
            "wang_rollout_enabled": getattr(args, "wang_rollout_enabled", False),
            "wang_rollout_strategy": getattr(args, "wang_rollout_strategy", "local"),
            "wang_rollout_inject_candidates": getattr(args, "wang_rollout_inject_candidates", False),
            "wang_rollout_width": getattr(args, "wang_rollout_width", 0),
            "wang_rollout_focus_n": getattr(args, "wang_rollout_focus_n", 8),
            "wang_rollout_candidate_limit": getattr(args, "wang_rollout_candidate_limit", 64),
            "wang_rollout_added_total": result.wang_rollout_added_total,
            "wang_pair_qubo_enabled": getattr(args, "wang_pair_qubo_enabled", False),
            "wang_pair_qubo_start_layer": getattr(args, "wang_pair_qubo_start_layer", 0),
            "wang_pair_qubo_calls": result.wang_pair_qubo_calls,
            "wang_pair_qubo_vars": result.wang_pair_qubo_vars,
            "wang_pair_qubo_time_sec": result.wang_pair_qubo_time_sec,
            "wang_feature_fixed_prefix_weight": getattr(args, "wang_feature_fixed_prefix_weight", 0.0),
            "wang_feature_fix_pair_weight": getattr(args, "wang_feature_fix_pair_weight", 0.0),
            "wang_feature_risk_weight": getattr(args, "wang_feature_risk_weight", 0.0),
            "wang_feature_same_parent_weight": getattr(args, "wang_feature_same_parent_weight", 0.0),
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
            "unified_wang_rollout_bonus": getattr(args, "unified_wang_rollout_bonus", 0.0),
            "unified_wang_qubo_calls": result.unified_wang_qubo_calls,
            "unified_wang_qubo_vars": result.unified_wang_qubo_vars,
            "unified_wang_qubo_time_sec": result.unified_wang_qubo_time_sec,
            "unified_wang_fixed_prefix_selected": result.unified_wang_fixed_prefix_selected,
            "unified_wang_fix_pair_selected": result.unified_wang_fix_pair_selected,
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
            "union_diversity_qubo_calls": result.union_diversity_qubo_calls,
            "union_diversity_qubo_vars": result.union_diversity_qubo_vars,
            "union_diversity_qubo_time_sec": result.union_diversity_qubo_time_sec,
            "union_diversity_qubo_selected_total": result.union_diversity_qubo_selected_total,
            "pareto_union_enabled": getattr(args, "pareto_union_enabled", False),
            "pareto_union_candidate_window": getattr(args, "pareto_union_candidate_window", 128),
            "pareto_union_seed_width": getattr(args, "pareto_union_seed_width", 0),
            "pareto_union_qubo_width": getattr(args, "pareto_union_qubo_width", 0),
            "pareto_union_max_vars": getattr(args, "pareto_union_max_vars", 96),
            "pareto_union_n_band_size": getattr(args, "pareto_union_n_band_size", 2),
            "pareto_union_quality_weight": getattr(args, "pareto_union_quality_weight", 1.0),
            "pareto_union_repair_weight": getattr(args, "pareto_union_repair_weight", 0.20),
            "pareto_union_space_weight": getattr(args, "pareto_union_space_weight", 0.12),
            "pareto_union_risk_weight": getattr(args, "pareto_union_risk_weight", 0.16),
            "pareto_union_same_parent_penalty": getattr(args, "pareto_union_same_parent_penalty", 0.55),
            "pareto_union_same_source_penalty": getattr(args, "pareto_union_same_source_penalty", 0.25),
            "pareto_union_same_dest_penalty": getattr(args, "pareto_union_same_dest_penalty", 0.20),
            "pareto_union_same_source_dest_penalty": getattr(args, "pareto_union_same_source_dest_penalty", 0.35),
            "pareto_union_signature_penalty": getattr(args, "pareto_union_signature_penalty", 0.20),
            "pareto_union_reduce_signature_penalty": getattr(args, "pareto_union_reduce_signature_penalty", 0.15),
            "pareto_union_n_band_penalty": getattr(args, "pareto_union_n_band_penalty", 0.08),
            "pareto_union_risk_bucket_penalty": getattr(args, "pareto_union_risk_bucket_penalty", 0.08),
            "pareto_union_layers": result.pareto_union_layers,
            "pareto_union_seed_selected_total": result.pareto_union_seed_selected_total,
            "pareto_union_qubo_calls": result.pareto_union_qubo_calls,
            "pareto_union_qubo_vars": result.pareto_union_qubo_vars,
            "pareto_union_qubo_time_sec": result.pareto_union_qubo_time_sec,
            "pareto_union_qubo_selected_total": result.pareto_union_qubo_selected_total,
            "pareto_union_quality_selected": result.pareto_union_quality_selected,
            "pareto_union_space_selected": result.pareto_union_space_selected,
            "pareto_union_repair_selected": result.pareto_union_repair_selected,
            "pareto_union_risk_selected": result.pareto_union_risk_selected,
            "pareto_union_distinct_parent_count_last": result.pareto_union_distinct_parent_count_last,
            "pareto_union_distinct_source_count_last": result.pareto_union_distinct_source_count_last,
            "pareto_union_distinct_dest_count_last": result.pareto_union_distinct_dest_count_last,
            "pareto_union_distinct_signature_count_last": result.pareto_union_distinct_signature_count_last,
            "union_rollout_probe_enabled": getattr(args, "union_rollout_probe_enabled", False),
            "union_rollout_probe_max_candidates": getattr(args, "union_rollout_probe_max_candidates", 0),
            "union_rollout_probe_time_limit": getattr(args, "union_rollout_probe_time_limit", 0.0),
            "union_rollout_probe_max_steps": getattr(args, "union_rollout_probe_max_steps", 0),
            "union_rollout_probe_feasible_bonus": getattr(args, "union_rollout_probe_feasible_bonus", 0.0),
            "union_rollout_probe_final_n_weight": getattr(args, "union_rollout_probe_final_n_weight", 0.0),
            "union_rollout_probe_moves_weight": getattr(args, "union_rollout_probe_moves_weight", 0.0),
            "union_rollout_probe_dead_end_penalty": getattr(args, "union_rollout_probe_dead_end_penalty", 0.0),
            "union_rollout_probe_calls": result.union_rollout_probe_calls,
            "union_rollout_probe_candidates": result.union_rollout_probe_candidates,
            "union_rollout_probe_feasible": result.union_rollout_probe_feasible,
            "union_rollout_probe_time_sec": result.union_rollout_probe_time_sec,
            "union_rollout_probe_cache_hits": result.union_rollout_probe_cache_hits,
            "union_rollout_probe_selected_total": result.union_rollout_probe_selected_total,
            "union_rollout_terminal_graft": getattr(args, "union_rollout_terminal_graft", False),
            "union_rollout_terminal_graft_respect_cap": getattr(
                args,
                "union_rollout_terminal_graft_respect_cap",
                True,
            ),
            "union_rollout_terminal_grafted_total": result.union_rollout_terminal_grafted_total,
            "union_rollout_terminal_graft_used": result.union_rollout_terminal_graft_used,
            "micro_rollout_enabled": getattr(args, "micro_rollout_enabled", False),
            "micro_rollout_trace_only": getattr(args, "micro_rollout_trace_only", False),
            "receiver_safety_enabled": getattr(args, "receiver_safety_enabled", False),
            "micro_rollout_window": getattr(args, "micro_rollout_window", 0),
            "micro_first_child_limit": getattr(args, "micro_first_child_limit", 0),
            "micro_second_child_limit": getattr(args, "micro_second_child_limit", 0),
            "micro_cache_enabled": getattr(args, "micro_cache_enabled", True),
            "w_micro_delta_n": getattr(args, "w_micro_delta_n", 0.0),
            "w_micro_safe_continuation": getattr(args, "w_micro_safe_continuation", 0.0),
            "w_receiver_safe_delta": getattr(args, "w_receiver_safe_delta", 0.0),
            "w_last_buffer_consumed": getattr(args, "w_last_buffer_consumed", 0.0),
            "w_height_pressure": getattr(args, "w_height_pressure", 0.0),
            "micro_rollout_calls": result.micro_rollout_calls,
            "micro_rollout_cache_hits": result.micro_rollout_cache_hits,
            "micro_rollout_candidates": result.micro_rollout_candidates,
            "micro_rollout_selected_total": result.micro_rollout_selected_total,
            "receiver_safety_scored_total": result.receiver_safety_scored_total,
            "frg_frontier_child_enabled": getattr(args, "frg_frontier_child_enabled", False),
            "frg_frontier_child_parent_width": getattr(args, "frg_frontier_child_parent_width", 0),
            "frg_frontier_child_limit": getattr(args, "frg_frontier_child_limit", 0),
            "frg_frontier_child_max_steps": getattr(args, "frg_frontier_child_max_steps", 1),
            "frg_frontier_child_append_only": getattr(args, "frg_frontier_child_append_only", False),
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
            "frg_frontier_child_calls": result.frg_frontier_child_calls,
            "frg_frontier_child_generated": result.frg_frontier_child_generated,
            "frg_frontier_child_added_total": result.frg_frontier_child_added_total,
            "feasibility_escape_enabled": getattr(args, "feasibility_escape_enabled", False),
            "feasibility_escape_supplement_width": getattr(args, "feasibility_escape_supplement_width", 0),
            "feasibility_escape_focus_n": getattr(args, "feasibility_escape_focus_n", 8),
            "feasibility_escape_n_window": getattr(args, "feasibility_escape_n_window", 1),
            "feasibility_escape_dead_end_threshold": getattr(args, "feasibility_escape_dead_end_threshold", 0.45),
            "feasibility_escape_min_temporary_slack": getattr(args, "feasibility_escape_min_temporary_slack", 1),
            "feasibility_escape_calls": result.feasibility_escape_calls,
            "feasibility_escape_added_total": result.feasibility_escape_added_total,
            "feasibility_escape_active_layers": getattr(
                result,
                "feasibility_escape_active_layers",
                getattr(args, "_feasibility_escape_active_layers", 0),
            ),
            "feasibility_escape_gate_reasons": getattr(
                result,
                "feasibility_escape_gate_reasons",
                _format_feasibility_escape_gate_reasons(getattr(args, "_feasibility_escape_gate_reasons", {})),
            ),
            "candidate_ledger_score_enabled": getattr(args, "candidate_ledger_score_enabled", False),
            "candidate_ledger_wang_fixed_weight": getattr(args, "candidate_ledger_wang_fixed_weight", 0.0),
            "candidate_ledger_wang_pair_weight": getattr(args, "candidate_ledger_wang_pair_weight", 0.0),
            "candidate_ledger_frg_fill_weight": getattr(args, "candidate_ledger_frg_fill_weight", 0.0),
            "candidate_ledger_frg_reduce_weight": getattr(args, "candidate_ledger_frg_reduce_weight", 0.0),
            "candidate_ledger_frg_unblocked_weight": getattr(args, "candidate_ledger_frg_unblocked_weight", 0.0),
            "candidate_ledger_frg_assignment_weight": getattr(args, "candidate_ledger_frg_assignment_weight", 0.0),
            "candidate_ledger_rollout_bonus": getattr(args, "candidate_ledger_rollout_bonus", 0.0),
            "candidate_ledger_rank_window": getattr(args, "candidate_ledger_rank_window", 0),
            "frg_seed_enabled": getattr(args, "frg_seed_enabled", False),
            "frg_seed_time_limit": getattr(args, "frg_seed_time_limit", 0.0),
            "frg_seed_min_max_steps": getattr(args, "frg_seed_min_max_steps", 0),
            "frg_seed_max_steps_multiplier": getattr(args, "frg_seed_max_steps_multiplier", 0),
            "frg_seed_max_steps_cap": getattr(args, "frg_seed_max_steps_cap", 0),
            "frg_seed_respect_cap": getattr(args, "frg_seed_respect_cap", True),
            "frg_seed_prune_to_incumbent": getattr(args, "frg_seed_prune_to_incumbent", True),
            "frg_seed_allow_large_over_cap": getattr(args, "frg_seed_allow_large_over_cap", False),
            "frg_seed_paper_pure": getattr(args, "frg_seed_paper_pure", True),
            "frg_seed_moves": result.frg_seed_moves,
            "frg_seed_runtime_sec": result.frg_seed_runtime_sec,
            "frg_seed_used": result.frg_seed_used,
            "frg_seed_status": result.frg_seed_status,
            "frg_seed_generated": result.frg_seed_generated,
            "frg_seed_cap_ok": result.frg_seed_cap_ok,
            "frg_seed_large_layout_ok": result.frg_seed_large_layout_ok,
            "frg_candidate_supplement_enabled": getattr(args, "frg_candidate_supplement_enabled", False),
            "frg_candidate_supplement_width": getattr(args, "frg_candidate_supplement_width", 0),
            "frg_candidate_focus_n": getattr(args, "frg_candidate_focus_n", -1),
            "frg_candidate_n_window": getattr(args, "frg_candidate_n_window", 2),
            "frg_candidate_max_ranked": getattr(args, "frg_candidate_max_ranked", 96),
            "frg_candidate_scarcity_gate_enabled": getattr(args, "frg_candidate_scarcity_gate_enabled", False),
            "frg_candidate_selected_total": result.frg_candidate_selected_total,
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
            "frg_compound_appended_total": result.frg_compound_appended_total,
            "dlts_value_supplement_enabled": getattr(args, "dlts_value_supplement_enabled", False),
            "dlts_value_supplement_width": getattr(args, "dlts_value_supplement_width", 0),
            "dlts_value_selected_total": result.dlts_value_selected_total,
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
            "bs_frg_qubo_calls": result.bs_frg_qubo_calls,
            "bs_frg_qubo_vars": result.bs_frg_qubo_vars,
            "bs_frg_qubo_time_sec": result.bs_frg_qubo_time_sec,
            "bs_frg_qubo_selected_total": result.bs_frg_qubo_selected_total,
            "wang2017_task_enabled": getattr(args, "wang2017_task_enabled", False),
            "wang2017_task_supplement_width": getattr(args, "wang2017_task_supplement_width", 0),
            "wang2017_task_focus_n": getattr(args, "wang2017_task_focus_n", 8),
            "wang2017_task_n_window": getattr(args, "wang2017_task_n_window", 1),
            "wang2017_task_plateau_window": getattr(args, "wang2017_task_plateau_window", 0),
            "wang2017_task_plateau_delta": getattr(args, "wang2017_task_plateau_delta", 0),
            "wang2017_task_calls": getattr(result, "wang2017_task_calls", 0),
            "wang2017_task_added_total": getattr(result, "wang2017_task_added_total", 0),
            "wang2017_task_active_layers": getattr(
                result,
                "wang2017_task_active_layers",
                getattr(args, "_wang2017_task_active_layers", 0),
            ),
            "wang2017_task_selected_tier_protection_total": getattr(
                result,
                "wang2017_task_selected_tier_protection_total",
                getattr(args, "_wang2017_task_selected_tier_protection_total", 0),
            ),
            "wang2017_task_selected_stable_height_total": getattr(
                result,
                "wang2017_task_selected_stable_height_total",
                getattr(args, "_wang2017_task_selected_stable_height_total", 0),
            ),
            "target_layout_count": len(getattr(args, "_layout_target_groups", ()) or ()),
            "beam_width": args.beam_width,
            "core_width": args.core_width,
            "parent_core_cap": getattr(args, "parent_core_cap", 0),
            "survivor_parent_cap_enabled": getattr(args, "survivor_parent_cap_enabled", False),
            "survivor_parent_cap_trigger_ratio": getattr(args, "survivor_parent_cap_trigger_ratio", 0.70),
            "survivor_parent_cap_release_ratio": getattr(args, "survivor_parent_cap_release_ratio", 0.55),
            "survivor_parent_cap_value": getattr(args, "survivor_parent_cap_value", 3),
            "survivor_parent_cap_window": getattr(args, "survivor_parent_cap_window", 2),
            "survivor_parent_cap_min_n": getattr(args, "survivor_parent_cap_min_n", 8),
            "survivor_parent_cap_layers": getattr(args, "_survivor_parent_cap_layers", 0),
            "survivor_parent_cap_last": getattr(args, "_survivor_parent_cap_last", getattr(args, "parent_core_cap", 0)),
            "survivor_parent_cap_last_reason": getattr(args, "_survivor_parent_cap_last_reason", ""),
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
            "scarcity_inner_rescue_enabled": getattr(args, "scarcity_inner_rescue_enabled", False),
            "scarcity_inner_rescue_risk_threshold": getattr(args, "scarcity_inner_rescue_risk_threshold", 0.0),
            "scarcity_inner_rescue_max_l0": getattr(args, "scarcity_inner_rescue_max_l0", 0),
            "scarcity_inner_rescue_min_one_safe_ratio": getattr(
                args, "scarcity_inner_rescue_min_one_safe_ratio", 0.0
            ),
            "scarcity_inner_rescue_min_n": getattr(args, "scarcity_inner_rescue_min_n", 0),
            "scarcity_inner_rescue_min_candidates": getattr(args, "scarcity_inner_rescue_min_candidates", 0),
            "scarcity_inner_rescue_layers": getattr(args, "_scarcity_inner_rescue_layers", 0),
            "scarcity_inner_rescue_gate_reasons": _format_scarcity_inner_rescue_gate_reasons(
                getattr(args, "_scarcity_inner_rescue_gate_reasons", {})
            ),
            "tail_lookahead_enabled": getattr(args, "tail_lookahead_enabled", False),
            "tail_lookahead_core_path_only": getattr(args, "tail_lookahead_core_path_only", False),
            "tail_lookahead_max_stack_count": getattr(args, "tail_lookahead_max_stack_count", 0),
            "tail_lookahead_focus_n": getattr(args, "tail_lookahead_focus_n", 2),
            "tail_lookahead_depth": getattr(args, "tail_lookahead_depth", 3),
            "tail_lookahead_max_seed_candidates": getattr(args, "tail_lookahead_max_seed_candidates", 0),
            "tail_lookahead_supplement_width": getattr(args, "tail_lookahead_supplement_width", 0),
            "tail_lookahead_max_expanded": getattr(args, "tail_lookahead_max_expanded", 0),
            "tail_lookahead_calls": result.tail_lookahead_calls,
            "tail_lookahead_selected_total": result.tail_lookahead_selected_total,
            "tail_lookahead_goal_reachable_total": result.tail_lookahead_goal_reachable_total,
            "tail_lookahead_best_delta_total": result.tail_lookahead_best_delta_total,
            "tail_lookahead_expanded_total": result.tail_lookahead_expanded_total,
            "tail_risk_reserve_enabled": getattr(args, "tail_risk_reserve_enabled", False),
            "tail_risk_focus_n": getattr(args, "tail_risk_focus_n", 2),
            "tail_risk_n_window": getattr(args, "tail_risk_n_window", 1),
            "tail_risk_reserve_width": getattr(args, "tail_risk_reserve_width", 0),
            "tail_risk_max_scan": getattr(args, "tail_risk_max_scan", 0),
            "tail_risk_min_candidates": getattr(args, "tail_risk_min_candidates", 0),
            "tail_risk_min_depth_slack": getattr(args, "tail_risk_min_depth_slack", 0),
            "tail_risk_max_depth_slack": getattr(args, "tail_risk_max_depth_slack", 0),
            "tail_risk_max_stack_count": getattr(args, "tail_risk_max_stack_count", 0),
            "tail_risk_reserve_calls": result.tail_risk_reserve_calls,
            "tail_risk_reserve_selected_total": result.tail_risk_reserve_selected_total,
            "tail_risk_reserve_scanned_total": result.tail_risk_reserve_scanned_total,
            "tail_risk_reserve_one_step_improved_total": result.tail_risk_reserve_one_step_improved_total,
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
            "window_polish_saved": int(sum(_numeric(row, "post_polish_window_moves_saved") for row in selected)),
            "window_polish_attempts": int(sum(_numeric(row, "post_polish_window_attempts") for row in selected)),
            "window_polish_successes": int(sum(_numeric(row, "post_polish_window_successes") for row in selected)),
            "peak_candidates_max": int(max((_numeric(row, "peak_candidates") for row in selected), default=0)),
            "peak_candidates_avg": (
                sum(_numeric(row, "peak_candidates") for row in selected) / len(selected) if selected else 0.0
            ),
            "stream_pruned": int(sum(_numeric(row, "stream_pruned_candidates") for row in selected)),
            "rank_pool_truncated": int(sum(_numeric(row, "rank_pool_truncated_candidates") for row in selected)),
            "rescue_layers": int(sum(_numeric(row, "rescue_layers_used") for row in selected)),
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
        "| scope | cases | solved | moves total | avg runtime sec | max runtime sec | qubo calls | qubo vars | qubo time sec | qubo path | qubo ancestor | group qcalls | group qtime sec | polish saved | window saved | window attempts | window successes | peak cand max | peak cand avg | stream pruned | rank-pool truncated | rescue layers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in [("all", all_summary), ("bay26", bay26_summary)]:
        lines.append(
            f"| {label} | {summary['cases']} | {summary['solved']} | {summary['moves_total']} | "
            f"{summary['avg_runtime']:.3f} | {summary['max_runtime']:.3f} | "
            f"{summary['qubo_calls']} | {summary['qubo_vars']} | {summary['qubo_time']:.3f} | "
            f"{summary['qubo_on_path']} | {summary['qubo_ancestor']} | "
            f"{summary['group_qubo_calls']} | {summary['group_qubo_time']:.3f} | "
            f"{summary['post_polish_saved']} | {summary['window_polish_saved']} | "
            f"{summary['window_polish_attempts']} | {summary['window_polish_successes']} | "
            f"{summary['peak_candidates_max']} | "
            f"{summary['peak_candidates_avg']:.1f} | {summary['stream_pruned']} | "
            f"{summary['rank_pool_truncated']} | {summary['rescue_layers']} |"
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
    parser.add_argument("--beam-survivor-telemetry-output", type=Path)
    parser.add_argument("--dlts-feature-output", type=Path)
    parser.add_argument("--frontier-trace-output", type=Path)
    parser.add_argument("--frontier-trace-top-k", type=int, default=64)
    parser.add_argument(
        "--solver-profile",
        "--profile",
        dest="solver_profile",
        choices=sorted(_SOLVER_PROFILES.keys()),
        default="speed_v2",
        help=(
            "Reusable solver baseline. speed_v2 is the main 30/30, 509-move QUBO Beam profile; "
            "post_polish_v1, frontier_quality_v1, and rescue_qubo_v1 are experimental QUBO influence probes."
        ),
    )
    parser.add_argument("--priority", choices=["n_then_depth", "astar", "weighted_astar", "target_guided"], default="n_then_depth")
    parser.add_argument("--g-weight", type=float, default=1.0)
    parser.add_argument("--n-weight", type=float, default=1.0)
    parser.add_argument("--target-guided-weight", type=float, default=0.0)
    parser.add_argument("--target-guided-focus-n", type=int, default=8, help="-1 applies target guidance at all N levels.")
    parser.add_argument("--target-guided-supplement-width", type=int, default=0)
    parser.add_argument("--target-guided-n-window", type=int, default=0)
    parser.add_argument("--wang-probe-supplement-width", type=int, default=0)
    parser.add_argument("--wang-probe-focus-n", type=int, default=8, help="-1 applies Wang-style fixed-prefix probing at all N levels.")
    parser.add_argument("--wang-probe-n-window", type=int, default=0)
    parser.add_argument("--wang-probe-fixed-weight", type=float, default=1.0)
    parser.add_argument("--wang-probe-pair-weight", type=float, default=0.0)
    parser.add_argument("--wang-probe-target-weight", type=float, default=0.0)
    parser.add_argument("--wang-probe-plateau-window", type=int, default=0)
    parser.add_argument("--wang-probe-plateau-delta", type=int, default=0)
    parser.add_argument("--wang-incumbent-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wang-incumbent-beam-width", type=int, default=100)
    parser.add_argument("--wang-incumbent-candidate-limit", type=int, default=64)
    parser.add_argument("--wang-incumbent-max-depth", type=int, default=0)
    parser.add_argument("--wang-incumbent-time-limit", type=float, default=0.0)
    parser.add_argument("--wang-cap-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wang-rollout-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wang-rollout-strategy", choices=["local", "baby_lookahead"], default="local")
    parser.add_argument("--wang-rollout-inject-candidates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wang-rollout-width", type=int, default=0)
    parser.add_argument("--wang-rollout-focus-n", type=int, default=8)
    parser.add_argument("--wang-rollout-plateau-window", type=int, default=3)
    parser.add_argument("--wang-rollout-plateau-delta", type=int, default=0)
    parser.add_argument("--wang-rollout-max-steps", type=int, default=1)
    parser.add_argument("--wang-rollout-candidate-limit", type=int, default=64)
    parser.add_argument("--wang-pair-qubo-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wang-pair-qubo-start-layer", type=int, default=0)
    parser.add_argument("--wang-pair-qubo-supplement-width", type=int, default=0)
    parser.add_argument("--wang-pair-qubo-focus-n", type=int, default=8)
    parser.add_argument("--wang-pair-qubo-n-window", type=int, default=1)
    parser.add_argument("--wang-pair-qubo-max-vars", type=int, default=32)
    parser.add_argument("--wang-pair-qubo-select", type=int, default=8)
    parser.add_argument("--wang-pair-qubo-same-parent-penalty", type=float, default=0.25)
    parser.add_argument("--wang-feature-fixed-prefix-weight", type=float, default=0.0)
    parser.add_argument("--wang-feature-fix-pair-weight", type=float, default=0.0)
    parser.add_argument("--wang-feature-risk-weight", type=float, default=0.0)
    parser.add_argument("--wang-feature-same-parent-weight", type=float, default=0.0)
    parser.add_argument("--unified-wang-qubo-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--unified-wang-qubo-supplement-width", type=int, default=None)
    parser.add_argument("--unified-wang-qubo-focus-n", type=int, default=None)
    parser.add_argument("--unified-wang-qubo-n-window", type=int, default=None)
    parser.add_argument("--unified-wang-qubo-max-vars", type=int, default=None)
    parser.add_argument("--unified-wang-qubo-select", type=int, default=None)
    parser.add_argument("--unified-wang-n-weight", type=float, default=None)
    parser.add_argument("--unified-wang-depth-weight", type=float, default=None)
    parser.add_argument("--unified-wang-fixed-prefix-weight", type=float, default=None)
    parser.add_argument("--unified-wang-fix-pair-weight", type=float, default=None)
    parser.add_argument("--unified-wang-target-gap-weight", type=float, default=None)
    parser.add_argument("--unified-wang-same-parent-penalty", type=float, default=None)
    parser.add_argument("--unified-wang-pair-same-parent-penalty", type=float, default=None)
    parser.add_argument("--unified-wang-signature-penalty", type=float, default=None)
    parser.add_argument("--unified-wang-complement-bonus", type=float, default=None)
    parser.add_argument("--unified-wang-rollout-bonus", type=float, default=None)
    parser.add_argument("--feasibility-escape-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--feasibility-escape-supplement-width", type=int, default=None)
    parser.add_argument("--feasibility-escape-focus-n", type=int, default=None)
    parser.add_argument("--feasibility-escape-n-window", type=int, default=None)
    parser.add_argument("--feasibility-escape-dead-end-threshold", type=float, default=None)
    parser.add_argument("--feasibility-escape-min-temporary-slack", type=int, default=None)
    parser.add_argument("--candidate-ledger-score-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--candidate-ledger-wang-fixed-weight", type=float, default=None)
    parser.add_argument("--candidate-ledger-wang-pair-weight", type=float, default=None)
    parser.add_argument("--candidate-ledger-frg-fill-weight", type=float, default=None)
    parser.add_argument("--candidate-ledger-frg-reduce-weight", type=float, default=None)
    parser.add_argument("--candidate-ledger-frg-unblocked-weight", type=float, default=None)
    parser.add_argument("--candidate-ledger-frg-assignment-weight", type=float, default=None)
    parser.add_argument("--candidate-ledger-rollout-bonus", type=float, default=None)
    parser.add_argument("--candidate-ledger-rank-window", type=int, default=None)
    parser.add_argument("--union-diversity-qubo-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--union-diversity-qubo-supplement-width", type=int, default=None)
    parser.add_argument("--union-diversity-qubo-focus-n", type=int, default=None)
    parser.add_argument("--union-diversity-qubo-n-window", type=int, default=None)
    parser.add_argument("--union-diversity-qubo-min-candidates", type=int, default=None)
    parser.add_argument("--union-diversity-qubo-max-vars", type=int, default=None)
    parser.add_argument("--union-diversity-qubo-select", type=int, default=None)
    parser.add_argument("--union-diversity-qubo-rank-weight", type=float, default=None)
    parser.add_argument("--union-diversity-qubo-wang-fixed-weight", type=float, default=None)
    parser.add_argument("--union-diversity-qubo-wang-pair-weight", type=float, default=None)
    parser.add_argument("--union-diversity-qubo-frg-weight", type=float, default=None)
    parser.add_argument("--union-diversity-qubo-rollout-bonus", type=float, default=None)
    parser.add_argument("--union-diversity-qubo-same-parent-penalty", type=float, default=None)
    parser.add_argument("--union-diversity-qubo-same-source-penalty", type=float, default=None)
    parser.add_argument("--union-diversity-qubo-signature-penalty", type=float, default=None)
    parser.add_argument("--union-diversity-qubo-reduce-signature-penalty", type=float, default=None)
    parser.add_argument("--union-diversity-qubo-n-band-penalty", type=float, default=None)
    parser.add_argument("--union-diversity-qubo-n-band-size", type=int, default=None)
    parser.add_argument("--union-diversity-qubo-origin-similarity-penalty", type=float, default=None)
    parser.add_argument("--union-diversity-qubo-same-origin-parent-n-band-penalty", type=float, default=None)
    parser.add_argument("--union-rollout-probe-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--union-rollout-probe-max-candidates", type=int, default=None)
    parser.add_argument("--union-rollout-probe-time-limit", type=float, default=None)
    parser.add_argument("--union-rollout-probe-max-steps", type=int, default=None)
    parser.add_argument("--union-rollout-probe-feasible-bonus", type=float, default=None)
    parser.add_argument("--union-rollout-probe-final-n-weight", type=float, default=None)
    parser.add_argument("--union-rollout-probe-moves-weight", type=float, default=None)
    parser.add_argument("--union-rollout-probe-dead-end-penalty", type=float, default=None)
    parser.add_argument("--union-rollout-terminal-graft", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--union-rollout-terminal-graft-respect-cap",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--micro-rollout-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--micro-rollout-trace-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--receiver-safety-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--micro-rollout-window", type=int, default=None)
    parser.add_argument("--micro-first-child-limit", type=int, default=None)
    parser.add_argument("--micro-second-child-limit", type=int, default=None)
    parser.add_argument("--micro-cache-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--w-micro-delta-n", type=float, default=None)
    parser.add_argument("--w-micro-safe-continuation", type=float, default=None)
    parser.add_argument("--w-receiver-safe-delta", type=float, default=None)
    parser.add_argument("--w-last-buffer-consumed", type=float, default=None)
    parser.add_argument("--w-height-pressure", type=float, default=None)
    parser.add_argument("--frg-frontier-child-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-frontier-child-parent-width", type=int, default=None)
    parser.add_argument("--frg-frontier-child-limit", type=int, default=None)
    parser.add_argument("--frg-frontier-child-max-steps", type=int, default=None)
    parser.add_argument("--frg-frontier-child-append-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-frontier-child-adaptive-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-frontier-child-adaptive-parent-width", type=int, default=None)
    parser.add_argument("--frg-frontier-child-adaptive-limit", type=int, default=None)
    parser.add_argument("--frg-frontier-child-adaptive-max-steps", type=int, default=None)
    parser.add_argument("--frg-frontier-child-adaptive-plateau-window", type=int, default=None)
    parser.add_argument("--frg-frontier-child-adaptive-plateau-delta", type=int, default=None)
    parser.add_argument("--frg-frontier-child-adaptive-min-n", type=int, default=None)
    parser.add_argument("--frg-frontier-child-adaptive-small-stack-max", type=int, default=None)
    parser.add_argument("--frg-frontier-child-adaptive-min-container-count", type=int, default=None)
    parser.add_argument("--frg-seed-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-seed-time-limit", type=float, default=None)
    parser.add_argument("--frg-seed-min-max-steps", type=int, default=None)
    parser.add_argument("--frg-seed-max-steps-multiplier", type=int, default=None)
    parser.add_argument("--frg-seed-max-steps-cap", type=int, default=None)
    parser.add_argument("--frg-seed-respect-cap", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-seed-prune-to-incumbent", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-seed-allow-large-over-cap", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-seed-paper-pure", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-candidate-supplement-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-candidate-supplement-width", type=int, default=None)
    parser.add_argument("--frg-candidate-focus-n", type=int, default=None)
    parser.add_argument("--frg-candidate-n-window", type=int, default=None)
    parser.add_argument("--frg-candidate-max-ranked", type=int, default=None)
    parser.add_argument("--frg-candidate-scarcity-gate-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-linear-score-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-linear-fill-weight", type=float, default=None)
    parser.add_argument("--frg-linear-reduce-weight", type=float, default=None)
    parser.add_argument("--frg-linear-unblocked-weight", type=float, default=None)
    parser.add_argument("--frg-linear-assignment-weight", type=float, default=None)
    parser.add_argument("--frg-linear-max-height-ordered-weight", type=float, default=None)
    parser.add_argument("--frg-linear-height-pressure-weight", type=float, default=None)
    parser.add_argument("--frg-rank-tiebreak-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-rank-tiebreak-weight", type=float, default=None)
    parser.add_argument("--frg-rank-tiebreak-focus-n", type=int, default=None)
    parser.add_argument("--frg-scarcity-gate-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-scarcity-gate-max-available-stacks", type=int, default=None)
    parser.add_argument("--frg-scarcity-gate-min-avg-height-pressure", type=float, default=None)
    parser.add_argument("--frg-scarcity-gate-max-reduction-receivers", type=int, default=None)
    parser.add_argument("--frg-scarcity-gate-min-unblocked-bad", type=int, default=None)
    parser.add_argument("--frg-scarcity-gate-min-candidates", type=int, default=None)
    parser.add_argument("--frg-scarcity-gate-cap-slack", type=int, default=None)
    parser.add_argument("--frg-compound-escape-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frg-compound-escape-width", type=int, default=None)
    parser.add_argument("--frg-compound-escape-focus-n", type=int, default=None)
    parser.add_argument("--frg-compound-escape-plateau-window", type=int, default=None)
    parser.add_argument("--frg-compound-escape-plateau-delta", type=int, default=None)
    parser.add_argument("--frg-compound-escape-cap-slack", type=int, default=None)
    parser.add_argument("--frg-compound-escape-min-candidates", type=int, default=None)
    parser.add_argument("--dlts-value-supplement-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dlts-value-supplement-width", type=int, default=None)
    parser.add_argument("--bs-frg-qubo-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bs-frg-qubo-supplement-width", type=int, default=None)
    parser.add_argument("--bs-frg-qubo-focus-n", type=int, default=None)
    parser.add_argument("--bs-frg-qubo-n-window", type=int, default=None)
    parser.add_argument("--bs-frg-qubo-min-candidates", type=int, default=None)
    parser.add_argument("--bs-frg-qubo-max-vars", type=int, default=None)
    parser.add_argument("--bs-frg-qubo-rank-weight", type=float, default=None)
    parser.add_argument("--bs-frg-qubo-same-parent-penalty", type=float, default=None)
    parser.add_argument("--bs-frg-qubo-same-source-penalty", type=float, default=None)
    parser.add_argument("--bs-frg-qubo-signature-penalty", type=float, default=None)
    parser.add_argument("--bs-frg-qubo-reduce-signature-penalty", type=float, default=None)
    parser.add_argument("--bs-frg-qubo-n-band-penalty", type=float, default=None)
    parser.add_argument("--bs-frg-qubo-n-band-size", type=int, default=None)
    parser.add_argument("--wang2017-task-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wang2017-task-supplement-width", type=int, default=None)
    parser.add_argument("--wang2017-task-focus-n", type=int, default=None)
    parser.add_argument("--wang2017-task-n-window", type=int, default=None)
    parser.add_argument("--wang2017-task-plateau-window", type=int, default=None)
    parser.add_argument("--wang2017-task-plateau-delta", type=int, default=None)
    parser.add_argument("--tail-lookahead-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tail-lookahead-core-path-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tail-lookahead-max-stack-count", type=int, default=None)
    parser.add_argument("--tail-lookahead-focus-n", type=int, default=None)
    parser.add_argument("--tail-lookahead-depth", type=int, default=None)
    parser.add_argument("--tail-lookahead-max-seed-candidates", type=int, default=None)
    parser.add_argument("--tail-lookahead-supplement-width", type=int, default=None)
    parser.add_argument("--tail-lookahead-max-expanded", type=int, default=None)
    parser.add_argument("--tail-risk-reserve-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tail-risk-focus-n", type=int, default=None)
    parser.add_argument("--tail-risk-n-window", type=int, default=None)
    parser.add_argument("--tail-risk-reserve-width", type=int, default=None)
    parser.add_argument("--tail-risk-max-scan", type=int, default=None)
    parser.add_argument("--tail-risk-min-candidates", type=int, default=None)
    parser.add_argument("--tail-risk-min-depth-slack", type=int, default=None)
    parser.add_argument("--tail-risk-max-depth-slack", type=int, default=None)
    parser.add_argument("--tail-risk-max-stack-count", type=int, default=None)
    parser.add_argument("--target-final-n", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--time-limit", type=float, default=0.0, help="Per-case limit in seconds; 0 means no limit.")
    parser.add_argument("--stream-candidate-limit", type=int, default=0, help="0 keeps the full per-layer candidate set.")
    parser.add_argument("--stream-candidate-trim-factor", type=float, default=2.0)
    parser.add_argument("--stream-bucket-topk", type=int, default=0, help="0 disables generation-time grouped top-k pruning.")
    parser.add_argument("--rank-pool-size", type=int, default=0, help="0 uses the old full candidate sort.")
    parser.add_argument("--rank-pool-min-candidates", type=int, default=0)
    parser.add_argument("--rank-bucket-topk", type=int, default=0, help="0 disables grouped bounded top-k candidate ranking.")
    parser.add_argument(
        "--rank-bucket-key",
        choices=["n", "n_depth_parent", "n_depth", "n_band_parent", "n_band", "parent"],
        default="n_depth_parent",
    )
    parser.add_argument("--rank-bucket-n-band-size", type=int, default=2)
    parser.add_argument("--rank-bucket-topk-high-n", type=int, default=0, help="0 applies grouped top-k at any best-N level.")
    parser.add_argument("--rank-bucket-topk-min-candidates", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=1200)
    parser.add_argument("--core-width", type=int, default=1080)
    parser.add_argument("--parent-core-cap", type=int, default=4)
    parser.add_argument("--survivor-parent-cap-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--survivor-parent-cap-trigger-ratio", type=float, default=0.70)
    parser.add_argument("--survivor-parent-cap-release-ratio", type=float, default=0.55)
    parser.add_argument("--survivor-parent-cap-value", type=int, default=3)
    parser.add_argument("--survivor-parent-cap-window", type=int, default=2)
    parser.add_argument("--survivor-parent-cap-min-n", type=int, default=8)
    parser.add_argument(
        "--selection-policy",
        choices=[
            "enhanced",
            "classic",
            "dynamic_qubo",
            "adaptive_qubo",
            "rescue_qubo",
            "scheduler_qubo",
            "scheduler_state_qubo",
            "qubo_bucket",
        ],
        default="enhanced",
    )
    parser.add_argument("--use-qubo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qubo-influence-profile", choices=sorted(_QUBO_INFLUENCE_PRESETS.keys()), default="stable")
    parser.add_argument("--qubo-mode", choices=["supplement", "focused_multi", "majority_multi"], default="focused_multi")
    parser.add_argument("--qubo-profile", default="frontier_cohort")
    parser.add_argument("--max-qubo-calls", type=int, default=32)
    parser.add_argument("--max-qubo-vars", type=int, default=64)
    parser.add_argument("--qubo-core-width", type=int, default=224)
    parser.add_argument("--qubo-supplement-width", type=int, default=24)
    parser.add_argument("--qubo-min-candidates", type=int, default=64)
    parser.add_argument("--qubo-start-layer", type=int, default=0)
    parser.add_argument("--qubo-priority-bonus", type=float, default=0.0001)
    parser.add_argument("--qubo-diversity-weight", type=float, default=0.10)
    parser.add_argument("--qubo-frontier-n-window", type=int, default=0)
    parser.add_argument("--qubo-frontier-depth-window", type=int, default=1)
    parser.add_argument("--qubo-focus-n", type=int, default=8, help="Only call QUBO when the layer best N is <= this value; -1 disables the gate.")
    parser.add_argument("--qubo-first", action=argparse.BooleanOptionalAction, default=False, help="Let QUBO fill its share before deterministic core fill.")
    parser.add_argument("--qubo-target-share", type=float, default=0.50, help="Target frontier share selected by QUBO in majority_multi mode.")
    parser.add_argument("--qubo-cohorts-per-layer", type=int, default=2)
    parser.add_argument("--qubo-cohort-stride", type=int, default=64, help="Candidate stride between QUBO cohorts; 0 uses max_qubo_vars.")
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
    parser.add_argument("--adaptive-high-n", type=int, default=16)
    parser.add_argument("--adaptive-low-n", type=int, default=8)
    parser.add_argument("--adaptive-high-frontier", type=int, default=800)
    parser.add_argument("--adaptive-mid-frontier", type=int, default=1000)
    parser.add_argument("--adaptive-low-frontier", type=int, default=1200)
    parser.add_argument("--adaptive-core-ratio", type=float, default=0.90)
    parser.add_argument("--adaptive-candidate-pool", type=int, default=0)
    parser.add_argument("--adaptive-controller", choices=["tiered", "signal_v1"], default="tiered")
    parser.add_argument("--adaptive-expand-frontier", type=int, default=1200)
    parser.add_argument("--adaptive-contract-pressure", type=float, default=0.0)
    parser.add_argument("--adaptive-mid-pressure", type=float, default=0.0)
    parser.add_argument("--adaptive-min-best-n-ratio", type=float, default=0.0)
    parser.add_argument("--adaptive-max-duplication-ratio", type=float, default=1.0)
    parser.add_argument("--adaptive-parent-top-share", type=float, default=0.0)
    parser.add_argument("--adaptive-plateau-window", type=int, default=0)
    parser.add_argument("--adaptive-plateau-delta", type=int, default=0)
    parser.add_argument("--adaptive-cap-slack", type=int, default=0)
    parser.add_argument("--adaptive-large-stack-min-stacks", type=int, default=0)
    parser.add_argument("--adaptive-large-stack-expand-frontier", type=int, default=0)
    parser.add_argument("--adaptive-large-stack-plateau-window", type=int, default=0)
    parser.add_argument("--adaptive-large-stack-cap-slack", type=int, default=0)
    parser.add_argument("--scheduler-adaptive-min-stacks", type=int, default=7)
    parser.add_argument("--scheduler-adaptive-high-n", type=int, default=16)
    parser.add_argument("--scheduler-adaptive-cv1-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scheduler-adaptive-max-duplication-ratio", type=float, default=1.0)
    parser.add_argument("--scheduler-adaptive-min-best-n-ratio", type=float, default=0.0)
    parser.add_argument("--scheduler-adaptive-max-candidate-pressure", type=float, default=0.0)
    parser.add_argument("--scheduler-adaptive-max-parent-top-share", type=float, default=0.0)
    parser.add_argument("--scheduler-adaptive-plateau-window", type=int, default=0)
    parser.add_argument("--scheduler-adaptive-cap-slack", type=int, default=0)
    parser.add_argument("--scheduler-state-controller", choices=["off", "state_v2"], default="off")
    parser.add_argument("--scheduler-state-contract-pressure", type=float, default=8.0)
    parser.add_argument("--scheduler-state-max-contract-pressure", type=float, default=40.0)
    parser.add_argument("--scheduler-state-min-best-n-ratio", type=float, default=0.0)
    parser.add_argument("--scheduler-state-progress-window", type=int, default=3)
    parser.add_argument("--scheduler-state-progress-delta", type=int, default=2)
    parser.add_argument("--scheduler-state-failed-contract-cooldown", type=int, default=0)
    parser.add_argument("--scheduler-state-post-rescue-plateau-window", type=int, default=0)
    parser.add_argument("--scheduler-state-post-rescue-plateau-delta", type=int, default=0)
    parser.add_argument("--scheduler-state-post-rescue-min-ratio", type=float, default=0.90)
    parser.add_argument("--scheduler-state-post-rescue-action", choices=["protect", "blend", "rescue"], default="protect")
    parser.add_argument("--scheduler-state-initial-plateau-window", type=int, default=0)
    parser.add_argument("--scheduler-state-initial-plateau-min-ratio", type=float, default=0.98)
    parser.add_argument("--scheduler-state-initial-plateau-action", choices=["protect", "blend", "rescue"], default="protect")
    parser.add_argument("--scheduler-state-initial-plateau-lock", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scheduler-state-initial-plateau-min-duplication", type=float, default=0.0)
    parser.add_argument("--scheduler-state-initial-plateau-unique-min-n", type=int, default=0)
    parser.add_argument("--scheduler-state-protect-parent-top-share", type=float, default=0.65)
    parser.add_argument("--scheduler-state-protect-n-band", type=int, default=2)
    parser.add_argument("--scheduler-state-top-window", type=int, default=64)
    parser.add_argument("--scheduler-state-rescue-plateau-window", type=int, default=3)
    parser.add_argument("--scheduler-state-rescue-plateau-delta", type=int, default=0)
    parser.add_argument("--scheduler-state-rescue-cap-slack", type=int, default=8)
    parser.add_argument("--scarcity-inner-rescue-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scarcity-inner-rescue-risk-threshold", type=float, default=0.95)
    parser.add_argument("--scarcity-inner-rescue-max-l0", type=int, default=5)
    parser.add_argument("--scarcity-inner-rescue-min-one-safe-ratio", type=float, default=0.90)
    parser.add_argument("--scarcity-inner-rescue-min-n", type=int, default=12)
    parser.add_argument("--scarcity-inner-rescue-min-candidates", type=int, default=64)
    parser.add_argument("--scarcity-group-qubo-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scarcity-group-qubo-threshold", type=float, default=0.95)
    parser.add_argument("--scarcity-group-qubo-plateau-window", type=int, default=2)
    parser.add_argument("--scarcity-group-qubo-plateau-delta", type=int, default=0)
    parser.add_argument("--scarcity-group-qubo-min-best-n-ratio", type=float, default=0.70)
    parser.add_argument("--scarcity-group-qubo-min-candidates", type=int, default=1000)
    parser.add_argument("--scarcity-group-qubo-min-n", type=int, default=12)
    parser.add_argument("--bucket-key", choices=["n_depth", "f_n", "n_band"], default="n_depth")
    parser.add_argument("--bucket-n-band-size", type=int, default=2)
    parser.add_argument("--bucket-core-ratio", type=float, default=0.50)
    parser.add_argument("--bucket-qubo-share", type=float, default=0.35)
    parser.add_argument("--bucket-max-qubo-vars", type=int, default=96)
    parser.add_argument("--bucket-qubo-select", type=int, default=24)
    parser.add_argument("--bucket-min-candidates", type=int, default=64)
    parser.add_argument("--rescue-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rescue-plateau-window", type=int, default=2)
    parser.add_argument("--rescue-plateau-delta", type=int, default=0)
    parser.add_argument("--rescue-cap-slack", type=int, default=26)
    parser.add_argument("--rescue-min-n", type=int, default=12)
    parser.add_argument("--rescue-min-candidates", type=int, default=5000)
    parser.add_argument("--rescue-max-layers", type=int, default=16)
    parser.add_argument("--rescue-beam-width", type=int, default=1350)
    parser.add_argument("--rescue-core-width", type=int, default=1215)
    parser.add_argument("--rescue-escalate-after", type=int, default=2)
    parser.add_argument("--rescue-escalate-beam-width", type=int, default=1600)
    parser.add_argument("--rescue-escalate-core-width", type=int, default=1440)
    parser.add_argument("--rescue-max-qubo-vars", type=int, default=96)
    parser.add_argument("--rescue-qubo-supplement-width", type=int, default=48)
    parser.add_argument("--rescue-qubo-cohorts-per-layer", type=int, default=4)
    parser.add_argument("--rescue-qubo-time-limit", type=float, default=0.002)
    parser.add_argument("--rescue-qubo-iterations", type=int, default=30)
    parser.add_argument("--rescue-next-n-weight", type=float, default=0.75)
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
    parser.add_argument("--qubo-next-n-weight", type=float, default=0.0)
    parser.add_argument("--qubo-penalty-a", type=float, default=30.0)
    parser.add_argument("--qubo-time-limit", type=float, default=0.001)
    parser.add_argument("--qubo-reads", type=int, default=2)
    parser.add_argument("--qubo-iterations", type=int, default=20)
    parser.add_argument("--deterministic-qubo-sampler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--post-polish-qubo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-polish-mode", choices=["legacy", "qubo_windows_v2"], default="legacy")
    parser.add_argument("--post-polish-accept", choices=["better", "equal_or_better"], default="better")
    parser.add_argument("--post-polish-window-min-stacks", type=int, default=0)
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
    parser.add_argument("--post-polish-window-min-size", type=int, default=3)
    parser.add_argument("--post-polish-window-max-size", type=int, default=10)
    parser.add_argument("--post-polish-window-stride", type=int, default=1)
    parser.add_argument("--post-polish-window-budget", type=int, default=96)
    parser.add_argument("--post-polish-window-qubo-vars", type=int, default=72)
    parser.add_argument("--post-polish-window-select", type=int, default=8)
    parser.add_argument("--post-polish-window-time-limit", type=float, default=0.001)
    parser.add_argument("--post-polish-window-repair-expanded", type=int, default=20000)
    parser.add_argument("--post-polish-window-repair-depth-extra", type=int, default=0)
    parser.add_argument("--post-polish-successor-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tail-order-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.beam_survivor_telemetry_output is not None and not args.beam_survivor_telemetry_output.is_absolute():
        args.beam_survivor_telemetry_output = ROOT / args.beam_survivor_telemetry_output
    if args.dlts_feature_output is not None and not args.dlts_feature_output.is_absolute():
        args.dlts_feature_output = ROOT / args.dlts_feature_output
    if args.frontier_trace_output is not None and not args.frontier_trace_output.is_absolute():
        args.frontier_trace_output = ROOT / args.frontier_trace_output
    _reset_beam_survivor_output(args.beam_survivor_telemetry_output)
    _reset_dlts_feature_output(args.dlts_feature_output)
    _reset_frontier_trace_output(args.frontier_trace_output)
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
