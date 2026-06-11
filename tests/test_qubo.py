from __future__ import annotations

import argparse
import sys
from pathlib import Path

from container_reshuffle.core import Instance, Move, ProblemVariant, State
from container_reshuffle.qubo.beam_selection import QuboBeamSelectionBuilder
from container_reshuffle.qubo.frontier_selection import QuboFrontierConfig, QuboFrontierSelector
from container_reshuffle.qubo.model import QuboModel
from container_reshuffle.qubo.path_selection import PathEdge, PathNode, QuboPathSelectionBuilder
from container_reshuffle.qubo.sampler import QuboSample, QuboSamplerConfig, SimulatedAnnealingQuboSampler
from container_reshuffle.qubo.qubo_solvers import QuboCoreBeamSearchSolver
from container_reshuffle.solvers.base import SolverConfig
from container_reshuffle.solvers.beam import Candidate

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_exactly_k_penalty_prefers_k_selected():
    model = QuboModel()
    vars_ = [model.add_binary_var(f"x{i}") for i in range(5)]
    model.add_exactly_k_penalty(vars_, 2, 5.0)
    assert model.evaluate({0: 1, 1: 1}) < model.evaluate({0: 1})


def test_qubo_evaluate():
    model = QuboModel()
    x = model.add_binary_var("x")
    y = model.add_binary_var("y")
    model.add_linear(x, 1.0)
    model.add_linear(y, 2.0)
    model.add_quadratic(x, y, 3.0)
    assert model.evaluate({x: 1, y: 1}) == 6.0


def test_sa_sampler_returns_samples():
    model = QuboModel()
    x = model.add_binary_var("x")
    model.add_linear(x, -1.0)
    ss = SimulatedAnnealingQuboSampler().sample(model, QuboSamplerConfig(num_reads=3, sa_iterations=50))
    assert ss.samples


def test_qubo_beam_selection_selects_b():
    inst = Instance("x", ProblemVariant.PREMARSHALLING_CPMP, 2, 2, 2, [[1, 2], []])
    candidates = [
        Candidate(State([[1], [2]], move_count=1), 1.0, None, (i,))
        for i in range(4)
    ]
    builder = QuboBeamSelectionBuilder()
    model, usable = builder.build(candidates, 2, 0.1, 10.0, 10)
    sample = QuboSample({0: 1, 1: 1}, model.evaluate({0: 1, 1: 1}))
    selected, repair = builder.decode_and_repair(sample, usable, 2)
    assert len(selected) == 2


def test_qubo_repair_too_many_too_few():
    candidates = [Candidate(State([[]]), float(i), None, (i,)) for i in range(5)]
    builder = QuboBeamSelectionBuilder()
    selected, repair = builder.decode_and_repair(QuboSample({0: 1, 1: 1, 2: 1}, 0), candidates, 2)
    assert len(selected) == 2 and repair
    selected, repair = builder.decode_and_repair(QuboSample({}, 0), candidates, 2)
    assert len(selected) == 2 and repair


def test_qubo_objective_aware_repair_improves_feasible_sample():
    candidates = [Candidate(State([[]]), float(i), None, (i,)) for i in [0, 1, 10, 11]]
    builder = QuboBeamSelectionBuilder()
    selected, repair = builder.decode_and_repair(
        QuboSample({2: 1, 3: 1}, 0),
        candidates,
        2,
        repair_hamming_slack=3,
    )
    assert [cand.score for cand in selected] == [0.0, 1.0]
    assert repair


def test_path_selection_repair_does_not_choose_empty_root_path():
    move = Move(step=0, box=1, from_stack=0, to_stack=1)
    nodes = [
        PathNode(0, State([[1], []]), 0, outgoing_edges=[0]),
        PathNode(1, State([[], [1]], move_count=1), 1, incoming_edges=[0]),
    ]
    edges = [PathEdge(0, 0, 1, move, 1.0)]
    path = QuboPathSelectionBuilder._best_classical_path(nodes, edges)
    assert path == [move]


def test_qubo_frontier_selector_keeps_core_and_qubo_supplement():
    candidates = [
        Candidate(State([[]], move_count=i), float(i), None, (i,), ("p", i // 2))
        for i in range(8)
    ]
    selector = QuboFrontierSelector()
    result = selector.select(
        candidates,
        QuboFrontierConfig(
            policy="enhanced",
            beam_width=4,
            core_width=2,
            qubo_min_candidates=1,
            qubo_max_vars=6,
            qubo_supplement_width=2,
            qubo_num_reads=2,
            qubo_sa_iterations=30,
        ),
    )
    assert len(result.selected) == 4
    assert result.selected[0].hash_value == (0,)
    assert result.qubo_calls >= 1


def test_frontier_config_mapping_includes_dynamic_parameters():
    config = SolverConfig(
        qubo_dynamic_adaptive_frontier=False,
        qubo_dynamic_stagnation_window=4,
        qubo_dynamic_stagnation_epsilon=1e-5,
        qubo_dynamic_similarity_window=32,
        qubo_dynamic_similarity_expand_threshold=0.6,
        qubo_dynamic_similarity_contract_threshold=0.4,
        qubo_dynamic_parent_share_expand_threshold=0.9,
        qubo_dynamic_score_gap_expand_ratio=0.05,
        qubo_dynamic_frontier_pressure_step=0.2,
        qubo_dynamic_frontier_pressure_decay=0.1,
    )
    from container_reshuffle.qubo.qubo_solvers import _frontier_config_from_solver_config

    frontier_cfg = _frontier_config_from_solver_config(config)
    assert frontier_cfg.dynamic_adaptive_frontier is False
    assert frontier_cfg.dynamic_stagnation_window == 4
    assert frontier_cfg.dynamic_similarity_window == 32
    assert frontier_cfg.dynamic_similarity_expand_threshold == 0.6
    assert frontier_cfg.dynamic_similarity_contract_threshold == 0.4
    assert frontier_cfg.dynamic_parent_share_expand_threshold == 0.9
    assert frontier_cfg.dynamic_score_gap_expand_ratio == 0.05
    assert frontier_cfg.dynamic_frontier_pressure_step == 0.2
    assert frontier_cfg.dynamic_frontier_pressure_decay == 0.1


def test_dynamic_frontier_selects_with_adaptive_off():
    candidates = [
        Candidate(State([[]], move_count=i), float(i), None, (i,), ("p", i // 2))
        for i in range(20)
    ]
    selector = QuboFrontierSelector()
    result = selector.select(
        candidates,
        QuboFrontierConfig(
            policy="dynamic_qubo",
            beam_width=10,
            dynamic_adaptive_frontier=False,
            dynamic_min_frontier=4,
            dynamic_max_frontier=8,
            dynamic_core_frontier=1,
            dynamic_candidate_pool=32,
            qubo_min_candidates=2,
            group_qubo_enabled=True,
            group_qubo_max_vars=16,
            group_qubo_select=4,
            group_qubo_take=2,
            qubo_num_reads=2,
            qubo_sa_iterations=20,
        ),
        call_index=0,
    )
    assert result.policy == "dynamic_qubo"
    assert result.frontier_width in (None, 8)
    assert len(result.selected) <= 8


def test_qubo_core_beam_solver_runs_tiny_cpmp():
    inst = Instance("cpmp", ProblemVariant.PREMARSHALLING_CPMP, 3, 2, 3, [[1, 3], [2], []])
    config = SolverConfig(
        max_depth=4,
        beam_width=4,
        max_candidates_per_layer=20,
        max_qubo_vars=8,
        qubo_min_candidates_for_call=1,
        qubo_num_reads=2,
        qubo_sa_iterations=30,
        qubo_time_limit_sec=0.001,
    )
    sol = QuboCoreBeamSearchSolver().solve(inst, config)
    assert sol.status.startswith(("FEASIBLE", "NO_FEASIBLE_FOUND"))


def test_qubo_repair_pair_matrix_matches_selection_value():
    from benchmark_target_astar_memory_n0 import (
        _pair_penalty_matrix,
        _qubo_selection_value,
        _qubo_selection_value_with_pairs,
        _repair_qubo_selection,
        _repair_qubo_selection_with_pairs,
    )

    payload = [
        (2, 5, b"p0", b"abcde12345"),
        (2, 5, b"p0", b"abcdf12345"),
        (3, 5, b"p1", b"uvwxy12345"),
        (2, 6, b"p2", b"abcde12399"),
        (4, 6, b"p3", b"zzzzz00000"),
    ]
    score_norm = [0.1, 0.2, 0.4, 0.25, 0.6]
    diversity_weight = 0.3
    pairs = _pair_penalty_matrix(payload, diversity_weight)

    selected = {0, 1, 4}
    assert _qubo_selection_value_with_pairs(selected, score_norm, pairs) == _qubo_selection_value(
        selected,
        score_norm,
        payload,
        diversity_weight,
    )

    repaired_old = _repair_qubo_selection({4}, 3, score_norm, payload, diversity_weight)
    repaired_new = _repair_qubo_selection_with_pairs({4}, 3, score_norm, pairs)
    assert repaired_new == repaired_old


def test_compact_cached_successors_match_reference_successors():
    from benchmark_compact_beam_qubo_n0 import _successors_with_n_cached
    from run_business_exact_target_search_sqlite import _encode, _successors_with_n
    from run_exact_astar_real_until_done import _instance_from_single_bay

    instance = _instance_from_single_bay("A1", "26")
    code, n_value = _encode(instance.stacks, instance.stack_count, instance.max_height)
    reference = sorted(_successors_with_n(code, n_value, instance.stack_count, instance.max_height))
    cached = sorted(_successors_with_n_cached(code, n_value, instance.stack_count, instance.max_height))
    assert cached == reference

    next_code, next_n = reference[0]
    reference_next = sorted(_successors_with_n(next_code, next_n, instance.stack_count, instance.max_height))
    cached_next = sorted(_successors_with_n_cached(next_code, next_n, instance.stack_count, instance.max_height))
    assert cached_next == reference_next


def test_compact_cached_successors_deduplicate_identical_stack_moves():
    from benchmark_compact_beam_qubo_n0 import _successors_with_n_cached
    from run_business_exact_target_search_sqlite import _encode, _successors_with_n

    stacks = [[3, 1], [3, 1], [], []]
    code, n_value = _encode(stacks, 4, 3)

    reference = sorted(_successors_with_n(code, n_value, 4, 3))
    cached = sorted(_successors_with_n_cached(code, n_value, 4, 3))

    assert cached == reference
    assert len(cached) < 4 * 3


def test_compact_cached_chunk_pop_push_match_manual_bytes():
    from benchmark_compact_beam_qubo_n0 import _cached_chunk_meta, _cached_pop_chunk, _cached_push_chunk

    chunk = bytes([9, 4, 7, 0, 0])

    assert _cached_chunk_meta(chunk, 5) == (3, 1)
    assert _cached_pop_chunk(chunk, 3, 5) == bytes([9, 4, 0, 0, 0])
    assert _cached_push_chunk(chunk, 3, 6, 5) == bytes([9, 4, 7, 6, 0])


def test_dlts_dat_parser_infers_open_cpmp_capacity(tmp_path):
    from open_cpmp_benchmarks import load_dlts_dat

    case_dir = tmp_path / "3x2_cv1_test"
    case_dir.mkdir()
    path = case_dir / "0.dat"
    path.write_text(
        "\n".join(
            [
                "3 6",
                "2 6 1",
                "2 5 2",
                "2 4 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    instance = load_dlts_dat(path)

    assert instance.instance_id == "3x2_cv1_test_0"
    assert instance.stack_count == 3
    assert instance.container_count == 6
    assert instance.max_height == 4
    assert instance.stacks == [[6, 1], [5, 2], [4, 3]]


def test_compact_beam_instance_entry_solves_tiny_open_cpmp():
    from benchmark_compact_beam_qubo_n0 import (
        _row_dict_from_instance_result,
        _validate_instance_result,
        make_compact_args,
        run_compact_beam_instance,
    )

    instance = Instance(
        "tiny_open_cpmp",
        ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=3,
        max_height=3,
        container_count=3,
        stacks=[[1, 3], [], [2]],
    )
    args = make_compact_args(
        alpha=10.0,
        beam_width=8,
        core_width=8,
        max_moves=4,
        max_qubo_calls=0,
        post_polish_qubo=False,
        solver_profile="speed_v2",
        time_limit=1.0,
        use_qubo=False,
    )

    result = run_compact_beam_instance(instance.instance_id, instance, args)
    row = _row_dict_from_instance_result(result, instance, args)
    business_ok, goal_ok, msg = _validate_instance_result(result, instance, business_alpha=10.0)

    assert row["final_n"] == 0
    assert row["moves"] == 1
    assert business_ok
    assert goal_ok, msg


def _compact_args(**overrides):
    values = {
        "priority": "n_then_depth",
        "solver_profile": "speed_v2",
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
        "qubo_target_share": 0.5,
        "qubo_cohorts_per_layer": 2,
        "qubo_cohort_stride": 64,
        "qubo_time_limit": 0.001,
        "qubo_iterations": 20,
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
        "reservoir_size": 0,
        "peak_reservoir": 0,
        "reservoir_added_total": 0,
        "reservoir_pulled_total": 0,
        "stream_candidate_limit": 0,
        "stream_candidate_trim_factor": 2.0,
        "rank_pool_size": 0,
        "rank_pool_min_candidates": 0,
        "_rank_pool_truncated_candidates": 0,
        "post_polish_qubo": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_solver_profile_speed_v2_restores_historical_baseline_without_overriding_explicit_flags(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(
        beam_width=7,
        core_width=3,
        rank_pool_size=6500,
        qubo_influence_profile="light",
        stream_candidate_limit=111,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            "--rank-pool-size",
            "6500",
            "--qubo-influence-profile",
            "light",
        ],
    )

    _apply_solver_profile(args)

    assert args.beam_width == 1200
    assert args.core_width == 1080
    assert args.selection_policy == "enhanced"
    assert args.qubo_mode == "focused_multi"
    assert args.qubo_focus_n == 8
    assert args.qubo_cohorts_per_layer == 2
    assert args.stream_candidate_limit == 0
    assert args.rank_pool_size == 6500
    assert args.qubo_influence_profile == "light"


def test_solver_profile_respects_programmatic_explicit_keys(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(
        beam_width=1600,
        core_width=1600,
        selection_policy="classic",
        use_qubo=False,
        max_qubo_calls=0,
        post_polish_qubo=False,
        _profile_explicit_keys={
            "beam_width",
            "core_width",
            "selection_policy",
            "use_qubo",
            "max_qubo_calls",
            "post_polish_qubo",
        },
    )
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.beam_width == 1600
    assert args.core_width == 1600
    assert args.selection_policy == "classic"
    assert args.use_qubo is False
    assert args.max_qubo_calls == 0
    assert args.post_polish_qubo is False
    assert args.qubo_mode == "focused_multi"


def test_ablation_no_qubo_args_are_marked_explicit():
    from benchmark_compact_qubo_ablation_time_aligned import _compact_args as _ablation_compact_args

    outer = argparse.Namespace(
        bays=["01", "26"],
        output=Path("dummy.csv"),
        priority="n_then_depth",
        g_weight=1.0,
        n_weight=1.0,
        stream_candidate_limit=0,
        stream_candidate_trim_factor=2.0,
        core_ratio=0.9,
        parent_core_cap=4,
        qubo_policy="enhanced",
        no_qubo_policy="classic",
        qubo_mode="focused_multi",
        qubo_selector="state",
        max_qubo_calls=32,
        max_qubo_vars=64,
        qubo_core_width=224,
        qubo_supplement_width=24,
        qubo_diversity_weight=0.10,
        qubo_focus_n=8,
        qubo_first=False,
        qubo_target_share=0.50,
        qubo_cohorts_per_layer=2,
        qubo_cohort_stride=64,
        max_group_qubo_calls=32,
        dynamic_profile="legacy",
        qubo_time_limit=0.001,
        qubo_reads=2,
        qubo_iterations=20,
        qubo_group_max_vars=96,
        qubo_group_select=24,
        qubo_group_take=4,
        qubo_group_same_n_penalty=0.04,
        qubo_group_lookahead_weight=0.0,
        qubo_group_branch_weight=0.0,
        group_qubo_risk_gate=False,
        group_qubo_risk_min_candidates=50000,
        group_qubo_risk_parent_share=0.12,
        group_qubo_risk_plateau_window=3,
        group_qubo_risk_plateau_delta=0,
        dynamic_min_frontier=128,
        dynamic_core_frontier=0,
        dynamic_max_frontier=0,
        dynamic_candidate_pool=4096,
        dynamic_qubo_keep_threshold=0.55,
        global_reservoir=False,
        reservoir_max_states=50000,
        reservoir_pull=800,
        reservoir_add_per_layer=4000,
        post_polish_qubo=True,
        post_polish_passes=2,
        post_polish_focus_n=8,
        post_polish_tail_span=14,
        post_polish_max_window=8,
        post_polish_max_windows=96,
        post_polish_select_windows=0,
        post_polish_max_qubo_vars=72,
        post_polish_max_bridge_depth=7,
        post_polish_bridge_expanded=20000,
        post_polish_suffix_focus_n=9,
        post_polish_suffix_tail_span=18,
        post_polish_suffix_max_candidates=48,
        post_polish_suffix_select=5,
        post_polish_suffix_max_qubo_vars=48,
        post_polish_suffix_max_depth=18,
        post_polish_suffix_expanded=60000,
        tail_order_steps=6,
        seed=37,
    )

    args = _ablation_compact_args(outer, beam_width=1600, time_limit=1.2, use_qubo=False)

    assert args.solver_profile == "speed_v2"
    assert args.use_qubo is False
    assert args.post_polish_qubo is False
    assert "use_qubo" in args._profile_explicit_keys
    assert "post_polish_qubo" in args._profile_explicit_keys
    assert "selection_policy" in args._profile_explicit_keys


def test_group_majority_args_are_marked_explicit():
    from benchmark_compact_group_qubo_majority import _build_args

    outer = argparse.Namespace(
        cases=["A2_26"],
        bays=["26"],
        output=Path("dummy.csv"),
        beam_width=1200,
        core_width=600,
        parent_core_cap=4,
        time_limit=20.0,
        max_qubo_calls=64,
        qubo_focus_n=8,
        qubo_group_max_vars=96,
        qubo_group_select=72,
        qubo_group_take=8,
        qubo_group_same_n_penalty=0.04,
        qubo_time_limit=0.001,
        qubo_reads=2,
        qubo_iterations=20,
        post_polish_qubo=True,
        seed=37,
    )

    args = _build_args(outer)

    assert args.solver_profile == "speed_v2"
    assert args.core_width == 600
    assert args.qubo_mode == "group_majority"
    assert "core_width" in args._profile_explicit_keys
    assert "qubo_group_select" in args._profile_explicit_keys
    assert "qubo_mode" in args._profile_explicit_keys


def test_qubo_influence_profile_applies_without_overriding_explicit_flags(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_qubo_influence_profile

    args = _compact_args(qubo_influence_profile="light", core_width=777)
    monkeypatch.setattr(sys, "argv", ["bench", "--core-width", "777"])

    _apply_qubo_influence_profile(args)

    assert args.core_width == 777
    assert args.max_qubo_vars == 96
    assert args.qubo_supplement_width == 48
    assert args.max_qubo_calls == 64


def test_rank_candidates_defaults_to_full_sorted_order_and_can_truncate():
    from benchmark_compact_beam_qubo_n0 import _rank_candidates

    candidates = {
        b"a": (4, 1, b"p0"),
        b"b": (1, 3, b"p1"),
        b"c": (2, 2, b"p2"),
        b"d": (1, 1, b"p3"),
    }
    args = _compact_args(rank_pool_size=0)

    ranked = _rank_candidates(candidates, args)
    assert [code for code, _data in ranked] == [b"d", b"b", b"c", b"a"]
    assert args._rank_pool_truncated_candidates == 0

    args = _compact_args(rank_pool_size=2, rank_pool_min_candidates=0)
    ranked = _rank_candidates(candidates, args)
    assert [code for code, _data in ranked] == [b"d", b"b"]
    assert args._rank_pool_truncated_candidates == 2


def test_candidate_sort_key_matches_candidate_rank_order_for_default_priority():
    from benchmark_compact_beam_qubo_n0 import _candidate_rank, _candidate_sort_key

    args = _compact_args(priority="n_then_depth")
    items = [
        (b"c", (2, 1, b"p1")),
        (b"a", (1, 5, b"p2")),
        (b"b", (1, 3, b"p3")),
        (b"d", (1, 3, b"p1")),
    ]

    assert sorted(items, key=lambda item: _candidate_sort_key(item, args)) == sorted(
        items,
        key=lambda item: _candidate_rank(item, args),
    )


def test_stream_worst_key_matches_candidate_rank_for_parent_ties():
    from benchmark_compact_beam_qubo_n0 import _candidate_rank, _stream_worst_key

    args = _compact_args()
    good = (b"z", (1, 1, b"a"))
    worse = (b"a", (1, 1, b"z"))

    assert max([good, worse], key=lambda item: _candidate_rank(item, args)) == worse
    assert min(
        [
            (_stream_worst_key(good[0], good[1], args, 0), good),
            (_stream_worst_key(worse[0], worse[1], args, 1), worse),
        ]
    )[1] == worse


def test_stream_candidate_limit_keeps_global_top_k():
    from benchmark_compact_beam_qubo_n0 import _offer_stream_candidate

    args = _compact_args(stream_candidate_limit=2, stream_candidate_trim_factor=1.0)
    candidates = {}
    heap = []
    rows = [
        (b"z", (1, 1, b"a")),
        (b"a", (1, 1, b"z")),
        (b"b", (2, 1, b"a")),
        (b"c", (0, 5, b"a")),
    ]
    pruned = 0
    for seq, (code, data) in enumerate(rows):
        pruned += _offer_stream_candidate(candidates, heap, code, data, args, seq)

    assert pruned == 2
    assert set(candidates) == {b"z", b"c"}


def test_post_polish_detail_rows_pair_old_and_new_suffix_moves():
    from trace_qubo_path_contributions import _post_polish_detail_rows

    step_rows = [
        {
            "case": "A2_26",
            "path_kind": "search_path_before_polish",
            "step": "18",
            "box_rank": "18",
            "from_stack": "4",
            "to_stack": "9",
            "reason": "DIRECT_DROP_N",
            "n_before": "9",
            "n_after": "8",
        },
        {
            "case": "A2_26",
            "path_kind": "search_path_before_polish",
            "step": "19",
            "box_rank": "5",
            "from_stack": "4",
            "to_stack": "9",
            "reason": "DIRECT_DROP_N",
            "n_before": "8",
            "n_after": "7",
        },
        {
            "case": "A2_26",
            "path_kind": "final_path_after_polish",
            "step": "18",
            "box_rank": "18",
            "from_stack": "4",
            "to_stack": "3",
            "reason": "DIRECT_DROP_N",
            "n_before": "9",
            "n_after": "8",
        },
    ]
    events = [
        {
            "case": "A2_26",
            "event_type": "post_polish_suffix",
            "start_step": "18",
            "end_step": "19",
            "old_moves": "2",
            "new_moves": "1",
            "moves_saved": "1",
        }
    ]

    rows = _post_polish_detail_rows(step_rows, events)

    assert len(rows) == 2
    assert rows[0]["old_move"] == "18:4->9"
    assert rows[0]["new_move"] == "18:4->3"
    assert rows[1]["old_move"] == "5:4->9"
    assert rows[1]["new_move"] == ""
    assert rows[1]["move_delta"] == "removed"


def test_compact_row_dict_exposes_qubo_and_pool_metrics():
    from benchmark_compact_beam_qubo_n0 import CompactBeamResult, _row_dict_from_result

    result = CompactBeamResult(
        case="A1_01_single",
        status="TARGET_REACHABLE",
        moves=[],
        runtime_sec=0.123,
        expanded=10,
        generated=20,
        visited=15,
        open_count=7,
        pruned_bound=1,
        pruned_business=2,
        dominated=3,
        peak_frontier=4,
        peak_candidates=5,
        depth_reached=6,
        qubo_calls=2,
        qubo_vars=8,
        qubo_time_sec=0.004,
        qubo_selected_total=9,
        qubo_selected_on_path=1,
        qubo_ancestor_on_path=3,
        group_qubo_calls=1,
        group_qubo_vars=11,
        group_qubo_time_sec=0.006,
        group_qubo_selected_total=12,
        post_polish_moves_saved=1,
        stream_pruned_candidates=13,
        rank_pool_truncated_candidates=14,
    )
    args = _compact_args(qubo_influence_profile="medium", rank_pool_size=5000)

    row = _row_dict_from_result(result, "A1", "01", args)

    assert row["profile"] == "speed_v2"
    assert row["solver_profile"] == "speed_v2"
    assert row["qubo_influence_profile"] == "medium"
    assert row["peak_candidates"] == 5
    assert row["open_count"] == 7
    assert row["qubo_selected_on_path"] == 1
    assert row["qubo_ancestor_on_path"] == 3
    assert row["rank_pool_size"] == 5000
    assert row["rank_pool_truncated_candidates"] == 14
    assert "peak_candidates=5" in row["status"]
