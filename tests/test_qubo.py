from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

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


def test_sa_sampler_can_ignore_tiny_time_limit_for_fixed_iterations():
    model = QuboModel()
    x = model.add_binary_var("x")
    model.add_linear(x, -1.0)

    ss = SimulatedAnnealingQuboSampler().sample(
        model,
        QuboSamplerConfig(
            num_reads=5,
            sa_iterations=20,
            time_limit_sec=0.0,
            respect_time_limit=False,
        ),
    )

    assert len(ss.samples) == 5


def test_compact_qubo_sampler_config_disables_wall_clock_limit_when_requested():
    from benchmark_compact_beam_qubo_n0 import _qubo_sampler_config

    args = _compact_args(deterministic_qubo_sampler=True)

    config = _qubo_sampler_config(
        args,
        num_reads=2,
        random_seed=37,
        time_limit_sec=0.001,
        sa_iterations=20,
    )

    assert config.respect_time_limit is False


def test_generic_qubo_boosted_respects_deterministic_sampler_flag(monkeypatch):
    import benchmark_target_astar_memory_n0 as target
    from benchmark_compact_beam_qubo_n0 import make_compact_args
    from container_reshuffle.qubo.sampler import QuboSampleSet

    captured: dict[str, bool] = {}

    def fake_sample(self, model, config):
        captured["respect_time_limit"] = config.respect_time_limit
        return QuboSampleSet(samples=[QuboSample({0: 1, 1: 1}, 0.0)], sampler_name="fake", runtime_sec=0.0)

    monkeypatch.setattr(target.SimulatedAnnealingQuboSampler, "sample", fake_sample)
    args = make_compact_args(
        use_qubo=True,
        qubo_min_candidates=1,
        qubo_core_width=0,
        qubo_supplement_width=1,
        max_qubo_vars=4,
        deterministic_qubo_sampler=True,
    )
    candidates = {
        bytes([idx + 1]): (1, idx, b"parent")
        for idx in range(4)
    }

    selected, qvars, _qtime = target._select_qubo_boosted(candidates, args)

    assert selected
    assert qvars == 2
    assert captured["respect_time_limit"] is False


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


def test_qubo_pair_policy_baseline_none_and_parent_only():
    from benchmark_target_astar_memory_n0 import _pair_penalty_matrix

    payload = [
        (2, 5, b"p0", b"abcde12345"),
        (2, 5, b"p0", b"abcdf12345"),
        (2, 5, b"p1", b"abcde12399"),
    ]
    diversity_weight = 0.3
    baseline_args = argparse.Namespace(qubo_pair_policy="baseline")
    none_args = argparse.Namespace(qubo_pair_policy="none")
    parent_args = argparse.Namespace(
        qubo_pair_policy="parent_only",
        qubo_parent_sim_weight=0.65,
    )

    assert _pair_penalty_matrix(payload, diversity_weight, baseline_args) == _pair_penalty_matrix(
        payload,
        diversity_weight,
    )
    assert _pair_penalty_matrix(payload, diversity_weight, none_args) == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]

    pairs = _pair_penalty_matrix(payload, diversity_weight, parent_args)

    assert pairs[0][1] == pytest.approx(diversity_weight * 0.65)
    assert pairs[0][2] == 0.0


def test_qubo_pair_policy_stack_signature_uses_structural_similarity():
    from benchmark_target_astar_memory_n0 import _pair_penalty_matrix
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    code_a, n_a = _encode([[1, 4], [2], [3]], stack_count, max_height)
    code_b, n_b = _encode([[1, 5], [2], [3]], stack_count, max_height)
    code_c, n_c = _encode([[1], [2, 5], [3]], stack_count, max_height)
    payload = [
        (n_a, 4, b"pa", code_a),
        (n_b, 4, b"pb", code_b),
        (n_c, 4, b"pc", code_c),
    ]
    args = argparse.Namespace(
        qubo_pair_policy="stack_signature",
        qubo_stack_sim_weight=0.40,
        qubo_sim_threshold=0.70,
        _stack_count=stack_count,
        _max_height=max_height,
    )

    pairs = _pair_penalty_matrix(payload, 0.5, args)

    assert pairs[0][1] > 0.0
    assert pairs[0][1] > pairs[0][2]


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


def test_compact_full_state_successor_cache_hits_on_repeated_state():
    from benchmark_compact_beam_qubo_n0 import _successors_with_n_cached
    from run_business_exact_target_search_sqlite import _encode, _successors_with_n

    _successors_with_n_cached.cache_clear()
    code, n_value = _encode([[5, 2], [4, 3], [1], []], 4, 3)

    first = tuple(_successors_with_n_cached(code, n_value, 4, 3))
    after_first = _successors_with_n_cached.cache_info()
    second = tuple(_successors_with_n_cached(code, n_value, 4, 3))
    after_second = _successors_with_n_cached.cache_info()

    assert first == tuple(_successors_with_n(code, n_value, 4, 3))
    assert second == first
    assert after_first.misses == 1
    assert after_second.hits == 1


def test_compact_uncached_successor_stream_matches_reference_without_filling_full_cache():
    from benchmark_compact_beam_qubo_n0 import _successors_with_n_cached, _successors_with_n_uncached
    from run_business_exact_target_search_sqlite import _encode, _successors_with_n

    _successors_with_n_cached.cache_clear()
    code, n_value = _encode([[5, 2], [4, 3], [1], []], 4, 3)

    stream = tuple(_successors_with_n_uncached(code, n_value, 4, 3))

    assert stream == tuple(_successors_with_n(code, n_value, 4, 3))
    assert _successors_with_n_cached.cache_info().currsize == 0


def test_post_polish_successor_cache_gate_uses_cache_only_when_enabled():
    from benchmark_compact_beam_qubo_n0 import _post_polish_successors, _successors_with_n_cached
    from run_business_exact_target_search_sqlite import _encode, _successors_with_n

    code, n_value = _encode([[5, 2], [4, 3], [1], []], 4, 3)
    reference = tuple(_successors_with_n(code, n_value, 4, 3))

    _successors_with_n_cached.cache_clear()
    uncached_args = _compact_args(post_polish_successor_cache=False)
    assert tuple(_post_polish_successors(code, n_value, 4, 3, uncached_args)) == reference
    assert _successors_with_n_cached.cache_info().currsize == 0

    cached_args = _compact_args(post_polish_successor_cache=True)
    assert tuple(_post_polish_successors(code, n_value, 4, 3, cached_args)) == reference
    assert _successors_with_n_cached.cache_info().currsize == 1


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


def test_dlts_dat_parser_allows_priority_group_instances(tmp_path):
    from open_cpmp_benchmarks import load_dlts_dat, requires_priority_group_semantics

    case_dir = tmp_path / "3x2_cv2_test"
    case_dir.mkdir()
    path = case_dir / "0.dat"
    path.write_text(
        "\n".join(
            [
                "3 6",
                "2 2 1",
                "2 2 3",
                "2 1 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    instance = load_dlts_dat(path)

    assert requires_priority_group_semantics(instance)
    assert instance.container_count == 6
    assert instance.stacks == [[2, 1], [2, 3], [1, 3]]
    with pytest.raises(ValueError):
        load_dlts_dat(path, allow_priority_groups=False)


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
    assert row["rescue_layers_used"] == 0
    assert business_ok
    assert goal_ok, msg


def test_open_row_serializes_post_polish_successor_cache_for_priority_group_instances():
    from benchmark_compact_beam_qubo_n0 import CompactBeamResult
    from open_cpmp_benchmarks import _open_row_dict_from_instance_result, requires_priority_group_semantics

    instance = Instance(
        "tiny_open_cache_row",
        ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=3,
        max_height=3,
        container_count=3,
        stacks=[[1, 2], [1], []],
    )
    result = CompactBeamResult(
        case=instance.instance_id,
        status="ok",
        moves=[],
        runtime_sec=0.0,
        expanded=0,
        generated=0,
        visited=0,
        open_count=0,
        pruned_bound=0,
        pruned_business=0,
        dominated=0,
        peak_frontier=0,
        peak_candidates=0,
        depth_reached=0,
    )
    args = _compact_args(post_polish_successor_cache=True)

    assert requires_priority_group_semantics(instance)

    row = _open_row_dict_from_instance_result(result, instance, args)

    assert row["post_polish_successor_cache"] is True


def test_unified_wang_profile_initializes_target_layouts():
    from benchmark_compact_beam_qubo_n0 import make_compact_args, run_compact_beam_instance

    instance = Instance(
        "tiny_unified_target_layouts",
        ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=3,
        max_height=3,
        container_count=4,
        stacks=[[1, 4], [2], [3]],
    )
    args = make_compact_args(
        alpha=10.0,
        beam_width=4,
        core_width=4,
        max_moves=0,
        max_qubo_calls=0,
        post_polish_qubo=False,
        solver_profile="scheduler_state_v7_wang_qubo_unified",
        time_limit=0.01,
        use_qubo=False,
    )

    run_compact_beam_instance(instance.instance_id, instance, args)

    assert args.unified_wang_target_gap_weight > 0.0
    assert len(args._layout_target_groups) > 0


def test_compact_row_serializes_unified_wang_qubo_config_fields():
    from benchmark_compact_beam_qubo_n0 import CompactBeamResult, _row_dict_from_instance_result

    instance = Instance(
        "tiny_unified_wang_config",
        ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=3,
        max_height=3,
        container_count=3,
        stacks=[[1], [2], [3]],
    )
    result = CompactBeamResult(
        case=instance.instance_id,
        status="ok",
        moves=[],
        runtime_sec=0.0,
        expanded=0,
        generated=0,
        visited=0,
        open_count=0,
        pruned_bound=0,
        pruned_business=0,
        dominated=0,
        peak_frontier=0,
        peak_candidates=0,
        depth_reached=0,
    )
    args = _compact_args(
        unified_wang_qubo_enabled=True,
        unified_wang_qubo_supplement_width=12,
        unified_wang_qubo_focus_n=-1,
        unified_wang_qubo_n_window=1,
        unified_wang_qubo_max_vars=64,
        unified_wang_qubo_select=12,
        unified_wang_n_weight=1000.0,
        unified_wang_depth_weight=1.0,
        unified_wang_fixed_prefix_weight=20.0,
        unified_wang_fix_pair_weight=8.0,
        unified_wang_target_gap_weight=1.0,
        unified_wang_same_parent_penalty=0.10,
        unified_wang_pair_same_parent_penalty=0.25,
        unified_wang_signature_penalty=0.10,
        unified_wang_complement_bonus=0.05,
    )

    row = _row_dict_from_instance_result(result, instance, args)

    expected = {
        "unified_wang_qubo_enabled": True,
        "unified_wang_qubo_supplement_width": 12,
        "unified_wang_qubo_focus_n": -1,
        "unified_wang_qubo_n_window": 1,
        "unified_wang_qubo_max_vars": 64,
        "unified_wang_qubo_select": 12,
        "unified_wang_n_weight": 1000.0,
        "unified_wang_depth_weight": 1.0,
        "unified_wang_fixed_prefix_weight": 20.0,
        "unified_wang_fix_pair_weight": 8.0,
        "unified_wang_target_gap_weight": 1.0,
        "unified_wang_same_parent_penalty": 0.10,
        "unified_wang_pair_same_parent_penalty": 0.25,
        "unified_wang_signature_penalty": 0.10,
        "unified_wang_complement_bonus": 0.05,
    }
    for field, value in expected.items():
        assert row[field] == value


def test_wang_incumbent_fallback_solves_when_compact_depth_is_zero():
    from benchmark_compact_beam_qubo_n0 import (
        _row_dict_from_instance_result,
        _validate_instance_result,
        make_compact_args,
        run_compact_beam_instance,
    )

    instance = Instance(
        "tiny_wang_fallback",
        ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=3,
        max_height=3,
        container_count=3,
        stacks=[[1, 3], [], [2]],
    )
    args = make_compact_args(
        alpha=10.0,
        beam_width=2,
        core_width=2,
        max_moves=0,
        max_qubo_calls=0,
        post_polish_qubo=False,
        solver_profile="speed_v2",
        time_limit=1.0,
        use_qubo=False,
        wang_incumbent_enabled=True,
        wang_incumbent_beam_width=8,
        wang_incumbent_candidate_limit=8,
        wang_incumbent_max_depth=8,
        wang_incumbent_time_limit=1.0,
    )

    result = run_compact_beam_instance(instance.instance_id, instance, args)
    row = _row_dict_from_instance_result(result, instance, args)
    business_ok, goal_ok, msg = _validate_instance_result(result, instance, business_alpha=10.0)

    assert row["final_n"] == 0
    assert row["moves"] == 1
    assert row["wang_incumbent_used"] is True
    assert "wang_incumbent_used=True" in row["status"]
    assert business_ok
    assert goal_ok, msg


def test_wang_search_depth_cap_targets_strict_improvement():
    from benchmark_compact_beam_qubo_n0 import _wang_search_depth_cap

    args = _compact_args(wang_cap_enabled=True)

    assert _wang_search_depth_cap(base_depth_cap=10, incumbent_moves=5, args=args) == 4
    assert _wang_search_depth_cap(base_depth_cap=3, incumbent_moves=5, args=args) == 3

    args.wang_cap_enabled = False
    assert _wang_search_depth_cap(base_depth_cap=10, incumbent_moves=5, args=args) == 10


def test_wang_feature_adjustment_preserves_group_score_with_zero_weights():
    from benchmark_compact_beam_qubo_n0 import _group_qubo_linear_score
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    code, n_value = _encode([[4], [3, 2], [1]], stack_count, max_height)
    rows = [(code, (n_value, 7, b"parent"))]
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        qubo_group_lookahead_weight=0.0,
        qubo_group_branch_weight=0.0,
        wang_feature_fixed_prefix_weight=0.0,
        wang_feature_fix_pair_weight=0.0,
        wang_feature_risk_weight=0.0,
        wang_feature_same_parent_weight=0.0,
    )

    assert _group_qubo_linear_score(rows, args) == float(n_value * 1000 + 7)


def test_wang_feature_fixed_prefix_weight_lowers_better_prefix_score():
    from benchmark_compact_beam_qubo_n0 import _group_qubo_linear_score
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    better_prefix, n_value = _encode([[5, 4], [3], [2, 1]], stack_count, max_height)
    worse_prefix, worse_n = _encode([[4, 5], [3], [2, 1]], stack_count, max_height)
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        qubo_group_lookahead_weight=0.0,
        qubo_group_branch_weight=0.0,
        wang_feature_fixed_prefix_weight=25.0,
        wang_feature_fix_pair_weight=0.0,
        wang_feature_risk_weight=0.0,
        wang_feature_same_parent_weight=0.0,
    )

    better_depth = 1004
    worse_depth = 4
    assert n_value * 1000 + better_depth == worse_n * 1000 + worse_depth

    better_score = _group_qubo_linear_score([(better_prefix, (n_value, better_depth, b"p1"))], args)
    worse_score = _group_qubo_linear_score([(worse_prefix, (worse_n, worse_depth, b"p2"))], args)

    assert better_score < worse_score


def test_open_runner_solves_tiny_priority_group_cpmp():
    from benchmark_compact_beam_qubo_n0 import make_compact_args, run_compact_beam_instance
    from open_cpmp_benchmarks import (
        _open_row_dict_from_instance_result,
        _open_validate_instance_result,
        _simulate_priority_group_moves,
        requires_priority_group_semantics,
    )

    sorted_instance = Instance(
        "tiny_priority_group_sorted",
        ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=3,
        max_height=3,
        container_count=3,
        stacks=[[2, 2], [1], []],
    )
    ok, msg, _stacks, n_trace, _max_g = _simulate_priority_group_moves(sorted_instance, [])
    assert ok, msg
    assert n_trace[-1] == 0

    instance = Instance(
        "tiny_priority_group",
        ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=3,
        max_height=3,
        container_count=3,
        stacks=[[1, 2], [1], []],
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
    row = _open_row_dict_from_instance_result(result, instance, args)
    business_ok, goal_ok, msg = _open_validate_instance_result(result, instance, business_alpha=10.0)

    assert requires_priority_group_semantics(instance)
    assert row["final_n"] == 0
    assert row["moves"] == 1
    assert row["qubo_pair_policy"] == "baseline"
    assert row["rescue_layers_used"] == 0
    assert business_ok
    assert goal_ok, msg


def test_open_priority_group_row_serializes_unified_wang_qubo_fields():
    from benchmark_compact_beam_qubo_n0 import CompactBeamResult
    from open_cpmp_benchmarks import _open_row_dict_from_instance_result, requires_priority_group_semantics

    instance = Instance(
        "tiny_priority_group_unified_row",
        ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=3,
        max_height=3,
        container_count=3,
        stacks=[[1, 2], [1], []],
    )
    result = CompactBeamResult(
        case=instance.instance_id,
        status="ok",
        moves=[],
        runtime_sec=0.0,
        expanded=0,
        generated=0,
        visited=0,
        open_count=0,
        pruned_bound=0,
        pruned_business=0,
        dominated=0,
        peak_frontier=0,
        peak_candidates=0,
        depth_reached=0,
        unified_wang_qubo_calls=3,
        unified_wang_qubo_vars=21,
        unified_wang_qubo_time_sec=0.125,
        unified_wang_fixed_prefix_selected=8,
        unified_wang_fix_pair_selected=5,
    )
    args = _compact_args(
        unified_wang_qubo_enabled=True,
        unified_wang_qubo_supplement_width=12,
        unified_wang_qubo_focus_n=-1,
        unified_wang_qubo_n_window=1,
        unified_wang_qubo_max_vars=64,
        unified_wang_qubo_select=12,
        unified_wang_n_weight=1000.0,
        unified_wang_depth_weight=1.0,
        unified_wang_fixed_prefix_weight=20.0,
        unified_wang_fix_pair_weight=8.0,
        unified_wang_target_gap_weight=1.0,
        unified_wang_same_parent_penalty=0.10,
        unified_wang_pair_same_parent_penalty=0.25,
        unified_wang_signature_penalty=0.10,
        unified_wang_complement_bonus=0.05,
    )
    args._layout_target_groups = (("target",),)

    row = _open_row_dict_from_instance_result(result, instance, args)

    assert requires_priority_group_semantics(instance)
    expected = {
        "unified_wang_qubo_enabled": True,
        "unified_wang_qubo_supplement_width": 12,
        "unified_wang_qubo_focus_n": -1,
        "unified_wang_qubo_n_window": 1,
        "unified_wang_qubo_max_vars": 64,
        "unified_wang_qubo_select": 12,
        "unified_wang_n_weight": 1000.0,
        "unified_wang_depth_weight": 1.0,
        "unified_wang_fixed_prefix_weight": 20.0,
        "unified_wang_fix_pair_weight": 8.0,
        "unified_wang_target_gap_weight": 1.0,
        "unified_wang_same_parent_penalty": 0.10,
        "unified_wang_pair_same_parent_penalty": 0.25,
        "unified_wang_signature_penalty": 0.10,
        "unified_wang_complement_bonus": 0.05,
        "unified_wang_qubo_calls": 3,
        "unified_wang_qubo_vars": 21,
        "unified_wang_qubo_time_sec": 0.125,
        "unified_wang_fixed_prefix_selected": 8,
        "unified_wang_fix_pair_selected": 5,
        "target_layout_count": 1,
    }
    for field, value in expected.items():
        assert row[field] == value


def test_open_priority_group_row_serializes_frg_dlts_fields():
    from benchmark_compact_beam_qubo_n0 import CompactBeamResult
    from open_cpmp_benchmarks import _open_row_dict_from_instance_result, requires_priority_group_semantics

    instance = Instance(
        "tiny_priority_group_frg_dlts_row",
        ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=3,
        max_height=3,
        container_count=3,
        stacks=[[1, 2], [1], []],
    )
    result = CompactBeamResult(
        case=instance.instance_id,
        status="ok",
        moves=[],
        runtime_sec=0.0,
        expanded=0,
        generated=0,
        visited=0,
        open_count=0,
        pruned_bound=0,
        pruned_business=0,
        dominated=0,
        peak_frontier=0,
        peak_candidates=0,
        depth_reached=0,
        wang_pair_qubo_calls=11,
        wang_pair_qubo_vars=22,
        wang_pair_qubo_time_sec=0.33,
        frg_candidate_selected_total=13,
        frg_compound_appended_total=3,
        dlts_value_selected_total=7,
    )
    args = _compact_args(
        wang_pair_qubo_enabled=True,
        wang_feature_fixed_prefix_weight=3.0,
        wang_feature_fix_pair_weight=2.0,
        wang_feature_risk_weight=1.0,
        wang_feature_same_parent_weight=0.5,
        frg_candidate_supplement_enabled=True,
        frg_candidate_supplement_width=8,
        frg_candidate_focus_n=-1,
        frg_candidate_n_window=2,
        frg_candidate_max_ranked=96,
        frg_compound_escape_enabled=True,
        frg_compound_escape_width=2,
        frg_compound_escape_focus_n=-1,
        frg_compound_escape_plateau_window=5,
        frg_compound_escape_plateau_delta=0,
        frg_compound_escape_cap_slack=24,
        frg_compound_escape_min_candidates=7000,
        dlts_value_supplement_enabled=True,
        dlts_value_supplement_width=4,
    )

    row = _open_row_dict_from_instance_result(result, instance, args)

    assert requires_priority_group_semantics(instance)
    expected = {
        "frg_candidate_supplement_enabled": True,
        "frg_candidate_supplement_width": 8,
        "frg_candidate_focus_n": -1,
        "frg_candidate_n_window": 2,
        "frg_candidate_max_ranked": 96,
        "frg_candidate_selected_total": 13,
        "frg_compound_escape_enabled": True,
        "frg_compound_escape_width": 2,
        "frg_compound_escape_focus_n": -1,
        "frg_compound_escape_plateau_window": 5,
        "frg_compound_escape_plateau_delta": 0,
        "frg_compound_escape_cap_slack": 24,
        "frg_compound_escape_min_candidates": 7000,
        "frg_compound_appended_total": 3,
        "dlts_value_supplement_enabled": True,
        "dlts_value_supplement_width": 4,
        "dlts_value_selected_total": 7,
        "wang_pair_qubo_enabled": True,
        "wang_pair_qubo_calls": 11,
        "wang_pair_qubo_vars": 22,
        "wang_pair_qubo_time_sec": 0.33,
        "wang_feature_fixed_prefix_weight": 3.0,
        "wang_feature_fix_pair_weight": 2.0,
        "wang_feature_risk_weight": 1.0,
        "wang_feature_same_parent_weight": 0.5,
    }
    for field, value in expected.items():
        assert row[field] == value


def test_open_runner_full_quality_summary_penalizes_invalid_rows():
    from open_cpmp_benchmarks import _open_full_quality_summary

    rows = [
        {
            "variant": "qubo",
            "family": "5x5_cv1_test",
            "moves": 10,
            "runtime_sec": 1.0,
            "reference_moves": 8,
            "move_gap": 2,
            "n0": 8,
            "cap_13": 10,
            "final_n": 0,
            "business_ok": True,
            "goal_validation_ok": True,
        },
        {
            "variant": "qubo",
            "family": "5x5_cv1_test",
            "moves": "",
            "runtime_sec": 1.5,
            "reference_moves": 8,
            "move_gap": "",
            "n0": 8,
            "cap_13": 10,
            "final_n": 3,
            "business_ok": False,
            "goal_validation_ok": False,
        },
    ]

    summary = _open_full_quality_summary(rows, penalty_factor=4)

    assert summary[("qubo", "5x5_cv1_test")]["valid"] == 1
    assert summary[("qubo", "5x5_cv1_test")]["invalid_or_unsolved"] == 1
    assert summary[("qubo", "5x5_cv1_test")]["full_quality_penalty_total"] == 50


def _compact_args(**overrides):
    values = {
        "priority": "n_then_depth",
        "solver_profile": "speed_v2",
        "g_weight": 1.0,
        "n_weight": 1.0,
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
        "_qubo_calls_done": 0,
        "max_qubo_vars": 64,
        "qubo_core_width": 224,
        "qubo_supplement_width": 24,
        "qubo_min_candidates": 64,
        "qubo_focus_n": 8,
        "qubo_first": False,
        "qubo_target_share": 0.5,
        "qubo_cohorts_per_layer": 2,
        "qubo_cohort_stride": 64,
        "qubo_pair_policy": "baseline",
        "qubo_parent_sim_weight": 0.65,
        "qubo_layout_sim_weight": 1.0,
        "qubo_action_sim_weight": 0.0,
        "qubo_stack_sim_weight": 0.35,
        "qubo_sim_threshold": 0.80,
        "qubo_time_limit": 0.001,
        "qubo_iterations": 20,
        "deterministic_qubo_sampler": False,
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
        "stream_bucket_topk": 0,
        "rank_pool_size": 0,
        "rank_pool_min_candidates": 0,
        "_rank_pool_truncated_candidates": 0,
        "post_polish_qubo": True,
        "post_polish_mode": "legacy",
        "post_polish_accept": "better",
        "post_polish_successor_cache": False,
        "adaptive_high_n": 16,
        "adaptive_low_n": 8,
        "adaptive_high_frontier": 800,
        "adaptive_mid_frontier": 1000,
        "adaptive_low_frontier": 1200,
        "adaptive_core_ratio": 0.90,
        "adaptive_candidate_pool": 0,
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
        "bucket_key": "n_depth",
        "bucket_n_band_size": 2,
        "bucket_core_ratio": 0.5,
        "bucket_qubo_share": 0.5,
        "bucket_max_qubo_vars": 96,
        "bucket_qubo_select": 24,
        "bucket_min_candidates": 64,
        "post_polish_window_min_size": 3,
        "post_polish_window_max_size": 10,
        "post_polish_window_stride": 1,
        "post_polish_window_budget": 96,
        "post_polish_window_qubo_vars": 72,
        "post_polish_window_select": 8,
        "post_polish_window_time_limit": 0.001,
        "post_polish_window_repair_expanded": 20000,
        "post_polish_window_repair_depth_extra": 0,
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
        "wang_rollout_inject_candidates": False,
        "feasibility_escape_enabled": False,
        "feasibility_escape_supplement_width": 0,
        "feasibility_escape_focus_n": 8,
        "feasibility_escape_n_window": 1,
        "feasibility_escape_dead_end_threshold": 0.45,
        "feasibility_escape_min_temporary_slack": 1,
        "frg_candidate_supplement_enabled": False,
        "frg_candidate_supplement_width": 0,
        "frg_candidate_focus_n": -1,
        "frg_candidate_n_window": 2,
        "frg_candidate_max_ranked": 96,
        "frg_compound_escape_enabled": False,
        "frg_compound_escape_width": 0,
        "frg_compound_escape_focus_n": -1,
        "frg_compound_escape_plateau_window": 5,
        "frg_compound_escape_plateau_delta": 0,
        "frg_compound_escape_cap_slack": 24,
        "frg_compound_escape_min_candidates": 7000,
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
        "bs_frg_qubo_signature_penalty": 0.15,
        "bs_frg_qubo_n_band_penalty": 0.05,
        "bs_frg_qubo_n_band_size": 1,
        "wang2017_task_enabled": False,
        "wang2017_task_supplement_width": 0,
        "wang2017_task_focus_n": 8,
        "wang2017_task_n_window": 1,
        "wang2017_task_plateau_window": 0,
        "wang2017_task_plateau_delta": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_beam_survivor_metrics_count_dead_frontier_parents():
    from benchmark_compact_beam_qubo_n0 import _beam_survivor_parent_metrics

    p0, p1, p2, p3 = b"p0", b"p1", b"p2", b"p3"
    c0, c1, c2, c3 = b"c0", b"c1", b"c2", b"c3"
    frontier = [(p0, 5, 0), (p1, 5, 0), (p2, 5, 0), (p3, 5, 0)]
    candidates = {
        c0: (4, 1, p0),
        c1: (4, 1, p0),
        c2: (4, 1, p2),
        c3: (4, 1, p3),
    }

    metrics = _beam_survivor_parent_metrics(
        frontier,
        candidates,
        selected_codes=[c0, c1, c2],
        selected_by_qubo={c1, c2},
        top_limits=(1, 2, 3),
    )

    assert metrics["frontier_size"] == 4
    assert metrics["candidate_count"] == 4
    assert metrics["selected_count"] == 3
    assert metrics["selected_parent_count"] == 2
    assert metrics["dead_parent_count"] == 2
    assert metrics["dead_parent_ratio"] == 0.5
    assert metrics["selected_children_top1_parent"] == 2
    assert metrics["selected_children_top2_parent"] == 2
    assert metrics["selected_children_top3_parent"] == 3
    assert metrics["selected_children_tail_half_parent"] == 1
    assert metrics["selected_parents_top1"] == 1
    assert metrics["selected_parents_top2"] == 1
    assert metrics["selected_parents_top3"] == 2
    assert metrics["selected_parents_tail_half"] == 1
    assert metrics["qubo_selected_count"] == 2
    assert metrics["qubo_selected_parent_count"] == 2


def test_beam_survivor_rows_mark_final_search_path_edges():
    from benchmark_compact_beam_qubo_n0 import _finalize_beam_survivor_rows

    root, c0, c1, off_path = b"root", b"c0", b"c1", b"off"
    records = [
        {
            "row": {"case": "toy", "layer": 0},
            "selected_rank_by_code": {c0: 1, off_path: 2},
            "parent_by_code": {c0: root, off_path: root},
            "parent_rank_by_code": {root: 1},
        },
        {
            "row": {"case": "toy", "layer": 1},
            "selected_rank_by_code": {c1: 1},
            "parent_by_code": {c1: c0},
            "parent_rank_by_code": {c0: 1},
        },
    ]

    rows = _finalize_beam_survivor_rows(records, [root, c0, c1])

    assert rows[0]["on_final_search_path"] == 1
    assert rows[0]["path_child_selected_rank"] == 1
    assert rows[0]["path_parent_frontier_rank"] == 1
    assert rows[1]["on_final_search_path"] == 1
    assert rows[1]["path_child_selected_rank"] == 1
    assert rows[1]["path_parent_frontier_rank"] == 1


def test_survivor_parent_core_cap_contracts_releases_and_skips_tail():
    from benchmark_compact_beam_qubo_n0 import _survivor_parent_core_cap

    args = _compact_args(
        parent_core_cap=4,
        survivor_parent_cap_enabled=True,
        survivor_parent_cap_value=2,
        survivor_parent_cap_trigger_ratio=0.70,
        survivor_parent_cap_release_ratio=0.55,
        survivor_parent_cap_window=2,
        survivor_parent_cap_min_n=8,
        _survivor_dead_parent_ratio_history=[0.72, 0.74],
    )

    cap, reason = _survivor_parent_core_cap(args, best_n=20)

    assert cap == 2
    assert reason == "survivor_contract"
    assert args._survivor_parent_cap_active is True

    args._survivor_dead_parent_ratio_history = [0.40, 0.50]
    cap, reason = _survivor_parent_core_cap(args, best_n=20)

    assert cap == 4
    assert reason == "survivor_release"
    assert args._survivor_parent_cap_active is False

    args._survivor_dead_parent_ratio_history = [0.90, 0.92]
    cap, reason = _survivor_parent_core_cap(args, best_n=5)

    assert cap == 4
    assert reason == "survivor_tail"
    assert args._survivor_parent_cap_active is False


def test_enhanced_frontier_uses_dynamic_survivor_parent_core_cap():
    from benchmark_compact_beam_qubo_n0 import _select_enhanced_qubo_frontier

    ranked = [
        (b"c00", (12, 1, b"p0")),
        (b"c01", (12, 1, b"p0")),
        (b"c10", (12, 1, b"p1")),
        (b"c20", (12, 1, b"p2")),
    ]
    args = _compact_args(
        use_qubo=False,
        beam_width=3,
        core_width=3,
        parent_core_cap=4,
        survivor_parent_cap_enabled=True,
        survivor_parent_cap_value=1,
        survivor_parent_cap_trigger_ratio=0.70,
        survivor_parent_cap_release_ratio=0.55,
        survivor_parent_cap_window=1,
        survivor_parent_cap_min_n=8,
        _survivor_dead_parent_ratio_history=[0.80],
    )

    selected, _selected_by_qubo, _qcalls, _qvars, _qtime = _select_enhanced_qubo_frontier(ranked, len(ranked), args)

    assert selected == [b"c00", b"c10", b"c20"]
    assert args._survivor_parent_cap_last == 1
    assert args._survivor_parent_cap_last_reason == "survivor_contract"


def test_feasibility_escape_supplement_prefers_lower_escape_score(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as bench
    from container_reshuffle.solvers.feasibility_signals import FeasibilitySignals

    low = b"low"
    high = b"high"
    args = _compact_args(feasibility_escape_min_temporary_slack=1)

    def fake_signals(code, selector_args):
        if code == low:
            return FeasibilitySignals(3, 3, 0, 2, 1, 3, 0, 0.20)
        return FeasibilitySignals(1, 1, 4, 0, 0, 0, 3, 0.80)

    monkeypatch.setattr(bench, "_feasibility_signals_for_code", fake_signals)

    selected = bench._select_feasibility_escape_supplement(
        [(high, (2, 7, b"p0")), (low, (2, 7, b"p1"))],
        args,
        target=1,
    )

    assert selected == [low]


def test_feasibility_escape_selector_is_used_inside_enhanced_frontier(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as bench
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    chosen_code, chosen_n = _encode([[5, 4], [3, 1], [2]], stack_count, max_height)
    other_code, other_n = _encode([[4, 5], [3, 1], [2]], stack_count, max_height)
    ranked = [
        (other_code, (other_n, 8, b"p0")),
        (chosen_code, (chosen_n, 8, b"p1")),
    ]
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        use_qubo=True,
        qubo_min_candidates=99,
        feasibility_escape_enabled=True,
        feasibility_escape_supplement_width=1,
        feasibility_escape_focus_n=-1,
        feasibility_escape_n_window=2,
        _best_n_history=[chosen_n, chosen_n, chosen_n],
        _layer_depth_slack=1,
        _layer_parent_top_share=0.50,
    )

    def fake_layer_active(selector_args, best_n):
        return True

    def fake_selector(rows, selector_args, target):
        assert rows
        assert target == 1
        return [chosen_code]

    monkeypatch.setattr(bench, "_feasibility_escape_layer_active", fake_layer_active)
    monkeypatch.setattr(bench, "_select_feasibility_escape_supplement", fake_selector)

    selected, selected_by_qubo, qcalls, qvars, qtime = bench._select_enhanced_qubo_frontier(
        ranked,
        candidate_count=len(ranked),
        args=args,
        beam_width=2,
        core_width=1,
        qubo_supplement_width=0,
    )

    assert chosen_code in selected
    assert chosen_code not in selected_by_qubo
    assert qcalls == 0
    assert qvars == 0
    assert qtime == 0.0
    assert args._feasibility_escape_calls_done == 1
    assert args._feasibility_escape_added_total == 1


def test_bs_frg_qubo_supplement_selects_quality_diverse_candidates(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as bench
    from container_reshuffle.qubo.sampler import QuboSampleSet
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 4
    max_height = 3
    parent_a, _ = _encode([[5, 1, 4], [3, 2], [], []], stack_count, max_height)
    parent_b, _ = _encode([[5, 4], [3, 1, 2], [], []], stack_count, max_height)
    low_frg_a, low_n = _encode([[5, 1], [3, 2], [4], []], stack_count, max_height)
    duplicate_parent_a, duplicate_n = _encode([[5, 1], [3], [4], [2]], stack_count, max_height)
    low_frg_b, low_b_n = _encode([[5], [3, 1, 2], [4], []], stack_count, max_height)
    weak, weak_n = _encode([[5, 1, 4], [3], [2], []], stack_count, max_height)
    ranked = [
        (weak, (weak_n, 4, parent_a)),
        (low_frg_a, (low_n, 4, parent_a)),
        (duplicate_parent_a, (duplicate_n, 4, parent_a)),
        (low_frg_b, (low_b_n, 4, parent_b)),
    ]
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        bs_frg_qubo_max_vars=4,
        bs_frg_qubo_same_parent_penalty=3.0,
        bs_frg_qubo_signature_penalty=1.0,
    )

    def fake_sample(self, model, config):
        values = {idx: 1 for idx in range(len(model.var_index))}
        return QuboSampleSet(samples=[QuboSample(values, model.evaluate(values))], sampler_name="fake", runtime_sec=0.25)

    monkeypatch.setattr(bench.SimulatedAnnealingQuboSampler, "sample", fake_sample)

    selected, qvars, qtime = bench._select_bs_frg_qubo_supplement(ranked, args, target=2)

    assert qvars == 4
    assert qtime == 0.25
    assert selected == [duplicate_parent_a, low_frg_b]


def test_bs_frg_qubo_selector_is_used_inside_enhanced_frontier(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as bench
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 4
    max_height = 3
    core, core_n = _encode([[5, 4], [3, 2], [1], []], stack_count, max_height)
    chosen, chosen_n = _encode([[5, 1], [3, 2], [4], []], stack_count, max_height)
    other, other_n = _encode([[5, 1, 4], [3], [2], []], stack_count, max_height)
    ranked = [
        (core, (core_n, 3, b"p0")),
        (other, (other_n, 3, b"p1")),
        (chosen, (chosen_n, 3, b"p2")),
    ]
    args = _compact_args(
        beam_width=2,
        core_width=1,
        parent_core_cap=0,
        use_qubo=True,
        qubo_min_candidates=99,
        bs_frg_qubo_enabled=True,
        bs_frg_qubo_supplement_width=1,
        bs_frg_qubo_focus_n=-1,
        bs_frg_qubo_min_candidates=1,
        _stack_count=stack_count,
        _max_height=max_height,
    )

    def fake_selector(rows, selector_args, target):
        assert rows
        assert target == 1
        return [chosen], 3, 0.125

    monkeypatch.setattr(bench, "_select_bs_frg_qubo_supplement", fake_selector)

    selected, selected_by_qubo, qcalls, qvars, qtime = bench._select_enhanced_qubo_frontier(
        ranked,
        candidate_count=len(ranked),
        args=args,
    )

    assert selected == [core, chosen]
    assert chosen in selected_by_qubo
    assert qcalls == 1
    assert qvars == 3
    assert qtime == 0.125
    assert args._bs_frg_qubo_calls_done == 1
    assert args._bs_frg_qubo_selected_total == 1


def test_feasibility_escape_layer_active_allows_low_parent_share_plateau():
    import benchmark_compact_beam_qubo_n0 as bench

    args = _compact_args(
        feasibility_escape_enabled=True,
        feasibility_escape_focus_n=-1,
        rescue_cap_slack=26,
        _best_n_history=[17, 17, 17],
        _layer_depth_slack=14,
        _layer_parent_top_share=0.006,
    )

    assert bench._feasibility_escape_gate_reason(args, best_n=17) == "active"
    assert bench._feasibility_escape_layer_active(args, best_n=17)


def test_feasibility_escape_gate_telemetry_is_serialized():
    from benchmark_compact_beam_qubo_n0 import CompactBeamResult, _row_dict_from_instance_result
    from container_reshuffle.core import Instance

    args = _compact_args()
    result = CompactBeamResult(
        case="case_a",
        status="done",
        moves=[],
        runtime_sec=0.0,
        expanded=0,
        generated=0,
        visited=0,
        open_count=0,
        pruned_bound=0,
        pruned_business=0,
        dominated=0,
        peak_frontier=0,
        peak_candidates=0,
        depth_reached=0,
        feasibility_escape_active_layers=3,
        feasibility_escape_gate_reasons="active:3;not_plateau:5",
    )
    row = _row_dict_from_instance_result(
        result,
        Instance("case_a", ProblemVariant.PREMARSHALLING_CPMP, 1, 2, 2, [[2, 1]]),
        args,
    )

    assert row["feasibility_escape_active_layers"] == 3
    assert row["feasibility_escape_gate_reasons"] == "active:3;not_plateau:5"


def test_frg_dlts_supplement_telemetry_is_serialized():
    from benchmark_compact_beam_qubo_n0 import CompactBeamResult, _row_dict_from_instance_result
    from container_reshuffle.core import Instance

    args = _compact_args(
        frg_candidate_supplement_enabled=True,
        frg_candidate_supplement_width=8,
        dlts_value_supplement_enabled=True,
        dlts_value_supplement_width=4,
    )
    result = CompactBeamResult(
        case="case_a",
        status="done",
        moves=[],
        runtime_sec=0.0,
        expanded=0,
        generated=0,
        visited=0,
        open_count=0,
        pruned_bound=0,
        pruned_business=0,
        dominated=0,
        peak_frontier=0,
        peak_candidates=0,
        depth_reached=0,
        frg_candidate_selected_total=2,
        frg_compound_appended_total=3,
        dlts_value_selected_total=1,
    )
    row = _row_dict_from_instance_result(
        result,
        Instance("case_a", ProblemVariant.PREMARSHALLING_CPMP, 1, 2, 2, [[2, 1]]),
        args,
    )

    assert row["frg_candidate_supplement_enabled"] is True
    assert row["frg_candidate_supplement_width"] == 8
    assert row["frg_candidate_selected_total"] == 2
    assert row["frg_compound_appended_total"] == 3
    assert row["dlts_value_supplement_enabled"] is True
    assert row["dlts_value_supplement_width"] == 4
    assert row["dlts_value_selected_total"] == 1


def test_bs_frg_qubo_telemetry_is_serialized():
    from benchmark_compact_beam_qubo_n0 import CompactBeamResult, _row_dict_from_instance_result
    from container_reshuffle.core import Instance

    args = _compact_args(
        bs_frg_qubo_enabled=True,
        bs_frg_qubo_supplement_width=4,
        bs_frg_qubo_focus_n=-1,
        bs_frg_qubo_n_window=2,
        bs_frg_qubo_max_vars=64,
    )
    result = CompactBeamResult(
        case="case_a",
        status="done",
        moves=[],
        runtime_sec=0.0,
        expanded=0,
        generated=0,
        visited=0,
        open_count=0,
        pruned_bound=0,
        pruned_business=0,
        dominated=0,
        peak_frontier=0,
        peak_candidates=0,
        depth_reached=0,
        bs_frg_qubo_calls=2,
        bs_frg_qubo_vars=8,
        bs_frg_qubo_time_sec=0.125,
        bs_frg_qubo_selected_total=3,
    )
    row = _row_dict_from_instance_result(
        result,
        Instance("case_a", ProblemVariant.PREMARSHALLING_CPMP, 1, 2, 2, [[2, 1]]),
        args,
    )

    assert row["bs_frg_qubo_enabled"] is True
    assert row["bs_frg_qubo_supplement_width"] == 4
    assert row["bs_frg_qubo_focus_n"] == -1
    assert row["bs_frg_qubo_calls"] == 2
    assert row["bs_frg_qubo_vars"] == 8
    assert row["bs_frg_qubo_time_sec"] == 0.125
    assert row["bs_frg_qubo_selected_total"] == 3


def test_deterministic_qubo_sampler_flag_is_serialized():
    from benchmark_compact_beam_qubo_n0 import CompactBeamResult, _row_dict_from_instance_result
    from container_reshuffle.core import Instance

    args = _compact_args(deterministic_qubo_sampler=True)
    result = CompactBeamResult(
        case="case_a",
        status="done",
        moves=[],
        runtime_sec=0.0,
        expanded=0,
        generated=0,
        visited=0,
        open_count=0,
        pruned_bound=0,
        pruned_business=0,
        dominated=0,
        peak_frontier=0,
        peak_candidates=0,
        depth_reached=0,
    )

    row = _row_dict_from_instance_result(
        result,
        Instance("case_a", ProblemVariant.PREMARSHALLING_CPMP, 1, 2, 2, [[2, 1]]),
        args,
    )

    assert row["deterministic_qubo_sampler"] is True


def test_wang2017_task_features_infer_move_and_penalize_stable_loss():
    import benchmark_compact_beam_qubo_n0 as bench
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 4
    max_height = 3
    parent_code, _parent_n = _encode([[5, 1, 4], [3, 2], [], []], stack_count, max_height)
    dirty_child, dirty_n = _encode([[5, 1], [3, 2], [4], []], stack_count, max_height)
    stable_child, stable_n = _encode([[5, 1, 4], [3], [2], []], stack_count, max_height)
    args = _compact_args(_stack_count=stack_count, _max_height=max_height)

    dirty_features = bench._wang2017_task_features_for_code(parent_code, dirty_child, (dirty_n, 1, parent_code), args)
    stable_features = bench._wang2017_task_features_for_code(parent_code, stable_child, (stable_n, 1, parent_code), args)

    assert dirty_features.source == 3
    assert dirty_features.target == 2
    assert dirty_features.moved_group == 4
    assert dirty_features.tier_protection == 0
    assert stable_features.tier_protection > dirty_features.tier_protection
    assert stable_features.stable_loss > dirty_features.stable_loss


def test_wang2017_task_supplement_prefers_dirty_top_continuation():
    import benchmark_compact_beam_qubo_n0 as bench
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 4
    max_height = 3
    parent_code, _parent_n = _encode([[5, 1, 4], [3, 2], [], []], stack_count, max_height)
    dirty_child, dirty_n = _encode([[5, 1], [3, 2], [4], []], stack_count, max_height)
    stable_child, stable_n = _encode([[5, 1, 4], [3], [2], []], stack_count, max_height)
    args = _compact_args(_stack_count=stack_count, _max_height=max_height)

    selected, metrics = bench._select_wang2017_task_supplement(
        [
            (stable_child, (stable_n, 1, parent_code)),
            (dirty_child, (dirty_n, 1, parent_code)),
        ],
        args,
        target=1,
    )

    assert selected == [dirty_child]
    assert metrics["selected_tier_protection_total"] == 0
    assert metrics["selected_stable_height_total"] >= 1


def test_solver_profile_scheduler_state_v5_wang2017_task_polish_extends_rescue_polish(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_wang2017_task_polish")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v5_wang2017_task_polish"
    assert args.rescue_enabled is True
    assert args.wang_pair_qubo_enabled is True
    assert args.post_polish_qubo is True
    assert args.wang2017_task_enabled is True
    assert args.wang2017_task_supplement_width > 0
    assert args.wang2017_task_focus_n < 0


def test_solver_profile_scheduler_state_v5_bs_frg_qubo_polish_extends_rescue_polish(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_bs_frg_qubo_polish")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v5_bs_frg_qubo_polish"
    assert args.rescue_enabled is True
    assert args.wang_pair_qubo_enabled is True
    assert args.post_polish_qubo is True
    assert args.bs_frg_qubo_enabled is True
    assert args.bs_frg_qubo_supplement_width > 0
    assert args.bs_frg_qubo_focus_n == 8
    assert args.bs_frg_qubo_max_vars <= 64


def test_solver_profile_scheduler_state_v5_wang_rescue_light_polish_is_bounded(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_wang_rescue_light_polish")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v5_wang_rescue_light_polish"
    assert args.rescue_enabled is True
    assert args.wang_pair_qubo_enabled is True
    assert args.wang_pair_qubo_focus_n == -1
    assert args.wang_pair_qubo_supplement_width == 8
    assert args.wang_pair_qubo_max_vars == 32
    assert args.wang_pair_qubo_select == 8
    assert args.rescue_qubo_supplement_width == 32
    assert args.rescue_max_qubo_vars == 64
    assert args.post_polish_qubo is True


def test_wang2017_task_gate_reason_requires_plateau_and_tight_slack():
    import benchmark_compact_beam_qubo_n0 as bench

    args = _compact_args(
        wang2017_task_enabled=True,
        wang2017_task_supplement_width=1,
        wang2017_task_focus_n=-1,
        wang2017_task_plateau_window=3,
        wang2017_task_plateau_delta=0,
        rescue_cap_slack=26,
        _best_n_history=[17, 17, 17],
        _layer_depth_slack=12,
    )

    assert bench._wang2017_task_gate_reason(args, best_n=17) == "active"
    assert bench._wang2017_task_layer_active(args, best_n=17)

    args._best_n_history = [19, 18, 17]
    assert bench._wang2017_task_gate_reason(args, best_n=17) == "not_plateau"

    args._best_n_history = [17, 17, 17]
    args._layer_depth_slack = 27
    assert bench._wang2017_task_gate_reason(args, best_n=17) == "slack_too_loose"

    args.wang2017_task_enabled = False
    assert bench._wang2017_task_gate_reason(args, best_n=17) == "disabled"


def test_solver_profile_scheduler_state_v5_wang2017_task_guarded_polish_is_conservative(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_wang2017_task_guarded_polish")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v5_wang2017_task_guarded_polish"
    assert args.rescue_enabled is True
    assert args.wang_pair_qubo_enabled is True
    assert args.post_polish_qubo is True
    assert args.wang2017_task_enabled is True
    assert args.wang2017_task_supplement_width == 1
    assert args.wang2017_task_n_window == 0
    assert args.wang2017_task_plateau_window == 7


def test_wang2017_task_selector_is_used_inside_enhanced_frontier(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as bench
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 4
    max_height = 3
    parent_code, _parent_n = _encode([[5, 1, 4], [3, 2], [], []], stack_count, max_height)
    chosen_code, chosen_n = _encode([[5, 1], [3, 2], [4], []], stack_count, max_height)
    other_code, other_n = _encode([[5, 1, 4], [3], [2], []], stack_count, max_height)
    ranked = [
        (other_code, (other_n, 1, parent_code)),
        (chosen_code, (chosen_n, 1, parent_code)),
    ]
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        use_qubo=True,
        qubo_min_candidates=99,
        wang2017_task_enabled=True,
        wang2017_task_supplement_width=1,
        wang2017_task_focus_n=-1,
        wang2017_task_n_window=3,
    )

    def fake_selector(rows, selector_args, target):
        assert rows
        assert target == 1
        return [chosen_code], {"selected_tier_protection_total": 0, "selected_stable_height_total": 1}

    monkeypatch.setattr(bench, "_select_wang2017_task_supplement", fake_selector)

    selected, selected_by_qubo, qcalls, qvars, qtime = bench._select_enhanced_qubo_frontier(
        ranked,
        candidate_count=len(ranked),
        args=args,
        beam_width=2,
        core_width=1,
        qubo_supplement_width=0,
    )

    assert chosen_code in selected
    assert chosen_code not in selected_by_qubo
    assert qcalls == 0
    assert qvars == 0
    assert qtime == 0.0
    assert args._wang2017_task_calls_done == 1
    assert args._wang2017_task_added_total == 1
    assert args._wang2017_task_selected_stable_height_total == 1


def test_wang2017_task_telemetry_is_serialized():
    from benchmark_compact_beam_qubo_n0 import CompactBeamResult, _row_dict_from_instance_result
    from container_reshuffle.core import Instance

    args = _compact_args()
    result = CompactBeamResult(
        case="case_a",
        status="done",
        moves=[],
        runtime_sec=0.0,
        expanded=0,
        generated=0,
        visited=0,
        open_count=0,
        pruned_bound=0,
        pruned_business=0,
        dominated=0,
        peak_frontier=0,
        peak_candidates=0,
        depth_reached=0,
        wang2017_task_calls=4,
        wang2017_task_added_total=8,
        wang2017_task_active_layers=4,
        wang2017_task_selected_tier_protection_total=2,
        wang2017_task_selected_stable_height_total=11,
    )
    row = _row_dict_from_instance_result(
        result,
        Instance("case_a", ProblemVariant.PREMARSHALLING_CPMP, 1, 2, 2, [[2, 1]]),
        args,
    )

    assert row["wang2017_task_calls"] == 4
    assert row["wang2017_task_added_total"] == 8
    assert row["wang2017_task_selected_stable_height_total"] == 11


def test_dlts_feature_row_contains_feasibility_and_selection_fields(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as bench
    from container_reshuffle.solvers.feasibility_signals import FeasibilitySignals

    code = b"candidate"
    args = _compact_args(_case_id="case_a")

    monkeypatch.setattr(
        bench,
        "_feasibility_signals_for_code",
        lambda row_code, row_args: FeasibilitySignals(4, 4, 2, 3, 1, 5, 0, 0.25),
    )

    row = bench._dlts_feature_row(
        case="case_a",
        layer=2,
        code=code,
        data=(3, 9, b"parent"),
        parent_rank=7,
        selected=True,
        args=args,
    )

    assert row["case"] == "case_a"
    assert row["layer"] == 2
    assert row["n_value"] == 3
    assert row["depth"] == 9
    assert row["parent_rank"] == 7
    assert row["selected"] == 1
    assert row["stable_prefix_total"] == 4
    assert row["temporary_slack_moves"] == 5
    assert row["on_final_path"] == 0


def test_make_compact_args_accepts_dlts_feature_output(tmp_path):
    from benchmark_compact_beam_qubo_n0 import make_compact_args

    output = tmp_path / "features.csv"
    args = make_compact_args(dlts_feature_output=output)

    assert args.dlts_feature_output == output
    assert "dlts_feature_output" in args._profile_explicit_keys


def test_open_runner_passes_beam_survivor_telemetry_output():
    from open_cpmp_benchmarks import _args_for_variant, build_parser

    outer = build_parser().parse_args(
        [
            "--variant",
            "qubo_scheduler_state_v5",
            "--beam-survivor-telemetry-output",
            "docs/experiments/beam_survivor.csv",
        ]
    )

    args = _args_for_variant(outer, "qubo_scheduler_state_v5")

    assert Path(args.beam_survivor_telemetry_output).as_posix().endswith("docs/experiments/beam_survivor.csv")


def test_open_runner_passes_deterministic_qubo_sampler_flag():
    from open_cpmp_benchmarks import _args_for_variant, build_parser

    outer = build_parser().parse_args(
        [
            "--variant",
            "qubo_scheduler_state_v5_wang_rescue_polish",
            "--deterministic-qubo-sampler",
        ]
    )

    args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang_rescue_polish")

    assert args.deterministic_qubo_sampler is True


def test_open_runner_passes_bs_frg_qubo_overrides():
    from open_cpmp_benchmarks import _args_for_variant, build_parser

    outer = build_parser().parse_args(
        [
            "--variant",
            "qubo_scheduler_state_v5_bs_frg_qubo_polish",
            "--bs-frg-qubo-supplement-width",
            "5",
            "--bs-frg-qubo-min-candidates",
            "12",
            "--bs-frg-qubo-max-vars",
            "32",
            "--bs-frg-qubo-same-parent-penalty",
            "0.7",
        ]
    )

    args = _args_for_variant(outer, "qubo_scheduler_state_v5_bs_frg_qubo_polish")

    assert args.bs_frg_qubo_supplement_width == 5
    assert args.bs_frg_qubo_min_candidates == 12
    assert args.bs_frg_qubo_max_vars == 32
    assert args.bs_frg_qubo_same_parent_penalty == 0.7


def test_open_runner_passes_frontier_width_overrides():
    from open_cpmp_benchmarks import _args_for_variant, build_parser

    outer = build_parser().parse_args(
        [
            "--variant",
            "qubo_scheduler_state_v5_wang_rescue",
            "--beam-width",
            "1000",
            "--core-width",
            "900",
            "--parent-core-cap",
            "2",
        ]
    )

    args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang_rescue")

    assert args.beam_width == 1000
    assert args.core_width == 900
    assert args.parent_core_cap == 2


def test_open_runner_passes_survivor_parent_cap_overrides():
    from open_cpmp_benchmarks import _args_for_variant, build_parser

    outer = build_parser().parse_args(
        [
            "--variant",
            "qubo_scheduler_state_v5_wang_rescue",
            "--survivor-parent-cap-enabled",
            "--survivor-parent-cap-trigger-ratio",
            "0.72",
            "--survivor-parent-cap-release-ratio",
            "0.50",
            "--survivor-parent-cap-value",
            "2",
            "--survivor-parent-cap-window",
            "3",
            "--survivor-parent-cap-min-n",
            "9",
        ]
    )

    args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang_rescue")

    assert args.survivor_parent_cap_enabled is True
    assert args.survivor_parent_cap_trigger_ratio == 0.72
    assert args.survivor_parent_cap_release_ratio == 0.50
    assert args.survivor_parent_cap_value == 2
    assert args.survivor_parent_cap_window == 3
    assert args.survivor_parent_cap_min_n == 9


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


def test_solver_profile_post_polish_v1_strengthens_suffix_polish_branch(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="post_polish_v1")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.selection_policy == "enhanced"
    assert args.beam_width == 1200
    assert args.core_width == 1080
    assert args.max_qubo_vars == 64
    assert args.max_qubo_calls == 32
    assert args.qubo_supplement_width == 24
    assert args.qubo_next_n_weight == 0.0
    assert args.post_polish_passes > 2
    assert args.post_polish_suffix_max_candidates > 48
    assert args.post_polish_suffix_expanded > 60000


def test_solver_profile_frontier_quality_v1_strengthens_qubo_quality_branch(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="frontier_quality_v1")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.selection_policy == "enhanced"
    assert args.beam_width == 1200
    assert args.core_width == 1080
    assert args.max_qubo_vars > 64
    assert args.max_qubo_calls > 32
    assert args.qubo_supplement_width > 24
    assert args.qubo_next_n_weight > 0
    assert args.qubo_parent_weight > 0.15
    assert args.post_polish_passes == 2
    assert args.post_polish_suffix_max_candidates == 48


def test_target_guided_rank_prefers_better_target_match_at_same_n():
    from benchmark_compact_beam_qubo_n0 import (
        _candidate_key_func,
        _compact_target_layouts,
    )
    from run_business_exact_target_search_sqlite import _encode

    instance = Instance(
        "target_rank",
        ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=3,
        max_height=3,
        container_count=5,
        stacks=[[5, 1], [4, 2], [3]],
    )
    targets = _compact_target_layouts(instance)
    better, better_n = _encode([[5, 4], [3, 2], [1]], instance.stack_count, instance.max_height)
    worse, worse_n = _encode([[5, 1], [4, 2], [3]], instance.stack_count, instance.max_height)
    assert better_n == worse_n
    args = _compact_args(
        priority="target_guided",
        target_guided_weight=1.0,
        _layout_target_groups=targets,
        _stack_count=instance.stack_count,
        _max_height=instance.max_height,
    )

    ranked = sorted(
        {
            worse: (worse_n, 3, b"p"),
            better: (better_n, 3, b"p"),
        }.items(),
        key=_candidate_key_func(args),
    )

    assert ranked[0][0] == better


def test_target_guided_rank_waits_until_focus_n_gate():
    from benchmark_compact_beam_qubo_n0 import (
        _candidate_key_func,
        _compact_target_layouts,
    )
    from run_business_exact_target_search_sqlite import _encode

    instance = Instance(
        "target_rank_gate",
        ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=3,
        max_height=3,
        container_count=5,
        stacks=[[5, 1], [4, 2], [3]],
    )
    targets = _compact_target_layouts(instance)
    better, better_n = _encode([[5, 4], [3, 2], [1]], instance.stack_count, instance.max_height)
    worse, worse_n = _encode([[5, 1], [4, 2], [3]], instance.stack_count, instance.max_height)
    args = _compact_args(
        priority="target_guided",
        target_guided_weight=1.0,
        target_guided_focus_n=2,
        _layout_target_groups=targets,
        _stack_count=instance.stack_count,
        _max_height=instance.max_height,
        _layer_best_n=9,
    )
    plain_args = _compact_args(priority="n_then_depth")

    better_item = (better, (better_n, 3, b"p"))
    worse_item = (worse, (worse_n, 3, b"p"))

    assert _candidate_key_func(args)(better_item) == _candidate_key_func(plain_args)(better_item)
    assert _candidate_key_func(args)(worse_item) == _candidate_key_func(plain_args)(worse_item)


def test_solver_profile_target_guided_v1_uses_protected_supplement(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="target_guided_v1")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.priority == "n_then_depth"
    assert args.target_guided_weight > 0
    assert args.target_guided_focus_n == 8
    assert args.target_guided_supplement_width > 0
    assert args.selection_policy == "enhanced"
    assert args.use_qubo is True


def test_target_guided_supplement_adds_target_close_candidate():
    from benchmark_compact_beam_qubo_n0 import (
        _compact_target_layouts,
        _select_enhanced_qubo_frontier,
    )
    from run_business_exact_target_search_sqlite import _encode

    instance = Instance(
        "target_supplement",
        ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=3,
        max_height=3,
        container_count=5,
        stacks=[[5, 1], [4, 2], [3]],
    )
    targets = _compact_target_layouts(instance)
    core_code, core_n = _encode([[5, 1], [4, 2], [3]], instance.stack_count, instance.max_height)
    target_code, target_n = _encode([[5, 4], [3, 2], [1]], instance.stack_count, instance.max_height)
    filler_code, filler_n = _encode([[5, 3], [4, 1], [2]], instance.stack_count, instance.max_height)
    ranked = [
        (core_code, (core_n, 3, b"p0")),
        (filler_code, (filler_n, 3, b"p1")),
        (target_code, (target_n, 3, b"p2")),
    ]
    args = _compact_args(
        beam_width=2,
        core_width=1,
        parent_core_cap=0,
        use_qubo=False,
        target_guided_weight=1.0,
        target_guided_focus_n=8,
        target_guided_supplement_width=1,
        _layout_target_groups=targets,
        _stack_count=instance.stack_count,
        _max_height=instance.max_height,
        _layer_best_n=0,
    )

    selected, _selected_by_qubo, _qcalls, _qvars, _qtime = _select_enhanced_qubo_frontier(
        ranked,
        len(ranked),
        args,
    )

    assert selected == [core_code, target_code]


def test_wang_probe_fixed_prefix_allows_repeated_priority_groups():
    from benchmark_compact_beam_qubo_n0 import _compact_wang_fixed_prefix_score
    from run_business_exact_target_search_sqlite import _encode

    code, _n = _encode([[5, 5, 3], [4, 2, 4], []], 3, 4)

    assert _compact_wang_fixed_prefix_score(code, stack_count=3, max_height=4) == 5


def test_wang_stack_feature_cache_reuses_shared_chunks():
    from benchmark_compact_beam_qubo_n0 import (
        _compact_wang_fix_pair_score,
        _compact_wang_fixed_prefix_score,
        _compact_wang_stack_features,
    )
    from run_business_exact_target_search_sqlite import _encode

    _compact_wang_fix_pair_score.cache_clear()
    _compact_wang_fixed_prefix_score.cache_clear()
    _compact_wang_stack_features.cache_clear()

    first, _first_n = _encode([[5, 4], [3, 1, 2], []], 3, 3)
    second, _second_n = _encode([[5, 4], [3, 2, 1], []], 3, 3)

    assert _compact_wang_fixed_prefix_score(first, 3, 3) >= 0
    assert _compact_wang_fix_pair_score(first, 3, 3) >= 0
    before = _compact_wang_stack_features.cache_info()

    assert _compact_wang_fixed_prefix_score(second, 3, 3) >= 0
    assert _compact_wang_fix_pair_score(second, 3, 3) >= 0
    after = _compact_wang_stack_features.cache_info()

    assert after.hits > before.hits


def test_solver_profile_wang_probe_v1_uses_protected_supplement(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="wang_probe_v1")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.priority == "n_then_depth"
    assert args.selection_policy == "enhanced"
    assert args.use_qubo is True
    assert args.wang_probe_supplement_width > 0
    assert args.wang_probe_focus_n >= 8
    assert args.wang_probe_pair_weight > 0


def test_wang_probe_supplement_adds_fix_pair_candidate():
    from benchmark_compact_beam_qubo_n0 import _select_enhanced_qubo_frontier
    from run_business_exact_target_search_sqlite import _encode

    core_code, core_n = _encode([[5, 4, 2], [1, 3], []], 3, 3)
    filler_code, filler_n = _encode([[5, 4, 3], [1, 2], []], 3, 3)
    fixed_prefix_code, fixed_prefix_n = _encode([[5, 4], [3, 1, 2], []], 3, 3)
    assert core_n == filler_n == fixed_prefix_n
    ranked = [
        (core_code, (core_n, 3, b"p0")),
        (filler_code, (filler_n, 3, b"p1")),
        (fixed_prefix_code, (fixed_prefix_n, 3, b"p2")),
    ]
    args = _compact_args(
        beam_width=2,
        core_width=1,
        parent_core_cap=0,
        use_qubo=False,
        wang_probe_supplement_width=1,
        wang_probe_focus_n=8,
        wang_probe_n_window=0,
        wang_probe_fixed_weight=0.0,
        wang_probe_pair_weight=1.0,
        wang_probe_target_weight=0.0,
        _stack_count=3,
        _max_height=3,
        _layer_best_n=0,
    )

    selected, _selected_by_qubo, _qcalls, _qvars, _qtime = _select_enhanced_qubo_frontier(
        ranked,
        len(ranked),
        args,
    )

    assert selected == [core_code, fixed_prefix_code]


def test_wang_probe_supplement_waits_for_plateau_gate():
    from benchmark_compact_beam_qubo_n0 import _select_enhanced_qubo_frontier
    from run_business_exact_target_search_sqlite import _encode

    core_code, core_n = _encode([[5, 4, 2], [1, 3], []], 3, 3)
    filler_code, filler_n = _encode([[5, 4, 3], [1, 2], []], 3, 3)
    probe_code, probe_n = _encode([[5, 4], [3, 1, 2], []], 3, 3)
    ranked = [
        (core_code, (core_n, 3, b"p0")),
        (filler_code, (filler_n, 3, b"p1")),
        (probe_code, (probe_n, 3, b"p2")),
    ]
    args = _compact_args(
        beam_width=2,
        core_width=1,
        parent_core_cap=0,
        use_qubo=False,
        wang_probe_supplement_width=1,
        wang_probe_focus_n=8,
        wang_probe_n_window=0,
        wang_probe_fixed_weight=0.0,
        wang_probe_pair_weight=1.0,
        wang_probe_target_weight=0.0,
        wang_probe_plateau_window=3,
        wang_probe_plateau_delta=0,
        _best_n_history=[5, 4],
        _stack_count=3,
        _max_height=3,
    )

    selected, *_ = _select_enhanced_qubo_frontier(ranked, len(ranked), args)
    assert selected == [core_code, filler_code]

    args._best_n_history = [4, 4, 4]
    selected, *_ = _select_enhanced_qubo_frontier(ranked, len(ranked), args)
    assert selected == [core_code, probe_code]


def test_wang_rollout_successor_uses_legal_top_move_and_parent_depth():
    from benchmark_compact_beam_qubo_n0 import _wang_rollout_successor
    from run_business_exact_target_search_sqlite import _encode

    code, n_value = _encode([[5, 4], [3, 1, 2], []], 3, 3)
    args = _compact_args(
        _stack_count=3,
        _max_height=3,
        wang_rollout_enabled=True,
        wang_rollout_max_steps=1,
    )

    child = _wang_rollout_successor(code, n_value, depth=4, args=args)

    assert child is not None
    child_code, child_data = child
    assert child_data[1] == 5
    assert child_data[2] == code
    assert child_code != code


def test_wang_rollout_successor_can_use_baby_lookahead_strategy(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact
    from run_business_exact_target_search_sqlite import _encode

    origin, origin_n = _encode([[5, 4], [3, 1, 2], []], 3, 3)
    lookahead_child, lookahead_n = _encode([[5, 4], [3, 1], [2]], 3, 3)
    args = _compact_args(
        _stack_count=3,
        _max_height=3,
        wang_rollout_strategy="baby_lookahead",
    )

    def fake_baby_lookahead(code, n_value, depth, helper_args):
        assert code == origin
        assert n_value == origin_n
        assert depth == 4
        assert helper_args is args
        return lookahead_child, (lookahead_n, 5, origin)

    monkeypatch.setattr(compact, "_wang_baby_lookahead_successor", fake_baby_lookahead)

    child = compact._wang_rollout_successor(origin, origin_n, depth=4, args=args)

    assert child == (lookahead_child, (lookahead_n, 5, origin))


def test_wang_baby_lookahead_successor_returns_one_legal_child():
    from benchmark_compact_beam_qubo_n0 import _wang_baby_lookahead_successor
    from run_business_exact_target_search_sqlite import _encode

    code, n_value = _encode([[5, 4], [3, 1, 2], []], 3, 3)
    args = _compact_args(
        _stack_count=3,
        _max_height=3,
        wang_rollout_candidate_limit=64,
    )

    child = _wang_baby_lookahead_successor(code, n_value, depth=4, args=args)

    assert child is not None
    child_code, child_data = child
    assert child_data[1] == 5
    assert child_data[2] == code
    assert child_code != code


def test_wang_rollout_frontier_skips_known_shallower_code(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact
    from run_business_exact_target_search_sqlite import _encode

    origin, origin_n = _encode([[3, 1], [2], []], 3, 3)
    known, known_n = _encode([[3], [2, 1], []], 3, 3)
    candidates = {origin: (origin_n, 4, b"root")}
    args = _compact_args(
        _stack_count=3,
        _max_height=3,
        wang_rollout_enabled=True,
        wang_rollout_width=1,
        wang_rollout_focus_n=99,
        wang_rollout_plateau_window=0,
    )
    monkeypatch.setattr(compact, "_wang_rollout_successor", lambda *_args: (known, (known_n, 5, origin)))

    added = compact._append_wang_rollout_frontier([origin], candidates, args, depth_cap=20, best_depth={known: 2})

    assert added == set()
    assert known not in candidates


def test_wang_rollout_candidate_injection_adds_child_before_selection(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact
    from run_business_exact_target_search_sqlite import _encode

    origin, origin_n = _encode([[3, 1], [2], []], 3, 3)
    child, child_n = _encode([[3], [2, 1], []], 3, 3)
    candidates = {origin: (origin_n, 4, b"root")}
    frontier = [(origin, origin_n, 4)]
    args = _compact_args(
        _stack_count=3,
        _max_height=3,
        wang_rollout_enabled=True,
        wang_rollout_inject_candidates=True,
        wang_rollout_width=1,
        wang_rollout_focus_n=99,
        wang_rollout_plateau_window=0,
    )
    monkeypatch.setattr(compact, "_wang_rollout_successor", lambda *_args: (child, (child_n, 5, origin)))

    added = compact._inject_wang_rollout_candidates(frontier, candidates, args, depth_cap=20, best_depth={})

    assert added == {child}
    assert candidates[child] == (child_n, 5, origin)


def test_wang_rollout_candidate_injection_skips_known_shallower_code(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact
    from run_business_exact_target_search_sqlite import _encode

    origin, origin_n = _encode([[3, 1], [2], []], 3, 3)
    known, known_n = _encode([[3], [2, 1], []], 3, 3)
    candidates = {origin: (origin_n, 4, b"root")}
    frontier = [(origin, origin_n, 4)]
    args = _compact_args(
        _stack_count=3,
        _max_height=3,
        wang_rollout_enabled=True,
        wang_rollout_inject_candidates=True,
        wang_rollout_width=1,
        wang_rollout_focus_n=99,
        wang_rollout_plateau_window=0,
    )
    monkeypatch.setattr(compact, "_wang_rollout_successor", lambda *_args: (known, (known_n, 5, origin)))

    added = compact._inject_wang_rollout_candidates(
        frontier,
        candidates,
        args,
        depth_cap=20,
        best_depth={known: 2},
    )

    assert added == set()
    assert known not in candidates


def test_wang_rollout_candidate_injection_marks_existing_child(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact
    from run_business_exact_target_search_sqlite import _encode

    origin, origin_n = _encode([[3, 1], [2], []], 3, 3)
    child, child_n = _encode([[3], [2, 1], []], 3, 3)
    candidates = {child: (child_n, 5, origin)}
    frontier = [(origin, origin_n, 4)]
    args = _compact_args(
        _stack_count=3,
        _max_height=3,
        wang_rollout_enabled=True,
        wang_rollout_inject_candidates=True,
        wang_rollout_width=1,
        wang_rollout_focus_n=99,
        wang_rollout_plateau_window=0,
    )
    monkeypatch.setattr(compact, "_wang_rollout_successor", lambda *_args: (child, (child_n, 5, origin)))

    marked = compact._inject_wang_rollout_candidates(frontier, candidates, args, depth_cap=20, best_depth={})

    assert marked == {child}
    assert candidates == {child: (child_n, 5, origin)}


def test_wang_rollout_candidate_injection_disables_post_selection_append(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact
    from run_business_exact_target_search_sqlite import _encode

    origin, origin_n = _encode([[3, 1], [2], []], 3, 3)
    child, child_n = _encode([[3], [2, 1], []], 3, 3)
    candidates = {origin: (origin_n, 4, b"root")}
    args = _compact_args(
        _stack_count=3,
        _max_height=3,
        wang_rollout_enabled=True,
        wang_rollout_inject_candidates=True,
        wang_rollout_width=1,
        wang_rollout_focus_n=99,
        wang_rollout_plateau_window=0,
    )
    monkeypatch.setattr(compact, "_wang_rollout_successor", lambda *_args: (child, (child_n, 5, origin)))

    added = compact._append_wang_rollout_frontier([origin], candidates, args, depth_cap=20, best_depth={})

    assert added == set()
    assert child not in candidates


def test_unified_wang_qubo_energy_rewards_rollout_candidate():
    from benchmark_compact_beam_qubo_n0 import _unified_wang_qubo_linear_energy
    from run_business_exact_target_search_sqlite import _encode

    plain, plain_n = _encode([[3, 1], [2], []], 3, 3)
    marked, marked_n = _encode([[3], [2, 1], []], 3, 3)
    entries = [
        (plain, (plain_n, 5, b"p1")),
        (marked, (marked_n, 5, b"p2")),
    ]
    args = _compact_args(
        _stack_count=3,
        _max_height=3,
        _wang_rollout_candidate_codes={marked},
        unified_wang_rollout_bonus=25.0,
    )

    plain_energy = _unified_wang_qubo_linear_energy(plain, entries[0][1], entries, args)
    marked_energy = _unified_wang_qubo_linear_energy(marked, entries[1][1], entries, args)

    assert marked_energy < plain_energy


def test_wang_pair_qubo_supplement_selects_high_fix_pair_candidate():
    from benchmark_compact_beam_qubo_n0 import _select_wang_pair_qubo_supplement
    from run_business_exact_target_search_sqlite import _encode

    filler_code, filler_n = _encode([[5, 4, 3], [1, 2], []], 3, 3)
    probe_code, probe_n = _encode([[5, 4], [3, 1, 2], []], 3, 3)
    ranked = [
        (filler_code, (filler_n, 3, b"p1")),
        (probe_code, (probe_n, 3, b"p2")),
    ]
    args = _compact_args(
        seed=37,
        qubo_penalty_a=30.0,
        qubo_reads=2,
        qubo_time_limit=0.001,
        qubo_iterations=20,
        wang_pair_qubo_max_vars=8,
        wang_pair_qubo_select=1,
        wang_pair_qubo_same_parent_penalty=0.25,
        _stack_count=3,
        _max_height=3,
    )

    selected, qvars, _qtime = _select_wang_pair_qubo_supplement(ranked, args, target=1)

    assert selected == [probe_code]
    assert qvars == 2


def test_frg_candidate_supplement_prefers_fill_transition_score():
    from benchmark_compact_beam_qubo_n0 import _select_frg_candidate_supplement
    from run_business_exact_target_search_sqlite import _encode

    parent_code, _parent_n = _encode([[6, 5], [4, 7, 2], [3], []], 4, 3)
    reduce_code, _reduce_n = _encode([[6], [4, 7, 2], [3, 5], []], 4, 3)
    fill_code, _fill_n = _encode([[6, 5, 2], [4, 7], [3], []], 4, 3)
    ranked = [
        (reduce_code, (3, 10, parent_code)),
        (fill_code, (3, 10, parent_code)),
    ]
    args = _compact_args(
        _stack_count=4,
        _max_height=3,
        frg_candidate_supplement_enabled=True,
        frg_candidate_focus_n=-1,
        frg_candidate_n_window=2,
        frg_candidate_max_ranked=8,
    )

    selected = _select_frg_candidate_supplement(ranked, args, target=1)

    assert selected == [fill_code]


def test_frg_and_dlts_supplements_are_used_inside_enhanced_frontier(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as bench
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 4
    max_height = 3
    core_code, core_n = _encode([[6, 5], [4, 7, 2], [3], []], stack_count, max_height)
    frg_code, frg_n = _encode([[6, 5, 2], [4, 7], [3], []], stack_count, max_height)
    dlts_code, dlts_n = _encode([[6, 5], [4, 7], [3, 2], []], stack_count, max_height)
    parent_code = core_code
    ranked = [
        (core_code, (core_n, 8, parent_code)),
        (frg_code, (frg_n, 9, parent_code)),
        (dlts_code, (dlts_n, 9, parent_code)),
    ]
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        use_qubo=True,
        qubo_min_candidates=99,
        frg_candidate_supplement_enabled=True,
        frg_candidate_supplement_width=1,
        dlts_value_supplement_enabled=True,
        dlts_value_supplement_width=1,
    )

    def fake_frg(rows, selector_args, target):
        assert rows
        assert selector_args is args
        assert target == 1
        return [frg_code]

    def fake_dlts(rows, selector_args, target):
        assert rows
        assert selector_args is args
        assert target == 1
        return [dlts_code]

    monkeypatch.setattr(bench, "_select_frg_candidate_supplement", fake_frg)
    monkeypatch.setattr(bench, "_select_dlts_value_supplement", fake_dlts)

    selected, _selected_by_qubo, qcalls, qvars, qtime = bench._select_enhanced_qubo_frontier(
        ranked,
        candidate_count=len(ranked),
        args=args,
        beam_width=3,
        core_width=1,
        qubo_supplement_width=0,
    )

    assert selected == [core_code, frg_code, dlts_code]
    assert qcalls == 0
    assert qvars == 0
    assert qtime == 0.0
    assert args._frg_candidate_selected_total == 1
    assert args._dlts_value_selected_total == 1


def test_frg_compound_escape_gate_requires_plateau_tight_slack_and_pressure():
    import benchmark_compact_beam_qubo_n0 as bench

    args = _compact_args(
        frg_compound_escape_enabled=True,
        frg_compound_escape_focus_n=-1,
        frg_compound_escape_plateau_window=4,
        frg_compound_escape_plateau_delta=0,
        frg_compound_escape_cap_slack=12,
        frg_compound_escape_min_candidates=5000,
        _best_n_history=[16, 16, 16, 16],
        _layer_candidate_count=6000,
        _layer_depth_slack=12,
    )

    assert bench._frg_compound_escape_active(args, best_n=16)

    args._layer_depth_slack = 13
    assert not bench._frg_compound_escape_active(args, best_n=16)

    args._layer_depth_slack = 12
    args._best_n_history = [18, 17, 16, 16]
    assert not bench._frg_compound_escape_active(args, best_n=16)

    args._best_n_history = [16, 16, 16, 16]
    args._layer_candidate_count = 4999
    assert not bench._frg_compound_escape_active(args, best_n=16)


def test_frg_compound_guard_requires_small_residual_pressure_and_parent_diversity():
    import benchmark_compact_beam_qubo_n0 as bench

    parent_a = b"a" * 25
    parent_b = b"b" * 25
    parent_c = b"c" * 25
    origins = [b"1" * 25, b"2" * 25, b"3" * 25, b"4" * 25]
    candidates = {
        origins[0]: (16, 8, parent_a),
        origins[1]: (16, 9, parent_b),
        origins[2]: (17, 8, parent_c),
        origins[3]: (17, 9, parent_c),
    }
    args = _compact_args(
        frg_compound_guard_enabled=True,
        frg_compound_guard_max_stack_count=5,
        frg_compound_guard_max_initial_n=18,
        frg_compound_guard_min_best_n=12,
        frg_compound_guard_max_best_n=18,
        frg_compound_guard_min_candidates=1000,
        frg_compound_guard_min_parent_count=3,
        frg_compound_guard_max_parent_share=0.75,
        _stack_count=5,
        _initial_n=15,
        _layer_candidate_count=5000,
    )

    assert bench._frg_compound_guard_accepts(args, best_n=16, origins=origins, candidates=candidates)

    args._stack_count = 7
    assert not bench._frg_compound_guard_accepts(args, best_n=16, origins=origins, candidates=candidates)

    args._stack_count = 5
    assert not bench._frg_compound_guard_accepts(args, best_n=20, origins=origins, candidates=candidates)

    args._initial_n = 20
    assert not bench._frg_compound_guard_accepts(args, best_n=16, origins=origins, candidates=candidates)

    args._initial_n = 15
    args._layer_candidate_count = 999
    assert not bench._frg_compound_guard_accepts(args, best_n=16, origins=origins, candidates=candidates)

    args._layer_candidate_count = 5000
    collapsed = {code: (16, 8, parent_a) for code in origins}
    assert not bench._frg_compound_guard_accepts(args, best_n=16, origins=origins, candidates=collapsed)


def test_append_frg_compound_frontier_adds_best_legal_successor():
    import benchmark_compact_beam_qubo_n0 as bench
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 4
    max_height = 3
    parent_code, _parent_n = _encode([[6, 5], [4, 7, 2], [3], []], stack_count, max_height)
    expected_child, _expected_n = _encode([[2], [3], [4, 7], [6, 5]], stack_count, max_height)
    selected_codes = [parent_code]
    candidates = {parent_code: (2, 10, b"root")}
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        frg_compound_escape_enabled=True,
        frg_compound_escape_width=1,
        frg_compound_escape_focus_n=-1,
        frg_compound_escape_plateau_window=4,
        frg_compound_escape_plateau_delta=0,
        frg_compound_escape_cap_slack=12,
        frg_compound_escape_min_candidates=5000,
        _best_n_history=[2, 2, 2, 2],
        _layer_candidate_count=6000,
        _layer_depth_slack=12,
    )

    added = bench._append_frg_compound_frontier(
        selected_codes,
        candidates,
        args,
        depth_cap=99,
        best_depth={},
    )

    assert added == {expected_child}
    assert selected_codes == [parent_code, expected_child]
    assert candidates[expected_child] == (1, 11, parent_code)
    assert args._frg_compound_appended_total == 1


def test_unified_wang_selector_is_used_inside_enhanced_frontier(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as bench
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    chosen_code, chosen_n = _encode([[5, 4], [3, 1, 2], []], stack_count, max_height)
    other_code, other_n = _encode([[4, 5], [3, 1, 2], []], stack_count, max_height)
    ranked = [(other_code, (other_n, 8, b"p0")), (chosen_code, (chosen_n, 8, b"p1"))]
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        use_qubo=True,
        qubo_min_candidates=1,
        max_qubo_calls=0,
        unified_wang_qubo_enabled=True,
        unified_wang_qubo_supplement_width=1,
        unified_wang_qubo_focus_n=-1,
        unified_wang_qubo_n_window=1,
        unified_wang_qubo_select=1,
    )

    def fake_selector(rows, selector_args, target):
        assert rows
        assert selector_args is args
        assert target == 1
        return [chosen_code], 3, 0.002, {"fixed_prefix_selected": 4, "fix_pair_selected": 2}

    monkeypatch.setattr(bench, "_select_unified_wang_qubo_supplement", fake_selector)

    selected, selected_by_qubo, qcalls, qvars, qtime = bench._select_enhanced_qubo_frontier(
        ranked,
        candidate_count=len(ranked),
        args=args,
        beam_width=2,
        core_width=1,
        qubo_supplement_width=0,
    )

    assert chosen_code in selected
    assert chosen_code in selected_by_qubo
    assert qcalls == 1
    assert qvars == 3
    assert qtime == 0.002
    assert args._unified_wang_qubo_calls_done == 1
    assert args._unified_wang_qubo_vars == 3
    assert args._unified_wang_qubo_time == 0.002
    assert args._unified_wang_fixed_prefix_selected == 4
    assert args._unified_wang_fix_pair_selected == 2


def test_unified_wang_selector_accepts_none_focus_default(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as bench
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    chosen_code, chosen_n = _encode([[5, 4], [3, 1, 2], []], stack_count, max_height)
    other_code, other_n = _encode([[4, 5], [3, 1, 2], []], stack_count, max_height)
    ranked = [(other_code, (other_n, 8, b"p0")), (chosen_code, (chosen_n, 8, b"p1"))]
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        use_qubo=True,
        qubo_min_candidates=1,
        max_qubo_calls=0,
        unified_wang_qubo_enabled=True,
        unified_wang_qubo_supplement_width=1,
        unified_wang_qubo_focus_n=None,
        unified_wang_qubo_n_window=1,
        unified_wang_qubo_select=1,
    )

    calls = []

    def fake_selector(rows, selector_args, target):
        calls.append((rows, selector_args, target))
        return [chosen_code], 2, 0.001, {"fixed_prefix_selected": 1, "fix_pair_selected": 1}

    monkeypatch.setattr(bench, "_select_unified_wang_qubo_supplement", fake_selector)

    selected, selected_by_qubo, qcalls, qvars, qtime = bench._select_enhanced_qubo_frontier(
        ranked,
        candidate_count=len(ranked),
        args=args,
        beam_width=2,
        core_width=1,
        qubo_supplement_width=0,
    )

    assert calls
    assert calls[0][1] is args
    assert calls[0][2] == 1
    assert chosen_code in selected
    assert chosen_code in selected_by_qubo
    assert qcalls == 1
    assert qvars == 2
    assert qtime == 0.001


def test_unified_wang_qubo_supplement_selects_best_energy_candidate():
    from benchmark_compact_beam_qubo_n0 import _select_unified_wang_qubo_supplement
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    preferred_code, preferred_n = _encode([[5, 4], [3, 1, 2], []], stack_count, max_height)
    other_code, other_n = _encode([[4, 5], [3, 1, 2], []], stack_count, max_height)
    rows = [
        (other_code, (other_n, 8, b"p0")),
        (preferred_code, (preferred_n, 8, b"p1")),
    ]
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        unified_wang_qubo_max_vars=8,
        unified_wang_qubo_select=1,
        unified_wang_n_weight=1000.0,
        unified_wang_depth_weight=1.0,
        unified_wang_fixed_prefix_weight=30.0,
        unified_wang_fix_pair_weight=10.0,
        unified_wang_target_gap_weight=1.0,
        unified_wang_same_parent_penalty=0.25,
        unified_wang_signature_penalty=0.0,
        unified_wang_complement_bonus=0.0,
        qubo_penalty_a=30.0,
        qubo_reads=2,
        qubo_iterations=20,
        qubo_time_limit=0.001,
        seed=37,
    )

    selected, qvars, qtime, metrics = _select_unified_wang_qubo_supplement(rows, args, target=1)

    assert selected == [preferred_code]
    assert qvars == 2
    assert qtime >= 0.0
    assert metrics["fixed_prefix_selected"] >= 2
    assert metrics["fix_pair_selected"] >= 1


def test_unified_wang_qubo_pair_same_parent_penalty_discourages_duplicate_parent():
    from benchmark_compact_beam_qubo_n0 import _select_unified_wang_qubo_supplement
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    same_parent_a, same_parent_a_n = _encode([[5], [4, 3], [2, 1]], stack_count, max_height)
    same_parent_b, same_parent_b_n = _encode([[5, 4], [3, 2], [1]], stack_count, max_height)
    other_parent, other_parent_n = _encode([[5, 4], [3, 1, 2], []], stack_count, max_height)
    rows = [
        (same_parent_a, (same_parent_a_n, 8, b"same")),
        (same_parent_b, (same_parent_b_n, 8, b"same")),
        (other_parent, (other_parent_n, 8, b"other")),
    ]
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        unified_wang_qubo_max_vars=8,
        unified_wang_n_weight=1000.0,
        unified_wang_depth_weight=1.0,
        unified_wang_fixed_prefix_weight=30.0,
        unified_wang_fix_pair_weight=10.0,
        unified_wang_target_gap_weight=1.0,
        unified_wang_same_parent_penalty=0.0,
        unified_wang_pair_same_parent_penalty=3.0,
        unified_wang_signature_penalty=0.0,
        unified_wang_complement_bonus=0.0,
        qubo_penalty_a=30.0,
        qubo_reads=2,
        qubo_iterations=20,
        qubo_time_limit=0.001,
        seed=37,
    )

    selected, qvars, _qtime, _metrics = _select_unified_wang_qubo_supplement(rows, args, target=2)

    parent_by_code = {code: data[2] for code, data in rows}
    assert qvars == 3
    assert other_parent in selected
    assert {parent_by_code[code] for code in selected} == {b"same", b"other"}


def test_unified_wang_qubo_features_capture_prefix_pair_and_diversity():
    from benchmark_compact_beam_qubo_n0 import (
        _compact_wang_fix_pair_score,
        _compact_wang_fixed_prefix_score,
        _unified_wang_qubo_features,
    )
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    good_code, good_n = _encode([[5, 4], [3, 1, 2], []], stack_count, max_height)
    peer_code, peer_n = _encode([[5], [4, 3], [2, 1]], stack_count, max_height)
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
    )

    features = _unified_wang_qubo_features(
        good_code,
        (good_n, 6, b"p0"),
        [
            (good_code, (good_n, 6, b"p0")),
            (good_code, (good_n, 8, b"p0")),
            (peer_code, (peer_n, 7, b"p0")),
        ],
        args,
    )

    expected_prefix = _compact_wang_fixed_prefix_score(good_code, stack_count, max_height)
    expected_pair = _compact_wang_fix_pair_score(good_code, stack_count, max_height)
    assert expected_prefix == 4
    assert expected_pair == 2
    assert features.fixed_prefix == expected_prefix
    assert features.fix_pair == expected_pair
    assert features.target_gap >= 0
    assert features.same_parent_count == 1
    assert features.depth == 6
    assert features.n_value == good_n


def test_unified_wang_qubo_energy_defaults_order_by_n_then_depth():
    from benchmark_compact_beam_qubo_n0 import _unified_wang_qubo_linear_energy
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    higher_n_code, higher_n = _encode([[5, 4], [3, 1, 2], []], stack_count, max_height)
    lower_n_code, lower_n = _encode([[5], [4, 3], [2, 1]], stack_count, max_height)
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
    )
    entries = [(higher_n_code, (higher_n, 6, b"pa")), (lower_n_code, (lower_n, 7, b"pb"))]

    higher_n_energy = _unified_wang_qubo_linear_energy(
        higher_n_code,
        (higher_n, 6, b"pa"),
        entries,
        args,
    )
    lower_n_energy = _unified_wang_qubo_linear_energy(
        lower_n_code,
        (lower_n, 7, b"pb"),
        entries,
        args,
    )

    assert higher_n_energy == higher_n * 1000.0 + 6.0
    assert lower_n_energy == lower_n * 1000.0 + 7.0
    assert lower_n_energy < higher_n_energy


def test_unified_wang_qubo_energy_treats_none_weights_as_defaults():
    from benchmark_compact_beam_qubo_n0 import _unified_wang_qubo_linear_energy
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    code, n_value = _encode([[5, 4], [3, 1, 2], []], stack_count, max_height)
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        unified_wang_n_weight=None,
        unified_wang_depth_weight=None,
        unified_wang_fixed_prefix_weight=None,
        unified_wang_fix_pair_weight=None,
        unified_wang_target_gap_weight=None,
        unified_wang_same_parent_penalty=None,
    )

    energy = _unified_wang_qubo_linear_energy(
        code,
        (n_value, 6, b"pa"),
        [(code, (n_value, 6, b"pa"))],
        args,
    )

    assert energy == n_value * 1000.0 + 6.0


def test_unified_wang_qubo_energy_applies_fixed_prefix_weight_independently():
    from benchmark_compact_beam_qubo_n0 import _unified_wang_qubo_linear_energy
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    lower_prefix_code, lower_prefix_n = _encode([[5, 4], [3, 1, 2], []], stack_count, max_height)
    higher_prefix_code, higher_prefix_n = _encode([[5], [4, 3], [2, 1]], stack_count, max_height)
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        unified_wang_n_weight=0.0,
        unified_wang_depth_weight=0.0,
        unified_wang_fixed_prefix_weight=2.0,
        unified_wang_fix_pair_weight=0.0,
        unified_wang_target_gap_weight=0.0,
        unified_wang_same_parent_penalty=0.0,
    )
    entries = [
        (lower_prefix_code, (lower_prefix_n, 6, b"pa")),
        (higher_prefix_code, (higher_prefix_n, 7, b"pb")),
    ]

    lower_prefix_energy = _unified_wang_qubo_linear_energy(
        lower_prefix_code,
        (lower_prefix_n, 6, b"pa"),
        entries,
        args,
    )
    higher_prefix_energy = _unified_wang_qubo_linear_energy(
        higher_prefix_code,
        (higher_prefix_n, 7, b"pb"),
        entries,
        args,
    )

    assert higher_prefix_energy < lower_prefix_energy


def test_unified_wang_qubo_energy_applies_fix_pair_weight_independently():
    from benchmark_compact_beam_qubo_n0 import _unified_wang_qubo_linear_energy
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    higher_pair_code, higher_pair_n = _encode([[5, 4], [3, 1, 2], []], stack_count, max_height)
    lower_pair_code, lower_pair_n = _encode([[5], [4, 3], [2, 1]], stack_count, max_height)
    args = _compact_args(
        _stack_count=stack_count,
        _max_height=max_height,
        unified_wang_n_weight=0.0,
        unified_wang_depth_weight=0.0,
        unified_wang_fixed_prefix_weight=0.0,
        unified_wang_fix_pair_weight=3.0,
        unified_wang_target_gap_weight=0.0,
        unified_wang_same_parent_penalty=0.0,
    )
    entries = [
        (higher_pair_code, (higher_pair_n, 6, b"pa")),
        (lower_pair_code, (lower_pair_n, 7, b"pb")),
    ]

    higher_pair_energy = _unified_wang_qubo_linear_energy(
        higher_pair_code,
        (higher_pair_n, 6, b"pa"),
        entries,
        args,
    )
    lower_pair_energy = _unified_wang_qubo_linear_energy(
        lower_pair_code,
        (lower_pair_n, 7, b"pb"),
        entries,
        args,
    )

    assert higher_pair_energy < lower_pair_energy


def test_solver_profile_scheduler_state_v5_wang_probe_inherits_state_controller(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_wang_probe")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.selection_policy == "scheduler_state_qubo"
    assert args.scheduler_state_controller == "state_v2"
    assert args.scheduler_state_initial_plateau_unique_min_n == 24
    assert args.wang_probe_supplement_width > 0
    assert args.wang_probe_plateau_window >= 3


def test_solver_profile_scheduler_state_v5_wang_hybrid_enables_all_wang_modules(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_wang_hybrid")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.selection_policy == "scheduler_state_qubo"
    assert args.scheduler_state_controller == "state_v2"
    assert args.wang_incumbent_enabled is True
    assert args.wang_cap_enabled is True
    assert args.wang_rollout_enabled is True
    assert args.wang_pair_qubo_enabled is True
    assert args.wang_pair_qubo_supplement_width > 0


def test_solver_profile_scheduler_state_v5_wang_rescue_uses_wang_pair_qubo(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_wang_rescue")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.selection_policy == "scheduler_state_qubo"
    assert args.rescue_enabled is True
    assert args.scheduler_state_rescue_plateau_window > 0
    assert args.wang_pair_qubo_enabled is True
    assert args.wang_pair_qubo_focus_n == -1
    assert args.wang_pair_qubo_supplement_width >= 12


def test_solver_profile_scheduler_state_v5_wang_features_enables_generic_features(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_wang_features")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v5_wang_features"
    assert args.scheduler_state_controller == "state_v2"
    assert args.group_qubo_supplement is True
    assert args.wang_feature_fixed_prefix_weight > 0.0
    assert args.wang_feature_fix_pair_weight > 0.0
    assert args.wang_pair_qubo_enabled is False


def test_solver_profile_scheduler_state_v7_wang_qubo_unified_enables_unified_selector(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile, _scheduler_state_action

    args = _compact_args(solver_profile="scheduler_state_v7_wang_qubo_unified")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v7_wang_qubo_unified"
    assert args.selection_policy == "scheduler_state_qubo"
    assert args.scheduler_state_controller == "state_v2"
    assert args.use_qubo is True
    assert args.unified_wang_qubo_enabled is True
    assert args.unified_wang_qubo_supplement_width == 1
    assert args.unified_wang_qubo_select == 1
    assert args.unified_wang_fixed_prefix_weight > 0.0
    assert args.unified_wang_fix_pair_weight > 0.0
    assert args.unified_wang_pair_same_parent_penalty > 0.0
    assert args.unified_wang_rollout_bonus == 0.0
    assert args.wang_incumbent_enabled is False
    assert args.wang_cap_enabled is False
    assert args.wang_rollout_enabled is True
    assert args.wang_rollout_strategy == "baby_lookahead"
    assert args.wang_rollout_inject_candidates is False
    assert args.wang_rollout_width == 2
    assert args.wang_rollout_focus_n == -1
    assert args.wang_rollout_plateau_window == 0
    assert args.wang_pair_qubo_enabled is False
    assert args.post_polish_qubo is False
    assert args.rescue_enabled is False
    assert args.rescue_qubo_supplement_width == 0
    assert args.rescue_max_layers == 0
    assert args.scheduler_state_rescue_cap_slack == 0
    assert args.scheduler_state_initial_plateau_action != "rescue"

    args._best_n_history = [30, 30, 30, 30, 30]
    args._initial_n = 30
    args._priority_duplication_ratio = 0.50
    args._layer_depth_slack = 0
    args._layer_candidate_count = 12000
    assert _scheduler_state_action(args, best_n=30) != "rescue"


def test_solver_profile_scheduler_state_v7_wang_qubo_pool_enables_rollout_candidate_pool(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v7_wang_qubo_pool")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v7_wang_qubo_pool"
    assert args.unified_wang_qubo_enabled is True
    assert args.wang_rollout_enabled is True
    assert args.wang_rollout_strategy == "baby_lookahead"
    assert args.wang_rollout_inject_candidates is True
    assert args.unified_wang_rollout_bonus > 0.0


def test_solver_profile_scheduler_state_v5_wang_rescue_polish_combines_rescue_and_window_polish(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_wang_rescue_polish")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.rescue_enabled is True
    assert args.wang_pair_qubo_enabled is True
    assert args.post_polish_qubo is True
    assert args.post_polish_mode == "qubo_windows_v2"
    assert args.post_polish_accept == "better"
    assert args.post_polish_window_select >= 1
    assert args.post_polish_successor_cache is False


def test_solver_profile_scheduler_state_v5_wang_rescue_successor_cache_polish_is_opt_in(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_wang_rescue_successor_cache_polish")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v5_wang_rescue_successor_cache_polish"
    assert args.wang_pair_qubo_enabled is True
    assert args.post_polish_qubo is True
    assert args.post_polish_mode == "qubo_windows_v2"
    assert args.post_polish_successor_cache is True


def test_solver_profile_scheduler_state_v5_feasibility_escape_polish_extends_rescue_polish(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_feasibility_escape_polish")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v5_feasibility_escape_polish"
    assert args.rescue_enabled is True
    assert args.wang_pair_qubo_enabled is True
    assert args.post_polish_qubo is True
    assert args.post_polish_mode == "qubo_windows_v2"
    assert args.feasibility_escape_enabled is True
    assert args.feasibility_escape_supplement_width > 0
    assert args.feasibility_escape_dead_end_threshold > 0.0


def test_solver_profile_scheduler_state_v5_frg_dlts_polish_extends_current_best(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_frg_dlts_polish")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v5_frg_dlts_polish"
    assert args.post_polish_qubo is True
    assert args.post_polish_mode == "qubo_windows_v2"
    assert args.wang_pair_qubo_enabled is True
    assert args.frg_candidate_supplement_enabled is True
    assert args.frg_candidate_supplement_width == 8
    assert args.frg_compound_escape_enabled is False
    assert args.dlts_value_supplement_enabled is True
    assert args.dlts_value_supplement_width == 4


def test_solver_profile_scheduler_state_v5_frg_compound_polish_is_gated(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_frg_compound_polish")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v5_frg_compound_polish"
    assert args.post_polish_qubo is True
    assert args.wang_pair_qubo_enabled is True
    assert args.frg_candidate_supplement_enabled is True
    assert args.dlts_value_supplement_enabled is True
    assert args.frg_compound_escape_enabled is True
    assert args.frg_compound_escape_width == 2
    assert args.frg_compound_escape_plateau_window == 5
    assert args.frg_compound_escape_cap_slack == 24
    assert args.frg_compound_escape_min_candidates == 7000


def test_solver_profile_scheduler_state_v5_frg_compound_loose_polish_is_residual_probe(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_frg_compound_loose_polish")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v5_frg_compound_loose_polish"
    assert args.frg_candidate_supplement_enabled is True
    assert args.dlts_value_supplement_enabled is True
    assert args.frg_compound_escape_enabled is True
    assert args.frg_compound_escape_width == 1
    assert args.frg_compound_escape_plateau_window == 2
    assert args.frg_compound_escape_cap_slack == 99
    assert args.frg_compound_escape_min_candidates == 1


def test_solver_profile_scheduler_state_v5_frg_compound_guarded_polish_uses_gate_stack(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_frg_compound_guarded_polish")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v5_frg_compound_guarded_polish"
    assert args.frg_candidate_supplement_enabled is False
    assert args.dlts_value_supplement_enabled is False
    assert args.frg_compound_escape_enabled is True
    assert args.frg_compound_escape_width == 1
    assert args.frg_compound_escape_plateau_window == 2
    assert args.frg_compound_escape_cap_slack == 99
    assert args.frg_compound_escape_min_candidates == 1000
    assert args.frg_compound_guard_enabled is True
    assert args.frg_compound_guard_max_stack_count == 5
    assert args.frg_compound_guard_max_initial_n == 18
    assert args.frg_compound_guard_min_best_n == 12
    assert args.frg_compound_guard_max_best_n == 18
    assert args.frg_compound_guard_min_parent_count == 3


def test_solver_profile_scheduler_state_v5_wang_rescue_suffix_polish_skips_windows(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_wang_rescue_suffix_polish")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.rescue_enabled is True
    assert args.wang_pair_qubo_enabled is True
    assert args.post_polish_qubo is True
    assert args.post_polish_mode == "qubo_windows_v2"
    assert args.post_polish_window_select == 0


def test_solver_profile_scheduler_state_v5_wang_rescue_dynamic_polish_enables_survivor_cap(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_wang_rescue_dynamic_polish")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.rescue_enabled is True
    assert args.post_polish_qubo is True
    assert args.post_polish_mode == "qubo_windows_v2"
    assert args.survivor_parent_cap_enabled is True
    assert args.survivor_parent_cap_value < args.parent_core_cap


def test_solver_profile_scheduler_state_v5_wang_rescue_risk_polish_adds_small_group_qubo(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_wang_rescue_risk_polish")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.rescue_enabled is True
    assert args.post_polish_qubo is True
    assert args.group_qubo_supplement is True
    assert args.qubo_group_select <= 12
    assert args.wang_feature_fixed_prefix_weight > 0.0
    assert args.wang_feature_same_parent_weight > 0.0


def test_solver_profile_scheduler_state_v5_wang_rescue_selective_window_polish_filters_windows(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_state_v5_wang_rescue_selective_window_polish")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.rescue_enabled is True
    assert args.post_polish_mode == "qubo_windows_v2"
    assert args.post_polish_window_select >= 2
    assert args.post_polish_window_min_score_ratio > 0.0


def test_post_polish_windows_v2_min_score_ratio_filters_low_signal_windows():
    from benchmark_compact_beam_qubo_n0 import PostPolishWindow, _select_post_polish_windows_v2_qubo

    windows = [
        PostPolishWindow(start=0, end=4, old_len=4, min_n=2, max_n=4, score=10.0),
        PostPolishWindow(start=4, end=8, old_len=4, min_n=2, max_n=5, score=7.0),
        PostPolishWindow(start=8, end=12, old_len=4, min_n=3, max_n=6, score=2.0),
    ]
    args = _compact_args(post_polish_window_select=3, post_polish_window_min_score_ratio=0.80)

    selected, qvars, qtime = _select_post_polish_windows_v2_qubo(windows, args)

    assert selected == [windows[0]]
    assert qvars == 0
    assert qtime == 0.0


def test_rescue_layer_gate_requires_plateau_near_cap_and_candidate_pressure():
    from benchmark_compact_beam_qubo_n0 import _rescue_layer_active

    args = _compact_args(
        rescue_enabled=True,
        rescue_plateau_window=4,
        rescue_plateau_delta=0,
        rescue_cap_slack=8,
        rescue_min_n=12,
        rescue_min_candidates=5000,
        rescue_max_layers=2,
    )
    args._best_n_history = [18, 16, 16, 16, 16]
    args._layer_depth_slack = 8
    args._layer_candidate_count = 6000
    args._rescue_layers_used = 0

    assert _rescue_layer_active(args, best_n=16)

    args._layer_depth_slack = 9
    assert not _rescue_layer_active(args, best_n=16)

    args._layer_depth_slack = 8
    args._best_n_history = [18, 17, 16, 16]
    assert not _rescue_layer_active(args, best_n=16)

    args._best_n_history = [18, 16, 16, 16, 16]
    args._layer_candidate_count = 4999
    assert not _rescue_layer_active(args, best_n=16)

    args._layer_candidate_count = 6000
    args._rescue_layers_used = 2
    assert not _rescue_layer_active(args, best_n=16)


def test_solver_profile_rescue_qubo_v1_is_local_rescue_not_speed_v2(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="rescue_qubo_v1")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.selection_policy == "rescue_qubo"
    assert args.beam_width == 1200
    assert args.core_width == 1080
    assert args.rescue_enabled is True
    assert args.rescue_beam_width > args.beam_width
    assert args.rescue_core_width > args.core_width
    assert args.rescue_escalate_after == 2
    assert args.rescue_escalate_beam_width >= args.rescue_beam_width
    assert args.rescue_max_qubo_vars > args.max_qubo_vars
    assert args.rescue_qubo_supplement_width > args.qubo_supplement_width
    assert args.qubo_focus_n == 8


def test_solver_profile_light_rescue_qubo_v1_is_more_conservative(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    full = _compact_args(solver_profile="rescue_qubo_v1")
    light = _compact_args(solver_profile="light_rescue_qubo_v1")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(full)
    _apply_solver_profile(light)

    assert light.selection_policy == "rescue_qubo"
    assert light.rescue_enabled is True
    assert light.rescue_max_layers < full.rescue_max_layers
    assert light.rescue_beam_width < full.rescue_beam_width
    assert light.rescue_core_width < full.rescue_core_width
    assert light.rescue_escalate_after > full.rescue_escalate_after
    assert light.rescue_qubo_supplement_width < full.rescue_qubo_supplement_width


def test_solver_profile_adaptive_rescue_v1_combines_adaptive_with_light_rescue(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="adaptive_rescue_v1")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.selection_policy == "adaptive_qubo"
    assert args.rescue_enabled is True
    assert args.adaptive_high_frontier == 800
    assert args.adaptive_mid_frontier == 1000
    assert args.adaptive_low_frontier == 1200
    assert args.rescue_plateau_window == 3
    assert args.rescue_cap_slack == 12
    assert args.rescue_min_candidates == 20000
    assert args.rescue_max_layers == 3
    assert args.rescue_beam_width == 1200
    assert args.rescue_core_width == 1080
    assert args.rescue_qubo_supplement_width >= 32
    assert args.rescue_max_qubo_vars >= 80


def test_solver_profile_adaptive_polish_v1_combines_adaptive_with_quality_polish(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="adaptive_polish_v1")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.selection_policy == "adaptive_qubo"
    assert args.adaptive_high_frontier == 800
    assert args.adaptive_mid_frontier == 1000
    assert args.adaptive_low_frontier == 1200
    assert args.post_polish_passes == 3
    assert args.post_polish_suffix_max_candidates == 72
    assert args.post_polish_suffix_select == 7


def test_solver_profile_scheduler_topk_combines_scheduler_with_bounded_rank(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="scheduler_topk_v1")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.selection_policy == "scheduler_qubo"
    assert args.rescue_enabled is True
    assert args.rank_bucket_topk == 64
    assert args.rank_bucket_key == "n_depth_parent"
    assert args.rank_bucket_topk_high_n == 18
    assert args.rank_bucket_topk_min_candidates == 10000
    assert args.scheduler_adaptive_min_stacks == 7
    assert args.scheduler_adaptive_cv1_only is False
    assert args.scheduler_adaptive_max_duplication_ratio == 0.0
    assert args.scheduler_adaptive_min_best_n_ratio == 0.45
    assert args.scheduler_adaptive_plateau_window == 3
    assert args.scheduler_adaptive_cap_slack == 12


def test_rescue_frontier_widths_escalate_after_repeated_rescue_layers():
    from benchmark_compact_beam_qubo_n0 import _rescue_frontier_widths

    args = _compact_args(
        rescue_beam_width=1350,
        rescue_core_width=1215,
        rescue_escalate_after=8,
        rescue_escalate_beam_width=1600,
        rescue_escalate_core_width=1440,
    )
    args._rescue_layers_used = 7

    assert _rescue_frontier_widths(args) == (1350, 1215)

    args._rescue_layers_used = 8

    assert _rescue_frontier_widths(args) == (1600, 1440)


def test_scheduler_qubo_prefers_rescue_when_plateau_is_active(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact

    ranked = [(b"a", (18, 7, b"p"))]
    args = _compact_args(
        selection_policy="scheduler_qubo",
        rescue_enabled=True,
        rescue_plateau_window=2,
        rescue_plateau_delta=0,
        rescue_cap_slack=30,
        rescue_min_n=12,
        rescue_min_candidates=100,
        rescue_max_layers=4,
        _best_n_history=[18, 18],
        _layer_candidate_count=1000,
        _layer_depth_slack=10,
        _rescue_layers_used=0,
        _stack_count=7,
    )
    calls: list[str] = []

    monkeypatch.setattr(compact, "_select_rescue_qubo_frontier", lambda *_args: calls.append("rescue") or ([b"r"], {b"r"}, 1, 3, 0.1))
    monkeypatch.setattr(compact, "_select_adaptive_qubo_frontier", lambda *_args: calls.append("adaptive") or ([b"a"], set(), 0, 0, 0.0))
    monkeypatch.setattr(compact, "_select_enhanced_qubo_frontier", lambda *_args: calls.append("enhanced") or ([b"e"], set(), 0, 0, 0.0))

    selected = compact._select_scheduler_qubo_frontier(ranked, len(ranked), args)

    assert selected[0] == [b"r"]
    assert calls == ["rescue"]


def test_scheduler_qubo_uses_adaptive_for_large_high_n_seven_stack_layers(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact

    ranked = [(b"a", (24, 3, b"p"))]
    args = _compact_args(
        selection_policy="scheduler_qubo",
        scheduler_adaptive_min_stacks=7,
        scheduler_adaptive_high_n=16,
        rescue_enabled=True,
        rescue_plateau_window=2,
        rescue_plateau_delta=0,
        rescue_cap_slack=4,
        rescue_min_n=12,
        rescue_min_candidates=5000,
        _best_n_history=[25, 24],
        _layer_candidate_count=1000,
        _layer_depth_slack=99,
        _rescue_layers_used=0,
        _stack_count=7,
    )
    calls: list[str] = []

    monkeypatch.setattr(compact, "_select_rescue_qubo_frontier", lambda *_args: calls.append("rescue") or ([b"r"], {b"r"}, 1, 3, 0.1))
    monkeypatch.setattr(compact, "_select_adaptive_qubo_frontier", lambda *_args: calls.append("adaptive") or ([b"a"], set(), 0, 0, 0.0))
    monkeypatch.setattr(compact, "_select_enhanced_qubo_frontier", lambda *_args: calls.append("enhanced") or ([b"e"], set(), 0, 0, 0.0))

    selected = compact._select_scheduler_qubo_frontier(ranked, len(ranked), args)

    assert selected[0] == [b"a"]
    assert calls == ["adaptive"]


def test_scheduler_qubo_keeps_small_instances_on_enhanced_path(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact

    ranked = [(b"a", (24, 3, b"p"))]
    args = _compact_args(
        selection_policy="scheduler_qubo",
        scheduler_adaptive_min_stacks=7,
        scheduler_adaptive_high_n=16,
        rescue_enabled=False,
        _best_n_history=[],
        _layer_candidate_count=1000,
        _layer_depth_slack=99,
        _rescue_layers_used=0,
        _stack_count=5,
    )
    calls: list[str] = []

    monkeypatch.setattr(compact, "_select_adaptive_qubo_frontier", lambda *_args: calls.append("adaptive") or ([b"a"], set(), 0, 0, 0.0))
    monkeypatch.setattr(compact, "_select_enhanced_qubo_frontier", lambda *_args: calls.append("enhanced") or ([b"e"], set(), 0, 0, 0.0))

    selected = compact._select_scheduler_qubo_frontier(ranked, len(ranked), args)

    assert selected[0] == [b"e"]
    assert calls == ["enhanced"]


def test_scheduler_qubo_avoids_adaptive_for_non_cv1_open_cases(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact

    ranked = [(b"a", (24, 3, b"p"))]
    args = _compact_args(
        selection_policy="scheduler_qubo",
        scheduler_adaptive_min_stacks=7,
        scheduler_adaptive_high_n=16,
        scheduler_adaptive_cv1_only=True,
        rescue_enabled=False,
        _best_n_history=[],
        _layer_candidate_count=1000,
        _layer_depth_slack=99,
        _rescue_layers_used=0,
        _stack_count=7,
        _case_id="7x5_cv2_test_2",
    )
    calls: list[str] = []

    monkeypatch.setattr(compact, "_select_adaptive_qubo_frontier", lambda *_args: calls.append("adaptive") or ([b"a"], set(), 0, 0, 0.0))
    monkeypatch.setattr(compact, "_select_enhanced_qubo_frontier", lambda *_args: calls.append("enhanced") or ([b"e"], set(), 0, 0, 0.0))

    selected = compact._select_scheduler_qubo_frontier(ranked, len(ranked), args)

    assert selected[0] == [b"e"]
    assert calls == ["enhanced"]


def test_adaptive_qubo_uses_rescue_when_plateau_is_active(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact

    ranked = [(b"a", (18, 7, b"p"))]
    args = _compact_args(
        selection_policy="adaptive_qubo",
        rescue_enabled=True,
        rescue_plateau_window=2,
        rescue_plateau_delta=0,
        rescue_cap_slack=30,
        rescue_min_n=12,
        rescue_min_candidates=100,
        rescue_max_layers=4,
        _best_n_history=[18, 18],
        _layer_candidate_count=1000,
        _layer_depth_slack=10,
        _rescue_layers_used=0,
    )
    calls: list[str] = []

    monkeypatch.setattr(compact, "_select_rescue_qubo_frontier", lambda *_args: calls.append("rescue") or ([b"r"], {b"r"}, 1, 3, 0.1))
    monkeypatch.setattr(compact, "_select_enhanced_qubo_frontier", lambda *_args, **_kwargs: calls.append("enhanced") or ([b"e"], set(), 0, 0, 0.0))

    selected = compact._select_adaptive_qubo_frontier(ranked, len(ranked), args)

    assert selected[0] == [b"r"]
    assert calls == ["rescue"]


def test_scheduler_adaptive_gate_uses_priority_duplication_not_case_id():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        scheduler_adaptive_cv1_only=False,
        scheduler_adaptive_max_duplication_ratio=0.0,
        scheduler_adaptive_min_best_n_ratio=0.45,
        scheduler_adaptive_high_n=16,
        _stack_count=7,
        _initial_n=32,
        _priority_duplication_ratio=0.25,
        _layer_candidate_count=12000,
        _layer_parent_top_share=0.05,
        _layer_depth_slack=99,
        _best_n_history=[24],
        _case_id="unlabeled_priority_group_case",
    )

    assert not compact._scheduler_uses_adaptive(args, best_n=24)

    args._priority_duplication_ratio = 0.0
    args._case_id = "7x5_cv2_test_2"

    assert compact._scheduler_uses_adaptive(args, best_n=24)


def test_scheduler_adaptive_gate_blocks_plateau_when_slack_is_tight():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        scheduler_adaptive_cv1_only=False,
        scheduler_adaptive_max_duplication_ratio=1.0,
        scheduler_adaptive_plateau_window=3,
        scheduler_adaptive_cap_slack=12,
        scheduler_adaptive_high_n=16,
        _stack_count=7,
        _initial_n=30,
        _priority_duplication_ratio=0.0,
        _layer_candidate_count=12000,
        _layer_parent_top_share=0.05,
        _layer_depth_slack=8,
        _best_n_history=[24, 24, 24],
    )

    assert not compact._scheduler_uses_adaptive(args, best_n=24)

    args._best_n_history = [26, 25, 24]

    assert compact._scheduler_uses_adaptive(args, best_n=24)


def test_scheduler_state_v2_contracts_only_when_search_is_progressing():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        scheduler_state_controller="state_v2",
        scheduler_state_contract_pressure=8.0,
        scheduler_state_progress_window=3,
        scheduler_state_progress_delta=2,
        scheduler_state_protect_parent_top_share=0.65,
        scheduler_state_protect_n_band=2,
        _initial_n=40,
        _best_n_history=[31, 28, 25],
        _layer_candidate_count=12000,
        _layer_depth_slack=40,
        _layer_parent_top_share=0.10,
        _layer_top_n_band=1,
    )

    assert compact._scheduler_state_action(args, best_n=25) == "contract"


def test_scheduler_state_v2_protects_uncertain_or_collapsed_frontiers():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        scheduler_state_controller="state_v2",
        scheduler_state_contract_pressure=8.0,
        scheduler_state_progress_window=3,
        scheduler_state_progress_delta=2,
        scheduler_state_protect_parent_top_share=0.65,
        scheduler_state_protect_n_band=2,
        _initial_n=40,
        _best_n_history=[31, 28, 25],
        _layer_candidate_count=12000,
        _layer_depth_slack=40,
        _layer_parent_top_share=0.70,
        _layer_top_n_band=1,
    )

    assert compact._scheduler_state_action(args, best_n=25) == "protect"

    args._layer_parent_top_share = 0.10
    args._layer_top_n_band = 4

    assert compact._scheduler_state_action(args, best_n=25) == "protect"


def test_scheduler_state_v2_protects_extreme_candidate_pressure():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        beam_width=1200,
        scheduler_state_controller="state_v2",
        scheduler_state_contract_pressure=8.0,
        scheduler_state_max_contract_pressure=40.0,
        scheduler_state_progress_window=3,
        scheduler_state_progress_delta=2,
        scheduler_state_protect_parent_top_share=0.65,
        scheduler_state_protect_n_band=2,
        _initial_n=40,
        _best_n_history=[31, 28, 25],
        _layer_candidate_count=60000,
        _layer_depth_slack=40,
        _layer_parent_top_share=0.10,
        _layer_top_n_band=1,
    )

    assert compact._scheduler_state_action(args, best_n=25) == "protect"


def test_scheduler_state_v2_protects_late_search_after_n_reduction():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        beam_width=1200,
        scheduler_state_controller="state_v2",
        scheduler_state_contract_pressure=8.0,
        scheduler_state_max_contract_pressure=40.0,
        scheduler_state_min_best_n_ratio=0.65,
        scheduler_state_progress_window=3,
        scheduler_state_progress_delta=0,
        scheduler_state_protect_parent_top_share=0.65,
        scheduler_state_protect_n_band=2,
        _initial_n=40,
        _best_n_history=[27, 26, 25],
        _layer_candidate_count=12000,
        _layer_depth_slack=40,
        _layer_parent_top_share=0.10,
        _layer_top_n_band=1,
    )

    assert compact._scheduler_state_action(args, best_n=25) == "protect"


def test_scheduler_state_v2_rescues_plateau_with_tight_slack():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        scheduler_state_controller="state_v2",
        scheduler_state_rescue_plateau_window=3,
        scheduler_state_rescue_plateau_delta=0,
        scheduler_state_rescue_cap_slack=8,
        scheduler_state_contract_pressure=8.0,
        _initial_n=40,
        _best_n_history=[24, 24, 24],
        _layer_candidate_count=12000,
        _layer_depth_slack=6,
        _layer_parent_top_share=0.10,
        _layer_top_n_band=1,
    )

    assert compact._scheduler_state_action(args, best_n=24) == "rescue"


def test_scheduler_state_v3_protects_after_failed_contract():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        beam_width=1200,
        scheduler_state_controller="state_v2",
        scheduler_state_contract_pressure=8.0,
        scheduler_state_max_contract_pressure=40.0,
        scheduler_state_progress_window=3,
        scheduler_state_progress_delta=0,
        scheduler_state_failed_contract_cooldown=1,
        scheduler_state_protect_parent_top_share=0.65,
        scheduler_state_protect_n_band=2,
        _initial_n=40,
        _best_n_history=[26, 25, 25],
        _layer_candidate_count=12000,
        _layer_depth_slack=40,
        _layer_parent_top_share=0.10,
        _layer_top_n_band=1,
        _scheduler_state_last_action="contract",
    )

    assert compact._scheduler_state_action(args, best_n=25) == "protect"

    args._best_n_history = [27, 26, 25]

    assert compact._scheduler_state_action(args, best_n=25) == "contract"


def test_scheduler_state_v3_protects_high_plateau_after_effective_rescue():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        beam_width=1200,
        scheduler_state_controller="state_v2",
        scheduler_state_contract_pressure=8.0,
        scheduler_state_max_contract_pressure=40.0,
        scheduler_state_progress_window=3,
        scheduler_state_progress_delta=0,
        scheduler_state_post_rescue_plateau_window=3,
        scheduler_state_post_rescue_plateau_delta=0,
        scheduler_state_post_rescue_min_ratio=0.90,
        scheduler_state_post_rescue_action="protect",
        scheduler_state_protect_parent_top_share=0.65,
        scheduler_state_protect_n_band=2,
        _initial_n=40,
        _best_n_history=[40, 40, 40],
        _layer_candidate_count=12000,
        _layer_depth_slack=40,
        _layer_parent_top_share=0.10,
        _layer_top_n_band=1,
        _scheduler_state_last_effective_action="rescue",
    )

    assert compact._scheduler_state_action(args, best_n=40) == "protect"

    args._best_n_history = [40, 39, 38]

    assert compact._scheduler_state_action(args, best_n=38) == "contract"


def test_scheduler_state_v3_protects_long_initial_plateau():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        beam_width=1200,
        scheduler_state_controller="state_v2",
        scheduler_state_contract_pressure=8.0,
        scheduler_state_max_contract_pressure=40.0,
        scheduler_state_progress_window=3,
        scheduler_state_progress_delta=0,
        scheduler_state_initial_plateau_window=5,
        scheduler_state_initial_plateau_min_ratio=0.98,
        scheduler_state_initial_plateau_action="protect",
        scheduler_state_protect_parent_top_share=0.65,
        scheduler_state_protect_n_band=2,
        _initial_n=40,
        _best_n_history=[40, 40, 40, 40, 40],
        _layer_candidate_count=12000,
        _layer_depth_slack=40,
        _layer_parent_top_share=0.10,
        _layer_top_n_band=1,
    )

    assert compact._scheduler_state_action(args, best_n=40) == "protect"

    args._best_n_history = [40, 40, 39, 39, 39]

    assert compact._scheduler_state_action(args, best_n=39) == "contract"


def test_scheduler_state_v3_blends_on_long_initial_plateau_when_configured():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        beam_width=1200,
        scheduler_state_controller="state_v2",
        scheduler_state_contract_pressure=8.0,
        scheduler_state_max_contract_pressure=40.0,
        scheduler_state_progress_window=3,
        scheduler_state_progress_delta=0,
        scheduler_state_initial_plateau_window=5,
        scheduler_state_initial_plateau_min_ratio=0.98,
        scheduler_state_initial_plateau_action="blend",
        scheduler_state_protect_parent_top_share=0.65,
        scheduler_state_protect_n_band=2,
        _initial_n=40,
        _best_n_history=[40, 40, 40, 40, 40],
        _layer_candidate_count=12000,
        _layer_depth_slack=40,
        _layer_parent_top_share=0.10,
        _layer_top_n_band=1,
    )

    assert compact._scheduler_state_action(args, best_n=40) == "blend"


def test_scheduler_state_v3_can_route_long_initial_plateau_to_rescue_selector():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        beam_width=1200,
        scheduler_state_controller="state_v2",
        scheduler_state_contract_pressure=8.0,
        scheduler_state_max_contract_pressure=40.0,
        scheduler_state_progress_window=3,
        scheduler_state_progress_delta=0,
        scheduler_state_initial_plateau_window=5,
        scheduler_state_initial_plateau_min_ratio=0.98,
        scheduler_state_initial_plateau_action="rescue",
        scheduler_state_protect_parent_top_share=0.65,
        scheduler_state_protect_n_band=2,
        _initial_n=40,
        _best_n_history=[40, 40, 40, 40, 40],
        _layer_candidate_count=12000,
        _layer_depth_slack=40,
        _layer_parent_top_share=0.10,
        _layer_top_n_band=1,
    )

    assert compact._scheduler_state_action(args, best_n=40) == "rescue"


def test_scheduler_state_v3_can_lock_rescue_after_long_initial_plateau():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        beam_width=1200,
        scheduler_state_controller="state_v2",
        scheduler_state_contract_pressure=8.0,
        scheduler_state_max_contract_pressure=40.0,
        scheduler_state_progress_window=3,
        scheduler_state_progress_delta=0,
        scheduler_state_initial_plateau_window=5,
        scheduler_state_initial_plateau_min_ratio=0.98,
        scheduler_state_initial_plateau_action="rescue",
        scheduler_state_initial_plateau_lock=True,
        scheduler_state_initial_plateau_min_duplication=0.10,
        scheduler_state_protect_parent_top_share=0.65,
        scheduler_state_protect_n_band=2,
        _initial_n=40,
        _priority_duplication_ratio=0.50,
        _best_n_history=[40, 40, 40, 40, 40],
        _layer_candidate_count=12000,
        _layer_depth_slack=40,
        _layer_parent_top_share=0.10,
        _layer_top_n_band=1,
    )

    assert compact._scheduler_state_action(args, best_n=40) == "rescue"
    assert args._scheduler_state_locked_action == "rescue"

    args._best_n_history = [40, 39, 38]

    assert compact._scheduler_state_action(args, best_n=38) == "rescue"


def test_scheduler_state_v3_does_not_lock_initial_plateau_for_unique_priorities():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        beam_width=1200,
        scheduler_state_controller="state_v2",
        scheduler_state_contract_pressure=8.0,
        scheduler_state_max_contract_pressure=40.0,
        scheduler_state_progress_window=3,
        scheduler_state_progress_delta=0,
        scheduler_state_initial_plateau_window=5,
        scheduler_state_initial_plateau_min_ratio=0.98,
        scheduler_state_initial_plateau_action="rescue",
        scheduler_state_initial_plateau_lock=True,
        scheduler_state_initial_plateau_min_duplication=0.10,
        scheduler_state_protect_parent_top_share=0.65,
        scheduler_state_protect_n_band=2,
        _initial_n=40,
        _priority_duplication_ratio=0.0,
        _best_n_history=[40, 40, 40, 40, 40],
        _layer_candidate_count=12000,
        _layer_depth_slack=40,
        _layer_parent_top_share=0.10,
        _layer_top_n_band=1,
    )

    assert compact._scheduler_state_action(args, best_n=40) == "contract"
    assert not hasattr(args, "_scheduler_state_locked_action")


def test_scheduler_state_v5_skips_low_n_unique_initial_plateau():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        beam_width=1200,
        scheduler_state_controller="state_v2",
        scheduler_state_contract_pressure=8.0,
        scheduler_state_max_contract_pressure=40.0,
        scheduler_state_progress_window=3,
        scheduler_state_progress_delta=0,
        scheduler_state_initial_plateau_window=5,
        scheduler_state_initial_plateau_min_ratio=0.98,
        scheduler_state_initial_plateau_action="rescue",
        scheduler_state_initial_plateau_lock=True,
        scheduler_state_initial_plateau_min_duplication=0.0,
        scheduler_state_initial_plateau_unique_min_n=24,
        scheduler_state_protect_parent_top_share=0.65,
        scheduler_state_protect_n_band=2,
        _initial_n=20,
        _priority_duplication_ratio=0.0,
        _best_n_history=[20, 20, 20, 20, 20],
        _layer_candidate_count=12000,
        _layer_depth_slack=40,
        _layer_parent_top_share=0.10,
        _layer_top_n_band=1,
    )

    assert compact._scheduler_state_action(args, best_n=20) == "contract"
    assert not hasattr(args, "_scheduler_state_locked_action")

    args._initial_n = 24
    args._best_n_history = [24, 24, 24, 24, 24]

    assert compact._scheduler_state_action(args, best_n=24) == "rescue"
    assert args._scheduler_state_locked_action == "rescue"


def test_scheduler_state_v5_keeps_low_n_grouped_initial_plateau_rescue():
    import benchmark_compact_beam_qubo_n0 as compact

    args = _compact_args(
        beam_width=1200,
        scheduler_state_controller="state_v2",
        scheduler_state_contract_pressure=8.0,
        scheduler_state_max_contract_pressure=40.0,
        scheduler_state_progress_window=3,
        scheduler_state_progress_delta=0,
        scheduler_state_initial_plateau_window=5,
        scheduler_state_initial_plateau_min_ratio=0.98,
        scheduler_state_initial_plateau_action="rescue",
        scheduler_state_initial_plateau_lock=True,
        scheduler_state_initial_plateau_min_duplication=0.0,
        scheduler_state_initial_plateau_unique_min_n=24,
        scheduler_state_protect_parent_top_share=0.65,
        scheduler_state_protect_n_band=2,
        _initial_n=17,
        _priority_duplication_ratio=0.50,
        _best_n_history=[17, 17, 17, 17, 17],
        _layer_candidate_count=12000,
        _layer_depth_slack=40,
        _layer_parent_top_share=0.10,
        _layer_top_n_band=1,
    )

    assert compact._scheduler_state_action(args, best_n=17) == "rescue"
    assert args._scheduler_state_locked_action == "rescue"


def test_scheduler_state_blend_backfills_adaptive_selection(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact

    ranked = [(bytes([idx]), (24, 3, b"p")) for idx in range(5)]
    args = _compact_args(selection_policy="scheduler_state_qubo", beam_width=4)
    calls: list[str] = []

    monkeypatch.setattr(compact, "_scheduler_state_action", lambda *_args: "blend")
    monkeypatch.setattr(compact, "_select_adaptive_qubo_frontier", lambda *_args: calls.append("adaptive") or ([b"\x00", b"\x02"], {b"\x02"}, 1, 3, 0.1))

    selected = compact._select_scheduler_state_qubo_frontier(ranked, len(ranked), args)

    assert selected == ([b"\x00", b"\x01", b"\x02", b"\x03"], {b"\x02"}, 1, 3, 0.1)
    assert calls == ["adaptive"]


def test_scheduler_state_qubo_dispatches_by_state_action(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact

    ranked = [(b"a", (24, 3, b"p"))]
    args = _compact_args(selection_policy="scheduler_state_qubo")
    calls: list[str] = []

    monkeypatch.setattr(compact, "_scheduler_state_action", lambda *_args: "contract")
    monkeypatch.setattr(compact, "_select_rescue_qubo_frontier", lambda *_args: calls.append("rescue") or ([b"r"], {b"r"}, 1, 3, 0.1))
    monkeypatch.setattr(compact, "_select_adaptive_qubo_frontier", lambda *_args: calls.append("adaptive") or ([b"a"], set(), 0, 0, 0.0))
    monkeypatch.setattr(compact, "_select_enhanced_qubo_frontier", lambda *_args: calls.append("enhanced") or ([b"e"], set(), 0, 0, 0.0))

    selected = compact._select_scheduler_state_qubo_frontier(ranked, len(ranked), args)

    assert selected[0] == [b"a"]
    assert calls == ["adaptive"]

    calls.clear()
    monkeypatch.setattr(compact, "_scheduler_state_action", lambda *_args: "protect")

    selected = compact._select_scheduler_state_qubo_frontier(ranked, len(ranked), args)

    assert selected[0] == [b"e"]
    assert calls == ["enhanced"]

    calls.clear()
    monkeypatch.setattr(compact, "_scheduler_state_action", lambda *_args: "rescue")

    selected = compact._select_scheduler_state_qubo_frontier(ranked, len(ranked), args)

    assert selected[0] == [b"r"]
    assert calls == ["rescue"]


def test_qubo_bucket_key_variants_group_candidates():
    from benchmark_compact_beam_qubo_n0 import _bucket_key_for_candidate

    data = (7, 4, b"parent")
    args = _compact_args(bucket_key="n_depth", bucket_n_band_size=3)

    assert _bucket_key_for_candidate(data, args) == (7, 4)

    args.bucket_key = "f_n"
    assert _bucket_key_for_candidate(data, args) == (11, 7)

    args.bucket_key = "n_band"
    assert _bucket_key_for_candidate(data, args) == (2, 4)


def test_qubo_bucket_frontier_uses_qubo_inside_bucket_and_backfills(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact

    ranked = [
        (b"a", (1, 0, b"p1")),
        (b"b", (1, 0, b"p2")),
        (b"c", (1, 0, b"p3")),
        (b"d", (2, 0, b"p4")),
        (b"e", (2, 0, b"p5")),
    ]
    args = _compact_args(
        selection_policy="qubo_bucket",
        beam_width=4,
        bucket_key="n_depth",
        bucket_core_ratio=0.34,
        bucket_qubo_share=0.5,
        bucket_min_candidates=3,
        bucket_max_qubo_vars=3,
        bucket_qubo_select=1,
        max_qubo_calls=2,
        qubo_min_candidates=64,
    )
    args._qubo_calls_done = 0

    def fake_select_qubo_boosted(candidates, qargs):
        assert qargs.qubo_core_width == 0
        assert qargs.qubo_supplement_width == 1
        assert qargs.max_qubo_vars == 3
        if b"c" in candidates:
            return {b"c"}, 3, 0.01
        return set(), 0, 0.0

    monkeypatch.setattr(compact, "_select_qubo_boosted", fake_select_qubo_boosted)

    selected, selected_by_qubo, qcalls, qvars, qtime = compact._select_qubo_bucket_frontier(
        ranked,
        len(ranked),
        args,
    )

    assert selected == [b"a", b"c", b"b", b"d"]
    assert selected_by_qubo == {b"c"}
    assert qcalls == 1
    assert qvars == 3
    assert qtime == 0.01
    assert args._qubo_calls_done == 1


def test_rank_bucket_topk_keeps_best_items_per_bucket_and_records_truncation():
    from benchmark_compact_beam_qubo_n0 import _rank_bucket_topk_candidates

    candidates = {
        b"a": (4, 2, b"p1"),
        b"b": (4, 2, b"p2"),
        b"c": (4, 2, b"p3"),
        b"d": (5, 2, b"p1"),
        b"e": (5, 2, b"p2"),
        b"f": (5, 2, b"p3"),
    }
    args = _compact_args(rank_bucket_topk=2, rank_bucket_key="n_depth")
    args._rank_pool_truncated_candidates = 0

    ranked = _rank_bucket_topk_candidates(candidates, args)

    assert [code for code, _data in ranked] == [b"a", b"b", b"d", b"e"]
    assert args._rank_pool_truncated_candidates == 2


def test_rank_bucket_topk_can_group_by_n_and_keep_shallow_candidates():
    from benchmark_compact_beam_qubo_n0 import _rank_bucket_topk_candidates

    candidates = {
        b"deep": (4, 7, b"p1"),
        b"shallow": (4, 2, b"p9"),
        b"mid": (4, 4, b"p2"),
        b"other": (5, 1, b"p1"),
    }
    args = _compact_args(rank_bucket_topk=2, rank_bucket_key="n")
    args._rank_pool_truncated_candidates = 0

    ranked = _rank_bucket_topk_candidates(candidates, args)

    assert [code for code, _data in ranked] == [b"shallow", b"mid", b"other"]
    assert args._rank_pool_truncated_candidates == 1


def test_rank_bucket_topk_gate_skips_low_n_layers_and_candidate_trickle():
    from benchmark_compact_beam_qubo_n0 import _rank_next_frontier_candidates

    candidates = {
        b"a": (4, 2, b"p1"),
        b"b": (4, 2, b"p2"),
        b"c": (4, 2, b"p3"),
    }
    args = _compact_args(
        rank_bucket_topk=1,
        rank_bucket_key="n_depth",
        rank_bucket_topk_high_n=10,
        rank_bucket_topk_min_candidates=1,
    )
    args._rank_pool_truncated_candidates = 0

    ranked = _rank_next_frontier_candidates(candidates, args)

    assert [code for code, _data in ranked] == [b"a", b"b", b"c"]
    assert args._rank_pool_truncated_candidates == 0

    args.rank_bucket_topk_high_n = 4
    args.rank_bucket_topk_min_candidates = 4
    ranked = _rank_next_frontier_candidates(candidates, args)

    assert [code for code, _data in ranked] == [b"a", b"b", b"c"]
    assert args._rank_pool_truncated_candidates == 0

    args.rank_bucket_topk_min_candidates = 1
    ranked = _rank_next_frontier_candidates(candidates, args)

    assert [code for code, _data in ranked] == [b"a"]
    assert args._rank_pool_truncated_candidates == 2


def test_qubo_linear_scores_can_use_one_step_future_n_bias():
    from benchmark_target_astar_memory_n0 import _qubo_linear_scores
    from run_business_exact_target_search_sqlite import _encode

    stack_count = 3
    max_height = 3
    better_future, current_n = _encode([[4], [2, 3], [1]], stack_count, max_height)
    worse_future, worse_n = _encode([[3, 4], [2], [1]], stack_count, max_height)
    assert worse_n == current_n

    args = _compact_args(
        qubo_profile="frontier_cohort",
        qubo_rank_weight=0.0,
        qubo_n_weight=0.0,
        qubo_depth_weight=0.0,
        qubo_parent_weight=0.0,
        qubo_next_n_weight=1.0,
        qubo_frontier_n_window=1,
        qubo_frontier_depth_window=1,
        _stack_count=stack_count,
        _max_height=max_height,
    )
    usable = [
        (better_future, (current_n, 4, b"parent-a")),
        (worse_future, (current_n, 4, b"parent-b")),
    ]

    scores = _qubo_linear_scores(usable, usable[0][1], args)

    assert scores[0] < scores[1]


def test_open_runner_accepts_experimental_solver_profiles():
    from benchmark_compact_beam_qubo_n0 import make_compact_args
    from open_cpmp_benchmarks import _args_for_variant

    outer = argparse.Namespace(
        adaptive_candidate_pool=0,
        adaptive_high_frontier=800,
        adaptive_high_n=16,
        adaptive_low_frontier=1200,
        adaptive_low_n=8,
        adaptive_mid_frontier=1000,
        alpha=10.0,
        max_moves=None,
        solver_profile="post_polish_v1",
        time_limit=3.0,
        no_qubo_beam_width=1200,
    )

    args = _args_for_variant(outer, "qubo")

    assert args.solver_profile == "post_polish_v1"
    assert make_compact_args(solver_profile="post_polish_v1").solver_profile == "post_polish_v1"
    assert make_compact_args(solver_profile="frontier_quality_v1").solver_profile == "frontier_quality_v1"
    assert make_compact_args(solver_profile="rescue_qubo_v1").solver_profile == "rescue_qubo_v1"
    assert make_compact_args(solver_profile="light_rescue_qubo_v1").solver_profile == "light_rescue_qubo_v1"
    assert make_compact_args(solver_profile="adaptive_controller_v1").solver_profile == "adaptive_controller_v1"
    assert make_compact_args(solver_profile="scheduler_v1").solver_profile == "scheduler_v1"
    assert make_compact_args(solver_profile="scheduler_topk_v1").solver_profile == "scheduler_topk_v1"
    assert make_compact_args(solver_profile="scheduler_state_v2").solver_profile == "scheduler_state_v2"
    assert make_compact_args(solver_profile="scheduler_state_v3").solver_profile == "scheduler_state_v3"
    assert make_compact_args(solver_profile="scheduler_state_v4").solver_profile == "scheduler_state_v4"
    assert make_compact_args(solver_profile="scheduler_state_v5").solver_profile == "scheduler_state_v5"
    assert (
        make_compact_args(solver_profile="scheduler_state_v5_window_polish").solver_profile
        == "scheduler_state_v5_window_polish"
    )
    assert make_compact_args(solver_profile="scheduler_state_v6").solver_profile == "scheduler_state_v6"
    assert make_compact_args(solver_profile="scheduler_state_v7").solver_profile == "scheduler_state_v7"
    assert make_compact_args(solver_profile="window_polish_v2").solver_profile == "window_polish_v2"


def test_window_polish_v2_profile_keeps_light_safe_window_settings(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="window_polish_v2")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.post_polish_mode == "qubo_windows_v2"
    assert args.post_polish_accept == "better"
    assert args.post_polish_window_select == 1
    assert args.post_polish_window_max_size == 8
    assert args.post_polish_window_repair_expanded == 3000


def test_adaptive_qubo_frontier_width_contracts_high_n_and_expands_low_n():
    from benchmark_compact_beam_qubo_n0 import _adaptive_frontier_width

    args = _compact_args(
        adaptive_high_n=16,
        adaptive_low_n=8,
        adaptive_high_frontier=800,
        adaptive_mid_frontier=1000,
        adaptive_low_frontier=1200,
    )

    assert _adaptive_frontier_width(args, best_n=24) == 800
    assert _adaptive_frontier_width(args, best_n=12) == 1000
    assert _adaptive_frontier_width(args, best_n=6) == 1200


def test_adaptive_signal_controller_contracts_only_under_candidate_pressure():
    from benchmark_compact_beam_qubo_n0 import _adaptive_frontier_width

    args = _compact_args(
        beam_width=1200,
        adaptive_controller="signal_v1",
        adaptive_high_n=16,
        adaptive_low_n=8,
        adaptive_high_frontier=800,
        adaptive_mid_frontier=1000,
        adaptive_low_frontier=1200,
        adaptive_expand_frontier=1400,
        adaptive_contract_pressure=12.0,
        adaptive_mid_pressure=6.0,
        adaptive_plateau_window=3,
        adaptive_cap_slack=12,
        adaptive_min_best_n_ratio=0.45,
    )
    args._initial_n = 40
    args._layer_candidate_count = 24000
    args._layer_depth_slack = 30
    args._layer_parent_top_share = 0.08
    args._best_n_history = [30, 27, 24]

    assert _adaptive_frontier_width(args, best_n=24) == 800
    assert args._adaptive_last_reason == "contract_high_pressure"

    args._layer_candidate_count = 3000

    assert _adaptive_frontier_width(args, best_n=24) == 1200
    assert args._adaptive_last_reason == "wide_low_pressure"


def test_adaptive_signal_controller_expands_when_plateau_slack_is_tight():
    from benchmark_compact_beam_qubo_n0 import _adaptive_frontier_width

    args = _compact_args(
        beam_width=1200,
        adaptive_controller="signal_v1",
        adaptive_high_n=16,
        adaptive_low_n=8,
        adaptive_high_frontier=800,
        adaptive_mid_frontier=1000,
        adaptive_low_frontier=1200,
        adaptive_expand_frontier=1400,
        adaptive_contract_pressure=12.0,
        adaptive_plateau_window=3,
        adaptive_plateau_delta=0,
        adaptive_cap_slack=12,
        adaptive_min_best_n_ratio=0.45,
    )
    args._initial_n = 40
    args._layer_candidate_count = 24000
    args._layer_depth_slack = 8
    args._layer_parent_top_share = 0.08
    args._best_n_history = [24, 24, 24]

    assert _adaptive_frontier_width(args, best_n=24) == 1400
    assert args._adaptive_last_reason == "expand_plateau_tight_slack"


def test_adaptive_signal_controller_uses_stronger_plateau_guard_for_large_stack_instances():
    from benchmark_compact_beam_qubo_n0 import _adaptive_frontier_width

    args = _compact_args(
        beam_width=1200,
        adaptive_controller="signal_v1",
        adaptive_high_n=16,
        adaptive_low_n=8,
        adaptive_high_frontier=800,
        adaptive_mid_frontier=1000,
        adaptive_low_frontier=1200,
        adaptive_expand_frontier=1200,
        adaptive_contract_pressure=0.5,
        adaptive_mid_pressure=0.25,
        adaptive_plateau_window=4,
        adaptive_plateau_delta=0,
        adaptive_cap_slack=4,
        adaptive_large_stack_min_stacks=8,
        adaptive_large_stack_expand_frontier=1400,
        adaptive_large_stack_plateau_window=3,
        adaptive_large_stack_cap_slack=12,
    )
    args._initial_n = 40
    args._layer_candidate_count = 24000
    args._layer_depth_slack = 8
    args._layer_parent_top_share = 0.08
    args._best_n_history = [24, 24, 24]
    args._stack_count = 7

    assert _adaptive_frontier_width(args, best_n=24) == 800
    assert args._adaptive_last_reason == "contract_high_pressure"

    args._stack_count = 10

    assert _adaptive_frontier_width(args, best_n=24) == 1400
    assert args._adaptive_last_reason == "expand_plateau_tight_slack"


def test_adaptive_controller_profile_enables_signal_width_control(monkeypatch):
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile

    args = _compact_args(solver_profile="adaptive_controller_v1")
    monkeypatch.setattr(sys, "argv", ["bench"])

    _apply_solver_profile(args)

    assert args.selection_policy == "adaptive_qubo"
    assert args.adaptive_controller == "signal_v1"
    assert args.adaptive_expand_frontier == 1200
    assert args.adaptive_contract_pressure == 0.5
    assert args.adaptive_plateau_window == 4
    assert args.adaptive_large_stack_min_stacks == 8
    assert args.adaptive_large_stack_expand_frontier == 1400


def test_open_runner_adaptive_variant_uses_adaptive_selection_policy():
    from open_cpmp_benchmarks import _args_for_variant

    instance = Instance(
        "open7",
        ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=7,
        max_height=7,
        container_count=35,
        stacks=[[35, 1], [34, 2], [33, 3], [32, 4], [31, 5], [30, 6], [29, 7]],
    )
    outer = argparse.Namespace(
        adaptive_candidate_pool=0,
        adaptive_high_frontier=800,
        adaptive_high_n=16,
        adaptive_low_frontier=1200,
        adaptive_low_n=8,
        adaptive_mid_frontier=1000,
        alpha=10.0,
        max_moves=None,
        solver_profile="speed_v2",
        time_limit=3.0,
        no_qubo_beam_width=1200,
    )

    args = _args_for_variant(outer, "qubo_adaptive", instance)

    assert args.selection_policy == "adaptive_qubo"
    assert args.adaptive_high_frontier == 800
    assert args.adaptive_low_frontier == 1200
    assert "selection_policy" in args._profile_explicit_keys

    small_instance = Instance(
        "open5",
        ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=5,
        max_height=7,
        container_count=25,
        stacks=[[25, 1], [24, 2], [23, 3], [22, 4], [21, 5]],
    )
    small_args = _args_for_variant(outer, "qubo_adaptive", small_instance)

    assert small_args.selection_policy == "adaptive_qubo"
    assert small_args.adaptive_high_frontier == 1200
    assert small_args.adaptive_mid_frontier == 1200
    assert small_args.adaptive_low_frontier == 1200


def test_open_runner_adaptive_controller_variant_uses_signal_profile():
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile
    from open_cpmp_benchmarks import _args_for_variant, build_parser

    outer = build_parser().parse_args(["--variant", "qubo_adaptive_controller"])

    args = _args_for_variant(outer, "qubo_adaptive_controller")
    _apply_solver_profile(args)

    assert args.solver_profile == "adaptive_controller_v1"
    assert args.selection_policy == "adaptive_qubo"
    assert args.adaptive_controller == "signal_v1"
    assert args.adaptive_expand_frontier == 1200
    assert args.adaptive_large_stack_expand_frontier == 1400


def test_open_runner_wang_rescue_variant_selects_reusable_profile():
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile
    from open_cpmp_benchmarks import _args_for_variant, build_parser

    outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v5_wang_rescue"])
    args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang_rescue")

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v5_wang_rescue"
    assert args.selection_policy == "scheduler_state_qubo"
    assert args.wang_pair_qubo_enabled is True


def test_open_runner_unified_wang_qubo_variant_selects_reusable_profile():
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile
    from open_cpmp_benchmarks import _args_for_variant, _variant_names, build_parser

    outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v7_wang_qubo_unified"])
    args = _args_for_variant(outer, "qubo_scheduler_state_v7_wang_qubo_unified")
    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v7_wang_qubo_unified"
    assert args.wang_rollout_strategy == "baby_lookahead"
    assert "qubo_scheduler_state_v7_wang_qubo_unified" in _variant_names("wang_all")

    outer = build_parser().parse_args(
        [
            "--variant",
            "qubo_scheduler_state_v7_wang_qubo_unified",
            "--no-wang-rollout-enabled",
            "--wang-rollout-strategy",
            "local",
            "--no-wang-rollout-inject-candidates",
            "--wang-rollout-candidate-limit",
            "7",
            "--unified-wang-rollout-bonus",
            "3.5",
        ]
    )
    args = _args_for_variant(outer, "qubo_scheduler_state_v7_wang_qubo_unified")
    _apply_solver_profile(args)

    assert args.wang_rollout_enabled is False
    assert args.wang_rollout_strategy == "local"
    assert args.wang_rollout_inject_candidates is False
    assert args.wang_rollout_candidate_limit == 7
    assert args.unified_wang_rollout_bonus == 3.5

    outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v7_wang_qubo_pool"])
    args = _args_for_variant(outer, "qubo_scheduler_state_v7_wang_qubo_pool")
    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v7_wang_qubo_pool"
    assert args.wang_rollout_inject_candidates is True


def test_open_runner_accepts_feasibility_escape_variant():
    from open_cpmp_benchmarks import _args_for_variant, _variant_names, build_parser

    outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v5_feasibility_escape_polish"])
    args = _args_for_variant(outer, "qubo_scheduler_state_v5_feasibility_escape_polish")

    assert args.solver_profile == "scheduler_state_v5_feasibility_escape_polish"
    assert "qubo_scheduler_state_v5_feasibility_escape_polish" in _variant_names("wang_research_all")


def test_open_runner_accepts_frg_dlts_variant():
    from open_cpmp_benchmarks import _args_for_variant, _variant_names, build_parser

    outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v5_frg_dlts_polish"])
    args = _args_for_variant(outer, "qubo_scheduler_state_v5_frg_dlts_polish")

    assert args.solver_profile == "scheduler_state_v5_frg_dlts_polish"
    assert "qubo_scheduler_state_v5_frg_dlts_polish" in _variant_names("wang_research_all")


def test_open_runner_accepts_bs_frg_qubo_variant():
    from open_cpmp_benchmarks import _args_for_variant, _variant_names, build_parser

    outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v5_bs_frg_qubo_polish"])
    args = _args_for_variant(outer, "qubo_scheduler_state_v5_bs_frg_qubo_polish")

    assert args.solver_profile == "scheduler_state_v5_bs_frg_qubo_polish"
    assert "qubo_scheduler_state_v5_bs_frg_qubo_polish" in _variant_names("wang_research_all")


def test_open_runner_accepts_frg_compound_variant():
    from open_cpmp_benchmarks import _args_for_variant, _variant_names, build_parser

    outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v5_frg_compound_polish"])
    args = _args_for_variant(outer, "qubo_scheduler_state_v5_frg_compound_polish")

    assert args.solver_profile == "scheduler_state_v5_frg_compound_polish"
    assert "qubo_scheduler_state_v5_frg_compound_polish" in _variant_names("wang_research_all")


def test_open_runner_accepts_frg_compound_loose_variant():
    from open_cpmp_benchmarks import _args_for_variant, _variant_names, build_parser

    outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v5_frg_compound_loose_polish"])
    args = _args_for_variant(outer, "qubo_scheduler_state_v5_frg_compound_loose_polish")

    assert args.solver_profile == "scheduler_state_v5_frg_compound_loose_polish"
    assert "qubo_scheduler_state_v5_frg_compound_loose_polish" in _variant_names("wang_research_all")


def test_open_runner_accepts_frg_compound_guarded_variant():
    from open_cpmp_benchmarks import _args_for_variant, _variant_names, build_parser

    outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v5_frg_compound_guarded_polish"])
    args = _args_for_variant(outer, "qubo_scheduler_state_v5_frg_compound_guarded_polish")

    assert args.solver_profile == "scheduler_state_v5_frg_compound_guarded_polish"
    assert "qubo_scheduler_state_v5_frg_compound_guarded_polish" in _variant_names("wang_research_all")


def test_open_runner_passes_frg_compound_overrides():
    from open_cpmp_benchmarks import _args_for_variant, build_parser

    outer = build_parser().parse_args(
        [
            "--variant",
            "qubo_scheduler_state_v5_frg_compound_polish",
            "--frg-compound-escape-width",
            "4",
            "--frg-compound-escape-plateau-window",
            "2",
            "--frg-compound-escape-cap-slack",
            "99",
            "--frg-compound-escape-min-candidates",
            "1",
        ]
    )
    args = _args_for_variant(outer, "qubo_scheduler_state_v5_frg_compound_polish")

    assert args.frg_compound_escape_width == 4
    assert args.frg_compound_escape_plateau_window == 2
    assert args.frg_compound_escape_cap_slack == 99
    assert args.frg_compound_escape_min_candidates == 1


def test_open_runner_accepts_wang2017_task_variant():
    from open_cpmp_benchmarks import _args_for_variant, _variant_names, build_parser

    outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v5_wang2017_task_polish"])
    args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang2017_task_polish")

    assert args.solver_profile == "scheduler_state_v5_wang2017_task_polish"
    assert "qubo_scheduler_state_v5_wang2017_task_polish" in _variant_names("wang_research_all")


def test_open_runner_accepts_wang2017_task_guarded_variant():
    from open_cpmp_benchmarks import _args_for_variant, _variant_names, build_parser

    outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v5_wang2017_task_guarded_polish"])
    args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang2017_task_guarded_polish")

    assert args.solver_profile == "scheduler_state_v5_wang2017_task_guarded_polish"
    assert "qubo_scheduler_state_v5_wang2017_task_guarded_polish" in _variant_names("wang_research_all")


def test_open_runner_accepts_wang_rescue_light_polish_variant():
    from open_cpmp_benchmarks import _args_for_variant, _variant_names, build_parser

    outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v5_wang_rescue_light_polish"])
    args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang_rescue_light_polish")

    assert args.solver_profile == "scheduler_state_v5_wang_rescue_light_polish"
    assert "qubo_scheduler_state_v5_wang_rescue_light_polish" in _variant_names("wang_research_all")


def test_open_runner_accepts_wang2017_task_guarded_portfolio_variant():
    from open_cpmp_benchmarks import _variant_names, build_parser

    build_parser().parse_args(["--variant", "qubo_scheduler_state_v5_wang2017_task_guarded_portfolio"])

    assert "qubo_scheduler_state_v5_wang2017_task_guarded_portfolio" in _variant_names("wang_research_all")


def test_open_no_regret_row_keeps_valid_baseline_and_selects_shorter_valid_guarded():
    from open_cpmp_benchmarks import _select_open_no_regret_row

    baseline = {
        "solver_profile": "scheduler_state_v5_wang_rescue_polish",
        "moves": 10,
        "final_n": 0,
        "runtime_sec": 1.0,
        "business_ok": True,
        "goal_validation_ok": True,
        "full_quality_penalty": 10,
    }
    invalid_guarded = {
        "solver_profile": "scheduler_state_v5_wang2017_task_guarded_polish",
        "moves": "",
        "final_n": 2,
        "runtime_sec": 2.0,
        "business_ok": False,
        "goal_validation_ok": False,
        "full_quality_penalty": 50,
    }
    shorter_guarded = {
        "solver_profile": "scheduler_state_v5_wang2017_task_guarded_polish",
        "moves": 9,
        "final_n": 0,
        "runtime_sec": 2.0,
        "business_ok": True,
        "goal_validation_ok": True,
        "full_quality_penalty": 9,
    }

    assert _select_open_no_regret_row([baseline, invalid_guarded]) is baseline
    assert _select_open_no_regret_row([baseline, shorter_guarded]) is shorter_guarded


def test_open_runner_case_ids_filter_accepts_full_or_numeric_ids():
    from open_cpmp_benchmarks import _case_id_selected, build_parser

    args = build_parser().parse_args(["--case-ids", "4,5x5_cv2_test_10"])

    assert _case_id_selected("5x5_cv1_test_4", args.case_ids)
    assert _case_id_selected("5x5_cv2_test_10", args.case_ids)
    assert not _case_id_selected("5x5_cv1_test_5", args.case_ids)


def test_open_runner_accepts_wang_rescue_successor_cache_polish_variant():
    from open_cpmp_benchmarks import _args_for_variant, _variant_names, build_parser

    outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v5_wang_rescue_successor_cache_polish"])
    args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang_rescue_successor_cache_polish")

    assert args.solver_profile == "scheduler_state_v5_wang_rescue_successor_cache_polish"
    assert "qubo_scheduler_state_v5_wang_rescue_successor_cache_polish" in _variant_names("wang_research_all")


def test_open_runner_wang_all_expands_current_and_wang_variants():
    from open_cpmp_benchmarks import _variant_names, build_parser

    build_parser().parse_args(["--variant", "wang_all"])
    build_parser().parse_args(["--variant", "wang_research_all"])

    assert _variant_names("wang_all") == [
        "qubo",
        "qubo_scheduler_state_v5",
        "qubo_scheduler_state_v5_wang_pair",
        "qubo_scheduler_state_v5_wang_features",
        "qubo_scheduler_state_v5_wang_rescue",
        "qubo_scheduler_state_v5_wang_rescue_suffix_polish",
        "qubo_scheduler_state_v5_wang_rescue_polish",
        "qubo_scheduler_state_v5_wang_rescue_successor_cache_polish",
        "qubo_scheduler_state_v5_wang_rescue_light_polish",
        "qubo_scheduler_state_v5_wang2017_task_polish",
        "qubo_scheduler_state_v5_wang2017_task_guarded_polish",
        "qubo_scheduler_state_v5_wang2017_task_guarded_portfolio",
        "qubo_scheduler_state_v5_wang_hybrid",
        "qubo_scheduler_state_v7_wang_qubo_unified",
    ]
    assert _variant_names("wang_research_all") == [
        "qubo_scheduler_state_v5_wang_rescue_polish",
        "qubo_scheduler_state_v5_wang_rescue_successor_cache_polish",
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


def test_open_runner_accepts_wang_feature_and_rescue_polish_variants():
    from open_cpmp_benchmarks import _args_for_variant, _variant_names

    outer = argparse.Namespace(
        adaptive_candidate_pool=0,
        adaptive_high_frontier=800,
        adaptive_high_n=16,
        adaptive_low_frontier=1200,
        adaptive_low_n=8,
        adaptive_mid_frontier=1000,
        alpha=10.0,
        max_moves=None,
        solver_profile="speed_v2",
        time_limit=10.0,
        no_qubo_beam_width=1200,
    )

    feature_args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang_features")
    suffix_polish_args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang_rescue_suffix_polish")
    polish_args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang_rescue_polish")
    dynamic_args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang_rescue_dynamic_polish")
    risk_args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang_rescue_risk_polish")
    selective_args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang_rescue_selective_window_polish")
    wang2017_task_args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang2017_task_polish")
    wang2017_guarded_args = _args_for_variant(outer, "qubo_scheduler_state_v5_wang2017_task_guarded_polish")

    assert feature_args.solver_profile == "scheduler_state_v5_wang_features"
    assert suffix_polish_args.solver_profile == "scheduler_state_v5_wang_rescue_suffix_polish"
    assert polish_args.solver_profile == "scheduler_state_v5_wang_rescue_polish"
    assert dynamic_args.solver_profile == "scheduler_state_v5_wang_rescue_dynamic_polish"
    assert risk_args.solver_profile == "scheduler_state_v5_wang_rescue_risk_polish"
    assert selective_args.solver_profile == "scheduler_state_v5_wang_rescue_selective_window_polish"
    assert wang2017_task_args.solver_profile == "scheduler_state_v5_wang2017_task_polish"
    assert wang2017_guarded_args.solver_profile == "scheduler_state_v5_wang2017_task_guarded_polish"
    assert "qubo_scheduler_state_v5_wang_features" in _variant_names("wang_all")
    assert "qubo_scheduler_state_v5_wang_rescue_suffix_polish" in _variant_names("wang_all")
    assert "qubo_scheduler_state_v5_wang_rescue_polish" in _variant_names("wang_all")


def test_open_runner_bucket_variant_passes_bucket_selection_args():
    from open_cpmp_benchmarks import _args_for_variant, build_parser

    outer = build_parser().parse_args(
        [
            "--variant",
            "qubo_bucket",
            "--bucket-key",
            "n_band",
            "--bucket-n-band-size",
            "3",
            "--bucket-core-ratio",
            "0.4",
            "--bucket-qubo-share",
            "0.6",
            "--bucket-max-qubo-vars",
            "80",
            "--bucket-qubo-select",
            "12",
            "--bucket-min-candidates",
            "24",
        ]
    )

    args = _args_for_variant(outer, "qubo_bucket")

    assert args.selection_policy == "qubo_bucket"
    assert args.bucket_key == "n_band"
    assert args.bucket_n_band_size == 3
    assert args.bucket_core_ratio == 0.4
    assert args.bucket_qubo_share == 0.6
    assert args.bucket_max_qubo_vars == 80
    assert args.bucket_qubo_select == 12
    assert args.bucket_min_candidates == 24
    assert "selection_policy" in args._profile_explicit_keys


def test_open_runner_scheduler_variants_select_reusable_profiles():
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile
    from open_cpmp_benchmarks import _args_for_variant, build_parser

    outer = build_parser().parse_args(
        [
            "--variant",
            "qubo_scheduler",
            "--scheduler-adaptive-min-stacks",
            "7",
            "--scheduler-adaptive-high-n",
            "18",
        ]
    )
    args = _args_for_variant(outer, "qubo_scheduler")

    assert args.solver_profile == "scheduler_v1"
    assert args.scheduler_adaptive_high_n == 18

    _apply_solver_profile(args)

    assert args.selection_policy == "scheduler_qubo"
    assert args.rescue_enabled is True
    assert args.scheduler_adaptive_cv1_only is False
    assert args.scheduler_adaptive_max_duplication_ratio == 0.0
    assert args.scheduler_adaptive_min_best_n_ratio == 0.45

    topk_outer = build_parser().parse_args(["--variant", "qubo_scheduler_topk"])
    topk_args = _args_for_variant(topk_outer, "qubo_scheduler_topk")

    _apply_solver_profile(topk_args)

    assert topk_args.solver_profile == "scheduler_topk_v1"
    assert topk_args.selection_policy == "scheduler_qubo"
    assert topk_args.rank_bucket_topk == 64
    assert topk_args.rank_bucket_topk_high_n == 18
    assert topk_args.rank_bucket_topk_min_candidates == 10000
    assert topk_args.scheduler_adaptive_cv1_only is False
    assert topk_args.scheduler_adaptive_max_duplication_ratio == 0.0

    state_outer = build_parser().parse_args(["--variant", "qubo_scheduler_state"])
    state_args = _args_for_variant(state_outer, "qubo_scheduler_state")

    _apply_solver_profile(state_args)

    assert state_args.solver_profile == "scheduler_state_v2"
    assert state_args.selection_policy == "scheduler_state_qubo"
    assert state_args.scheduler_state_controller == "state_v2"
    assert state_args.scheduler_state_progress_delta == 0
    assert state_args.scheduler_state_max_contract_pressure == 40.0
    assert state_args.scheduler_state_failed_contract_cooldown == 0

    state_v3_outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v3"])
    state_v3_args = _args_for_variant(state_v3_outer, "qubo_scheduler_state_v3")

    _apply_solver_profile(state_v3_args)

    assert state_v3_args.solver_profile == "scheduler_state_v3"
    assert state_v3_args.selection_policy == "scheduler_state_qubo"
    assert state_v3_args.scheduler_state_failed_contract_cooldown == 0
    assert state_v3_args.scheduler_state_post_rescue_plateau_window == 0
    assert state_v3_args.scheduler_state_post_rescue_min_ratio == 0.90
    assert state_v3_args.scheduler_state_post_rescue_action == "protect"
    assert state_v3_args.scheduler_state_initial_plateau_window == 5
    assert state_v3_args.scheduler_state_initial_plateau_min_ratio == 0.98
    assert state_v3_args.scheduler_state_initial_plateau_action == "rescue"
    assert state_v3_args.scheduler_state_initial_plateau_lock is True
    assert state_v3_args.scheduler_state_initial_plateau_min_duplication == 0.10

    state_v4_outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v4"])
    state_v4_args = _args_for_variant(state_v4_outer, "qubo_scheduler_state_v4")

    _apply_solver_profile(state_v4_args)

    assert state_v4_args.solver_profile == "scheduler_state_v4"
    assert state_v4_args.selection_policy == "scheduler_state_qubo"
    assert state_v4_args.scheduler_state_post_rescue_plateau_window == 0
    assert state_v4_args.scheduler_state_initial_plateau_window == 5
    assert state_v4_args.scheduler_state_initial_plateau_action == "rescue"
    assert state_v4_args.scheduler_state_initial_plateau_lock is True
    assert state_v4_args.scheduler_state_initial_plateau_min_duplication == 0.0

    state_v5_outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v5"])
    state_v5_args = _args_for_variant(state_v5_outer, "qubo_scheduler_state_v5")

    _apply_solver_profile(state_v5_args)

    assert state_v5_args.solver_profile == "scheduler_state_v5"
    assert state_v5_args.selection_policy == "scheduler_state_qubo"
    assert state_v5_args.scheduler_state_post_rescue_plateau_window == 0
    assert state_v5_args.scheduler_state_initial_plateau_window == 5
    assert state_v5_args.scheduler_state_initial_plateau_action == "rescue"
    assert state_v5_args.scheduler_state_initial_plateau_lock is True
    assert state_v5_args.scheduler_state_initial_plateau_min_duplication == 0.0
    assert state_v5_args.scheduler_state_initial_plateau_unique_min_n == 24

    state_v5_window_outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v5_window"])
    state_v5_window_args = _args_for_variant(state_v5_window_outer, "qubo_scheduler_state_v5_window")

    _apply_solver_profile(state_v5_window_args)

    assert state_v5_window_args.solver_profile == "scheduler_state_v5_window_polish"
    assert state_v5_window_args.selection_policy == "scheduler_state_qubo"
    assert state_v5_window_args.scheduler_state_initial_plateau_unique_min_n == 24
    assert state_v5_window_args.post_polish_mode == "qubo_windows_v2"
    assert state_v5_window_args.post_polish_window_select == 1

    state_v6_outer = build_parser().parse_args(["--variant", "qubo_scheduler_state_v6"])
    state_v6_args = _args_for_variant(state_v6_outer, "qubo_scheduler_state_v6")

    _apply_solver_profile(state_v6_args)

    assert state_v6_args.solver_profile == "scheduler_state_v6"
    assert state_v6_args.selection_policy == "scheduler_state_qubo"
    assert state_v6_args.post_polish_mode == "qubo_windows_v2"
    assert state_v6_args.post_polish_window_min_stacks == 7

    stream_override_outer = build_parser().parse_args(
        ["--variant", "qubo_scheduler_state_v5", "--stream-bucket-topk", "96", "--rank-bucket-key", "n_depth"]
    )
    stream_override_args = _args_for_variant(stream_override_outer, "qubo_scheduler_state_v5")

    _apply_solver_profile(stream_override_args)

    assert stream_override_args.solver_profile == "scheduler_state_v5"
    assert stream_override_args.stream_bucket_topk == 96
    assert stream_override_args.rank_bucket_key == "n_depth"

    global_stream_outer = build_parser().parse_args(
        [
            "--variant",
            "qubo_scheduler_state_v5",
            "--stream-candidate-limit",
            "9000",
            "--stream-candidate-trim-factor",
            "1.5",
        ]
    )
    global_stream_args = _args_for_variant(global_stream_outer, "qubo_scheduler_state_v5")

    _apply_solver_profile(global_stream_args)

    assert global_stream_args.solver_profile == "scheduler_state_v5"
    assert global_stream_args.stream_candidate_limit == 9000
    assert global_stream_args.stream_candidate_trim_factor == 1.5

    adaptive_rescue_outer = build_parser().parse_args(["--variant", "qubo_adaptive_rescue"])
    adaptive_rescue_args = _args_for_variant(adaptive_rescue_outer, "qubo_adaptive_rescue")

    _apply_solver_profile(adaptive_rescue_args)

    assert adaptive_rescue_args.solver_profile == "adaptive_rescue_v1"
    assert adaptive_rescue_args.selection_policy == "adaptive_qubo"
    assert adaptive_rescue_args.rescue_enabled is True

    adaptive_polish_outer = build_parser().parse_args(["--variant", "qubo_adaptive_polish"])
    adaptive_polish_args = _args_for_variant(adaptive_polish_outer, "qubo_adaptive_polish")

    _apply_solver_profile(adaptive_polish_args)

    assert adaptive_polish_args.solver_profile == "adaptive_polish_v1"
    assert adaptive_polish_args.selection_policy == "adaptive_qubo"
    assert adaptive_polish_args.post_polish_passes == 3

    explicit_outer = build_parser().parse_args(
        [
            "--variant",
            "qubo_scheduler_topk",
            "--rank-bucket-topk",
            "2",
            "--rank-bucket-key",
            "n_depth",
            "--rank-bucket-topk-high-n",
            "20",
            "--rank-bucket-topk-min-candidates",
            "123",
            "--scheduler-adaptive-max-duplication-ratio",
            "0.5",
            "--scheduler-adaptive-min-best-n-ratio",
            "0.4",
            "--scheduler-adaptive-plateau-window",
            "4",
            "--scheduler-adaptive-cap-slack",
            "9",
        ]
    )
    explicit_args = _args_for_variant(explicit_outer, "qubo_scheduler_topk")

    _apply_solver_profile(explicit_args)

    assert explicit_args.rank_bucket_topk == 2
    assert explicit_args.rank_bucket_key == "n_depth"
    assert explicit_args.rank_bucket_topk_high_n == 20
    assert explicit_args.rank_bucket_topk_min_candidates == 123
    assert explicit_args.scheduler_adaptive_max_duplication_ratio == 0.5
    assert explicit_args.scheduler_adaptive_min_best_n_ratio == 0.4
    assert explicit_args.scheduler_adaptive_plateau_window == 4
    assert explicit_args.scheduler_adaptive_cap_slack == 9


def test_no_regret_portfolio_keeps_valid_baseline_against_worse_or_invalid_challengers():
    from benchmark_qubo_no_regret_portfolio_n0 import PortfolioCandidate, _select_no_regret_candidate

    baseline = PortfolioCandidate("stable", moves=10, runtime_sec=1.0, valid=True, final_n=0, status="ok")
    invalid_better = PortfolioCandidate("state_light", moves=8, runtime_sec=2.0, valid=False, final_n=1, status="bad")
    worse_valid = PortfolioCandidate("state_medium", moves=12, runtime_sec=0.5, valid=True, final_n=0, status="ok")

    selected = _select_no_regret_candidate([baseline, invalid_better, worse_valid])

    assert selected.profile == "stable"


def test_no_regret_portfolio_selects_shorter_valid_challenger_and_prefers_stable_tie():
    from benchmark_qubo_no_regret_portfolio_n0 import PortfolioCandidate, _select_no_regret_candidate

    baseline = PortfolioCandidate("stable", moves=10, runtime_sec=1.0, valid=True, final_n=0, status="ok")
    shorter = PortfolioCandidate("state_light", moves=9, runtime_sec=3.0, valid=True, final_n=0, status="ok")

    assert _select_no_regret_candidate([baseline, shorter]).profile == "state_light"

    same_quality = PortfolioCandidate("state_medium", moves=10, runtime_sec=1.0, valid=True, final_n=0, status="ok")

    assert _select_no_regret_candidate([same_quality, baseline]).profile == "stable"


def test_no_regret_portfolio_full_quality_penalizes_invalid_and_unsolved_rows():
    from benchmark_qubo_no_regret_portfolio_n0 import _profile_full_quality_summary

    rows = [
        {
            "all_profile_moves_json": '{"stable": 10, "wang_pair": 9}',
            "all_profile_runtime_json": '{"stable": 1.0, "wang_pair": 1.5}',
            "all_profile_valid_json": '{"stable": true, "wang_pair": false}',
            "all_profile_final_n_json": '{"stable": 0, "wang_pair": 1}',
            "n0": 8,
            "cap_13": 10,
        },
        {
            "all_profile_moves_json": '{"stable": 12, "wang_pair": 11}',
            "all_profile_runtime_json": '{"stable": 1.2, "wang_pair": 1.3}',
            "all_profile_valid_json": '{"stable": true, "wang_pair": true}',
            "all_profile_final_n_json": '{"stable": 0, "wang_pair": 0}',
            "n0": 8,
            "cap_13": 10,
        },
    ]

    summary = _profile_full_quality_summary(rows, penalty_factor=4)

    assert summary["stable"]["invalid_or_unsolved"] == 0
    assert summary["stable"]["penalized_moves_total"] == 22
    assert summary["wang_pair"]["invalid_or_unsolved"] == 1
    assert summary["wang_pair"]["penalized_moves_total"] == 53


def test_no_regret_portfolio_accepts_wang_candidate_profiles():
    from benchmark_qubo_no_regret_portfolio_n0 import _args_for_portfolio_profile, _parse_portfolio_profiles

    profiles = _parse_portfolio_profiles(
        "scheduler_state_v5,wang_incumbent,scheduler_state_v5_wang_pair,scheduler_state_v5_wang_hybrid"
    )

    assert profiles == [
        "stable",
        "scheduler_state_v5",
        "wang_incumbent",
        "scheduler_state_v5_wang_pair",
        "scheduler_state_v5_wang_hybrid",
    ]

    outer = argparse.Namespace(
        alpha=1.3,
        max_moves=None,
        solver_profile="speed_v2",
        time_limit=2.0,
        portfolio_time_limit_per_variant=None,
    )
    wang_args = _args_for_portfolio_profile(outer, "wang_incumbent")
    pair_args = _args_for_portfolio_profile(outer, "scheduler_state_v5_wang_pair")

    assert wang_args.solver_profile == "speed_v2"
    assert wang_args.wang_incumbent_enabled is True
    assert pair_args.solver_profile == "scheduler_state_v5_wang_pair_qubo"
    assert pair_args.wang_pair_qubo_enabled is True


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


def test_stream_candidate_batch_trim_keeps_global_top_k():
    from benchmark_compact_beam_qubo_n0 import _trim_stream_candidates_batch

    args = _compact_args()
    candidates = {
        b"z": (1, 1, b"a"),
        b"a": (1, 1, b"z"),
        b"b": (2, 1, b"a"),
        b"c": (0, 5, b"a"),
    }

    pruned = _trim_stream_candidates_batch(candidates, 2, args)

    assert pruned == 2
    assert set(candidates) == {b"z", b"c"}


def test_stream_bucket_topk_keeps_top_k_per_bucket_during_generation():
    from benchmark_compact_beam_qubo_n0 import _offer_stream_bucket_candidate

    args = _compact_args(stream_bucket_topk=2, rank_bucket_key="n_depth")
    candidates = {}
    bucket_heaps = {}
    bucket_members = {}
    rows = [
        (b"a", (4, 2, b"p1")),
        (b"b", (4, 2, b"p2")),
        (b"c", (4, 2, b"p3")),
        (b"d", (5, 2, b"p1")),
        (b"e", (5, 2, b"p2")),
        (b"f", (5, 2, b"p3")),
    ]

    pruned = 0
    for seq, (code, data) in enumerate(rows):
        pruned += _offer_stream_bucket_candidate(
            candidates,
            bucket_heaps,
            bucket_members,
            code,
            data,
            args,
            seq,
        )

    assert pruned == 2
    assert set(candidates) == {b"a", b"b", b"d", b"e"}
    assert bucket_members[(4, 2)] == {b"a", b"b"}
    assert bucket_members[(5, 2)] == {b"d", b"e"}


def test_stream_bucket_topk_updates_members_when_candidate_moves_bucket():
    from benchmark_compact_beam_qubo_n0 import _offer_stream_bucket_candidate

    args = _compact_args(stream_bucket_topk=1, rank_bucket_key="n_depth")
    candidates = {}
    bucket_heaps = {}
    bucket_members = {}

    _offer_stream_bucket_candidate(candidates, bucket_heaps, bucket_members, b"x", (5, 2, b"p2"), args, 0)
    _offer_stream_bucket_candidate(candidates, bucket_heaps, bucket_members, b"y", (5, 2, b"p1"), args, 1)
    _offer_stream_bucket_candidate(candidates, bucket_heaps, bucket_members, b"x", (4, 1, b"p1"), args, 2)

    assert candidates[b"x"] == (4, 1, b"p1")
    assert set(candidates) == {b"x", b"y"}
    assert bucket_members[(4, 1)] == {b"x"}
    assert bucket_members[(5, 2)] == {b"y"}


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


def test_post_polish_windows_v2_scores_plateaus_and_early_windows():
    from benchmark_compact_beam_qubo_n0 import _post_polish_windows_v2

    codes = [bytes([idx]) for idx in range(8)]
    n_values = [7, 7, 8, 7, 6, 6, 4, 0]
    args = _compact_args(
        post_polish_focus_n=0,
        post_polish_tail_span=2,
        post_polish_window_min_size=3,
        post_polish_window_max_size=4,
        post_polish_window_stride=1,
        post_polish_window_budget=20,
    )

    windows = _post_polish_windows_v2(codes, n_values, args)

    assert windows
    assert any(window.start == 0 for window in windows)
    assert windows[0].start == 0
    assert windows[0].plateau_count >= 1
    assert windows[0].increase_count >= 1


def test_post_polish_windows_v2_rejects_equal_length_bridge_by_default(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact

    codes = [b"s0", b"s1", b"s2", b"s3"]
    args = _compact_args(
        post_polish_mode="qubo_windows_v2",
        post_polish_qubo=True,
        post_polish_passes=1,
    )
    window = compact.PostPolishWindow(start=0, end=2, old_len=2, min_n=2, max_n=3, score=10.0)

    monkeypatch.setattr(compact, "_path_n_values", lambda *_args: [3, 3, 2, 0])
    monkeypatch.setattr(compact, "_post_polish_windows_v2", lambda *_args: [window])
    monkeypatch.setattr(compact, "_select_post_polish_windows_v2_qubo", lambda windows, _args: (windows, 0, 0.0))
    monkeypatch.setattr(compact, "_bridge_shorter_path", lambda *_args: ([b"s0", b"x", b"s2"], 4))
    monkeypatch.setattr(compact, "_path_business_ok", lambda *_args: True)

    polished, stats = compact._post_polish_codes_with_qubo(
        codes,
        stack_count=3,
        max_height=3,
        cap=99,
        args=args,
        deadline=float("inf"),
    )

    assert polished == codes
    assert stats.window_attempts == 1
    assert stats.window_successes == 0
    assert stats.window_moves_saved == 0


def test_post_polish_windows_v2_select_zero_skips_window_repair():
    from benchmark_compact_beam_qubo_n0 import PostPolishWindow, _select_post_polish_windows_v2_qubo

    windows = [
        PostPolishWindow(start=0, end=3, old_len=3, min_n=2, max_n=4, score=12.0),
        PostPolishWindow(start=3, end=6, old_len=3, min_n=1, max_n=3, score=10.0),
    ]
    args = _compact_args(post_polish_window_select=0)

    selected, qvars, qtime = _select_post_polish_windows_v2_qubo(windows, args)

    assert selected == []
    assert qvars == 0
    assert qtime == 0.0


def test_post_polish_window_min_stacks_falls_back_to_legacy_for_small_instances(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact

    codes = [b"s0", b"s1", b"s2", b"s3"]
    args = _compact_args(
        post_polish_mode="qubo_windows_v2",
        post_polish_qubo=True,
        post_polish_window_min_stacks=7,
        post_polish_passes=1,
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("window polish should be skipped below post_polish_window_min_stacks")

    monkeypatch.setattr(compact, "_post_polish_codes_with_qubo_windows_v2", fail_if_called)
    monkeypatch.setattr(compact, "_path_n_values", lambda *_args: [3, 2, 1, 0])
    monkeypatch.setattr(compact, "_post_polish_suffixes", lambda *_args: [])
    monkeypatch.setattr(compact, "_post_polish_windows", lambda *_args: [])

    polished, stats = compact._post_polish_codes_with_qubo(
        codes,
        stack_count=5,
        max_height=5,
        cap=99,
        args=args,
        deadline=float("inf"),
    )

    assert polished == codes
    assert stats.attempts == 0
    assert stats.window_attempts == 0


def test_post_polish_windows_v2_accepts_strictly_shorter_bridge_and_records_metrics(monkeypatch):
    import benchmark_compact_beam_qubo_n0 as compact

    codes = [b"s0", b"s1", b"s2", b"s3"]
    args = _compact_args(
        post_polish_mode="qubo_windows_v2",
        post_polish_qubo=True,
        post_polish_passes=1,
    )
    window = compact.PostPolishWindow(start=0, end=2, old_len=2, min_n=2, max_n=3, score=10.0)

    monkeypatch.setattr(compact, "_path_n_values", lambda *_args: [3, 3, 2, 0])
    monkeypatch.setattr(compact, "_post_polish_windows_v2", lambda *_args: [window])
    monkeypatch.setattr(compact, "_select_post_polish_windows_v2_qubo", lambda windows, _args: (windows, 5, 0.01))
    monkeypatch.setattr(compact, "_bridge_shorter_path", lambda *_args: ([b"s0", b"s2"], 4))
    monkeypatch.setattr(compact, "_path_business_ok", lambda *_args: True)

    polished, stats = compact._post_polish_codes_with_qubo(
        codes,
        stack_count=3,
        max_height=3,
        cap=99,
        args=args,
        deadline=float("inf"),
    )

    assert polished == [b"s0", b"s2", b"s3"]
    assert stats.qcalls == 1
    assert stats.qvars == 5
    assert stats.window_candidates == 1
    assert stats.window_qubo_selected == 1
    assert stats.window_attempts == 1
    assert stats.window_successes == 1
    assert stats.window_moves_saved == 1


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
    args = _compact_args(qubo_influence_profile="medium", rank_pool_size=5000, stream_bucket_topk=24)

    row = _row_dict_from_result(result, "A1", "01", args)

    assert row["profile"] == "speed_v2"
    assert row["solver_profile"] == "speed_v2"
    assert row["qubo_influence_profile"] == "medium"
    assert row["peak_candidates"] == 5
    assert row["open_count"] == 7
    assert row["qubo_selected_on_path"] == 1
    assert row["qubo_ancestor_on_path"] == 3
    assert row["stream_bucket_topk"] == 24
    assert row["rank_pool_size"] == 5000
    assert row["rank_pool_truncated_candidates"] == 14
    assert "peak_candidates=5" in row["status"]
