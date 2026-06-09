from __future__ import annotations

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
