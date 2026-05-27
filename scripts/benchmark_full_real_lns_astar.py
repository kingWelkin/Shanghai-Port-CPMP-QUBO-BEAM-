"""LNS and weighted-A* benchmarks for full real paired business cases."""

from __future__ import annotations

import csv
import heapq
import time
from pathlib import Path

from benchmark_full_real_business_algorithms import (
    ROOT,
    TrajectoryResult,
    _instance_from_zone,
    _result_from_state,
    discover_paired_cases,
)

from container_reshuffle.core import Instance, State, apply_move, create_initial_state, generate_legal_moves, hash_state
from container_reshuffle.features import (
    bad_count,
    dirty_stack_count,
    disorder_box_count,
    entropy_limit,
    entropy_value,
    inversion_count,
    settled_box_count,
    well_prefix_gain,
)
from container_reshuffle.qubo.beam_selection import QuboBeamSelectionBuilder
from container_reshuffle.qubo.sampler import QuboSamplerConfig, SimulatedAnnealingQuboSampler
from container_reshuffle.scoring import cpmp_business_feasible, dominance_value, score_state
from container_reshuffle.solvers.base import SolverConfig
from container_reshuffle.solvers.beam import Candidate


def _rank_state(instance: Instance, state: State) -> tuple[int, int, int, int, int]:
    return (
        disorder_box_count(instance, state),
        state.entropy_max,
        state.move_count,
        state.disorder_total,
        inversion_count(instance, state),
    )


def _move_key(instance: Instance, current: State, nxt: State, config: SolverConfig) -> tuple:
    n_current = disorder_box_count(instance, current)
    n_next = disorder_box_count(instance, nxt)
    return (
        n_next,
        -(n_current - n_next),
        entropy_value(instance, nxt),
        nxt.entropy_max,
        inversion_count(instance, nxt),
        bad_count(instance, nxt),
        dirty_stack_count(instance, nxt),
        -settled_box_count(instance, nxt),
        score_state(instance, nxt, config.score_weights),
        nxt.move_count,
    )


def _state_priority(instance: Instance, state: State, weight: float) -> tuple:
    n_value = disorder_box_count(instance, state)
    return (
        n_value * 10_000
        + weight * inversion_count(instance, state)
        + 0.35 * bad_count(instance, state)
        + 0.25 * dirty_stack_count(instance, state)
        - 0.08 * settled_box_count(instance, state)
        + 0.015 * state.disorder_total
        + 0.10 * state.move_count
        - 0.03 * entropy_value(instance, state),
        n_value,
        state.entropy_max,
        state.move_count,
    )


AGGRESSIVE_ASTAR_MODES = ("n_budget", "score", "inversion", "short", "entropy")
ASTAR_MODES = ("legacy",) + AGGRESSIVE_ASTAR_MODES


def _astar_priority(instance: Instance, state: State, config: SolverConfig, mode: str) -> tuple:
    n_value = disorder_box_count(instance, state)
    g_value = entropy_value(instance, state)
    cap = entropy_limit(instance, config.entropy_alpha)
    inv = inversion_count(instance, state)
    bad = bad_count(instance, state)
    dirty = dirty_stack_count(instance, state)
    settled = settled_box_count(instance, state)
    if mode == "legacy":
        return _state_priority(instance, state, weight=0.45)
    common = (
        n_value,
        -g_value,
        state.entropy_max,
        state.disorder_total,
        inv,
        bad,
        dirty,
        -settled,
        state.move_count,
    )
    if mode == "score":
        return (score_state(instance, state, config.score_weights),) + common
    if mode == "inversion":
        return (n_value, inv, bad, dirty, -g_value, state.move_count, state.disorder_total)
    if mode == "short":
        return (n_value, state.move_count, inv, bad, -g_value, state.disorder_total)
    if mode == "entropy":
        return (n_value, state.entropy_max, cap - g_value, -g_value, inv, bad, state.move_count)
    return common


def _weighted_astar_successor_keys(
    instance: Instance,
    current: State,
    nxt: State,
    config: SolverConfig,
    *,
    include_legacy_rule: bool,
) -> list[tuple]:
    n_current = disorder_box_count(instance, current)
    n_next = disorder_box_count(instance, nxt)
    g_next = entropy_value(instance, nxt)
    cap = entropy_limit(instance, config.entropy_alpha)
    gain = n_current - n_next
    prefix_gain = well_prefix_gain(instance, current, nxt)
    inv = inversion_count(instance, nxt)
    bad = bad_count(instance, nxt)
    dirty = dirty_stack_count(instance, nxt)
    settled = settled_box_count(instance, nxt)
    score = score_state(instance, nxt, config.score_weights)
    common = (
        n_next,
        -g_next,
        nxt.entropy_max,
        nxt.disorder_total,
        inv,
        bad,
        dirty,
        -settled,
        nxt.move_count,
    )
    keys = [
        common,
        (score,) + common,
        (-gain, n_next, -g_next, inv, bad, nxt.move_count),
        (-prefix_gain, n_next, -g_next, inv, bad, dirty, nxt.move_count),
        (inv, bad, dirty, n_next, -g_next, nxt.move_count),
        (cap - g_next, n_next, inv, bad, nxt.move_count),
    ]
    if include_legacy_rule:
        keys.insert(0, _move_key(instance, current, nxt, config))
    return keys


def _select_weighted_astar_successors(
    instance: Instance,
    state: State,
    config: SolverConfig,
    *,
    per_rule_limit: int,
    include_legacy_rule: bool = True,
) -> tuple[list[State], int]:
    evaluated = []
    for mv in generate_legal_moves(instance, state):
        try:
            nxt = apply_move(instance, state, mv)
        except ValueError:
            continue
        if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
            continue
        evaluated.append(
            (
                nxt,
                _weighted_astar_successor_keys(
                    instance,
                    state,
                    nxt,
                    config,
                    include_legacy_rule=include_legacy_rule,
                ),
            )
        )

    selected: dict[tuple, State] = {}
    rule_count = len(evaluated[0][1]) if evaluated else 0
    for rule_idx in range(rule_count):
        ranked = sorted(evaluated, key=lambda item: item[1][rule_idx])
        for nxt, _keys in ranked[:per_rule_limit]:
            selected[hash_state(instance, nxt, config.enable_canonical_hash)] = nxt
    return list(selected.values()), len(evaluated)


def _replay_prefix(instance: Instance, moves, prefix_len: int) -> State:
    state = create_initial_state(instance)
    for move in moves[:prefix_len]:
        state = apply_move(instance, state, move)
    return state


def _qbo_dense_config(time_limit: float = 8.0) -> SolverConfig:
    return SolverConfig(
        max_depth=60,
        beam_width=10,
        max_candidates_per_layer=120,
        candidate_limit_per_state=3,
        time_limit_sec=time_limit,
        max_qubo_vars=80,
        qubo_penalty_a=50.0,
        diversity_weight=0.075,
        qubo_num_reads=12,
        qubo_time_limit_sec=0.012,
        qubo_sa_iterations=80,
        qubo_call_interval=1,
        qubo_min_candidates_for_call=1,
        qubo_repair_hamming_slack=1,
        entropy_alpha=1.3,
        random_seed=37,
    )


def _beam_suffix_config(time_limit: float) -> SolverConfig:
    return SolverConfig(
        max_depth=60,
        beam_width=14,
        max_candidates_per_layer=220,
        candidate_limit_per_state=5,
        time_limit_sec=time_limit,
        entropy_alpha=1.3,
        random_seed=37,
    )


def _qbo_suffix_config(time_limit: float) -> SolverConfig:
    cfg = _qbo_dense_config(time_limit)
    cfg.beam_width = 12
    cfg.max_candidates_per_layer = 160
    cfg.candidate_limit_per_state = 4
    cfg.max_qubo_vars = 100
    cfg.qubo_num_reads = 14
    cfg.qubo_sa_iterations = 100
    cfg.qubo_time_limit_sec = 0.015
    return cfg


def _run_beam_from_state(instance: Instance, start_state: State, config: SolverConfig) -> tuple[State, int, int, int]:
    start = time.perf_counter()
    best = start_state
    h0 = hash_state(instance, start_state, config.enable_canonical_hash)
    beam = [Candidate(start_state, score_state(instance, start_state, config.score_weights), None, h0)]
    visited = {h0: dominance_value(instance, start_state)}
    expanded = generated = 0
    for _depth in range(config.max_depth):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        candidates: list[Candidate] = []
        for cand in beam:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expanded += 1
            scored = []
            for mv in generate_legal_moves(instance, cand.state):
                try:
                    nxt = apply_move(instance, cand.state, mv)
                except ValueError:
                    continue
                if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                    continue
                scored.append((_move_key(instance, cand.state, nxt, config), mv, nxt))
            scored.sort(key=lambda item: item[0])
            scored = scored[: config.candidate_limit_per_state or len(scored)]
            for key, mv, nxt in scored:
                h = hash_state(instance, nxt, config.enable_canonical_hash)
                dom = dominance_value(instance, nxt)
                if config.enable_visited_pruning and visited.get(h, float("inf")) <= dom:
                    continue
                visited[h] = dom
                if _rank_state(instance, nxt) < _rank_state(instance, best):
                    best = nxt
                candidates.append(Candidate(nxt, float(key[0]), mv, h))
        generated += len(candidates)
        if not candidates or disorder_box_count(instance, best) == 0:
            break
        candidates.sort(key=lambda cand: (_rank_state(instance, cand.state), cand.score))
        beam = candidates[: config.max_candidates_per_layer][: config.beam_width]
    return best, expanded, generated, len(visited)


def _run_qbo_from_state(instance: Instance, start_state: State, config: SolverConfig) -> tuple[State, int, int, int, int, int, float]:
    start = time.perf_counter()
    best = start_state
    h0 = hash_state(instance, start_state, config.enable_canonical_hash)
    beam = [Candidate(start_state, score_state(instance, start_state, config.score_weights), None, h0)]
    visited = {h0: dominance_value(instance, start_state)}
    expanded = generated = qubo_calls = qubo_vars = 0
    qubo_time = 0.0
    builder = QuboBeamSelectionBuilder()
    sampler = SimulatedAnnealingQuboSampler()
    for depth in range(config.max_depth):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        candidates: list[Candidate] = []
        for cand in beam:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expanded += 1
            scored = []
            for mv in generate_legal_moves(instance, cand.state):
                try:
                    nxt = apply_move(instance, cand.state, mv)
                except ValueError:
                    continue
                if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                    continue
                scored.append((_move_key(instance, cand.state, nxt, config), mv, nxt))
            scored.sort(key=lambda item: item[0])
            scored = scored[: config.candidate_limit_per_state or len(scored)]
            for key, mv, nxt in scored:
                h = hash_state(instance, nxt, config.enable_canonical_hash)
                dom = dominance_value(instance, nxt)
                if config.enable_visited_pruning and visited.get(h, float("inf")) <= dom:
                    continue
                visited[h] = dom
                if _rank_state(instance, nxt) < _rank_state(instance, best):
                    best = nxt
                candidates.append(Candidate(nxt, float(key[0]), mv, h))
        generated += len(candidates)
        if not candidates or disorder_box_count(instance, best) == 0:
            break
        candidates.sort(key=lambda cand: (_rank_state(instance, cand.state), cand.score))
        candidates = candidates[: config.max_candidates_per_layer]
        use_qubo = depth % max(1, config.qubo_call_interval) == 0 and len(candidates) >= config.qubo_min_candidates_for_call
        if use_qubo:
            model, usable = builder.build(
                candidates,
                config.beam_width,
                config.diversity_weight,
                config.qubo_penalty_a,
                config.max_qubo_vars,
            )
            qubo_calls += 1
            qubo_vars += len(model.var_index)
            samples = sampler.sample(
                model,
                QuboSamplerConfig(
                    num_reads=config.qubo_num_reads,
                    random_seed=config.random_seed + qubo_calls + depth,
                    time_limit_sec=config.qubo_time_limit_sec or 0.01,
                    sa_iterations=config.qubo_sa_iterations,
                ),
            )
            qubo_time += samples.runtime_sec
            beam, _repair = builder.decode_samples_and_repair(
                samples.samples,
                usable,
                config.beam_width,
                elite_keep_count=max(1, int(round(config.beam_width * 0.75))),
                diversity_weight=config.diversity_weight,
                repair_hamming_slack=config.qubo_repair_hamming_slack,
            )
        else:
            beam = candidates[: config.beam_width]
    return best, expanded, generated, len(visited), qubo_calls, qubo_vars, qubo_time


def _run_weighted_astar_legacy_state(instance: Instance, time_limit: float) -> tuple[State, int, int, int, str]:
    config = SolverConfig(max_depth=60, candidate_limit_per_state=18, time_limit_sec=time_limit, entropy_alpha=1.3, random_seed=37)
    initial = create_initial_state(instance)
    best = initial
    heap: list[tuple[tuple, int, State]] = []
    counter = 0
    heapq.heappush(heap, (_state_priority(instance, initial, weight=0.45), counter, initial))
    visited = {hash_state(instance, initial, config.enable_canonical_hash): dominance_value(instance, initial)}
    expanded = generated = 0
    start = time.perf_counter()
    while heap and time.perf_counter() - start <= time_limit:
        _priority, _idx, state = heapq.heappop(heap)
        expanded += 1
        if _rank_state(instance, state) < _rank_state(instance, best):
            best = state
            if disorder_box_count(instance, best) == 0:
                break
        scored = []
        for mv in generate_legal_moves(instance, state):
            try:
                nxt = apply_move(instance, state, mv)
            except ValueError:
                continue
            if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                continue
            scored.append((_move_key(instance, state, nxt, config), nxt))
        scored.sort(key=lambda item: item[0])
        for _key, nxt in scored[: config.candidate_limit_per_state or len(scored)]:
            h = hash_state(instance, nxt, config.enable_canonical_hash)
            dom = dominance_value(instance, nxt)
            if config.enable_visited_pruning and visited.get(h, float("inf")) <= dom:
                continue
            visited[h] = dom
            generated += 1
            if _rank_state(instance, nxt) < _rank_state(instance, best):
                best = nxt
                if disorder_box_count(instance, best) == 0:
                    break
            counter += 1
            heapq.heappush(heap, (_state_priority(instance, nxt, weight=0.45), counter, nxt))
        if disorder_box_count(instance, best) == 0:
            break
    return best, expanded, generated, len(visited), f"legacy; open={len(heap)}"


def _run_weighted_astar_multi_state(
    instance: Instance,
    time_limit: float,
    *,
    modes: tuple[str, ...],
    include_legacy_rule: bool,
) -> tuple[State, int, int, int, str]:
    config = SolverConfig(max_depth=60, candidate_limit_per_state=18, time_limit_sec=time_limit, entropy_alpha=1.3, random_seed=37)
    initial = create_initial_state(instance)
    best = initial
    heaps: dict[str, list[tuple[tuple, int, State]]] = {mode: [] for mode in ASTAR_MODES}
    counter = 0
    for mode in modes:
        heapq.heappush(heaps[mode], (_astar_priority(instance, initial, config, mode), counter, initial))
    visited = {hash_state(instance, initial, config.enable_canonical_hash): dominance_value(instance, initial)}
    expanded_dom: dict[tuple, float] = {}
    expanded = generated = 0
    mode_index = 0
    start = time.perf_counter()
    while any(heaps.values()) and time.perf_counter() - start <= time_limit:
        state = None
        for _ in range(len(modes)):
            mode = modes[mode_index % len(modes)]
            mode_index += 1
            heap = heaps[mode]
            while heap:
                _priority, _idx, cand_state = heapq.heappop(heap)
                h = hash_state(instance, cand_state, config.enable_canonical_hash)
                dom = dominance_value(instance, cand_state)
                if expanded_dom.get(h, float("inf")) <= dom:
                    continue
                expanded_dom[h] = dom
                state = cand_state
                break
            if state is not None:
                break
        if state is None:
            break
        expanded += 1
        if _rank_state(instance, state) < _rank_state(instance, best):
            best = state
            if disorder_box_count(instance, best) == 0:
                break
        successors, evaluated_count = _select_weighted_astar_successors(
            instance,
            state,
            config,
            per_rule_limit=3,
            include_legacy_rule=include_legacy_rule,
        )
        generated += evaluated_count
        for nxt in successors[: config.candidate_limit_per_state or len(successors)]:
            h = hash_state(instance, nxt, config.enable_canonical_hash)
            dom = dominance_value(instance, nxt)
            if config.enable_visited_pruning and visited.get(h, float("inf")) <= dom:
                continue
            visited[h] = dom
            if _rank_state(instance, nxt) < _rank_state(instance, best):
                best = nxt
                if disorder_box_count(instance, best) == 0:
                    break
            counter += 1
            for mode in modes:
                heapq.heappush(heaps[mode], (_astar_priority(instance, nxt, config, mode), counter, nxt))
        if disorder_box_count(instance, best) == 0:
            break
    open_count = sum(len(heap) for heap in heaps.values())
    return best, expanded, generated, len(visited), f"multi_queue={len(modes)}; open={open_count}"


def run_weighted_astar_business(case: str, instance: Instance, time_limit: float = 12.0) -> TrajectoryResult:
    start = time.perf_counter()
    best, expanded, generated, visited, status = _run_weighted_astar_multi_state(
        instance,
        time_limit,
        modes=ASTAR_MODES,
        include_legacy_rule=True,
    )
    return _result_from_state(
        case,
        "weighted_astar_business",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=visited,
        status=f"BEST_EFFORT; {status}",
    )


def run_weighted_astar_business_portfolio(case: str, instance: Instance, time_limit: float = 30.0) -> TrajectoryResult:
    """Run a portfolio of weighted-A* variants and keep the best route."""

    start = time.perf_counter()
    component_time = max(1.0, time_limit / 3.0)
    components = [
        ("legacy", _run_weighted_astar_legacy_state(instance, component_time)),
        (
            "aggressive_multi",
            _run_weighted_astar_multi_state(
                instance,
                component_time,
                modes=AGGRESSIVE_ASTAR_MODES,
                include_legacy_rule=False,
            ),
        ),
        (
            "hybrid_multi",
            _run_weighted_astar_multi_state(
                instance,
                component_time,
                modes=ASTAR_MODES,
                include_legacy_rule=True,
            ),
        ),
    ]
    best_name, (best, expanded, generated, visited, _status) = min(
        components,
        key=lambda item: _rank_state(instance, item[1][0]),
    )
    total_expanded = sum(item[1][1] for item in components)
    total_generated = sum(item[1][2] for item in components)
    total_visited = sum(item[1][3] for item in components)
    component_summary = ",".join(
        f"{name}:N{disorder_box_count(instance, state_stats[0])}/M{state_stats[0].move_count}"
        for name, state_stats in components
    )
    return _result_from_state(
        case,
        "weighted_astar_business_portfolio",
        instance,
        best,
        start,
        expanded_states=total_expanded,
        generated_states=total_generated,
        visited_states=total_visited,
        status=f"BEST_EFFORT; selected={best_name}; {component_summary}",
    )


def _prefix_lengths(move_count: int) -> list[int]:
    windows = [4, 6, 8, 10, 12, 16]
    prefixes = {max(0, move_count - window) for window in windows if move_count > window}
    prefixes.update({0, move_count // 3, move_count // 2, (2 * move_count) // 3})
    return sorted(prefixes)


def run_lns_route(
    case: str,
    instance: Instance,
    *,
    solver_name: str,
    use_qubo: bool,
    total_time: float = 30.0,
) -> TrajectoryResult:
    start = time.perf_counter()
    incumbent, e0, g0, v0, q0, qv0, qt0 = _run_qbo_from_state(instance, create_initial_state(instance), _qbo_dense_config(8.0))
    best = incumbent
    expanded, generated, visited = e0, g0, v0
    qubo_calls, qubo_vars, qubo_time = q0, qv0, qt0
    attempts = 0
    prefixes = _prefix_lengths(incumbent.move_count)
    while time.perf_counter() - start < total_time and prefixes:
        prefix_len = prefixes[attempts % len(prefixes)]
        attempts += 1
        try:
            prefix_state = _replay_prefix(instance, best.moves, prefix_len)
        except ValueError:
            continue
        remaining = max(0.5, min(3.0, total_time - (time.perf_counter() - start)))
        if use_qubo:
            state, e, g, v, qc, qv, qt = _run_qbo_from_state(instance, prefix_state, _qbo_suffix_config(remaining))
            qubo_calls += qc
            qubo_vars += qv
            qubo_time += qt
        else:
            state, e, g, v = _run_beam_from_state(instance, prefix_state, _beam_suffix_config(remaining))
        expanded += e
        generated += g
        visited += v
        if _rank_state(instance, state) < _rank_state(instance, best):
            best = state
        if disorder_box_count(instance, best) == 0:
            break
    return _result_from_state(
        case,
        solver_name,
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=visited,
        qubo_calls=qubo_calls,
        qubo_total_vars=qubo_vars,
        qubo_time_sec=qubo_time,
        status=f"BEST_EFFORT; incumbent=qbo_beam_dense; lns_attempts={attempts}; suffix={'qbo' if use_qubo else 'beam'}",
    )


def _write_csv(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrajectoryResult.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_comparison(new_rows: list[TrajectoryResult]) -> None:
    base_path = ROOT / "docs" / "experiments" / "full_real_business_focused_beam_comparison.csv"
    rows = _read_rows(base_path)
    rows.extend({key: str(value) for key, value in row.__dict__.items()} for row in new_rows)
    seen: set[tuple[str, str]] = set()
    dedup = []
    for row in rows:
        key = (row["case"], row["solver"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)

    cases = sorted({row["case"] for row in dedup})
    by_solver: dict[str, list[dict[str, str]]] = {}
    for row in dedup:
        by_solver.setdefault(row["solver"], []).append(row)
    summary = []
    for solver, solver_rows in by_solver.items():
        if len(solver_rows) != len(cases):
            continue
        summary.append(
            {
                "solver": solver,
                "cases": len(solver_rows),
                "avg_final_n": sum(float(r["final_n"]) for r in solver_rows) / len(solver_rows),
                "avg_reduction": sum(float(r["reduction"]) for r in solver_rows) / len(solver_rows),
                "avg_moves": sum(float(r["moves"]) for r in solver_rows) / len(solver_rows),
                "avg_runtime": sum(float(r["runtime_sec"]) for r in solver_rows) / len(solver_rows),
                "avg_qubo_calls": sum(float(r["qubo_calls"]) for r in solver_rows) / len(solver_rows),
                "ok13": sum(str(r["g_ok_13"]).lower() == "true" for r in solver_rows),
            }
        )
    summary.sort(key=lambda item: (item["avg_final_n"], item["avg_moves"], item["avg_runtime"]))
    solvers = [item["solver"] for item in summary]
    lookup = {(row["case"], row["solver"]): row for row in dedup}

    csv_path = ROOT / "docs" / "experiments" / "full_real_business_lns_astar_comparison.csv"
    md_path = ROOT / "docs" / "experiments" / "full_real_business_lns_astar_comparison.md"
    fields = list(TrajectoryResult.__annotations__.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in dedup:
            writer.writerow({field: row.get(field, "") for field in fields})

    lines = [
        "# Full Real Business LNS / Weighted-A* Benchmark",
        "",
        "This report tests exact-inspired non-MIP benchmarks that are intended to be stronger than simple heuristics but faster than the full 4D MIP.",
        "",
        f"CSV: `{csv_path.relative_to(ROOT)}`",
        "",
        "## Aggregate",
        "",
        "| solver | cases | avg final N | avg reduction | avg moves | avg runtime s | avg QUBO calls | ok 1.3N0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['solver']} | {item['cases']} | {item['avg_final_n']:.2f} | {item['avg_reduction']:.2f} | "
            f"{item['avg_moves']:.2f} | {item['avg_runtime']:.3f} | {item['avg_qubo_calls']:.2f} | {item['ok13']} |"
        )
    lines.extend(["", "## Per Case Final N", ""])
    lines.append("| case | " + " | ".join(solvers) + " |")
    lines.append("|---|" + "|".join("---:" for _ in solvers) + "|")
    for case in cases:
        values = []
        for solver in solvers:
            row = lookup.get((case, solver))
            values.append("" if row is None else row["final_n"])
        lines.append(f"| {case} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `lns_beam_repair` starts from `qbo_beam_dense`, rolls back selected suffix windows, and replans the suffix with beam search.",
            "- `lns_qubo_repair` uses the same prefix rollback scheme but replans each suffix with QUBO beam.",
            "- `weighted_astar_business` is a best-first search over legal business-feasible states with a weighted disorder/inversion heuristic.",
            "- These are benchmark routes, not optimality proofs.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def main() -> None:
    rows: list[TrajectoryResult] = []
    for zone, _suffixes, _paths in discover_paired_cases(ROOT / "data" / "input", ["01", "26"]):
        instance = _instance_from_zone(zone)
        case = instance.instance_id
        print(f"running {case}", flush=True)
        rows.append(run_weighted_astar_business(case, instance, 10.0))
        rows.append(run_lns_route(case, instance, solver_name="lns_beam_repair", use_qubo=False, total_time=24.0))
        rows.append(run_lns_route(case, instance, solver_name="lns_qubo_repair", use_qubo=True, total_time=30.0))
    csv_path = ROOT / "docs" / "experiments" / "full_real_business_lns_astar.csv"
    _write_csv(rows, csv_path)
    _write_comparison(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
