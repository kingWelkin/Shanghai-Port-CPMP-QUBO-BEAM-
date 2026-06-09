"""Pre-pruning slack-aware QUBO beam experiment.

This variant uses QUBO before local move pruning. For each beam state it puts a
large legal-move reservoir into a QUBO and selects a small diverse set of moves
to expand. The QUBO objective is deliberately interpretable: reduce N(Future),
preserve slack under the business G cap, and avoid selecting many moves with
the same source or destination.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam import _write_explain_csv  # noqa: E402
from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from benchmark_full_real_business_algorithms import TrajectoryResult, _result_from_state  # noqa: E402
from benchmark_macro_reservoir_qubo_beam import tail_repair_to_n_zero  # noqa: E402
from container_reshuffle.core import Instance, Move, State, apply_move, create_initial_state, generate_legal_moves, hash_state  # noqa: E402
from container_reshuffle.features import (  # noqa: E402
    bad_count,
    dirty_stack_count,
    disorder_box_count,
    entropy_limit,
    entropy_value,
    inversion_count,
    settled_box_count,
    well_prefix_gain,
)
from container_reshuffle.qubo.model import QuboModel  # noqa: E402
from container_reshuffle.qubo.sampler import QuboSamplerConfig, QuboSample, SimulatedAnnealingQuboSampler  # noqa: E402
from container_reshuffle.scoring import cpmp_business_feasible, dominance_value, score_state  # noqa: E402
from container_reshuffle.similarity import combined_similarity  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.beam import Candidate  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


@dataclass(frozen=True)
class MoveOption:
    move: Move
    state: State
    energy: float


def _auto_config(instance: Instance, time_limit: float) -> tuple[SolverConfig, dict[str, int | float]]:
    cap = entropy_limit(instance, 1.3)
    config = SolverConfig(
        max_depth=cap,
        beam_width=60,
        max_candidates_per_layer=600,
        time_limit_sec=time_limit,
        entropy_alpha=1.3,
        random_seed=37,
    )
    params: dict[str, int | float] = {
        "move_keep": 6,
        "max_qubo_vars": 90,
        "qubo_reads": 4,
        "qubo_iterations": 80,
        "qubo_time_limit": 0.006,
        "cardinality_penalty": 8.0,
        "same_source_penalty": 0.16,
        "same_dest_penalty": 0.08,
        "similarity_penalty": 0.04,
        "tail_repair_n": 5,
        "tail_repair_expanded": 200_000,
        "tail_states_per_layer": 4,
    }
    return config, params


def _state_rank(instance: Instance, state: State) -> tuple:
    return (
        disorder_box_count(instance, state),
        state.entropy_max,
        state.disorder_total,
        state.move_count,
        inversion_count(instance, state),
        bad_count(instance, state),
        dirty_stack_count(instance, state),
        -settled_box_count(instance, state),
    )


def _move_energy(instance: Instance, current: State, nxt: State) -> float:
    cap = entropy_limit(instance, 1.3)
    current_n = disorder_box_count(instance, current)
    n_value = disorder_box_count(instance, nxt)
    g_value = entropy_value(instance, nxt)
    slack = cap - g_value
    remaining_moves = max(0, cap - nxt.move_count)
    clearance_risk = max(0, n_value - remaining_moves)
    low_slack_risk = max(0, 2 - slack) if n_value > 4 else max(0, 1 - slack)
    gain = current_n - n_value
    return (
        1_000_000.0 * n_value
        + 80_000.0 * clearance_risk * clearance_risk
        + 40_000.0 * low_slack_risk * low_slack_risk
        + 25_000.0 * nxt.entropy_max
        + 8.0 * nxt.disorder_total
        + 25.0 * nxt.move_count
        + 1.0 * inversion_count(instance, nxt)
        + 2.0 * bad_count(instance, nxt)
        + 8.0 * dirty_stack_count(instance, nxt)
        - 50.0 * settled_box_count(instance, nxt)
        - 5_000.0 * max(0, gain)
        - 500.0 * max(0, well_prefix_gain(instance, current, nxt))
        - 150.0 * max(0, slack)
    )


def _generate_move_options(
    instance: Instance,
    state: State,
    config: SolverConfig,
    max_qubo_vars: int,
) -> list[MoveOption]:
    options: list[MoveOption] = []
    for move in generate_legal_moves(instance, state):
        try:
            nxt = apply_move(instance, state, move)
        except ValueError:
            continue
        if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
            continue
        options.append(MoveOption(move, nxt, _move_energy(instance, state, nxt)))
    if len(options) <= max_qubo_vars:
        return sorted(options, key=lambda opt: (opt.energy, _state_rank(instance, opt.state)))

    selected: dict[tuple[int, int, int], MoveOption] = {}

    def add(rows: list[MoveOption]) -> None:
        for opt in rows:
            key = (opt.move.box, opt.move.from_stack, opt.move.to_stack)
            old = selected.get(key)
            if old is None or opt.energy < old.energy:
                selected[key] = opt

    ranked = sorted(options, key=lambda opt: (opt.energy, _state_rank(instance, opt.state)))
    add(ranked[: max_qubo_vars // 2])
    for src in sorted({opt.move.from_stack for opt in options}):
        bucket = [opt for opt in options if opt.move.from_stack == src]
        bucket.sort(key=lambda opt: (opt.energy, _state_rank(instance, opt.state)))
        add(bucket[:2])
    for dst in sorted({opt.move.to_stack for opt in options}):
        bucket = [opt for opt in options if opt.move.to_stack == dst]
        bucket.sort(key=lambda opt: (opt.energy, _state_rank(instance, opt.state)))
        add(bucket[:1])
    add(ranked)
    out = sorted(selected.values(), key=lambda opt: (opt.energy, _state_rank(instance, opt.state)))
    return out[:max_qubo_vars]


def _normalized(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _build_move_qubo(
    instance: Instance,
    options: list[MoveOption],
    target: int,
    params: dict[str, int | float],
) -> QuboModel:
    model = QuboModel()
    vars_ = [model.add_binary_var(f"m_{idx}") for idx in range(len(options))]
    for var, value in zip(vars_, _normalized([opt.energy for opt in options])):
        model.add_linear(var, value)
    for i in range(len(options)):
        for j in range(i + 1, len(options)):
            oi = options[i]
            oj = options[j]
            penalty = 0.0
            if oi.move.from_stack == oj.move.from_stack:
                penalty += float(params["same_source_penalty"])
            if oi.move.to_stack == oj.move.to_stack:
                penalty += float(params["same_dest_penalty"])
            penalty += float(params["similarity_penalty"]) * combined_similarity(oi.state, oj.state)
            if penalty:
                model.add_quadratic(vars_[i], vars_[j], penalty)
    model.add_exactly_k_penalty(vars_, target, float(params["cardinality_penalty"]))
    return model


def _selection_objective(
    instance: Instance,
    options: list[MoveOption],
    selected: set[int],
    params: dict[str, int | float],
) -> float:
    scores = _normalized([opt.energy for opt in options])
    total = sum(scores[i] for i in selected)
    for pos, i in enumerate(selected):
        oi = options[i]
        for j in list(selected)[pos + 1 :]:
            oj = options[j]
            if oi.move.from_stack == oj.move.from_stack:
                total += float(params["same_source_penalty"])
            if oi.move.to_stack == oj.move.to_stack:
                total += float(params["same_dest_penalty"])
            total += float(params["similarity_penalty"]) * combined_similarity(oi.state, oj.state)
    return total


def _repair_selection(
    instance: Instance,
    options: list[MoveOption],
    sample: QuboSample | None,
    target: int,
    params: dict[str, int | float],
) -> list[int]:
    selected = {idx for idx in range(len(options)) if sample is not None and sample.binary_values.get(idx, 0) == 1}
    while len(selected) > target:
        drop = min(
            selected,
            key=lambda idx: (
                _selection_objective(instance, options, selected - {idx}, params),
                -options[idx].energy,
                idx,
            ),
        )
        selected.remove(drop)
    while len(selected) < target:
        missing = [idx for idx in range(len(options)) if idx not in selected]
        if not missing:
            break
        add = min(
            missing,
            key=lambda idx: (
                _selection_objective(instance, options, selected | {idx}, params),
                options[idx].energy,
                idx,
            ),
        )
        selected.add(add)
    for _ in range(2):
        current = _selection_objective(instance, options, selected, params)
        best_swap = None
        best_value = current
        for remove in list(selected):
            base = selected - {remove}
            for add in range(len(options)):
                if add in selected:
                    continue
                value = _selection_objective(instance, options, base | {add}, params)
                if value + 1e-12 < best_value:
                    best_value = value
                    best_swap = (remove, add)
        if best_swap is None:
            break
        selected.remove(best_swap[0])
        selected.add(best_swap[1])
    return sorted(selected, key=lambda idx: (options[idx].energy, idx))


def _select_moves_with_qubo(
    instance: Instance,
    options: list[MoveOption],
    config: SolverConfig,
    params: dict[str, int | float],
    qcall: int,
) -> tuple[list[MoveOption], int, float]:
    if not options:
        return [], 0, 0.0
    target = min(int(params["move_keep"]), len(options))
    model = _build_move_qubo(instance, options, target, params)
    sampler = SimulatedAnnealingQuboSampler()
    sample_set = sampler.sample(
        model,
        QuboSamplerConfig(
            num_reads=int(params["qubo_reads"]),
            random_seed=config.random_seed + qcall,
            time_limit_sec=float(params["qubo_time_limit"]),
            sa_iterations=int(params["qubo_iterations"]),
        ),
    )
    candidates = sample_set.samples or [None]
    best_idx: list[int] | None = None
    best_value = math.inf
    for sample in candidates:
        idxs = _repair_selection(instance, options, sample, target, params)
        value = _selection_objective(instance, options, set(idxs), params)
        if value < best_value:
            best_value = value
            best_idx = idxs
    if best_idx is None:
        best_idx = list(range(target))
    return [options[idx] for idx in best_idx], len(model.var_index), sample_set.runtime_sec


def _tail_candidates(instance: Instance, beam: list[Candidate], params: dict[str, int | float]) -> list[State]:
    rows = [cand.state for cand in beam if disorder_box_count(instance, cand.state) <= int(params["tail_repair_n"])]
    rows.sort(key=lambda state: _state_rank(instance, state))
    return rows[: int(params["tail_states_per_layer"])]


def run_pre_prune_slack_qubo_beam(
    case: str,
    instance: Instance,
    config: SolverConfig,
    params: dict[str, int | float],
) -> tuple[TrajectoryResult, State]:
    start = time.perf_counter()
    initial = create_initial_state(instance)
    best = initial
    h0 = hash_state(instance, initial, config.enable_canonical_hash)
    beam = [Candidate(initial, score_state(instance, initial, config.score_weights), None, h0)]
    visited = {h0: dominance_value(instance, initial)}
    expanded = generated = qcalls = qvars = 0
    qtime = 0.0
    tail_expanded = tail_generated = 0
    tail_found = False

    for depth in range(config.max_depth):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        candidates: list[Candidate] = []
        for cand in beam:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expanded += 1
            options = _generate_move_options(instance, cand.state, config, int(params["max_qubo_vars"]))
            selected, var_count, runtime = _select_moves_with_qubo(instance, options, config, params, qcalls + 1)
            qcalls += 1 if options else 0
            qvars += var_count
            qtime += runtime
            for opt in selected:
                h = hash_state(instance, opt.state, config.enable_canonical_hash)
                dom = dominance_value(instance, opt.state)
                if config.enable_visited_pruning and visited.get(h, math.inf) <= dom:
                    continue
                visited[h] = dom
                if _state_rank(instance, opt.state) < _state_rank(instance, best):
                    best = opt.state
                candidates.append(Candidate(opt.state, opt.energy, opt.move, h))
        generated += len(candidates)
        if not candidates:
            break
        candidates.sort(key=lambda cand: (_state_rank(instance, cand.state), cand.score))
        beam = candidates[: config.beam_width]

        for state in _tail_candidates(instance, beam, params):
            repaired, exp_count, gen_count = tail_repair_to_n_zero(
                instance,
                state,
                config,
                deadline=start + config.time_limit_sec,
                max_expanded=int(params["tail_repair_expanded"]),
            )
            tail_expanded += exp_count
            tail_generated += gen_count
            if repaired is not None:
                best = repaired
                tail_found = True
                break

        print(
            "LAYER "
            f"depth={depth + 1} beam={len(beam)} bestN={disorder_box_count(instance, best)} "
            f"beamBestN={disorder_box_count(instance, beam[0].state)} "
            f"qcalls={qcalls} elapsed={time.perf_counter() - start:.2f}s",
            flush=True,
        )
        if disorder_box_count(instance, best) == 0:
            break

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += (
        "; pre_prune_qubo=move_reservoir/slack/source_dest_diversity"
        f"; move_keep={params['move_keep']}; max_qubo_vars={params['max_qubo_vars']}; "
        f"tail_expanded={tail_expanded}; tail_generated={tail_generated}; tail_found={tail_found}"
    )
    row = _result_from_state(
        case,
        "pre_prune_slack_qubo_beam",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=len(visited),
        qubo_calls=qcalls,
        qubo_total_vars=qvars,
        qubo_time_sec=qtime,
        status=status + f"; qubo_calls={qcalls}",
    )
    return row, best


def _write_result_csv(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrajectoryResult.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _run_case(zone: str, bay: str, time_limit: float, output_prefix: Path | None) -> TrajectoryResult:
    instance = _instance_from_single_bay(zone, bay)
    config, params = _auto_config(instance, time_limit)
    initial = create_initial_state(instance)
    print(
        "START "
        f"case={instance.instance_id} N0={disorder_box_count(instance, initial)} "
        f"cap={entropy_limit(instance, 1.3)} params={params}",
        flush=True,
    )
    row, best = run_pre_prune_slack_qubo_beam(instance.instance_id, instance, config, params)
    print(f"FINAL {row}", flush=True)
    if output_prefix is not None:
        _write_result_csv([row], output_prefix.with_suffix(".csv"))
        _write_explain_csv(instance, best.moves, output_prefix.with_name(output_prefix.name + "_moves.csv"))
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="C1")
    parser.add_argument("--bay", default="26")
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "pre_prune_slack_qubo_beam_all_single_bays.csv",
    )
    parser.add_argument(
        "--moves-dir",
        type=Path,
        default=ROOT / "docs" / "experiments" / "pre_prune_slack_qubo_beam_moves",
    )
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()

    if not args.all:
        _run_case(args.zone, args.bay, args.time_limit, args.output_prefix)
        return 0

    rows: list[TrajectoryResult] = []
    start = time.perf_counter()
    cases = discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))
    for idx, (zone, bay) in enumerate(cases, start=1):
        instance = _instance_from_single_bay(zone, bay)
        config, params = _auto_config(instance, args.time_limit)
        print(f"CASE_START {idx}/{len(cases)} {instance.instance_id}", flush=True)
        row, best = run_pre_prune_slack_qubo_beam(instance.instance_id, instance, config, params)
        rows.append(row)
        _write_result_csv(rows, args.output)
        _write_explain_csv(instance, best.moves, args.moves_dir / f"{instance.instance_id}_moves.csv")
        print(
            f"CASE_DONE {idx}/{len(cases)} {row.case} finalN={row.final_n} moves={row.moves} "
            f"runtime={row.runtime_sec:.2f}s",
            flush=True,
        )
    solved = sum(1 for row in rows if row.final_n == 0)
    print(
        f"SUMMARY cases={len(rows)} solved={solved} runtime={sum(row.runtime_sec for row in rows):.2f}s "
        f"wall={time.perf_counter() - start:.2f}s",
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
