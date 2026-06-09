"""Compare stronger QUBO-style variants on real single-bay business cases.

This script is intentionally experimental.  It keeps production solvers
untouched and adapts existing QUBO builders to the real single-bay
``N(Future)`` business objective by returning the best partial trajectory.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_fast_slack_beam import run_fast_slack_beam  # noqa: E402
from benchmark_full_real_business_algorithms import TrajectoryResult, _result_from_state  # noqa: E402
from benchmark_macro_reservoir_qubo_beam import run_macro_reservoir_qubo_beam  # noqa: E402
from benchmark_rule_qubo_hybrid_beam import _auto_config as hybrid_auto_config  # noqa: E402
from benchmark_rule_qubo_hybrid_beam import run_rule_qubo_hybrid_beam  # noqa: E402
from container_reshuffle.core import (  # noqa: E402
    Instance,
    State,
    apply_move,
    create_initial_state,
    generate_legal_moves,
    hash_state,
)
from container_reshuffle.features import (  # noqa: E402
    bad_count,
    dirty_stack_count,
    disorder_box_count,
    entropy_limit,
    inversion_count,
    settled_box_count,
)
from container_reshuffle.qubo.path_selection import QuboPathSelectionBuilder  # noqa: E402
from container_reshuffle.qubo.sampler import QuboSamplerConfig, SimulatedAnnealingQuboSampler  # noqa: E402
from container_reshuffle.qubo.stack_assignment import QuboStackAssignmentBuilder  # noqa: E402
from container_reshuffle.scoring import cpmp_business_feasible, dominance_value, score_state  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.beam import Candidate  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


def _best_key(instance: Instance, state: State) -> tuple:
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


def run_path_flow_qubo_business(case: str, instance: Instance, config: SolverConfig) -> TrajectoryResult:
    """Use the existing path-flow QUBO as a rolling business improver."""

    start = time.perf_counter()
    state = create_initial_state(instance)
    best = state
    builder = QuboPathSelectionBuilder()
    sampler = SimulatedAnnealingQuboSampler()
    calls = vars_total = repairs = expanded = generated = 0
    qtime = 0.0

    while time.perf_counter() - start <= config.time_limit_sec and state.move_count < config.max_depth:
        if disorder_box_count(instance, best) == 0:
            break
        nodes, edges = builder.build_tree(instance, state, config)
        expanded += len(nodes)
        generated += len(edges)
        if not edges:
            break
        model = builder.build_qubo(nodes, edges, config.qubo_penalty_a)
        calls += 1
        vars_total += len(model.var_index)
        sample_set = sampler.sample(
            model,
            QuboSamplerConfig(
                num_reads=config.qubo_num_reads,
                random_seed=config.random_seed + calls,
                time_limit_sec=max(0.001, config.qubo_time_limit_sec or 0.02),
                sa_iterations=max(1, int(config.qubo_sa_iterations)),
            ),
        )
        qtime += sample_set.runtime_sec
        path, repair = builder.decode_or_repair(nodes, edges, sample_set.samples[0] if sample_set.samples else None)
        repairs += repair
        if not path:
            break
        for move in path[: max(1, config.prefix_commit_length)]:
            state = apply_move(instance, state, move)
            if _best_key(instance, state) < _best_key(instance, best):
                best = state
            if disorder_box_count(instance, best) == 0 or state.move_count >= config.max_depth:
                break
        print(
            "PATH_FLOW "
            f"step={state.move_count} bestN={disorder_box_count(instance, best)} "
            f"currentN={disorder_box_count(instance, state)} qvars={len(model.var_index)} "
            f"elapsed={time.perf_counter() - start:.2f}s",
            flush=True,
        )

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += f"; path_flow_rolling_qubo; repairs={repairs}"
    return _result_from_state(
        case,
        "path_flow_qubo_rolling_business",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=0,
        qubo_calls=calls,
        qubo_total_vars=vars_total,
        qubo_time_sec=qtime,
        status=status,
    )


def _off_target(instance: Instance, state: State, target: dict[int, int]) -> int:
    current = {box: idx for idx, stack in enumerate(state.stacks) for box in stack}
    return sum(1 for box, idx in current.items() if target.get(box) != idx)


def run_target_stack_qubo_business(case: str, instance: Instance, config: SolverConfig) -> TrajectoryResult:
    """Use target-stack assignment QUBO, then run guided beam for best business state."""

    start = time.perf_counter()
    initial = create_initial_state(instance)
    builder = QuboStackAssignmentBuilder()
    model, var_for = builder.build(instance, initial, config.qubo_penalty_a, keep_weight=0.8, order_weight=1.2)
    sampler = SimulatedAnnealingQuboSampler()
    sample_set = sampler.sample(
        model,
        QuboSamplerConfig(
            num_reads=config.qubo_num_reads,
            random_seed=config.random_seed,
            time_limit_sec=max(0.001, config.qubo_time_limit_sec or 0.05),
            sa_iterations=max(1, int(config.qubo_sa_iterations)),
        ),
    )
    target, repairs = builder.decode_and_repair(instance, sample_set.samples[0] if sample_set.samples else None, var_for)
    beam = [Candidate(initial, score_state(instance, initial, config.score_weights), None, hash_state(instance, initial))]
    visited = {beam[0].hash_value: dominance_value(instance, initial)}
    best = initial
    expanded = generated = 0

    for _depth in range(config.max_depth):
        if time.perf_counter() - start > config.time_limit_sec or disorder_box_count(instance, best) == 0:
            break
        candidates: list[Candidate] = []
        for cand in beam:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expanded += 1
            scored = []
            for move in generate_legal_moves(instance, cand.state):
                nxt = apply_move(instance, cand.state, move)
                if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                    continue
                guided = (
                    score_state(instance, nxt, config.score_weights)
                    + 25.0 * _off_target(instance, nxt, target)
                    + 10_000.0 * disorder_box_count(instance, nxt)
                )
                scored.append((guided, move, nxt))
            scored.sort(key=lambda row: row[0])
            if config.candidate_limit_per_state is not None:
                scored = scored[: config.candidate_limit_per_state]
            for score, move, nxt in scored:
                h = hash_state(instance, nxt, config.enable_canonical_hash)
                dom = dominance_value(instance, nxt)
                if config.enable_visited_pruning and visited.get(h, math.inf) <= dom:
                    continue
                visited[h] = dom
                if _best_key(instance, nxt) < _best_key(instance, best):
                    best = nxt
                candidates.append(Candidate(nxt, score, move, h))
        generated += len(candidates)
        if not candidates:
            break
        candidates.sort(key=lambda cand: (_best_key(instance, cand.state), cand.score))
        beam = candidates[: config.beam_width]
        print(
            "TARGET_STACK "
            f"depth={beam[0].state.move_count if beam else 0} beam={len(beam)} "
            f"bestN={disorder_box_count(instance, best)} elapsed={time.perf_counter() - start:.2f}s",
            flush=True,
        )

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += f"; target_stack_assignment_qubo; repairs={repairs}"
    return _result_from_state(
        case,
        "target_stack_qubo_guided_business",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=len(visited),
        qubo_calls=1,
        qubo_total_vars=len(model.var_index),
        qubo_time_sec=sample_set.runtime_sec,
        status=status,
    )


def _write_rows(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrajectoryResult.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def run_case(args: argparse.Namespace, zone: str, bay: str) -> list[TrajectoryResult]:
    instance = _instance_from_single_bay(zone, bay)
    cap = entropy_limit(instance, 1.3)
    initial = create_initial_state(instance)
    print(
        "START "
        f"case={instance.instance_id} stacks={instance.stack_count} H={instance.max_height} "
        f"N0={disorder_box_count(instance, initial)} cap={cap}",
        flush=True,
    )
    rows: list[TrajectoryResult] = []

    fast_cfg = SolverConfig(
        max_depth=cap,
        beam_width=args.beam_width,
        time_limit_sec=args.time_limit,
        entropy_alpha=1.3,
        random_seed=37,
    )
    row, _ = run_fast_slack_beam(
        instance.instance_id,
        instance,
        fast_cfg,
        tail_repair_n=args.tail_repair_n,
        tail_repair_expanded=args.tail_repair_expanded,
    )
    rows.append(row)

    path_cfg = SolverConfig(
        max_depth=cap,
        beam_width=max(8, min(30, args.beam_width // 4)),
        max_candidates_per_layer=args.path_candidates,
        rolling_depth=args.rolling_depth,
        prefix_commit_length=args.prefix_commit_length,
        time_limit_sec=args.time_limit,
        max_qubo_vars=args.max_qubo_vars,
        qubo_penalty_a=50.0,
        qubo_num_reads=args.qubo_reads,
        qubo_time_limit_sec=args.qubo_time_limit,
        qubo_sa_iterations=args.qubo_iterations,
        entropy_alpha=1.3,
        random_seed=37,
    )
    rows.append(run_path_flow_qubo_business(instance.instance_id, instance, path_cfg))

    target_cfg = SolverConfig(
        max_depth=cap,
        beam_width=args.beam_width,
        max_candidates_per_layer=2500,
        candidate_limit_per_state=args.candidate_limit_per_state,
        time_limit_sec=args.time_limit,
        max_qubo_vars=args.max_qubo_vars,
        qubo_penalty_a=50.0,
        qubo_num_reads=args.qubo_reads,
        qubo_time_limit_sec=args.qubo_time_limit,
        qubo_sa_iterations=args.qubo_iterations,
        entropy_alpha=1.3,
        random_seed=37,
    )
    rows.append(run_target_stack_qubo_business(instance.instance_id, instance, target_cfg))

    macro_cfg = SolverConfig(
        max_depth=cap,
        beam_width=max(12, min(36, args.beam_width // 8)),
        max_candidates_per_layer=500,
        time_limit_sec=args.time_limit,
        max_qubo_vars=args.max_qubo_vars,
        qubo_penalty_a=40.0,
        qubo_num_reads=args.qubo_reads,
        qubo_time_limit_sec=args.qubo_time_limit,
        qubo_sa_iterations=args.qubo_iterations,
        entropy_alpha=1.3,
        random_seed=37,
    )
    row, _ = run_macro_reservoir_qubo_beam(
        instance.instance_id,
        instance,
        macro_cfg,
        macro_depth=args.macro_depth,
        commit_length=args.commit_length,
        global_keep=5,
        source_keep=2,
        dest_keep=1,
        plateau_keep=6,
        max_reservoir=18,
        max_macros_per_state=36,
        state_similarity_weight=0.08,
        same_first_weight=0.60,
        same_source_weight=0.20,
        same_dest_weight=0.06,
        tail_repair_n=args.tail_repair_n,
        tail_repair_expanded=args.tail_repair_expanded,
    )
    rows.append(row)

    hybrid_cfg, hybrid_params = hybrid_auto_config(instance, args.time_limit)
    hybrid_cfg.qubo_time_limit_sec = args.qubo_time_limit
    hybrid_cfg.qubo_num_reads = args.qubo_reads
    hybrid_cfg.qubo_sa_iterations = args.qubo_iterations
    hybrid_params["tail_repair_n"] = args.tail_repair_n
    hybrid_params["tail_repair_expanded"] = args.tail_repair_expanded
    row, _ = run_rule_qubo_hybrid_beam(instance.instance_id, instance, hybrid_cfg, hybrid_params)
    rows.append(row)

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="C1")
    parser.add_argument("--bay", default="26")
    parser.add_argument("--time-limit", type=float, default=12.0)
    parser.add_argument("--beam-width", type=int, default=120)
    parser.add_argument("--candidate-limit-per-state", type=int, default=28)
    parser.add_argument("--rolling-depth", type=int, default=3)
    parser.add_argument("--prefix-commit-length", type=int, default=1)
    parser.add_argument("--path-candidates", type=int, default=90)
    parser.add_argument("--macro-depth", type=int, default=2)
    parser.add_argument("--commit-length", type=int, default=2)
    parser.add_argument("--max-qubo-vars", type=int, default=120)
    parser.add_argument("--qubo-time-limit", type=float, default=0.02)
    parser.add_argument("--qubo-reads", type=int, default=6)
    parser.add_argument("--qubo-iterations", type=int, default=120)
    parser.add_argument("--tail-repair-n", type=int, default=6)
    parser.add_argument("--tail-repair-expanded", type=int, default=40000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "strong_qubo_variants_real_single.csv",
    )
    args = parser.parse_args()

    rows = run_case(args, args.zone, args.bay)
    _write_rows(rows, args.output)
    print(f"wrote {args.output}", flush=True)
    for row in rows:
        print(
            f"RESULT {row.solver} finalN={row.final_n} moves={row.moves} "
            f"runtime={row.runtime_sec:.3f}s qcalls={row.qubo_calls}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
