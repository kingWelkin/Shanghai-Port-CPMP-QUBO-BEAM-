"""Fast slack-preserving one-step beam experiment.

This is a low-latency counterpart to the macro/QUBO experiments. It avoids
macro expansion and instead uses wide one-step beam selection with explicit
rule buckets:

- best objective states,
- low-entropy/slack-preserving states,
- buffer-release moves,
- source representatives.

It is intended to test whether a seconds-level route can still reach N=0.
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

from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from benchmark_explainable_rule_macro_beam import _write_explain_csv  # noqa: E402
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
    well_prefix_len,
)
from container_reshuffle.scoring import cpmp_business_feasible, dominance_value  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.beam import Candidate  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


@dataclass(frozen=True)
class StepCandidate:
    state: State
    move: Move
    hash_value: tuple
    score: tuple
    bucket: str


def _best_key(instance: Instance, state: State) -> tuple:
    return (
        disorder_box_count(instance, state),
        state.entropy_max,
        state.move_count,
        inversion_count(instance, state),
        bad_count(instance, state),
        dirty_stack_count(instance, state),
        -settled_box_count(instance, state),
    )


def _score(instance: Instance, before: State, after: State, move: Move) -> tuple:
    cap = entropy_limit(instance, 1.3)
    n_before = disorder_box_count(instance, before)
    n_after = disorder_box_count(instance, after)
    remaining = cap - after.move_count
    slack = cap - entropy_value(instance, after)
    clearance_risk = max(0, n_after - remaining)
    source_remaining = len(after.stacks[move.from_stack])
    buffer_release = 1 if len(before.stacks[move.from_stack]) <= 2 and n_after <= n_before else 0
    return (
        clearance_risk,
        n_after,
        max(0, 2 - slack),
        after.entropy_max,
        -buffer_release,
        source_remaining,
        -well_prefix_gain(instance, before, after),
        inversion_count(instance, after),
        bad_count(instance, after),
        dirty_stack_count(instance, after),
        after.move_count,
    )


def _is_immediate_undo(first: Move, second: Move) -> bool:
    return first.box == second.box and first.from_stack == second.to_stack and first.to_stack == second.from_stack


def _lookahead_score(instance: Instance, candidate: StepCandidate, entropy_alpha: float) -> tuple:
    """Rank a first move by the best feasible state reachable in one more move."""

    best = (
        disorder_box_count(instance, candidate.state),
        candidate.state.entropy_max,
        entropy_value(instance, candidate.state),
        inversion_count(instance, candidate.state),
        bad_count(instance, candidate.state),
        dirty_stack_count(instance, candidate.state),
        candidate.score,
    )
    for move in generate_legal_moves(instance, candidate.state):
        if _is_immediate_undo(candidate.move, move):
            continue
        try:
            nxt = apply_move(instance, candidate.state, move)
        except ValueError:
            continue
        if not cpmp_business_feasible(instance, nxt, entropy_alpha):
            continue
        key = (
            disorder_box_count(instance, nxt),
            nxt.entropy_max,
            entropy_value(instance, nxt),
            inversion_count(instance, nxt),
            bad_count(instance, nxt),
            dirty_stack_count(instance, nxt),
            candidate.score,
        )
        if key < best:
            best = key
    return best


def _select_bucketed(
    instance: Instance,
    candidates: list[StepCandidate],
    width: int,
    *,
    lookahead_pool: int = 0,
    lookahead_keep: int = 0,
    entropy_alpha: float = 1.3,
) -> list[StepCandidate]:
    if len(candidates) <= width:
        return sorted(candidates, key=lambda c: c.score)
    ranked = sorted(candidates, key=lambda c: c.score)
    selected: dict[tuple, StepCandidate] = {}

    def add(rows: list[StepCandidate]) -> None:
        for cand in rows:
            if len(selected) >= width:
                return
            old = selected.get(cand.hash_value)
            if old is None or cand.score < old.score:
                selected[cand.hash_value] = cand

    if lookahead_pool > 0 and lookahead_keep > 0:
        pool: dict[tuple, StepCandidate] = {}

        def seed(rows: list[StepCandidate]) -> None:
            for cand in rows:
                pool.setdefault(cand.hash_value, cand)
                if len(pool) >= lookahead_pool:
                    return

        seed(ranked[:lookahead_pool])
        low_entropy_pool = sorted(candidates, key=lambda c: (c.state.entropy_max, entropy_value(instance, c.state), c.score))
        seed(low_entropy_pool[: max(1, lookahead_pool // 3)])
        lookahead_ranked = sorted(pool.values(), key=lambda c: (_lookahead_score(instance, c, entropy_alpha), c.score))
        add(lookahead_ranked[:lookahead_keep])

    add(ranked[: max(8, width // 3)])
    low_entropy = sorted(candidates, key=lambda c: (c.state.entropy_max, entropy_value(instance, c.state), c.score))
    add(low_entropy[: max(4, width // 5)])
    buffer_release = [c for c in candidates if len(c.state.stacks[c.move.from_stack]) <= 1]
    add(sorted(buffer_release, key=lambda c: c.score)[: max(4, width // 5)])
    for src, dst in sorted({(c.move.from_stack, c.move.to_stack) for c in candidates}):
        bucket = [c for c in candidates if c.move.from_stack == src and c.move.to_stack == dst]
        add(sorted(bucket, key=lambda c: c.score)[:1])
    for src in sorted({c.move.from_stack for c in candidates}):
        bucket = [c for c in candidates if c.move.from_stack == src]
        add(sorted(bucket, key=lambda c: c.score)[:1])
    for dst in sorted({c.move.to_stack for c in candidates}):
        bucket = [c for c in candidates if c.move.to_stack == dst]
        add(sorted(bucket, key=lambda c: c.score)[:1])
    add(ranked)
    out = list(selected.values())
    out.sort(key=lambda c: c.score)
    return out[:width]


def run_fast_slack_beam(
    case: str,
    instance: Instance,
    config: SolverConfig,
    *,
    tail_repair_n: int,
    tail_repair_expanded: int,
    tail_repair_candidates: int,
    lookahead_pool: int,
    lookahead_keep: int,
    rescue_repair_n: int,
    rescue_repair_candidates: int,
    rescue_reservoir_size: int,
) -> tuple[TrajectoryResult, State]:
    start = time.perf_counter()
    initial = create_initial_state(instance)
    best = initial
    h0 = hash_state(instance, initial, config.enable_canonical_hash)
    beam = [Candidate(initial, 0.0, None, h0)]
    visited = {h0: dominance_value(instance, initial)}
    expanded = generated = tail_expanded = tail_generated = 0
    tail_found = False
    rescue_reservoir: dict[tuple, State] = {}

    def remember_for_rescue(states: list[State]) -> None:
        if rescue_repair_n <= 0 or rescue_repair_candidates <= 0 or rescue_reservoir_size <= 0:
            return
        for state in states:
            if disorder_box_count(instance, state) > rescue_repair_n:
                continue
            h = hash_state(instance, state, config.enable_canonical_hash)
            old = rescue_reservoir.get(h)
            if old is None or _best_key(instance, state) < _best_key(instance, old):
                rescue_reservoir[h] = state
        if len(rescue_reservoir) > rescue_reservoir_size:
            keep = sorted(rescue_reservoir.values(), key=lambda state: _best_key(instance, state))[:rescue_reservoir_size]
            rescue_reservoir.clear()
            for state in keep:
                rescue_reservoir[hash_state(instance, state, config.enable_canonical_hash)] = state

    for depth in range(config.max_depth):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        step_candidates: list[StepCandidate] = []
        for cand in beam:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expanded += 1
            for move in generate_legal_moves(instance, cand.state):
                try:
                    nxt = apply_move(instance, cand.state, move)
                except ValueError:
                    continue
                if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                    continue
                h = hash_state(instance, nxt, config.enable_canonical_hash)
                dom = dominance_value(instance, nxt)
                if config.enable_visited_pruning and visited.get(h, math.inf) <= dom:
                    continue
                step_candidates.append(StepCandidate(nxt, move, h, _score(instance, cand.state, nxt, move), "step"))
        generated += len(step_candidates)
        if not step_candidates:
            break
        selected = _select_bucketed(
            instance,
            step_candidates,
            config.beam_width,
            lookahead_pool=lookahead_pool,
            lookahead_keep=lookahead_keep,
            entropy_alpha=config.entropy_alpha,
        )
        next_beam: list[Candidate] = []
        for item in selected:
            visited[item.hash_value] = dominance_value(instance, item.state)
            if _best_key(instance, item.state) < _best_key(instance, best):
                best = item.state
            next_beam.append(Candidate(item.state, float(item.score[1]), item.move, item.hash_value))
        beam = next_beam
        remember_for_rescue([cand.state for cand in beam])

        low_n = [cand.state for cand in beam if disorder_box_count(instance, cand.state) <= tail_repair_n]
        low_n.sort(key=lambda state: _best_key(instance, state))
        for state in low_n[: max(1, tail_repair_candidates)]:
            repaired, exp_count, gen_count = tail_repair_to_n_zero(
                instance,
                state,
                config,
                deadline=start + config.time_limit_sec,
                max_expanded=tail_repair_expanded,
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
            f"beamBestN={disorder_box_count(instance, beam[0].state)} elapsed={time.perf_counter() - start:.3f}s",
            flush=True,
        )
        if disorder_box_count(instance, best) == 0:
            break

    if disorder_box_count(instance, best) > 0 and rescue_reservoir:
        rescue_states = sorted(rescue_reservoir.values(), key=lambda state: _best_key(instance, state))
        for state in rescue_states[:rescue_repair_candidates]:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            repaired, exp_count, gen_count = tail_repair_to_n_zero(
                instance,
                state,
                config,
                deadline=start + config.time_limit_sec,
                max_expanded=tail_repair_expanded,
            )
            tail_expanded += exp_count
            tail_generated += gen_count
            if repaired is not None:
                best = repaired
                tail_found = True
                break

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += (
        f"; fast_slack_bucketed_beam; lookahead_pool={lookahead_pool}; lookahead_keep={lookahead_keep}; "
        f"rescue_repair_n={rescue_repair_n}; rescue_candidates={rescue_repair_candidates}; "
        f"tail_expanded={tail_expanded}; tail_generated={tail_generated}; tail_found={tail_found}"
    )
    row = _result_from_state(
        case,
        "fast_slack_beam",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=len(visited),
        status=status,
    )
    return row, best


def _write_result_csv(row: TrajectoryResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrajectoryResult.__annotations__.keys()))
        writer.writeheader()
        writer.writerow(row.__dict__)


def _write_rows_csv(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrajectoryResult.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _run_one(args: argparse.Namespace, zone: str, bay: str) -> tuple[TrajectoryResult, State]:
    instance = _instance_from_single_bay(zone, bay)
    config = SolverConfig(
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        time_limit_sec=args.time_limit,
        entropy_alpha=1.3,
        random_seed=37,
    )
    initial = create_initial_state(instance)
    print(
        "START "
        f"case={instance.instance_id} N0={disorder_box_count(instance, initial)} "
        f"cap={entropy_limit(instance, 1.3)} beam={config.beam_width}",
        flush=True,
    )
    return run_fast_slack_beam(
        instance.instance_id,
        instance,
        config,
        tail_repair_n=args.tail_repair_n,
        tail_repair_expanded=args.tail_repair_expanded,
        tail_repair_candidates=args.tail_repair_candidates,
        lookahead_pool=args.lookahead_pool,
        lookahead_keep=args.lookahead_keep,
        rescue_repair_n=args.rescue_repair_n,
        rescue_repair_candidates=args.rescue_repair_candidates,
        rescue_reservoir_size=args.rescue_reservoir_size,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="C1")
    parser.add_argument("--bay", default="26")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument("--beam-width", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--tail-repair-n", type=int, default=5)
    parser.add_argument("--tail-repair-expanded", type=int, default=10000)
    parser.add_argument("--tail-repair-candidates", type=int, default=2)
    parser.add_argument("--lookahead-pool", type=int, default=0)
    parser.add_argument("--lookahead-keep", type=int, default=0)
    parser.add_argument("--rescue-repair-n", type=int, default=0)
    parser.add_argument("--rescue-repair-candidates", type=int, default=0)
    parser.add_argument("--rescue-reservoir-size", type=int, default=0)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "fast_slack_beam_all_single_bays.csv",
    )
    parser.add_argument(
        "--moves-dir",
        type=Path,
        default=ROOT / "docs" / "experiments" / "fast_slack_beam_moves",
    )
    args = parser.parse_args()

    if not args.all:
        instance = _instance_from_single_bay(args.zone, args.bay)
        row, best = _run_one(args, args.zone, args.bay)
        print(f"FINAL {row}", flush=True)
        if args.output_prefix is not None:
            _write_result_csv(row, args.output_prefix.with_suffix(".csv"))
            _write_explain_csv(
                instance,
                best.moves,
                args.output_prefix.with_name(args.output_prefix.name + "_moves.csv"),
            )
        return 0

    rows: list[TrajectoryResult] = []
    start = time.perf_counter()
    cases = discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))
    for idx, (zone, bay) in enumerate(cases, start=1):
        print(f"CASE_START {idx}/{len(cases)} {zone}_{bay}", flush=True)
        instance = _instance_from_single_bay(zone, bay)
        row, best = _run_one(args, zone, bay)
        rows.append(row)
        _write_rows_csv(rows, args.output)
        _write_explain_csv(instance, best.moves, args.moves_dir / f"{instance.instance_id}_moves.csv")
        print(
            f"CASE_DONE {idx}/{len(cases)} {row.case} finalN={row.final_n} "
            f"moves={row.moves} runtime={row.runtime_sec:.3f}s",
            flush=True,
        )
    solved = sum(1 for row in rows if row.final_n == 0)
    print(
        f"SUMMARY cases={len(rows)} solved={solved} runtime={sum(row.runtime_sec for row in rows):.3f}s "
        f"wall={time.perf_counter() - start:.3f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
