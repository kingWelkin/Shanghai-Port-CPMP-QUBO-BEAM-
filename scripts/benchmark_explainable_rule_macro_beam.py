"""Explainable rule-based macro beam experiment.

This variant intentionally avoids QUBO sampling and hand-tuned per-case knobs.
It uses deterministic rule buckets:

- direct N(Future) reduction,
- source-stack representatives,
- destination representatives,
- safe staging / plateau moves,
- low-N exact tail repair.

The default sizes are derived from the number of stacks, making the run easier
to explain than a parameter sweep.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_full_real_business_algorithms import TrajectoryResult, _result_from_state  # noqa: E402
from benchmark_macro_reservoir_qubo_beam import (  # noqa: E402
    MacroCandidate,
    _best_state_key,
    _state_rank,
    build_macro_candidates,
    tail_repair_to_n_zero,
)
from container_reshuffle.core import (  # noqa: E402
    Instance,
    Move,
    State,
    apply_move,
    create_initial_state,
    hash_state,
)
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
from container_reshuffle.scoring import dominance_value, score_state  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.beam import Candidate  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


def _auto_config(instance: Instance, time_limit: float) -> tuple[SolverConfig, dict[str, int]]:
    stacks = instance.stack_count
    cap = entropy_limit(instance, 1.3)
    config = SolverConfig(
        max_depth=cap,
        beam_width=max(24, 6 * stacks),
        time_limit_sec=time_limit,
        entropy_alpha=1.3,
        random_seed=37,
    )
    rule_sizes = {
        "macro_depth": 2,
        "commit_length": 2,
        "global_keep": max(4, stacks // 3),
        "source_keep": 2,
        "dest_keep": 1,
        "plateau_keep": max(5, stacks // 2),
        "max_reservoir": stacks + 4,
        "max_macros_per_state": 2 * stacks,
        "tail_repair_n": max(3, stacks // 2),
        "tail_repair_expanded": 200_000,
    }
    return config, rule_sizes


def _add_macros(
    selected: dict[tuple, MacroCandidate],
    macros: Iterable[MacroCandidate],
) -> None:
    for macro in macros:
        key = (macro.commit_hash, macro.leaf_hash)
        old = selected.get(key)
        if old is None or (macro.score, _state_rank_from_macro(macro)) < (old.score, _state_rank_from_macro(old)):
            selected[key] = macro


def _state_rank_from_macro(macro: MacroCandidate) -> tuple:
    # Local helper so representative selection is deterministic without needing
    # the instance inside _add_macros.
    state = macro.leaf_state
    return (
        macro.leaf_n,
        state.entropy_max,
        state.disorder_total,
        state.move_count,
    )


def select_explainable_macros(
    instance: Instance,
    macros: list[MacroCandidate],
    width: int,
) -> list[MacroCandidate]:
    """Select macro paths by named deterministic rule buckets."""

    ranked = sorted(macros, key=lambda m: (m.score, _state_rank(instance, m.leaf_state)))
    selected: dict[tuple, MacroCandidate] = {}

    elite_count = max(1, width // 2)
    _add_macros(selected, ranked[:elite_count])

    for src in sorted({macro.first_move.from_stack for macro in macros}):
        bucket = [macro for macro in macros if macro.first_move.from_stack == src]
        bucket.sort(key=lambda m: (m.score, _state_rank(instance, m.leaf_state)))
        _add_macros(selected, bucket[:2])

    for dst in sorted({macro.first_move.to_stack for macro in macros}):
        bucket = [macro for macro in macros if macro.first_move.to_stack == dst]
        bucket.sort(key=lambda m: (m.score, _state_rank(instance, m.leaf_state)))
        _add_macros(selected, bucket[:1])

    # Keep representatives for each reachable terminal-N bucket. This is the
    # interpretable replacement for a diversity penalty.
    seen_n: set[int] = set()
    for macro in ranked:
        if macro.leaf_n in seen_n:
            continue
        seen_n.add(macro.leaf_n)
        _add_macros(selected, [macro])

    if len(selected) < width:
        _add_macros(selected, ranked)

    out = list(selected.values())
    out.sort(key=lambda m: (m.score, _state_rank(instance, m.leaf_state)))
    return out[:width]


def _commit_move_index(macro: MacroCandidate, commit_length: int) -> int:
    return min(len(macro.moves), max(1, commit_length)) - 1


def run_explainable_rule_macro_beam(
    case: str,
    instance: Instance,
    config: SolverConfig,
    rule_sizes: dict[str, int],
) -> tuple[TrajectoryResult, State]:
    """Run deterministic explainable macro beam."""

    start = time.perf_counter()
    initial = create_initial_state(instance)
    best = initial
    h0 = hash_state(instance, initial, config.enable_canonical_hash)
    beam = [Candidate(initial, score_state(instance, initial, config.score_weights), None, h0)]
    visited = {h0: dominance_value(instance, initial)}
    expanded = generated = tail_expanded = tail_generated = 0
    tail_found = False

    for depth in range(config.max_depth):
        if time.perf_counter() - start > config.time_limit_sec:
            break

        all_macros: list[MacroCandidate] = []
        for cand in beam:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expanded += 1
            all_macros.extend(
                build_macro_candidates(
                    instance,
                    cand.state,
                    config,
                    macro_depth=rule_sizes["macro_depth"],
                    commit_length=rule_sizes["commit_length"],
                    global_keep=rule_sizes["global_keep"],
                    source_keep=rule_sizes["source_keep"],
                    dest_keep=rule_sizes["dest_keep"],
                    plateau_keep=rule_sizes["plateau_keep"],
                    max_reservoir=rule_sizes["max_reservoir"],
                    max_macros_per_state=rule_sizes["max_macros_per_state"],
                )
            )
        generated += len(all_macros)
        if not all_macros:
            break

        selected = select_explainable_macros(instance, all_macros, config.beam_width)
        filler = sorted(all_macros, key=lambda m: (m.score, _state_rank(instance, m.leaf_state)))
        next_by_state: dict[tuple, Candidate] = {}
        for macro in selected + filler:
            h = macro.commit_hash
            dom = dominance_value(instance, macro.commit_state)
            if config.enable_visited_pruning and visited.get(h, math.inf) <= dom:
                continue
            move = macro.moves[_commit_move_index(macro, rule_sizes["commit_length"])]
            old = next_by_state.get(h)
            candidate = Candidate(macro.commit_state, macro.score, move, h)
            if old is None or candidate.score < old.score:
                next_by_state[h] = candidate
            if len(next_by_state) >= config.beam_width:
                break
        if not next_by_state:
            break

        for candidate in next_by_state.values():
            visited[candidate.hash_value] = dominance_value(instance, candidate.state)
            if _best_state_key(instance, candidate.state) < _best_state_key(instance, best):
                best = candidate.state

        beam = sorted(next_by_state.values(), key=lambda c: (_best_state_key(instance, c.state), c.score))[
            : config.beam_width
        ]

        low_n_states = [
            cand.state for cand in beam if disorder_box_count(instance, cand.state) <= rule_sizes["tail_repair_n"]
        ]
        low_n_states.sort(key=lambda state: _best_state_key(instance, state))
        for state in low_n_states[: min(6, len(low_n_states))]:
            repaired, exp_count, gen_count = tail_repair_to_n_zero(
                instance,
                state,
                config,
                deadline=start + config.time_limit_sec,
                max_expanded=rule_sizes["tail_repair_expanded"],
            )
            tail_expanded += exp_count
            tail_generated += gen_count
            if repaired is not None:
                best = repaired
                tail_found = True
                break

        print(
            "LAYER "
            f"depth={depth + 1} beam={len(beam)} macros={len(all_macros)} "
            f"bestN={disorder_box_count(instance, best)} beamBestN={disorder_box_count(instance, beam[0].state)} "
            f"elapsed={time.perf_counter() - start:.2f}s",
            flush=True,
        )
        if disorder_box_count(instance, best) == 0:
            break

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += (
        "; explainable_rules=direct/source/dest/n_bucket/tail"
        f"; auto_width={config.beam_width}; macro_depth={rule_sizes['macro_depth']}; "
        f"commit_length={rule_sizes['commit_length']}; tail_repair_n={rule_sizes['tail_repair_n']}; "
        f"tail_expanded={tail_expanded}; tail_generated={tail_generated}; tail_found={tail_found}"
    )
    row = _result_from_state(
        case,
        "explainable_rule_macro_beam",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=len(visited),
        status=status,
    )
    return row, best


def classify_move(instance: Instance, before: State, after: State, move: Move) -> str:
    n_before = disorder_box_count(instance, before)
    n_after = disorder_box_count(instance, after)
    cap = entropy_limit(instance, 1.3)
    if n_after < n_before:
        return "DIRECT_DROP_N"
    if bad_count(instance, after) < bad_count(instance, before):
        return "UNLOCK_BAD_STACK"
    if well_prefix_gain(instance, before, after) > 0:
        return "INCREASE_WELL_PREFIX"
    dst_before = before.stacks[move.to_stack]
    dst_safe = not dst_before or dst_before[-1] > move.box
    src_after = after.stacks[move.from_stack]
    src_clean_or_freed = well_prefix_len(src_after) == len(src_after)
    if dst_safe and src_clean_or_freed:
        return "SAFE_STAGING_SLOT"
    if entropy_value(instance, after) <= cap - 2:
        return "SLACK_PRESERVING_STAGE"
    return "BUDGETED_STAGE"


def _write_explain_csv(instance: Instance, moves: list[Move], path: Path) -> None:
    def label(idx: int) -> str:
        return str(idx + 1) if idx < 9 else "A"

    rows = []
    state = create_initial_state(instance)
    for step, move in enumerate(moves, start=1):
        before = state
        after = apply_move(instance, state, move)
        rows.append(
            {
                "step": step,
                "box_rank": move.box,
                "from_stack": label(move.from_stack),
                "to_stack": label(move.to_stack),
                "reason": classify_move(instance, before, after, move),
                "n_before": disorder_box_count(instance, before),
                "n_after": disorder_box_count(instance, after),
                "g_after": entropy_value(instance, after),
                "max_g_after": after.entropy_max,
                "bad_after": bad_count(instance, after),
                "dirty_after": dirty_stack_count(instance, after),
                "inversions_after": inversion_count(instance, after),
                "settled_after": settled_box_count(instance, after),
            }
        )
        state = after

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else ["step"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_trace(instance: Instance, moves: list[Move]) -> None:
    def label(idx: int) -> str:
        return str(idx + 1) if idx < 9 else "A"

    state = create_initial_state(instance)
    print(f"TRACE 0 N={disorder_box_count(instance, state)} G={entropy_value(instance, state)}")
    for step, move in enumerate(moves, start=1):
        before = state
        state = apply_move(instance, state, move)
        print(
            "TRACE "
            f"{step} {move.box}:{label(move.from_stack)}->{label(move.to_stack)} "
            f"reason={classify_move(instance, before, state, move)} "
            f"N={disorder_box_count(instance, state)} G={entropy_value(instance, state)}",
            flush=True,
        )


def _write_result_csv(row: TrajectoryResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrajectoryResult.__annotations__.keys()))
        writer.writeheader()
        writer.writerow(row.__dict__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="C1")
    parser.add_argument("--bay", default="26")
    parser.add_argument("--time-limit", type=float, default=80.0)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()

    instance = _instance_from_single_bay(args.zone, args.bay)
    config, rule_sizes = _auto_config(instance, args.time_limit)
    initial = create_initial_state(instance)
    print(
        "START "
        f"case={instance.instance_id} stacks={instance.stack_count} H={instance.max_height} "
        f"N0={disorder_box_count(instance, initial)} cap={entropy_limit(instance, 1.3)} "
        f"auto_width={config.beam_width} rules={rule_sizes}",
        flush=True,
    )
    row, best = run_explainable_rule_macro_beam(instance.instance_id, instance, config, rule_sizes)
    print(f"FINAL {row}", flush=True)
    _print_trace(instance, best.moves)

    prefix = args.output_prefix or ROOT / "docs" / "experiments" / f"explainable_rule_macro_beam_{instance.instance_id}"
    _write_result_csv(row, prefix.with_suffix(".csv"))
    _write_explain_csv(instance, best.moves, prefix.with_name(prefix.name + "_moves.csv"))
    print(f"wrote {prefix.with_suffix('.csv')}", flush=True)
    print(f"wrote {prefix.with_name(prefix.name + '_moves.csv')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
