"""Explainable rule macro beam V2 with slack-protected selection.

V1 optimizes N(Future) aggressively. That can spend the whole G budget before
the state is close enough for exact tail repair. V2 keeps the same
interpretable macro buckets, but ranks frontier states by a target-completion
risk that preserves enough slack for the remaining disorder.
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

from benchmark_explainable_rule_macro_beam import (  # noqa: E402
    _commit_move_index,
    _state_rank_from_macro,
    _write_explain_csv,
    classify_move,
)
from benchmark_full_real_business_algorithms import TrajectoryResult, _result_from_state  # noqa: E402
from benchmark_macro_reservoir_qubo_beam import MacroCandidate, build_macro_candidates, tail_repair_to_n_zero  # noqa: E402
from container_reshuffle.core import Instance, Move, State, apply_move, create_initial_state, hash_state  # noqa: E402
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
from container_reshuffle.scoring import dominance_value, score_state  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.beam import Candidate  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


def _auto_config_v2(instance: Instance, time_limit: float) -> tuple[SolverConfig, dict[str, int]]:
    stacks = instance.stack_count
    cap = entropy_limit(instance, 1.3)
    width = max(48, 8 * stacks)
    config = SolverConfig(
        max_depth=cap,
        beam_width=width,
        time_limit_sec=time_limit,
        entropy_alpha=1.3,
        random_seed=37,
    )
    rule_sizes = {
        "macro_depth": 2,
        "commit_length": 2,
        "global_keep": max(5, stacks // 2),
        "source_keep": 3,
        "dest_keep": 1,
        "plateau_keep": max(6, stacks // 2),
        "max_reservoir": stacks + 8,
        "max_macros_per_state": 3 * stacks,
        "tail_repair_n": max(8, stacks),
        "tail_repair_expanded": 500_000,
        "tail_states_per_layer": 4,
    }
    return config, rule_sizes


def _desired_slack(n_value: int) -> int:
    if n_value <= 0:
        return 0
    if n_value <= 3:
        return 1
    if n_value <= 8:
        return 2
    if n_value <= 14:
        return 3
    return 4


def _slack_features(instance: Instance, state: State) -> tuple[int, int, int, int]:
    cap = entropy_limit(instance, 1.3)
    n_value = disorder_box_count(instance, state)
    g_value = entropy_value(instance, state)
    slack = cap - g_value
    low_slack = max(0, _desired_slack(n_value) - slack)
    remaining_moves = max(0, cap - state.move_count)
    clearance_risk = max(0, n_value - remaining_moves - 2)
    return n_value, slack, low_slack, clearance_risk


def _state_key_v2(instance: Instance, state: State) -> tuple:
    n_value, slack, low_slack, clearance_risk = _slack_features(instance, state)
    return (
        clearance_risk,
        low_slack,
        n_value,
        state.entropy_max,
        state.disorder_total,
        inversion_count(instance, state),
        bad_count(instance, state),
        dirty_stack_count(instance, state),
        -settled_box_count(instance, state),
        state.move_count,
        -slack,
    )


def _best_state_key(instance: Instance, state: State) -> tuple:
    return (
        disorder_box_count(instance, state),
        state.entropy_max,
        state.disorder_total,
        state.move_count,
    )


def _macro_score_v2(instance: Instance, macro: MacroCandidate) -> float:
    state = macro.leaf_state
    n_value, slack, low_slack, clearance_risk = _slack_features(instance, state)
    gain = macro.root_n - macro.leaf_n
    immediate_gain = macro.root_n - macro.first_n
    plateau_bonus = 1.0 if macro.first_n == macro.root_n and gain > 0 else 0.0
    return (
        1_000_000.0 * n_value
        + 2_000_000.0 * clearance_risk
        + 900_000.0 * low_slack * low_slack
        + 65_000.0 * state.entropy_max
        + 8.0 * state.disorder_total
        + 25.0 * state.move_count
        + 1.0 * inversion_count(instance, state)
        + 2.0 * bad_count(instance, state)
        + 8.0 * dirty_stack_count(instance, state)
        - 50.0 * settled_box_count(instance, state)
        - 7_500.0 * gain
        - 1_000.0 * max(0, immediate_gain)
        - 1_200.0 * plateau_bonus
        - 200.0 * max(0, slack)
    )


def _add_macros(
    selected: dict[tuple, MacroCandidate],
    macros: Iterable[MacroCandidate],
    instance: Instance,
) -> None:
    for macro in macros:
        key = (macro.commit_hash, macro.leaf_hash)
        old = selected.get(key)
        if old is None or (_macro_score_v2(instance, macro), _state_rank_from_macro(macro)) < (
            _macro_score_v2(instance, old),
            _state_rank_from_macro(old),
        ):
            selected[key] = macro


def select_explainable_macros_v2(
    instance: Instance,
    macros: list[MacroCandidate],
    width: int,
) -> list[MacroCandidate]:
    ranked = sorted(macros, key=lambda m: (_macro_score_v2(instance, m), _state_key_v2(instance, m.leaf_state)))
    selected: dict[tuple, MacroCandidate] = {}

    _add_macros(selected, ranked[: max(1, width // 2)], instance)

    for src in sorted({macro.first_move.from_stack for macro in macros}):
        bucket = [macro for macro in macros if macro.first_move.from_stack == src]
        bucket.sort(key=lambda m: (_macro_score_v2(instance, m), _state_key_v2(instance, m.leaf_state)))
        _add_macros(selected, bucket[:2], instance)

    for dst in sorted({macro.first_move.to_stack for macro in macros}):
        bucket = [macro for macro in macros if macro.first_move.to_stack == dst]
        bucket.sort(key=lambda m: (_macro_score_v2(instance, m), _state_key_v2(instance, m.leaf_state)))
        _add_macros(selected, bucket[:1], instance)

    # Completion-risk representatives keep branches that have enough slack even
    # when their current N is not the best in the layer.
    seen: set[tuple[int, int, int]] = set()
    for macro in ranked:
        n_value, slack, low_slack, clearance_risk = _slack_features(instance, macro.leaf_state)
        bucket = (clearance_risk, low_slack, min(n_value // 3, 8))
        if bucket in seen:
            continue
        seen.add(bucket)
        _add_macros(selected, [macro], instance)

    if len(selected) < width:
        _add_macros(selected, ranked, instance)

    out = list(selected.values())
    out.sort(key=lambda m: (_macro_score_v2(instance, m), _state_key_v2(instance, m.leaf_state)))
    return out[:width]


def _tail_candidates(instance: Instance, beam: list[Candidate], rule_sizes: dict[str, int]) -> list[State]:
    out: list[State] = []
    cap = entropy_limit(instance, 1.3)
    for cand in beam:
        state = cand.state
        n_value = disorder_box_count(instance, state)
        remaining = cap - state.move_count
        if n_value <= rule_sizes["tail_repair_n"] or (n_value <= remaining + 2 and n_value <= 14):
            out.append(state)
    out.sort(key=lambda state: (_state_key_v2(instance, state), _best_state_key(instance, state)))
    return out[: rule_sizes["tail_states_per_layer"]]


def run_explainable_rule_macro_beam_v2(
    case: str,
    instance: Instance,
    config: SolverConfig,
    rule_sizes: dict[str, int],
) -> tuple[TrajectoryResult, State]:
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

        selected = select_explainable_macros_v2(instance, all_macros, config.beam_width)
        filler = sorted(all_macros, key=lambda m: (_macro_score_v2(instance, m), _state_key_v2(instance, m.leaf_state)))
        next_by_state: dict[tuple, Candidate] = {}
        for macro in selected + filler:
            h = macro.commit_hash
            dom = dominance_value(instance, macro.commit_state)
            if config.enable_visited_pruning and visited.get(h, math.inf) <= dom:
                continue
            move = macro.moves[_commit_move_index(macro, rule_sizes["commit_length"])]
            candidate = Candidate(macro.commit_state, _macro_score_v2(instance, macro), move, h)
            old = next_by_state.get(h)
            if old is None or (candidate.score, _state_key_v2(instance, candidate.state)) < (
                old.score,
                _state_key_v2(instance, old.state),
            ):
                next_by_state[h] = candidate
            if len(next_by_state) >= config.beam_width:
                break
        if not next_by_state:
            break

        for candidate in next_by_state.values():
            visited[candidate.hash_value] = dominance_value(instance, candidate.state)
            if _best_state_key(instance, candidate.state) < _best_state_key(instance, best):
                best = candidate.state

        beam = sorted(next_by_state.values(), key=lambda c: (_state_key_v2(instance, c.state), c.score))[
            : config.beam_width
        ]

        for state in _tail_candidates(instance, beam, rule_sizes):
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
            f"beamSlack={_slack_features(instance, beam[0].state)[1]} "
            f"elapsed={time.perf_counter() - start:.2f}s",
            flush=True,
        )
        if disorder_box_count(instance, best) == 0:
            break

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += (
        "; explainable_rules_v2=slack_protected/direct/source/dest/risk_bucket/tail"
        f"; auto_width={config.beam_width}; macro_depth={rule_sizes['macro_depth']}; "
        f"commit_length={rule_sizes['commit_length']}; tail_repair_n={rule_sizes['tail_repair_n']}; "
        f"tail_expanded={tail_expanded}; tail_generated={tail_generated}; tail_found={tail_found}"
    )
    row = _result_from_state(
        case,
        "explainable_rule_macro_beam_v2",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=len(visited),
        status=status,
    )
    return row, best


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
    config, rule_sizes = _auto_config_v2(instance, args.time_limit)
    initial = create_initial_state(instance)
    print(
        "START "
        f"case={instance.instance_id} stacks={instance.stack_count} H={instance.max_height} "
        f"N0={disorder_box_count(instance, initial)} cap={entropy_limit(instance, 1.3)} "
        f"auto_width={config.beam_width} rules={rule_sizes}",
        flush=True,
    )
    row, best = run_explainable_rule_macro_beam_v2(instance.instance_id, instance, config, rule_sizes)
    print(f"FINAL {row}", flush=True)
    _print_trace(instance, best.moves)

    prefix = args.output_prefix or ROOT / "docs" / "experiments" / f"explainable_rule_macro_beam_v2_{instance.instance_id}"
    _write_result_csv(row, prefix.with_suffix(".csv"))
    _write_explain_csv(instance, best.moves, prefix.with_name(prefix.name + "_moves.csv"))
    print(f"wrote {prefix.with_suffix('.csv')}", flush=True)
    print(f"wrote {prefix.with_name(prefix.name + '_moves.csv')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
