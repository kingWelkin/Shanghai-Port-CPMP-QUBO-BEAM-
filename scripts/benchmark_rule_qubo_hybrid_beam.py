"""Rule-core plus QUBO-supplement macro beam experiment.

This variant keeps the explainable deterministic macro beam as the protected
core, then uses QUBO only to add a small set of diverse supplemental branches.
The point is to combine QUBO without letting it prune away the known-good rule
trajectory.
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

from benchmark_explainable_rule_macro_beam import (  # noqa: E402
    _commit_move_index,
    _write_explain_csv,
    select_explainable_macros,
)
from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from benchmark_full_real_business_algorithms import TrajectoryResult, _result_from_state  # noqa: E402
from benchmark_macro_reservoir_qubo_beam import (  # noqa: E402
    MacroCandidate,
    _best_state_key,
    _state_rank,
    build_macro_candidates,
    select_macros_with_qubo,
    tail_repair_to_n_zero,
)
from container_reshuffle.core import Instance, State, create_initial_state, hash_state  # noqa: E402
from container_reshuffle.features import disorder_box_count, entropy_limit  # noqa: E402
from container_reshuffle.scoring import dominance_value, score_state  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.beam import Candidate  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


def _auto_config(instance: Instance, time_limit: float) -> tuple[SolverConfig, dict[str, int | float]]:
    stacks = instance.stack_count
    cap = entropy_limit(instance, 1.3)
    rule_core_width = max(24, 3 * stacks + 4)
    slack_core_width = max(5, stacks - 3)
    qubo_supplement_width = max(5, stacks - 3)
    config = SolverConfig(
        max_depth=cap,
        beam_width=max(36, rule_core_width + slack_core_width + qubo_supplement_width),
        time_limit_sec=time_limit,
        max_qubo_vars=75,
        qubo_penalty_a=30.0,
        qubo_num_reads=4,
        qubo_time_limit_sec=0.01,
        qubo_sa_iterations=80,
        entropy_alpha=1.3,
        random_seed=37,
    )
    params: dict[str, int | float] = {
        "rule_core_width": rule_core_width,
        "slack_core_width": slack_core_width,
        "qubo_supplement_width": qubo_supplement_width,
        "macro_depth": 2,
        "commit_length": 2,
        "global_keep": max(4, stacks // 3),
        "source_keep": 2,
        "dest_keep": 1,
        "plateau_keep": max(5, stacks // 2),
        "max_reservoir": max(18, 2 * stacks + 4),
        "max_macros_per_state": max(20, 2 * stacks + 8),
        "tail_repair_n": max(5, stacks - 2),
        "tail_repair_expanded": 120_000,
    }
    return config, params


def _apply_cli_overrides(config: SolverConfig, params: dict[str, int | float], args: argparse.Namespace) -> None:
    for name in (
        "rule_core_width",
        "slack_core_width",
        "qubo_supplement_width",
        "macro_depth",
        "commit_length",
        "global_keep",
        "source_keep",
        "dest_keep",
        "plateau_keep",
        "max_reservoir",
        "max_macros_per_state",
        "tail_repair_n",
        "tail_repair_expanded",
    ):
        value = getattr(args, name, None)
        if value is not None:
            params[name] = value
    if args.beam_width is not None:
        config.beam_width = args.beam_width
    else:
        config.beam_width = int(params["rule_core_width"]) + int(params["slack_core_width"]) + int(
            params["qubo_supplement_width"]
        )
    if args.max_qubo_vars is not None:
        config.max_qubo_vars = args.max_qubo_vars
    if args.qubo_reads is not None:
        config.qubo_num_reads = args.qubo_reads
    if args.qubo_iterations is not None:
        config.qubo_sa_iterations = args.qubo_iterations
    if args.qubo_time_limit is not None:
        config.qubo_time_limit_sec = args.qubo_time_limit


def _macro_key(macro: MacroCandidate) -> tuple:
    return (macro.commit_hash, macro.leaf_hash)


def _slack_key(instance: Instance, macro: MacroCandidate) -> tuple:
    root_slack_used = macro.leaf_state.entropy_max - macro.root_n
    return (
        macro.leaf_state.entropy_max,
        root_slack_used,
        macro.leaf_n,
        macro.leaf_state.disorder_total,
        _state_rank(instance, macro.leaf_state),
        macro.score,
    )


def _select_slack_macros(
    instance: Instance,
    macros: list[MacroCandidate],
    width: int,
    excluded: set[tuple],
) -> list[MacroCandidate]:
    slack_pool = [
        macro
        for macro in macros
        if _macro_key(macro) not in excluded
        and macro.leaf_n <= macro.root_n
        and macro.leaf_state.entropy_max <= entropy_limit(instance, 1.3) - 3
    ]
    if len(slack_pool) < width:
        slack_pool = [macro for macro in macros if _macro_key(macro) not in excluded]
    slack_pool.sort(key=lambda macro: _slack_key(instance, macro))
    return slack_pool[:width]


def _select_hybrid_macros(
    instance: Instance,
    macros: list[MacroCandidate],
    config: SolverConfig,
    params: dict[str, int | float],
) -> tuple[list[MacroCandidate], int, float, int]:
    rule_core_width = int(params["rule_core_width"])
    slack_core_width = int(params["slack_core_width"])
    qubo_width = int(params["qubo_supplement_width"])
    rule_core = select_explainable_macros(instance, macros, rule_core_width)
    rule_keys = {_macro_key(macro) for macro in rule_core}
    slack_core = _select_slack_macros(instance, macros, slack_core_width, rule_keys)
    protected_keys = rule_keys | {_macro_key(macro) for macro in slack_core}
    supplement_pool = [macro for macro in macros if _macro_key(macro) not in protected_keys]
    if len(supplement_pool) < qubo_width + 2:
        supplement_pool = macros

    qubo_selected, qvars, qtime = select_macros_with_qubo(
        instance,
        supplement_pool,
        config,
        target_count=qubo_width,
        state_similarity_weight=0.05,
        same_first_weight=0.45,
        same_source_weight=0.16,
        same_dest_weight=0.05,
    )

    selected: dict[tuple, MacroCandidate] = {}
    ordered: list[MacroCandidate] = []
    for macro in rule_core + slack_core + qubo_selected:
        old = selected.get(_macro_key(macro))
        if old is None or macro.score < old.score:
            selected[_macro_key(macro)] = macro
            ordered.append(macro)
    return ordered, qvars, qtime, len(rule_core)


def _tail_budget(n_value: int, params: dict[str, int | float]) -> int:
    full_budget = int(params["tail_repair_expanded"])
    if n_value <= 5:
        return full_budget
    if n_value <= 7:
        return min(full_budget, 50_000)
    return min(full_budget, 20_000)


def run_rule_qubo_hybrid_beam(
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
    expanded = generated = qcalls = qvars_total = 0
    tail_expanded = tail_generated = 0
    qtime_total = 0.0
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
                    macro_depth=int(params["macro_depth"]),
                    commit_length=int(params["commit_length"]),
                    global_keep=int(params["global_keep"]),
                    source_keep=int(params["source_keep"]),
                    dest_keep=int(params["dest_keep"]),
                    plateau_keep=int(params["plateau_keep"]),
                    max_reservoir=int(params["max_reservoir"]),
                    max_macros_per_state=int(params["max_macros_per_state"]),
                )
            )
        generated += len(all_macros)
        if not all_macros:
            break

        selected, qvars, qtime, rule_count = _select_hybrid_macros(instance, all_macros, config, params)
        qcalls += 1 if qvars else 0
        qvars_total += qvars
        qtime_total += qtime

        filler = sorted(all_macros, key=lambda macro: (macro.score, _state_rank(instance, macro.leaf_state)))
        next_by_state: dict[tuple, Candidate] = {}
        for macro in selected + filler:
            h = macro.commit_hash
            dom = dominance_value(instance, macro.commit_state)
            if config.enable_visited_pruning and visited.get(h, math.inf) <= dom:
                continue
            move = macro.moves[_commit_move_index(macro, int(params["commit_length"]))]
            candidate = Candidate(macro.commit_state, macro.score, move, h)
            old = next_by_state.get(h)
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
            cand.state for cand in beam if disorder_box_count(instance, cand.state) <= int(params["tail_repair_n"])
        ]
        low_n_states.sort(key=lambda state: _best_state_key(instance, state))
        if low_n_states:
            best_tail_n = disorder_box_count(instance, low_n_states[0])
            tail_state_limit = 1 if best_tail_n >= 8 else 3 if best_tail_n >= 6 else 6
        else:
            tail_state_limit = 0
        for state in low_n_states[: min(tail_state_limit, len(low_n_states))]:
            n_value = disorder_box_count(instance, state)
            repaired, exp_count, gen_count = tail_repair_to_n_zero(
                instance,
                state,
                config,
                deadline=start + config.time_limit_sec,
                max_expanded=_tail_budget(n_value, params),
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
            f"rule_core={rule_count} qvars={qvars} "
            f"bestN={disorder_box_count(instance, best)} "
            f"beamBestN={disorder_box_count(instance, beam[0].state)} "
            f"elapsed={time.perf_counter() - start:.2f}s",
            flush=True,
        )
        if disorder_box_count(instance, best) == 0:
            break

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += (
        "; hybrid=protected_rule_core+qubo_supplement"
        f"; rule_core_width={params['rule_core_width']}; slack_core_width={params['slack_core_width']}; "
        f"qubo_supplement_width={params['qubo_supplement_width']}; "
        f"tail_expanded={tail_expanded}; tail_generated={tail_generated}; tail_found={tail_found}"
    )
    row = _result_from_state(
        case,
        "rule_qubo_hybrid_beam",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=len(visited),
        qubo_calls=qcalls,
        qubo_total_vars=qvars_total,
        qubo_time_sec=qtime_total,
        status=status,
    )
    return row, best


def _write_result_csv(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrajectoryResult.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _run_case(args: argparse.Namespace, zone: str, bay: str, output_prefix: Path | None) -> TrajectoryResult:
    instance = _instance_from_single_bay(zone, bay)
    config, params = _auto_config(instance, args.time_limit)
    _apply_cli_overrides(config, params, args)
    initial = create_initial_state(instance)
    print(
        "START "
        f"case={instance.instance_id} N0={disorder_box_count(instance, initial)} "
        f"cap={entropy_limit(instance, 1.3)} beam={config.beam_width} params={params}",
        flush=True,
    )
    row, best = run_rule_qubo_hybrid_beam(instance.instance_id, instance, config, params)
    print(f"FINAL {row}", flush=True)
    if output_prefix is not None:
        _write_result_csv([row], output_prefix.with_suffix(".csv"))
        _write_explain_csv(instance, best.moves, output_prefix.with_name(output_prefix.name + "_moves.csv"))
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="C1")
    parser.add_argument("--bay", default="26")
    parser.add_argument("--time-limit", type=float, default=80.0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--beam-width", type=int)
    parser.add_argument("--rule-core-width", type=int)
    parser.add_argument("--slack-core-width", type=int)
    parser.add_argument("--qubo-supplement-width", type=int)
    parser.add_argument("--macro-depth", type=int)
    parser.add_argument("--commit-length", type=int)
    parser.add_argument("--global-keep", type=int)
    parser.add_argument("--source-keep", type=int)
    parser.add_argument("--dest-keep", type=int)
    parser.add_argument("--plateau-keep", type=int)
    parser.add_argument("--max-reservoir", type=int)
    parser.add_argument("--max-macros-per-state", type=int)
    parser.add_argument("--tail-repair-n", type=int)
    parser.add_argument("--tail-repair-expanded", type=int)
    parser.add_argument("--max-qubo-vars", type=int)
    parser.add_argument("--qubo-time-limit", type=float)
    parser.add_argument("--qubo-reads", type=int)
    parser.add_argument("--qubo-iterations", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "rule_qubo_hybrid_beam_all_single_bays.csv",
    )
    parser.add_argument(
        "--moves-dir",
        type=Path,
        default=ROOT / "docs" / "experiments" / "rule_qubo_hybrid_beam_moves",
    )
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()

    if not args.all:
        _run_case(args, args.zone, args.bay, args.output_prefix)
        return 0

    rows: list[TrajectoryResult] = []
    start = time.perf_counter()
    cases = discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))
    for idx, (zone, bay) in enumerate(cases, start=1):
        instance = _instance_from_single_bay(zone, bay)
        config, params = _auto_config(instance, args.time_limit)
        _apply_cli_overrides(config, params, args)
        print(f"CASE_START {idx}/{len(cases)} {instance.instance_id}", flush=True)
        row, best = run_rule_qubo_hybrid_beam(instance.instance_id, instance, config, params)
        rows.append(row)
        _write_result_csv(rows, args.output)
        _write_explain_csv(instance, best.moves, args.moves_dir / f"{instance.instance_id}_moves.csv")
        print(
            f"CASE_DONE {idx}/{len(cases)} {row.case} finalN={row.final_n} moves={row.moves} "
            f"runtime={row.runtime_sec:.2f}s qcalls={row.qubo_calls}",
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
