"""Fixed adaptive fast/QUBO rescue benchmark.

This script is intentionally a single fixed policy:

1. Run a fast slack beam with a protected, tiny QUBO supplement.
2. If and only if that pass does not reach N=0, restart a wider rescue beam.
3. The rescue beam embeds QUBO locally by selecting diverse first-move seeds
   before the deterministic rescue search proceeds.

It is not a best-of portfolio: the first N=0 stage is returned, and the second
stage only exists as a fixed failure-recovery branch.
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
from benchmark_fast_qubo_integrated_v1 import (  # noqa: E402
    ACCEPTANCE_FIELDS,
    _exact_moves_by_case,
    _move_gap,
    _parse_case,
    _rel,
    _validate_final,
)
from benchmark_fast_slack_beam import StepCandidate, _best_key, _score, _select_bucketed  # noqa: E402
from benchmark_fast_slack_qubo_beam import (  # noqa: E402
    _select_qubo_indices,
    _select_with_qubo_supplement,
    run_fast_slack_qubo_beam,
)
from benchmark_full_real_business_algorithms import TrajectoryResult, _result_from_state  # noqa: E402
from benchmark_macro_reservoir_qubo_beam import tail_repair_to_n_zero  # noqa: E402
from container_reshuffle.core import (  # noqa: E402
    Instance,
    State,
    apply_move,
    create_initial_state,
    generate_legal_moves,
    hash_state,
)
from container_reshuffle.features import disorder_box_count, entropy_limit  # noqa: E402
from container_reshuffle.scoring import cpmp_business_feasible, dominance_value  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.beam import Candidate  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


SOLVER_NAME = "fast_qubo_adaptive_rescue_v1"


@dataclass
class AdaptiveRun:
    row: TrajectoryResult
    best_state: State
    business_ok: bool
    goal_validation_ok: bool
    validation_msg: str


def _case_filter(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return [case if case.endswith("_single") else f"{case}_single" for case in args.cases]
    return [f"{zone}_{bay}_single" for zone, bay in discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))]


def _build_config(
    *,
    beam_width: int,
    time_limit: float,
    max_depth: int,
    args: argparse.Namespace,
) -> SolverConfig:
    return SolverConfig(
        max_depth=max_depth,
        beam_width=beam_width,
        time_limit_sec=time_limit,
        max_qubo_vars=args.max_qubo_vars,
        qubo_penalty_a=args.qubo_penalty_a,
        diversity_weight=args.diversity_weight,
        qubo_num_reads=args.qubo_reads,
        qubo_time_limit_sec=args.qubo_time_limit,
        qubo_sa_iterations=args.qubo_iterations,
        qubo_repair_hamming_slack=2,
        entropy_alpha=1.3,
        random_seed=args.seed,
    )


def _qubo_seeded_initial_beam(
    instance: Instance,
    config: SolverConfig,
    *,
    seed_width: int,
) -> tuple[list[Candidate], set[tuple], int, int, float]:
    initial = create_initial_state(instance)
    h0 = hash_state(instance, initial, config.enable_canonical_hash)
    if seed_width <= 0:
        return [Candidate(initial, 0.0, None, h0)], {h0}, 0, 0, 0.0
    candidates: list[StepCandidate] = []
    for move in generate_legal_moves(instance, initial):
        try:
            nxt = apply_move(instance, initial, move)
        except ValueError:
            continue
        if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
            continue
        h = hash_state(instance, nxt, config.enable_canonical_hash)
        candidates.append(StepCandidate(nxt, move, h, _score(instance, initial, nxt, move), "qubo_seed"))

    selected, qvars, qtime = _select_qubo_indices(candidates, config, min(seed_width, len(candidates)))
    beam = [Candidate(initial, 0.0, None, h0)]
    seen = {h0}
    for item in selected:
        if item.hash_value in seen:
            continue
        seen.add(item.hash_value)
        beam.append(Candidate(item.state, float(item.score[1]), item.move, item.hash_value))
    return beam, seen, 1 if qvars else 0, qvars, qtime


def run_qubo_seeded_rescue_beam(
    case: str,
    instance: Instance,
    config: SolverConfig,
    *,
    seed_width: int,
    core_width: int,
    supplement_width: int,
    qubo_min_depth: int,
    qubo_call_interval: int,
    qubo_plateau_only: bool,
    tail_repair_n: int,
    tail_repair_expanded: int,
    tail_repair_candidates: int,
    rescue_repair_n: int,
    rescue_repair_candidates: int,
    rescue_reservoir_size: int,
) -> tuple[TrajectoryResult, State]:
    start = time.perf_counter()
    initial = create_initial_state(instance)
    best = initial
    beam, seed_hashes, qcalls, qvars, qtime = _qubo_seeded_initial_beam(instance, config, seed_width=seed_width)
    visited = {hash_state(instance, initial, config.enable_canonical_hash): dominance_value(instance, initial)}
    for cand in beam:
        visited[cand.hash_value] = dominance_value(instance, cand.state)
        if _best_key(instance, cand.state) < _best_key(instance, best):
            best = cand.state
    expanded = 0
    generated = len(seed_hashes) - 1
    tail_expanded = tail_generated = 0
    tail_found = False
    plateau_layers = 0
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
        best_n_before = disorder_box_count(instance, best)
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
                step_candidates.append(StepCandidate(nxt, move, h, _score(instance, cand.state, nxt, move), "rescue"))
        generated += len(step_candidates)
        if not step_candidates:
            break

        layer_no = depth + 1
        use_qubo = (
            supplement_width > 0
            and layer_no >= qubo_min_depth
            and (layer_no - qubo_min_depth) % max(1, qubo_call_interval) == 0
            and (not qubo_plateau_only or plateau_layers > 0)
        )
        if use_qubo:
            selected, calls, vars_count, sample_time = _select_with_qubo_supplement(
                instance,
                step_candidates,
                config,
                core_width=core_width,
                supplement_width=supplement_width,
            )
        else:
            selected = _select_bucketed(instance, step_candidates, config.beam_width)
            calls = vars_count = 0
            sample_time = 0.0
        qcalls += calls
        qvars += vars_count
        qtime += sample_time
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
            "RESCUE_LAYER "
            f"depth={depth + 1} beam={len(beam)} bestN={disorder_box_count(instance, best)} "
            f"beamBestN={disorder_box_count(instance, beam[0].state)} elapsed={time.perf_counter() - start:.3f}s",
            flush=True,
        )
        if disorder_box_count(instance, best) == 0:
            break
        best_n_after = disorder_box_count(instance, best)
        plateau_layers = plateau_layers + 1 if best_n_after >= best_n_before else 0

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
        "; qubo_seeded_rescue_beam"
        f"; qubo_seed_width={seed_width}; tail_repair_n={tail_repair_n}; "
        f"core_width={core_width}; supplement_width={supplement_width}; "
        f"qubo_min_depth={qubo_min_depth}; qubo_call_interval={qubo_call_interval}; "
        f"qubo_plateau_only={qubo_plateau_only}; "
        f"rescue_repair_n={rescue_repair_n}; rescue_candidates={rescue_repair_candidates}; "
        f"tail_expanded={tail_expanded}; tail_generated={tail_generated}; tail_found={tail_found}"
    )
    row = _result_from_state(
        case,
        "qubo_seeded_rescue_beam",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=len(visited),
        qubo_calls=qcalls,
        qubo_total_vars=qvars,
        qubo_time_sec=qtime,
        status=status,
    )
    return row, best


def run_adaptive(case: str, instance: Instance, args: argparse.Namespace) -> AdaptiveRun:
    total_start = time.perf_counter()
    fast_config = _build_config(
        beam_width=args.fast_beam_width,
        time_limit=args.fast_time_limit,
        max_depth=args.max_depth,
        args=args,
    )
    fast_row, fast_state = run_fast_slack_qubo_beam(
        case,
        instance,
        fast_config,
        core_width=args.fast_core_width,
        supplement_width=args.fast_qubo_supplement_width,
        qubo_min_depth=args.fast_qubo_min_depth,
        qubo_call_interval=args.fast_qubo_call_interval,
        qubo_plateau_only=args.fast_qubo_plateau_only,
        tail_repair_n=args.fast_tail_repair_n,
        tail_repair_expanded=args.fast_tail_repair_expanded,
    )

    stage_rows = [fast_row]
    best = fast_state
    selected_stage = "fast_qubo"
    if disorder_box_count(instance, best) != 0:
        rescue_config = _build_config(
            beam_width=args.rescue_beam_width,
            time_limit=args.rescue_time_limit,
            max_depth=args.max_depth,
            args=args,
        )
        rescue_row, rescue_state = run_qubo_seeded_rescue_beam(
            case,
            instance,
            rescue_config,
            seed_width=args.rescue_qubo_seed_width,
            core_width=args.rescue_core_width,
            supplement_width=args.rescue_qubo_supplement_width,
            qubo_min_depth=args.rescue_qubo_min_depth,
            qubo_call_interval=args.rescue_qubo_call_interval,
            qubo_plateau_only=args.rescue_qubo_plateau_only,
            tail_repair_n=args.rescue_tail_repair_n,
            tail_repair_expanded=args.rescue_tail_repair_expanded,
            tail_repair_candidates=args.rescue_tail_repair_candidates,
            rescue_repair_n=args.rescue_repair_n,
            rescue_repair_candidates=args.rescue_repair_candidates,
            rescue_reservoir_size=args.rescue_reservoir_size,
        )
        stage_rows.append(rescue_row)
        best = rescue_state
        selected_stage = "qubo_seeded_rescue"

    runtime = time.perf_counter() - total_start
    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += (
        f"; {SOLVER_NAME}; fixed_adaptive_policy; selected_stage={selected_stage}; "
        + " -> ".join(
            f"{row.solver}(N={row.final_n},moves={row.moves},t={row.runtime_sec:.3f},q={row.qubo_calls})"
            for row in stage_rows
        )
    )
    row = _result_from_state(
        case,
        SOLVER_NAME,
        instance,
        best,
        total_start,
        expanded_states=sum(item.expanded_states for item in stage_rows),
        generated_states=sum(item.generated_states for item in stage_rows),
        visited_states=sum(item.visited_states for item in stage_rows),
        qubo_calls=sum(item.qubo_calls for item in stage_rows),
        qubo_total_vars=sum(item.qubo_total_vars for item in stage_rows),
        qubo_time_sec=sum(item.qubo_time_sec for item in stage_rows),
        status=status,
    )
    object.__setattr__(row, "runtime_sec", runtime)
    business_ok, goal_ok, validation_msg = _validate_final(instance, best, runtime)
    return AdaptiveRun(row, best, business_ok, goal_ok, validation_msg)


def _write_rows(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrajectoryResult.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_acceptance(runs: list[AdaptiveRun], path: Path, moves_dir: Path, exact_refs: dict[str, int]) -> None:
    rows = []
    moves_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        move_path = moves_dir / f"{run.row.case}_moves.csv"
        zone, bay = _parse_case(run.row.case)
        _write_explain_csv(_instance_from_single_bay(zone, bay), run.best_state.moves, move_path)
        rows.append(
            {
                "case": run.row.case,
                "selected_solver": run.row.solver,
                "final_n": run.row.final_n,
                "moves": run.row.moves,
                "runtime_sec": f"{run.row.runtime_sec:.6f}",
                "exact_moves": exact_refs.get(run.row.case, ""),
                "move_gap": _move_gap(run.row, exact_refs),
                "qubo_used": run.row.qubo_calls > 0,
                "selected_reason": run.row.status,
                "business_ok": run.business_ok,
                "goal_validation_ok": run.goal_validation_ok,
                "validation_msg": run.validation_msg,
                "moves_path": _rel(move_path),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ACCEPTANCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(runs: list[AdaptiveRun], output: Path, acceptance: Path, report: Path, exact_refs: dict[str, int]) -> None:
    solved = sum(1 for run in runs if run.row.final_n == 0)
    runtimes = [run.row.runtime_sec for run in runs]
    gaps = [int(_move_gap(run.row, exact_refs)) for run in runs if _move_gap(run.row, exact_refs) != ""]
    qubo_used = sum(1 for run in runs if run.row.qubo_calls > 0)
    lines = [
        "# Fast QUBO Adaptive Rescue V1",
        "",
        "Fixed adaptive algorithm: fast QUBO pass, then QUBO-seeded rescue only on failure.",
        "",
        f"Result CSV: `{_rel(output)}`",
        f"Acceptance CSV: `{_rel(acceptance)}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| cases | {len(runs)} |",
        f"| N=0 solved | {solved}/{len(runs)} |",
        f"| avg runtime sec | {sum(runtimes) / len(runtimes):.3f} |" if runtimes else "| avg runtime sec |  |",
        f"| max runtime sec | {max(runtimes):.3f} |" if runtimes else "| max runtime sec |  |",
        f"| avg move gap | {sum(gaps) / len(gaps):.3f} |" if gaps else "| avg move gap |  |",
        f"| QUBO executed | {qubo_used}/{len(runs)} |",
        f"| all selected routes valid | {all(run.business_ok and run.goal_validation_ok for run in runs)} |",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--acceptance-output", type=Path)
    parser.add_argument("--moves-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--fast-time-limit", type=float, default=8.0)
    parser.add_argument("--fast-beam-width", type=int, default=122)
    parser.add_argument("--fast-core-width", type=int, default=120)
    parser.add_argument("--fast-qubo-supplement-width", type=int, default=2)
    parser.add_argument("--fast-qubo-min-depth", type=int, default=10)
    parser.add_argument("--fast-qubo-call-interval", type=int, default=3)
    parser.add_argument("--fast-qubo-plateau-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-tail-repair-n", type=int, default=5)
    parser.add_argument("--fast-tail-repair-expanded", type=int, default=30000)
    parser.add_argument("--rescue-time-limit", type=float, default=16.0)
    parser.add_argument("--rescue-beam-width", type=int, default=240)
    parser.add_argument("--rescue-qubo-seed-width", type=int, default=0)
    parser.add_argument("--rescue-core-width", type=int, default=232)
    parser.add_argument("--rescue-qubo-supplement-width", type=int, default=8)
    parser.add_argument("--rescue-qubo-min-depth", type=int, default=6)
    parser.add_argument("--rescue-qubo-call-interval", type=int, default=4)
    parser.add_argument("--rescue-qubo-plateau-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rescue-tail-repair-n", type=int, default=3)
    parser.add_argument("--rescue-tail-repair-expanded", type=int, default=30000)
    parser.add_argument("--rescue-tail-repair-candidates", type=int, default=2)
    parser.add_argument("--rescue-repair-n", type=int, default=8)
    parser.add_argument("--rescue-repair-candidates", type=int, default=80)
    parser.add_argument("--rescue-reservoir-size", type=int, default=120)
    parser.add_argument("--max-qubo-vars", type=int, default=60)
    parser.add_argument("--qubo-time-limit", type=float, default=0.01)
    parser.add_argument("--qubo-reads", type=int, default=2)
    parser.add_argument("--qubo-iterations", type=int, default=80)
    parser.add_argument("--qubo-penalty-a", type=float, default=30.0)
    parser.add_argument("--diversity-weight", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "docs" / "experiments" / f"{SOLVER_NAME}_{stamp}.csv"
    acceptance = args.acceptance_output or output.with_name(output.stem + "_acceptance.csv")
    moves_dir = args.moves_dir or output.with_name(output.stem + "_moves")
    report = args.report or output.with_suffix(".md")
    if not output.is_absolute():
        output = ROOT / output
    if not acceptance.is_absolute():
        acceptance = ROOT / acceptance
    if not moves_dir.is_absolute():
        moves_dir = ROOT / moves_dir
    if not report.is_absolute():
        report = ROOT / report

    exact_refs = _exact_moves_by_case()
    runs: list[AdaptiveRun] = []
    for idx, case in enumerate(_case_filter(args), start=1):
        zone, bay = _parse_case(case)
        instance = _instance_from_single_bay(zone, bay)
        print(f"CASE_START {idx} {case} solver={SOLVER_NAME}", flush=True)
        run = run_adaptive(case, instance, args)
        runs.append(run)
        _write_rows([item.row for item in runs], output)
        _write_acceptance(runs, acceptance, moves_dir, exact_refs)
        _write_report(runs, output, acceptance, report, exact_refs)
        print(
            "CASE_DONE "
            f"{case} final_n={run.row.final_n} moves={run.row.moves} "
            f"runtime={run.row.runtime_sec:.3f}s qcalls={run.row.qubo_calls} "
            f"business_ok={run.business_ok} goal_ok={run.goal_validation_ok}",
            flush=True,
        )
    print(f"WROTE {_rel(output)}", flush=True)
    print(f"WROTE {_rel(acceptance)}", flush=True)
    print(f"WROTE {_rel(report)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
