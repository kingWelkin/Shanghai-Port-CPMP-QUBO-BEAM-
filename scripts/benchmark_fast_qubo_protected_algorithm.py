"""Honest fixed-flow fast + protected-QUBO business algorithm benchmark.

This is a real online algorithm rather than an oracle best-of sweep.  Each case
uses the same staged policy, each stage may update the incumbent, and total
runtime is the sum of every executed stage.
"""

from __future__ import annotations

import argparse
import csv
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
from benchmark_fast_slack_beam import run_fast_slack_beam  # noqa: E402
from benchmark_fast_slack_qubo_beam import run_fast_slack_qubo_beam  # noqa: E402
from benchmark_full_real_business_algorithms import TrajectoryResult  # noqa: E402
from container_reshuffle.core import (  # noqa: E402
    Instance,
    Move,
    Solution,
    State,
    apply_move,
    create_initial_state,
    validate_solution,
)
from container_reshuffle.features import disorder_box_count, entropy_limit, entropy_value  # noqa: E402
from container_reshuffle.scoring import cpmp_business_feasible  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


EXACT_REFERENCE = ROOT / "docs" / "experiments" / "fast_slack_vs_exact_all_single_bays_8s_w120.csv"
RESULT_FIELDS = list(TrajectoryResult.__annotations__.keys())
ACCEPTANCE_FIELDS = [
    "case",
    "selected_solver",
    "final_n",
    "moves",
    "runtime_sec",
    "exact_moves",
    "move_gap",
    "qubo_used",
    "selected_reason",
    "business_ok",
    "goal_validation_ok",
    "validation_msg",
    "moves_path",
]


@dataclass(frozen=True)
class StageSpec:
    """One predetermined stage in the online algorithm."""

    name: str
    family: str
    supplement_width: int = 0
    qubo_min_depth: int = 0
    qubo_plateau_only: bool = False


@dataclass
class StageCandidate:
    """A stage result and its terminal state."""

    spec: StageSpec
    result: TrajectoryResult
    state: State


@dataclass
class ProtectedRun:
    """Final aggregate for one case."""

    row: TrajectoryResult
    best_state: State
    selected_stage: str
    stage_trace: str
    business_ok: bool
    goal_validation_ok: bool
    validation_msg: str


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _exact_moves_by_case() -> dict[str, int]:
    refs: dict[str, int] = {}
    for row in _read_rows(EXACT_REFERENCE):
        value = row.get("exact_moves_to_n0") or row.get("exact_moves")
        if value not in (None, ""):
            refs[row["case"]] = int(float(value))
    return refs


def _parse_case(case: str) -> tuple[str, str]:
    parts = case.split("_")
    if len(parts) < 2:
        raise ValueError(f"cannot parse case name: {case}")
    return parts[0], parts[1]


def _case_filter(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return [case if case.endswith("_single") else f"{case}_single" for case in args.cases]
    return [f"{zone}_{bay}_single" for zone, bay in discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))]


def _candidate_key(candidate: StageCandidate) -> tuple[int, int, int]:
    final_n = candidate.result.final_n if candidate.result.final_n is not None else 10**9
    moves = candidate.result.moves if candidate.result.moves is not None else 10**9
    return final_n, moves, 0 if candidate.spec.family == "qubo" else 1


def _state_business_ok(instance: Instance, moves: list[Move]) -> tuple[bool, str, State]:
    state = create_initial_state(instance)
    cap = entropy_limit(instance, 1.3)
    try:
        for move in moves:
            state = apply_move(instance, state, move)
            if state.entropy_max > cap:
                return False, f"G_LIMIT_EXCEEDED max_g={state.entropy_max} cap={cap}", state
            if not cpmp_business_feasible(instance, state, 1.3):
                return False, "BUSINESS_INFEASIBLE", state
        return True, "ok", state
    except Exception as exc:  # noqa: BLE001 - keep benchmark diagnostics in-row.
        return False, str(exc), state


def _validate_final(instance: Instance, state: State, solver_name: str, runtime_sec: float) -> tuple[bool, bool, str]:
    business_ok, business_msg, simulated = _state_business_ok(instance, state.moves)
    if disorder_box_count(instance, simulated) != disorder_box_count(instance, state):
        return False, False, (
            f"{business_msg}; FINAL_N_MISMATCH simulated={disorder_box_count(instance, simulated)} "
            f"state={disorder_box_count(instance, state)}"
        )
    if disorder_box_count(instance, state) != 0:
        return business_ok, False, business_msg
    solution = Solution(
        solver_name=solver_name,
        feasible=True,
        optimal=False,
        objective=float(len(state.moves)),
        runtime_sec=runtime_sec,
        moves=state.moves,
        status="N_ZERO",
    )
    goal_ok, goal_msg, _ = validate_solution(instance, solution)
    return business_ok, goal_ok, goal_msg


def _fast_config(instance: Instance, *, beam_width: int, time_limit: float) -> SolverConfig:
    return SolverConfig(
        max_depth=80,
        beam_width=beam_width,
        time_limit_sec=time_limit,
        entropy_alpha=1.3,
        random_seed=37,
    )


def _qubo_config(instance: Instance, *, time_limit: float) -> SolverConfig:
    return SolverConfig(
        max_depth=80,
        beam_width=120,
        time_limit_sec=time_limit,
        max_qubo_vars=60,
        qubo_penalty_a=30.0,
        diversity_weight=0.08,
        qubo_num_reads=4,
        qubo_time_limit_sec=0.01,
        qubo_sa_iterations=80,
        entropy_alpha=1.3,
        random_seed=37,
    )


def _run_fast_stage(case: str, instance: Instance, spec: StageSpec, *, time_limit: float, rescue: bool) -> StageCandidate:
    config = _fast_config(instance, beam_width=120 if not rescue else 240, time_limit=time_limit)
    result, state = run_fast_slack_beam(
        case,
        instance,
        config,
        tail_repair_n=5,
        tail_repair_expanded=30_000 if rescue else 10_000,
        tail_repair_candidates=2,
        lookahead_pool=0,
        lookahead_keep=0,
        rescue_repair_n=8 if rescue else 0,
        rescue_repair_candidates=80 if rescue else 0,
        rescue_reservoir_size=240 if rescue else 0,
    )
    result.solver = spec.name
    return StageCandidate(spec, result, state)


def _run_qubo_stage(case: str, instance: Instance, spec: StageSpec, *, time_limit: float) -> StageCandidate:
    config = _qubo_config(instance, time_limit=time_limit)
    result, state = run_fast_slack_qubo_beam(
        case,
        instance,
        config,
        core_width=120 - spec.supplement_width,
        supplement_width=spec.supplement_width,
        qubo_min_depth=spec.qubo_min_depth,
        qubo_call_interval=3,
        qubo_plateau_only=spec.qubo_plateau_only,
        tail_repair_n=5,
        tail_repair_expanded=10_000,
    )
    result.solver = spec.name
    return StageCandidate(spec, result, state)


def _n0(instance: Instance) -> int:
    return disorder_box_count(instance, create_initial_state(instance))


def _needs_low_move_probe(instance: Instance, best: StageCandidate) -> bool:
    n0 = disorder_box_count(instance, create_initial_state(instance))
    final_n = best.result.final_n if best.result.final_n is not None else 10**9
    moves = best.result.moves if best.result.moves is not None else 10**9
    return final_n > 0 or (moves >= n0 + 5 and moves < 28)


def _needs_high_move_probe(instance: Instance, best: StageCandidate) -> bool:
    final_n = best.result.final_n if best.result.final_n is not None else 10**9
    moves = best.result.moves if best.result.moves is not None else 10**9
    return final_n == 0 and moves > _n0(instance) + 5 and moves >= 30


def _good_enough_without_qubo(instance: Instance, best: StageCandidate) -> bool:
    final_n = best.result.final_n if best.result.final_n is not None else 10**9
    moves = best.result.moves if best.result.moves is not None else 10**9
    n0 = _n0(instance)
    if final_n != 0:
        return False
    if moves >= 29 and n0 >= 25:
        return False
    return moves <= n0 + 5


def _needs_rescue(instance: Instance, best: StageCandidate, profile: str) -> bool:
    final_n = best.result.final_n if best.result.final_n is not None else 10**9
    if final_n > 0:
        return True
    if profile != "quality":
        return False
    n0 = disorder_box_count(instance, create_initial_state(instance))
    moves = best.result.moves if best.result.moves is not None else 10**9
    return moves >= n0 + 7 and moves >= 30


def _maybe_update(best: StageCandidate, candidate: StageCandidate) -> StageCandidate:
    return candidate if _candidate_key(candidate) < _candidate_key(best) else best


def run_fast_qubo_protected(case: str, instance: Instance, *, profile: str) -> ProtectedRun:
    """Run the fixed staged algorithm and return a fair aggregate row."""

    stages: list[StageCandidate] = []
    core_spec = StageSpec("fast_qubo_protected_core_w120_t8", "fast")
    best = _run_fast_stage(case, instance, core_spec, time_limit=8.0, rescue=False)
    stages.append(best)

    q_specs = [
        StageSpec("fast_qubo_probe_s36_d6_any", "qubo", supplement_width=36, qubo_min_depth=6, qubo_plateau_only=False),
        StageSpec("fast_qubo_probe_s36_d8_any", "qubo", supplement_width=36, qubo_min_depth=8, qubo_plateau_only=False),
        StageSpec("fast_qubo_probe_s24_d8_plateau", "qubo", supplement_width=24, qubo_min_depth=8, qubo_plateau_only=True),
        StageSpec("fast_qubo_probe_s36_d10_any", "qubo", supplement_width=36, qubo_min_depth=10, qubo_plateau_only=False),
    ]
    if not _good_enough_without_qubo(instance, best):
        candidate = _run_qubo_stage(case, instance, q_specs[0], time_limit=8.0)
        stages.append(candidate)
        best = _maybe_update(best, candidate)

        if best.result.final_n != 0:
            candidate = _run_qubo_stage(case, instance, q_specs[1], time_limit=8.0)
            stages.append(candidate)
            best = _maybe_update(best, candidate)

        if best.result.final_n == 0 and _needs_low_move_probe(instance, best):
            candidate = _run_qubo_stage(case, instance, q_specs[2], time_limit=8.0)
            stages.append(candidate)
            best = _maybe_update(best, candidate)

        if _needs_high_move_probe(instance, best):
            candidate = _run_qubo_stage(case, instance, q_specs[3], time_limit=8.0)
            stages.append(candidate)
            best = _maybe_update(best, candidate)

    if _needs_rescue(instance, best, profile):
        rescue_spec = StageSpec("fast_qubo_rescue_w240_t18_tail30000", "rescue")
        candidate = _run_fast_stage(case, instance, rescue_spec, time_limit=18.0, rescue=True)
        stages.append(candidate)
        best = _maybe_update(best, candidate)

    total_runtime = sum(stage.result.runtime_sec for stage in stages)
    expanded = sum(stage.result.expanded_states for stage in stages)
    generated = sum(stage.result.generated_states for stage in stages)
    visited = sum(stage.result.visited_states for stage in stages)
    qcalls = sum(stage.result.qubo_calls for stage in stages)
    qvars = sum(stage.result.qubo_total_vars for stage in stages)
    qtime = sum(stage.result.qubo_time_sec for stage in stages)
    initial = create_initial_state(instance)
    n0 = disorder_box_count(instance, initial)
    final_n = disorder_box_count(instance, best.state)
    max_g = best.state.entropy_max if best.state.moves else entropy_value(instance, best.state)
    cap13 = entropy_limit(instance, 1.3)
    cap15 = entropy_limit(instance, 1.5)
    business_ok, goal_ok, validation_msg = _validate_final(
        instance,
        best.state,
        "fast_qubo_protected",
        total_runtime,
    )
    stage_trace = " -> ".join(
        f"{stage.spec.name}(N={stage.result.final_n},moves={stage.result.moves},t={stage.result.runtime_sec:.3f})"
        for stage in stages
    )
    row = TrajectoryResult(
        case=case,
        solver=f"fast_qubo_protected_{profile}",
        n0=n0,
        cap_13=cap13,
        cap_15=cap15,
        moves=best.state.move_count,
        final_n=final_n,
        reduction=n0 - final_n,
        g_end=best.state.move_count + final_n,
        max_g=max_g,
        g_ok_13=max_g <= cap13,
        g_ok_15=max_g <= cap15,
        runtime_sec=total_runtime,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=visited,
        qubo_calls=qcalls,
        qubo_total_vars=qvars,
        qubo_time_sec=qtime,
        status=(
            f"selected_stage={best.spec.name}; stages={len(stages)}; "
            f"profile={profile}; validation={validation_msg}; trace={stage_trace}"
        ),
    )
    return ProtectedRun(
        row=row,
        best_state=best.state,
        selected_stage=best.spec.name,
        stage_trace=stage_trace,
        business_ok=business_ok,
        goal_validation_ok=goal_ok,
        validation_msg=validation_msg,
    )


def _write_rows(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _move_gap(row: TrajectoryResult, exact_refs: dict[str, int]) -> str:
    exact = exact_refs.get(row.case)
    if exact is None or row.final_n != 0 or row.moves is None:
        return ""
    return str(row.moves - exact)


def _write_acceptance(runs: list[ProtectedRun], path: Path, moves_dir: Path, exact_refs: dict[str, int]) -> None:
    rows = []
    moves_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        move_path = moves_dir / f"{run.row.case}_moves.csv"
        zone, bay = _parse_case(run.row.case)
        _write_explain_csv(_instance_from_single_bay(zone, bay), run.best_state.moves, move_path)
        rows.append(
            {
                "case": run.row.case,
                "selected_solver": run.selected_stage,
                "final_n": run.row.final_n,
                "moves": run.row.moves,
                "runtime_sec": f"{run.row.runtime_sec:.6f}",
                "exact_moves": exact_refs.get(run.row.case, ""),
                "move_gap": _move_gap(run.row, exact_refs),
                "qubo_used": str(run.row.qubo_calls > 0),
                "selected_reason": f"fixed staged policy; {run.stage_trace}",
                "business_ok": str(run.business_ok),
                "goal_validation_ok": str(run.goal_validation_ok),
                "validation_msg": run.validation_msg,
                "moves_path": _rel(move_path),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ACCEPTANCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(runs: list[ProtectedRun], output: Path, acceptance: Path, report: Path, exact_refs: dict[str, int]) -> None:
    solved = sum(1 for run in runs if run.row.final_n == 0)
    avg_runtime = sum(run.row.runtime_sec for run in runs) / len(runs) if runs else 0.0
    gaps = [
        run.row.moves - exact_refs[run.row.case]
        for run in runs
        if run.row.final_n == 0 and run.row.moves is not None and run.row.case in exact_refs
    ]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
    qubo_used = sum(1 for run in runs if run.row.qubo_calls > 0)
    lines = [
        "# Fast QUBO Protected Algorithm",
        "",
        "This is a fixed staged algorithm. Runtime is the sum of all executed stages, not the selected stage alone.",
        "",
        f"Result CSV: `{_rel(output)}`",
        f"Acceptance CSV: `{_rel(acceptance)}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| cases | {len(runs)} |",
        f"| N=0 solved | {solved}/{len(runs)} |",
        f"| avg runtime sec | {avg_runtime:.3f} |",
        f"| avg move gap | {avg_gap:.3f} |",
        f"| QUBO executed | {qubo_used}/{len(runs)} |",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--profile", choices=["balanced", "quality"], default="balanced")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--acceptance-output", type=Path)
    parser.add_argument("--moves-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "docs" / "experiments" / f"fast_qubo_protected_{args.profile}_{stamp}.csv"
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
    runs: list[ProtectedRun] = []
    for idx, case in enumerate(_case_filter(args), start=1):
        zone, bay = _parse_case(case)
        instance = _instance_from_single_bay(zone, bay)
        print(f"CASE_START {idx} {case} profile={args.profile}", flush=True)
        run = run_fast_qubo_protected(case, instance, profile=args.profile)
        runs.append(run)
        _write_rows([item.row for item in runs], output)
        _write_acceptance(runs, acceptance, moves_dir, exact_refs)
        _write_report(runs, output, acceptance, report, exact_refs)
        print(
            "CASE_DONE "
            f"{case} final_n={run.row.final_n} moves={run.row.moves} "
            f"runtime={run.row.runtime_sec:.3f}s qcalls={run.row.qubo_calls} selected={run.selected_stage}",
            flush=True,
        )
    print(f"WROTE {_rel(output)}", flush=True)
    print(f"WROTE {_rel(acceptance)}", flush=True)
    print(f"WROTE {_rel(report)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
