"""Target-reaching A* comparison for real single-bay N=0 cases.

This is a satisficing A* benchmark, not an optimality proof.  It keeps the
single open-list / best-g A* search shape, uses a non-admissible quality
heuristic to prioritize N=0 reachability, and stops as soon as a valid N=0
route is found.  A low-N deterministic tail repair is used as a completion
step from states popped by the A* queue.
"""

from __future__ import annotations

import argparse
import csv
import heapq
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
from benchmark_fast_qubo_integrated_v1 import _exact_moves_by_case, _move_gap, _parse_case, _rel  # noqa: E402
from benchmark_fast_slack_beam import StepCandidate, _score, _select_bucketed  # noqa: E402
from benchmark_full_real_business_algorithms import TrajectoryResult, _result_from_state  # noqa: E402
from benchmark_macro_reservoir_qubo_beam import tail_repair_to_n_zero  # noqa: E402
from container_reshuffle.core import (  # noqa: E402
    Instance,
    Move,
    Solution,
    State,
    apply_move,
    create_initial_state,
    generate_legal_moves,
    hash_state,
    validate_solution,
)
from container_reshuffle.features import (  # noqa: E402
    bad_count,
    dirty_stack_count,
    disorder_box_count,
    entropy_limit,
    entropy_value,
    inversion_count,
    settled_box_count,
    well_prefix_len,
)
from container_reshuffle.scoring import cpmp_business_feasible  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


SOLVER_NAME = "target_astar_n0_v1"
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


@dataclass
class TargetAStarRun:
    row: TrajectoryResult
    best_state: State
    business_ok: bool
    goal_validation_ok: bool
    validation_msg: str


def _case_filter(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return [case if case.endswith("_single") else f"{case}_single" for case in args.cases]
    return [f"{zone}_{bay}_single" for zone, bay in discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))]


def _is_clean_stack(stack: list[int]) -> bool:
    return well_prefix_len(stack) == len(stack)


def _clean_open_slots(instance: Instance, state: State) -> int:
    return sum(1 for stack in state.stacks if len(stack) < instance.max_height and _is_clean_stack(stack))


def _height_pressure(instance: Instance, state: State) -> int:
    return sum(1 for stack in state.stacks if len(stack) >= instance.max_height)


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


def _repair_seed_key(instance: Instance, state: State, alpha: float) -> tuple:
    n_value = disorder_box_count(instance, state)
    cap = entropy_limit(instance, alpha)
    remaining_moves = cap - state.move_count
    clearance_risk = max(0, n_value - remaining_moves)
    return (
        clearance_risk,
        n_value,
        entropy_value(instance, state),
        state.move_count,
        state.entropy_max,
        inversion_count(instance, state),
        bad_count(instance, state),
        dirty_stack_count(instance, state),
        -settled_box_count(instance, state),
    )


def _heuristic_quality(instance: Instance, state: State, args: argparse.Namespace) -> float:
    n_value = disorder_box_count(instance, state)
    cap = entropy_limit(instance, args.alpha)
    slack = cap - entropy_value(instance, state)
    remaining_moves = cap - state.move_count
    clearance_risk = max(0, n_value - remaining_moves)
    inv = inversion_count(instance, state)
    bad = bad_count(instance, state)
    dirty = dirty_stack_count(instance, state)
    settled = settled_box_count(instance, state)
    clean_slots = _clean_open_slots(instance, state)
    pressure = _height_pressure(instance, state)

    low_n_boost = 1.0 if n_value <= args.tail_repair_n else 0.0
    return (
        args.n_weight * n_value
        + args.risk_weight * clearance_risk
        + args.bad_weight * bad
        + args.inv_weight * inv
        + args.dirty_weight * dirty
        + args.pressure_weight * pressure
        + args.slack_penalty * max(0, args.min_slack - slack)
        - args.settled_weight * settled
        - args.clean_slot_weight * clean_slots
        - args.low_n_bonus * low_n_boost
    )


def _priority(instance: Instance, state: State, args: argparse.Namespace) -> tuple:
    h_quality = _heuristic_quality(instance, state, args)
    n_value = disorder_box_count(instance, state)
    return (
        args.g_weight * state.move_count + h_quality,
        h_quality,
        n_value,
        entropy_value(instance, state),
        state.entropy_max,
        state.move_count,
        inversion_count(instance, state),
        bad_count(instance, state),
        dirty_stack_count(instance, state),
        -settled_box_count(instance, state),
    )


def _state_business_ok(instance: Instance, moves: list[Move], alpha: float) -> tuple[bool, str, State]:
    state = create_initial_state(instance)
    cap = entropy_limit(instance, alpha)
    try:
        for move in moves:
            state = apply_move(instance, state, move)
            if state.entropy_max > cap:
                return False, f"G_LIMIT_EXCEEDED max_g={state.entropy_max} cap={cap}", state
            if not cpmp_business_feasible(instance, state, alpha):
                return False, "BUSINESS_INFEASIBLE", state
        return True, "ok", state
    except Exception as exc:  # noqa: BLE001 - benchmark diagnostics should stay in-row.
        return False, str(exc), state


def _validate_final(instance: Instance, state: State, runtime_sec: float, alpha: float) -> tuple[bool, bool, str]:
    business_ok, business_msg, simulated = _state_business_ok(instance, state.moves, alpha)
    if disorder_box_count(instance, simulated) != disorder_box_count(instance, state):
        return False, False, (
            f"{business_msg}; FINAL_N_MISMATCH simulated={disorder_box_count(instance, simulated)} "
            f"state={disorder_box_count(instance, state)}"
        )
    if disorder_box_count(instance, state) != 0:
        return business_ok, False, business_msg
    solution = Solution(
        solver_name=SOLVER_NAME,
        feasible=True,
        optimal=False,
        objective=float(len(state.moves)),
        runtime_sec=runtime_sec,
        moves=state.moves,
        status="TARGET_REACHABLE",
    )
    goal_ok, goal_msg, _ = validate_solution(instance, solution)
    return business_ok, goal_ok, goal_msg


def _build_config(args: argparse.Namespace) -> SolverConfig:
    return SolverConfig(
        max_depth=args.max_depth,
        time_limit_sec=args.time_limit,
        entropy_alpha=args.alpha,
        enable_canonical_hash=args.canonical,
        enable_visited_pruning=True,
        random_seed=args.seed,
    )


def run_target_astar_n0(case: str, instance: Instance, args: argparse.Namespace) -> TargetAStarRun:
    start = time.perf_counter()
    deadline = start + args.time_limit
    search_deadline = max(start, deadline - max(0.0, args.repair_reserve_sec))
    config = _build_config(args)
    initial = create_initial_state(instance)
    best = initial
    cap = entropy_limit(instance, args.alpha)

    counter = 0
    initial_hash = hash_state(instance, initial, args.canonical)
    best_g: dict[tuple, int] = {initial_hash: 0}
    heap: list[tuple[tuple, int, State]] = []
    heapq.heappush(heap, (_priority(instance, initial, args), counter, initial))

    expanded = generated = dominated = depth_pruned = business_pruned = successor_pruned = 0
    tail_expanded = tail_generated = tail_attempts = 0
    tail_found = False
    goal_source = "none"
    tail_probed: set[tuple] = set()
    tail_reservoir: dict[tuple, State] = {}

    def remember_tail_seed(state: State, state_hash: tuple) -> None:
        if args.tail_reservoir_size <= 0:
            return
        if disorder_box_count(instance, state) > args.tail_reservoir_n:
            return
        old = tail_reservoir.get(state_hash)
        if old is None or _repair_seed_key(instance, state, args.alpha) < _repair_seed_key(instance, old, args.alpha):
            tail_reservoir[state_hash] = state
        if len(tail_reservoir) > args.tail_reservoir_size:
            keep = sorted(tail_reservoir.values(), key=lambda item: _repair_seed_key(instance, item, args.alpha))[
                : args.tail_reservoir_size
            ]
            tail_reservoir.clear()
            for item in keep:
                tail_reservoir[hash_state(instance, item, args.canonical)] = item

    while heap and time.perf_counter() < search_deadline:
        _rank, _idx, state = heapq.heappop(heap)
        state_hash = hash_state(instance, state, args.canonical)
        if state.move_count != best_g.get(state_hash, math.inf):
            dominated += 1
            continue

        if _best_key(instance, state) < _best_key(instance, best):
            best = state
        remember_tail_seed(state, state_hash)

        state_n = disorder_box_count(instance, state)
        if state_n == 0:
            best = state
            goal_source = "astar_pop"
            break

        if (
            args.tail_repair_n > 0
            and state_n <= args.tail_repair_n
            and tail_attempts < args.max_tail_attempts
            and state_hash not in tail_probed
        ):
            tail_probed.add(state_hash)
            tail_attempts += 1
            repaired, exp_count, gen_count = tail_repair_to_n_zero(
                instance,
                state,
                config,
                deadline=deadline,
                max_expanded=args.tail_repair_expanded,
            )
            tail_expanded += exp_count
            tail_generated += gen_count
            if repaired is not None:
                best = repaired
                tail_found = True
                goal_source = "tail_repair"
                break

        if state.move_count >= args.max_depth or state.move_count >= cap:
            depth_pruned += 1
            continue

        expanded += 1
        moves = generate_legal_moves(instance, state)
        successors: list[StepCandidate] = []
        for move in moves:
            try:
                nxt = apply_move(instance, state, move)
            except ValueError:
                continue
            generated += 1
            if not cpmp_business_feasible(instance, nxt, args.alpha):
                business_pruned += 1
                continue
            if nxt.move_count > args.max_depth:
                depth_pruned += 1
                continue
            nxt_hash = hash_state(instance, nxt, args.canonical)
            if nxt.move_count >= best_g.get(nxt_hash, math.inf):
                dominated += 1
                continue
            if disorder_box_count(instance, nxt) == 0:
                best = nxt
                goal_source = "astar_generate"
                heap.clear()
                successors.clear()
                break
            successors.append(StepCandidate(nxt, move, nxt_hash, _score(instance, state, nxt, move), "astar_successor"))

        if goal_source == "astar_generate":
            break

        if args.candidate_limit_per_state > 0:
            selected = _select_bucketed(
                instance,
                successors,
                args.candidate_limit_per_state,
                lookahead_pool=args.lookahead_pool,
                lookahead_keep=args.lookahead_keep,
                entropy_alpha=args.alpha,
            )
            successor_pruned += max(0, len(successors) - len(selected))
        else:
            selected = sorted(successors, key=lambda cand: (_priority(instance, cand.state, args), cand.score))

        selected.sort(key=lambda cand: (_priority(instance, cand.state, args), cand.score))
        for cand in selected:
            nxt = cand.state
            best_g[cand.hash_value] = nxt.move_count
            remember_tail_seed(nxt, cand.hash_value)
            if (
                args.tail_probe_generated
                and args.tail_repair_n > 0
                and disorder_box_count(instance, nxt) <= args.tail_repair_n
                and tail_attempts < args.max_tail_attempts
                and cand.hash_value not in tail_probed
                and time.perf_counter() < deadline
            ):
                tail_probed.add(cand.hash_value)
                tail_attempts += 1
                repaired, exp_count, gen_count = tail_repair_to_n_zero(
                    instance,
                    nxt,
                    config,
                    deadline=deadline,
                    max_expanded=args.tail_repair_expanded,
                )
                tail_expanded += exp_count
                tail_generated += gen_count
                if repaired is not None:
                    best = repaired
                    tail_found = True
                    goal_source = "tail_repair_generated"
                    heap.clear()
                    break
            counter += 1
            heapq.heappush(heap, (_priority(instance, nxt, args), counter, nxt))
        if goal_source == "tail_repair_generated":
            break

    if disorder_box_count(instance, best) != 0 and time.perf_counter() < deadline:
        for state in sorted(tail_reservoir.values(), key=lambda item: _repair_seed_key(instance, item, args.alpha)):
            if time.perf_counter() >= deadline or tail_attempts >= args.max_tail_attempts:
                break
            state_hash = hash_state(instance, state, args.canonical)
            if state_hash in tail_probed:
                continue
            tail_probed.add(state_hash)
            tail_attempts += 1
            repaired, exp_count, gen_count = tail_repair_to_n_zero(
                instance,
                state,
                config,
                deadline=deadline,
                max_expanded=args.tail_repair_expanded,
            )
            tail_expanded += exp_count
            tail_generated += gen_count
            if repaired is not None:
                best = repaired
                tail_found = True
                goal_source = "tail_repair_reservoir"
                break

    final_n = disorder_box_count(instance, best)
    status = "TARGET_REACHABLE" if final_n == 0 else "BEST_EFFORT"
    status += (
        f"; solver={SOLVER_NAME}; goal_source={goal_source}; "
        f"heuristic=weighted_quality_astar; g_weight={args.g_weight}; "
        f"candidate_limit_per_state={args.candidate_limit_per_state}; "
        f"repair_reserve_sec={args.repair_reserve_sec}; tail_reservoir={len(tail_reservoir)}; "
        f"tail_repair_n={args.tail_repair_n}; tail_attempts={tail_attempts}; "
        f"tail_found={tail_found}; tail_expanded={tail_expanded}; tail_generated={tail_generated}; "
        f"expanded={expanded}; generated={generated}; visited={len(best_g)}; open={len(heap)}; "
        f"dominated={dominated}; business_pruned={business_pruned}; depth_pruned={depth_pruned}; "
        f"successor_pruned={successor_pruned}"
    )
    row = _result_from_state(
        case,
        SOLVER_NAME,
        instance,
        best,
        start,
        expanded_states=expanded + tail_expanded,
        generated_states=generated + tail_generated,
        visited_states=len(best_g),
        status=status,
    )
    business_ok, goal_ok, msg = _validate_final(instance, best, row.runtime_sec, args.alpha)
    return TargetAStarRun(row=row, best_state=best, business_ok=business_ok, goal_validation_ok=goal_ok, validation_msg=msg)


def _write_rows(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_acceptance(runs: list[TargetAStarRun], path: Path, moves_dir: Path, exact_refs: dict[str, int]) -> None:
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
                "qubo_used": "False",
                "selected_reason": run.row.status,
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


def _write_report(runs: list[TargetAStarRun], output: Path, acceptance: Path, report: Path, exact_refs: dict[str, int]) -> None:
    solved = sum(1 for run in runs if run.row.final_n == 0)
    avg_runtime = sum(run.row.runtime_sec for run in runs) / len(runs) if runs else 0.0
    gaps = [
        run.row.moves - exact_refs[run.row.case]
        for run in runs
        if run.row.final_n == 0 and run.row.moves is not None and run.row.case in exact_refs
    ]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
    all_valid = all(run.business_ok and run.goal_validation_ok for run in runs if run.row.final_n == 0)
    lines = [
        f"# {SOLVER_NAME}",
        "",
        "Satisficing target A* comparison. This is not an optimality proof.",
        "",
        f"Result CSV: `{_rel(output)}`",
        f"Acceptance CSV: `{_rel(acceptance)}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| cases | {len(runs)} |",
        f"| N=0 solved | {solved}/{len(runs)} |",
        f"| avg runtime sec | {avg_runtime:.3f} |",
        f"| max runtime sec | {max((run.row.runtime_sec for run in runs), default=0.0):.3f} |",
        f"| avg move gap | {avg_gap:.3f} |",
        f"| all solved routes valid | {all_valid} |",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def _output_paths(args: argparse.Namespace, stamp: str) -> tuple[Path, Path, Path, Path]:
    output = args.output or args.output_dir / f"target_astar_n0_{stamp}.csv"
    if not output.is_absolute():
        output = ROOT / output
    acceptance = output.with_name(output.stem + "_acceptance.csv")
    moves_dir = output.with_name(output.stem + "_moves")
    report = output.with_suffix(".md")
    return output, acceptance, moves_dir, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "experiments")
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--g-weight", type=float, default=0.20)
    parser.add_argument("--n-weight", type=float, default=1000.0)
    parser.add_argument("--risk-weight", type=float, default=250.0)
    parser.add_argument("--bad-weight", type=float, default=18.0)
    parser.add_argument("--inv-weight", type=float, default=2.0)
    parser.add_argument("--dirty-weight", type=float, default=14.0)
    parser.add_argument("--pressure-weight", type=float, default=5.0)
    parser.add_argument("--settled-weight", type=float, default=1.2)
    parser.add_argument("--clean-slot-weight", type=float, default=4.0)
    parser.add_argument("--slack-penalty", type=float, default=35.0)
    parser.add_argument("--min-slack", type=int, default=2)
    parser.add_argument("--low-n-bonus", type=float, default=80.0)
    parser.add_argument("--tail-repair-n", type=int, default=10)
    parser.add_argument("--tail-repair-expanded", type=int, default=60000)
    parser.add_argument("--max-tail-attempts", type=int, default=80)
    parser.add_argument("--repair-reserve-sec", type=float, default=8.0)
    parser.add_argument("--tail-reservoir-n", type=int, default=12)
    parser.add_argument("--tail-reservoir-size", type=int, default=400)
    parser.add_argument("--candidate-limit-per-state", type=int, default=36)
    parser.add_argument("--lookahead-pool", type=int, default=0)
    parser.add_argument("--lookahead-keep", type=int, default=0)
    parser.add_argument("--tail-probe-generated", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir

    exact_refs = _exact_moves_by_case()
    cases = _case_filter(args)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output, acceptance, moves_dir, report = _output_paths(args, stamp)

    runs: list[TargetAStarRun] = []
    for idx, case in enumerate(cases, start=1):
        zone, bay = _parse_case(case)
        instance = _instance_from_single_bay(zone, bay)
        print(f"CASE_START {idx}/{len(cases)} {case} solver={SOLVER_NAME}", flush=True)
        run = run_target_astar_n0(case, instance, args)
        runs.append(run)
        print(
            "CASE_DONE "
            f"{case} final_n={run.row.final_n} moves={run.row.moves} runtime={run.row.runtime_sec:.3f}s "
            f"valid={run.business_ok and run.goal_validation_ok} status={run.row.status}",
            flush=True,
        )
        _write_rows([item.row for item in runs], output)
        _write_acceptance(runs, acceptance, moves_dir, exact_refs)
        _write_report(runs, output, acceptance, report, exact_refs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
