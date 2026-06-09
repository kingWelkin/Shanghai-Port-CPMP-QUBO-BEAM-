"""Target-layout QUBO upper-bound experiment for single-bay CPMP cases.

The QUBO is the primary planning model: it assigns every box to a final target
stack while enforcing one-stack-per-box and stack capacity through quadratic
penalties with slack variables. A deterministic target-guided beam then tries
to realize the sampled target layout and writes validated move routes.
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
from benchmark_fast_qubo_integrated_v1 import _exact_moves_by_case, _move_gap, _parse_case, _rel  # noqa: E402
from benchmark_full_real_business_algorithms import TrajectoryResult  # noqa: E402
from container_reshuffle.core import (  # noqa: E402
    Instance,
    Move,
    Solution,
    State,
    apply_move,
    create_initial_state,
    generate_legal_moves,
    hash_state,
    is_goal,
    validate_solution,
)
from container_reshuffle.features import disorder_box_count, entropy_limit, max_entropy_along_moves  # noqa: E402
from container_reshuffle.qubo.model import QuboModel  # noqa: E402
from container_reshuffle.qubo.sampler import QuboSample, QuboSamplerConfig, SimulatedAnnealingQuboSampler  # noqa: E402
from container_reshuffle.scoring import cpmp_business_feasible, dominance_value, score_state  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


SOLVER_NAME = "target_layout_qubo_upper_bound_v1"
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
class DecodedTarget:
    assignment: dict[int, int]
    repairs: int
    energy: float
    sample_index: int
    target_cost: float


@dataclass
class LayoutRun:
    case: str
    state: State | None
    runtime_sec: float
    expanded: int
    generated: int
    visited: int
    qvars: int
    qtime: float
    qrepairs: int
    qsample_index: int
    target_exact: bool
    status: str


def _add_square_penalty(model: QuboModel, terms: list[tuple[int, float]], constant: float, penalty: float) -> None:
    """Add ``penalty * (constant + sum(coeff_i*x_i))^2``."""

    model.offset += penalty * constant * constant
    for idx, (var, coeff) in enumerate(terms):
        model.add_linear(var, penalty * (coeff * coeff + 2.0 * constant * coeff))
        for other, other_coeff in terms[idx + 1 :]:
            model.add_quadratic(var, other, penalty * 2.0 * coeff * other_coeff)


class TargetLayoutQuboBuilder:
    def build(self, instance: Instance, args: argparse.Namespace) -> tuple[QuboModel, dict[tuple[int, int], int]]:
        state = create_initial_state(instance)
        current_stack = {box: i for i, stack in enumerate(state.stacks) for box in stack}
        removal_cost = _removal_costs(instance)
        settled = _settled_prefix_boxes(instance)
        model = QuboModel()
        var_for: dict[tuple[int, int], int] = {}

        for box in range(1, instance.container_count + 1):
            vars_box: list[int] = []
            for stack_idx in range(instance.stack_count):
                var = model.add_binary_var(f"x_{box}_{stack_idx}")
                var_for[(box, stack_idx)] = var
                vars_box.append(var)
                move_cost = 0.0 if current_stack[box] == stack_idx else removal_cost[box]
                keep_cost = 0.0 if current_stack[box] == stack_idx else 1.0
                settled_cost = 0.0 if box not in settled or current_stack[box] == stack_idx else args.settled_break_weight
                smaller_current = sum(1 for q in instance.stacks[stack_idx] if q < box)
                model.add_linear(
                    var,
                    args.move_weight * move_cost
                    + args.keep_weight * keep_cost
                    + settled_cost
                    + args.stack_order_weight * smaller_current,
                )
            model.add_one_hot_penalty(vars_box, args.assignment_penalty)

        for stack_idx in range(instance.stack_count):
            load_terms = [(var_for[(box, stack_idx)], 1.0) for box in range(1, instance.container_count + 1)]
            slack_terms: list[tuple[int, float]] = []
            slack_vars: list[int] = []
            for slack in range(instance.max_height + 1):
                var = model.add_binary_var(f"slack_{stack_idx}_{slack}")
                slack_vars.append(var)
                slack_terms.append((var, float(slack)))
                model.add_linear(var, args.empty_slot_weight * slack)
            model.add_one_hot_penalty(slack_vars, args.capacity_penalty)
            _add_square_penalty(model, load_terms + slack_terms, -float(instance.max_height), args.capacity_penalty)

        for stack in instance.stacks:
            pos = {box: idx for idx, box in enumerate(stack)}
            for p in stack:
                for q in stack:
                    if p < q and pos[p] < pos[q]:
                        for stack_idx in range(instance.stack_count):
                            model.add_quadratic(var_for[(p, stack_idx)], var_for[(q, stack_idx)], args.conflict_weight)
        return model, var_for


def _decode_target(
    instance: Instance,
    sample: QuboSample | None,
    var_for: dict[tuple[int, int], int],
    sample_index: int,
) -> DecodedTarget:
    target: dict[int, int] = {}
    repairs = 0
    for box in range(1, instance.container_count + 1):
        chosen = [
            stack
            for stack in range(instance.stack_count)
            if sample is not None and sample.binary_values.get(var_for[(box, stack)], 0) == 1
        ]
        if len(chosen) != 1:
            repairs += 1
            chosen = [
                max(
                    range(instance.stack_count),
                    key=lambda stack: sample.binary_values.get(var_for[(box, stack)], 0) if sample is not None else 0,
                )
            ]
        target[box] = chosen[0]

    loads = {stack: 0 for stack in range(instance.stack_count)}
    for stack in target.values():
        loads[stack] += 1
    underfull = [stack for stack, load in loads.items() for _ in range(max(0, instance.max_height - load))]
    for stack in range(instance.stack_count):
        boxes = [box for box, assigned in target.items() if assigned == stack]
        boxes.sort(reverse=True)
        while len(boxes) > instance.max_height and underfull:
            box = boxes.pop()
            target[box] = underfull.pop(0)
            repairs += 1
    return DecodedTarget(
        assignment=target,
        repairs=repairs,
        energy=sample.energy if sample is not None else math.inf,
        sample_index=sample_index,
        target_cost=_target_assignment_cost(instance, target),
    )


def _target_assignment_cost(instance: Instance, target: dict[int, int]) -> float:
    current = {box: i for i, stack in enumerate(instance.stacks) for box in stack}
    off = sum(1 for box, stack in current.items() if target.get(box) != stack)
    conflict = 0
    for stack in instance.stacks:
        pos = {box: idx for idx, box in enumerate(stack)}
        for p in stack:
            for q in stack:
                if p < q and pos[p] < pos[q] and target.get(p) == target.get(q):
                    conflict += 1
    loads = [sum(1 for stack in target.values() if stack == i) for i in range(instance.stack_count)]
    overflow = sum(max(0, load - instance.max_height) for load in loads)
    return float(off + 3 * conflict + 1000 * overflow)


def _target_stacks(instance: Instance, target: dict[int, int]) -> list[list[int]]:
    stacks: list[list[int]] = [[] for _ in range(instance.stack_count)]
    for box, stack in target.items():
        stacks[stack].append(box)
    return [sorted(stack, reverse=True) for stack in stacks]


def _off_target_count(state: State, target: dict[int, int]) -> int:
    return sum(1 for stack_idx, stack in enumerate(state.stacks) for box in stack if target.get(box) != stack_idx)


def _target_exact(instance: Instance, state: State, target: dict[int, int]) -> bool:
    return state.stacks == _target_stacks(instance, target)


def _move_target_score(instance: Instance, state: State, move: Move, target: dict[int, int]) -> tuple:
    try:
        nxt = apply_move(instance, state, move)
    except ValueError:
        return (999, 999, 999, 999)
    box_target = target.get(move.box)
    to_target = 0 if box_target == move.to_stack else 1
    away_from_target = 1 if box_target == move.from_stack and box_target != move.to_stack else 0
    off = _off_target_count(nxt, target)
    n_value = disorder_box_count(instance, nxt)
    return (to_target, away_from_target, off, n_value)


def _state_score(instance: Instance, state: State, config: SolverConfig, target: dict[int, int], require_target: bool) -> tuple:
    n_value = disorder_box_count(instance, state)
    off = _off_target_count(state, target)
    exact = 0 if _target_exact(instance, state, target) else 1
    if require_target:
        return (off, exact, n_value, state.move_count, state.entropy_max)
    return (n_value, off, state.move_count, state.entropy_max, score_state(instance, state, config.score_weights))


def _realize_target(
    instance: Instance,
    target: dict[int, int],
    args: argparse.Namespace,
    deadline: float,
    require_target: bool,
) -> tuple[State | None, int, int, int, bool]:
    initial = create_initial_state(instance)
    config = SolverConfig(
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        max_candidates_per_layer=args.max_candidates_per_layer,
        candidate_limit_per_state=args.candidate_limit_per_state,
        entropy_alpha=args.alpha,
        enable_visited_pruning=True,
        enable_canonical_hash=False,
        random_seed=args.seed,
    )
    beam = [initial]
    visited = {hash_state(instance, initial, False): dominance_value(instance, initial)}
    expanded = generated = 0
    best_goal: State | None = None
    best_goal_exact = False

    for _depth in range(args.max_depth):
        if time.perf_counter() >= deadline:
            break
        candidates: list[tuple[tuple, State]] = []
        for state in beam:
            expanded += 1
            moves = generate_legal_moves(instance, state)
            moves.sort(key=lambda mv: _move_target_score(instance, state, mv, target))
            if args.candidate_limit_per_state is not None:
                moves = moves[: args.candidate_limit_per_state]
            for move in moves:
                try:
                    nxt = apply_move(instance, state, move)
                except ValueError:
                    continue
                if not cpmp_business_feasible(instance, nxt, args.alpha):
                    continue
                h = hash_state(instance, nxt, False)
                dom = dominance_value(instance, nxt)
                if visited.get(h, math.inf) <= dom:
                    continue
                visited[h] = dom
                generated += 1
                exact = _target_exact(instance, nxt, target)
                if is_goal(instance, nxt) and (exact or not require_target):
                    if best_goal is None or len(nxt.moves) < len(best_goal.moves) or (exact and not best_goal_exact):
                        best_goal = nxt
                        best_goal_exact = exact
                candidates.append((_state_score(instance, nxt, config, target, require_target), nxt))
        if best_goal is not None:
            return best_goal, expanded, generated, len(visited), best_goal_exact
        if not candidates:
            break
        candidates.sort(key=lambda item: item[0])
        if args.max_candidates_per_layer:
            candidates = candidates[: args.max_candidates_per_layer]
        beam = [state for _score, state in candidates[: args.beam_width]]
    return None, expanded, generated, len(visited), False


def _run_case(case: str, zone: str, bay: str, args: argparse.Namespace) -> LayoutRun:
    start = time.perf_counter()
    deadline = start + args.time_limit if args.time_limit > 0 else math.inf
    instance = _instance_from_single_bay(zone, bay)
    builder = TargetLayoutQuboBuilder()
    model, var_for = builder.build(instance, args)
    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        QuboSamplerConfig(
            num_reads=args.qubo_reads,
            random_seed=args.seed,
            time_limit_sec=args.qubo_time_limit,
            sa_iterations=args.qubo_iterations,
        ),
    )
    decoded = [
        _decode_target(instance, sample, var_for, idx)
        for idx, sample in enumerate(sample_set.samples[: args.target_samples])
    ]
    decoded.sort(key=lambda target: (target.repairs, target.target_cost, target.energy, target.sample_index))
    if not decoded:
        decoded = [_decode_target(instance, None, var_for, 0)]

    total_expanded = total_generated = max_visited = 0
    last_status = "NO_TARGET_DECODED"
    for target in decoded:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        state, expanded, generated, visited, exact = _realize_target(
            instance,
            target.assignment,
            args,
            deadline,
            args.require_target,
        )
        total_expanded += expanded
        total_generated += generated
        max_visited = max(max_visited, visited)
        if state is not None:
            return LayoutRun(
                case=case,
                state=state,
                runtime_sec=time.perf_counter() - start,
                expanded=total_expanded,
                generated=total_generated,
                visited=max_visited,
                qvars=len(model.var_index),
                qtime=sample_set.runtime_sec,
                qrepairs=target.repairs,
                qsample_index=target.sample_index,
                target_exact=exact,
                status=(
                    f"FEASIBLE; qsample={target.sample_index}; qenergy={target.energy:.3f}; "
                    f"target_cost={target.target_cost:.3f}; target_exact={exact}; "
                    f"require_target={args.require_target}; qrepairs={target.repairs}"
                ),
            )
        last_status = (
            f"REALIZATION_FAILED; tried_sample={target.sample_index}; "
            f"target_cost={target.target_cost:.3f}; qrepairs={target.repairs}"
        )
    return LayoutRun(
        case=case,
        state=None,
        runtime_sec=time.perf_counter() - start,
        expanded=total_expanded,
        generated=total_generated,
        visited=max_visited,
        qvars=len(model.var_index),
        qtime=sample_set.runtime_sec,
        qrepairs=sum(target.repairs for target in decoded),
        qsample_index=-1,
        target_exact=False,
        status=last_status,
    )


def _removal_costs(instance: Instance) -> dict[int, int]:
    costs: dict[int, int] = {}
    for stack in instance.stacks:
        for idx, box in enumerate(stack):
            costs[box] = 1 + len(stack) - idx - 1
    return costs


def _settled_prefix_boxes(instance: Instance) -> set[int]:
    out: set[int] = set()
    for stack in instance.stacks:
        prefix: list[int] = []
        for box in stack:
            if prefix and prefix[-1] < box:
                break
            prefix.append(box)
        out.update(prefix)
    return out


def _case_filter(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return [case if case.endswith("_single") else f"{case}_single" for case in args.cases]
    return [f"{zone}_{bay}_single" for zone, bay in discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))]


def _row_from_run(run: LayoutRun, zone: str, bay: str) -> TrajectoryResult:
    instance = _instance_from_single_bay(zone, bay)
    initial = create_initial_state(instance)
    n0 = disorder_box_count(instance, initial)
    if run.state is not None:
        final_n = disorder_box_count(instance, run.state)
        moves = len(run.state.moves)
        max_g = max_entropy_along_moves(instance, run.state.moves)
    else:
        final_n = n0
        moves = None
        max_g = n0
    cap13 = entropy_limit(instance, 1.3)
    cap15 = entropy_limit(instance, 1.5)
    return TrajectoryResult(
        case=run.case,
        solver=SOLVER_NAME,
        n0=n0,
        cap_13=cap13,
        cap_15=cap15,
        moves=moves,
        final_n=final_n,
        reduction=n0 - final_n if final_n is not None else None,
        g_end=(moves + final_n) if moves is not None and final_n is not None else None,
        max_g=max_g,
        g_ok_13=max_g <= cap13,
        g_ok_15=max_g <= cap15,
        runtime_sec=run.runtime_sec,
        expanded_states=run.expanded,
        generated_states=run.generated,
        visited_states=run.visited,
        qubo_calls=1,
        qubo_total_vars=run.qvars,
        qubo_time_sec=run.qtime,
        status=run.status,
    )


def _validate_run(run: LayoutRun, zone: str, bay: str) -> tuple[bool, bool, str]:
    if run.state is None:
        return False, False, run.status
    instance = _instance_from_single_bay(zone, bay)
    solution = Solution(
        solver_name=SOLVER_NAME,
        feasible=True,
        optimal=False,
        objective=float(len(run.state.moves)),
        runtime_sec=run.runtime_sec,
        moves=run.state.moves,
        status=run.status,
    )
    goal_ok, goal_msg, _ = validate_solution(instance, solution)
    business_ok = max_entropy_along_moves(instance, run.state.moves) <= entropy_limit(instance, 1.3)
    return business_ok, goal_ok, goal_msg


def _write_rows(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_acceptance(
    rows: list[TrajectoryResult],
    validation: dict[str, tuple[bool, bool, str]],
    moves_paths: dict[str, Path],
    exact_refs: dict[str, int],
    path: Path,
) -> None:
    out = []
    for row in rows:
        business_ok, goal_ok, msg = validation[row.case]
        out.append(
            {
                "case": row.case,
                "selected_solver": row.solver,
                "final_n": row.final_n,
                "moves": row.moves,
                "runtime_sec": f"{row.runtime_sec:.6f}",
                "exact_moves": exact_refs.get(row.case, ""),
                "move_gap": _move_gap(row, exact_refs),
                "qubo_used": "True",
                "selected_reason": row.status,
                "business_ok": str(business_ok),
                "goal_validation_ok": str(goal_ok),
                "validation_msg": msg,
                "moves_path": _rel(moves_paths.get(row.case, Path(""))),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ACCEPTANCE_FIELDS)
        writer.writeheader()
        writer.writerows(out)


def _write_report(rows: list[TrajectoryResult], output: Path, acceptance: Path, report: Path, exact_refs: dict[str, int]) -> None:
    solved = sum(1 for row in rows if row.final_n == 0)
    runtimes = [row.runtime_sec for row in rows]
    gaps = [
        row.moves - exact_refs[row.case]
        for row in rows
        if row.final_n == 0 and row.moves is not None and row.case in exact_refs
    ]
    lines = [
        f"# {SOLVER_NAME}",
        "",
        "Target-layout QUBO upper-bound benchmark. QUBO selects final stack assignment; deterministic beam realizes it.",
        "",
        f"Result CSV: `{_rel(output)}`",
        f"Acceptance CSV: `{_rel(acceptance)}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| cases | {len(rows)} |",
        f"| N=0 solved | {solved}/{len(rows)} |",
        f"| avg runtime sec | {sum(runtimes) / len(runtimes) if runtimes else 0.0:.3f} |",
        f"| max runtime sec | {max(runtimes, default=0.0):.3f} |",
        f"| avg move gap | {sum(gaps) / len(gaps) if gaps else 0.0:.3f} |",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "target_layout_qubo_upper_bound_v1.csv")
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--beam-width", type=int, default=400)
    parser.add_argument("--max-candidates-per-layer", type=int, default=30000)
    parser.add_argument("--candidate-limit-per-state", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--require-target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-samples", type=int, default=8)
    parser.add_argument("--qubo-time-limit", type=float, default=0.08)
    parser.add_argument("--qubo-reads", type=int, default=24)
    parser.add_argument("--qubo-iterations", type=int, default=400)
    parser.add_argument("--assignment-penalty", type=float, default=45.0)
    parser.add_argument("--capacity-penalty", type=float, default=45.0)
    parser.add_argument("--move-weight", type=float, default=0.9)
    parser.add_argument("--keep-weight", type=float, default=0.6)
    parser.add_argument("--conflict-weight", type=float, default=3.5)
    parser.add_argument("--stack-order-weight", type=float, default=0.08)
    parser.add_argument("--settled-break-weight", type=float, default=1.0)
    parser.add_argument("--empty-slot-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    acceptance = output.with_name(output.stem + "_acceptance.csv")
    moves_dir = output.with_name(output.stem + "_moves")
    report = output.with_suffix(".md")
    moves_dir.mkdir(parents=True, exist_ok=True)

    cases = _case_filter(args)
    exact_refs = _exact_moves_by_case()
    rows: list[TrajectoryResult] = []
    validation: dict[str, tuple[bool, bool, str]] = {}
    moves_paths: dict[str, Path] = {}

    for idx, case in enumerate(cases, start=1):
        zone, bay = _parse_case(case)
        print(f"CASE_START {idx}/{len(cases)} {case} solver={SOLVER_NAME}", flush=True)
        run = _run_case(case, zone, bay, args)
        row = _row_from_run(run, zone, bay)
        business_ok, goal_ok, msg = _validate_run(run, zone, bay)
        if run.state is not None:
            move_path = moves_dir / f"{case}_moves.csv"
            _write_explain_csv(_instance_from_single_bay(zone, bay), run.state.moves, move_path)
            moves_paths[case] = move_path
        rows.append(row)
        validation[case] = (business_ok, goal_ok, msg)
        _write_rows(rows, output)
        _write_acceptance(rows, validation, moves_paths, exact_refs, acceptance)
        _write_report(rows, output, acceptance, report, exact_refs)
        print(
            "CASE_DONE "
            f"{case} status={run.status.split(';')[0]} final_n={row.final_n} moves={row.moves} "
            f"runtime={row.runtime_sec:.3f}s valid={business_ok and goal_ok} "
            f"qvars={run.qvars} target_exact={run.target_exact}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
