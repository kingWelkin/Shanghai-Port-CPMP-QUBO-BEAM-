"""Compare QUBO upper-bound generation routes on real single-bay CPMP cases.

The routes here intentionally keep feasibility in the classical state engine:

* one_step_qubo: build a one-hot QUBO over legal next moves.
* sequence_pool_qubo: build a one-hot QUBO over short legal move sequences.
* seeded_astar: use the best validated N=0 QUBO route as an initial incumbent
  for exact A* pruning.

Rows with final_n == 0 are strict CPMP upper bounds on the target-reaching move
count. Rows with final_n > 0 are business-feasible incumbents for the current
project objective, but not strict terminal CPMP upper bounds.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import random
import sys
import time
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam import _write_explain_csv  # noqa: E402
from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
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
    is_goal,
    validate_solution,
)
from container_reshuffle.features import (  # noqa: E402
    bad_count,
    dirty_stack_count,
    disorder_box_count,
    entropy_limit,
    entropy_value,
    inversion_count,
    max_entropy_along_moves,
    settled_box_count,
)
from container_reshuffle.qubo.model import QuboModel  # noqa: E402
from container_reshuffle.qubo.sampler import QuboSample, QuboSamplerConfig, SimulatedAnnealingQuboSampler  # noqa: E402
from container_reshuffle.scoring import cpmp_business_feasible, score_state  # noqa: E402
from container_reshuffle.solvers.astar import ExactAStarSolver  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


RESULT_FIELDS = list(TrajectoryResult.__annotations__.keys())
ACCEPTANCE_FIELDS = [
    "case",
    "solver",
    "final_n",
    "moves",
    "runtime_sec",
    "exact_moves",
    "move_gap",
    "business_ok",
    "goal_validation_ok",
    "validation_msg",
    "moves_path",
    "status",
]


@dataclass(frozen=True)
class MoveOption:
    state: State
    moves: tuple[Move, ...]
    score: float
    hash_value: tuple


@dataclass
class MethodRun:
    row: TrajectoryResult
    state: State
    business_ok: bool
    goal_ok: bool
    validation_msg: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _exact_moves_by_case() -> dict[str, int]:
    refs: dict[str, int] = {}
    for row in _read_csv(ROOT / "docs" / "experiments" / "exact_target_baseline_reference.csv"):
        value = row.get("exact_moves_witness")
        if value not in (None, ""):
            refs[row["case"]] = int(float(value))
    for row in _read_csv(ROOT / "docs" / "experiments" / "fast_slack_vs_exact_all_single_bays_8s_w120.csv"):
        value = row.get("exact_moves_to_n0") or row.get("exact_moves")
        if value not in (None, ""):
            refs.setdefault(row["case"], int(float(value)))
    return refs


def _parse_case(case: str) -> tuple[str, str]:
    parts = case.split("_")
    if len(parts) < 2:
        raise ValueError(f"cannot parse case name: {case}")
    return parts[0], parts[1]


def _normalize_case(case: str) -> str:
    return case if case.endswith("_single") else f"{case}_single"


def _case_filter(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return [_normalize_case(case) for case in args.cases]
    return [
        f"{zone}_{bay}_single"
        for zone, bay in discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))
    ]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _best_state_key(instance: Instance, state: State) -> tuple:
    return (
        disorder_box_count(instance, state),
        state.move_count,
        state.entropy_max,
        state.disorder_total,
        inversion_count(instance, state),
        bad_count(instance, state),
        dirty_stack_count(instance, state),
        -settled_box_count(instance, state),
    )


def _numeric_state_score(instance: Instance, state: State, config: SolverConfig) -> float:
    n_value = disorder_box_count(instance, state)
    g_value = entropy_value(instance, state)
    return (
        1_000_000.0 * n_value
        + 3_000.0 * g_value
        + 50.0 * state.move_count
        + 0.05 * state.disorder_total
        + 12.0 * bad_count(instance, state)
        + 4.0 * inversion_count(instance, state)
        + 8.0 * dirty_stack_count(instance, state)
        - 2.0 * settled_box_count(instance, state)
        + 0.001 * score_state(instance, state, config.score_weights)
    )


def _tail_preview(
    instance: Instance,
    state: State,
    config: SolverConfig,
    *,
    depth: int,
    max_children: int,
) -> State:
    """Greedy preview used only as a candidate score proxy."""

    preview = state
    seen = {hash_state(instance, preview, config.enable_canonical_hash)}
    for _ in range(max(0, depth)):
        if is_goal(instance, preview):
            break
        children = []
        for move in generate_legal_moves(instance, preview):
            try:
                nxt = apply_move(instance, preview, move)
            except ValueError:
                continue
            if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                continue
            h = hash_state(instance, nxt, config.enable_canonical_hash)
            if h in seen:
                continue
            children.append((_numeric_state_score(instance, nxt, config), nxt, h))
        if not children:
            break
        children.sort(key=lambda item: item[0])
        children = children[: max(1, max_children)]
        _score, preview, h = children[0]
        seen.add(h)
    return preview


def _option_score(
    instance: Instance,
    state: State,
    config: SolverConfig,
    *,
    tail_depth: int,
    max_tail_children: int,
    jitter: float,
    rng: random.Random,
) -> float:
    preview = _tail_preview(
        instance,
        state,
        config,
        depth=tail_depth,
        max_children=max_tail_children,
    )
    score = _numeric_state_score(instance, state, config)
    if preview is not state:
        score += 0.35 * _numeric_state_score(instance, preview, config)
        score += 1_000.0 * (preview.move_count - state.move_count)
    if jitter:
        score += rng.uniform(0.0, jitter)
    return score


def _sample_one_hot(
    scores: list[float],
    *,
    penalty: float,
    seed: int,
    reads: int,
    iterations: int,
    time_limit: float,
) -> tuple[int, int, int, float, float]:
    """Return selected index, repair count, var count, QUBO runtime, energy."""

    model = QuboModel()
    vars_ = [model.add_binary_var(f"y_{i}") for i in range(len(scores))]
    if not scores:
        return -1, 0, 0, 0.0, math.inf
    lo, hi = min(scores), max(scores)
    span = max(1e-9, hi - lo)
    for var, score in zip(vars_, scores):
        model.add_linear(var, (score - lo) / span)
    model.add_one_hot_penalty(vars_, penalty)
    ss = SimulatedAnnealingQuboSampler().sample(
        model,
        QuboSamplerConfig(
            num_reads=reads,
            random_seed=seed,
            time_limit_sec=time_limit,
            sa_iterations=iterations,
        ),
    )
    best_idx = min(range(len(scores)), key=lambda idx: (scores[idx], idx))
    best_repair = 1
    best_value = (scores[best_idx], best_repair, math.inf)
    for sample in ss.samples or [QuboSample({}, math.inf)]:
        selected = [idx for idx in range(len(scores)) if sample.binary_values.get(idx, 0) == 1]
        repair = 0
        if len(selected) != 1:
            repair = 1
            selected = [min(selected, key=lambda idx: (scores[idx], idx))] if selected else [best_idx]
        idx = selected[0]
        value = (scores[idx], repair, sample.energy)
        if value < best_value:
            best_idx = idx
            best_repair = repair
            best_value = value
    return best_idx, best_repair, len(model.var_index), ss.runtime_sec, float(best_value[2])


def _dedupe_options(options: Iterable[MoveOption]) -> list[MoveOption]:
    best: dict[tuple, MoveOption] = {}
    for option in options:
        old = best.get(option.hash_value)
        if old is None or (option.score, option.state.move_count) < (old.score, old.state.move_count):
            best[option.hash_value] = option
    out = list(best.values())
    out.sort(key=lambda opt: (opt.score, opt.state.move_count))
    return out


def _build_one_step_options(
    instance: Instance,
    state: State,
    config: SolverConfig,
    args: argparse.Namespace,
    seen: set[tuple],
    rng: random.Random,
) -> tuple[list[MoveOption], int]:
    options: list[MoveOption] = []
    generated = 0
    for move in generate_legal_moves(instance, state):
        try:
            nxt = apply_move(instance, state, move)
        except ValueError:
            continue
        if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
            continue
        h = hash_state(instance, nxt, config.enable_canonical_hash)
        if h in seen:
            continue
        generated += 1
        score = _option_score(
            instance,
            nxt,
            config,
            tail_depth=args.tail_preview_depth,
            max_tail_children=args.tail_preview_children,
            jitter=args.jitter,
            rng=rng,
        )
        options.append(MoveOption(nxt, (move,), score, h))
    options.sort(key=lambda opt: (opt.score, _best_state_key(instance, opt.state)))
    return options[: args.max_options], generated


def _build_sequence_options(
    instance: Instance,
    state: State,
    config: SolverConfig,
    args: argparse.Namespace,
    seen: set[tuple],
    rng: random.Random,
) -> tuple[list[MoveOption], int, int]:
    """Enumerate a truncated pool of legal short sequences."""

    root_key = hash_state(instance, state, config.enable_canonical_hash)
    frontier: list[tuple[State, tuple[Move, ...], set[tuple]]] = [(state, tuple(), {root_key})]
    options: list[MoveOption] = []
    expanded = generated = 0
    for depth in range(max(1, args.sequence_depth)):
        next_frontier: list[tuple[State, tuple[Move, ...], set[tuple]]] = []
        for current, path, path_keys in frontier:
            expanded += 1
            children: list[MoveOption] = []
            for move in generate_legal_moves(instance, current):
                try:
                    nxt = apply_move(instance, current, move)
                except ValueError:
                    continue
                if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                    continue
                h = hash_state(instance, nxt, config.enable_canonical_hash)
                if h in seen or h in path_keys:
                    continue
                generated += 1
                new_path = path + (move,)
                score = _option_score(
                    instance,
                    nxt,
                    config,
                    tail_depth=args.tail_preview_depth,
                    max_tail_children=args.tail_preview_children,
                    jitter=args.jitter,
                    rng=rng,
                )
                children.append(MoveOption(nxt, new_path, score, h))
            children.sort(key=lambda opt: (opt.score, _best_state_key(instance, opt.state)))
            for child in children[: args.max_children_per_node]:
                next_keys = set(path_keys)
                next_keys.add(child.hash_value)
                next_frontier.append((child.state, child.moves, next_keys))
                if depth + 1 >= args.sequence_min_depth or is_goal(instance, child.state):
                    options.append(child)
        if not next_frontier:
            break
        next_frontier.sort(key=lambda item: (_numeric_state_score(instance, item[0], config), item[0].move_count))
        frontier = next_frontier[: args.sequence_frontier_width]
        if len(options) >= args.max_options and depth + 1 >= args.sequence_min_depth:
            break
    options = _dedupe_options(options)
    return options[: args.max_options], expanded, generated


def _run_qubo_route(
    case: str,
    instance: Instance,
    config: SolverConfig,
    args: argparse.Namespace,
    *,
    solver_name: str,
    sequence_mode: bool,
) -> MethodRun:
    start = time.perf_counter()
    deadline = start + args.time_limit
    best = create_initial_state(instance)
    expanded = generated = qcalls = qvars = qrepairs = 0
    tail_expanded = tail_generated = 0
    tail_found = False
    qtime = 0.0
    last_energy = math.inf

    for rollout in range(max(1, args.rollouts)):
        if time.perf_counter() > deadline:
            break
        rng = random.Random(args.seed + 1009 * rollout + (17 if sequence_mode else 0))
        state = create_initial_state(instance)
        seen = {hash_state(instance, state, config.enable_canonical_hash)}

        while state.move_count < config.max_depth and time.perf_counter() <= deadline:
            if is_goal(instance, state):
                break
            if sequence_mode:
                options, exp_count, gen_count = _build_sequence_options(instance, state, config, args, seen, rng)
                expanded += exp_count
                generated += gen_count
            else:
                options, gen_count = _build_one_step_options(instance, state, config, args, seen, rng)
                expanded += 1
                generated += gen_count
            if not options:
                break
            scores = [opt.score for opt in options]
            idx, repair, var_count, sample_time, energy = _sample_one_hot(
                scores,
                penalty=args.qubo_penalty,
                seed=args.seed + qcalls + 31 * rollout,
                reads=args.qubo_reads,
                iterations=args.qubo_iterations,
                time_limit=args.qubo_time_limit,
            )
            qcalls += 1
            qvars += var_count
            qtime += sample_time
            qrepairs += repair
            last_energy = energy
            if idx < 0:
                break
            chosen = options[idx]
            state = chosen.state
            seen.add(chosen.hash_value)
            if _best_state_key(instance, state) < _best_state_key(instance, best):
                best = state
            tail_probe = [chosen]
            if args.tail_probe_options > 0:
                seen_probe_hashes = {chosen.hash_value}
                for option in options[: args.tail_probe_options]:
                    if option.hash_value in seen_probe_hashes:
                        continue
                    seen_probe_hashes.add(option.hash_value)
                    tail_probe.append(option)
            for probe in tail_probe:
                if disorder_box_count(instance, probe.state) > args.tail_repair_n:
                    continue
                if time.perf_counter() > deadline:
                    break
                repaired, exp_count, gen_count = tail_repair_to_n_zero(
                    instance,
                    probe.state,
                    config,
                    deadline=deadline,
                    max_expanded=args.tail_repair_expanded,
                )
                tail_expanded += exp_count
                tail_generated += gen_count
                if repaired is not None and _best_state_key(instance, repaired) < _best_state_key(instance, best):
                    best = repaired
                    state = repaired
                    tail_found = True
                    break
            if tail_found:
                break
            if is_goal(instance, state):
                best = state
                break

    status = "N_ZERO" if is_goal(instance, best) else "BEST_EFFORT"
    status += (
        f"; rollouts={args.rollouts}; qrepairs={qrepairs}; last_energy={last_energy:.6g}; "
        f"mode={'sequence_pool' if sequence_mode else 'one_step'}; "
        f"tail_expanded={tail_expanded}; tail_generated={tail_generated}; tail_found={tail_found}"
    )
    row = _result_from_state(
        case,
        solver_name,
        instance,
        best,
        start,
        expanded_states=expanded + tail_expanded,
        generated_states=generated + tail_generated,
        visited_states=0,
        qubo_calls=calls_safe(qcalls),
        qubo_total_vars=qvars,
        qubo_time_sec=qtime,
        status=status,
    )
    business_ok, goal_ok, msg = _validate_state(instance, best, row.runtime_sec)
    return MethodRun(row, best, business_ok, goal_ok, msg)


def calls_safe(value: int) -> int:
    return max(0, int(value))


def _validate_state(instance: Instance, state: State, runtime_sec: float) -> tuple[bool, bool, str]:
    cap = entropy_limit(instance, 1.3)
    try:
        max_g = max_entropy_along_moves(instance, state.moves) if state.moves else entropy_value(instance, state)
    except Exception as exc:  # noqa: BLE001
        return False, False, str(exc)
    business_ok = max_g <= cap
    if not is_goal(instance, state):
        return business_ok, False, "best-effort state is not a terminal CPMP goal"
    sol = Solution(
        solver_name="validation",
        feasible=True,
        optimal=False,
        objective=float(len(state.moves)),
        runtime_sec=runtime_sec,
        moves=state.moves,
        status="N_ZERO",
    )
    goal_ok, goal_msg, _ = validate_solution(instance, sol)
    return business_ok, goal_ok, goal_msg


def _run_seeded_astar(
    case: str,
    instance: Instance,
    config: SolverConfig,
    args: argparse.Namespace,
    seed_state: State | None,
) -> MethodRun:
    start = time.perf_counter()
    initial = create_initial_state(instance)
    if seed_state is None or not is_goal(instance, seed_state):
        row = _result_from_state(
            case,
            "seeded_astar_from_qubo",
            instance,
            initial,
            start,
            status="NO_VALID_N_ZERO_QUBO_SEED",
        )
        business_ok, goal_ok, msg = _validate_state(instance, initial, row.runtime_sec)
        return MethodRun(row, initial, business_ok, goal_ok, msg)

    incumbent = seed_state
    initial_key = hash_state(instance, initial, config.enable_canonical_hash)
    best_g = {initial_key: 0}
    heap: list[tuple[float, int, int, int, State]] = []
    counter = 0
    h0 = ExactAStarSolver._lower_bound(instance, initial)
    heapq.heappush(heap, (h0, h0, 0, counter, initial))
    expanded = generated = pruned_by_seed = 0
    depth_limited = False
    deadline = start + args.astar_time_limit

    while heap:
        if time.perf_counter() > deadline:
            row = _result_from_state(
                case,
                "seeded_astar_from_qubo",
                instance,
                incumbent,
                start,
                expanded_states=expanded,
                generated_states=generated,
                visited_states=len(best_g),
                status=f"TIME_LIMIT_FEASIBLE_NOT_PROVEN; seed_moves={seed_state.move_count}; pruned_by_seed={pruned_by_seed}",
            )
            business_ok, goal_ok, msg = _validate_state(instance, incumbent, row.runtime_sec)
            return MethodRun(row, incumbent, business_ok, goal_ok, msg)

        f_value, _h, g_value, _idx, state = heapq.heappop(heap)
        state_key = hash_state(instance, state, config.enable_canonical_hash)
        if g_value != best_g.get(state_key, inf):
            continue
        if f_value >= incumbent.move_count:
            row = _result_from_state(
                case,
                "seeded_astar_from_qubo",
                instance,
                incumbent,
                start,
                expanded_states=expanded,
                generated_states=generated,
                visited_states=len(best_g),
                status=f"OPTIMAL; seed_moves={seed_state.move_count}; pruned_by_seed={pruned_by_seed}",
            )
            business_ok, goal_ok, msg = _validate_state(instance, incumbent, row.runtime_sec)
            return MethodRun(row, incumbent, business_ok, goal_ok, msg)
        if is_goal(instance, state):
            if state.move_count < incumbent.move_count:
                incumbent = state
            continue
        if state.move_count >= config.max_depth:
            depth_limited = True
            continue

        expanded += 1
        for move in generate_legal_moves(instance, state):
            try:
                nxt = apply_move(instance, state, move)
            except ValueError:
                continue
            if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                continue
            generated += 1
            nxt_key = hash_state(instance, nxt, config.enable_canonical_hash)
            nxt_g = nxt.move_count
            if nxt_g >= best_g.get(nxt_key, inf):
                continue
            best_g[nxt_key] = nxt_g
            if is_goal(instance, nxt) and nxt.move_count < incumbent.move_count:
                incumbent = nxt
            nxt_h = ExactAStarSolver._lower_bound(instance, nxt)
            if nxt_g + nxt_h >= incumbent.move_count:
                pruned_by_seed += 1
                continue
            counter += 1
            heapq.heappush(heap, (nxt_g + nxt_h, nxt_h, nxt_g, counter, nxt))

    status = "OPTIMAL" if not depth_limited else "DEPTH_LIMIT_WITH_INCUMBENT"
    row = _result_from_state(
        case,
        "seeded_astar_from_qubo",
        instance,
        incumbent,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=len(best_g),
        status=f"{status}; seed_moves={seed_state.move_count}; pruned_by_seed={pruned_by_seed}",
    )
    business_ok, goal_ok, msg = _validate_state(instance, incumbent, row.runtime_sec)
    return MethodRun(row, incumbent, business_ok, goal_ok, msg)


def _write_rows(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _move_gap(row: TrajectoryResult, exact_refs: dict[str, int]) -> str:
    exact = exact_refs.get(row.case)
    if exact is None or row.final_n != 0 or row.moves is None:
        return ""
    return str(row.moves - exact)


def _write_acceptance(
    runs: list[MethodRun],
    exact_refs: dict[str, int],
    move_paths: dict[tuple[str, str], Path],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ACCEPTANCE_FIELDS)
        writer.writeheader()
        for run in runs:
            row = run.row
            writer.writerow(
                {
                    "case": row.case,
                    "solver": row.solver,
                    "final_n": row.final_n,
                    "moves": row.moves,
                    "runtime_sec": f"{row.runtime_sec:.6f}",
                    "exact_moves": exact_refs.get(row.case, ""),
                    "move_gap": _move_gap(row, exact_refs),
                    "business_ok": str(run.business_ok),
                    "goal_validation_ok": str(run.goal_ok),
                    "validation_msg": run.validation_msg,
                    "moves_path": _rel(move_paths.get((row.case, row.solver), Path(""))),
                    "status": row.status,
                }
            )


def _write_report(runs: list[MethodRun], output: Path, acceptance: Path, report: Path, exact_refs: dict[str, int]) -> None:
    by_solver: dict[str, list[TrajectoryResult]] = {}
    for run in runs:
        by_solver.setdefault(run.row.solver, []).append(run.row)
    lines = [
        "# QUBO Upper-Bound Methods",
        "",
        f"Result CSV: `{_rel(output)}`",
        f"Acceptance CSV: `{_rel(acceptance)}`",
        "",
        "| solver | cases | N=0 solved | avg final N | avg moves | avg runtime sec | avg move gap when N=0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for solver, rows in sorted(by_solver.items()):
        solved = [row for row in rows if row.final_n == 0]
        gaps = [
            row.moves - exact_refs[row.case]
            for row in solved
            if row.moves is not None and row.case in exact_refs
        ]
        avg_final = sum(float(row.final_n or 0) for row in rows) / len(rows)
        move_rows = [row.moves for row in rows if row.moves is not None]
        avg_moves = sum(move_rows) / len(move_rows) if move_rows else 0.0
        avg_runtime = sum(row.runtime_sec for row in rows) / len(rows)
        avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
        lines.append(
            f"| `{solver}` | {len(rows)} | {len(solved)}/{len(rows)} | {avg_final:.3f} | "
            f"{avg_moves:.3f} | {avg_runtime:.3f} | {avg_gap:.3f} |"
        )
    lines.append("")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def _write_moves(instance: Instance, run: MethodRun, moves_dir: Path) -> Path:
    path = moves_dir / f"{run.row.case}_{run.row.solver}_moves.csv"
    _write_explain_csv(instance, run.state.moves, path)
    return path


def _run_case(case: str, args: argparse.Namespace) -> list[MethodRun]:
    zone, bay = _parse_case(case)
    instance = _instance_from_single_bay(zone, bay)
    cap = entropy_limit(instance, args.alpha)
    config = SolverConfig(
        max_depth=min(args.max_depth, cap),
        time_limit_sec=args.time_limit,
        entropy_alpha=args.alpha,
        random_seed=args.seed,
        enable_canonical_hash=args.canonical,
    )
    runs: list[MethodRun] = []
    if "one_step" in args.methods:
        runs.append(
            _run_qubo_route(
                case,
                instance,
                config,
                args,
                solver_name="one_step_qubo_upper_bound",
                sequence_mode=False,
            )
        )
    if "sequence_pool" in args.methods:
        runs.append(
            _run_qubo_route(
                case,
                instance,
                config,
                args,
                solver_name="sequence_pool_qubo_upper_bound",
                sequence_mode=True,
            )
        )
    if "seeded_astar" in args.methods:
        seed_candidates = [
            run.state
            for run in runs
            if run.row.final_n == 0 and run.business_ok and run.goal_ok
        ]
        seed_state = min(seed_candidates, key=lambda state: state.move_count, default=None)
        astar_config = SolverConfig(
            max_depth=min(args.max_depth, cap),
            time_limit_sec=args.astar_time_limit,
            entropy_alpha=args.alpha,
            enable_canonical_hash=args.canonical,
        )
        runs.append(_run_seeded_astar(case, instance, astar_config, args, seed_state))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["one_step", "sequence_pool", "seeded_astar"],
        default=["one_step", "sequence_pool", "seeded_astar"],
    )
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "qubo_upper_bound_methods_probe.csv")
    parser.add_argument("--time-limit", type=float, default=8.0)
    parser.add_argument("--astar-time-limit", type=float, default=8.0)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--max-options", type=int, default=80)
    parser.add_argument("--max-children-per-node", type=int, default=12)
    parser.add_argument("--sequence-depth", type=int, default=2)
    parser.add_argument("--sequence-min-depth", type=int, default=2)
    parser.add_argument("--sequence-frontier-width", type=int, default=60)
    parser.add_argument("--tail-preview-depth", type=int, default=1)
    parser.add_argument("--tail-preview-children", type=int, default=8)
    parser.add_argument("--tail-repair-n", type=int, default=6)
    parser.add_argument("--tail-repair-expanded", type=int, default=200_000)
    parser.add_argument("--tail-probe-options", type=int, default=8)
    parser.add_argument("--qubo-penalty", type=float, default=12.0)
    parser.add_argument("--qubo-reads", type=int, default=16)
    parser.add_argument("--qubo-iterations", type=int, default=250)
    parser.add_argument("--qubo-time-limit", type=float, default=0.05)
    parser.add_argument("--jitter", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--canonical", action="store_true")
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    acceptance = output.with_name(output.stem + "_acceptance.csv")
    report = output.with_suffix(".md")
    moves_dir = output.with_name(output.stem + "_moves")
    moves_dir.mkdir(parents=True, exist_ok=True)

    exact_refs = _exact_moves_by_case()
    all_runs: list[MethodRun] = []
    move_paths: dict[tuple[str, str], Path] = {}
    cases = _case_filter(args)
    for index, case in enumerate(cases, start=1):
        zone, bay = _parse_case(case)
        instance = _instance_from_single_bay(zone, bay)
        print(
            "CASE_START "
            f"{index}/{len(cases)} case={case} N0={disorder_box_count(instance, create_initial_state(instance))} "
            f"cap={entropy_limit(instance, args.alpha)} methods={','.join(args.methods)}",
            flush=True,
        )
        runs = _run_case(case, args)
        for run in runs:
            move_paths[(run.row.case, run.row.solver)] = _write_moves(instance, run, moves_dir)
            all_runs.append(run)
            print(
                "CASE_METHOD_DONE "
                f"case={run.row.case} solver={run.row.solver} finalN={run.row.final_n} "
                f"moves={run.row.moves} runtime={run.row.runtime_sec:.3f}s "
                f"business={run.business_ok} goal={run.goal_ok} status={run.row.status.split(';')[0]}",
                flush=True,
            )
        _write_rows([run.row for run in all_runs], output)
        _write_acceptance(all_runs, exact_refs, move_paths, acceptance)
        _write_report(all_runs, output, acceptance, report, exact_refs)

    print(f"wrote {output}", flush=True)
    print(f"wrote {acceptance}", flush=True)
    print(f"wrote {report}", flush=True)
    print(f"wrote moves to {moves_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
