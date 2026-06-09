"""Single-pass fast/QUBO integrated beam benchmark.

This experiment keeps one fixed online search flow:

1. generate one-step legal beam candidates,
2. rank them with a phase-aware quality score,
3. use bounded lookahead inside the same layer selection,
4. protect an elite deterministic core and let QUBO select supplements.

It is intentionally separate from the earlier staged/portfolio scripts so the
runtime and route quality are attributable to one algorithm run.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

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
    well_prefix_gain,
    well_prefix_len,
)
from container_reshuffle.qubo.model import QuboModel  # noqa: E402
from container_reshuffle.qubo.sampler import QuboSamplerConfig, SimulatedAnnealingQuboSampler  # noqa: E402
from container_reshuffle.scoring import cpmp_business_feasible, dominance_value  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.beam import Candidate  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


SOLVER_NAME = "fast_qubo_integrated_v1"
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
class StepCandidate:
    state: State
    move: Move
    hash_value: tuple
    score: tuple
    legacy_score: tuple
    quality: float
    contaminates_clean_dest: bool
    destination_dirty_after: bool
    tail_ready: bool
    bucket: str = "step"


@dataclass(frozen=True)
class StateFeatures:
    n: int
    inv: int
    bad: int
    dirty: int
    settled: int
    clean_slots: int


@dataclass
class IntegratedRun:
    row: TrajectoryResult
    best_state: State
    business_ok: bool
    goal_validation_ok: bool
    validation_msg: str


@dataclass(frozen=True)
class TailRepairStats:
    expanded: int = 0
    generated: int = 0
    attempts: int = 0
    found: bool = False


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


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


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


def _validate_final(instance: Instance, state: State, runtime_sec: float) -> tuple[bool, bool, str]:
    business_ok, business_msg, simulated = _state_business_ok(instance, state.moves)
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
        status="N_ZERO",
    )
    goal_ok, goal_msg, _ = validate_solution(instance, solution)
    return business_ok, goal_ok, goal_msg


def _is_clean_stack(stack: list[int]) -> bool:
    return well_prefix_len(stack) == len(stack)


def _clean_open_slots(instance: Instance, state: State) -> int:
    return sum(1 for stack in state.stacks if len(stack) < instance.max_height and _is_clean_stack(stack))


def _state_features(instance: Instance, state: State) -> StateFeatures:
    return StateFeatures(
        n=disorder_box_count(instance, state),
        inv=inversion_count(instance, state),
        bad=bad_count(instance, state),
        dirty=dirty_stack_count(instance, state),
        settled=settled_box_count(instance, state),
        clean_slots=0,
    )


def _contaminates_clean_destination(before: State, after: State, move: Move) -> bool:
    before_dst = before.stacks[move.to_stack]
    after_dst = after.stacks[move.to_stack]
    return _is_clean_stack(before_dst) and not _is_clean_stack(after_dst)


def _destination_dirty_after(after: State, move: Move) -> bool:
    return not _is_clean_stack(after.stacks[move.to_stack])


def _best_key(instance: Instance, state: State) -> tuple:
    n = disorder_box_count(instance, state)
    return (
        n,
        state.move_count,
        state.entropy_max,
        state.disorder_total,
        inversion_count(instance, state),
        bad_count(instance, state),
        dirty_stack_count(instance, state),
        -settled_box_count(instance, state),
    )


def _tail_buffer(n: int, dirty: int, bad: int, inv: int, clean_slot_loss: int = 0) -> int:
    if n > 8:
        return 0
    structural = int(dirty > 0) + int(bad > 0) + int(inv > 0) + clean_slot_loss
    return min(4, structural)


def _repair_seed_key(instance: Instance, state: State) -> tuple:
    cap = entropy_limit(instance, 1.3)
    n = disorder_box_count(instance, state)
    remaining = cap - state.move_count
    dirty = dirty_stack_count(instance, state)
    bad = bad_count(instance, state)
    inv = inversion_count(instance, state)
    tail_buffer = _tail_buffer(n, dirty, bad, inv)
    return (
        n,
        max(0, n + tail_buffer - remaining),
        state.entropy_max,
        entropy_value(instance, state),
        dirty,
        bad,
        inv,
        state.move_count,
        -settled_box_count(instance, state),
    )


def _tail_reservoir_signature(instance: Instance, state: State) -> tuple:
    last = state.moves[-1] if state.moves else None
    dirty_shape = tuple(
        sorted(
            (
                len(stack),
                len(stack) - well_prefix_len(stack),
                stack[-1] if stack else 0,
            )
            for stack in state.stacks
        )
    )
    return (
        disorder_box_count(instance, state),
        dirty_stack_count(instance, state),
        last.from_stack if last else -1,
        last.to_stack if last else -1,
        dirty_shape,
    )


def _remember_low_n_states(
    instance: Instance,
    config: SolverConfig,
    reservoir: dict[tuple, State],
    states: list[State],
    *,
    max_n: int,
    max_size: int,
) -> None:
    if max_n <= 0 or max_size <= 0:
        return
    for state in states:
        if disorder_box_count(instance, state) > max_n:
            continue
        h = hash_state(instance, state, config.enable_canonical_hash)
        old = reservoir.get(h)
        if old is None or _best_key(instance, state) < _best_key(instance, old):
            reservoir[h] = state
    if len(reservoir) <= max_size:
        return

    ranked = sorted(reservoir.values(), key=lambda state: (_repair_seed_key(instance, state), _best_key(instance, state)))
    keep: dict[tuple, State] = {}

    def add(state: State) -> None:
        if len(keep) >= max_size:
            return
        keep[hash_state(instance, state, config.enable_canonical_hash)] = state

    for state in ranked[: max(8, max_size // 3)]:
        add(state)
    for state in sorted(reservoir.values(), key=lambda state: _best_key(instance, state))[: max(8, max_size // 3)]:
        add(state)
    seen_signatures: set[tuple] = set()
    for state in ranked:
        sig = _tail_reservoir_signature(instance, state)
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        add(state)
        if len(keep) >= max_size:
            break
    for state in ranked:
        add(state)
        if len(keep) >= max_size:
            break

    reservoir.clear()
    reservoir.update(keep)


def _tail_repair_candidate_states(
    instance: Instance,
    config: SolverConfig,
    beam: list[Candidate],
    reservoir: dict[tuple, State],
    *,
    max_n: int,
    limit: int,
) -> list[State]:
    if limit <= 0:
        return []
    pool: dict[tuple, State] = {}

    def add_state(state: State) -> None:
        if disorder_box_count(instance, state) > max_n:
            return
        h = hash_state(instance, state, config.enable_canonical_hash)
        old = pool.get(h)
        if old is None or _best_key(instance, state) < _best_key(instance, old):
            pool[h] = state

    for cand in beam:
        add_state(cand.state)
    for state in reservoir.values():
        add_state(state)

    ranked = sorted(pool.values(), key=lambda state: (_repair_seed_key(instance, state), _best_key(instance, state)))
    selected: dict[tuple, State] = {}

    def add_rows(rows: list[State]) -> None:
        for state in rows:
            selected.setdefault(hash_state(instance, state, config.enable_canonical_hash), state)
            if len(selected) >= limit:
                return

    add_rows(ranked[: max(4, limit // 3)])
    add_rows(sorted(pool.values(), key=lambda state: _best_key(instance, state))[: max(4, limit // 3)])
    by_signature: dict[tuple, State] = {}
    for state in ranked:
        sig = _tail_reservoir_signature(instance, state)
        old = by_signature.get(sig)
        if old is None or _best_key(instance, state) < _best_key(instance, old):
            by_signature[sig] = state
    add_rows(sorted(by_signature.values(), key=lambda state: _best_key(instance, state)))
    add_rows(ranked)
    out = list(selected.values())
    out.sort(key=lambda state: _best_key(instance, state))
    return out[:limit]


def _repair_budget_for_state(
    instance: Instance,
    state: State,
    *,
    plateau_layers: int,
    per_candidate: int,
    max_expanded: int,
    force: bool,
) -> int:
    n = disorder_box_count(instance, state)
    if n == 0:
        return 0
    cap = entropy_limit(instance, 1.3)
    if state.move_count >= cap:
        return 0
    dirty = dirty_stack_count(instance, state)
    bad = bad_count(instance, state)
    inv = inversion_count(instance, state)
    tail_buffer = _tail_buffer(n, dirty, bad, inv)
    tail_surplus = (cap - state.move_count) - (n + tail_buffer)
    if n <= 3:
        if not force and tail_surplus < 1:
            return 0
        return min(max_expanded, max(per_candidate * 2, per_candidate))
    if n <= 5:
        if force:
            return min(max_expanded, max(500, per_candidate // 2))
        if tail_surplus < 2:
            return 0
        return min(max_expanded, per_candidate)
    if n == 6 and (force or plateau_layers > 0):
        return min(max_expanded, max(500, per_candidate // 2))
    if n <= 8 and force:
        return min(max_expanded, max(250, per_candidate // 4))
    return 0


def _run_multi_start_tail_repair(
    instance: Instance,
    config: SolverConfig,
    states: list[State],
    incumbent: State,
    *,
    deadline: float,
    plateau_layers: int,
    per_candidate: int,
    max_expanded: int,
    force: bool,
) -> tuple[State, TailRepairStats]:
    best = incumbent
    expanded = generated = attempts = 0
    found = False
    seen: set[tuple] = set()
    for state in states:
        if time.perf_counter() >= deadline:
            break
        h = hash_state(instance, state, config.enable_canonical_hash)
        if h in seen:
            continue
        seen.add(h)
        budget = _repair_budget_for_state(
            instance,
            state,
            plateau_layers=plateau_layers,
            per_candidate=per_candidate,
            max_expanded=max_expanded,
            force=force,
        )
        if budget <= 0:
            continue
        attempts += 1
        repaired, exp_count, gen_count = tail_repair_to_n_zero(
            instance,
            state,
            config,
            deadline=deadline,
            max_expanded=budget,
        )
        expanded += exp_count
        generated += gen_count
        if repaired is not None and _best_key(instance, repaired) < _best_key(instance, best):
            best = repaired
            found = True
    return best, TailRepairStats(expanded=expanded, generated=generated, attempts=attempts, found=found)


def _quality_scalar(
    instance: Instance,
    before: State,
    after: State,
    move: Move,
    *,
    cap: int,
    before_features: StateFeatures,
    after_features: StateFeatures,
    plateau_layers: int,
) -> tuple[float, tuple, bool, bool, bool]:
    n_before = before_features.n
    n_after = after_features.n
    remaining = cap - after.move_count
    slack = cap - (after.move_count + n_after)
    low_phase = n_after <= 8
    plateau = plateau_layers > 0

    inv_before = before_features.inv
    inv_after = after_features.inv
    bad_before = before_features.bad
    bad_after = after_features.bad
    dirty_before = before_features.dirty
    dirty_after = after_features.dirty
    settled_before = before_features.settled
    settled_after = after_features.settled
    well_gain = settled_after - settled_before

    contamination = _contaminates_clean_destination(before, after, move)
    dest_dirty = _destination_dirty_after(after, move)
    clean_slot_loss = 1 if contamination else 0
    settled_loss = max(0, settled_before - settled_after)
    buffer_release = 1 if len(before.stacks[move.from_stack]) <= 2 and n_after <= n_before else 0
    source_remaining = len(after.stacks[move.from_stack])
    tail_buffer = _tail_buffer(n_after, dirty_after, bad_after, inv_after, clean_slot_loss)
    clearance_risk = max(0, n_after + tail_buffer - remaining)
    tail_ready = n_after <= 8 and slack >= max(1, min(4, n_after + tail_buffer))
    phase_move = after.move_count if low_phase else 0

    score_tuple = (
        clearance_risk,
        n_after,
        max(0, 2 - slack),
        after.entropy_max,
        -buffer_release,
        source_remaining,
        -well_gain,
        inv_after,
        bad_after,
        dirty_after,
        phase_move,
        1 if contamination and (low_phase or plateau) else 0,
        settled_loss if low_phase or plateau else 0,
        clean_slot_loss,
        max(0, dirty_after - dirty_before),
        max(0, bad_after - bad_before),
        max(0, inv_after - inv_before),
        -well_gain if plateau else 0,
        -int(tail_ready),
        after.move_count,
    )

    scalar = (
        2_000.0 * clearance_risk
        + 500.0 * n_after
        + 16.0 * max(0, 2 - slack)
        + (10.0 if low_phase else 2.0) * after.move_count
        + (18.0 if contamination else 0.0)
        + 10.0 * settled_loss
        + 6.0 * clean_slot_loss
        + 4.0 * max(0, dirty_after - dirty_before)
        + 3.0 * max(0, bad_after - bad_before)
        + 0.5 * max(0, inv_after - inv_before)
        + 2.0 * dirty_after
        + 1.5 * bad_after
        + 0.2 * inv_after
        - 10.0 * well_gain
        - 5.0 * buffer_release
        - 14.0 * int(tail_ready)
    )
    if plateau:
        scalar -= 5.0 * well_gain
        scalar += 5.0 * clean_slot_loss
    return scalar, score_tuple, contamination, dest_dirty, tail_ready


def _legacy_score(
    instance: Instance,
    before: State,
    after: State,
    move: Move,
    *,
    before_features: StateFeatures,
    after_features: StateFeatures,
) -> tuple:
    cap = entropy_limit(instance, 1.3)
    n_before = before_features.n
    n_after = after_features.n
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
        -(after_features.settled - before_features.settled),
        after_features.inv,
        after_features.bad,
        after_features.dirty,
        after.move_count,
    )


def _make_step_candidate(
    instance: Instance,
    before: State,
    after: State,
    move: Move,
    hash_value: tuple,
    *,
    cap: int,
    before_features: StateFeatures,
    after_features: StateFeatures,
    plateau_layers: int,
) -> StepCandidate:
    scalar, score, contamination, dest_dirty, tail_ready = _quality_scalar(
        instance,
        before,
        after,
        move,
        cap=cap,
        before_features=before_features,
        after_features=after_features,
        plateau_layers=plateau_layers,
    )
    return StepCandidate(
        state=after,
        move=move,
        hash_value=hash_value,
        score=score,
        legacy_score=_legacy_score(
            instance,
            before,
            after,
            move,
            before_features=before_features,
            after_features=after_features,
        ),
        quality=scalar,
        contaminates_clean_dest=contamination,
        destination_dirty_after=dest_dirty,
        tail_ready=tail_ready,
    )


def _is_immediate_undo(first: Move, second: Move) -> bool:
    return first.box == second.box and first.from_stack == second.to_stack and first.to_stack == second.from_stack


def _lookahead_state_key(instance: Instance, state: State, root_score: tuple) -> tuple:
    cap = entropy_limit(instance, 1.3)
    n = disorder_box_count(instance, state)
    remaining = cap - state.move_count
    slack = cap - entropy_value(instance, state)
    dirty = dirty_stack_count(instance, state)
    bad = bad_count(instance, state)
    inv = inversion_count(instance, state)
    tail_buffer = _tail_buffer(n, dirty, bad, inv)
    return (
        max(0, n + tail_buffer - remaining),
        n,
        max(0, 2 - slack),
        state.move_count,
        state.entropy_max,
        dirty,
        bad,
        inv,
        -settled_box_count(instance, state),
        root_score,
    )


def _lookahead_key(
    instance: Instance,
    candidate: StepCandidate,
    config: SolverConfig,
    *,
    deadline: float,
    allow_two_step: bool,
    two_step_branch: int,
) -> tuple[tuple, int]:
    best = _lookahead_state_key(instance, candidate.state, candidate.score)
    expanded = 0
    children: list[tuple[tuple, State, Move]] = []

    for move in generate_legal_moves(instance, candidate.state):
        if time.perf_counter() >= deadline:
            break
        if _is_immediate_undo(candidate.move, move):
            continue
        try:
            nxt = apply_move(instance, candidate.state, move)
        except ValueError:
            continue
        if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
            continue
        expanded += 1
        key = _lookahead_state_key(instance, nxt, candidate.score)
        if key < best:
            best = key
        if allow_two_step:
            children.append((key, nxt, move))

    if not allow_two_step:
        return best, expanded

    children.sort(key=lambda row: row[0])
    for _key, child, second_move in children[: max(0, two_step_branch)]:
        for move in generate_legal_moves(instance, child):
            if time.perf_counter() >= deadline:
                break
            if _is_immediate_undo(second_move, move):
                continue
            try:
                nxt = apply_move(instance, child, move)
            except ValueError:
                continue
            if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                continue
            expanded += 1
            key = _lookahead_state_key(instance, nxt, candidate.score)
            if key < best:
                best = key
    return best, expanded


def _select_bucketed_integrated(
    instance: Instance,
    candidates: list[StepCandidate],
    width: int,
    config: SolverConfig,
    *,
    deadline: float,
    lookahead_pool: int,
    lookahead_keep: int,
    two_step_pool: int,
    two_step_branch: int,
    plateau_layers: int,
    legacy_lane_width: int,
    repair_lane_width: int,
    legacy_primary_all: bool,
) -> tuple[list[StepCandidate], int, int]:
    def primary_key(cand: StepCandidate) -> tuple:
        if legacy_primary_all or cand.score[1] <= 8:
            return (cand.legacy_score, cand.score, cand.quality)
        return (cand.score, cand.quality)

    if len(candidates) <= width:
        return sorted(candidates, key=primary_key), 0, 0

    ranked = sorted(candidates, key=primary_key)
    selected: dict[tuple, StepCandidate] = {}

    def add(rows: list[StepCandidate]) -> None:
        for cand in rows:
            if len(selected) >= width:
                return
            old = selected.get(cand.hash_value)
            if old is None or primary_key(cand) < primary_key(old):
                selected[cand.hash_value] = cand

    lookahead_calls = 0
    lookahead_expanded = 0
    if lookahead_pool > 0 and lookahead_keep > 0 and time.perf_counter() < deadline:
        pool: dict[tuple, StepCandidate] = {}

        def seed(rows: list[StepCandidate]) -> None:
            for cand in rows:
                pool.setdefault(cand.hash_value, cand)
                if len(pool) >= lookahead_pool:
                    return

        seed(ranked[:lookahead_pool])
        low_entropy = sorted(candidates, key=lambda c: (c.state.entropy_max, c.state.move_count + c.score[1], c.score))
        seed(low_entropy[: max(1, lookahead_pool // 4)])
        tail_ready = sorted((c for c in candidates if c.tail_ready), key=lambda c: (c.state.move_count, c.score))
        seed(tail_ready[: max(1, lookahead_pool // 5)])

        min_n = min(disorder_box_count(instance, cand.state) for cand in pool.values())
        allow_two = plateau_layers > 0 or min_n <= 8
        scored: list[tuple[tuple, StepCandidate]] = []
        for idx, cand in enumerate(pool.values()):
            if time.perf_counter() >= deadline:
                break
            key, expanded = _lookahead_key(
                instance,
                cand,
                config,
                deadline=deadline,
                allow_two_step=allow_two and idx < two_step_pool,
                two_step_branch=two_step_branch,
            )
            lookahead_calls += 1
            lookahead_expanded += expanded
            scored.append((key, cand))
        scored.sort(key=lambda row: (row[0], primary_key(row[1])))
        add([cand for _key, cand in scored[:lookahead_keep]])

    add(ranked[: max(12, width // 3)])
    if legacy_lane_width > 0:
        legacy_ranked = sorted(candidates, key=lambda c: (c.legacy_score, c.score, c.quality))
        add(legacy_ranked[:legacy_lane_width])
    if repair_lane_width > 0:
        repair_ready = sorted(candidates, key=lambda c: (_repair_seed_key(instance, c.state), c.score, c.quality))
        add(repair_ready[:repair_lane_width])
    low_entropy = sorted(candidates, key=lambda c: (c.state.entropy_max, c.state.move_count + c.score[1], c.score))
    add(low_entropy[: max(6, width // 5)])
    tail_ready = sorted((c for c in candidates if c.tail_ready), key=lambda c: (c.score, c.quality))
    add(tail_ready[: max(6, width // 6)])
    clean_preserving = sorted((c for c in candidates if not c.contaminates_clean_dest), key=lambda c: (c.score, c.quality))
    add(clean_preserving[: max(6, width // 6)])
    buffer_release = [c for c in candidates if len(c.state.stacks[c.move.from_stack]) <= 1]
    add(sorted(buffer_release, key=lambda c: (c.score, c.quality))[: max(6, width // 6)])

    for src, dst in sorted({(c.move.from_stack, c.move.to_stack) for c in candidates}):
        bucket = [c for c in candidates if c.move.from_stack == src and c.move.to_stack == dst]
        add(sorted(bucket, key=lambda c: (c.score, c.quality))[:1])
    for src in sorted({c.move.from_stack for c in candidates}):
        bucket = [c for c in candidates if c.move.from_stack == src]
        add(sorted(bucket, key=lambda c: (c.score, c.quality))[:1])
    for dst in sorted({c.move.to_stack for c in candidates}):
        bucket = [c for c in candidates if c.move.to_stack == dst]
        add(sorted(bucket, key=lambda c: (c.score, c.quality))[:1])

    add(ranked)
    out = list(selected.values())
    out.sort(key=primary_key)
    return out[:width], lookahead_calls, lookahead_expanded


def _usable_supplement_pool(
    candidates: list[StepCandidate],
    *,
    max_vars: int,
    target: int,
) -> list[StepCandidate]:
    ranked = sorted(candidates, key=lambda cand: (cand.quality, cand.score))
    selected: dict[tuple, StepCandidate] = {}

    def add(rows: list[StepCandidate]) -> None:
        for cand in rows:
            old = selected.get(cand.hash_value)
            if old is None or (cand.quality, cand.score) < (old.quality, old.score):
                selected[cand.hash_value] = cand

    add(ranked[: max(target * 2, target + 12)])
    add(sorted((c for c in candidates if c.tail_ready), key=lambda c: (c.quality, c.score))[: max(4, target // 2)])
    add(sorted((c for c in candidates if not c.contaminates_clean_dest), key=lambda c: (c.quality, c.score))[: max(4, target // 2)])
    for src in sorted({cand.move.from_stack for cand in candidates}):
        bucket = [cand for cand in candidates if cand.move.from_stack == src]
        add(sorted(bucket, key=lambda cand: (cand.quality, cand.score))[:1])
    for dst in sorted({cand.move.to_stack for cand in candidates}):
        bucket = [cand for cand in candidates if cand.move.to_stack == dst]
        add(sorted(bucket, key=lambda cand: (cand.quality, cand.score))[:1])
    add(ranked)
    usable = list(selected.values())
    usable.sort(key=lambda cand: (cand.quality, cand.score))
    return usable[:max_vars]


def _pair_penalty(a: StepCandidate, b: StepCandidate, diversity_weight: float) -> float:
    penalty = 0.0
    if a.move.from_stack == b.move.from_stack:
        penalty += 0.35
    if a.move.to_stack == b.move.to_stack:
        penalty += 0.25
    if a.move.box == b.move.box:
        penalty += 0.30
    if a.move.from_stack == b.move.from_stack and a.move.to_stack == b.move.to_stack:
        penalty += 0.70
    if a.contaminates_clean_dest and b.contaminates_clean_dest:
        penalty += 0.15
        if a.move.to_stack == b.move.to_stack:
            penalty += 0.55
    if a.destination_dirty_after and b.destination_dirty_after and a.move.to_stack == b.move.to_stack:
        penalty += 0.20
    return diversity_weight * penalty


def _selection_value(
    usable: list[StepCandidate],
    selected: set[int],
    score_norm: list[float],
    diversity_weight: float,
) -> float:
    value = sum(score_norm[idx] for idx in selected)
    selected_list = sorted(selected)
    for pos, i in enumerate(selected_list):
        for j in selected_list[pos + 1 :]:
            value += _pair_penalty(usable[i], usable[j], diversity_weight)
    return value


def _repair_indices(
    usable: list[StepCandidate],
    selected: set[int],
    target: int,
    score_norm: list[float],
    diversity_weight: float,
) -> set[int]:
    selected = set(selected)
    target = min(target, len(usable))
    while len(selected) > target:
        remove = min(
            selected,
            key=lambda idx: (
                _selection_value(usable, selected - {idx}, score_norm, diversity_weight),
                -score_norm[idx],
                idx,
            ),
        )
        selected.remove(remove)
    while len(selected) < target:
        missing = [idx for idx in range(len(usable)) if idx not in selected]
        if not missing:
            break
        add = min(
            missing,
            key=lambda idx: (
                _selection_value(usable, selected | {idx}, score_norm, diversity_weight),
                score_norm[idx],
                idx,
            ),
        )
        selected.add(add)
    return selected


def _select_qubo_indices(
    usable: list[StepCandidate],
    config: SolverConfig,
    target: int,
) -> tuple[list[StepCandidate], int, float]:
    target = min(target, len(usable))
    if target <= 0:
        return [], 0, 0.0
    if len(usable) <= target:
        return usable, 0, 0.0

    raw_scores = [cand.quality for cand in usable]
    lo, hi = min(raw_scores), max(raw_scores)
    score_norm = [0.0 if hi == lo else (score - lo) / (hi - lo) for score in raw_scores]

    if config.qubo_num_reads <= 0:
        repaired = _repair_indices(usable, set(), target, score_norm, config.diversity_weight)
        return [usable[idx] for idx in sorted(repaired, key=lambda idx: (usable[idx].score, idx))], len(usable), 0.0

    model = QuboModel()
    vars_ = [model.add_binary_var(f"s_{idx}") for idx in range(len(usable))]
    for idx, var in enumerate(vars_):
        model.add_linear(var, score_norm[idx])
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            coeff = _pair_penalty(usable[i], usable[j], config.diversity_weight)
            if coeff:
                model.add_quadratic(vars_[i], vars_[j], coeff)
    model.add_exactly_k_penalty(vars_, target, config.qubo_penalty_a)

    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        QuboSamplerConfig(
            num_reads=config.qubo_num_reads,
            random_seed=config.random_seed,
            time_limit_sec=max(0.001, config.qubo_time_limit_sec or 0.015),
            sa_iterations=max(1, int(config.qubo_sa_iterations)),
        ),
    )

    best: set[int] | None = None
    best_value = float("inf")
    for sample in sample_set.samples or []:
        selected = {idx for idx in range(len(usable)) if sample.binary_values.get(idx, 0) == 1}
        repaired = _repair_indices(usable, selected, target, score_norm, config.diversity_weight)
        value = _selection_value(usable, repaired, score_norm, config.diversity_weight)
        if value < best_value:
            best = repaired
            best_value = value
    if best is None:
        best = _repair_indices(usable, set(), target, score_norm, config.diversity_weight)
    out = [usable[idx] for idx in sorted(best, key=lambda idx: (usable[idx].score, idx))]
    return out, len(model.var_index), sample_set.runtime_sec


def _select_with_qubo_supplement(
    instance: Instance,
    candidates: list[StepCandidate],
    config: SolverConfig,
    *,
    core_width: int,
    supplement_width: int,
    deadline: float,
    lookahead_pool: int,
    lookahead_keep: int,
    two_step_pool: int,
    two_step_branch: int,
    plateau_layers: int,
    legacy_lane_width: int,
    repair_lane_width: int,
    legacy_primary_all: bool,
) -> tuple[list[StepCandidate], int, int, float, int, int]:
    if len(candidates) <= config.beam_width:
        selected = sorted(candidates, key=lambda cand: (cand.score, cand.quality))
        return selected, 0, 0, 0.0, 0, 0

    core_width = max(0, min(core_width, config.beam_width))
    supplement_width = max(0, min(supplement_width, config.beam_width - core_width))
    protected, lh_calls, lh_expanded = _select_bucketed_integrated(
        instance,
        candidates,
        core_width,
        config,
        deadline=deadline,
        lookahead_pool=lookahead_pool,
        lookahead_keep=lookahead_keep,
        two_step_pool=two_step_pool,
        two_step_branch=two_step_branch,
        plateau_layers=plateau_layers,
        legacy_lane_width=legacy_lane_width,
        repair_lane_width=repair_lane_width,
        legacy_primary_all=legacy_primary_all,
    )
    selected_by_hash = {cand.hash_value: cand for cand in protected}
    if not supplement_width or time.perf_counter() >= deadline:
        selected, calls2, expanded2 = _select_bucketed_integrated(
            instance,
            candidates,
            config.beam_width,
            config,
            deadline=deadline,
            lookahead_pool=0,
            lookahead_keep=0,
            two_step_pool=0,
            two_step_branch=0,
            plateau_layers=plateau_layers,
            legacy_lane_width=0,
            repair_lane_width=0,
            legacy_primary_all=False,
        )
        return selected, 0, 0, 0.0, lh_calls + calls2, lh_expanded + expanded2

    supplement_pool = [cand for cand in candidates if cand.hash_value not in selected_by_hash]
    usable = _usable_supplement_pool(
        supplement_pool,
        max_vars=config.max_qubo_vars,
        target=supplement_width,
    )
    if not usable:
        selected = sorted(protected, key=lambda cand: (cand.score, cand.quality))
        return selected, 0, 0, 0.0, lh_calls, lh_expanded

    selected_qubo, qvars, qtime = _select_qubo_indices(usable, config, supplement_width)
    for cand in selected_qubo:
        selected_by_hash.setdefault(cand.hash_value, cand)

    fill, _calls2, _expanded2 = _select_bucketed_integrated(
        instance,
        candidates,
        config.beam_width,
        config,
        deadline=deadline,
        lookahead_pool=0,
        lookahead_keep=0,
        two_step_pool=0,
        two_step_branch=0,
        plateau_layers=plateau_layers,
        legacy_lane_width=0,
        repair_lane_width=0,
        legacy_primary_all=False,
    )
    for cand in fill:
        selected_by_hash.setdefault(cand.hash_value, cand)
        if len(selected_by_hash) >= config.beam_width:
            break

    selected = list(selected_by_hash.values())
    selected.sort(key=lambda cand: (cand.score, cand.quality))
    return selected[: config.beam_width], 1 if qvars else 0, qvars, qtime, lh_calls, lh_expanded


def run_fast_qubo_integrated_v1(
    case: str,
    instance: Instance,
    config: SolverConfig,
    *,
    core_width: int,
    supplement_width: int,
    qubo_min_depth: int,
    qubo_call_interval: int,
    qubo_plateau_only: bool,
    tail_repair_n: int,
    tail_repair_expanded: int,
    tail_repair_candidates: int,
    tail_repair_per_candidate: int,
    low_n_reservoir_n: int,
    low_n_reservoir_size: int,
    tail_mode_n: int,
    tail_mode_width: int,
    tail_mode_plateau_layers: int,
    tail_mode_min_beam: int,
    tail_mode_sticky: bool,
    legacy_lane_width: int,
    repair_lane_width: int,
    legacy_primary_all: bool,
    lookahead_pool: int,
    lookahead_keep: int,
    two_step_pool: int,
    two_step_branch: int,
) -> IntegratedRun:
    start = time.perf_counter()
    deadline = start + config.time_limit_sec
    initial = create_initial_state(instance)
    best = initial
    h0 = hash_state(instance, initial, config.enable_canonical_hash)
    beam = [Candidate(initial, 0.0, None, h0)]
    visited = {h0: dominance_value(instance, initial)}
    cap = entropy_limit(instance, config.entropy_alpha)
    feature_cache: dict[tuple, StateFeatures] = {h0: _state_features(instance, initial)}
    expanded = generated = qcalls = qvars = 0
    qtime = 0.0
    lookahead_calls = lookahead_expanded = 0
    tail_expanded = tail_generated = tail_attempts = tail_mode_layers = 0
    tail_found = False
    plateau_layers = 0
    low_n_reservoir: dict[tuple, State] = {}
    loop_exhausted = False
    sticky_tail_mode = False

    for depth in range(config.max_depth):
        if time.perf_counter() >= deadline:
            break
        best_n_before = disorder_box_count(instance, best)
        step_candidates: list[StepCandidate] = []
        for cand in beam:
            if time.perf_counter() >= deadline:
                break
            expanded += 1
            before_features = feature_cache.get(cand.hash_value)
            if before_features is None:
                before_features = _state_features(instance, cand.state)
                feature_cache[cand.hash_value] = before_features
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
                after_features = feature_cache.get(h)
                if after_features is None:
                    after_features = _state_features(instance, nxt)
                    feature_cache[h] = after_features
                step_candidates.append(
                    _make_step_candidate(
                        instance,
                        cand.state,
                        nxt,
                        move,
                        h,
                        cap=cap,
                        before_features=before_features,
                        after_features=after_features,
                        plateau_layers=plateau_layers,
                    )
                )
        generated += len(step_candidates)
        if not step_candidates:
            loop_exhausted = True
            break
        if low_n_reservoir_n > 0 and low_n_reservoir_size > 0 and best_n_before <= low_n_reservoir_n + 2:
            low_candidates = [cand for cand in step_candidates if disorder_box_count(instance, cand.state) <= low_n_reservoir_n]
            generated_low_n = [
                cand.state
                for cand in sorted(low_candidates, key=lambda cand: (_repair_seed_key(instance, cand.state), cand.score))[
                    : max(8, low_n_reservoir_size // 2)
                ]
            ]
            _remember_low_n_states(
                instance,
                config,
                low_n_reservoir,
                generated_low_n,
                max_n=low_n_reservoir_n,
                max_size=low_n_reservoir_size,
            )

        layer_no = depth + 1
        current_best_n = disorder_box_count(instance, best)
        tail_mode = (
            current_best_n <= tail_mode_n
            or (tail_mode_plateau_layers > 0 and plateau_layers >= tail_mode_plateau_layers)
            or (depth > 0 and tail_mode_min_beam > 0 and len(beam) <= tail_mode_min_beam)
        )
        if tail_mode:
            sticky_tail_mode = True
        if tail_mode_sticky and sticky_tail_mode:
            tail_mode = True
        if tail_mode and disorder_box_count(instance, best) != 0:
            tail_mode_layers += 1
        active_config = replace(config, beam_width=max(config.beam_width, tail_mode_width)) if tail_mode else config
        active_supplement_width = supplement_width
        active_core_width = core_width
        if tail_mode:
            active_supplement_width = min(supplement_width, max(0, active_config.beam_width - 1))
            active_core_width = max(core_width, active_config.beam_width - active_supplement_width)

        use_qubo = (
            layer_no >= qubo_min_depth
            and (layer_no - qubo_min_depth) % max(1, qubo_call_interval) == 0
            and (not qubo_plateau_only or plateau_layers > 0)
            and len(step_candidates) > active_config.beam_width
            and time.perf_counter() < deadline
        )
        if use_qubo:
            selected, calls, vars_count, sample_time, lh_calls, lh_expanded = _select_with_qubo_supplement(
                instance,
                step_candidates,
                active_config,
                core_width=active_core_width,
                supplement_width=active_supplement_width,
                deadline=deadline,
                lookahead_pool=lookahead_pool,
                lookahead_keep=lookahead_keep,
                two_step_pool=two_step_pool,
                two_step_branch=two_step_branch,
                plateau_layers=plateau_layers,
                legacy_lane_width=legacy_lane_width,
                repair_lane_width=repair_lane_width,
                legacy_primary_all=legacy_primary_all,
            )
        else:
            active_width = active_core_width if active_supplement_width > 0 else active_config.beam_width
            selected, lh_calls, lh_expanded = _select_bucketed_integrated(
                instance,
                step_candidates,
                active_width,
                active_config,
                deadline=deadline,
                lookahead_pool=lookahead_pool,
                lookahead_keep=lookahead_keep,
                two_step_pool=two_step_pool,
                two_step_branch=two_step_branch,
                plateau_layers=plateau_layers,
                legacy_lane_width=legacy_lane_width,
                repair_lane_width=repair_lane_width,
                legacy_primary_all=legacy_primary_all,
            )
            calls = vars_count = 0
            sample_time = 0.0
        qcalls += calls
        qvars += vars_count
        qtime += sample_time
        lookahead_calls += lh_calls
        lookahead_expanded += lh_expanded

        next_beam: list[Candidate] = []
        for item in selected:
            visited[item.hash_value] = dominance_value(instance, item.state)
            if _best_key(instance, item.state) < _best_key(instance, best):
                best = item.state
            next_beam.append(Candidate(item.state, item.quality, item.move, item.hash_value))
        beam = next_beam
        _remember_low_n_states(
            instance,
            config,
            low_n_reservoir,
            [cand.state for cand in beam],
            max_n=low_n_reservoir_n,
            max_size=low_n_reservoir_size,
        )

        repair_states = _tail_repair_candidate_states(
            instance,
            config,
            beam,
            low_n_reservoir,
            max_n=tail_repair_n,
            limit=tail_repair_candidates,
        )
        best, stats = _run_multi_start_tail_repair(
            instance,
            config,
            repair_states,
            best,
            deadline=deadline,
            plateau_layers=plateau_layers,
            per_candidate=tail_repair_per_candidate,
            max_expanded=tail_repair_expanded,
            force=False,
        )
        tail_expanded += stats.expanded
        tail_generated += stats.generated
        tail_attempts += stats.attempts
        tail_found = tail_found or stats.found

        best_n_after = disorder_box_count(instance, best)
        plateau_layers = plateau_layers + 1 if best_n_after >= best_n_before else 0

        print(
            "LAYER "
            f"depth={depth + 1} beam={len(beam)} bestN={best_n_after} "
            f"bestMoves={best.move_count} qvars={vars_count} tailMode={tail_mode} "
            f"lookahead={lh_calls}/{lh_expanded} elapsed={time.perf_counter() - start:.3f}s",
            flush=True,
        )

        if best_n_after == 0:
            break

    if disorder_box_count(instance, best) != 0 and time.perf_counter() < deadline:
        final_states = _tail_repair_candidate_states(
            instance,
            config,
            beam,
            low_n_reservoir,
            max_n=max(tail_repair_n, low_n_reservoir_n),
            limit=max(tail_repair_candidates, low_n_reservoir_size),
        )
        best, stats = _run_multi_start_tail_repair(
            instance,
            config,
            final_states,
            best,
            deadline=deadline,
            plateau_layers=max(plateau_layers, 1),
            per_candidate=tail_repair_per_candidate,
            max_expanded=tail_repair_expanded,
            force=True,
        )
        tail_expanded += stats.expanded
        tail_generated += stats.generated
        tail_attempts += stats.attempts
        tail_found = tail_found or stats.found

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += (
        f"; {SOLVER_NAME}; single_pass_fixed_policy"
        f"; beam_width={config.beam_width}; core_width={core_width}; supplement_width={supplement_width}; "
        f"qubo_min_depth={qubo_min_depth}; qubo_call_interval={qubo_call_interval}; "
        f"qubo_plateau_only={qubo_plateau_only}; "
        f"lookahead_calls={lookahead_calls}; lookahead_expanded={lookahead_expanded}; "
        f"tail_mode_layers={tail_mode_layers}; reservoir_size={len(low_n_reservoir)}; "
        f"tail_mode_sticky={tail_mode_sticky}; "
        f"legacy_lane_width={legacy_lane_width}; repair_lane_width={repair_lane_width}; "
        f"legacy_primary_all={legacy_primary_all}; "
        f"tail_attempts={tail_attempts}; tail_expanded={tail_expanded}; "
        f"tail_generated={tail_generated}; tail_found={tail_found}; loop_exhausted={loop_exhausted}"
    )
    row = _result_from_state(
        case,
        SOLVER_NAME,
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
    business_ok, goal_ok, validation_msg = _validate_final(instance, best, row.runtime_sec)
    return IntegratedRun(
        row=row,
        best_state=best,
        business_ok=business_ok,
        goal_validation_ok=goal_ok,
        validation_msg=validation_msg,
    )


def _write_acceptance(runs: list[IntegratedRun], path: Path, moves_dir: Path, exact_refs: dict[str, int]) -> None:
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
                "qubo_used": str(run.row.qubo_calls > 0),
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


def _write_report(runs: list[IntegratedRun], output: Path, acceptance: Path, report: Path, exact_refs: dict[str, int]) -> None:
    solved = sum(1 for run in runs if run.row.final_n == 0)
    avg_runtime = sum(run.row.runtime_sec for run in runs) / len(runs) if runs else 0.0
    gaps = [
        run.row.moves - exact_refs[run.row.case]
        for run in runs
        if run.row.final_n == 0 and run.row.moves is not None and run.row.case in exact_refs
    ]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
    qubo_used = sum(1 for run in runs if run.row.qubo_calls > 0)
    all_valid = all(run.business_ok and run.goal_validation_ok for run in runs)
    lines = [
        "# Fast QUBO Integrated V1",
        "",
        "This is a single-pass fixed algorithm, not a best-of portfolio.",
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
        f"| all selected routes valid | {all_valid} |",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def _build_config(args: argparse.Namespace) -> SolverConfig:
    return SolverConfig(
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        time_limit_sec=args.time_limit,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--acceptance-output", type=Path)
    parser.add_argument("--moves-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--time-limit", type=float, default=12.0)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--core-width", type=int, default=120)
    parser.add_argument("--qubo-supplement-width", type=int, default=8)
    parser.add_argument("--qubo-min-depth", type=int, default=10)
    parser.add_argument("--qubo-call-interval", type=int, default=3)
    parser.add_argument("--qubo-plateau-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-qubo-vars", type=int, default=30)
    parser.add_argument("--qubo-time-limit", type=float, default=0.005)
    parser.add_argument("--qubo-reads", type=int, default=2)
    parser.add_argument("--qubo-iterations", type=int, default=60)
    parser.add_argument("--qubo-penalty-a", type=float, default=30.0)
    parser.add_argument("--diversity-weight", type=float, default=0.08)
    parser.add_argument("--tail-repair-n", type=int, default=5)
    parser.add_argument("--tail-repair-expanded", type=int, default=30000)
    parser.add_argument("--tail-repair-candidates", type=int, default=24)
    parser.add_argument("--tail-repair-per-candidate", type=int, default=6000)
    parser.add_argument("--low-n-reservoir-n", type=int, default=8)
    parser.add_argument("--low-n-reservoir-size", type=int, default=120)
    parser.add_argument("--tail-mode-n", type=int, default=7)
    parser.add_argument("--tail-mode-width", type=int, default=240)
    parser.add_argument("--tail-mode-plateau-layers", type=int, default=1)
    parser.add_argument("--tail-mode-min-beam", type=int, default=12)
    parser.add_argument("--tail-mode-sticky", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--legacy-lane-width", type=int, default=0)
    parser.add_argument("--repair-lane-width", type=int, default=0)
    parser.add_argument("--legacy-primary-all", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lookahead-pool", type=int, default=0)
    parser.add_argument("--lookahead-keep", type=int, default=0)
    parser.add_argument("--two-step-pool", type=int, default=0)
    parser.add_argument("--two-step-branch", type=int, default=0)
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
    config = _build_config(args)
    runs: list[IntegratedRun] = []
    for idx, case in enumerate(_case_filter(args), start=1):
        zone, bay = _parse_case(case)
        instance = _instance_from_single_bay(zone, bay)
        print(f"CASE_START {idx} {case} solver={SOLVER_NAME}", flush=True)
        run = run_fast_qubo_integrated_v1(
            case,
            instance,
            config,
            core_width=args.core_width,
            supplement_width=args.qubo_supplement_width,
            qubo_min_depth=args.qubo_min_depth,
            qubo_call_interval=args.qubo_call_interval,
            qubo_plateau_only=args.qubo_plateau_only,
            tail_repair_n=args.tail_repair_n,
            tail_repair_expanded=args.tail_repair_expanded,
            tail_repair_candidates=args.tail_repair_candidates,
            tail_repair_per_candidate=args.tail_repair_per_candidate,
            low_n_reservoir_n=args.low_n_reservoir_n,
            low_n_reservoir_size=args.low_n_reservoir_size,
            tail_mode_n=args.tail_mode_n,
            tail_mode_width=args.tail_mode_width,
            tail_mode_plateau_layers=args.tail_mode_plateau_layers,
            tail_mode_min_beam=args.tail_mode_min_beam,
            tail_mode_sticky=args.tail_mode_sticky,
            legacy_lane_width=args.legacy_lane_width,
            repair_lane_width=args.repair_lane_width,
            legacy_primary_all=args.legacy_primary_all,
            lookahead_pool=args.lookahead_pool,
            lookahead_keep=args.lookahead_keep,
            two_step_pool=args.two_step_pool,
            two_step_branch=args.two_step_branch,
        )
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
