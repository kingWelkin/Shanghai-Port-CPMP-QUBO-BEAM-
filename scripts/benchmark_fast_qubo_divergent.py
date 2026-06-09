"""Divergent strict-clear Fast/QUBO single-algorithm experiments.

Each solver in this file is a fixed policy, not a runtime portfolio:

* fast_qubo_strict_clear_v2 keeps the known fast QUBO pass and adds a
  protected-core QUBO rescue beam.
* fast_qubo_tail_repair_v1 switches from global search to QUBO-ranked tail
  repair once the fast pass stalls at low N.
* fast_qubo_tail_repair_v4 uses a fixed depth checkpoint to decide whether the
  current layout should switch into QUBO quality tail repair or continue the
  protected fast beam.
* fast_qubo_lns_v1 tries QUBO-guided local perturbations before the same fixed
  strict-clear rescue safety path.
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam import _write_explain_csv  # noqa: E402
from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from benchmark_fast_qubo_adaptive_rescue_v1 import (  # noqa: E402
    run_adaptive,
    run_qubo_seeded_rescue_beam,
)
from benchmark_fast_qubo_integrated_v1 import (  # noqa: E402
    ACCEPTANCE_FIELDS,
    _exact_moves_by_case,
    _move_gap,
    _parse_case,
    _rel,
    _validate_final,
)
from benchmark_fast_slack_beam import StepCandidate, _best_key, _score, _select_bucketed  # noqa: E402
from benchmark_fast_slack_qubo_beam import _select_qubo_indices, run_fast_slack_qubo_beam  # noqa: E402
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
from container_reshuffle.features import dirty_stack_count, disorder_box_count, entropy_limit, inversion_count  # noqa: E402
from container_reshuffle.scoring import cpmp_business_feasible  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


STRICT_SOLVER = "fast_qubo_strict_clear_v2"
TAIL_SOLVER = "fast_qubo_tail_repair_v1"
TAIL_V2_SOLVER = "fast_qubo_tail_repair_v2"
TAIL_V3_SOLVER = "fast_qubo_tail_repair_v3"
TAIL_V4_SOLVER = "fast_qubo_tail_repair_v4"
LNS_SOLVER = "fast_qubo_lns_v1"
FROZEN_SOLVER = TAIL_V4_SOLVER
SOLVERS = (STRICT_SOLVER, TAIL_SOLVER, TAIL_V2_SOLVER, TAIL_V3_SOLVER, TAIL_V4_SOLVER, LNS_SOLVER)
RESULT_FIELDS = list(TrajectoryResult.__annotations__.keys())


@dataclass
class DivergentRun:
    row: TrajectoryResult
    best_state: State
    business_ok: bool
    goal_validation_ok: bool
    validation_msg: str


@dataclass
class LocalRepairResult:
    best_state: State
    qcalls: int = 0
    qvars: int = 0
    qtime: float = 0.0
    expanded: int = 0
    generated: int = 0
    attempts: int = 0
    found: bool = False
    status: str = ""


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


def _strict_adaptive_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        max_depth=args.max_depth,
        fast_time_limit=args.fast_time_limit,
        fast_beam_width=122,
        fast_core_width=120,
        fast_qubo_supplement_width=2,
        fast_qubo_min_depth=10,
        fast_qubo_call_interval=3,
        fast_qubo_plateau_only=False,
        fast_tail_repair_n=5,
        fast_tail_repair_expanded=30000,
        rescue_time_limit=args.rescue_time_limit,
        rescue_beam_width=256,
        rescue_qubo_seed_width=0,
        rescue_core_width=240,
        rescue_qubo_supplement_width=16,
        rescue_qubo_min_depth=6,
        rescue_qubo_call_interval=4,
        rescue_qubo_plateau_only=False,
        rescue_tail_repair_n=3,
        rescue_tail_repair_expanded=30000,
        rescue_tail_repair_candidates=2,
        rescue_repair_n=8,
        rescue_repair_candidates=80,
        rescue_reservoir_size=120,
        max_qubo_vars=args.max_qubo_vars,
        qubo_time_limit=args.qubo_time_limit,
        qubo_reads=args.qubo_reads,
        qubo_iterations=args.qubo_iterations,
        qubo_penalty_a=args.qubo_penalty_a,
        diversity_weight=args.diversity_weight,
        seed=args.seed,
    )


def _local_candidates(instance: Instance, state: State, config: SolverConfig, bucket: str) -> list[StepCandidate]:
    out: list[StepCandidate] = []
    for move in generate_legal_moves(instance, state):
        try:
            nxt = apply_move(instance, state, move)
        except ValueError:
            continue
        if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
            continue
        out.append(
            StepCandidate(
                nxt,
                move,
                hash_state(instance, nxt, config.enable_canonical_hash),
                _score(instance, state, nxt, move),
                bucket,
            )
        )
    return out


def _two_step_lns_candidates(
    instance: Instance,
    state: State,
    config: SolverConfig,
    *,
    first_width: int,
    second_width: int,
    max_candidates: int,
) -> list[StepCandidate]:
    first = _select_bucketed(instance, _local_candidates(instance, state, config, "lns_first"), first_width)
    out: dict[tuple, StepCandidate] = {}
    for first_cand in first:
        second = _select_bucketed(
            instance,
            _local_candidates(instance, first_cand.state, config, "lns_second"),
            second_width,
        )
        for cand in second:
            if first_cand.move.box == cand.move.box and first_cand.move.from_stack == cand.move.to_stack:
                continue
            old = out.get(cand.hash_value)
            if old is None or cand.score < old.score:
                out[cand.hash_value] = cand
            if len(out) >= max_candidates:
                break
        if len(out) >= max_candidates:
            break
    selected = list(out.values())
    selected.sort(key=lambda cand: cand.score)
    return selected


def _qubo_seed_states(
    instance: Instance,
    candidates: list[StepCandidate],
    config: SolverConfig,
    *,
    deterministic_width: int,
    qubo_width: int,
) -> tuple[list[State], int, int, float]:
    if not candidates:
        return [], 0, 0, 0.0
    deterministic = _select_bucketed(instance, candidates, deterministic_width)
    usable = sorted(candidates, key=lambda cand: cand.score)[: config.max_qubo_vars]
    qubo_selected, qvars, qtime = _select_qubo_indices(usable, config, min(qubo_width, len(usable)))
    qcalls = 1 if qvars else 0
    states: dict[tuple, State] = {}
    for cand in qubo_selected + deterministic:
        states.setdefault(cand.hash_value, cand.state)
    return list(states.values()), qcalls, qvars, qtime


def _tail_quality_rank(instance: Instance, state: State) -> tuple:
    n_value = disorder_box_count(instance, state)
    return (
        state.move_count + n_value,
        n_value,
        state.move_count,
        state.entropy_max,
        _best_key(instance, state),
    )


def _tail_goal_first_quality_rank(instance: Instance, state: State) -> tuple:
    n_value = disorder_box_count(instance, state)
    return (
        n_value,
        state.move_count + n_value,
        state.move_count,
        state.entropy_max,
        _best_key(instance, state),
    )


def _tail_repair_to_n_zero_quality(
    instance: Instance,
    start_state: State,
    config: SolverConfig,
    *,
    deadline: float,
    max_expanded: int,
    goal_first: bool = False,
) -> tuple[State | None, int, int]:
    """Tail repair that keeps searching for fewer-move N=0 routes within budget."""

    if disorder_box_count(instance, start_state) == 0:
        return start_state, 0, 0
    cap = entropy_limit(instance, config.entropy_alpha)
    if start_state.move_count >= cap:
        return None, 0, 0

    start_hash = hash_state(instance, start_state, config.enable_canonical_hash)
    visited = {start_hash: start_state.move_count}
    heap: list[tuple[tuple, int, State]] = []
    counter = 0
    rank_fn = _tail_goal_first_quality_rank if goal_first else _tail_quality_rank
    heapq.heappush(heap, (rank_fn(instance, start_state), counter, start_state))
    expanded = generated = 0
    best_goal: State | None = None

    while heap and expanded < max_expanded and time.perf_counter() < deadline:
        _rank, _idx, state = heapq.heappop(heap)
        if best_goal is not None and state.move_count >= best_goal.move_count:
            continue
        n_value = disorder_box_count(instance, state)
        if n_value == 0:
            if best_goal is None or state.move_count < best_goal.move_count:
                best_goal = state
            continue
        if state.move_count >= cap:
            continue
        expanded += 1
        for move in generate_legal_moves(instance, state):
            try:
                nxt = apply_move(instance, state, move)
            except ValueError:
                continue
            if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                continue
            if best_goal is not None and nxt.move_count >= best_goal.move_count:
                continue
            generated += 1
            h = hash_state(instance, nxt, config.enable_canonical_hash)
            if visited.get(h, math.inf) <= nxt.move_count:
                continue
            visited[h] = nxt.move_count
            counter += 1
            heapq.heappush(heap, (rank_fn(instance, nxt), counter, nxt))
    return best_goal, expanded, generated


def _repair_from_seed_states(
    instance: Instance,
    start_state: State,
    seed_states: list[State],
    config: SolverConfig,
    *,
    deadline: float,
    max_expanded_per_seed: int,
    quality_tail: bool = False,
    quality_seed_incumbent: bool = False,
    quality_goal_first: bool = False,
) -> LocalRepairResult:
    best = start_state
    best_goal: State | None = None
    expanded = generated = attempts = 0
    for state in [start_state] + seed_states:
        if time.perf_counter() >= deadline:
            break
        attempts += 1
        if _best_key(instance, state) < _best_key(instance, best):
            best = state
        if quality_tail:
            if quality_seed_incumbent:
                seed_budget = max(1000, max_expanded_per_seed // 4)
                repaired_seed, exp_seed, gen_seed = tail_repair_to_n_zero(
                    instance, state, config, deadline=deadline, max_expanded=seed_budget
                )
                if repaired_seed is not None and (best_goal is None or repaired_seed.move_count < best_goal.move_count):
                    best_goal = repaired_seed
                    best = repaired_seed
                remaining_budget = max(1000, max_expanded_per_seed - exp_seed)
            else:
                exp_seed = gen_seed = 0
                remaining_budget = max_expanded_per_seed
            repaired, exp_quality, gen_quality = _tail_repair_to_n_zero_quality(
                instance,
                state,
                config,
                deadline=deadline,
                max_expanded=remaining_budget,
                goal_first=quality_goal_first,
            )
            exp_count = exp_seed + exp_quality
            gen_count = gen_seed + gen_quality
        else:
            repaired, exp_count, gen_count = tail_repair_to_n_zero(
                instance, state, config, deadline=deadline, max_expanded=max_expanded_per_seed
            )
        expanded += exp_count
        generated += gen_count
        if repaired is not None:
            if best_goal is None or repaired.move_count < best_goal.move_count:
                best_goal = repaired
                best = repaired
            if quality_tail:
                continue
            return LocalRepairResult(
                best_state=repaired,
                expanded=expanded,
                generated=generated,
                attempts=attempts,
                found=True,
                status="local_tail_repair_found",
            )
    if best_goal is not None:
        return LocalRepairResult(
            best_state=best_goal,
            expanded=expanded,
            generated=generated,
            attempts=attempts,
            found=True,
            status="local_tail_quality_found",
        )
    return LocalRepairResult(
        best_state=best,
        expanded=expanded,
        generated=generated,
        attempts=attempts,
        found=False,
        status="local_tail_repair_exhausted",
    )


def _run_qubo_local_tail_repair(
    case: str,
    instance: Instance,
    start_state: State,
    config: SolverConfig,
    *,
    local_time_limit: float,
    deterministic_width: int,
    qubo_width: int,
    max_expanded_per_seed: int,
    quality_tail: bool = False,
    quality_seed_incumbent: bool = False,
    quality_goal_first: bool = False,
) -> LocalRepairResult:
    del case
    deadline = time.perf_counter() + local_time_limit
    candidates = _local_candidates(instance, start_state, config, "tail_qubo")
    seed_states, qcalls, qvars, qtime = _qubo_seed_states(
        instance,
        candidates,
        config,
        deterministic_width=deterministic_width,
        qubo_width=qubo_width,
    )
    result = _repair_from_seed_states(
        instance,
        start_state,
        seed_states,
        config,
        deadline=deadline,
        max_expanded_per_seed=max_expanded_per_seed,
        quality_tail=quality_tail,
        quality_seed_incumbent=quality_seed_incumbent,
        quality_goal_first=quality_goal_first,
    )
    result.qcalls += qcalls
    result.qvars += qvars
    result.qtime += qtime
    result.status += f"; qubo_tail_candidates={len(candidates)}; qubo_seed_states={len(seed_states)}"
    return result


def _run_qubo_lns_repair(
    case: str,
    instance: Instance,
    start_state: State,
    config: SolverConfig,
    *,
    local_time_limit: float,
    deterministic_width: int,
    qubo_width: int,
    max_expanded_per_seed: int,
) -> LocalRepairResult:
    del case
    deadline = time.perf_counter() + local_time_limit
    candidates = _two_step_lns_candidates(
        instance,
        start_state,
        config,
        first_width=24,
        second_width=8,
        max_candidates=120,
    )
    seed_states, qcalls, qvars, qtime = _qubo_seed_states(
        instance,
        candidates,
        config,
        deterministic_width=deterministic_width,
        qubo_width=qubo_width,
    )
    result = _repair_from_seed_states(
        instance,
        start_state,
        seed_states,
        config,
        deadline=deadline,
        max_expanded_per_seed=max_expanded_per_seed,
    )
    result.qcalls += qcalls
    result.qvars += qvars
    result.qtime += qtime
    result.status += f"; lns_candidates={len(candidates)}; qubo_seed_states={len(seed_states)}"
    return result


def _run_rescue_stage(
    case: str,
    instance: Instance,
    args: argparse.Namespace,
) -> tuple[TrajectoryResult, State]:
    rescue_args = _strict_adaptive_args(args)
    rescue_config = _build_config(
        beam_width=rescue_args.rescue_beam_width,
        time_limit=rescue_args.rescue_time_limit,
        max_depth=rescue_args.max_depth,
        args=args,
    )
    return run_qubo_seeded_rescue_beam(
        case,
        instance,
        rescue_config,
        seed_width=rescue_args.rescue_qubo_seed_width,
        core_width=rescue_args.rescue_core_width,
        supplement_width=rescue_args.rescue_qubo_supplement_width,
        qubo_min_depth=rescue_args.rescue_qubo_min_depth,
        qubo_call_interval=rescue_args.rescue_qubo_call_interval,
        qubo_plateau_only=rescue_args.rescue_qubo_plateau_only,
        tail_repair_n=rescue_args.rescue_tail_repair_n,
        tail_repair_expanded=rescue_args.rescue_tail_repair_expanded,
        tail_repair_candidates=rescue_args.rescue_tail_repair_candidates,
        rescue_repair_n=rescue_args.rescue_repair_n,
        rescue_repair_candidates=rescue_args.rescue_repair_candidates,
        rescue_reservoir_size=rescue_args.rescue_reservoir_size,
    )


def _run_core_only_rescue_stage(
    case: str,
    instance: Instance,
    args: argparse.Namespace,
) -> tuple[TrajectoryResult, State]:
    rescue_args = _strict_adaptive_args(args)
    rescue_config = _build_config(
        beam_width=240,
        time_limit=rescue_args.rescue_time_limit,
        max_depth=rescue_args.max_depth,
        args=args,
    )
    return run_qubo_seeded_rescue_beam(
        case,
        instance,
        rescue_config,
        seed_width=0,
        core_width=240,
        supplement_width=0,
        qubo_min_depth=rescue_args.rescue_qubo_min_depth,
        qubo_call_interval=rescue_args.rescue_qubo_call_interval,
        qubo_plateau_only=rescue_args.rescue_qubo_plateau_only,
        tail_repair_n=rescue_args.rescue_tail_repair_n,
        tail_repair_expanded=rescue_args.rescue_tail_repair_expanded,
        tail_repair_candidates=rescue_args.rescue_tail_repair_candidates,
        rescue_repair_n=rescue_args.rescue_repair_n,
        rescue_repair_candidates=rescue_args.rescue_repair_candidates,
        rescue_reservoir_size=rescue_args.rescue_reservoir_size,
    )


def _run_rescue_safety_stages(
    case: str,
    instance: Instance,
    args: argparse.Namespace,
) -> tuple[list[TrajectoryResult], State, bool]:
    qubo_row, qubo_state = _run_rescue_stage(case, instance, args)
    rows = [qubo_row]
    best = qubo_state
    core_shadow_used = False
    if disorder_box_count(instance, best) != 0:
        core_row, core_state = _run_core_only_rescue_stage(case, instance, args)
        rows.append(core_row)
        core_shadow_used = True
        if _best_key(instance, core_state) < _best_key(instance, best):
            best = core_state
    return rows, best, core_shadow_used


def _run_fast_stage(case: str, instance: Instance, args: argparse.Namespace) -> tuple[TrajectoryResult, State]:
    config = _build_config(
        beam_width=122,
        time_limit=args.fast_time_limit,
        max_depth=args.max_depth,
        args=args,
    )
    return run_fast_slack_qubo_beam(
        case,
        instance,
        config,
        core_width=120,
        supplement_width=2,
        qubo_min_depth=10,
        qubo_call_interval=3,
        qubo_plateau_only=False,
        tail_repair_n=5,
        tail_repair_expanded=30000,
    )


def _run_fast_checkpoint_stage(case: str, instance: Instance, args: argparse.Namespace) -> tuple[TrajectoryResult, State]:
    config = _build_config(
        beam_width=122,
        time_limit=args.fast_time_limit,
        max_depth=args.tail_switch_depth,
        args=args,
    )
    return run_fast_slack_qubo_beam(
        case,
        instance,
        config,
        core_width=120,
        supplement_width=2,
        qubo_min_depth=10,
        qubo_call_interval=3,
        qubo_plateau_only=False,
        tail_repair_n=5,
        tail_repair_expanded=30000,
    )


def _should_switch_checkpoint_to_quality_tail(instance: Instance, state: State) -> bool:
    n_value = disorder_box_count(instance, state)
    if n_value == 0:
        return False
    low_height_stacks = sum(1 for stack in state.stacks if len(stack) <= 1)
    inversions = inversion_count(instance, state)
    dirty = dirty_stack_count(instance, state)
    if n_value <= 5:
        return True
    if n_value == 6:
        return not (low_height_stacks == 0 and dirty >= 3 and inversions >= 9)
    if n_value == 7:
        return low_height_stacks == 0 and inversions <= 6
    return False


def run_strict_clear_v2(case: str, instance: Instance, args: argparse.Namespace) -> DivergentRun:
    total_start = time.perf_counter()
    run = run_adaptive(case, instance, _strict_adaptive_args(args))
    stage_rows = [run.row]
    best = run.best_state
    core_shadow_used = False
    if disorder_box_count(instance, best) != 0:
        core_row, core_state = _run_core_only_rescue_stage(case, instance, args)
        stage_rows.append(core_row)
        core_shadow_used = True
        if _best_key(instance, core_state) < _best_key(instance, best):
            best = core_state

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += (
        f"; {STRICT_SOLVER}; fixed_policy; route=strict_clear_v2; "
        "rescue_core_protected=True; rescue_qubo_supplement=16; "
        f"core_shadow_used={core_shadow_used}; "
        + " -> ".join(
            f"{row.solver}(N={row.final_n},moves={row.moves},t={row.runtime_sec:.3f},q={row.qubo_calls})"
            for row in stage_rows
        )
    )
    row = _result_from_state(
        case,
        STRICT_SOLVER,
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
    business_ok, goal_ok, validation_msg = _validate_final(instance, best, row.runtime_sec)
    return DivergentRun(row, best, business_ok, goal_ok, validation_msg)


def run_tail_repair_v1(case: str, instance: Instance, args: argparse.Namespace) -> DivergentRun:
    total_start = time.perf_counter()
    fast_row, fast_state = _run_fast_stage(case, instance, args)
    stage_rows = [fast_row]
    best = fast_state
    local = LocalRepairResult(best_state=best, status="local_tail_skipped")
    fallback_used = False

    if disorder_box_count(instance, best) != 0:
        local_config = _build_config(beam_width=64, time_limit=args.local_time_limit, max_depth=args.max_depth, args=args)
        local = _run_qubo_local_tail_repair(
            case,
            instance,
            best,
            local_config,
            local_time_limit=args.local_time_limit,
            deterministic_width=12,
            qubo_width=12,
            max_expanded_per_seed=args.local_repair_expanded,
        )
        best = local.best_state

    if disorder_box_count(instance, best) != 0:
        rescue_rows, rescue_state, core_shadow_used = _run_rescue_safety_stages(case, instance, args)
        stage_rows.extend(rescue_rows)
        best = rescue_state
        fallback_used = True
    else:
        core_shadow_used = False

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += (
        f"; {TAIL_SOLVER}; fixed_policy; fast_stage=N{fast_row.final_n}; "
        f"qubo_stage=local_tail_repair; local_status={local.status}; "
        f"local_attempts={local.attempts}; fallback_rescue={fallback_used}; "
        f"core_shadow_used={core_shadow_used}; "
        + " -> ".join(
            f"{row.solver}(N={row.final_n},moves={row.moves},t={row.runtime_sec:.3f},q={row.qubo_calls})"
            for row in stage_rows
        )
    )
    row = _result_from_state(
        case,
        TAIL_SOLVER,
        instance,
        best,
        total_start,
        expanded_states=sum(item.expanded_states for item in stage_rows) + local.expanded,
        generated_states=sum(item.generated_states for item in stage_rows) + local.generated,
        visited_states=sum(item.visited_states for item in stage_rows),
        qubo_calls=sum(item.qubo_calls for item in stage_rows) + local.qcalls,
        qubo_total_vars=sum(item.qubo_total_vars for item in stage_rows) + local.qvars,
        qubo_time_sec=sum(item.qubo_time_sec for item in stage_rows) + local.qtime,
        status=status,
    )
    business_ok, goal_ok, validation_msg = _validate_final(instance, best, row.runtime_sec)
    return DivergentRun(row, best, business_ok, goal_ok, validation_msg)


def run_tail_repair_v2(case: str, instance: Instance, args: argparse.Namespace) -> DivergentRun:
    total_start = time.perf_counter()
    fast_row, fast_state = _run_fast_stage(case, instance, args)
    stage_rows = [fast_row]
    best = fast_state
    local = LocalRepairResult(best_state=best, status="local_tail_quality_skipped")
    fallback_used = False

    if disorder_box_count(instance, best) != 0:
        local_config = _build_config(beam_width=80, time_limit=args.local_time_limit, max_depth=args.max_depth, args=args)
        local = _run_qubo_local_tail_repair(
            case,
            instance,
            best,
            local_config,
            local_time_limit=args.local_time_limit,
            deterministic_width=16,
            qubo_width=16,
            max_expanded_per_seed=args.local_repair_expanded,
            quality_tail=True,
            quality_seed_incumbent=args.tail_quality_seed_incumbent,
        )
        best = local.best_state

    if disorder_box_count(instance, best) != 0:
        rescue_rows, rescue_state, core_shadow_used = _run_rescue_safety_stages(case, instance, args)
        stage_rows.extend(rescue_rows)
        best = rescue_state
        fallback_used = True
    else:
        core_shadow_used = False

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += (
        f"; {TAIL_V2_SOLVER}; fixed_policy; fast_stage=N{fast_row.final_n}; "
        f"qubo_stage=local_tail_quality_repair; local_status={local.status}; "
        f"local_attempts={local.attempts}; fallback_rescue={fallback_used}; "
        f"core_shadow_used={core_shadow_used}; "
        + " -> ".join(
            f"{row.solver}(N={row.final_n},moves={row.moves},t={row.runtime_sec:.3f},q={row.qubo_calls})"
            for row in stage_rows
        )
    )
    row = _result_from_state(
        case,
        TAIL_V2_SOLVER,
        instance,
        best,
        total_start,
        expanded_states=sum(item.expanded_states for item in stage_rows) + local.expanded,
        generated_states=sum(item.generated_states for item in stage_rows) + local.generated,
        visited_states=sum(item.visited_states for item in stage_rows),
        qubo_calls=sum(item.qubo_calls for item in stage_rows) + local.qcalls,
        qubo_total_vars=sum(item.qubo_total_vars for item in stage_rows) + local.qvars,
        qubo_time_sec=sum(item.qubo_time_sec for item in stage_rows) + local.qtime,
        status=status,
    )
    business_ok, goal_ok, validation_msg = _validate_final(instance, best, row.runtime_sec)
    return DivergentRun(row, best, business_ok, goal_ok, validation_msg)


def run_tail_repair_v3(case: str, instance: Instance, args: argparse.Namespace) -> DivergentRun:
    total_start = time.perf_counter()
    fast_row, fast_state = _run_fast_stage(case, instance, args)
    stage_rows = [fast_row]
    best = fast_state
    local = LocalRepairResult(best_state=best, status="local_tail_goal_first_quality_skipped")
    fallback_used = False

    if disorder_box_count(instance, best) != 0:
        local_config = _build_config(beam_width=80, time_limit=args.local_time_limit, max_depth=args.max_depth, args=args)
        local = _run_qubo_local_tail_repair(
            case,
            instance,
            best,
            local_config,
            local_time_limit=args.local_time_limit,
            deterministic_width=16,
            qubo_width=16,
            max_expanded_per_seed=args.local_repair_expanded,
            quality_tail=True,
            quality_seed_incumbent=args.tail_quality_seed_incumbent,
            quality_goal_first=True,
        )
        best = local.best_state

    if disorder_box_count(instance, best) != 0:
        rescue_rows, rescue_state, core_shadow_used = _run_rescue_safety_stages(case, instance, args)
        stage_rows.extend(rescue_rows)
        best = rescue_state
        fallback_used = True
    else:
        core_shadow_used = False

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += (
        f"; {TAIL_V3_SOLVER}; fixed_policy; fast_stage=N{fast_row.final_n}; "
        f"qubo_stage=local_tail_goal_first_quality_repair; local_status={local.status}; "
        f"local_attempts={local.attempts}; fallback_rescue={fallback_used}; "
        f"core_shadow_used={core_shadow_used}; "
        + " -> ".join(
            f"{row.solver}(N={row.final_n},moves={row.moves},t={row.runtime_sec:.3f},q={row.qubo_calls})"
            for row in stage_rows
        )
    )
    row = _result_from_state(
        case,
        TAIL_V3_SOLVER,
        instance,
        best,
        total_start,
        expanded_states=sum(item.expanded_states for item in stage_rows) + local.expanded,
        generated_states=sum(item.generated_states for item in stage_rows) + local.generated,
        visited_states=sum(item.visited_states for item in stage_rows),
        qubo_calls=sum(item.qubo_calls for item in stage_rows) + local.qcalls,
        qubo_total_vars=sum(item.qubo_total_vars for item in stage_rows) + local.qvars,
        qubo_time_sec=sum(item.qubo_time_sec for item in stage_rows) + local.qtime,
        status=status,
    )
    business_ok, goal_ok, validation_msg = _validate_final(instance, best, row.runtime_sec)
    return DivergentRun(row, best, business_ok, goal_ok, validation_msg)


def run_tail_repair_v4(case: str, instance: Instance, args: argparse.Namespace) -> DivergentRun:
    total_start = time.perf_counter()
    checkpoint_row, checkpoint_state = _run_fast_checkpoint_stage(case, instance, args)
    stage_rows = [checkpoint_row]
    best = checkpoint_state
    local = LocalRepairResult(best_state=best, status="checkpoint_tail_skipped")
    fallback_used = False
    continued_fast = False
    switch_to_tail = _should_switch_checkpoint_to_quality_tail(instance, checkpoint_state)

    if disorder_box_count(instance, best) != 0 and switch_to_tail:
        local_config = _build_config(beam_width=80, time_limit=args.local_time_limit, max_depth=args.max_depth, args=args)
        local = _run_qubo_local_tail_repair(
            case,
            instance,
            best,
            local_config,
            local_time_limit=args.local_time_limit,
            deterministic_width=16,
            qubo_width=16,
            max_expanded_per_seed=args.local_repair_expanded,
            quality_tail=True,
            quality_seed_incumbent=args.tail_quality_seed_incumbent,
        )
        best = local.best_state

    if disorder_box_count(instance, best) != 0 and not switch_to_tail:
        fast_row, fast_state = _run_fast_stage(case, instance, args)
        stage_rows.append(fast_row)
        best = fast_state
        continued_fast = True

    if disorder_box_count(instance, best) != 0:
        rescue_rows, rescue_state, core_shadow_used = _run_rescue_safety_stages(case, instance, args)
        stage_rows.extend(rescue_rows)
        best = rescue_state
        fallback_used = True
    else:
        core_shadow_used = False

    checkpoint_n = disorder_box_count(instance, checkpoint_state)
    checkpoint_low_height = sum(1 for stack in checkpoint_state.stacks if len(stack) <= 1)
    checkpoint_inv = inversion_count(instance, checkpoint_state)
    checkpoint_dirty = dirty_stack_count(instance, checkpoint_state)
    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += (
        f"; {TAIL_V4_SOLVER}; fixed_policy; checkpoint_depth={args.tail_switch_depth}; "
        f"checkpoint=N{checkpoint_n}/M{checkpoint_state.move_count}/dirty{checkpoint_dirty}/"
        f"inv{checkpoint_inv}/lowheight{checkpoint_low_height}; "
        f"switch_to_qubo_tail={switch_to_tail}; continued_fast={continued_fast}; "
        f"qubo_stage=checkpoint_quality_tail; local_status={local.status}; "
        f"local_attempts={local.attempts}; fallback_rescue={fallback_used}; "
        f"core_shadow_used={core_shadow_used}; "
        + " -> ".join(
            f"{row.solver}(N={row.final_n},moves={row.moves},t={row.runtime_sec:.3f},q={row.qubo_calls})"
            for row in stage_rows
        )
    )
    row = _result_from_state(
        case,
        TAIL_V4_SOLVER,
        instance,
        best,
        total_start,
        expanded_states=sum(item.expanded_states for item in stage_rows) + local.expanded,
        generated_states=sum(item.generated_states for item in stage_rows) + local.generated,
        visited_states=sum(item.visited_states for item in stage_rows),
        qubo_calls=sum(item.qubo_calls for item in stage_rows) + local.qcalls,
        qubo_total_vars=sum(item.qubo_total_vars for item in stage_rows) + local.qvars,
        qubo_time_sec=sum(item.qubo_time_sec for item in stage_rows) + local.qtime,
        status=status,
    )
    business_ok, goal_ok, validation_msg = _validate_final(instance, best, row.runtime_sec)
    return DivergentRun(row, best, business_ok, goal_ok, validation_msg)


def run_lns_v1(case: str, instance: Instance, args: argparse.Namespace) -> DivergentRun:
    total_start = time.perf_counter()
    fast_row, fast_state = _run_fast_stage(case, instance, args)
    stage_rows = [fast_row]
    best = fast_state
    local = LocalRepairResult(best_state=best, status="lns_skipped")
    fallback_used = False

    if disorder_box_count(instance, best) != 0:
        local_config = _build_config(beam_width=96, time_limit=args.local_time_limit, max_depth=args.max_depth, args=args)
        local = _run_qubo_lns_repair(
            case,
            instance,
            best,
            local_config,
            local_time_limit=args.local_time_limit,
            deterministic_width=12,
            qubo_width=16,
            max_expanded_per_seed=args.local_repair_expanded,
        )
        best = local.best_state

    if disorder_box_count(instance, best) != 0:
        rescue_rows, rescue_state, core_shadow_used = _run_rescue_safety_stages(case, instance, args)
        stage_rows.extend(rescue_rows)
        best = rescue_state
        fallback_used = True
    else:
        core_shadow_used = False

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += (
        f"; {LNS_SOLVER}; fixed_policy; fast_stage=N{fast_row.final_n}; "
        f"qubo_stage=lns_perturb_repair; lns_status={local.status}; "
        f"local_attempts={local.attempts}; fallback_rescue={fallback_used}; "
        f"core_shadow_used={core_shadow_used}; "
        + " -> ".join(
            f"{row.solver}(N={row.final_n},moves={row.moves},t={row.runtime_sec:.3f},q={row.qubo_calls})"
            for row in stage_rows
        )
    )
    row = _result_from_state(
        case,
        LNS_SOLVER,
        instance,
        best,
        total_start,
        expanded_states=sum(item.expanded_states for item in stage_rows) + local.expanded,
        generated_states=sum(item.generated_states for item in stage_rows) + local.generated,
        visited_states=sum(item.visited_states for item in stage_rows),
        qubo_calls=sum(item.qubo_calls for item in stage_rows) + local.qcalls,
        qubo_total_vars=sum(item.qubo_total_vars for item in stage_rows) + local.qvars,
        qubo_time_sec=sum(item.qubo_time_sec for item in stage_rows) + local.qtime,
        status=status,
    )
    business_ok, goal_ok, validation_msg = _validate_final(instance, best, row.runtime_sec)
    return DivergentRun(row, best, business_ok, goal_ok, validation_msg)


def _run_one_solver(solver: str, case: str, instance: Instance, args: argparse.Namespace) -> DivergentRun:
    if solver == STRICT_SOLVER:
        return run_strict_clear_v2(case, instance, args)
    if solver == TAIL_SOLVER:
        return run_tail_repair_v1(case, instance, args)
    if solver == TAIL_V2_SOLVER:
        return run_tail_repair_v2(case, instance, args)
    if solver == TAIL_V3_SOLVER:
        return run_tail_repair_v3(case, instance, args)
    if solver == TAIL_V4_SOLVER:
        return run_tail_repair_v4(case, instance, args)
    if solver == LNS_SOLVER:
        return run_lns_v1(case, instance, args)
    raise ValueError(f"unknown solver: {solver}")


def _write_rows(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_acceptance(runs: list[DivergentRun], path: Path, moves_dir: Path, exact_refs: dict[str, int]) -> None:
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


def _write_report(runs: list[DivergentRun], output: Path, acceptance: Path, report: Path, exact_refs: dict[str, int]) -> None:
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
        f"# {runs[0].row.solver if runs else 'fast_qubo_divergent'}",
        "",
        "Fixed Fast/QUBO divergent strict-clear experiment.",
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
        f"| QUBO executed | {qubo_used}/{len(runs)} |",
        f"| all selected routes valid | {all_valid} |",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def _output_paths(args: argparse.Namespace, solver: str, stamp: str) -> tuple[Path, Path, Path, Path]:
    if args.output and args.solver != "all":
        output = args.output
    else:
        output = args.output_dir / f"fast_qubo_divergent_{solver}_{stamp}.csv"
    if not output.is_absolute():
        output = ROOT / output
    acceptance = output.with_name(output.stem + "_acceptance.csv")
    moves_dir = output.with_name(output.stem + "_moves")
    report = output.with_suffix(".md")
    return output, acceptance, moves_dir, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=[*SOLVERS, "all"], default=FROZEN_SOLVER)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "experiments")
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--fast-time-limit", type=float, default=8.0)
    parser.add_argument("--tail-switch-depth", type=int, default=23)
    parser.add_argument("--local-time-limit", type=float, default=6.0)
    parser.add_argument("--local-repair-expanded", type=int, default=16000)
    parser.add_argument("--tail-quality-seed-incumbent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rescue-time-limit", type=float, default=20.0)
    parser.add_argument("--max-qubo-vars", type=int, default=60)
    parser.add_argument("--qubo-time-limit", type=float, default=0.01)
    parser.add_argument("--qubo-reads", type=int, default=2)
    parser.add_argument("--qubo-iterations", type=int, default=80)
    parser.add_argument("--qubo-penalty-a", type=float, default=30.0)
    parser.add_argument("--diversity-weight", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir

    exact_refs = _exact_moves_by_case()
    cases = _case_filter(args)
    solvers = SOLVERS if args.solver == "all" else (args.solver,)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for solver in solvers:
        output, acceptance, moves_dir, report = _output_paths(args, solver, stamp)
        runs: list[DivergentRun] = []
        for idx, case in enumerate(cases, start=1):
            zone, bay = _parse_case(case)
            instance = _instance_from_single_bay(zone, bay)
            print(f"CASE_START {idx} {case} solver={solver}", flush=True)
            run = _run_one_solver(solver, case, instance, args)
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
