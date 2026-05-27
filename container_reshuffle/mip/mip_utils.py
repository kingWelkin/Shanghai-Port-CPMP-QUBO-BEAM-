"""Shared MIP helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core import Instance, Move, ProblemVariant, Solution, State, apply_move, create_initial_state, deep_copy_stacks, validate_solution
from ..features import entropy_limit, future_disorder_count_stacks


def require_cpmp(instance: Instance) -> None:
    """Raise if the instance is not CPMP."""

    if instance.variant is not ProblemVariant.PREMARSHALLING_CPMP:
        raise ValueError("MIP CPMP solvers only support PREMARSHALLING_CPMP")


def gurobi_unavailable_solution(name: str) -> Solution:
    """Return a structured result when gurobipy is unavailable."""

    return Solution(name, False, False, float("inf"), 0.0, [], "GUROBI_NOT_AVAILABLE")


def initial_positions(instance: Instance) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Return box -> stack and box -> one-based level maps."""

    stack_of: Dict[int, int] = {}
    level_of: Dict[int, int] = {}
    for i, stack in enumerate(instance.stacks):
        for level, box in enumerate(stack, start=1):
            stack_of[int(box)] = i
            level_of[int(box)] = level
    return stack_of, level_of


def write_iis(model, prefix: str, diagnostics_dir: str | None) -> None:
    """Write LP and IIS artifacts for an infeasible model."""

    out = Path(diagnostics_dir or "mip_diagnostics")
    out.mkdir(parents=True, exist_ok=True)
    model.write(str(out / f"{prefix}.lp"))
    model.computeIIS()
    model.write(str(out / f"{prefix}.ilp"))


def validate_mip_solution(instance: Instance, solution: Solution) -> Solution:
    """Validate a MIP solution and downgrade status if invalid."""

    if solution.feasible:
        ok, msg, _ = validate_solution(instance, solution)
        if not ok:
            solution.feasible = False
            solution.optimal = False
            solution.status = f"VALIDATION_FAILED: {msg}"
            solution.objective = float("inf")
    return solution


def build_start_trace(instance: Instance, moves: List[Move], max_steps: int) -> List[State]:
    """Build a state trace from a validated CPMP move sequence for MIP starts."""

    if len(moves) > max_steps:
        raise ValueError("MIP start has more moves than max_steps")
    probe = Solution("mip_start", True, False, float(len(moves)), 0.0, moves, "MIP_START")
    ok, msg, _ = validate_solution(instance, probe)
    if not ok:
        raise ValueError(f"invalid MIP start: {msg}")
    state = create_initial_state(instance)
    trace = [state]
    for move in moves:
        state = apply_move(instance, state, move)
        trace.append(state)
    while len(trace) < max_steps + 1:
        trace.append(
            State(
                stacks=deep_copy_stacks(trace[-1].stacks),
                next_retrieval_id=trace[-1].next_retrieval_id,
                move_count=trace[-1].move_count,
                moves=list(trace[-1].moves),
                disorder_current=trace[-1].disorder_current,
                disorder_total=trace[-1].disorder_total,
                entropy_max=trace[-1].entropy_max,
            )
        )
    return trace


def positions_from_stacks(stacks: List[List[int]]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Return box -> stack and box -> one-based level maps for a stack layout."""

    stack_of: Dict[int, int] = {}
    level_of: Dict[int, int] = {}
    for i, stack in enumerate(stacks):
        for level, box in enumerate(stack, start=1):
            stack_of[int(box)] = i
            level_of[int(box)] = level
    return stack_of, level_of


def add_entropy_constraints_from_slots(
    model,
    gp,
    GRB,
    instance: Instance,
    max_steps: int,
    boxes,
    stacks,
    levels,
    times,
    steps,
    slot,
    active,
    entropy_alpha: Optional[float],
    prefix: str,
    entropy_times=None,
) -> Tuple[object, object]:
    """Add business entropy expressions and optional hard alpha constraints.

    Returns ``(max_t G(t), sum_t N(t))``.

    ``slot[p,i,h,t]`` must be binary and equal 1 when box ``p`` is at stack
    ``i``, zero-based level ``h``, after step ``t``.
    """

    n = instance.container_count
    limit = entropy_limit(instance, entropy_alpha) if entropy_alpha is not None else None
    max_entropy = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_steps + n, name=f"{prefix}_max_entropy")
    modeled_times = list(times if entropy_times is None else entropy_times)
    violate = {
        (i, h, t): model.addVar(vtype=GRB.BINARY, name=f"{prefix}_viol_{i}_{h}_{t}")
        for i in stacks
        for h in range(1, instance.max_height)
        for t in modeled_times
    }
    suffix = {
        (i, h, t): model.addVar(vtype=GRB.BINARY, name=f"{prefix}_suffix_{i}_{h}_{t}")
        for i in stacks
        for h in levels
        for t in modeled_times
    }
    bad = {
        (i, h, t): model.addVar(vtype=GRB.BINARY, name=f"{prefix}_bad_{i}_{h}_{t}")
        for i in stacks
        for h in levels
        for t in modeled_times
    }

    total_disorder_terms = []
    disorder_by_time = {}
    for t in modeled_times:
        for i in stacks:
            model.addConstr(suffix[i, 0, t] == 0, name=f"{prefix}_suffix_base_{i}_{t}")
            model.addConstr(bad[i, 0, t] == 0, name=f"{prefix}_bad_base_{i}_{t}")
            for h in range(1, instance.max_height):
                lower_value = gp.quicksum(p * slot[p, i, h - 1, t] for p in boxes)
                upper_value = gp.quicksum(p * slot[p, i, h, t] for p in boxes)
                upper_occ = gp.quicksum(slot[p, i, h, t] for p in boxes)
                diff = upper_value - lower_value
                model.addConstr(violate[i, h, t] <= upper_occ, name=f"{prefix}_viol_occ_{i}_{h}_{t}")
                model.addConstr(diff <= n * violate[i, h, t] + n * (1 - upper_occ), name=f"{prefix}_viol_force_{i}_{h}_{t}")
                model.addConstr(
                    diff >= 1 - n * (1 - violate[i, h, t]) - n * (1 - upper_occ),
                    name=f"{prefix}_viol_exact_{i}_{h}_{t}",
                )
                model.addConstr(suffix[i, h, t] >= suffix[i, h - 1, t], name=f"{prefix}_suffix_prev_{i}_{h}_{t}")
                model.addConstr(suffix[i, h, t] >= violate[i, h, t], name=f"{prefix}_suffix_viol_{i}_{h}_{t}")
                model.addConstr(
                    suffix[i, h, t] <= suffix[i, h - 1, t] + violate[i, h, t],
                    name=f"{prefix}_suffix_upper_{i}_{h}_{t}",
                )
                occ = gp.quicksum(slot[p, i, h, t] for p in boxes)
                model.addConstr(bad[i, h, t] <= occ, name=f"{prefix}_bad_occ_{i}_{h}_{t}")
                model.addConstr(bad[i, h, t] <= suffix[i, h, t], name=f"{prefix}_bad_suffix_{i}_{h}_{t}")
                model.addConstr(bad[i, h, t] >= occ + suffix[i, h, t] - 1, name=f"{prefix}_bad_lower_{i}_{h}_{t}")
        disorder_expr = gp.quicksum(bad[i, h, t] for i in stacks for h in levels)
        disorder_by_time[t] = disorder_expr
        total_disorder_terms.append(disorder_expr)
        entropy_expr = gp.quicksum(active[s] for s in range(min(t, max_steps))) + disorder_expr
        model.addConstr(entropy_expr <= max_entropy, name=f"{prefix}_max_{t}")
        if limit is not None:
            model.addConstr(entropy_expr <= limit, name=f"{prefix}_limit_{t}")
    initial_disorder = int(future_disorder_count_stacks(instance.stacks))
    total_active = gp.quicksum(active[s] for s in steps)
    model.addConstr(total_active >= initial_disorder, name=f"{prefix}_move_lb_initial_disorder")
    for t in steps:
        current_disorder = initial_disorder if t == 0 else disorder_by_time.get(t)
        next_disorder = 0 if t + 1 == max_steps else disorder_by_time.get(t + 1)
        if current_disorder is not None and next_disorder is not None:
            model.addConstr(
                next_disorder >= current_disorder - active[t],
                name=f"{prefix}_disorder_drop_lb_{t}",
            )
    if limit is not None:
        model.addConstr(total_active <= limit, name=f"{prefix}_final_limit")
    return max_entropy, gp.quicksum(total_disorder_terms)
