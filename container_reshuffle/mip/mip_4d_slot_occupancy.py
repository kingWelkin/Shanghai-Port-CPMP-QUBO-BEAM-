"""4D slot-occupancy MIP for CPMP."""

from __future__ import annotations

import time
from typing import Dict, Optional

from ..core import Instance, Move, Solution
from ..features import disorder_trace, entropy_limit, max_entropy_along_moves, total_disorder_along_moves
from .mip_utils import (
    add_entropy_constraints_from_slots,
    build_start_trace,
    gurobi_unavailable_solution,
    require_cpmp,
    validate_mip_solution,
    write_iis,
)

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover
    gp = None
    GRB = None


def solve_cpmp_4d_slot_occupancy(
    instance: Instance,
    max_steps: int,
    time_limit_sec: float = 3600,
    mip_gap: float = 0.0,
    output_flag: bool = False,
    diagnostics_dir: Optional[str] = None,
    start_moves: Optional[list[Move]] = None,
    entropy_alpha: Optional[float] = 1.3,
    minimize_entropy: bool = True,
    minimize_future_disorder: Optional[bool] = None,
    target_stack: Optional[Dict[int, int]] = None,
) -> Solution:
    """Solve CPMP with the explicit slot-occupancy Gurobi model."""

    name = "mip_4d"
    if gp is None or GRB is None:
        return gurobi_unavailable_solution(name)
    require_cpmp(instance)
    use_future_disorder_objective = minimize_entropy if minimize_future_disorder is None else minimize_future_disorder
    start = time.perf_counter()
    m, H, n, T = instance.stack_count, instance.max_height, instance.container_count, max_steps
    boxes = range(1, n + 1)
    stacks = range(m)
    levels = range(H)
    times = range(T + 1)
    steps = range(T)

    model = gp.Model("cpmp_4d_slot_occupancy")
    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.TimeLimit = float(time_limit_sec)
    model.Params.MIPGap = float(mip_gap)

    x = {(p, i, h, t): model.addVar(vtype=GRB.BINARY, name=f"x_{p}_{i}_{h}_{t}") for p in boxes for i in stacks for h in levels for t in times}
    out = {(p, i, h, t): model.addVar(vtype=GRB.BINARY, name=f"out_{p}_{i}_{h}_{t}") for p in boxes for i in stacks for h in levels for t in steps}
    inn = {(p, i, h, t): model.addVar(vtype=GRB.BINARY, name=f"in_{p}_{i}_{h}_{t}") for p in boxes for i in stacks for h in levels for t in steps}
    z = {(p, t): model.addVar(vtype=GRB.BINARY, name=f"z_{p}_{t}") for p in boxes for t in steps}
    active = {t: model.addVar(vtype=GRB.BINARY, name=f"active_{t}") for t in steps}

    for p in boxes:
        for i in stacks:
            for h in levels:
                init = 1 if h < len(instance.stacks[i]) and instance.stacks[i][h] == p else 0
                model.addConstr(x[p, i, h, 0] == init, name=f"init_{p}_{i}_{h}")

    for t in times:
        for p in boxes:
            model.addConstr(gp.quicksum(x[p, i, h, t] for i in stacks for h in levels) == 1, name=f"unique_pos_{p}_{t}")
        for i in stacks:
            for h in levels:
                model.addConstr(gp.quicksum(x[p, i, h, t] for p in boxes) <= 1, name=f"slot_cap_{i}_{h}_{t}")
            for h in range(H - 1):
                model.addConstr(
                    gp.quicksum(x[p, i, h + 1, t] for p in boxes) <= gp.quicksum(x[p, i, h, t] for p in boxes),
                    name=f"no_float_{i}_{h}_{t}",
                )

    for t in steps:
        model.addConstr(gp.quicksum(z[p, t] for p in boxes) == active[t], name=f"active_def_{t}")
        model.addConstr(active[t] <= 1, name=f"active_ub_{t}")
        if t < T - 1:
            model.addConstr(active[t] >= active[t + 1], name=f"no_late_idle_{t}")
        for p in boxes:
            model.addConstr(gp.quicksum(out[p, i, h, t] for i in stacks for h in levels) == z[p, t], name=f"out_def_{p}_{t}")
            model.addConstr(gp.quicksum(inn[p, i, h, t] for i in stacks for h in levels) == z[p, t], name=f"in_def_{p}_{t}")
            for i in stacks:
                model.addConstr(
                    gp.quicksum(out[p, i, h, t] for h in levels) + gp.quicksum(inn[p, i, h, t] for h in levels) <= 1,
                    name=f"no_same_{p}_{i}_{t}",
                )
                for h in levels:
                    model.addConstr(out[p, i, h, t] <= x[p, i, h, t], name=f"out_match_{p}_{i}_{h}_{t}")
                    if h < H - 1:
                        model.addConstr(
                            out[p, i, h, t] + gp.quicksum(x[q, i, h + 1, t] for q in boxes) <= 1,
                            name=f"top_only_{p}_{i}_{h}_{t}",
                        )
                    model.addConstr(
                        inn[p, i, h, t] + gp.quicksum(x[q, i, h, t] for q in boxes) <= 1,
                        name=f"in_empty_{p}_{i}_{h}_{t}",
                    )
                    if h >= 1:
                        model.addConstr(
                            inn[p, i, h, t] <= gp.quicksum(x[q, i, h - 1, t] for q in boxes),
                            name=f"in_supported_{p}_{i}_{h}_{t}",
                        )
                    model.addConstr(
                        x[p, i, h, t + 1] == x[p, i, h, t] - out[p, i, h, t] + inn[p, i, h, t],
                        name=f"transition_{p}_{i}_{h}_{t}",
                    )

    for i in stacks:
        for h in range(H - 1):
            for p in boxes:
                for q in range(p + 1, n + 1):
                    model.addConstr(x[p, i, h, T] + x[q, i, h + 1, T] <= 1, name=f"final_{p}_{q}_{i}_{h}")
    if target_stack is not None:
        for p, i in target_stack.items():
            if p not in boxes or i not in stacks:
                raise ValueError(f"invalid target assignment: box {p} -> stack {i}")
            model.addConstr(gp.quicksum(x[p, i, h, T] for h in levels) == 1, name=f"target_stack_{p}_{i}")

    max_entropy, total_disorder_expr = add_entropy_constraints_from_slots(
        model,
        gp,
        GRB,
        instance,
        T,
        boxes,
        stacks,
        levels,
        times,
        steps,
        x,
        active,
        entropy_alpha,
        "entropy4d",
    )
    move_count_expr = gp.quicksum(active[t] for t in steps)
    if use_future_disorder_objective:
        model.setObjective((T + 1) * total_disorder_expr + move_count_expr, GRB.MINIMIZE)
    else:
        model.setObjective(move_count_expr, GRB.MINIMIZE)
    if start_moves:
        _apply_mip_start(instance, T, boxes, stacks, levels, times, steps, start_moves, x, out, inn, z, active)
    model.optimize()
    if model.Status == GRB.INF_OR_UNBD:
        model.Params.DualReductions = 0
        model.optimize()
    if model.Status == GRB.INFEASIBLE:
        write_iis(model, "mip4d_model", diagnostics_dir)

    feasible = model.SolCount > 0
    moves = _extract_moves(T, boxes, stacks, levels, active, out, inn) if feasible else []
    entropy_max = float(max_entropy_along_moves(instance, moves)) if feasible else None
    entropy_cap = float(entropy_limit(instance, entropy_alpha)) if entropy_alpha is not None else None
    disorder_values = disorder_trace(instance, moves) if feasible else []
    sol = Solution(
        name,
        feasible,
        model.Status == GRB.OPTIMAL,
        float(len(moves)) if feasible else float("inf"),
        time.perf_counter() - start,
        moves,
        _status_name(model.Status),
        mip_best_bound=float(model.ObjBound) if feasible or model.Status in {GRB.TIME_LIMIT, GRB.OPTIMAL} else None,
        mip_gap=float(model.MIPGap) if feasible else None,
        mip_node_count=float(model.NodeCount),
        mip_var_count=int(model.NumVars),
        mip_constr_count=int(model.NumConstrs),
        entropy_max=entropy_max,
        entropy_limit=entropy_cap,
        entropy_alpha=float(entropy_alpha) if entropy_alpha is not None else None,
        disorder_total=float(total_disorder_along_moves(instance, moves)) if feasible else None,
        disorder_initial=float(disorder_values[0]) if feasible else None,
        disorder_final=float(disorder_values[-1]) if feasible else None,
    )
    return validate_mip_solution(instance, sol)


def _extract_moves(T, boxes, stacks, levels, active, out, inn) -> list[Move]:
    moves: list[Move] = []
    for t in range(T):
        if active[t].X < 0.5:
            continue
        src = [(p, i, h) for p in boxes for i in stacks for h in levels if out[p, i, h, t].X > 0.5]
        dst = [(p, i, h) for p in boxes for i in stacks for h in levels if inn[p, i, h, t].X > 0.5]
        if len(src) != 1 or len(dst) != 1 or src[0][0] != dst[0][0]:
            raise RuntimeError(f"invalid MIP move extraction at t={t}: {src}, {dst}")
        p, i, h = src[0]
        _, j, e = dst[0]
        moves.append(Move(t, p, i, j, h + 1, e + 1))
    return moves


def _apply_mip_start(instance, T, boxes, stacks, levels, times, steps, start_moves, x, out, inn, z, active) -> None:
    trace = build_start_trace(instance, start_moves, T)
    for t in times:
        for p in boxes:
            for i in stacks:
                for h in levels:
                    x[p, i, h, t].Start = 1 if h < len(trace[t].stacks[i]) and trace[t].stacks[i][h] == p else 0
    for t in steps:
        move = start_moves[t] if t < len(start_moves) else None
        active[t].Start = 1 if move is not None else 0
        for p in boxes:
            z[p, t].Start = 1 if move is not None and move.box == p else 0
            for i in stacks:
                for h in levels:
                    out[p, i, h, t].Start = (
                        1
                        if move is not None
                        and move.box == p
                        and move.from_stack == i
                        and move.from_level == h + 1
                        else 0
                    )
                    inn[p, i, h, t].Start = (
                        1
                        if move is not None
                        and move.box == p
                        and move.to_stack == i
                        and move.to_level == h + 1
                        else 0
                    )


def _status_name(status: int) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(status, str(status))
