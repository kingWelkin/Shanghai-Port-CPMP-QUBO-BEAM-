"""Archived 3D integer-position MIP for CPMP.

This formulation is kept for reproducibility and model-comparison tests.  For
the current business objective based on ``N(Future)``, the 4D slot-occupancy
model is the recommended exact benchmark because it expresses adjacent-slot
disorder directly and has a stronger relaxation on the tested cases.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from ..core import Instance, Move, ProblemVariant, Solution
from ..features import disorder_trace, entropy_limit, max_entropy_along_moves, total_disorder_along_moves
from .mip_utils import (
    add_entropy_constraints_from_slots,
    build_start_trace,
    gurobi_unavailable_solution,
    initial_positions,
    positions_from_stacks,
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


def solve_cpmp_3d_integer_position(
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
) -> Solution:
    """Solve CPMP with the archived 3D integer-position Gurobi model."""

    name = "mip_3d"
    if gp is None or GRB is None:
        return gurobi_unavailable_solution(name)
    require_cpmp(instance)
    use_future_disorder_objective = minimize_entropy if minimize_future_disorder is None else minimize_future_disorder
    start = time.perf_counter()
    m, H, n, T = instance.stack_count, instance.max_height, instance.container_count, max_steps
    boxes = range(1, n + 1)
    stacks = range(m)
    times = range(T + 1)
    steps = range(T)
    stack_of, level_of = initial_positions(instance)

    model = gp.Model("cpmp_3d_integer_position")
    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.TimeLimit = float(time_limit_sec)
    model.Params.MIPGap = float(mip_gap)

    a = {(p, i, t): model.addVar(vtype=GRB.BINARY, name=f"a_{p}_{i}_{t}") for p in boxes for i in stacks for t in times}
    lev = {(p, t): model.addVar(vtype=GRB.INTEGER, lb=1, ub=H, name=f"lev_{p}_{t}") for p in boxes for t in times}
    out = {(p, i, t): model.addVar(vtype=GRB.BINARY, name=f"out_{p}_{i}_{t}") for p in boxes for i in stacks for t in steps}
    inn = {(p, i, t): model.addVar(vtype=GRB.BINARY, name=f"in_{p}_{i}_{t}") for p in boxes for i in stacks for t in steps}
    active = {t: model.addVar(vtype=GRB.BINARY, name=f"active_{t}") for t in steps}
    needs_entropy_slots = use_future_disorder_objective or entropy_alpha is not None
    ord_times = [T] if needs_entropy_slots and T > 0 else range(1, T + 1)
    ordv = {
        (p, q, t): model.addVar(vtype=GRB.BINARY, name=f"ord_{p}_{q}_{t}")
        for p in boxes
        for q in range(p + 1, n + 1)
        for t in ord_times
    }
    entropy_slot = None
    entropy_times = range(1, T)
    if needs_entropy_slots:
        levels = range(H)
        entropy_slot = {
            (p, i, h, t): model.addVar(vtype=GRB.BINARY, name=f"entropy_slot_{p}_{i}_{h}_{t}")
            for p in boxes
            for i in stacks
            for h in levels
            for t in entropy_times
        }

    for p in boxes:
        for i in stacks:
            model.addConstr(a[p, i, 0] == (1 if stack_of[p] == i else 0), name=f"init_a_{p}_{i}")
        model.addConstr(lev[p, 0] == level_of[p], name=f"init_lev_{p}")

    def height_expr(stack: int, t: int):
        return gp.quicksum(a[q, stack, t] for q in boxes)

    def pos_expr(box: int, t: int):
        return H * (gp.quicksum((i + 1) * a[box, i, t] for i in stacks) - 1) + lev[box, t]

    for t in times:
        for p in boxes:
            model.addConstr(gp.quicksum(a[p, i, t] for i in stacks) == 1, name=f"unique_stack_{p}_{t}")
        for p in boxes:
            for i in stacks:
                model.addConstr(lev[p, t] <= height_expr(i, t) + H * (1 - a[p, i, t]), name=f"lev_ub_{p}_{i}_{t}")
        if t in ord_times:
            for p in boxes:
                for q in range(p + 1, n + 1):
                    model.addConstr(pos_expr(p, t) - pos_expr(q, t) + m * H * ordv[p, q, t] >= 1, name=f"neq_a_{p}_{q}_{t}")
                    model.addConstr(pos_expr(q, t) - pos_expr(p, t) + m * H * (1 - ordv[p, q, t]) >= 1, name=f"neq_b_{p}_{q}_{t}")
        if entropy_slot is not None and t in entropy_times:
            for p in boxes:
                for i in stacks:
                    model.addConstr(
                        gp.quicksum(entropy_slot[p, i, h, t] for h in levels) == a[p, i, t],
                        name=f"entropy_slot_stack_{p}_{i}_{t}",
                    )
                model.addConstr(
                    lev[p, t] == gp.quicksum((h + 1) * entropy_slot[p, i, h, t] for i in stacks for h in levels),
                    name=f"entropy_slot_level_{p}_{t}",
                )
            for i in stacks:
                for h in levels:
                    model.addConstr(
                        gp.quicksum(entropy_slot[p, i, h, t] for p in boxes) <= 1,
                        name=f"entropy_slot_cap_{i}_{h}_{t}",
                    )
                for h in range(H - 1):
                    model.addConstr(
                        gp.quicksum(entropy_slot[p, i, h + 1, t] for p in boxes)
                        <= gp.quicksum(entropy_slot[p, i, h, t] for p in boxes),
                        name=f"entropy_no_float_{i}_{h}_{t}",
                    )

    for t in steps:
        model.addConstr(gp.quicksum(out[p, i, t] for p in boxes for i in stacks) == active[t], name=f"active_out_def_{t}")
        model.addConstr(gp.quicksum(inn[p, i, t] for p in boxes for i in stacks) == active[t], name=f"active_in_def_{t}")
        model.addConstr(active[t] <= 1, name=f"active_ub_{t}")
        if t < T - 1:
            model.addConstr(active[t] >= active[t + 1], name=f"no_late_idle_{t}")
        for p in boxes:
            moved_p = gp.quicksum(out[p, i, t] for i in stacks)
            model.addConstr(gp.quicksum(out[p, i, t] for i in stacks) == gp.quicksum(inn[p, i, t] for i in stacks), name=f"in_out_match_{p}_{t}")
            model.addConstr(lev[p, t + 1] - lev[p, t] <= H * moved_p, name=f"unmoved_lev_pos_{p}_{t}")
            model.addConstr(lev[p, t] - lev[p, t + 1] <= H * moved_p, name=f"unmoved_lev_neg_{p}_{t}")
            for i in stacks:
                model.addConstr(out[p, i, t] + inn[p, i, t] <= 1, name=f"no_same_{p}_{i}_{t}")
                model.addConstr(out[p, i, t] <= a[p, i, t], name=f"out_stack_{p}_{i}_{t}")
                model.addConstr(lev[p, t] - height_expr(i, t) <= H * (1 - out[p, i, t]), name=f"top_pos_{p}_{i}_{t}")
                model.addConstr(height_expr(i, t) - lev[p, t] <= H * (1 - out[p, i, t]), name=f"top_neg_{p}_{i}_{t}")
                model.addConstr(height_expr(i, t) <= H - 1 + H * (1 - inn[p, i, t]), name=f"dest_not_full_{p}_{i}_{t}")
                model.addConstr(a[p, i, t + 1] == a[p, i, t] - out[p, i, t] + inn[p, i, t], name=f"a_trans_{p}_{i}_{t}")
                model.addConstr(lev[p, t + 1] - (height_expr(i, t) + 1) <= H * (1 - inn[p, i, t]), name=f"in_lev_pos_{p}_{i}_{t}")
                model.addConstr((height_expr(i, t) + 1) - lev[p, t + 1] <= H * (1 - inn[p, i, t]), name=f"in_lev_neg_{p}_{i}_{t}")

    for p in boxes:
        for q in range(p + 1, n + 1):
            for i in stacks:
                model.addConstr(
                    lev[p, T] - lev[q, T] >= 1 - H * (2 - a[p, i, T] - a[q, i, T]),
                    name=f"final_dec_{p}_{q}_{i}",
                )

    move_count_expr = gp.quicksum(active[t] for t in steps)
    if entropy_slot is not None:
        max_entropy, total_disorder_expr = add_entropy_constraints_from_slots(
            model,
            gp,
            GRB,
            instance,
            T,
            boxes,
            stacks,
            range(H),
            times,
            steps,
            entropy_slot,
            active,
            entropy_alpha,
            "entropy3d",
            entropy_times=entropy_times,
        )
    else:
        max_entropy = None
        total_disorder_expr = None
    if use_future_disorder_objective and max_entropy is not None:
        model.setObjective((T + 1) * total_disorder_expr + move_count_expr, GRB.MINIMIZE)
    else:
        model.setObjective(move_count_expr, GRB.MINIMIZE)
    if start_moves:
        _apply_mip_start(instance, T, boxes, stacks, times, steps, start_moves, a, lev, out, inn, active, ordv, entropy_slot)
    model.optimize()
    if model.Status == GRB.INF_OR_UNBD:
        model.Params.DualReductions = 0
        model.optimize()
    if model.Status == GRB.INFEASIBLE:
        write_iis(model, "mip3d_model", diagnostics_dir)

    feasible = model.SolCount > 0
    moves = _extract_moves(T, boxes, stacks, active, out, inn, lev) if feasible else []
    status = _status_name(model.Status)
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
        status,
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


def _extract_moves(T, boxes, stacks, active, out, inn, lev) -> list[Move]:
    moves: list[Move] = []
    for t in range(T):
        if active[t].X < 0.5:
            continue
        src = [(p, i) for p in boxes for i in stacks if out[p, i, t].X > 0.5]
        dst = [(p, i) for p in boxes for i in stacks if inn[p, i, t].X > 0.5]
        if len(src) != 1 or len(dst) != 1 or src[0][0] != dst[0][0]:
            raise RuntimeError(f"invalid MIP move extraction at t={t}: {src}, {dst}")
        p, i = src[0]
        _, j = dst[0]
        moves.append(Move(t, p, i, j, round(lev[p, t].X), round(lev[p, t + 1].X)))
    return moves


def _apply_mip_start(instance, T, boxes, stacks, times, steps, start_moves, a, lev, out, inn, active, ordv, entropy_slot=None) -> None:
    trace = build_start_trace(instance, start_moves, T)
    pos_values: Dict[Tuple[int, int], int] = {}
    for t in times:
        stack_of, level_of = positions_from_stacks(trace[t].stacks)
        for p in boxes:
            for i in stacks:
                a[p, i, t].Start = 1 if stack_of[p] == i else 0
                if entropy_slot is not None:
                    for h in range(instance.max_height):
                        if (p, i, h, t) in entropy_slot:
                            entropy_slot[p, i, h, t].Start = 1 if stack_of[p] == i and level_of[p] == h + 1 else 0
            lev[p, t].Start = level_of[p]
            encoded = instance.max_height * stack_of[p] + level_of[p]
            pos_values[p, t] = encoded
        for p in boxes:
            for q in range(p + 1, instance.container_count + 1):
                if (p, q, t) in ordv:
                    ordv[p, q, t].Start = 1 if pos_values[p, t] < pos_values[q, t] else 0

    for t in steps:
        move = start_moves[t] if t < len(start_moves) else None
        active[t].Start = 1 if move is not None else 0
        for p in boxes:
            for i in stacks:
                out[p, i, t].Start = 1 if move is not None and move.box == p and move.from_stack == i else 0
                inn[p, i, t].Start = 1 if move is not None and move.box == p and move.to_stack == i else 0


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
