"""Solve one single-bay business objective exactly with Gurobi.

Objective: minimize final N(Future), then move count, subject to every
intermediate G = moves_so_far + N(Future) satisfying the business cap.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from container_reshuffle.core import Move, Solution, State, apply_move, validate_instance  # noqa: E402
from container_reshuffle.features import disorder_trace, entropy_limit, entropy_trace, future_disorder_count_stacks  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover
    gp = None
    GRB = None


def _add_disorder(model, boxes, stacks, levels, times, slot, n: int, H: int, prefix: str):
    violate = {
        (i, h, t): model.addVar(vtype=GRB.BINARY, name=f"{prefix}_viol_{i}_{h}_{t}")
        for i in stacks
        for h in range(1, H)
        for t in times
    }
    suffix = {
        (i, h, t): model.addVar(vtype=GRB.BINARY, name=f"{prefix}_suffix_{i}_{h}_{t}")
        for i in stacks
        for h in levels
        for t in times
    }
    bad = {
        (i, h, t): model.addVar(vtype=GRB.BINARY, name=f"{prefix}_bad_{i}_{h}_{t}")
        for i in stacks
        for h in levels
        for t in times
    }
    disorder = {}
    for t in times:
        for i in stacks:
            model.addConstr(suffix[i, 0, t] == 0, name=f"{prefix}_suffix_base_{i}_{t}")
            model.addConstr(bad[i, 0, t] == 0, name=f"{prefix}_bad_base_{i}_{t}")
            for h in range(1, H):
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
        disorder[t] = gp.quicksum(bad[i, h, t] for i in stacks for h in levels)
    return disorder


def _extract_moves(T, boxes, stacks, levels, active, out, inn) -> list[Move]:
    moves: list[Move] = []
    for t in range(T):
        if active[t].X < 0.5:
            continue
        src = [(p, i, h) for p in boxes for i in stacks for h in levels if out[p, i, h, t].X > 0.5]
        dst = [(p, i, h) for p in boxes for i in stacks for h in levels if inn[p, i, h, t].X > 0.5]
        if len(src) != 1 or len(dst) != 1 or src[0][0] != dst[0][0]:
            raise RuntimeError(f"invalid move extraction at step {t}: {src}, {dst}")
        p, i, h = src[0]
        _, j, e = dst[0]
        moves.append(Move(t, p, i, j, h + 1, e + 1))
    return moves


def solve(
    zone: str,
    bay: str,
    alpha: float,
    time_limit: float,
    output_flag: bool,
    target_final_n: int | None,
) -> int:
    if gp is None or GRB is None:
        print("GUROBI_NOT_AVAILABLE", flush=True)
        return 2

    instance = _instance_from_single_bay(zone, bay)
    validate_instance(instance)
    cap = entropy_limit(instance, alpha)
    initial_n = future_disorder_count_stacks(instance.stacks)
    m, H, n, T = instance.stack_count, instance.max_height, instance.container_count, cap
    boxes = range(1, n + 1)
    stacks = range(m)
    levels = range(H)
    times = range(T + 1)
    steps = range(T)
    start = time.perf_counter()

    print(
        f"START case={instance.instance_id} stacks={m} H={H} containers={n} initial_N={initial_n} cap={cap}",
        flush=True,
    )

    model = gp.Model("business_final_n_single_bay")
    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = 0.0

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
            model.addConstr(gp.quicksum(x[p, i, h, t] for i in stacks for h in levels) == 1, name=f"unique_{p}_{t}")
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
                    model.addConstr(x[p, i, h, t + 1] == x[p, i, h, t] - out[p, i, h, t] + inn[p, i, h, t], name=f"trans_{p}_{i}_{h}_{t}")

    disorder = _add_disorder(model, boxes, stacks, levels, times, x, n, H, "business")
    for t in times:
        moves_so_far = gp.quicksum(active[s] for s in range(min(t, T)))
        model.addConstr(moves_so_far + disorder[t] <= cap, name=f"g_cap_{t}")

    move_count = gp.quicksum(active[t] for t in steps)
    if target_final_n is None:
        model.setObjective((T + 1) * disorder[T] + move_count, GRB.MINIMIZE)
    else:
        model.addConstr(disorder[T] <= int(target_final_n), name="target_final_n")
        model.setObjective(move_count, GRB.MINIMIZE)
    model.optimize()

    status_name = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
    }.get(model.Status, str(model.Status))

    feasible = model.SolCount > 0
    moves = _extract_moves(T, boxes, stacks, levels, active, out, inn) if feasible else []
    state = State(stacks=[list(stack) for stack in instance.stacks])
    for move in moves:
        state = apply_move(instance, state, move)
    final_n = future_disorder_count_stacks(state.stacks) if feasible else None
    entropy_values = entropy_trace(instance, moves) if feasible else []
    disorder_values = disorder_trace(instance, moves) if feasible else []
    ok = feasible and max(entropy_values) <= cap
    obj_bound = float(model.ObjBound) if model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT} or feasible else None
    proven_final_bound = None if obj_bound is None else int(obj_bound // (T + 1))
    print(
        "FINAL "
        f"status={status_name} feasible={feasible} optimal={model.Status == GRB.OPTIMAL} "
        f"target_final_N={target_final_n} "
        f"final_N={final_n} moves={len(moves) if feasible else None} "
        f"max_G={max(entropy_values) if entropy_values else None} ok_cap={ok} "
        f"obj={float(model.ObjVal) if feasible else None} bound={obj_bound} final_N_bound={proven_final_bound} "
        f"gap={float(model.MIPGap) if feasible else None} nodes={float(model.NodeCount)} "
        f"runtime={time.perf_counter() - start:.3f}s",
        flush=True,
    )
    if feasible:
        print("DISORDER_TRACE " + ",".join(str(v) for v in disorder_values), flush=True)
        print("MOVES " + "; ".join(f"{mv.step}:{mv.box} {mv.from_stack}->{mv.to_stack}" for mv in moves), flush=True)
        sol = Solution("business_mip", True, model.Status == GRB.OPTIMAL, float(len(moves)), 0.0, moves, "BUSINESS")
        _ = sol
    return 0 if model.Status == GRB.OPTIMAL else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="C1")
    parser.add_argument("--bay", default="26")
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--time-limit", type=float, default=1_000_000_000.0)
    parser.add_argument("--output-flag", action="store_true")
    parser.add_argument("--target-final-n", type=int)
    args = parser.parse_args()
    return solve(args.zone, args.bay, args.alpha, args.time_limit, args.output_flag, args.target_final_n)


if __name__ == "__main__":
    raise SystemExit(main())
