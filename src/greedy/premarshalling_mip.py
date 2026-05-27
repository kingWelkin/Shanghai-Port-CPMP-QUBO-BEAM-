"""Exact Gurobi MIP models for container pre-marshalling.

The public API keeps the names from the original implementation request:
``Solve3DIntegerPositionMip`` and ``Solve4DSlotMip``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:  # pragma: no cover - exercised only without Gurobi.
    gp = None
    GRB = None
    _GUROBI_IMPORT_ERROR = exc
else:
    _GUROBI_IMPORT_ERROR = None


@dataclass
class PremarshallingInstance:
    StackCount: int
    MaxHeight: int
    ContainerCount: int
    Stacks: List[List[int]]


@dataclass
class Move:
    Step: int
    Box: int
    FromStack: int
    ToStack: int
    FromLevel: int
    ToLevel: int


@dataclass
class PremarshallingSolution:
    Feasible: bool
    Optimal: bool
    Objective: float
    Moves: List[Move] = field(default_factory=list)
    RuntimeSec: float = 0.0
    NodeCount: int = 0
    Status: str = ""


def Solve3DIntegerPositionMip(
    instance: PremarshallingInstance,
    maxSteps: int,
    timeLimitSec: Optional[float] = None,
    diagnosticsDir: Optional[str] = None,
) -> PremarshallingSolution:
    """Solve the integer-position MIP model."""
    _require_gurobi()
    _validate_instance(instance, maxSteps)

    model_data = _build_3d_model(instance, maxSteps, timeLimitSec)
    model = model_data["model"]
    status = _optimize_with_diagnostics(model, "premarshalling_3d", diagnosticsDir)
    return _solution_from_model(
        model,
        status,
        lambda: _extract_3d_moves(model_data, instance, maxSteps),
        instance,
    )


def Solve4DSlotMip(
    instance: PremarshallingInstance,
    maxSteps: int,
    timeLimitSec: Optional[float] = None,
    diagnosticsDir: Optional[str] = None,
) -> PremarshallingSolution:
    """Solve the explicit slot-occupancy MIP model."""
    _require_gurobi()
    _validate_instance(instance, maxSteps)

    model_data = _build_4d_model(instance, maxSteps, timeLimitSec)
    model = model_data["model"]
    status = _optimize_with_diagnostics(model, "premarshalling_4d", diagnosticsDir)
    return _solution_from_model(
        model,
        status,
        lambda: _extract_4d_moves(model_data, maxSteps),
        instance,
    )


def validate_move_sequence(
    instance: PremarshallingInstance,
    moves: Iterable[Move],
    *,
    expected_objective: Optional[float] = None,
) -> List[List[int]]:
    """Simulate and validate a move sequence.

    Returns the final stacks when valid; raises ``ValueError`` otherwise.
    """
    _validate_instance(instance, maxSteps=0)
    ordered = sorted(list(moves), key=lambda mv: mv.Step)
    stacks = [list(st) for st in instance.Stacks]

    if expected_objective is not None and len(ordered) != int(round(expected_objective)):
        raise ValueError(
            f"extracted move count {len(ordered)} != objective {expected_objective}"
        )

    last_step = 0
    for mv in ordered:
        if mv.Step <= last_step:
            raise ValueError(f"move steps must be strictly increasing: {mv.Step}")
        last_step = mv.Step
        if not (1 <= mv.FromStack <= instance.StackCount):
            raise ValueError(f"source stack out of range: {mv.FromStack}")
        if not (1 <= mv.ToStack <= instance.StackCount):
            raise ValueError(f"destination stack out of range: {mv.ToStack}")
        if mv.FromStack == mv.ToStack:
            raise ValueError("source and destination stacks must differ")

        src = stacks[mv.FromStack - 1]
        dst = stacks[mv.ToStack - 1]
        if not src:
            raise ValueError(f"source stack {mv.FromStack} is empty")
        if len(dst) >= instance.MaxHeight:
            raise ValueError(f"destination stack {mv.ToStack} is full")
        if mv.FromLevel != len(src):
            raise ValueError(
                f"from level mismatch at step {mv.Step}: "
                f"{mv.FromLevel} != {len(src)}"
            )
        if mv.ToLevel != len(dst) + 1:
            raise ValueError(
                f"to level mismatch at step {mv.Step}: "
                f"{mv.ToLevel} != {len(dst) + 1}"
            )
        if src[-1] != mv.Box:
            raise ValueError(
                f"moved box is not source top at step {mv.Step}: "
                f"{mv.Box} != {src[-1]}"
            )
        dst.append(src.pop())

    if not _all_stacks_decreasing(stacks):
        raise ValueError(f"final stacks are not bottom->top decreasing: {stacks}")
    return stacks


def _require_gurobi() -> None:
    if gp is None or GRB is None:
        raise ImportError("gurobipy is required for exact MIP solving") from _GUROBI_IMPORT_ERROR


def _validate_instance(instance: PremarshallingInstance, maxSteps: int) -> None:
    if maxSteps < 0:
        raise ValueError("maxSteps must be non-negative")
    if instance.StackCount <= 0:
        raise ValueError("StackCount must be positive")
    if instance.MaxHeight <= 0:
        raise ValueError("MaxHeight must be positive")
    if instance.ContainerCount < 0:
        raise ValueError("ContainerCount must be non-negative")
    if len(instance.Stacks) != instance.StackCount:
        raise ValueError(
            f"Stacks.Count {len(instance.Stacks)} != StackCount {instance.StackCount}"
        )

    seen: List[int] = []
    for idx, stack in enumerate(instance.Stacks, start=1):
        if len(stack) > instance.MaxHeight:
            raise ValueError(f"stack {idx} height exceeds MaxHeight")
        seen.extend(int(x) for x in stack)

    expected = list(range(1, instance.ContainerCount + 1))
    actual = sorted(seen)
    if actual != expected:
        raise ValueError(
            f"container IDs must be exactly 1..{instance.ContainerCount}; got {actual}"
        )


def _all_stacks_decreasing(stacks: List[List[int]]) -> bool:
    return all(all(stack[i] > stack[i + 1] for i in range(len(stack) - 1)) for stack in stacks)


def _initial_maps(instance: PremarshallingInstance) -> Tuple[Dict[int, int], Dict[int, int]]:
    stack_of: Dict[int, int] = {}
    level_of: Dict[int, int] = {}
    for i, stack in enumerate(instance.Stacks, start=1):
        for level, box in enumerate(stack, start=1):
            stack_of[int(box)] = i
            level_of[int(box)] = level
    return stack_of, level_of


def _configure_model(model: "gp.Model", timeLimitSec: Optional[float]) -> None:
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.0
    if timeLimitSec is not None:
        model.Params.TimeLimit = float(timeLimitSec)


def _status_name(status: int) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    }
    return names.get(status, str(status))


def _optimize_with_diagnostics(
    model: "gp.Model",
    model_name: str,
    diagnosticsDir: Optional[str],
) -> int:
    model.optimize()
    if model.Status == GRB.INF_OR_UNBD:
        model.Params.DualReductions = 0
        model.optimize()
    if model.Status == GRB.INFEASIBLE:
        out_dir = Path(diagnosticsDir or "mip_diagnostics")
        out_dir.mkdir(parents=True, exist_ok=True)
        model.write(str(out_dir / f"{model_name}.lp"))
        model.computeIIS()
        model.write(str(out_dir / f"{model_name}.ilp"))
    return int(model.Status)


def _solution_from_model(
    model: "gp.Model",
    status: int,
    extractor,
    instance: PremarshallingInstance,
) -> PremarshallingSolution:
    feasible = int(getattr(model, "SolCount", 0)) > 0
    optimal = status == GRB.OPTIMAL
    objective = float(model.ObjVal) if feasible else float("inf")
    moves: List[Move] = []
    if feasible:
        moves = extractor()
        validate_move_sequence(instance, moves, expected_objective=objective)
    return PremarshallingSolution(
        Feasible=feasible,
        Optimal=optimal,
        Objective=objective,
        Moves=moves,
        RuntimeSec=float(model.Runtime),
        NodeCount=int(round(float(model.NodeCount))),
        Status=_status_name(status),
    )


def _build_4d_model(
    instance: PremarshallingInstance,
    maxSteps: int,
    timeLimitSec: Optional[float],
) -> Dict[str, object]:
    m, H, n, T = (
        instance.StackCount,
        instance.MaxHeight,
        instance.ContainerCount,
        maxSteps,
    )
    boxes = range(1, n + 1)
    stacks = range(1, m + 1)
    levels = range(1, H + 1)
    times = range(0, T + 1)
    steps = range(1, T + 1)

    model = gp.Model("premarshalling_4d_slot")
    _configure_model(model, timeLimitSec)

    x = {
        (p, i, l, t): model.addVar(vtype=GRB.BINARY, name=f"x_{p}_{i}_{l}_{t}")
        for p in boxes
        for i in stacks
        for l in levels
        for t in times
    }
    a = {
        (p, i, l, k, r, s): model.addVar(
            vtype=GRB.BINARY, name=f"a_{p}_{i}_{l}_{k}_{r}_{s}"
        )
        for p in boxes
        for i in stacks
        for l in levels
        for k in stacks
        if k != i
        for r in levels
        for s in steps
    }
    active = {s: model.addVar(vtype=GRB.BINARY, name=f"active_{s}") for s in steps}
    idle = {s: model.addVar(vtype=GRB.BINARY, name=f"idle_{s}") for s in steps}

    stack_of, level_of = _initial_maps(instance)
    for p in boxes:
        for i in stacks:
            for l in levels:
                model.addConstr(
                    x[p, i, l, 0] == (1 if stack_of[p] == i and level_of[p] == l else 0),
                    name=f"init_{p}_{i}_{l}",
                )

    for t in times:
        for p in boxes:
            model.addConstr(
                gp.quicksum(x[p, i, l, t] for i in stacks for l in levels) == 1,
                name=f"box_once_{p}_{t}",
            )
        for i in stacks:
            for l in levels:
                model.addConstr(
                    gp.quicksum(x[p, i, l, t] for p in boxes) <= 1,
                    name=f"slot_cap_{i}_{l}_{t}",
                )
            for l in range(1, H):
                model.addConstr(
                    gp.quicksum(x[p, i, l + 1, t] for p in boxes)
                    <= gp.quicksum(x[p, i, l, t] for p in boxes),
                    name=f"no_float_{i}_{l}_{t}",
                )

    for s in steps:
        model.addConstr(active[s] + idle[s] == 1, name=f"active_idle_{s}")
        model.addConstr(
            gp.quicksum(a[key] for key in a if key[5] == s) == active[s],
            name=f"one_move_{s}",
        )
        if s < T:
            model.addConstr(idle[s] <= idle[s + 1], name=f"no_late_idle_{s}")

        for key, var in a.items():
            p, i, l, k, r, step = key
            if step != s:
                continue
            model.addConstr(var <= x[p, i, l, s - 1], name=f"src_occ_{p}_{i}_{l}_{k}_{r}_{s}")
            for upper in range(l + 1, H + 1):
                model.addConstr(
                    var + gp.quicksum(x[q, i, upper, s - 1] for q in boxes) <= 1,
                    name=f"src_top_{p}_{i}_{l}_{upper}_{k}_{r}_{s}",
                )
            model.addConstr(
                var + gp.quicksum(x[q, k, r, s - 1] for q in boxes) <= 1,
                name=f"dst_empty_{p}_{i}_{l}_{k}_{r}_{s}",
            )
            for below_level in range(1, r):
                model.addConstr(
                    var <= gp.quicksum(x[q, k, below_level, s - 1] for q in boxes),
                    name=f"dst_supported_{p}_{i}_{l}_{k}_{r}_{below_level}_{s}",
                )

        for p in boxes:
            for i in stacks:
                for l in levels:
                    out_expr = gp.quicksum(
                        a[p, i, l, k, r, s] for k in stacks if k != i for r in levels
                    )
                    in_expr = gp.quicksum(
                        a[p, h, u, i, l, s] for h in stacks if h != i for u in levels
                    )
                    model.addConstr(
                        x[p, i, l, s] == x[p, i, l, s - 1] - out_expr + in_expr,
                        name=f"transition_{p}_{i}_{l}_{s}",
                    )

    for i in stacks:
        for l in range(1, H):
            for lower in boxes:
                for upper in range(lower + 1, n + 1):
                    model.addConstr(
                        x[lower, i, l, T] + x[upper, i, l + 1, T] <= 1,
                        name=f"final_sorted_{i}_{l}_{lower}_{upper}",
                    )

    model.setObjective(gp.quicksum(active[s] for s in steps), GRB.MINIMIZE)
    return {"model": model, "x": x, "a": a, "active": active, "idle": idle}


def _build_3d_model(
    instance: PremarshallingInstance,
    maxSteps: int,
    timeLimitSec: Optional[float],
) -> Dict[str, object]:
    m, H, n, T = (
        instance.StackCount,
        instance.MaxHeight,
        instance.ContainerCount,
        maxSteps,
    )
    boxes = range(1, n + 1)
    stacks = range(1, m + 1)
    times = range(0, T + 1)
    steps = range(1, T + 1)
    ordered_pairs = [(p, q) for p in boxes for q in boxes if p != q]
    unordered_pairs = [(p, q) for p in boxes for q in range(p + 1, n + 1)]

    model = gp.Model("premarshalling_3d_integer_position")
    _configure_model(model, timeLimitSec)

    z = {
        (p, i, t): model.addVar(vtype=GRB.BINARY, name=f"z_{p}_{i}_{t}")
        for p in boxes
        for i in stacks
        for t in times
    }
    level = {
        (p, t): model.addVar(vtype=GRB.INTEGER, lb=1, ub=H, name=f"level_{p}_{t}")
        for p in boxes
        for t in times
    }
    below = {
        (p, q, t): model.addVar(vtype=GRB.BINARY, name=f"below_{p}_{q}_{t}")
        for p, q in ordered_pairs
        for t in times
    }
    same = {
        (p, q, t): model.addVar(vtype=GRB.BINARY, name=f"same_{p}_{q}_{t}")
        for p, q in unordered_pairs
        for t in times
    }
    move = {
        (p, i, k, s): model.addVar(vtype=GRB.BINARY, name=f"move_{p}_{i}_{k}_{s}")
        for p in boxes
        for i in stacks
        for k in stacks
        if k != i
        for s in steps
    }
    active = {s: model.addVar(vtype=GRB.BINARY, name=f"active_{s}") for s in steps}
    idle = {s: model.addVar(vtype=GRB.BINARY, name=f"idle_{s}") for s in steps}

    stack_of, level_of = _initial_maps(instance)
    for p in boxes:
        for i in stacks:
            model.addConstr(z[p, i, 0] == (1 if stack_of[p] == i else 0), name=f"init_z_{p}_{i}")
        model.addConstr(level[p, 0] == level_of[p], name=f"init_level_{p}")
    for p, q in ordered_pairs:
        init_below = (
            stack_of[p] == stack_of[q]
            and level_of[p] < level_of[q]
        )
        model.addConstr(below[p, q, 0] == (1 if init_below else 0), name=f"init_below_{p}_{q}")

    for t in times:
        for p in boxes:
            model.addConstr(
                gp.quicksum(z[p, i, t] for i in stacks) == 1,
                name=f"box_stack_once_{p}_{t}",
            )
            model.addConstr(
                level[p, t]
                == 1 + gp.quicksum(below[q, p, t] for q in boxes if q != p),
                name=f"level_count_{p}_{t}",
            )
        for i in stacks:
            model.addConstr(
                gp.quicksum(z[p, i, t] for p in boxes) <= H,
                name=f"stack_cap_{i}_{t}",
            )
        for p, q in ordered_pairs:
            model.addConstr(
                level[p, t] + 1 <= level[q, t] + H * (1 - below[p, q, t]),
                name=f"below_level_{p}_{q}_{t}",
            )
        for p, q in unordered_pairs:
            spq = same[p, q, t]
            for i in stacks:
                model.addConstr(spq >= z[p, i, t] + z[q, i, t] - 1, name=f"same_lb_{p}_{q}_{i}_{t}")
                model.addConstr(spq <= 1 - z[p, i, t] + z[q, i, t], name=f"same_ub_p_{p}_{q}_{i}_{t}")
                model.addConstr(spq <= 1 - z[q, i, t] + z[p, i, t], name=f"same_ub_q_{p}_{q}_{i}_{t}")
            model.addConstr(
                below[p, q, t] + below[q, p, t] == spq,
                name=f"below_same_{p}_{q}_{t}",
            )
        for p in boxes:
            for q in boxes:
                if q == p:
                    continue
                for r in boxes:
                    if r == p or r == q:
                        continue
                    model.addConstr(
                        below[p, q, t] + below[q, r, t] - 1 <= below[p, r, t],
                        name=f"transitive_{p}_{q}_{r}_{t}",
                    )

    for s in steps:
        model.addConstr(active[s] + idle[s] == 1, name=f"active_idle_{s}")
        model.addConstr(
            gp.quicksum(move[key] for key in move if key[3] == s) == active[s],
            name=f"one_move_{s}",
        )
        if s < T:
            model.addConstr(idle[s] <= idle[s + 1], name=f"no_late_idle_{s}")

        for p in boxes:
            moved_box = gp.quicksum(move[p, i, k, s] for i in stacks for k in stacks if k != i)
            model.addConstr(level[p, s] - level[p, s - 1] <= H * moved_box, name=f"level_hold_pos_{p}_{s}")
            model.addConstr(level[p, s - 1] - level[p, s] <= H * moved_box, name=f"level_hold_neg_{p}_{s}")

        for p in boxes:
            for i in stacks:
                out_expr = gp.quicksum(move[p, i, k, s] for k in stacks if k != i)
                in_expr = gp.quicksum(move[p, h, i, s] for h in stacks if h != i)
                model.addConstr(
                    z[p, i, s] == z[p, i, s - 1] - out_expr + in_expr,
                    name=f"z_transition_{p}_{i}_{s}",
                )

        for key, var in move.items():
            p, i, k, step = key
            if step != s:
                continue
            height_i_prev = gp.quicksum(z[q, i, s - 1] for q in boxes)
            height_k_prev = gp.quicksum(z[q, k, s - 1] for q in boxes)
            model.addConstr(var <= z[p, i, s - 1], name=f"move_src_{p}_{i}_{k}_{s}")
            model.addConstr(
                level[p, s - 1] - height_i_prev <= H * (1 - var),
                name=f"src_top_pos_{p}_{i}_{k}_{s}",
            )
            model.addConstr(
                height_i_prev - level[p, s - 1] <= H * (1 - var),
                name=f"src_top_neg_{p}_{i}_{k}_{s}",
            )
            model.addConstr(
                height_k_prev <= (H - 1) + H * (1 - var),
                name=f"dst_not_full_{p}_{i}_{k}_{s}",
            )
            model.addConstr(
                level[p, s] - (height_k_prev + 1) <= H * (1 - var),
                name=f"dst_level_pos_{p}_{i}_{k}_{s}",
            )
            model.addConstr(
                (height_k_prev + 1) - level[p, s] <= H * (1 - var),
                name=f"dst_level_neg_{p}_{i}_{k}_{s}",
            )

    for p in boxes:
        for q in range(p + 1, n + 1):
            model.addConstr(below[p, q, T] == 0, name=f"final_sorted_{p}_{q}")

    model.setObjective(gp.quicksum(active[s] for s in steps), GRB.MINIMIZE)
    return {
        "model": model,
        "z": z,
        "level": level,
        "below": below,
        "same": same,
        "move": move,
        "active": active,
        "idle": idle,
    }


def _extract_4d_moves(model_data: Dict[str, object], maxSteps: int) -> List[Move]:
    a: Dict[Tuple[int, int, int, int, int, int], "gp.Var"] = model_data["a"]  # type: ignore[assignment]
    active: Dict[int, "gp.Var"] = model_data["active"]  # type: ignore[assignment]
    moves: List[Move] = []
    for s in range(1, maxSteps + 1):
        if active[s].X <= 0.5:
            continue
        chosen = [key for key, var in a.items() if key[5] == s and var.X > 0.5]
        if len(chosen) != 1:
            raise RuntimeError(f"expected exactly one 4D move at step {s}, got {chosen}")
        p, i, l, k, r, _ = chosen[0]
        moves.append(Move(Step=s, Box=p, FromStack=i, ToStack=k, FromLevel=l, ToLevel=r))
    return moves


def _extract_3d_moves(
    model_data: Dict[str, object],
    instance: PremarshallingInstance,
    maxSteps: int,
) -> List[Move]:
    move: Dict[Tuple[int, int, int, int], "gp.Var"] = model_data["move"]  # type: ignore[assignment]
    level: Dict[Tuple[int, int], "gp.Var"] = model_data["level"]  # type: ignore[assignment]
    active: Dict[int, "gp.Var"] = model_data["active"]  # type: ignore[assignment]
    moves: List[Move] = []
    for s in range(1, maxSteps + 1):
        if active[s].X <= 0.5:
            continue
        chosen = [key for key, var in move.items() if key[3] == s and var.X > 0.5]
        if len(chosen) != 1:
            raise RuntimeError(f"expected exactly one 3D move at step {s}, got {chosen}")
        p, i, k, _ = chosen[0]
        from_level = int(round(level[p, s - 1].X))
        to_level = int(round(level[p, s].X))
        moves.append(
            Move(
                Step=s,
                Box=p,
                FromStack=i,
                ToStack=k,
                FromLevel=from_level,
                ToLevel=to_level,
            )
        )
    return moves


# Pythonic aliases for callers that prefer snake_case.
solve_3d_integer_position_mip = Solve3DIntegerPositionMip
solve_4d_slot_mip = Solve4DSlotMip
