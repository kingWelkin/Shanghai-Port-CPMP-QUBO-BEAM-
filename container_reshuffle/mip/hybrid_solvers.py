"""Experimental MIP hybrid solvers for larger CPMP workflows."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..core import (
    Instance,
    Move,
    ProblemVariant,
    State,
    apply_move,
    create_initial_state,
    future_disorder_count_stacks,
    generate_legal_moves,
    is_goal,
    validate_solution,
)
from ..features import entropy_limit
from ..scoring import cpmp_business_feasible
from ..solvers.base import Solver, SolverConfig, failure_solution, not_applicable, solution_from_state
from ..solvers.beam import BeamSearchSolver
from .mip_4d_slot_occupancy import solve_cpmp_4d_slot_occupancy
from .mip_utils import gurobi_unavailable_solution, require_cpmp

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover
    gp = None
    GRB = None


class RollingHorizonMipSolver(Solver):
    """Repeatedly solve a short 4D slot MIP window and commit a prefix."""

    name = "rolling_horizon_mip"

    def solve(self, instance: Instance, config: SolverConfig):
        """Run rolling-horizon MIP for CPMP."""

        if instance.variant is not ProblemVariant.PREMARSHALLING_CPMP:
            return not_applicable(self.name)
        if gp is None or GRB is None:
            return gurobi_unavailable_solution(self.name)
        start = time.perf_counter()
        deadline = start + max(0.001, config.time_limit_sec)
        state = create_initial_state(instance)
        calls = vars_total = constr_total = 0
        nodes_total = 0.0
        window_statuses: List[str] = []

        while not is_goal(instance, state) and state.move_count < config.max_depth:
            remaining_global = config.max_depth - state.move_count
            window = max(1, min(config.rolling_depth, remaining_global))
            time_left = deadline - time.perf_counter()
            if time_left <= 0:
                break
            result = _solve_window_4d(
                instance,
                state,
                window,
                min(time_left, _mip_window_time_limit(config)),
                config.entropy_alpha,
            )
            calls += 1
            vars_total += result.var_count
            constr_total += result.constr_count
            nodes_total += result.node_count
            window_statuses.append(result.status)
            if not result.moves:
                break
            committed = 0
            for mv in result.moves:
                if committed >= max(1, config.prefix_commit_length):
                    break
                if state.move_count >= config.max_depth:
                    break
                try:
                    state = apply_move(instance, state, mv)
                except ValueError:
                    return failure_solution(
                        self.name,
                        start,
                        "WINDOW_MOVE_INVALID",
                        mip_node_count=nodes_total,
                        mip_var_count=vars_total,
                        mip_constr_count=constr_total,
                    )
                committed += 1
                if not cpmp_business_feasible(instance, state, config.entropy_alpha):
                    return failure_solution(
                        self.name,
                        start,
                        "BUSINESS_CAP_VIOLATED",
                        mip_node_count=nodes_total,
                        mip_var_count=vars_total,
                        mip_constr_count=constr_total,
                    )
                if is_goal(instance, state):
                    sol = solution_from_state(
                        self.name,
                        instance,
                        state,
                        start,
                        mip_node_count=nodes_total,
                        mip_var_count=vars_total,
                        mip_constr_count=constr_total,
                    )
                    sol.status += f"; mip_calls={calls}; window_statuses={','.join(sorted(set(window_statuses)))}"
                    return sol
        if is_goal(instance, state):
            return solution_from_state(self.name, instance, state, start)
        return failure_solution(
            self.name,
            start,
            f"NO_FEASIBLE_FOUND; mip_calls={calls}; window_statuses={','.join(sorted(set(window_statuses))) or 'none'}",
            mip_node_count=nodes_total,
            mip_var_count=vars_total,
            mip_constr_count=constr_total,
        )


class FinalStackAssignmentMipSolver(Solver):
    """Master stack-assignment MIP followed by 4D realization MIP."""

    name = "final_assignment_mip"

    def solve(self, instance: Instance, config: SolverConfig):
        """Solve target stack assignment, then realize it with 4D MIP."""

        if instance.variant is not ProblemVariant.PREMARSHALLING_CPMP:
            return not_applicable(self.name)
        if gp is None or GRB is None:
            return gurobi_unavailable_solution(self.name)
        start = time.perf_counter()
        target, master = _solve_assignment_master(instance, min(config.time_limit_sec, _master_time_limit(config)))
        if target is None:
            return failure_solution(
                self.name,
                start,
                f"MASTER_{master.status}",
                mip_node_count=master.node_count,
                mip_var_count=master.var_count,
                mip_constr_count=master.constr_count,
            )
        remaining = max(0.001, config.time_limit_sec - (time.perf_counter() - start))
        sol = solve_cpmp_4d_slot_occupancy(
            instance,
            config.max_depth,
            min(remaining, _realization_attempt_time(config)),
            0.0,
            entropy_alpha=config.entropy_alpha,
            target_stack=target,
        )
        if not sol.feasible:
            repaired = _beam_repair_target(instance, config)
            if repaired is not None:
                repair_target, repair_moves = repaired
                remaining = max(0.001, config.time_limit_sec - (time.perf_counter() - start))
                repaired_sol = solve_cpmp_4d_slot_occupancy(
                    instance,
                    config.max_depth,
                    remaining,
                    0.0,
                    entropy_alpha=config.entropy_alpha,
                    target_stack=repair_target,
                    start_moves=repair_moves,
                )
                repaired_sol.status = f"{repaired_sol.status}; first_target={sol.status}; repair=beam_target"
                sol = repaired_sol
        return _rename_mip_solution(self.name, sol, start, master)


class PatternColumnGenerationSolver(Solver):
    """Pattern-set master MIP followed by 4D realization MIP.

    This is a compact prototype rather than a full branch-and-price
    implementation: columns are generated upfront as legal decreasing final
    stack patterns.
    """

    name = "pattern_column_mip"

    def solve(self, instance: Instance, config: SolverConfig):
        """Select final stack patterns, then realize them with 4D MIP."""

        if instance.variant is not ProblemVariant.PREMARSHALLING_CPMP:
            return not_applicable(self.name)
        if gp is None or GRB is None:
            return gurobi_unavailable_solution(self.name)
        start = time.perf_counter()
        target, master = _solve_pattern_master(instance, config, min(config.time_limit_sec, _master_time_limit(config)))
        if target is None:
            return failure_solution(
                self.name,
                start,
                f"MASTER_{master.status}",
                mip_node_count=master.node_count,
                mip_var_count=master.var_count,
                mip_constr_count=master.constr_count,
            )
        remaining = max(0.001, config.time_limit_sec - (time.perf_counter() - start))
        sol = solve_cpmp_4d_slot_occupancy(
            instance,
            config.max_depth,
            min(remaining, _realization_attempt_time(config)),
            0.0,
            entropy_alpha=config.entropy_alpha,
            target_stack=target,
        )
        if not sol.feasible:
            repaired = _beam_repair_target(instance, config)
            if repaired is not None:
                repair_target, repair_moves = repaired
                remaining = max(0.001, config.time_limit_sec - (time.perf_counter() - start))
                repaired_sol = solve_cpmp_4d_slot_occupancy(
                    instance,
                    config.max_depth,
                    remaining,
                    0.0,
                    entropy_alpha=config.entropy_alpha,
                    target_stack=repair_target,
                    start_moves=repair_moves,
                )
                repaired_sol.status = f"{repaired_sol.status}; first_target={sol.status}; repair=beam_target"
                sol = repaired_sol
        return _rename_mip_solution(self.name, sol, start, master)


@dataclass
class _MipStats:
    status: str
    var_count: int = 0
    constr_count: int = 0
    node_count: float = 0.0


@dataclass
class _WindowResult(_MipStats):
    moves: List[Move] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.moves is None:
            self.moves = []


def _solve_window_4d(
    instance: Instance,
    root: State,
    window: int,
    time_limit_sec: float,
    alpha: float,
) -> _WindowResult:
    """Solve a short 4D MIP from an arbitrary state without forcing final goal."""

    require_cpmp(instance)
    m, H, n, T = instance.stack_count, instance.max_height, instance.container_count, window
    boxes = range(1, n + 1)
    stacks = range(m)
    levels = range(H)
    times = range(T + 1)
    steps = range(T)
    model = gp.Model("cpmp_4d_window")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(time_limit_sec)
    model.Params.MIPGap = 0.0

    x = {(p, i, h, t): model.addVar(vtype=GRB.BINARY, name=f"x_{p}_{i}_{h}_{t}") for p in boxes for i in stacks for h in levels for t in times}
    out = {(p, i, h, t): model.addVar(vtype=GRB.BINARY, name=f"out_{p}_{i}_{h}_{t}") for p in boxes for i in stacks for h in levels for t in steps}
    inn = {(p, i, h, t): model.addVar(vtype=GRB.BINARY, name=f"in_{p}_{i}_{h}_{t}") for p in boxes for i in stacks for h in levels for t in steps}
    z = {(p, t): model.addVar(vtype=GRB.BINARY, name=f"z_{p}_{t}") for p in boxes for t in steps}
    active = {t: model.addVar(vtype=GRB.BINARY, name=f"active_{t}") for t in steps}

    for p in boxes:
        for i in stacks:
            for h in levels:
                init = 1 if h < len(root.stacks[i]) and root.stacks[i][h] == p else 0
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
        model.addConstr(gp.quicksum(z[p, t] for p in boxes) == active[t], name=f"active_{t}")
        if t == 0 and not is_goal(instance, root):
            model.addConstr(active[t] == 1, name="first_move")
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
                        name=f"trans_{p}_{i}_{h}_{t}",
                    )

    disorder = _add_window_disorder(model, instance, boxes, stacks, levels, times, x)
    limit = entropy_limit(instance, alpha)
    for t in times:
        prefix_moves = gp.quicksum(active[s] for s in range(min(t, T)))
        model.addConstr(root.move_count + prefix_moves + disorder[t] <= limit, name=f"business_cap_{t}")
    terminal_weight = 1000.0
    model.setObjective(
        terminal_weight * disorder[T]
        + gp.quicksum(disorder[t] for t in times if t > 0)
        + gp.quicksum(active[t] for t in steps),
        GRB.MINIMIZE,
    )
    model.optimize()
    status = _status_name(model.Status)
    stats = _WindowResult(status, int(model.NumVars), int(model.NumConstrs), float(model.NodeCount), [])
    if model.SolCount <= 0:
        return stats
    stats.moves = _extract_window_moves(T, boxes, stacks, levels, active, out, inn)
    return stats


def _add_window_disorder(model, instance: Instance, boxes, stacks, levels, times, x):
    n = instance.container_count
    H = instance.max_height
    violate = {
        (i, h, t): model.addVar(vtype=GRB.BINARY, name=f"wd_viol_{i}_{h}_{t}")
        for i in stacks
        for h in range(1, H)
        for t in times
    }
    suffix = {
        (i, h, t): model.addVar(vtype=GRB.BINARY, name=f"wd_suffix_{i}_{h}_{t}")
        for i in stacks
        for h in levels
        for t in times
    }
    bad = {
        (i, h, t): model.addVar(vtype=GRB.BINARY, name=f"wd_bad_{i}_{h}_{t}")
        for i in stacks
        for h in levels
        for t in times
    }
    disorder = {}
    for t in times:
        for i in stacks:
            model.addConstr(suffix[i, 0, t] == 0, name=f"wd_suffix_base_{i}_{t}")
            model.addConstr(bad[i, 0, t] == 0, name=f"wd_bad_base_{i}_{t}")
            for h in range(1, H):
                lower_value = gp.quicksum(p * x[p, i, h - 1, t] for p in boxes)
                upper_value = gp.quicksum(p * x[p, i, h, t] for p in boxes)
                upper_occ = gp.quicksum(x[p, i, h, t] for p in boxes)
                diff = upper_value - lower_value
                model.addConstr(violate[i, h, t] <= upper_occ, name=f"wd_viol_occ_{i}_{h}_{t}")
                model.addConstr(diff <= n * violate[i, h, t] + n * (1 - upper_occ), name=f"wd_viol_force_{i}_{h}_{t}")
                model.addConstr(
                    diff >= 1 - n * (1 - violate[i, h, t]) - n * (1 - upper_occ),
                    name=f"wd_viol_exact_{i}_{h}_{t}",
                )
                model.addConstr(suffix[i, h, t] >= suffix[i, h - 1, t], name=f"wd_suffix_prev_{i}_{h}_{t}")
                model.addConstr(suffix[i, h, t] >= violate[i, h, t], name=f"wd_suffix_viol_{i}_{h}_{t}")
                model.addConstr(suffix[i, h, t] <= suffix[i, h - 1, t] + violate[i, h, t], name=f"wd_suffix_upper_{i}_{h}_{t}")
                occ = gp.quicksum(x[p, i, h, t] for p in boxes)
                model.addConstr(bad[i, h, t] <= occ, name=f"wd_bad_occ_{i}_{h}_{t}")
                model.addConstr(bad[i, h, t] <= suffix[i, h, t], name=f"wd_bad_suffix_{i}_{h}_{t}")
                model.addConstr(bad[i, h, t] >= occ + suffix[i, h, t] - 1, name=f"wd_bad_lower_{i}_{h}_{t}")
        disorder[t] = gp.quicksum(bad[i, h, t] for i in stacks for h in levels)
    return disorder


def _extract_window_moves(T, boxes, stacks, levels, active, out, inn) -> List[Move]:
    moves: List[Move] = []
    for t in range(T):
        if active[t].X < 0.5:
            continue
        src = [(p, i, h) for p in boxes for i in stacks for h in levels if out[p, i, h, t].X > 0.5]
        dst = [(p, i, h) for p in boxes for i in stacks for h in levels if inn[p, i, h, t].X > 0.5]
        if len(src) != 1 or len(dst) != 1 or src[0][0] != dst[0][0]:
            return []
        p, i, h = src[0]
        _, j, e = dst[0]
        moves.append(Move(t, p, i, j, h + 1, e + 1))
    return moves


def _solve_assignment_master(instance: Instance, time_limit_sec: float) -> Tuple[Optional[Dict[int, int]], _MipStats]:
    m, H, n = instance.stack_count, instance.max_height, instance.container_count
    boxes = range(1, n + 1)
    stacks = range(m)
    current = {box: i for i, stack in enumerate(instance.stacks) for box in stack}
    removal_cost = _removal_costs(instance)
    model = gp.Model("final_stack_assignment_master")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(time_limit_sec)
    y = {(p, i): model.addVar(vtype=GRB.BINARY, name=f"y_{p}_{i}") for p in boxes for i in stacks}
    for p in boxes:
        model.addConstr(gp.quicksum(y[p, i] for i in stacks) == 1, name=f"assign_{p}")
    for i in stacks:
        model.addConstr(gp.quicksum(y[p, i] for p in boxes) <= H, name=f"cap_{i}")
    smaller_in_stack = {
        (p, i): sum(1 for q in instance.stacks[i] if q < p)
        for p in boxes
        for i in stacks
    }
    obj = gp.quicksum(
        ((0 if current[p] == i else removal_cost[p]) + 2.0 * smaller_in_stack[p, i] * (0 if current[p] == i else 1)) * y[p, i]
        for p in boxes
        for i in stacks
    )
    conflict_terms = []
    for stack in instance.stacks:
        pos = {box: idx for idx, box in enumerate(stack)}
        for p in stack:
            for q in stack:
                if p < q and pos[p] < pos[q]:
                    for i in stacks:
                        c = model.addVar(vtype=GRB.BINARY, name=f"conflict_{p}_{q}_{i}")
                        model.addConstr(c <= y[p, i], name=f"conflict_ub_p_{p}_{q}_{i}")
                        model.addConstr(c <= y[q, i], name=f"conflict_ub_q_{p}_{q}_{i}")
                        model.addConstr(c >= y[p, i] + y[q, i] - 1, name=f"conflict_lb_{p}_{q}_{i}")
                        conflict_terms.append(c)
    if conflict_terms:
        obj += 3.0 * gp.quicksum(conflict_terms)
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    stats = _MipStats(_status_name(model.Status), int(model.NumVars), int(model.NumConstrs), float(model.NodeCount))
    if model.SolCount <= 0:
        return None, stats
    target = {p: max(stacks, key=lambda i: y[p, i].X) for p in boxes}
    return target, stats


def _solve_pattern_master(instance: Instance, config: SolverConfig, time_limit_sec: float) -> Tuple[Optional[Dict[int, int]], _MipStats]:
    patterns = _generate_patterns(instance)
    m, n = instance.stack_count, instance.container_count
    boxes = range(1, n + 1)
    stacks = range(m)
    current_sets = {i: set(instance.stacks[i]) for i in stacks}
    model = gp.Model("pattern_column_master")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(time_limit_sec)
    y = {(i, c): model.addVar(vtype=GRB.BINARY, name=f"pat_{i}_{c}") for i in stacks for c in range(len(patterns))}
    for i in stacks:
        model.addConstr(gp.quicksum(y[i, c] for c in range(len(patterns))) == 1, name=f"one_pattern_{i}")
    for p in boxes:
        model.addConstr(
            gp.quicksum(y[i, c] for i in stacks for c, pat in enumerate(patterns) if p in pat) == 1,
            name=f"cover_{p}",
        )
    obj = gp.quicksum(
        _pattern_cost(patterns[c], current_sets[i], instance.stacks[i], _removal_costs(instance)) * y[i, c]
        for i in stacks
        for c in range(len(patterns))
    )
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    stats = _MipStats(_status_name(model.Status), int(model.NumVars), int(model.NumConstrs), float(model.NodeCount))
    if model.SolCount <= 0:
        return None, stats
    target: Dict[int, int] = {}
    for i in stacks:
        chosen = max(range(len(patterns)), key=lambda c: y[i, c].X)
        for p in patterns[chosen]:
            target[p] = i
    if len(target) != n:
        return None, _MipStats("DECODE_FAILED", stats.var_count, stats.constr_count, stats.node_count)
    return target, stats


def _generate_patterns(instance: Instance) -> List[Tuple[int, ...]]:
    boxes = list(range(1, instance.container_count + 1))
    patterns: List[Tuple[int, ...]] = [()]
    for size in range(1, instance.max_height + 1):
        for comb in itertools.combinations(boxes, size):
            patterns.append(tuple(sorted(comb, reverse=True)))
    return patterns


def _pattern_cost(pattern: Tuple[int, ...], current: set[int], current_stack: List[int], removal_cost: Dict[int, int]) -> float:
    pattern_set = set(pattern)
    pos = {box: idx for idx, box in enumerate(current_stack)}
    conflict = 0
    for p in pattern_set:
        for q in pattern_set:
            if p < q and p in pos and q in pos and pos[p] < pos[q]:
                conflict += 1
    removed_cost = sum(removal_cost[p] for p in current - pattern_set)
    return float(len(pattern_set - current) + 0.25 * removed_cost + 3.0 * conflict)


def _removal_costs(instance: Instance) -> Dict[int, int]:
    costs: Dict[int, int] = {}
    for stack in instance.stacks:
        for idx, box in enumerate(stack):
            costs[box] = 1 + len(stack) - idx - 1
    return costs


def _rename_mip_solution(name: str, sol, start_time: float, master: _MipStats):
    realization_optimal = sol.optimal
    sol.solver_name = name
    sol.optimal = False
    sol.runtime_sec = time.perf_counter() - start_time
    sol.mip_node_count = (sol.mip_node_count or 0.0) + master.node_count
    sol.mip_var_count = (sol.mip_var_count or 0) + master.var_count
    sol.mip_constr_count = (sol.mip_constr_count or 0) + master.constr_count
    sol.status = f"{sol.status}; realization_optimal={realization_optimal}; master={master.status}"
    return sol


def _beam_repair_target(instance: Instance, config: SolverConfig) -> Optional[Tuple[Dict[int, int], List[Move]]]:
    warm_cfg = SolverConfig(
        max_depth=config.max_depth,
        beam_width=max(1, min(config.beam_width, 300)),
        max_candidates_per_layer=max(1, min(config.max_candidates_per_layer, 5000)),
        candidate_limit_per_state=config.candidate_limit_per_state,
        time_limit_sec=max(0.001, min(config.time_limit_sec, config.mip_warm_start_time_limit_sec)),
        entropy_alpha=config.entropy_alpha,
        random_seed=config.random_seed,
        enable_visited_pruning=config.enable_visited_pruning,
        enable_canonical_hash=config.enable_canonical_hash,
        score_weights=config.score_weights,
    )
    sol = BeamSearchSolver().solve(instance, warm_cfg)
    ok, _, _ = validate_solution(instance, sol) if sol.feasible else (False, "", [])
    if not ok:
        return None
    state = create_initial_state(instance)
    for move in sol.moves:
        state = apply_move(instance, state, move)
    target = {box: i for i, stack in enumerate(state.stacks) for box in stack}
    return target, sol.moves


def _master_time_limit(config: SolverConfig) -> float:
    return max(0.1, min(10.0, config.time_limit_sec * 0.25))


def _realization_attempt_time(config: SolverConfig) -> float:
    return max(0.1, min(10.0, config.time_limit_sec * 0.25))


def _mip_window_time_limit(config: SolverConfig) -> float:
    return max(0.1, min(10.0, config.time_limit_sec))


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
