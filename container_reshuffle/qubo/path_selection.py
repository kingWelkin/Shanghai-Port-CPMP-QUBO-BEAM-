"""QUBO rolling-horizon path selection."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..core import Move, State, apply_move, create_initial_state, generate_legal_moves, is_goal
from ..features import (
    bad_count,
    bad_relocation_count,
    blocking_count,
    current_target_blockers,
    dirty_stack_count,
    disorder_box_count,
    inversion_count,
    retrieval_progress,
    well_prefix_gain,
)
from ..scoring import cpmp_business_feasible, score_state
from ..solvers.base import Solver, SolverConfig, failure_solution, solution_from_state
from .model import QuboModel
from .sampler import IQuboSampler, QuboSamplerConfig, SimulatedAnnealingQuboSampler


@dataclass
class PathEdge:
    """Action-tree edge."""

    id: int
    from_node: int
    to_node: int
    move: Move
    cost: float


@dataclass
class PathNode:
    """Action-tree node."""

    id: int
    state: State
    depth: int
    incoming_edges: List[int] = field(default_factory=list)
    outgoing_edges: List[int] = field(default_factory=list)


class QuboPathSelectionBuilder:
    """Build a QUBO for selecting one root-to-depth path."""

    def build_tree(self, instance, root: State, config: SolverConfig) -> Tuple[List[PathNode], List[PathEdge]]:
        """Build a truncated action tree."""

        nodes = [PathNode(0, root, 0)]
        edges: List[PathEdge] = []
        frontier = [0]
        for depth in range(config.rolling_depth):
            next_frontier = []
            scored_children = []
            for node_id in frontier:
                node = nodes[node_id]
                moves = generate_legal_moves(instance, node.state)
                for mv in moves:
                    ns = apply_move(instance, node.state, mv)
                    if not cpmp_business_feasible(instance, ns, config.entropy_alpha):
                        continue
                    cost = self.edge_cost(instance, node.state, ns, config)
                    scored_children.append((score_state(instance, ns, config.score_weights), node_id, mv, ns, cost))
            scored_children.sort(key=lambda x: x[0])
            scored_children = scored_children[: config.max_candidates_per_layer]
            for _, parent_id, mv, ns, cost in scored_children:
                nid = len(nodes)
                eid = len(edges)
                nodes.append(PathNode(nid, ns, depth + 1, incoming_edges=[eid]))
                edges.append(PathEdge(eid, parent_id, nid, mv, cost))
                nodes[parent_id].outgoing_edges.append(eid)
                next_frontier.append(nid)
            frontier = next_frontier[: config.beam_width]
            if not frontier:
                break
        return nodes, edges

    @staticmethod
    def edge_cost(instance, current: State, nxt: State, config: SolverConfig) -> float:
        """Variant-aware edge cost."""

        w = config.score_weights
        if instance.variant.value == "retrieval_crp":
            return (
                1
                + w.w_blocking * blocking_count(instance, nxt)
                + w.w_target * current_target_blockers(instance, nxt)
                + w.w_bad * bad_relocation_count(instance, nxt)
                - w.w_progress * retrieval_progress(instance, nxt)
            )
        return (
            100.0 * disorder_box_count(instance, nxt)
            + 1
            + w.w_badcount * bad_count(instance, nxt)
            + w.w_inv * inversion_count(instance, nxt)
            + w.w_dirty * dirty_stack_count(instance, nxt)
            - w.w_gain * well_prefix_gain(instance, current, nxt)
        )

    def build_qubo(self, nodes: List[PathNode], edges: List[PathEdge], penalty_a: float) -> QuboModel:
        """Build path-flow QUBO over edge variables."""

        model = QuboModel()
        edge_vars = {edge.id: model.add_binary_var(f"z_{edge.id}") for edge in edges}
        for edge in edges:
            model.add_linear(edge_vars[edge.id], edge.cost)
        root_out = [edge_vars[eid] for eid in nodes[0].outgoing_edges]
        if root_out:
            model.add_one_hot_penalty(root_out, penalty_a)
        for node in nodes[1:]:
            if not node.outgoing_edges:
                continue
            incoming = [edge_vars[eid] for eid in node.incoming_edges]
            outgoing = [edge_vars[eid] for eid in node.outgoing_edges]
            # (sum_in - sum_out)^2
            for v in incoming + outgoing:
                model.add_linear(v, penalty_a)
            for idx, vi in enumerate(incoming):
                for vj in incoming[idx + 1 :]:
                    model.add_quadratic(vi, vj, 2 * penalty_a)
            for idx, vi in enumerate(outgoing):
                for vj in outgoing[idx + 1 :]:
                    model.add_quadratic(vi, vj, 2 * penalty_a)
            for vi in incoming:
                for vj in outgoing:
                    model.add_quadratic(vi, vj, -2 * penalty_a)
        return model

    def decode_or_repair(self, nodes: List[PathNode], edges: List[PathEdge], sample) -> Tuple[List[Move], int]:
        """Decode a selected path or repair via lowest-cost tree path."""

        if not edges:
            return [], 0
        selected = {eid for eid in range(len(edges)) if sample and sample.binary_values.get(eid, 0) == 1}
        path: List[Move] = []
        node_id = 0
        repair = 0
        while nodes[node_id].outgoing_edges:
            chosen = [eid for eid in nodes[node_id].outgoing_edges if eid in selected]
            if len(chosen) != 1:
                repair += 1
                return self._best_classical_path(nodes, edges), repair
            edge = edges[chosen[0]]
            path.append(edge.move)
            node_id = edge.to_node
        return path, repair

    @staticmethod
    def _best_classical_path(nodes: List[PathNode], edges: List[PathEdge]) -> List[Move]:
        best_cost: Dict[int, float] = {0: 0.0}
        best_path: Dict[int, List[Move]] = {0: []}
        for node in nodes:
            if node.id not in best_cost:
                continue
            for eid in node.outgoing_edges:
                edge = edges[eid]
                cost = best_cost[node.id] + edge.cost
                if cost < best_cost.get(edge.to_node, float("inf")):
                    best_cost[edge.to_node] = cost
                    best_path[edge.to_node] = best_path[node.id] + [edge.move]
        if not best_path:
            return []
        nonempty_paths = [nid for nid, path in best_path.items() if path]
        if not nonempty_paths:
            return []
        leaf = min(nonempty_paths, key=lambda nid: (best_cost[nid], -len(best_path[nid])))
        return best_path[leaf]


class QuboRollingHorizonSolver(Solver):
    """Rolling horizon with QUBO path selection."""

    name = "qubo_rolling_horizon"

    def __init__(self, sampler: Optional[IQuboSampler] = None) -> None:
        self.sampler = sampler or SimulatedAnnealingQuboSampler()

    def solve(self, instance, config: SolverConfig):
        """Run QUBO-assisted rolling horizon."""

        start = time.perf_counter()
        state = create_initial_state(instance)
        builder = QuboPathSelectionBuilder()
        calls = vars_total = repairs = 0
        qtime = 0.0
        qubo_time_limit = _qubo_time_limit(config)
        while not is_goal(instance, state) and state.move_count < config.max_depth:
            nodes, edges = builder.build_tree(instance, state, config)
            model = builder.build_qubo(nodes, edges, config.qubo_penalty_a)
            calls += 1
            vars_total += len(model.var_index)
            ss = self.sampler.sample(
                model,
                QuboSamplerConfig(
                    config.qubo_num_reads,
                    config.random_seed + calls,
                    qubo_time_limit,
                    sa_iterations=max(1, int(getattr(config, "qubo_sa_iterations", 5000))),
                ),
            )
            qtime += ss.runtime_sec
            path, repair = builder.decode_or_repair(nodes, edges, ss.samples[0] if ss.samples else None)
            repairs += repair
            if not path:
                break
            for mv in path[: max(1, config.prefix_commit_length)]:
                state = apply_move(instance, state, mv)
                if is_goal(instance, state) or state.move_count >= config.max_depth:
                    break
        if is_goal(instance, state):
            return solution_from_state(
                self.name,
                instance,
                state,
                start,
                qubo_calls=calls,
                qubo_total_vars=vars_total,
                qubo_total_time_sec=qtime,
                qubo_repair_count=repairs,
            )
        return failure_solution(
            self.name,
            start,
            "NO_FEASIBLE_FOUND",
            qubo_calls=calls,
            qubo_total_vars=vars_total,
            qubo_total_time_sec=qtime,
            qubo_repair_count=repairs,
        )


def _qubo_time_limit(config: SolverConfig) -> float:
    value = getattr(config, "qubo_time_limit_sec", None)
    if value is None:
        value = min(5.0, config.time_limit_sec)
    return max(0.001, float(value))
