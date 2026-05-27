"""QUBO route C: CRP batch relocation destination assignment."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..core import Instance, Move, ProblemVariant, State, apply_move, create_initial_state, is_goal, normalize_retrieval_state
from ..features import find_box, min_priority
from ..solvers.base import Solver, SolverConfig, failure_solution, not_applicable, solution_from_state
from ..solvers.greedy import _greedy_retrieval_destination
from .model import QuboModel
from .sampler import IQuboSampler, QuboSamplerConfig, QuboSample, SimulatedAnnealingQuboSampler


@dataclass
class BatchRelocationQuboData:
    """Builder side data for decoding a batch-relocation QUBO."""

    blockers: List[int]
    source: int
    destinations: List[int]
    var_for: Dict[Tuple[int, int], int]


class QuboBatchRelocationBuilder:
    """Build QUBO for assigning current-target blockers to destinations."""

    def build(self, instance: Instance, state: State, penalty_a: float, pair_weight: float = 1.0) -> Tuple[QuboModel, BatchRelocationQuboData]:
        """Build a batch-relocation assignment QUBO."""

        normalize_retrieval_state(instance, state)
        loc = find_box(state.stacks, state.next_retrieval_id)
        model = QuboModel()
        if loc is None:
            return model, BatchRelocationQuboData([], -1, [], {})
        source, level = loc
        blockers = list(reversed(state.stacks[source][level + 1 :]))
        destinations = [q for q, st in enumerate(state.stacks) if q != source and len(st) < instance.max_height]
        var_for: Dict[Tuple[int, int], int] = {}
        for a, box in enumerate(blockers):
            vars_for_box = []
            for q in destinations:
                v = model.add_binary_var(f"x_{a}_{q}")
                var_for[(a, q)] = v
                vars_for_box.append(v)
                single_cost = 0.0 if box < min_priority(state.stacks[q]) else 1.0
                model.add_linear(v, single_cost)
            if vars_for_box:
                model.add_one_hot_penalty(vars_for_box, penalty_a)
        for q in destinations:
            cap = instance.max_height - len(state.stacks[q])
            q_vars = [var_for[(a, q)] for a in range(len(blockers)) if (a, q) in var_for]
            if cap <= 1:
                model.add_at_most_one_penalty(q_vars, penalty_a)
            for a in range(len(blockers)):
                for c in range(a + 1, len(blockers)):
                    if (a, q) not in var_for or (c, q) not in var_for:
                        continue
                    ba, bc = blockers[a], blockers[c]
                    if bc > ba:
                        model.add_quadratic(var_for[(a, q)], var_for[(c, q)], pair_weight)
        return model, BatchRelocationQuboData(blockers, source, destinations, var_for)

    def decode(self, sample: Optional[QuboSample], data: BatchRelocationQuboData) -> Tuple[Dict[int, int], int]:
        """Decode blocker index -> destination, repairing missing/multiple choices."""

        assignment: Dict[int, int] = {}
        repairs = 0
        for a, _box in enumerate(data.blockers):
            chosen = []
            for q in data.destinations:
                v = data.var_for.get((a, q))
                if v is not None and sample is not None and sample.binary_values.get(v, 0) == 1:
                    chosen.append(q)
            if len(chosen) != 1:
                repairs += 1
            if chosen:
                assignment[a] = chosen[0]
        return assignment, repairs


class QuboBatchRelocationRetrievalSolver(Solver):
    """CRP solver using QUBO to batch-assign blockers above the current target."""

    name = "qubo_batch_relocation"

    def __init__(self, sampler: Optional[IQuboSampler] = None) -> None:
        self.sampler = sampler or SimulatedAnnealingQuboSampler()

    def solve(self, instance: Instance, config: SolverConfig):
        """Run QUBO batch relocation for retrieval CRP."""

        if instance.variant is not ProblemVariant.RETRIEVAL_CRP:
            return not_applicable(self.name)
        start = time.perf_counter()
        state = create_initial_state(instance)
        builder = QuboBatchRelocationBuilder()
        calls = vars_total = repairs = 0
        qtime = 0.0
        batch_sizes: List[int] = []
        qubo_time_limit = _qubo_time_limit(config)

        while not is_goal(instance, state) and state.move_count < config.max_depth:
            normalize_retrieval_state(instance, state)
            if is_goal(instance, state):
                break
            loc = find_box(state.stacks, state.next_retrieval_id)
            if loc is None:
                break
            source, level = loc
            if level == len(state.stacks[source]) - 1:
                normalize_retrieval_state(instance, state)
                continue
            model, data = builder.build(instance, state, config.qubo_penalty_a)
            batch_sizes.append(len(data.blockers))
            if not data.blockers:
                break
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
            assignment, repair = builder.decode(ss.samples[0] if ss.samples else None, data)
            repairs += repair
            for a, box in enumerate(data.blockers):
                if state.move_count >= config.max_depth:
                    break
                src = data.source
                dest = assignment.get(a)
                if dest is None or dest == src or len(state.stacks[dest]) >= instance.max_height:
                    dest = _greedy_retrieval_destination(instance, state, src, box)
                    repairs += 1
                if dest is None:
                    break
                if not state.stacks[src] or state.stacks[src][-1] != box:
                    repairs += 1
                    box = state.stacks[src][-1]
                state = apply_move(instance, state, Move(state.move_count, box, src, dest, len(state.stacks[src]), len(state.stacks[dest]) + 1))
            normalize_retrieval_state(instance, state)
        if is_goal(instance, state):
            sol = solution_from_state(
                self.name,
                instance,
                state,
                start,
                qubo_calls=calls,
                qubo_total_vars=vars_total,
                qubo_total_time_sec=qtime,
                qubo_repair_count=repairs,
            )
            if batch_sizes:
                sol.status += f"; avg_batch_size={sum(batch_sizes)/len(batch_sizes):.2f}"
            return sol
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
