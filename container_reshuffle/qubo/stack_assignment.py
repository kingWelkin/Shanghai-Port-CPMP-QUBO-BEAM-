"""QUBO route D: CPMP target stack assignment guidance."""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from ..core import Instance, Move, ProblemVariant, State, apply_move, create_initial_state, generate_legal_moves, hash_state, is_goal
from ..scoring import cpmp_business_feasible, dominance_value, score_state
from ..solvers.base import Solver, SolverConfig, failure_solution, not_applicable, solution_from_state
from .model import QuboModel
from .sampler import IQuboSampler, QuboSamplerConfig, QuboSample, SimulatedAnnealingQuboSampler


class QuboStackAssignmentBuilder:
    """Build QUBO assigning each container to a target final stack."""

    def build(
        self,
        instance: Instance,
        state: State,
        penalty_a: float,
        keep_weight: float = 1.0,
        order_weight: float = 0.2,
    ) -> Tuple[QuboModel, Dict[Tuple[int, int], int]]:
        """Build stack-assignment QUBO."""

        current_stack = {box: i for i, st in enumerate(state.stacks) for box in st}
        model = QuboModel()
        var_for: Dict[Tuple[int, int], int] = {}
        for p in range(1, instance.container_count + 1):
            vars_p = []
            for i in range(instance.stack_count):
                v = model.add_binary_var(f"x_{p}_{i}")
                var_for[(p, i)] = v
                vars_p.append(v)
                model.add_linear(v, keep_weight * (0 if current_stack.get(p) == i else 1))
            model.add_one_hot_penalty(vars_p, penalty_a)
        for i in range(instance.stack_count):
            vars_i = [var_for[(p, i)] for p in range(1, instance.container_count + 1)]
            if instance.max_height <= 1:
                model.add_at_most_one_penalty(vars_i, penalty_a)
        for stack in state.stacks:
            pos = {box: idx for idx, box in enumerate(stack)}
            for p in stack:
                for q in stack:
                    if p < q and pos[p] < pos[q]:
                        # p is currently below q, opposite of final target if grouped.
                        for i in range(instance.stack_count):
                            model.add_quadratic(var_for[(p, i)], var_for[(q, i)], order_weight)
        return model, var_for

    def decode_and_repair(
        self,
        instance: Instance,
        sample: Optional[QuboSample],
        var_for: Dict[Tuple[int, int], int],
    ) -> Tuple[Dict[int, int], int]:
        """Decode target stack for each container and repair capacity overflow."""

        target: Dict[int, int] = {}
        repairs = 0
        for p in range(1, instance.container_count + 1):
            chosen = [
                i
                for i in range(instance.stack_count)
                if sample is not None and sample.binary_values.get(var_for[(p, i)], 0) == 1
            ]
            if len(chosen) != 1:
                repairs += 1
                chosen = [min(range(instance.stack_count), key=lambda i: var_for[(p, i)])]
            target[p] = chosen[0]
        loads = {i: 0 for i in range(instance.stack_count)}
        for i in target.values():
            loads[i] += 1
        underfull = [i for i, load in loads.items() for _ in range(instance.max_height - load) if load < instance.max_height]
        for i in range(instance.stack_count):
            boxes = [p for p, q in target.items() if q == i]
            while len(boxes) > instance.max_height and underfull:
                p = boxes.pop()
                target[p] = underfull.pop(0)
                repairs += 1
        return target, repairs


class StackAssignmentGuidedPremarshallingSolver(Solver):
    """CPMP beam search guided by a QUBO target stack assignment."""

    name = "stack_assignment_guided_cpmp"

    def __init__(self, sampler: Optional[IQuboSampler] = None, guide_weight: float = 0.5) -> None:
        self.sampler = sampler or SimulatedAnnealingQuboSampler()
        self.guide_weight = guide_weight

    def solve(self, instance: Instance, config: SolverConfig):
        """Run guided CPMP beam search."""

        if instance.variant is not ProblemVariant.PREMARSHALLING_CPMP:
            return not_applicable(self.name)
        start = time.perf_counter()
        initial = create_initial_state(instance)
        builder = QuboStackAssignmentBuilder()
        model, var_for = builder.build(instance, initial, config.qubo_penalty_a)
        ss = self.sampler.sample(
            model,
            QuboSamplerConfig(
                config.qubo_num_reads,
                config.random_seed,
                _qubo_time_limit(config),
                sa_iterations=max(1, int(getattr(config, "qubo_sa_iterations", 5000))),
            ),
        )
        target, repairs = builder.decode_and_repair(instance, ss.samples[0] if ss.samples else None, var_for)
        beam = [initial]
        visited = {hash_state(instance, initial): dominance_value(instance, initial)}
        expanded = generated = 0

        for _depth in range(config.max_depth):
            candidates = []
            for state in beam:
                expanded += 1
                moves = generate_legal_moves(instance, state)
                moves.sort(key=lambda mv: (0 if target.get(mv.box) == mv.to_stack else 1, mv.to_stack))
                if config.candidate_limit_per_state is not None:
                    moves = moves[: config.candidate_limit_per_state]
                for mv in moves:
                    ns = apply_move(instance, state, mv)
                    if not cpmp_business_feasible(instance, ns, config.entropy_alpha):
                        continue
                    h = hash_state(instance, ns, config.enable_canonical_hash)
                    dom = dominance_value(instance, ns)
                    if config.enable_visited_pruning and visited.get(h, float("inf")) <= dom:
                        continue
                    visited[h] = dom
                    if is_goal(instance, ns):
                        return solution_from_state(
                            self.name,
                            instance,
                            ns,
                            start,
                            expanded_states=expanded,
                            generated_states=generated + len(candidates) + 1,
                            visited_states=len(visited),
                            qubo_calls=1,
                            qubo_total_vars=len(model.var_index),
                            qubo_total_time_sec=ss.runtime_sec,
                            qubo_repair_count=repairs,
                        )
                    candidates.append((self._guided_score(instance, ns, config, target), ns))
            generated += len(candidates)
            if not candidates:
                break
            candidates.sort(key=lambda x: x[0])
            beam = [state for _, state in candidates[: config.beam_width]]
        return failure_solution(
            self.name,
            start,
            "NO_FEASIBLE_FOUND",
            expanded_states=expanded,
            generated_states=generated,
            visited_states=len(visited),
            qubo_calls=1,
            qubo_total_vars=len(model.var_index),
            qubo_total_time_sec=ss.runtime_sec,
            qubo_repair_count=repairs,
        )

    def _guided_score(self, instance: Instance, state: State, config: SolverConfig, target: Dict[int, int]) -> float:
        current = {box: i for i, st in enumerate(state.stacks) for box in st}
        off_target = sum(1 for box, stack in current.items() if target.get(box) != stack)
        return score_state(instance, state, config.score_weights) + self.guide_weight * off_target


def _qubo_time_limit(config: SolverConfig) -> float:
    value = getattr(config, "qubo_time_limit_sec", None)
    if value is None:
        value = min(5.0, config.time_limit_sec)
    return max(0.001, float(value))
