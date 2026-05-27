"""QUBO-assisted solvers."""

from __future__ import annotations

import time
from typing import List

from ..core import apply_move, create_initial_state, generate_legal_moves, hash_state, is_goal
from ..scoring import cpmp_business_feasible, dominance_value, score_state
from ..similarity import combined_similarity
from ..solvers.base import SolverConfig, failure_solution, solution_from_state
from ..solvers.beam import BeamSearchSolver, Candidate
from .beam_selection import QuboBeamSelectionBuilder
from .sampler import IQuboSampler, QuboSamplerConfig, SimulatedAnnealingQuboSampler


class QuboBeamSearchSolver(BeamSearchSolver):
    """Beam search using QUBO to select each layer's beam."""

    name = "qubo_beam"

    def __init__(self, sampler: IQuboSampler | None = None, elite_keep_ratio: float = 0.0) -> None:
        self.sampler = sampler or SimulatedAnnealingQuboSampler()
        self.elite_keep_ratio = max(0.0, min(1.0, elite_keep_ratio))

    def solve(self, instance, config: SolverConfig):
        """Run QUBO-assisted beam search."""

        start = time.perf_counter()
        initial = create_initial_state(instance)
        if is_goal(instance, initial):
            return solution_from_state(self.name, instance, initial, start)
        h0 = hash_state(instance, initial, config.enable_canonical_hash)
        beam = [Candidate(initial, score_state(instance, initial, config.score_weights), None, h0)]
        visited = {h0: dominance_value(instance, initial)}
        expanded = generated = qubo_calls = qubo_vars = repairs = 0
        qubo_time = 0.0
        selected_sims: List[float] = []
        sampler_names: List[str] = []
        builder = QuboBeamSelectionBuilder()

        call_interval = max(1, int(getattr(config, "qubo_call_interval", 1)))
        min_candidates_for_call = max(0, int(getattr(config, "qubo_min_candidates_for_call", 0)))
        qubo_time_limit = _qubo_time_limit(config)

        for depth in range(config.max_depth):
            candidates: List[Candidate] = []
            for cand in beam:
                expanded += 1
                moves = generate_legal_moves(instance, cand.state)
                if config.candidate_limit_per_state is not None:
                    moves = self._ordered_moves(instance, cand.state, moves, config)[: config.candidate_limit_per_state]
                for mv in moves:
                    ns = apply_move(instance, cand.state, mv)
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
                            qubo_calls=qubo_calls,
                            qubo_total_vars=qubo_vars,
                            qubo_total_time_sec=qubo_time,
                            qubo_repair_count=repairs,
                            selected_avg_similarity=_mean(selected_sims),
                            status=_sampler_status(sampler_names),
                        )
                    candidates.append(Candidate(ns, score_state(instance, ns, config.score_weights), mv, h))
            generated += len(candidates)
            if not candidates:
                break
            candidates.sort(key=lambda c: c.score)
            candidates = candidates[: config.max_candidates_per_layer]
            use_qubo = depth % call_interval == 0 and len(candidates) >= min_candidates_for_call
            if not use_qubo:
                beam = candidates[: config.beam_width]
                selected_sims.append(_avg_similarity(beam))
                continue
            model, usable = builder.build(
                candidates,
                config.beam_width,
                config.diversity_weight,
                config.qubo_penalty_a,
                config.max_qubo_vars,
            )
            qubo_calls += 1
            qubo_vars += len(model.var_index)
            ss = self.sampler.sample(
                model,
                QuboSamplerConfig(
                    num_reads=config.qubo_num_reads,
                    random_seed=config.random_seed + qubo_calls,
                    time_limit_sec=qubo_time_limit,
                    sa_iterations=max(1, int(getattr(config, "qubo_sa_iterations", 5000))),
                ),
            )
            qubo_time += ss.runtime_sec
            sampler_names.append(ss.sampler_name)
            elite_keep_count = int(round(config.beam_width * self.elite_keep_ratio))
            beam, repair = builder.decode_samples_and_repair(
                ss.samples,
                usable,
                config.beam_width,
                elite_keep_count=elite_keep_count,
                diversity_weight=config.diversity_weight,
                repair_hamming_slack=int(getattr(config, "qubo_repair_hamming_slack", 3)),
            )
            repairs += repair
            selected_sims.append(_avg_similarity(beam))

        return failure_solution(
            self.name,
            start,
            _sampler_status(sampler_names, "NO_FEASIBLE_FOUND"),
            expanded_states=expanded,
            generated_states=generated,
            visited_states=len(visited),
            qubo_calls=qubo_calls,
            qubo_total_vars=qubo_vars,
            qubo_total_time_sec=qubo_time,
            qubo_repair_count=repairs,
            selected_avg_similarity=_mean(selected_sims),
        )


def _avg_similarity(candidates: List[Candidate]) -> float:
    if len(candidates) < 2:
        return 0.0
    total = count = 0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            total += combined_similarity(candidates[i].state, candidates[j].state)
            count += 1
    return total / count if count else 0.0


def _mean(values: List[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _qubo_time_limit(config: SolverConfig) -> float:
    value = getattr(config, "qubo_time_limit_sec", None)
    if value is None:
        value = min(5.0, config.time_limit_sec)
    return max(0.001, float(value))


def _sampler_status(sampler_names: List[str], base: str = "FEASIBLE") -> str:
    if not sampler_names:
        return base
    unique = sorted(set(sampler_names))
    return f"{base}; samplers={','.join(unique)}"
