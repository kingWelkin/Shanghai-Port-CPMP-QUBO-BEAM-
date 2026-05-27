"""Diversity-aware beam search."""

from __future__ import annotations

import time
from typing import List

from ..core import apply_move, create_initial_state, generate_legal_moves, hash_state, is_goal
from ..scoring import cpmp_business_feasible, dominance_value, score_state
from ..similarity import combined_similarity
from .base import SolverConfig, failure_solution, solution_from_state
from .beam import BeamSearchSolver, Candidate


class DiverseBeamSearchSolver(BeamSearchSolver):
    """Beam search that penalizes similarity among selected states."""

    name = "diverse_beam"

    def solve(self, instance, config: SolverConfig):
        """Run diversity-aware beam search."""

        start = time.perf_counter()
        initial = create_initial_state(instance)
        if is_goal(instance, initial):
            return solution_from_state(self.name, instance, initial, start, selected_avg_similarity=0.0)
        h0 = hash_state(instance, initial, config.enable_canonical_hash)
        beam = [Candidate(initial, score_state(instance, initial, config.score_weights), None, h0)]
        visited = {h0: dominance_value(instance, initial)}
        expanded = generated = 0
        last_similarity = None

        for _depth in range(config.max_depth):
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
                    sc = score_state(instance, ns, config.score_weights)
                    if is_goal(instance, ns):
                        return solution_from_state(
                            self.name,
                            instance,
                            ns,
                            start,
                            expanded_states=expanded,
                            generated_states=generated + len(candidates) + 1,
                            visited_states=len(visited),
                            selected_avg_similarity=last_similarity,
                        )
                    candidates.append(Candidate(ns, sc, mv, h))
            generated += len(candidates)
            if not candidates:
                break
            candidates.sort(key=lambda c: (c.score, c.state.move_count))
            candidates = candidates[: config.max_candidates_per_layer]
            beam = self._select_diverse(candidates, config.beam_width, config.diversity_weight)
            last_similarity = self._avg_similarity(beam)

        return failure_solution(
            self.name,
            start,
            "NO_FEASIBLE_FOUND",
            expanded_states=expanded,
            generated_states=generated,
            visited_states=len(visited),
            selected_avg_similarity=last_similarity,
        )

    @staticmethod
    def _select_diverse(candidates: List[Candidate], width: int, diversity_weight: float) -> List[Candidate]:
        if len(candidates) <= width:
            return list(candidates)
        scores = [c.score for c in candidates]
        lo, hi = min(scores), max(scores)

        def norm(c: Candidate) -> float:
            return 0.0 if hi == lo else (c.score - lo) / (hi - lo)

        selected = [candidates[0]]
        remaining = candidates[1:]
        while remaining and len(selected) < width:
            best_idx = 0
            best_value = float("inf")
            for idx, cand in enumerate(remaining):
                sim = sum(combined_similarity(cand.state, s.state) for s in selected) / len(selected)
                value = norm(cand) + diversity_weight * sim
                if value < best_value:
                    best_value = value
                    best_idx = idx
            selected.append(remaining.pop(best_idx))
        return selected

    @staticmethod
    def _avg_similarity(candidates: List[Candidate]) -> float:
        if len(candidates) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                total += combined_similarity(candidates[i].state, candidates[j].state)
                count += 1
        return total / count if count else 0.0
