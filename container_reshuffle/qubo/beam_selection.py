"""QUBO candidate selection for beam search."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from ..similarity import combined_similarity
from ..solvers.beam import Candidate
from .model import QuboModel
from .sampler import QuboSample


class QuboBeamSelectionBuilder:
    """Build and decode a QUBO that selects a diverse beam subset."""

    def build(
        self,
        candidates: List[Candidate],
        beam_width: int,
        diversity_weight: float,
        penalty_a: float,
        max_qubo_vars: int,
    ) -> Tuple[QuboModel, List[Candidate]]:
        """Build a beam-selection QUBO."""

        usable = sorted(candidates, key=lambda c: (c.score, c.state.move_count))[:max_qubo_vars]
        model = QuboModel()
        if not usable:
            return model, usable
        scores = [c.score for c in usable]
        lo, hi = min(scores), max(scores)
        vars_ = []
        for idx, cand in enumerate(usable):
            v = model.add_binary_var(f"u_{idx}")
            vars_.append(v)
            normalized = 0.0 if hi == lo else (cand.score - lo) / (hi - lo)
            model.add_linear(v, normalized)
        for i in range(len(usable)):
            for j in range(i + 1, len(usable)):
                sim = combined_similarity(usable[i].state, usable[j].state)
                model.add_quadratic(vars_[i], vars_[j], diversity_weight * sim)
        model.add_exactly_k_penalty(vars_, min(beam_width, len(usable)), penalty_a)
        return model, usable

    def decode_and_repair(
        self,
        sample: QuboSample | None,
        candidates: List[Candidate],
        beam_width: int,
        elite_keep_count: int = 0,
        diversity_weight: float = 0.0,
        repair_hamming_slack: int = 3,
        objective_aware_repair: bool = True,
    ) -> Tuple[List[Candidate], int]:
        """Decode one sample and repair it to a low-cost exact-size beam.

        The repair mirrors the hardware-QUBO workflow used in the sibling
        Ising project: API samples are treated as hints, then projected back to
        the hard cardinality constraint and locally improved under the original
        beam-selection objective.
        """

        selected, repair = self._decode_indices(
            sample,
            candidates,
            beam_width,
            elite_keep_count=elite_keep_count,
            diversity_weight=diversity_weight,
            repair_hamming_slack=repair_hamming_slack,
            objective_aware_repair=objective_aware_repair,
        )
        return [candidates[i] for i in selected], repair

    def decode_samples_and_repair(
        self,
        samples: Sequence[QuboSample] | None,
        candidates: List[Candidate],
        beam_width: int,
        elite_keep_count: int = 0,
        diversity_weight: float = 0.0,
        repair_hamming_slack: int = 3,
        objective_aware_repair: bool = True,
    ) -> Tuple[List[Candidate], int]:
        """Repair all samples and return the repaired selection with best objective."""

        sample_list = list(samples or [None])
        best_selected: List[int] | None = None
        best_repair = 0
        best_value = float("inf")
        for sample in sample_list:
            selected, repair = self._decode_indices(
                sample,
                candidates,
                beam_width,
                elite_keep_count=elite_keep_count,
                diversity_weight=diversity_weight,
                repair_hamming_slack=repair_hamming_slack,
                objective_aware_repair=objective_aware_repair,
            )
            value = _selection_objective(candidates, selected, diversity_weight)
            tie = sum(candidates[i].score for i in selected)
            current = (value, tie, repair)
            incumbent = (
                best_value,
                sum(candidates[i].score for i in best_selected) if best_selected is not None else float("inf"),
                best_repair,
            )
            if current < incumbent:
                best_selected = selected
                best_repair = repair
                best_value = value
        if best_selected is None:
            return [], 0
        return [candidates[i] for i in best_selected], best_repair

    def _decode_indices(
        self,
        sample: QuboSample | None,
        candidates: List[Candidate],
        beam_width: int,
        *,
        elite_keep_count: int,
        diversity_weight: float,
        repair_hamming_slack: int,
        objective_aware_repair: bool,
    ) -> Tuple[List[int], int]:
        if not candidates:
            return [], 0
        repair = 0
        target = min(beam_width, len(candidates))
        selected_idx: List[int] = []
        if sample is not None:
            selected_idx = [i for i in range(len(candidates)) if sample.binary_values.get(i, 0) == 1]
        selected_set = set(selected_idx)
        elite_keep_count = max(0, min(elite_keep_count, target, len(candidates)))
        protected = set()
        if elite_keep_count:
            elite_idx = sorted(range(len(candidates)), key=lambda i: (candidates[i].score, i))[:elite_keep_count]
            protected = set(elite_idx)
            missing_elite = [i for i in elite_idx if i not in selected_set]
            if missing_elite:
                repair += 1
                selected_idx.extend(missing_elite)
                selected_set.update(missing_elite)
        if len(selected_idx) > target:
            repair += 1
            selected_idx = _drop_to_target(candidates, selected_idx, target, protected, diversity_weight)
        if len(selected_idx) < target:
            repair += 1
            selected_idx = _fill_to_target(candidates, selected_idx, target, diversity_weight)
        if not selected_idx:
            repair += 1
            selected_idx = list(range(target))
        if objective_aware_repair and target > 0:
            improved, changed = _swap_local_repair(
                candidates,
                selected_idx,
                protected,
                diversity_weight,
                max_swaps=max(0, int(repair_hamming_slack)),
            )
            if changed:
                repair += 1
                selected_idx = improved
        selected_idx = sorted(set(selected_idx), key=lambda i: (candidates[i].score, i))[:target]
        return selected_idx, repair


def _drop_to_target(
    candidates: List[Candidate],
    selected: Iterable[int],
    target: int,
    protected: set[int],
    diversity_weight: float,
) -> List[int]:
    """Remove selected indices until cardinality is exact, using objective deltas."""

    selected_set = set(selected)
    while len(selected_set) > target:
        removable = [i for i in selected_set if i not in protected]
        if not removable:
            removable = list(selected_set)
        remove = min(
            removable,
            key=lambda i: (
                _selection_objective(candidates, selected_set - {i}, diversity_weight),
                -candidates[i].score,
                i,
            ),
        )
        selected_set.remove(remove)
    return list(selected_set)


def _fill_to_target(
    candidates: List[Candidate],
    selected: Iterable[int],
    target: int,
    diversity_weight: float,
) -> List[int]:
    """Add indices until cardinality is exact, using objective deltas."""

    selected_set = set(selected)
    while len(selected_set) < target:
        missing = [i for i in range(len(candidates)) if i not in selected_set]
        if not missing:
            break
        add = min(
            missing,
            key=lambda i: (
                _selection_objective(candidates, selected_set | {i}, diversity_weight),
                candidates[i].score,
                i,
            ),
        )
        selected_set.add(add)
    return list(selected_set)


def _swap_local_repair(
    candidates: List[Candidate],
    selected: Iterable[int],
    protected: set[int],
    diversity_weight: float,
    max_swaps: int,
) -> Tuple[List[int], bool]:
    """Improve an exact-cardinality selection with objective-aware 1-swaps."""

    selected_set = set(selected)
    changed = False
    current = _selection_objective(candidates, selected_set, diversity_weight)
    for _ in range(max_swaps):
        removable = [i for i in selected_set if i not in protected]
        addable = [i for i in range(len(candidates)) if i not in selected_set]
        if not removable or not addable:
            break
        best_move = None
        best_value = current
        for remove in removable:
            base = selected_set - {remove}
            for add in addable:
                value = _selection_objective(candidates, base | {add}, diversity_weight)
                if value + 1e-12 < best_value:
                    best_value = value
                    best_move = (remove, add)
        if best_move is None:
            break
        selected_set.remove(best_move[0])
        selected_set.add(best_move[1])
        current = best_value
        changed = True
    return list(selected_set), changed


def _selection_objective(
    candidates: List[Candidate],
    selected: Iterable[int],
    diversity_weight: float,
) -> float:
    """Evaluate the non-penalty beam-selection objective for selected indices."""

    selected_idx = list(selected)
    if not selected_idx:
        return 0.0
    scores = [c.score for c in candidates]
    lo, hi = min(scores), max(scores)
    total = 0.0
    for i in selected_idx:
        total += 0.0 if hi == lo else (candidates[i].score - lo) / (hi - lo)
    if diversity_weight:
        for pos, i in enumerate(selected_idx):
            for j in selected_idx[pos + 1 :]:
                total += diversity_weight * combined_similarity(candidates[i].state, candidates[j].state)
    return float(total)
