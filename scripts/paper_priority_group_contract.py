"""Scoped CPMP priority-group semantics for previously unique-ID research helpers."""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import benchmark_compact_beam_qubo_n0 as beam
import open_cpmp_benchmarks as opened
from container_reshuffle.core import State, ProblemVariant, future_disorder_count_stacks


def is_group(instance):
    return instance.variant is ProblemVariant.PREMARSHALLING_CPMP and opened.requires_priority_group_semantics(instance)


class PriorityGroupContract:
    """Preserve labels, physical moves and coordinates; delegate unique-ID behavior."""
    def __init__(self):
        self.originals = {}

    def install(self):
        if self.originals:
            raise RuntimeError('Priority-group contract already installed')
        for name, callback in {
            'create_initial_state': self.initial,
            'validate_solution': self.validate,
            'max_entropy_along_moves': self.peak,
        }.items():
            self.originals[name] = getattr(beam, name)
            setattr(beam, name, callback)

    def restore(self):
        for name, original in self.originals.items():
            setattr(beam, name, original)
        self.originals.clear()

    def initial(self, instance):
        if not is_group(instance):
            return self.originals['create_initial_state'](instance)
        opened._validate_open_shape(instance, Path(instance.instance_id))
        stacks = [list(s) for s in instance.stacks]
        n = future_disorder_count_stacks(stacks)
        return State(stacks=stacks, disorder_current=n, disorder_total=n, entropy_max=n)

    def validate(self, instance, solution):
        if not is_group(instance):
            return self.originals['validate_solution'](instance, solution)
        try:
            opened._validate_open_shape(instance, Path(instance.instance_id))
            ok, msg, stacks, _, _ = opened._simulate_priority_group_moves(instance, solution.moves)
            if ok and math.isfinite(solution.objective) and len(solution.moves) != int(round(solution.objective)):
                return False, 'move count does not equal objective', stacks
            return ok, msg, stacks
        except (ValueError, RuntimeError, IndexError) as ex:
            return False, str(ex), [list(s) for s in instance.stacks]

    def peak(self, instance, moves):
        if not is_group(instance):
            return self.originals['max_entropy_along_moves'](instance, moves)
        opened._validate_open_shape(instance, Path(instance.instance_id))
        normalized = [replace(m, step=i) for i,m in enumerate(moves)]
        _, msg, _, _, peak = opened._simulate_priority_group_moves(instance, normalized)
        if msg not in ('ok', 'final state is not a priority-group goal'):
            raise ValueError(msg)
        return peak
