from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from categorical_propagating_qubo import solve
from complete_trajectory_domains import UnsupportedDomain
from conditioned_trajectory_domains import PhysicalTrace
from factorized_trajectory_qubo import FactorizedConditionedDomains
from fast_reduced_trajectory_domains import FastReducedConditionedDomains
from test_trajectory_subsequence_qubo import sample


def test_exact_factors_energy_and_physical_optima_including_repeated_priorities():
    rng = np.random.default_rng(20261012)
    for seed in range(12):
        stacks, path = sample(seed, steps=7 + seed % 4)
        if seed % 2:
            stacks = tuple(tuple(value // 2 for value in stack) for stack in stacks)
            path = tuple(replace(move, box=move.box // 2) for move in path)
        trace = PhysicalTrace(stacks, 3, path)
        released = [box for box, events in enumerate(trace.events) if events]
        old = FactorizedConditionedDomains(stacks, 3, path, released, trace=trace)
        new = FastReducedConditionedDomains(stacks, 3, path, released, trace=trace)
        assert old.routes == new.routes
        assert old.domains == new.domains
        assert old.route_ids == new.route_ids
        assert old.current == new.current
        assert old.capacity == new.capacity
        assert old.constant == new.constant and old.penalty == new.penalty
        for name in ('unary', 'pair', 'loads', 'caps', 'occupancy'):
            np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
        q0, q1 = old.compile_qubo(), new.compile_qubo()
        assert q0.var_index == q1.var_index
        assert q0.offset == q1.offset
        for vector in rng.integers(0, 2, size=(96, len(q0.var_index))):
            bits = dict(enumerate(map(int, vector)))
            assert q0.evaluate(bits) == q1.evaluate(bits)
        outputs = []
        for model in (old, new):
            chosen, meta = solve(model.unary, model.groups, model.pair, model.loads,
                                 model.caps, model.penalty, model.constant, model.current)
            physical = model.replay(chosen)
            assert model.qubo.evaluate(model.bits(chosen)) == len(physical) == meta['energy']
            outputs.append((chosen, physical, meta['nodes'], meta['optimal']))
        assert outputs[0] == outputs[1]


def test_same_resource_gate_including_route_and_slack_overflow():
    seen_slack_gate = False
    for seed in range(20):
        stacks, path = sample(seed, steps=9)
        trace = PhysicalTrace(stacks, 3, path)
        released = [box for box, events in enumerate(trace.events) if events]
        old = FactorizedConditionedDomains(stacks, 3, path, released, trace=trace)
        count = len(old.compile_qubo().var_index)
        route_count = len(old.route_ids)
        for limit in {0, max(0, route_count - 1), max(0, count - 1), count}:
            if count > limit:
                with pytest.raises(UnsupportedDomain, match='REDUCED_QUBO_OVERSIZE'):
                    old.compile_qubo(max_variables=limit)
                with pytest.raises(UnsupportedDomain, match='REDUCED_QUBO_OVERSIZE'):
                    FastReducedConditionedDomains(stacks, 3, path, released, trace=trace, max_variables=limit)
                seen_slack_gate |= route_count <= limit < count
            else:
                new = FastReducedConditionedDomains(stacks, 3, path, released, trace=trace, max_variables=limit)
                assert len(new.compile_qubo().var_index) == count
    assert seen_slack_gate


def test_oversize_route_domain_rejected_before_conflict_matrix(monkeypatch):
    stacks, path = sample(4, steps=8)
    trace = PhysicalTrace(stacks, 3, path)
    released = [box for box, events in enumerate(trace.events) if events]
    old = FactorizedConditionedDomains(stacks, 3, path, released, trace=trace)
    assert len(old.route_ids) > 0
    original = np.zeros
    def checked(shape, *args, **kwargs):
        assert shape != (len(old.route_ids), len(old.route_ids)), 'Oversize quadratic matrix allocated'
        return original(shape, *args, **kwargs)
    monkeypatch.setattr(np, 'zeros', checked)
    with pytest.raises(UnsupportedDomain, match='REDUCED_QUBO_OVERSIZE'):
        FastReducedConditionedDomains(stacks, 3, path, released, trace=trace,
                                      max_variables=len(old.route_ids) - 1)
