from pathlib import Path
import itertools
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from conditioned_trajectory_domains import ConditionedTrajectoryDomains,PhysicalTrace
from complete_trajectory_domains import UnsupportedDomain
from factorized_trajectory_qubo import FactorizedTrajectoryQubo,FactorizedConditionedDomains
from categorical_propagating_qubo import solve
from test_trajectory_subsequence_qubo import sample


def test_all_binary_energies_including_invalid_choices_and_slack_match_explicit_qubo():
    rng=np.random.default_rng(98731)
    for seed in range(10):
        stacks,path=sample(seed,steps=5);trace=PhysicalTrace(stacks,3,path)
        released=[b for b,events in enumerate(trace.events) if events]
        model=ConditionedTrajectoryDomains(stacks,3,path,released,trace=trace)
        explicit=model.compile_qubo();factored=FactorizedTrajectoryQubo(model)
        exported=factored.materialize()
        assert factored.var_index==explicit.var_index==exported.var_index
        assert factored.offset==explicit.offset==exported.offset
        assert exported.Q==explicit.Q
        n=len(explicit.var_index)
        assignments=itertools.product((0,1),repeat=n) if n<=12 else rng.integers(0,2,size=(300,n))
        for bits in assignments:
            values=dict(enumerate(map(int,bits)))
            assert factored.evaluate(values)==explicit.evaluate(values)==exported.evaluate(values)
        # Missing bits retain the explicit evaluator's zero-default behavior.
        assert factored.evaluate({})==explicit.evaluate({})


def test_factorized_adapter_preserves_seed_bits_backend_energy_and_physical_output():
    for seed in range(8):
        stacks,path=sample(seed,steps=6);trace=PhysicalTrace(stacks,3,path)
        released=[b for b,events in enumerate(trace.events) if events]
        dense=ConditionedTrajectoryDomains(stacks,3,path,released,trace=trace)
        fast=FactorizedConditionedDomains(stacks,3,path,released,trace=trace)
        q0=dense.compile_qubo();q1=fast.compile_qubo()
        outputs=[]
        for m,q in ((dense,q0),(fast,q1)):
            chosen,meta=solve(m.unary,m.groups,m.pair,m.loads,m.caps,m.penalty,m.constant,m.current)
            physical=m.replay(chosen)
            assert q.evaluate(m.bits(chosen))==len(physical)==meta['energy']
            outputs.append((chosen,physical,m.bits(chosen)))
        assert outputs[0]==outputs[1]


def test_same_whole_variable_gate_and_explicit_binary_scope():
    stacks,path=sample(4,steps=8);trace=PhysicalTrace(stacks,3,path)
    released=[b for b,events in enumerate(trace.events) if events]
    model=ConditionedTrajectoryDomains(stacks,3,path,released,trace=trace)
    dense=model.compile_qubo();n=len(dense.var_index)
    assert n>0
    with pytest.raises(UnsupportedDomain,match='REDUCED_QUBO_OVERSIZE'):
        FactorizedTrajectoryQubo(model,max_variables=n-1)
    with pytest.raises(ValueError,match='binary'):
        FactorizedTrajectoryQubo(model).evaluate({0:2})
