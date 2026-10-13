from pathlib import Path
import sys
import itertools
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from categorical_budgeted_qubo import solve
from categorical_propagating_qubo import solve as reference


def models():
    rng=np.random.default_rng(20261012)
    for _ in range(40):
        groups=[list(range(g*3,g*3+3)) for g in range(5)];n=15
        unary=rng.integers(0,10,n);inc=[g[0] for g in groups]
        pair=np.triu(rng.random((n,n))<.15,1);pair|=pair.T.copy()
        for a in inc:
            for b in inc:pair[a,b]=False
        loads=rng.integers(0,3,(n,3));caps=loads[inc].sum(axis=0)
        yield unary,groups,pair,loads,caps,int(unary[inc].sum())+1,7,inc


def test_no_deadline_matches_all_original_choices_and_certificates():
    for args in models():
        chosen,meta=solve(*args);old,ref=reference(*args)
        assert chosen==old and meta['energy']==ref['energy'] and meta['optimal']==ref['optimal']
        assert meta['nodes']==ref['nodes'] and meta['propagation_rounds']==ref['propagation_rounds']


def test_deadline_cutoffs_always_return_a_feasible_nonworse_energy():
    for args in models():
        unary,groups,pair,loads,caps,penalty,constant,inc=args
        optimum=reference(*args)[1]['energy']
        for limit in (0,1,5,20,60):
            ticks=[0]
            def clock():ticks[0]+=1;return ticks[0]
            chosen,meta=solve(*args,deadline=limit,clock=clock)
            assert all(i in g for i,g in zip(chosen,groups))
            assert not any(pair[a,b] for a,b in itertools.combinations(chosen,2))
            assert np.all(loads[chosen].sum(axis=0)<=caps)
            assert meta['energy']==constant+unary[chosen].sum()<=constant+unary[inc].sum()
            if meta.get('wall_cutoff'):assert not meta['optimal']
            if meta['optimal']:assert meta['energy']==optimum


def test_improvement_found_before_deadline_is_kept():
    # Independent cheap domains give a strict feasible improvement assoon as
    # the first complete assignment is reached;later deadline cuts keep it.
    args=(np.array([3,2,0]*5),[list(range(g*3,g*3+3)) for g in range(5)],
        np.zeros((15,15),dtype=bool),np.zeros((15,0),dtype=int),np.zeros(0,dtype=int),16,0,[0,3,6,9,12])
    found=False
    for limit in range(1,200):
        ticks=[0]
        def clock():ticks[0]+=1;return ticks[0]
        chosen,meta=solve(*args,deadline=limit,clock=clock)
        if meta.get('wall_cutoff') and meta['energy']<15:found=True
    assert found


def test_capped_controller_restores_hooks_and_returns_valid_full_paths():
    from test_bounded_conditioned_refinement import example
    import bounded_conditioned_refinement as original
    from capped_conditioned_refinement import run
    instance,path,releases=example();saved=(original.solve,original.exact.solve,original.exact.milp)
    for mode in ('capped_qubo_frg','capped_oracle_frg'):
        published=[]
        result=run(instance,path,mode,time.perf_counter(),2.,published.append,releases=releases,continue_frg=False)
        assert result['moves']==2 and (original.solve,original.exact.solve,original.exact.milp)==saved
        assert all(r['nonoracle_milp_calls']==0 for r in result['models'])
        assert any(e['type']=='INCUMBENT' and e['source']=='pair_seed' for e in published)
