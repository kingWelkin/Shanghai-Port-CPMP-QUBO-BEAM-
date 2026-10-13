"""Compose fixed release neighborhoods with actual classical warm starts."""
from __future__ import annotations

from dataclasses import asdict

import categorical_exact_control as exact
from categorical_exchange_controls import descend as exchange
from categorical_greedy_controls import descend as single
from categorical_qubo_search import solve
from conditioned_trajectory_domains import PhysicalTrace,ConditionedTrajectoryDomains
from complete_trajectory_domains import UnsupportedDomain
from hybrid_trajectory_refiner import no_milp
from production_selector_replay import REAL_TIME
from research_receiver_allocation import physical_replay


def classical_seed(model,pairs=True):
    started=REAL_TIME()
    singles=[single(model.unary,model.groups,model.pair,model.loads,model.caps,model.current,s)
             for s in ('best','forward','reverse')]
    if pairs:
        trials=[exchange(model.unary,model.groups,model.pair,model.loads,model.caps,start)
                for start in [model.current]+[chosen for chosen,meta in singles]]
    else:trials=singles
    chosen,meta=min(trials,key=lambda a:(float(model.unary[a[0]].sum()),a[0]))
    return chosen,dict(runtime_sec=REAL_TIME()-started,single_trials=[m for c,m in singles],
                       pair_trials=[m for c,m in trials] if pairs else [],selected_metadata=meta)


def run(instance,path,mode,releases):
    if mode not in ('single','pair','pair_qubo','pair_oracle'):raise ValueError(mode)
    started=REAL_TIME();current=tuple(path)
    physical_replay(instance.stacks,instance.max_height,current,goal=True)
    records=[];events=[]
    for k,release in enumerate(releases):
        before=len(current);began=REAL_TIME()
        try:
            with no_milp():
                trace=PhysicalTrace(instance.stacks,instance.max_height,current)
                model=ConditionedTrajectoryDomains(instance.stacks,instance.max_height,current,release['boxes'],trace=trace)
                q=model.compile_qubo()
        except UnsupportedDomain as error:
            records.append(dict(index=k,release=release,status='UNSUPPORTED',reason=str(error),before=before,
                                setup_sec=REAL_TIME()-began));continue
        setup=REAL_TIME()-began
        original=exact.milp
        def forbidden(*args,**kwargs):raise RuntimeError('MILP called in classical/QUBO refinement')
        try:
            exact.milp=forbidden
            with no_milp() as guard:
                selected,seed_meta=classical_seed(model,pairs=mode!='single')
                seed=model.replay(selected)
                if abs(q.evaluate(model.bits(selected))-len(seed))>1e-5:
                    raise ValueError('Classical seed fails matrix/physical validation')
                backend=None
                if mode=='pair_qubo':
                    selected,backend=solve(model.unary,model.groups,model.pair,model.loads,model.caps,
                        model.penalty,model.constant,selected,50000)
                calls=guard['calls']
        finally:exact.milp=original
        if mode=='pair_oracle':
            selected,backend=exact.solve(model.unary,model.groups,model.pair,model.loads,model.caps,selected,2.)
        candidate=model.replay(selected)
        energy=q.evaluate(model.bits(selected))
        if abs(energy-len(candidate))>1e-5 or len(candidate)>len(seed) or len(seed)>before:
            raise ValueError('Sequential refinement lost its feasible classical incumbent')
        if len(candidate)<before:
            current=candidate
            events.append(dict(index=k,before=before,after=len(current),seed_moves=len(seed),
                coupled_extra=len(seed)-len(candidate),ready_sec=REAL_TIME()-started,path=[asdict(m) for m in current]))
        records.append(dict(index=k,release=release,status='SUPPORTED',before=before,seed_moves=len(seed),after=len(current),
            setup_sec=setup,variables=len(q.var_index),model=model.stats,seed_metadata=seed_meta,
            backend=backend,nonoracle_milp_calls=calls,selected=list(map(int,selected)),energy=energy))
    physical_replay(instance.stacks,instance.max_height,current,goal=True)
    return dict(mode=mode,input=len(path),moves=len(current),path=[asdict(m) for m in current],
                events=events,models=records,elapsed_sec=REAL_TIME()-started,
                passes=1,converged=False,global_cpmp_optimal=False)
