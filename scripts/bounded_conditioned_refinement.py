"""Anytime conditioned route refinement with an externally enforced deadline."""
from __future__ import annotations

from dataclasses import asdict,replace

import categorical_exact_control as exact
from categorical_propagating_qubo import solve
from complete_trajectory_domains import UnsupportedDomain
from conditioned_trajectory_domains import PhysicalTrace,contact_releases
from factorized_trajectory_qubo import FactorizedConditionedDomains
from hybrid_trajectory_refiner import no_milp
from sequential_conditioned_refiner import classical_seed
from anytime_trajectory_refinement import frg_descent,bs_frg_continuation
from production_selector_replay import REAL_TIME
from research_receiver_allocation import physical_replay

MODES=('pair_frg','qubo_frg','oracle_frg','frg','bs_frg')


class PublishedTape:
    def __init__(self,instance,path,start,budget,publish):
        self.instance=instance;self.start=start;self.budget=budget;self.deadline=start+budget
        self.best=tuple(path);self.initial=len(path);self.events=[];self.late=[];self.publish=publish
        physical_replay(instance.stacks,instance.max_height,self.best,goal=True)

    def offer(self,path,source):
        path=tuple(replace(m,step=i) for i,m in enumerate(path))
        if len(path)>=len(self.best):return False
        physical_replay(self.instance.stacks,self.instance.max_height,path,goal=True)
        event=dict(type='INCUMBENT',ready_sec=REAL_TIME()-self.start,moves=len(path),
                   source=source,path=[asdict(m) for m in path])
        if REAL_TIME()>self.deadline:
            self.late.append(dict(ready_sec=event['ready_sec'],moves=len(path),source=source));return False
        self.best=path;self.events.append(event);self.publish(event)
        return True

    def phase(self,phase,**fields):
        self.publish(dict(type='PHASE',phase=phase,elapsed_sec=REAL_TIME()-self.start,**fields))


def run(instance,path,mode,start,budget,publish,*,releases=None,continue_frg=True):
    if mode not in MODES:raise ValueError(mode)
    tape=PublishedTape(instance,path,start,budget,publish);records=[];stages={}
    if mode in ('pair_frg','qubo_frg','oracle_frg') and REAL_TIME()<tape.deadline:
        tape.phase('release_selection')
        if releases is None:releases=contact_releases(PhysicalTrace(instance.stacks,instance.max_height,path),8,12)
        for k,release in enumerate(releases):
            if REAL_TIME()>=tape.deadline:break
            before=len(tape.best);began=REAL_TIME();tape.phase('domain',index=k,before=before)
            try:
                with no_milp():
                    trace=PhysicalTrace(instance.stacks,instance.max_height,tape.best)
                    model=FactorizedConditionedDomains(instance.stacks,instance.max_height,tape.best,release['boxes'],trace=trace)
                    q=model.compile_qubo()
            except UnsupportedDomain as error:
                records.append(dict(index=k,status='UNSUPPORTED',reason=str(error),setup_sec=REAL_TIME()-began));continue
            setup=REAL_TIME()-began
            if REAL_TIME()>=tape.deadline:break
            tape.phase('pair_seed',index=k,variables=len(q.var_index));original=exact.milp
            def forbidden(*args,**kwargs):raise RuntimeError('MILP called in pair/QUBO refinement')
            try:
                exact.milp=forbidden
                with no_milp() as guard:
                    chosen,seed_meta=classical_seed(model,pairs=True)
                    seed=model.replay(chosen)
                    if q.evaluate(model.bits(chosen))!=len(seed) or len(seed)>before:
                        raise ValueError('Invalid classical seed energy/path')
                    # Publish the actually obtained seed before any additional search.
                    tape.offer(seed,'pair_seed');backend=None
                    if mode=='qubo_frg' and REAL_TIME()<tape.deadline:
                        tape.phase('qubo_search',index=k,seed_moves=len(seed))
                        chosen,backend=solve(model.unary,model.groups,model.pair,model.loads,model.caps,
                            model.penalty,model.constant,chosen,50000)
                    calls=guard['calls']
            finally:exact.milp=original
            if mode=='oracle_frg' and REAL_TIME()<tape.deadline:
                tape.phase('milp_search',index=k,seed_moves=len(seed))
                chosen,backend=exact.solve(model.unary,model.groups,model.pair,model.loads,model.caps,
                    chosen,max(.001,min(2.,tape.deadline-REAL_TIME())))
            if REAL_TIME()>=tape.deadline:break
            candidate=model.replay(chosen);energy=q.evaluate(model.bits(chosen))
            if energy!=len(candidate) or len(candidate)>len(seed):raise ValueError('Invalid joint energy/path')
            tape.offer(candidate,'qubo_joint' if mode=='qubo_frg' else 'milp_joint' if mode=='oracle_frg' else 'pair')
            record=dict(index=k,status='SUPPORTED',before=before,seed_moves=len(seed),after=len(tape.best),
                setup_sec=setup,variables=len(q.var_index),model=model.stats,seed_metadata=seed_meta,
                backend=backend,nonoracle_milp_calls=calls,selected=list(map(int,chosen)),energy=energy)
            records.append(record);tape.publish(dict(type='MODEL',record=record,elapsed_sec=REAL_TIME()-start))
    backend_end=REAL_TIME()-start
    if continue_frg and REAL_TIME()<tape.deadline:
        tape.phase('bs_frg' if mode=='bs_frg' else 'frg')
        stages['continuation']=(bs_frg_continuation(instance,tape,max(0,len(path)-1)) if mode=='bs_frg'
                                else frg_descent(instance,tape))
    return dict(mode=mode,moves=len(tape.best),path=[asdict(m) for m in tape.best],models=records,
        stages=stages,backend_end_sec=backend_end,elapsed_sec=REAL_TIME()-start,
        late=tape.late,passes=1,global_cpmp_optimal=False)
