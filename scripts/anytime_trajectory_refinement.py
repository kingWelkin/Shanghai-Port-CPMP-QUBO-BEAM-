"""Equal extra-time refinement with timestamped,validated feasible incumbents."""
from __future__ import annotations
from dataclasses import asdict,replace
import inspect
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp

from container_reshuffle.core import Move
from container_reshuffle.solvers import bs_frg as frg
from container_reshuffle.solvers.base import SolverConfig
from classical_trajectory_simplify import simplify
from complete_trajectory_domains import CompleteTrajectoryDomains,UnsupportedDomain
from categorical_qubo_search import solve
from categorical_greedy_controls import descend
from trajectory_group_flow_qubo import GroupFlowQubo
from hybrid_trajectory_refiner import no_milp
from research_incumbent_action_refinement import suffix_states
from research_receiver_allocation import physical_replay
from production_selector_replay import REAL_TIME


class FeasibleTape:
    def __init__(self,instance,path,budget):
        self.instance=instance;self.start=REAL_TIME();self.deadline=self.start+budget;self.budget=budget
        self.best=tuple(path);self.events=[];self.late=[]
        physical_replay(instance.stacks,instance.max_height,self.best,goal=True)
        self.initial=len(self.best)

    def offer(self,path,source):
        path=tuple(replace(m,step=i) for i,m in enumerate(path))
        if len(path)>=len(self.best):return False
        physical_replay(self.instance.stacks,self.instance.max_height,path,goal=True)
        ready=REAL_TIME()-self.start
        event=dict(ready_sec=ready,moves=len(path),source=source,path=[asdict(m) for m in path])
        if ready>self.budget:
            self.late.append(event);return False
        self.best=path;self.events.append(event);return True


def timed_groups(model,tape):
    kept=tuple(range(model.count));bits=model.encode(kept);solves=0;passes=0;converged=False;limited=False
    priorities=np.array([m.box for m in model.moves]+[a.value for a in model.arcs])
    groups=sorted({m.box for m in model.moves})
    while solves<2048 and REAL_TIME()<tape.deadline:
        before=len(kept);complete=True;certified=True;passes+=1
        for group in groups:
            if solves>=2048 or REAL_TIME()>=tape.deadline-.003:complete=False;limited=True;break
            fixed=priorities!=group
            values=np.array([bits.get(i,0) for i in range(model.core_variables)],dtype=float)
            lower=np.where(fixed,values,0.);upper=np.where(fixed,values,1.)
            allowance=min(.1,tape.deadline-REAL_TIME()-.003)
            if allowance<=0:complete=False;limited=True;break
            result=milp(model.cost,integrality=np.ones(model.core_variables),bounds=Bounds(lower,upper),
                constraints=LinearConstraint(model.matrix,model.lower,model.upper),
                options=dict(time_limit=allowance,mip_rel_gap=0.,presolve=True))
            solves+=1;certified=certified and result.status==0
            if result.x is not None:
                x=np.rint(result.x);lhs=model.matrix@x
                if (np.max(abs(x-result.x))>1e-5 or np.any(x<lower) or np.any(x>upper)
                    or np.any(lhs<np.array(model.lower)-1e-6) or np.any(lhs>np.array(model.upper)+1e-6)):
                    raise ValueError('Timed group control violates fixed blocks or physical constraints')
                candidate=tuple(j for j in range(model.count) if x[model.z[j]])
                if len(candidate)<len(kept):
                    # The checked MILP vector already supplies the next fixed
                    # group state. Materialize/validate once before publication.
                    state=[list(s) for s in model.stacks];path=[]
                    for j in candidate:
                        move=model.moves[j]
                        path.append(Move(len(path),move.box,move.from_stack,move.to_stack,
                                         len(state[move.from_stack]),len(state[move.to_stack])+1))
                        state[move.to_stack].append(state[move.from_stack].pop())
                    bits={i:int(v) for i,v in enumerate(x)};kept=candidate
                    tape.offer(path,'coordinate')
        if complete and len(kept)==before:
            converged=certified and REAL_TIME()<=tape.deadline;break
    return dict(solves=solves,passes=passes,converged=converged,limited=limited or REAL_TIME()>=tape.deadline)


def frg_descent(instance,tape):
    attempts=0;passes=0;converged=False
    while REAL_TIME()<tape.deadline:
        path=tape.best;before=len(path);passes+=1;complete=True
        for prefix,stacks,n in suffix_states(instance,path):
            if REAL_TIME()>=tape.deadline:complete=False;break
            result=frg._frg_completion(frg._FrgLayout(stacks,None,(0,)*instance.stack_count),instance.max_height,
                start_step=prefix,max_steps=len(path)-prefix-1,deadline=tape.deadline,
                max_generated=None,paper_pure=True)
            attempts+=1
            if result.feasible:tape.offer(path[:prefix]+result.moves,'frg_descent')
        if complete and len(tape.best)==before:
            converged=REAL_TIME()<tape.deadline;break
    return dict(attempts=attempts,passes=passes,converged=converged)


def bs_frg_continuation(instance,tape,depth_cap):
    original=frg._frg_completion_cached;captures=0
    def observed(*args,**kwargs):
        nonlocal captures
        result=original(*args,**kwargs)
        if result.feasible:
            frame=inspect.currentframe().f_back
            if frame.f_code is frg.BsFrgSolver.solve.__code__ and frame.f_locals.get('instance') is instance:
                prefix=tuple(frame.f_locals.get('child_moves',()))
                if len(prefix)!=kwargs.get('start_step',0):raise ValueError('Timed BS-FRG prefix mismatch')
                captures+=1;tape.offer(prefix+result.moves,'bs_frg')
        return result
    remaining=tape.deadline-REAL_TIME()
    if remaining<=0:return dict(captures=0,skipped=True)
    try:
        frg._frg_completion_cached=observed
        result=frg.BsFrgSolver(paper_pure=True).solve(instance,SolverConfig(max_depth=depth_cap,
            beam_width=1000,candidate_limit_per_state=3,time_limit_sec=remaining,random_seed=37,
            exact_incumbent_moves=list(tape.best)))
    finally:frg._frg_completion_cached=original
    if result.feasible:tape.offer(result.moves,'bs_frg')
    return dict(captures=captures,expanded=result.expanded_states,generated=result.generated_states,status=result.status)


def run(instance,path,mode,budget=2.):
    if mode not in ('frg','bs_frg','greedy_frg','coordinate_frg','qubo_frg','oracle_frg'):raise ValueError(mode)
    tape=FeasibleTape(instance,path,budget);stages={};q_calls=0;classical_calls=0;fallback=None
    common,common_meta=simplify(instance.stacks,instance.max_height,path)
    tape.offer(common,'common_simplification');stages['common']=common_meta
    common_late=REAL_TIME()>tape.deadline
    if not common_late and mode not in ('frg','bs_frg'):
        if mode in ('qubo_frg','greedy_frg'):
            try:
                with no_milp() as guard:
                    model=CompleteTrajectoryDomains(instance.stacks,instance.max_height,common)
                    stages['domain']=dict(model.stats)
                    if mode=='qubo_frg':
                        q=model.compile_qubo()
                        selected,meta=solve(model.unary,model.groups,model.pair,model.loads,model.caps,
                            model.penalty,model.constant,model.current,50000)
                        candidate=model.replay(selected)
                        if abs(q.evaluate(model.bits(selected))-len(candidate))>1e-5:
                            raise ValueError('Timed QUBO energy/path mismatch')
                        tape.offer(candidate,'qubo')
                        stages['backend']=meta;stages['q_variables']=len(q.var_index)
                    else:
                        trials=[descend(model.unary,model.groups,model.pair,model.loads,model.caps,
                                        model.current,order) for order in ('best','forward','reverse')]
                        selected,meta=min(trials,key=lambda r:(model.unary[r[0]].sum(),r[0]))
                        tape.offer(model.replay(selected),'greedy3')
                        stages['backend']=dict(orders=[meta for selected,meta in trials])
                    q_calls=guard['calls']
            except UnsupportedDomain as e:fallback=str(e)
        if (mode in ('coordinate_frg','oracle_frg') or fallback is not None) and REAL_TIME()<tape.deadline:
            build=REAL_TIME();model=GroupFlowQubo(instance.stacks,instance.max_height,common)
            stages['group_build_sec']=REAL_TIME()-build
            if REAL_TIME()<tape.deadline-.003:
                if mode=='oracle_frg':
                    kept,meta=model.exact(min(2.,tape.deadline-REAL_TIME()-.003))
                    tape.offer(model.replay(kept),'joint_milp_control');classical_calls=1
                    stages['backend']=meta
                else:
                    meta=timed_groups(model,tape);classical_calls=meta['solves'];stages['backend']=meta
    backend_end=REAL_TIME()-tape.start
    if mode=='bs_frg':stages['continuation']=bs_frg_continuation(instance,tape,max(0,len(common)-1))
    else:stages['continuation']=frg_descent(instance,tape)
    elapsed=REAL_TIME()-tape.start
    physical_replay(instance.stacks,instance.max_height,tape.best,goal=True)
    return dict(mode=mode,budget_sec=budget,initial=len(path),moves=len(tape.best),
        path=[asdict(m) for m in tape.best],events=tape.events,late=tape.late,stages=stages,
        q_milp_calls=q_calls,classical_milp_calls=classical_calls,fallback=fallback,
        common_late=common_late,backend_end_sec=backend_end,elapsed_sec=elapsed,
        overrun_sec=max(0.,elapsed-budget),global_cpmp_optimal=False)
