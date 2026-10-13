"""Opt-in complete-path refinement with an explicit classical fallback.

Production search defaults are untouched. QUBO,common simplification,and
classical fallback effects are reported separately;all returned moves replay.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict

from classical_trajectory_simplify import simplify
from complete_trajectory_domains import CompleteTrajectoryDomains,UnsupportedDomain
from categorical_qubo_search import solve
from categorical_greedy_controls import descend
from trajectory_group_flow_qubo import GroupFlowQubo
from research_receiver_allocation import physical_replay
from production_selector_replay import REAL_TIME


@contextmanager
def no_milp():
    import scipy.optimize as opt
    import trajectory_subsequence_qubo as subsequence
    import trajectory_group_flow_qubo as flow
    import two_wave_action_qubo as action
    saved=[];stats=dict(calls=0)
    def forbidden(*args,**kwargs):
        stats['calls']+=1
        raise RuntimeError('MILP is forbidden inside the QUBO/greedy route backend')
    try:
        for module in (opt,subsequence,flow,action):
            saved.append((module,module.milp));module.milp=forbidden
        yield stats
    finally:
        for module,original in reversed(saved):module.milp=original


def refine(instance,moves,mode='qubo',*,max_routes=20000,max_variables=4096,node_limit=50000):
    """Natural completion;timed qualification is a separate experimental contract."""
    if mode not in ('qubo','greedy','coordinate','oracle'):raise ValueError(mode)
    started=REAL_TIME();original=tuple(moves)
    physical_replay(instance.stacks,instance.max_height,original,goal=True)
    common,common_meta=simplify(instance.stacks,instance.max_height,original)
    phase_started=REAL_TIME();candidate=common;fallback_reason=None;backend={};model_meta={}
    q_milp_calls=0;fallback_milp_calls=0;source=mode
    if mode in ('qubo','greedy'):
        try:
            with no_milp() as guard:
                model=CompleteTrajectoryDomains(instance.stacks,instance.max_height,common,max_routes=max_routes)
                model_meta=dict(model.stats)
                if mode=='qubo':
                    build=REAL_TIME();q=model.compile_qubo(max_variables)
                    model_meta.update(compile_sec=REAL_TIME()-build,variables=len(q.var_index),terms=len(q.Q))
                    selected,backend=solve(model.unary,model.groups,model.pair,model.loads,model.caps,
                        model.penalty,model.constant,model.current,node_limit)
                    candidate=model.replay(selected)
                    energy=q.evaluate(model.bits(selected))
                    if abs(energy-len(candidate))>1e-5 or abs(energy-backend['energy'])>1e-5:
                        raise ValueError('Hybrid QUBO matrix/factor/path mismatch')
                    backend.update(bits=model.bits(selected),selected=selected)
                    if not model.groups:source='propagation_certificate'
                else:
                    trials=[descend(model.unary,model.groups,model.pair,model.loads,model.caps,
                                    model.current,order) for order in ('best','forward','reverse')]
                    selected,best=min(trials,key=lambda row:(model.unary[row[0]].sum(),row[0]))
                    candidate=model.replay(selected)
                    backend=dict(orders=[meta for chosen,meta in trials],
                        runtime_sec=sum(meta['runtime_sec'] for chosen,meta in trials),optimal=False)
                q_milp_calls=guard['calls']
        except UnsupportedDomain as e:
            fallback_reason=str(e);source='classical_group_fallback'
            model_meta['unsupported_attempt_sec']=REAL_TIME()-phase_started
    if mode in ('coordinate','oracle') or fallback_reason is not None:
        build=REAL_TIME();model=GroupFlowQubo(instance.stacks,instance.max_height,common)
        model_meta['group_model_sec']=REAL_TIME()-build
        if mode=='oracle':
            kept,backend=model.exact(2.)
            fallback_milp_calls=1;source='joint_milp_control'
        else:
            kept,backend=model.group_descent(seconds_per_group=.1,max_solves=2048)
            fallback_milp_calls=backend['solves']
        candidate=model.replay(kept)
        if abs(model.qubo.evaluate(model.encode(kept))-len(candidate))>1e-4:
            raise ValueError('Classical group fallback failed complete-path validation')
    physical_replay(instance.stacks,instance.max_height,candidate,goal=True)
    if len(candidate)>len(common):raise ValueError('Refinement lost the common feasible incumbent')
    result=candidate if len(candidate)<len(original) else original
    retained_source=(source if len(candidate)<len(common) else
                     'common_simplification' if len(common)<len(original) else 'native_retained')
    return dict(path=[asdict(m) for m in result],native_moves=len(original),common_moves=len(common),
        moves=len(result),mode=mode,backend_source=source,retained_source=retained_source,
        common_saved=len(original)-len(common),backend_saved=len(common)-len(result),
        fallback_reason=fallback_reason,q_milp_calls=q_milp_calls,classical_milp_calls=fallback_milp_calls,
        common=common_meta,model=model_meta,backend=backend,
        refinement_sec=REAL_TIME()-phase_started,total_sec=REAL_TIME()-started,
        global_cpmp_optimal=False)
