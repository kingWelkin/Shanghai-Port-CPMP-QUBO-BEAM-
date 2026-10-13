"""Uniform50ms joint-search cap within the unchanged2-second outer controller."""
from __future__ import annotations

import bounded_conditioned_refinement as original
from categorical_budgeted_qubo import solve
from production_selector_replay import REAL_TIME


def run(instance,path,mode,start,budget,publish,**kwargs):
    if mode not in ('capped_qubo_frg','capped_oracle_frg'):
        return original.run(instance,path,mode,start,budget,publish,**kwargs)
    old_q,old_milp=original.solve,original.exact.solve
    def q(*args,**kw):
        deadline=min(start+budget,REAL_TIME()+.05)
        selected,meta=solve(*args,**kw,deadline=deadline)
        meta['local_time_limit_sec']=.05
        return selected,meta
    def milp(unary,groups,pair,loads,caps,incumbent,seconds=2.):
        chosen,meta=old_milp(unary,groups,pair,loads,caps,incumbent,min(.05,seconds))
        meta['local_solver_time_limit_sec']=.05
        # SciPy's solver limit excludes Python construction andmay overrun;
        # the unchanged parent still enforces the whole2-second allowance.
        return chosen,meta
    try:
        original.solve=q;original.exact.solve=milp
        return original.run(instance,path,'qubo_frg' if mode=='capped_qubo_frg' else 'oracle_frg',
            start,budget,publish,**kwargs)
    finally:original.solve,original.exact.solve=old_q,old_milp
