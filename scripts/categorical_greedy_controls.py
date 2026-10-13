"""Classical feasible single-group greedy controls on the same full domains."""
from __future__ import annotations
import numpy as np
from production_selector_replay import REAL_TIME


def descend(unary,groups,pair,loads,caps,incumbent,order='best'):
    began=REAL_TIME();chosen=list(map(int,incumbent));evaluations=0;updates=0
    unary=np.asarray(unary);groups=[np.asarray(g,dtype=int) for g in groups]
    def improve(g):
        nonlocal evaluations
        domain=groups[g];fixed=[v for h,v in enumerate(chosen) if h!=g]
        fixed_load=loads[fixed].sum(axis=0) if fixed else np.zeros(len(caps),dtype=np.int64)
        allowed=np.all(fixed_load+loads[domain]<=caps,axis=1)
        if fixed:allowed&=~pair[np.ix_(domain,fixed)].any(axis=1)
        options=domain[allowed];evaluations+=len(domain)
        if not len(options):raise ValueError('A feasible greedy incumbent disappeared')
        option=min(map(int,options),key=lambda i:(unary[i],i))
        delta=float(unary[option]-unary[chosen[g]])
        return (delta,g,option) if delta<0 else None
    if order=='best':
        while True:
            changes=[v for g in range(len(groups)) if (v:=improve(g)) is not None]
            if not changes:break
            delta,g,option=min(changes);chosen[g]=option;updates+=1
    else:
        sequence=list(range(len(groups)))
        if order=='reverse':sequence.reverse()
        elif order!='forward':raise ValueError(order)
        while True:
            changed=False
            for g in sequence:
                result=improve(g)
                if result is not None:
                    _,_,option=result;chosen[g]=option;updates+=1;changed=True
            if not changed:break
    return chosen,dict(order=order,evaluations=evaluations,updates=updates,runtime_sec=REAL_TIME()-began)
