"""Deterministic energy branch-and-bound for nonnegative categorical QUBOs.

One-hot blocks are assigned directly. Capacity slack is minimized analytically.
The solver sees costs,penalty couplings and loads,not container moves or MILP.
"""
from __future__ import annotations
import numpy as np
from production_selector_replay import REAL_TIME


def solve(unary,groups,pair,loads,caps,penalty,constant,incumbent,node_limit=50000):
    began=REAL_TIME()
    unary=np.asarray(unary,dtype=float);pair=np.asarray(pair,dtype=bool)
    loads=np.asarray(loads,dtype=np.int16);caps=np.asarray(caps,dtype=np.int16)
    if np.any(unary<0) or np.any(loads<0):raise ValueError('This energy bound requires nonnegative costs/loads')
    groups=[np.asarray(g,dtype=int) for g in groups]
    incumbent=list(map(int,incumbent))
    if len(incumbent)!=len(groups) or any(i not in g for i,g in zip(incumbent,groups)):
        raise ValueError('Incumbent must select exactly one option from every group')
    def energy(chosen):
        if not chosen:return float(constant)
        selected=np.asarray(chosen,dtype=int)
        violations=sum(pair[i,j] for k,i in enumerate(chosen) for j in chosen[k+1:])
        overflow=np.maximum(0,loads[selected].sum(axis=0)-caps)
        return float(constant+unary[selected].sum()+penalty*(violations+np.square(overflow).sum()))
    best=incumbent;best_energy=energy(best)
    if best_energy>=constant+penalty:
        raise ValueError('A feasible incumbent below the dominant penalty is required')
    nodes=0;cutoff=False;pruned=0
    degrees=[]
    for g,indices in enumerate(groups):
        degrees.append(sum(bool(np.any(pair[np.ix_(indices,other)])) for h,other in enumerate(groups) if h!=g))
    assigned=[None]*len(groups)
    def visit(domains,chosen,cost,load):
        nonlocal nodes,cutoff,pruned,best,best_energy
        if nodes>=node_limit:cutoff=True;return
        nodes+=1
        if not domains:
            candidate=[int(v) for v in assigned]
            value=energy(candidate)
            if value<best_energy-1e-9:best,best_energy=candidate,value
            return
        reduced={}
        for g,domain in domains.items():
            # Any selected pair penalty alone exceeds the feasible incumbent.
            options=domain[~pair[np.ix_(domain,chosen)].any(axis=1)] if chosen else domain
            if not len(options):pruned+=1;return
            reduced[g]=options
        minimum_load=load.copy()
        lower=float(constant+cost)
        for domain in reduced.values():
            lower+=float(unary[domain].min())
            minimum_load+=loads[domain].min(axis=0)
        overflow=np.maximum(0,minimum_load-caps)
        lower+=float(penalty*np.square(overflow).sum())
        if lower>=best_energy-1e-9:pruned+=1;return
        g=min(reduced,key=lambda g:(len(reduced[g]),-degrees[g],g))
        rest={h:d for h,d in reduced.items() if h!=g}
        for option in sorted(map(int,reduced[g]),key=lambda i:(unary[i],i)):
            assigned[g]=option
            visit(rest,chosen+[option],cost+unary[option],load+loads[option])
            if cutoff:break
        assigned[g]=None
    visit({g:d for g,d in enumerate(groups)},[],0.,np.zeros(len(caps),dtype=np.int64))
    return best,dict(energy=best_energy,nodes=nodes,pruned=pruned,node_limit=node_limit,
                     optimal=not cutoff,cutoff=cutoff,runtime_sec=REAL_TIME()-began)
