"""Cooperatively bounded structured QUBO search preserving every feasible best.

Scope:nonnegative integer unary costs/loads,one-hot groups,and a penalty larger
than a supplied feasible residual cost. Under that dominance bound,any energy
improvement is feasible;incompatible or overflowing options may be removed.
The kernel receives factors only andnever calls a CPMP orMILP solver.
"""
from __future__ import annotations

import numpy as np
from production_selector_replay import REAL_TIME


def _integer_array(value,name):
    raw=np.asarray(value)
    if not np.all(np.isfinite(raw)) or np.any(raw!=np.rint(raw)):
        raise ValueError(name+' must contain finite integers')
    return raw.astype(np.int64)


def solve(unary,groups,pair,loads,caps,penalty,constant,incumbent,node_limit=50000,*,deadline=None,clock=None):
    began=REAL_TIME()
    clock=REAL_TIME if clock is None else clock
    unary=_integer_array(unary,'unary');loads=_integer_array(loads,'loads');caps=_integer_array(caps,'caps')
    pair=np.asarray(pair,dtype=bool)
    groups=[list(map(int,g)) for g in groups];incumbent=list(map(int,incumbent))
    n=len(unary);ng=len(groups)
    if (np.any(unary<0) or np.any(loads<0) or np.any(caps<0) or
        pair.shape!=(n,n) or loads.shape!=(n,len(caps)) or not np.array_equal(pair,pair.T)):
        raise ValueError('Unsupported nonnegative categorical QUBO factors')
    if sorted(i for g in groups for i in g)!=list(range(n)) or any(not g for g in groups):
        raise ValueError('One-hot groups must partition the options')
    if len(incumbent)!=ng or any(i not in g for i,g in zip(incumbent,groups)):
        raise ValueError('Invalid categorical incumbent')
    if not float(penalty).is_integer() or not float(constant).is_integer() or node_limit<0:
        raise ValueError('Integer penalty/constant andnonnegative node limit required')
    penalty,constant=int(penalty),int(constant)
    if (any(pair[a,b] for k,a in enumerate(incumbent) for b in incumbent[k+1:]) or
        (incumbent and np.any(loads[incumbent].sum(axis=0)>caps))):
        raise ValueError('The warm incumbent must be feasible')
    initial_cost=sum(int(unary[i]) for i in incumbent)
    if initial_cost>=penalty:raise ValueError('Penalty does not dominate the feasible residual cost')
    best=list(incumbent);best_energy=constant+initial_cost
    nodes=pruned=removed=rounds=0;cutoff=False;support_tests=0;wall_cutoff=False
    class DeadlineReached(Exception):pass
    def check():
        if deadline is not None and clock()>=deadline:raise DeadlineReached
    initial_bound=constant+sum(min(int(unary[i]) for i in g) for g in groups)
    if node_limit and initial_bound>=best_energy:
        return best,dict(energy=best_energy,nodes=1,pruned=1,node_limit=node_limit,optimal=True,cutoff=False,
            removed=0,propagation_rounds=0,support_tests=0,initial_bound_closed=True,runtime_sec=REAL_TIME()-began)
    blocked=[];masks=[];neighbors=[]
    supports={}
    costs=list(map(int,unary))
    def indices(mask):
        result=[]
        while mask:
            bit=mask&-mask;result.append(bit.bit_length()-1);mask^=bit
        return result
    def propagate(domains):
        nonlocal removed,rounds,support_tests
        domains=list(domains)
        while True:
            check()
            rounds+=1;changed=False
            for g,ns in enumerate(neighbors):
                for h in ns:
                    check()
                    available=domains[h];kept=0
                    for number,i in enumerate(indices(domains[g])):
                        if number%64==0:check()
                        support=supports.get((i,h),0)
                        if not support&available:
                            support_tests+=1
                            options=available&~blocked[i]
                            support=options&-options
                            supports[i,h]=support
                        if support:kept|=1<<i
                    if kept!=domains[g]:
                        removed+=domains[g].bit_count()-kept.bit_count();domains[g]=kept;changed=True
                        if not kept:return None
            lists=[indices(d) for d in domains]
            minima=[min(costs[i] for i in ids) for ids in lists]
            lower=constant+sum(minima)
            if lower>=best_energy:return None
            minloads=[loads[ids].min(axis=0) for ids in lists] if len(caps) else []
            total_min=np.sum(minloads,axis=0,dtype=np.int64) if len(caps) else np.zeros(0,dtype=np.int64)
            if len(caps) and np.any(total_min>caps):return None
            for g,ids in enumerate(lists):
                check()
                allowed=unary[ids]+lower-minima[g]<best_energy
                if len(caps):allowed&=np.all(loads[ids]+total_min-minloads[g]<=caps,axis=1)
                kept=sum(1<<i for i,ok in zip(ids,allowed) if ok)
                if kept!=domains[g]:
                    removed+=domains[g].bit_count()-kept.bit_count();domains[g]=kept;changed=True
                    if not kept:return None
            if not changed:return domains
    def visit(domains):
        nonlocal nodes,pruned,cutoff,best,best_energy
        check()
        if nodes>=node_limit:cutoff=True;return
        nodes+=1
        reduced=propagate(domains)
        if reduced is None:pruned+=1;return
        pending=[g for g,d in enumerate(reduced) if d.bit_count()>1]
        if not pending:
            chosen=[d.bit_length()-1 for d in reduced]
            value=constant+sum(costs[i] for i in chosen)
            if value<best_energy:best,best_energy=chosen,value
            return
        g=min(pending,key=lambda h:(reduced[h].bit_count(),-len(neighbors[h]),h))
        for i in sorted(indices(reduced[g]),key=lambda i:(costs[i],i)):
            child=list(reduced);child[g]=1<<i;visit(child)
            if cutoff:break
    try:
        check()
        for j,row in enumerate(pair):
            if j%64==0:check()
            blocked.append(int.from_bytes(np.packbits(row,bitorder='little').tobytes(),'little'))
        masks=[sum(1<<i for i in g) for g in groups]
        for g,options in enumerate(groups):
            check();conflicts=0
            for i in options:conflicts|=blocked[i]
            neighbors.append([h for h,m in enumerate(masks) if h!=g and conflicts&m])
        visit(masks)
    except DeadlineReached:
        cutoff=True;wall_cutoff=True
    if (any(pair[a,b] for k,a in enumerate(best) for b in best[k+1:]) or
        (best and np.any(loads[best].sum(axis=0)>caps))):
        raise ValueError('Propagation returned an infeasible QUBO incumbent')
    return best,dict(energy=best_energy,nodes=nodes,pruned=pruned,node_limit=node_limit,
        optimal=not cutoff,cutoff=cutoff,removed=removed,propagation_rounds=rounds,support_tests=support_tests,
        initial_bound_closed=False,wall_cutoff=wall_cutoff,runtime_sec=REAL_TIME()-began)
