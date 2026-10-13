"""Equivalent feasible categorical MILP,kept separate from QUBO execution."""
from __future__ import annotations

import itertools
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from production_selector_replay import REAL_TIME


def solve(unary, groups, pair, loads, caps, incumbent, seconds=2.):
    started = REAL_TIME()
    costs = np.asarray(unary, dtype=float)
    if not len(costs):
        return [], dict(optimal=True, status=0, runtime_sec=REAL_TIME()-started, milp_calls=0)
    rr, cc, vv, lower, upper = [], [], [], [], []
    def row(terms, lo, hi):
        k = len(lower)
        lower.append(lo)
        upper.append(hi)
        for i, value in terms:
            rr.append(k); cc.append(int(i)); vv.append(float(value))
    for group in groups:
        row(((i,1) for i in group),1,1)
    for i,j in zip(*np.where(np.triu(pair,1))):
        row(((i,1),(j,1)),-np.inf,1)
    for j,cap in enumerate(caps):
        row(((i,loads[i,j]) for i in range(len(costs)) if loads[i,j]),-np.inf,cap)
    matrix = coo_matrix((vv,(rr,cc)),shape=(len(lower),len(costs))).tocsc()
    result = milp(costs,integrality=np.ones(len(costs)),bounds=Bounds(0,1),
                  constraints=LinearConstraint(matrix,lower,upper),
                  options=dict(time_limit=seconds,mip_rel_gap=0.,presolve=True))
    chosen = list(map(int,incumbent))
    if result.x is not None:
        bits=np.rint(result.x)
        values=matrix@bits
        if (np.max(abs(bits-result.x))>1e-5 or np.any(bits<0) or np.any(bits>1) or
            np.any(values<np.asarray(lower)-1e-6) or np.any(values>np.asarray(upper)+1e-6)):
            raise ValueError('Joint classical control produced an invalid integral assignment')
        candidate=[next(int(i) for i in g if bits[i]) for g in groups]
        if costs[candidate].sum()<costs[chosen].sum():
            chosen=candidate
    return chosen,dict(optimal=result.status==0,status=int(result.status),message=result.message,
                       runtime_sec=REAL_TIME()-started,milp_calls=1)
