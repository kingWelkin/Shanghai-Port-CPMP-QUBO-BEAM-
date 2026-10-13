"""Complete priority-group flow QUBO over all retained move subsequences.

Residence arcs match initial/push arrivals to later same-group pops or terminal
residence. Flow,LIFO conflicts,terminal order,and capacity characterize physical
subsequence feasibility without a per-container route pool or fixed identities.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import coo_matrix

from container_reshuffle.core import Move
from container_reshuffle.solvers import bs_frg as frg
from container_reshuffle.qubo.model import QuboModel
from two_wave_action_qubo import binary_weights
from research_receiver_allocation import physical_replay
from production_selector_replay import REAL_TIME


@dataclass(frozen=True)
class Arrival:
    stack:int
    value:int
    time:int
    move:int|None


@dataclass(frozen=True)
class Residence:
    arrival:int
    departure:int|None
    stack:int
    value:int
    start:int
    end:int


class GroupFlowQubo:
    def __init__(self,stacks,height,moves):
        self.stacks=tuple(tuple(s) for s in stacks)
        self.height,self.moves=height,tuple(moves)
        physical_replay(stacks,height,moves,goal=True)
        self.count=len(moves);self.end=2*self.count+2;self.penalty=self.count+1
        self.qubo=QuboModel()
        self.z=[self.qubo.add_binary_var(f'move_{j}') for j in range(self.count)]
        for z in self.z:self.qubo.add_linear(z,1.)
        self.arrivals=[];self.initial=[];self.push={}
        for s,stack in enumerate(stacks):
            ids=[]
            for level,value in enumerate(stack):
                ids.append(len(self.arrivals))
                self.arrivals.append(Arrival(s,value,-2*height+2*level,None))
            self.initial.append(ids)
        for j,move in enumerate(moves):
            self.push[j]=len(self.arrivals)
            self.arrivals.append(Arrival(move.to_stack,move.box,2*j+1,j))
        self.arcs=[];self.y=[];self.arc_index={};self.outgoing=[[] for _ in self.arrivals]
        self.incoming=[[] for _ in moves]
        for a,arrival in enumerate(self.arrivals):
            departures=[j for j,move in enumerate(moves) if move.from_stack==arrival.stack
                        and move.box==arrival.value and arrival.time<2*j]+[None]
            for d in departures:
                arc=Residence(a,d,arrival.stack,arrival.value,arrival.time,self.end if d is None else 2*d)
                v=self.qubo.add_binary_var(f'reside_{a}_{d}')
                self.arc_index[a,d]=v;self.y.append(v);self.arcs.append(arc);self.outgoing[a].append(v)
                if d is not None:self.incoming[d].append(v)
        self.core_variables=len(self.qubo.var_index)
        self.cost=np.r_[np.ones(self.count),np.zeros(len(self.arcs))]
        self.rows=[];self.lower=[];self.upper=[];self.capacity=[];self.conflicts=[]
        for a,arrival in enumerate(self.arrivals):
            row={v:1 for v in self.outgoing[a]}
            if arrival.move is None:self.equal(row,1)
            else:
                row[self.z[arrival.move]]=-1
                self.equal(row,0)
        for j,vs in enumerate(self.incoming):
            self.equal({**{v:1 for v in vs},self.z[j]:-1},0)
        for i,left in enumerate(self.arcs):
            for j in range(i+1,len(self.arcs)):
                right=self.arcs[j]
                if left.stack!=right.stack:continue
                a,b,c,d=left.start,left.end,right.start,right.end
                crossing=a<c<b<d or c<a<d<b
                bad_final=(b==d==self.end and ((a<c and left.value<right.value) or (c<a and right.value<left.value)))
                if crossing or bad_final:
                    u,v=self.y[i],self.y[j]
                    self.conflicts.append((u,v))
                    self.rows.append({u:1,v:1});self.lower.append(-np.inf);self.upper.append(1)
                    self.qubo.add_quadratic(u,v,self.penalty)
        for s,stack in enumerate(stacks):
            row={}
            for j,move in enumerate(moves):
                if move.from_stack==s:row[self.z[j]]=-1
                if move.to_stack==s:
                    row[self.z[j]]=1
                    rhs=height-len(stack)
                    self.rows.append(dict(row));self.lower.append(-np.inf);self.upper.append(rhs)
                    slack=[(self.qubo.add_binary_var(f'cap_{s}_{j}_{k}'),w)
                           for k,w in enumerate(binary_weights(height))]
                    self.capacity.append((dict(row),rhs,slack))
                    self.square({**row,**dict(slack)},rhs)
        self.matrix=self.constraint_matrix()
        original=self.encode(tuple(range(self.count)))
        if abs(self.qubo.evaluate(original)-self.count)>1e-5:raise ValueError('Original group-flow energy mismatch')

    def square(self,row,rhs):
        p=self.penalty
        self.qubo.offset+=p*rhs*rhs
        for i,a in row.items():self.qubo.add_linear(i,p*(a*a-2*rhs*a))
        for (i,a),(j,b) in itertools.combinations(row.items(),2):self.qubo.add_quadratic(i,j,2*p*a*b)

    def equal(self,row,rhs):
        self.rows.append(row);self.lower.append(rhs);self.upper.append(rhs);self.square(row,rhs)

    def constraint_matrix(self):
        rr,cc,vv=[],[],[]
        for r,row in enumerate(self.rows):
            for i,v in row.items():rr.append(r);cc.append(i);vv.append(v)
        return coo_matrix((vv,(rr,cc)),shape=(len(self.rows),self.core_variables)).tocsc()

    def capacity_bits(self,values):
        values=dict(values)
        for row,rhs,slack in self.capacity:
            remaining=rhs-sum(a*values.get(i,0) for i,a in row.items())
            if not 0<=remaining<=self.height:raise ValueError('Invalid physical stack occupancy')
            for var,weight in reversed(slack):
                if weight<=remaining:values[var]=1;remaining-=weight
            if remaining:raise ValueError('Cannot represent capacity slack')
        return values

    def encode(self,kept):
        kept=tuple(kept)
        if kept!=tuple(sorted(set(kept))) or any(t<0 or t>=self.count for t in kept):
            raise ValueError('Retained events must be an ordered subset')
        state=[list(s) for s in self.initial];values={};path=[]
        for j in kept:
            move=self.moves[j]
            if not state[move.from_stack] or self.arrivals[state[move.from_stack][-1]].value!=move.box:
                raise ValueError('Retained priority group is not at the stack top')
            if len(state[move.to_stack])>=self.height:raise ValueError('Retained move exceeds capacity')
            a=state[move.from_stack].pop()
            values[self.z[j]]=1;values[self.arc_index[a,j]]=1
            path.append(Move(len(path),move.box,move.from_stack,move.to_stack,
                             len(state[move.from_stack])+1,len(state[move.to_stack])+1))
            state[move.to_stack].append(self.push[j])
        for stack in state:
            if not frg._is_ordered_stack([self.arrivals[a].value for a in stack]):
                raise ValueError('Retained events do not reach an ordered goal')
            for a in stack:values[self.arc_index[a,None]]=1
        x=np.array([values.get(i,0) for i in range(self.core_variables)])
        lhs=self.matrix@x
        if np.any(lhs<np.array(self.lower)-1e-6) or np.any(lhs>np.array(self.upper)+1e-6):
            raise ValueError('Physical replay violates group-flow constraints')
        physical_replay(self.stacks,self.height,path,goal=True)
        return self.capacity_bits(values)

    def replay(self,kept):
        self.encode(kept)
        stacks=[list(s) for s in self.stacks];path=[]
        for j in kept:
            move=self.moves[j]
            path.append(Move(len(path),move.box,move.from_stack,move.to_stack,
                             len(stacks[move.from_stack]),len(stacks[move.to_stack])+1))
            stacks[move.to_stack].append(stacks[move.from_stack].pop())
        return tuple(path)

    def exact(self,seconds=2.,incumbent=None):
        began=REAL_TIME()
        incumbent=tuple(range(self.count)) if incumbent is None else tuple(incumbent)
        self.encode(incumbent)
        result=milp(self.cost,integrality=np.ones(self.core_variables),bounds=Bounds(0.,1.),
            constraints=LinearConstraint(self.matrix,self.lower,self.upper),
            options=dict(time_limit=seconds,mip_rel_gap=0.,presolve=True))
        chosen=incumbent;raw_energy=None
        if result.x is not None:
            x=np.rint(result.x)
            lhs=self.matrix@x
            if np.max(abs(x-result.x))>1e-5 or np.any(lhs<np.array(self.lower)-1e-6) or np.any(lhs>np.array(self.upper)+1e-6):
                raise ValueError('Structured group-flow reference is not integral/feasible')
            kept=tuple(j for j in range(self.count) if x[self.z[j]])
            raw_bits=self.capacity_bits({i:int(v) for i,v in enumerate(x)})
            raw_energy=self.qubo.evaluate(raw_bits)
            if abs(raw_energy-len(kept))>1e-4:raise ValueError('Reference energy differs from retained moves')
            self.replay(kept)
            if result.status==0 and len(kept)>len(incumbent):
                raise ValueError('Certified reference is worse than a feasible incumbent')
            if len(kept)<len(chosen):chosen=kept
        return chosen,dict(optimal=result.status==0,status=int(result.status),raw_energy=raw_energy,
            dual_bound=float(result.mip_dual_bound) if getattr(result,'mip_dual_bound',None) is not None else None,
            runtime_sec=REAL_TIME()-began)

    def group_descent(self,seconds_per_group=.1,max_solves=256):
        """Classical exact single-priority block control,with other groups fixed."""
        began=REAL_TIME();kept=tuple(range(self.count));solves=0;passes=[];converged=False
        priorities=np.array([m.box for m in self.moves]+[a.value for a in self.arcs])
        groups=sorted({m.box for m in self.moves})
        bits=self.encode(kept)
        while solves<max_solves:
            before=len(kept);records=[];complete=True;certified=True
            for group in groups:
                if solves>=max_solves:complete=False;break
                fixed=priorities!=group
                values=np.array([bits.get(i,0) for i in range(self.core_variables)],dtype=float)
                lower=np.where(fixed,values,0.)
                upper=np.where(fixed,values,1.)
                result=milp(self.cost,integrality=np.ones(self.core_variables),bounds=Bounds(lower,upper),
                    constraints=LinearConstraint(self.matrix,self.lower,self.upper),
                    options=dict(time_limit=seconds_per_group,mip_rel_gap=0.,presolve=True))
                solves+=1;old=len(kept)
                certified=certified and result.status==0
                if result.x is not None:
                    x=np.rint(result.x);lhs=self.matrix@x
                    if (np.max(abs(x-result.x))>1e-5 or np.any(x<lower) or np.any(x>upper)
                            or np.any(lhs<np.array(self.lower)-1e-6) or np.any(lhs>np.array(self.upper)+1e-6)):
                        raise ValueError('Single-group reference violated fixed blocks or physical constraints')
                    candidate=tuple(j for j in range(self.count) if x[self.z[j]])
                    if len(candidate)<len(kept):
                        bits=self.encode(candidate);kept=candidate
                records.append(dict(group=group,before=old,after=len(kept),optimal=result.status==0,status=int(result.status)))
            passes.append(dict(before=before,after=len(kept),complete=complete,certified=certified,groups=records))
            if complete and before==len(kept):
                converged=certified
                break
        return kept,dict(solves=solves,passes=passes,coordinate_converged=converged,runtime_sec=REAL_TIME()-began)
