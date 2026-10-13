"""Exact QUBO factor storage without eager quadratic coefficient expansion.

The binary variables,penalties andenergy are unchanged. materialize() exports
the ordinary explicit QuboModel when a matrix consumer actually needs it.
"""
from __future__ import annotations

import itertools
import numpy as np

from conditioned_trajectory_domains import ConditionedTrajectoryDomains
from complete_trajectory_domains import UnsupportedDomain
from container_reshuffle.qubo.model import QuboModel
from two_wave_action_qubo import binary_weights


class FactorizedTrajectoryQubo:
    def __init__(self,model,max_variables=4096):
        weights=[tuple(map(int,binary_weights(int(cap)))) for indices,cap in model.capacity]
        size=len(model.route_ids)+sum(map(len,weights))
        if size>max_variables:raise UnsupportedDomain('REDUCED_QUBO_OVERSIZE')
        self.constant=float(model.constant);self.penalty=float(model.penalty)
        self.unary=np.array(model.unary,dtype=float,copy=True)
        self.groups=tuple(np.asarray(g,dtype=np.int64).copy() for g in model.groups)
        self.pair=np.array(model.pair,dtype=bool,copy=True)
        self.loads=np.array(model.loads,copy=True)
        self.caps=np.asarray([int(cap) for indices,cap in model.capacity],dtype=np.int64)
        self.capacity=tuple((tuple(map(int,indices)),int(cap)) for indices,cap in model.capacity)
        self.var_index={f'route_{route}':i for i,route in enumerate(model.route_ids)}
        self.slack=[]
        for row,ws in enumerate(weights):
            variables=[]
            for j,w in enumerate(ws):
                index=len(self.var_index);self.var_index[f'capacity_{row}_{j}']=index
                variables.append((index,w))
            self.slack.append(variables)
        self.index_var={i:name for name,i in self.var_index.items()}
        self.offset=self.constant+self.penalty*(len(self.groups)+sum(int(c)*int(c) for c in self.caps))
        self._slack_rows=np.array([r for r,variables in enumerate(self.slack) for i,w in variables],dtype=np.int64)
        self._slack_weights=np.array([w for variables in self.slack for i,w in variables],dtype=np.int64)

    def evaluate(self,binary_values):
        values=np.fromiter((int(binary_values.get(i,0)) for i in range(len(self.var_index))),
                           dtype=np.int64,count=len(self.var_index))
        if np.any((values!=0)&(values!=1)):raise ValueError('QUBO evaluation requires binary assignments')
        x=values[:len(self.unary)];selected=np.flatnonzero(x)
        onehot=sum((int(x[g].sum())-1)**2 for g in self.groups)
        conflicts=int(np.count_nonzero(np.triu(self.pair[np.ix_(selected,selected)],1)))
        load=self.loads[selected].sum(axis=0,dtype=np.int64) if len(selected) else np.zeros(len(self.caps),dtype=np.int64)
        slack=np.zeros(len(self.caps),dtype=np.int64)
        np.add.at(slack,self._slack_rows,values[len(self.unary):]*self._slack_weights)
        residual=load+slack-self.caps
        return float(self.constant+float(self.unary@x)+self.penalty*(onehot+conflicts+int(residual@residual)))

    def materialize(self):
        """Export exactly the original sparse coefficient dictionary on demand."""
        q=QuboModel();q.offset=self.constant
        for name in self.var_index:q.add_binary_var(name)
        for i,c in enumerate(self.unary):q.add_linear(i,float(c))
        for group in self.groups:q.add_one_hot_penalty(list(map(int,group)),self.penalty)
        for i,j in zip(*np.where(np.triu(self.pair,1))):q.add_quadratic(int(i),int(j),self.penalty)
        for row,(indices,cap) in enumerate(self.capacity):
            terms={i:1 for i in indices};terms.update(dict(self.slack[row]))
            q.offset+=self.penalty*cap*cap
            for i,a in terms.items():q.add_linear(i,self.penalty*(a*a-2*cap*a))
            for (i,a),(j,b) in itertools.combinations(terms.items(),2):q.add_quadratic(i,j,2*self.penalty*a*b)
        return q


class FactorizedConditionedDomains(ConditionedTrajectoryDomains):
    def compile_qubo(self,max_variables=4096):
        q=FactorizedTrajectoryQubo(self,max_variables)
        self.slack=q.slack;self.qubo=q
        self.stats['qubo_representation']='exact_factors_with_on_demand_coefficients'
        return q
