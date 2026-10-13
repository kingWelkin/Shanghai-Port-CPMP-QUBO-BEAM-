"""Alternate classical state-graph shortcuts and physical one-box deletion."""
from __future__ import annotations
from trajectory_subsequence_qubo import SubsequenceModel
from trajectory_state_crossover import state_crossover
from research_receiver_allocation import physical_replay
from production_selector_replay import REAL_TIME


def simplify(stacks,height,path,max_passes=20):
    began=REAL_TIME();path=tuple(path);passes=[];converged=False
    physical_replay(stacks,height,path,goal=True)
    for _ in range(max_passes):
        before=len(path)
        graph,gm=state_crossover(stacks,height,[path])
        model=SubsequenceModel(stacks,height,graph)
        chosen,meta=model.classical_descent()
        path=model.replay(chosen)
        if len(path)>before:raise ValueError('Classical simplification increased path cost')
        passes.append(dict(before=before,after_graph=len(graph),after=len(path),graph=gm))
        if len(path)==before:
            converged=True;break
    physical_replay(stacks,height,path,goal=True)
    return path,dict(passes=passes,converged=converged,runtime_sec=REAL_TIME()-began)
