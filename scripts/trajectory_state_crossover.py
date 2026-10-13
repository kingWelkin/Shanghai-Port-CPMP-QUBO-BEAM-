"""Classical shortest-path crossover over physically identical parent states."""
from __future__ import annotations
from collections import deque
from dataclasses import replace
from container_reshuffle.core import Move
from container_reshuffle.solvers import bs_frg as frg
from research_receiver_allocation import physical_replay


def state_crossover(stacks,height,paths):
    initial=tuple(tuple(s) for s in stacks)
    edges={}
    for path in paths:
        physical_replay(stacks,height,path,goal=True)
        state=[list(s) for s in stacks]
        for move in path:
            before=tuple(tuple(s) for s in state)
            state[move.to_stack].append(state[move.from_stack].pop())
            after=tuple(tuple(s) for s in state)
            edges.setdefault(before,{})[after]=(move.box,move.from_stack,move.to_stack)
    queue=deque([initial]);parents={initial:None};goal=None
    while queue:
        state=queue.popleft()
        if frg._is_goal(state):goal=state;break
        for nxt,move in sorted(edges.get(state,{}).items()):
            if nxt not in parents:
                parents[nxt]=(state,move)
                queue.append(nxt)
    if goal is None:raise ValueError('Validated parent graph has no goal')
    transitions=[];current=goal
    while parents[current] is not None:
        before,action=parents[current]
        transitions.append((before,action))
        current=before
    result=[]
    for before,(box,s,d) in reversed(transitions):
        result.append(Move(len(result),box,s,d,len(before[s]),len(before[d])+1))
    physical_replay(stacks,height,result,goal=True)
    if len(result)>min(map(len,paths)):raise ValueError('Crossover lost its parent incumbent')
    return tuple(result),dict(states=len(parents),edges=sum(map(len,edges.values())))


def split_shared_prefix(stacks,height,parents):
    from frg_rollout_coordinates import move_core
    common=0
    for a,b in zip(*parents):
        if move_core(a)!=move_core(b):break
        common+=1
    prefix=tuple(parents[0][:common])
    state=physical_replay(stacks,height,prefix)
    tails=[tuple(replace(m,step=j) for j,m in enumerate(path[common:])) for path in parents]
    for tail in tails:physical_replay(state,height,tail,goal=True)
    return prefix,state,tails
