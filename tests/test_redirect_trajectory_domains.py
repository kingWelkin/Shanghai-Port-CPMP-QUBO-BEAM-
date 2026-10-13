from pathlib import Path
import itertools
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from advance_trajectory_domains import AdvanceTrajectoryDomains
from redirect_trajectory_domains import RedirectTrajectoryDomains
from complete_trajectory_domains import UnsupportedDomain
from test_advance_trajectory_domains import optimum
from test_trajectory_subsequence_qubo import sample


def all_unpruned_routes(model,box):
    paths={(model.origins[box][0],0):[()]}
    for t in model.events[box]:
        nxt={key:list(ps) for key,ps in paths.items()}
        for (s,used),ps in paths.items():
            for d in range(len(model.stacks)):
                change=int(d!=model.moves[t].to_stack)
                if s!=d and used+change<=1:
                    nxt.setdefault((d,used+change),[]).extend(p+((t,d),) for p in ps)
        paths=nxt
    return [model._route(box,p) for ps in paths.values() for p in ps]


def test_unary_pruning_preserves_every_physically_feasible_small_assignment():
    for seed in range(6):
        original,path=sample(seed,steps=4)
        for repeated in (False,True):
            stacks=[[((v+1)//2 if repeated else v) for v in s] for s in original]
            from container_reshuffle.core import Move
            moves=[Move(m.step,((m.box+1)//2 if repeated else m.box),m.from_stack,m.to_stack,m.from_level,m.to_level) for m in path]
            model=RedirectTrajectoryDomains(stacks,3,moves)
            raw=[all_unpruned_routes(model,b) for b in range(len(model.values))]
            reduced=[{model.routes[i].actions for i in g} for g in model.domains]
            best=len(moves)
            # Independent replay of all raw products also tests routes removed
            # before the usual compatibility/capacity propagation stage.
            for selected in itertools.product(*raw):
                ids=[list(s) for s in model.identities];locations=[s for s,t in model.origins]
                events=sorted((t,r.box,d) for r in selected for t,d in r.actions)
                valid=True
                for t,b,d in events:
                    s=locations[b]
                    if not ids[s] or ids[s][-1]!=b or s==d or len(ids[d])>=3:valid=False;break
                    ids[d].append(ids[s].pop());locations[b]=d
                if valid:
                    valid=all(all(model.values[a]>=model.values[b] for a,b in zip(s,s[1:])) for s in ids)
                if valid:
                    assert all(r.actions in domain for r,domain in zip(selected,reduced))
                    best=min(best,len(events))
            chosen,result=optimum(model)
            assert len(result)==best
            assert model.action_changes(chosen)['destination_redirects']<=len(model.groups)


def test_early_domain_legal_routes_remain_represented_and_counts_are_complete():
    for seed in range(6):
        stacks,moves=sample(seed,steps=5)
        early=AdvanceTrajectoryDomains(stacks,3,moves)
        model=RedirectTrajectoryDomains(stacks,3,moves)
        represented=[{model.routes[i].actions for i in g} for g in model.all_groups]
        for chosen in itertools.product(*early.all_groups):
            try:early.replay_full(chosen)
            except ValueError:continue
            assert all(early.routes[i].actions in target for i,target in zip(chosen,represented))


def test_redirect_domain_is_never_rank_truncated():
    stacks,moves=sample(2)
    with pytest.raises(UnsupportedDomain,match='COMPLETE_ONE_REDIRECT_DOMAIN_OVERSIZE'):
        RedirectTrajectoryDomains(stacks,3,moves,max_routes=1)
