from pathlib import Path
import itertools
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from advance_trajectory_domains import AdvanceTrajectoryDomains
from bypass_trajectory_domains import BypassTrajectoryDomains
from categorical_qubo_search import solve
from complete_trajectory_domains import UnsupportedDomain
from container_reshuffle.core import Move
from test_trajectory_subsequence_qubo import sample


def optimum(model):
    q=model.compile_qubo()
    selected,meta=solve(model.unary,model.groups,model.pair,model.loads,model.caps,
                        model.penalty,model.constant,model.current)
    path=model.replay(selected)
    assert meta['optimal'] and q.evaluate(model.bits(selected))==len(path)
    return selected,path


def test_early_destination_improves_when_delaying_would_block_the_next_container():
    stacks=[[6,1,3,4],[2],[5],[7]]
    path=(Move(0,4,0,1,4,2),Move(1,3,0,3,3,2),Move(2,4,1,2,2,2))
    _,late=optimum(BypassTrajectoryDomains(stacks,4,path))
    model=AdvanceTrajectoryDomains(stacks,4,path)
    chosen,early=optimum(model)
    assert len(late)==3 and len(early)==2
    assert (early[0].box,early[0].from_stack,early[0].to_stack)==(4,0,2)
    assert model.action_changes(chosen)['early_destination_moves']==1


def test_all_small_routes_match_physical_feasibility_and_preserve_late_domains():
    for seed in range(8):
        original,path=sample(seed,steps=5)
        for repeated in (False,True):
            stacks=[[((v+1)//2 if repeated else v) for v in s] for s in original]
            moves=[Move(m.step,((m.box+1)//2 if repeated else m.box),m.from_stack,m.to_stack,m.from_level,m.to_level) for m in path]
            model=AdvanceTrajectoryDomains(stacks,3,moves)
            late=BypassTrajectoryDomains(stacks,3,moves)
            for old_group,new_group in zip(late.all_groups,model.all_groups):
                new_actions={model.routes[i].actions for i in new_group}
                assert all(tuple((t,moves[t].to_stack) for t in late.routes[i].kept) in new_actions for i in old_group)
            feasible=[]
            for chosen in itertools.product(*model.all_groups):
                physical=True
                try:replayed=model.replay_full(chosen)
                except ValueError:physical=False
                logical=(not any(model.conflict(a,b) for a,b in itertools.combinations(chosen,2)) and
                         np.all(model.occupancy[list(chosen)].sum(axis=0)<=3))
                assert physical==logical
                if physical:
                    assert all(i in g for i,g in zip(chosen,model.domains))
                    feasible.append(len(replayed))
            selected,result=optimum(model)
            assert len(result)==min(feasible)
            for i in model.full_selection(selected):
                assert sum(d!=moves[t].to_stack for t,d in model.routes[i].actions)<=1


def test_whole_domain_oversize_rejection():
    stacks,moves=sample(3)
    with pytest.raises(UnsupportedDomain,match='COMPLETE_ONE_ADVANCE_DOMAIN_OVERSIZE'):
        AdvanceTrajectoryDomains(stacks,3,moves,max_routes=1)
