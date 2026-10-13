from pathlib import Path
import itertools
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from conditioned_trajectory_domains import PhysicalTrace,ConditionedTrajectoryDomains,contact_releases
from container_reshuffle.core import Move
from test_redirect_trajectory_domains import all_unpruned_routes
from test_advance_trajectory_domains import optimum
from test_trajectory_subsequence_qubo import sample


def test_fixed_route_conditioning_keeps_all_and_only_legal_small_combinations():
    for seed in range(6):
        original,path=sample(seed,steps=5)
        for repeated in (False,True):
            stacks=[[((v+1)//2 if repeated else v) for v in s] for s in original]
            moves=[Move(m.step,((m.box+1)//2 if repeated else m.box),m.from_stack,m.to_stack,m.from_level,m.to_level) for m in path]
            trace=PhysicalTrace(stacks,3,moves)
            active=[b for b,events in enumerate(trace.events) if events]
            for released in ((),tuple(active[:2]),tuple(active)):
                model=ConditionedTrajectoryDomains(stacks,3,moves,released,trace=trace)
                raw=[all_unpruned_routes(model,b) if b in released else [trace.original_routes[b]]
                     for b in range(len(model.values))]
                retained=[{model.routes[i].actions for i in g} for g in model.domains]
                best=len(moves)
                for choices in itertools.product(*raw):
                    state=[list(s) for s in trace.identities];where=[s for s,t in trace.origins]
                    events=sorted((t,r.box,d) for r in choices for t,d in r.actions);valid=True
                    for t,b,d in events:
                        s=where[b]
                        if not state[s] or state[s][-1]!=b or s==d or len(state[d])>=3:valid=False;break
                        state[d].append(state[s].pop());where[b]=d
                    if valid:valid=all(all(trace.values[a]>=trace.values[b] for a,b in zip(s,s[1:])) for s in state)
                    if valid:
                        assert all(r.actions in domain for r,domain in zip(choices,retained))
                        best=min(best,len(events))
                chosen,result=optimum(model)
                assert len(result)==best
                full=model.full_selection(chosen)
                assert all(model.routes[full[b]].actions==trace.original_routes[b].actions
                           for b in range(len(trace.values)) if b not in released)


def test_contact_releases_are_deterministic_bounded_and_do_not_use_answers():
    stacks,moves=sample(23,steps=14)
    trace=PhysicalTrace(stacks,3,moves)
    first=contact_releases(trace,width=3,maximum=4)
    assert first==contact_releases(trace,width=3,maximum=4)
    assert len(first)<=4
    assert len({r['boxes'] for r in first})==len(first)
    assert all(r['seed'] in r['boxes'] and len(r['boxes'])<=3 for r in first)
    assert all(trace.events[b] for r in first for b in r['boxes'])
