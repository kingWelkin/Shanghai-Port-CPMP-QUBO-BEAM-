"""Physical route domains with at most one early destination per container.

Each original event may retain its destination or use the destination of the
same container's next original event. All delayed bypasses remain available.
The complete declared domain is counted before materialization; resource
limits reject the whole domain rather than keeping a ranked option prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from container_reshuffle.core import Move
from complete_trajectory_domains import CompleteTrajectoryDomains, UnsupportedDomain
from production_selector_replay import REAL_TIME
from research_receiver_allocation import physical_replay


@dataclass(frozen=True)
class AdvanceRoute:
    box: int
    actions: tuple[tuple[int,int], ...]
    intervals: tuple[tuple[int,int,int], ...]

    @property
    def kept(self):
        return tuple(t for t,d in self.actions)


class AdvanceTrajectoryDomains(CompleteTrajectoryDomains):
    def __init__(self, stacks, height, moves, max_routes=20000):
        began=REAL_TIME()
        self.stacks=tuple(tuple(s) for s in stacks)
        self.height,self.moves=height,tuple(moves)
        physical_replay(stacks,height,self.moves,goal=True)
        self.end=2*len(moves)+2
        self.identities,self.values,self.origins,self.events=[],[],[],[]
        for source,stack in enumerate(stacks):
            ids=[]
            for level,value in enumerate(stack):
                ids.append(len(self.values));self.values.append(value)
                self.origins.append((source,-2*height+2*level));self.events.append([])
            self.identities.append(ids)
        state=[list(s) for s in self.identities]
        self.move_box=[]
        for t,m in enumerate(moves):
            box=state[m.from_stack].pop();state[m.to_stack].append(box)
            self.move_box.append(box);self.events[box].append(t)
        self.event_options={}
        counts=[]
        for box,steps in enumerate(self.events):
            ways={(self.origins[box][0],0):1}
            for j,t in enumerate(steps):
                targets=[(moves[t].to_stack,0)]
                if j+1<len(steps):
                    destination=moves[steps[j+1]].to_stack
                    if destination!=targets[0][0]:targets.append((destination,1))
                self.event_options[t]=tuple(targets)
                nxt=dict(ways)
                for (source,used),number in ways.items():
                    for destination,advance in targets:
                        if source!=destination and used+advance<=1:
                            key=(destination,used+advance)
                            nxt[key]=nxt.get(key,0)+number
                ways=nxt
            counts.append(sum(ways.values()))
        if sum(counts)>max_routes:
            raise UnsupportedDomain('COMPLETE_ONE_ADVANCE_DOMAIN_OVERSIZE')
        self.routes,self.all_groups,self.original=[],[],[]
        for box,steps in enumerate(self.events):
            paths={(self.origins[box][0],0):[()]}
            for t in steps:
                nxt={key:list(ps) for key,ps in paths.items()}
                for (source,used),ps in paths.items():
                    for destination,advance in self.event_options[t]:
                        if source!=destination and used+advance<=1:
                            nxt.setdefault((destination,used+advance),[]).extend(p+((t,destination),) for p in ps)
                paths=nxt
            pool=sorted((p for ps in paths.values() for p in ps),key=lambda p:(len(p),p))
            if len(pool)!=counts[box] or len(pool)!=len(set(pool)):
                raise ValueError('Complete early-destination enumeration mismatch')
            start=len(self.routes)
            self.routes.extend(self._route(box,p) for p in pool)
            group=list(range(start,len(self.routes)));self.all_groups.append(group)
            original=tuple((t,moves[t].to_stack) for t in steps)
            self.original.append(next(i for i in group if self.routes[i].actions==original))
        self.incumbent=list(self.original)
        self.replay_full(self.original)
        self.cost_all=np.array([len(r.actions) for r in self.routes],dtype=float)
        arc_index,arcs,route_arcs={},[],[]
        for route in self.routes:
            ids=[]
            for s,a,b in route.intervals:
                key=(route.box,s,a,b)
                if key not in arc_index:arc_index[key]=len(arcs);arcs.append(key)
                ids.append(arc_index[key])
            route_arcs.append(ids)
        masks=[0]*len(arcs)
        for i,(box,s,a,b) in enumerate(arcs):
            for j in range(i+1,len(arcs)):
                other,t,c,d=arcs[j]
                if box==other or s!=t:continue
                bad=(a<c<b<d or c<a<d<b or (b==d==self.end and
                    ((a<c and self.values[box]<self.values[other]) or
                     (c<a and self.values[other]<self.values[box]))))
                if bad:masks[i]|=1<<j;masks[j]|=1<<i
        self.arc_masks,self.block_masks=[],[]
        for ids in route_arcs:
            present=blocked=0
            for i in ids:present|=1<<i;blocked|=masks[i]
            self.arc_masks.append(present);self.block_masks.append(blocked)
        # New destination choices require occupancy checks at those early
        # push times too,not only the original destination/time pairs.
        points=[(s,-1) for s in range(len(stacks))]+sorted(
            {(destination,2*t+1) for t,targets in self.event_options.items() for destination,advance in targets})
        point_stack=np.array([s for s,t in points]);point_time=np.array([t for s,t in points])
        self.occupancy=np.zeros((len(self.routes),len(points)),dtype=np.int8)
        for i,route in enumerate(self.routes):
            for s,a,b in route.intervals:
                self.occupancy[i,(point_stack==s)&(point_time>=a)&(point_time<b)]=1
        self.domains=[list(g) for g in self.all_groups]
        self.stats=dict(total_routes=len(self.routes),max_routes=max(counts,default=0),arc_count=len(arcs),
            enumeration_sec=REAL_TIME()-began,repeated_priorities=len(set(self.values))!=len(self.values),
            domain='all_delayed_bypasses_plus_at_most_one_next_destination_advance_per_identity',
            capacity_points=len(points))
        self._propagate();self._prepare_reduced();self.stats['total_build_sec']=REAL_TIME()-began

    def _route(self,box,actions):
        source,start=self.origins[box];intervals=[]
        for t,destination in actions:
            if source==destination:raise ValueError('Early-destination route contains a self move')
            intervals.append((source,start,2*t));source,start=destination,2*t+1
        intervals.append((source,start,self.end))
        return AdvanceRoute(box,tuple(actions),tuple(intervals))

    def replay_full(self,chosen):
        if len(chosen)!=len(self.all_groups) or any(i not in g for i,g in zip(chosen,self.all_groups)):
            raise ValueError('Invalid full early-destination selection')
        state=[list(s) for s in self.identities];locations=[s for s,t in self.origins]
        events=sorted((t,route.box,destination) for i in chosen for route in [self.routes[i]]
                      for t,destination in route.actions)
        path=[]
        for t,box,destination in events:
            source=locations[box]
            if not state[source] or state[source][-1]!=box:
                raise ValueError('Early-destination source is inaccessible')
            if source==destination or len(state[destination])>=self.height:
                raise ValueError('Early-destination push is invalid')
            path.append(Move(len(path),self.values[box],source,destination,len(state[source]),len(state[destination])+1))
            state[destination].append(state[source].pop());locations[box]=destination
        physical_replay(self.stacks,self.height,path,goal=True)
        return tuple(path)

    def replay(self,chosen):
        return self.replay_full(self.full_selection(chosen))

    def action_changes(self,chosen):
        changed=early=0
        for i in self.full_selection(chosen):
            route=self.routes[i];source=self.origins[route.box][0]
            for t,destination in route.actions:
                changed += (source,destination)!=(self.moves[t].from_stack,self.moves[t].to_stack)
                early += destination!=self.moves[t].to_stack
                source=destination
        return dict(changed_moves=changed,early_destination_moves=early)
