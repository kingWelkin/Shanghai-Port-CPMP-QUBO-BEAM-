"""Complete one-redirect route neighborhoods conditioned on fixed box routes."""
from __future__ import annotations

from functools import lru_cache
import numpy as np

from advance_trajectory_domains import AdvanceRoute, AdvanceTrajectoryDomains
from complete_trajectory_domains import UnsupportedDomain
from production_selector_replay import REAL_TIME
from research_receiver_allocation import physical_replay


class PhysicalTrace:
    def __init__(self,stacks,height,moves):
        self.stacks=tuple(tuple(s) for s in stacks);self.height=height;self.moves=tuple(moves)
        physical_replay(stacks,height,self.moves,goal=True)
        self.end=2*len(moves)+2
        self.identities,self.values,self.origins,self.events=[],[],[],[]
        for source,stack in enumerate(stacks):
            ids=[]
            for level,value in enumerate(stack):
                ids.append(len(self.values));self.values.append(value)
                self.origins.append((source,-2*height+2*level));self.events.append([])
            self.identities.append(ids)
        state=[list(s) for s in self.identities];self.move_box=[]
        self.contacts=np.zeros((len(self.values),len(self.values)),dtype=int)
        def contact(a,b):
            self.contacts[a,b]+=1;self.contacts[b,a]+=1
        for t,m in enumerate(moves):
            box=state[m.from_stack].pop()
            if state[m.from_stack]:contact(box,state[m.from_stack][-1])
            if state[m.to_stack]:contact(box,state[m.to_stack][-1])
            state[m.to_stack].append(box)
            self.move_box.append(box);self.events[box].append(t)
        self.original_routes=[]
        for b,steps in enumerate(self.events):
            source,start=self.origins[b];intervals=[];actions=[]
            for t in steps:
                intervals.append((source,start,2*t));source,start=moves[t].to_stack,2*t+1
                actions.append((t,source))
            intervals.append((source,start,self.end))
            self.original_routes.append(AdvanceRoute(b,tuple(actions),tuple(intervals)))


def contact_releases(trace,width=8,maximum=12):
    """Original-trace contacts and move counts only;no optimized-answer input."""
    active=[b for b,steps in enumerate(trace.events) if steps]
    seeds=sorted((b for b in active if len(trace.events[b])>=2),key=lambda b:(-len(trace.events[b]),b))
    releases=[];seen=set()
    for seed in seeds:
        group=[seed]
        while len(group)<min(width,len(active)):
            options=[b for b in active if b not in group]
            if not options:break
            candidate=max(options,key=lambda b:(sum(int(trace.contacts[b,g]) for g in group),len(trace.events[b]),-b))
            if sum(int(trace.contacts[candidate,g]) for g in group)==0:break
            group.append(candidate)
        key=tuple(sorted(group))
        if key not in seen:
            releases.append(dict(seed=seed,boxes=key));seen.add(key)
        if len(releases)>=maximum:break
    return releases


class ConditionedTrajectoryDomains(AdvanceTrajectoryDomains):
    def __init__(self,stacks,height,moves,released,*,trace=None,max_routes=20000):
        began=REAL_TIME()
        trace=PhysicalTrace(stacks,height,moves) if trace is None else trace
        if trace.stacks!=tuple(tuple(s) for s in stacks) or trace.height!=height or trace.moves!=tuple(moves):
            raise ValueError('Conditioning trace does not match the input')
        for name in ('stacks','height','moves','end','identities','values','origins','events','move_box'):
            setattr(self,name,getattr(trace,name))
        released=set(map(int,released))
        if not released<=set(range(len(self.values))):raise ValueError('Unknown released identity')
        self.released=released
        fixed=[r for b,r in enumerate(trace.original_routes) if b not in released]
        fixed_intervals=[[] for s in stacks]
        times=np.arange(-1,self.end)
        fixed_occupancy=np.zeros((len(stacks),len(times)),dtype=np.int16)
        for route in fixed:
            for s,a,b in route.intervals:
                fixed_intervals[s].append((route.box,a,b))
                fixed_occupancy[s,(times>=a)&(times<b)]+=1
        self.event_options={t:tuple((d,int(d!=m.to_stack)) for d in range(len(stacks)))
                            for t,m in enumerate(moves)}
        self.next_destinations={steps[j]:moves[steps[j+1]].to_stack for steps in self.events for j in range(len(steps)-1)}
        @lru_cache(None)
        def interval_ok(box,s,a,b):
            for other,c,d in fixed_intervals[s]:
                if a<c<b<d or c<a<d<b:return False
                if b==d==self.end and ((a<c and self.values[box]<self.values[other]) or
                                       (c<a and self.values[other]<self.values[box])):return False
            lo=max(-1,a)+1;hi=min(self.end,b)+1
            return lo>=hi or bool(np.max(fixed_occupancy[s,lo:hi])+1<=height)
        counters={};counts=[]
        def make_counter(box):
            steps=self.events[box]
            @lru_cache(None)
            def count(j,source,start,used):
                if j==len(steps):return int(interval_ok(box,source,start,self.end))
                t=steps[j]
                total=count(j+1,source,start,used)
                if interval_ok(box,source,start,2*t):
                    for destination,redirect in self.event_options[t]:
                        if (destination!=source and used+redirect<=1 and
                            fixed_occupancy[destination,2*t+2]<height):
                            total+=count(j+1,destination,2*t+1,used+redirect)
                return total
            return count
        for box,steps in enumerate(self.events):
            if box not in released:counts.append(1);continue
            count=make_counter(box);counters[box]=count
            source,start=self.origins[box]
            number=count(0,source,start,0)
            if number<=0:raise ValueError('Fixed-route conditioning excluded the original path')
            counts.append(number)
        if sum(counts)>max_routes:raise UnsupportedDomain('COMPLETE_CONDITIONED_ROUTE_DOMAIN_OVERSIZE')
        self.routes,self.all_groups,self.original=[],[],[]
        for box,steps in enumerate(self.events):
            if box not in released:
                pool=[trace.original_routes[box].actions]
            else:
                count=counters[box];pool=[]
                def visit(j,source,start,used,actions):
                    if count(j,source,start,used)==0:return
                    if j==len(steps):pool.append(actions);return
                    visit(j+1,source,start,used,actions)
                    t=steps[j]
                    if interval_ok(box,source,start,2*t):
                        for destination,redirect in self.event_options[t]:
                            if (destination!=source and used+redirect<=1 and
                                fixed_occupancy[destination,2*t+2]<height):
                                visit(j+1,destination,2*t+1,used+redirect,actions+((t,destination),))
                source,start=self.origins[box];visit(0,source,start,0,())
                pool.sort(key=lambda p:(len(p),p))
            if len(pool)!=counts[box] or len(pool)!=len(set(pool)):
                raise ValueError('Conditioned complete route count mismatch')
            start=len(self.routes);self.routes.extend(self._route(box,p) for p in pool)
            group=list(range(start,len(self.routes)));self.all_groups.append(group)
            original=trace.original_routes[box].actions
            self.original.append(next(i for i in group if self.routes[i].actions==original))
        self.incumbent=list(self.original);self.replay_full(self.original)
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
                if (a<c<b<d or c<a<d<b or (b==d==self.end and
                    ((a<c and self.values[box]<self.values[other]) or
                     (c<a and self.values[other]<self.values[box])))):
                    masks[i]|=1<<j;masks[j]|=1<<i
        self.arc_masks,self.block_masks=[],[]
        for ids in route_arcs:
            present=blocked=0
            for i in ids:present|=1<<i;blocked|=masks[i]
            self.arc_masks.append(present);self.block_masks.append(blocked)
        points=[(s,-1) for s in range(len(stacks))]
        for t,m in enumerate(moves):
            destinations=range(len(stacks)) if self.move_box[t] in released else (m.to_stack,)
            points.extend((d,2*t+1) for d in destinations)
        point_stack=np.array([s for s,t in points]);point_time=np.array([t for s,t in points])
        self.occupancy=np.zeros((len(self.routes),len(points)),dtype=np.int8)
        for i,route in enumerate(self.routes):
            for s,a,b in route.intervals:
                self.occupancy[i,(point_stack==s)&(point_time>=a)&(point_time<b)]=1
        self.domains=[list(g) for g in self.all_groups]
        self.stats=dict(total_routes=len(self.routes),max_routes=max(counts,default=0),arc_count=len(arcs),
            enumeration_sec=REAL_TIME()-began,repeated_priorities=len(set(self.values))!=len(self.values),
            released_boxes=sorted(released),fixed_boxes=len(self.values)-len(released),
            domain='complete_one_redirect_conditioned_on_all_unreleased_original_routes',
            interval_checks=interval_ok.cache_info().misses,
            count_states=sum(f.cache_info().misses for f in counters.values()),capacity_points=len(points))
        self._propagate();self._prepare_reduced();self.stats['total_build_sec']=REAL_TIME()-began

    def action_changes(self,chosen):
        changes=super().action_changes(chosen)
        redirects=changes['early_destination_moves'];early=0
        for i in self.full_selection(chosen):
            for t,d in self.routes[i].actions:
                early += d!=self.moves[t].to_stack and d==self.next_destinations.get(t)
        return dict(changed_moves=changes['changed_moves'],early_destination_moves=early,destination_redirects=redirects)
