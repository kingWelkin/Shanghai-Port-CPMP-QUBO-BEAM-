"""One destination redirection per physical identity with sound unary pruning."""
from __future__ import annotations

import numpy as np

from advance_trajectory_domains import AdvanceTrajectoryDomains
from complete_trajectory_domains import UnsupportedDomain
from production_selector_replay import REAL_TIME
from research_receiver_allocation import physical_replay


class RedirectTrajectoryDomains(AdvanceTrajectoryDomains):
    def __init__(self,stacks,height,moves,max_routes=20000):
        began=REAL_TIME()
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
        for t,m in enumerate(moves):
            box=state[m.from_stack].pop();state[m.to_stack].append(box)
            self.move_box.append(box);self.events[box].append(t)
        fixed={b for b,steps in enumerate(self.events) if not steps}
        fixed_heights=[];fixed_tops=[]
        for ids in self.identities:
            immutable=[b for b in ids if b in fixed]
            if ids[:len(immutable)]!=immutable:
                raise ValueError('A feasible trace cannot move below a permanently fixed container')
            fixed_heights.append(len(immutable))
            fixed_tops.append(self.values[immutable[-1]] if immutable else float('inf'))
        self.event_options={};self.next_destinations={};suffix_tables=[];counts=[]
        for box,steps in enumerate(self.events):
            allowed_ends={s for s in range(len(stacks))
                          if fixed_heights[s]<height and self.values[box]<=fixed_tops[s]}
            if not steps:allowed_ends={self.origins[box][0]}
            for j,t in enumerate(steps):
                original=moves[t].to_stack
                self.event_options[t]=tuple((s,int(s!=original)) for s in range(len(stacks))
                                            if fixed_heights[s]<height)
                if j+1<len(steps):self.next_destinations[t]=moves[steps[j+1]].to_stack
            # Count suffixes before allocating routes. States with no legal
            # terminal suffix are also omitted during enumeration.
            tables=[None]*(len(steps)+1)
            tables[-1]={(s,used):int(s in allowed_ends) for s in range(len(stacks)) for used in (0,1)}
            for j in range(len(steps)-1,-1,-1):
                t=steps[j];following=tables[j+1];current={}
                for source in range(len(stacks)):
                    for used in (0,1):
                        count=following[source,used]
                        for destination,redirect in self.event_options[t]:
                            if source!=destination and used+redirect<=1:
                                count+=following[destination,used+redirect]
                        current[source,used]=count
                tables[j]=current
            count=tables[0][self.origins[box][0],0]
            if count<=0:raise ValueError('Unary pruning excluded a known feasible original route')
            suffix_tables.append(tables);counts.append(count)
        if sum(counts)>max_routes:
            raise UnsupportedDomain('COMPLETE_ONE_REDIRECT_DOMAIN_OVERSIZE')
        self.routes,self.all_groups,self.original=[],[],[]
        for box,steps in enumerate(self.events):
            pool=[];tables=suffix_tables[box]
            def visit(j,source,used,actions):
                if tables[j][source,used]==0:return
                if j==len(steps):pool.append(actions);return
                visit(j+1,source,used,actions)
                t=steps[j]
                for destination,redirect in self.event_options[t]:
                    if source!=destination and used+redirect<=1:
                        visit(j+1,destination,used+redirect,actions+((t,destination),))
            visit(0,self.origins[box][0],0,())
            pool.sort(key=lambda p:(len(p),p))
            if len(pool)!=counts[box] or len(pool)!=len(set(pool)):
                raise ValueError('Complete redirect route count mismatch')
            start=len(self.routes);self.routes.extend(self._route(box,p) for p in pool)
            group=list(range(start,len(self.routes)));self.all_groups.append(group)
            original=tuple((t,moves[t].to_stack) for t in steps)
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
                bad=(a<c<b<d or c<a<d<b or (b==d==self.end and
                    ((a<c and self.values[box]<self.values[other]) or
                     (c<a and self.values[other]<self.values[box]))))
                if bad:masks[i]|=1<<j;masks[j]|=1<<i
        self.arc_masks,self.block_masks=[],[]
        for ids in route_arcs:
            present=blocked=0
            for i in ids:present|=1<<i;blocked|=masks[i]
            self.arc_masks.append(present);self.block_masks.append(blocked)
        points=[(s,-1) for s in range(len(stacks))]+sorted(
            {(d,2*t+1) for t,targets in self.event_options.items() for d,redirect in targets})
        point_stack=np.array([s for s,t in points]);point_time=np.array([t for s,t in points])
        self.occupancy=np.zeros((len(self.routes),len(points)),dtype=np.int8)
        for i,route in enumerate(self.routes):
            for s,a,b in route.intervals:
                self.occupancy[i,(point_stack==s)&(point_time>=a)&(point_time<b)]=1
        self.domains=[list(g) for g in self.all_groups]
        self.stats=dict(total_routes=len(self.routes),max_routes=max(counts,default=0),arc_count=len(arcs),
            enumeration_sec=REAL_TIME()-began,repeated_priorities=len(set(self.values))!=len(self.values),
            domain='all_delayed_bypasses_plus_one_arbitrary_destination_redirect_per_identity',
            unary_pruning='permanently_full_destinations_and_immutable_prefix_terminal_order',
            fixed_heights=fixed_heights,capacity_points=len(points))
        self._propagate();self._prepare_reduced();self.stats['total_build_sec']=REAL_TIME()-began

    def action_changes(self,chosen):
        changes=super().action_changes(chosen)
        redirects=changes['early_destination_moves']
        early=0
        for i in self.full_selection(chosen):
            for t,d in self.routes[i].actions:
                early += d!=self.moves[t].to_stack and d==self.next_destinations.get(t)
        return dict(changed_moves=changes['changed_moves'],early_destination_moves=early,
                    destination_redirects=redirects)
