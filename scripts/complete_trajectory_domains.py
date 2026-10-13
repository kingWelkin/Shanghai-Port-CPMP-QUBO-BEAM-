"""Complete unique-priority route domains and sound physical propagation."""
from __future__ import annotations
from collections import deque
import itertools
import numpy as np

from container_reshuffle.core import Move
from container_reshuffle.qubo.model import QuboModel
from trajectory_subsequence_qubo import Route,SubsequenceModel
from research_receiver_allocation import physical_replay
from two_wave_action_qubo import binary_weights
from production_selector_replay import REAL_TIME


class UnsupportedDomain(ValueError):pass


class CompleteTrajectoryDomains:
    def __init__(self,stacks,height,moves,incumbent_kept=None,max_routes=20000):
        began=REAL_TIME()
        self.stacks=tuple(tuple(s) for s in stacks);self.height=height;self.moves=tuple(moves)
        physical_replay(stacks,height,moves,goal=True)
        flat=[v for s in stacks for v in s]
        if len(set(flat))!=len(flat):raise UnsupportedDomain('REPEATED_PRIORITIES_REQUIRE_GROUP_FLOW')
        self.end=2*len(moves)+2;self.identities=[];self.values=[];self.origins=[];events=[]
        for source,stack in enumerate(stacks):
            ids=[]
            for level,value in enumerate(stack):
                ids.append(len(self.values));self.values.append(value)
                self.origins.append((source,-2*height+2*level));events.append([])
            self.identities.append(ids)
        state=[list(s) for s in self.identities];self.move_box=[]
        for t,move in enumerate(moves):
            box=state[move.from_stack].pop();state[move.to_stack].append(box)
            self.move_box.append(box);events[box].append(t)
        # Count first;never truncate a domain to fit the resource bound.
        counts=[]
        for box,steps in enumerate(events):
            ways={self.origins[box][0]:1}
            for t in steps:
                m=self.moves[t];ways[m.to_stack]=ways.get(m.to_stack,0)+ways.get(m.from_stack,0)
            counts.append(sum(ways.values()))
        if sum(counts)>max_routes:raise UnsupportedDomain('COMPLETE_ROUTE_DOMAIN_OVERSIZE')
        self.routes=[];self.all_groups=[];self.original=[]
        for box,steps in enumerate(events):
            paths={self.origins[box][0]:[()]}
            for t in steps:
                move=self.moves[t];nxt={s:list(ps) for s,ps in paths.items()}
                nxt.setdefault(move.to_stack,[]).extend(p+(t,) for p in paths.get(move.from_stack,[]))
                paths=nxt
            pool=sorted((p for ps in paths.values() for p in ps),key=lambda p:(len(p),p))
            if len(pool)!=counts[box] or len(set(pool))!=len(pool):raise ValueError('Complete route enumeration mismatch')
            start=len(self.routes)
            self.routes.extend(SubsequenceModel.route(self,box,p) for p in pool)
            group=list(range(start,len(self.routes)));self.all_groups.append(group)
            self.original.append(next(i for i in group if self.routes[i].kept==tuple(steps)))
        current=tuple(range(len(moves))) if incumbent_kept is None else tuple(incumbent_kept)
        wanted=[[] for _ in events]
        for t in current:wanted[self.move_box[t]].append(t)
        self.incumbent=[next((i for i in g if self.routes[i].kept==tuple(wanted[b])),-1)
                        for b,g in enumerate(self.all_groups)]
        if min(self.incumbent,default=0)<0:raise ValueError('Incumbent route is absent from a complete domain')
        SubsequenceModel.replay(self,self.incumbent)
        self.cost_all=np.array([len(r.kept) for r in self.routes],dtype=float)
        # Arc bitsets make repeated route compatibility tests cheap.
        arc_index={};arc_values=[];route_arc_ids=[]
        for route in self.routes:
            ids=[]
            for s,a,b in route.intervals:
                key=(route.box,s,a,b)
                if key not in arc_index:
                    arc_index[key]=len(arc_values);arc_values.append(key)
                ids.append(arc_index[key])
            route_arc_ids.append(ids)
        masks=[0]*len(arc_values)
        for i,(box,s,a,b) in enumerate(arc_values):
            for j in range(i+1,len(arc_values)):
                other,t,c,d=arc_values[j]
                if box==other or s!=t:continue
                bad=(a<c<b<d or c<a<d<b or (b==d==self.end and
                    ((a<c and self.values[box]<self.values[other]) or (c<a and self.values[other]<self.values[box]))))
                if bad:masks[i]|=1<<j;masks[j]|=1<<i
        self.arc_masks=[];self.block_masks=[]
        for ids in route_arc_ids:
            present=0;blocked=0
            for i in ids:present|=1<<i;blocked|=masks[i]
            self.arc_masks.append(present);self.block_masks.append(blocked)
        points=[(s,-1) for s in range(len(stacks))]+[(m.to_stack,2*j+1) for j,m in enumerate(moves)]
        point_stack=np.array([p[0] for p in points]);point_time=np.array([p[1] for p in points])
        self.occupancy=np.zeros((len(self.routes),len(points)),dtype=np.int8)
        for i,route in enumerate(self.routes):
            for s,a,b in route.intervals:
                self.occupancy[i,(point_stack==s)&(point_time>=a)&(point_time<b)]=1
        self.domains=[list(g) for g in self.all_groups]
        self.stats=dict(total_routes=len(self.routes),max_routes=max(counts,default=0),
                        enumeration_sec=REAL_TIME()-began,arc_count=len(arc_values))
        self._propagate()
        self._prepare_reduced()
        self.stats['total_build_sec']=REAL_TIME()-began

    def conflict(self,i,j):return bool(self.block_masks[i]&self.arc_masks[j])

    def _propagate(self):
        began=REAL_TIME();supports={};tests=0;binary_removed=0;capacity_removed=0
        occupied_stacks=[set(s for i in g for s,a,b in self.routes[i].intervals) for g in self.all_groups]
        neighbors=[[h for h in range(len(self.domains)) if h!=g and occupied_stacks[g]&occupied_stacks[h]]
                   for g in range(len(self.domains))]
        queue=deque((g,h) for g,ns in enumerate(neighbors) for h in ns)
        while True:
            while queue:
                g,h=queue.popleft();other=set(self.domains[h]);kept=[]
                for i in self.domains[g]:
                    support=supports.get((i,h))
                    if support not in other:
                        support=None
                        for j in self.domains[h]:
                            tests+=1
                            if not self.conflict(i,j):support=j;break
                        supports[i,h]=support
                    if support is not None:kept.append(i)
                if len(kept)!=len(self.domains[g]):
                    binary_removed+=len(self.domains[g])-len(kept);self.domains[g]=kept
                    if not kept:raise ValueError('Sound propagation excluded a known feasible trajectory')
                    queue.extend((k,g) for k in neighbors[g] if k!=h)
            minima=[self.occupancy[g].min(axis=0).astype(np.int16) for g in self.domains]
            total=np.sum(minima,axis=0);changed=[]
            for g,domain in enumerate(self.domains):
                possible=self.occupancy[domain]+total-minima[g]
                kept=[i for i,valid in zip(domain,np.all(possible<=self.height,axis=1)) if valid]
                if len(kept)!=len(domain):
                    capacity_removed+=len(domain)-len(kept);self.domains[g]=kept;changed.append(g)
                    if not kept:raise ValueError('Capacity propagation excluded the original trajectory')
            if not changed:break
            for g in changed:queue.extend((h,g) for h in neighbors[g])
        for g,domain in enumerate(self.domains):
            if self.original[g] not in domain or self.incumbent[g] not in domain:
                raise ValueError('Propagation removed a replay-validated feasible option')
        self.stats.update(binary_removed=binary_removed,capacity_removed=capacity_removed,
            support_tests=tests,surviving_routes=sum(map(len,self.domains)),
            active_groups=sum(len(g)>1 for g in self.domains),propagate_sec=REAL_TIME()-began)

    def _prepare_reduced(self):
        self.active_boxes=[b for b,d in enumerate(self.domains) if len(d)>1]
        self.fixed={b:d[0] for b,d in enumerate(self.domains) if len(d)==1}
        self.constant=float(sum(self.cost_all[i] for i in self.fixed.values()))
        self.route_ids=[];self.groups=[];self.current=[]
        for b in self.active_boxes:
            start=len(self.route_ids);self.route_ids.extend(self.domains[b])
            self.groups.append(np.arange(start,len(self.route_ids),dtype=int))
            self.current.append(start+self.domains[b].index(self.incumbent[b]))
        self.unary=self.cost_all[self.route_ids]
        self.pair=np.zeros((len(self.route_ids),len(self.route_ids)),dtype=bool)
        for left,right in itertools.combinations(self.groups,2):
            for i in left:
                for j in right:
                    self.pair[i,j]=self.pair[j,i]=self.conflict(self.route_ids[i],self.route_ids[j])
        rows=set()
        for point in range(self.occupancy.shape[1]):
            fixed=sum(int(self.occupancy[i,point]) for i in self.fixed.values());variables=[];varying=0
            for indices in self.groups:
                loaded=[int(i) for i in indices if self.occupancy[self.route_ids[i],point]]
                if len(loaded)==len(indices):fixed+=1
                elif loaded:variables.extend(loaded);varying+=1
            cap=self.height-fixed
            if cap<varying:
                if cap<0:raise ValueError('Fixed occupancy exceeds capacity')
                rows.add((tuple(variables),cap))
        self.capacity=sorted(rows)
        self.loads=np.zeros((len(self.route_ids),len(self.capacity)),dtype=np.int16)
        for r,(indices,cap) in enumerate(self.capacity):self.loads[list(indices),r]=1
        self.caps=np.array([cap for indices,cap in self.capacity],dtype=np.int16)
        self.penalty=int(self.cost_all[self.incumbent].sum()-self.constant)+1
        self.qubo=None

    def compile_qubo(self,max_variables=4096):
        auxiliary=sum(len(binary_weights(cap)) for _,cap in self.capacity)
        if len(self.route_ids)+auxiliary>max_variables:raise UnsupportedDomain('REDUCED_QUBO_OVERSIZE')
        q=QuboModel();q.offset=self.constant
        for i,c in enumerate(self.unary):q.add_linear(q.add_binary_var(f'route_{self.route_ids[i]}'),c)
        for group in self.groups:q.add_one_hot_penalty(list(map(int,group)),self.penalty)
        for i,j in zip(*np.where(np.triu(self.pair,1))):q.add_quadratic(int(i),int(j),self.penalty)
        self.slack=[]
        for r,(indices,cap) in enumerate(self.capacity):
            terms={i:1 for i in indices}
            slack=[(q.add_binary_var(f'capacity_{r}_{j}'),w) for j,w in enumerate(binary_weights(cap))]
            self.slack.append(slack);terms.update(dict(slack));q.offset+=self.penalty*cap*cap
            for i,a in terms.items():q.add_linear(i,self.penalty*(a*a-2*cap*a))
            for (i,a),(j,b) in itertools.combinations(terms.items(),2):q.add_quadratic(i,j,2*self.penalty*a*b)
        self.qubo=q;return q

    def bits(self,chosen):
        if self.qubo is None:self.compile_qubo()
        values={int(i):1 for i in chosen}
        load=self.loads[list(chosen)].sum(axis=0) if chosen else np.zeros(len(self.caps),dtype=int)
        for r,slack in enumerate(self.slack):
            remaining=max(0,int(self.caps[r]-load[r]))
            for var,w in reversed(slack):
                if w<=remaining:values[var]=1;remaining-=w
            if remaining:raise ValueError('Residual capacity slack cannot be encoded')
        return values

    def full_selection(self,chosen):
        if len(chosen)!=len(self.groups) or any(i not in g for i,g in zip(chosen,self.groups)):
            raise ValueError('Invalid categorical one-hot choice')
        full=[None]*len(self.domains)
        for b,i in self.fixed.items():full[b]=i
        for b,i in zip(self.active_boxes,chosen):full[b]=self.route_ids[i]
        return full

    def replay(self,chosen):
        return SubsequenceModel.replay(self,self.full_selection(chosen))
