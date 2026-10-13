"""Complete delayed-bypass route domains on a fixed incumbent event clock.

At an event originally assigned to a container, it may stay or move from its
current stack to that event's original destination. Skipping earlier events
can therefore create direct moves that were absent from the original path.
Repeated priorities retain the original physical identities in this domain;
the domain does not claim completeness over arbitrary identity rematching.
"""
from __future__ import annotations

import numpy as np

from container_reshuffle.core import Move
from complete_trajectory_domains import CompleteTrajectoryDomains, UnsupportedDomain
from production_selector_replay import REAL_TIME
from research_receiver_allocation import physical_replay
from trajectory_subsequence_qubo import Route


class BypassTrajectoryDomains(CompleteTrajectoryDomains):
    def __init__(self, stacks, height, moves, max_routes=20000):
        started = REAL_TIME()
        self.stacks = tuple(tuple(s) for s in stacks)
        self.height, self.moves = height, tuple(moves)
        physical_replay(stacks, height, self.moves, goal=True)
        self.end = 2*len(moves)+2
        self.identities, self.values, self.origins, self.events = [], [], [], []
        for source, stack in enumerate(stacks):
            ids = []
            for level, value in enumerate(stack):
                ids.append(len(self.values))
                self.values.append(value)
                self.origins.append((source, -2*height+2*level))
                self.events.append([])
            self.identities.append(ids)
        state = [list(s) for s in self.identities]
        self.move_box = []
        for t, move in enumerate(moves):
            box = state[move.from_stack].pop()
            state[move.to_stack].append(box)
            self.move_box.append(box)
            self.events[box].append(t)
        counts = []
        for box, steps in enumerate(self.events):
            ways = {self.origins[box][0]: 1}
            for t in steps:
                destination = self.moves[t].to_stack
                incoming = sum(v for s, v in ways.items() if s != destination)
                ways[destination] = ways.get(destination, 0) + incoming
            counts.append(sum(ways.values()))
        if sum(counts) > max_routes:
            raise UnsupportedDomain('COMPLETE_BYPASS_ROUTE_DOMAIN_OVERSIZE')
        self.routes, self.all_groups, self.original = [], [], []
        for box, steps in enumerate(self.events):
            paths = {self.origins[box][0]: [()]}
            for t in steps:
                destination = self.moves[t].to_stack
                incoming = [p+(t,) for s, ps in paths.items() if s != destination for p in ps]
                nxt = {s: list(ps) for s, ps in paths.items()}
                nxt.setdefault(destination, []).extend(incoming)
                paths = nxt
            pool = sorted((p for ps in paths.values() for p in ps), key=lambda p: (len(p), p))
            if len(pool) != counts[box] or len(pool) != len(set(pool)):
                raise ValueError('Complete bypass enumeration mismatch')
            start = len(self.routes)
            self.routes.extend(self._route(box, p) for p in pool)
            group = list(range(start, len(self.routes)))
            self.all_groups.append(group)
            self.original.append(next(i for i in group if self.routes[i].kept == tuple(steps)))
        self.incumbent = list(self.original)
        self.replay_full(self.original)
        self.cost_all = np.array([len(r.kept) for r in self.routes], dtype=float)
        arc_index, arcs, route_arcs = {}, [], []
        for route in self.routes:
            ids = []
            for s, a, b in route.intervals:
                key = (route.box, s, a, b)
                if key not in arc_index:
                    arc_index[key] = len(arcs)
                    arcs.append(key)
                ids.append(arc_index[key])
            route_arcs.append(ids)
        masks = [0]*len(arcs)
        for i, (box, s, a, b) in enumerate(arcs):
            for j in range(i+1, len(arcs)):
                other, t, c, d = arcs[j]
                if box == other or s != t:
                    continue
                bad = (a<c<b<d or c<a<d<b or (b==d==self.end and
                       ((a<c and self.values[box]<self.values[other]) or
                        (c<a and self.values[other]<self.values[box]))))
                if bad:
                    masks[i] |= 1 << j
                    masks[j] |= 1 << i
        self.arc_masks, self.block_masks = [], []
        for ids in route_arcs:
            present = blocked = 0
            for i in ids:
                present |= 1 << i
                blocked |= masks[i]
            self.arc_masks.append(present)
            self.block_masks.append(blocked)
        points = [(s,-1) for s in range(len(stacks))] + [(m.to_stack,2*t+1) for t,m in enumerate(moves)]
        point_stack = np.array([s for s,t in points])
        point_time = np.array([t for s,t in points])
        self.occupancy = np.zeros((len(self.routes),len(points)),dtype=np.int8)
        for i, route in enumerate(self.routes):
            for s,a,b in route.intervals:
                self.occupancy[i,(point_stack==s)&(point_time>=a)&(point_time<b)] = 1
        self.domains = [list(g) for g in self.all_groups]
        self.stats = dict(total_routes=len(self.routes),max_routes=max(counts,default=0),
                          enumeration_sec=REAL_TIME()-started,arc_count=len(arcs),
                          repeated_priorities=len(set(self.values)) != len(self.values),
                          domain='all_delayed_bypasses_at_original_container_event_times')
        self._propagate()
        self._prepare_reduced()
        self.stats['total_build_sec'] = REAL_TIME()-started

    def _route(self, box, kept):
        source, start = self.origins[box]
        intervals = []
        for t in kept:
            destination = self.moves[t].to_stack
            if source == destination:
                raise ValueError('Bypass route contains a self move')
            intervals.append((source,start,2*t))
            source,start = destination,2*t+1
        intervals.append((source,start,self.end))
        return Route(box,tuple(kept),tuple(intervals))

    def replay_full(self, chosen):
        if len(chosen)!=len(self.all_groups) or any(i not in g for i,g in zip(chosen,self.all_groups)):
            raise ValueError('Invalid full bypass route selection')
        state = [list(s) for s in self.identities]
        locations = [s for s,t in self.origins]
        retained = sorted(t for i in chosen for t in self.routes[i].kept)
        path = []
        for t in retained:
            box = self.move_box[t]
            source, destination = locations[box], self.moves[t].to_stack
            if not state[source] or state[source][-1]!=box:
                raise ValueError('Bypass source is not physically accessible')
            if source==destination or len(state[destination])>=self.height:
                raise ValueError('Bypass destination is invalid')
            path.append(Move(len(path),self.values[box],source,destination,len(state[source]),len(state[destination])+1))
            state[destination].append(state[source].pop())
            locations[box] = destination
        physical_replay(self.stacks,self.height,path,goal=True)
        return tuple(path)

    def replay(self, chosen):
        return self.replay_full(self.full_selection(chosen))
