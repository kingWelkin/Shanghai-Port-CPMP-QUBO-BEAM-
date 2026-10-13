"""Exact physical compatibility of per-container incumbent subsequences.

Each option follows one identified container through a subsequence of its
original moves. Residence intervals on one stack must be noncrossing, final
residents must be ordered, and stack occupancy must respect capacity. The
objective is the number of retained elementary moves,not a search proxy.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from container_reshuffle.core import Move
from container_reshuffle.qubo.model import QuboModel
from container_reshuffle.solvers import bs_frg as frg
from production_selector_replay import REAL_TIME
from research_receiver_allocation import physical_replay
from two_wave_action_qubo import binary_weights


@dataclass(frozen=True)
class Route:
    box: int
    kept: tuple[int, ...]
    intervals: tuple[tuple[int, int, int], ...]


class SubsequenceModel:
    def __init__(self, stacks, height, moves, max_options=12):
        self.stacks = tuple(tuple(s) for s in stacks)
        self.height, self.moves = height, tuple(moves)
        physical_replay(stacks, height, self.moves, goal=True)
        self.end = 2*len(moves)+2
        self.identities, self.origins, self.values, self.events = [], [], [], []
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
            self.events[box].append(t)
            self.move_box.append(box)
        self.routes, self.groups, self.original = [], [], []
        for box, events in enumerate(self.events):
            pool = self.enumerate_routes(box, max_options)
            start = len(self.routes)
            self.routes.extend(pool)
            group = tuple(range(start, len(self.routes)))
            self.groups.append(group)
            self.original.append(next(i for i in group if self.routes[i].kept == tuple(events)))
        self.cost = np.array([len(r.kept) for r in self.routes], dtype=float)
        self.conflicts = []
        self.pair = np.zeros((len(self.routes), len(self.routes)), dtype=bool)
        for ga, gb in itertools.combinations(self.groups, 2):
            for i in ga:
                for j in gb:
                    if self.conflict(self.routes[i], self.routes[j]):
                        self.conflicts.append((i, j))
                        self.pair[i, j] = self.pair[j, i] = True
        # Remove box-constant occupancy and duplicate capacity rows.
        capacity = set()
        for source in range(len(stacks)):
            for t in [-1]+[2*j+1 for j, m in enumerate(moves) if m.to_stack == source]:
                varying, fixed, possible = [], 0, 0
                for group in self.groups:
                    occupied = [i for i in group if any(s == source and a <= t < b for s, a, b in self.routes[i].intervals)]
                    if len(occupied) == len(group):
                        fixed += 1
                    elif occupied:
                        varying.extend(occupied)
                        possible += 1
                cap = height-fixed
                if cap < possible:
                    if cap < 0:
                        raise ValueError('Original trajectory has impossible fixed occupancy')
                    capacity.add((tuple(varying), cap))
        self.capacity = sorted(capacity)
        self.loads = np.zeros((len(self.routes), len(self.capacity)), dtype=np.int16)
        for r, (vs, cap) in enumerate(self.capacity):
            self.loads[list(vs), r] = 1
        self.caps = np.array([cap for vs, cap in self.capacity], dtype=np.int16)
        self.penalty = len(moves)+1
        self.qubo = None
        if not self.feasible(self.original):
            raise ValueError('Physical incumbent excluded by compatibility model')

    def route(self, box, kept):
        source, start = self.origins[box]
        intervals = []
        for t in kept:
            move = self.moves[t]
            if move.from_stack != source:
                raise ValueError('Subsequence changes a container location discontinuously')
            intervals.append((source, start, 2*t))
            source, start = move.to_stack, 2*t+1
        intervals.append((source, start, self.end))
        return Route(box, tuple(kept), tuple(intervals))

    def enumerate_routes(self, box, maximum):
        source, _ = self.origins[box]
        paths = {source:[()]}
        for t in self.events[box]:
            move = self.moves[t]
            nxt = {s:list(rows) for s, rows in paths.items()}
            for path in paths.get(move.from_stack, []):
                nxt.setdefault(move.to_stack, []).append(path+(t,))
            paths = {s:sorted(set(rows), key=lambda p:(len(p), p))[:maximum] for s, rows in nxt.items()}
        original = tuple(self.events[box])
        mandatory = {original}
        for path in sorted((rows[0] for rows in paths.values()), key=lambda p:(len(p), p)):
            if len(mandatory) >= maximum:
                break
            mandatory.add(path)
        all_paths = sorted(set(p for rows in paths.values() for p in rows), key=lambda p:(len(p), p))
        for path in all_paths:
            if len(mandatory) >= maximum:
                break
            mandatory.add(path)
        return [self.route(box, p) for p in sorted(mandatory, key=lambda p:(len(p), p))]

    def conflict(self, left, right):
        for s, a, b in left.intervals:
            for t, c, d in right.intervals:
                if s != t:
                    continue
                if a < c < b < d or c < a < d < b:
                    return True
                if b == d == self.end:
                    if a < c and self.values[left.box] < self.values[right.box]:
                        return True
                    if c < a and self.values[right.box] < self.values[left.box]:
                        return True
        return False

    def feasible(self, chosen):
        if len(chosen) != len(self.groups) or any(i not in g for i, g in zip(chosen, self.groups)):
            return False
        if any(self.pair[i, j] for i, j in itertools.combinations(chosen, 2)):
            return False
        return bool(np.all(self.loads[list(chosen)].sum(axis=0) <= self.caps))

    def energy(self, chosen):
        violations = sum(self.pair[i, j] for i, j in itertools.combinations(chosen, 2))
        overflow = np.maximum(0, self.loads[list(chosen)].sum(axis=0)-self.caps)
        return float(self.cost[list(chosen)].sum()+self.penalty*(violations+np.square(overflow).sum()))

    def replay(self, chosen):
        state = [list(s) for s in self.identities]
        kept = sorted(t for i in chosen for t in self.routes[i].kept)
        result = []
        for t in kept:
            move = self.moves[t]
            box = self.move_box[t]
            if not state[move.from_stack] or state[move.from_stack][-1] != box:
                raise ValueError('Retained move is not at the physical stack top')
            if len(state[move.to_stack]) >= self.height:
                raise ValueError('Retained move exceeds physical capacity')
            result.append(Move(len(result), self.values[box], move.from_stack, move.to_stack,
                               len(state[move.from_stack]), len(state[move.to_stack])+1))
            state[move.to_stack].append(state[move.from_stack].pop())
        final = [[self.values[b] for b in s] for s in state]
        if not all(frg._is_ordered_stack(s) for s in final):
            raise ValueError('Retained trajectory does not reach an ordered layout')
        physical_replay(self.stacks, self.height, result, goal=True)
        return tuple(result)

    def compile_qubo(self):
        q = QuboModel()
        for i, c in enumerate(self.cost):
            q.add_linear(q.add_binary_var(f'route_{i}'), c)
        for group in self.groups:
            q.add_one_hot_penalty(list(group), self.penalty)
        for i, j in self.conflicts:
            q.add_quadratic(i, j, self.penalty)
        self.slack = []
        for r, (vs, cap) in enumerate(self.capacity):
            terms = {i:1 for i in vs}
            slack = [(q.add_binary_var(f'capacity_{r}_{j}'), w) for j, w in enumerate(binary_weights(cap))]
            self.slack.append(slack)
            terms.update(dict(slack))
            q.offset += self.penalty*cap*cap
            for i, c in terms.items():
                q.add_linear(i, self.penalty*(c*c-2*cap*c))
            for (i, a), (j, b) in itertools.combinations(terms.items(), 2):
                q.add_quadratic(i, j, 2*self.penalty*a*b)
        self.qubo = q
        return q

    def bits(self, chosen):
        if self.qubo is None:
            self.compile_qubo()
        values = {i:1 for i in chosen}
        loads = self.loads[list(chosen)].sum(axis=0)
        for r, slack in enumerate(self.slack):
            remaining = max(0, int(self.caps[r]-loads[r]))
            for var, weight in reversed(slack):
                if weight <= remaining:
                    values[var] = 1
                    remaining -= weight
            if remaining:
                raise ValueError('Cannot encode occupancy slack')
        return values

    def exact(self, seconds=1.):
        began = REAL_TIME()
        rows, lo, hi = [], [], []
        for group in self.groups:
            rows.append({i:1 for i in group}); lo.append(1); hi.append(1)
        for i, j in self.conflicts:
            rows.append({i:1, j:1}); lo.append(-np.inf); hi.append(1)
        for vs, cap in self.capacity:
            rows.append({i:1 for i in vs}); lo.append(-np.inf); hi.append(cap)
        rr, cc, vv = [], [], []
        for r, row in enumerate(rows):
            for i, v in row.items():
                rr.append(r); cc.append(i); vv.append(v)
        a = coo_matrix((vv, (rr, cc)), shape=(len(rows), len(self.routes))).tocsc()
        result = milp(self.cost, integrality=np.ones(len(self.routes)),
            bounds=Bounds(0., 1.), constraints=LinearConstraint(a, lo, hi),
            options=dict(time_limit=seconds, mip_rel_gap=0., presolve=True))
        chosen = list(self.original)
        if result.x is not None:
            trial = [next((i for i in group if result.x[i] > .5), -1) for group in self.groups]
            if not self.feasible(trial):
                raise ValueError('MILP returned an invalid subsequence selection')
            if self.cost[trial].sum() < self.cost[chosen].sum():
                chosen = trial
        return chosen, dict(optimal=result.status == 0, status=int(result.status),
                            runtime_sec=REAL_TIME()-began)

    def descend(self, blocks=1):
        began = REAL_TIME()
        chosen = list(self.original)
        value = self.energy(chosen)
        active = [g for g, group in enumerate(self.groups) if len(group) > 1]
        evaluations = 0
        while True:
            best = None
            for size in range(1, blocks+1):
                for block in itertools.combinations(active, size):
                    fixed = [i for g, i in enumerate(chosen) if g not in block]
                    fixed_cost = self.cost[fixed].sum()
                    fixed_load = self.loads[fixed].sum(axis=0)
                    left = np.array(self.groups[block[0]], dtype=int)
                    left_cost = self.cost[left]+self.penalty*self.pair[np.ix_(left, fixed)].sum(axis=1)
                    if size == 1:
                        load = fixed_load+self.loads[left]-self.caps
                        scores = fixed_cost+left_cost+self.penalty*np.square(np.maximum(0, load)).sum(axis=1)
                        index = int(np.argmin(scores))
                        alternatives = (int(left[index]),)
                        candidate = float(scores[index])
                    else:
                        right = np.array(self.groups[block[1]], dtype=int)
                        right_cost = self.cost[right]+self.penalty*self.pair[np.ix_(right, fixed)].sum(axis=1)
                        load = fixed_load+self.loads[left, None, :]+self.loads[None, right, :]-self.caps
                        scores = (fixed_cost+left_cost[:, None]+right_cost[None, :]+
                            self.penalty*self.pair[np.ix_(left, right)]+
                            self.penalty*np.square(np.maximum(0, load)).sum(axis=2))
                        index = int(np.argmin(scores))
                        li, ri = divmod(index, len(right))
                        alternatives = (int(left[li]), int(right[ri]))
                        candidate = float(scores[li, ri])
                    evaluations += int(scores.size)
                    if candidate < value-1e-9 and (best is None or (candidate, block, alternatives) < best):
                        best = candidate, block, alternatives
            if best is None:
                break
            value, block, alternatives = best
            for g, i in zip(block, alternatives):
                chosen[g] = i
            if not self.feasible(chosen):
                raise ValueError('Penalty descent lost physical feasibility')
        return chosen, dict(evaluations=evaluations, runtime_sec=REAL_TIME()-began, optimal=False)

    def classical_descent(self):
        """Independent greedy baseline uses physical replay,not QUBO constraints."""
        began = REAL_TIME()
        chosen = list(self.original)
        value = int(self.cost[chosen].sum())
        evaluations = 0
        while True:
            best = None
            for g, group in enumerate(self.groups):
                for option in group:
                    candidate = int(value-self.cost[chosen[g]]+self.cost[option])
                    if candidate >= value:
                        continue
                    trial = list(chosen)
                    trial[g] = option
                    evaluations += 1
                    try:
                        self.replay(trial)
                    except ValueError:
                        continue
                    if best is None or (candidate, g, option) < best:
                        best = candidate, g, option
            if best is None:
                break
            value, g, option = best
            chosen[g] = option
        return chosen, dict(evaluations=evaluations, runtime_sec=REAL_TIME()-began, optimal=False)
