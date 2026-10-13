"""A physically serializable two-wave bad-to-good relocation QUBO.

The exact classical oracle solves the equivalent feasibility model. This
separates model value from annealer performance and supplies optimal references.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import itertools
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from container_reshuffle.core import Move, future_disorder_count_stacks
from container_reshuffle.solvers import bs_frg as frg
from container_reshuffle.qubo.model import QuboModel


@dataclass(frozen=True)
class Box:
    source: int
    position: int
    value: int
    segment: int


@dataclass
class Plan:
    moves: tuple[Move, ...]
    stacks: tuple[tuple[int, ...], ...]
    selected: list[dict]
    objective: float
    energy: float
    optimal: bool
    status: int
    solve_sec: float
    gap: float | None


def prefix_runs(stack, maximum=None):
    stable = frg._stable_prefix_len(stack)
    pop = list(reversed(stack[stable:]))
    if maximum is not None:
        pop = pop[:maximum]
    first = next((i for i in range(1, len(pop)) if pop[i - 1] < pop[i]), len(pop))
    end = next((i for i in range(first + 1, len(pop)) if pop[i - 1] < pop[i]), len(pop))
    return stable, pop[:end], first


def binary_weights(capacity):
    weights, power = [], 1
    while capacity:
        value = min(power, capacity)
        weights.append(value)
        capacity -= value
        power *= 2
    return weights


class ActionQubo:
    def __init__(self, stacks, height, waves=2, allowed=None):
        self.stacks = tuple(tuple(s) for s in stacks)
        self.height, self.waves = height, waves
        self.boxes, self.by_source, self.first, self.stable = [], {}, {}, {}
        for source, stack in enumerate(stacks):
            stable, pop, first = prefix_runs(stack, None if allowed is None else allowed.get(source, 0))
            self.stable[source] = stable
            self.first[source] = first
            if waves == 1:
                pop = pop[:first]
            indices = []
            for position, value in enumerate(pop):
                indices.append(len(self.boxes))
                self.boxes.append(Box(source, position, int(value), int(position >= first)))
            if indices:
                self.by_source[source] = indices
        self.qubo = QuboModel()
        self.cost, self.rows, self.lower, self.upper = [], [], [], []
        self.penalty = len(self.boxes) + 2
        self.x, self.y, self.u = {}, {}, {}
        self.cross_conflicts = self.activation_links = 0
        self.max_group = max((max(s, default=0) for s in stacks), default=1) or 1
        self.initial_receivers = {j for j, s in enumerate(stacks)
                                  if self.stable[j] == len(s) and len(s) < height}
        self.new_receivers = {j for j, s in enumerate(stacks)
                              if 0 < len(s) - self.stable[j] == self.first[j] and j in self.by_source}
        for b, box in enumerate(self.boxes):
            self.u[b] = self.var(f'u_{b}', -1.)
            for wave in range(waves):
                self.y[b, wave] = self.var(f'y_{b}_{wave}')
        for b, box in enumerate(self.boxes):
            for wave in range(waves):
                if wave == 0 and box.segment:
                    continue
                receivers = self.initial_receivers | (self.new_receivers if wave else set())
                for d in sorted(receivers):
                    if d == box.source:
                        continue
                    threshold = self.receiver_top(d)
                    if box.value > threshold:
                        continue
                    tie = (threshold - box.value) / (self.max_group * (len(self.boxes) + 1))
                    self.x[b, d, wave] = self.var(f'x_{b}_{d}_{wave}', tie)
        for b, box in enumerate(self.boxes):
            self.equality({self.u[b]: 1, **{self.y[b, w]: -1 for w in range(waves)}}, 0)
            for wave in range(waves):
                row = {var: 1 for (bb, d, w), var in self.x.items() if bb == b and w == wave}
                row[self.y[b, wave]] = -1
                self.equality(row, 0)
        for source, indices in self.by_source.items():
            for prev, curr in zip(indices, indices[1:]):
                self.implies(self.u[curr], self.u[prev])
                self.implies(self.y[curr, 0], self.y[prev, 0])
            if waves > 1 and self.first[source] < len(indices):
                anchor = self.y[indices[self.first[source] - 1], 0]
                for b in indices[self.first[source]:]:
                    self.implies(self.y[b, 1], anchor)
        if waves > 1:
            for (b, d, wave), var in self.x.items():
                if wave == 1 and d in self.new_receivers:
                    last_bad = len(self.stacks[d]) - self.stable[d] - 1
                    anchor = self.y[self.by_source[d][last_bad], 0]
                    self.implies(var, anchor)
                    self.activation_links += 1
            for d in self.initial_receivers:
                early = [(b, var) for (b, dd, w), var in self.x.items() if dd == d and w == 0]
                late = [(b, var) for (b, dd, w), var in self.x.items() if dd == d and w == 1]
                for (b, v), (c, z) in itertools.product(early, late):
                    if self.boxes[b].value < self.boxes[c].value:
                        self.mutex(v, z)
                        self.cross_conflicts += 1
        for d in sorted(self.initial_receivers | (self.new_receivers if waves > 1 else set())):
            capacity = height - (len(self.stacks[d]) if d in self.initial_receivers else self.stable[d])
            row = {var: 1 for (b, dd, wave), var in self.x.items() if dd == d}
            for i, weight in enumerate(binary_weights(capacity)):
                row[self.var(f'slack_{d}_{i}')] = weight
            self.equality(row, capacity)

    def receiver_top(self, destination):
        s = self.stacks[destination]
        length = len(s) if destination in self.initial_receivers else self.stable[destination]
        return s[length - 1] if length else self.max_group

    def var(self, name, cost=0.):
        index = self.qubo.add_binary_var(name)
        if index != len(self.cost):
            raise ValueError('Repeated variable')
        self.cost.append(cost)
        self.qubo.add_linear(index, cost)
        return index

    def equality(self, coefficients, target):
        self.rows.append(dict(coefficients))
        self.lower.append(target)
        self.upper.append(target)
        p = self.penalty
        self.qubo.offset += p * target * target
        for v, a in coefficients.items():
            self.qubo.add_linear(v, p * (a*a - 2*target*a))
        for (v, a), (w, b) in itertools.combinations(coefficients.items(), 2):
            self.qubo.add_quadratic(v, w, 2*p*a*b)

    def implies(self, consequent, prerequisite):
        self.rows.append({consequent: 1, prerequisite: -1})
        self.lower.append(-np.inf)
        self.upper.append(0)
        self.qubo.add_linear(consequent, self.penalty)
        self.qubo.add_quadratic(consequent, prerequisite, -self.penalty)

    def mutex(self, left, right):
        self.rows.append({left: 1, right: 1})
        self.lower.append(-np.inf)
        self.upper.append(1)
        self.qubo.add_quadratic(left, right, self.penalty)

    def matrix(self):
        rr, cc, vv = [], [], []
        for i, row in enumerate(self.rows):
            for j, value in row.items():
                rr.append(i)
                cc.append(j)
                vv.append(value)
        return coo_matrix((vv, (rr, cc)), shape=(len(self.rows), len(self.cost))).tocsc()

    def solve(self, time_limit=2.):
        started = time.perf_counter()
        if not self.cost:
            return Plan((), self.stacks, [], 0., 0., True, 0, time.perf_counter() - started, 0.)
        a = self.matrix()
        result = milp(np.asarray(self.cost), integrality=np.ones(len(self.cost)),
            bounds=Bounds(np.zeros(len(self.cost)), np.ones(len(self.cost))),
            constraints=LinearConstraint(a, self.lower, self.upper),
            options=dict(time_limit=time_limit, mip_rel_gap=0., presolve=True))
        if result.x is None:
            return Plan((), self.stacks, [], 0., float('nan'), False, int(result.status),
                        time.perf_counter() - started, None)
        values = np.rint(result.x)
        lhs = a @ values
        if np.max(abs(values - result.x)) > 1e-5 or any(
            v < lo - 1e-6 or v > hi + 1e-6 for v, lo, hi in zip(lhs, self.lower, self.upper)):
            raise ValueError('Classical oracle returned an infeasible plan')
        selected = []
        for (b, d, w), v in self.x.items():
            if values[v]:
                selected.append(dict(box_index=b, source=self.boxes[b].source, position=self.boxes[b].position,
                                     value=self.boxes[b].value, destination=d, wave=w))
        selected.sort(key=lambda a: (a['wave'], -a['value'], a['source'], a['position']))
        stacks, moves = [list(s) for s in self.stacks], []
        for step, action in enumerate(selected):
            source, destination, box = action['source'], action['destination'], action['value']
            if not stacks[source] or stacks[source][-1] != box or len(stacks[destination]) >= self.height:
                raise ValueError('Selected actions are not physically serializable')
            if frg._is_ordered_stack(stacks[source]) or not frg._can_accept_well_placed(
                    stacks[destination], box, self.height, self.max_group):
                raise ValueError('Selected action is not bad-to-good')
            moves.append(Move(step, box, source, destination, len(stacks[source]), len(stacks[destination]) + 1))
            stacks[destination].append(stacks[source].pop())
        if future_disorder_count_stacks(stacks) != future_disorder_count_stacks(self.stacks) - len(moves):
            raise ValueError('Real N reduction differs from model count')
        objective = float(np.dot(self.cost, values))
        energy = self.qubo.evaluate({i: int(v) for i, v in enumerate(values)})
        if abs(energy - objective) > 1e-6:
            raise ValueError('Compiled QUBO energy differs from feasible objective')
        return Plan(tuple(moves), tuple(tuple(s) for s in stacks), selected, objective, energy,
                    result.status == 0, int(result.status), time.perf_counter() - started,
                    float(result.mip_gap) if getattr(result, 'mip_gap', None) is not None else None)

    def summary(self):
        return dict(boxes=len(self.boxes), variables=len(self.cost), actions=len(self.x),
                    cross_wave_conflicts=self.cross_conflicts, activation_links=self.activation_links,
                    receivers=len(self.initial_receivers), new_receivers=len(self.new_receivers),
                    sources=len(self.by_source), penalty=self.penalty)


def sequential_exact(stacks, height, time_limit=2., allowed=None):
    joint_domain = ActionQubo(stacks, height, 2, allowed)
    first_model = ActionQubo(stacks, height, 1, allowed)
    first = first_model.solve(time_limit / 2)
    moved = {s: sum(a['source'] == s for a in first.selected) for s in joint_domain.by_source}
    allowed = {s: len(boxes) - moved[s] for s, boxes in joint_domain.by_source.items()}
    second_model = ActionQubo(first.stacks, height, 1, allowed)
    second = second_model.solve(time_limit / 2)
    return first, second


def greedy_two_wave(stacks, height, allowed=None):
    domain = ActionQubo(stacks, height, 2, allowed)
    current = [list(s) for s in stacks]
    counts = {s: 0 for s in domain.by_source}
    all_moves = []
    for wave in (0, 1):
        before_wave = dict(counts)
        receivers = {j for j, s in enumerate(current) if frg._is_ordered_stack(s) and len(s) < height}
        while True:
            options = []
            for source, indices in domain.by_source.items():
                count = counts[source]
                if count >= len(indices):
                    continue
                b = domain.boxes[indices[count]]
                if wave == 0 and b.segment:
                    continue
                if wave == 1 and b.segment and before_wave[source] < domain.first[source]:
                    continue
                for d in receivers:
                    if d != source and frg._can_accept_well_placed(current[d], b.value, height, domain.max_group):
                        gap = frg._stack_group(current[d], domain.max_group) - b.value
                        options.append((gap, source, d, b))
            if not options:
                break
            _, source, d, b = min(options, key=lambda row: row[:3])
            all_moves.append(Move(len(all_moves), b.value, source, d, len(current[source]), len(current[d]) + 1))
            current[d].append(current[source].pop())
            counts[source] += 1
    return tuple(all_moves), tuple(tuple(s) for s in current)


def bounded_sources(stacks, height, max_sources=6, max_boxes=18, per_source=4):
    receivers = [s for s in stacks if frg._is_ordered_stack(s) and len(s) < height]
    group = max((max(s, default=0) for s in stacks), default=1) or 1
    candidates = []
    for source, stack in enumerate(stacks):
        stable, pop, first = prefix_runs(stack)
        if not pop:
            continue
        direct = any(frg._can_accept_well_placed(list(s), pop[0], height, group) for s in receivers)
        candidates.append(((first >= len(pop), not direct, len(stack) - stable, source), source, len(pop)))
    allowed = {}
    for _, source, count in sorted(candidates)[:max_sources]:
        use = min(count, per_source, max_boxes - sum(allowed.values()))
        if use:
            allowed[source] = use
    return allowed
