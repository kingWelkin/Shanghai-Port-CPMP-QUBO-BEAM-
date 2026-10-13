"""Joint, capacity-feasible destinations for a fixed source pop sequence.

The quadratic quality term counts actual new priority-inversion pairs. This
module is opt-in; it does not change FRG or production defaults on import.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import time

from container_reshuffle.core import Move, future_disorder_count_stacks
from container_reshuffle.qubo.model import QuboModel
from container_reshuffle.solvers import bs_frg as frg


@dataclass(frozen=True)
class Allocation:
    stacks: tuple[tuple[int, ...], ...]
    source: int
    height: int
    boxes: tuple[int, ...]
    receivers: tuple[int, ...]
    capacity: tuple[int, ...]
    unary: tuple[tuple[int, ...], ...]
    clashes: tuple[tuple[int, int], ...]

    @classmethod
    def build(cls, stacks, source, height, max_pop=10):
        frozen = tuple(tuple(s) for s in stacks)
        receivers = tuple(d for d, s in enumerate(frozen) if d != source and len(s) < height)
        capacities = tuple(height - len(frozen[d]) for d in receivers)
        count = min(len(frozen[source]), sum(capacities), max_pop)
        boxes = tuple(reversed(frozen[source][-count:])) if count else ()
        unary = tuple(tuple(sum(v < b for v in frozen[d]) for d in receivers) for b in boxes)
        clashes = tuple((i, j) for i in range(count) for j in range(i + 1, count) if boxes[i] < boxes[j])
        return cls(frozen, source, height, boxes, receivers, capacities, unary, clashes)

    def feasible(self, assignment):
        return len(assignment) == len(self.boxes) and all(0 <= d < len(self.receivers) for d in assignment) and all(
            assignment.count(d) <= capacity for d, capacity in enumerate(self.capacity))

    def value(self, assignment, pair=True):
        if not self.feasible(assignment):
            raise ValueError('Invalid destination assignment')
        return sum(self.unary[i][d] for i, d in enumerate(assignment)) + (
            sum(assignment[i] == assignment[j] for i, j in self.clashes) if pair else 0)

    def replay(self, assignment):
        if not self.feasible(assignment):
            raise ValueError('Cannot replay infeasible assignment')
        stacks = [list(s) for s in self.stacks]
        moves = []
        for step, (box, d) in enumerate(zip(self.boxes, assignment)):
            target = self.receivers[d]
            if stacks[self.source][-1] != box or len(stacks[target]) >= self.height:
                raise ValueError('Physical assignment mismatch')
            moves.append(Move(step, box, self.source, target, len(stacks[self.source]), len(stacks[target]) + 1))
            stacks[self.source].pop()
            stacks[target].append(box)
        return tuple(tuple(s) for s in stacks), tuple(moves)

    def exact(self, pair=True):
        """Exact subset DP oracle for this small structured quadratic problem."""
        n, m = len(self.boxes), len(self.receivers)
        if not n:
            return (), dict(states=0, algorithm='subset_dp')
        size = 1 << n
        radix = m + 1
        scale = radix ** n
        powers = [radix ** (n - 1 - i) for i in range(n)]
        bits = [[i for i in range(n) if mask >> i & 1] for mask in range(size)]
        pair_cost = [sum((mask >> i & 1) and (mask >> j & 1) for i, j in self.clashes) if pair else 0
                     for mask in range(size)]
        infinity = 10 ** 100
        dp = [infinity] * size
        dp[0] = 0
        history = []
        states = 0
        for d in range(m):
            cost = [scale * (sum(self.unary[i][d] for i in positions) + pair_cost[mask])
                    + d * sum(powers[i] for i in positions) for mask, positions in enumerate(bits)]
            nxt = [infinity] * size
            chosen = [-1] * size
            for mask in range(size):
                sub = mask
                while True:
                    if sub.bit_count() <= self.capacity[d] and dp[mask ^ sub] < infinity:
                        states += 1
                        value = dp[mask ^ sub] + cost[sub]
                        if value < nxt[mask]:
                            nxt[mask], chosen[mask] = value, sub
                    if not sub:
                        break
                    sub = (sub - 1) & mask
            dp = nxt
            history.append(chosen)
        if dp[-1] == infinity:
            raise ValueError('Insufficient capacity')
        mask = size - 1
        result = [-1] * n
        for d in reversed(range(m)):
            sub = history[d][mask]
            for i in bits[sub]:
                result[i] = d
            mask ^= sub
        result = tuple(result)
        if not self.feasible(result):
            raise ValueError('DP produced an invalid assignment')
        return result, dict(states=states, algorithm='subset_dp', objective=self.value(result, pair))

    def marginal(self):
        assignment = []
        counts = [0] * len(self.receivers)
        for i, box in enumerate(self.boxes):
            d = min((d for d, cap in enumerate(self.capacity) if counts[d] < cap),
                    key=lambda d: (self.unary[i][d] + sum(self.boxes[j] < box and old == d for j, old in enumerate(assignment)), d))
            assignment.append(d)
            counts[d] += 1
        return tuple(assignment)

    def local(self, start):
        current = tuple(start)
        evaluations = 0
        while True:
            best = (self.value(current), current)
            for i in range(len(current)):
                for d in range(len(self.receivers)):
                    trial = current[:i] + (d,) + current[i + 1:]
                    if self.feasible(trial):
                        evaluations += 1
                        candidate = (self.value(trial), trial)
                        if candidate < best:
                            best = candidate
                for j in range(i + 1, len(current)):
                    trial = list(current)
                    trial[i], trial[j] = trial[j], trial[i]
                    trial = tuple(trial)
                    evaluations += 1
                    candidate = (self.value(trial), trial)
                    if candidate < best:
                        best = candidate
            if best[1] == current:
                return current, dict(evaluations=evaluations, algorithm='one_move_and_swap_descent')
            current = best[1]

    def native(self):
        """Use native dense-sequence assignment and native physical fallback."""
        stacks = [list(s) for s in self.stacks]
        plan, spare = frg._unblocking_assignment(stacks, self.source, self.height, paper_pure=True)
        result = []
        for box in self.boxes:
            index = len(stacks[self.source]) - 1
            target = plan[index] if index < len(plan) else None
            allowed = {target} if target is not None else set(spare)
            choices = frg._destination_choices(stacks, box, self.source, self.height, allowed=allowed)
            if not choices and target is not None:
                choices = frg._destination_choices(stacks, box, self.source, self.height, allowed=set(spare) or None)
            if not choices:
                choices = frg._destination_choices(stacks, box, self.source, self.height)
            if not choices:
                raise ValueError('Native assignment could not fill existing capacity')
            target = choices[0][1]
            result.append(self.receivers.index(target))
            stacks[self.source].pop()
            stacks[target].append(box)
        return tuple(result)

    def qubo(self, pair=True):
        """Exact quality plus one-hot and binary-slack capacity penalties."""
        n, m = len(self.boxes), len(self.receivers)
        upper = sum(max(row, default=0) for row in self.unary) + len(self.clashes)
        penalty = upper + 1
        model = QuboModel()
        x = [[model.add_binary_var(f'x_{i}_{d}') for d in range(m)] for i in range(n)]
        for i, row in enumerate(x):
            model.add_one_hot_penalty(row, penalty)
            for d, var in enumerate(row):
                model.add_linear(var, self.unary[i][d])
        if pair:
            for i, j in self.clashes:
                for d in range(m):
                    model.add_quadratic(x[i][d], x[j][d], 1)
        slack = []
        for d, cap in enumerate(self.capacity):
            weights = []
            remaining, weight = cap, 1
            while remaining:
                take = min(weight, remaining)
                weights.append(take)
                remaining -= take
                weight *= 2
            variables = [(x[i][d], 1) for i in range(n)]
            svars = [(model.add_binary_var(f's_{d}_{j}'), w) for j, w in enumerate(weights)]
            slack.append(svars)
            variables += svars
            model.offset += penalty * cap * cap
            for var, w in variables:
                model.add_linear(var, penalty * (w * w - 2 * cap * w))
            for (a, wa), (b, wb) in itertools.combinations(variables, 2):
                model.add_quadratic(a, b, 2 * penalty * wa * wb)
        return model, x, slack, penalty

    def encoded(self, assignment, x, slack):
        if not self.feasible(assignment):
            raise ValueError('Cannot encode infeasible assignment')
        values = {x[i][d]: 1 for i, d in enumerate(assignment)}
        for d, variables in enumerate(slack):
            remaining = self.capacity[d] - assignment.count(d)
            for var, weight in reversed(variables):
                if weight <= remaining:
                    values[var] = 1
                    remaining -= weight
            if remaining:
                raise ValueError('Slack encoding failure')
        return values


def inversion_count(stacks):
    return sum(a < b for s in stacks for i, a in enumerate(s) for b in s[i + 1:])
