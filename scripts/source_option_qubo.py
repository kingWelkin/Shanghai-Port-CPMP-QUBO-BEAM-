"""QUBO over legal per-source two-wave options, solved by block energy moves."""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import time
import numpy as np
from container_reshuffle.core import Move, future_disorder_count_stacks
from container_reshuffle.qubo.model import QuboModel
from container_reshuffle.solvers import bs_frg as frg
from two_wave_action_qubo import ActionQubo, binary_weights, Plan


@dataclass(frozen=True)
class Option:
    source: int
    actions: tuple[tuple[int, int, int], ...]
    cost: float
    loads: tuple[int, ...]
    minimum_first: tuple[int, ...]
    maximum_second: tuple[int, ...]
    requires: frozenset[int]
    clears_first: bool
    first_count: int


class SourceOptionQubo:
    def __init__(self, action_model, local_beam=12, max_options=24):
        self.action = action_model
        self.receivers = sorted(action_model.initial_receivers | action_model.new_receivers)
        self.rindex = {d: i for i, d in enumerate(self.receivers)}
        self.capacity = np.array([action_model.height - (
            len(action_model.stacks[d]) if d in action_model.initial_receivers else action_model.stable[d])
            for d in self.receivers], dtype=int)
        self.options, self.groups = [], []
        for source, indices in action_model.by_source.items():
            empty = self.option(source, (), 0)
            mandatory, alternatives = [empty], []
            first_run = action_model.first[source]
            for count in range(1, len(indices) + 1):
                for first in range(min(count, first_run) + 1):
                    if count > first_run and first != first_run:
                        continue
                    beam = [()]
                    for pos, b in enumerate(indices[:count]):
                        wave = int(pos >= first)
                        targets = sorted(d for bb, d, w in action_model.x if bb == b and w == wave)
                        expanded = {}
                        for actions in beam:
                            for d in targets:
                                choice = actions + ((b, d, wave),)
                                option = self.option(source, choice, first)
                                if option is not None:
                                    expanded[choice] = option
                        candidates = sorted(expanded.values(), key=lambda p: (p.cost, p.actions))[:local_beam]
                        beam = [p.actions for p in candidates]
                        if not beam:
                            break
                    if beam:
                        finished = [self.option(source, actions, first) for actions in beam]
                        finished.sort(key=lambda p: (p.cost, p.actions))
                        mandatory.append(finished[0])
                        alternatives.extend(finished[1:])
            unique = {p.actions: p for p in mandatory}
            for p in sorted(alternatives, key=lambda p: (-len(p.actions), p.cost, p.first_count, p.actions)):
                if len(unique) >= max_options:
                    break
                unique.setdefault(p.actions, p)
            chosen = [empty] + sorted((p for a, p in unique.items() if a), key=lambda p: (-len(p.actions), p.cost, p.actions))
            start = len(self.options)
            self.options.extend(chosen)
            self.groups.append(np.arange(start, len(self.options), dtype=int))
        self.unary = np.array([p.cost for p in self.options])
        self.loads = np.array([p.loads for p in self.options], dtype=int).reshape(len(self.options), len(self.receivers))
        self.penalty = 2. + sum(len(p.actions) for p in self.options)
        n = len(self.options)
        self.pair = np.zeros((n, n))
        for i, p in enumerate(self.options):
            for j in range(i + 1, n):
                q = self.options[j]
                if p.source == q.source:
                    continue
                violations = int(q.source in p.requires and not q.clears_first) + int(
                    p.source in q.requires and not p.clears_first)
                for r, d in enumerate(self.receivers):
                    if d in action_model.initial_receivers:
                        violations += p.minimum_first[r] < q.maximum_second[r]
                        violations += q.minimum_first[r] < p.maximum_second[r]
                self.pair[i, j] = self.pair[j, i] = self.penalty * violations
        self.qubo = QuboModel()
        self.variables = [self.qubo.add_binary_var(f'option_{i}') for i in range(n)]
        for i, c in enumerate(self.unary):
            self.qubo.add_linear(i, c)
        for group in self.groups:
            self.qubo.add_one_hot_penalty(list(map(int, group)), self.penalty)
        for i in range(n):
            for j in range(i + 1, n):
                if self.pair[i, j]:
                    self.qubo.add_quadratic(i, j, self.pair[i, j])
        self.slack = []
        for r, cap in enumerate(self.capacity):
            terms = [(i, int(load)) for i, load in enumerate(self.loads[:, r]) if load]
            slack = [(self.qubo.add_binary_var(f'capacity_{r}_{j}'), weight) for j, weight in enumerate(binary_weights(int(cap)))]
            self.slack.append(slack)
            terms += slack
            self.qubo.offset += self.penalty * cap * cap
            for i, value in terms:
                self.qubo.add_linear(i, self.penalty * (value*value - 2*cap*value))
            for (i, a), (j, b) in itertools.combinations(terms, 2):
                self.qubo.add_quadratic(i, j, 2*self.penalty*a*b)

    def option(self, source, actions, first):
        loads = [0] * len(self.receivers)
        minimum = [10**9] * len(self.receivers)
        maximum = [-1] * len(self.receivers)
        requires = set()
        cost = -float(len(actions))
        for b, d, wave in actions:
            r = self.rindex[d]
            loads[r] += 1
            if loads[r] > self.capacity[r]:
                return None
            value = self.action.boxes[b].value
            if wave == 0:
                minimum[r] = min(minimum[r], value)
            else:
                maximum[r] = max(maximum[r], value)
                if d in self.action.new_receivers:
                    requires.add(d)
            if d in self.action.initial_receivers and minimum[r] < maximum[r]:
                return None
            cost += self.action.cost[self.action.x[b, d, wave]]
        bad = len(self.action.stacks[source]) - self.action.stable[source]
        return Option(source, tuple(actions), cost, tuple(loads), tuple(minimum), tuple(maximum),
                      frozenset(requires), first == bad, first)

    def energy(self, chosen):
        load = self.loads[list(chosen)].sum(axis=0) if chosen else np.zeros(len(self.receivers), dtype=int)
        return float(sum(self.unary[i] for i in chosen) +
            sum(self.pair[i, j] for i, j in itertools.combinations(chosen, 2)) +
            self.penalty * np.square(np.maximum(0, load - self.capacity)).sum())

    def bits(self, chosen):
        bits = {int(i): 1 for i in chosen}
        load = self.loads[list(chosen)].sum(axis=0) if chosen else np.zeros(len(self.receivers), dtype=int)
        for r, variables in enumerate(self.slack):
            remaining = max(0, int(self.capacity[r] - load[r]))
            for index, weight in reversed(variables):
                if weight <= remaining:
                    bits[index] = 1
                    remaining -= weight
            if remaining:
                raise ValueError('Capacity slack representation failed')
        return bits

    def neighborhood(self, chosen, block):
        fixed = [v for i, v in enumerate(chosen) if i not in block]
        constant = sum(self.unary[i] for i in fixed) + sum(self.pair[i, j] for i, j in itertools.combinations(fixed, 2))
        base_load = self.loads[fixed].sum(axis=0) if fixed else np.zeros(len(self.receivers), dtype=int)
        left = self.groups[block[0]]
        left_cost = self.unary[left] + (self.pair[np.ix_(left, fixed)].sum(axis=1) if fixed else 0)
        if len(block) == 1:
            load = base_load + self.loads[left] - self.capacity
            scores = constant + left_cost + self.penalty * np.square(np.maximum(0, load)).sum(axis=1)
            index = int(np.argmin(scores))
            return float(scores[index]), (int(left[index]),), len(scores)
        right = self.groups[block[1]]
        right_cost = self.unary[right] + (self.pair[np.ix_(right, fixed)].sum(axis=1) if fixed else 0)
        load = base_load + self.loads[left, None, :] + self.loads[None, right, :] - self.capacity
        scores = (constant + left_cost[:, None] + right_cost[None, :] + self.pair[np.ix_(left, right)] +
                  self.penalty * np.square(np.maximum(0, load)).sum(axis=2))
        flat = int(np.argmin(scores))
        i, j = divmod(flat, len(right))
        return float(scores[i, j]), (int(left[i]), int(right[j])), int(scores.size)

    def solve(self, blocks=2):
        started = time.perf_counter()
        empty = [int(group[0]) for group in self.groups]
        evaluations, candidates = 0, []
        for order in (range(len(self.groups)), reversed(range(len(self.groups)))):
            order = list(order)
            current = list(empty)
            value = self.energy(current)
            for _ in range(5):
                changed = False
                for source in order:
                    trial, choices, count = self.neighborhood(current, (source,))
                    evaluations += count
                    if trial < value - 1e-9:
                        current[source], value = choices[0], trial
                        changed = True
                if not changed:
                    break
            candidates.append((value, tuple(current)))
        value, chosen_tuple = min(candidates)
        chosen = list(chosen_tuple)
        exchanges = 0
        if blocks >= 2:
            for _ in range(12):
                best = None
                for block in itertools.combinations(range(len(self.groups)), 2):
                    trial, choices, count = self.neighborhood(chosen, block)
                    evaluations += count
                    if trial < value - 1e-9 and (best is None or (trial, block, choices) < best):
                        best = (trial, block, choices)
                if best is None:
                    break
                value, block, choices = best
                for source, choice in zip(block, choices):
                    chosen[source] = choice
                exchanges += 1
        bits = self.bits(chosen)
        full_energy = self.qubo.evaluate(bits)
        if abs(full_energy - self.energy(chosen)) > 1e-5 or value > 1e-6:
            raise ValueError('Reduced-factor QUBO energy mismatch or lost feasible incumbent')
        actions = [a for i in chosen for a in self.options[i].actions]
        actions.sort(key=lambda a: (a[2], -self.action.boxes[a[0]].value,
                                    self.action.boxes[a[0]].source, self.action.boxes[a[0]].position))
        stacks, moves = [list(s) for s in self.action.stacks], []
        for step, (b, d, wave) in enumerate(actions):
            box = self.action.boxes[b]
            source = box.source
            if not stacks[source] or stacks[source][-1] != box.value or frg._is_ordered_stack(stacks[source]):
                raise ValueError('Source-option pop order is invalid')
            if not frg._can_accept_well_placed(stacks[d], box.value, self.action.height, self.action.max_group):
                raise ValueError('Source-option receiver is invalid')
            moves.append(Move(step, box.value, source, d, len(stacks[source]), len(stacks[d]) + 1))
            stacks[d].append(stacks[source].pop())
        if future_disorder_count_stacks(stacks) != future_disorder_count_stacks(self.action.stacks) - len(moves):
            raise ValueError('Source-option actual N count differs')
        return dict(moves=tuple(moves), stacks=tuple(tuple(s) for s in stacks), energy=full_energy,
            chosen=chosen, bits=bits, evaluations=evaluations, exchanges=exchanges,
            runtime_sec=time.perf_counter() - started, variables=len(self.qubo.var_index),
            groups=len(self.groups), options=len(self.options), certified=False)
