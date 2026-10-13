"""Classical feasible one/two-group exchange on complete route domains.

The control minimizes retained move costs under explicit compatibility and
capacity constraints. It neither compiles nor searches a QUBO penalty energy.
"""
from __future__ import annotations

import itertools
import numpy as np

from production_selector_replay import REAL_TIME


def descend(unary, groups, conflicts, loads, caps, incumbent):
    started = REAL_TIME()
    costs = np.asarray(unary)
    groups = [np.asarray(g, dtype=int) for g in groups]
    loads = np.asarray(loads, dtype=np.int64)
    caps = np.asarray(caps, dtype=np.int64)
    chosen = list(map(int, incumbent))
    if len(chosen) != len(groups) or any(i not in g for i, g in zip(chosen, groups)):
        raise ValueError('Invalid exchange incumbent')
    if any(conflicts[a, b] for a, b in itertools.combinations(chosen, 2)) or (
        chosen and np.any(loads[chosen].sum(axis=0) > caps)):
        raise ValueError('Infeasible exchange incumbent')
    blocks = [(g,) for g in range(len(groups))] + list(itertools.combinations(range(len(groups)), 2))
    evaluations = passes = 0
    updates = []
    while True:
        passes += 1
        best = None
        for block in blocks:
            fixed = [i for g, i in enumerate(chosen) if g not in block]
            occupancy = loads[fixed].sum(axis=0) if fixed else np.zeros(len(caps), dtype=np.int64)
            domains = []
            for g in block:
                options = groups[g]
                allowed = np.all(occupancy + loads[options] <= caps, axis=1)
                if fixed:
                    allowed &= ~conflicts[np.ix_(options, fixed)].any(axis=1)
                domains.append(options[allowed])
            before = sum(float(costs[chosen[g]]) for g in block)
            for choices in itertools.product(*domains):
                evaluations += 1
                after = sum(float(costs[i]) for i in choices)
                if after >= before:
                    continue
                if len(choices) == 2 and conflicts[choices[0], choices[1]]:
                    continue
                if np.any(occupancy + loads[list(choices)].sum(axis=0) > caps):
                    continue
                candidate = (after - before, block, tuple(map(int, choices)))
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            break
        delta, block, choices = best
        changed = sum(chosen[g] != i for g, i in zip(block, choices))
        for g, i in zip(block, choices):
            chosen[g] = i
        updates.append(dict(delta=delta, block=block, choices=choices, changed_groups=changed))
    return chosen, dict(evaluations=evaluations, passes=passes, updates=updates,
                        pair_local_optimal=True, global_optimal=False,
                        runtime_sec=REAL_TIME() - started)
