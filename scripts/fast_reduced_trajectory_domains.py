"""Exact reduced factors with early size gates and bitset conflict construction."""
from __future__ import annotations

import numpy as np

from complete_trajectory_domains import UnsupportedDomain
from factorized_trajectory_qubo import FactorizedConditionedDomains
from production_selector_replay import REAL_TIME
from two_wave_action_qubo import binary_weights


class FastReducedConditionedDomains(FactorizedConditionedDomains):
    """Keep the complete route domain and enforce its declared QUBO size limit.

    Propagation is unchanged. After propagation the number of route bits alone
    is a lower bound on total variables, permitting an exact early rejection.
    Capacity rows and their unchanged slack weights give the final size before
    allocating or evaluating the quadratic route-conflict matrix.
    """

    def __init__(self, stacks, height, moves, released, *, trace=None,
                 max_routes=20000, max_variables=4096):
        self.variable_limit = max_variables
        super().__init__(stacks, height, moves, released, trace=trace, max_routes=max_routes)

    def _prepare_reduced(self):
        began = REAL_TIME()
        self.active_boxes = [b for b, domain in enumerate(self.domains) if len(domain) > 1]
        self.fixed = {b: domain[0] for b, domain in enumerate(self.domains) if len(domain) == 1}
        self.constant = float(sum(self.cost_all[i] for i in self.fixed.values()))
        self.route_ids, self.groups, self.current = [], [], []
        for box in self.active_boxes:
            start = len(self.route_ids)
            self.route_ids.extend(self.domains[box])
            self.groups.append(np.arange(start, len(self.route_ids), dtype=int))
            self.current.append(start + self.domains[box].index(self.incumbent[box]))
        count = len(self.route_ids)
        if count > self.variable_limit:
            raise UnsupportedDomain('REDUCED_QUBO_OVERSIZE')
        self.unary = self.cost_all[self.route_ids]

        # Advanced row indexing creates a copy. Removing fully loaded groups
        # from this copy does not mutate the original propagation occupancy.
        active = self.occupancy[np.asarray(self.route_ids, dtype=int), :].copy()
        fixed = self.occupancy[np.asarray(list(self.fixed.values()), dtype=int), :].sum(axis=0, dtype=np.int16)
        varying = np.zeros(self.occupancy.shape[1], dtype=np.int16)
        for group in self.groups:
            block = active[int(group[0]):int(group[-1]) + 1]
            loaded = block.sum(axis=0, dtype=np.int32)
            full = loaded == len(group)
            fixed += full
            varying += (loaded > 0) & ~full
            block[:, full] = 0
        caps = self.height - fixed
        if np.any(caps < 0):
            raise ValueError('Fixed occupancy exceeds capacity')
        rows = {(tuple(map(int, np.flatnonzero(active[:, point]))), int(caps[point]))
                for point in np.flatnonzero(caps < varying)}
        self.capacity = sorted(rows)
        total_variables = count + sum(len(binary_weights(cap)) for _, cap in self.capacity)
        if total_variables > self.variable_limit:
            raise UnsupportedDomain('REDUCED_QUBO_OVERSIZE')

        # Invert route/arc incidence. An incompatible arc's route bitset gives
        # every route blocked by that arc; unions produce an entire matrix row.
        routes_at_arc = {}
        used_arcs = 0
        group_masks = []
        for group in self.groups:
            group_masks.append(((1 << len(group)) - 1) << int(group[0]))
        for index, route in enumerate(self.route_ids):
            arcs = self.arc_masks[route]
            used_arcs |= arcs
            while arcs:
                bit = arcs & -arcs
                routes_at_arc[bit] = routes_at_arc.get(bit, 0) | (1 << index)
                arcs -= bit
        self.pair = np.zeros((count, count), dtype=bool)
        bytes_per_row = (count + 7) // 8
        cached = {}
        for group, own in zip(self.groups, group_masks):
            for index in group:
                route = self.route_ids[int(index)]
                blocked = self.block_masks[route] & used_arcs
                conflicts = cached.get(blocked)
                if conflicts is None:
                    conflicts = 0
                    remaining = blocked
                    while remaining:
                        bit = remaining & -remaining
                        conflicts |= routes_at_arc[bit]
                        remaining -= bit
                    cached[blocked] = conflicts
                conflicts &= ~own
                packed = np.frombuffer(conflicts.to_bytes(bytes_per_row, 'little'), dtype=np.uint8)
                self.pair[index] = np.unpackbits(packed, bitorder='little', count=count)
        self.loads = np.zeros((count, len(self.capacity)), dtype=np.int16)
        for row, (indices, _) in enumerate(self.capacity):
            self.loads[list(indices), row] = 1
        self.caps = np.array([cap for _, cap in self.capacity], dtype=np.int16)
        self.penalty = int(self.cost_all[self.incumbent].sum() - self.constant) + 1
        self.qubo = None
        self.stats.update(reduced_build_sec=REAL_TIME() - began,
                          reduced_constructor='bitset_conflicts_vectorized_capacity',
                          resource_variable_limit=self.variable_limit)

    def compile_qubo(self, max_variables=None):
        limit = self.variable_limit if max_variables is None else max_variables
        return super().compile_qubo(max_variables=limit)
