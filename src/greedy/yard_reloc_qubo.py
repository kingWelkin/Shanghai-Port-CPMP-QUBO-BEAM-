"""
全场翻箱 QUBO：仅允许跨堆移动（i≠k），禁止同堆内翻箱；不提箱（无 v/y）。

x[i,j,k,l,n,t]=1：时刻 t 将箱 n 从堆 i 层 j 搬到堆 k 层 l，且 i≠k。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .cross_stack_reloc import (
    cross_stack_time_horizon,
    plan_cross_stack_reloc_moves,
)
from .metrics import effective_move_cap, g_cap_from_n0, n0_from_problem
from .qubo_accum import (
    ListCouplingAccumulator,
    SubQuboAccumulator,
    add_compact_idle_reloc_only,
    add_compact_transition_reloc_only,
    add_penalty_linear_eq,
    add_unsafe_stack_adjacent,
    auto_time_horizon,
)
from .utils import irange


@dataclass
class YardRelocQuboSpec:
    problem: object
    W: int
    H: int
    N: int
    T: int
    var_index: Dict[Tuple, int]
    names: List[Tuple]
    x_keys: Set[Tuple]
    penalty: float
    reloc_scale: float
    terminal_stack_order_scale: float
    stack_order_scale: float
    meta: Dict
    linear: List[Tuple[int, float]] = field(default_factory=list)
    quads: List[Tuple[int, int, float]] = field(default_factory=list)

    @property
    def n_var(self) -> int:
        return len(self.var_index)

    def energy(self, assignment: np.ndarray) -> float:
        e = 0.0
        for i, w in self.linear:
            e += float(w) * float(assignment[i])
        for i, j, w in self.quads:
            e += float(w) * float(assignment[i]) * float(assignment[j])
        return float(e)

    def subqubo(
        self,
        key_variables: Sequence[int],
        assignment: np.ndarray,
    ) -> np.ndarray:
        key_set = {int(v) for v in key_variables}
        fixed = {i: int(assignment[i]) for i in range(self.n_var) if i not in key_set}
        acc = SubQuboAccumulator(key_variables, fixed)
        for i, w in self.linear:
            if i in key_set:
                acc.add_h(i, w)
        for i, j, w in self.quads:
            if i in key_set or j in key_set:
                acc.add_J(i, j, w)
        subQ, _ = acc.finalize()
        return subQ

    def _vid(self, key: Tuple) -> int:
        return self.var_index[key]

    def _accumulate_all(self, acc) -> None:
        W, H, N, T = self.W, self.H, self.N, self.T
        vid = self._vid
        P = self.penalty
        rs = self.reloc_scale
        problem = self.problem
        x_keys = self.x_keys

        if rs != 0.0:
            for key in x_keys:
                acc.add_h(vid(key), rs)

        # (1') 每箱始终在堆场
        for n, t in product(irange(1, N), irange(1, T)):
            terms = [(vid(("b", i, j, n, t)), 1.0) for i, j in product(irange(1, W), irange(1, H))]
            add_penalty_linear_eq(acc, terms, 1.0, P)

        for i, j, t in product(irange(1, W), irange(1, H), irange(1, T)):
            terms = [(vid(("s_cell", i, j, t)), 1.0)]
            for n in irange(1, N):
                terms.append((vid(("b", i, j, n, t)), 1.0))
            add_penalty_linear_eq(acc, terms, 1.0, P)

        for i, j, t in product(irange(1, W), irange(1, H - 1), irange(1, T)):
            terms = [(vid(("s_stack", i, j, t)), -1.0)]
            for n in irange(1, N):
                terms.append((vid(("b", i, j, n, t)), 1.0))
                terms.append((vid(("b", i, j + 1, n, t)), -1.0))
            add_penalty_linear_eq(acc, terms, 0.0, P)

        add_compact_idle_reloc_only(
            acc, vid, W, H, N, T, P, x_keys=x_keys
        )

        add_compact_transition_reloc_only(
            acc, vid, W, H, N, T, P, x_keys=x_keys
        )

        # 源堆顶箱才可移出
        P_top = P * 4.0
        for t in irange(1, T):
            for i, j, k, l, n in product(
                irange(1, W), irange(1, H), irange(1, W), irange(1, H), irange(1, N)
            ):
                if i == k or (i, j) == (k, l):
                    continue
                key = ("x", i, j, k, l, n, t)
                if key not in x_keys:
                    continue
                xv = vid(key)
                for j_up in irange(j + 1, H):
                    for n_up in irange(1, N):
                        acc.add_J(xv, vid(("b", i, j_up, n_up, t)), P_top)

        for i, j, n in product(irange(1, W), irange(1, H), irange(1, N)):
            target = 1.0 if problem.bay.pri[i - 1][j - 1] == n else 0.0
            add_penalty_linear_eq(acc, [(vid(("b", i, j, n, 1)), 1.0)], target, P * 5.0)

        w_stack = float(self.stack_order_scale)
        if w_stack != 0.0:
            add_unsafe_stack_adjacent(acc, vid, W, H, N, T, w_stack)

        w_term = float(self.terminal_stack_order_scale)
        if w_term != 0.0:
            add_unsafe_stack_adjacent(
                acc, vid, W, H, N, T, w_term, time_steps=[T]
            )


def build_yard_reloc_spec(
    problem,
    *,
    penalty: float = 30.0,
    reloc_scale: float = 0.5,
    stack_order_scale: float = 5.0,
    terminal_stack_order_scale: float = 50.0,
    time_horizon: Optional[int] = None,
    auto_T: bool = True,
    clip_time_to_cap: bool = True,
    verbose_build: bool = False,
) -> YardRelocQuboSpec:
    W, H, N, T_full = problem.W, problem.H, problem.N, problem.T
    n0 = n0_from_problem(problem)
    move_cap, T_cap = effective_move_cap(problem)
    planned = plan_cross_stack_reloc_moves(problem, max_moves=move_cap)
    T_moves = min(cross_stack_time_horizon(problem, planned_moves=planned), T_cap)
    if time_horizon is not None:
        T = max(2, int(time_horizon))
        if clip_time_to_cap:
            T = min(T, T_cap)
    elif auto_T:
        T = min(max(T_moves, auto_time_horizon(N, T_full, W, H)), T_cap)
    else:
        T = min(max(T_moves, int(T_full)), T_cap)
    var_index: Dict[Tuple, int] = {}
    names: List[Tuple] = []
    x_keys: Set[Tuple] = set()

    def vid(key: Tuple) -> int:
        if key not in var_index:
            var_index[key] = len(names)
            names.append(key)
        return var_index[key]

    for i, j, n, t in product(irange(1, W), irange(1, H), irange(1, N), irange(1, T)):
        vid(("b", i, j, n, t))
    for i, j, k, l, n, t in product(
        irange(1, W),
        irange(1, H),
        irange(1, W),
        irange(1, H),
        irange(1, N),
        irange(1, T),
    ):
        if i == k or (i, j) == (k, l):
            continue
        key = ("x", i, j, k, l, n, t)
        x_keys.add(key)
        vid(key)
    for i, j, t in product(irange(1, W), irange(1, H), irange(1, T)):
        vid(("s_cell", i, j, t))
    for i, j, t in product(irange(1, W), irange(1, H - 1), irange(1, T)):
        vid(("s_stack", i, j, t))
    for t in irange(1, T):
        vid(("s_idle", t))
        vid(("active", t))

    layout_mode = (
        "multi_bay_reloc"
        if getattr(problem, "combined_bays", None)
        else "yard_reloc"
    )
    meta = {
        "mode": layout_mode,
        "cross_stack_only": True,
        "n0": n0,
        "g_cap": g_cap_from_n0(n0, 1.3),
        "cap_ratio": 1.3,
        "T_qubo": T,
        "n_planned_moves": len(planned),
        "max_planned_moves": move_cap,
        "T_cap": T_cap,
        "lazy": True,
    }

    spec = YardRelocQuboSpec(
        problem=problem,
        W=W,
        H=H,
        N=N,
        T=T,
        var_index=var_index,
        names=names,
        x_keys=x_keys,
        penalty=penalty,
        reloc_scale=reloc_scale,
        terminal_stack_order_scale=terminal_stack_order_scale,
        stack_order_scale=stack_order_scale,
        meta=meta,
    )

    t0 = time.perf_counter()
    acc = ListCouplingAccumulator()
    spec._accumulate_all(acc)
    spec.linear, spec.quads = acc.finalize()
    if verbose_build:
        print(
            f"  [跨堆翻箱] 耦合表: |linear|={len(spec.linear)}, |quads|={len(spec.quads)}, "
            f"T={T}, |x|={len(x_keys)}, build={time.perf_counter() - t0:.1f}s"
        )
    meta["n_couplings"] = len(spec.linear) + len(spec.quads)
    return spec
