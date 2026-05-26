"""QUBO 累积器。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import product
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .utils import irange

Pair = Tuple[int, int]


class QuboAccumulator(ABC):
    @abstractmethod
    def add_h(self, i: int, w: float) -> None: ...

    @abstractmethod
    def add_J(self, i: int, j: int, w: float) -> None: ...


class SubQuboAccumulator(QuboAccumulator):
    """仅生成 key 变量上的对称 subQ 与常数 offset（fixed 变量已赋值）。"""

    def __init__(
        self,
        key_variables: Sequence[int],
        fixed: Dict[int, int],
    ) -> None:
        self.key_pos = {int(v): i for i, v in enumerate(key_variables)}
        self.fixed = {int(k): int(v) for k, v in fixed.items()}
        k = len(key_variables)
        self.subQ = np.zeros((k, k), dtype=np.float64)
        self.offset = 0.0

    def add_h(self, i: int, w: float) -> None:
        if w == 0.0:
            return
        if i in self.key_pos:
            self.subQ[self.key_pos[i], self.key_pos[i]] += float(w)
        elif i in self.fixed:
            self.offset += float(w) * float(self.fixed[i])

    def add_J(self, i: int, j: int, w: float) -> None:
        if w == 0.0:
            return
        w = float(w)
        xi_k = i in self.key_pos
        xj_k = j in self.key_pos
        xi_f = i in self.fixed
        xj_f = j in self.fixed
        if xi_k and xj_k:
            r, s = self.key_pos[i], self.key_pos[j]
            if r == s:
                self.subQ[r, r] += w
            elif r > s:
                r, s = s, r
            self.subQ[r, s] += w
        elif xi_k and xj_f:
            self.subQ[self.key_pos[i], self.key_pos[i]] += w * float(self.fixed[j])
        elif xi_f and xj_k:
            self.subQ[self.key_pos[j], self.key_pos[j]] += w * float(self.fixed[i])
        elif xi_f and xj_f:
            self.offset += w * float(self.fixed[i]) * float(self.fixed[j])

    def finalize(self) -> Tuple[np.ndarray, float]:
        k = self.subQ.shape[0]
        for i in range(k):
            for j in range(i + 1, k):
                if self.subQ[i, j] != 0.0:
                    half = 0.5 * self.subQ[i, j]
                    self.subQ[i, j] = half
                    self.subQ[j, i] += half
        return self.subQ, float(self.offset)


def add_penalty_linear_eq(
    acc: QuboAccumulator,
    terms: Sequence[Tuple[int, float]],
    rhs: float,
    weight: float,
) -> None:
    """weight * (sum a_i x_i - rhs)^2"""
    P = float(weight)
    for i, ai in terms:
        acc.add_h(i, P * (ai * ai - 2.0 * rhs * ai))
    lst = list(terms)
    for u in range(len(lst)):
        i, ai = lst[u]
        for v in range(u + 1, len(lst)):
            j, aj = lst[v]
            acc.add_J(i, j, P * (2.0 * ai * aj))


def add_binary_leq(
    acc: QuboAccumulator,
    x: int,
    z: int,
    weight: float,
) -> None:
    """惩罚 P * x * (1-z)  =>  x<=z（均为 0/1）。"""
    P = float(weight)
    acc.add_h(x, P)
    acc.add_J(x, z, -P)


def add_unsafe_stack_adjacent(
    acc: QuboAccumulator,
    vid,
    W: int,
    H: int,
    N: int,
    T: int,
    weight: float,
    *,
    time_steps: Optional[Sequence[int]] = None,
) -> None:
    """惩罚相邻层「下层出库编号 < 上层」的占位。"""
    w = float(weight)
    if w == 0.0:
        return
    times = list(time_steps) if time_steps is not None else list(irange(1, T))
    for i in irange(1, W):
        for j in irange(1, H - 1):
            for t in times:
                if t < 1 or t > T:
                    continue
                for n_lo in irange(1, N):
                    for n_hi in irange(1, N):
                        if not (n_lo < n_hi):
                            continue
                        acc.add_J(vid(("b", i, j, n_lo, t)), vid(("b", i, j + 1, n_hi, t)), w)


def add_compact_idle_reloc_only(
    acc: QuboAccumulator,
    vid,
    W: int,
    H: int,
    N: int,
    T: int,
    penalty: float,
    *,
    x_keys: Optional[Set[Tuple]] = None,
) -> None:
    """每时刻至多一次翻箱 x（无提箱 y）。"""
    P = float(penalty)
    for t in irange(1, T):
        active = vid(("active", t))
        s_idle = vid(("s_idle", t))
        add_penalty_linear_eq(acc, [(active, 1.0), (s_idle, 1.0)], rhs=1.0, weight=P)
        for i, j, k, l, n in product(
            irange(1, W),
            irange(1, H),
            irange(1, W),
            irange(1, H),
            irange(1, N),
        ):
            if i == k or (i, j) == (k, l):
                continue
            key = ("x", i, j, k, l, n, t)
            if x_keys is not None and key not in x_keys:
                continue
            add_binary_leq(acc, vid(key), active, P)


def add_compact_transition_reloc_only(
    acc: QuboAccumulator,
    vid,
    W: int,
    H: int,
    N: int,
    T: int,
    penalty: float,
    *,
    x_keys: Optional[Set[Tuple]] = None,
) -> None:
    """b_t - b_{t-1} + in - out = 0（全场翻箱，无 y）。"""
    P = float(penalty)
    for i, j, n, t in product(irange(1, W), irange(1, H), irange(1, N), irange(2, T)):
        tau = t - 1
        bt = vid(("b", i, j, n, t))
        bm = vid(("b", i, j, n, tau))
        add_penalty_linear_eq(acc, [(bt, 1.0), (bm, -1.0)], 0.0, P)
        for k, l in product(irange(1, W), irange(1, H)):
            key_in = ("x", k, l, i, j, n, tau)
            if k != i and (k, l) != (i, j) and (x_keys is None or key_in in x_keys):
                u = vid(key_in)
                acc.add_h(u, P)
                acc.add_J(bt, u, 2.0 * P)
                acc.add_J(bm, u, -2.0 * P)
            key_out = ("x", i, j, k, l, n, tau)
            if k != i and (k, l) != (i, j) and (x_keys is None or key_out in x_keys):
                v = vid(key_out)
                acc.add_h(v, P)
                acc.add_J(bt, v, -2.0 * P)
                acc.add_J(bm, v, 2.0 * P)


def auto_time_horizon(N: int, T_full: int, W: int, H: int) -> int:
    if N <= 8 and W * H * T_full <= 800:
        return int(T_full)
    return max(2, min(int(T_full), 8, N + 2))


class ListCouplingAccumulator(QuboAccumulator):
    """构建阶段汇总线性/二次项，供 lazy 耦合表使用。"""

    def __init__(self) -> None:
        self._h: Dict[int, float] = {}
        self._J: Dict[Pair, float] = {}

    def add_h(self, i: int, w: float) -> None:
        if w == 0.0:
            return
        self._h[int(i)] = self._h.get(int(i), 0.0) + float(w)

    def add_J(self, i: int, j: int, w: float) -> None:
        if w == 0.0:
            return
        i, j = int(i), int(j)
        if i == j:
            self._h[i] = self._h.get(i, 0.0) + float(w)
            return
        a, b = (i, j) if i < j else (j, i)
        self._J[(a, b)] = self._J.get((a, b), 0.0) + float(w)

    def finalize(
        self,
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, int, float]]]:
        linear = list(self._h.items())
        quads = [(a, b, w) for (a, b), w in self._J.items()]
        return linear, quads
