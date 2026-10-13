"""将贪心轨迹编码为 QUBO 全部变量（含 active / s_idle / s_cell / s_stack）。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .cross_stack_reloc import Move5
from .in_stack_shuffle import _column_grid, _encode_b
from .utils import irange

BKey = Tuple[int, int, int, int]
XKey = Tuple[int, int, int, int, int, int]


def _encode_slacks_from_cols(
    cols: List[List[Optional[int]]],
    W: int,
    H: int,
    t: int,
    s_cell: Dict[Tuple[int, int, int], int],
    s_stack: Dict[Tuple[int, int, int], int],
) -> None:
    for i in irange(1, W):
        for j in irange(1, H):
            occ = cols[i - 1][j - 1] is not None
            s_cell[(i, j, t)] = 0 if occ else 1
        for j in irange(1, H - 1):
            has_lo = cols[i - 1][j - 1] is not None
            has_hi = cols[i - 1][j] is not None
            s_stack[(i, j, t)] = 1 if has_lo and not has_hi else 0


def split_moves_for_qubo_window(
    planned_moves: List[Move5],
    T_window: int,
) -> Tuple[List[Move5], List[Move5]]:
    """长轨迹裁到 QUBO 窗口：前缀重放为初态，后缀在窗口内编码。"""
    max_in_window = max(0, int(T_window) - 2)
    if len(planned_moves) <= max_in_window:
        return [], list(planned_moves)
    prefix = planned_moves[: len(planned_moves) - max_in_window]
    suffix = planned_moves[len(planned_moves) - max_in_window :]
    return prefix, suffix


def construct_full_yard_solution(
    problem,
    planned_moves: List[Move5],
    *,
    T: Optional[int] = None,
    initial_stacks: Optional[List[List[int]]] = None,
) -> Dict[str, Dict]:
    """跨堆轨迹 → b/x/active/s_idle/s_cell/s_stack（与 yard_reloc QUBO 一致）。"""
    W, H, N = problem.W, problem.H, problem.N
    m = len(planned_moves)
    need_t = m + 2
    T = max(int(T), need_t) if T is not None else need_t

    b: Dict[BKey, int] = {}
    x: Dict[XKey, int] = {}
    active: Dict[int, int] = {}
    s_idle: Dict[int, int] = {}
    s_cell: Dict[Tuple[int, int, int], int] = {}
    s_stack: Dict[Tuple[int, int, int], int] = {}

    if initial_stacks is not None:
        stacks_work = [list(s) for s in initial_stacks]
    else:
        stacks_work = [
            [int(z) for z in problem.bay.pri[i - 1] if z is not None]
            for i in irange(1, W)
        ]
    cols = _column_grid(stacks_work, H, W)
    _encode_b(problem, cols, 1, b)
    _encode_slacks_from_cols(cols, W, H, 1, s_cell, s_stack)

    for step, (i, j_from, k, l_to, n) in enumerate(planned_moves):
        if i == k:
            raise RuntimeError(f"同堆动作: 堆{i} 箱{n}")
        t = step + 1
        active[t] = 1
        s_idle[t] = 0
        x[(i, j_from, k, l_to, n, t)] = 1
        st = stacks_work[i - 1]
        if not st or st[-1] != n:
            raise RuntimeError(
                f"重放不一致 t={t} 堆{i}→堆{k}: 顶={st[-1] if st else None} 期望 {n}"
            )
        st.pop()
        dst = stacks_work[k - 1]
        if len(dst) >= H:
            raise RuntimeError(f"目标堆{k}已满 t={t}")
        dst.append(n)
        cols = _column_grid(stacks_work, H, W)
        _encode_b(problem, cols, t + 1, b)
        _encode_slacks_from_cols(cols, W, H, t + 1, s_cell, s_stack)

    for t in irange(m + 1, T):
        active[t] = 0
        s_idle[t] = 1

    for t in irange(m + 2, T + 1):
        _encode_b(problem, cols, t, b)
        _encode_slacks_from_cols(cols, W, H, t, s_cell, s_stack)

    return {
        "b": b,
        "x": x,
        "active": active,
        "s_idle": s_idle,
        "s_cell": s_cell,
        "s_stack": s_stack,
    }


def flatten_full_yard_solution(
    sol: Dict[str, Dict],
    var_index: Dict[Tuple, int],
    n_var: int,
) -> np.ndarray:
    vec = np.zeros(n_var, dtype=np.int8)
    for prefix, table in (
        ("b", sol["b"]),
        ("x", sol["x"]),
        ("s_cell", sol["s_cell"]),
        ("s_stack", sol["s_stack"]),
    ):
        for key, val in table.items():
            if prefix == "x":
                i, j, k, l, n, t = key
                if i == k:
                    continue
            idx = var_index.get((prefix,) + key)
            if idx is not None:
                vec[idx] = int(val)
    for t, val in sol["active"].items():
        idx = var_index.get(("active", t))
        if idx is not None:
            vec[idx] = int(val)
    for t, val in sol["s_idle"].items():
        idx = var_index.get(("s_idle", t))
        if idx is not None:
            vec[idx] = int(val)
    return vec


def count_feasibility_violations(
    problem,
    sol: Dict[str, Dict],
    *,
    T: int,
) -> Dict[str, int]:
    """粗检硬约束违反次数（用于诊断初解）。"""
    from itertools import product

    W, H, N = problem.W, problem.H, problem.N
    b, x = sol["b"], sol["x"]
    active, s_idle = sol["active"], sol["s_idle"]
    s_cell, s_stack = sol["s_cell"], sol["s_stack"]
    v: Dict[str, int] = {}

    for n, t in product(irange(1, N), irange(1, T)):
        s = sum(b.get((i, j, n, t), 0) for i, j in product(irange(1, W), irange(1, H)))
        if s != 1:
            v["box_presence"] = v.get("box_presence", 0) + 1

    for i, j, t in product(irange(1, W), irange(1, H), irange(1, T)):
        sc = s_cell.get((i, j, t), -1)
        sb = sum(b.get((i, j, n, t), 0) for n in irange(1, N))
        if sc + sb != 1:
            v["s_cell"] = v.get("s_cell", 0) + 1

    for i, j, t in product(irange(1, W), irange(1, H - 1), irange(1, T)):
        ss = s_stack.get((i, j, t), -1)
        blo = sum(b.get((i, j, n, t), 0) for n in irange(1, N))
        bhi = sum(b.get((i, j + 1, n, t), 0) for n in irange(1, N))
        if ss - blo + bhi != 0:
            v["s_stack"] = v.get("s_stack", 0) + 1

    for t in irange(1, T + 1):
        if active.get(t, 0) + s_idle.get(t, 0) != 1:
            v["idle_active"] = v.get("idle_active", 0) + 1
        xsum = sum(val for key, val in x.items() if key[5] == t)
        if xsum > 1:
            v["multi_move"] = v.get("multi_move", 0) + 1
        if xsum > active.get(t, 0):
            v["x_active"] = v.get("x_active", 0) + 1

    for i, j, n, t in product(
        irange(1, W), irange(1, H), irange(1, N), irange(2, T + 1)
    ):
        tau = t - 1
        bt = b.get((i, j, n, t), 0)
        bm = b.get((i, j, n, tau), 0)
        delta = bt - bm
        for k, l in product(irange(1, W), irange(1, H)):
            if k == i and l == j:
                continue
            if x.get((k, l, i, j, n, tau), 0):
                delta += 1
            if x.get((i, j, k, l, n, tau), 0):
                delta -= 1
        if delta != 0:
            v["transition"] = v.get("transition", 0) + 1

    return v
