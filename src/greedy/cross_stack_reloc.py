"""跨堆翻箱：仅允许从堆 i 顶箱搬到堆 k≠i（禁止同堆内调整）。"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .in_stack_shuffle import (
    _column_grid,
    _encode_b,
    build_rank_to_seq,
    stack_order_ok_list,
    stack_sorted_bottom_up,
)
from .metrics import max_planned_moves_limit
from .utils import irange

BKey = Tuple[int, int, int, int]
XKey = Tuple[int, int, int, int, int, int]
Move5 = Tuple[int, int, int, int, int]  # i, j_from, k, l_to, n


def _stacks_from_problem(problem) -> List[List[int]]:
    return [
        [int(z) for z in problem.bay.pri[i - 1] if z is not None]
        for i in irange(1, problem.W)
    ]


def _choose_dest_stack(
    W: int,
    H: int,
    stacks: List[List[int]],
    src_i: int,
    rng: random.Random,
) -> int:
    cands = [
        k
        for k in irange(1, W)
        if k != src_i and len(stacks[k - 1]) < H
    ]
    if not cands:
        return -1
    cands.sort(key=lambda k: len(stacks[k - 1]))
    return cands[rng.randint(0, min(2, len(cands) - 1))]


def plan_cross_stack_reloc_moves(
    problem,
    seed: int = 0,
    *,
    max_moves: Optional[int] = None,
) -> List[Move5]:
    """
    贪心规划跨堆翻箱：仅从「尚未良序」的堆移出顶箱；已良序的堆不动。
    返回 (源堆 i, 源层 j, 目标堆 k, 目标层 l, 箱 n)。
    """
    rng = random.Random(seed)
    W, H = problem.W, problem.H
    rank_to_seq = build_rank_to_seq(problem)
    stacks = _stacks_from_problem(problem)
    moves: List[Move5] = []
    limit = (
        int(max_moves)
        if max_moves is not None
        else max_planned_moves_limit(problem)
    )

    for _ in range(limit):
        if len(moves) >= limit:
            break
        if all(
            stack_order_ok_list(stacks[i - 1], rank_to_seq)
            for i in irange(1, W)
            if stacks[i - 1]
        ):
            break

        src_candidates = [
            i
            for i in irange(1, W)
            if stacks[i - 1]
            and not stack_order_ok_list(stacks[i - 1], rank_to_seq)
        ]
        # 各堆均已良序则停止，不从已排好序的堆再搬箱
        if not src_candidates:
            break

        src_i = rng.choice(src_candidates)
        st = stacks[src_i - 1]
        n = int(st[-1])
        j_from = len(st)
        dest_i = _choose_dest_stack(W, H, stacks, src_i, rng)
        if dest_i < 0:
            break
        dst = stacks[dest_i - 1]
        l_to = len(dst) + 1
        st.pop()
        dst.append(n)
        moves.append((src_i, j_from, dest_i, l_to, n))

    return moves


def cross_stack_time_horizon(
    problem,
    planned_moves: Optional[List[Move5]] = None,
) -> int:
    if planned_moves is None:
        planned_moves = plan_cross_stack_reloc_moves(problem)
    return max(2, len(planned_moves) + 2)


def construct_one_cross_stack_solution(
    problem,
    seed: int = 0,
    *,
    T: Optional[int] = None,
    planned_moves: Optional[List[Move5]] = None,
) -> Dict[str, Dict]:
    """构造仅含跨堆 x[i,j,k,l,n,t]（i≠k）的可行初解。"""
    W, H, N = problem.W, problem.H, problem.N
    all_moves = (
        planned_moves
        if planned_moves is not None
        else plan_cross_stack_reloc_moves(problem, seed=seed)
    )
    need_t = len(all_moves) + 2
    T = max(int(T), need_t) if T is not None else need_t

    b: Dict[BKey, int] = {}
    x: Dict[XKey, int] = {}
    stacks_work = _stacks_from_problem(problem)
    cols = _column_grid(stacks_work, H, W)
    _encode_b(problem, cols, 1, b)

    for step, (i, j_from, k, l_to, n) in enumerate(all_moves):
        if i == k:
            raise RuntimeError(f"规划含同堆动作: 堆{i} 箱{n}")
        t = step + 1
        x[(i, j_from, k, l_to, n, t)] = 1
        st = stacks_work[i - 1]
        if not st or st[-1] != n:
            raise RuntimeError(
                f"跨堆重放不一致 t={t} 堆{i}→堆{k}: 顶箱={st[-1] if st else None} 期望 n={n}"
            )
        st.pop()
        dst = stacks_work[k - 1]
        if len(dst) >= H:
            raise RuntimeError(f"目标堆{k}已满 t={t}")
        dst.append(n)
        if l_to != len(dst):
            raise RuntimeError(
                f"层位不一致 t={t} 堆{k}: 记录 l={l_to} 实际高={len(dst)}"
            )
        cols = _column_grid(stacks_work, H, W)
        _encode_b(problem, cols, t + 1, b)

    for t in irange(len(all_moves) + 2, T + 1):
        _encode_b(problem, cols, t, b)

    return {"b": b, "x": x}


def flatten_cross_stack_solution(
    sol: Dict[str, Dict],
    var_index: Dict[Tuple, int],
    n_var: int,
) -> np.ndarray:
    vec = np.zeros(n_var, dtype=np.int8)
    for key, val in sol["b"].items():
        idx = var_index.get(("b",) + key)
        if idx is not None:
            vec[idx] = int(val)
    for key, val in sol["x"].items():
        i, j, k, l, n, t = key
        if i == k:
            continue
        idx = var_index.get(("x",) + key)
        if idx is not None:
            vec[idx] = int(val)
    return vec
