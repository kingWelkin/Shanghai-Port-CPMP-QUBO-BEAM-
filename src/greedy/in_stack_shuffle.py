"""同堆翻箱：BFS 规划与可行初解构造。"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from .metrics import max_planned_moves_limit
from .utils import irange

BKey = Tuple[int, int, int, int]
XKey = Tuple[int, int, int, int, int, int]


def home_stack_of_boxes(problem) -> Dict[int, int]:
    home: Dict[int, int] = {}
    pri = problem.bay.pri
    for i in irange(1, problem.W):
        for j in irange(1, problem.H):
            n = pri[i - 1][j - 1]
            if n is not None:
                home[int(n)] = int(i)
    return home


def build_rank_to_seq(problem) -> Dict[int, int]:
    rank_to_seq: Dict[int, int] = {}
    pri = problem.bay.pri
    seq = problem.bay.seq if problem.bay.seq is not None else problem.bay.pri
    for i in range(problem.W):
        for j in range(problem.H):
            r = pri[i][j]
            s = seq[i][j]
            if r is not None and s is not None:
                rank_to_seq[int(r)] = int(s)
    return rank_to_seq


def stack_sorted_bottom_up(stack: List[int], rank_to_seq: Dict[int, int]) -> List[int]:
    return sorted(stack, key=lambda n: rank_to_seq.get(n, n), reverse=True)


def stack_order_ok_list(stack: List[int], rank_to_seq: Dict[int, int]) -> bool:
    """自下而上关序号递减（下层 > 上层）。"""
    for k in range(len(stack) - 1):
        s_lo = rank_to_seq.get(stack[k], stack[k])
        s_hi = rank_to_seq.get(stack[k + 1], stack[k + 1])
        if int(s_hi) > int(s_lo):
            return False
    return True


def _list_to_col(stack: List[int], H: int) -> List[Optional[int]]:
    col: List[Optional[int]] = [None] * H
    for k, n in enumerate(stack):
        if k < H:
            col[k] = n
    return col


def _move_top_insert(
    stack: List[int],
    insert_pos: int,
) -> Tuple[int, int, int]:
    if not stack:
        raise ValueError("空堆无法翻箱")
    if insert_pos < 0 or insert_pos > len(stack) - 1:
        raise ValueError(
            f"插入位置非法 insert_pos={insert_pos}, stack_len={len(stack)}"
        )
    j_from = len(stack)
    n = int(stack[-1])
    stack.pop()
    stack.insert(insert_pos, n)
    return j_from, insert_pos + 1, n


def _sort_stack_moves_bfs(
    stack: List[int],
    rank_to_seq: Dict[int, int],
) -> List[Tuple[int, int, int]]:
    if len(stack) <= 1:
        return []
    target = tuple(stack_sorted_bottom_up(stack, rank_to_seq))
    start = tuple(stack)
    if start == target:
        return []

    q: deque = deque([(start, [])])
    seen = {start}
    while q:
        state, path = q.popleft()
        st = list(state)
        j_from = len(st)
        n_top = int(st[-1])
        for insert_pos in range(len(st) - 1):
            new_st = st[:-1]
            new_st.insert(insert_pos, n_top)
            new_state = tuple(new_st)
            move = (j_from, insert_pos + 1, n_top)
            new_path = path + [move]
            if new_state == target:
                return new_path
            if new_state not in seen:
                seen.add(new_state)
                q.append((new_state, new_path))
    raise RuntimeError(f"BFS 未找到单堆排序路径 h={len(stack)} stack={list(stack)}")


def sort_stack_moves_fixed(
    stack: List[int],
    rank_to_seq: Dict[int, int],
    H: int,
) -> List[Tuple[int, int, int]]:
    if len(stack) > H:
        raise ValueError(f"堆高 {len(stack)} 超过最大层数 H={H}")
    work = list(stack)
    moves = _sort_stack_moves_bfs(work, rank_to_seq)
    for _j_from, j_to, _n in moves:
        _move_top_insert(work, j_to - 1)
    stack[:] = work
    return moves


def plan_all_shuffle_moves(
    problem,
    *,
    max_moves: Optional[int] = None,
) -> List[Tuple[int, int, int, int]]:
    rank_to_seq = build_rank_to_seq(problem)
    cap = (
        int(max_moves)
        if max_moves is not None
        else max_planned_moves_limit(problem)
    )
    all_moves: List[Tuple[int, int, int, int]] = []
    for i in irange(1, problem.W):
        st = [int(z) for z in problem.bay.pri[i - 1] if z is not None]
        if not st:
            continue
        if stack_order_ok_list(st, rank_to_seq):
            continue
        for j_from, j_to, n in sort_stack_moves_fixed(st, rank_to_seq, problem.H):
            if len(all_moves) >= cap:
                return all_moves
            all_moves.append((i, j_from, j_to, n))
    return all_moves


def _column_grid(stacks: List[List[int]], H: int, W: int) -> List[List[Optional[int]]]:
    if len(stacks) < W:
        raise ValueError(f"stacks 长度 {len(stacks)} < W={W}")
    return [_list_to_col(stacks[i - 1], H) for i in irange(1, W)]


def _encode_b(problem, cols: List[List[Optional[int]]], t: int, b: Dict[BKey, int]) -> None:
    for i in irange(1, problem.W):
        for j in irange(1, problem.H):
            n_here = cols[i - 1][j - 1]
            for n in irange(1, problem.N):
                b[(i, j, n, t)] = 1 if n_here == n else 0


def construct_one_in_stack_solution(
    problem,
    seed: int = 0,
    *,
    T: Optional[int] = None,
    planned_moves: Optional[List[Tuple[int, int, int, int]]] = None,
    check_home: bool = True,
) -> Dict[str, Dict]:
    _ = seed
    W, H, N = problem.W, problem.H, problem.N
    home = home_stack_of_boxes(problem)

    all_moves = (
        planned_moves if planned_moves is not None else plan_all_shuffle_moves(problem)
    )
    need_t = len(all_moves) + 2
    if T is None:
        T = need_t
    else:
        T = max(int(T), need_t)

    b: Dict[BKey, int] = {}
    x: Dict[XKey, int] = {}
    stacks_work: List[List[int]] = [
        [int(z) for z in problem.bay.pri[i - 1] if z is not None] for i in irange(1, W)
    ]
    cols = _column_grid(stacks_work, H, W)
    _encode_b(problem, cols, 1, b)

    for step, (i, j_from, j_to, n) in enumerate(all_moves):
        t = step + 1
        x[(i, j_from, i, j_to, n, t)] = 1
        st = stacks_work[i - 1]
        if not st or st[-1] != n:
            raise RuntimeError(
                f"重放翻箱不一致 t={t} 堆{i}: 顶箱={st[-1] if st else None} 期望 n={n}"
            )
        _move_top_insert(st, j_to - 1)
        cols = _column_grid(stacks_work, H, W)
        _encode_b(problem, cols, t + 1, b)

    for t in irange(len(all_moves) + 2, T + 1):
        _encode_b(problem, cols, t, b)

    if check_home:
        for n in irange(1, N):
            i0 = home[n]
            for t in irange(1, T + 1):
                on_home = sum(b[(i0, j, n, t)] for j in irange(1, H))
                if on_home != 1:
                    raise RuntimeError(
                        f"箱 {n} 在 t={t} 不在初始堆 {i0} (on_home={on_home})"
                    )
                elsewhere = sum(
                    b[(i, j, n, t)]
                    for i in irange(1, W)
                    for j in irange(1, H)
                    if i != i0
                )
                if elsewhere != 0:
                    raise RuntimeError(f"箱 {n} 在 t={t} 离开了初始堆 {i0}")

    return {"b": b, "x": x}


def is_valid_top_out_move(
    b_vals: Dict[BKey, int],
    stack_i: int,
    j_from: int,
    n: int,
    t: int,
    H: int,
    N: int,
) -> bool:
    """时刻 t 翻箱前，箱 n 须在堆 stack_i 的顶箱层 j_from。"""
    top_j = top_tier_of_stack_at(b_vals, stack_i, t, H, N)
    if top_j is None or top_j != j_from:
        return False
    return int(b_vals.get((stack_i, j_from, n, t), 0)) == 1


def flatten_in_stack_solution(
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
        idx = var_index.get(("x",) + key)
        if idx is not None:
            vec[idx] = int(val)
    return vec


def top_tier_of_stack_at(
    b_vals: Dict[BKey, int],
    stack_i: int,
    t: int,
    H: int,
    N: int,
) -> Optional[int]:
    """时刻 t 堆 stack_i 顶箱所在层（1-based），空堆返回 None。"""
    for j in range(H, 0, -1):
        for n in irange(1, N):
            if int(b_vals.get((stack_i, j, n, t), 0)) == 1:
                return j
    return None


def is_valid_top_insert_move(
    b_vals: Dict[BKey, int],
    stack_i: int,
    j_from: int,
    j_to: int,
    n: int,
    t: int,
    H: int,
    N: int,
) -> bool:
    """翻箱前须为顶箱，且只能插入比原顶位更低的层（提起→插入）。"""
    if j_from == j_to:
        return False
    if j_to >= j_from:
        return False
    top_j = top_tier_of_stack_at(b_vals, stack_i, t, H, N)
    if top_j is None or top_j != j_from:
        return False
    return int(b_vals.get((stack_i, j_from, n, t), 0)) == 1


def shuffle_time_horizon(
    problem,
    planned_moves: Optional[List[Tuple[int, int, int, int]]] = None,
) -> int:
    if planned_moves is None:
        planned_moves = plan_all_shuffle_moves(problem)
    return max(2, len(planned_moves) + 2)
