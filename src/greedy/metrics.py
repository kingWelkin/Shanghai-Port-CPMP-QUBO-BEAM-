"""G(x)=M+N 经典度量与解向量解码。"""

from __future__ import annotations

import math
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import irange


def stack_boxes_from_pri_row(row: Sequence[Optional[int]], H: int) -> List[Optional[int]]:
    return [row[j - 1] if j - 1 < len(row) else None for j in irange(1, H)]


def _tier_seq_value(
    rank_or_seq: Optional[int],
    rank_to_seq: Optional[Dict[int, int]],
) -> Optional[int]:
    if rank_or_seq is None:
        return None
    if rank_to_seq is not None:
        return int(rank_to_seq.get(int(rank_or_seq), int(rank_or_seq)))
    return int(rank_or_seq)


def n_future_single_stack(
    boxes_by_tier: Sequence[Optional[int]],
    H: int,
    *,
    rank_to_seq: Optional[Dict[int, int]] = None,
) -> int:
    """
    单堆逆序箱数 N(Future)：

    从第 2 层向上 (2,3,4,5,6)，找**第一个**「上层关序号 > 下层关序号」的层 j；
    将该层及其上方所有非空箱计入逆序箱数。

    ``boxes_by_tier[j-1]`` 为第 j 层箱标识（通常为秩）；若提供 ``rank_to_seq`` 则按关序号比较。
    良序要求：下层关序号 > 上层（即上层关序号 < 下层）。
    """
    boxes = list(boxes_by_tier)
    first_inv: Optional[int] = None
    for j in irange(2, H):
        lo = _tier_seq_value(boxes[j - 2], rank_to_seq)
        hi = _tier_seq_value(boxes[j - 1], rank_to_seq)
        if lo is None or hi is None:
            continue
        if hi > lo:
            first_inv = j
            break
    if first_inv is None:
        return 0
    cnt = 0
    for j in irange(first_inv, H):
        if boxes[j - 1] is not None:
            cnt += 1
    return cnt


def first_inversion_tier(
    boxes_by_tier: Sequence[Optional[int]],
    H: int,
    *,
    rank_to_seq: Optional[Dict[int, int]] = None,
) -> Optional[int]:
    """首个逆序层 j（上层关序号 > 下层）；无则 None。"""
    boxes = list(boxes_by_tier)
    for j in irange(2, H):
        lo = _tier_seq_value(boxes[j - 2], rank_to_seq)
        hi = _tier_seq_value(boxes[j - 1], rank_to_seq)
        if lo is None or hi is None:
            continue
        if hi > lo:
            return j
    return None


def adjacent_order_violations(
    boxes_by_tier: Sequence[Optional[int]], H: int
) -> List[Tuple[int, int, int, int]]:
    return adjacent_order_violations_by_seq(boxes_by_tier, H)


def adjacent_order_violations_by_seq(
    boxes_by_tier: Sequence[Optional[int]],
    H: int,
    *,
    rank_to_seq: Optional[Dict[int, int]] = None,
) -> List[Tuple[int, int, int, int]]:
    """相邻层关序号违例（仅诊断；逆序箱数用 n_future_single_stack）。"""
    boxes = list(boxes_by_tier)
    out: List[Tuple[int, int, int, int]] = []
    for j in irange(2, H):
        lo = _tier_seq_value(boxes[j - 2], rank_to_seq)
        hi = _tier_seq_value(boxes[j - 1], rank_to_seq)
        if lo is None or hi is None:
            continue
        if hi > lo:
            out.append((j - 1, j, lo, hi))
    return out


def format_single_stack_order_line(
    boxes_by_tier: Sequence[Optional[int]],
    H: int,
    label: str,
    *,
    rank_to_seq: Optional[Dict[int, int]] = None,
) -> str:
    n = n_future_single_stack(boxes_by_tier, H, rank_to_seq=rank_to_seq)
    if n == 0:
        return f"  堆{label}: OK"
    start = first_inversion_tier(boxes_by_tier, H, rank_to_seq=rank_to_seq)
    viol = adjacent_order_violations_by_seq(boxes_by_tier, H, rank_to_seq=rank_to_seq)
    parts = [f"层{v[0]}<{v[1]} 下{v[2]}上{v[3]}" for v in viol]
    detail = "; ".join(parts)
    if start is not None:
        return f"  堆{label}: 逆序 {n} 箱（自层{start}） — {detail}"
    return f"  堆{label}: 逆序 {n} 箱 — {detail}"


def stack_order_satisfied(
    boxes_by_tier: Sequence[Optional[int]],
    H: int,
    *,
    rank_to_seq: Optional[Dict[int, int]] = None,
) -> bool:
    return n_future_single_stack(boxes_by_tier, H, rank_to_seq=rank_to_seq) == 0


def stack_row_label(i: int) -> str:
    return "A" if int(i) == 10 else str(int(i))


def format_stack_order_report(
    grid: List[List[Optional[int]]],
    W: int,
    H: int,
) -> List[str]:
    lines = [
        "【堆叠验收】下层关序号 > 上层；逆序箱数：自第2层向上首个上层>下层，该层及以上计入",
    ]
    all_ok = True
    for i in irange(1, W):
        row = grid[i - 1] if i - 1 < len(grid) else []
        boxes = stack_boxes_from_pri_row(row, H)
        label = stack_row_label(i)
        line = format_single_stack_order_line(boxes, H, label)
        if not line.endswith(": OK"):
            all_ok = False
        lines.append(line)
    lines.append(f"  全场良序: {'是' if all_ok else '否'}")
    return lines


def n_future_bay(
    pri: List[List[Optional[int]]],
    W: int,
    H: int,
    *,
    seq_grid: Optional[List[List[Optional[int]]]] = None,
    rank_to_seq: Optional[Dict[int, int]] = None,
) -> int:
    total = 0
    for i in irange(1, W):
        row = pri[i - 1] if i - 1 < len(pri) else []
        boxes = stack_boxes_from_pri_row(row, H)
        rts = None if seq_grid is not None else rank_to_seq
        if seq_grid is not None:
            row_s = seq_grid[i - 1] if i - 1 < len(seq_grid) else []
            boxes = stack_boxes_from_pri_row(row_s, H)
        total += n_future_single_stack(boxes, H, rank_to_seq=rts)
    return total


def n0_from_problem(problem) -> int:
    from .in_stack_shuffle import build_rank_to_seq

    seq = getattr(problem.bay, "seq", None)
    rank_to_seq = build_rank_to_seq(problem)
    return n_future_bay(
        problem.bay.pri,
        problem.W,
        problem.H,
        seq_grid=seq,
        rank_to_seq=rank_to_seq,
    )


def max_planned_moves_limit(problem) -> int:
    """规划翻箱步数上限：由 T < 1.5*N(0) 得 T_cap，最多 T_cap-2 步。"""
    return max_moves_for_time_cap(max_time_horizon_for_problem(problem))


def max_time_horizon_from_n0(n0: int) -> int:
    """最大时刻 T：整数 T 满足 T < 1.5 * N(0)。"""
    n0 = int(n0)
    if n0 <= 0:
        return 2
    return max(2, int(math.floor(1.5 * float(n0) - 1e-9)))


def max_time_horizon_for_problem(problem) -> int:
    return max_time_horizon_from_n0(n0_from_problem(problem))


def max_moves_for_time_cap(T_cap: int) -> int:
    """T 步时间轴下最多翻箱次数（初态 + 各步 + 末态 → 步数 ≤ T-2）。"""
    return max(0, int(T_cap))


def effective_move_cap(problem) -> Tuple[int, int]:
    """返回 (翻箱步数上限, T 上限)；均来自 T < 1.5*N(0)。"""
    n0 = n0_from_problem(problem)
    t_cap = max_time_horizon_from_n0(n0)
    move_cap = max_moves_for_time_cap(t_cap)
    return move_cap, t_cap


def g_cap_from_n0(n0: int, ratio: float) -> int:
    r = float(ratio)
    if r <= 0.0:
        raise ValueError("ratio 须为正")
    if r > 2.0:
        r = 2.0
    if n0 <= 0:
        return max(0, int(r) - 1) if r < 1.0 else 0
    return max(0, int(n0 * r + 1e-9))


def layout_at_time_from_b(
    pri_init: List[List[Optional[int]]],
    b_vals: Dict[Tuple[int, int, int, int], int],
    W: int,
    H: int,
    N: int,
    t: int,
) -> List[List[Optional[int]]]:
    layout: List[List[Optional[int]]] = [[None for _ in range(H)] for _ in range(W)]
    for i in irange(1, W):
        for j in irange(1, H):
            found: Optional[int] = None
            for n in irange(1, N):
                if int(b_vals.get((i, j, n, t), 0)) == 1:
                    found = n
                    break
            layout[i - 1][j - 1] = found
    if t == 1 and all(layout[i][j] is None for i in range(W) for j in range(H)):
        return [list(row) for row in pri_init]
    return layout


def cumulative_reloc_until_t(
    x_vals: Dict[Tuple[int, int, int, int, int, int], int],
    W: int,
    H: int,
    N: int,
    t: int,
) -> int:
    if t <= 1:
        return 0
    cnt = 0
    for tau in irange(1, t - 1):
        for i, j, k, l, n in product(
            irange(1, W), irange(1, H), irange(1, W), irange(1, H), irange(1, N)
        ):
            if (i, j) == (k, l):
                continue
            if int(x_vals.get((i, j, k, l, n, tau), 0)) == 1:
                cnt += 1
    return cnt


def evaluate_g_trajectory(
    pri: List[List[Optional[int]]],
    W: int,
    H: int,
    N: int,
    T: int,
    b_vals: Dict[Tuple[int, int, int, int], int],
    x_vals: Dict[Tuple[int, int, int, int, int, int], int],
) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    for t in irange(1, T):
        layout = layout_at_time_from_b(pri, b_vals, W, H, N, t)
        n_t = n_future_bay(layout, W, H)
        m_t = cumulative_reloc_until_t(x_vals, W, H, N, t)
        out.append((t, m_t, n_t, m_t + n_t))
    return out


def extract_bvx_from_var_index(
    best: np.ndarray,
    var_index: Dict[Tuple, int],
) -> Tuple[
    Dict[Tuple[int, int, int, int], int],
    Dict[Tuple[int, int, int, int, int, int], int],
]:
    b_vals: Dict[Tuple[int, int, int, int], int] = {}
    x_vals: Dict[Tuple[int, int, int, int, int, int], int] = {}
    for key, idx in var_index.items():
        if int(best[idx]) != 1:
            continue
        if key[0] == "b":
            b_vals[key[1:]] = 1  # type: ignore[misc]
        elif key[0] == "x":
            x_vals[key[1:]] = 1  # type: ignore[misc]
    return b_vals, x_vals
