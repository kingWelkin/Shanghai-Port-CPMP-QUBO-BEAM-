"""贪心动作：仅跨堆/跨贝搬顶箱（禁止同堆内翻箱）。"""

from __future__ import annotations

import math
from typing import List, Tuple

# (cross, 源堆 i, 源层 j, 目标堆 k, 目标层 l, 箱秩 n)
GreedyMove = Tuple


def g_cap_from_n0(n0: int, ratio: float = 1.5) -> int:
    """Return ``floor(ratio * N(0))`` for the business G cap."""
    if n0 <= 0:
        return 0
    r = min(float(ratio), 2.0)
    if r <= 0:
        raise ValueError("ratio must be positive")
    return max(0, int(math.floor(r * float(n0) + 1e-9)))


def g_value(num_moves: int, n_future: int) -> int:
    return int(num_moves) + int(n_future)


def apply_move(stacks: List[List[int]], move: GreedyMove) -> None:
    if move[0] != "cross":
        raise ValueError(f"仅允许跨堆动作，收到: {move[0]}")
    _, i, j_from, k, l_to, n = move
    st = stacks[i - 1]
    if not st or st[-1] != n:
        raise RuntimeError(f"跨堆重放失败 堆{i}")
    st.pop()
    stacks[k - 1].append(n)


def copy_stacks(stacks: List[List[int]]) -> List[List[int]]:
    return [list(s) for s in stacks]


def simulate_move(
    stacks: List[List[int]], move: GreedyMove
) -> List[List[int]]:
    new = copy_stacks(stacks)
    apply_move(new, move)
    return new
