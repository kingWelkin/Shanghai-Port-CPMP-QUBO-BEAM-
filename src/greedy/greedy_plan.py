"""
贪心规划：仅跨堆/跨贝搬顶箱（禁止同堆内翻箱）；
约束 G = 翻箱次数 + N(Future) ≤ 1.5·N(0)。
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple, Union

from .in_stack_shuffle import build_rank_to_seq, stack_order_ok_list
from .metrics import n_future_single_stack, stack_boxes_from_pri_row

from .greedy_moves import (
    GreedyMove,
    apply_move,
    g_cap_from_n0,
    g_value,
    simulate_move,
)
from .cross_stack_reloc import _stacks_from_problem
from .utils import irange


def _all_stacks_ok(
    stacks: List[List[int]],
    rank_to_seq: dict,
    W: int,
) -> bool:
    return all(
        stack_order_ok_list(stacks[i - 1], rank_to_seq)
        for i in irange(1, W)
        if stacks[i - 1]
    )


def _stacks_state_key(stacks: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(s) for s in stacks)


def _stack_bay(problem, global_i: int) -> Optional[int]:
    sb = getattr(problem, "stack_bay", None)
    if sb and 1 <= global_i <= len(sb):
        return int(sb[global_i - 1])
    return None


def _stack_n_future(
    stack: List[int],
    H: int,
    rank_to_seq: dict,
) -> int:
    if not stack:
        return 0
    col = stack_boxes_from_pri_row(
        [stack[k] if k < len(stack) else None for k in range(H)],
        H,
    )
    return n_future_single_stack(col, H, rank_to_seq=rank_to_seq)


def total_n_future(
    stacks: List[List[int]],
    W: int,
    H: int,
    rank_to_seq: dict,
) -> int:
    return sum(
        _stack_n_future(stacks[i - 1], H, rank_to_seq)
        for i in irange(1, W)
        if stacks[i - 1]
    )


def _move_tie_bonus_cross(
    problem,
    src_i: int,
    dest_i: int,
    *,
    drain_bay: Optional[int],
    fill_bay: Optional[int],
) -> int:
    b = 0
    if drain_bay is not None and _stack_bay(problem, src_i) == drain_bay:
        b += 2
    if fill_bay is not None and _stack_bay(problem, dest_i) == fill_bay:
        b += 1
    return b


def _enumerate_cross(
    stacks: List[List[int]], src_i: int, W: int, H: int
) -> List[GreedyMove]:
    st = stacks[src_i - 1]
    if not st:
        return []
    n = int(st[-1])
    j_from = len(st)
    out: List[GreedyMove] = []
    for k in irange(1, W):
        if k == src_i or len(stacks[k - 1]) >= H:
            continue
        l_to = len(stacks[k - 1]) + 1
        out.append(("cross", src_i, j_from, k, l_to, n))
    return out


def _src_stacks(
    stacks: List[List[int]],
    rank_to_seq: dict,
    W: int,
    H: int,
) -> List[int]:
    out: List[int] = []
    for i in irange(1, W):
        st = stacks[i - 1]
        if not st:
            continue
        if _stack_n_future(st, H, rank_to_seq) > 0 or not stack_order_ok_list(
            st, rank_to_seq
        ):
            out.append(i)
    return out


def _pick_best_move(
    stacks: List[List[int]],
    problem,
    W: int,
    H: int,
    n_moves: int,
    n_now: int,
    g_limit: int,
    seen: set,
    rng: random.Random,
    *,
    drain_bay: Optional[int],
    fill_bay: Optional[int],
) -> Optional[GreedyMove]:
    """
    硬约束（唯一）：搬箱后 ``G = 翻箱次数 + N(Future) ≤ floor(1.5 × N(0))``。
    不要求 G 或 N 相对初值单调减少，只不许超过 1.5 倍初始逆序。

    选步：在合法动作中优先 **N 不升高**，再 **G 尽量小**，tie-break 26→01。
    """
    rank_to_seq = build_rank_to_seq(problem)
    ranked: List[Tuple] = []

    for src_i in _src_stacks(stacks, rank_to_seq, W, H):
        for mv in _enumerate_cross(stacks, src_i, W, H):
            new_st = simulate_move(stacks, mv)
            key = _stacks_state_key(new_st)
            if key in seen:
                continue
            n_after = total_n_future(new_st, W, H, rank_to_seq)
            g_after = g_value(n_moves + 1, n_after)
            if g_after > g_limit:
                continue
            delta_n = n_now - n_after
            if delta_n < 0:
                continue
            tie = _move_tie_bonus_cross(
                problem, mv[1], mv[3], drain_bay=drain_bay, fill_bay=fill_bay
            )
            ranked.append((delta_n, g_after, tie, n_after, mv, key))

    if not ranked:
        return None
    ranked.sort(key=lambda t: (-t[0], t[1], -t[2], t[3]))
    best_dn = ranked[0][0]
    tier = [r for r in ranked if r[0] == best_dn]
    return tier[rng.randint(0, len(tier) - 1)][4]


def plan_greedy_until_terminal(
    problem,
    seed: int = 0,
    *,
    max_moves: Optional[int] = None,
    verbose: bool = False,
    drain_bay: Optional[int] = None,
    fill_bay: Optional[int] = None,
    use_bay_bias: bool = True,
    g_ratio: float = 1.5,
) -> List[GreedyMove]:
    """
    仅跨堆/跨贝（禁止同堆内）。硬约束：全程 ``G=翻箱次数+N ≤ floor(g_ratio×N(0))``（默认 1.5），
    **不是**要求 G 或 N 必须比初始值更小。
    """
    rng = random.Random(seed)
    W, H, N = problem.W, problem.H, problem.N
    rank_to_seq = build_rank_to_seq(problem)
    stacks = _stacks_from_problem(problem)
    moves: List[GreedyMove] = []

    n0 = total_n_future(stacks, W, H, rank_to_seq)
    g_limit = g_cap_from_n0(n0, g_ratio)

    if use_bay_bias and getattr(problem, "combined_bays", None):
        drain_bay = drain_bay if drain_bay is not None else getattr(
            problem, "drain_bay", None
        )
        fill_bay = fill_bay if fill_bay is not None else getattr(
            problem, "fill_bay", None
        )
    else:
        drain_bay = None
        fill_bay = None

    if verbose:
        print(
            f"  [贪心] N(0)={n0}  G上限={g_limit} (={g_ratio}×N0，不得超过)  "
            f"仅跨堆 i≠k"
        )
        if drain_bay is not None:
            print(f"  tie-break: 贝{drain_bay:02d}→贝{fill_bay:02d}")

    hard_cap = (
        int(max_moves) if max_moves is not None else max(500, W * H * N * 30)
    )
    seen: set = set()
    seen.add(_stacks_state_key(stacks))

    while True:
        if len(moves) >= hard_cap:
            break
        if _all_stacks_ok(stacks, rank_to_seq, W):
            break

        n_now = total_n_future(stacks, W, H, rank_to_seq)
        g_now = g_value(len(moves), n_now)
        if g_now > g_limit and not moves:
            break

        mv = _pick_best_move(
            stacks,
            problem,
            W,
            H,
            len(moves),
            n_now,
            g_limit,
            seen,
            rng,
            drain_bay=drain_bay,
            fill_bay=fill_bay,
        )
        if mv is None:
            if verbose:
                print(
                    f"  [贪心] 无满足 G≤{g_limit} 的动作，停止  "
                    f"步={len(moves)} N={n_now} G={g_now}"
                )
            break

        apply_move(stacks, mv)
        moves.append(mv)
        seen.add(_stacks_state_key(stacks))

        if verbose and len(moves) % 10 == 0:
            nn = total_n_future(stacks, W, H, rank_to_seq)
            print(
                f"  [贪心] 步={len(moves)} N={nn} G={g_value(len(moves), nn)}/{g_limit}"
            )

    if verbose:
        nn = total_n_future(stacks, W, H, rank_to_seq)
        print(
            f"  [贪心] 结束 步={len(moves)} N={nn} G={g_value(len(moves), nn)}/{g_limit} "
            f"良序={_all_stacks_ok(stacks, rank_to_seq, W)}"
        )
    return moves


def terminal_sort_reached(problem, moves: List[GreedyMove]) -> bool:
    rank_to_seq = build_rank_to_seq(problem)
    stacks = _stacks_from_problem(problem)
    W = problem.W
    for mv in moves:
        apply_move(stacks, mv)
    return _all_stacks_ok(stacks, rank_to_seq, W)


def greedy_plan_stats(problem, moves: List[GreedyMove], g_ratio: float = 1.5) -> dict:
    W, H = problem.W, problem.H
    stacks = _stacks_from_problem(problem)
    rank_to_seq = build_rank_to_seq(problem)
    n_init = total_n_future(stacks, W, H, rank_to_seq)
    g_limit = g_cap_from_n0(n_init, g_ratio)
    for mv in moves:
        apply_move(stacks, mv)
    n_end = total_n_future(stacks, W, H, rank_to_seq)
    m = len(moves)
    return {
        "n_future_init": n_init,
        "n_future_end": n_end,
        "n_future_delta": n_init - n_end,
        "greedy_moves": m,
        "g_cap": g_limit,
        "g_ratio": float(g_ratio),
        "g_init": g_value(0, n_init),
        "g_end": g_value(m, n_end),
        "g_ok": g_value(m, n_end) <= g_limit,
    }
