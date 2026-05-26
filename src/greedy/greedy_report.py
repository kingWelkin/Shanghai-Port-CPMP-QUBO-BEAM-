"""由贪心轨迹直接生成报告（仅跨堆/跨贝）。"""

from __future__ import annotations

from typing import Dict, List

from .in_stack_shuffle import _column_grid, build_rank_to_seq
from .input_loader import format_layout_table
from .metrics import (
    format_single_stack_order_line,
    format_stack_order_report,
    stack_boxes_from_pri_row,
)
from .report import layout_rank_to_seq
from .utils import irange

from .greedy_moves import GreedyMove, apply_move
from .greedy_plan import _stacks_from_problem
from .multi_bay_loader import format_multi_bay_grid, format_multi_bay_layout


def stack_name(problem, global_i: int) -> str:
    labels = getattr(problem, "stack_labels", None)
    if labels and 1 <= global_i <= len(labels):
        return labels[global_i - 1]
    return str(global_i)


def format_move_line(problem, move: GreedyMove) -> str:
    if move[0] != "cross":
        raise ValueError(f"非法动作（仅跨堆）: {move}")
    _, i, j_from, k, l_to, n = move
    return (
        f"{stack_name(problem, i)}→{stack_name(problem, k)} "
        f"顶秩{n} 层{j_from}→层{l_to}"
    )


def replay_moves_to_stacks(problem, moves: List[GreedyMove]) -> List[List[int]]:
    stacks = _stacks_from_problem(problem)
    for mv in moves:
        apply_move(stacks, mv)
    return stacks


def format_stack_order_report_multi(problem, grid_seq, W: int, H: int) -> List[str]:
    lines = [
        "【堆叠验收】下层关序号 > 上层；逆序箱数：自第2层向上首个上层>下层，该层及以上计入",
    ]
    all_ok = True
    for i in irange(1, W):
        row = grid_seq[i - 1] if i - 1 < len(grid_seq) else []
        boxes = stack_boxes_from_pri_row(row, H)
        label = stack_name(problem, i)
        line = format_single_stack_order_line(boxes, H, label)
        if not line.endswith(": OK"):
            all_ok = False
        lines.append(line)
    lines.append(f"  全场良序: {'是' if all_ok else '否'}")
    return lines


def format_greedy_placement_report(
    problem,
    moves: List[GreedyMove],
    meta: Dict,
    *,
    zone: str = "A1",
    max_steps: int = 120,
) -> List[str]:
    W, H = problem.W, problem.H
    rank_to_seq = build_rank_to_seq(problem)
    stacks = _stacks_from_problem(problem)
    combined = getattr(problem, "combined_bays", None)
    bays_txt = (
        "+".join(f"{b:02d}" for b in combined) if combined else "单贝"
    )
    g_end = meta.get("g_end", "?")
    g_cap = meta.get("g_cap", "?")
    lines = [
        "--- 箱位摆放 (贪心轨迹) ---",
        f"箱区={zone}  合并贝位={bays_txt}  步数={len(moves)}  "
        f"终态良序={meta.get('terminal_sorted')}",
        f"G=翻箱+N  末态 G={g_end}  须≤{g_cap}(1.5×N0)  合规={meta.get('g_ok')}",
        "说明: 仅跨堆/跨贝(i≠k)；G=翻箱次数+逆序不得超过1.5×初始逆序",
        "",
        "【初始】关序号:",
    ]
    if combined:
        lines.extend(format_multi_bay_layout(problem).splitlines())
    else:
        lines.extend(format_layout_table(problem).splitlines())
    lines.extend(
        [
            "",
            "=" * 60,
            "【每时刻翻箱说明】",
            "=" * 60,
        ]
    )
    n_show = min(len(moves), max_steps)
    if len(moves) > max_steps:
        lines.append(f"  （仅展示前 {max_steps} 步，共 {len(moves)} 步）")
    for step in range(n_show):
        mv = moves[step]
        t = step + 1
        lines.append(f"  t={t} → t={t+1}  {format_move_line(problem, mv)}")
        apply_move(stacks, mv)

    final_rank = _column_grid(replay_moves_to_stacks(problem, moves), H, W)
    final_seq = layout_rank_to_seq(final_rank, rank_to_seq)
    lines.extend(["", f"【末态 t={len(moves)+1}】关序号:"])
    if combined:
        lines.extend(format_multi_bay_grid(final_seq, problem).splitlines())
    else:
        from .report import format_layout_grid

        lines.extend(format_layout_grid(final_seq, W, H).splitlines())
    lines.append("")
    if combined:
        lines.extend(format_stack_order_report_multi(problem, final_seq, W, H))
    else:
        lines.extend(format_stack_order_report(final_seq, W, H))
    return lines
