"""解向量 → 箱位摆放与每时刻翻箱说明。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .input_loader import format_layout_table
from .in_stack_shuffle import is_valid_top_out_move, top_tier_of_stack_at
from .metrics import (
    extract_bvx_from_var_index,
    format_stack_order_report,
    layout_at_time_from_b,
)
from .utils import irange


def _is_yard_reloc_mode(meta: Dict) -> bool:
    return meta.get("mode") in ("yard_reloc", "multi_bay_reloc", "greedy_yard_reloc")


def stack_row_to_char(stack_row: int) -> str:
    return "A" if int(stack_row) == 10 else str(int(stack_row))


def stack_display_name(problem, global_i: int) -> str:
    labels = getattr(problem, "stack_labels", None)
    if labels and 1 <= global_i <= len(labels) and labels[global_i - 1]:
        return labels[global_i - 1]
    return f"堆{stack_row_to_char(global_i)}"


def bay_for_global_stack(problem, global_i: int) -> int:
    sb = getattr(problem, "stack_bay", None)
    if sb and 1 <= global_i <= len(sb):
        return int(sb[global_i - 1])
    return int(getattr(problem, "bay_no", None) or 1)


def local_stack_row_for_global(problem, global_i: int) -> int:
    labels = getattr(problem, "stack_labels", None)
    if labels and 1 <= global_i <= len(labels) and labels[global_i - 1]:
        tail = labels[global_i - 1].split("-", 1)[1]
        return 10 if tail == "A" else int(tail)
    return int(global_i) if int(global_i) <= 10 else int(global_i) - 10


def format_position_code(zone: str, bay: int, stack_row: int, tier: int) -> str:
    return f"{zone.upper()}{int(bay):02d}{stack_row_to_char(stack_row)}{int(tier)}"


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


def layout_rank_to_seq(
    layout_rank: List[List[Optional[int]]],
    rank_to_seq: Dict[int, int],
) -> List[List[Optional[int]]]:
    out: List[List[Optional[int]]] = []
    for row in layout_rank:
        out.append([rank_to_seq.get(int(v), v) if v is not None else None for v in row])
    return out


def format_layout_grid(
    grid: List[List[Optional[int]]],
    W: int,
    H: int,
    *,
    empty: str = ".",
) -> str:
    lines = ["堆\\层"] + [str(j) for j in irange(1, H)]
    rows = ["\t".join(lines)]
    for i in irange(1, W):
        cells = [stack_row_to_char(i) if i == 10 else str(i)]
        for j in irange(1, H):
            v = grid[i - 1][j - 1]
            cells.append(str(v) if v is not None else empty)
        rows.append("\t".join(cells))
    return "\n".join(rows)


def format_placement_lines(
    grid_seq: List[List[Optional[int]]],
    zone: str,
    bay: int,
    W: int,
    H: int,
) -> List[str]:
    lines: List[str] = []
    for i in irange(1, W):
        for j in irange(1, H):
            sid = grid_seq[i - 1][j - 1]
            if sid is None:
                continue
            pos = format_position_code(zone, bay, i, j)
            lines.append(f"{pos} {sid}")
    return lines


def _stack_snapshot_lines(
    grid_seq: List[List[Optional[int]]],
    stack_i: int,
    H: int,
    zone: str,
    bay: int,
) -> List[str]:
    lines: List[str] = []
    row = grid_seq[stack_i - 1]
    for j in irange(1, H):
        sid = row[j - 1]
        if sid is None:
            continue
        pos = format_position_code(zone, bay, stack_i, j)
        lines.append(f"      层{j}  {pos}  关序号={sid}")
    if not lines:
        lines.append("      (空堆)")
    return lines


def _collect_reloc_moves(
    x_vals: Dict[Tuple[int, int, int, int, int, int], int],
    T_qubo: int,
) -> Dict[int, List[Tuple[int, int, int, int, int, int]]]:
    by_t: Dict[int, List[Tuple[int, int, int, int, int, int]]] = {}
    for i, j, k, l, n, t in sorted(x_vals.keys()):
        if int(t) > T_qubo or i == k or (i, j) == (k, l):
            continue
        if x_vals.get((i, j, k, l, n, t), 0) != 1:
            continue
        by_t.setdefault(int(t), []).append((int(i), int(j), int(k), int(l), int(n)))
    return by_t


def _describe_move(
    i_from: int,
    j_from: int,
    i_to: int,
    j_to: int,
    sid: int,
    pos_from: str,
    pos_to: str,
    *,
    valid_top: bool = True,
) -> str:
    si = stack_row_to_char(i_from)
    sk = stack_row_to_char(i_to)
    if not valid_top:
        return (
            f"【不可行】非顶箱移出：关序号 {sid} 堆{si}层{j_from}({pos_from})"
            f" → 堆{sk}层{j_to}({pos_to})"
        )
    if i_from == i_to:
        return (
            f"【禁止】同堆翻箱不可行：关序号 {sid} 堆{si} 层{j_from}→层{j_to}"
        )
    return (
        f"① 从堆{si} 层{j_from}（{pos_from}）提起顶箱（关序号 {sid}）"
        f" → ② 放到堆{sk} 层{j_to}（{pos_to}）"
    )


def format_shuffle_step_trace(
    problem,
    best: np.ndarray,
    var_index: Dict[Tuple, int],
    meta: Dict,
    *,
    zone: str = "A1",
    bay: int = 1,
    max_detail_t: int = 200,
) -> List[str]:
    W, H, N = problem.W, problem.H, problem.N
    T_qubo = int(meta.get("T_qubo", problem.T))
    rank_to_seq = build_rank_to_seq(problem)

    b_vals, x_vals = extract_bvx_from_var_index(best, var_index)
    x_vals = {k: v for k, v in x_vals.items() if (k[0], k[1]) != (k[2], k[3])}
    yard_mode = _is_yard_reloc_mode(meta)

    layouts_seq: Dict[int, List[List[Optional[int]]]] = {}
    t_max = min(T_qubo, max_detail_t)
    for t in irange(1, t_max + 1):
        layout_rank = layout_at_time_from_b(problem.bay.pri, b_vals, W, H, N, t)
        layouts_seq[t] = layout_rank_to_seq(layout_rank, rank_to_seq)

    moves_by_t = _collect_reloc_moves(x_vals, T_qubo)

    lines: List[str] = [
        "=" * 60,
        "【每时刻翻箱说明】",
        "=" * 60,
        "阅读方式：",
        "  · 下面按模型时刻 t=1,2,… 依次说明；在时刻 t 翻箱后，堆场进入 t+1。",
    ]
    if yard_mode:
        lines.append(
            "  · 仅允许跨堆翻箱（堆 i → 堆 k，i≠k）；禁止在同堆内调整层位。"
        )
        lines.append("  · 不提箱出场；每步从源堆顶箱提起，放到目标堆位。")
    else:
        lines.append("  · 仅在同堆内移动，不提箱、不跨堆。")
        lines.append("  · 每步只能移出源堆顶箱（提起），再放入本堆更低层位。")
    lines.extend(
        [
            "  · 目标：各堆下层关序号 > 上层关序号。",
            "",
            "【时刻索引】",
        ]
    )

    for t in irange(1, T_qubo):
        lst = moves_by_t.get(t, [])
        if not lst:
            lines.append(f"  t={t}  →  t={t + 1}   无翻箱（idle，堆场不变）")
        else:
            parts = []
            for i, j_from, k, j_to, n in lst:
                sid = rank_to_seq.get(n, n)
                ok = is_valid_top_out_move(b_vals, i, j_from, n, t, H, N)
                tag = "顶箱" if ok else "非顶箱?"
                if i == k:
                    parts.append(
                        f"{stack_display_name(problem, i)} {tag}关序号{sid} "
                        f"提起层{j_from}→插入层{j_to}"
                    )
                else:
                    parts.append(
                        f"{stack_display_name(problem, i)}→{stack_display_name(problem, k)} "
                        f"{tag}关序号{sid} 层{j_from}→层{j_to}"
                    )
            lines.append(f"  t={t}  →  t={t + 1}   " + "；".join(parts))
    lines.append(f"  t={T_qubo}           终态（不再翻箱）")
    lines.append("")
    lines.append("-" * 60)
    lines.append("")

    step_no = 0
    for t in irange(1, T_qubo):
        lines.append("=" * 60)
        lines.append(f"时刻 t = {t}")
        if t < T_qubo:
            lines.append(f"  （本时刻结束时堆场进入 t = {t + 1}）")
        else:
            lines.append("  （终态，流程结束）")
        lines.append("=" * 60)

        lst = moves_by_t.get(t, []) if t < T_qubo else []

        if t >= T_qubo:
            lines.append("  无翻箱操作。")
            lines.append("")
            lines.append(f"  【全场堆场】t={t} 关序号网格:")
            if t in layouts_seq:
                lines.append(format_layout_grid(layouts_seq[t], W, H))
            lines.append("")
            continue

        if not lst:
            lines.append("  本时刻操作: 无（idle）")
            lines.append("  说明: 本步未移动任何箱子，全场布局与上一时刻相同。")
            lines.append("")
            if t in layouts_seq:
                lines.append(f"  【全场堆场】t={t}（与 t={t+1} 相同）:")
                lines.append(format_layout_grid(layouts_seq[t], W, H))
            lines.append("")
            continue

        for i, j_from, k, j_to, n in lst:
            step_no += 1
            sid = int(rank_to_seq.get(n, n))
            sl = stack_display_name(problem, i)
            sk = stack_display_name(problem, k)
            pos_from = format_position_code(
                zone,
                bay_for_global_stack(problem, i),
                local_stack_row_for_global(problem, i),
                j_from,
            )
            pos_to = format_position_code(
                zone,
                bay_for_global_stack(problem, k),
                local_stack_row_for_global(problem, k),
                j_to,
            )
            ok = is_valid_top_out_move(b_vals, i, j_from, n, t, H, N)
            top_j = top_tier_of_stack_at(b_vals, i, t, H, N)
            desc = _describe_move(
                i, j_from, k, j_to, sid, pos_from, pos_to, valid_top=ok
            )

            lines.append(f"  --- 翻箱动作 #{step_no}（在时刻 t={t} 执行）---")
            if top_j is not None:
                lines.append(f"  源堆{sl} 翻箱前顶箱层位: 第 {top_j} 层")
            if i == k:
                lines.append(f"  范围: 第 {sl} 堆内")
            else:
                lines.append(f"  范围: 堆{sl} → 堆{sk}（跨堆）")
            lines.append(f"  箱号: 关序号 {sid}  (出库编号 n={n})")
            lines.append(f"  操作: {desc}")
            lines.append("")

            before = layouts_seq.get(t)
            after = layouts_seq.get(t + 1)
            if before is not None:
                lines.append(f"  【翻箱前】t={t} — 源堆 {sl}（自下而上）:")
                lines.extend(_stack_snapshot_lines(before, i, H, zone, bay))
                lines.append("")
            if after is not None and i != k:
                lines.append(f"  【翻箱后】t={t + 1} — 目标堆 {sk}:")
                lines.extend(_stack_snapshot_lines(after, k, H, zone, bay))
                lines.append("")
            elif after is not None:
                lines.append(f"  【翻箱后】t={t + 1} — 第 {sl} 堆:")
                lines.extend(_stack_snapshot_lines(after, i, H, zone, bay))
                lines.append("")

        if t + 1 in layouts_seq and t + 1 < T_qubo:
            lines.append(f"  【本步结束后全场】t={t + 1} 关序号网格:")
            lines.append(format_layout_grid(layouts_seq[t + 1], W, H))
            lines.append("")

    if T_qubo in layouts_seq:
        lines.append("=" * 60)
        lines.append(f"【最终堆场】时刻 t = {T_qubo}")
        lines.append("=" * 60)
        lines.append(format_layout_grid(layouts_seq[T_qubo], W, H))
        lines.append("")

    return lines


def format_solution_placement_report(
    problem,
    best: np.ndarray,
    var_index: Dict[Tuple, int],
    meta: Dict,
    *,
    zone: str = "A1",
    bay: int = 1,
    include_retrieval: bool = True,
) -> List[str]:
    W, H, N = problem.W, problem.H, problem.N
    T_qubo = int(meta.get("T_qubo", problem.T))
    rank_to_seq = build_rank_to_seq(problem)

    b_vals, x_vals = extract_bvx_from_var_index(best, var_index)
    x_vals = {k: v for k, v in x_vals.items() if (k[0], k[1]) != (k[2], k[3])}
    yard_reloc = _is_yard_reloc_mode(meta)

    mode = meta.get("mode", "brp")
    combined = getattr(problem, "combined_bays", None)
    bays_txt = (
        "+".join(f"{int(b):02d}" for b in combined) if combined else f"{bay:02d}"
    )
    lines: List[str] = [
        "--- 箱位摆放 ---",
        f"箱区={zone} 贝位={bays_txt}  T_qubo={T_qubo}  模式={mode}",
        "",
        "【初始】关序号 (来自 input):",
    ]
    if combined:
        from .multi_bay_loader import format_multi_bay_layout

        lines.extend(format_multi_bay_layout(problem).splitlines())
    else:
        lines.append(format_layout_table(problem))
    lines.append("")

    if not include_retrieval:
        lines.extend(
            format_shuffle_step_trace(
                problem,
                best,
                var_index,
                meta,
                zone=zone,
                bay=bay,
            )
        )

    for label, t in [("末态", T_qubo), ("t=1", 1)]:
        if t > T_qubo:
            continue
        layout_rank = layout_at_time_from_b(problem.bay.pri, b_vals, W, H, N, t)
        layout_seq = layout_rank_to_seq(layout_rank, rank_to_seq)
        lines.append(f"【{label}】t={t} 关序号网格 (堆\\层):")
        if combined:
            from .multi_bay_loader import format_multi_bay_grid

            lines.extend(format_multi_bay_grid(layout_seq, problem).splitlines())
        else:
            lines.append(format_layout_grid(layout_seq, W, H))
        lines.append("")
        lines.append(f"【{label}】t={t} 非空格列表 (位置码 关序号):")
        if combined and problem.stack_bay:
            for gi in irange(1, W):
                bno = bay_for_global_stack(problem, gi)
                local_row = local_stack_row_for_global(problem, gi)
                row = layout_seq[gi - 1]
                for j in irange(1, H):
                    sid = row[j - 1]
                    if sid is None:
                        continue
                    pos = format_position_code(zone, bno, local_row, j)
                    lines.append(f"{pos} {sid}")
        else:
            lines.extend(format_placement_lines(layout_seq, zone, bay, W, H))
        lines.append("")
        if label == "末态":
            lines.extend(format_stack_order_report(layout_seq, W, H))
            lines.append("")

    if not include_retrieval:
        if yard_reloc:
            lines.append("【操作】仅跨堆翻箱（禁止同堆 i=k），无提箱 (v/y 未使用)")
        else:
            lines.append("【操作】仅同堆翻箱，无提箱 (v/y 未使用)")

    if x_vals and not include_retrieval:
        title = "【翻箱 x 编码索引】" if yard_reloc else "【同堆翻箱 x 编码索引】"
        lines.append(title + "（与上文「每时刻翻箱说明」一致）")
        for i, j, k, l, n, t in sorted(x_vals.keys()):
            if i == k or (i, j) == (k, l) or t > T_qubo:
                continue
            sid = rank_to_seq.get(n, n)
            if i == k:
                lines.append(
                    f"  t={t}  {stack_display_name(problem, i)} 层{j}→层{l} "
                    f"关序号={sid} (n={n})"
                )
            else:
                lines.append(
                    f"  t={t}  {stack_display_name(problem, i)}层{j}"
                    f"→{stack_display_name(problem, k)}层{l} "
                    f"关序号={sid} (n={n})"
                )
        lines.append("")
    return lines
