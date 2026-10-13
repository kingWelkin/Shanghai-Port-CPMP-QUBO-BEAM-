"""多贝位合并：如 A1_01 + A1_26 合成一场，允许跨贝位搬顶箱。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

from .input_loader import (
    H_FIXED,
    ROW_A,
    _retrieval_rank_from_seq_ids,
    parse_position_code,
)
from .problem_types import BayLike, CompatProblem

Placement = Tuple[str, int, int, int, int]  # zone, bay, stack_row, tier, seq_id
STACKS_PER_BAY = ROW_A  # 10


def _read_placements(txt_path: Path) -> List[Placement]:
    lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: List[Placement] = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        zone, bay, stack_row, tier = parse_position_code(parts[0])
        out.append((zone, bay, stack_row, tier, int(parts[1])))
    if not out:
        raise ValueError(f"无有效数据行: {txt_path}")
    return out


def _bay_suffix_to_int(s: str) -> int:
    return int(s)


def stack_label(bay_no: int, local_row: int) -> str:
    ch = "A" if local_row == STACKS_PER_BAY else str(local_row)
    return f"{bay_no:02d}-{ch}"


def global_stack_index(bay_slot: int, local_row: int) -> int:
    """bay_slot: 0..B-1；返回 1..W 全局堆号。"""
    return bay_slot * STACKS_PER_BAY + local_row


def load_combined_zone_bays(
    input_dir: Path,
    zone: str,
    bay_suffixes: Sequence[str],
) -> CompatProblem:
    """
    将同一箱区多个贝位（如 01 与 26）合并为一个 problem。

    - W = 10 * 贝位数；堆 1..10 属第一贝，11..20 属第二贝；
    - 任意两堆 i≠k 可搬顶箱（含 01 贝 ↔ 26 贝）；
    - 全场共用一套关序号秩（按 seq_id 统一排序）。
    """
    input_dir = Path(input_dir)
    zone = zone.strip().upper()
    suffixes = [str(b).strip() for b in bay_suffixes]
    paths = [input_dir / f"{zone}_{s}_input.txt" for s in suffixes]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"缺少输入文件: {p}")

    placements: List[Placement] = []
    for p in paths:
        placements.extend(_read_placements(p))

    bay_nums = [_bay_suffix_to_int(s) for s in suffixes]
    bay_slot = {b: i for i, b in enumerate(bay_nums)}

    seq_ids = [p[4] for p in placements]
    rank_map = _retrieval_rank_from_seq_ids(seq_ids)
    H = H_FIXED
    W = STACKS_PER_BAY * len(suffixes)
    N = len(rank_map)

    pri: List[List[int | None]] = [[None for _ in range(H)] for _ in range(W)]
    seq: List[List[int | None]] = [[None for _ in range(H)] for _ in range(W)]
    stack_bay: List[int] = [0] * W
    stack_labels: List[str] = [""] * W

    for b in bay_nums:
        slot = bay_slot[b]
        for local in range(1, STACKS_PER_BAY + 1):
            gi = global_stack_index(slot, local) - 1
            stack_bay[gi] = b
            stack_labels[gi] = stack_label(b, local)

    for zone_p, bay, stack_row, tier, seq_id in placements:
        if zone_p.upper() != zone:
            raise ValueError(f"箱区不一致: 期望 {zone} 收到 {zone_p}")
        if bay not in bay_slot:
            raise ValueError(f"未在合并列表中的贝位: {bay}")
        gi = global_stack_index(bay_slot[bay], stack_row) - 1
        pri[gi][tier - 1] = rank_map[seq_id]
        seq[gi][tier - 1] = seq_id

    prob = CompatProblem(
        W=W,
        H=H,
        N=N,
        T=N + 2 * N,
        bay=BayLike(pri=pri, seq=seq),
        zone=zone,
        bay_no=None,
        combined_bays=bay_nums,
        bay_suffixes=suffixes,
        source_files=[str(p) for p in paths],
        stack_bay=stack_bay,
        stack_labels=stack_labels,
    )
    _attach_bay_n0_hints(prob)
    return prob


def compute_bay_n0_initial(problem: CompatProblem) -> dict[int, int]:
    from .in_stack_shuffle import build_rank_to_seq
    from .metrics import n_future_single_stack, stack_boxes_from_pri_row

    bays = getattr(problem, "combined_bays", None) or []
    if not bays or not problem.stack_bay:
        return {}
    rank_to_seq = build_rank_to_seq(problem)
    scores: dict[int, int] = {int(b): 0 for b in bays}
    for i in range(problem.W):
        row = problem.bay.pri[i]
        boxes = stack_boxes_from_pri_row(row, problem.H)
        b = int(problem.stack_bay[i])
        scores[b] = scores.get(b, 0) + n_future_single_stack(
            boxes, problem.H, rank_to_seq=rank_to_seq
        )
    return scores


def _attach_bay_n0_hints(problem: CompatProblem) -> None:
    scores = compute_bay_n0_initial(problem)
    if not scores:
        return
    problem.bay_n0_initial = scores
    problem.drain_bay = max(scores, key=lambda b: scores[b])
    problem.fill_bay = min(scores, key=lambda b: scores[b])


def discover_paired_cases(
    input_dir: Path,
    bay_suffixes: Sequence[str] = ("01", "26"),
) -> List[Tuple[str, List[str], List[Path]]]:
    """扫描 input，找出同时存在 {zone}_01 与 {zone}_26 的箱区。"""
    input_dir = Path(input_dir)
    suffix_set = {str(s).strip() for s in bay_suffixes}
    zones: set[str] = set()
    for p in input_dir.glob("*_input.txt"):
        name = p.stem
        if name.endswith("_input"):
            name = name[: -len("_input")]
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in suffix_set:
            zones.add(parts[0])
    cases: List[Tuple[str, List[str], List[Path]]] = []
    suffix_list = [str(s).strip() for s in bay_suffixes]
    for z in sorted(zones):
        paths = [input_dir / f"{z}_{s}_input.txt" for s in suffix_list]
        if all(p.is_file() for p in paths):
            cases.append((z, suffix_list, paths))
    return cases


def format_multi_bay_layout(problem: CompatProblem) -> str:
    grid = problem.bay.seq if problem.bay.seq is not None else problem.bay.pri
    bays = getattr(problem, "combined_bays", None) or [getattr(problem, "bay_no", 1)]
    lines: List[str] = []
    bay_slot = {b: i for i, b in enumerate(bays)}
    for bay_no in bays:
        lines.append(f"--- 贝位 {int(bay_no):02d} ---")
        header = ["堆\\层"] + [str(j) for j in range(1, problem.H + 1)]
        lines.append("\t".join(header))
        slot = bay_slot[bay_no]
        for local in range(1, STACKS_PER_BAY + 1):
            gi = global_stack_index(slot, local) - 1
            ch = "A" if local == STACKS_PER_BAY else str(local)
            cells = [ch]
            for j in range(problem.H):
                v = grid[gi][j]
                cells.append(str(v) if v is not None else ".")
            lines.append("\t".join(cells))
        lines.append("")
    return "\n".join(lines).rstrip()


def format_multi_bay_grid(
    grid: List[List[int | None]],
    problem: CompatProblem,
) -> str:
    lines: List[str] = []
    for bay_no in problem.combined_bays or []:
        lines.append(f"--- 贝位 {int(bay_no):02d} ---")
        header = ["堆\\层"] + [str(j) for j in range(1, problem.H + 1)]
        lines.append("\t".join(header))
        for idx, b in enumerate(problem.stack_bay or []):
            if b != bay_no:
                continue
            cells = [problem.stack_labels[idx].split("-", 1)[1]]
            for j in range(problem.H):
                v = grid[idx][j]
                cells.append(str(v) if v is not None else ".")
            lines.append("\t".join(cells))
        lines.append("")
    return "\n".join(lines).rstrip()
