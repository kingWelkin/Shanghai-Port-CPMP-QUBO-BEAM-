"""读取 data/input/*_input.txt。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from .problem_types import BayLike, CompatProblem
from .utils import irange

H_FIXED = 6
ROW_A = 10


def parse_position_code(code: str) -> Tuple[str, int, int, int]:
    code = code.strip().upper()
    if len(code) != 6:
        raise ValueError(f"位置编码须为 6 位，收到 {code!r}")
    zone = code[0:2]
    bay = int(code[2:4])
    stack_ch = code[4]
    tier = int(code[5])
    if tier < 1 or tier > H_FIXED:
        raise ValueError(f"层号须在 1..{H_FIXED}: {code!r}")
    if stack_ch == "A":
        stack_row = ROW_A
    elif stack_ch.isdigit():
        stack_row = int(stack_ch)
        if stack_row < 1 or stack_row > 9:
            raise ValueError(f"堆号须在 1..9 或 A(=10): {code!r}")
    else:
        raise ValueError(f"无法解析堆号(第5位): {code!r} in {code!r}")
    return zone, bay, stack_row, tier


def _retrieval_rank_from_seq_ids(seq_ids: List[int]) -> Dict[int, int]:
    uniq = sorted(set(seq_ids))
    return {c: i + 1 for i, c in enumerate(uniq)}


def load_problem_from_input_txt(txt_path: Path) -> CompatProblem:
    txt_path = Path(txt_path)
    lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    placements: List[Tuple[str, int, int, int, int]] = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        zone, bay, stack_row, tier = parse_position_code(parts[0])
        seq_id = int(parts[1])
        placements.append((zone, bay, stack_row, tier, seq_id))
    if not placements:
        raise ValueError(f"无有效数据行: {txt_path}")
    W = max(p[2] for p in placements)
    H = H_FIXED
    rank_map = _retrieval_rank_from_seq_ids([p[4] for p in placements])
    N = len(rank_map)
    pri: List[List[int | None]] = [[None for _ in range(H)] for _ in range(W)]
    seq: List[List[int | None]] = [[None for _ in range(H)] for _ in range(W)]
    for _zone, _bay, stack_row, tier, seq_id in placements:
        pri[stack_row - 1][tier - 1] = rank_map[seq_id]
        seq[stack_row - 1][tier - 1] = seq_id
    zone0, bay0 = placements[0][0], placements[0][1]
    return CompatProblem(
        W=W,
        H=H,
        N=N,
        T=N + 2 * N,
        bay=BayLike(pri=pri, seq=seq),
        zone=zone0,
        bay_no=bay0,
    )


def format_layout_table(problem: CompatProblem) -> str:
    grid = problem.bay.seq if problem.bay.seq is not None else problem.bay.pri
    header = ["堆\\层"] + [str(j) for j in range(1, problem.H + 1)]
    rows = ["\t".join(header)]
    for i in range(1, problem.W + 1):
        cells = [str(i)]
        for j in range(1, problem.H + 1):
            v = grid[i - 1][j - 1]
            cells.append(str(v) if v is not None else ".")
        rows.append("\t".join(cells))
    return "\n".join(rows)
