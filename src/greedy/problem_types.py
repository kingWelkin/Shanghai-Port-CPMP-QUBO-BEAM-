"""问题数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BayLike:
    pri: List[List[int | None]]
    seq: Optional[List[List[int | None]]] = None


@dataclass
class CompatProblem:
    W: int
    H: int
    N: int
    T: int
    bay: BayLike
    zone: Optional[str] = None
    bay_no: Optional[int] = None
    # 多贝位合并（greedy multi_bay）
    combined_bays: Optional[List[int]] = None
    bay_suffixes: Optional[List[str]] = None
    source_files: Optional[List[str]] = None
    stack_bay: Optional[List[int]] = None
    stack_labels: Optional[List[str]] = None
