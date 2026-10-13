"""逆序箱数定义（与业务一致，供报告说明）。"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .metrics import n_future_single_stack
from .utils import irange


def describe_n_future_rule() -> str:
    return (
        "逆序箱数 N(Future)：自第2层起向上(2,3,4,5,6)，"
        "找第一个「上层关序号>下层关序号」的层，"
        "该层及其上方所有箱计入逆序。"
    )


def count_stack_inversions(
    boxes_by_tier: Sequence[Optional[int]],
    H: int,
    rank_to_seq: Optional[Dict[int, int]] = None,
) -> int:
    return n_future_single_stack(boxes_by_tier, H, rank_to_seq=rank_to_seq)


def first_inversion_tier(
    boxes_by_tier: Sequence[Optional[int]],
    H: int,
    rank_to_seq: Optional[Dict[int, int]] = None,
) -> Optional[int]:
    """返回第一个逆序层的层号 j（1-based），无则 None。"""
    from .metrics import _tier_seq_value

    boxes = list(boxes_by_tier)
    for j in irange(2, H):
        lo = _tier_seq_value(boxes[j - 2], rank_to_seq)
        hi = _tier_seq_value(boxes[j - 1], rank_to_seq)
        if lo is None or hi is None:
            continue
        if hi > lo:
            return j
    return None
