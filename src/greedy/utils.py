"""工具函数（包内自包含）。"""

from __future__ import annotations


def irange(start: int, end: int) -> range:
    """闭区间 [start, end]。"""
    return range(start, end + 1)
