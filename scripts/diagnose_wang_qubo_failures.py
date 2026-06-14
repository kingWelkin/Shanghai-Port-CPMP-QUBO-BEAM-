"""Focused diagnostics for Wang-vs-QUBO open CPMP failure cases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

DEFAULT_CASES = ("5x5_cv1_test_4", "5x5_cv1_test_6")
FIELDS = [
    "case_id",
    "family",
    "compact_status",
    "compact_final_n",
    "compact_moves",
    "compact_plateau_tail",
    "fixed_prefix",
    "fix_pair",
    "parent_top_share",
    "wang_moves",
    "wang_first_moves",
]


def _case_selected(case_id: str, exact_cases: set[str]) -> bool:
    return not exact_cases or case_id in exact_cases


def _plateau_tail(history: Iterable[int], width: int = 3) -> str:
    values = list(history)[-max(1, width):]
    return ",".join(str(value) for value in values)


def _diagnostic_row(
    *,
    case_id: str,
    family: str,
    compact_status: str,
    compact_final_n: int | None,
    compact_moves: int | None,
    compact_best_n_history: list[int],
    fixed_prefix: int,
    fix_pair: int,
    parent_top_share: float,
    wang_moves: int | None,
    wang_first_moves: list[str],
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "family": family,
        "compact_status": compact_status,
        "compact_final_n": "" if compact_final_n is None else compact_final_n,
        "compact_moves": "" if compact_moves is None else compact_moves,
        "compact_plateau_tail": _plateau_tail(compact_best_n_history),
        "fixed_prefix": fixed_prefix,
        "fix_pair": fix_pair,
        "parent_top_share": f"{parent_top_share:.6f}",
        "wang_moves": "" if wang_moves is None else wang_moves,
        "wang_first_moves": ";".join(wang_first_moves),
    }


def _write_rows(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/wang_qubo_failure_autopsy.csv"),
    )
    args = parser.parse_args()
    exact = set(args.cases)
    rows = [
        _diagnostic_row(
            case_id=case_id,
            family=case_id.rsplit("_", 1)[0],
            compact_status="diagnostic_not_run",
            compact_final_n=None,
            compact_moves=None,
            compact_best_n_history=[],
            fixed_prefix=0,
            fix_pair=0,
            parent_top_share=0.0,
            wang_moves=None,
            wang_first_moves=[],
        )
        for case_id in DEFAULT_CASES
        if _case_selected(case_id, exact)
    ]
    _write_rows(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
