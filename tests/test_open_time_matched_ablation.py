from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _row(case: str, variant: str, final_n: int, moves: int | str = "", runtime: float = 1.0, **extra):
    row = {
        "case": case,
        "variant": variant,
        "final_n": str(final_n),
        "moves": str(moves),
        "runtime_sec": str(runtime),
        "move_gap": str(extra.pop("move_gap", "")),
        "matched_width": str(extra.pop("matched_width", "")),
        "post_polish_moves_saved": str(extra.pop("post_polish_moves_saved", 0)),
        "qubo_selected_on_path": str(extra.pop("qubo_selected_on_path", 0)),
        "qubo_ancestor_on_path": str(extra.pop("qubo_ancestor_on_path", 0)),
        "qubo_selected_total": str(extra.pop("qubo_selected_total", 0)),
    }
    row.update(extra)
    return row


def test_best_classic_prefers_solved_then_lower_moves_then_width():
    from open_cpmp_time_matched_ablation import best_classic_by_case

    rows = [
        _row("c1", "classic_tm_w1200", 5, "", matched_width=1200),
        _row("c1", "classic_tm_w2400", 0, 31, matched_width=2400),
        _row("c1", "classic_tm_w4800", 0, 29, matched_width=4800),
        _row("c2", "classic_tm_w1200", 4, "", runtime=0.5, matched_width=1200),
        _row("c2", "classic_tm_w2400", 3, "", runtime=1.0, matched_width=2400),
    ]

    best = best_classic_by_case(rows)

    assert best["c1"]["matched_width"] == "4800"
    assert best["c2"]["final_n"] == "3"


def test_case_category_covers_qubo_classic_win_loss_and_ties():
    from open_cpmp_time_matched_ablation import case_category

    assert case_category(_row("c", "qubo", 0, 20), _row("c", "classic", 2, "")) == "qubo_only"
    assert case_category(_row("c", "qubo", 2, ""), _row("c", "classic", 0, 20)) == "classic_only"
    assert case_category(_row("c", "qubo", 0, 18), _row("c", "classic", 0, 20)) == "both_qubo_fewer"
    assert case_category(_row("c", "qubo", 0, 22), _row("c", "classic", 0, 20)) == "both_classic_fewer"
    assert case_category(_row("c", "qubo", 0, 20), _row("c", "classic", 0, 20)) == "both_tie"
    assert case_category(_row("c", "qubo", 3, ""), _row("c", "classic", 4, "")) == "neither_solved"


def test_contribution_flags_report_frontier_polish_and_adaptive():
    from open_cpmp_time_matched_ablation import contribution_flags

    row = _row(
        "c",
        "qubo_adaptive",
        0,
        20,
        post_polish_moves_saved=2,
        qubo_selected_on_path=1,
        qubo_ancestor_on_path=3,
        qubo_selected_total=20,
    )

    assert contribution_flags(row) == "post_polish;frontier_direct;frontier_ancestor;frontier_selected;adaptive"
    assert contribution_flags(_row("c", "qubo", 0, 20)) == "no_visible_qubo_path_marker"
