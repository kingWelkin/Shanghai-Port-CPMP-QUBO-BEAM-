import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_case_selector_accepts_exact_open_case_ids():
    from diagnose_wang_qubo_failures import _case_selected

    assert _case_selected("5x5_cv1_test_4", {"5x5_cv1_test_4"})
    assert not _case_selected("5x5_cv1_test_5", {"5x5_cv1_test_4"})
    assert _case_selected("5x5_cv1_test_6", set())


def test_diagnostic_row_degrades_when_wang_data_is_missing():
    from diagnose_wang_qubo_failures import _diagnostic_row

    row = _diagnostic_row(
        case_id="5x5_cv1_test_4",
        family="5x5_cv1_test",
        compact_status="NO_FEASIBLE_FOUND",
        compact_final_n=7,
        compact_moves=None,
        compact_best_n_history=[12, 10, 10, 10, 9],
        fixed_prefix=4,
        fix_pair=2,
        parent_top_share=0.42,
        wang_moves=None,
        wang_first_moves=[],
    )

    assert row["case_id"] == "5x5_cv1_test_4"
    assert row["compact_plateau_tail"] == "10,10,9"
    assert row["wang_moves"] == ""
    assert row["wang_first_moves"] == ""


def test_format_first_moves_keeps_stack_and_box_fields():
    from container_reshuffle.core import Move
    from diagnose_wang_qubo_failures import _format_first_moves

    moves = [
        Move(step=1, box=7, from_stack=0, to_stack=2, from_level=3, to_level=1),
        Move(step=2, box=5, from_stack=1, to_stack=0, from_level=2, to_level=4),
    ]

    assert _format_first_moves(moves, limit=2) == ["1:7:0->2", "2:5:1->0"]
