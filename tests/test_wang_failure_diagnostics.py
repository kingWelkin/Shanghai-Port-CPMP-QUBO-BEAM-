import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_case_selector_accepts_exact_open_case_ids():
    from diagnose_wang_qubo_failures import DEFAULT_CASES, _case_selected

    assert _case_selected("5x5_cv1_test_4", {"5x5_cv1_test_4"})
    assert not _case_selected("5x5_cv1_test_5", {"5x5_cv1_test_4"})
    assert _case_selected("5x5_cv1_test_6", set())
    assert DEFAULT_CASES == ("5x5_cv1_test_4", "5x5_cv2_test_10")


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


def test_main_uses_real_open_case_flow_without_network_clone(monkeypatch, tmp_path):
    import diagnose_wang_qubo_failures as diag
    from container_reshuffle.core import Instance, Move, ProblemVariant, Solution

    calls = {}
    source = tmp_path / "DLTS"
    output = tmp_path / "diag.csv"
    instance = Instance(
        instance_id="tiny",
        variant=ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=3,
        max_height=3,
        container_count=3,
        stacks=[[1], [4], [3]],
    )
    compact_move = Move(step=0, box=3, from_stack=2, to_stack=0, from_level=1, to_level=2)
    wang_move = Move(step=0, box=3, from_stack=2, to_stack=0, from_level=1, to_level=2)
    compact_args = SimpleNamespace()
    rows = []

    def fake_ensure_dlts_source(path, *, auto_clone=True, refresh=True):
        calls["ensure"] = (path, auto_clone, refresh)
        return source

    def fake_discover_dlts_cases(path, *, families, limit_per_family, max_total):
        calls["discover"] = (path, families, limit_per_family, max_total)
        return [
            SimpleNamespace(
                family="tiny_family",
                case_id="tiny_family_1",
                instance_path=tmp_path / "1.dat",
                reference=SimpleNamespace(max_height=3),
            ),
            SimpleNamespace(
                family="tiny_family",
                case_id="tiny_family_2",
                instance_path=tmp_path / "2.dat",
                reference=SimpleNamespace(max_height=3),
            ),
        ]

    def fake_load_dlts_dat(path, *, max_height):
        calls.setdefault("loaded", []).append((path, max_height))
        return instance

    def fake_make_compact_args(**kwargs):
        calls["compact_kwargs"] = kwargs
        return compact_args

    def fake_run_compact_beam_instance(case_id, loaded_instance, args):
        calls["compact_run"] = (case_id, loaded_instance, args)
        args._best_n_history = [1, 0]
        args._layer_parent_top_share = 0.5
        args._stack_count = loaded_instance.stack_count
        args._max_height = loaded_instance.max_height
        return SimpleNamespace(status="FEASIBLE", moves=[compact_move])

    class FakeWangSolver:
        def solve(self, loaded_instance, config):
            calls["wang"] = (loaded_instance, config)
            return Solution(
                solver_name="wang_bs_b",
                feasible=True,
                optimal=False,
                objective=1,
                runtime_sec=0.0,
                moves=[wang_move],
                status="FEASIBLE",
            )

    def fake_write_rows(emitted_rows, emitted_output):
        rows.extend(emitted_rows)
        calls["output"] = emitted_output

    monkeypatch.setattr(diag, "ensure_dlts_source", fake_ensure_dlts_source)
    monkeypatch.setattr(diag, "discover_dlts_cases", fake_discover_dlts_cases)
    monkeypatch.setattr(diag, "load_dlts_dat", fake_load_dlts_dat)
    monkeypatch.setattr(diag, "make_compact_args", fake_make_compact_args)
    monkeypatch.setattr(diag, "run_compact_beam_instance", fake_run_compact_beam_instance)
    monkeypatch.setattr(diag, "WangBsBabyMovesSolver", FakeWangSolver)
    monkeypatch.setattr(diag, "_write_rows", fake_write_rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose",
            "--source",
            str(source),
            "--families",
            "tiny_family",
            "--cases",
            "tiny_family_1",
            "--compact-solver-profile",
            "scheduler_state_v5_wang_rescue_polish",
            "--time-limit",
            "1",
            "--output",
            str(output),
        ],
    )

    assert diag.main() == 0

    assert calls["ensure"] == (source, False, False)
    assert calls["discover"] == (source, ["tiny_family"], None, None)
    assert [item[0].name for item in calls["loaded"]] == ["1.dat"]
    assert calls["compact_kwargs"]["solver_profile"] == "scheduler_state_v5_wang_rescue_polish"
    assert calls["compact_run"] == ("tiny_family_1", instance, compact_args)
    assert calls["wang"][0] is instance
    assert calls["wang"][1].max_candidates_per_layer == 64
    assert calls["output"] == output
    assert len(rows) == 1
    assert rows[0]["case_id"] == "tiny_family_1"
    assert rows[0]["fixed_prefix"] != 0
    assert rows[0]["fix_pair"] != 0
    assert rows[0]["compact_final_n"] == 1
