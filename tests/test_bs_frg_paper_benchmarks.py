from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_load_bf_bay_parses_bortfeldt_forster_format(tmp_path: Path):
    from benchmark_bs_frg_paper import load_bf_bay

    path = tmp_path / "cpmp_3_4_sample.bay"
    path.write_text(
        "\n".join(
            [
                "Label : cpmp_3_4_sample.bay",
                "Width : 3",
                "Height : 4",
                "Containers : 5",
                "Stack 1 : 5 3",
                "Stack 2 :",
                "Stack 3 : 4 2 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    instance = load_bf_bay(path)

    assert instance.instance_id == "cpmp_3_4_sample"
    assert instance.stack_count == 3
    assert instance.max_height == 4
    assert instance.container_count == 5
    assert instance.stacks == [[5, 3], [], [4, 2, 1]]


def test_load_cv_dat_uses_filename_height_plus_two_for_relocation_buffer(tmp_path: Path):
    from benchmark_bs_frg_paper import load_cv_dat

    path = tmp_path / "data4-3-1.dat"
    path.write_text(
        "\n".join(
            [
                "3 7",
                "2 7 4",
                "3 6 5 3",
                "2 2 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    instance = load_cv_dat(path)

    assert instance.instance_id == "data4-3-1"
    assert instance.stack_count == 3
    assert instance.max_height == 6
    assert instance.container_count == 7
    assert instance.stacks == [[7, 4], [6, 5, 3], [2, 1]]


def test_discover_sampled_paper_cases_caps_bf_families_with_natural_order(tmp_path: Path):
    from benchmark_compact_paper_set import discover_sampled_paper_cases

    bf1 = tmp_path / "BF" / "BF" / "BF1"
    bf2 = tmp_path / "BF" / "BF" / "BF2"
    bf1.mkdir(parents=True)
    bf2.mkdir(parents=True)
    for idx in [1, 10, 2, 11, 3, 12, 4, 5, 6, 7, 8, 9]:
        (bf1 / f"case_{idx}.bay").write_text("", encoding="utf-8")
    for idx in [1, 2, 3]:
        (bf2 / f"case_{idx}.bay").write_text("", encoding="utf-8")

    cases = discover_sampled_paper_cases(tmp_path, datasets=("BF",), sample_per_family=10)

    assert [(case.family, case.case_id) for case in cases[:10]] == [
        ("BF1", f"case_{idx}") for idx in range(1, 11)
    ]
    assert [(case.family, case.case_id) for case in cases[10:]] == [
        ("BF2", "case_1"),
        ("BF2", "case_2"),
        ("BF2", "case_3"),
    ]


def test_discover_sampled_paper_cases_groups_cv_by_height_and_stack(tmp_path: Path):
    from benchmark_compact_paper_set import discover_sampled_paper_cases

    cv_root = tmp_path / "CV" / "CRPTestcases_Caserta"
    cv_root.mkdir(parents=True)
    for idx in [1, 10, 2, 11, 3, 12, 4, 5, 6, 7, 8, 9]:
        (cv_root / f"data4-3-{idx}.dat").write_text("", encoding="utf-8")
    for idx in [1, 2]:
        (cv_root / f"data4-4-{idx}.dat").write_text("", encoding="utf-8")

    cases = discover_sampled_paper_cases(tmp_path, datasets=("CV",), sample_per_family=10)

    assert [(case.family, case.case_id) for case in cases[:10]] == [
        ("4-3", f"data4-3-{idx}") for idx in range(1, 11)
    ]
    assert [(case.family, case.case_id) for case in cases[10:]] == [
        ("4-4", "data4-4-1"),
        ("4-4", "data4-4-2"),
    ]


def test_discover_sampled_paper_cases_filters_selected_families(tmp_path: Path):
    from benchmark_compact_paper_set import discover_sampled_paper_cases

    for family in ("BF14", "BF16"):
        family_root = tmp_path / "BF" / "BF" / family
        family_root.mkdir(parents=True)
        for idx in [1, 10, 2]:
            (family_root / f"case_{idx}.bay").write_text("", encoding="utf-8")

    cv_root = tmp_path / "CV" / "CRPTestcases_Caserta"
    cv_root.mkdir(parents=True)
    for family in ("10-6", "10-10"):
        height, stacks = family.split("-")
        for idx in [1, 10, 2]:
            (cv_root / f"data{height}-{stacks}-{idx}.dat").write_text("", encoding="utf-8")

    cases = discover_sampled_paper_cases(
        tmp_path,
        datasets=("BF", "CV"),
        sample_per_family=2,
        families=("BF16", "10-6"),
    )

    assert [(case.family, case.case_id) for case in cases] == [
        ("BF16", "case_1"),
        ("BF16", "case_2"),
        ("10-6", "data10-6-1"),
        ("10-6", "data10-6-2"),
    ]


def test_paper_exception_row_preserves_failed_case_metadata(tmp_path: Path):
    from argparse import Namespace

    from benchmark_bs_frg_paper import PaperCase
    from benchmark_compact_paper_set import row_from_paper_exception

    case = PaperCase("BF", "BFx", "case_1", tmp_path / "case_1.bay")
    args = Namespace(alpha=10.0)

    row = row_from_paper_exception(case, args, "candidate", RuntimeError("lift failed"))

    assert row["dataset"] == "BF"
    assert row["family"] == "BFx"
    assert row["case"] == "case_1"
    assert row["variant"] == "candidate"
    assert row["goal_validation_ok"] is False
    assert row["business_ok"] is False
    assert row["feasibility_ok"] is False
    assert row["time_limit_ok"] is False
    assert row["primary_ok"] is False
    assert row["status"] == "ERROR"
    assert row["validation_msg"] == "RuntimeError: lift failed"
    assert row["full_quality_penalty"] == 1000000


@pytest.mark.parametrize(
    ("row", "time_limit", "expected"),
    [
        (
            {"moves": 17, "final_n": 0, "goal_validation_ok": True, "business_ok": False, "runtime_sec": 1.0},
            20.0,
            {"feasibility_ok": True, "time_limit_ok": True, "primary_ok": False},
        ),
        (
            {"moves": 17, "final_n": 0, "goal_validation_ok": True, "business_ok": True, "runtime_sec": 20.01},
            20.0,
            {"feasibility_ok": True, "time_limit_ok": False, "primary_ok": False},
        ),
        (
            {"moves": 17, "final_n": 0, "goal_validation_ok": True, "business_ok": True, "runtime_sec": 20.0},
            20.0,
            {"feasibility_ok": True, "time_limit_ok": True, "primary_ok": True},
        ),
        (
            {"moves": 17, "final_n": 0, "goal_validation_ok": True, "business_ok": True, "runtime_sec": "bad"},
            20.0,
            {"feasibility_ok": True, "time_limit_ok": False, "primary_ok": False},
        ),
    ],
)
def test_paper_scorecard_distinguishes_feasibility_cap_and_time(row, time_limit, expected):
    from benchmark_compact_paper_set import _apply_paper_scorecard

    scored = _apply_paper_scorecard(dict(row), time_limit=time_limit)

    assert {key: scored[key] for key in expected} == expected


def test_paper_summary_reports_primary_and_feasibility_counts(tmp_path: Path):
    from argparse import Namespace

    from benchmark_compact_paper_set import _write_csv, _write_summary

    rows = [
        {
            "dataset": "CV",
            "family": "10-6",
            "case": "strict",
            "moves": 9,
            "final_n": 0,
            "runtime_sec": 1.0,
            "business_ok": True,
            "goal_validation_ok": True,
            "feasibility_ok": True,
            "time_limit_ok": True,
            "primary_ok": True,
            "full_quality_penalty": 9,
        },
        {
            "dataset": "CV",
            "family": "10-6",
            "case": "over_cap",
            "moves": 20,
            "final_n": 0,
            "runtime_sec": 1.0,
            "business_ok": False,
            "goal_validation_ok": True,
            "feasibility_ok": True,
            "time_limit_ok": True,
            "primary_ok": False,
            "full_quality_penalty": 99,
        },
        {
            "dataset": "CV",
            "family": "10-6",
            "case": "over_time",
            "moves": 21,
            "final_n": 0,
            "runtime_sec": 20.1,
            "business_ok": True,
            "goal_validation_ok": True,
            "feasibility_ok": True,
            "time_limit_ok": False,
            "primary_ok": False,
            "full_quality_penalty": 99,
        },
    ]
    output = tmp_path / "paper.csv"
    args = Namespace(variant="union", sample_per_family=10, time_limit=20.0)

    _write_csv(rows, output)
    summary = _write_summary(rows, output, tmp_path, args)

    assert "| ALL | 3 | 1 | 9 |" in summary.read_text(encoding="utf-8")
    assert "| feasibility-only over cap | 1 |" in summary.read_text(encoding="utf-8")
    assert "| feasibility-only over time | 1 |" in summary.read_text(encoding="utf-8")
    assert "primary_ok" in output.read_text(encoding="utf-8").splitlines()[0]


def test_paper_summary_records_family_filter(tmp_path: Path):
    from argparse import Namespace

    from benchmark_compact_paper_set import _write_summary

    rows = [
        {
            "dataset": "BF",
            "family": "BF16",
            "case": "case_1",
            "moves": 4,
            "final_n": 0,
            "runtime_sec": 1.0,
            "business_ok": True,
            "goal_validation_ok": True,
            "feasibility_ok": True,
            "time_limit_ok": True,
            "primary_ok": True,
            "full_quality_penalty": 4,
        }
    ]
    output = tmp_path / "paper.csv"
    args = Namespace(variant="union", sample_per_family=2, time_limit=20.0, families=["BF16", "10-6"])

    summary = _write_summary(rows, output, tmp_path, args)

    assert "families: `BF16,10-6`" in summary.read_text(encoding="utf-8")


def test_paper_runner_rejects_nonpositive_time_limit():
    from benchmark_compact_paper_set import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--time-limit", "0"])


def test_paper_parser_accepts_family_filter():
    from benchmark_compact_paper_set import build_parser

    args = build_parser().parse_args(["--families", "BF14", "10-6"])

    assert args.families == ["BF14", "10-6"]


def test_paper_runner_terminal_escape_lazy_portfolio_stops_after_valid_bridge(monkeypatch, tmp_path: Path):
    from argparse import Namespace

    from benchmark_bs_frg_paper import PaperCase
    from benchmark_compact_paper_set import PaperRunTask, _run_task

    bridge_variant = "qubo_scheduler_state_v5_qubo_diversity_union_terminal_bridge_polish"
    wide_variant = "qubo_scheduler_state_v5_qubo_diversity_union_tail_lookahead_wide_polish"
    lazy_variant = "qubo_scheduler_state_v5_qubo_diversity_union_terminal_escape_lazy_portfolio"
    case = PaperCase("BF", "BFx", "case_1", tmp_path / "case_1.bay")
    args = Namespace(alpha=10.0)
    calls = []

    def fake_load(case_arg):
        return object()

    def fake_run(case_arg, args_arg, variant, instance=None):
        calls.append(variant)
        return {
            "dataset": case_arg.dataset,
            "family": case_arg.family,
            "case": case_arg.case_id,
            "variant": variant,
            "solver_profile": "scheduler_state_v5_qubo_diversity_union_terminal_bridge_polish",
            "moves": 31,
            "final_n": 0,
            "runtime_sec": 3.0,
            "business_ok": True,
            "goal_validation_ok": True,
            "full_quality_penalty": 31,
        }

    monkeypatch.setattr("benchmark_compact_paper_set._load_paper_case", fake_load)
    monkeypatch.setattr("benchmark_compact_paper_set._run_paper_variant_row", fake_run)

    result = _run_task(PaperRunTask(0, case, lazy_variant), vars(args))

    assert calls == [bridge_variant]
    assert wide_variant in result.row["portfolio_profiles"]
    assert result.row["variant"] == lazy_variant
    assert result.row["portfolio_selected_profile"] == (
        "scheduler_state_v5_qubo_diversity_union_terminal_bridge_polish"
    )
    assert result.row["portfolio_runtime_sec"] == "3.000000"


def test_paper_runner_terminal_escape_lazy_portfolio_tries_wide_after_invalid_bridge(monkeypatch, tmp_path: Path):
    from argparse import Namespace

    from benchmark_bs_frg_paper import PaperCase
    from benchmark_compact_paper_set import PaperRunTask, _run_task

    bridge_variant = "qubo_scheduler_state_v5_qubo_diversity_union_terminal_bridge_polish"
    wide_variant = "qubo_scheduler_state_v5_qubo_diversity_union_tail_lookahead_wide_polish"
    lazy_variant = "qubo_scheduler_state_v5_qubo_diversity_union_terminal_escape_lazy_portfolio"
    case = PaperCase("BF", "BFx", "case_1", tmp_path / "case_1.bay")
    args = Namespace(alpha=10.0)
    calls = []

    def fake_load(case_arg):
        return object()

    def fake_run(case_arg, args_arg, variant, instance=None):
        calls.append(variant)
        valid = variant == wide_variant
        return {
            "dataset": case_arg.dataset,
            "family": case_arg.family,
            "case": case_arg.case_id,
            "variant": variant,
            "solver_profile": "scheduler_state_v5_qubo_diversity_union_tail_lookahead_wide_polish"
            if valid
            else "scheduler_state_v5_qubo_diversity_union_terminal_bridge_polish",
            "moves": 27 if valid else "",
            "final_n": 0 if valid else 15,
            "runtime_sec": 5.0 if valid else 1.0,
            "business_ok": valid,
            "goal_validation_ok": valid,
            "full_quality_penalty": 27 if valid else 88,
        }

    monkeypatch.setattr("benchmark_compact_paper_set._load_paper_case", fake_load)
    monkeypatch.setattr("benchmark_compact_paper_set._run_paper_variant_row", fake_run)

    result = _run_task(PaperRunTask(0, case, lazy_variant), vars(args))

    assert calls == [bridge_variant, wide_variant]
    assert result.row["variant"] == lazy_variant
    assert result.row["portfolio_selected_profile"] == (
        "scheduler_state_v5_qubo_diversity_union_tail_lookahead_wide_polish"
    )
    assert result.row["portfolio_runtime_sec"] == "6.000000"
    assert result.row["full_quality_penalty"] == 27


def test_paper_lazy_portfolio_skips_over_time_feasibility_child(monkeypatch, tmp_path: Path):
    from argparse import Namespace

    from benchmark_bs_frg_paper import PaperCase
    from benchmark_compact_paper_set import PaperRunTask, _run_task

    bridge_variant = "qubo_scheduler_state_v5_qubo_diversity_union_terminal_bridge_polish"
    wide_variant = "qubo_scheduler_state_v5_qubo_diversity_union_tail_lookahead_wide_polish"
    lazy_variant = "qubo_scheduler_state_v5_qubo_diversity_union_terminal_escape_lazy_portfolio"
    case = PaperCase("BF", "BFx", "case_1", tmp_path / "case_1.bay")
    calls = []

    def fake_load(_case_arg):
        return object()

    def fake_run(case_arg, _args_arg, variant, instance=None):
        del instance
        calls.append(variant)
        primary = variant == wide_variant
        return {
            "dataset": case_arg.dataset,
            "family": case_arg.family,
            "case": case_arg.case_id,
            "variant": variant,
            "solver_profile": variant,
            "moves": 9 if primary else 7,
            "final_n": 0,
            "runtime_sec": 1.0 if primary else 21.0,
            "business_ok": True,
            "goal_validation_ok": True,
            "feasibility_ok": True,
            "time_limit_ok": primary,
            "primary_ok": primary,
            "full_quality_penalty": 9 if primary else 99,
        }

    monkeypatch.setattr("benchmark_compact_paper_set._load_paper_case", fake_load)
    monkeypatch.setattr("benchmark_compact_paper_set._run_paper_variant_row", fake_run)

    result = _run_task(PaperRunTask(0, case, lazy_variant), vars(Namespace(alpha=10.0)))

    assert calls == [bridge_variant, wide_variant]
    assert result.row["portfolio_selected_profile"] == wide_variant
    assert result.row["primary_ok"] is True
    assert result.row["portfolio_runtime_sec"] == "22.000000"
