from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import tune_v4_optuna as tuner  # noqa: E402


class FakeTrial:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def suggest_categorical(self, name, choices):
        self.calls.append(("categorical", name))
        return choices[0]

    def suggest_float(self, name, low, high, *, step=None, log=False):
        self.calls.append(("float", name))
        return low if step is None else low + step

    def suggest_int(self, name, low, high, *, step=1, log=False):
        self.calls.append(("int", name))
        return low


class FixedTrial:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def suggest_categorical(self, name, choices):
        value = self.values[name]
        assert value in choices
        return value

    def suggest_float(self, name, low, high, *, step=None, log=False):
        value = float(self.values[name])
        assert low <= value <= high
        if step is not None:
            ticks = round((value - low) / step)
            assert abs((low + ticks * step) - value) < 1e-9
        return value

    def suggest_int(self, name, low, high, *, step=1, log=False):
        value = int(self.values[name])
        assert low <= value <= high
        assert (value - low) % step == 0
        return value


class FakeStudy:
    def __init__(self, trials=None) -> None:
        self.trials = list(trials or [])
        self.enqueued: list[tuple[dict[str, object], dict[str, object] | None]] = []

    def enqueue_trial(self, params, user_attrs=None) -> None:
        self.enqueued.append((dict(params), dict(user_attrs) if user_attrs is not None else None))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_csv_counts_primary_ok_and_full_penalty(tmp_path: Path) -> None:
    result_csv = tmp_path / "paper.csv"
    _write_rows(
        result_csv,
        [
            {"primary_ok": "True", "moves": "10", "full_quality_penalty": "3", "runtime_sec": "1.5"},
            {"primary_ok": "false", "moves": "", "full_quality_penalty": "1000000", "runtime_sec": "2.0"},
            {"business_ok": "True", "goal_validation_ok": "True", "moves": "20", "runtime_sec": "0.5"},
        ],
    )

    summary = tuner.summarize_result_csv(result_csv, label="paper")

    assert summary.cases == 3
    assert summary.solved == 2
    assert summary.moves == 30
    assert summary.full_quality_penalty == 1_000_003
    assert summary.runtime_sec == 4.0


def test_constraint_objective_missing_solution_dominates_shorter_moves() -> None:
    all_solved = [
        tuner.BenchmarkSummary("open", cases=2, solved=2, moves=5000, full_quality_penalty=100, runtime_sec=10.0),
        tuner.BenchmarkSummary("BF", cases=2, solved=2, moves=5000, full_quality_penalty=100, runtime_sec=10.0),
        tuner.BenchmarkSummary("CV", cases=2, solved=2, moves=5000, full_quality_penalty=100, runtime_sec=10.0),
    ]
    one_missing_but_short = [
        tuner.BenchmarkSummary("open", cases=2, solved=2, moves=100, full_quality_penalty=0, runtime_sec=1.0),
        tuner.BenchmarkSummary("BF", cases=2, solved=1, moves=100, full_quality_penalty=0, runtime_sec=1.0),
        tuner.BenchmarkSummary("CV", cases=2, solved=2, moves=100, full_quality_penalty=0, runtime_sec=1.0),
    ]

    expected = {"open": 2, "BF": 2, "CV": 2}

    assert tuner.constraint_objective_score(one_missing_but_short, expected) > tuner.constraint_objective_score(
        all_solved,
        expected,
    )


def test_command_builders_pin_v4_variant_no_business_cap_and_params(tmp_path: Path) -> None:
    params = {
        "beam_width": 1200,
        "core_width": 1040,
        "parent_core_cap": 4,
        "union_diversity_qubo_macro_ancestry_penalty_weight": 36.0,
        "macro_chain_witness_parent_width": 4,
        "union_diversity_qubo_supplement_width": 80,
        "union_diversity_qubo_max_vars": 128,
        "union_diversity_qubo_select": 24,
        "union_diversity_qubo_rank_weight": 0.03,
        "union_diversity_qubo_same_parent_penalty": 0.42,
        "union_diversity_qubo_same_source_penalty": 0.18,
        "union_diversity_qubo_signature_penalty": 0.22,
        "union_diversity_qubo_reduce_signature_penalty": 0.10,
        "union_diversity_qubo_n_band_penalty": 0.06,
        "unified_macro_supply_parent_width": 4,
        "unified_macro_supply_parent_offset": 5,
        "unified_macro_supply_max_generated": 1024,
        "quality_reservoir_enabled": False,
    }
    open_cmd = tuner.build_open_command(
        trial_name="trial_0007",
        params=params,
        output_dir=tmp_path,
        time_limit=20.0,
        jobs=4,
    )
    paper_cmd = tuner.build_paper_command(
        trial_name="trial_0007",
        params=params,
        output_dir=tmp_path,
        time_limit=20.0,
        jobs=4,
        datasets=("BF", "CV"),
        families=("BF14", "10-10"),
        sample_per_family=3,
    )

    assert tuner.V4_BASE_VARIANT in open_cmd
    assert tuner.V4_BASE_VARIANT in paper_cmd
    assert "--no-business-cap" in open_cmd
    assert "--no-business-cap" in paper_cmd
    assert open_cmd[open_cmd.index("--beam-width") + 1] == "1200"
    assert paper_cmd[paper_cmd.index("--core-width") + 1] == "1040"
    assert open_cmd[open_cmd.index("--macro-chain-witness-parent-width") + 1] == "4"
    assert "--union-diversity-qubo-macro-ancestry-penalty-weight" in paper_cmd
    assert open_cmd[open_cmd.index("--union-diversity-qubo-max-vars") + 1] == "128"
    assert paper_cmd[paper_cmd.index("--union-diversity-qubo-max-vars") + 1] == "128"
    assert open_cmd[open_cmd.index("--union-diversity-qubo-same-parent-penalty") + 1] == "0.42"
    assert paper_cmd[paper_cmd.index("--union-diversity-qubo-same-parent-penalty") + 1] == "0.42"
    assert open_cmd[open_cmd.index("--unified-macro-supply-parent-width") + 1] == "4"
    assert paper_cmd[paper_cmd.index("--unified-macro-supply-parent-width") + 1] == "4"
    assert open_cmd[open_cmd.index("--unified-macro-supply-max-generated") + 1] == "1024"
    assert paper_cmd[paper_cmd.index("--unified-macro-supply-max-generated") + 1] == "1024"
    assert "--no-quality-reservoir-enabled" not in open_cmd
    assert str(tmp_path / "trial_0007_open.csv") in open_cmd
    assert str(tmp_path / "trial_0007_paper.csv") in paper_cmd


def test_open_and_paper_runners_parse_shared_optuna_v4_parameters() -> None:
    import benchmark_compact_paper_set as paper_runner
    import open_cpmp_benchmarks as open_runner

    shared = [
        "--variant",
        tuner.V4_BASE_VARIANT,
        "--union-diversity-qubo-supplement-width",
        "80",
        "--union-diversity-qubo-max-vars",
        "128",
        "--union-diversity-qubo-select",
        "24",
        "--union-diversity-qubo-rank-weight",
        "0.03",
        "--union-diversity-qubo-same-parent-penalty",
        "0.42",
        "--union-diversity-qubo-same-source-penalty",
        "0.18",
        "--union-diversity-qubo-signature-penalty",
        "0.22",
        "--union-diversity-qubo-reduce-signature-penalty",
        "0.10",
        "--union-diversity-qubo-n-band-penalty",
        "0.06",
        "--unified-macro-supply-parent-width",
        "4",
        "--unified-macro-supply-parent-offset",
        "5",
        "--unified-macro-supply-max-generated",
        "1024",
        "--no-post-polish-qubo",
    ]

    open_args = open_runner.build_parser().parse_args(shared)
    paper_args = paper_runner.build_parser().parse_args(shared)
    compact_paper_args = paper_runner._args_for_variant(paper_args, tuner.V4_BASE_VARIANT)

    assert open_args.union_diversity_qubo_supplement_width == 80
    assert paper_args.union_diversity_qubo_supplement_width == 80
    assert open_args.union_diversity_qubo_same_parent_penalty == 0.42
    assert paper_args.union_diversity_qubo_same_parent_penalty == 0.42
    assert open_args.unified_macro_supply_parent_width == 4
    assert paper_args.unified_macro_supply_parent_width == 4
    assert open_args.unified_macro_supply_max_generated == 1024
    assert paper_args.unified_macro_supply_max_generated == 1024
    assert open_args.post_polish_qubo is False
    assert paper_args.post_polish_qubo is False
    assert compact_paper_args.post_polish_qubo is False


def test_conservative_sampler_does_not_toggle_solver_family_or_postcompress() -> None:
    params = tuner.suggest_conservative_v4_params(FakeTrial())

    assert params
    assert "bs_frg_postcompress_enabled" not in params
    assert "frg_seed_enabled" not in params
    assert "union_diversity_qubo_max_vars" in params
    assert "union_diversity_qubo_same_parent_penalty" in params
    assert "unified_macro_supply_parent_width" in params
    assert "solver_profile" not in params
    assert "variant" not in params
    assert set(params).issubset(tuner.CONSERVATIVE_TUNABLE_PARAMS)


def test_v4_baseline_anchor_is_searchable_and_replays_default_cli_params() -> None:
    params = tuner.suggest_conservative_v4_params(FixedTrial(tuner.V4_BASELINE_OPTUNA_PARAMS))

    assert params["beam_width"] == 1200
    assert params["core_width"] == 1080
    assert params["union_diversity_qubo_supplement_width"] == 4
    assert params["union_diversity_qubo_max_vars"] == 48
    assert params["union_diversity_qubo_select"] == 4
    assert params["union_diversity_qubo_unified_macro_pair_penalty"] == 0.0
    assert params["macro_chain_witness_max_generated"] == 0
    assert params["post_polish_max_attempts"] == 0


def test_baseline_anchor_is_enqueued_only_for_empty_studies() -> None:
    empty = FakeStudy()
    existing = FakeStudy(trials=[object()])

    assert tuner.enqueue_baseline_anchor_if_needed(empty, enabled=True)
    assert empty.enqueued == [
        (dict(tuner.V4_BASELINE_OPTUNA_PARAMS), {"anchor": "v4.0_default"}),
    ]
    assert not tuner.enqueue_baseline_anchor_if_needed(existing, enabled=True)
    assert not existing.enqueued
    assert not tuner.enqueue_baseline_anchor_if_needed(FakeStudy(), enabled=False)


def test_best_cli_params_are_loaded_from_trial_outcome_not_raw_optuna_params(tmp_path: Path) -> None:
    outcome_path = tmp_path / "trial_0003_outcome.json"
    outcome_path.write_text(
        json.dumps(
            {
                "params": {"beam_width": 1120, "core_width": 960},
                "score": 123.0,
                "summaries": [],
            }
        ),
        encoding="utf-8",
    )
    study = SimpleNamespace(
        best_params={"beam_width": 1120, "core_ratio": 0.86},
        best_trial=SimpleNamespace(user_attrs={"outcome_json": str(outcome_path)}),
    )

    best_path = tuner.write_best_cli_params(study, tmp_path, "study_x")
    payload = json.loads(best_path.read_text(encoding="utf-8"))

    assert payload == {"beam_width": 1120, "core_width": 960}


def test_safe_sampler_keeps_qubo_selector_small_and_macro_supply_bounded() -> None:
    params = tuner.suggest_safe_v4_params(FakeTrial())

    assert params["union_diversity_qubo_max_vars"] == 48
    assert params["union_diversity_qubo_select"] in {4, 8}
    assert params["union_diversity_qubo_supplement_width"] <= 48
    assert params["unified_macro_supply_parent_width"] in {3, 4, 5}
    assert params["macro_chain_witness_max_generated"] <= 512
    assert set(params).issubset(tuner.CONSERVATIVE_TUNABLE_PARAMS)


def test_family_safe_objective_penalizes_family_regression_even_when_total_moves_match() -> None:
    summaries = [
        tuner.BenchmarkSummary("open", cases=2, solved=2, moves=100, full_quality_penalty=100, runtime_sec=1.0),
        tuner.BenchmarkSummary("BF", cases=2, solved=2, moves=200, full_quality_penalty=200, runtime_sec=1.0),
        tuner.BenchmarkSummary("CV", cases=2, solved=2, moves=300, full_quality_penalty=300, runtime_sec=1.0),
    ]
    baseline_family = {
        "CV/10-6": tuner.BenchmarkSummary(
            "CV/10-6", cases=1, solved=1, moves=250, full_quality_penalty=250, runtime_sec=1.0
        ),
        "CV/10-10": tuner.BenchmarkSummary(
            "CV/10-10", cases=1, solved=1, moves=50, full_quality_penalty=50, runtime_sec=1.0
        ),
    }
    current_family = {
        "CV/10-6": tuner.BenchmarkSummary(
            "CV/10-6", cases=1, solved=1, moves=150, full_quality_penalty=150, runtime_sec=1.0
        ),
        "CV/10-10": tuner.BenchmarkSummary(
            "CV/10-10", cases=1, solved=1, moves=150, full_quality_penalty=150, runtime_sec=1.0
        ),
    }
    aggregate_score = tuner.constraint_objective_score(summaries, {"open": 2, "BF": 2, "CV": 2})
    safe_score = tuner.family_safe_objective_score(
        summaries,
        {"open": 2, "BF": 2, "CV": 2},
        current_family=current_family,
        baseline_family=baseline_family,
        move_tolerance=20,
        move_regression_weight=1000.0,
    )

    assert safe_score == aggregate_score + 80_000.0


def test_family_breakdown_uses_dataset_and_family_keys(tmp_path: Path) -> None:
    result_csv = tmp_path / "paper.csv"
    _write_rows(
        result_csv,
        [
            {
                "dataset": "CV",
                "family": "10-10",
                "primary_ok": "True",
                "moves": "40",
                "full_quality_penalty": "40",
                "runtime_sec": "1",
            },
            {
                "dataset": "CV",
                "family": "10-10",
                "primary_ok": "False",
                "moves": "",
                "full_quality_penalty": "100",
                "runtime_sec": "2",
            },
            {
                "dataset": "BF",
                "family": "BF14",
                "primary_ok": "True",
                "moves": "70",
                "full_quality_penalty": "70",
                "runtime_sec": "3",
            },
        ],
    )

    metrics = tuner.summarize_csv_by_family(result_csv, default_dataset="open")

    assert metrics["CV/10-10"].cases == 2
    assert metrics["CV/10-10"].solved == 1
    assert metrics["CV/10-10"].moves == 40
    assert metrics["CV/10-10"].full_quality_penalty == 140
    assert metrics["BF/BF14"].solved == 1


def test_empty_storage_normalizes_to_in_memory_optuna_storage(tmp_path: Path) -> None:
    assert tuner.normalize_storage_url("", tmp_path) is None
    assert tuner.normalize_storage_url("sqlite:///already.db", tmp_path) == "sqlite:///already.db"
    assert tuner.normalize_storage_url("study.db", tmp_path) == f"sqlite:///{(tmp_path / 'study.db').as_posix()}"
