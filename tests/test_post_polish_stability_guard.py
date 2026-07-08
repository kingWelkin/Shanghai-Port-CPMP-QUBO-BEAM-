from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_stability_guard_uses_solved_beam_fallback_and_accumulates_runtime() -> None:
    from open_cpmp_benchmarks import _apply_post_polish_stability_guard_row

    repaired = {
        "primary_ok": False,
        "goal_validation_ok": False,
        "final_n": 80,
        "moves": "",
        "runtime_sec": 120.5,
        "full_quality_penalty": 1000320,
        "validation_msg": "TIME_LIMIT_OR_BEAM_EXHAUSTED",
    }
    beam = {
        "primary_ok": True,
        "goal_validation_ok": True,
        "business_ok": True,
        "final_n": 0,
        "moves": 383,
        "runtime_sec": 21.25,
        "full_quality_penalty": 383,
        "validation_msg": "ok",
    }

    selected = _apply_post_polish_stability_guard_row(
        repaired,
        beam,
        enabled=True,
        attempted=True,
    )

    assert selected["moves"] == 383
    assert selected["final_n"] == 0
    assert selected["runtime_sec"] == "141.750000"
    assert selected["post_polish_stability_guard_enabled"] is True
    assert selected["post_polish_stability_guard_attempted"] is True
    assert selected["post_polish_stability_guard_used"] is True
    assert selected["post_polish_stability_guard_selected_source"] == "beam_fallback"
    assert selected["post_polish_stability_guard_original_final_n"] == 80
    assert selected["post_polish_stability_guard_original_runtime_sec"] == "120.500000"
    assert selected["post_polish_stability_guard_fallback_runtime_sec"] == "21.250000"


def test_stability_guard_keeps_repaired_solution_without_fallback_attempt() -> None:
    from open_cpmp_benchmarks import _apply_post_polish_stability_guard_row

    repaired = {
        "primary_ok": True,
        "goal_validation_ok": True,
        "business_ok": True,
        "final_n": 0,
        "moves": 300,
        "runtime_sec": 15.0,
        "full_quality_penalty": 300,
        "validation_msg": "ok",
    }

    selected = _apply_post_polish_stability_guard_row(
        repaired,
        None,
        enabled=True,
        attempted=False,
    )

    assert selected["moves"] == 300
    assert selected["runtime_sec"] == 15.0
    assert selected["post_polish_stability_guard_enabled"] is True
    assert selected["post_polish_stability_guard_attempted"] is False
    assert selected["post_polish_stability_guard_used"] is False
    assert selected["post_polish_stability_guard_selected_source"] == "repaired"


def test_open_runner_guard_retries_with_beam_only_when_repaired_run_fails(monkeypatch) -> None:
    import open_cpmp_benchmarks as open_runner

    calls: list[bool] = []

    def fake_args_for_variant(_args, _variant, _instance):
        return SimpleNamespace(
            post_polish_qubo=True,
            business_cap_enabled=False,
            _profile_explicit_keys=set(),
        )

    def fake_run(_case_id, _instance, compact_args):
        calls.append(bool(compact_args.post_polish_qubo))
        return SimpleNamespace(kind="repaired" if compact_args.post_polish_qubo else "beam")

    def fake_row(result, _instance, compact_args):
        if result.kind == "repaired":
            return {
                "moves": "",
                "final_n": 80,
                "runtime_sec": 120.0,
                "full_quality_penalty": 1000320,
                "post_polish_qubo": compact_args.post_polish_qubo,
            }
        return {
            "moves": 383,
            "final_n": 0,
            "runtime_sec": 21.0,
            "full_quality_penalty": 383,
            "post_polish_qubo": compact_args.post_polish_qubo,
        }

    def fake_validate(result, *_args, **_kwargs):
        return (True, result.kind == "beam", "ok" if result.kind == "beam" else "failed")

    monkeypatch.setattr(open_runner, "_args_for_variant", fake_args_for_variant)
    monkeypatch.setattr(open_runner, "run_compact_beam_instance", fake_run)
    monkeypatch.setattr(open_runner, "_open_row_dict_from_instance_result", fake_row)
    monkeypatch.setattr(open_runner, "_open_validate_instance_result", fake_validate)

    case = SimpleNamespace(
        case_id="case_1",
        family="fam",
        instance_path=Path("case.dat"),
        reference_path=None,
        reference=SimpleNamespace(optimal_moves=None, max_height=None),
    )
    row = open_runner._run_open_variant_row(
        case,
        object(),
        SimpleNamespace(alpha=10.0, post_polish_stability_guard=True),
        "variant",
    )

    assert calls == [True, False]
    assert row["moves"] == 383
    assert row["runtime_sec"] == "141.000000"
    assert row["post_polish_stability_guard_used"] is True
    assert row["post_polish_stability_guard_selected_source"] == "beam_fallback"
