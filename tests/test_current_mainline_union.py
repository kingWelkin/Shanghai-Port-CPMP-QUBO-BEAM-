from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_union_diversity_profile_is_current_mainline_candidate():
    from benchmark_compact_beam_qubo_n0 import _apply_solver_profile, make_compact_args

    args = make_compact_args(solver_profile="scheduler_state_v5_qubo_diversity_union_polish")

    _apply_solver_profile(args)

    assert args.solver_profile == "scheduler_state_v5_qubo_diversity_union_polish"
    assert args.frg_seed_enabled is True
    assert args.union_diversity_qubo_enabled is True
    assert args.union_diversity_qubo_supplement_width == 4
    assert args.candidate_ledger_score_enabled is False


def test_open_runner_defaults_to_v43_mainline():
    from open_cpmp_benchmarks import _args_for_variant, build_parser

    outer = build_parser().parse_args([])
    args = _args_for_variant(outer, outer.variant)

    assert outer.variant == (
        "qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_"
        "wang_route_v43_polish"
    )
    assert args.solver_profile == (
        "scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_"
        "short_incumbent_v42_polish"
    )


def test_paper_set_runner_defaults_to_v43_mainline():
    from benchmark_compact_paper_set import build_parser as build_paper_parser
    from open_cpmp_benchmarks import _args_for_variant

    outer = build_paper_parser().parse_args([])
    args = _args_for_variant(outer, outer.variant)

    assert outer.variant == (
        "qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_"
        "wang_route_v43_polish"
    )
    assert args.solver_profile == (
        "scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_"
        "short_incumbent_v42_polish"
    )
