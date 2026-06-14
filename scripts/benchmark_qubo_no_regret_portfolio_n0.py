"""No-regret wrapper for compact QUBO Beam N=0 experiments.

The wrapper runs the stable compact Beam baseline first, then optional stronger
QUBO challengers.  The final row always selects the shortest validated solution;
if challengers fail or degrade, the stable baseline remains the incumbent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_compact_beam_qubo_n0 import (  # noqa: E402
    _SOLVER_PROFILES,
    _case_filter,
    _row_dict_from_instance_result,
    _validate_instance_result,
    make_compact_args,
    run_compact_beam_instance,
)
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


PORTFOLIO_FIELDS = [
    "case_id",
    "bay",
    "n0",
    "cap_13",
    "selected_profile",
    "selected_moves",
    "selected_full_quality_penalty",
    "baseline_moves",
    "improvement_over_baseline",
    "selected_runtime_sec",
    "total_portfolio_runtime_sec",
    "all_profile_moves_json",
    "all_profile_runtime_json",
    "all_profile_valid_json",
    "all_profile_final_n_json",
    "all_profile_full_quality_penalty_json",
    "all_profile_status_json",
    "validated",
    "final_n",
]

PORTFOLIO_PROFILE_OVERRIDES: dict[str, dict[str, object]] = {
    "stable": {},
    "speed_v2": {
        "solver_profile": "speed_v2",
    },
    "scheduler_state_v5": {
        "solver_profile": "scheduler_state_v5",
    },
    "scheduler_state_v5_window": {
        "solver_profile": "scheduler_state_v5_window_polish",
    },
    "scheduler_state_v6": {
        "solver_profile": "scheduler_state_v6",
    },
    "wang_incumbent": {
        "wang_incumbent_enabled": True,
        "wang_incumbent_beam_width": 100,
        "wang_incumbent_candidate_limit": 64,
        "wang_incumbent_max_depth": 0,
        "wang_incumbent_time_limit": 2.0,
        "wang_cap_enabled": True,
    },
    "scheduler_state_v5_wang_pair": {
        "solver_profile": "scheduler_state_v5_wang_pair_qubo",
        "wang_pair_qubo_enabled": True,
        "wang_pair_qubo_supplement_width": 8,
        "wang_pair_qubo_focus_n": 8,
        "wang_pair_qubo_n_window": 1,
        "wang_pair_qubo_max_vars": 32,
        "wang_pair_qubo_select": 8,
        "wang_pair_qubo_same_parent_penalty": 0.25,
    },
    "scheduler_state_v5_wang_rescue": {
        "solver_profile": "scheduler_state_v5_wang_rescue",
        "wang_pair_qubo_enabled": True,
        "wang_pair_qubo_supplement_width": 12,
        "wang_pair_qubo_focus_n": -1,
        "wang_pair_qubo_n_window": 1,
        "wang_pair_qubo_max_vars": 48,
        "wang_pair_qubo_select": 12,
        "wang_pair_qubo_same_parent_penalty": 0.35,
    },
    "wang_rescue": {
        "solver_profile": "scheduler_state_v5_wang_rescue",
        "wang_pair_qubo_enabled": True,
        "wang_pair_qubo_supplement_width": 12,
        "wang_pair_qubo_focus_n": -1,
        "wang_pair_qubo_n_window": 1,
        "wang_pair_qubo_max_vars": 48,
        "wang_pair_qubo_select": 12,
        "wang_pair_qubo_same_parent_penalty": 0.35,
    },
    "wang_pair": {
        "solver_profile": "scheduler_state_v5_wang_pair_qubo",
        "wang_pair_qubo_enabled": True,
        "wang_pair_qubo_supplement_width": 8,
        "wang_pair_qubo_focus_n": 8,
        "wang_pair_qubo_n_window": 1,
        "wang_pair_qubo_max_vars": 32,
        "wang_pair_qubo_select": 8,
        "wang_pair_qubo_same_parent_penalty": 0.25,
    },
    "scheduler_state_v5_wang_hybrid": {
        "solver_profile": "scheduler_state_v5_wang_hybrid",
        "wang_incumbent_time_limit": 2.0,
    },
    "wang_hybrid": {
        "solver_profile": "scheduler_state_v5_wang_hybrid",
        "wang_incumbent_time_limit": 2.0,
    },
    "state_light": {
        "qubo_influence_profile": "light",
    },
    "state_medium": {
        "qubo_influence_profile": "medium",
    },
    "protected_group_medium": {
        "qubo_influence_profile": "medium",
        "core_width": 840,
        "group_qubo_supplement": True,
        "max_group_qubo_calls": 96,
        "qubo_group_max_vars": 128,
        "qubo_group_select": 48,
        "qubo_group_take": 6,
        "qubo_group_lookahead_weight": 0.20,
        "qubo_group_branch_weight": 0.05,
    },
    "qubo_bucket": {
        "selection_policy": "qubo_bucket",
        "bucket_key": "n_depth",
        "bucket_core_ratio": 0.50,
        "bucket_qubo_share": 0.35,
        "bucket_max_qubo_vars": 96,
        "bucket_qubo_select": 24,
        "bucket_min_candidates": 64,
    },
}


@dataclass
class PortfolioCandidate:
    profile: str
    moves: int | None
    runtime_sec: float
    valid: bool
    final_n: int | None
    status: str
    row: dict[str, object] = field(default_factory=dict, repr=False)


def _is_true(value: object) -> bool:
    return value in (True, "True", "true", "1", 1)


def _json_object(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object, got {type(parsed).__name__}")
    return parsed


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _case_full_quality_penalty(
    moves: int | None,
    valid: bool,
    final_n: int | None,
    n0: int,
    cap_13: int,
    penalty_factor: int = 4,
) -> int:
    if valid and moves is not None and final_n == 0:
        return int(moves)
    return max(n0, cap_13) + int(penalty_factor) * max(1, n0)


def _profile_full_quality_summary(
    rows: list[dict[str, object]],
    penalty_factor: int = 4,
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for row in rows:
        moves_by_profile = _json_object(row.get("all_profile_moves_json"))
        runtime_by_profile = _json_object(row.get("all_profile_runtime_json"))
        valid_by_profile = _json_object(row.get("all_profile_valid_json"))
        final_n_by_profile = _json_object(row.get("all_profile_final_n_json"))
        profiles = set(moves_by_profile) | set(runtime_by_profile) | set(valid_by_profile) | set(final_n_by_profile)
        n0 = int(float(row.get("n0") or 0))
        cap_13 = int(float(row.get("cap_13") or n0))
        for profile in profiles:
            moves = _optional_int(moves_by_profile.get(profile))
            final_n = _optional_int(final_n_by_profile.get(profile))
            valid = _is_true(valid_by_profile.get(profile))
            penalty = _case_full_quality_penalty(
                moves,
                valid,
                final_n,
                n0=n0,
                cap_13=cap_13,
                penalty_factor=penalty_factor,
            )
            item = summary.setdefault(
                str(profile),
                {
                    "cases": 0,
                    "valid": 0,
                    "invalid_or_unsolved": 0,
                    "penalized_moves_total": 0,
                    "runtime_total": 0.0,
                },
            )
            item["cases"] = int(item["cases"]) + 1
            item["valid"] = int(item["valid"]) + (1 if valid and final_n == 0 and moves is not None else 0)
            item["invalid_or_unsolved"] = int(item["invalid_or_unsolved"]) + (
                0 if valid and final_n == 0 and moves is not None else 1
            )
            item["penalized_moves_total"] = int(item["penalized_moves_total"]) + penalty
            item["runtime_total"] = float(item["runtime_total"]) + float(runtime_by_profile.get(profile) or 0.0)
    return summary


def _select_no_regret_candidate(candidates: list[PortfolioCandidate]) -> PortfolioCandidate:
    if not candidates:
        raise ValueError("portfolio has no candidates")
    valid = [candidate for candidate in candidates if candidate.valid and candidate.moves is not None]
    if not valid:
        return candidates[0]
    return min(
        valid,
        key=lambda candidate: (
            candidate.moves if candidate.moves is not None else math.inf,
            candidate.runtime_sec,
            0 if candidate.profile == "stable" else 1,
        ),
    )


def _parse_portfolio_profiles(raw: str) -> list[str]:
    profiles = [part.strip() for part in raw.split(",") if part.strip()]
    if not profiles:
        profiles = ["stable"]
    unknown = [profile for profile in profiles if profile not in PORTFOLIO_PROFILE_OVERRIDES]
    if unknown:
        raise ValueError(f"unknown portfolio profile(s): {', '.join(unknown)}")
    if "stable" in profiles:
        profiles = ["stable"] + [profile for profile in profiles if profile != "stable"]
    else:
        profiles.insert(0, "stable")
    return profiles


def _split_case(case: str) -> tuple[str, str]:
    base = case.removesuffix("_single")
    zone, bay = base.rsplit("_", 1)
    return zone, bay


def _args_for_portfolio_profile(
    args: argparse.Namespace,
    profile: str,
) -> argparse.Namespace:
    overrides: dict[str, object] = {
        "alpha": args.alpha,
        "max_moves": args.max_moves,
        "solver_profile": args.solver_profile,
        "target_final_n": 0,
        "time_limit": args.portfolio_time_limit_per_variant
        if args.portfolio_time_limit_per_variant is not None
        else args.time_limit,
    }
    overrides.update(PORTFOLIO_PROFILE_OVERRIDES[profile])
    return make_compact_args(**overrides)


def _candidate_from_result(
    profile: str,
    result,
    instance,
    compact_args: argparse.Namespace,
    business_alpha: float,
) -> PortfolioCandidate:
    row = _row_dict_from_instance_result(result, instance, compact_args)
    business_ok, goal_ok, validation_msg = _validate_instance_result(
        result,
        instance,
        business_alpha=business_alpha,
    )
    moves_value = row.get("moves")
    moves = int(moves_value) if moves_value not in (None, "") else None
    final_n_value = row.get("final_n")
    final_n = int(final_n_value) if final_n_value not in (None, "") else None
    valid = bool(business_ok and goal_ok and moves is not None and final_n == 0)
    status = f"{result.status}; validation={validation_msg}; valid={valid}"
    return PortfolioCandidate(
        profile=profile,
        moves=moves,
        runtime_sec=float(result.runtime_sec),
        valid=valid,
        final_n=final_n,
        status=status,
        row=row,
    )


def run_portfolio_case(case: str, args: argparse.Namespace) -> dict[str, object]:
    zone, bay = _split_case(case)
    instance = _instance_from_single_bay(zone, bay)
    profiles = _parse_portfolio_profiles(args.portfolio_profiles)
    candidates: list[PortfolioCandidate] = []

    for index, profile in enumerate(profiles):
        compact_args = _args_for_portfolio_profile(args, profile)
        result = run_compact_beam_instance(case, instance, compact_args)
        candidate = _candidate_from_result(profile, result, instance, compact_args, args.alpha)
        candidates.append(candidate)
        if index == 0 and args.portfolio_stop_on_baseline_fail and not candidate.valid:
            break

    selected = _select_no_regret_candidate(candidates)
    baseline = candidates[0]
    profile_moves = {candidate.profile: candidate.moves for candidate in candidates}
    profile_runtime = {candidate.profile: candidate.runtime_sec for candidate in candidates}
    profile_valid = {candidate.profile: candidate.valid for candidate in candidates}
    profile_final_n = {candidate.profile: candidate.final_n for candidate in candidates}
    n0 = int(baseline.row.get("n0") or 0)
    cap_13 = int(baseline.row.get("cap_13") or n0)
    profile_penalty = {
        candidate.profile: _case_full_quality_penalty(candidate.moves, candidate.valid, candidate.final_n, n0, cap_13)
        for candidate in candidates
    }
    profile_status = {candidate.profile: candidate.status for candidate in candidates}

    improvement: int | str
    if baseline.moves is None or selected.moves is None:
        improvement = ""
    else:
        improvement = baseline.moves - selected.moves

    return {
        "case_id": case,
        "bay": bay,
        "n0": n0,
        "cap_13": cap_13,
        "selected_profile": selected.profile,
        "selected_moves": selected.moves if selected.moves is not None else "",
        "selected_full_quality_penalty": profile_penalty[selected.profile],
        "baseline_moves": baseline.moves if baseline.moves is not None else "",
        "improvement_over_baseline": improvement,
        "selected_runtime_sec": selected.runtime_sec,
        "total_portfolio_runtime_sec": sum(candidate.runtime_sec for candidate in candidates),
        "all_profile_moves_json": json.dumps(profile_moves, ensure_ascii=False, sort_keys=True),
        "all_profile_runtime_json": json.dumps(profile_runtime, ensure_ascii=False, sort_keys=True),
        "all_profile_valid_json": json.dumps(profile_valid, ensure_ascii=False, sort_keys=True),
        "all_profile_final_n_json": json.dumps(profile_final_n, ensure_ascii=False, sort_keys=True),
        "all_profile_full_quality_penalty_json": json.dumps(profile_penalty, ensure_ascii=False, sort_keys=True),
        "all_profile_status_json": json.dumps(profile_status, ensure_ascii=False, sort_keys=True),
        "validated": selected.valid,
        "final_n": selected.final_n if selected.final_n is not None else "",
    }


def run_portfolio(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cases = _case_filter(args)
    for index, case in enumerate(cases, start=1):
        print(f"CASE_START {index}/{len(cases)} {case}", flush=True)
        row = run_portfolio_case(case, args)
        rows.append(row)
        print(
            f"CASE_DONE {case} selected={row['selected_profile']} moves={row['selected_moves']} "
            f"baseline={row['baseline_moves']} improved={row['improvement_over_baseline']} "
            f"valid={row['validated']} total_runtime={float(row['total_portfolio_runtime_sec']):.3f}s",
            flush=True,
        )
    return rows


def _write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PORTFOLIO_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(rows: list[dict[str, object]], output: Path) -> Path:
    summary = output.with_name(output.stem + "_summary.md")
    wins: dict[str, int] = {}
    improved = 0
    validated = 0
    for row in rows:
        profile = str(row["selected_profile"])
        wins[profile] = wins.get(profile, 0) + 1
        validated += 1 if row["validated"] in (True, "True", "true", 1) else 0
        improvement = row.get("improvement_over_baseline")
        if improvement not in (None, "") and int(improvement) > 0:
            improved += 1
    lines = [
        "# No-Regret QUBO Portfolio Summary",
        "",
        f"cases: `{len(rows)}`",
        f"validated: `{validated}`",
        f"improved_over_baseline_cases: `{improved}`",
        "",
        "## Selected Winners",
        "",
        "| selected profile | wins |",
        "|---|---:|",
    ]
    for profile, count in sorted(wins.items()):
        lines.append(f"| {profile} | {count} |")
    lines.extend(
        [
            "",
            "## Full-Case Quality",
            "",
            "| profile | cases | valid | invalid/unsolved | penalized moves total | avg penalized moves | total runtime sec |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    quality = _profile_full_quality_summary(rows)
    for profile, item in sorted(
        quality.items(),
        key=lambda entry: (
            int(entry[1]["invalid_or_unsolved"]),
            int(entry[1]["penalized_moves_total"]),
            float(entry[1]["runtime_total"]),
            entry[0],
        ),
    ):
        cases = int(item["cases"])
        total = int(item["penalized_moves_total"])
        lines.append(
            f"| {profile} | {cases} | {int(item['valid'])} | {int(item['invalid_or_unsolved'])} | "
            f"{total} | {total / cases if cases else 0.0:.3f} | {float(item['runtime_total']):.3f} |"
        )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--solver-profile", choices=sorted(_SOLVER_PROFILES.keys()), default="speed_v2")
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--time-limit", type=float, default=0.0)
    parser.add_argument(
        "--portfolio-profiles",
        default=(
            "stable,scheduler_state_v5,wang_incumbent,scheduler_state_v5_wang_pair,"
            "scheduler_state_v5_wang_rescue,scheduler_state_v5_wang_hybrid,"
            "scheduler_state_v5_window,scheduler_state_v6"
        ),
        help="Comma-separated portfolio profile list. Stable is always run first.",
    )
    parser.add_argument("--portfolio-stop-on-baseline-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--portfolio-time-limit-per-variant", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "qubo_no_regret_portfolio_n0.csv",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    rows = run_portfolio(args)
    _write_csv(rows, args.output)
    summary = _write_summary(rows, args.output)
    print(f"WROTE {args.output}", flush=True)
    print(f"WROTE {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
