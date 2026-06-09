"""Run prior search baselines on all single-bay real cases."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from benchmark_full_real_business_algorithms import (  # noqa: E402
    TrajectoryResult,
    _run_business_beam,
    _run_business_qubo_beam,
)
from benchmark_full_real_enhanced_beam import run_enhanced_business_beam  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


def _base_config(time_limit: float) -> SolverConfig:
    return SolverConfig(
        max_depth=60,
        beam_width=10,
        max_candidates_per_layer=100,
        candidate_limit_per_state=3,
        time_limit_sec=time_limit,
        max_qubo_vars=60,
        qubo_penalty_a=50.0,
        diversity_weight=0.1,
        qubo_num_reads=16,
        qubo_time_limit_sec=0.02,
        qubo_sa_iterations=120,
        qubo_call_interval=5,
        qubo_min_candidates_for_call=1,
        qubo_repair_hamming_slack=0,
        entropy_alpha=1.3,
        random_seed=37,
    )


def _enhanced_fast_config(time_limit: float) -> SolverConfig:
    return SolverConfig(
        max_depth=60,
        beam_width=14,
        max_candidates_per_layer=180,
        time_limit_sec=time_limit,
        entropy_alpha=1.3,
        random_seed=37,
    )


def _enhanced_wide_config(time_limit: float) -> SolverConfig:
    return SolverConfig(
        max_depth=60,
        beam_width=24,
        max_candidates_per_layer=320,
        time_limit_sec=time_limit,
        entropy_alpha=1.3,
        random_seed=37,
    )


def _write_rows(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrajectoryResult.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _int_or_none(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _float_or_zero(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _best_baseline(rows: list[dict[str, str]]) -> dict[str, str] | None:
    valid = [row for row in rows if _int_or_none(row.get("final_n")) is not None]
    if not valid:
        return None
    missing = 10**9
    return min(
        valid,
        key=lambda row: (
            missing if _int_or_none(row.get("final_n")) is None else _int_or_none(row.get("final_n")),
            missing if _int_or_none(row.get("moves")) is None else _int_or_none(row.get("moves")),
            _float_or_zero(row.get("runtime_sec")),
        ),
    )


def _write_comparison(baseline_path: Path, new_path: Path, out_path: Path) -> None:
    baseline_rows = _read_rows(baseline_path)
    new_rows = _read_rows(new_path)
    if not baseline_rows or not new_rows:
        return

    by_case: dict[str, list[dict[str, str]]] = {}
    for row in baseline_rows:
        by_case.setdefault(row["case"], []).append(row)
    new_by_case = {row["case"]: row for row in new_rows}

    fields = [
        "case",
        "n0",
        "best_baseline_solver",
        "best_baseline_final_n",
        "best_baseline_moves",
        "best_baseline_runtime_sec",
        "new_solver",
        "new_final_n",
        "new_moves",
        "new_runtime_sec",
        "new_minus_baseline_final_n",
        "new_minus_baseline_moves",
        "new_is_better_final_n",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in sorted(by_case):
            new = new_by_case.get(case)
            best = _best_baseline(by_case[case])
            if new is None or best is None:
                continue
            best_final = _int_or_none(best.get("final_n"))
            new_final = _int_or_none(new.get("final_n"))
            best_moves = _int_or_none(best.get("moves"))
            new_moves = _int_or_none(new.get("moves"))
            writer.writerow(
                {
                    "case": case,
                    "n0": new.get("n0") or best.get("n0"),
                    "best_baseline_solver": best["solver"],
                    "best_baseline_final_n": best_final,
                    "best_baseline_moves": best_moves,
                    "best_baseline_runtime_sec": best.get("runtime_sec"),
                    "new_solver": new.get("solver"),
                    "new_final_n": new_final,
                    "new_moves": new_moves,
                    "new_runtime_sec": new.get("runtime_sec"),
                    "new_minus_baseline_final_n": None
                    if best_final is None or new_final is None
                    else new_final - best_final,
                    "new_minus_baseline_moves": None
                    if best_moves is None or new_moves is None
                    else new_moves - best_moves,
                    "new_is_better_final_n": None
                    if best_final is None or new_final is None
                    else new_final < best_final,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "single_bay_search_baselines.csv",
    )
    parser.add_argument(
        "--new-results",
        type=Path,
        default=ROOT / "docs" / "experiments" / "explainable_rule_macro_beam_all_single_bays.csv",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=ROOT / "docs" / "experiments" / "single_bay_search_baselines_vs_explainable_rule_macro_beam.csv",
    )
    args = parser.parse_args()

    cases = discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))
    rows: list[TrajectoryResult] = []
    for case_idx, (zone, bay) in enumerate(cases, start=1):
        instance = _instance_from_single_bay(zone, bay)
        case = instance.instance_id
        print(f"CASE_START {case_idx}/{len(cases)} {case}", flush=True)

        base_config = _base_config(args.time_limit)
        row = _run_business_beam(case, instance, base_config)
        rows.append(row)
        print(f"  {row.solver}: finalN={row.final_n} moves={row.moves} runtime={row.runtime_sec:.2f}s", flush=True)

        row = _run_business_qubo_beam(case, instance, base_config, solver_name="qubo_beam")
        rows.append(row)
        print(f"  {row.solver}: finalN={row.final_n} moves={row.moves} runtime={row.runtime_sec:.2f}s", flush=True)

        row = run_enhanced_business_beam(
            case,
            instance,
            _enhanced_fast_config(args.time_limit),
            solver_name="enhanced_beam_fast",
            per_rule_limit=2,
            diversity_weight=0.06,
            elite_ratio=0.60,
        )
        rows.append(row)
        print(f"  {row.solver}: finalN={row.final_n} moves={row.moves} runtime={row.runtime_sec:.2f}s", flush=True)

        row = run_enhanced_business_beam(
            case,
            instance,
            _enhanced_wide_config(args.time_limit),
            solver_name="enhanced_beam_wide",
            per_rule_limit=3,
            diversity_weight=0.08,
            elite_ratio=0.55,
        )
        rows.append(row)
        print(f"  {row.solver}: finalN={row.final_n} moves={row.moves} runtime={row.runtime_sec:.2f}s", flush=True)

        _write_rows(rows, args.output)

    _write_rows(rows, args.output)
    _write_comparison(args.output, args.new_results, args.comparison)

    solved = sum(1 for row in rows if row.final_n == 0)
    total_runtime = sum(row.runtime_sec for row in rows)
    print(
        "SUMMARY "
        f"rows={len(rows)} cases={len(cases)} solved_rows={solved} total_runtime={total_runtime:.2f}s",
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)
    if args.comparison.exists():
        print(f"wrote {args.comparison}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
