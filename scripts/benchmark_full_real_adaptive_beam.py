"""Adaptive beam racing for full real paired business cases."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Callable

from benchmark_full_real_business_algorithms import (
    ROOT,
    TrajectoryResult,
    _instance_from_zone,
    _run_business_beam,
    _run_business_qubo_beam,
    discover_paired_cases,
)
from benchmark_full_real_enhanced_beam import run_enhanced_business_beam, run_hybrid_business_beam

from container_reshuffle.core import Instance
from container_reshuffle.solvers.base import SolverConfig


VariantRunner = Callable[[str, Instance, float], TrajectoryResult]


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


def _wide_config(time_limit: float) -> SolverConfig:
    return SolverConfig(
        max_depth=60,
        beam_width=30,
        max_candidates_per_layer=300,
        candidate_limit_per_state=5,
        time_limit_sec=time_limit,
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


def _hybrid_config(time_limit: float) -> SolverConfig:
    return SolverConfig(
        max_depth=60,
        beam_width=18,
        max_candidates_per_layer=260,
        time_limit_sec=time_limit,
        entropy_alpha=1.3,
        random_seed=37,
    )


def _run_plain(case: str, instance: Instance, time_limit: float) -> TrajectoryResult:
    row = _run_business_beam(case, instance, _base_config(time_limit))
    row.solver = "race_plain_beam"
    return row


def _run_qubo(case: str, instance: Instance, time_limit: float) -> TrajectoryResult:
    return _run_business_qubo_beam(case, instance, _base_config(time_limit), solver_name="race_qubo_beam")


def _run_wide(case: str, instance: Instance, time_limit: float) -> TrajectoryResult:
    row = _run_business_beam(case, instance, _wide_config(time_limit))
    row.solver = "race_wide_beam"
    return row


def _run_enhanced(case: str, instance: Instance, time_limit: float) -> TrajectoryResult:
    return run_enhanced_business_beam(
        case,
        instance,
        _enhanced_fast_config(time_limit),
        solver_name="race_enhanced_fast",
        per_rule_limit=2,
        diversity_weight=0.06,
        elite_ratio=0.60,
    )


def _run_hybrid(case: str, instance: Instance, time_limit: float) -> TrajectoryResult:
    return run_hybrid_business_beam(
        case,
        instance,
        _hybrid_config(time_limit),
        solver_name="race_hybrid_beam",
        per_rule_limit=2,
        diversity_weight=0.05,
    )


VARIANTS: list[tuple[str, VariantRunner]] = [
    ("plain", _run_plain),
    ("qubo", _run_qubo),
    ("wide", _run_wide),
    ("enhanced", _run_enhanced),
    ("hybrid", _run_hybrid),
]


def _result_key(row: TrajectoryResult) -> tuple[int, int, int, float]:
    return (
        int(row.final_n if row.final_n is not None else 10**9),
        int(row.max_g if row.max_g is not None else 10**9),
        int(row.moves if row.moves is not None else 10**9),
        float(row.runtime_sec),
    )


def _sum_rows(case: str, solver: str, selected: TrajectoryResult, rows: list[TrajectoryResult]) -> TrajectoryResult:
    out = replace(selected)
    out.case = case
    out.solver = solver
    out.runtime_sec = sum(row.runtime_sec for row in rows)
    out.expanded_states = sum(row.expanded_states for row in rows)
    out.generated_states = sum(row.generated_states for row in rows)
    out.visited_states = sum(row.visited_states for row in rows)
    out.qubo_calls = sum(row.qubo_calls for row in rows)
    out.qubo_total_vars = sum(row.qubo_total_vars for row in rows)
    out.qubo_time_sec = sum(row.qubo_time_sec for row in rows)
    out.status = (
        f"ADAPTIVE_SELECTED={selected.solver}; "
        f"components={','.join(row.solver for row in rows)}"
    )
    return out


def run_adaptive_beam(
    case: str,
    instance: Instance,
    *,
    probe_time: float = 0.8,
    top1_time: float = 8.0,
    top2_time: float = 5.0,
) -> list[TrajectoryResult]:
    """Run short-race adaptive beam portfolios."""

    probes: list[tuple[str, TrajectoryResult]] = []
    for name, runner in VARIANTS:
        row = runner(case, instance, probe_time)
        row.solver = f"probe_{row.solver}"
        probes.append((name, row))
    ordered = sorted(probes, key=lambda item: _result_key(item[1]))

    top1_name = ordered[0][0]
    top1_full = dict(VARIANTS)[top1_name](case, instance, top1_time)
    top1_full.solver = f"full_{top1_full.solver}"
    top1_components = [row for _name, row in probes] + [top1_full]
    top1_selected = min(top1_components, key=_result_key)

    top2_full_rows = []
    for name, _probe in ordered[:2]:
        full = dict(VARIANTS)[name](case, instance, top2_time)
        full.solver = f"full2_{full.solver}"
        top2_full_rows.append(full)
    top2_components = [row for _name, row in probes] + top2_full_rows
    top2_selected = min(top2_components, key=_result_key)

    return [
        _sum_rows(case, "adaptive_beam_top1", top1_selected, top1_components),
        _sum_rows(case, "adaptive_beam_top2", top2_selected, top2_components),
    ]


def _write_csv(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrajectoryResult.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_comparison(adaptive_rows: list[TrajectoryResult]) -> None:
    base_path = ROOT / "docs" / "experiments" / "full_real_business_beam_portfolio_comparison.csv"
    rows = _read_rows(base_path)
    rows.extend({k: str(v) for k, v in row.__dict__.items()} for row in adaptive_rows)

    seen: set[tuple[str, str]] = set()
    dedup = []
    for row in rows:
        key = (row["case"], row["solver"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)

    by_solver: dict[str, list[dict[str, str]]] = {}
    for row in dedup:
        by_solver.setdefault(row["solver"], []).append(row)
    cases = sorted({row["case"] for row in dedup})
    summary = []
    for solver, solver_rows in by_solver.items():
        if len(solver_rows) != len(cases):
            continue
        summary.append(
            {
                "solver": solver,
                "cases": len(solver_rows),
                "avg_final_n": sum(float(r["final_n"]) for r in solver_rows) / len(solver_rows),
                "avg_reduction": sum(float(r["reduction"]) for r in solver_rows) / len(solver_rows),
                "avg_moves": sum(float(r["moves"]) for r in solver_rows) / len(solver_rows),
                "avg_runtime": sum(float(r["runtime_sec"]) for r in solver_rows) / len(solver_rows),
                "ok13": sum(str(r["g_ok_13"]).lower() == "true" for r in solver_rows),
            }
        )
    summary.sort(key=lambda item: (item["avg_final_n"], item["avg_moves"]))
    solvers = [item["solver"] for item in summary]
    lookup = {(row["case"], row["solver"]): row for row in dedup}

    csv_path = ROOT / "docs" / "experiments" / "full_real_business_adaptive_beam_comparison.csv"
    md_path = ROOT / "docs" / "experiments" / "full_real_business_adaptive_beam_comparison.md"
    fields = list(TrajectoryResult.__annotations__.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in dedup:
            writer.writerow({key: row.get(key, "") for key in fields})

    lines = [
        "# Full Real Business Adaptive Beam Comparison",
        "",
        "This report adds short-race adaptive beam portfolios. A short probe is run for each beam variant, then the best one or two variants receive the main budget. Portfolio runtime is the sum of all probes and selected full runs.",
        "",
        f"CSV: `{csv_path.relative_to(ROOT)}`",
        "",
        "## Aggregate",
        "",
        "| solver | cases | avg final N | avg reduction | avg moves | avg runtime s | ok 1.3N0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['solver']} | {item['cases']} | {item['avg_final_n']:.2f} | {item['avg_reduction']:.2f} | {item['avg_moves']:.2f} | {item['avg_runtime']:.3f} | {item['ok13']} |"
        )
    lines.extend(["", "## Per Case Final N", ""])
    lines.append("| case | " + " | ".join(solvers) + " |")
    lines.append("|---|" + "|".join("---:" for _ in solvers) + "|")
    for case in cases:
        values = []
        for solver in solvers:
            row = lookup.get((case, solver))
            values.append("" if row is None else row["final_n"])
        lines.append(f"| {case} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `adaptive_beam_top1` is the low-cost adaptive version: short race plus one full selected run.",
            "- `adaptive_beam_top2` keeps two selected full runs and is more robust but slower.",
            "- The full sequential beam portfolio remains a quality upper bound; adaptive racing tests how much of that gain can be recovered without running every variant to completion.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def main() -> None:
    rows: list[TrajectoryResult] = []
    for zone, _suffixes, _paths in discover_paired_cases(ROOT / "data" / "input", ["01", "26"]):
        instance = _instance_from_zone(zone)
        case = instance.instance_id
        print(f"running {case}", flush=True)
        rows.extend(run_adaptive_beam(case, instance))
    csv_path = ROOT / "docs" / "experiments" / "full_real_business_adaptive_beam.csv"
    _write_csv(rows, csv_path)
    _write_comparison(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
