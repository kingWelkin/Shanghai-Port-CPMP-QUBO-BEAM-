"""Tune standalone QUBO-beam variants on full real paired business cases."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from benchmark_full_real_business_algorithms import (
    ROOT,
    TrajectoryResult,
    _instance_from_zone,
    _run_business_qubo_beam,
    discover_paired_cases,
)

from container_reshuffle.core import Instance
from container_reshuffle.solvers.base import SolverConfig


def _base_config(seed: int = 37) -> SolverConfig:
    """Current standalone QUBO-beam baseline."""

    return SolverConfig(
        max_depth=60,
        beam_width=10,
        max_candidates_per_layer=100,
        candidate_limit_per_state=3,
        time_limit_sec=10.0,
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
        random_seed=seed,
    )


def _low_div_config(seed: int = 37) -> SolverConfig:
    """More conservative diversity pressure with slightly larger beam."""

    return SolverConfig(
        max_depth=60,
        beam_width=12,
        max_candidates_per_layer=140,
        candidate_limit_per_state=4,
        time_limit_sec=10.0,
        max_qubo_vars=80,
        qubo_penalty_a=60.0,
        diversity_weight=0.035,
        qubo_num_reads=20,
        qubo_time_limit_sec=0.02,
        qubo_sa_iterations=140,
        qubo_call_interval=4,
        qubo_min_candidates_for_call=1,
        qubo_repair_hamming_slack=1,
        entropy_alpha=1.3,
        random_seed=seed,
    )


def _wide_config(seed: int = 37) -> SolverConfig:
    """Wider standalone QUBO beam with more per-state actions."""

    return SolverConfig(
        max_depth=60,
        beam_width=16,
        max_candidates_per_layer=220,
        candidate_limit_per_state=5,
        time_limit_sec=10.0,
        max_qubo_vars=120,
        qubo_penalty_a=70.0,
        diversity_weight=0.055,
        qubo_num_reads=24,
        qubo_time_limit_sec=0.03,
        qubo_sa_iterations=180,
        qubo_call_interval=4,
        qubo_min_candidates_for_call=1,
        qubo_repair_hamming_slack=2,
        entropy_alpha=1.3,
        random_seed=seed,
    )


def _dense_config(seed: int = 37) -> SolverConfig:
    """Call QUBO more often while keeping each QUBO small."""

    return SolverConfig(
        max_depth=60,
        beam_width=10,
        max_candidates_per_layer=120,
        candidate_limit_per_state=3,
        time_limit_sec=10.0,
        max_qubo_vars=80,
        qubo_penalty_a=50.0,
        diversity_weight=0.075,
        qubo_num_reads=12,
        qubo_time_limit_sec=0.012,
        qubo_sa_iterations=80,
        qubo_call_interval=1,
        qubo_min_candidates_for_call=1,
        qubo_repair_hamming_slack=1,
        entropy_alpha=1.3,
        random_seed=seed,
    )


VARIANTS = [
    ("qbo_beam_base_rerun", _base_config),
    ("qbo_beam_low_div", _low_div_config),
    ("qbo_beam_wide", _wide_config),
    ("qbo_beam_dense", _dense_config),
]


def _result_key(row: TrajectoryResult) -> tuple[int, int, int, float]:
    return (
        int(row.final_n if row.final_n is not None else 10**9),
        int(row.max_g if row.max_g is not None else 10**9),
        int(row.moves if row.moves is not None else 10**9),
        float(row.runtime_sec),
    )


def _portfolio_row(case: str, rows: list[TrajectoryResult]) -> TrajectoryResult:
    """Build a QUBO-only portfolio row from standalone QUBO variants."""

    selected = min(rows, key=_result_key)
    out = replace(selected)
    out.case = case
    out.solver = "qbo_beam_standalone_portfolio"
    out.runtime_sec = sum(row.runtime_sec for row in rows)
    out.expanded_states = sum(row.expanded_states for row in rows)
    out.generated_states = sum(row.generated_states for row in rows)
    out.visited_states = sum(row.visited_states for row in rows)
    out.qubo_calls = sum(row.qubo_calls for row in rows)
    out.qubo_total_vars = sum(row.qubo_total_vars for row in rows)
    out.qubo_time_sec = sum(row.qubo_time_sec for row in rows)
    out.status = f"QBO_STANDALONE_PORTFOLIO_SELECTED={selected.solver}; components={','.join(row.solver for row in rows)}"
    return out


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


def _write_comparison(rows_new: list[TrajectoryResult]) -> None:
    base_path = ROOT / "docs" / "experiments" / "full_real_business_enhanced_qubo_beam_comparison.csv"
    rows = _read_rows(base_path)
    rows.extend({k: str(v) for k, v in row.__dict__.items()} for row in rows_new)

    seen: set[tuple[str, str]] = set()
    dedup: list[dict[str, str]] = []
    for row in rows:
        key = (row["case"], row["solver"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)

    cases = sorted({row["case"] for row in dedup})
    by_solver: dict[str, list[dict[str, str]]] = {}
    for row in dedup:
        by_solver.setdefault(row["solver"], []).append(row)
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
                "avg_qubo_calls": sum(float(r["qubo_calls"]) for r in solver_rows) / len(solver_rows),
                "ok13": sum(str(r["g_ok_13"]).lower() == "true" for r in solver_rows),
            }
        )
    summary.sort(key=lambda item: (item["avg_final_n"], item["avg_moves"], item["avg_runtime"]))
    solvers = [item["solver"] for item in summary]
    lookup = {(row["case"], row["solver"]): row for row in dedup}

    csv_path = ROOT / "docs" / "experiments" / "full_real_business_qubo_standalone_tuning_comparison.csv"
    md_path = ROOT / "docs" / "experiments" / "full_real_business_qubo_standalone_tuning_comparison.md"
    fields = list(TrajectoryResult.__annotations__.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in dedup:
            writer.writerow({field: row.get(field, "") for field in fields})

    lines = [
        "# Full Real Business Standalone QUBO Beam Tuning",
        "",
        "This report tests whether the QUBO branch itself can be strengthened without using plain/enhanced beam as separate portfolio components.",
        "",
        f"CSV: `{csv_path.relative_to(ROOT)}`",
        "",
        "## Aggregate",
        "",
        "| solver | cases | avg final N | avg reduction | avg moves | avg runtime s | avg QUBO calls | ok 1.3N0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['solver']} | {item['cases']} | {item['avg_final_n']:.2f} | {item['avg_reduction']:.2f} | "
            f"{item['avg_moves']:.2f} | {item['avg_runtime']:.3f} | {item['avg_qubo_calls']:.2f} | {item['ok13']} |"
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
            "- `qbo_beam_low_div` reduces diversity pressure and slightly widens the beam.",
            "- `qbo_beam_wide` expands both beam width and per-state action retention.",
            "- `qbo_beam_dense` calls QUBO every layer with smaller annealing budgets.",
            "- `qbo_beam_standalone_portfolio` selects the best result among the four QUBO-only variants per case; it does not use plain or enhanced beam results.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def run_case(case: str, instance: Instance) -> list[TrajectoryResult]:
    """Run all standalone QUBO variants for one case."""

    rows = []
    for solver_name, cfg_factory in VARIANTS:
        print(f"  {solver_name}", flush=True)
        rows.append(_run_business_qubo_beam(case, instance, cfg_factory(), solver_name=solver_name))
    rows.append(_portfolio_row(case, rows))
    return rows


def main() -> None:
    rows: list[TrajectoryResult] = []
    for zone, _suffixes, _paths in discover_paired_cases(ROOT / "data" / "input", ["01", "26"]):
        instance = _instance_from_zone(zone)
        case = instance.instance_id
        print(f"running {case}", flush=True)
        rows.extend(run_case(case, instance))
    csv_path = ROOT / "docs" / "experiments" / "full_real_business_qubo_standalone_tuning.csv"
    _write_csv(rows, csv_path)
    _write_comparison(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
