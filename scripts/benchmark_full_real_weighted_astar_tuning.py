"""Tune weighted-A* variants on full real paired business cases."""

from __future__ import annotations

import csv
from pathlib import Path

from benchmark_full_real_business_algorithms import ROOT, TrajectoryResult, _instance_from_zone, discover_paired_cases
from benchmark_full_real_lns_astar import run_weighted_astar_business_portfolio


def _read_previous_weighted_astar() -> list[dict[str, str]]:
    path = ROOT / "docs" / "experiments" / "full_real_business_lns_astar.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            if row.get("solver") == "weighted_astar_business":
                row = dict(row)
                row["solver"] = "weighted_astar_business_previous"
                rows.append(row)
        return rows


def _row_to_dict(row: TrajectoryResult) -> dict[str, str]:
    return {key: str(value) for key, value in row.__dict__.items()}


def _write_outputs(rows: list[dict[str, str]]) -> None:
    csv_path = ROOT / "docs" / "experiments" / "full_real_business_weighted_astar_tuning.csv"
    md_path = ROOT / "docs" / "experiments" / "full_real_business_weighted_astar_tuning.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(TrajectoryResult.__annotations__.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    cases = sorted({row["case"] for row in rows})
    by_solver: dict[str, list[dict[str, str]]] = {}
    for row in rows:
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
                "ok13": sum(str(r["g_ok_13"]).lower() == "true" for r in solver_rows),
            }
        )
    summary.sort(key=lambda item: (item["avg_final_n"], item["avg_moves"], item["avg_runtime"]))
    solvers = [item["solver"] for item in summary]
    lookup = {(row["case"], row["solver"]): row for row in rows}

    lines = [
        "# Full Real Business Weighted-A* Tuning",
        "",
        "This report compares the previous single-queue weighted-A* benchmark with the enhanced portfolio weighted-A* route.",
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
            f"| {item['solver']} | {item['cases']} | {item['avg_final_n']:.2f} | "
            f"{item['avg_reduction']:.2f} | {item['avg_moves']:.2f} | {item['avg_runtime']:.3f} | {item['ok13']} |"
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
            "- `weighted_astar_business_previous` is the earlier single-queue weighted-A* result from the LNS/A* benchmark.",
            "- `weighted_astar_business_portfolio` runs legacy, aggressive multi-queue, and hybrid multi-queue A* branches, then keeps the best validated route.",
            "- The portfolio is still a best-effort heuristic benchmark; it is not an optimality proof.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def main() -> None:
    rows = _read_previous_weighted_astar()
    for zone, _suffixes, _paths in discover_paired_cases(ROOT / "data" / "input", ["01", "26"]):
        instance = _instance_from_zone(zone)
        print(f"running {instance.instance_id}", flush=True)
        rows.append(_row_to_dict(run_weighted_astar_business_portfolio(instance.instance_id, instance, 30.0)))
    _write_outputs(rows)


if __name__ == "__main__":
    main()
