"""Build focused beam portfolio rows from completed beam runs."""

from __future__ import annotations

import csv
from pathlib import Path

from benchmark_full_real_business_algorithms import ROOT, TrajectoryResult


SOURCE_FILES = [
    ROOT / "docs" / "experiments" / "full_real_business_beam_portfolio_comparison.csv",
    ROOT / "docs" / "experiments" / "full_real_business_enhanced_beam.csv",
    ROOT / "docs" / "experiments" / "full_real_business_qubo_standalone_tuning.csv",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _num(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return 0.0 if value == "" else float(value)


def _make_portfolio(case: str, rows: list[dict[str, str]], solver: str, components: set[str]) -> dict[str, str]:
    case_rows = [row for row in rows if row["case"] == case and row["solver"] in components]
    if len(case_rows) != len(components):
        raise ValueError(f"missing components for {case}: {components}")
    best = min(
        case_rows,
        key=lambda row: (
            int(float(row["final_n"])),
            int(float(row["max_g"])),
            int(float(row["moves"])),
            float(row["runtime_sec"]),
        ),
    )
    out = dict(best)
    out["solver"] = solver
    out["runtime_sec"] = str(sum(_num(row, "runtime_sec") for row in case_rows))
    out["expanded_states"] = str(int(sum(_num(row, "expanded_states") for row in case_rows)))
    out["generated_states"] = str(int(sum(_num(row, "generated_states") for row in case_rows)))
    out["visited_states"] = str(int(sum(_num(row, "visited_states") for row in case_rows)))
    out["qubo_calls"] = str(int(sum(_num(row, "qubo_calls") for row in case_rows)))
    out["qubo_total_vars"] = str(int(sum(_num(row, "qubo_total_vars") for row in case_rows)))
    out["qubo_time_sec"] = str(sum(_num(row, "qubo_time_sec") for row in case_rows))
    out["status"] = f"FOCUSED_PORTFOLIO_SELECTED={best['solver']}; components={','.join(sorted(components))}"
    return out


def main() -> None:
    rows: list[dict[str, str]] = []
    for path in SOURCE_FILES:
        rows.extend(_read_rows(path))
    unique = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["case"], row["solver"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    rows = unique

    cases = sorted({row["case"] for row in rows})
    focused_rows = []
    for case in cases:
        focused_rows.append(_make_portfolio(case, rows, "beam_focused_plain", {"beam", "enhanced_beam_fast"}))
        focused_rows.append(_make_portfolio(case, rows, "beam_focused_with_qubo", {"beam", "qubo_beam", "enhanced_beam_fast"}))
        focused_rows.append(
            _make_portfolio(
                case,
                rows,
                "beam_focused_with_dense_qubo",
                {"beam", "enhanced_beam_fast", "qbo_beam_dense"},
            )
        )
        focused_rows.append(
            _make_portfolio(
                case,
                rows,
                "beam_focused_with_tuned_qubo",
                {"beam", "enhanced_beam_fast", "qubo_beam", "qbo_beam_dense"},
            )
        )
        focused_rows.append(
            _make_portfolio(
                case,
                rows,
                "beam_focused_with_qubo_portfolio",
                {"beam", "enhanced_beam_fast", "qbo_beam_standalone_portfolio"},
            )
        )

    combined = rows + focused_rows
    by_solver: dict[str, list[dict[str, str]]] = {}
    for row in combined:
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
    summary.sort(key=lambda item: (item["avg_final_n"], item["avg_moves"]))
    solvers = [item["solver"] for item in summary]
    lookup = {(row["case"], row["solver"]): row for row in combined}

    csv_path = ROOT / "docs" / "experiments" / "full_real_business_focused_beam_comparison.csv"
    md_path = ROOT / "docs" / "experiments" / "full_real_business_focused_beam_comparison.md"
    fields = list(TrajectoryResult.__annotations__.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in combined:
            writer.writerow({key: row.get(key, "") for key in fields})

    lines = [
        "# Full Real Business Focused Beam Comparison",
        "",
        "This report adds focused beam portfolios selected from the completed beam experiments. They are cheaper than the full beam portfolio and avoid the short-probe selection failure observed in adaptive racing.",
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
            "- `beam_focused_plain` runs only `beam` and `enhanced_beam_fast`.",
            "- `beam_focused_with_qubo` adds `qubo_beam` to recover cases where QUBO improves the standard beam.",
            "- `beam_focused_with_dense_qubo` replaces the original QUBO branch with the denser standalone QUBO configuration.",
            "- `beam_focused_with_tuned_qubo` keeps both original `qubo_beam` and `qbo_beam_dense`.",
            "- `beam_focused_with_qubo_portfolio` plugs the QUBO-only mini-portfolio into the original focused algorithm.",
            "- Runtime is conservatively summed across the selected components.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
