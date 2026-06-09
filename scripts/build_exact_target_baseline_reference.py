"""Build a compact exact target-search baseline reference.

The source exact run proves whether each split single-bay case can reach
``final_N <= target_final_n`` under the business ``moves + N <= cap_13`` bound.
For ``target_final_n=0``, a reachable row proves optimal final N, since N is
non-negative. The reported move count is the returned witness path length, not
a proof of minimum moves.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

OUT_FIELDS = [
    "case",
    "zone",
    "bay",
    "n0",
    "cap_13",
    "target_final_n",
    "status",
    "exact_final_n",
    "exact_moves_witness",
    "g_end",
    "g_cap_ok",
    "exact_runtime_sec",
    "exact_wall_sec",
    "expanded",
    "generated",
    "visited",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    return int(value) if value not in {"", None} else default


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else default


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _compact_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        final_n = _int(row, "final_n")
        moves = _int(row, "moves")
        cap = _int(row, "cap_13")
        g_end = moves + final_n
        out.append(
            {
                "case": row["case"],
                "zone": row["zone"],
                "bay": row["bay"],
                "n0": _int(row, "n0"),
                "cap_13": cap,
                "target_final_n": _int(row, "target_final_n"),
                "status": row["status"],
                "exact_final_n": final_n,
                "exact_moves_witness": moves,
                "g_end": g_end,
                "g_cap_ok": str(g_end <= cap),
                "exact_runtime_sec": _format_float(_float(row, "runtime_sec")),
                "exact_wall_sec": _format_float(_float(row, "wall_time_sec")),
                "expanded": _int(row, "expanded"),
                "generated": _int(row, "generated"),
                "visited": _int(row, "visited"),
            }
        )
    return out


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _runtime(row: dict[str, object]) -> float:
    return float(row["exact_runtime_sec"])


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _write_report(rows: list[dict[str, object]], source: Path, csv_path: Path, report_path: Path) -> None:
    solved = [row for row in rows if row["status"] == "TARGET_REACHABLE"]
    runtimes = [_runtime(row) for row in rows]
    moves = [float(row["exact_moves_witness"]) for row in solved]
    generated = [float(row["generated"]) for row in rows]
    visited = [float(row["visited"]) for row in rows]
    by_bay: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_bay.setdefault(str(row["bay"]), []).append(row)

    lines = [
        "# Exact Target Baseline Reference",
        "",
        f"Source: `{_display_path(source)}`",
        f"CSV: `{_display_path(csv_path)}`",
        "",
        "Scope: split single-bay real cases for bays `01` and `26`.",
        "Exact target: `final_N=0` under the business cap `moves + N <= cap_13`.",
        "",
        "Interpretation:",
        "",
        "- `exact_final_n=0` is an exact final-N optimum because `N` cannot be negative.",
        "- `exact_moves_witness` is the first target-reaching path returned by exact best-first search; it is not a shortest-move proof.",
        "- Use `case` as the join key when comparing heuristic, beam, QUBO, or portfolio results.",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| cases | {len(rows)} |",
        f"| target reachable | {len(solved)}/{len(rows)} |",
        f"| cap-feasible witnesses | {sum(1 for row in rows if row['g_cap_ok'] == 'True')}/{len(rows)} |",
        f"| total exact runtime sec | {_format_float(sum(runtimes))} |",
        f"| avg exact runtime sec | {_format_float(_avg(runtimes))} |",
        f"| median exact runtime sec | {_format_float(statistics.median(runtimes)) if runtimes else '0.000'} |",
        f"| max exact runtime sec | {_format_float(max(runtimes) if runtimes else 0.0)} |",
        f"| avg witness moves | {_format_float(_avg(moves))} |",
        f"| avg generated states | {_format_float(_avg(generated))} |",
        f"| avg visited states | {_format_float(_avg(visited))} |",
        "",
        "## By Bay",
        "",
        "| bay | cases | reachable | avg runtime sec | median runtime sec | avg witness moves |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for bay in sorted(by_bay):
        items = by_bay[bay]
        bay_runtimes = [_runtime(row) for row in items]
        bay_moves = [float(row["exact_moves_witness"]) for row in items]
        bay_solved = sum(1 for row in items if row["status"] == "TARGET_REACHABLE")
        lines.append(
            "| "
            f"{bay} | {len(items)} | {bay_solved}/{len(items)} | "
            f"{_format_float(_avg(bay_runtimes))} | "
            f"{_format_float(statistics.median(bay_runtimes))} | "
            f"{_format_float(_avg(bay_moves))} |"
        )

    lines.extend(
        [
            "",
            "## Hardest Cases By Runtime",
            "",
            "| case | n0 | cap_13 | exact moves witness | runtime sec | generated | visited |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=_runtime, reverse=True)[:10]:
        lines.append(
            "| "
            f"{row['case']} | {row['n0']} | {row['cap_13']} | "
            f"{row['exact_moves_witness']} | {row['exact_runtime_sec']} | "
            f"{row['generated']} | {row['visited']} |"
        )
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "docs" / "experiments" / "exact_target_bestfirst_all_single_bays_long.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "exact_target_baseline_reference.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs" / "experiments" / "exact_target_baseline_reference.md",
    )
    args = parser.parse_args()

    rows = _compact_rows(_read_rows(args.source))
    _write_csv(rows, args.output)
    _write_report(rows, args.source, args.output, args.report)
    print(f"WROTE csv={args.output}")
    print(f"WROTE report={args.report}")
    print(f"SUMMARY cases={len(rows)} reachable={sum(1 for row in rows if row['status'] == 'TARGET_REACHABLE')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
