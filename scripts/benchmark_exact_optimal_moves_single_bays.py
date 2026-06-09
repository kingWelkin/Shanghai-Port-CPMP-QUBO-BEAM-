"""Benchmark exact shortest-move proofs on split real single-bay cases."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from container_reshuffle.core import create_initial_state  # noqa: E402
from container_reshuffle.features import (  # noqa: E402
    disorder_box_count,
    disorder_trace,
    entropy_limit,
    entropy_trace,
)
from container_reshuffle.solvers.astar import ExactAStarSolver  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


FIELDS = [
    "case",
    "zone",
    "bay",
    "n0",
    "cap_13",
    "status",
    "feasible",
    "proof_shortest",
    "optimal_moves",
    "final_n",
    "g_end",
    "max_g",
    "g_cap_ok",
    "runtime_sec",
    "expanded",
    "generated",
    "visited",
]


def _run_one(zone: str, bay: str, time_limit: float, canonical: bool) -> dict[str, object]:
    instance = _instance_from_single_bay(zone, bay)
    initial = create_initial_state(instance)
    n0 = disorder_box_count(instance, initial)
    cap = entropy_limit(instance, 1.3)
    config = SolverConfig(
        max_depth=cap,
        time_limit_sec=time_limit,
        entropy_alpha=1.3,
        enable_canonical_hash=canonical,
    )
    sol = ExactAStarSolver().solve(instance, config)
    if sol.feasible:
        n_values = disorder_trace(instance, sol.moves)
        g_values = entropy_trace(instance, sol.moves)
        final_n = n_values[-1] if n_values else n0
        max_g = max(g_values) if g_values else n0
        g_end = len(sol.moves) + final_n
        moves = len(sol.moves)
        g_cap_ok = max_g <= cap
    else:
        final_n = ""
        max_g = ""
        g_end = ""
        moves = ""
        g_cap_ok = ""
    return {
        "case": instance.instance_id,
        "zone": zone,
        "bay": bay,
        "n0": n0,
        "cap_13": cap,
        "status": sol.status,
        "feasible": str(sol.feasible),
        "proof_shortest": str(sol.feasible and sol.optimal and sol.status == "OPTIMAL"),
        "optimal_moves": moves,
        "final_n": final_n,
        "g_end": g_end,
        "max_g": max_g,
        "g_cap_ok": str(g_cap_ok) if g_cap_ok != "" else "",
        "runtime_sec": f"{sol.runtime_sec:.6f}",
        "expanded": sol.expanded_states,
        "generated": sol.generated_states,
        "visited": sol.visited_states,
    }


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, object], key: str) -> float:
    value = row[key]
    return float(value) if value != "" else 0.0


def _write_report(rows: list[dict[str, object]], csv_path: Path, report_path: Path, wall_time: float) -> None:
    proven = [row for row in rows if row["proof_shortest"] == "True"]
    runtimes = [_float(row, "runtime_sec") for row in rows]
    moves = [_float(row, "optimal_moves") for row in proven]
    lines = [
        "# Exact Shortest-Move Proof Baseline",
        "",
        f"CSV: `{csv_path.resolve().relative_to(ROOT).as_posix()}`",
        "",
        "Scope: split real single-bay cases selected by the benchmark command.",
        "Solver: exact A* over legal relocations, with admissible remaining-disorder lower bound.",
        "A row has a shortest-move proof only when `proof_shortest=True` and `status=OPTIMAL`.",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| cases | {len(rows)} |",
        f"| proven shortest | {len(proven)}/{len(rows)} |",
        f"| wall time sec | {wall_time:.3f} |",
        f"| total solver runtime sec | {sum(runtimes):.3f} |",
        f"| avg runtime sec | {(sum(runtimes) / len(runtimes)) if runtimes else 0.0:.3f} |",
        f"| median runtime sec | {statistics.median(runtimes) if runtimes else 0.0:.3f} |",
        f"| avg optimal moves | {(sum(moves) / len(moves)) if moves else 0.0:.3f} |",
        "",
        "## Detail",
        "",
        "| case | n0 | cap_13 | proof shortest | optimal moves | runtime sec | expanded | generated | visited |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['case']} | {row['n0']} | {row['cap_13']} | {row['proof_shortest']} | "
            f"{row['optimal_moves']} | {float(row['runtime_sec']):.3f} | "
            f"{row['expanded']} | {row['generated']} | {row['visited']} |"
        )
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bays", nargs="+", default=["01"])
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--canonical", action="store_true", default=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "exact_optimal_moves_single_bays_01.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs" / "experiments" / "exact_optimal_moves_single_bays_01.md",
    )
    args = parser.parse_args()

    cases = discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))
    rows: list[dict[str, object]] = []
    start_all = time.perf_counter()
    for idx, (zone, bay) in enumerate(cases, start=1):
        print(f"CASE_START {idx}/{len(cases)} {zone}_{bay}_single", flush=True)
        row = _run_one(zone, bay, args.time_limit, args.canonical)
        rows.append(row)
        _write_csv(rows, args.output)
        print(
            "CASE_DONE "
            f"{idx}/{len(cases)} case={row['case']} status={row['status']} "
            f"proof={row['proof_shortest']} moves={row['optimal_moves']} runtime={row['runtime_sec']}s",
            flush=True,
        )

    wall_time = time.perf_counter() - start_all
    _write_csv(rows, args.output)
    _write_report(rows, args.output, args.report, wall_time)
    print(
        "SUMMARY "
        f"cases={len(rows)} proven={sum(1 for row in rows if row['proof_shortest'] == 'True')} "
        f"wall={wall_time:.3f}s output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
