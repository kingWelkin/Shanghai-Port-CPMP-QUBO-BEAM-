"""Run the standalone BS-FRG solver on open CPMP/DLTS instances."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_classical_cpmp_baselines import FIELDS, row_from_solution  # noqa: E402
from container_reshuffle.core import validate_instance  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.bs_frg import BsFrgSolver  # noqa: E402
from open_cpmp_benchmarks import discover_dlts_cases, load_dlts_dat  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "external" / "DLTS")
    parser.add_argument("--families", nargs="+", default=["5x5_cv1_test"])
    parser.add_argument("--limit-per-family", type=int, default=20)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--max-depth", type=int, default=160)
    parser.add_argument("--beam-width", type=int, default=1000)
    parser.add_argument("--k", type=int, default=3, help="candidate moves kept per beam node")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "bs_frg_open_cpmp.csv")
    return parser


def _is_open_valid(row: dict[str, object]) -> bool:
    return row.get("goal_validation_ok") in (True, "True") and row.get("final_n") == 0 and row.get("moves") != ""


def _write_summary(output: Path, rows: list[dict[str, object]]) -> None:
    solved = [row for row in rows if _is_open_valid(row)]
    moves = sum(int(row["moves"]) for row in solved)
    gap = sum(int(row["move_gap"]) for row in solved if row.get("move_gap") not in ("", None))
    runtime = sum(float(row["runtime_sec"]) for row in rows) / max(1, len(rows))
    invalid = [row for row in rows if not _is_open_valid(row)]
    lines = [
        "# BS-FRG Open CPMP Summary",
        "",
        f"Result CSV: `{output}`",
        "",
        "| solver | solved | moves_sum | gap_sum | avg_runtime_sec |",
        "|---|---:|---:|---:|---:|",
        f"| bs_frg | {len(solved)}/{len(rows)} | {moves if solved else ''} | {gap if solved else ''} | {runtime:.3f} |",
        "",
        "## Unsolved",
        "",
    ]
    if invalid:
        lines.extend(f"- `{row['case']}`: {row['status']}" for row in invalid)
    else:
        lines.append("- none")
    output.with_name(output.stem + "_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    cases = discover_dlts_cases(source, families=args.families, limit_per_family=args.limit_per_family)
    solver = BsFrgSolver()
    rows: list[dict[str, object]] = []

    for case in cases:
        instance = load_dlts_dat(case.instance_path, max_height=case.reference.max_height)
        try:
            validate_instance(instance)
            allow_priority_groups = False
        except ValueError:
            allow_priority_groups = True
        config = SolverConfig(
            max_depth=args.max_depth,
            beam_width=args.beam_width,
            candidate_limit_per_state=args.k,
            time_limit_sec=args.time_limit,
            entropy_alpha=10.0,
        )
        print(f"CASE_START {case.case_id}", flush=True)
        solution = solver.solve(instance, config)
        row = row_from_solution(
            instance,
            case.family,
            case.case_id,
            solution,
            reference_moves=case.reference.optimal_moves,
            allow_priority_groups=allow_priority_groups,
        )
        rows.append(row)
        print(
            f"CASE_DONE {case.case_id} final_n={row['final_n']} moves={row['moves']} "
            f"runtime={float(row['runtime_sec']):.3f}s status={row['status']}",
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(output, rows)
    print(f"WROTE {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
