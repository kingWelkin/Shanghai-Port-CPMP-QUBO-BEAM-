"""Run the two BF14 pure BS-FRG failures without a solver time limit."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_bs_frg_paper import load_bf_bay  # noqa: E402
from benchmark_classical_cpmp_baselines import FIELDS, row_from_solution  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.bs_frg import BsFrgSolver  # noqa: E402


CASES = [
    "cpmp_16_8_103_21_78_1",
    "cpmp_16_8_103_21_78_20",
]


def main() -> int:
    source = ROOT / "external" / "paper_cpmp" / "BF" / "BF" / "BF14"
    output = ROOT / "docs" / "experiments" / "bs_frg_pure_failed_bf14_unlimited_20260618.csv"
    solver = BsFrgSolver(paper_pure=True)
    config = SolverConfig(
        max_depth=1000,
        beam_width=1000,
        candidate_limit_per_state=3,
        time_limit_sec=0.0,
        entropy_alpha=10.0,
    )
    rows = []
    for case_id in CASES:
        instance = load_bf_bay(source / f"{case_id}.bay")
        print(f"CASE_START BF BF14/{case_id}", flush=True)
        solution = solver.solve(instance, config)
        row = row_from_solution(
            instance,
            "BF14",
            case_id,
            solution,
            reference_moves=None,
            allow_priority_groups=True,
        )
        rows.append(row)
        print(
            f"CASE_DONE BF BF14/{case_id} final_n={row['final_n']} moves={row['moves']} "
            f"runtime={float(row['runtime_sec']):.3f}s status={row['status']}",
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
