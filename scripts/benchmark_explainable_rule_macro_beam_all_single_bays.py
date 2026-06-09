"""Run explainable rule macro beam on all single-bay real cases."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam import (  # noqa: E402
    _auto_config,
    _write_explain_csv,
    run_explainable_rule_macro_beam,
)
from benchmark_full_real_business_algorithms import TrajectoryResult  # noqa: E402
from container_reshuffle.core import create_initial_state  # noqa: E402
from container_reshuffle.features import disorder_box_count, entropy_limit  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


def discover_single_bay_cases(input_dir: Path, bays: set[str]) -> list[tuple[str, str]]:
    pattern = re.compile(r"^([ABC][1-5])_(\d{2})_input\.txt$")
    cases: list[tuple[str, str]] = []
    for path in input_dir.glob("*_input.txt"):
        match = pattern.match(path.name)
        if match is None:
            continue
        zone, bay = match.group(1), match.group(2)
        if bay in bays:
            cases.append((zone, bay))
    return sorted(cases, key=lambda item: (item[0][0], int(item[0][1:]), item[1]))


def _write_summary(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrajectoryResult.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _print_summary(rows: list[TrajectoryResult], wall_time: float) -> None:
    solved = sum(1 for row in rows if row.final_n == 0)
    total_runtime = sum(row.runtime_sec for row in rows)
    best_effort = len(rows) - solved
    print(
        "SUMMARY "
        f"cases={len(rows)} solved={solved} best_effort={best_effort} "
        f"solver_runtime={total_runtime:.2f}s wall_time={wall_time:.2f}s",
        flush=True,
    )
    for row in rows:
        print(
            "SUMMARY_CASE "
            f"{row.case} finalN={row.final_n} moves={row.moves} "
            f"maxG={row.max_g} runtime={row.runtime_sec:.2f}s status={row.status}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=80.0)
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "explainable_rule_macro_beam_all_single_bays.csv",
    )
    parser.add_argument(
        "--moves-dir",
        type=Path,
        default=ROOT / "docs" / "experiments" / "explainable_rule_macro_beam_all_single_moves",
    )
    args = parser.parse_args()

    cases = discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))
    if not cases:
        print("NO_CASES_FOUND", flush=True)
        return 1

    rows: list[TrajectoryResult] = []
    start_all = time.perf_counter()
    for index, (zone, bay) in enumerate(cases, start=1):
        instance = _instance_from_single_bay(zone, bay)
        config, rule_sizes = _auto_config(instance, args.time_limit)
        initial = create_initial_state(instance)
        print(
            "CASE_START "
            f"{index}/{len(cases)} case={instance.instance_id} "
            f"N0={disorder_box_count(instance, initial)} cap={entropy_limit(instance, 1.3)} "
            f"auto_width={config.beam_width} rules={rule_sizes}",
            flush=True,
        )
        row, best = run_explainable_rule_macro_beam(instance.instance_id, instance, config, rule_sizes)
        rows.append(row)
        _write_explain_csv(instance, best.moves, args.moves_dir / f"{instance.instance_id}_moves.csv")
        _write_summary(rows, args.output)
        print(
            "CASE_DONE "
            f"{index}/{len(cases)} case={row.case} finalN={row.final_n} moves={row.moves} "
            f"maxG={row.max_g} runtime={row.runtime_sec:.2f}s",
            flush=True,
        )

    _print_summary(rows, time.perf_counter() - start_all)
    print(f"wrote {args.output}", flush=True)
    print(f"wrote moves to {args.moves_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
