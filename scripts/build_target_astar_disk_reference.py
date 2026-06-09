"""Build a validated target-search A* reference table from SQLite witnesses.

The source run is the disk-backed target best-first/A*-style search that stops
when N=0 is first reached.  This helper does not rerun the expensive search and
does not write to the D: database directory; it reads the recorded SQLite
visited sets and found-code witnesses, reconstructs physical move routes, then
writes standard result/acceptance/moves artifacts under docs/experiments.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam import _write_explain_csv  # noqa: E402
from benchmark_fast_qubo_integrated_v1 import _move_gap, _parse_case, _rel  # noqa: E402
from benchmark_full_real_business_algorithms import TrajectoryResult  # noqa: E402
from container_reshuffle.core import Move, Solution, validate_solution  # noqa: E402
from container_reshuffle.features import disorder_box_count, entropy_limit, max_entropy_along_moves  # noqa: E402
from extract_business_path_from_sqlite import (  # noqa: E402
    _backtrack_codes,
    _lift_to_physical,
    _read_found,
)
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


SOLVER_NAME = "target_astar_disk_n0_reference"
RESULT_FIELDS = list(TrajectoryResult.__annotations__.keys())
ACCEPTANCE_FIELDS = [
    "case",
    "selected_solver",
    "final_n",
    "moves",
    "runtime_sec",
    "exact_moves",
    "move_gap",
    "qubo_used",
    "selected_reason",
    "business_ok",
    "goal_validation_ok",
    "validation_msg",
    "moves_path",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _moves_from_witness(zone: str, bay: str, db_path: Path, found_code_path: Path) -> list[Move]:
    import sqlite3

    instance = _instance_from_single_bay(zone, bay)
    target_depth, target_code, parent_depth, parent_code = _read_found(found_code_path)
    conn = sqlite3.connect(str(db_path))
    try:
        codes = _backtrack_codes(
            conn,
            target_code,
            target_depth,
            parent_code,
            parent_depth,
            instance.stack_count,
            instance.max_height,
        )
    finally:
        conn.close()
    physical_rows = _lift_to_physical(instance.stacks, codes, instance.stack_count, instance.max_height)
    return [
        Move(
            step=int(row["step"]) - 1,
            box=int(row["box_rank"]),
            from_stack=int(row["from_stack"]),
            to_stack=int(row["to_stack"]),
            from_level=int(row["from_level"]),
            to_level=int(row["to_level"]),
        )
        for row in physical_rows
    ]


def _row_from_moves(case: str, zone: str, bay: str, moves: list[Move], runtime_sec: float, source_status: str) -> TrajectoryResult:
    instance = _instance_from_single_bay(zone, bay)
    state = None
    from container_reshuffle.core import apply_move, create_initial_state

    state = create_initial_state(instance)
    for move in moves:
        state = apply_move(instance, state, move)
    n0 = disorder_box_count(instance, create_initial_state(instance))
    final_n = disorder_box_count(instance, state)
    cap13 = entropy_limit(instance, 1.3)
    cap15 = entropy_limit(instance, 1.5)
    max_g = max_entropy_along_moves(instance, moves)
    return TrajectoryResult(
        case=case,
        solver=SOLVER_NAME,
        n0=n0,
        cap_13=cap13,
        cap_15=cap15,
        moves=len(moves),
        final_n=final_n,
        reduction=n0 - final_n,
        g_end=len(moves) + final_n,
        max_g=max_g,
        g_ok_13=max_g <= cap13,
        g_ok_15=max_g <= cap15,
        runtime_sec=runtime_sec,
        status=(
            f"{source_status}; solver={SOLVER_NAME}; source=disk_target_bestfirst; "
            "target_stop=N0; proof=not_optimality_proof"
        ),
    )


def _write_rows(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_acceptance(
    rows: list[TrajectoryResult],
    validation: dict[str, tuple[bool, bool, str]],
    exact_refs: dict[str, int],
    moves_paths: dict[str, Path],
    path: Path,
) -> None:
    out = []
    for row in rows:
        business_ok, goal_ok, msg = validation[row.case]
        out.append(
            {
                "case": row.case,
                "selected_solver": row.solver,
                "final_n": row.final_n,
                "moves": row.moves,
                "runtime_sec": f"{row.runtime_sec:.6f}",
                "exact_moves": exact_refs.get(row.case, ""),
                "move_gap": _move_gap(row, exact_refs),
                "qubo_used": "False",
                "selected_reason": row.status,
                "business_ok": str(business_ok),
                "goal_validation_ok": str(goal_ok),
                "validation_msg": msg,
                "moves_path": _rel(moves_paths[row.case]),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ACCEPTANCE_FIELDS)
        writer.writeheader()
        writer.writerows(out)


def _write_report(rows: list[TrajectoryResult], acceptance: Path, output: Path, report: Path, exact_refs: dict[str, int]) -> None:
    solved = sum(1 for row in rows if row.final_n == 0)
    avg_runtime = sum(row.runtime_sec for row in rows) / len(rows) if rows else 0.0
    gaps = [
        row.moves - exact_refs[row.case]
        for row in rows
        if row.final_n == 0 and row.moves is not None and row.case in exact_refs
    ]
    lines = [
        f"# {SOLVER_NAME}",
        "",
        "Validated N=0 target-search reference reconstructed from disk SQLite witnesses.",
        "",
        f"Result CSV: `{_rel(output)}`",
        f"Acceptance CSV: `{_rel(acceptance)}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| cases | {len(rows)} |",
        f"| N=0 solved | {solved}/{len(rows)} |",
        f"| avg runtime sec | {avg_runtime:.3f} |",
        f"| max runtime sec | {max((row.runtime_sec for row in rows), default=0.0):.3f} |",
        f"| avg move gap | {sum(gaps) / len(gaps) if gaps else 0.0:.3f} |",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "docs" / "experiments" / "exact_target_bestfirst_all_single_bays_long.csv",
    )
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "target_astar_disk_reference.csv")
    args = parser.parse_args()

    source_rows = _read_rows(args.source)
    wanted = {case if case.endswith("_single") else f"{case}_single" for case in args.cases} if args.cases else None
    source_rows = [row for row in source_rows if wanted is None or row["case"] in wanted]

    output = args.output if args.output.is_absolute() else ROOT / args.output
    acceptance = output.with_name(output.stem + "_acceptance.csv")
    moves_dir = output.with_name(output.stem + "_moves")
    report = output.with_suffix(".md")
    moves_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[TrajectoryResult] = []
    validation: dict[str, tuple[bool, bool, str]] = {}
    exact_refs: dict[str, int] = {}
    moves_paths: dict[str, Path] = {}

    for idx, source in enumerate(source_rows, start=1):
        case = source["case"]
        if source.get("status") != "TARGET_REACHABLE":
            print(f"CASE_SKIP {idx}/{len(source_rows)} {case} status={source.get('status')}", flush=True)
            continue
        zone, bay = _parse_case(case)
        db_path = Path(source["db_path"])
        found_code_path = Path(source["log_path"]).with_name(Path(source["log_path"]).stem + "_found_code.txt")
        print(f"CASE_START {idx}/{len(source_rows)} {case}", flush=True)
        start = time.perf_counter()
        moves = _moves_from_witness(zone, bay, db_path, found_code_path)
        instance = _instance_from_single_bay(zone, bay)
        solution = Solution(
            solver_name=SOLVER_NAME,
            feasible=True,
            optimal=False,
            objective=float(len(moves)),
            runtime_sec=time.perf_counter() - start,
            moves=moves,
            status="TARGET_REACHABLE",
        )
        goal_ok, goal_msg, _ = validate_solution(instance, solution)
        max_g = max_entropy_along_moves(instance, moves)
        business_ok = max_g <= entropy_limit(instance, 1.3)
        row = _row_from_moves(case, zone, bay, moves, float(source["runtime_sec"]), source["status"])
        move_path = moves_dir / f"{case}_moves.csv"
        _write_explain_csv(instance, moves, move_path)
        result_rows.append(row)
        validation[case] = (business_ok, goal_ok, goal_msg if goal_ok else goal_msg)
        exact_refs[case] = int(float(source["moves"]))
        moves_paths[case] = move_path
        _write_rows(result_rows, output)
        _write_acceptance(result_rows, validation, exact_refs, moves_paths, acceptance)
        _write_report(result_rows, acceptance, output, report, exact_refs)
        print(
            f"CASE_DONE {case} moves={len(moves)} final_n={row.final_n} "
            f"business_ok={business_ok} goal_ok={goal_ok}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
