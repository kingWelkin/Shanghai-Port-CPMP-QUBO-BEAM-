"""Fresh disk-backed target A*/best-first benchmark for N=0.

This is the fair-runtime counterpart to the historical target witness files:
every case starts from the initial layout, uses SQLite only for this run's
open/best/parent tables, stops when N=0 is reached, reconstructs the route from
the freshly written parent table, and includes search + reconstruction +
validation in the reported runtime.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam import _write_explain_csv  # noqa: E402
from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from benchmark_fast_qubo_integrated_v1 import _exact_moves_by_case, _move_gap, _parse_case, _rel  # noqa: E402
from benchmark_full_real_business_algorithms import TrajectoryResult  # noqa: E402
from container_reshuffle.core import Move, Solution, create_initial_state, validate_solution  # noqa: E402
from container_reshuffle.features import disorder_box_count, entropy_limit, max_entropy_along_moves  # noqa: E402
from extract_business_path_from_sqlite import _lift_to_physical  # noqa: E402
from run_business_exact_target_search_sqlite import (  # noqa: E402
    _encode,
    _remaining_lower_bound,
    _successors_with_n,
)
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


SOLVER_NAME = "target_astar_fresh_n0_v1"
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


@dataclass
class SearchResult:
    case: str
    status: str
    target_code: bytes | None
    moves: list[Move]
    runtime_sec: float
    expanded: int
    generated: int
    visited: int
    open_count: int
    pruned_bound: int
    pruned_business: int
    dominated: int
    db_path: Path


def _connect(db_path: Path, cache_mb: int) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(f"PRAGMA cache_size={-1024 * max(16, cache_mb)}")
    return conn


def _reset_db(conn: sqlite3.Connection, initial: bytes, n0: int, priority: float) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS best;
        DROP TABLE IF EXISTS open;
        DROP TABLE IF EXISTS candidate;
        CREATE TABLE best(
            code BLOB PRIMARY KEY,
            n INTEGER NOT NULL,
            depth INTEGER NOT NULL,
            parent BLOB
        ) WITHOUT ROWID;
        CREATE TABLE open(
            code BLOB PRIMARY KEY,
            n INTEGER NOT NULL,
            depth INTEGER NOT NULL,
            priority REAL NOT NULL
        ) WITHOUT ROWID;
        CREATE TEMP TABLE candidate(
            code BLOB PRIMARY KEY,
            n INTEGER NOT NULL,
            depth INTEGER NOT NULL,
            priority REAL NOT NULL,
            parent BLOB NOT NULL
        ) WITHOUT ROWID;
        """
    )
    conn.execute(
        "INSERT INTO best(code, n, depth, parent) VALUES (?, ?, 0, NULL)",
        (sqlite3.Binary(initial), n0),
    )
    conn.execute(
        "INSERT INTO open(code, n, depth, priority) VALUES (?, ?, 0, ?)",
        (sqlite3.Binary(initial), n0, priority),
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_open_order ON open(priority, n, depth, code)")
    conn.commit()


def _priority(n_value: int, depth: int, args: argparse.Namespace) -> float:
    if args.priority == "astar":
        return float(depth + n_value)
    if args.priority == "weighted_astar":
        return float(args.g_weight * depth + args.n_weight * n_value)
    return float(n_value * 1_000_000 + depth)


def _select_batch(conn: sqlite3.Connection, batch_size: int) -> list[tuple[bytes, int, int]]:
    rows = conn.execute(
        """
        SELECT o.code, o.n, o.depth
        FROM open o
        JOIN best b ON b.code = o.code AND b.depth = o.depth
        ORDER BY o.priority, o.n, o.depth, o.code
        LIMIT ?
        """,
        (batch_size,),
    ).fetchall()
    if rows:
        return [(bytes(code), int(n_value), int(depth)) for code, n_value, depth in rows]
    conn.execute("DELETE FROM open WHERE EXISTS (SELECT 1 FROM best b WHERE b.code = open.code AND b.depth < open.depth)")
    conn.commit()
    rows = conn.execute(
        """
        SELECT o.code, o.n, o.depth
        FROM open o
        JOIN best b ON b.code = o.code AND b.depth = o.depth
        ORDER BY o.priority, o.n, o.depth, o.code
        LIMIT ?
        """,
        (batch_size,),
    ).fetchall()
    return [(bytes(code), int(n_value), int(depth)) for code, n_value, depth in rows]


def _delete_open_rows(conn: sqlite3.Connection, codes: list[bytes]) -> None:
    if not codes:
        return
    conn.executemany("DELETE FROM open WHERE code = ?", ((sqlite3.Binary(code),) for code in codes))
    conn.commit()


def _flush_candidates(conn: sqlite3.Connection, candidates: dict[bytes, tuple[int, int, float, bytes]]) -> tuple[int, int]:
    if not candidates:
        return 0, 0
    conn.execute("DELETE FROM candidate")
    conn.executemany(
        "INSERT INTO candidate(code, n, depth, priority, parent) VALUES (?, ?, ?, ?, ?)",
        (
            (sqlite3.Binary(code), n_value, depth, priority, sqlite3.Binary(parent))
            for code, (n_value, depth, priority, parent) in candidates.items()
        ),
    )
    before = conn.total_changes
    conn.execute(
        """
        UPDATE open
        SET
            n = (SELECT c.n FROM candidate c WHERE c.code = open.code),
            depth = (SELECT c.depth FROM candidate c WHERE c.code = open.code),
            priority = (SELECT c.priority FROM candidate c WHERE c.code = open.code)
        WHERE EXISTS (
            SELECT 1 FROM candidate c
            WHERE c.code = open.code AND c.depth < open.depth
        )
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO open(code, n, depth, priority)
        SELECT c.code, c.n, c.depth, c.priority
        FROM candidate c
        LEFT JOIN best b ON b.code = c.code
        WHERE b.code IS NULL OR c.depth < b.depth
        """
    )
    opened_or_updated = conn.total_changes - before

    before = conn.total_changes
    conn.execute(
        """
        UPDATE best
        SET
            n = (SELECT c.n FROM candidate c WHERE c.code = best.code),
            depth = (SELECT c.depth FROM candidate c WHERE c.code = best.code),
            parent = (SELECT c.parent FROM candidate c WHERE c.code = best.code)
        WHERE EXISTS (
            SELECT 1 FROM candidate c
            WHERE c.code = best.code AND c.depth < best.depth
        )
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO best(code, n, depth, parent)
        SELECT c.code, c.n, c.depth, c.parent
        FROM candidate c
        """
    )
    best_inserted_or_updated = conn.total_changes - before
    conn.commit()
    candidates.clear()
    return opened_or_updated, best_inserted_or_updated


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _insert_target_parent(conn: sqlite3.Connection, target: bytes, n_value: int, depth: int, parent: bytes) -> None:
    conn.execute(
        """
        INSERT INTO best(code, n, depth, parent) VALUES (?, ?, ?, ?)
        ON CONFLICT(code) DO UPDATE SET
            n = excluded.n,
            depth = excluded.depth,
            parent = excluded.parent
        WHERE excluded.depth <= best.depth
        """,
        (sqlite3.Binary(target), n_value, depth, sqlite3.Binary(parent)),
    )
    conn.commit()


def _backtrack_codes(conn: sqlite3.Connection, target_code: bytes) -> list[bytes]:
    codes = [target_code]
    current = target_code
    while True:
        row = conn.execute("SELECT parent, depth FROM best WHERE code = ?", (sqlite3.Binary(current),)).fetchone()
        if row is None:
            raise RuntimeError("target code missing from parent table")
        parent, depth = row
        if parent is None or int(depth) == 0:
            break
        current = bytes(parent)
        codes.append(current)
    codes.reverse()
    return codes


def _moves_from_codes(zone: str, bay: str, codes: list[bytes]) -> list[Move]:
    instance = _instance_from_single_bay(zone, bay)
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


def _search_case(case: str, zone: str, bay: str, db_path: Path, args: argparse.Namespace) -> SearchResult:
    start = time.perf_counter()
    deadline = start + args.time_limit if args.time_limit > 0 else float("inf")
    instance = _instance_from_single_bay(zone, bay)
    initial, n0 = _encode(instance.stacks, instance.stack_count, instance.max_height)
    cap = entropy_limit(instance, args.alpha)
    depth_cap = min(cap, args.max_moves) if args.max_moves is not None else cap
    conn = _connect(db_path, args.cache_mb)
    _reset_db(conn, initial, n0, _priority(n0, 0, args))

    expanded = generated = pruned_bound = pruned_business = dominated = opened_total = 0
    visited_count = 1
    open_count = 1
    candidates: dict[bytes, tuple[int, int, float, bytes]] = {}
    target_code: bytes | None = initial if n0 <= args.target_final_n else None

    if target_code is None:
        while time.perf_counter() < deadline:
            rows = _select_batch(conn, args.pop_batch_size)
            if not rows:
                break
            processed: list[bytes] = []
            for code, current_n, depth in rows:
                processed.append(code)
                if time.perf_counter() >= deadline:
                    break
                if current_n <= args.target_final_n:
                    target_code = code
                    break
                if _remaining_lower_bound(current_n, depth, depth_cap) > args.target_final_n:
                    pruned_bound += 1
                    continue
                if depth >= depth_cap:
                    pruned_bound += 1
                    continue

                expanded += 1
                next_depth = depth + 1
                for nxt, nxt_n in _successors_with_n(code, current_n, instance.stack_count, instance.max_height):
                    generated += 1
                    if next_depth + nxt_n > cap:
                        pruned_business += 1
                        continue
                    if nxt_n <= args.target_final_n:
                        _flush_candidates(conn, candidates)
                        _insert_target_parent(conn, nxt, nxt_n, next_depth, code)
                        target_code = nxt
                        break
                    if _remaining_lower_bound(nxt_n, next_depth, depth_cap) > args.target_final_n:
                        pruned_bound += 1
                        continue
                    existing = candidates.get(nxt)
                    if existing is None or next_depth < existing[1]:
                        candidates[nxt] = (nxt_n, next_depth, _priority(nxt_n, next_depth, args), code)
                    else:
                        dominated += 1
                    if len(candidates) >= args.insert_batch_size:
                        opened, best_changed = _flush_candidates(conn, candidates)
                        opened_total += opened
                        visited_count += best_changed
                        open_count = _table_count(conn, "open")
                if target_code is not None:
                    break

            _delete_open_rows(conn, processed)
            opened, best_changed = _flush_candidates(conn, candidates)
            opened_total += opened
            visited_count += best_changed
            open_count = _table_count(conn, "open")
            if target_code is not None:
                break

    moves: list[Move] = []
    status = "TARGET_REACHABLE" if target_code is not None else "TIME_LIMIT_OR_EXHAUSTED"
    if target_code is not None:
        codes = _backtrack_codes(conn, target_code)
        moves = _moves_from_codes(zone, bay, codes)
    runtime = time.perf_counter() - start
    open_count = _table_count(conn, "open")
    visited_count = _table_count(conn, "best")
    conn.close()
    return SearchResult(
        case=case,
        status=status,
        target_code=target_code,
        moves=moves,
        runtime_sec=runtime,
        expanded=expanded,
        generated=generated,
        visited=visited_count,
        open_count=open_count,
        pruned_bound=pruned_bound,
        pruned_business=pruned_business,
        dominated=dominated,
        db_path=db_path,
    )


def _case_filter(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return [case if case.endswith("_single") else f"{case}_single" for case in args.cases]
    return [f"{zone}_{bay}_single" for zone, bay in discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))]


def _row_from_result(result: SearchResult, zone: str, bay: str, args: argparse.Namespace) -> TrajectoryResult:
    instance = _instance_from_single_bay(zone, bay)
    initial = create_initial_state(instance)
    n0 = disorder_box_count(instance, initial)
    if result.moves:
        from container_reshuffle.core import apply_move

        state = initial
        for move in result.moves:
            state = apply_move(instance, state, move)
        final_n = disorder_box_count(instance, state)
        moves_count = len(result.moves)
        max_g = max_entropy_along_moves(instance, result.moves)
    else:
        final_n = n0
        moves_count = None
        max_g = n0
    cap13 = entropy_limit(instance, 1.3)
    cap15 = entropy_limit(instance, 1.5)
    return TrajectoryResult(
        case=result.case,
        solver=SOLVER_NAME,
        n0=n0,
        cap_13=cap13,
        cap_15=cap15,
        moves=moves_count,
        final_n=final_n,
        reduction=n0 - final_n,
        g_end=(moves_count + final_n) if moves_count is not None else None,
        max_g=max_g,
        g_ok_13=max_g <= cap13,
        g_ok_15=max_g <= cap15,
        runtime_sec=result.runtime_sec,
        expanded_states=result.expanded,
        generated_states=result.generated,
        visited_states=result.visited,
        status=(
            f"{result.status}; solver={SOLVER_NAME}; fresh_run=True; db_path={_rel(result.db_path)}; "
            f"priority={args.priority}; target_stop=N0; proof=not_optimality_proof; "
            f"open={result.open_count}; pruned_bound={result.pruned_bound}; "
            f"pruned_business={result.pruned_business}; dominated={result.dominated}"
        ),
    )


def _validate_result(result: SearchResult, zone: str, bay: str) -> tuple[bool, bool, str]:
    if not result.moves:
        return False, False, result.status
    instance = _instance_from_single_bay(zone, bay)
    solution = Solution(
        solver_name=SOLVER_NAME,
        feasible=True,
        optimal=False,
        objective=float(len(result.moves)),
        runtime_sec=result.runtime_sec,
        moves=result.moves,
        status=result.status,
    )
    goal_ok, goal_msg, _ = validate_solution(instance, solution)
    business_ok = max_entropy_along_moves(instance, result.moves) <= entropy_limit(instance, 1.3)
    return business_ok, goal_ok, goal_msg


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
    moves_paths: dict[str, Path],
    exact_refs: dict[str, int],
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
                "moves_path": _rel(moves_paths.get(row.case, Path(""))),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ACCEPTANCE_FIELDS)
        writer.writeheader()
        writer.writerows(out)


def _write_report(rows: list[TrajectoryResult], output: Path, acceptance: Path, report: Path, exact_refs: dict[str, int]) -> None:
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
        "Fresh disk-backed target A*/best-first benchmark. No historical witness reuse.",
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


def _output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    output = args.output or args.output_dir / f"target_astar_fresh_n0_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    if not output.is_absolute():
        output = ROOT / output
    acceptance = output.with_name(output.stem + "_acceptance.csv")
    moves_dir = output.with_name(output.stem + "_moves")
    report = output.with_suffix(".md")
    db_dir = args.db_dir if args.db_dir.is_absolute() else ROOT / args.db_dir
    return output, acceptance, moves_dir, report, db_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "experiments")
    parser.add_argument("--db-dir", type=Path, default=ROOT / "tmp" / "target_astar_fresh_n0")
    parser.add_argument("--priority", choices=["n_then_depth", "astar", "weighted_astar"], default="n_then_depth")
    parser.add_argument("--g-weight", type=float, default=1.0)
    parser.add_argument("--n-weight", type=float, default=1.0)
    parser.add_argument("--target-final-n", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--time-limit", type=float, default=0.0, help="Per-case limit in seconds; 0 means no limit.")
    parser.add_argument("--cache-mb", type=int, default=512)
    parser.add_argument("--pop-batch-size", type=int, default=5000)
    parser.add_argument("--insert-batch-size", type=int, default=50000)
    args = parser.parse_args()

    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir

    output, acceptance, moves_dir, report, db_dir = _output_paths(args)
    moves_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)
    cases = _case_filter(args)
    exact_refs = _exact_moves_by_case()
    rows: list[TrajectoryResult] = []
    validation: dict[str, tuple[bool, bool, str]] = {}
    moves_paths: dict[str, Path] = {}

    for idx, case in enumerate(cases, start=1):
        zone, bay = _parse_case(case)
        db_path = db_dir / f"{case}_target{args.target_final_n}_{args.priority}.sqlite"
        print(f"CASE_START {idx}/{len(cases)} {case} solver={SOLVER_NAME} priority={args.priority}", flush=True)
        result = _search_case(case, zone, bay, db_path, args)
        row = _row_from_result(result, zone, bay, args)
        business_ok, goal_ok, msg = _validate_result(result, zone, bay)
        if result.moves:
            move_path = moves_dir / f"{case}_moves.csv"
            _write_explain_csv(_instance_from_single_bay(zone, bay), result.moves, move_path)
            moves_paths[case] = move_path
        rows.append(row)
        validation[case] = (business_ok, goal_ok, msg)
        _write_rows(rows, output)
        _write_acceptance(rows, validation, moves_paths, exact_refs, acceptance)
        _write_report(rows, output, acceptance, report, exact_refs)
        print(
            "CASE_DONE "
            f"{case} status={result.status} final_n={row.final_n} moves={row.moves} "
            f"runtime={row.runtime_sec:.3f}s valid={business_ok and goal_ok}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
