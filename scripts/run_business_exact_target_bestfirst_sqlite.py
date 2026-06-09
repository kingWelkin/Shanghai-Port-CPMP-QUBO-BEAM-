"""Disk-backed exact best-first target search for business final N(Future)."""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from container_reshuffle.features import entropy_limit  # noqa: E402
from run_business_exact_target_search_sqlite import (  # noqa: E402
    _connect,
    _encode,
    _remaining_lower_bound,
    _successors_with_n,
    _table_count,
)
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402

Code = bytes


def _reset_for_resume(conn: sqlite3.Connection, resume_depth: int) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS next_frontier;
        DROP TABLE IF EXISTS open;
        DROP TABLE IF EXISTS candidate;
        CREATE TABLE IF NOT EXISTS visited(code BLOB PRIMARY KEY, depth INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS frontier(code BLOB PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE open(code BLOB PRIMARY KEY, n INTEGER NOT NULL, depth INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TEMP TABLE candidate(code BLOB PRIMARY KEY, n INTEGER NOT NULL, depth INTEGER NOT NULL) WITHOUT ROWID;
        """
    )
    conn.execute("DELETE FROM visited WHERE depth > ?", (resume_depth,))
    conn.execute("INSERT INTO open(code, n, depth) SELECT code, n, ? FROM frontier", (resume_depth,))
    conn.execute("CREATE INDEX IF NOT EXISTS idx_open_order ON open(n, depth, code)")
    conn.commit()


def _reset_fresh(conn: sqlite3.Connection, initial: Code, n0: int) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS visited;
        DROP TABLE IF EXISTS frontier;
        DROP TABLE IF EXISTS next_frontier;
        DROP TABLE IF EXISTS open;
        DROP TABLE IF EXISTS candidate;
        CREATE TABLE visited(code BLOB PRIMARY KEY, depth INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE open(code BLOB PRIMARY KEY, n INTEGER NOT NULL, depth INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TEMP TABLE candidate(code BLOB PRIMARY KEY, n INTEGER NOT NULL, depth INTEGER NOT NULL) WITHOUT ROWID;
        """
    )
    conn.execute("INSERT INTO visited(code, depth) VALUES (?, 0)", (sqlite3.Binary(initial),))
    conn.execute("INSERT INTO open(code, n, depth) VALUES (?, ?, 0)", (sqlite3.Binary(initial), n0))
    conn.execute("CREATE INDEX IF NOT EXISTS idx_open_order ON open(n, depth, code)")
    conn.commit()


def _select_batch(conn: sqlite3.Connection, batch_size: int) -> list[tuple[bytes, int, int]]:
    rows = conn.execute(
        """
        SELECT o.code, o.n, o.depth
        FROM open o
        JOIN visited v ON v.code = o.code AND v.depth = o.depth
        ORDER BY o.n, o.depth, o.code
        LIMIT ?
        """,
        (batch_size,),
    ).fetchall()
    if not rows:
        conn.execute(
            "DELETE FROM open WHERE EXISTS (SELECT 1 FROM visited v WHERE v.code = open.code AND v.depth < open.depth)"
        )
        conn.commit()
        rows = conn.execute(
            """
            SELECT o.code, o.n, o.depth
            FROM open o
            JOIN visited v ON v.code = o.code AND v.depth = o.depth
            ORDER BY o.n, o.depth, o.code
            LIMIT ?
            """,
            (batch_size,),
        ).fetchall()
    return [(bytes(code), int(n_value), int(depth)) for code, n_value, depth in rows]


def _delete_open_rows(conn: sqlite3.Connection, codes: list[Code]) -> None:
    if not codes:
        return
    conn.executemany("DELETE FROM open WHERE code = ?", ((sqlite3.Binary(code),) for code in codes))
    conn.commit()


def _flush_candidates(conn: sqlite3.Connection, candidates: dict[Code, tuple[int, int]]) -> int:
    if not candidates:
        return 0
    conn.execute("DELETE FROM candidate")
    conn.executemany(
        "INSERT INTO candidate(code, n, depth) VALUES (?, ?, ?)",
        ((sqlite3.Binary(code), n_value, depth) for code, (n_value, depth) in candidates.items()),
    )
    conn.execute(
        """
        UPDATE open
        SET
            n = (SELECT c.n FROM candidate c WHERE c.code = open.code),
            depth = (SELECT c.depth FROM candidate c WHERE c.code = open.code)
        WHERE EXISTS (
            SELECT 1 FROM candidate c
            WHERE c.code = open.code AND c.depth < open.depth
        )
        """
    )
    before = conn.total_changes
    conn.execute(
        """
        INSERT OR IGNORE INTO open(code, n, depth)
        SELECT c.code, c.n, c.depth
        FROM candidate c
        LEFT JOIN visited v ON v.code = c.code
        WHERE v.code IS NULL OR c.depth < v.depth
        """
    )
    opened = conn.total_changes - before
    conn.execute(
        """
        UPDATE visited
        SET depth = (SELECT c.depth FROM candidate c WHERE c.code = visited.code)
        WHERE EXISTS (
            SELECT 1 FROM candidate c
            WHERE c.code = visited.code AND c.depth < visited.depth
        )
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO visited(code, depth)
        SELECT c.code, c.depth
        FROM candidate c
        """
    )
    conn.commit()
    candidates.clear()
    return opened


def run(
    zone: str,
    bay: str,
    target_final_n: int,
    alpha: float,
    db_path: Path,
    cache_mb: int,
    pop_batch_size: int,
    insert_batch_size: int,
    progress_interval: float,
    resume_depth: int | None,
    continue_open: bool,
    found_code_path: Path | None,
    max_moves: int | None,
) -> int:
    instance = _instance_from_single_bay(zone, bay)
    initial, n0 = _encode(instance.stacks, instance.stack_count, instance.max_height)
    cap = entropy_limit(instance, alpha)
    depth_cap = min(cap, int(max_moves)) if max_moves is not None else cap
    start = time.perf_counter()
    next_progress = start

    conn = _connect(db_path, cache_mb)
    if continue_open:
        conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS candidate(code BLOB PRIMARY KEY, n INTEGER NOT NULL, depth INTEGER NOT NULL) WITHOUT ROWID"
        )
        conn.commit()
    elif resume_depth is None:
        _reset_fresh(conn, initial, n0)
    else:
        _reset_for_resume(conn, int(resume_depth))

    visited_count = _table_count(conn, "visited")
    open_count = _table_count(conn, "open")
    print(
        f"START case={instance.instance_id} target_final_N={target_final_n} "
        f"stacks={instance.stack_count} H={instance.max_height} containers={instance.container_count} "
        f"initial_N={n0} cap={cap} depth_cap={depth_cap} db={db_path} resume_depth={resume_depth} "
        f"continue_open={continue_open} visited={visited_count} open={open_count}",
        flush=True,
    )

    if n0 <= target_final_n:
        if found_code_path is not None:
            found_code_path.parent.mkdir(parents=True, exist_ok=True)
            found_code_path.write_text(f"depth=0 n={n0} code={initial.hex()}\n", encoding="utf-8")
        print(
            f"FINAL status=TARGET_REACHABLE target_final_N={target_final_n} moves=0 final_N={n0} "
            f"expanded=0 generated=0 visited={visited_count} open={open_count} runtime={time.perf_counter() - start:.3f}s",
            flush=True,
        )
        return 0

    expanded = generated = pruned_bound = pruned_business = dominated = opened_total = 0
    candidates: dict[Code, tuple[int, int]] = {}

    while True:
        rows = _select_batch(conn, pop_batch_size)
        if not rows:
            break
        processed: list[Code] = []
        for code, current_n, depth in rows:
            processed.append(code)
            now = time.perf_counter()
            if now >= next_progress:
                print(
                    "PROGRESS "
                    f"elapsed={now - start:.1f}s min_batch_N={current_n} depth={depth} open={open_count} "
                    f"expanded={expanded} generated={generated} visited={visited_count} opened={opened_total} "
                    f"pruned_bound={pruned_bound} pruned_business={pruned_business} dominated={dominated}",
                    flush=True,
                )
                next_progress = now + progress_interval

            if current_n <= target_final_n:
                _delete_open_rows(conn, processed)
                opened_total += _flush_candidates(conn, candidates)
                if found_code_path is not None:
                    found_code_path.parent.mkdir(parents=True, exist_ok=True)
                    found_code_path.write_text(f"depth={depth} n={current_n} code={code.hex()}\n", encoding="utf-8")
                print(
                    "FINAL "
                    f"status=TARGET_REACHABLE target_final_N={target_final_n} moves={depth} final_N={current_n} "
                    f"expanded={expanded} generated={generated} visited={visited_count} open={open_count} "
                    f"pruned_bound={pruned_bound} pruned_business={pruned_business} dominated={dominated} "
                    f"runtime={time.perf_counter() - start:.3f}s",
                    flush=True,
                )
                return 0
            if _remaining_lower_bound(current_n, depth, depth_cap) > target_final_n:
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
                if nxt_n <= target_final_n:
                    _delete_open_rows(conn, processed)
                    opened_total += _flush_candidates(conn, candidates)
                    if found_code_path is not None:
                        found_code_path.parent.mkdir(parents=True, exist_ok=True)
                        found_code_path.write_text(
                            f"depth={next_depth} n={nxt_n} code={nxt.hex()} parent_depth={depth} parent={code.hex()}\n",
                            encoding="utf-8",
                        )
                    print(
                        "FINAL "
                        f"status=TARGET_REACHABLE target_final_N={target_final_n} moves={next_depth} final_N={nxt_n} "
                        f"expanded={expanded} generated={generated} visited={visited_count} open={open_count} "
                        f"pruned_bound={pruned_bound} pruned_business={pruned_business} dominated={dominated} "
                        f"runtime={time.perf_counter() - start:.3f}s",
                        flush=True,
                    )
                    return 0
                if _remaining_lower_bound(nxt_n, next_depth, depth_cap) > target_final_n:
                    pruned_bound += 1
                    continue
                previous = candidates.get(nxt)
                if previous is None or next_depth < previous[1]:
                    candidates[nxt] = (nxt_n, next_depth)
                else:
                    dominated += 1
                if len(candidates) >= insert_batch_size:
                    opened = _flush_candidates(conn, candidates)
                    opened_total += opened
                    visited_count = _table_count(conn, "visited")
                    open_count = _table_count(conn, "open")

        _delete_open_rows(conn, processed)
        opened = _flush_candidates(conn, candidates)
        opened_total += opened
        if opened:
            visited_count = _table_count(conn, "visited")
            open_count = _table_count(conn, "open")

    print(
        "FINAL "
        f"status=TARGET_INFEASIBLE target_final_N={target_final_n} expanded={expanded} generated={generated} "
        f"visited={visited_count} open={open_count} opened={opened_total} pruned_bound={pruned_bound} "
        f"pruned_business={pruned_business} dominated={dominated} runtime={time.perf_counter() - start:.3f}s",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="C1")
    parser.add_argument("--bay", default="26")
    parser.add_argument("--target-final-n", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--db-path", type=Path, default=ROOT / "tmp" / "exact_target_bestfirst.sqlite")
    parser.add_argument("--cache-mb", type=int, default=256)
    parser.add_argument("--pop-batch-size", type=int, default=5000)
    parser.add_argument("--insert-batch-size", type=int, default=50_000)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument("--resume-depth", type=int)
    parser.add_argument("--continue-open", action="store_true")
    parser.add_argument("--found-code-path", type=Path)
    parser.add_argument("--max-moves", type=int)
    args = parser.parse_args()
    return run(
        args.zone,
        args.bay,
        args.target_final_n,
        args.alpha,
        args.db_path,
        args.cache_mb,
        args.pop_batch_size,
        args.insert_batch_size,
        args.progress_interval,
        args.resume_depth,
        args.continue_open,
        args.found_code_path,
        args.max_moves,
    )


if __name__ == "__main__":
    raise SystemExit(main())
