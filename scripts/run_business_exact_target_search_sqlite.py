"""Disk-backed exact target search for single-bay business final N(Future)."""

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
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402

Code = bytes


def _height(chunk: bytes, max_height: int) -> int:
    zero = chunk.find(0)
    return max_height if zero < 0 else zero


def _stack_n(chunk: bytes, max_height: int) -> int:
    height = _height(chunk, max_height)
    for h in range(1, height):
        if chunk[h] > chunk[h - 1]:
            return height - h
    return 0


def _split(code: Code, stack_count: int, max_height: int) -> list[bytes]:
    return [code[i * max_height : (i + 1) * max_height] for i in range(stack_count)]


def _canonical(chunks: list[bytes]) -> Code:
    return b"".join(sorted(chunks))


def _encode(stacks: list[list[int]], stack_count: int, max_height: int) -> tuple[Code, int]:
    chunks: list[bytes] = []
    for stack in stacks:
        data = bytearray(max_height)
        for h, box in enumerate(stack):
            data[h] = int(box)
        chunks.append(bytes(data))
    code = _canonical(chunks)
    n_value = sum(_stack_n(chunk, max_height) for chunk in _split(code, stack_count, max_height))
    return code, n_value


def _remaining_lower_bound(current_n: int, move_count: int, cap: int) -> int:
    return max(0, current_n - (cap - move_count))


def _successors_with_n(code: Code, current_n: int, stack_count: int, max_height: int):
    chunks = _split(code, stack_count, max_height)
    heights = [_height(chunk, max_height) for chunk in chunks]
    contrib = [_stack_n(chunk, max_height) for chunk in chunks]
    seen: set[Code] = set()
    for i, src_h in enumerate(heights):
        if src_h == 0:
            continue
        src_top = chunks[i][src_h - 1]
        for j, dst_h in enumerate(heights):
            if i == j or dst_h >= max_height:
                continue
            src = bytearray(chunks[i])
            dst = bytearray(chunks[j])
            src[src_h - 1] = 0
            dst[dst_h] = src_top
            src_b = bytes(src)
            dst_b = bytes(dst)
            nxt_n = current_n - contrib[i] - contrib[j] + _stack_n(src_b, max_height) + _stack_n(dst_b, max_height)
            nxt_chunks = list(chunks)
            nxt_chunks[i] = src_b
            nxt_chunks[j] = dst_b
            nxt = _canonical(nxt_chunks)
            if nxt in seen:
                continue
            seen.add(nxt)
            yield nxt, nxt_n


def _connect(db_path: Path, cache_mb: int, durable_sqlite: bool) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    if durable_sqlite:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    else:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(f"PRAGMA cache_size={-1024 * max(16, cache_mb)}")
    return conn


def _reset_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS visited;
        DROP TABLE IF EXISTS frontier;
        DROP TABLE IF EXISTS next_frontier;
        DROP TABLE IF EXISTS candidate;
        CREATE TABLE visited(code BLOB PRIMARY KEY, depth INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE frontier(code BLOB PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE next_frontier(code BLOB PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TEMP TABLE candidate(code BLOB PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID;
        """
    )
    conn.commit()


def _prepare_resume_db(conn: sqlite3.Connection, resume_depth: int) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS next_frontier;
        DROP TABLE IF EXISTS candidate;
        CREATE TABLE IF NOT EXISTS visited(code BLOB PRIMARY KEY, depth INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS frontier(code BLOB PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE next_frontier(code BLOB PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TEMP TABLE candidate(code BLOB PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID;
        """
    )
    conn.execute("DELETE FROM visited WHERE depth > ?", (resume_depth,))
    conn.commit()


def _flush_candidates(
    conn: sqlite3.Connection,
    candidates: dict[Code, int],
    depth: int,
) -> int:
    if not candidates:
        return 0
    conn.execute("DELETE FROM candidate")
    conn.executemany(
        "INSERT OR IGNORE INTO candidate(code, n) VALUES (?, ?)",
        ((sqlite3.Binary(code), n_value) for code, n_value in candidates.items()),
    )
    before = conn.total_changes
    conn.execute(
        """
        INSERT OR IGNORE INTO next_frontier(code, n)
        SELECT c.code, c.n
        FROM candidate c
        WHERE NOT EXISTS (SELECT 1 FROM visited v WHERE v.code = c.code)
        """
    )
    inserted = conn.total_changes - before
    conn.execute(
        """
        INSERT OR IGNORE INTO visited(code, depth)
        SELECT code, ? FROM candidate
        """,
        (depth,),
    )
    conn.commit()
    candidates.clear()
    return inserted


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def run(
    zone: str,
    bay: str,
    target_final_n: int,
    alpha: float,
    db_path: Path,
    cache_mb: int,
    batch_size: int,
    progress_interval: float,
    resume_depth: int | None,
    order_by_n: bool,
    durable_sqlite: bool,
) -> int:
    instance = _instance_from_single_bay(zone, bay)
    initial, n0 = _encode(instance.stacks, instance.stack_count, instance.max_height)
    cap = entropy_limit(instance, alpha)
    start = time.perf_counter()
    next_progress = start

    conn = _connect(db_path, cache_mb, durable_sqlite)
    if resume_depth is None:
        _reset_db(conn)
        conn.execute("INSERT INTO visited(code, depth) VALUES (?, 0)", (sqlite3.Binary(initial),))
        conn.execute("INSERT INTO frontier(code, n) VALUES (?, ?)", (sqlite3.Binary(initial), n0))
        conn.commit()
        start_depth = 0
        visited_count = 1
        frontier_count = 1
    else:
        _prepare_resume_db(conn, resume_depth)
        start_depth = int(resume_depth)
        visited_count = _table_count(conn, "visited")
        frontier_count = _table_count(conn, "frontier")

    print(
        f"START case={instance.instance_id} target_final_N={target_final_n} "
        f"stacks={instance.stack_count} H={instance.max_height} containers={instance.container_count} "
        f"initial_N={n0} cap={cap} db={db_path} start_depth={start_depth} "
        f"order_by_n={order_by_n} durable_sqlite={durable_sqlite}",
        flush=True,
    )

    expanded = generated = pruned_bound = pruned_business = 0
    target_reached = n0 <= target_final_n

    if target_reached:
        print(
            f"FINAL status=TARGET_REACHABLE target_final_N={target_final_n} moves=0 final_N={n0} "
            f"expanded=0 generated=0 visited=1 runtime={time.perf_counter() - start:.3f}s",
            flush=True,
        )
        return 0

    for depth in range(start_depth, cap):
        next_depth = depth + 1
        next_count = 0
        candidates: dict[Code, int] = {}
        if order_by_n:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_frontier_n_code ON frontier(n, code)")
            conn.commit()
            cursor = conn.execute("SELECT code, n FROM frontier ORDER BY n, code")
        else:
            cursor = conn.execute("SELECT code, n FROM frontier")
        layer_expanded = layer_generated = 0

        for code, current_n in cursor:
            now = time.perf_counter()
            if now >= next_progress:
                print(
                    "PROGRESS "
                    f"elapsed={now - start:.1f}s depth={depth} frontier={frontier_count} next_frontier={next_count} "
                    f"expanded={expanded} generated={generated} visited={visited_count} "
                    f"pruned_bound={pruned_bound} pruned_business={pruned_business}",
                    flush=True,
                )
                next_progress = now + progress_interval

            code_b = bytes(code)
            if _remaining_lower_bound(int(current_n), depth, cap) > target_final_n:
                pruned_bound += 1
                continue

            expanded += 1
            layer_expanded += 1
            for nxt, nxt_n in _successors_with_n(code_b, int(current_n), instance.stack_count, instance.max_height):
                generated += 1
                layer_generated += 1
                if next_depth + nxt_n > cap:
                    pruned_business += 1
                    continue
                if nxt_n <= target_final_n:
                    print(
                        "FINAL "
                        f"status=TARGET_REACHABLE target_final_N={target_final_n} moves={next_depth} final_N={nxt_n} "
                        f"expanded={expanded} generated={generated} visited={visited_count} "
                        f"pruned_bound={pruned_bound} pruned_business={pruned_business} "
                        f"runtime={time.perf_counter() - start:.3f}s",
                        flush=True,
                    )
                    return 0
                if _remaining_lower_bound(nxt_n, next_depth, cap) > target_final_n:
                    pruned_bound += 1
                    continue
                previous = candidates.get(nxt)
                if previous is None or nxt_n < previous:
                    candidates[nxt] = nxt_n
                if len(candidates) >= batch_size:
                    inserted = _flush_candidates(conn, candidates, next_depth)
                    next_count += inserted
                    visited_count += inserted

        inserted = _flush_candidates(conn, candidates, next_depth)
        next_count += inserted
        visited_count += inserted
        print(
            "LAYER_DONE "
            f"elapsed={time.perf_counter() - start:.1f}s depth={depth} expanded={layer_expanded} "
            f"generated={layer_generated} next_frontier={next_count} visited={visited_count}",
            flush=True,
        )

        conn.execute("DROP TABLE frontier")
        conn.execute("ALTER TABLE next_frontier RENAME TO frontier")
        conn.execute("CREATE TABLE next_frontier(code BLOB PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID")
        conn.commit()
        frontier_count = next_count
        if frontier_count == 0:
            break

    frontier_left = _table_count(conn, "frontier")
    print(
        "FINAL "
        f"status=TARGET_INFEASIBLE target_final_N={target_final_n} expanded={expanded} generated={generated} "
        f"visited={visited_count} frontier_left={frontier_left} pruned_bound={pruned_bound} "
        f"pruned_business={pruned_business} runtime={time.perf_counter() - start:.3f}s",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="C1")
    parser.add_argument("--bay", default="26")
    parser.add_argument("--target-final-n", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--db-path", type=Path, default=ROOT / "tmp" / "exact_target_search.sqlite")
    parser.add_argument("--cache-mb", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument("--resume-depth", type=int)
    parser.add_argument("--order-by-n", action="store_true")
    parser.add_argument("--durable-sqlite", action="store_true")
    args = parser.parse_args()
    return run(
        args.zone,
        args.bay,
        args.target_final_n,
        args.alpha,
        args.db_path,
        args.cache_mb,
        args.batch_size,
        args.progress_interval,
        args.resume_depth,
        args.order_by_n,
        args.durable_sqlite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
