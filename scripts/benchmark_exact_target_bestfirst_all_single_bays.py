"""Batch run disk-backed exact best-first target search on single-bay cases."""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from container_reshuffle.core import create_initial_state  # noqa: E402
from container_reshuffle.features import disorder_box_count, entropy_limit  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


FIELDS = [
    "case",
    "zone",
    "bay",
    "target_final_n",
    "n0",
    "cap_13",
    "status",
    "moves",
    "final_n",
    "runtime_sec",
    "wall_time_sec",
    "expanded",
    "generated",
    "visited",
    "open",
    "opened",
    "pruned_bound",
    "pruned_business",
    "dominated",
    "db_path",
    "log_path",
]


def _parse_pairs(line: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for key, value in re.findall(r"([A-Za-z_]+)=([^ ]+)", line):
        pairs[key] = value[:-1] if value.endswith("s") and key in {"runtime", "elapsed"} else value
    return pairs


def _parse_log(log_path: Path) -> dict[str, str]:
    if not log_path.exists():
        return {}
    start_pairs: dict[str, str] = {}
    final_pairs: dict[str, str] = {}
    progress_pairs: dict[str, str] = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("START "):
            start_pairs = _parse_pairs(line)
        elif line.startswith("PROGRESS "):
            progress_pairs = _parse_pairs(line)
        elif line.startswith("FINAL "):
            final_pairs = _parse_pairs(line)
    out = {}
    out.update(start_pairs)
    out.update(progress_pairs)
    out.update(final_pairs)
    return out


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _db_has_open_table(db_path: Path) -> bool:
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='open'").fetchone()
            return row is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def _row_from_run(
    *,
    zone: str,
    bay: str,
    target_final_n: int,
    db_path: Path,
    log_path: Path,
    wall_time: float,
    timed_out: bool,
) -> dict[str, object]:
    instance = _instance_from_single_bay(zone, bay)
    parsed = _parse_log(log_path)
    status = "TIME_LIMIT_WRAPPER" if timed_out else parsed.get("status", "NO_FINAL")
    return {
        "case": instance.instance_id,
        "zone": zone,
        "bay": bay,
        "target_final_n": target_final_n,
        "n0": disorder_box_count(instance, create_initial_state(instance)),
        "cap_13": entropy_limit(instance, 1.3),
        "status": status,
        "moves": parsed.get("moves", ""),
        "final_n": parsed.get("final_N", ""),
        "runtime_sec": parsed.get("runtime", ""),
        "wall_time_sec": f"{wall_time:.3f}",
        "expanded": parsed.get("expanded", ""),
        "generated": parsed.get("generated", ""),
        "visited": parsed.get("visited", ""),
        "open": parsed.get("open", ""),
        "opened": parsed.get("opened", ""),
        "pruned_bound": parsed.get("pruned_bound", ""),
        "pruned_business": parsed.get("pruned_business", ""),
        "dominated": parsed.get("dominated", ""),
        "db_path": str(db_path),
        "log_path": str(log_path),
    }


def _run_one(
    *,
    zone: str,
    bay: str,
    target_final_n: int,
    per_case_time_limit: float,
    cache_mb: int,
    pop_batch_size: int,
    insert_batch_size: int,
    progress_interval: float,
    db_dir: Path,
    log_dir: Path,
    continue_open_existing: bool,
) -> dict[str, object]:
    instance = _instance_from_single_bay(zone, bay)
    db_path = db_dir / f"{instance.instance_id}_target{target_final_n}.sqlite"
    log_path = log_dir / f"{instance.instance_id}_target{target_final_n}.log"
    err_path = log_dir / f"{instance.instance_id}_target{target_final_n}.err.txt"
    found_path = log_dir / f"{instance.instance_id}_target{target_final_n}_found_code.txt"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_business_exact_target_bestfirst_sqlite.py"),
        "--zone",
        zone,
        "--bay",
        bay,
        "--target-final-n",
        str(target_final_n),
        "--db-path",
        str(db_path),
        "--cache-mb",
        str(cache_mb),
        "--pop-batch-size",
        str(pop_batch_size),
        "--insert-batch-size",
        str(insert_batch_size),
        "--progress-interval",
        str(progress_interval),
        "--found-code-path",
        str(found_path),
    ]
    continuing = continue_open_existing and _db_has_open_table(db_path)
    if continuing:
        cmd.append("--continue-open")

    start = time.perf_counter()
    timed_out = False
    log_mode = "a" if continuing else "w"
    err_mode = "a" if continuing else "w"
    with log_path.open(log_mode, encoding="utf-8") as out, err_path.open(err_mode, encoding="utf-8") as err:
        if continuing:
            out.write(f"\nWRAPPER_RESUME elapsed=0.000s db={db_path}\n")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=out, stderr=err, text=True)
        try:
            proc.wait(timeout=per_case_time_limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
            out.write(f"\nWRAPPER_TIMEOUT elapsed={time.perf_counter() - start:.3f}s\n")
    wall_time = time.perf_counter() - start
    return _row_from_run(
        zone=zone,
        bay=bay,
        target_final_n=target_final_n,
        db_path=db_path,
        log_path=log_path,
        wall_time=wall_time,
        timed_out=timed_out,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-final-n", type=int, default=0)
    parser.add_argument("--per-case-time-limit", type=float, default=7200.0)
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--cache-mb", type=int, default=512)
    parser.add_argument("--pop-batch-size", type=int, default=5000)
    parser.add_argument("--insert-batch-size", type=int, default=50_000)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument("--continue-open-existing", action="store_true")
    parser.add_argument("--skip-solved-from", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "exact_target_bestfirst_all_single_bays.csv",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=ROOT / "tmp" / "exact_target_bestfirst_all_single_bays",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "docs" / "experiments" / "exact_target_bestfirst_all_single_bays_logs",
    )
    args = parser.parse_args()

    cases = discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))
    rows: list[dict[str, object]] = []
    previous_by_case = {
        row["case"]: row for row in _read_rows(args.skip_solved_from) if row.get("case")
    }
    start_all = time.perf_counter()
    for idx, (zone, bay) in enumerate(cases, start=1):
        case_id = f"{zone}_{bay}_single"
        previous = previous_by_case.get(case_id)
        if previous is not None and previous.get("status") == "TARGET_REACHABLE":
            rows.append(previous)
            _write_rows(rows, args.output)
            print(f"CASE_SKIP {idx}/{len(cases)} {case_id} previous_status=TARGET_REACHABLE", flush=True)
            continue
        print(
            f"CASE_START {idx}/{len(cases)} {case_id} "
            f"target={args.target_final_n} limit={args.per_case_time_limit:.1f}s",
            flush=True,
        )
        row = _run_one(
            zone=zone,
            bay=bay,
            target_final_n=args.target_final_n,
            per_case_time_limit=args.per_case_time_limit,
            cache_mb=args.cache_mb,
            pop_batch_size=args.pop_batch_size,
            insert_batch_size=args.insert_batch_size,
            progress_interval=args.progress_interval,
            db_dir=args.db_dir,
            log_dir=args.log_dir,
            continue_open_existing=args.continue_open_existing,
        )
        rows.append(row)
        _write_rows(rows, args.output)
        print(
            "CASE_DONE "
            f"{idx}/{len(cases)} case={row['case']} status={row['status']} "
            f"moves={row['moves']} finalN={row['final_n']} wall={row['wall_time_sec']}s",
            flush=True,
        )

    solved = sum(1 for row in rows if row.get("status") == "TARGET_REACHABLE")
    print(
        "SUMMARY "
        f"cases={len(rows)} solved={solved} wall_time={time.perf_counter() - start_all:.3f}s "
        f"output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
