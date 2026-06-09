"""Attach to a running bay-26 disk BFS proof and continue the remaining cases."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from run_exact_optimal_moves_26_disk_bfs_long import (  # noqa: E402
    _case_metadata,
    _read_rows,
    _row_from_run,
    _run_one,
    _write_report,
    _write_rows,
)


def _pid_exists(pid: int) -> bool:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attach-pid", type=int, required=True)
    parser.add_argument("--attach-zone", default="A1")
    parser.add_argument("--attach-bay", default="26")
    parser.add_argument("--poll-interval", type=float, default=120.0)
    parser.add_argument("--cache-mb", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--progress-interval", type=float, default=120.0)
    parser.add_argument("--order-by-n", action="store_true")
    parser.add_argument("--cleanup-db", action="store_true")
    parser.add_argument("--durable-sqlite", action="store_true")
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=Path("D:/exact_optimal_moves_single_bays_26_disk_bfs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "exact_optimal_moves_single_bays_26_disk_bfs_long.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs" / "experiments" / "exact_optimal_moves_single_bays_26_disk_bfs_long.md",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "docs" / "experiments" / "exact_optimal_moves_single_bays_26_disk_bfs_long_logs",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = [dict(row) for row in _read_rows(args.output)]
    existing_by_case = {str(row["case"]): row for row in rows if row.get("case")}
    attach_case, _n0, _cap = _case_metadata(args.attach_zone, args.attach_bay)
    start_all = time.perf_counter()

    if existing_by_case.get(attach_case, {}).get("proof_shortest") != "True":
        print(f"ATTACH_WAIT pid={args.attach_pid} case={attach_case}", flush=True)
        while _pid_exists(args.attach_pid):
            time.sleep(args.poll_interval)
            print(f"ATTACH_STILL_RUNNING pid={args.attach_pid} elapsed={time.perf_counter() - start_all:.1f}s", flush=True)
        db_path = args.db_dir / f"{attach_case}_target0_shortest_bfs.sqlite"
        log_path = args.log_dir / f"{attach_case}_target0_shortest_bfs.log"
        row = _row_from_run(
            zone=args.attach_zone,
            bay=args.attach_bay,
            db_path=db_path,
            log_path=log_path,
            exit_code="ATTACHED_PROCESS_EXITED",
            wall_time=time.perf_counter() - start_all,
            timed_out=False,
        )
        rows = [r for r in rows if r.get("case") != attach_case]
        rows.append(row)
        _write_rows(rows, args.output)
        existing_by_case = {str(r["case"]): r for r in rows if r.get("case")}
        print(
            "ATTACH_DONE "
            f"case={row['case']} status={row['status']} proof={row['proof_shortest']} "
            f"moves={row['optimal_moves']}",
            flush=True,
        )

    cases = discover_single_bay_cases(ROOT / "data" / "input", {"26"})
    for index, (zone, bay) in enumerate(cases, start=1):
        case, _n0, _cap = _case_metadata(zone, bay)
        previous = existing_by_case.get(case)
        if previous is not None and previous.get("proof_shortest") == "True":
            print(f"CASE_SKIP {index}/{len(cases)} {case} proof_shortest=True", flush=True)
            continue
        if case == attach_case and previous is not None:
            print(f"CASE_SKIP {index}/{len(cases)} {case} already_attempted status={previous.get('status')}", flush=True)
            continue
        print(f"CASE_START {index}/{len(cases)} {case} limit=none", flush=True)
        row = _run_one(
            zone=zone,
            bay=bay,
            per_case_time_limit=0,
            cache_mb=args.cache_mb,
            batch_size=args.batch_size,
            progress_interval=args.progress_interval,
            db_dir=args.db_dir,
            log_dir=args.log_dir,
            order_by_n=args.order_by_n,
            cleanup_db=args.cleanup_db,
            durable_sqlite=args.durable_sqlite,
        )
        rows = [r for r in rows if r.get("case") != case]
        rows.append(row)
        _write_rows(rows, args.output)
        existing_by_case[case] = row
        print(
            "CASE_DONE "
            f"{index}/{len(cases)} case={row['case']} status={row['status']} "
            f"proof={row['proof_shortest']} moves={row['optimal_moves']} "
            f"runtime={row['runtime_sec']}s",
            flush=True,
        )

    _write_report(rows, args.output, args.report, time.perf_counter() - start_all)
    print(
        "SUMMARY "
        f"cases={len(rows)} proven={sum(1 for row in rows if row.get('proof_shortest') == 'True')} "
        f"output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
