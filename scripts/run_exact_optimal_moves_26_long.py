"""Long-run exact shortest-move proof benchmark for bay 26 cases."""

from __future__ import annotations

import argparse
import csv
import re
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
    "n0",
    "cap_13",
    "status",
    "feasible",
    "proof_shortest",
    "optimal_moves",
    "final_n",
    "g_end",
    "runtime_sec",
    "expanded",
    "generated",
    "visited",
    "exit_code",
    "log_path",
]


def _parse_pairs(line: str) -> dict[str, str]:
    return {key: value for key, value in re.findall(r"([A-Za-z_]+)=([^ ]+)", line)}


def _parse_final(log_path: Path) -> dict[str, str]:
    if not log_path.exists():
        return {}
    final: dict[str, str] = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("FINAL "):
            final = _parse_pairs(line)
    return final


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _case_metadata(zone: str, bay: str) -> tuple[str, int, int]:
    instance = _instance_from_single_bay(zone, bay)
    initial = create_initial_state(instance)
    return instance.instance_id, disorder_box_count(instance, initial), entropy_limit(instance, 1.3)


def _row_from_run(
    *,
    zone: str,
    bay: str,
    log_path: Path,
    exit_code: int | str,
    wall_time: float,
    timed_out: bool,
) -> dict[str, object]:
    case, n0, cap = _case_metadata(zone, bay)
    final = _parse_final(log_path)
    status = "TIME_LIMIT_WRAPPER" if timed_out else final.get("status", "NO_FINAL")
    feasible = final.get("feasible", "False")
    optimal = final.get("optimal", "False")
    objective = final.get("objective", "")
    proof = status == "OPTIMAL" and feasible == "True" and optimal == "True" and objective not in {"", "inf"}
    moves = int(float(objective)) if proof else ""
    final_n = 0 if proof else ""
    return {
        "case": case,
        "zone": zone,
        "bay": bay,
        "n0": n0,
        "cap_13": cap,
        "status": status,
        "feasible": feasible,
        "proof_shortest": str(proof),
        "optimal_moves": moves,
        "final_n": final_n,
        "g_end": moves if proof else "",
        "runtime_sec": f"{wall_time:.3f}",
        "expanded": final.get("expanded", ""),
        "generated": final.get("generated", ""),
        "visited": final.get("visited", ""),
        "exit_code": exit_code,
        "log_path": str(log_path),
    }


def _run_one(
    *,
    zone: str,
    bay: str,
    per_case_time_limit: float,
    progress_interval: float,
    log_dir: Path,
) -> dict[str, object]:
    case, _n0, _cap = _case_metadata(zone, bay)
    log_path = log_dir / f"{case}_exact_astar_progress.log"
    err_path = log_dir / f"{case}_exact_astar_progress.err.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_exact_astar_real_until_done.py"),
        "--zone",
        zone,
        "--bay",
        bay,
        "--algorithm",
        "exact_astar_progress",
        "--canonical",
        "--progress-interval",
        str(progress_interval),
        "--time-limit",
        str(per_case_time_limit),
    ]
    start = time.perf_counter()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=out, stderr=err, text=True)
        try:
            exit_code: int | str = proc.wait(timeout=per_case_time_limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            exit_code = "TIMEOUT_KILLED"
            out.write(f"\nWRAPPER_TIMEOUT elapsed={time.perf_counter() - start:.3f}s\n")
    return _row_from_run(
        zone=zone,
        bay=bay,
        log_path=log_path,
        exit_code=exit_code,
        wall_time=time.perf_counter() - start,
        timed_out=timed_out,
    )


def _write_report(rows: list[dict[str, object]], output: Path, report: Path, wall_time: float) -> None:
    proven = [row for row in rows if row.get("proof_shortest") == "True"]
    lines = [
        "# Exact Shortest-Move Proof Baseline: Bay 26 Long Run",
        "",
        f"CSV: `{output.resolve().relative_to(ROOT).as_posix()}`",
        "",
        "A row is a shortest-move proof only when `proof_shortest=True` and `status=OPTIMAL`.",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| cases attempted/listed | {len(rows)} |",
        f"| proven shortest | {len(proven)}/{len(rows)} |",
        f"| wall time sec | {wall_time:.3f} |",
        "",
        "## Detail",
        "",
        "| case | n0 | cap_13 | status | proof shortest | optimal moves | runtime sec | expanded | generated | visited |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['case']} | {row['n0']} | {row['cap_13']} | {row['status']} | "
            f"{row['proof_shortest']} | {row['optimal_moves']} | {row['runtime_sec']} | "
            f"{row['expanded']} | {row['generated']} | {row['visited']} |"
        )
    lines.append("")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-case-time-limit", type=float, default=7200.0)
    parser.add_argument("--progress-interval", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "exact_optimal_moves_single_bays_26_long.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs" / "experiments" / "exact_optimal_moves_single_bays_26_long.md",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "docs" / "experiments" / "exact_optimal_moves_single_bays_26_long_logs",
    )
    args = parser.parse_args()

    cases = discover_single_bay_cases(ROOT / "data" / "input", {"26"})
    existing = _read_rows(args.output) if args.resume else []
    existing_by_case = {row["case"]: row for row in existing if row.get("case")}
    rows: list[dict[str, object]] = []
    start_all = time.perf_counter()

    for index, (zone, bay) in enumerate(cases, start=1):
        case, _n0, _cap = _case_metadata(zone, bay)
        previous = existing_by_case.get(case)
        if previous is not None and previous.get("proof_shortest") == "True":
            rows.append(previous)
            _write_rows(rows, args.output)
            print(f"CASE_SKIP {index}/{len(cases)} {case} proof_shortest=True", flush=True)
            continue
        print(
            f"CASE_START {index}/{len(cases)} {case} "
            f"limit={args.per_case_time_limit:.0f}s",
            flush=True,
        )
        row = _run_one(
            zone=zone,
            bay=bay,
            per_case_time_limit=args.per_case_time_limit,
            progress_interval=args.progress_interval,
            log_dir=args.log_dir,
        )
        rows.append(row)
        _write_rows(rows, args.output)
        print(
            "CASE_DONE "
            f"{index}/{len(cases)} case={row['case']} status={row['status']} "
            f"proof={row['proof_shortest']} moves={row['optimal_moves']} "
            f"wall={row['runtime_sec']}s",
            flush=True,
        )

    wall_time = time.perf_counter() - start_all
    _write_rows(rows, args.output)
    _write_report(rows, args.output, args.report, wall_time)
    print(
        "SUMMARY "
        f"cases={len(rows)} proven={sum(1 for row in rows if row.get('proof_shortest') == 'True')} "
        f"wall={wall_time:.3f}s output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
