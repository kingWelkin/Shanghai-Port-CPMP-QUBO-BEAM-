"""Time-aligned QUBO ablation for the fast slack beam.

The experiment answers one narrow question:

If the QUBO module is disabled, can a wider deterministic beam spend roughly
the same wall time and recover the same solution quality?

The reference rows are read from an existing QUBO benchmark CSV.  For every
case, this script increases the no-QUBO beam width until the runtime reaches
the reference runtime neighborhood, then reports both:

- the no-QUBO row closest to the QUBO runtime,
- the best no-QUBO row found within the same runtime budget.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
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
from benchmark_fast_qubo_adaptive_rescue_v1 import run_adaptive  # noqa: E402
from benchmark_fast_qubo_integrated_v1 import _exact_moves_by_case, _move_gap, _parse_case, _rel, _validate_final  # noqa: E402
from benchmark_fast_slack_beam import run_fast_slack_beam  # noqa: E402
from benchmark_full_real_business_algorithms import TrajectoryResult  # noqa: E402
from container_reshuffle.core import State  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


REFERENCE_RESULTS = ROOT / "docs" / "experiments" / "fast_qubo_adaptive_rescue_v1_full_v3_fastq_only.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "experiments" / "qubo_beam_ablation_time_aligned_full.csv"
RESULT_FIELDS = list(TrajectoryResult.__annotations__.keys())
COMPARISON_FIELDS = [
    "case",
    "role",
    "solver",
    "selected_width",
    "reference_runtime_sec",
    "runtime_gap_sec",
    "runtime_ratio",
    *RESULT_FIELDS[2:],
    "business_ok",
    "goal_validation_ok",
    "validation_msg",
    "moves_path",
]
SWEEP_FIELDS = [
    "case",
    "width",
    "reference_runtime_sec",
    "runtime_gap_sec",
    "runtime_ratio",
    *RESULT_FIELDS[1:],
]


@dataclass
class AblationRun:
    row: TrajectoryResult
    best_state: State
    width: int
    business_ok: bool
    goal_validation_ok: bool
    validation_msg: str


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _reference_by_case(path: Path, solver: str | None) -> dict[str, dict[str, str]]:
    rows = _read_csv_rows(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if solver and row.get("solver") != solver:
            continue
        out[row["case"]] = row
    return out


def _case_filter(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return [case if case.endswith("_single") else f"{case}_single" for case in args.cases]
    return [f"{zone}_{bay}_single" for zone, bay in discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))]


def _as_float(row: dict[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def _as_int(row: dict[str, object], key: str) -> int | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return int(float(value))


def _qubo_reference_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        max_depth=args.max_depth,
        fast_time_limit=8.0,
        fast_beam_width=122,
        fast_core_width=120,
        fast_qubo_supplement_width=2,
        fast_qubo_min_depth=10,
        fast_qubo_call_interval=3,
        fast_qubo_plateau_only=True,
        fast_tail_repair_n=5,
        fast_tail_repair_expanded=30000,
        rescue_time_limit=16.0,
        rescue_beam_width=240,
        rescue_qubo_seed_width=0,
        rescue_core_width=232,
        rescue_qubo_supplement_width=8,
        rescue_qubo_min_depth=6,
        rescue_qubo_call_interval=4,
        rescue_qubo_plateau_only=False,
        rescue_tail_repair_n=3,
        rescue_tail_repair_expanded=30000,
        rescue_tail_repair_candidates=2,
        rescue_repair_n=8,
        rescue_repair_candidates=80,
        rescue_reservoir_size=120,
        max_qubo_vars=60,
        qubo_time_limit=0.01,
        qubo_reads=2,
        qubo_iterations=80,
        qubo_penalty_a=30.0,
        diversity_weight=0.08,
        seed=args.seed,
    )


def _runtime_ratio(runtime: float, target: float) -> str:
    if target <= 0:
        return ""
    return f"{runtime / target:.6f}"


def _runtime_gap(runtime: float, target: float) -> str:
    return f"{runtime - target:.6f}"


def _run_no_qubo_width(args: argparse.Namespace, case: str, width: int, target_runtime: float) -> AblationRun:
    zone, bay = _parse_case(case)
    instance = _instance_from_single_bay(zone, bay)
    config = SolverConfig(
        max_depth=args.max_depth,
        beam_width=width,
        time_limit_sec=max(args.min_time_limit, target_runtime * args.time_limit_factor),
        entropy_alpha=1.3,
        random_seed=args.seed,
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        row, best = run_fast_slack_beam(
            case,
            instance,
            config,
            tail_repair_n=args.tail_repair_n,
            tail_repair_expanded=args.tail_repair_expanded,
            tail_repair_candidates=args.tail_repair_candidates,
            lookahead_pool=0,
            lookahead_keep=0,
            rescue_repair_n=args.rescue_repair_n,
            rescue_repair_candidates=args.rescue_repair_candidates,
            rescue_reservoir_size=args.rescue_reservoir_size,
        )
    row.solver = "no_qubo_wide_fast_slack_beam"
    row.status += f"; qubo_disabled; width={width}; target_runtime_sec={target_runtime:.6f}"
    business_ok, goal_ok, validation_msg = _validate_final(instance, best, row.runtime_sec)
    return AblationRun(row, best, width, business_ok, goal_ok, validation_msg)


def _runtime_distance(run: AblationRun, target_runtime: float) -> tuple[float, int]:
    return abs(run.row.runtime_sec - target_runtime), run.width


def _quality_key(run: AblationRun, target_runtime: float) -> tuple[int, int, float, int]:
    final_n = run.row.final_n if run.row.final_n is not None else 10**9
    moves = run.row.moves if run.row.moves is not None else 10**9
    return final_n, moves, abs(run.row.runtime_sec - target_runtime), run.width


def _write_comparison_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_sweep_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row_dict_for_reference(
    ref: dict[str, object],
    exact_refs: dict[str, int],
    *,
    business_ok: bool = True,
    goal_validation_ok: bool | None = None,
    validation_msg: str = "reference_csv",
    moves_path: Path | None = None,
) -> dict[str, object]:
    target_runtime = _as_float(ref, "runtime_sec")
    final_n = _as_int(ref, "final_n")
    moves = _as_int(ref, "moves")
    if goal_validation_ok is None:
        goal_validation_ok = final_n == 0
    row = {
        "case": ref["case"],
        "role": "qubo_reference",
        "solver": ref.get("solver", ""),
        "selected_width": "",
        "reference_runtime_sec": f"{target_runtime:.6f}",
        "runtime_gap_sec": "0.000000",
        "runtime_ratio": "1.000000",
        "business_ok": business_ok,
        "goal_validation_ok": goal_validation_ok,
        "validation_msg": validation_msg,
        "moves_path": _rel(moves_path) if moves_path is not None else "",
    }
    for field in RESULT_FIELDS[2:]:
        row[field] = ref.get(field, "")
    if final_n == 0:
        exact = exact_refs.get(ref["case"])
        if exact is not None and moves is not None:
            row["status"] = f"{ref.get('status', '')}; exact_move_gap={moves - exact}"
    return row


def _row_dict_for_run(
    run: AblationRun,
    *,
    role: str,
    target_runtime: float,
    moves_dir: Path,
    exact_refs: dict[str, int],
) -> dict[str, object]:
    zone, bay = _parse_case(run.row.case)
    instance = _instance_from_single_bay(zone, bay)
    move_path = moves_dir / f"{run.row.case}_{role}_w{run.width}_moves.csv"
    _write_explain_csv(instance, run.best_state.moves, move_path)
    row = {
        "case": run.row.case,
        "role": role,
        "solver": run.row.solver,
        "selected_width": run.width,
        "reference_runtime_sec": f"{target_runtime:.6f}",
        "runtime_gap_sec": _runtime_gap(run.row.runtime_sec, target_runtime),
        "runtime_ratio": _runtime_ratio(run.row.runtime_sec, target_runtime),
        "business_ok": run.business_ok,
        "goal_validation_ok": run.goal_validation_ok,
        "validation_msg": run.validation_msg,
        "moves_path": _rel(move_path),
    }
    for field in RESULT_FIELDS[2:]:
        row[field] = getattr(run.row, field)
    gap = _move_gap(run.row, exact_refs)
    if gap:
        row["status"] = f"{run.row.status}; exact_move_gap={gap}"
    return row


def _sweep_row(run: AblationRun, target_runtime: float) -> dict[str, object]:
    row = {
        "case": run.row.case,
        "width": run.width,
        "reference_runtime_sec": f"{target_runtime:.6f}",
        "runtime_gap_sec": _runtime_gap(run.row.runtime_sec, target_runtime),
        "runtime_ratio": _runtime_ratio(run.row.runtime_sec, target_runtime),
    }
    for field in RESULT_FIELDS[1:]:
        row[field] = getattr(run.row, field)
    return row


def _summarize(rows: list[dict[str, object]], *, bay_suffix: str | None = None) -> list[dict[str, object]]:
    roles = ["qubo_reference", "no_qubo_time_matched", "no_qubo_budget_best"]
    summary = []
    for role in roles:
        selected = [row for row in rows if row["role"] == role]
        if bay_suffix:
            selected = [row for row in selected if str(row["case"]).split("_")[1] == bay_suffix]
        if not selected:
            continue
        final_ns = [int(float(row["final_n"])) for row in selected if row.get("final_n") not in (None, "")]
        moves = [int(float(row["moves"])) for row in selected if row.get("moves") not in (None, "")]
        solved_moves = [
            int(float(row["moves"]))
            for row in selected
            if row.get("final_n") not in (None, "") and int(float(row["final_n"])) == 0 and row.get("moves") not in (None, "")
        ]
        runtimes = [float(row["runtime_sec"]) for row in selected if row.get("runtime_sec") not in (None, "")]
        ratios = [float(row["runtime_ratio"]) for row in selected if row.get("runtime_ratio") not in (None, "")]
        widths = [int(float(row["selected_width"])) for row in selected if row.get("selected_width") not in (None, "")]
        summary.append(
            {
                "role": role,
                "cases": len(selected),
                "solved": sum(1 for value in final_ns if value == 0),
                "avg_final_n": sum(final_ns) / len(final_ns) if final_ns else 0.0,
                "avg_moves": sum(moves) / len(moves) if moves else 0.0,
                "avg_solved_moves": sum(solved_moves) / len(solved_moves) if solved_moves else 0.0,
                "avg_runtime": sum(runtimes) / len(runtimes) if runtimes else 0.0,
                "avg_runtime_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
                "avg_width": sum(widths) / len(widths) if widths else 0.0,
            }
        )
    return summary


def _format_summary_table(summary: list[dict[str, object]]) -> list[str]:
    lines = [
        "| role | solved | avg final N | avg moves | avg solved moves | avg runtime sec | avg runtime ratio | avg width |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['role']} | {item['solved']}/{item['cases']} | "
            f"{item['avg_final_n']:.3f} | {item['avg_moves']:.3f} | {item['avg_solved_moves']:.3f} | "
            f"{item['avg_runtime']:.3f} | {item['avg_runtime_ratio']:.3f} | {item['avg_width']:.1f} |"
        )
    return lines


def _write_report(
    comparison_rows: list[dict[str, object]],
    sweep_rows: list[dict[str, object]],
    output: Path,
    sweep_output: Path,
    report: Path,
) -> None:
    all_summary = _summarize(comparison_rows)
    bay26_summary = _summarize(comparison_rows, bay_suffix="26")
    qubo_active_cases = {
        str(row["case"])
        for row in comparison_rows
        if row["role"] == "qubo_reference" and row.get("qubo_calls") not in (None, "") and int(float(row["qubo_calls"])) > 0
    }
    qubo_active_summary = _summarize([row for row in comparison_rows if str(row["case"]) in qubo_active_cases])
    lines = [
        "# QUBO Beam Ablation: Time-Aligned Wider No-QUBO Beam",
        "",
        "Reference is the existing fast QUBO adaptive rescue result. The ablation disables QUBO and increases beam width until the runtime is close to the reference runtime.",
        "",
        f"Comparison CSV: `{_rel(output)}`",
        f"Width sweep CSV: `{_rel(sweep_output)}`",
        "",
        "## All 30 Cases",
        "",
        *_format_summary_table(all_summary),
        "",
        "## Bay 26 Cases",
        "",
        *_format_summary_table(bay26_summary),
        "",
        "## QUBO-Active Cases",
        "",
        *_format_summary_table(qubo_active_summary),
        "",
        "## Per-Case Comparison",
        "",
        "| case | QUBO N/moves/t/qcalls | no-QUBO time-matched width/N/moves/t | no-QUBO budget-best width/N/moves/t |",
        "|---|---:|---:|---:|",
    ]
    by_case: dict[str, dict[str, dict[str, object]]] = {}
    for row in comparison_rows:
        by_case.setdefault(str(row["case"]), {})[str(row["role"])] = row
    for case in sorted(by_case):
        rows = by_case[case]
        ref = rows.get("qubo_reference", {})
        matched = rows.get("no_qubo_time_matched", {})
        best = rows.get("no_qubo_budget_best", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    case,
                    f"{ref.get('final_n', '')}/{ref.get('moves', '')}/{float(ref.get('runtime_sec', 0) or 0):.3f}/{ref.get('qubo_calls', '')}",
                    f"w{matched.get('selected_width', '')}/{matched.get('final_n', '')}/{matched.get('moves', '')}/{float(matched.get('runtime_sec', 0) or 0):.3f}",
                    f"w{best.get('selected_width', '')}/{best.get('final_n', '')}/{best.get('moves', '')}/{float(best.get('runtime_sec', 0) or 0):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Sweep rows: {len(sweep_rows)}",
            "- `no_qubo_time_matched` chooses the tried no-QUBO width with runtime closest to the QUBO reference runtime.",
            "- `no_qubo_budget_best` chooses the best no-QUBO result among tried widths not exceeding the reference runtime by more than the configured tolerance.",
            "- Very small 01 cases may finish before any beam width can meaningfully consume the reference runtime.",
            "- QUBO-active means the reference QUBO run actually made at least one QUBO call.",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(comparison_rows: list[dict[str, object]]) -> None:
    print("\nSUMMARY all cases", flush=True)
    for line in _format_summary_table(_summarize(comparison_rows)):
        print(line, flush=True)
    print("\nSUMMARY bay 26", flush=True)
    for line in _format_summary_table(_summarize(comparison_rows, bay_suffix="26")):
        print(line, flush=True)
    qubo_active_cases = {
        str(row["case"])
        for row in comparison_rows
        if row["role"] == "qubo_reference" and row.get("qubo_calls") not in (None, "") and int(float(row["qubo_calls"])) > 0
    }
    print("\nSUMMARY QUBO-active cases", flush=True)
    active_rows = [row for row in comparison_rows if str(row["case"]) in qubo_active_cases]
    for line in _format_summary_table(_summarize(active_rows)):
        print(line, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-results", type=Path, default=REFERENCE_RESULTS)
    parser.add_argument("--reference-solver", default="fast_qubo_adaptive_rescue_v1")
    parser.add_argument("--rerun-reference", action="store_true")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--widths", nargs="+", type=int, default=[160, 240, 360, 480, 640, 960, 1280])
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--time-tolerance", type=float, default=0.15)
    parser.add_argument("--time-limit-factor", type=float, default=1.0)
    parser.add_argument("--min-time-limit", type=float, default=0.10)
    parser.add_argument("--small-target-sec", type=float, default=0.10)
    parser.add_argument("--tail-repair-n", type=int, default=5)
    parser.add_argument("--tail-repair-expanded", type=int, default=30000)
    parser.add_argument("--tail-repair-candidates", type=int, default=2)
    parser.add_argument("--rescue-repair-n", type=int, default=8)
    parser.add_argument("--rescue-repair-candidates", type=int, default=80)
    parser.add_argument("--rescue-reservoir-size", type=int, default=120)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sweep-output", type=Path)
    parser.add_argument("--moves-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.sweep_output is None:
        args.sweep_output = args.output.with_name(args.output.stem + "_sweep.csv")
    if args.moves_dir is None:
        args.moves_dir = args.output.with_name(args.output.stem + "_moves")
    if args.report is None:
        args.report = args.output.with_suffix(".md")

    refs = {} if args.rerun_reference else _reference_by_case(args.reference_results, args.reference_solver)
    if not args.rerun_reference and not refs:
        raise SystemExit(f"no reference rows found in {args.reference_results}")
    exact_refs = _exact_moves_by_case()
    qargs = _qubo_reference_args(args)

    comparison_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    cases = _case_filter(args)
    start = time.perf_counter()
    for idx, case in enumerate(cases, start=1):
        ref_run = None
        ref_moves_path = None
        if args.rerun_reference:
            zone, bay = _parse_case(case)
            instance = _instance_from_single_bay(zone, bay)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                ref_run = run_adaptive(case, instance, qargs)
            ref = {field: getattr(ref_run.row, field) for field in RESULT_FIELDS}
            ref_moves_path = args.moves_dir / f"{case}_qubo_reference_moves.csv"
            _write_explain_csv(instance, ref_run.best_state.moves, ref_moves_path)
        else:
            ref = refs.get(case)
        if ref is None:
            print(f"SKIP {idx}/{len(cases)} {case}: no reference row", flush=True)
            continue
        target = _as_float(ref, "runtime_sec")
        tried: list[AblationRun] = []
        target_low = target * (1.0 - args.time_tolerance)
        target_high = target * (1.0 + args.time_tolerance)
        widths = sorted(dict.fromkeys(width for width in args.widths if width > 0))
        if target <= args.small_target_sec:
            widths = widths[:1]

        print(
            f"CASE_START {idx}/{len(cases)} {case} target={target:.3f}s "
            f"reference={'rerun_current_code' if args.rerun_reference else 'csv'}",
            flush=True,
        )
        for width in widths:
            run = _run_no_qubo_width(args, case, width, target)
            tried.append(run)
            sweep_rows.append(_sweep_row(run, target))
            print(
                f"  width={width} noQUBO finalN={run.row.final_n} moves={run.row.moves} "
                f"runtime={run.row.runtime_sec:.3f}s ratio={_runtime_ratio(run.row.runtime_sec, target)}",
                flush=True,
            )
            if target > args.small_target_sec and run.row.runtime_sec >= target_low:
                break

        time_matched = min(tried, key=lambda run: _runtime_distance(run, target))
        budget_candidates = [run for run in tried if run.row.runtime_sec <= target_high]
        if not budget_candidates:
            budget_candidates = tried
        budget_best = min(budget_candidates, key=lambda run: _quality_key(run, target))

        comparison_rows.append(
            _row_dict_for_reference(
                ref,
                exact_refs,
                business_ok=True if ref_run is None else ref_run.business_ok,
                goal_validation_ok=None if ref_run is None else ref_run.goal_validation_ok,
                validation_msg="reference_csv" if ref_run is None else ref_run.validation_msg,
                moves_path=ref_moves_path,
            )
        )
        comparison_rows.append(
            _row_dict_for_run(
                time_matched,
                role="no_qubo_time_matched",
                target_runtime=target,
                moves_dir=args.moves_dir,
                exact_refs=exact_refs,
            )
        )
        comparison_rows.append(
            _row_dict_for_run(
                budget_best,
                role="no_qubo_budget_best",
                target_runtime=target,
                moves_dir=args.moves_dir,
                exact_refs=exact_refs,
            )
        )
        _write_comparison_rows(comparison_rows, args.output)
        _write_sweep_rows(sweep_rows, args.sweep_output)
        print(
            f"CASE_DONE {case} QUBO finalN={ref.get('final_n')} moves={ref.get('moves')} "
            f"matched=w{time_matched.width}/N{time_matched.row.final_n}/t{time_matched.row.runtime_sec:.3f}s "
            f"best=w{budget_best.width}/N{budget_best.row.final_n}/t{budget_best.row.runtime_sec:.3f}s",
            flush=True,
        )

    _write_comparison_rows(comparison_rows, args.output)
    _write_sweep_rows(sweep_rows, args.sweep_output)
    _write_report(comparison_rows, sweep_rows, args.output, args.sweep_output, args.report)
    _print_summary(comparison_rows)
    print(
        f"\nDONE cases={len(comparison_rows) // 3} sweep_runs={len(sweep_rows)} "
        f"wall={time.perf_counter() - start:.3f}s",
        flush=True,
    )
    print(f"OUTPUT comparison={_rel(args.output)} sweep={_rel(args.sweep_output)} report={_rel(args.report)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
