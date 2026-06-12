"""Run strict time-matched classic no-QUBO ablations for open CPMP rows.

The script consumes one or more ``open_cpmp_benchmarks.py`` CSV files.  QUBO
rows define per-case runtime budgets; each selected classic no-QUBO run gets
the same budget for that case.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_compact_beam_qubo_n0 import make_compact_args, run_compact_beam_instance  # noqa: E402
from open_cpmp_benchmarks import (  # noqa: E402
    OPEN_RESULT_FIELDS,
    _open_row_dict_from_instance_result,
    _open_validate_instance_result,
    load_dlts_dat,
)


TIME_MATCHED_EXTRA_FIELDS = [
    "matched_budget_variant",
    "matched_budget_sec",
    "matched_width",
    "matched_time_scale",
    "matched_source_runtime_sec",
    "matched_source_variant",
]
TIME_MATCHED_FIELDS = OPEN_RESULT_FIELDS + [field for field in TIME_MATCHED_EXTRA_FIELDS if field not in OPEN_RESULT_FIELDS]
BREAKDOWN_FIELDS = [
    "family",
    "case",
    "category",
    "qubo_solved",
    "classic_solved",
    "qubo_moves",
    "classic_moves",
    "qubo_gap",
    "classic_gap",
    "qubo_runtime_sec",
    "classic_runtime_sec",
    "classic_width",
    "qubo_contribution_flags",
    "adaptive_solved",
    "adaptive_moves",
    "adaptive_runtime_sec",
    "adaptive_note",
]


def numeric(row: dict[str, object], field: str, default: float = 0.0) -> float:
    value = row.get(field)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_solved(row: dict[str, object]) -> bool:
    return int(numeric(row, "final_n", 10**9)) == 0 and row.get("moves") not in (None, "")


def case_category(qubo_row: dict[str, object], classic_row: dict[str, object] | None) -> str:
    qubo_solved = is_solved(qubo_row)
    classic_solved = bool(classic_row and is_solved(classic_row))
    if qubo_solved and not classic_solved:
        return "qubo_only"
    if classic_solved and not qubo_solved:
        return "classic_only"
    if not qubo_solved and not classic_solved:
        return "neither_solved"
    qubo_moves = int(numeric(qubo_row, "moves", 10**9))
    classic_moves = int(numeric(classic_row or {}, "moves", 10**9))
    if qubo_moves < classic_moves:
        return "both_qubo_fewer"
    if classic_moves < qubo_moves:
        return "both_classic_fewer"
    return "both_tie"


def contribution_flags(row: dict[str, object]) -> str:
    flags: list[str] = []
    if numeric(row, "post_polish_moves_saved") > 0:
        flags.append("post_polish")
    if numeric(row, "qubo_selected_on_path") > 0:
        flags.append("frontier_direct")
    if numeric(row, "qubo_ancestor_on_path") > 0:
        flags.append("frontier_ancestor")
    if numeric(row, "qubo_selected_total") > 0:
        flags.append("frontier_selected")
    if "adaptive" in str(row.get("variant", "")):
        flags.append("adaptive")
    return ";".join(flags) if flags else "no_visible_qubo_path_marker"


def _classic_rank(row: dict[str, object]) -> tuple[object, ...]:
    solved = is_solved(row)
    if solved:
        return (
            0,
            int(numeric(row, "moves", 10**9)),
            numeric(row, "move_gap", 10**9),
            numeric(row, "runtime_sec", 10**9),
            numeric(row, "matched_width", 10**9),
        )
    return (
        1,
        int(numeric(row, "final_n", 10**9)),
        numeric(row, "runtime_sec", 10**9),
        numeric(row, "matched_width", 10**9),
    )


def best_classic_by_case(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case"])].append(row)
    return {case: sorted(selected, key=_classic_rank)[0] for case, selected in grouped.items() if selected}


def _read_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows.extend(dict(row) for row in csv.DictReader(f))
    return rows


def _selected_budget_rows(rows: list[dict[str, object]], args: argparse.Namespace) -> list[dict[str, object]]:
    families = set(args.families or [])
    per_family_seen: dict[str, int] = defaultdict(int)
    selected: list[dict[str, object]] = []
    seen_cases: set[str] = set()
    for row in rows:
        if row.get("variant") != args.budget_variant:
            continue
        family = str(row.get("family", ""))
        if families and family not in families:
            continue
        case = str(row.get("case", ""))
        if case in seen_cases:
            continue
        if args.max_cases_per_family is not None and per_family_seen[family] >= args.max_cases_per_family:
            continue
        seen_cases.add(case)
        per_family_seen[family] += 1
        selected.append(row)
    return selected


def _rows_by_case(rows: list[dict[str, object]], variant: str) -> dict[str, dict[str, object]]:
    return {str(row.get("case", "")): row for row in rows if row.get("variant") == variant}


def _reference_copy(row: dict[str, object], args: argparse.Namespace, variant_name: str) -> dict[str, object]:
    budget = max(float(args.time_floor), numeric(row, "runtime_sec") * float(args.time_scale))
    copied = {field: row.get(field, "") for field in OPEN_RESULT_FIELDS}
    copied.update(
        {
            "variant": variant_name,
            "matched_budget_variant": args.budget_variant,
            "matched_budget_sec": f"{budget:.6f}",
            "matched_width": "",
            "matched_time_scale": args.time_scale,
            "matched_source_runtime_sec": row.get("runtime_sec", ""),
            "matched_source_variant": row.get("variant", ""),
        }
    )
    return copied


def _load_instance_for_row(row: dict[str, object]):
    source = Path(str(row["source_file"]))
    max_height = int(numeric(row, "reference_tiers", 0)) or None
    return load_dlts_dat(source, max_height=max_height)


def _classic_args_for_row(row: dict[str, object], width: int, budget_sec: float, alpha: float):
    return make_compact_args(
        alpha=alpha,
        beam_width=width,
        core_width=width,
        max_moves=None,
        max_qubo_calls=0,
        post_polish_qubo=False,
        selection_policy="classic",
        solver_profile="speed_v2",
        target_final_n=0,
        time_limit=budget_sec,
        use_qubo=False,
    )


def run_classic_time_matched(
    budget_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = len(budget_rows) * len(args.widths)
    run_idx = 0
    for budget_row in budget_rows:
        instance = _load_instance_for_row(budget_row)
        alpha = numeric(budget_row, "business_alpha", args.alpha) or args.alpha
        source_runtime = numeric(budget_row, "runtime_sec")
        budget_sec = max(float(args.time_floor), source_runtime * float(args.time_scale))
        for width in args.widths:
            run_idx += 1
            compact_args = _classic_args_for_row(budget_row, int(width), budget_sec, alpha)
            print(
                f"TM_START {run_idx}/{total} {budget_row['case']} width={width} budget={budget_sec:.3f}s",
                flush=True,
            )
            result = run_compact_beam_instance(str(budget_row["case"]), instance, compact_args)
            row = _open_row_dict_from_instance_result(result, instance, compact_args)
            business_ok, goal_ok, msg = _open_validate_instance_result(result, instance, business_alpha=alpha)
            reference_moves = budget_row.get("reference_moves", "")
            moves = row.get("moves")
            row.update(
                {
                    "dataset": budget_row.get("dataset", "DLTS"),
                    "family": budget_row.get("family", ""),
                    "variant": f"classic_tm_w{width}",
                    "source_file": budget_row.get("source_file", ""),
                    "reference_file": budget_row.get("reference_file", ""),
                    "reference_moves": reference_moves,
                    "move_gap": (
                        int(numeric(row, "moves")) - int(numeric(budget_row, "reference_moves"))
                        if moves not in (None, "") and reference_moves not in (None, "")
                        else ""
                    ),
                    "reference_tiers": budget_row.get("reference_tiers", ""),
                    "business_alpha": alpha,
                    "business_ok": business_ok,
                    "goal_validation_ok": goal_ok,
                    "validation_msg": msg,
                    "matched_budget_variant": args.budget_variant,
                    "matched_budget_sec": f"{budget_sec:.6f}",
                    "matched_width": width,
                    "matched_time_scale": args.time_scale,
                    "matched_source_runtime_sec": f"{source_runtime:.6f}",
                    "matched_source_variant": budget_row.get("variant", ""),
                }
            )
            rows.append(row)
            print(
                f"TM_DONE {budget_row['case']} width={width} final_n={row['final_n']} "
                f"moves={row['moves']} runtime={float(row['runtime_sec']):.3f}s",
                flush=True,
            )
    return rows


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIME_MATCHED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize(selected: list[dict[str, object]]) -> dict[str, float | int]:
    solved = [row for row in selected if is_solved(row)]
    gaps = [numeric(row, "move_gap") for row in solved if row.get("move_gap") not in (None, "")]
    budgets = [numeric(row, "matched_budget_sec") for row in selected if row.get("matched_budget_sec") not in (None, "")]
    return {
        "cases": len(selected),
        "solved": len(solved),
        "moves_total": int(sum(numeric(row, "moves") for row in solved)),
        "avg_runtime": sum(numeric(row, "runtime_sec") for row in selected) / len(selected) if selected else 0.0,
        "avg_gap": sum(gaps) / len(gaps) if gaps else 0.0,
        "avg_budget": sum(budgets) / len(budgets) if budgets else 0.0,
    }


def _family_variant_groups(rows: list[dict[str, object]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("family", "")), str(row.get("variant", "")))].append(row)
    return groups


def build_breakdown_rows(
    qubo_rows: list[dict[str, object]],
    classic_rows: list[dict[str, object]],
    adaptive_rows: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    best = best_classic_by_case(classic_rows)
    out: list[dict[str, object]] = []
    for qubo in qubo_rows:
        case = str(qubo.get("case", ""))
        classic = best.get(case)
        adaptive = adaptive_rows.get(case)
        adaptive_note = ""
        if adaptive:
            if is_solved(adaptive) and not is_solved(qubo):
                adaptive_note = "adaptive_only_vs_qubo"
            elif is_solved(qubo) and not is_solved(adaptive):
                adaptive_note = "adaptive_lost_solution"
            elif is_solved(adaptive) and is_solved(qubo):
                if numeric(adaptive, "moves", 10**9) < numeric(qubo, "moves", 10**9):
                    adaptive_note = "adaptive_fewer_moves"
                elif numeric(adaptive, "runtime_sec", 10**9) < numeric(qubo, "runtime_sec", 10**9):
                    adaptive_note = "adaptive_faster"
        out.append(
            {
                "family": qubo.get("family", ""),
                "case": case,
                "category": case_category(qubo, classic),
                "qubo_solved": is_solved(qubo),
                "classic_solved": bool(classic and is_solved(classic)),
                "qubo_moves": qubo.get("moves", ""),
                "classic_moves": classic.get("moves", "") if classic else "",
                "qubo_gap": qubo.get("move_gap", ""),
                "classic_gap": classic.get("move_gap", "") if classic else "",
                "qubo_runtime_sec": qubo.get("runtime_sec", ""),
                "classic_runtime_sec": classic.get("runtime_sec", "") if classic else "",
                "classic_width": classic.get("matched_width", "") if classic else "",
                "qubo_contribution_flags": contribution_flags(qubo),
                "adaptive_solved": is_solved(adaptive) if adaptive else "",
                "adaptive_moves": adaptive.get("moves", "") if adaptive else "",
                "adaptive_runtime_sec": adaptive.get("runtime_sec", "") if adaptive else "",
                "adaptive_note": adaptive_note,
            }
        )
    return out


def _write_breakdown(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BREAKDOWN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_summary(
    rows: list[dict[str, object]],
    breakdown_rows: list[dict[str, object]],
    path: Path,
    output: Path,
    breakdown: Path,
) -> None:
    groups = _family_variant_groups(rows)
    lines = [
        "# Open CPMP Time-Matched No-QUBO Ablation",
        "",
        f"result_csv: `{_display_path(output)}`",
        f"breakdown_csv: `{_display_path(breakdown)}`",
        "",
        "| family | variant | cases | N=0 solved | moves total | avg runtime sec | avg budget sec | avg ref gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (family, variant), selected in sorted(groups.items()):
        summary = _summarize(selected)
        lines.append(
            f"| {family} | {variant} | {summary['cases']} | {summary['solved']} | "
            f"{summary['moves_total']} | {summary['avg_runtime']:.3f} | "
            f"{summary['avg_budget']:.3f} | {summary['avg_gap']:.3f} |"
        )

    best_rows = [row for row in rows if row.get("variant") == "classic_time_matched_best"]
    if best_rows:
        lines.extend(
            [
                "",
                "## Case Breakdown",
                "",
                "| family | category | cases |",
                "|---|---|---:|",
            ]
        )
        counts: Counter[tuple[str, str]] = Counter((str(row["family"]), str(row["category"])) for row in breakdown_rows)
        for (family, category), count in sorted(counts.items()):
            lines.append(f"| {family} | {category} | {count} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _with_best_classic_rows(rows: list[dict[str, object]], classic_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = list(rows)
    for row in best_classic_by_case(classic_rows).values():
        copied = dict(row)
        copied["variant"] = "classic_time_matched_best"
        out.append(copied)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-csv", nargs="+", type=Path, required=True)
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--max-cases-per-family", type=int)
    parser.add_argument("--budget-variant", default="qubo")
    parser.add_argument("--widths", nargs="+", type=int, default=[1200, 2400, 4800])
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--time-floor", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "open_cpmp_time_matched_noqubo.csv",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.reference_csv = [path if path.is_absolute() else ROOT / path for path in args.reference_csv]
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    reference_rows = _read_rows(args.reference_csv)
    budget_rows = _selected_budget_rows(reference_rows, args)
    if not budget_rows:
        raise SystemExit("no budget rows selected")

    qubo_reference_rows = [_reference_copy(row, args, "qubo_reference") for row in budget_rows]
    adaptive_by_case = _rows_by_case(reference_rows, "qubo_adaptive")
    adaptive_reference_rows = [
        _reference_copy(adaptive_by_case[str(row["case"])], args, "qubo_adaptive_reference")
        for row in budget_rows
        if str(row["case"]) in adaptive_by_case
    ]

    classic_rows = run_classic_time_matched(budget_rows, args)
    rows = _with_best_classic_rows(qubo_reference_rows + adaptive_reference_rows + classic_rows, classic_rows)
    breakdown_rows = build_breakdown_rows(qubo_reference_rows, classic_rows, adaptive_by_case)

    _write_rows(rows, args.output)
    breakdown = args.output.with_name(args.output.stem + "_breakdown.csv")
    _write_breakdown(breakdown_rows, breakdown)
    summary = args.output.with_name(args.output.stem + "_summary.md")
    _write_summary(rows, breakdown_rows, summary, args.output, breakdown)
    print(f"WROTE {args.output}", flush=True)
    print(f"WROTE {breakdown}", flush=True)
    print(f"WROTE {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
