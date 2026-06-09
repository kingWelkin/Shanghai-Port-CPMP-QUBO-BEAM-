"""Strict QUBO/no-QUBO ablation for the latest compact beam algorithm.

This driver runs the current compact byte-state beam implementation twice on
the same instances in the same process:

1. QUBO-on: the latest compact beam defaults.
2. QUBO-off: frontier QUBO and post-polish QUBO disabled, with larger beam
   widths swept until runtime is close to the QUBO-on runtime.

The goal is to isolate the QUBO module's contribution under a matched wall-time
budget without mixing in historical CSV results from older code.
"""

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

import benchmark_compact_beam_qubo_n0 as compact  # noqa: E402
from benchmark_compact_beam_qubo_n0 import (  # noqa: E402
    RESULT_FIELDS,
    _row_from_result,
    _select_next_frontier as _state_frontier_selector,
    _validate_result,
)
from benchmark_compact_group_qubo_majority import _select_group_qubo_frontier  # noqa: E402
from benchmark_explainable_rule_macro_beam import _write_explain_csv  # noqa: E402
from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from benchmark_fast_qubo_integrated_v1 import _exact_moves_by_case, _move_gap, _parse_case, _rel  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


DEFAULT_OUTPUT = ROOT / "docs" / "experiments" / "compact_qubo_ablation_time_aligned_full.csv"
ROW_FIELDS = [field for field in RESULT_FIELDS if field != "case"]
COMPARISON_FIELDS = [
    "case",
    "role",
    "selected_width",
    "selection_policy",
    "target_runtime_sec",
    "runtime_gap_sec",
    "runtime_ratio",
    "exact_moves",
    "move_gap",
    "business_ok",
    "goal_validation_ok",
    "validation_msg",
    "moves_path",
    "qubo_selected_total",
    "qubo_selected_on_path",
    "qubo_path_share",
    "qubo_ancestor_on_path",
    "qubo_ancestor_path_share",
    "tail_n_trace",
    "tail_ordered_trace",
    "tail_ordered_gain",
    *ROW_FIELDS,
]
SWEEP_FIELDS = [
    "case",
    "width",
    "selection_policy",
    "target_runtime_sec",
    "runtime_gap_sec",
    "runtime_ratio",
    "exact_moves",
    "move_gap",
    "business_ok",
    "goal_validation_ok",
    "validation_msg",
    "qubo_selected_total",
    "qubo_selected_on_path",
    "qubo_path_share",
    "qubo_ancestor_on_path",
    "qubo_ancestor_path_share",
    "tail_n_trace",
    "tail_ordered_trace",
    "tail_ordered_gain",
    *ROW_FIELDS,
]


def _case_filter(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return [case if case.endswith("_single") else f"{case}_single" for case in args.cases]
    return [f"{zone}_{bay}_single" for zone, bay in discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))]


def _compact_args(
    outer: argparse.Namespace,
    *,
    beam_width: int,
    time_limit: float,
    use_qubo: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        cases=None,
        bays=outer.bays,
        output=outer.output,
        priority=outer.priority,
        g_weight=outer.g_weight,
        n_weight=outer.n_weight,
        target_final_n=0,
        alpha=1.3,
        max_moves=None,
        time_limit=time_limit,
        stream_candidate_limit=outer.stream_candidate_limit if use_qubo else 0,
        stream_candidate_trim_factor=outer.stream_candidate_trim_factor,
        beam_width=beam_width,
        core_width=max(0, min(beam_width, int(round(beam_width * outer.core_ratio)))),
        parent_core_cap=outer.parent_core_cap,
        selection_policy=outer.qubo_policy if use_qubo else outer.no_qubo_policy,
        use_qubo=use_qubo,
        qubo_mode=outer.qubo_mode,
        qubo_profile=(
            "group_majority"
            if outer.qubo_selector == "group"
            else "frontier_cohort_protected_group"
            if outer.qubo_selector == "protected_group"
            else "frontier_cohort"
        ),
        max_qubo_calls=outer.max_qubo_calls if use_qubo else 0,
        max_qubo_vars=outer.max_qubo_vars,
        qubo_core_width=outer.qubo_core_width,
        qubo_supplement_width=outer.qubo_supplement_width,
        qubo_min_candidates=64,
        qubo_priority_bonus=0.0001,
        qubo_diversity_weight=outer.qubo_diversity_weight,
        qubo_frontier_n_window=0,
        qubo_frontier_depth_window=1,
        qubo_focus_n=outer.qubo_focus_n,
        qubo_first=outer.qubo_first,
        qubo_target_share=outer.qubo_target_share,
        qubo_cohorts_per_layer=outer.qubo_cohorts_per_layer,
        qubo_cohort_stride=outer.qubo_cohort_stride,
        group_qubo_supplement=use_qubo and outer.qubo_selector == "protected_group",
        max_group_qubo_calls=outer.max_group_qubo_calls if use_qubo else 0,
        dynamic_profile=getattr(outer, "dynamic_profile", "none"),
        qubo_rank_weight=1.0,
        qubo_n_weight=2.0,
        qubo_depth_weight=1.0,
        qubo_parent_weight=0.15,
        qubo_penalty_a=30.0,
        qubo_time_limit=outer.qubo_time_limit,
        qubo_reads=outer.qubo_reads,
        qubo_iterations=outer.qubo_iterations,
        qubo_group_max_vars=outer.qubo_group_max_vars,
        qubo_group_select=outer.qubo_group_select,
        qubo_group_take=outer.qubo_group_take,
        qubo_group_same_n_penalty=outer.qubo_group_same_n_penalty,
        qubo_group_lookahead_weight=outer.qubo_group_lookahead_weight,
        qubo_group_branch_weight=outer.qubo_group_branch_weight,
        group_qubo_risk_gate=outer.group_qubo_risk_gate,
        group_qubo_risk_min_candidates=outer.group_qubo_risk_min_candidates,
        group_qubo_risk_parent_share=outer.group_qubo_risk_parent_share,
        group_qubo_risk_plateau_window=outer.group_qubo_risk_plateau_window,
        group_qubo_risk_plateau_delta=outer.group_qubo_risk_plateau_delta,
        dynamic_min_frontier=outer.dynamic_min_frontier,
        dynamic_core_frontier=outer.dynamic_core_frontier,
        dynamic_max_frontier=outer.dynamic_max_frontier,
        dynamic_candidate_pool=outer.dynamic_candidate_pool,
        dynamic_qubo_keep_threshold=outer.dynamic_qubo_keep_threshold,
        global_reservoir=outer.global_reservoir if use_qubo else False,
        reservoir_max_states=outer.reservoir_max_states,
        reservoir_pull=outer.reservoir_pull,
        reservoir_add_per_layer=outer.reservoir_add_per_layer,
        post_polish_qubo=use_qubo and outer.post_polish_qubo,
        post_polish_passes=outer.post_polish_passes,
        post_polish_focus_n=outer.post_polish_focus_n,
        post_polish_tail_span=outer.post_polish_tail_span,
        post_polish_max_window=outer.post_polish_max_window,
        post_polish_max_windows=outer.post_polish_max_windows,
        post_polish_select_windows=outer.post_polish_select_windows,
        post_polish_max_qubo_vars=outer.post_polish_max_qubo_vars,
        post_polish_max_bridge_depth=outer.post_polish_max_bridge_depth,
        post_polish_bridge_expanded=outer.post_polish_bridge_expanded,
        post_polish_suffix_focus_n=outer.post_polish_suffix_focus_n,
        post_polish_suffix_tail_span=outer.post_polish_suffix_tail_span,
        post_polish_suffix_max_candidates=outer.post_polish_suffix_max_candidates,
        post_polish_suffix_select=outer.post_polish_suffix_select,
        post_polish_suffix_max_qubo_vars=outer.post_polish_suffix_max_qubo_vars,
        post_polish_suffix_max_depth=outer.post_polish_suffix_max_depth,
        post_polish_suffix_expanded=outer.post_polish_suffix_expanded,
        post_polish_suffix_near_gap=2,
        post_polish_suffix_near_penalty=0.30,
        post_polish_overlap_penalty=8.0,
        post_polish_near_window_gap=2,
        post_polish_penalty_a=18.0,
        post_polish_qubo_time_limit=outer.qubo_time_limit,
        post_polish_qubo_reads=outer.qubo_reads,
        post_polish_qubo_iterations=outer.qubo_iterations,
        tail_order_steps=outer.tail_order_steps,
        seed=outer.seed,
    )


def _run_one(
    case: str,
    args: argparse.Namespace,
    *,
    beam_width: int,
    time_limit: float,
    use_qubo: bool,
    solver_suffix: str,
) -> tuple[dict[str, object], list]:
    zone, bay = _parse_case(case)
    compact_args = _compact_args(args, beam_width=beam_width, time_limit=time_limit, use_qubo=use_qubo)
    selector = (
        _state_frontier_selector
        if compact_args.selection_policy == "dynamic_qubo"
        else _select_group_qubo_frontier
        if args.qubo_selector == "group"
        else _state_frontier_selector
    )
    previous_selector = compact._select_next_frontier
    compact._select_next_frontier = selector
    try:
        result = compact.run_compact_beam(case, zone, bay, compact_args)
    finally:
        compact._select_next_frontier = previous_selector
    row = _row_from_result(result, zone, bay, compact_args)
    row.solver = f"compact_beam_qubo_n0_v1_{args.qubo_selector}_{compact_args.selection_policy}_{solver_suffix}"
    business_ok, goal_ok, validation_msg = _validate_result(result, zone, bay)
    return (
        {
            "row": row,
            "result": result,
            "args": compact_args,
            "business_ok": business_ok,
            "goal_validation_ok": goal_ok,
            "validation_msg": validation_msg,
            "width": beam_width,
        },
        result.moves,
    )


def _ratio(runtime: float, target: float) -> str:
    if target <= 0:
        return ""
    return f"{runtime / target:.6f}"


def _gap(runtime: float, target: float) -> str:
    return f"{runtime - target:.6f}"


def _is_final_n_zero(value: object) -> bool:
    if value is None:
        return False
    try:
        return int(float(value)) == 0
    except (TypeError, ValueError):
        return False


def _quality_key(run: dict[str, object], target_runtime: float) -> tuple[int, int, float, int]:
    row = run["row"]
    final_n = row.final_n if row.final_n is not None else 10**9
    moves = row.moves if row.moves is not None else 10**9
    return final_n, moves, abs(row.runtime_sec - target_runtime), int(run["width"])


def _time_key(run: dict[str, object], target_runtime: float) -> tuple[float, int]:
    row = run["row"]
    return abs(row.runtime_sec - target_runtime), int(run["width"])


def _write_csv(rows: list[dict[str, object]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _moves_path(moves_dir: Path, case: str, role: str, width: int | None) -> Path:
    suffix = f"_w{width}" if width is not None else ""
    return moves_dir / f"{case}_{role}{suffix}_moves.csv"


def _comparison_row(
    run: dict[str, object],
    *,
    role: str,
    target_runtime: float,
    moves_path: Path | None,
    exact_refs: dict[str, int],
) -> dict[str, object]:
    row = run["row"]
    qubo_selected_total, qubo_selected_on_path, qubo_path_share, qubo_ancestor_on_path, qubo_ancestor_path_share = _qubo_path_metrics(row.status, row.moves)
    tail_n_trace = _status_text(row.status, "tail_n_trace")
    tail_ordered_trace = _status_text(row.status, "tail_ordered_trace")
    tail_ordered_gain = _status_text(row.status, "tail_ordered_gain")
    out = {
        "case": row.case,
        "role": role,
        "selected_width": run["width"],
        "selection_policy": run["args"].selection_policy,
        "target_runtime_sec": f"{target_runtime:.6f}",
        "runtime_gap_sec": _gap(row.runtime_sec, target_runtime),
        "runtime_ratio": _ratio(row.runtime_sec, target_runtime),
        "exact_moves": exact_refs.get(row.case, ""),
        "move_gap": _move_gap(row, exact_refs),
        "business_ok": run["business_ok"],
        "goal_validation_ok": run["goal_validation_ok"],
        "validation_msg": run["validation_msg"],
        "moves_path": _rel(moves_path) if moves_path is not None else "",
        "qubo_selected_total": qubo_selected_total,
        "qubo_selected_on_path": qubo_selected_on_path,
        "qubo_path_share": qubo_path_share,
        "qubo_ancestor_on_path": qubo_ancestor_on_path,
        "qubo_ancestor_path_share": qubo_ancestor_path_share,
        "tail_n_trace": tail_n_trace,
        "tail_ordered_trace": tail_ordered_trace,
        "tail_ordered_gain": tail_ordered_gain,
    }
    for field in RESULT_FIELDS:
        out[field] = getattr(row, field)
    return out


def _sweep_row(run: dict[str, object], target_runtime: float, exact_refs: dict[str, int]) -> dict[str, object]:
    row = run["row"]
    qubo_selected_total, qubo_selected_on_path, qubo_path_share, qubo_ancestor_on_path, qubo_ancestor_path_share = _qubo_path_metrics(row.status, row.moves)
    tail_n_trace = _status_text(row.status, "tail_n_trace")
    tail_ordered_trace = _status_text(row.status, "tail_ordered_trace")
    tail_ordered_gain = _status_text(row.status, "tail_ordered_gain")
    out = {
        "case": row.case,
        "width": run["width"],
        "selection_policy": run["args"].selection_policy,
        "target_runtime_sec": f"{target_runtime:.6f}",
        "runtime_gap_sec": _gap(row.runtime_sec, target_runtime),
        "runtime_ratio": _ratio(row.runtime_sec, target_runtime),
        "exact_moves": exact_refs.get(row.case, ""),
        "move_gap": _move_gap(row, exact_refs),
        "business_ok": run["business_ok"],
        "goal_validation_ok": run["goal_validation_ok"],
        "validation_msg": run["validation_msg"],
        "qubo_selected_total": qubo_selected_total,
        "qubo_selected_on_path": qubo_selected_on_path,
        "qubo_path_share": qubo_path_share,
        "qubo_ancestor_on_path": qubo_ancestor_on_path,
        "qubo_ancestor_path_share": qubo_ancestor_path_share,
        "tail_n_trace": tail_n_trace,
        "tail_ordered_trace": tail_ordered_trace,
        "tail_ordered_gain": tail_ordered_gain,
    }
    for field in RESULT_FIELDS:
        out[field] = getattr(row, field)
    return out


def _status_int(status: str, key: str) -> int:
    match = re.search(rf"{re.escape(key)}=(\d+)", status)
    return int(match.group(1)) if match else 0


def _status_text(status: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)}=([^;]*)", status)
    return match.group(1).strip() if match else ""


def _qubo_path_metrics(status: str, moves: int | None) -> tuple[int, int, str, int, str]:
    total = _status_int(status, "qubo_selected_total")
    on_path = _status_int(status, "qubo_selected_on_path")
    ancestor_on_path = _status_int(status, "qubo_ancestor_on_path")
    if not moves:
        return total, on_path, "", ancestor_on_path, ""
    return total, on_path, f"{on_path / moves:.6f}", ancestor_on_path, f"{ancestor_on_path / moves:.6f}"


def _summarize(rows: list[dict[str, object]], *, bay: str | None = None) -> list[dict[str, object]]:
    out = []
    for role in ["qubo_on", "no_qubo_time_matched", "no_qubo_budget_best"]:
        selected = [row for row in rows if row["role"] == role]
        if bay is not None:
            selected = [row for row in selected if str(row["case"]).split("_")[1] == bay]
        if not selected:
            continue
        final_ns = [int(float(row["final_n"])) for row in selected if row.get("final_n") not in (None, "")]
        moves = [int(float(row["moves"])) for row in selected if row.get("moves") not in (None, "")]
        gaps = [int(float(row["move_gap"])) for row in selected if row.get("move_gap") not in (None, "")]
        runtimes = [float(row["runtime_sec"]) for row in selected if row.get("runtime_sec") not in (None, "")]
        ratios = [float(row["runtime_ratio"]) for row in selected if row.get("runtime_ratio") not in (None, "")]
        widths = [int(float(row["selected_width"])) for row in selected if row.get("selected_width") not in (None, "")]
        qcalls = [int(float(row["qubo_calls"])) for row in selected if row.get("qubo_calls") not in (None, "")]
        path_shares = [float(row["qubo_path_share"]) for row in selected if row.get("qubo_path_share") not in (None, "")]
        ancestor_shares = [
            float(row["qubo_ancestor_path_share"])
            for row in selected
            if row.get("qubo_ancestor_path_share") not in (None, "")
        ]
        out.append(
            {
                "role": role,
                "cases": len(selected),
                "solved": sum(1 for value in final_ns if value == 0),
                "avg_final_n": sum(final_ns) / len(final_ns) if final_ns else 0.0,
                "avg_moves": sum(moves) / len(moves) if moves else 0.0,
                "avg_move_gap": sum(gaps) / len(gaps) if gaps else 0.0,
                "avg_runtime": sum(runtimes) / len(runtimes) if runtimes else 0.0,
                "avg_runtime_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
                "avg_width": sum(widths) / len(widths) if widths else 0.0,
                "avg_qcalls": sum(qcalls) / len(qcalls) if qcalls else 0.0,
                "avg_qubo_path_share": sum(path_shares) / len(path_shares) if path_shares else 0.0,
                "avg_qubo_ancestor_path_share": sum(ancestor_shares) / len(ancestor_shares) if ancestor_shares else 0.0,
            }
        )
    return out


def _summary_table(summary: list[dict[str, object]]) -> list[str]:
    lines = [
        "| role | solved | avg final N | avg moves | avg move gap | avg runtime sec | avg runtime ratio | avg width | avg qcalls | avg QUBO direct path share | avg QUBO ancestry share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['role']} | {row['solved']}/{row['cases']} | {row['avg_final_n']:.3f} | "
            f"{row['avg_moves']:.3f} | {row['avg_move_gap']:.3f} | {row['avg_runtime']:.3f} | "
            f"{row['avg_runtime_ratio']:.3f} | {row['avg_width']:.1f} | {row['avg_qcalls']:.2f} | "
            f"{row['avg_qubo_path_share']:.3f} | {row['avg_qubo_ancestor_path_share']:.3f} |"
        )
    return lines


def _write_report(comparison_rows: list[dict[str, object]], sweep_rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    by_case: dict[str, dict[str, dict[str, object]]] = {}
    for row in comparison_rows:
        by_case.setdefault(str(row["case"]), {})[str(row["role"])] = row

    lines = [
        "# Compact QUBO Strict Ablation",
        "",
        "Latest compact byte-state beam run in the same script and current codebase. QUBO-off disables frontier and post-polish QUBO, then uses wider beam widths to match QUBO-on wall time.",
        f"QUBO policy: `{args.qubo_policy}`. no-QUBO policy: `{args.no_qubo_policy}`.",
        "",
        f"Comparison CSV: `{_rel(args.output)}`",
        f"Sweep CSV: `{_rel(args.sweep_output)}`",
        "",
        "## All Cases",
        "",
        *_summary_table(_summarize(comparison_rows)),
        "",
        "## Bay 26 Cases",
        "",
        *_summary_table(_summarize(comparison_rows, bay="26")),
        "",
        "## Per-Case",
        "",
        "| case | QUBO width/N/moves/gap/t/qcalls/direct/ancestry | no-QUBO matched width/N/moves/gap/t | no-QUBO best width/N/moves/gap/t |",
        "|---|---:|---:|---:|",
    ]
    for case in sorted(by_case):
        q = by_case[case].get("qubo_on", {})
        matched = by_case[case].get("no_qubo_time_matched", {})
        best = by_case[case].get("no_qubo_budget_best", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    case,
                    f"w{q.get('selected_width', '')}/{q.get('final_n', '')}/{q.get('moves', '')}/{q.get('move_gap', '')}/{float(q.get('runtime_sec', 0) or 0):.3f}/{q.get('qubo_calls', '')}/{q.get('qubo_path_share', '')}/{q.get('qubo_ancestor_path_share', '')}",
                    f"w{matched.get('selected_width', '')}/{matched.get('final_n', '')}/{matched.get('moves', '')}/{matched.get('move_gap', '')}/{float(matched.get('runtime_sec', 0) or 0):.3f}",
                    f"w{best.get('selected_width', '')}/{best.get('final_n', '')}/{best.get('moves', '')}/{best.get('move_gap', '')}/{float(best.get('runtime_sec', 0) or 0):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Tail Ordered Traces",
            "",
            "| case | QUBO tail N | QUBO tail ordered | matched tail ordered | best tail ordered |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for case in sorted(by_case):
        q = by_case[case].get("qubo_on", {})
        matched = by_case[case].get("no_qubo_time_matched", {})
        best = by_case[case].get("no_qubo_budget_best", {})

        def tail(row: dict[str, object], key: str) -> str:
            final_n = row.get("final_n")
            return str(row.get(key, "")) if final_n not in (None, "") and int(float(final_n)) == 0 else ""

        lines.append(
            "| "
            + " | ".join(
                [
                    case,
                    tail(q, "tail_n_trace"),
                    tail(q, "tail_ordered_trace"),
                    tail(matched, "tail_ordered_trace"),
                    tail(best, "tail_ordered_trace"),
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
            "- `no_qubo_time_matched` is the tried QUBO-off width with runtime closest to the QUBO-on runtime.",
            "- `no_qubo_budget_best` is the best QUBO-off row among tried widths within the configured runtime tolerance.",
            "- `--require-final-zero` defaults to true, so matched/best baselines are selected from rows with final N=0 when available.",
            "- `classic` selection keeps the compact state representation, successor generation, dominated-state pruning, and route reconstruction, but truncates each layer by plain candidate rank without parent caps, diversity supplements, QUBO, or post-polish QUBO.",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(comparison_rows: list[dict[str, object]]) -> None:
    print("\nSUMMARY all cases", flush=True)
    for line in _summary_table(_summarize(comparison_rows)):
        print(line, flush=True)
    print("\nSUMMARY bay 26", flush=True)
    for line in _summary_table(_summarize(comparison_rows, bay="26")):
        print(line, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sweep-output", type=Path)
    parser.add_argument("--moves-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--qubo-beam-width", type=int, default=1200)
    parser.add_argument("--no-qubo-widths", nargs="+", type=int, default=[1300, 1400, 1600, 2000, 2400, 3200, 4800, 6400])
    parser.add_argument("--qubo-policy", choices=["enhanced", "classic", "dynamic_qubo"], default="enhanced")
    parser.add_argument("--no-qubo-policy", choices=["classic", "enhanced", "dynamic_qubo"], default="classic")
    parser.add_argument("--core-ratio", type=float, default=0.90)
    parser.add_argument("--parent-core-cap", type=int, default=4)
    parser.add_argument("--priority", choices=["n_then_depth", "astar", "weighted_astar"], default="n_then_depth")
    parser.add_argument("--g-weight", type=float, default=1.0)
    parser.add_argument("--n-weight", type=float, default=1.0)
    parser.add_argument("--time-tolerance", type=float, default=0.05)
    parser.add_argument("--time-limit-factor", type=float, default=1.0)
    parser.add_argument("--stream-candidate-limit", type=int, default=0)
    parser.add_argument("--stream-candidate-trim-factor", type=float, default=2.0)
    parser.add_argument("--small-target-sec", type=float, default=0.10)
    parser.add_argument("--sweep-all-no-qubo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-final-zero", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-qubo-calls", type=int, default=32)
    parser.add_argument("--qubo-selector", choices=["state", "group", "protected_group"], default="state")
    parser.add_argument("--qubo-mode", choices=["supplement", "focused_multi", "majority_multi"], default="focused_multi")
    parser.add_argument("--max-qubo-vars", type=int, default=64)
    parser.add_argument("--qubo-core-width", type=int, default=224)
    parser.add_argument("--qubo-supplement-width", type=int, default=24)
    parser.add_argument("--qubo-focus-n", type=int, default=8)
    parser.add_argument("--qubo-first", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qubo-target-share", type=float, default=0.50)
    parser.add_argument("--qubo-cohorts-per-layer", type=int, default=2)
    parser.add_argument("--qubo-cohort-stride", type=int, default=64)
    parser.add_argument("--qubo-diversity-weight", type=float, default=0.10)
    parser.add_argument("--qubo-time-limit", type=float, default=0.001)
    parser.add_argument("--qubo-reads", type=int, default=2)
    parser.add_argument("--qubo-iterations", type=int, default=20)
    parser.add_argument("--qubo-group-max-vars", type=int, default=96)
    parser.add_argument("--qubo-group-select", type=int, default=72)
    parser.add_argument("--qubo-group-take", type=int, default=8)
    parser.add_argument("--qubo-group-same-n-penalty", type=float, default=0.04)
    parser.add_argument("--qubo-group-lookahead-weight", type=float, default=0.0)
    parser.add_argument("--qubo-group-branch-weight", type=float, default=0.0)
    parser.add_argument("--group-qubo-risk-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--group-qubo-risk-min-candidates", type=int, default=50000)
    parser.add_argument("--group-qubo-risk-parent-share", type=float, default=0.12)
    parser.add_argument("--group-qubo-risk-plateau-window", type=int, default=3)
    parser.add_argument("--group-qubo-risk-plateau-delta", type=int, default=0)
    parser.add_argument("--dynamic-min-frontier", type=int, default=128)
    parser.add_argument("--dynamic-core-frontier", type=int, default=0)
    parser.add_argument("--dynamic-max-frontier", type=int, default=0)
    parser.add_argument("--dynamic-candidate-pool", type=int, default=4096)
    parser.add_argument("--dynamic-qubo-keep-threshold", type=float, default=0.55)
    parser.add_argument(
        "--dynamic-profile",
        choices=sorted(compact._DYNAMIC_QUBO_PRESETS.keys()),
        default="legacy",
        help=(
            "Reusable preset for dynamic frontier. Use 'legacy' for quality, "
            "'speed2000' for A*-speed alignment, or 'balanced2200' for the tuned speed-quality profile."
        ),
    )
    parser.add_argument("--global-reservoir", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reservoir-max-states", type=int, default=50000)
    parser.add_argument("--reservoir-pull", type=int, default=800)
    parser.add_argument("--reservoir-add-per-layer", type=int, default=4000)
    parser.add_argument("--max-group-qubo-calls", type=int, default=32)
    parser.add_argument("--post-polish-qubo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-polish-passes", type=int, default=2)
    parser.add_argument("--post-polish-focus-n", type=int, default=8)
    parser.add_argument("--post-polish-tail-span", type=int, default=14)
    parser.add_argument("--post-polish-max-window", type=int, default=8)
    parser.add_argument("--post-polish-max-windows", type=int, default=96)
    parser.add_argument("--post-polish-select-windows", type=int, default=0)
    parser.add_argument("--post-polish-max-qubo-vars", type=int, default=72)
    parser.add_argument("--post-polish-max-bridge-depth", type=int, default=7)
    parser.add_argument("--post-polish-bridge-expanded", type=int, default=20000)
    parser.add_argument("--post-polish-suffix-focus-n", type=int, default=9)
    parser.add_argument("--post-polish-suffix-tail-span", type=int, default=18)
    parser.add_argument("--post-polish-suffix-max-candidates", type=int, default=48)
    parser.add_argument("--post-polish-suffix-select", type=int, default=5)
    parser.add_argument("--post-polish-suffix-max-qubo-vars", type=int, default=48)
    parser.add_argument("--post-polish-suffix-max-depth", type=int, default=18)
    parser.add_argument("--post-polish-suffix-expanded", type=int, default=60000)
    parser.add_argument("--tail-order-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    if not args.output.is_absolute():
        args.output = ROOT / args.output
    if args.sweep_output is None:
        args.sweep_output = args.output.with_name(args.output.stem + "_sweep.csv")
    elif not args.sweep_output.is_absolute():
        args.sweep_output = ROOT / args.sweep_output
    if args.moves_dir is None:
        args.moves_dir = args.output.with_name(args.output.stem + "_moves")
    elif not args.moves_dir.is_absolute():
        args.moves_dir = ROOT / args.moves_dir
    if args.report is None:
        args.report = args.output.with_suffix(".md")
    elif not args.report.is_absolute():
        args.report = ROOT / args.report

    args.moves_dir.mkdir(parents=True, exist_ok=True)
    exact_refs = _exact_moves_by_case()
    comparison_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    start = time.perf_counter()
    cases = _case_filter(args)

    for idx, case in enumerate(cases, start=1):
        zone, bay = _parse_case(case)
        instance = _instance_from_single_bay(zone, bay)
        print(f"CASE_START {idx}/{len(cases)} {case} strict_compact_qubo", flush=True)

        qubo_run, qubo_moves = _run_one(
            case,
            args,
            beam_width=args.qubo_beam_width,
            time_limit=0.0,
            use_qubo=True,
            solver_suffix="qubo_on",
        )
        target_runtime = qubo_run["row"].runtime_sec
        qubo_path = None
        if qubo_moves:
            qubo_path = _moves_path(args.moves_dir, case, "qubo_on", args.qubo_beam_width)
            _write_explain_csv(instance, qubo_moves, qubo_path)

        tried: list[dict[str, object]] = []
        target_low = target_runtime * (1.0 - args.time_tolerance)
        target_high = target_runtime * (1.0 + args.time_tolerance)
        widths = sorted(dict.fromkeys(width for width in args.no_qubo_widths if width > args.qubo_beam_width))
        if target_runtime <= args.small_target_sec:
            widths = widths[:1]
        for width in widths:
            no_qubo_run, _moves = _run_one(
                case,
                args,
                beam_width=width,
                time_limit=max(0.01, target_runtime * args.time_limit_factor),
                use_qubo=False,
                solver_suffix="no_qubo_wide",
            )
            tried.append(no_qubo_run)
            sweep_rows.append(_sweep_row(no_qubo_run, target_runtime, exact_refs))
            row = no_qubo_run["row"]
            print(
                f"  noQUBO width={width} finalN={row.final_n} moves={row.moves} "
                f"runtime={row.runtime_sec:.3f}s ratio={_ratio(row.runtime_sec, target_runtime)}",
                flush=True,
            )
            if (
                not args.sweep_all_no_qubo
                and target_runtime > args.small_target_sec
                and row.runtime_sec >= target_low
                and _is_final_n_zero(row.final_n)
            ):
                break

        if args.require_final_zero:
            solved_tried = [run for run in tried if _is_final_n_zero(run["row"].final_n)]
            time_source = solved_tried if solved_tried else tried
        else:
            time_source = tried
        time_matched = min(time_source, key=lambda run: _time_key(run, target_runtime))
        if args.require_final_zero:
            budget_candidates = [
                run for run in solved_tried if run["row"].runtime_sec <= target_high
            ] if solved_tried else []
            if not budget_candidates:
                budget_candidates = solved_tried
        else:
            budget_candidates = [run for run in tried if run["row"].runtime_sec <= target_high]
        if not budget_candidates:
            budget_candidates = tried
        budget_best = min(budget_candidates, key=lambda run: _quality_key(run, target_runtime))

        for role, run in [
            ("qubo_on", qubo_run),
            ("no_qubo_time_matched", time_matched),
            ("no_qubo_budget_best", budget_best),
        ]:
            move_path = qubo_path if role == "qubo_on" else None
            if role != "qubo_on" and run["result"].moves:
                move_path = _moves_path(args.moves_dir, case, role, int(run["width"]))
                _write_explain_csv(instance, run["result"].moves, move_path)
            comparison_rows.append(
                _comparison_row(
                    run,
                    role=role,
                    target_runtime=target_runtime,
                    moves_path=move_path,
                    exact_refs=exact_refs,
                )
            )

        _write_csv(comparison_rows, args.output, COMPARISON_FIELDS)
        _write_csv(sweep_rows, args.sweep_output, SWEEP_FIELDS)
        print(
            f"CASE_DONE {case} QUBO N={qubo_run['row'].final_n} moves={qubo_run['row'].moves} "
            f"t={target_runtime:.3f}s qcalls={qubo_run['row'].qubo_calls} "
            f"matched=w{time_matched['width']}/N{time_matched['row'].final_n}/moves{time_matched['row'].moves}/t{time_matched['row'].runtime_sec:.3f}s "
            f"best=w{budget_best['width']}/N{budget_best['row'].final_n}/moves{budget_best['row'].moves}/t{budget_best['row'].runtime_sec:.3f}s",
            flush=True,
        )

    _write_csv(comparison_rows, args.output, COMPARISON_FIELDS)
    _write_csv(sweep_rows, args.sweep_output, SWEEP_FIELDS)
    _write_report(comparison_rows, sweep_rows, args)
    _print_summary(comparison_rows)
    print(
        f"\nDONE cases={len(comparison_rows) // 3} sweep_runs={len(sweep_rows)} wall={time.perf_counter() - start:.3f}s",
        flush=True,
    )
    print(f"OUTPUT comparison={_rel(args.output)} sweep={_rel(args.sweep_output)} report={_rel(args.report)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
