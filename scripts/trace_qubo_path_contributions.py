"""Trace action-level QUBO contributions in the stable compact QUBO beam.

The production benchmark records aggregate path counters such as
``qubo_selected_on_path``.  This diagnostic script reruns the same stable
configuration and writes the concrete moves that correspond to those counters.
It does not change the solver.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import benchmark_compact_beam_qubo_n0 as compact  # noqa: E402
from benchmark_explainable_rule_macro_beam import classify_move  # noqa: E402
from benchmark_fast_qubo_integrated_v1 import _parse_case, _rel  # noqa: E402
from container_reshuffle.core import apply_move, create_initial_state  # noqa: E402
from container_reshuffle.features import disorder_box_count, entropy_limit, entropy_value  # noqa: E402
from run_business_exact_target_search_sqlite import _encode, _remaining_lower_bound  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


DEFAULT_CASES = ["A2_26_single", "C1_26_single", "C2_26_single"]
STEP_FIELDS = [
    "case",
    "path_kind",
    "step",
    "box_rank",
    "from_stack",
    "to_stack",
    "reason",
    "n_before",
    "n_after",
    "g_after",
    "source",
    "direct_qubo_selected",
    "qubo_ancestor",
]
EVENT_FIELDS = [
    "case",
    "event_type",
    "pass_index",
    "start_step",
    "end_step",
    "old_moves",
    "new_moves",
    "moves_saved",
    "n_at_start",
    "selected_by_qubo",
    "score",
    "expanded",
]
DETAIL_FIELDS = [
    "case",
    "event_type",
    "event_start_step",
    "event_end_step",
    "local_index",
    "old_step",
    "old_move",
    "old_reason",
    "old_n",
    "new_step",
    "new_move",
    "new_reason",
    "new_n",
    "move_delta",
]


def _default_solver_args() -> argparse.Namespace:
    return argparse.Namespace(
        priority="n_then_depth",
        solver_profile="speed_v2",
        g_weight=1.0,
        n_weight=1.0,
        target_final_n=0,
        alpha=1.3,
        max_moves=None,
        time_limit=0.0,
        stream_candidate_limit=0,
        stream_candidate_trim_factor=2.0,
        beam_width=1200,
        core_width=1080,
        parent_core_cap=4,
        selection_policy="enhanced",
        use_qubo=True,
        qubo_mode="focused_multi",
        qubo_influence_profile="stable",
        qubo_profile="frontier_cohort",
        max_qubo_calls=32,
        max_qubo_vars=64,
        qubo_core_width=224,
        qubo_supplement_width=24,
        qubo_min_candidates=64,
        qubo_priority_bonus=0.0001,
        qubo_diversity_weight=0.10,
        qubo_frontier_n_window=0,
        qubo_frontier_depth_window=1,
        qubo_focus_n=8,
        qubo_first=False,
        qubo_target_share=0.50,
        qubo_cohorts_per_layer=2,
        qubo_cohort_stride=64,
        group_qubo_supplement=False,
        max_group_qubo_calls=32,
        qubo_group_max_vars=96,
        qubo_group_select=24,
        qubo_group_take=4,
        qubo_group_same_n_penalty=0.04,
        qubo_group_lookahead_weight=0.0,
        qubo_group_branch_weight=0.0,
        group_qubo_risk_gate=False,
        group_qubo_risk_min_candidates=50000,
        group_qubo_risk_parent_share=0.12,
        group_qubo_risk_plateau_window=3,
        group_qubo_risk_plateau_delta=0,
        dynamic_min_frontier=128,
        dynamic_core_frontier=0,
        dynamic_max_frontier=0,
        dynamic_candidate_pool=4096,
        dynamic_qubo_keep_threshold=0.55,
        dynamic_profile="legacy",
        rank_pool_size=0,
        rank_pool_min_candidates=0,
        global_reservoir=False,
        reservoir_max_states=50000,
        reservoir_pull=800,
        reservoir_add_per_layer=4000,
        qubo_rank_weight=1.0,
        qubo_n_weight=2.0,
        qubo_depth_weight=1.0,
        qubo_parent_weight=0.15,
        qubo_penalty_a=30.0,
        qubo_time_limit=0.001,
        qubo_reads=2,
        qubo_iterations=20,
        post_polish_qubo=True,
        post_polish_passes=2,
        post_polish_focus_n=8,
        post_polish_tail_span=14,
        post_polish_max_window=8,
        post_polish_max_windows=96,
        post_polish_select_windows=0,
        post_polish_max_qubo_vars=72,
        post_polish_max_bridge_depth=7,
        post_polish_bridge_expanded=20000,
        post_polish_suffix_focus_n=9,
        post_polish_suffix_tail_span=18,
        post_polish_suffix_max_candidates=48,
        post_polish_suffix_select=5,
        post_polish_suffix_max_qubo_vars=48,
        post_polish_suffix_max_depth=18,
        post_polish_suffix_expanded=60000,
        post_polish_suffix_near_gap=2,
        post_polish_suffix_near_penalty=0.30,
        post_polish_overlap_penalty=8.0,
        post_polish_near_window_gap=2,
        post_polish_penalty_a=18.0,
        post_polish_qubo_time_limit=0.001,
        post_polish_qubo_reads=2,
        post_polish_qubo_iterations=20,
        tail_order_steps=6,
        seed=37,
    )


def _label(idx: int) -> str:
    return str(idx + 1) if idx < 9 else "A"


def _path_step_rows(
    case: str,
    path_kind: str,
    zone: str,
    bay: str,
    codes: list[bytes],
    source_by_code: dict[bytes, str],
    qubo_ancestor_by_code: dict[bytes, bool],
) -> list[dict[str, object]]:
    instance = _instance_from_single_bay(zone, bay)
    moves = compact._codes_to_moves(zone, bay, codes)
    rows: list[dict[str, object]] = []
    state = create_initial_state(instance)
    for step, move in enumerate(moves, start=1):
        before = state
        after = apply_move(instance, state, move)
        code_after = codes[step]
        source = source_by_code.get(code_after, "post_polish")
        rows.append(
            {
                "case": case,
                "path_kind": path_kind,
                "step": step,
                "box_rank": move.box,
                "from_stack": _label(move.from_stack),
                "to_stack": _label(move.to_stack),
                "reason": classify_move(instance, before, after, move),
                "n_before": disorder_box_count(instance, before),
                "n_after": disorder_box_count(instance, after),
                "g_after": entropy_value(instance, after),
                "source": source,
                "direct_qubo_selected": source == "qubo",
                "qubo_ancestor": qubo_ancestor_by_code.get(code_after, False),
            }
        )
        state = after
    return rows


def _trace_post_polish(
    case: str,
    codes: list[bytes],
    stack_count: int,
    max_height: int,
    cap: int,
    args: argparse.Namespace,
    deadline: float,
) -> tuple[list[bytes], compact.PostPolishStats, list[dict[str, object]]]:
    stats = compact.PostPolishStats()
    events: list[dict[str, object]] = []
    if not args.post_polish_qubo or len(codes) <= 3:
        return codes, stats, events

    current = list(codes)
    for pass_idx in range(max(1, args.post_polish_passes)):
        if time.perf_counter() >= deadline:
            break
        n_values = compact._path_n_values(current, stack_count, max_height)
        suffixes = compact._post_polish_suffixes(current, n_values, args)
        selected_suffixes, suffix_qvars, suffix_qtime = compact._select_post_polish_suffixes_qubo(suffixes, args)
        if suffix_qvars:
            stats.qcalls += 1
            stats.qvars += suffix_qvars
            stats.qtime += suffix_qtime
        improved_this_pass = False
        for suffix in selected_suffixes:
            if time.perf_counter() >= deadline:
                break
            if suffix.start >= len(current) - 1:
                continue
            old_remaining = len(current) - 1 - suffix.start
            max_depth = min(old_remaining - 1, args.post_polish_suffix_max_depth)
            stats.attempts += 1
            stats.suffix_attempts += 1
            bridge, expanded = compact._suffix_shorter_path_to_n0(
                current[suffix.start],
                compact._code_n(current[suffix.start], stack_count, max_height),
                suffix.start,
                cap,
                stack_count,
                max_height,
                max_depth,
                deadline,
                args,
            )
            stats.suffix_expanded += expanded
            if bridge is None:
                continue
            saved = old_remaining - (len(bridge) - 1)
            if saved <= 0:
                continue
            candidate = current[: suffix.start] + bridge
            if not compact._path_business_ok(candidate, stack_count, max_height, cap):
                continue
            events.append(
                {
                    "case": case,
                    "event_type": "post_polish_suffix",
                    "pass_index": pass_idx,
                    "start_step": suffix.start + 1,
                    "end_step": len(current) - 1,
                    "old_moves": old_remaining,
                    "new_moves": len(bridge) - 1,
                    "moves_saved": saved,
                    "n_at_start": suffix.n_value,
                    "selected_by_qubo": suffix_qvars > 0,
                    "score": f"{suffix.score:.6f}",
                    "expanded": expanded,
                }
            )
            current = candidate
            stats.improvements += 1
            stats.suffix_improvements += 1
            stats.moves_saved += saved
            improved_this_pass = True
            break
        if improved_this_pass:
            continue

        n_values = compact._path_n_values(current, stack_count, max_height)
        windows = compact._post_polish_windows(current, n_values, args)
        selected, qvars, qtime = compact._select_post_polish_windows_qubo(windows, args)
        if qvars:
            stats.qcalls += 1
            stats.qvars += qvars
            stats.qtime += qtime
        if not selected:
            break
        for window in selected:
            if time.perf_counter() >= deadline:
                break
            if window.end >= len(current):
                continue
            old_len = window.end - window.start
            max_depth = min(old_len - 1, args.post_polish_max_bridge_depth)
            stats.attempts += 1
            bridge, expanded = compact._bridge_shorter_path(
                current[window.start],
                current[window.end],
                window.start,
                cap,
                stack_count,
                max_height,
                max_depth,
                deadline,
                args,
            )
            stats.bridge_expanded += expanded
            if bridge is None:
                continue
            saved = old_len - (len(bridge) - 1)
            if saved <= 0:
                continue
            candidate = current[: window.start] + bridge + current[window.end + 1 :]
            if not compact._path_business_ok(candidate, stack_count, max_height, cap):
                continue
            events.append(
                {
                    "case": case,
                    "event_type": "post_polish_window",
                    "pass_index": pass_idx,
                    "start_step": window.start + 1,
                    "end_step": window.end,
                    "old_moves": old_len,
                    "new_moves": len(bridge) - 1,
                    "moves_saved": saved,
                    "n_at_start": n_values[window.start],
                    "selected_by_qubo": qvars > 0,
                    "score": f"{window.score:.6f}",
                    "expanded": expanded,
                }
            )
            current = candidate
            stats.improvements += 1
            stats.moves_saved += saved
            improved_this_pass = True
        if not improved_this_pass:
            break
    return current, stats, events


def _trace_case(case: str, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    compact._apply_dynamic_profile(args)
    start = time.perf_counter()
    deadline = start + args.time_limit if args.time_limit > 0 else math.inf
    zone, bay = _parse_case(case)
    instance = _instance_from_single_bay(zone, bay)
    initial, n0 = _encode(instance.stacks, instance.stack_count, instance.max_height)
    cap = entropy_limit(instance, args.alpha)
    depth_cap = min(cap, args.max_moves) if args.max_moves is not None else cap

    parent: dict[bytes, bytes | None] = {initial: None}
    best_depth: dict[bytes, int] = {initial: 0}
    source_by_code: dict[bytes, str] = {initial: "initial"}
    qubo_ancestor_by_code: dict[bytes, bool] = {initial: False}
    reservoir: dict[bytes, tuple[int, int, bytes]] = {}
    frontier: list[tuple[bytes, int, int]] = [(initial, n0, 0)]
    target: bytes | None = initial if n0 <= args.target_final_n else None

    args._qubo_calls_done = 0
    args._group_qubo_calls_done = 0
    args._group_qubo_vars = 0
    args._group_qubo_time = 0.0
    args._group_qubo_selected_total = 0
    args._group_lookahead_cache = {}
    args._best_n_history = []
    args._layout_target_groups = []
    args._stack_count = instance.stack_count
    args._max_height = instance.max_height

    qcalls = qvars = 0
    qtime = 0.0
    expanded = generated = pruned_bound = pruned_business = dominated = 0
    stream_candidate_limit = max(0, getattr(args, "stream_candidate_limit", 0))
    stream_trim_enabled = stream_candidate_limit > 0
    stream_seq = 0

    while target is None and frontier and time.perf_counter() < deadline:
        candidates: dict[bytes, tuple[int, int, bytes]] = {}
        candidate_worst_heap: list[tuple[tuple[float, int, int, int, bytes], bytes, tuple[int, int, bytes]]] = []
        goals: dict[bytes, tuple[int, int, bytes]] = {}
        for code, current_n, depth in frontier:
            if time.perf_counter() >= deadline:
                break
            if current_n <= args.target_final_n:
                target = code
                break
            if _remaining_lower_bound(current_n, depth, depth_cap) > args.target_final_n:
                pruned_bound += 1
                continue
            if depth >= depth_cap:
                pruned_bound += 1
                continue
            expanded += 1
            next_depth = depth + 1
            for nxt, nxt_n in compact._successors_with_n_cached(
                code,
                current_n,
                instance.stack_count,
                instance.max_height,
            ):
                generated += 1
                if next_depth + nxt_n > cap:
                    pruned_business += 1
                    continue
                old_depth = best_depth.get(nxt)
                if old_depth is not None and next_depth >= old_depth:
                    dominated += 1
                    continue
                old_candidate = candidates.get(nxt)
                if old_candidate is not None and next_depth >= old_candidate[1]:
                    dominated += 1
                    continue
                data = (nxt_n, next_depth, code)
                if nxt_n <= args.target_final_n:
                    goals[nxt] = data
                if stream_trim_enabled:
                    stream_seq += 1
                    compact._offer_stream_candidate(candidates, candidate_worst_heap, nxt, data, args, stream_seq)
                else:
                    candidates[nxt] = data
        if target is not None:
            break
        if goals:
            target, data = min(goals.items(), key=lambda item: compact._candidate_rank(item, args))
            parent[target] = data[2]
            best_depth[target] = data[1]
            source_by_code[target] = "goal"
            qubo_ancestor_by_code[target] = qubo_ancestor_by_code.get(data[2], False)
            break
        if not candidates:
            break
        compact._reservoir_pull(reservoir, candidates, args)
        if stream_trim_enabled:
            compact._trim_stream_candidates(candidates, candidate_worst_heap, stream_candidate_limit, args)
        parent_counts: dict[bytes, int] = {}
        for _code, data in candidates.items():
            parent_counts[data[2]] = parent_counts.get(data[2], 0) + 1
        args._layer_candidate_count = len(candidates)
        args._layer_parent_count = len(parent_counts)
        args._layer_parent_top_share = (max(parent_counts.values()) / len(candidates)) if candidates else 0.0
        args._best_n_history.append(min(data[0] for data in candidates.values()))
        selected_codes, selected_by_qubo, add_qcalls, add_qvars, add_qtime = compact._select_next_frontier(candidates, args)
        qcalls += add_qcalls
        qvars += add_qvars
        qtime += add_qtime
        selected_set = set(selected_codes)
        if getattr(args, "global_reservoir", False):
            max_add = max(0, getattr(args, "reservoir_add_per_layer", 0))
            added_this_layer = 0
            for code, data in sorted(candidates.items(), key=lambda item: compact._candidate_rank(item, args)):
                if code in selected_set:
                    continue
                n_value, depth, parent_code = data
                old_depth = best_depth.get(code)
                if old_depth is not None and depth >= old_depth:
                    continue
                parent[code] = parent_code
                best_depth[code] = depth
                source_by_code.setdefault(code, "reservoir")
                qubo_ancestor_by_code.setdefault(code, qubo_ancestor_by_code.get(parent_code, False))
                reservoir[code] = data
                added_this_layer += 1
                if max_add > 0 and added_this_layer >= max_add:
                    break
            compact._reservoir_trim(reservoir, args)
        next_frontier: list[tuple[bytes, int, int]] = []
        for code in selected_codes:
            n_value, depth, parent_code = candidates[code]
            parent[code] = parent_code
            best_depth[code] = depth
            source_by_code[code] = "qubo" if code in selected_by_qubo else "deterministic"
            qubo_ancestor_by_code[code] = code in selected_by_qubo or qubo_ancestor_by_code.get(parent_code, False)
            next_frontier.append((code, n_value, depth))
        frontier = next_frontier

    if target is None:
        raise RuntimeError(f"{case} did not reach N=0")

    original_codes = compact._backtrack(parent, target)
    polished_codes, polish_stats, events = _trace_post_polish(
        case,
        original_codes,
        instance.stack_count,
        instance.max_height,
        cap,
        args,
        deadline,
    )
    rows = _path_step_rows(case, "search_path_before_polish", zone, bay, original_codes, source_by_code, qubo_ancestor_by_code)
    rows += _path_step_rows(case, "final_path_after_polish", zone, bay, polished_codes, source_by_code, qubo_ancestor_by_code)
    summary = {
        "case": case,
        "search_moves": len(original_codes) - 1,
        "final_moves": len(polished_codes) - 1,
        "direct_qubo_steps_search": sum(1 for code in original_codes[1:] if source_by_code.get(code) == "qubo"),
        "qubo_ancestor_steps_search": sum(1 for code in original_codes[1:] if qubo_ancestor_by_code.get(code, False)),
        "direct_qubo_steps_final": sum(1 for code in polished_codes[1:] if source_by_code.get(code) == "qubo"),
        "qubo_ancestor_steps_final": sum(1 for code in polished_codes[1:] if qubo_ancestor_by_code.get(code, False)),
        "post_polish_improvements": polish_stats.improvements,
        "post_polish_moves_saved": polish_stats.moves_saved,
        "qubo_calls": qcalls + polish_stats.qcalls,
        "qubo_vars": qvars + polish_stats.qvars,
        "qubo_time_sec": f"{qtime + polish_stats.qtime:.6f}",
        "runtime_sec": f"{time.perf_counter() - start:.6f}",
    }
    return rows, events, summary


def _write_csv(rows: list[dict[str, object]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _move_text(row: dict[str, object] | None) -> str:
    if row is None:
        return ""
    return f"{row['box_rank']}:{row['from_stack']}->{row['to_stack']}"


def _n_text(row: dict[str, object] | None) -> str:
    if row is None:
        return ""
    return f"{row['n_before']}->{row['n_after']}"


def _post_polish_detail_rows(
    step_rows: list[dict[str, object]],
    events: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_path_step: dict[tuple[str, str, int], dict[str, object]] = {}
    for row in step_rows:
        by_path_step[(str(row["case"]), str(row["path_kind"]), int(row["step"]))] = row

    details: list[dict[str, object]] = []
    for event in events:
        case = str(event["case"])
        start = int(event["start_step"])
        old_moves = int(event["old_moves"])
        new_moves = int(event["new_moves"])
        span = max(old_moves, new_moves)
        for local_idx in range(span):
            old_step = start + local_idx if local_idx < old_moves else None
            new_step = start + local_idx if local_idx < new_moves else None
            old_row = (
                by_path_step.get((case, "search_path_before_polish", old_step))
                if old_step is not None
                else None
            )
            new_row = (
                by_path_step.get((case, "final_path_after_polish", new_step))
                if new_step is not None
                else None
            )
            if old_row is None:
                delta = "added"
            elif new_row is None:
                delta = "removed"
            elif _move_text(old_row) == _move_text(new_row):
                delta = "same"
            else:
                delta = "changed"
            details.append(
                {
                    "case": case,
                    "event_type": event["event_type"],
                    "event_start_step": event["start_step"],
                    "event_end_step": event["end_step"],
                    "local_index": local_idx + 1,
                    "old_step": "" if old_step is None else old_step,
                    "old_move": _move_text(old_row),
                    "old_reason": "" if old_row is None else old_row["reason"],
                    "old_n": _n_text(old_row),
                    "new_step": "" if new_step is None else new_step,
                    "new_move": _move_text(new_row),
                    "new_reason": "" if new_row is None else new_row["reason"],
                    "new_n": _n_text(new_row),
                    "move_delta": delta,
                }
            )
    return details


def _write_report(
    summaries: list[dict[str, object]],
    step_rows: list[dict[str, object]],
    events: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    report: Path,
    step_path: Path,
    event_path: Path,
    detail_path: Path,
) -> None:
    lines = [
        "# QUBO Path Contribution Trace",
        "",
        "Stable compact QUBO Beam rerun with action-level QUBO provenance.",
        "",
        f"Step CSV: `{_rel(step_path)}`",
        f"Event CSV: `{_rel(event_path)}`",
        f"Detail CSV: `{_rel(detail_path)}`",
        "",
        "## Summary",
        "",
        "| case | search moves | final moves | direct QUBO steps | QUBO ancestor steps | polish saved | QUBO calls | runtime sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['case']} | {row['search_moves']} | {row['final_moves']} | "
            f"{row['direct_qubo_steps_final']} | {row['qubo_ancestor_steps_final']} | "
            f"{row['post_polish_moves_saved']} | {row['qubo_calls']} | {float(row['runtime_sec']):.3f} |"
        )
    lines.extend(["", "## Direct QUBO Moves", "", "| case | path | step | move | reason | N | source | ancestor |", "|---|---|---:|---|---|---|---|---|"])
    for row in step_rows:
        if row["direct_qubo_selected"] == "True" or row["direct_qubo_selected"] is True:
            lines.append(
                f"| {row['case']} | {row['path_kind']} | {row['step']} | "
                f"{row['box_rank']}:{row['from_stack']}->{row['to_stack']} | {row['reason']} | "
                f"{row['n_before']}->{row['n_after']} | {row['source']} | {row['qubo_ancestor']} |"
            )
    lines.extend(["", "## Post-Polish Events", "", "| case | type | start-end | old/new/saved | N at start | selected by QUBO | expanded |", "|---|---|---:|---:|---:|---:|---:|"])
    for event in events:
        lines.append(
            f"| {event['case']} | {event['event_type']} | {event['start_step']}-{event['end_step']} | "
            f"{event['old_moves']}/{event['new_moves']}/{event['moves_saved']} | "
            f"{event['n_at_start']} | {event['selected_by_qubo']} | {event['expanded']} |"
        )
    lines.extend(
        [
            "",
            "## Post-Polish Replacement Details",
            "",
            "| case | local | old step | old move | old N | new step | new move | new N | delta |",
            "|---|---:|---:|---|---|---:|---|---|---|",
        ]
    )
    for row in detail_rows:
        lines.append(
            f"| {row['case']} | {row['local_index']} | {row['old_step']} | {row['old_move']} | "
            f"{row['old_n']} | {row['new_step']} | {row['new_move']} | {row['new_n']} | "
            f"{row['move_delta']} |"
        )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "qubo_path_contributions.csv")
    parser.add_argument(
        "--no-frontier-qubo",
        action="store_true",
        help="Disable frontier QUBO while leaving post-polish QUBO enabled.",
    )
    parser.add_argument(
        "--no-post-polish-qubo",
        action="store_true",
        help="Disable post-polish QUBO as well.",
    )
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    event_output = output.with_name(output.stem + "_events.csv")
    detail_output = output.with_name(output.stem + "_details.csv")
    summary_output = output.with_name(output.stem + "_summary.csv")
    report = output.with_suffix(".md")

    step_rows: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for idx, case in enumerate(args.cases, start=1):
        solver_args = _default_solver_args()
        if args.no_frontier_qubo:
            solver_args.use_qubo = False
            solver_args.max_qubo_calls = 0
        if args.no_post_polish_qubo:
            solver_args.post_polish_qubo = False
        print(f"TRACE_START {idx}/{len(args.cases)} {case}", flush=True)
        rows, case_events, summary = _trace_case(case, solver_args)
        step_rows.extend(rows)
        events.extend(case_events)
        summaries.append(summary)
        print(
            f"TRACE_DONE {case} moves={summary['final_moves']} "
            f"direct_qubo={summary['direct_qubo_steps_final']} "
            f"ancestor={summary['qubo_ancestor_steps_final']} "
            f"polish_saved={summary['post_polish_moves_saved']}",
            flush=True,
        )

    detail_rows = _post_polish_detail_rows(step_rows, events)
    _write_csv(step_rows, output, STEP_FIELDS)
    _write_csv(events, event_output, EVENT_FIELDS)
    _write_csv(detail_rows, detail_output, DETAIL_FIELDS)
    _write_csv(summaries, summary_output, list(summaries[0].keys()) if summaries else ["case"])
    _write_report(summaries, step_rows, events, detail_rows, report, output, event_output, detail_output)
    print(f"wrote {_rel(output)}", flush=True)
    print(f"wrote {_rel(event_output)}", flush=True)
    print(f"wrote {_rel(detail_output)}", flush=True)
    print(f"wrote {_rel(summary_output)}", flush=True)
    print(f"wrote {_rel(report)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
