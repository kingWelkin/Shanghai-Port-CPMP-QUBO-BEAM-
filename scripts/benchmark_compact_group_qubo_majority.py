"""Experimental group-level QUBO-majority frontier selector.

This keeps the latest compact byte-state search intact and only replaces the
frontier selector.  QUBO variables represent parent groups rather than
individual states.  A selected group contributes several ranked child states,
so QUBO can control a large share of the next frontier with a moderate number
of binary variables.
"""

from __future__ import annotations

import argparse
import csv
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
from benchmark_fast_qubo_integrated_v1 import _exact_moves_by_case, _parse_case, _rel  # noqa: E402
from container_reshuffle.qubo.model import QuboModel  # noqa: E402
from container_reshuffle.qubo.sampler import QuboSamplerConfig, SimulatedAnnealingQuboSampler  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


def _repair_selection(linear: list[float], selected: set[int], target: int) -> set[int]:
    selected = set(selected)
    target = min(target, len(linear))
    while len(selected) > target:
        selected.remove(max(selected, key=lambda idx: (linear[idx], idx)))
    while len(selected) < target:
        missing = [idx for idx in range(len(linear)) if idx not in selected]
        if not missing:
            break
        selected.add(min(missing, key=lambda idx: (linear[idx], idx)))
    return selected


def _select_group_qubo_frontier(
    candidates: dict[bytes, tuple[int, int, bytes]],
    args: argparse.Namespace,
) -> tuple[list[bytes], set[bytes], int, int, float]:
    ranked = sorted(candidates.items(), key=lambda item: compact._candidate_rank(item, args))
    if getattr(args, "selection_policy", "enhanced") == "classic":
        return [code for code, _data in ranked[: args.beam_width]], set(), 0, 0, 0.0

    selected: dict[bytes, None] = {}
    selected_by_qubo: set[bytes] = set()

    core_target = max(0, min(args.core_width, args.beam_width))
    if args.parent_core_cap > 0:
        parent_counts: dict[bytes, int] = {}
        for code, data in ranked:
            parent_code = data[2]
            if parent_counts.get(parent_code, 0) >= args.parent_core_cap:
                continue
            selected.setdefault(code, None)
            parent_counts[parent_code] = parent_counts.get(parent_code, 0) + 1
            if len(selected) >= core_target:
                break
    for code, _data in ranked:
        selected.setdefault(code, None)
        if len(selected) >= core_target:
            break

    qcalls = qvars = 0
    qtime = 0.0
    best_n = ranked[0][1][0] if ranked else math.inf
    enabled = (
        args.use_qubo
        and len(candidates) >= args.qubo_min_candidates
        and args._qubo_calls_done < args.max_qubo_calls
        and (args.qubo_focus_n < 0 or best_n <= args.qubo_focus_n)
        and len(selected) < args.beam_width
    )
    if enabled:
        remaining_ranked = [(code, data) for code, data in ranked if code not in selected]
        groups: dict[bytes, list[tuple[bytes, tuple[int, int, bytes]]]] = {}
        for code, data in remaining_ranked:
            groups.setdefault(data[2], []).append((code, data))
        group_rows = sorted(
            groups.items(),
            key=lambda item: compact._candidate_rank(item[1][0], args),
        )[: max(1, args.qubo_group_max_vars)]
        if group_rows:
            target_groups = max(1, min(args.qubo_group_select, len(group_rows)))
            raw = []
            for _parent, rows in group_rows:
                best_code, best_data = rows[0]
                rank_key = compact._candidate_rank((best_code, best_data), args)
                raw.append(float(best_data[0] * 1000 + best_data[1]) + 1e-6 * len(rank_key[-1]))
            lo, hi = min(raw), max(raw)
            linear = [0.0 if hi == lo else (value - lo) / (hi - lo) for value in raw]

            model = QuboModel()
            vars_ = [model.add_binary_var(f"g_{idx}") for idx in range(len(group_rows))]
            for idx, var in enumerate(vars_):
                model.add_linear(var, linear[idx])
            for i in range(len(group_rows)):
                ni = group_rows[i][1][0][1][0]
                for j in range(i + 1, len(group_rows)):
                    nj = group_rows[j][1][0][1][0]
                    if ni == nj:
                        model.add_quadratic(vars_[i], vars_[j], args.qubo_group_same_n_penalty)
            model.add_exactly_k_penalty(vars_, target_groups, args.qubo_penalty_a)
            sample_set = SimulatedAnnealingQuboSampler().sample(
                model,
                QuboSamplerConfig(
                    num_reads=args.qubo_reads,
                    random_seed=args.seed + args._qubo_calls_done,
                    time_limit_sec=max(0.001, args.qubo_time_limit),
                    sa_iterations=max(1, args.qubo_iterations),
                ),
            )
            picked: set[int] = set()
            if sample_set.samples:
                picked = {idx for idx in range(len(group_rows)) if sample_set.samples[0].binary_values.get(idx, 0) == 1}
            picked = _repair_selection(linear, picked, target_groups)
            for idx in sorted(picked, key=lambda i: linear[i]):
                for code, _data in group_rows[idx][1][: max(1, args.qubo_group_take)]:
                    if code not in selected:
                        selected_by_qubo.add(code)
                    selected.setdefault(code, None)
                    if len(selected) >= args.beam_width:
                        break
                if len(selected) >= args.beam_width:
                    break
            qcalls = 1
            qvars = len(model.var_index)
            qtime = sample_set.runtime_sec
            args._qubo_calls_done += 1

    for code, _data in ranked:
        selected.setdefault(code, None)
        if len(selected) >= args.beam_width:
            break
    selected_codes = list(selected.keys())[: args.beam_width]
    selected_set = set(selected_codes)
    return selected_codes, selected_by_qubo & selected_set, qcalls, qvars, qtime


def _build_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        cases=args.cases,
        bays=args.bays,
        output=args.output,
        priority="n_then_depth",
        g_weight=1.0,
        n_weight=1.0,
        target_final_n=0,
        alpha=1.3,
        max_moves=None,
        time_limit=args.time_limit,
        beam_width=args.beam_width,
        core_width=args.core_width,
        parent_core_cap=args.parent_core_cap,
        selection_policy="enhanced",
        use_qubo=True,
        qubo_mode="group_majority",
        qubo_profile="group_majority",
        max_qubo_calls=args.max_qubo_calls,
        max_qubo_vars=args.qubo_group_max_vars,
        qubo_core_width=0,
        qubo_supplement_width=args.beam_width - args.core_width,
        qubo_min_candidates=64,
        qubo_priority_bonus=0.0001,
        qubo_diversity_weight=0.10,
        qubo_frontier_n_window=0,
        qubo_frontier_depth_window=1,
        qubo_focus_n=args.qubo_focus_n,
        qubo_cohorts_per_layer=1,
        qubo_cohort_stride=64,
        qubo_rank_weight=1.0,
        qubo_n_weight=2.0,
        qubo_depth_weight=1.0,
        qubo_parent_weight=0.15,
        qubo_penalty_a=30.0,
        qubo_time_limit=args.qubo_time_limit,
        qubo_reads=args.qubo_reads,
        qubo_iterations=args.qubo_iterations,
        qubo_group_max_vars=args.qubo_group_max_vars,
        qubo_group_select=args.qubo_group_select,
        qubo_group_take=args.qubo_group_take,
        qubo_group_same_n_penalty=args.qubo_group_same_n_penalty,
        post_polish_qubo=args.post_polish_qubo,
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
        post_polish_qubo_time_limit=args.qubo_time_limit,
        post_polish_qubo_reads=args.qubo_reads,
        post_polish_qubo_iterations=args.qubo_iterations,
        seed=args.seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["26"])
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "compact_group_qubo_majority.csv")
    parser.add_argument("--beam-width", type=int, default=1200)
    parser.add_argument("--core-width", type=int, default=600)
    parser.add_argument("--parent-core-cap", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--max-qubo-calls", type=int, default=64)
    parser.add_argument("--qubo-focus-n", type=int, default=8)
    parser.add_argument("--qubo-group-max-vars", type=int, default=96)
    parser.add_argument("--qubo-group-select", type=int, default=72)
    parser.add_argument("--qubo-group-take", type=int, default=8)
    parser.add_argument("--qubo-group-same-n-penalty", type=float, default=0.04)
    parser.add_argument("--qubo-time-limit", type=float, default=0.001)
    parser.add_argument("--qubo-reads", type=int, default=2)
    parser.add_argument("--qubo-iterations", type=int, default=20)
    parser.add_argument("--post-polish-qubo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()
    if not args.output.is_absolute():
        args.output = ROOT / args.output

    compact._select_next_frontier = _select_group_qubo_frontier
    run_args = _build_args(args)
    cases = compact._case_filter(run_args)
    rows = []
    validation = {}
    move_paths = {}
    moves_dir = args.output.with_name(args.output.stem + "_moves")
    moves_dir.mkdir(parents=True, exist_ok=True)
    exact_refs = _exact_moves_by_case()
    for idx, case in enumerate(cases, start=1):
        zone, bay = _parse_case(case)
        print(f"CASE_START {idx}/{len(cases)} {case}", flush=True)
        result = compact.run_compact_beam(case, zone, bay, run_args)
        row = compact._row_from_result(result, zone, bay, run_args)
        row.solver = "compact_group_qubo_majority"
        rows.append(row)
        validation[case] = compact._validate_result(result, zone, bay)
        if result.moves:
            move_path = moves_dir / f"{case}_moves.csv"
            compact._write_explain_csv(_instance_from_single_bay(zone, bay), result.moves, move_path)
            move_paths[case] = move_path
        print(
            f"CASE_DONE {case} final_n={row.final_n} moves={row.moves} runtime={row.runtime_sec:.3f}s "
            f"qcalls={row.qubo_calls} qvars={row.qubo_total_vars} business={validation[case][0]} goal={validation[case][1]}",
            flush=True,
        )
    compact._write_rows(rows, args.output)
    compact._write_acceptance(rows, validation, move_paths, exact_refs, args.output.with_name(args.output.stem + "_acceptance.csv"))
    compact._write_report(rows, args.output, args.output.with_name(args.output.stem + "_acceptance.csv"), args.output.with_suffix(".md"), exact_refs)
    print(f"WROTE {_rel(args.output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
