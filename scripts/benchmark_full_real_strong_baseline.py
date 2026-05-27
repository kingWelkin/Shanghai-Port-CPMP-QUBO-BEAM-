"""Run stronger multi-start greedy baselines on full real paired cases."""

from __future__ import annotations

import csv
import re
import random
import time
from pathlib import Path

from benchmark_full_real_business_algorithms import (
    ROOT,
    TrajectoryResult,
    _instance_from_zone,
    _result_from_state,
    _run_business_beam,
    _run_business_qubo_beam,
    discover_paired_cases,
)

from container_reshuffle.core import Instance, State, apply_move, create_initial_state, generate_legal_moves, hash_state
from container_reshuffle.features import (
    bad_count,
    dirty_stack_count,
    disorder_box_count,
    entropy_value,
    inversion_count,
    settled_box_count,
    well_prefix_gain,
)
from container_reshuffle.scoring import cpmp_business_feasible, score_state
from container_reshuffle.solvers.base import SolverConfig


RULES = (
    "n_first",
    "score_first",
    "gain_first",
    "settle_first",
    "budget_first",
    "lookahead_n",
)


def _rank_state(instance: Instance, state: State) -> tuple[int, int, int, int]:
    return (
        disorder_box_count(instance, state),
        state.entropy_max,
        state.disorder_total,
        state.move_count,
    )


def _top_is_clean_destination(stack: list[int], box: int) -> int:
    if not stack:
        return 1
    return int(stack[-1] > box)


def _candidate_key(instance: Instance, prev: State, nxt: State, rule: str, config: SolverConfig) -> tuple:
    n_prev = disorder_box_count(instance, prev)
    n_next = disorder_box_count(instance, nxt)
    g_next = entropy_value(instance, nxt)
    gain = n_prev - n_next
    prefix_gain = well_prefix_gain(instance, prev, nxt)
    common = (
        n_next,
        g_next,
        nxt.entropy_max,
        nxt.disorder_total,
        nxt.move_count,
        inversion_count(instance, nxt),
        bad_count(instance, nxt),
        dirty_stack_count(instance, nxt),
        -settled_box_count(instance, nxt),
    )
    if rule == "score_first":
        return (score_state(instance, nxt, config.score_weights),) + common
    if rule == "gain_first":
        return (-gain, n_next, g_next, inversion_count(instance, nxt), nxt.move_count)
    if rule == "settle_first":
        return (-prefix_gain, n_next, inversion_count(instance, nxt), g_next, nxt.move_count)
    if rule == "budget_first":
        return (g_next, n_next, nxt.entropy_max, nxt.disorder_total, inversion_count(instance, nxt))
    return common


def _lookahead_n(instance: Instance, state: State, config: SolverConfig) -> int:
    best = disorder_box_count(instance, state)
    for mv in generate_legal_moves(instance, state):
        try:
            nxt = apply_move(instance, state, mv)
        except ValueError:
            continue
        if cpmp_business_feasible(instance, nxt, config.entropy_alpha):
            best = min(best, disorder_box_count(instance, nxt))
    return best


def _run_one_rollout(
    instance: Instance,
    config: SolverConfig,
    *,
    rule: str,
    seed: int,
    top_k: int,
) -> tuple[State, int, int]:
    rng = random.Random(seed)
    state = create_initial_state(instance)
    best = state
    seen = {hash_state(instance, state, config.enable_canonical_hash)}
    expanded = generated = 0

    for _ in range(config.max_depth):
        expanded += 1
        scored = []
        for mv in generate_legal_moves(instance, state):
            try:
                nxt = apply_move(instance, state, mv)
            except ValueError:
                continue
            if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                continue
            h = hash_state(instance, nxt, config.enable_canonical_hash)
            if h in seen:
                continue
            key = _candidate_key(instance, state, nxt, rule, config)
            if rule == "lookahead_n":
                key = (_lookahead_n(instance, nxt, config),) + key
            box = mv.box
            dest = state.stacks[mv.to_stack]
            move_bias = (
                -_top_is_clean_destination(dest, box),
                len(dest),
                rng.random(),
            )
            scored.append((key + move_bias, mv, nxt, h))
        generated += len(scored)
        if not scored:
            break
        scored.sort(key=lambda item: item[0])
        pool = scored[: max(1, min(top_k, len(scored)))]
        # Mostly exploit, but keep enough randomized tie-breaking to diversify starts.
        if len(pool) > 1 and rng.random() < 0.35:
            _, _mv, state, h = rng.choice(pool)
        else:
            _, _mv, state, h = pool[0]
        seen.add(h)
        if _rank_state(instance, state) < _rank_state(instance, best):
            best = state
        if disorder_box_count(instance, best) == 0:
            break
    return best, expanded, generated


def run_multistart_greedy_repair(
    case: str,
    instance: Instance,
    config: SolverConfig,
    *,
    starts_per_rule: int = 10,
    top_k: int = 5,
) -> TrajectoryResult:
    """Run a portfolio of stochastic greedy rollouts and keep the best trajectory."""

    start = time.perf_counter()
    best = create_initial_state(instance)
    expanded = generated = 0
    visited = 0
    rollout_count = 0

    for rule in RULES:
        for rep in range(starts_per_rule):
            if time.perf_counter() - start > config.time_limit_sec:
                break
            rollout_count += 1
            state, e, g = _run_one_rollout(
                instance,
                config,
                rule=rule,
                seed=config.random_seed * 10_000 + 997 * rep + 37 * RULES.index(rule),
                top_k=top_k,
            )
            expanded += e
            generated += g
            visited += state.move_count + 1
            if _rank_state(instance, state) < _rank_state(instance, best):
                best = state
        if time.perf_counter() - start > config.time_limit_sec:
            break

    status = f"BEST_EFFORT; rollouts={rollout_count}; rules={len(RULES)}"
    if disorder_box_count(instance, best) == 0:
        status = f"N_ZERO; rollouts={rollout_count}; rules={len(RULES)}"
    return _result_from_state(
        case,
        "multi_start_greedy_repair",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=visited,
        status=status,
    )


def _parse_greedy_1p3_result(case: str, instance: Instance) -> TrajectoryResult:
    path = ROOT / "data" / "results_greedy_1p3" / f"result_{case}.txt"
    text = path.read_text(encoding="utf-8")
    moves = _parse_int(text, r"greedy_moves:\s*(\d+)")
    n0 = _parse_int(text, r"N\(0\)=(\d+)") or disorder_box_count(instance, State([list(s) for s in instance.stacks]))
    g_end = _parse_int(text, r"G_end=(\d+)")
    g_cap = _parse_int(text, r"G_cap=(\d+)")
    final_n = None
    match = re.search(r"N\(Future\)\s*(\d+)\D+(\d+)", text)
    if match:
        final_n = int(match.group(2))
    if final_n is None and moves is not None and g_end is not None:
        final_n = g_end - moves
    cap13 = int(g_cap or (1.3 * n0))
    cap15 = int(1.5 * n0)
    return TrajectoryResult(
        case=case,
        solver="chen_greedy_algorithm_1p3",
        n0=n0,
        cap_13=cap13,
        cap_15=cap15,
        moves=moves,
        final_n=final_n,
        reduction=None if final_n is None else n0 - final_n,
        g_end=g_end,
        max_g=g_end,
        g_ok_13=None if g_end is None else g_end <= cap13,
        g_ok_15=None if g_end is None else g_end <= cap15,
        runtime_sec=_parse_float(text, r"greedy_sec:\s*([0-9.]+)") or 0.0,
        status="rerun; cap=1.3*N0",
    )


def _parse_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    return None if not match else int(match.group(1))


def _parse_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    return None if not match else float(match.group(1))


def _write_comparison(rows: list[TrajectoryResult], csv_path: Path, md_path: Path) -> None:
    from benchmark_full_real_business_algorithms import _write_csv

    _write_csv(rows, csv_path)
    by_solver: dict[str, list[TrajectoryResult]] = {}
    for row in rows:
        by_solver.setdefault(row.solver, []).append(row)
    summary = []
    for solver, solver_rows in by_solver.items():
        summary.append(
            (
                solver,
                len(solver_rows),
                sum(float(r.final_n or 0) for r in solver_rows) / len(solver_rows),
                sum(float(r.reduction or 0) for r in solver_rows) / len(solver_rows),
                sum(float(r.moves or 0) for r in solver_rows) / len(solver_rows),
                sum(r.runtime_sec for r in solver_rows) / len(solver_rows),
                sum(1 for r in solver_rows if r.g_ok_13),
            )
        )
    summary.sort(key=lambda item: (item[2], item[4]))
    cases = sorted({r.case for r in rows})
    solvers = [row[0] for row in summary]
    lookup = {(r.case, r.solver): r for r in rows}

    lines = [
        "# Full Real Business Strong Baseline Comparison",
        "",
        "This table compares the stronger multi-start greedy repair baseline against current beam and QUBO-beam results under the 1.3*N0 business cap.",
        "",
        f"CSV: `{csv_path.relative_to(ROOT)}`",
        "",
        "## Aggregate",
        "",
        "| solver | cases | avg final N | avg reduction | avg moves | avg runtime s | ok 1.3N0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for solver, count, avg_n, avg_red, avg_moves, avg_runtime, ok_count in summary:
        lines.append(
            f"| {solver} | {count} | {avg_n:.2f} | {avg_red:.2f} | {avg_moves:.2f} | {avg_runtime:.3f} | {ok_count} |"
        )
    lines.extend(["", "## Per Case Final N", ""])
    lines.append("| case | " + " | ".join(solvers) + " |")
    lines.append("|---|" + "|".join("---:" for _ in solvers) + "|")
    for case in cases:
        values = []
        for solver in solvers:
            row = lookup.get((case, solver))
            values.append("" if row is None or row.final_n is None else str(row.final_n))
        lines.append(f"| {case} | " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = SolverConfig(
        max_depth=60,
        beam_width=10,
        max_candidates_per_layer=100,
        candidate_limit_per_state=3,
        time_limit_sec=10.0,
        max_qubo_vars=60,
        qubo_penalty_a=50.0,
        diversity_weight=0.1,
        qubo_num_reads=16,
        qubo_time_limit_sec=0.02,
        qubo_sa_iterations=120,
        qubo_call_interval=5,
        qubo_min_candidates_for_call=1,
        qubo_repair_hamming_slack=0,
        entropy_alpha=1.3,
        random_seed=37,
    )
    wide_config = SolverConfig(
        max_depth=60,
        beam_width=30,
        max_candidates_per_layer=300,
        candidate_limit_per_state=5,
        time_limit_sec=10.0,
        entropy_alpha=1.3,
        random_seed=37,
    )
    rows: list[TrajectoryResult] = []
    for zone, _suffixes, _paths in discover_paired_cases(ROOT / "data" / "input", ["01", "26"]):
        instance = _instance_from_zone(zone)
        case = instance.instance_id
        print(f"running {case}", flush=True)
        rows.append(_parse_greedy_1p3_result(case, instance))
        rows.append(run_multistart_greedy_repair(case, instance, config, starts_per_rule=10, top_k=5))
        rows.append(_run_business_beam(case, instance, config))
        wide_row = _run_business_beam(case, instance, wide_config)
        wide_row.solver = "wide_beam_baseline"
        rows.append(wide_row)
        rows.append(_run_business_qubo_beam(case, instance, config, solver_name="qubo_beam"))
    csv_path = ROOT / "docs" / "experiments" / "full_real_business_strong_baseline_comparison.csv"
    md_path = ROOT / "docs" / "experiments" / "full_real_business_strong_baseline_comparison.md"
    _write_comparison(rows, csv_path, md_path)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
