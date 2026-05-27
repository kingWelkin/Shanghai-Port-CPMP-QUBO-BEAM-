"""Tabu-search baseline for full real paired business cases."""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

from benchmark_full_real_business_algorithms import (
    ROOT,
    TrajectoryResult,
    _instance_from_zone,
    _result_from_state,
    discover_paired_cases,
)
from benchmark_full_real_strong_baseline import _parse_greedy_1p3_result

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


TABU_RULES = ("n_first", "score_first", "gain_first", "budget_first")


def _rank_state(instance: Instance, state: State) -> tuple[int, int, int, int]:
    return (
        disorder_box_count(instance, state),
        state.entropy_max,
        state.disorder_total,
        state.move_count,
    )


def _tabu_candidate_key(
    instance: Instance,
    current: State,
    nxt: State,
    config: SolverConfig,
    rule: str,
    noise: float,
) -> tuple:
    n_next = disorder_box_count(instance, nxt)
    g_next = entropy_value(instance, nxt)
    gain = disorder_box_count(instance, current) - n_next
    prefix_gain = well_prefix_gain(instance, current, nxt)
    secondary = (
        nxt.entropy_max,
        inversion_count(instance, nxt),
        bad_count(instance, nxt),
        dirty_stack_count(instance, nxt),
        -settled_box_count(instance, nxt),
        nxt.move_count,
        noise,
    )
    if rule == "score_first":
        return (score_state(instance, nxt, config.score_weights), n_next, g_next) + secondary
    if rule == "gain_first":
        return (-gain, n_next, -prefix_gain, g_next) + secondary
    if rule == "budget_first":
        return (g_next, n_next, score_state(instance, nxt, config.score_weights)) + secondary
    return (n_next, g_next, score_state(instance, nxt, config.score_weights)) + secondary


def run_tabu_search_baseline(
    case: str,
    instance: Instance,
    config: SolverConfig,
    *,
    restarts: int = 12,
    tabu_tenure: int = 9,
    candidate_pool: int = 8,
) -> TrajectoryResult:
    """Run restart-based tabu search under the business G cap."""

    start = time.perf_counter()
    initial = create_initial_state(instance)
    best = initial
    expanded = generated = visited_total = 0
    completed_restarts = 0

    for restart in range(max(1, restarts)):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        completed_restarts += 1
        rng = random.Random(config.random_seed * 1009 + 7919 * restart)
        rule = TABU_RULES[restart % len(TABU_RULES)]
        current = initial
        local_seen = {hash_state(instance, current, config.enable_canonical_hash)}
        tabu_until: dict[tuple[int, int, int], int] = {}
        stale = 0

        for iteration in range(config.max_depth):
            if time.perf_counter() - start > config.time_limit_sec:
                break
            if current.move_count >= config.max_depth:
                break
            expanded += 1
            candidates = []
            for mv in generate_legal_moves(instance, current):
                try:
                    nxt = apply_move(instance, current, mv)
                except ValueError:
                    continue
                if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                    continue
                generated += 1
                h = hash_state(instance, nxt, config.enable_canonical_hash)
                move_attr = (mv.box, mv.from_stack, mv.to_stack)
                tabu = tabu_until.get(move_attr, -1) > iteration
                aspiration = _rank_state(instance, nxt) < _rank_state(instance, best)
                if h in local_seen and not aspiration:
                    continue
                if tabu and not aspiration:
                    continue
                key = _tabu_candidate_key(instance, current, nxt, config, rule, rng.random())
                candidates.append((key, mv, nxt, h))

            if not candidates:
                if tabu_until:
                    tabu_until.clear()
                    continue
                break

            candidates.sort(key=lambda item: item[0])
            pool = candidates[: max(1, min(candidate_pool, len(candidates)))]
            # Later restarts explore more aggressively while still favoring best moves.
            explore_prob = 0.15 + 0.25 * min(1.0, restart / max(1, restarts - 1))
            if len(pool) > 1 and rng.random() < explore_prob:
                _key, mv, nxt, h = rng.choice(pool)
            else:
                _key, mv, nxt, h = pool[0]

            reverse_attr = (mv.box, mv.to_stack, mv.from_stack)
            tabu_until[reverse_attr] = iteration + tabu_tenure + rng.randint(0, 3)
            current = nxt
            local_seen.add(h)
            visited_total += 1

            if _rank_state(instance, current) < _rank_state(instance, best):
                best = current
                stale = 0
            else:
                stale += 1
            if disorder_box_count(instance, best) == 0:
                break
            if stale >= 18:
                break

    status = f"BEST_EFFORT; restarts={completed_restarts}; tenure={tabu_tenure}"
    if disorder_box_count(instance, best) == 0:
        status = f"N_ZERO; restarts={completed_restarts}; tenure={tabu_tenure}"
    return _result_from_state(
        case,
        "tabu_search_baseline",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=visited_total,
        status=status,
    )


def _write_csv(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrajectoryResult.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_comparison(tabu_rows: list[TrajectoryResult]) -> None:
    strong_path = ROOT / "docs" / "experiments" / "full_real_business_strong_baseline_comparison.csv"
    rows = _read_rows(strong_path) if strong_path.exists() else []
    rows.extend({k: str(v) for k, v in row.__dict__.items()} for row in tabu_rows)

    seen: set[tuple[str, str]] = set()
    dedup: list[dict[str, str]] = []
    for row in rows:
        key = (row["case"], row["solver"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)

    by_solver: dict[str, list[dict[str, str]]] = {}
    for row in dedup:
        by_solver.setdefault(row["solver"], []).append(row)
    summary = []
    for solver, solver_rows in by_solver.items():
        summary.append(
            {
                "solver": solver,
                "cases": len(solver_rows),
                "avg_final_n": sum(float(r["final_n"]) for r in solver_rows) / len(solver_rows),
                "avg_reduction": sum(float(r["reduction"]) for r in solver_rows) / len(solver_rows),
                "avg_moves": sum(float(r["moves"]) for r in solver_rows) / len(solver_rows),
                "avg_runtime": sum(float(r["runtime_sec"]) for r in solver_rows) / len(solver_rows),
                "ok13": sum(str(r["g_ok_13"]).lower() == "true" for r in solver_rows),
            }
        )
    summary.sort(key=lambda item: (item["avg_final_n"], item["avg_moves"]))
    solvers = [item["solver"] for item in summary]
    cases = sorted({row["case"] for row in dedup})
    lookup = {(row["case"], row["solver"]): row for row in dedup}

    csv_path = ROOT / "docs" / "experiments" / "full_real_business_tabu_baseline_comparison.csv"
    md_path = ROOT / "docs" / "experiments" / "full_real_business_tabu_baseline_comparison.md"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = list(TrajectoryResult.__annotations__.keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in dedup:
            writer.writerow({k: row.get(k, "") for k in fields})

    lines = [
        "# Full Real Business Tabu Baseline Comparison",
        "",
        "This report adds a restart-based tabu search baseline to the previous strong-baseline table. Every intermediate state is filtered by the 1.3*N0 business G cap.",
        "",
        f"CSV: `{csv_path.relative_to(ROOT)}`",
        "",
        "## Aggregate",
        "",
        "| solver | cases | avg final N | avg reduction | avg moves | avg runtime s | ok 1.3N0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['solver']} | {item['cases']} | {item['avg_final_n']:.2f} | {item['avg_reduction']:.2f} | {item['avg_moves']:.2f} | {item['avg_runtime']:.3f} | {item['ok13']} |"
        )
    lines.extend(["", "## Per Case Final N", ""])
    lines.append("| case | " + " | ".join(solvers) + " |")
    lines.append("|---|" + "|".join("---:" for _ in solvers) + "|")
    for case in cases:
        values = []
        for solver in solvers:
            row = lookup.get((case, solver))
            values.append("" if row is None else row["final_n"])
        lines.append(f"| {case} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `tabu_search_baseline` uses restart diversification, tabu reverse-move attributes, aspiration for globally improving states, and stochastic top-candidate selection.",
            "- It is a useful non-QUBO local-search baseline, but on this run it is weaker than beam/QUBO-beam and the multi-start greedy repair baseline.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def main() -> None:
    config = SolverConfig(
        max_depth=60,
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
        rows.append(run_tabu_search_baseline(case, instance, config, restarts=12, tabu_tenure=9, candidate_pool=8))
    csv_path = ROOT / "docs" / "experiments" / "full_real_business_tabu_baseline.csv"
    _write_csv(rows, csv_path)
    _write_comparison(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
