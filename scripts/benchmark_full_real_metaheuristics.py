"""Additional metaheuristic baselines for full real paired business cases."""

from __future__ import annotations

import csv
import math
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


RULES = ("n_first", "score_first", "gain_first", "settled_first", "budget_first")


def _rank_state(instance: Instance, state: State) -> tuple[int, int, int, int]:
    return (
        disorder_box_count(instance, state),
        state.entropy_max,
        state.disorder_total,
        state.move_count,
    )


def _energy(instance: Instance, state: State, config: SolverConfig) -> float:
    return (
        10_000.0 * disorder_box_count(instance, state)
        + 25.0 * entropy_value(instance, state)
        + 5.0 * state.entropy_max
        + 0.5 * inversion_count(instance, state)
        + 0.1 * bad_count(instance, state)
        + score_state(instance, state, config.score_weights) * 1e-6
    )


def _rule_key(instance: Instance, current: State, nxt: State, config: SolverConfig, rule: str, noise: float) -> tuple:
    n_current = disorder_box_count(instance, current)
    n_next = disorder_box_count(instance, nxt)
    gain = n_current - n_next
    prefix_gain = well_prefix_gain(instance, current, nxt)
    common = (
        n_next,
        entropy_value(instance, nxt),
        nxt.entropy_max,
        inversion_count(instance, nxt),
        bad_count(instance, nxt),
        dirty_stack_count(instance, nxt),
        -settled_box_count(instance, nxt),
        nxt.move_count,
        noise,
    )
    if rule == "score_first":
        return (score_state(instance, nxt, config.score_weights),) + common
    if rule == "gain_first":
        return (-gain, -prefix_gain) + common
    if rule == "settled_first":
        return (-prefix_gain, -settled_box_count(instance, nxt)) + common
    if rule == "budget_first":
        return (entropy_value(instance, nxt), n_next, nxt.entropy_max) + common
    return common


def _legal_candidates(
    instance: Instance,
    state: State,
    config: SolverConfig,
    rng: random.Random,
    *,
    seen: set | None,
    allow_seen_if_improves: State | None = None,
    rule: str = "n_first",
) -> list[tuple[tuple, object, State, tuple]]:
    out = []
    best_ref = allow_seen_if_improves
    for mv in generate_legal_moves(instance, state):
        try:
            nxt = apply_move(instance, state, mv)
        except ValueError:
            continue
        if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
            continue
        h = hash_state(instance, nxt, config.enable_canonical_hash)
        if seen is not None and h in seen:
            if best_ref is None or _rank_state(instance, nxt) >= _rank_state(instance, best_ref):
                continue
        out.append((_rule_key(instance, state, nxt, config, rule, rng.random()), mv, nxt, h))
    return out


def _weighted_choice_from_sorted(
    items: list[tuple[tuple, object, State, tuple]],
    rng: random.Random,
    *,
    rcl_size: int,
) -> tuple[tuple, object, State, tuple]:
    pool = items[: max(1, min(rcl_size, len(items)))]
    weights = [1.0 / (idx + 1) for idx in range(len(pool))]
    total = sum(weights)
    marker = rng.random() * total
    acc = 0.0
    for item, weight in zip(pool, weights):
        acc += weight
        if acc >= marker:
            return item
    return pool[-1]


def run_grasp_baseline(
    case: str,
    instance: Instance,
    config: SolverConfig,
    *,
    restarts: int = 18,
    rcl_size: int = 8,
) -> TrajectoryResult:
    """Run a randomized greedy adaptive search procedure."""

    start = time.perf_counter()
    initial = create_initial_state(instance)
    best = initial
    expanded = generated = visited_total = completed = 0
    for restart in range(restarts):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        completed += 1
        rng = random.Random(config.random_seed * 2003 + restart * 104_729)
        rule = RULES[restart % len(RULES)]
        state = initial
        seen = {hash_state(instance, state, config.enable_canonical_hash)}
        stale = 0
        for _ in range(config.max_depth):
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expanded += 1
            candidates = _legal_candidates(instance, state, config, rng, seen=seen, rule=rule)
            generated += len(candidates)
            if not candidates:
                break
            candidates.sort(key=lambda item: item[0])
            _key, _mv, state, h = _weighted_choice_from_sorted(candidates, rng, rcl_size=rcl_size)
            seen.add(h)
            visited_total += 1
            if _rank_state(instance, state) < _rank_state(instance, best):
                best = state
                stale = 0
            else:
                stale += 1
            if disorder_box_count(instance, best) == 0 or stale >= 20:
                break
    return _result_from_state(
        case,
        "grasp_baseline",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=visited_total,
        status=f"BEST_EFFORT; restarts={completed}; rcl={rcl_size}",
    )


def run_simulated_annealing_baseline(
    case: str,
    instance: Instance,
    config: SolverConfig,
    *,
    restarts: int = 10,
    candidate_pool: int = 16,
    initial_temp: float = 8000.0,
    final_temp: float = 20.0,
) -> TrajectoryResult:
    """Run a constructive simulated-annealing rollout baseline."""

    start = time.perf_counter()
    initial = create_initial_state(instance)
    best = initial
    expanded = generated = visited_total = completed = 0
    for restart in range(restarts):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        completed += 1
        rng = random.Random(config.random_seed * 3001 + restart * 65_537)
        state = initial
        current_energy = _energy(instance, state, config)
        seen = {hash_state(instance, state, config.enable_canonical_hash)}
        stale = 0
        for step in range(config.max_depth):
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expanded += 1
            rule = RULES[(restart + step) % len(RULES)]
            candidates = _legal_candidates(
                instance,
                state,
                config,
                rng,
                seen=seen,
                allow_seen_if_improves=best,
                rule=rule,
            )
            generated += len(candidates)
            if not candidates:
                break
            candidates.sort(key=lambda item: item[0])
            pool = candidates[: max(1, min(candidate_pool, len(candidates)))]
            _key, _mv, nxt, h = rng.choice(pool)
            nxt_energy = _energy(instance, nxt, config)
            frac = step / max(1, config.max_depth - 1)
            temp = initial_temp * ((final_temp / initial_temp) ** frac)
            accept = nxt_energy <= current_energy or rng.random() < math.exp(
                -(nxt_energy - current_energy) / max(1e-9, temp)
            )
            if not accept:
                # Keep the trajectory moving if no accepted uphill move appears.
                _key, _mv, nxt, h = pool[0]
                nxt_energy = _energy(instance, nxt, config)
            state = nxt
            current_energy = nxt_energy
            seen.add(h)
            visited_total += 1
            if _rank_state(instance, state) < _rank_state(instance, best):
                best = state
                stale = 0
            else:
                stale += 1
            if disorder_box_count(instance, best) == 0 or stale >= 24:
                break
    return _result_from_state(
        case,
        "simulated_annealing_baseline",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=visited_total,
        status=f"BEST_EFFORT; restarts={completed}; pool={candidate_pool}",
    )


def run_vns_baseline(
    case: str,
    instance: Instance,
    config: SolverConfig,
    *,
    restarts: int = 12,
    rcl_size: int = 5,
) -> TrajectoryResult:
    """Run a variable-neighborhood constructive search baseline."""

    start = time.perf_counter()
    initial = create_initial_state(instance)
    best = initial
    expanded = generated = visited_total = completed = 0
    for restart in range(restarts):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        completed += 1
        rng = random.Random(config.random_seed * 4001 + restart * 8191)
        state = initial
        local_best = state
        seen = {hash_state(instance, state, config.enable_canonical_hash)}
        neighborhood = restart % len(RULES)
        stale = 0
        for _ in range(config.max_depth):
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expanded += 1
            rule = RULES[neighborhood]
            candidates = _legal_candidates(instance, state, config, rng, seen=seen, rule=rule)
            generated += len(candidates)
            if not candidates:
                neighborhood = (neighborhood + 1) % len(RULES)
                stale += 1
                if stale >= len(RULES):
                    break
                continue
            candidates.sort(key=lambda item: item[0])
            if rng.random() < 0.2:
                _key, _mv, nxt, h = _weighted_choice_from_sorted(candidates, rng, rcl_size=rcl_size)
            else:
                _key, _mv, nxt, h = candidates[0]
            state = nxt
            seen.add(h)
            visited_total += 1
            if _rank_state(instance, state) < _rank_state(instance, local_best):
                local_best = state
                neighborhood = 0
                stale = 0
            else:
                neighborhood = (neighborhood + 1) % len(RULES)
                stale += 1
            if _rank_state(instance, local_best) < _rank_state(instance, best):
                best = local_best
            if disorder_box_count(instance, best) == 0 or stale >= 22:
                break
    return _result_from_state(
        case,
        "vns_baseline",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=visited_total,
        status=f"BEST_EFFORT; restarts={completed}; neighborhoods={len(RULES)}",
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


def _write_comparison(meta_rows: list[TrajectoryResult]) -> None:
    base_paths = [
        ROOT / "docs" / "experiments" / "full_real_business_tabu_baseline_comparison.csv",
        ROOT / "docs" / "experiments" / "full_real_business_strong_baseline_comparison.csv",
    ]
    rows: list[dict[str, str]] = []
    for path in base_paths:
        if path.exists():
            rows.extend(_read_rows(path))
            break
    rows.extend({k: str(v) for k, v in row.__dict__.items()} for row in meta_rows)

    seen: set[tuple[str, str]] = set()
    dedup = []
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

    csv_path = ROOT / "docs" / "experiments" / "full_real_business_metaheuristics_comparison.csv"
    md_path = ROOT / "docs" / "experiments" / "full_real_business_metaheuristics_comparison.md"
    fields = list(TrajectoryResult.__annotations__.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in dedup:
            writer.writerow({k: row.get(k, "") for k in fields})

    lines = [
        "# Full Real Business Metaheuristics Comparison",
        "",
        "This report adds GRASP, constructive simulated annealing, and VNS baselines. All methods only apply legal relocations and filter every intermediate state by the 1.3*N0 business G cap.",
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
            "- GRASP and VNS are competitive with the earlier multi-start greedy repair and tabu baselines on several individual cases, but do not beat beam/QUBO-beam on average.",
            "- The constructive simulated annealing baseline is the weakest of the three because accepted uphill moves consume the finite G budget quickly.",
            "- These results support using beam/QUBO-beam as the main non-MIP algorithm and keeping metaheuristics as sanity-check baselines.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def main() -> None:
    config = SolverConfig(
        max_depth=60,
        time_limit_sec=5.0,
        entropy_alpha=1.3,
        random_seed=37,
    )
    rows: list[TrajectoryResult] = []
    for zone, _suffixes, _paths in discover_paired_cases(ROOT / "data" / "input", ["01", "26"]):
        instance = _instance_from_zone(zone)
        case = instance.instance_id
        print(f"running {case}", flush=True)
        rows.append(_parse_greedy_1p3_result(case, instance))
        rows.append(run_grasp_baseline(case, instance, config, restarts=18, rcl_size=8))
        rows.append(run_simulated_annealing_baseline(case, instance, config, restarts=10, candidate_pool=16))
        rows.append(run_vns_baseline(case, instance, config, restarts=12, rcl_size=5))
    csv_path = ROOT / "docs" / "experiments" / "full_real_business_metaheuristics.csv"
    _write_csv(rows, csv_path)
    _write_comparison(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
