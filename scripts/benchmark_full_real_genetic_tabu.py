"""Hybrid genetic + tabu baseline for full real paired business cases."""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
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
from container_reshuffle.scoring import cpmp_business_feasible
from container_reshuffle.solvers.base import SolverConfig


@dataclass
class Chromosome:
    """Policy weights used to rank legal moves in a constructive rollout."""

    weights: list[float]


@dataclass
class Evaluation:
    """One chromosome evaluation result."""

    chromosome: Chromosome
    state: State
    fitness: float
    expanded: int
    generated: int
    visited: int


def _rank_state(instance: Instance, state: State) -> tuple[int, int, int, int]:
    return (
        disorder_box_count(instance, state),
        state.entropy_max,
        state.disorder_total,
        state.move_count,
    )


def _fitness(instance: Instance, state: State) -> float:
    return (
        1_000_000.0 * disorder_box_count(instance, state)
        + 1_000.0 * state.entropy_max
        + 0.2 * state.disorder_total
        + float(state.move_count)
    )


def _feature_vector(instance: Instance, current: State, nxt: State) -> list[float]:
    n_current = disorder_box_count(instance, current)
    n_next = disorder_box_count(instance, nxt)
    gain = n_current - n_next
    prefix_gain = well_prefix_gain(instance, current, nxt)
    return [
        float(n_next),
        float(entropy_value(instance, nxt)),
        float(nxt.entropy_max),
        float(inversion_count(instance, nxt)),
        float(bad_count(instance, nxt)),
        float(dirty_stack_count(instance, nxt)),
        -float(settled_box_count(instance, nxt)),
        float(nxt.move_count),
        -float(gain),
        -float(prefix_gain),
    ]


def _move_score(instance: Instance, current: State, nxt: State, chromosome: Chromosome) -> float:
    features = _feature_vector(instance, current, nxt)
    return sum(w * x for w, x in zip(chromosome.weights, features))


def _initial_population(rng: random.Random, size: int) -> list[Chromosome]:
    seeds = [
        [1000, 10, 5, 0.5, 0.5, 0.2, 0.2, 1, 120, 30],
        [800, 40, 20, 0.4, 0.4, 0.2, 0.1, 1, 80, 20],
        [500, 20, 10, 1.5, 1.0, 0.5, 0.4, 1, 250, 50],
        [300, 80, 30, 0.5, 0.5, 0.2, 0.1, 1, 60, 15],
        [1200, 5, 5, 2.0, 1.5, 0.8, 0.5, 1, 100, 100],
    ]
    population = [Chromosome(list(weights)) for weights in seeds[:size]]
    while len(population) < size:
        base = rng.choice(seeds)
        population.append(Chromosome([max(0.001, w * rng.uniform(0.35, 2.4)) for w in base]))
    return population


def _decode_policy(
    instance: Instance,
    config: SolverConfig,
    chromosome: Chromosome,
    rng: random.Random,
    *,
    rcl_size: int,
) -> tuple[State, int, int, int]:
    state = create_initial_state(instance)
    best = state
    seen = {hash_state(instance, state, config.enable_canonical_hash)}
    expanded = generated = visited = stale = 0

    for _ in range(config.max_depth):
        expanded += 1
        candidates = []
        for mv in generate_legal_moves(instance, state):
            try:
                nxt = apply_move(instance, state, mv)
            except ValueError:
                continue
            if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                continue
            h = hash_state(instance, nxt, config.enable_canonical_hash)
            if h in seen and _rank_state(instance, nxt) >= _rank_state(instance, best):
                continue
            score = _move_score(instance, state, nxt, chromosome)
            candidates.append((score, rng.random(), mv, nxt, h))
        generated += len(candidates)
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1]))
        pool = candidates[: max(1, min(rcl_size, len(candidates)))]
        if len(pool) > 1 and rng.random() < 0.25:
            _score, _noise, _mv, state, h = rng.choice(pool)
        else:
            _score, _noise, _mv, state, h = pool[0]
        seen.add(h)
        visited += 1
        if _rank_state(instance, state) < _rank_state(instance, best):
            best = state
            stale = 0
        else:
            stale += 1
        if disorder_box_count(instance, best) == 0 or stale >= 18:
            break
    return best, expanded, generated, visited


def _tabu_intensify(
    instance: Instance,
    config: SolverConfig,
    chromosome: Chromosome,
    start_state: State,
    incumbent: State,
    rng: random.Random,
    *,
    max_steps: int = 16,
    tenure: int = 7,
) -> tuple[State, int, int, int]:
    current = start_state
    best = incumbent
    local_seen = {hash_state(instance, current, config.enable_canonical_hash)}
    tabu_until: dict[tuple[int, int, int], int] = {}
    expanded = generated = visited = stale = 0

    for iteration in range(max_steps):
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
            attr = (mv.box, mv.from_stack, mv.to_stack)
            tabu = tabu_until.get(attr, -1) > iteration
            aspiration = _rank_state(instance, nxt) < _rank_state(instance, best)
            if h in local_seen and not aspiration:
                continue
            if tabu and not aspiration:
                continue
            candidates.append((_move_score(instance, current, nxt, chromosome), rng.random(), mv, nxt, h))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1]))
        _score, _noise, mv, current, h = candidates[0]
        tabu_until[(mv.box, mv.to_stack, mv.from_stack)] = iteration + tenure + rng.randint(0, 2)
        local_seen.add(h)
        visited += 1
        if _rank_state(instance, current) < _rank_state(instance, best):
            best = current
            stale = 0
        else:
            stale += 1
        if disorder_box_count(instance, best) == 0 or stale >= 8:
            break
    return best, expanded, generated, visited


def _evaluate(
    instance: Instance,
    config: SolverConfig,
    chromosome: Chromosome,
    rng: random.Random,
    *,
    rcl_size: int,
    tabu: bool,
) -> Evaluation:
    state, expanded, generated, visited = _decode_policy(instance, config, chromosome, rng, rcl_size=rcl_size)
    if tabu:
        state, e, g, v = _tabu_intensify(instance, config, chromosome, state, state, rng)
        expanded += e
        generated += g
        visited += v
    return Evaluation(chromosome, state, _fitness(instance, state), expanded, generated, visited)


def _tournament(evaluations: list[Evaluation], rng: random.Random, size: int = 3) -> Chromosome:
    sample = rng.sample(evaluations, k=min(size, len(evaluations)))
    return min(sample, key=lambda ev: ev.fitness).chromosome


def _crossover_mutate(parent_a: Chromosome, parent_b: Chromosome, rng: random.Random) -> Chromosome:
    child = []
    for wa, wb in zip(parent_a.weights, parent_b.weights):
        alpha = rng.random()
        value = alpha * wa + (1.0 - alpha) * wb
        if rng.random() < 0.35:
            value *= rng.uniform(0.55, 1.75)
        child.append(max(0.001, value))
    return Chromosome(child)


def run_genetic_tabu_baseline(
    case: str,
    instance: Instance,
    config: SolverConfig,
    *,
    population_size: int = 18,
    generations: int = 6,
    elite_count: int = 4,
    rcl_size: int = 5,
) -> TrajectoryResult:
    """Run a policy-genetic algorithm with tabu intensification."""

    start = time.perf_counter()
    rng = random.Random(config.random_seed * 5003 + hash(case) % 100_000)
    population = _initial_population(rng, population_size)
    best = create_initial_state(instance)
    expanded = generated = visited = evaluated = completed_generations = 0

    for generation in range(generations):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        completed_generations += 1
        evaluations: list[Evaluation] = []
        for idx, chromosome in enumerate(population):
            if time.perf_counter() - start > config.time_limit_sec:
                break
            ev = _evaluate(
                instance,
                config,
                chromosome,
                rng,
                rcl_size=rcl_size,
                tabu=idx < max(1, population_size // 3) or generation == generations - 1,
            )
            evaluations.append(ev)
            evaluated += 1
            expanded += ev.expanded
            generated += ev.generated
            visited += ev.visited
            if _rank_state(instance, ev.state) < _rank_state(instance, best):
                best = ev.state
        if not evaluations or disorder_box_count(instance, best) == 0:
            break
        evaluations.sort(key=lambda ev: ev.fitness)
        next_population = [Chromosome(list(ev.chromosome.weights)) for ev in evaluations[:elite_count]]
        while len(next_population) < population_size:
            pa = _tournament(evaluations, rng)
            pb = _tournament(evaluations, rng)
            next_population.append(_crossover_mutate(pa, pb, rng))
        population = next_population

    status = f"BEST_EFFORT; pop={population_size}; generations={completed_generations}; evals={evaluated}"
    if disorder_box_count(instance, best) == 0:
        status = f"N_ZERO; pop={population_size}; generations={completed_generations}; evals={evaluated}"
    return _result_from_state(
        case,
        "genetic_tabu_baseline",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=visited,
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


def _write_comparison(genetic_rows: list[TrajectoryResult]) -> None:
    base_path = ROOT / "docs" / "experiments" / "full_real_business_metaheuristics_comparison.csv"
    rows = _read_rows(base_path) if base_path.exists() else []
    rows.extend({k: str(v) for k, v in row.__dict__.items()} for row in genetic_rows)

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

    csv_path = ROOT / "docs" / "experiments" / "full_real_business_genetic_tabu_comparison.csv"
    md_path = ROOT / "docs" / "experiments" / "full_real_business_genetic_tabu_comparison.md"
    fields = list(TrajectoryResult.__annotations__.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in dedup:
            writer.writerow({k: row.get(k, "") for k in fields})

    lines = [
        "# Full Real Business Genetic + Tabu Comparison",
        "",
        "This report adds a memetic genetic + tabu baseline. The genetic layer evolves move-ranking policies; each policy is decoded through legal constructive rollouts and selected individuals receive tabu intensification. Every intermediate state is filtered by the 1.3*N0 business G cap.",
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
            "- The hybrid improves over plain tabu and the original greedy baseline, but it does not beat beam/QUBO-beam on this run.",
            "- Policy-level genetic search is safer than move-sequence crossover for this problem, because every decoded trajectory remains legal by construction.",
            "- The result reinforces that retaining multiple live states, as beam does, is more valuable here than optimizing a single constructive policy.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def main() -> None:
    config = SolverConfig(
        max_depth=60,
        time_limit_sec=6.0,
        entropy_alpha=1.3,
        random_seed=37,
    )
    rows: list[TrajectoryResult] = []
    for zone, _suffixes, _paths in discover_paired_cases(ROOT / "data" / "input", ["01", "26"]):
        instance = _instance_from_zone(zone)
        case = instance.instance_id
        print(f"running {case}", flush=True)
        rows.append(_parse_greedy_1p3_result(case, instance))
        rows.append(run_genetic_tabu_baseline(case, instance, config))
    csv_path = ROOT / "docs" / "experiments" / "full_real_business_genetic_tabu.csv"
    _write_csv(rows, csv_path)
    _write_comparison(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
