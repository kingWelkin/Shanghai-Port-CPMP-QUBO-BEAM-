"""Test QUBO selection on top of the strongest enhanced beam candidate set."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import replace
from pathlib import Path

from benchmark_full_real_business_algorithms import (
    ROOT,
    TrajectoryResult,
    _instance_from_zone,
    _result_from_state,
    discover_paired_cases,
)
from benchmark_full_real_enhanced_beam import (
    _beam_primary_key,
    _select_enhanced_beam,
    _select_state_expansions,
    _state_rank,
)

from container_reshuffle.core import Instance, State, create_initial_state, hash_state
from container_reshuffle.features import disorder_box_count
from container_reshuffle.qubo.beam_selection import QuboBeamSelectionBuilder
from container_reshuffle.qubo.sampler import QuboSamplerConfig, SimulatedAnnealingQuboSampler
from container_reshuffle.scoring import dominance_value, score_state
from container_reshuffle.solvers.base import SolverConfig
from container_reshuffle.solvers.beam import Candidate


def _ranked_for_qubo(candidates: list[Candidate], instance: Instance, config: SolverConfig) -> list[Candidate]:
    """Replace raw score with business-rank score before QUBO construction."""

    ordered = sorted(candidates, key=lambda c: _beam_primary_key(instance, c.state, config))
    ranked: list[Candidate] = []
    for rank, cand in enumerate(ordered):
        ranked.append(replace(cand, score=float(rank)))
    return ranked


def _run_enhanced_qubo_state(
    instance: Instance,
    config: SolverConfig,
    *,
    per_rule_limit: int,
    qbo_interval: int,
    qbo_elite_ratio: float,
    diversity_weight: float,
    fallback_diversity_weight: float,
    fallback_elite_ratio: float,
) -> tuple[State, int, int, int, int, int, float, int]:
    """Run enhanced beam where QUBO selects beam layers from business-ranked candidates."""

    start = time.perf_counter()
    initial = create_initial_state(instance)
    best = initial
    h0 = hash_state(instance, initial, config.enable_canonical_hash)
    beam = [Candidate(initial, score_state(instance, initial, config.score_weights), None, h0)]
    visited = {h0: dominance_value(instance, initial)}
    expanded = generated = evaluated_moves = 0
    qubo_calls = qubo_vars = 0
    qubo_time = 0.0
    builder = QuboBeamSelectionBuilder()
    sampler = SimulatedAnnealingQuboSampler()
    qbo_interval = max(1, qbo_interval)

    for depth in range(config.max_depth):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        candidates: list[Candidate] = []
        for cand in beam:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expanded += 1
            expansions, eval_count = _select_state_expansions(
                instance,
                cand.state,
                config,
                per_rule_limit=per_rule_limit,
            )
            evaluated_moves += eval_count
            for score, mv, nxt in expansions:
                h = hash_state(instance, nxt, config.enable_canonical_hash)
                dom = dominance_value(instance, nxt)
                if config.enable_visited_pruning and visited.get(h, float("inf")) <= dom:
                    continue
                visited[h] = dom
                if _state_rank(instance, nxt) < _state_rank(instance, best):
                    best = nxt
                candidates.append(Candidate(nxt, score, mv, h))
        generated += len(candidates)
        if not candidates or disorder_box_count(instance, best) == 0:
            break

        candidates = _ranked_for_qubo(candidates, instance, config)
        candidates = candidates[: config.max_candidates_per_layer]
        use_qubo = depth % qbo_interval == 0 and len(candidates) >= max(2, config.beam_width)
        if not use_qubo:
            beam = _select_enhanced_beam(
                candidates,
                instance,
                config,
                diversity_weight=fallback_diversity_weight,
                elite_ratio=fallback_elite_ratio,
            )
            continue

        model, usable = builder.build(
            candidates,
            config.beam_width,
            diversity_weight,
            config.qubo_penalty_a,
            config.max_qubo_vars,
        )
        qubo_calls += 1
        qubo_vars += len(model.var_index)
        sample_set = sampler.sample(
            model,
            QuboSamplerConfig(
                num_reads=config.qubo_num_reads,
                random_seed=config.random_seed + 1009 * qubo_calls + depth,
                time_limit_sec=config.qubo_time_limit_sec or 0.03,
                sa_iterations=config.qubo_sa_iterations,
            ),
        )
        qubo_time += sample_set.runtime_sec
        elite_keep_count = max(1, min(config.beam_width, int(math.ceil(config.beam_width * qbo_elite_ratio))))
        beam, _repair = builder.decode_samples_and_repair(
            sample_set.samples,
            usable,
            config.beam_width,
            elite_keep_count=elite_keep_count,
            diversity_weight=diversity_weight,
            repair_hamming_slack=config.qubo_repair_hamming_slack,
        )

    return best, expanded, generated, len(visited), qubo_calls, qubo_vars, qubo_time, evaluated_moves


def run_enhanced_qubo_beam(
    case: str,
    instance: Instance,
    config: SolverConfig,
    *,
    solver_name: str,
    per_rule_limit: int,
    qbo_interval: int,
    qbo_elite_ratio: float,
    diversity_weight: float,
    fallback_diversity_weight: float,
    fallback_elite_ratio: float,
) -> TrajectoryResult:
    """Run one enhanced+QUBO beam variant and convert it to a benchmark row."""

    start = time.perf_counter()
    state, expanded, generated, visited, qcalls, qvars, qtime, evaluated = _run_enhanced_qubo_state(
        instance,
        config,
        per_rule_limit=per_rule_limit,
        qbo_interval=qbo_interval,
        qbo_elite_ratio=qbo_elite_ratio,
        diversity_weight=diversity_weight,
        fallback_diversity_weight=fallback_diversity_weight,
        fallback_elite_ratio=fallback_elite_ratio,
    )
    status = "N_ZERO" if disorder_box_count(instance, state) == 0 else "BEST_EFFORT"
    status += (
        f"; enhanced_candidates=True; qbo_selection=True; per_rule={per_rule_limit}; "
        f"qbo_interval={qbo_interval}; qbo_elite={qbo_elite_ratio:.2f}; evaluated_moves={evaluated}"
    )
    return _result_from_state(
        case,
        solver_name,
        instance,
        state,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=visited,
        qubo_calls=qcalls,
        qubo_total_vars=qvars,
        qubo_time_sec=qtime,
        status=status,
    )


def _config(time_limit: float) -> SolverConfig:
    return SolverConfig(
        max_depth=60,
        beam_width=14,
        max_candidates_per_layer=180,
        time_limit_sec=time_limit,
        max_qubo_vars=120,
        qubo_penalty_a=50.0,
        qubo_num_reads=20,
        qubo_time_limit_sec=0.03,
        qubo_sa_iterations=160,
        qubo_repair_hamming_slack=2,
        entropy_alpha=1.3,
        random_seed=37,
    )


def _write_csv(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrajectoryResult.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_comparison(new_rows: list[TrajectoryResult]) -> None:
    base_path = ROOT / "docs" / "experiments" / "full_real_business_focused_beam_comparison.csv"
    rows = _read_rows(base_path)
    rows.extend({key: str(value) for key, value in row.__dict__.items()} for row in new_rows)
    seen: set[tuple[str, str]] = set()
    dedup: list[dict[str, str]] = []
    for row in rows:
        key = (row["case"], row["solver"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)

    cases = sorted({row["case"] for row in dedup})
    by_solver: dict[str, list[dict[str, str]]] = {}
    for row in dedup:
        by_solver.setdefault(row["solver"], []).append(row)
    summary = []
    for solver, solver_rows in by_solver.items():
        if len(solver_rows) != len(cases):
            continue
        summary.append(
            {
                "solver": solver,
                "cases": len(solver_rows),
                "avg_final_n": sum(float(r["final_n"]) for r in solver_rows) / len(solver_rows),
                "avg_reduction": sum(float(r["reduction"]) for r in solver_rows) / len(solver_rows),
                "avg_moves": sum(float(r["moves"]) for r in solver_rows) / len(solver_rows),
                "avg_runtime": sum(float(r["runtime_sec"]) for r in solver_rows) / len(solver_rows),
                "avg_qubo_calls": sum(float(r["qubo_calls"]) for r in solver_rows) / len(solver_rows),
                "ok13": sum(str(r["g_ok_13"]).lower() == "true" for r in solver_rows),
            }
        )
    summary.sort(key=lambda item: (item["avg_final_n"], item["avg_moves"], item["avg_runtime"]))
    solvers = [item["solver"] for item in summary]
    lookup = {(row["case"], row["solver"]): row for row in dedup}

    csv_path = ROOT / "docs" / "experiments" / "full_real_business_enhanced_qubo_beam_comparison.csv"
    md_path = ROOT / "docs" / "experiments" / "full_real_business_enhanced_qubo_beam_comparison.md"
    fields = list(TrajectoryResult.__annotations__.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in dedup:
            writer.writerow({field: row.get(field, "") for field in fields})

    lines = [
        "# Full Real Business Enhanced QUBO Beam Comparison",
        "",
        "This report tests QUBO selection after the improved enhanced-beam candidate generation and business ranking.",
        "The QUBO samples are only layer-selection hints; selections are repaired to exact beam width and all moves remain classical legal moves.",
        "",
        f"CSV: `{csv_path.relative_to(ROOT)}`",
        "",
        "## Aggregate",
        "",
        "| solver | cases | avg final N | avg reduction | avg moves | avg runtime s | avg QUBO calls | ok 1.3N0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['solver']} | {item['cases']} | {item['avg_final_n']:.2f} | {item['avg_reduction']:.2f} | "
            f"{item['avg_moves']:.2f} | {item['avg_runtime']:.3f} | {item['avg_qubo_calls']:.2f} | {item['ok13']} |"
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
            "- `enhanced_qubo_beam_ranked` calls QUBO more often and uses a moderate elite lock.",
            "- `enhanced_qubo_beam_elite` is conservative: most elite beam states are fixed before QUBO repair.",
            "- Compare these rows against `beam_focused_plain` and `beam_focused_with_qubo` to decide whether QUBO should be integrated into the best beam itself or kept as a separate portfolio component.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def main() -> None:
    rows: list[TrajectoryResult] = []
    for zone, _suffixes, _paths in discover_paired_cases(ROOT / "data" / "input", ["01", "26"]):
        instance = _instance_from_zone(zone)
        case = instance.instance_id
        print(f"running {case}", flush=True)
        rows.append(
            run_enhanced_qubo_beam(
                case,
                instance,
                _config(10.0),
                solver_name="enhanced_qubo_beam_ranked",
                per_rule_limit=2,
                qbo_interval=3,
                qbo_elite_ratio=0.65,
                diversity_weight=0.04,
                fallback_diversity_weight=0.06,
                fallback_elite_ratio=0.60,
            )
        )
        rows.append(
            run_enhanced_qubo_beam(
                case,
                instance,
                _config(10.0),
                solver_name="enhanced_qubo_beam_elite",
                per_rule_limit=2,
                qbo_interval=5,
                qbo_elite_ratio=0.80,
                diversity_weight=0.02,
                fallback_diversity_weight=0.06,
                fallback_elite_ratio=0.60,
            )
        )
    csv_path = ROOT / "docs" / "experiments" / "full_real_business_enhanced_qubo_beam.csv"
    _write_csv(rows, csv_path)
    _write_comparison(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
