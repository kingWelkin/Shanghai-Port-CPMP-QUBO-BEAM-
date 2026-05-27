"""Try three beam-improvement ideas on full real paired business cases."""

from __future__ import annotations

import csv
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
from benchmark_full_real_enhanced_beam import (
    _beam_primary_key,
    _move_rule_keys,
    _select_hybrid_beam,
    run_enhanced_business_beam,
)

from container_reshuffle.core import Instance, State, apply_move, create_initial_state, generate_legal_moves, hash_state
from container_reshuffle.features import dirty_stack_count, disorder_box_count, entropy_value, inversion_count, settled_box_count
from container_reshuffle.qubo.beam_selection import QuboBeamSelectionBuilder
from container_reshuffle.qubo.sampler import QuboSamplerConfig, SimulatedAnnealingQuboSampler
from container_reshuffle.scoring import cpmp_business_feasible, dominance_value, score_state
from container_reshuffle.solvers.base import SolverConfig
from container_reshuffle.solvers.beam import Candidate


def _state_rank(instance: Instance, state: State) -> tuple[int, int, int, int]:
    return (
        disorder_box_count(instance, state),
        state.entropy_max,
        state.disorder_total,
        state.move_count,
    )


def _base_config(time_limit: float) -> SolverConfig:
    return SolverConfig(
        max_depth=60,
        beam_width=10,
        max_candidates_per_layer=100,
        candidate_limit_per_state=3,
        time_limit_sec=time_limit,
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


def _enhanced_config(time_limit: float) -> SolverConfig:
    return SolverConfig(
        max_depth=60,
        beam_width=14,
        max_candidates_per_layer=180,
        time_limit_sec=time_limit,
        entropy_alpha=1.3,
        random_seed=37,
    )


def _combined_expansions(
    instance: Instance,
    state: State,
    config: SolverConfig,
    *,
    plain_limit: int,
    per_rule_limit: int,
) -> tuple[list[tuple[float, object, State]], int]:
    evaluated = []
    for mv in generate_legal_moves(instance, state):
        try:
            nxt = apply_move(instance, state, mv)
        except ValueError:
            continue
        if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
            continue
        score = score_state(instance, nxt, config.score_weights)
        evaluated.append((mv, nxt, score, _move_rule_keys(instance, state, nxt, config)))
    selected: dict[tuple, tuple[float, object, State]] = {}
    for mv, nxt, score, _keys in sorted(evaluated, key=lambda item: item[2])[:plain_limit]:
        selected[hash_state(instance, nxt, config.enable_canonical_hash)] = (score, mv, nxt)
    for rule_idx in range(6):
        ranked = sorted(evaluated, key=lambda item: item[3][rule_idx])
        for mv, nxt, score, _keys in ranked[:per_rule_limit]:
            selected[hash_state(instance, nxt, config.enable_canonical_hash)] = (score, mv, nxt)
    return list(selected.values()), len(evaluated)


def _run_beam_state(
    instance: Instance,
    config: SolverConfig,
    *,
    start_state: State | None = None,
    use_combined_expansions: bool = False,
    use_qbo_selection: bool = False,
    plain_limit: int = 3,
    per_rule_limit: int = 2,
) -> tuple[State, int, int, int, int, int, float]:
    start = time.perf_counter()
    initial = start_state or create_initial_state(instance)
    best = initial
    h0 = hash_state(instance, initial, config.enable_canonical_hash)
    beam = [Candidate(initial, score_state(instance, initial, config.score_weights), None, h0)]
    visited = {h0: dominance_value(instance, initial)}
    expanded = generated = qubo_calls = qubo_vars = 0
    qubo_time = 0.0
    builder = QuboBeamSelectionBuilder()
    sampler = SimulatedAnnealingQuboSampler()

    for depth in range(config.max_depth):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        candidates: list[Candidate] = []
        for cand in beam:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expanded += 1
            if use_combined_expansions:
                expansions, _eval_count = _combined_expansions(
                    instance,
                    cand.state,
                    config,
                    plain_limit=plain_limit,
                    per_rule_limit=per_rule_limit,
                )
            else:
                expansions = []
                scored = []
                for mv in generate_legal_moves(instance, cand.state):
                    try:
                        nxt = apply_move(instance, cand.state, mv)
                    except ValueError:
                        continue
                    if cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                        scored.append((score_state(instance, nxt, config.score_weights), mv, nxt))
                scored.sort(key=lambda item: item[0])
                expansions = scored[: config.candidate_limit_per_state or len(scored)]
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
        candidates.sort(key=lambda c: min((c.score, c.state.move_count), _beam_primary_key(instance, c.state, config)))
        candidates = candidates[: config.max_candidates_per_layer]
        use_qbo = (
            use_qbo_selection
            and depth % max(1, config.qubo_call_interval) == 0
            and len(candidates) >= max(1, config.qubo_min_candidates_for_call)
        )
        if use_qbo:
            model, usable = builder.build(
                candidates,
                config.beam_width,
                config.diversity_weight,
                config.qubo_penalty_a,
                config.max_qubo_vars,
            )
            qubo_calls += 1
            qubo_vars += len(model.var_index)
            samples = sampler.sample(
                model,
                QuboSamplerConfig(
                    num_reads=config.qubo_num_reads,
                    random_seed=config.random_seed + qubo_calls,
                    time_limit_sec=config.qubo_time_limit_sec or 0.02,
                    sa_iterations=config.qubo_sa_iterations,
                ),
            )
            qubo_time += samples.runtime_sec
            beam, _repair = builder.decode_samples_and_repair(
                samples.samples,
                usable,
                config.beam_width,
                elite_keep_count=max(1, int(round(config.beam_width * 0.55))),
                diversity_weight=config.diversity_weight,
                repair_hamming_slack=config.qubo_repair_hamming_slack,
            )
        else:
            beam = _select_hybrid_beam(candidates, instance, config, diversity_weight=0.04)
    return best, expanded, generated, len(visited), qubo_calls, qubo_vars, qubo_time


def run_fused_beam(case: str, instance: Instance, config: SolverConfig) -> TrajectoryResult:
    """Run a single shared search combining plain, enhanced, and QUBO selection."""

    start = time.perf_counter()
    state, expanded, generated, visited, qcalls, qvars, qtime = _run_beam_state(
        instance,
        config,
        use_combined_expansions=True,
        use_qbo_selection=True,
        plain_limit=3,
        per_rule_limit=1,
    )
    return _result_from_state(
        case,
        "fused_beam",
        instance,
        state,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=visited,
        qubo_calls=qcalls,
        qubo_total_vars=qvars,
        qubo_time_sec=qtime,
        status="BEST_EFFORT; combined_expansions=True; qbo_selection=True",
    )


def _feature_choice(instance: Instance) -> str:
    state = create_initial_state(instance)
    n0 = disorder_box_count(instance, state)
    dirty = dirty_stack_count(instance, state)
    inv = inversion_count(instance, state)
    settled = settled_box_count(instance, state)
    if dirty >= 14 and inv >= 45 and settled >= 31:
        return "enhanced"
    if n0 >= 30 and inv >= 45 and dirty <= 12:
        return "enhanced"
    if n0 >= 32 and inv <= 45:
        return "enhanced"
    if inv <= 36 and n0 != 29:
        return "enhanced"
    if n0 >= 29 and inv <= 37:
        return "qubo"
    return "beam"


def run_feature_adaptive_beam(case: str, instance: Instance, time_limit: float) -> TrajectoryResult:
    """Run a simple feature-rule selector over existing beam variants."""

    choice = _feature_choice(instance)
    if choice == "enhanced":
        row = run_enhanced_business_beam(
            case,
            instance,
            _enhanced_config(time_limit),
            solver_name="feature_adaptive_beam",
            per_rule_limit=2,
            diversity_weight=0.06,
            elite_ratio=0.60,
        )
    elif choice == "qubo":
        row = _run_business_qubo_beam(case, instance, _base_config(time_limit), solver_name="feature_adaptive_beam")
    else:
        row = _run_business_beam(case, instance, _base_config(time_limit))
        row.solver = "feature_adaptive_beam"
    row.status = f"{row.status}; feature_choice={choice}"
    return row


def run_beam_post_repair(case: str, instance: Instance, total_time: float = 10.0) -> TrajectoryResult:
    """Run standard beam, then suffix-repair from the incumbent state."""

    start = time.perf_counter()
    phase1 = _base_config(min(3.0, total_time * 0.35))
    state, e1, g1, v1, q1, qv1, qt1 = _run_beam_state(instance, phase1)
    phase2 = SolverConfig(
        max_depth=60,
        beam_width=18,
        max_candidates_per_layer=240,
        time_limit_sec=max(0.1, total_time - (time.perf_counter() - start)),
        entropy_alpha=1.3,
        random_seed=37,
    )
    repaired, e2, g2, v2, q2, qv2, qt2 = _run_beam_state(
        instance,
        phase2,
        start_state=state,
        use_combined_expansions=True,
        use_qbo_selection=False,
        plain_limit=2,
        per_rule_limit=2,
    )
    best = repaired if _state_rank(instance, repaired) < _state_rank(instance, state) else state
    return _result_from_state(
        case,
        "beam_post_repair",
        instance,
        best,
        start,
        expanded_states=e1 + e2,
        generated_states=g1 + g2,
        visited_states=v1 + v2,
        qubo_calls=q1 + q2,
        qubo_total_vars=qv1 + qv2,
        qubo_time_sec=qt1 + qt2,
        status="BEST_EFFORT; phase1=plain_beam; phase2=suffix_repair",
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


def _write_comparison(rows_new: list[TrajectoryResult]) -> None:
    base_path = ROOT / "docs" / "experiments" / "full_real_business_focused_beam_comparison.csv"
    rows = _read_rows(base_path)
    rows.extend({k: str(v) for k, v in row.__dict__.items()} for row in rows_new)
    seen: set[tuple[str, str]] = set()
    dedup = []
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
                "ok13": sum(str(r["g_ok_13"]).lower() == "true" for r in solver_rows),
            }
        )
    summary.sort(key=lambda item: (item["avg_final_n"], item["avg_moves"]))
    solvers = [item["solver"] for item in summary]
    lookup = {(row["case"], row["solver"]): row for row in dedup}

    csv_path = ROOT / "docs" / "experiments" / "full_real_business_beam_improvements_comparison.csv"
    md_path = ROOT / "docs" / "experiments" / "full_real_business_beam_improvements_comparison.md"
    fields = list(TrajectoryResult.__annotations__.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in dedup:
            writer.writerow({key: row.get(key, "") for key in fields})

    lines = [
        "# Full Real Business Beam Improvements Comparison",
        "",
        "This report tests three additional beam improvement ideas: fused single-search beam, feature-rule adaptive beam, and post-repair suffix beam.",
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
            "- `fused_beam` shares one visited set and one beam while combining plain, enhanced, and QUBO selection signals.",
            "- `feature_adaptive_beam` uses initial layout features to choose one beam variant.",
            "- `beam_post_repair` first gets an incumbent from standard beam, then runs suffix beam repair from that state.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def main() -> None:
    rows: list[TrajectoryResult] = []
    fused_config = SolverConfig(
        max_depth=60,
        beam_width=16,
        max_candidates_per_layer=220,
        time_limit_sec=10.0,
        max_qubo_vars=80,
        qubo_penalty_a=50.0,
        diversity_weight=0.08,
        qubo_num_reads=16,
        qubo_time_limit_sec=0.02,
        qubo_sa_iterations=120,
        qubo_call_interval=5,
        qubo_min_candidates_for_call=1,
        qubo_repair_hamming_slack=0,
        entropy_alpha=1.3,
        random_seed=37,
    )
    for zone, _suffixes, _paths in discover_paired_cases(ROOT / "data" / "input", ["01", "26"]):
        instance = _instance_from_zone(zone)
        case = instance.instance_id
        print(f"running {case}", flush=True)
        rows.append(run_fused_beam(case, instance, fused_config))
        rows.append(run_feature_adaptive_beam(case, instance, 10.0))
        rows.append(run_beam_post_repair(case, instance, 10.0))
    csv_path = ROOT / "docs" / "experiments" / "full_real_business_beam_improvements.csv"
    _write_csv(rows, csv_path)
    _write_comparison(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
