"""Enhanced beam variants for full real paired business cases."""

from __future__ import annotations

import csv
import math
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
    entropy_limit,
    entropy_value,
    inversion_count,
    settled_box_count,
    well_prefix_gain,
)
from container_reshuffle.scoring import cpmp_business_feasible, dominance_value, score_state
from container_reshuffle.similarity import combined_similarity
from container_reshuffle.solvers.base import SolverConfig
from container_reshuffle.solvers.beam import Candidate


def _state_rank(instance: Instance, state: State) -> tuple[int, int, int, int, int, int]:
    return (
        disorder_box_count(instance, state),
        state.entropy_max,
        state.disorder_total,
        state.move_count,
        inversion_count(instance, state),
        bad_count(instance, state),
    )


def _beam_primary_key(instance: Instance, state: State, config: SolverConfig) -> tuple:
    n = disorder_box_count(instance, state)
    g = entropy_value(instance, state)
    cap = entropy_limit(instance, config.entropy_alpha)
    slack = cap - g
    return (
        n,
        -g,
        state.entropy_max,
        state.disorder_total,
        inversion_count(instance, state),
        bad_count(instance, state),
        dirty_stack_count(instance, state),
        -settled_box_count(instance, state),
        state.move_count,
        abs(slack),
    )


def _move_rule_keys(instance: Instance, current: State, nxt: State, config: SolverConfig) -> list[tuple]:
    n_current = disorder_box_count(instance, current)
    n_next = disorder_box_count(instance, nxt)
    g_next = entropy_value(instance, nxt)
    gain = n_current - n_next
    prefix_gain = well_prefix_gain(instance, current, nxt)
    inv = inversion_count(instance, nxt)
    bad = bad_count(instance, nxt)
    dirty = dirty_stack_count(instance, nxt)
    settled = settled_box_count(instance, nxt)
    score = score_state(instance, nxt, config.score_weights)
    common = (
        n_next,
        -g_next,
        nxt.entropy_max,
        nxt.disorder_total,
        inv,
        bad,
        dirty,
        -settled,
        nxt.move_count,
    )
    return [
        (score,) + common,
        common,
        (-gain, n_next, -g_next, inv, bad, nxt.move_count),
        (-prefix_gain, n_next, inv, bad, dirty, nxt.move_count),
        (g_next, n_next, nxt.entropy_max, inv, bad, nxt.move_count),
        (inv, bad, n_next, -g_next, dirty, nxt.move_count),
    ]


def _select_state_expansions(
    instance: Instance,
    state: State,
    config: SolverConfig,
    *,
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
    for rule_idx in range(6):
        ranked = sorted(evaluated, key=lambda item: item[3][rule_idx])
        for mv, nxt, score, _keys in ranked[:per_rule_limit]:
            selected[hash_state(instance, nxt, config.enable_canonical_hash)] = (score, mv, nxt)
    return list(selected.values()), len(evaluated)


def _normalized_ranks(candidates: list[Candidate], instance: Instance, config: SolverConfig) -> dict[int, float]:
    ordered = sorted(range(len(candidates)), key=lambda idx: _beam_primary_key(instance, candidates[idx].state, config))
    if len(ordered) <= 1:
        return {idx: 0.0 for idx in ordered}
    return {idx: pos / (len(ordered) - 1) for pos, idx in enumerate(ordered)}


def _select_enhanced_beam(
    candidates: list[Candidate],
    instance: Instance,
    config: SolverConfig,
    *,
    diversity_weight: float,
    elite_ratio: float,
) -> list[Candidate]:
    if len(candidates) <= config.beam_width:
        return list(candidates)
    candidates = sorted(candidates, key=lambda c: _beam_primary_key(instance, c.state, config))
    candidates = candidates[: config.max_candidates_per_layer]
    target = min(config.beam_width, len(candidates))
    selected_idx: list[int] = []
    selected_set: set[int] = set()

    def add(idx: int) -> None:
        if idx not in selected_set and len(selected_idx) < target:
            selected_idx.append(idx)
            selected_set.add(idx)

    elite_count = max(1, min(target, int(math.ceil(target * elite_ratio))))
    for idx in range(elite_count):
        add(idx)

    # Preserve the best representatives across N(Future) buckets. This reduces
    # the chance of pruning a branch that is slightly worse now but structurally different.
    seen_n: set[int] = set()
    for idx, cand in enumerate(candidates):
        n = disorder_box_count(instance, cand.state)
        if n in seen_n:
            continue
        seen_n.add(n)
        add(idx)
        if len(selected_idx) >= max(elite_count, target // 2):
            break

    norm = _normalized_ranks(candidates, instance, config)
    while len(selected_idx) < target:
        best_idx = None
        best_value = float("inf")
        for idx, cand in enumerate(candidates):
            if idx in selected_set:
                continue
            if selected_idx:
                sim = sum(combined_similarity(cand.state, candidates[j].state) for j in selected_idx) / len(selected_idx)
            else:
                sim = 0.0
            value = norm[idx] + diversity_weight * sim
            if value < best_value:
                best_value = value
                best_idx = idx
        if best_idx is None:
            break
        add(best_idx)
    return [candidates[idx] for idx in selected_idx]


def _select_hybrid_beam(
    candidates: list[Candidate],
    instance: Instance,
    config: SolverConfig,
    *,
    diversity_weight: float,
) -> list[Candidate]:
    """Select a beam with explicit quotas for score-first and business-rank fronts."""

    if len(candidates) <= config.beam_width:
        return list(candidates)
    target = min(config.beam_width, len(candidates))
    candidates = candidates[:]
    score_order = sorted(range(len(candidates)), key=lambda idx: (candidates[idx].score, candidates[idx].state.move_count))
    primary_order = sorted(range(len(candidates)), key=lambda idx: _beam_primary_key(instance, candidates[idx].state, config))
    selected_idx: list[int] = []
    selected_set: set[int] = set()

    def add(idx: int) -> None:
        if idx not in selected_set and len(selected_idx) < target:
            selected_idx.append(idx)
            selected_set.add(idx)

    score_quota = max(1, int(round(0.45 * target)))
    primary_quota = max(1, int(round(0.35 * target)))
    for idx in score_order[:score_quota]:
        add(idx)
    for idx in primary_order[:primary_quota]:
        add(idx)

    seen_n: set[int] = set()
    for idx in primary_order:
        n = disorder_box_count(instance, candidates[idx].state)
        if n in seen_n:
            continue
        seen_n.add(n)
        add(idx)
        if len(selected_idx) >= int(round(0.75 * target)):
            break

    norm = _normalized_ranks(candidates, instance, config)
    while len(selected_idx) < target:
        best_idx = None
        best_value = float("inf")
        for idx, cand in enumerate(candidates):
            if idx in selected_set:
                continue
            sim = sum(combined_similarity(cand.state, candidates[j].state) for j in selected_idx) / len(selected_idx)
            score_pos = score_order.index(idx) / max(1, len(score_order) - 1)
            value = 0.55 * norm[idx] + 0.25 * score_pos + diversity_weight * sim
            if value < best_value:
                best_value = value
                best_idx = idx
        if best_idx is None:
            break
        add(best_idx)
    return [candidates[idx] for idx in selected_idx]


def run_enhanced_business_beam(
    case: str,
    instance: Instance,
    config: SolverConfig,
    *,
    solver_name: str,
    per_rule_limit: int = 2,
    diversity_weight: float = 0.08,
    elite_ratio: float = 0.55,
) -> TrajectoryResult:
    """Run enhanced multi-rule, diversity-preserving beam search."""

    start = time.perf_counter()
    initial = create_initial_state(instance)
    best = initial
    h0 = hash_state(instance, initial, config.enable_canonical_hash)
    beam = [Candidate(initial, score_state(instance, initial, config.score_weights), None, h0)]
    visited = {h0: dominance_value(instance, initial)}
    expanded = generated = evaluated_moves = 0

    for _depth in range(config.max_depth):
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
        beam = _select_enhanced_beam(
            candidates,
            instance,
            config,
            diversity_weight=diversity_weight,
            elite_ratio=elite_ratio,
        )

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += f"; per_rule={per_rule_limit}; evaluated_moves={evaluated_moves}"
    return _result_from_state(
        case,
        solver_name,
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=len(visited),
        status=status,
    )


def run_hybrid_business_beam(
    case: str,
    instance: Instance,
    config: SolverConfig,
    *,
    solver_name: str = "hybrid_enhanced_beam",
    per_rule_limit: int = 2,
    diversity_weight: float = 0.05,
) -> TrajectoryResult:
    """Run mixed-frontier beam search that keeps score and business-rank fronts."""

    start = time.perf_counter()
    initial = create_initial_state(instance)
    best = initial
    h0 = hash_state(instance, initial, config.enable_canonical_hash)
    beam = [Candidate(initial, score_state(instance, initial, config.score_weights), None, h0)]
    visited = {h0: dominance_value(instance, initial)}
    expanded = generated = evaluated_moves = 0

    for _depth in range(config.max_depth):
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
        candidates.sort(key=lambda c: min((c.score, c.state.move_count), _beam_primary_key(instance, c.state, config)))
        candidates = candidates[: config.max_candidates_per_layer]
        beam = _select_hybrid_beam(
            candidates,
            instance,
            config,
            diversity_weight=diversity_weight,
        )

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += f"; per_rule={per_rule_limit}; evaluated_moves={evaluated_moves}"
    return _result_from_state(
        case,
        solver_name,
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=len(visited),
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


def _write_comparison(enhanced_rows: list[TrajectoryResult]) -> None:
    base_path = ROOT / "docs" / "experiments" / "full_real_business_genetic_tabu_comparison.csv"
    rows = _read_rows(base_path) if base_path.exists() else []
    rows.extend({k: str(v) for k, v in row.__dict__.items()} for row in enhanced_rows)

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

    csv_path = ROOT / "docs" / "experiments" / "full_real_business_enhanced_beam_comparison.csv"
    md_path = ROOT / "docs" / "experiments" / "full_real_business_enhanced_beam_comparison.md"
    fields = list(TrajectoryResult.__annotations__.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in dedup:
            writer.writerow({k: row.get(k, "") for k in fields})

    lines = [
        "# Full Real Business Enhanced Beam Comparison",
        "",
        "This report adds enhanced beam variants. The enhancement uses multi-rule action retention per state and diversity-preserving layer selection instead of a single score-only truncation.",
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
            "- The enhanced variants test whether current beam quality is limited by score-only move pruning.",
            "- If enhanced beam improves only on selected cases, the next step is case-adaptive rule selection rather than simply increasing beam width.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def main() -> None:
    base_config = SolverConfig(
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
    enhanced_fast = SolverConfig(
        max_depth=60,
        beam_width=14,
        max_candidates_per_layer=180,
        time_limit_sec=10.0,
        entropy_alpha=1.3,
        random_seed=37,
    )
    enhanced_wide = SolverConfig(
        max_depth=60,
        beam_width=24,
        max_candidates_per_layer=320,
        time_limit_sec=10.0,
        entropy_alpha=1.3,
        random_seed=37,
    )

    rows: list[TrajectoryResult] = []
    for zone, _suffixes, _paths in discover_paired_cases(ROOT / "data" / "input", ["01", "26"]):
        instance = _instance_from_zone(zone)
        case = instance.instance_id
        print(f"running {case}", flush=True)
        rows.append(_run_business_beam(case, instance, base_config))
        rows.append(_run_business_qubo_beam(case, instance, base_config, solver_name="qubo_beam"))
        rows.append(
            run_enhanced_business_beam(
                case,
                instance,
                enhanced_fast,
                solver_name="enhanced_beam_fast",
                per_rule_limit=2,
                diversity_weight=0.06,
                elite_ratio=0.60,
            )
        )
        rows.append(
            run_enhanced_business_beam(
                case,
                instance,
                enhanced_wide,
                solver_name="enhanced_beam_wide",
                per_rule_limit=3,
                diversity_weight=0.08,
                elite_ratio=0.55,
            )
        )

    csv_path = ROOT / "docs" / "experiments" / "full_real_business_enhanced_beam.csv"
    _write_csv(rows, csv_path)
    _write_comparison(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
