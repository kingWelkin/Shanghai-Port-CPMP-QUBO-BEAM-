"""Benchmark full paired real cases with business-trajectory solvers.

This script compares the historical chen_greedy_algorithm outputs in
``data/results`` against beam and QUBO-beam trajectory search on the same
``data/input`` paired 01+26 cases.

The full real cases are business-improvement problems: the aim is to reduce
``N(Future)`` while respecting ``G = moves + N(Future)`` at every intermediate
state. They are not treated as exact CPMP terminal-sort benchmarks here.
"""

from __future__ import annotations

import csv
import re
import sys
import time
from dataclasses import dataclass
from math import floor, inf
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from greedy.multi_bay_loader import discover_paired_cases, load_combined_zone_bays  # noqa: E402

from container_reshuffle.core import (  # noqa: E402
    Instance,
    ProblemVariant,
    State,
    apply_move,
    create_initial_state,
    generate_legal_moves,
    hash_state,
)
from container_reshuffle.features import disorder_box_count, entropy_limit, entropy_value  # noqa: E402
from container_reshuffle.qubo.beam_selection import QuboBeamSelectionBuilder  # noqa: E402
from container_reshuffle.qubo.sampler import IQuboSampler, QuboSamplerConfig, SimulatedAnnealingQuboSampler  # noqa: E402
from container_reshuffle.scoring import cpmp_business_feasible, dominance_value, score_state  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.beam import Candidate  # noqa: E402


@dataclass
class TrajectoryResult:
    """One algorithm row for a full real case."""

    case: str
    solver: str
    n0: int
    cap_13: int
    cap_15: int
    moves: int | None
    final_n: int | None
    reduction: int | None
    g_end: int | None
    max_g: int | None
    g_ok_13: bool | None
    g_ok_15: bool | None
    runtime_sec: float
    expanded_states: int = 0
    generated_states: int = 0
    visited_states: int = 0
    qubo_calls: int = 0
    qubo_total_vars: int = 0
    qubo_time_sec: float = 0.0
    status: str = ""


def _instance_from_zone(zone: str) -> Instance:
    problem = load_combined_zone_bays(ROOT / "data" / "input", zone, ["01", "26"])
    placements: list[tuple[int, int, int]] = []
    for stack_idx, row in enumerate(problem.bay.seq):
        for level, seq in enumerate(row):
            if seq is not None:
                placements.append((int(seq), stack_idx, level))
    rank_by_slot = {
        (stack_idx, level): rank
        for rank, (_seq, stack_idx, level) in enumerate(sorted(placements), start=1)
    }
    stacks = []
    for stack_idx, row in enumerate(problem.bay.seq):
        stack = []
        for level, seq in enumerate(row):
            if seq is not None:
                stack.append(rank_by_slot[(stack_idx, level)])
        stacks.append(stack)
    return Instance(
        instance_id=f"{zone}_01_26_paired",
        variant=ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=problem.W,
        max_height=problem.H,
        container_count=len(placements),
        stacks=stacks,
    )


def _state_rank(instance: Instance, state: State) -> tuple[int, int, int, int]:
    return (
        disorder_box_count(instance, state),
        state.disorder_total,
        state.entropy_max,
        state.move_count,
    )


def _result_from_state(
    case: str,
    solver: str,
    instance: Instance,
    state: State,
    start: float,
    *,
    expanded_states: int = 0,
    generated_states: int = 0,
    visited_states: int = 0,
    qubo_calls: int = 0,
    qubo_total_vars: int = 0,
    qubo_time_sec: float = 0.0,
    status: str = "OK",
) -> TrajectoryResult:
    n0 = disorder_box_count(instance, State([list(s) for s in instance.stacks]))
    final_n = disorder_box_count(instance, state)
    max_g = state.entropy_max if state.moves else entropy_value(instance, state)
    g_end = state.move_count + final_n
    cap13 = entropy_limit(instance, 1.3)
    cap15 = entropy_limit(instance, 1.5)
    return TrajectoryResult(
        case=case,
        solver=solver,
        n0=n0,
        cap_13=cap13,
        cap_15=cap15,
        moves=state.move_count,
        final_n=final_n,
        reduction=n0 - final_n,
        g_end=g_end,
        max_g=max_g,
        g_ok_13=max_g <= cap13,
        g_ok_15=max_g <= cap15,
        runtime_sec=time.perf_counter() - start,
        expanded_states=expanded_states,
        generated_states=generated_states,
        visited_states=visited_states,
        qubo_calls=qubo_calls,
        qubo_total_vars=qubo_total_vars,
        qubo_time_sec=qubo_time_sec,
        status=status,
    )


def _run_business_beam(
    case: str,
    instance: Instance,
    config: SolverConfig,
) -> TrajectoryResult:
    start = time.perf_counter()
    initial = create_initial_state(instance)
    best = initial
    h0 = hash_state(instance, initial, config.enable_canonical_hash)
    beam = [Candidate(initial, score_state(instance, initial, config.score_weights), None, h0)]
    visited = {h0: dominance_value(instance, initial)}
    expanded = generated = 0

    for _depth in range(config.max_depth):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        candidates: list[Candidate] = []
        for cand in beam:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expanded += 1
            moves = generate_legal_moves(instance, cand.state)
            scored_moves = []
            for mv in moves:
                try:
                    ns = apply_move(instance, cand.state, mv)
                except ValueError:
                    continue
                if not cpmp_business_feasible(instance, ns, config.entropy_alpha):
                    continue
                scored_moves.append((score_state(instance, ns, config.score_weights), mv, ns))
            scored_moves.sort(key=lambda x: x[0])
            if config.candidate_limit_per_state is not None:
                scored_moves = scored_moves[: config.candidate_limit_per_state]
            for sc, mv, ns in scored_moves:
                h = hash_state(instance, ns, config.enable_canonical_hash)
                dom = dominance_value(instance, ns)
                if config.enable_visited_pruning and visited.get(h, inf) <= dom:
                    continue
                visited[h] = dom
                if _state_rank(instance, ns) < _state_rank(instance, best):
                    best = ns
                candidates.append(Candidate(ns, sc, mv, h))
        generated += len(candidates)
        if not candidates or disorder_box_count(instance, best) == 0:
            break
        candidates.sort(key=lambda c: (c.score, c.state.move_count))
        candidates = candidates[: config.max_candidates_per_layer]
        beam = candidates[: config.beam_width]

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    return _result_from_state(
        case,
        "beam",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=len(visited),
        status=status,
    )


def _run_business_qubo_beam(
    case: str,
    instance: Instance,
    config: SolverConfig,
    *,
    sampler: IQuboSampler | None = None,
    solver_name: str = "qubo_beam",
) -> TrajectoryResult:
    start = time.perf_counter()
    initial = create_initial_state(instance)
    best = initial
    h0 = hash_state(instance, initial, config.enable_canonical_hash)
    beam = [Candidate(initial, score_state(instance, initial, config.score_weights), None, h0)]
    visited = {h0: dominance_value(instance, initial)}
    expanded = generated = qubo_calls = qubo_vars = 0
    qubo_time = 0.0
    sampler_names: list[str] = []
    builder = QuboBeamSelectionBuilder()
    sampler = sampler or SimulatedAnnealingQuboSampler()
    call_interval = max(1, config.qubo_call_interval)
    min_candidates = max(0, config.qubo_min_candidates_for_call)
    q_time_limit = max(0.001, config.qubo_time_limit_sec or min(1.0, config.time_limit_sec))

    for depth in range(config.max_depth):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        candidates: list[Candidate] = []
        for cand in beam:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expanded += 1
            scored_moves = []
            for mv in generate_legal_moves(instance, cand.state):
                try:
                    ns = apply_move(instance, cand.state, mv)
                except ValueError:
                    continue
                if not cpmp_business_feasible(instance, ns, config.entropy_alpha):
                    continue
                scored_moves.append((score_state(instance, ns, config.score_weights), mv, ns))
            scored_moves.sort(key=lambda x: x[0])
            if config.candidate_limit_per_state is not None:
                scored_moves = scored_moves[: config.candidate_limit_per_state]
            for sc, mv, ns in scored_moves:
                h = hash_state(instance, ns, config.enable_canonical_hash)
                dom = dominance_value(instance, ns)
                if config.enable_visited_pruning and visited.get(h, inf) <= dom:
                    continue
                visited[h] = dom
                if _state_rank(instance, ns) < _state_rank(instance, best):
                    best = ns
                candidates.append(Candidate(ns, sc, mv, h))
        generated += len(candidates)
        if not candidates or disorder_box_count(instance, best) == 0:
            break
        candidates.sort(key=lambda c: (c.score, c.state.move_count))
        candidates = candidates[: config.max_candidates_per_layer]
        use_qubo = depth % call_interval == 0 and len(candidates) >= min_candidates
        if use_qubo:
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
                    time_limit_sec=q_time_limit,
                    sa_iterations=config.qubo_sa_iterations,
                ),
            )
            qubo_time += samples.runtime_sec
            sampler_names.append(samples.sampler_name)
            elite_keep_count = max(1, int(round(config.beam_width * 0.8)))
            beam, _repair = builder.decode_samples_and_repair(
                samples.samples,
                usable,
                config.beam_width,
                elite_keep_count=elite_keep_count,
                diversity_weight=config.diversity_weight,
                repair_hamming_slack=int(getattr(config, "qubo_repair_hamming_slack", 3)),
            )
        else:
            beam = candidates[: config.beam_width]

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    if sampler_names:
        status = f"{status}; samplers={','.join(sorted(set(sampler_names)))}"
    return _result_from_state(
        case,
        solver_name,
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=len(visited),
        qubo_calls=qubo_calls,
        qubo_total_vars=qubo_vars,
        qubo_time_sec=qubo_time,
        status=status,
    )


def _parse_legacy_result(case: str, instance: Instance) -> TrajectoryResult:
    path = ROOT / "data" / "results" / f"result_{case}.txt"
    text = path.read_text(encoding="utf-8")
    moves = _parse_int(text, r"greedy_moves:\s*(\d+)")
    n0 = _parse_int(text, r"N\(0\)=(\d+)")
    g_end = _parse_int(text, r"G_end=(\d+)")
    g_cap = _parse_int(text, r"G_cap=(\d+)")
    final_n = None
    match = re.search(r"N\(Future\)\s*(\d+)\D+(\d+)", text)
    if match:
        final_n = int(match.group(2))
    if final_n is None and moves is not None and g_end is not None:
        final_n = g_end - moves
    if n0 is None:
        n0 = disorder_box_count(instance, State([list(s) for s in instance.stacks]))
    cap13 = floor(1.3 * n0 + 1e-9)
    cap15 = floor(1.5 * n0 + 1e-9)
    return TrajectoryResult(
        case=case,
        solver="chen_greedy_algorithm_legacy",
        n0=n0,
        cap_13=cap13,
        cap_15=g_cap or cap15,
        moves=moves,
        final_n=final_n,
        reduction=None if final_n is None else n0 - final_n,
        g_end=g_end,
        max_g=g_end,
        g_ok_13=None if g_end is None else g_end <= cap13,
        g_ok_15=None if g_end is None else g_end <= (g_cap or cap15),
        runtime_sec=_parse_float(text, r"greedy_sec:\s*([0-9.]+)") or 0.0,
        status="historical; cap=1.5*N0",
    )


def _parse_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    return None if not match else int(match.group(1))


def _parse_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    return None if not match else float(match.group(1))


def _write_csv(rows: Iterable[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrajectoryResult.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_markdown(rows: list[TrajectoryResult], path: Path, csv_path: Path) -> None:
    aggregate = _aggregate_rows(rows)
    lines = [
        "# Full Real Cases: Greedy vs Beam vs QUBO Beam",
        "",
        "Scope: 15 paired `data/input` cases (`01+26`, `W=20,H=6,N=60`).",
        "Metric: best business trajectory under `G=M+N(Future)<=floor(1.3*N0)` for beam/QUBO beam.",
        "The historical greedy rows are parsed from `data/results` and were generated under the old `1.5*N0` cap.",
        "",
        f"CSV: `{csv_path.as_posix()}`",
        "",
        "Run config for beam/QUBO beam: `10s` per solver per case, `beam_width=10`, `max_candidates_per_layer=100`, `candidate_limit_per_state=3`, `entropy_alpha=1.3`; QUBO uses `diversity_weight=0.1` with elite retention. Duplicate original sequence IDs are tie-broken by position to satisfy the unique-container state engine.",
        "",
        "## Aggregate",
        "",
        "| Solver | OK13 cases | Avg final N | Avg reduction | Avg moves | Avg runtime(s) | Avg QUBO calls | Best/tied among OK13 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for solver in ["chen_greedy_algorithm_legacy", "beam", "qubo_beam"]:
        agg = aggregate[solver]
        lines.append(
            "| "
            + " | ".join(
                [
                    solver,
                    f"{agg['ok13']}/15",
                    f"{agg['avg_final_n']:.2f}",
                    f"{agg['avg_reduction']:.2f}",
                    f"{agg['avg_moves']:.2f}",
                    f"{agg['avg_runtime']:.3f}",
                    f"{agg['avg_qubo_calls']:.2f}",
                    f"{agg['best_ok13']}/15",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Detail",
            "",
        "| Case | Solver | N0 | Cap13 | Moves | Final N | Reduction | G_end | Max G | OK13 | Runtime(s) | Expanded | Generated | QUBO calls | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.case,
                    row.solver,
                    _fmt(row.n0),
                    _fmt(row.cap_13),
                    _fmt(row.moves),
                    _fmt(row.final_n),
                    _fmt(row.reduction),
                    _fmt(row.g_end),
                    _fmt(row.max_g),
                    _fmt(row.g_ok_13),
                    _fmt(row.runtime_sec),
                    _fmt(row.expanded_states),
                    _fmt(row.generated_states),
                    _fmt(row.qubo_calls),
                    row.status,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- Lower `Final N` is better; `Reduction=N0-Final N` is the direct business improvement.",
            "- `OK13` checks the current stricter `1.3*N0` cap. Historical greedy used `1.5*N0`, so it can fail `OK13` even when its original report said compliant.",
            "- `BEST_EFFORT` means the search returned the best trajectory seen within the time/depth budget, not a proof of optimality.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_rows(rows: list[TrajectoryResult]) -> dict[str, dict[str, float]]:
    solvers = sorted({row.solver for row in rows})
    by_solver = {solver: [row for row in rows if row.solver == solver] for solver in solvers}
    by_case: dict[str, list[TrajectoryResult]] = {}
    for row in rows:
        by_case.setdefault(row.case, []).append(row)
    best_ok13: dict[str, int] = {solver: 0 for solver in solvers}
    for case_rows in by_case.values():
        ok_rows = [row for row in case_rows if row.g_ok_13 and row.final_n is not None]
        if not ok_rows:
            continue
        best = min(int(row.final_n) for row in ok_rows)
        for row in ok_rows:
            if int(row.final_n) == best:
                best_ok13[row.solver] += 1
    out = {}
    for solver, solver_rows in by_solver.items():
        out[solver] = {
            "ok13": sum(1 for row in solver_rows if row.g_ok_13),
            "avg_final_n": sum(float(row.final_n or 0) for row in solver_rows) / len(solver_rows),
            "avg_reduction": sum(float(row.reduction or 0) for row in solver_rows) / len(solver_rows),
            "avg_moves": sum(float(row.moves or 0) for row in solver_rows) / len(solver_rows),
            "avg_runtime": sum(row.runtime_sec for row in solver_rows) / len(solver_rows),
            "avg_qubo_calls": sum(row.qubo_calls for row in solver_rows) / len(solver_rows),
            "best_ok13": best_ok13.get(solver, 0),
        }
    return out


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
        entropy_alpha=1.3,
        random_seed=37,
    )
    rows: list[TrajectoryResult] = []
    cases = discover_paired_cases(ROOT / "data" / "input", ["01", "26"])
    for zone, _suffixes, _paths in cases:
        instance = _instance_from_zone(zone)
        case = instance.instance_id
        print(f"running {case}")
        rows.append(_parse_legacy_result(case, instance))
        rows.append(_run_business_beam(case, instance, config))
        rows.append(_run_business_qubo_beam(case, instance, config))
    csv_path = ROOT / "docs" / "experiments" / "full_real_business_beam_qubo_vs_greedy.csv"
    md_path = ROOT / "docs" / "experiments" / "full_real_business_beam_qubo_vs_greedy.md"
    _write_csv(rows, csv_path)
    _write_markdown(rows, md_path, csv_path.relative_to(ROOT))
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
