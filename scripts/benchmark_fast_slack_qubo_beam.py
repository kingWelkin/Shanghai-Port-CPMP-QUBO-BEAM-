"""Fast slack beam with protected QUBO supplemental selection.

The deterministic fast-slack beam is kept as the protected core.  QUBO is used
only to select diverse supplemental one-step candidates from the remaining
pool, so it cannot prune away the known-good rule trajectory.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam import _write_explain_csv  # noqa: E402
from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from benchmark_fast_slack_beam import StepCandidate, _best_key, _score, _select_bucketed  # noqa: E402
from benchmark_full_real_business_algorithms import TrajectoryResult, _result_from_state  # noqa: E402
from benchmark_macro_reservoir_qubo_beam import tail_repair_to_n_zero  # noqa: E402
from container_reshuffle.core import Instance, State, apply_move, create_initial_state, generate_legal_moves, hash_state  # noqa: E402
from container_reshuffle.features import disorder_box_count, entropy_limit  # noqa: E402
from container_reshuffle.qubo.model import QuboModel  # noqa: E402
from container_reshuffle.qubo.sampler import QuboSamplerConfig, SimulatedAnnealingQuboSampler  # noqa: E402
from container_reshuffle.scoring import cpmp_business_feasible, dominance_value  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.beam import Candidate  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


def _candidate_rank_scores(candidates: list[StepCandidate]) -> dict[tuple, float]:
    ranked = sorted(candidates, key=lambda cand: cand.score)
    return {cand.hash_value: float(idx) for idx, cand in enumerate(ranked)}


def _usable_supplement_pool(
    candidates: list[StepCandidate],
    *,
    max_vars: int,
    target: int,
) -> list[StepCandidate]:
    ranked = sorted(candidates, key=lambda cand: cand.score)
    selected: dict[tuple, StepCandidate] = {}

    def add(rows: list[StepCandidate]) -> None:
        for cand in rows:
            old = selected.get(cand.hash_value)
            if old is None or cand.score < old.score:
                selected[cand.hash_value] = cand

    add(ranked[: max(target * 2, target + 8)])
    for src in sorted({cand.move.from_stack for cand in candidates}):
        bucket = [cand for cand in candidates if cand.move.from_stack == src]
        add(sorted(bucket, key=lambda cand: cand.score)[:1])
    for dst in sorted({cand.move.to_stack for cand in candidates}):
        bucket = [cand for cand in candidates if cand.move.to_stack == dst]
        add(sorted(bucket, key=lambda cand: cand.score)[:1])
    add(ranked)
    usable = list(selected.values())
    usable.sort(key=lambda cand: cand.score)
    return usable[:max_vars]


def _pair_penalty(a: StepCandidate, b: StepCandidate, diversity_weight: float) -> float:
    penalty = 0.0
    if a.move.from_stack == b.move.from_stack:
        penalty += 0.35
    if a.move.to_stack == b.move.to_stack:
        penalty += 0.20
    if a.move.box == b.move.box:
        penalty += 0.30
    if a.move.from_stack == b.move.from_stack and a.move.to_stack == b.move.to_stack:
        penalty += 0.60
    return diversity_weight * penalty


def _selection_value(
    usable: list[StepCandidate],
    selected: set[int],
    score_norm: list[float],
    diversity_weight: float,
) -> float:
    value = sum(score_norm[idx] for idx in selected)
    selected_list = sorted(selected)
    for pos, i in enumerate(selected_list):
        for j in selected_list[pos + 1 :]:
            value += _pair_penalty(usable[i], usable[j], diversity_weight)
    return value


def _repair_indices(
    usable: list[StepCandidate],
    selected: set[int],
    target: int,
    score_norm: list[float],
    diversity_weight: float,
) -> set[int]:
    selected = set(selected)
    target = min(target, len(usable))
    while len(selected) > target:
        remove = min(
            selected,
            key=lambda idx: (
                _selection_value(usable, selected - {idx}, score_norm, diversity_weight),
                -score_norm[idx],
                idx,
            ),
        )
        selected.remove(remove)
    while len(selected) < target:
        missing = [idx for idx in range(len(usable)) if idx not in selected]
        if not missing:
            break
        add = min(
            missing,
            key=lambda idx: (
                _selection_value(usable, selected | {idx}, score_norm, diversity_weight),
                score_norm[idx],
                idx,
            ),
        )
        selected.add(add)
    return selected


def _select_qubo_indices(
    usable: list[StepCandidate],
    config: SolverConfig,
    target: int,
) -> tuple[list[StepCandidate], int, float]:
    target = min(target, len(usable))
    if target <= 0:
        return [], 0, 0.0
    if len(usable) <= target:
        return usable, 0, 0.0

    ranks = _candidate_rank_scores(usable)
    raw_scores = [ranks[cand.hash_value] for cand in usable]
    lo, hi = min(raw_scores), max(raw_scores)
    score_norm = [0.0 if hi == lo else (score - lo) / (hi - lo) for score in raw_scores]

    model = QuboModel()
    vars_ = [model.add_binary_var(f"s_{idx}") for idx in range(len(usable))]
    for idx, var in enumerate(vars_):
        model.add_linear(var, score_norm[idx])
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            coeff = _pair_penalty(usable[i], usable[j], config.diversity_weight)
            if coeff:
                model.add_quadratic(vars_[i], vars_[j], coeff)
    model.add_exactly_k_penalty(vars_, target, config.qubo_penalty_a)

    sample_set = SimulatedAnnealingQuboSampler().sample(
        model,
        QuboSamplerConfig(
            num_reads=config.qubo_num_reads,
            random_seed=config.random_seed,
            time_limit_sec=max(0.001, config.qubo_time_limit_sec or 0.01),
            sa_iterations=max(1, int(config.qubo_sa_iterations)),
        ),
    )

    best: set[int] | None = None
    best_value = float("inf")
    for sample in sample_set.samples or []:
        selected = {idx for idx in range(len(usable)) if sample.binary_values.get(idx, 0) == 1}
        repaired = _repair_indices(usable, selected, target, score_norm, config.diversity_weight)
        value = _selection_value(usable, repaired, score_norm, config.diversity_weight)
        if value < best_value:
            best = repaired
            best_value = value
    if best is None:
        best = _repair_indices(usable, set(), target, score_norm, config.diversity_weight)
    out = [usable[idx] for idx in sorted(best, key=lambda idx: (usable[idx].score, idx))]
    return out, len(model.var_index), sample_set.runtime_sec


def _select_with_qubo_supplement(
    instance: Instance,
    candidates: list[StepCandidate],
    config: SolverConfig,
    *,
    core_width: int,
    supplement_width: int,
) -> tuple[list[StepCandidate], int, int, float]:
    """Select a protected fast-slack core plus QUBO-diverse supplement."""

    if len(candidates) <= config.beam_width:
        return sorted(candidates, key=lambda cand: cand.score), 0, 0, 0.0

    core_width = max(0, min(core_width, config.beam_width))
    supplement_width = max(0, min(supplement_width, config.beam_width - core_width))
    protected = _select_bucketed(instance, candidates, core_width) if core_width else []
    selected_by_hash = {cand.hash_value: cand for cand in protected}
    if not supplement_width:
        selected = _select_bucketed(instance, candidates, config.beam_width)
        return selected, 0, 0, 0.0

    supplement_pool = [cand for cand in candidates if cand.hash_value not in selected_by_hash]
    usable = _usable_supplement_pool(
        supplement_pool,
        max_vars=config.max_qubo_vars,
        target=supplement_width,
    )
    if not usable:
        selected = _select_bucketed(instance, candidates, config.beam_width)
        return selected, 0, 0, 0.0

    selected_qubo, qvars, qtime = _select_qubo_indices(usable, config, supplement_width)
    for cand in selected_qubo:
        selected_by_hash.setdefault(cand.hash_value, cand)

    # Fill any cardinality gap deterministically.  This keeps the beam width
    # stable if QUBO returns duplicate/invalid branches after repair.
    for cand in _select_bucketed(instance, candidates, config.beam_width):
        selected_by_hash.setdefault(cand.hash_value, cand)
        if len(selected_by_hash) >= config.beam_width:
            break

    selected = list(selected_by_hash.values())
    selected.sort(key=lambda cand: cand.score)
    return selected[: config.beam_width], 1 if qvars else 0, qvars, qtime


def run_fast_slack_qubo_beam(
    case: str,
    instance: Instance,
    config: SolverConfig,
    *,
    core_width: int,
    supplement_width: int,
    qubo_min_depth: int,
    qubo_call_interval: int,
    qubo_plateau_only: bool,
    tail_repair_n: int,
    tail_repair_expanded: int,
) -> tuple[TrajectoryResult, State]:
    start = time.perf_counter()
    initial = create_initial_state(instance)
    best = initial
    h0 = hash_state(instance, initial, config.enable_canonical_hash)
    beam = [Candidate(initial, 0.0, None, h0)]
    visited = {h0: dominance_value(instance, initial)}
    expanded = generated = qcalls = qvars = tail_expanded = tail_generated = 0
    qtime = 0.0
    tail_found = False
    plateau_layers = 0

    for depth in range(config.max_depth):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        best_n_before = disorder_box_count(instance, best)
        step_candidates: list[StepCandidate] = []
        for cand in beam:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expanded += 1
            for move in generate_legal_moves(instance, cand.state):
                try:
                    nxt = apply_move(instance, cand.state, move)
                except ValueError:
                    continue
                if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                    continue
                h = hash_state(instance, nxt, config.enable_canonical_hash)
                dom = dominance_value(instance, nxt)
                if config.enable_visited_pruning and visited.get(h, math.inf) <= dom:
                    continue
                step_candidates.append(StepCandidate(nxt, move, h, _score(instance, cand.state, nxt, move), "step"))
        generated += len(step_candidates)
        if not step_candidates:
            break

        layer_no = depth + 1
        use_qubo = (
            layer_no >= qubo_min_depth
            and (layer_no - qubo_min_depth) % max(1, qubo_call_interval) == 0
            and (not qubo_plateau_only or plateau_layers > 0)
        )
        if use_qubo:
            selected, calls, vars_count, sample_time = _select_with_qubo_supplement(
                instance,
                step_candidates,
                config,
                core_width=core_width,
                supplement_width=supplement_width,
            )
        else:
            selected = _select_bucketed(instance, step_candidates, core_width)
            calls = vars_count = 0
            sample_time = 0.0
        qcalls += calls
        qvars += vars_count
        qtime += sample_time

        next_beam: list[Candidate] = []
        for item in selected:
            visited[item.hash_value] = dominance_value(instance, item.state)
            if _best_key(instance, item.state) < _best_key(instance, best):
                best = item.state
            next_beam.append(Candidate(item.state, float(item.score[1]), item.move, item.hash_value))
        beam = next_beam

        low_n = [cand.state for cand in beam if disorder_box_count(instance, cand.state) <= tail_repair_n]
        low_n.sort(key=lambda state: _best_key(instance, state))
        for state in low_n[:2]:
            repaired, exp_count, gen_count = tail_repair_to_n_zero(
                instance,
                state,
                config,
                deadline=start + config.time_limit_sec,
                max_expanded=tail_repair_expanded,
            )
            tail_expanded += exp_count
            tail_generated += gen_count
            if repaired is not None:
                best = repaired
                tail_found = True
                break

        best_n_after = disorder_box_count(instance, best)
        plateau_layers = plateau_layers + 1 if best_n_after >= best_n_before else 0

        print(
            "LAYER "
            f"depth={depth + 1} beam={len(beam)} bestN={disorder_box_count(instance, best)} "
            f"beamBestN={disorder_box_count(instance, beam[0].state)} qvars={vars_count} "
            f"elapsed={time.perf_counter() - start:.3f}s",
            flush=True,
        )
        if disorder_box_count(instance, best) == 0:
            break

    status = "N_ZERO" if disorder_box_count(instance, best) == 0 else "BEST_EFFORT"
    status += (
        "; fast_slack+protected_qubo_supplement"
        f"; core_width={core_width}; supplement_width={supplement_width}; "
        f"qubo_min_depth={qubo_min_depth}; qubo_call_interval={qubo_call_interval}; "
        f"qubo_plateau_only={qubo_plateau_only}; "
        f"tail_expanded={tail_expanded}; tail_generated={tail_generated}; tail_found={tail_found}"
    )
    row = _result_from_state(
        case,
        "fast_slack_qubo_beam",
        instance,
        best,
        start,
        expanded_states=expanded,
        generated_states=generated,
        visited_states=len(visited),
        qubo_calls=qcalls,
        qubo_total_vars=qvars,
        qubo_time_sec=qtime,
        status=status,
    )
    return row, best


def _write_result_csv(row: TrajectoryResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrajectoryResult.__annotations__.keys()))
        writer.writeheader()
        writer.writerow(row.__dict__)


def _write_rows_csv(rows: list[TrajectoryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrajectoryResult.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _run_one(args: argparse.Namespace, zone: str, bay: str) -> tuple[TrajectoryResult, State]:
    instance = _instance_from_single_bay(zone, bay)
    cap = entropy_limit(instance, 1.3)
    supplement_width = args.qubo_supplement_width
    core_width = args.core_width
    if supplement_width is None:
        supplement_width = max(8, args.beam_width // 4)
    if core_width is None:
        core_width = max(0, args.beam_width - supplement_width)
    config = SolverConfig(
        max_depth=min(args.max_depth, cap),
        beam_width=args.beam_width,
        time_limit_sec=args.time_limit,
        max_qubo_vars=args.max_qubo_vars,
        qubo_penalty_a=args.qubo_penalty_a,
        diversity_weight=args.diversity_weight,
        qubo_num_reads=args.qubo_reads,
        qubo_time_limit_sec=args.qubo_time_limit,
        qubo_sa_iterations=args.qubo_iterations,
        qubo_repair_hamming_slack=args.qubo_repair_hamming_slack,
        entropy_alpha=1.3,
        random_seed=37,
    )
    initial = create_initial_state(instance)
    print(
        "START "
        f"case={instance.instance_id} N0={disorder_box_count(instance, initial)} "
        f"cap={cap} beam={config.beam_width} core={core_width} supplement={supplement_width}",
        flush=True,
    )
    return run_fast_slack_qubo_beam(
        instance.instance_id,
        instance,
        config,
        core_width=core_width,
        supplement_width=supplement_width,
        qubo_min_depth=args.qubo_min_depth,
        qubo_call_interval=args.qubo_call_interval,
        qubo_plateau_only=args.qubo_plateau_only,
        tail_repair_n=args.tail_repair_n,
        tail_repair_expanded=args.tail_repair_expanded,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="C1")
    parser.add_argument("--bay", default="26")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument("--beam-width", type=int, default=120)
    parser.add_argument("--core-width", type=int)
    parser.add_argument("--qubo-supplement-width", type=int)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--max-qubo-vars", type=int, default=60)
    parser.add_argument("--qubo-time-limit", type=float, default=0.01)
    parser.add_argument("--qubo-reads", type=int, default=4)
    parser.add_argument("--qubo-iterations", type=int, default=80)
    parser.add_argument("--qubo-penalty-a", type=float, default=30.0)
    parser.add_argument("--diversity-weight", type=float, default=0.08)
    parser.add_argument("--qubo-repair-hamming-slack", type=int, default=2)
    parser.add_argument("--qubo-min-depth", type=int, default=10)
    parser.add_argument("--qubo-call-interval", type=int, default=3)
    parser.add_argument("--qubo-plateau-only", action="store_true")
    parser.add_argument("--tail-repair-n", type=int, default=5)
    parser.add_argument("--tail-repair-expanded", type=int, default=10000)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "experiments" / "fast_slack_qubo_beam_all_single_bays.csv",
    )
    parser.add_argument(
        "--moves-dir",
        type=Path,
        default=ROOT / "docs" / "experiments" / "fast_slack_qubo_beam_moves",
    )
    args = parser.parse_args()

    if not args.all:
        instance = _instance_from_single_bay(args.zone, args.bay)
        row, best = _run_one(args, args.zone, args.bay)
        print(f"FINAL {row}", flush=True)
        if args.output_prefix is not None:
            _write_result_csv(row, args.output_prefix.with_suffix(".csv"))
            _write_explain_csv(
                instance,
                best.moves,
                args.output_prefix.with_name(args.output_prefix.name + "_moves.csv"),
            )
        return 0

    rows: list[TrajectoryResult] = []
    start = time.perf_counter()
    cases = discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))
    for idx, (zone, bay) in enumerate(cases, start=1):
        print(f"CASE_START {idx}/{len(cases)} {zone}_{bay}", flush=True)
        instance = _instance_from_single_bay(zone, bay)
        row, best = _run_one(args, zone, bay)
        rows.append(row)
        _write_rows_csv(rows, args.output)
        _write_explain_csv(instance, best.moves, args.moves_dir / f"{instance.instance_id}_moves.csv")
        print(
            f"CASE_DONE {idx}/{len(cases)} {row.case} finalN={row.final_n} "
            f"moves={row.moves} runtime={row.runtime_sec:.3f}s qcalls={row.qubo_calls}",
            flush=True,
        )
    solved = sum(1 for row in rows if row.final_n == 0)
    print(
        f"SUMMARY cases={len(rows)} solved={solved} runtime={sum(row.runtime_sec for row in rows):.3f}s "
        f"wall={time.perf_counter() - start:.3f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
