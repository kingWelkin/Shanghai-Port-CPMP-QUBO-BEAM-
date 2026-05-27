"""Independently audit legality of selected focused-route move sequences."""

from __future__ import annotations

import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (ROOT, SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_full_real_business_algorithms import _instance_from_zone, discover_paired_cases  # noqa: E402
from benchmark_full_real_enhanced_beam import (  # noqa: E402
    _select_enhanced_beam,
    _select_state_expansions,
    _state_rank as enhanced_state_rank,
)
from benchmark_full_real_qubo_standalone_tuning import (  # noqa: E402
    _base_config as qbo_base_config,
    _dense_config as qbo_dense_config,
    _low_div_config as qbo_low_div_config,
    _wide_config as qbo_wide_config,
)
from container_reshuffle.core import (  # noqa: E402
    Instance,
    State,
    apply_move,
    create_initial_state,
    future_disorder_count_stacks,
    generate_legal_moves,
    hash_state,
)
from container_reshuffle.features import disorder_box_count, entropy_limit  # noqa: E402
from container_reshuffle.qubo.beam_selection import QuboBeamSelectionBuilder  # noqa: E402
from container_reshuffle.qubo.sampler import QuboSamplerConfig, SimulatedAnnealingQuboSampler  # noqa: E402
from container_reshuffle.scoring import cpmp_business_feasible, dominance_value, score_state  # noqa: E402
from container_reshuffle.solvers.base import SolverConfig  # noqa: E402
from container_reshuffle.solvers.beam import Candidate  # noqa: E402


DIRECT_QBO_CONFIGS = {
    "qubo_beam": qbo_base_config,
    "qbo_beam_base_rerun": qbo_base_config,
    "qbo_beam_dense": qbo_dense_config,
    "qbo_beam_low_div": qbo_low_div_config,
    "qbo_beam_wide": qbo_wide_config,
}

FOCUSED_SOLVERS = [
    "beam_focused_with_qubo_portfolio",
    "beam_focused_with_tuned_qubo",
    "beam_focused_with_dense_qubo",
    "beam_focused_with_qubo",
]


@dataclass
class AuditRow:
    """One legality audit row."""

    case: str
    route: str
    selected_solver: str
    legal: bool
    moves: int
    final_n: int
    max_g: int
    cap_13: int
    checked_moves: int
    issue: str
    runtime_sec: float


def _plain_config() -> SolverConfig:
    return SolverConfig(
        max_depth=60,
        beam_width=10,
        max_candidates_per_layer=100,
        candidate_limit_per_state=3,
        time_limit_sec=10.0,
        entropy_alpha=1.3,
        random_seed=37,
    )


def _enhanced_config() -> SolverConfig:
    return SolverConfig(
        max_depth=60,
        beam_width=14,
        max_candidates_per_layer=180,
        time_limit_sec=10.0,
        entropy_alpha=1.3,
        random_seed=37,
    )


def _state_rank(instance: Instance, state: State) -> tuple[int, int, int, int]:
    return (
        disorder_box_count(instance, state),
        state.disorder_total,
        state.entropy_max,
        state.move_count,
    )


def _run_plain_state(instance: Instance) -> State:
    start = time.perf_counter()
    config = _plain_config()
    initial = create_initial_state(instance)
    best = initial
    h0 = hash_state(instance, initial, config.enable_canonical_hash)
    beam = [Candidate(initial, score_state(instance, initial, config.score_weights), None, h0)]
    visited = {h0: dominance_value(instance, initial)}
    for _depth in range(config.max_depth):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        candidates: list[Candidate] = []
        for cand in beam:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            scored_moves = []
            for mv in generate_legal_moves(instance, cand.state):
                try:
                    nxt = apply_move(instance, cand.state, mv)
                except ValueError:
                    continue
                if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                    continue
                scored_moves.append((score_state(instance, nxt, config.score_weights), mv, nxt))
            scored_moves.sort(key=lambda item: item[0])
            scored_moves = scored_moves[: config.candidate_limit_per_state or len(scored_moves)]
            for score, mv, nxt in scored_moves:
                h = hash_state(instance, nxt, config.enable_canonical_hash)
                dom = dominance_value(instance, nxt)
                if config.enable_visited_pruning and visited.get(h, float("inf")) <= dom:
                    continue
                visited[h] = dom
                if _state_rank(instance, nxt) < _state_rank(instance, best):
                    best = nxt
                candidates.append(Candidate(nxt, score, mv, h))
        if not candidates or disorder_box_count(instance, best) == 0:
            break
        candidates.sort(key=lambda cand: (cand.score, cand.state.move_count))
        beam = candidates[: config.max_candidates_per_layer][: config.beam_width]
    return best


def _run_enhanced_state(instance: Instance) -> State:
    start = time.perf_counter()
    config = _enhanced_config()
    initial = create_initial_state(instance)
    best = initial
    h0 = hash_state(instance, initial, config.enable_canonical_hash)
    beam = [Candidate(initial, score_state(instance, initial, config.score_weights), None, h0)]
    visited = {h0: dominance_value(instance, initial)}
    for _depth in range(config.max_depth):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        candidates: list[Candidate] = []
        for cand in beam:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            expansions, _eval_count = _select_state_expansions(
                instance,
                cand.state,
                config,
                per_rule_limit=2,
            )
            for score, mv, nxt in expansions:
                h = hash_state(instance, nxt, config.enable_canonical_hash)
                dom = dominance_value(instance, nxt)
                if config.enable_visited_pruning and visited.get(h, float("inf")) <= dom:
                    continue
                visited[h] = dom
                if enhanced_state_rank(instance, nxt) < enhanced_state_rank(instance, best):
                    best = nxt
                candidates.append(Candidate(nxt, score, mv, h))
        if not candidates or disorder_box_count(instance, best) == 0:
            break
        beam = _select_enhanced_beam(candidates, instance, config, diversity_weight=0.06, elite_ratio=0.60)
    return best


def _run_qbo_state(instance: Instance, solver_name: str) -> State:
    start = time.perf_counter()
    config = DIRECT_QBO_CONFIGS[solver_name]()
    initial = create_initial_state(instance)
    best = initial
    h0 = hash_state(instance, initial, config.enable_canonical_hash)
    beam = [Candidate(initial, score_state(instance, initial, config.score_weights), None, h0)]
    visited = {h0: dominance_value(instance, initial)}
    builder = QuboBeamSelectionBuilder()
    sampler = SimulatedAnnealingQuboSampler()
    call_interval = max(1, config.qubo_call_interval)
    min_candidates = max(0, config.qubo_min_candidates_for_call)
    q_time_limit = max(0.001, config.qubo_time_limit_sec or min(1.0, config.time_limit_sec))
    qubo_calls = 0

    for depth in range(config.max_depth):
        if time.perf_counter() - start > config.time_limit_sec:
            break
        candidates: list[Candidate] = []
        for cand in beam:
            if time.perf_counter() - start > config.time_limit_sec:
                break
            scored_moves = []
            for mv in generate_legal_moves(instance, cand.state):
                try:
                    nxt = apply_move(instance, cand.state, mv)
                except ValueError:
                    continue
                if not cpmp_business_feasible(instance, nxt, config.entropy_alpha):
                    continue
                scored_moves.append((score_state(instance, nxt, config.score_weights), mv, nxt))
            scored_moves.sort(key=lambda item: item[0])
            scored_moves = scored_moves[: config.candidate_limit_per_state or len(scored_moves)]
            for score, mv, nxt in scored_moves:
                h = hash_state(instance, nxt, config.enable_canonical_hash)
                dom = dominance_value(instance, nxt)
                if config.enable_visited_pruning and visited.get(h, float("inf")) <= dom:
                    continue
                visited[h] = dom
                if _state_rank(instance, nxt) < _state_rank(instance, best):
                    best = nxt
                candidates.append(Candidate(nxt, score, mv, h))
        if not candidates or disorder_box_count(instance, best) == 0:
            break
        candidates.sort(key=lambda cand: (cand.score, cand.state.move_count))
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
            sample_set = sampler.sample(
                model,
                QuboSamplerConfig(
                    num_reads=config.qubo_num_reads,
                    random_seed=config.random_seed + qubo_calls,
                    time_limit_sec=q_time_limit,
                    sa_iterations=config.qubo_sa_iterations,
                ),
            )
            elite_keep_count = max(1, int(round(config.beam_width * 0.8)))
            beam, _repair = builder.decode_samples_and_repair(
                sample_set.samples,
                usable,
                config.beam_width,
                elite_keep_count=elite_keep_count,
                diversity_weight=config.diversity_weight,
                repair_hamming_slack=config.qubo_repair_hamming_slack,
            )
        else:
            beam = candidates[: config.beam_width]
    return best


def _run_direct_state(instance: Instance, solver_name: str) -> State:
    if solver_name == "beam":
        return _run_plain_state(instance)
    if solver_name == "enhanced_beam_fast":
        return _run_enhanced_state(instance)
    if solver_name in DIRECT_QBO_CONFIGS:
        return _run_qbo_state(instance, solver_name)
    raise ValueError(f"cannot audit selected solver: {solver_name}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _selected_from_status(status: str, marker: str) -> str:
    match = re.search(rf"{marker}=([^;]+)", status)
    if not match:
        raise ValueError(f"cannot parse {marker} from status: {status}")
    return match.group(1)


def _direct_solver_for(case: str, route: str, focused_rows: list[dict[str, str]], qbo_rows: list[dict[str, str]]) -> str:
    focused = next(row for row in focused_rows if row["case"] == case and row["solver"] == route)
    selected = _selected_from_status(focused["status"], "FOCUSED_PORTFOLIO_SELECTED")
    if selected != "qbo_beam_standalone_portfolio":
        return selected
    qbo_portfolio = next(row for row in qbo_rows if row["case"] == case and row["solver"] == selected)
    return _selected_from_status(qbo_portfolio["status"], "QBO_STANDALONE_PORTFOLIO_SELECTED")


def _manual_audit(instance: Instance, state: State) -> tuple[bool, str, int, int, int]:
    stacks = [list(stack) for stack in instance.stacks]
    expected = sorted(box for stack in stacks for box in stack)
    cap = entropy_limit(instance, 1.3)
    max_g = future_disorder_count_stacks(stacks)
    checked = 0
    for idx, move in enumerate(state.moves):
        if move.step != idx:
            return False, f"step mismatch at {idx}: move.step={move.step}", checked, future_disorder_count_stacks(stacks), max_g
        if move.from_stack == move.to_stack:
            return False, f"same source/destination at step {idx}", checked, future_disorder_count_stacks(stacks), max_g
        if not (0 <= move.from_stack < instance.stack_count):
            return False, f"source out of range at step {idx}", checked, future_disorder_count_stacks(stacks), max_g
        if not (0 <= move.to_stack < instance.stack_count):
            return False, f"destination out of range at step {idx}", checked, future_disorder_count_stacks(stacks), max_g
        src = stacks[move.from_stack]
        dst = stacks[move.to_stack]
        if not src:
            return False, f"empty source at step {idx}", checked, future_disorder_count_stacks(stacks), max_g
        if len(dst) >= instance.max_height:
            return False, f"full destination at step {idx}", checked, future_disorder_count_stacks(stacks), max_g
        if src[-1] != move.box:
            return False, f"box {move.box} is not source top {src[-1]} at step {idx}", checked, future_disorder_count_stacks(stacks), max_g
        if move.from_level != len(src):
            return False, f"from_level mismatch at step {idx}", checked, future_disorder_count_stacks(stacks), max_g
        if move.to_level != len(dst) + 1:
            return False, f"to_level mismatch at step {idx}", checked, future_disorder_count_stacks(stacks), max_g
        dst.append(src.pop())
        checked += 1
        actual = sorted(box for stack in stacks for box in stack)
        if actual != expected:
            return False, f"container multiset changed at step {idx}", checked, future_disorder_count_stacks(stacks), max_g
        if any(len(stack) > instance.max_height for stack in stacks):
            return False, f"capacity violation after step {idx}", checked, future_disorder_count_stacks(stacks), max_g
        n_value = future_disorder_count_stacks(stacks)
        g_value = checked + n_value
        max_g = max(max_g, g_value)
        if g_value > cap:
            return False, f"G cap violation after step {idx}: {g_value}>{cap}", checked, n_value, max_g
    if stacks != state.stacks:
        return False, "final stacks do not match replayed moves", checked, future_disorder_count_stacks(stacks), max_g
    return True, "OK", checked, future_disorder_count_stacks(stacks), max_g


def main() -> None:
    focused_rows = _read_rows(ROOT / "docs" / "experiments" / "full_real_business_focused_beam_comparison.csv")
    qbo_rows = _read_rows(ROOT / "docs" / "experiments" / "full_real_business_qubo_standalone_tuning.csv")
    cases = [zone for zone, _suffixes, _paths in discover_paired_cases(ROOT / "data" / "input", ["01", "26"])]
    cache: dict[tuple[str, str], State] = {}
    audit_rows: list[AuditRow] = []
    for zone in cases:
        instance = _instance_from_zone(zone)
        case = instance.instance_id
        print(f"auditing {case}", flush=True)
        for route in FOCUSED_SOLVERS:
            selected = _direct_solver_for(case, route, focused_rows, qbo_rows)
            key = (case, selected)
            start = time.perf_counter()
            if key not in cache:
                print(f"  rerun {selected}", flush=True)
                cache[key] = _run_direct_state(instance, selected)
            state = cache[key]
            legal, issue, checked, final_n, max_g = _manual_audit(instance, state)
            audit_rows.append(
                AuditRow(
                    case=case,
                    route=route,
                    selected_solver=selected,
                    legal=legal,
                    moves=state.move_count,
                    final_n=final_n,
                    max_g=max_g,
                    cap_13=entropy_limit(instance, 1.3),
                    checked_moves=checked,
                    issue=issue,
                    runtime_sec=time.perf_counter() - start,
                )
            )

    csv_path = ROOT / "docs" / "experiments" / "full_real_business_focused_legality_audit.csv"
    md_path = ROOT / "docs" / "experiments" / "full_real_business_focused_legality_audit.md"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(AuditRow.__annotations__.keys()))
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(row.__dict__)

    total = len(audit_rows)
    passed = sum(row.legal for row in audit_rows)
    by_route = {}
    for row in audit_rows:
        by_route.setdefault(row.route, []).append(row)
    lines = [
        "# Full Real Focused Route Legality Audit",
        "",
        "This audit reruns the selected direct component for each focused route and independently replays every move from the original stacks.",
        "",
        f"CSV: `{csv_path.relative_to(ROOT)}`",
        "",
        "## Summary",
        "",
        f"- Checked rows: `{total}`",
        f"- Passed rows: `{passed}`",
        f"- Failed rows: `{total - passed}`",
        "",
        "| route | rows | passed | avg final N | avg moves | max observed G |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route, rows in by_route.items():
        lines.append(
            f"| {route} | {len(rows)} | {sum(r.legal for r in rows)} | "
            f"{sum(r.final_n for r in rows) / len(rows):.2f} | {sum(r.moves for r in rows) / len(rows):.2f} | "
            f"{max(r.max_g for r in rows)} |"
        )
    lines.extend(["", "## Checks", ""])
    lines.extend(
        [
            "- source and destination stack indices are in range",
            "- source and destination differ",
            "- source stack is non-empty",
            "- moved box equals the current source top",
            "- destination stack is not full",
            "- `from_level` and `to_level` match replayed stack heights",
            "- container multiset is unchanged after every move",
            "- every intermediate `G = moves + N(Future)` satisfies `G <= floor(1.3*N0)`",
            "- final replayed stacks equal the solver state",
        ]
    )
    bad = [row for row in audit_rows if not row.legal]
    if bad:
        lines.extend(["", "## Failures", ""])
        for row in bad:
            lines.append(f"- `{row.case}` / `{row.route}` / `{row.selected_solver}`: {row.issue}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
