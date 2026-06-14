"""Paper-style bounded beam search probe adapted to CPMP N=0.

The reference Bounded Beam Search (Bacci, Mattia, Ventura) is defined for the
restricted Block Relocation Problem, where blocks are eventually retrieved from
the yard.  This probe keeps the paper's control skeleton.  The default
successor rule also keeps the paper's restricted branching spirit by moving
only the top box of the first stack that still contributes to N.  A full legal
top-move adapter remains available for comparison because this project cannot
remove boxes from the bay.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_compact_beam_qubo_n0 import (  # noqa: E402
    _backtrack,
    _cached_chunk_meta,
    _cached_pop_chunk,
    _cached_push_chunk,
    _cached_stack_n,
    _code_n,
    _codes_to_moves_instance,
    _successors_with_n_cached,
)
from container_reshuffle.core import Instance, Move  # noqa: E402
from container_reshuffle.features import entropy_limit  # noqa: E402
from run_business_exact_target_search_sqlite import _encode  # noqa: E402


@dataclass(frozen=True)
class PaperBbsCandidate:
    code: bytes
    n_value: int
    depth: int
    parent: bytes
    ub_total: float
    lb_total: int


@dataclass
class PaperBbsResult:
    case: str
    status: str
    moves: list[Move]
    final_n: int
    runtime_sec: float
    expanded: int
    generated: int
    visited: int
    open_count: int
    pruned_bound: int
    pruned_business: int
    dominated: int
    peak_frontier: int
    peak_candidates: int
    depth_reached: int
    beta: int
    incumbent_moves: int | None = None


def make_paper_bbs_args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "alpha": 1.3,
        "beta": 100,
        "greedy_depth": 64,
        "max_moves": None,
        "target_final_n": 0,
        "time_limit": 0.0,
        "branching": "incumbent",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _select_paper_bbs_frontier(
    candidates: list[PaperBbsCandidate],
    beta: int,
    incumbent: float,
) -> tuple[list[PaperBbsCandidate], int]:
    eligible = [candidate for candidate in candidates if candidate.lb_total < incumbent]
    pruned = len(candidates) - len(eligible)
    selected = sorted(
        eligible,
        key=lambda item: (item.ub_total, item.lb_total, item.depth, item.code),
    )[: max(0, beta)]
    return selected, pruned


def _first_dirty_stack(chunks: tuple[bytes, ...], max_height: int) -> int | None:
    for idx, chunk in enumerate(chunks):
        height, _contrib = _cached_chunk_meta(chunk, max_height)
        for level in range(1, height):
            if chunk[level] > chunk[level - 1]:
                return idx
    return None


def _paper_bbs_successors(
    code: bytes,
    current_n: int,
    stack_count: int,
    max_height: int,
    branching: str = "incumbent",
) -> list[tuple[bytes, int, tuple[int, int]]]:
    if branching == "full":
        return [
            (nxt, nxt_n, (-1, -1))
            for nxt, nxt_n in _successors_with_n_cached(code, current_n, stack_count, max_height)
        ]
    if branching != "incumbent":
        raise ValueError(f"unknown paper BBS branching mode: {branching}")

    chunks = tuple(code[i * max_height : (i + 1) * max_height] for i in range(stack_count))
    chunk_meta = [_cached_chunk_meta(chunk, max_height) for chunk in chunks]
    src_idx = _first_dirty_stack(chunks, max_height)
    if src_idx is None:
        return []

    src_h, src_contrib = chunk_meta[src_idx]
    if src_h == 0:
        return []
    src_chunk = chunks[src_idx]
    src_top = src_chunk[src_h - 1]
    src_b = _cached_pop_chunk(src_chunk, src_h, max_height)
    src_after_n = _cached_stack_n(src_b, max_height)

    successors: list[tuple[bytes, int, tuple[int, int]]] = []
    seen: set[bytes] = set()
    for dst_idx, (dst_h, dst_contrib) in enumerate(chunk_meta):
        if src_idx == dst_idx or dst_h >= max_height:
            continue
        dst_b = _cached_push_chunk(chunks[dst_idx], dst_h, src_top, max_height)
        nxt_n = current_n - src_contrib - dst_contrib + src_after_n + _cached_stack_n(dst_b, max_height)
        nxt_chunks = list(chunks)
        nxt_chunks[src_idx] = src_b
        nxt_chunks[dst_idx] = dst_b
        nxt = b"".join(sorted(nxt_chunks))
        if nxt in seen:
            continue
        seen.add(nxt)
        successors.append((nxt, nxt_n, (src_idx, dst_idx)))
    return successors


def _greedy_cpmp_completion(
    code: bytes,
    n_value: int,
    stack_count: int,
    max_height: int,
    max_steps: int,
    target_final_n: int,
) -> list[bytes] | None:
    path = [code]
    current = code
    current_n = n_value
    seen = {code}
    for _step in range(max(0, max_steps)):
        if current_n <= target_final_n:
            return path
        choices = [
            (nxt_n, nxt)
            for nxt, nxt_n in _successors_with_n_cached(current, current_n, stack_count, max_height)
            if nxt not in seen
        ]
        if not choices:
            return None
        current_n, current = min(choices, key=lambda item: (item[0], item[1]))
        seen.add(current)
        path.append(current)
    return path if current_n <= target_final_n else None


def _best_seen_n(frontier: list[tuple[bytes, int, int]], initial_n: int) -> int:
    if not frontier:
        return initial_n
    return min(n_value for _code, n_value, _depth in frontier)


def run_paper_bbs_cpmp_instance(case: str, instance: Instance, args: argparse.Namespace) -> PaperBbsResult:
    start = time.perf_counter()
    deadline = start + args.time_limit if args.time_limit > 0 else math.inf
    initial, n0 = _encode(instance.stacks, instance.stack_count, instance.max_height)
    cap = entropy_limit(instance, args.alpha)
    depth_cap = min(cap, args.max_moves) if args.max_moves is not None else cap
    beta = max(1, int(args.beta))
    target_final_n = int(args.target_final_n)

    parent: dict[bytes, bytes | None] = {initial: None}
    best_depth: dict[bytes, int] = {initial: 0}
    frontier: list[tuple[bytes, int, int]] = [(initial, n0, 0)]
    best_codes: list[bytes] | None = [initial] if n0 <= target_final_n else None
    incumbent = 0 if best_codes is not None else math.inf

    greedy_initial = _greedy_cpmp_completion(
        initial,
        n0,
        instance.stack_count,
        instance.max_height,
        min(int(args.greedy_depth), depth_cap),
        target_final_n,
    )
    if greedy_initial is not None and _code_n(greedy_initial[-1], instance.stack_count, instance.max_height) <= target_final_n:
        incumbent = len(greedy_initial) - 1
        best_codes = greedy_initial

    expanded = generated = pruned_bound = pruned_business = dominated = 0
    peak_frontier = len(frontier)
    peak_candidates = 0
    depth_reached = 0
    best_final_n = n0

    while frontier and time.perf_counter() < deadline:
        candidates: list[PaperBbsCandidate] = []
        for code, current_n, depth in frontier:
            if time.perf_counter() >= deadline:
                break
            depth_reached = max(depth_reached, depth)
            best_final_n = min(best_final_n, current_n)
            if depth >= depth_cap:
                pruned_bound += 1
                continue
            if depth + current_n >= incumbent:
                pruned_bound += 1
                continue
            expanded += 1
            next_depth = depth + 1
            for nxt, nxt_n, _move in _paper_bbs_successors(
                code,
                current_n,
                instance.stack_count,
                instance.max_height,
                getattr(args, "branching", "incumbent"),
            ):
                generated += 1
                if next_depth + nxt_n > cap:
                    pruned_business += 1
                    continue
                old_depth = best_depth.get(nxt)
                if old_depth is not None and next_depth >= old_depth:
                    dominated += 1
                    continue
                lb_total = next_depth + nxt_n
                if lb_total >= incumbent:
                    pruned_bound += 1
                    continue

                ub_total = math.inf
                greedy_path: list[bytes] | None = [nxt] if nxt_n <= target_final_n else None
                if greedy_path is None:
                    greedy_path = _greedy_cpmp_completion(
                        nxt,
                        nxt_n,
                        instance.stack_count,
                        instance.max_height,
                        min(int(args.greedy_depth), depth_cap - next_depth),
                        target_final_n,
                    )
                if greedy_path is not None and _code_n(greedy_path[-1], instance.stack_count, instance.max_height) <= target_final_n:
                    ub_total = next_depth + len(greedy_path) - 1
                    if ub_total < incumbent:
                        incumbent = ub_total
                        best_codes = _backtrack(parent, code) + greedy_path

                candidates.append(
                    PaperBbsCandidate(
                        code=nxt,
                        n_value=nxt_n,
                        depth=next_depth,
                        parent=code,
                        ub_total=ub_total,
                        lb_total=lb_total,
                    )
                )

        peak_candidates = max(peak_candidates, len(candidates))
        if not candidates:
            break

        selected, pruned = _select_paper_bbs_frontier(candidates, beta, incumbent)
        pruned_bound += pruned
        frontier = []
        for candidate in selected:
            parent[candidate.code] = candidate.parent
            best_depth[candidate.code] = candidate.depth
            frontier.append((candidate.code, candidate.n_value, candidate.depth))
            best_final_n = min(best_final_n, candidate.n_value)
        peak_frontier = max(peak_frontier, len(frontier))

    solved = best_codes is not None and _code_n(best_codes[-1], instance.stack_count, instance.max_height) <= target_final_n
    moves = _codes_to_moves_instance(instance, best_codes) if solved and best_codes is not None else []
    final_n = 0 if solved else _best_seen_n(frontier, best_final_n)
    status = "FEASIBLE" if solved else "TIME_LIMIT_OR_BEAM_EXHAUSTED"
    return PaperBbsResult(
        case=case,
        status=(
            f"{status}; solver=paper_bounded_beam_cpmp_probe; beta={beta}; "
            f"paper_control=UB_LB_beta; branching={getattr(args, 'branching', 'incumbent')}; "
            "cpmp_adapter=legal_top_moves_no_retrieval"
        ),
        moves=moves,
        final_n=final_n,
        runtime_sec=time.perf_counter() - start,
        expanded=expanded,
        generated=generated,
        visited=len(best_depth),
        open_count=len(frontier),
        pruned_bound=pruned_bound,
        pruned_business=pruned_business,
        dominated=dominated,
        peak_frontier=peak_frontier,
        peak_candidates=peak_candidates,
        depth_reached=depth_reached,
        beta=beta,
        incumbent_moves=len(moves) if solved else None,
    )


def _row_from_result(result: PaperBbsResult, family: str = "", reference_moves: int | None = None) -> dict[str, object]:
    moves = len(result.moves) if result.moves else ""
    return {
        "case": result.case,
        "family": family,
        "solver": "paper_bounded_beam_cpmp_probe",
        "beta": result.beta,
        "moves": moves,
        "final_n": result.final_n,
        "reference_moves": reference_moves if reference_moves is not None else "",
        "move_gap": (int(moves) - reference_moves) if moves != "" and reference_moves is not None else "",
        "runtime_sec": result.runtime_sec,
        "expanded_states": result.expanded,
        "generated_states": result.generated,
        "visited_states": result.visited,
        "open_count": result.open_count,
        "pruned_bound": result.pruned_bound,
        "pruned_business": result.pruned_business,
        "dominated": result.dominated,
        "peak_frontier": result.peak_frontier,
        "peak_candidates": result.peak_candidates,
        "depth_reached": result.depth_reached,
        "status": result.status,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "external" / "DLTS")
    parser.add_argument("--families", nargs="+", default=["7x5_cv1_test"])
    parser.add_argument("--limit-per-family", type=int, default=10)
    parser.add_argument("--beta", type=int, default=100)
    parser.add_argument("--greedy-depth", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--time-limit", type=float, default=3.0)
    parser.add_argument("--branching", choices=["incumbent", "full"], default="incumbent")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "paper_bbs_cpmp_probe.csv")
    return parser


def main() -> int:
    from open_cpmp_benchmarks import discover_dlts_cases, load_dlts_dat

    parser = build_parser()
    cli = parser.parse_args()
    source = cli.source if cli.source.is_absolute() else ROOT / cli.source
    output = cli.output if cli.output.is_absolute() else ROOT / cli.output
    cases = discover_dlts_cases(source, families=cli.families, limit_per_family=cli.limit_per_family)
    args = make_paper_bbs_args(
        alpha=cli.alpha,
        beta=cli.beta,
        greedy_depth=cli.greedy_depth,
        max_moves=cli.max_moves,
        time_limit=cli.time_limit,
        branching=cli.branching,
    )
    rows: list[dict[str, object]] = []
    for idx, case in enumerate(cases, start=1):
        instance = load_dlts_dat(case.instance_path, max_height=case.reference.max_height)
        print(f"CASE_START {idx}/{len(cases)} paper_bbs {case.case_id}", flush=True)
        result = run_paper_bbs_cpmp_instance(case.case_id, instance, args)
        rows.append(_row_from_result(result, family=case.family, reference_moves=case.reference.optimal_moves))
        print(
            f"CASE_DONE paper_bbs {case.case_id} final_n={result.final_n} moves={len(result.moves) if result.moves else None} "
            f"runtime={result.runtime_sec:.3f}s",
            flush=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else list(_row_from_result(PaperBbsResult("", "", [], 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)).keys())
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
