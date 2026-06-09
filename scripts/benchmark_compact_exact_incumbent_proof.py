"""Compact exact A* proof search with incumbent upper bounds.

This proof engine is intended for large single-bay CPMP cases where open-list
``ExactAStarSolver`` spends most memory storing full ``State`` objects and move
histories.  It stores only stack tuples in the frontier and uses a validated
incumbent N=0 trajectory as an upper bound.  When the open list is exhausted
below that bound, the incumbent move count is proven optimal.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import sys
import time
from math import floor, inf
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_explainable_rule_macro_beam_all_single_bays import discover_single_bay_cases  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402


FIELDS = [
    "case",
    "n0",
    "cap_13",
    "incumbent_moves",
    "status",
    "proven",
    "best_moves",
    "runtime_sec",
    "expanded",
    "generated",
    "visited",
    "open",
    "pruned_bound",
    "pruned_business",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _stack_label_to_index(label: str) -> int:
    value = str(label).strip().upper()
    if value == "A":
        return 9
    return int(value) - 1


def _read_move_pairs(path: Path) -> list[tuple[int, int, int]]:
    moves: list[tuple[int, int, int]] = []
    for row in _read_rows(path):
        if not row.get("box_rank"):
            continue
        moves.append(
            (
                int(float(row["box_rank"])),
                _stack_label_to_index(row["from_stack"]),
                _stack_label_to_index(row["to_stack"]),
            )
        )
    return moves


def _normalize_case(case: str) -> str:
    return case if case.endswith("_single") else f"{case}_single"


def _case_filter(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return [_normalize_case(case) for case in args.cases]
    return [
        f"{zone}_{bay}_single"
        for zone, bay in discover_single_bay_cases(ROOT / "data" / "input", set(args.bays))
    ]


def _incumbent_moves_by_case(path: Path | None, cases: Iterable[str]) -> dict[str, list[tuple[int, int, int]]]:
    if path is None:
        return {}
    ref = path if path.is_absolute() else ROOT / path
    wanted = set(cases)
    out: dict[str, list[tuple[int, int, int]]] = {}
    for row in _read_rows(ref):
        case = row.get("case", "")
        if case not in wanted:
            continue
        if row.get("goal_validation_ok") != "True" or row.get("final_n") not in {"0", "0.0"}:
            continue
        moves_path_text = row.get("moves_path", "")
        if not moves_path_text:
            continue
        moves_path = Path(moves_path_text)
        if not moves_path.is_absolute():
            moves_path = ROOT / moves_path
        if moves_path.exists():
            out[case] = _read_move_pairs(moves_path)
    return out


def _stacks_key(stacks: tuple[tuple[int, ...], ...], canonical: bool) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(stacks)) if canonical else stacks


def _future_disorder(stacks: tuple[tuple[int, ...], ...]) -> int:
    total = 0
    for stack in stacks:
        for idx in range(1, len(stack)):
            if stack[idx] > stack[idx - 1]:
                total += len(stack) - idx
                break
    return total


def _is_goal(stacks: tuple[tuple[int, ...], ...]) -> bool:
    return all(all(stack[i] > stack[i + 1] for i in range(len(stack) - 1)) for stack in stacks)


def _apply_move(stacks: tuple[tuple[int, ...], ...], box: int, src: int, dst: int, height: int) -> tuple[tuple[int, ...], ...]:
    if src == dst or src < 0 or dst < 0 or src >= len(stacks) or dst >= len(stacks):
        raise ValueError("invalid stack index")
    if not stacks[src] or stacks[src][-1] != box or len(stacks[dst]) >= height:
        raise ValueError("invalid move")
    next_stacks = [list(stack) for stack in stacks]
    next_stacks[dst].append(next_stacks[src].pop())
    return tuple(tuple(stack) for stack in next_stacks)


def _validate_incumbent(
    initial: tuple[tuple[int, ...], ...],
    moves: list[tuple[int, int, int]],
    *,
    height: int,
    cap: int,
) -> int | None:
    stacks = initial
    for depth, (box, src, dst) in enumerate(moves, start=1):
        try:
            stacks = _apply_move(stacks, box, src, dst, height)
        except ValueError:
            return None
        if depth + _future_disorder(stacks) > cap:
            return None
    return len(moves) if _is_goal(stacks) else None


def _successors(
    stacks: tuple[tuple[int, ...], ...],
    *,
    height: int,
):
    for src, stack in enumerate(stacks):
        if not stack:
            continue
        box = stack[-1]
        for dst, target in enumerate(stacks):
            if src == dst or len(target) >= height:
                continue
            next_stacks = [list(item) for item in stacks]
            next_stacks[dst].append(next_stacks[src].pop())
            yield tuple(tuple(item) for item in next_stacks)


def _prove_case(case: str, args: argparse.Namespace, incumbent_moves: list[tuple[int, int, int]] | None) -> dict[str, object]:
    zone, bay, *_ = case.split("_")
    instance = _instance_from_single_bay(zone, bay)
    initial = tuple(tuple(stack) for stack in instance.stacks)
    n0 = _future_disorder(initial)
    cap = int(floor(min(args.alpha, 2.0) * n0 + 1e-9))
    start = time.perf_counter()
    deadline = start + args.time_limit

    initial_incumbent = None
    if incumbent_moves:
        initial_incumbent = _validate_incumbent(initial, incumbent_moves, height=instance.max_height, cap=cap)

    if _is_goal(initial):
        initial_incumbent = 0
    incumbent = inf if initial_incumbent is None else initial_incumbent

    h0 = n0
    heap: list[tuple[int, int, int, int, tuple[tuple[int, ...], ...]]] = []
    best_g = {_stacks_key(initial, args.canonical): 0}
    counter = 0
    if h0 < incumbent:
        heapq.heappush(heap, (h0, h0, 0, counter, initial))

    expanded = generated = pruned_bound = pruned_business = 0
    status = "NO_FEASIBLE_INCUMBENT" if incumbent == inf else "INCUMBENT_FEASIBLE_NOT_PROVEN"
    proven = False

    while heap:
        if time.perf_counter() > deadline:
            status = "TIME_LIMIT_" + ("FEASIBLE_NOT_PROVEN" if incumbent < inf else "NO_FEASIBLE_FOUND")
            break
        if args.max_visited_states and len(best_g) >= args.max_visited_states:
            status = "STATE_LIMIT_" + ("FEASIBLE_NOT_PROVEN" if incumbent < inf else "NO_FEASIBLE_FOUND")
            break

        f_value, _h, g_value, _idx, stacks = heapq.heappop(heap)
        if f_value >= incumbent:
            status = "OPTIMAL_SHORTER_PROVEN" if initial_incumbent is not None and incumbent < initial_incumbent else "OPTIMAL_INCUMBENT_PROVEN"
            proven = True
            break
        key = _stacks_key(stacks, args.canonical)
        if g_value != best_g.get(key):
            continue
        if _is_goal(stacks):
            incumbent = g_value
            status = "SHORTER_GOAL_FOUND"
            continue

        expanded += 1
        for nxt in _successors(stacks, height=instance.max_height):
            next_g = g_value + 1
            next_n = _future_disorder(nxt)
            generated += 1
            if next_g + next_n > cap:
                pruned_business += 1
                continue
            if next_g + next_n >= incumbent:
                pruned_bound += 1
                continue
            next_key = _stacks_key(nxt, args.canonical)
            if next_g >= best_g.get(next_key, inf):
                continue
            best_g[next_key] = next_g
            counter += 1
            heapq.heappush(heap, (next_g + next_n, next_n, next_g, counter, nxt))
    else:
        if incumbent < inf:
            status = "OPTIMAL_SHORTER_PROVEN" if initial_incumbent is not None and incumbent < initial_incumbent else "OPTIMAL_INCUMBENT_PROVEN"
            proven = True
        else:
            status = "NO_FEASIBLE_FOUND_EXHAUSTIVE"

    return {
        "case": case,
        "n0": n0,
        "cap_13": cap,
        "incumbent_moves": "" if initial_incumbent is None else int(initial_incumbent),
        "status": status,
        "proven": str(proven),
        "best_moves": "" if incumbent == inf else int(incumbent),
        "runtime_sec": f"{time.perf_counter() - start:.6f}",
        "expanded": expanded,
        "generated": generated,
        "visited": len(best_g),
        "open": len(heap),
        "pruned_bound": pruned_bound,
        "pruned_business": pruned_business,
    }


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--bays", nargs="+", default=["01", "26"])
    parser.add_argument("--incumbent-acceptance", type=Path)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--max-visited-states", type=int, default=1_000_000)
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "experiments" / "compact_exact_incumbent_proof.csv")
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    cases = _case_filter(args)
    incumbents = _incumbent_moves_by_case(args.incumbent_acceptance, cases)
    rows: list[dict[str, object]] = []
    for idx, case in enumerate(cases, start=1):
        print(f"CASE_START {idx}/{len(cases)} case={case} incumbent_moves={len(incumbents.get(case, []))}", flush=True)
        row = _prove_case(case, args, incumbents.get(case))
        rows.append(row)
        print(
            "  COMPACT_PROOF "
            f"status={row['status']} proven={row['proven']} best_moves={row['best_moves']} "
            f"runtime={float(row['runtime_sec']):.3f}s visited={row['visited']} open={row['open']}",
            flush=True,
        )
        _write_rows(rows, output)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
