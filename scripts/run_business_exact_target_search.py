"""Exact target search for single-bay business final N(Future)."""

from __future__ import annotations

import argparse
import heapq
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from container_reshuffle.features import entropy_limit  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402

Code = bytes


def _encode(stacks, stack_count: int, max_height: int) -> Code:
    data = bytearray(stack_count * max_height)
    for i, stack in enumerate(stacks):
        base = i * max_height
        for h, box in enumerate(stack):
            data[base + h] = int(box)
    return _canonical(bytes(data), stack_count, max_height)


def _canonical(code: Code, stack_count: int, max_height: int) -> Code:
    chunks = [code[i * max_height : (i + 1) * max_height] for i in range(stack_count)]
    return b"".join(sorted(chunks))


def _height(code: Code, stack: int, max_height: int) -> int:
    base = stack * max_height
    for h in range(max_height):
        if code[base + h] == 0:
            return h
    return max_height


def _future_n(code: Code, stack_count: int, max_height: int) -> int:
    total = 0
    for i in range(stack_count):
        base = i * max_height
        height = _height(code, i, max_height)
        for h in range(1, height):
            if code[base + h] > code[base + h - 1]:
                total += height - h
                break
    return total


def _apply(code: Code, from_stack: int, to_stack: int, stack_count: int, max_height: int) -> Code:
    src_h = _height(code, from_stack, max_height)
    dst_h = _height(code, to_stack, max_height)
    src_pos = from_stack * max_height + src_h - 1
    dst_pos = to_stack * max_height + dst_h
    data = bytearray(code)
    box = data[src_pos]
    data[src_pos] = 0
    data[dst_pos] = box
    return _canonical(bytes(data), stack_count, max_height)


def _successors(code: Code, stack_count: int, max_height: int):
    heights = [_height(code, i, max_height) for i in range(stack_count)]
    for i, src_h in enumerate(heights):
        if src_h == 0:
            continue
        for j, dst_h in enumerate(heights):
            if i == j or dst_h >= max_height:
                continue
            yield _apply(code, i, j, stack_count, max_height)


def _remaining_lower_bound(current_n: int, move_count: int, cap: int) -> int:
    return max(0, current_n - (cap - move_count))


def run(zone: str, bay: str, target_final_n: int, alpha: float, time_limit: float, progress_interval: float) -> int:
    instance = _instance_from_single_bay(zone, bay)
    initial = _encode(instance.stacks, instance.stack_count, instance.max_height)
    cap = entropy_limit(instance, alpha)
    n0 = _future_n(initial, instance.stack_count, instance.max_height)
    start = time.perf_counter()
    next_progress = start
    counter = 0
    expanded = generated = pruned_bound = pruned_business = dominated = 0

    heap: list[tuple[int, int, int, int, Code]] = []
    visited = {initial: 0}
    heapq.heappush(heap, (n0, 0, n0, counter, initial))
    print(
        f"START case={instance.instance_id} target_final_N={target_final_n} "
        f"stacks={instance.stack_count} H={instance.max_height} containers={instance.container_count} "
        f"initial_N={n0} cap={cap}",
        flush=True,
    )

    while heap:
        now = time.perf_counter()
        if now - start > time_limit:
            print(
                "FINAL "
                f"status=TIME_LIMIT_NO_PROOF target_final_N={target_final_n} expanded={expanded} "
                f"generated={generated} visited={len(visited)} open={len(heap)} "
                f"pruned_bound={pruned_bound} pruned_business={pruned_business} dominated={dominated} "
                f"runtime={now - start:.3f}s",
                flush=True,
            )
            return 1
        if now >= next_progress:
            print(
                "PROGRESS "
                f"elapsed={now - start:.1f}s expanded={expanded} generated={generated} "
                f"visited={len(visited)} open={len(heap)} pruned_bound={pruned_bound} "
                f"pruned_business={pruned_business} dominated={dominated}",
                flush=True,
            )
            next_progress = now + progress_interval

        _lb, move_count, current_n, _idx, stacks = heapq.heappop(heap)
        if move_count != visited.get(stacks):
            dominated += 1
            continue
        if current_n <= target_final_n:
            print(
                "FINAL "
                f"status=TARGET_REACHABLE target_final_N={target_final_n} moves={move_count} "
                f"final_N={current_n} expanded={expanded} generated={generated} visited={len(visited)} "
                f"runtime={time.perf_counter() - start:.3f}s",
                flush=True,
            )
            return 0
        if _remaining_lower_bound(current_n, move_count, cap) > target_final_n:
            pruned_bound += 1
            continue
        if move_count >= cap:
            pruned_bound += 1
            continue

        expanded += 1
        next_move_count = move_count + 1
        for nxt in _successors(stacks, instance.stack_count, instance.max_height):
            generated += 1
            nxt_n = _future_n(nxt, instance.stack_count, instance.max_height)
            if next_move_count + nxt_n > cap:
                pruned_business += 1
                continue
            if _remaining_lower_bound(nxt_n, next_move_count, cap) > target_final_n:
                pruned_bound += 1
                continue
            if visited.get(nxt, cap + 1) <= next_move_count:
                dominated += 1
                continue
            visited[nxt] = next_move_count
            counter += 1
            lb = _remaining_lower_bound(nxt_n, next_move_count, cap)
            heapq.heappush(heap, (lb, next_move_count, nxt_n, counter, nxt))

    print(
        "FINAL "
        f"status=TARGET_INFEASIBLE target_final_N={target_final_n} expanded={expanded} generated={generated} "
        f"visited={len(visited)} pruned_bound={pruned_bound} pruned_business={pruned_business} "
        f"dominated={dominated} runtime={time.perf_counter() - start:.3f}s",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="C1")
    parser.add_argument("--bay", default="26")
    parser.add_argument("--target-final-n", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--time-limit", type=float, default=1_000_000_000.0)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    args = parser.parse_args()
    return run(args.zone, args.bay, args.target_final_n, args.alpha, args.time_limit, args.progress_interval)


if __name__ == "__main__":
    raise SystemExit(main())
