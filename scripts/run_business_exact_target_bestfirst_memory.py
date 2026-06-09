"""In-memory exact best-first target search for business final N(Future)."""

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
from run_business_exact_target_search_sqlite import _encode, _remaining_lower_bound, _successors_with_n  # noqa: E402
from run_exact_astar_real_until_done import _instance_from_single_bay  # noqa: E402

Code = bytes


def run(
    zone: str,
    bay: str,
    target_final_n: int,
    alpha: float,
    time_limit: float,
    progress_interval: float,
    max_moves: int | None,
) -> int:
    instance = _instance_from_single_bay(zone, bay)
    initial, n0 = _encode(instance.stacks, instance.stack_count, instance.max_height)
    cap = entropy_limit(instance, alpha)
    depth_cap = min(cap, int(max_moves)) if max_moves is not None else cap
    start = time.perf_counter()
    next_progress = start
    counter = 0
    expanded = generated = pruned_bound = pruned_business = dominated = 0

    visited: dict[Code, int] = {initial: 0}
    heap: list[tuple[int, int, int, Code]] = [(n0, 0, counter, initial)]

    print(
        f"START case={instance.instance_id} target_final_N={target_final_n} "
        f"stacks={instance.stack_count} H={instance.max_height} containers={instance.container_count} "
        f"initial_N={n0} cap={cap} depth_cap={depth_cap}",
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
            min_n = heap[0][0] if heap else -1
            min_depth = heap[0][1] if heap else -1
            print(
                "PROGRESS "
                f"elapsed={now - start:.1f}s min_open_N={min_n} depth={min_depth} expanded={expanded} "
                f"generated={generated} visited={len(visited)} open={len(heap)} "
                f"pruned_bound={pruned_bound} pruned_business={pruned_business} dominated={dominated}",
                flush=True,
            )
            next_progress = now + progress_interval

        current_n, depth, _idx, code = heapq.heappop(heap)
        if depth != visited.get(code):
            dominated += 1
            continue
        if current_n <= target_final_n:
            print(
                "FINAL "
                f"status=TARGET_REACHABLE target_final_N={target_final_n} moves={depth} final_N={current_n} "
                f"expanded={expanded} generated={generated} visited={len(visited)} open={len(heap)} "
                f"pruned_bound={pruned_bound} pruned_business={pruned_business} dominated={dominated} "
                f"runtime={time.perf_counter() - start:.3f}s",
                flush=True,
            )
            return 0
        if _remaining_lower_bound(current_n, depth, depth_cap) > target_final_n or depth >= depth_cap:
            pruned_bound += 1
            continue

        expanded += 1
        next_depth = depth + 1
        for nxt, nxt_n in _successors_with_n(code, current_n, instance.stack_count, instance.max_height):
            generated += 1
            if next_depth + nxt_n > cap:
                pruned_business += 1
                continue
            if nxt_n <= target_final_n:
                print(
                    "FINAL "
                    f"status=TARGET_REACHABLE target_final_N={target_final_n} moves={next_depth} final_N={nxt_n} "
                    f"expanded={expanded} generated={generated} visited={len(visited)} open={len(heap)} "
                    f"pruned_bound={pruned_bound} pruned_business={pruned_business} dominated={dominated} "
                    f"runtime={time.perf_counter() - start:.3f}s",
                    flush=True,
                )
                return 0
            if _remaining_lower_bound(nxt_n, next_depth, depth_cap) > target_final_n:
                pruned_bound += 1
                continue
            if visited.get(nxt, cap + 1) <= next_depth:
                dominated += 1
                continue
            visited[nxt] = next_depth
            counter += 1
            heapq.heappush(heap, (nxt_n, next_depth, counter, nxt))

    print(
        "FINAL "
        f"status=TARGET_INFEASIBLE target_final_N={target_final_n} expanded={expanded} generated={generated} "
        f"visited={len(visited)} open=0 pruned_bound={pruned_bound} pruned_business={pruned_business} "
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
    parser.add_argument("--max-moves", type=int)
    args = parser.parse_args()
    return run(
        args.zone,
        args.bay,
        args.target_final_n,
        args.alpha,
        args.time_limit,
        args.progress_interval,
        args.max_moves,
    )


if __name__ == "__main__":
    raise SystemExit(main())
