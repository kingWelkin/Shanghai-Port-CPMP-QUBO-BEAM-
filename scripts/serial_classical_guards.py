"""Cyclic Q followed by both independently paid classical trajectories.

The two orders are explicit alternatives. Every branch starts from the same
actual input path, and every pass uses the original caller's absolute deadline.
"""
from __future__ import annotations

from dataclasses import asdict

from container_reshuffle.core import Move
from production_selector_replay import REAL_TIME, fingerprint
from single_seed_conditioned_refinement import run as pass_run

POLICIES = dict(qubo_qsp=('single_qubo', 'single', 'pair'),
                qubo_qps=('single_qubo', 'pair', 'single'))
MODES = tuple(POLICIES)


def run(instance, path, mode, start, budget, publish):
    if mode not in MODES:
        raise ValueError(mode)
    policies = POLICIES[mode]
    trajectories = [tuple(path) for _ in policies]
    active = [bool(path) for _ in policies]
    passes = [0 for _ in policies]
    global_best = tuple(path)
    records = []
    q_search_starts = credited_joint_paths = 0
    deadline = start + budget
    while any(active) and global_best and REAL_TIME() < deadline:
        branch = next(index for index, available in enumerate(active) if available)
        before = len(trajectories[branch])
        round_index = passes[branch]
        publish(dict(type='TRAJECTORY_BEGIN', branch=branch, round=round_index,
            policy=policies[branch], before=before, global_moves=len(global_best),
            absolute_start=start, budget_sec=budget,
            path_signature=fingerprint([asdict(move) for move in trajectories[branch]]),
            elapsed_sec=REAL_TIME()-start, schedule='serial_to_stall'))

        def emit(event):
            nonlocal global_best, q_search_starts, credited_joint_paths
            event = dict(event, trajectory_branch=branch, trajectory_round=round_index,
                         trajectory_policy=policies[branch])
            if event['type'] == 'PHASE' and event.get('phase') == 'qubo_search':
                q_search_starts += 1
            if event['type'] == 'INCUMBENT':
                if event['moves'] >= len(global_best):
                    return
                candidate = tuple(Move(**value) for value in event['path'])
                if len(candidate) != event['moves']:
                    raise ValueError('Wrong complete-path length')
                ready = REAL_TIME()-start
                if ready > budget:
                    return
                global_best = candidate
                credited_joint_paths += event['source'] == 'qubo_joint'
                event.update(ready_sec=ready, source=f'branch{branch}:'+event['source'],
                             original_source=event['source'])
            elif event['type'] == 'MODEL':
                event['record'] = dict(event['record'], trajectory_branch=branch,
                    trajectory_round=round_index, trajectory_policy=policies[branch],
                    global_moves=len(global_best))
            publish(event)

        last = pass_run(instance, trajectories[branch], policies[branch], start, budget, emit)
        candidate = tuple(Move(**value) for value in last['path'])
        if len(candidate) > before:
            raise ValueError('Branch lost its own incumbent')
        trajectories[branch] = candidate
        passes[branch] += 1
        active[branch] = bool(candidate and len(candidate) < before)
        record = dict(branch=branch, round=round_index, policy=policies[branch],
            before=before, after=len(candidate), active=active[branch],
            global_moves=len(global_best), elapsed_sec=REAL_TIME()-start)
        records.append(record)
        publish(dict(type='TRAJECTORY_END', path=[asdict(move) for move in candidate], **record))
    return dict(mode=mode, moves=len(global_best), path=[asdict(move) for move in global_best],
        seed_kind='explicit_three_trajectory', passes=sum(passes), branch_passes=passes,
        branch_moves=list(map(len, trajectories)), branch_active=active, trajectory_records=records,
        backend_end_sec=REAL_TIME()-start, elapsed_sec=REAL_TIME()-start,
        stop_reason='deadline' if REAL_TIME() >= deadline else 'all_trajectories_stalled',
        global_cpmp_optimal=False, configured_policies=policies,
        used_qubo=bool(q_search_starts), q_search_starts=q_search_starts,
        worker_timely_joint_publications=credited_joint_paths, schedule='serial_to_stall')
