"""Run the published v5.0 Q+Pair algorithm on a tracked CPMP instance.

The release adapter reuses the measured producer, continuation and deadline
protocol. It does not require historical experiment logs or machine CPU IDs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'scripts', ROOT / 'src'):
    sys.path.insert(0, str(directory))

INVENTORY = ROOT / 'configs/hardset169.json'
MODES = ('qubo_pair', 'qubo_cycle', 'single_pair')


def entries():
    return json.loads(INVENTORY.read_text(encoding='utf-8'))['cases']


def find_case(case):
    return next((entry for entry in entries() if entry['case'] == case), None)


def worker(entry, role, mode):
    # Imports and input identity checks are prewarmed before the shared START.
    import research_paper_v43_panel as broad
    from paired_shared_pipeline import run_continuation
    from paper_priority_group_contract import PriorityGroupContract
    from production_selector_replay import REAL_TIME
    from run_beam_selector_research import digest
    from shared_prefix_cpmp_pipeline import run_prefix

    if digest(ROOT / entry['path']) != entry['sha256']:
        raise ValueError('Benchmark input differs from the released inventory')
    output = sys.stdout

    def publish(event):
        # Capture stdout before the Beam redirects its diagnostic output.
        output.write(json.dumps(event, separators=(',', ':')) + '\n')
        output.flush()

    publish(dict(type='READY', role=role, case=entry['case']))
    clock = json.loads(sys.stdin.readline())
    start, budget = clock['start'], clock['budget']
    publish(dict(type='START_ACK', absolute_start=start, budget_sec=budget))
    try:
        if role == 'prefix':
            result = run_prefix(entry, start, budget, publish, reserve=2.)
        else:
            contract = PriorityGroupContract()
            contract.install()
            try:
                instance = broad.load_case(entry)
                message = json.loads(sys.stdin.readline())
                if message['type'] == 'STOP':
                    result = dict(mode=mode, skipped=True, reason=message['reason'])
                else:
                    if message['start'] != start or message['budget'] != budget:
                        raise ValueError('Continuation clock was rebased')
                    result = run_continuation(instance, mode, message['handoff'],
                                              start, budget, publish)
            finally:
                contract.restore()
        publish(dict(type='DONE', **result))
    except Exception as error:
        publish(dict(type='ERROR', detail=str(error), elapsed_sec=REAL_TIME() - start))
        raise


def solve(entry, mode, budget):
    import research_paper_v43_panel as broad
    from container_reshuffle.core import Move
    from research_receiver_allocation import physical_replay
    from run_beam_selector_research import ENVIRONMENT
    from shared_wall_deadline_protocol_full import credited_path, run_shared

    commands = {
        role: [sys.executable, '-u', str(Path(__file__).resolve()), '--case',
               entry['case'], '--mode', mode, '--worker-role', role]
        for role in ('prefix', 'refinement')
    }
    data = run_shared(commands, cwd=ROOT, env={**os.environ, **ENVIRONMENT}, budget=budget)
    prefix = data['actors']['prefix']['messages']
    continuation = data['actors']['refinement']['messages']
    best = credited_path(prefix, continuation, budget)
    final_state = None
    if best is not None:
        instance = broad.load_case(entry)
        path = tuple(Move(**value) for value in best['path'])
        if len(path) != best['moves']:
            raise ValueError('Output action count does not match the actual path')
        # Delivery replay only verifies the already credited path. It cannot
        # accept a late candidate or improve the measured result.
        final_state = physical_replay(instance.stacks, instance.max_height, path, goal=True)
    issues = data['issues']
    return dict(
        algorithm='QUBO-Beam v5.0', mode=mode, case=entry['case'],
        feasible=best is not None, global_cpmp_optimal=False,
        status=('FEASIBLE_WITH_WORKER_ERROR' if issues else 'FEASIBLE') if best else
               ('WORKER_ERROR' if issues else 'NO_FEASIBLE_SOLUTION_WITHIN_BUDGET'),
        moves=None if best is None else best['moves'],
        path=[] if best is None else best['path'], final_stacks=final_state,
        source=None if best is None else best['source'], budget_sec=budget,
        ready_sec=None if best is None else best['ready_sec'],
        received_sec=None if best is None else best['received_sec'],
        physical_replay_passed=best is not None,
        worker_status={role: actor['status'] for role, actor in data['actors'].items()},
        issues=issues, termination_lag_sec=data['termination_lag_sec'],
        cleanup_sec=data['cleanup_sec'],
        timing='Shared prewarmed START; final delivery replay and JSON writing are outside the scoring budget.',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', help='Case ID from --list-cases')
    parser.add_argument('--list-cases', action='store_true')
    parser.add_argument('--mode', choices=MODES, default='qubo_pair')
    parser.add_argument('--budget-sec', type=float, default=20.)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker-role', choices=('prefix', 'refinement'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.list_cases:
        print('\n'.join(entry['case'] for entry in entries()))
        return 0
    if not args.case:
        parser.error('--case is required; use --list-cases for available inputs')
    entry = find_case(args.case)
    if entry is None:
        parser.error('Unknown --case; use --list-cases for the frozen inventory')
    if not math.isfinite(args.budget_sec) or args.budget_sec <= 0:
        parser.error('--budget-sec must be positive and finite')
    if args.worker_role:
        worker(entry, args.worker_role, args.mode)
        return 0
    if args.output and args.output.exists():
        parser.error('--output already exists; choose a new result path')
    result = solve(entry, args.mode, args.budget_sec)
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as output:
            output.write(text)
    print(text, end='')
    return 0 if result['feasible'] and not result['issues'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
