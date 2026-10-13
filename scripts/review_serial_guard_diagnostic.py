"""Offline verification of credited paths and actual serial-guard trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT/'scripts', ROOT/'src'):
    sys.path.insert(0, str(directory))
import research_paper_v43_panel as broad
from container_reshuffle.core import Move
from cpmp_cpu_layout import cpu_sets
from production_selector_replay import fingerprint
from research_receiver_allocation import physical_replay
from research_hardset_qubo_utility import write_json
from shared_wall_deadline_protocol_full import credited_path

BASE = ROOT/'docs/experiments/v50_budget_tail_20261013'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def trajectory(actor, branch, budget):
    return [dict(round=event['round'], moves=event['after'], signature=fingerprint(event['path']),
                 elapsed_sec=event['elapsed_sec']) for event in actor['messages']
            if event['type']=='TRAJECTORY_END' and event['branch']==branch and event['received_sec']<=budget]


def identity(left, right):
    overlap = min(len(left), len(right))
    matched = 0
    for a, b in zip(left, right):
        if a['signature'] != b['signature']:
            break
        matched += 1
    return dict(compared=overlap, matched_leading_paths=matched,
                all_overlapping_identical=matched==overlap)


def review(scope):
    folder = BASE/('serial_guard_'+scope)
    if read(folder/'status.json')['status'] != 'COMPLETE':
        raise ValueError('Review only a completed diagnostic')
    manifest = read(folder/'manifest.json')
    rows, errors = [], []
    checks = 0
    for index, spec in enumerate(manifest['inputs']):
        item = manifest['cases'][spec['original_index']]
        instance = broad.load_case(item['entry'])
        for repeat in range(manifest['repeats']):
            data = read(folder/'runs'/f'i{index}__r{repeat}.json')
            budget = spec['budget_sec']
            key = {name: data[name] for name in ('case', 'role', 'source_repeat', 'nominal_total_sec', 'repeat')}
            actors = data['actors']
            for mode, answer in data['answers'].items():
                best = credited_path(actors['prefix']['messages'], actors[mode]['messages'], budget)
                if best is None or answer['moves'] != best['moves'] or answer['path'] != best['path']:
                    errors.append(dict(key, mode=mode, reason='Wrong credited answer'))
                    continue
                physical_replay(instance.stacks, instance.max_height,
                    tuple(Move(**value) for value in best['path']), goal=True)
                checks += 1
            q = trajectory(actors['qubo_qsp'], 0, budget)
            reference_q = trajectory(actors['qubo_deferred_pair'], 0, budget)
            single = trajectory(actors['qubo_qsp'], 1, budget)
            reference_single = trajectory(actors['single_deferred_pair'], 0, budget)
            pair = trajectory(actors['qubo_qsp'], 2, budget)
            reference_pair = trajectory(actors['qubo_deferred_pair'], 1, budget)
            begins = [dict(branch=event['branch'], policy=event['policy'], elapsed_sec=event['elapsed_sec'])
                for event in actors['qubo_qsp']['messages']
                if event['type']=='TRAJECTORY_BEGIN' and event['round']==0 and event['received_sec']<=budget]
            rows.append(dict(key, budget_sec=budget, moves={mode: answer['moves'] for mode, answer in data['answers'].items()},
                q_identity=identity(q, reference_q), single_identity=identity(single, reference_single),
                pair_identity=identity(pair, reference_pair), qsp_branch_starts=begins,
                qsp_q_cycles=q, qsp_single_cycles=single, qsp_pair_cycles=pair,
                final_provenance=data['answers']['qubo_qsp'].get('provenance')))
    qualified = [row for row in rows if row['q_identity']['compared']]
    single_repairs = [row for row in rows if row['moves']['qubo_qsp'] < row['moves']['qubo_deferred_pair']
                      and (row['final_provenance'] or {}).get('trajectory_policy') == 'single']
    result = dict(scope=scope, credited_paths_replayed=checks, errors=errors,
        compared_primary_runs=len(qualified), identical_overlapping_primary_runs=sum(
            row['q_identity']['all_overlapping_identical'] for row in qualified),
        actual_single_guard_winning_runs=len(single_repairs), rows=rows)
    write_json(folder/'independent_review.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=('smoke', 'diagnostic'), required=True)
    args = parser.parse_args()
    spare = [row['logical_processor'] for row in cpu_sets() if row['group']==0 and row['logical_processor']>=12]
    if spare:
        broad.pin_cpu(spare[-1])
    review(args.scope)
