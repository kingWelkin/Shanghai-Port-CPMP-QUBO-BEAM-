"""Offline path identity and paid-tail attribution for completed budget studies."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT/'scripts', ROOT/'src'):
    sys.path.insert(0, str(directory))
from production_selector_replay import fingerprint
from shared_wall_deadline_protocol_full import credited_path
from research_hardset_qubo_utility import write_json
from research_receiver_allocation import physical_replay
from container_reshuffle.core import Move
import research_paper_v43_panel as broad

BASE = ROOT/'docs/experiments/v50_budget_tail_20261013'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def primary_paths(actor, budget, solo=False):
    events = [event for event in actor['messages'] if event['received_sec'] <= budget]
    entry = next((event for event in events if event['type'] == 'BRANCH_INPUT'), None)
    if entry is None or entry['model_input_signature'] is None:
        return []
    initial = next((event['path'] for event in events if event['type'] == 'INCUMBENT' and
                    fingerprint(event['path']) == entry['model_input_signature']), None)
    if initial is None:
        raise ValueError('Initial trajectory lacks a reported physical path')
    current = initial
    completed = []
    for event in events:
        if event['type'] == 'REFINEMENT_INCUMBENT' and event['ready_sec'] <= budget:
            if solo or event.get('trajectory_branch') == 0:
                current = event['path']
        if solo and event['type'] == 'CYCLE_END':
            if len(current) != event['after']:
                raise ValueError('Completed solo cycle cannot be reconciled to its raw path')
            completed.append(dict(round=event['cycle'], moves=len(current),
                signature=fingerprint(current), elapsed_sec=event['elapsed_sec']))
        elif not solo and event['type'] == 'TRAJECTORY_END' and event['branch'] == 0:
            completed.append(dict(round=event['round'], moves=event['after'],
                signature=fingerprint(event['path']), elapsed_sec=event['elapsed_sec']))
    return completed


def compare_prefix(left, right):
    matched = 0
    for a, b in zip(left, right):
        if a['signature'] != b['signature']:
            break
        matched += 1
    return dict(left_completed=len(left), right_completed=len(right),
        compared=min(len(left), len(right)), matched_leading_paths=matched,
        all_overlapping_paths_identical=matched == min(len(left), len(right)))


def review(scope):
    folder = BASE/scope
    if read(folder/'status.json')['status'] != 'COMPLETE':
        raise ValueError('Review only a completed study')
    frozen = read(folder/'manifest.json')
    rows = []
    errors = []
    physical = 0
    for index, item in enumerate(frozen['cases']):
        instance = broad.load_case(item['entry'])
        for repeat in range(frozen['repeats']):
            data = read(folder/'runs'/f'c{index}__r{repeat}.json')
            budget = frozen['budget_sec']
            key = dict(case=item['entry']['case'], repeat=repeat, cohorts=item['cohorts'])
            if data['status'] != 'COMPLETE':
                errors.append(dict(key, reason='Incomplete execution'))
                continue
            prefix = data['actors']['prefix']['messages']
            for mode, answer in data['answers'].items():
                best = credited_path([] if mode == 'native' else prefix,
                                     data['actors'][mode]['messages'], budget)
                if answer['valid'] != (best is not None) or answer['moves'] != (None if best is None else best['moves']):
                    errors.append(dict(key, mode=mode, reason='Credited answer mismatch'))
                if best is not None:
                    physical_replay(instance.stacks, instance.max_height,
                        tuple(Move(**value) for value in best['path']), goal=True)
                    physical += 1
            actors = data['actors']
            q = primary_paths(actors['qubo_cycle'], budget, solo=True)
            qp = primary_paths(actors['qubo_pair'], budget)
            deferred = primary_paths(actors['qubo_deferred_pair'], budget)
            begins = [event for event in actors['qubo_deferred_pair']['messages']
                      if event['type'] == 'TRAJECTORY_BEGIN' and event['branch'] == 1
                      and event['received_sec'] <= budget]
            best = credited_path(prefix, actors['qubo_deferred_pair']['messages'], budget)
            rows.append(dict(key, moves={mode: answer['moves'] for mode, answer in data['answers'].items()},
                solo_q_cycles=q, original_qp_primary=qp, deferred_primary=deferred,
                deferred_vs_solo=compare_prefix(deferred, q), deferred_vs_qp=compare_prefix(deferred, qp),
                pair_started=bool(begins), pair_start_sec=begins[0]['elapsed_sec'] if begins else None,
                final_provenance=None if best is None else best.get('refinement_provenance')))
    overlap = [row for row in rows if row['deferred_vs_solo']['compared']]
    result = dict(scope=scope, budget_sec=frozen['budget_sec'], credited_paths_replayed=physical,
        errors=errors, overlapping_solo_primary_runs=len(overlap),
        identical_overlapping_primary_runs=sum(row['deferred_vs_solo']['all_overlapping_paths_identical'] for row in overlap),
        paid_pair_starts=sum(row['pair_started'] for row in rows), rows=rows)
    write_json(folder/'trajectory_identity_review.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', required=True)
    args = parser.parse_args()
    # Offline review uses an otherwise unused efficiency core when available.
    from cpmp_cpu_layout import cpu_sets
    spare = [row['logical_processor'] for row in cpu_sets() if row['group'] == 0 and row['logical_processor'] >= 12]
    if spare:
        broad.pin_cpu(spare[-1])
    review(args.scope)
