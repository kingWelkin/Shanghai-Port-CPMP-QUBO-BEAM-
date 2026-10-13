"""Read-only lineage audit from equal fresh inputs to the first joint change.

This compares complete received model/seed metadata and pre-joint selections.
It does not reconstruct the matrices or replace the driver's physical replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/experiments/action_coupling_qubo_20261011'
FOLDERS = dict(pilot='shared_prefix_wall_pilot', full='shared_prefix_full_hardset169')


def semantic(value):
    if isinstance(value, dict):
        return {key: semantic(item) for key, item in value.items() if not key.endswith('_sec')}
    if isinstance(value, list):
        return [semantic(item) for item in value]
    return value


def compare_block(data, budget):
    if data['status'] != 'COMPLETE':
        return dict(status=data['status'])
    actors = data['actors']
    q_events = [event for event in actors['qubo']['messages'] if event['received_sec'] <= budget]
    p_events = [event for event in actors['pair']['messages'] if event['received_sec'] <= budget]
    q_models = [event['record'] for event in q_events if event['type'] == 'MODEL'
                and event['record']['status'] == 'SUPPORTED']
    p_models = {event['record']['index']: event['record'] for event in p_events
                if event['type'] == 'MODEL' and event['record']['status'] == 'SUPPORTED'}
    joint_models = [model for model in q_models if model['energy'] < model['seed_moves']]
    q_input = next((event for event in q_events if event['type'] == 'BRANCH_INPUT'), None)
    p_input = next((event for event in p_events if event['type'] == 'BRANCH_INPUT'), None)
    input_match = bool(q_input and p_input and q_input['model_input_signature'] is not None
                       and q_input['model_input_signature'] == p_input['model_input_signature'])
    first = joint_models[0] if joint_models else None
    checks = []
    first_seed_match = False
    changed_groups = None
    status = 'NO_COMPLETED_JOINT_GAIN'
    if first is not None:
        for q in q_models:
            if q['index'] > first['index']:
                break
            p = p_models.get(q['index'])
            if p is None:
                checks.append(dict(index=q['index'], matched=False, reason='PAIR_MODEL_NOT_RECEIVED_BY_DEADLINE'))
                continue
            fields = ('before', 'seed_moves', 'variables')
            matching = (all(q[field] == p[field] for field in fields)
                and semantic(q['model']) == semantic(p['model'])
                and semantic(q['seed_metadata']) == semantic(p['seed_metadata']))
            if q['index'] < first['index']:
                matching = matching and q['selected'] == p['selected'] and q['energy'] == p['energy']
            else:
                matching = matching and p['after'] == p['seed_moves'] == q['seed_moves']
                first_seed_match = matching
                if matching and len(q['selected']) == len(p['selected']):
                    changed_groups = sum(a != b for a, b in zip(q['selected'], p['selected']))
            checks.append(dict(index=q['index'], matched=matching,
                reason='MATCHED_RECEIVED_PREFIX_METADATA' if matching else 'MODEL_OR_SEED_METADATA_DIFFERENCE'))
        matched = input_match and all(check['matched'] for check in checks) and first_seed_match
        status = 'JOINT_AFTER_MATCHED_PAIR_PREFIX' if matched else 'JOINT_WITH_INCOMPLETE_PREFIX_METADATA'
    else:
        matched = False
    joint_paths = [event for event in q_events if event['type'] == 'INCUMBENT'
                   and event['ready_sec'] <= budget
                   and event['source'] in ('refine:qubo_joint', 'common_graph:refine:qubo_joint')]
    q, p = data['answers']['qubo'], data['answers']['pair']
    delta = q['moves'] - p['moves'] if q['valid'] and p['valid'] else None
    return dict(status=status, shared_model_input=input_match, delta=delta,
        completed_joint_models=len(joint_models), credited_joint_paths=len(joint_paths),
        first_joint=None if first is None else dict(index=first['index'], before=first['before'],
            seed_moves=first['seed_moves'], energy=first['energy'], changed_groups_from_received_pair=changed_groups,
            pair_local_optimal=first['seed_metadata']['selected_metadata']['pair_local_optimal']),
        matching_reference_seed_metadata=matched, prior_checks=checks)


def main(scope):
    folder = BASE / FOLDERS[scope]
    frozen = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    rows, missing, original_checks, violations = [], [], [], []
    for index, item in enumerate(frozen['cases']):
        paired = []
        for repeat in range(frozen['repeats']):
            path = folder / 'runs' / f'c{index}__r{repeat}.json'
            if not path.exists():
                missing.append(dict(index=index, repeat=repeat))
                continue
            data = json.loads(path.read_text(encoding='utf-8'))
            row = dict(case=item['entry']['case'], repeat=repeat, **compare_block(data, frozen['budget_sec']))
            first = row.get('first_joint')
            if (row.get('matching_reference_seed_metadata') and first is not None
                    and first['pair_local_optimal'] and first['changed_groups_from_received_pair'] is not None
                    and first['changed_groups_from_received_pair'] <= 2):
                violations.append(dict(case=row['case'], repeat=repeat, first_joint=first,
                                       reason='Strict gain differs in at most two groups from a matched Pair optimum'))
            rows.append(row)
            paired.append(row)
        if len(paired) == frozen['repeats']:
            valid = all(row.get('delta') is not None for row in paired)
            stable = valid and all(row['delta'] < 0 for row in paired)
            original_checks.append(dict(case=item['entry']['case'], valid=valid, stable_win=stable,
                deltas=[row.get('delta') for row in paired],
                joint_paths_all=all(row.get('credited_joint_paths', 0) for row in paired),
                matched_first_joint_all=all(row.get('matching_reference_seed_metadata') for row in paired)))
    summary = dict(scope=scope, finished_blocks=len(rows), expected_blocks=len(frozen['cases']) * frozen['repeats'],
        complete=not missing, missing=missing, violations=violations,
        statuses=dict(Counter(row['status'] for row in rows)),
        stable_wins=sum(row['stable_win'] for row in original_checks),
        stable_wins_with_matched_first_joint=sum(row['stable_win'] and row['matched_first_joint_all'] for row in original_checks),
        stable_wins_without_joint_paths=[row['case'] for row in original_checks if row['stable_win'] and not row['joint_paths_all']],
        stable_wins_without_matched_first_joint=[row for row in original_checks if row['stable_win'] and not row['matched_first_joint_all']],
        rows=rows, original_checks=original_checks,
        note='Metadata and selected-prefix audit only. The frozen driver independently physically replays every '
             'published path. Missing Pair model records may reflect the actual deadline;they are retained as '
             'incomplete attribution,not proof of a changed model. No hidden or late answer is credited.')
    name = 'lineage_audit.json' if not missing else 'lineage_partial_audit.json'
    (folder / name).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items()
                     if key not in ('rows', 'original_checks', 'missing', 'note')}))
    if violations:
        raise ValueError('Joint/Pair local-optimum lineage invariant failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=tuple(FOLDERS), required=True)
    main(parser.parse_args().scope)
