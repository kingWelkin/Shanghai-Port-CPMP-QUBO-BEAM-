"""Independently check deferred scheduling, seeds, lineage and original clocks."""
from __future__ import annotations

from deferred_pair_refinement import PRIMARY
from production_selector_replay import fingerprint


def audit_answer(answer, initial, mode, budget):
    if mode not in PRIMARY:
        raise ValueError(mode)
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    events = [event for event in answer['messages'] if event['received_sec'] <= budget]
    acks = [event for event in events if event['type'] == 'START_ACK']
    require(len(acks) == 1 and acks[0]['budget_sec'] == budget, 'Missing/wrong original clock')
    original_start = acks[0]['absolute_start'] if acks else None
    paths = [initial, initial]
    active = [bool(initial), bool(initial)]
    rounds = [0, 0]
    pending = None
    best = len(initial)
    models = strict = requests = joint_publications = pair_publications = 0
    inputs = []
    policies = (PRIMARY[mode], 'pair')
    for event in events:
        kind = event['type']
        if kind == 'INCUMBENT':
            if event['ready_sec'] > budget:
                continue
            require(event['moves'] < best, 'Nonmonotone raw global incumbent')
            require(event['moves'] == len(event['path']), 'Incorrect raw full-path length')
            require(pending is not None and event.get('trajectory_branch') == pending[0],
                    'Publication not from the active trajectory')
            best = event['moves']
            joint_publications += event.get('original_source', event['source']) == 'qubo_joint'
            pair_publications += event.get('trajectory_branch') == 1
        elif kind == 'TRAJECTORY_BEGIN':
            branch = event['branch']
            require(branch in (0, 1), 'Unknown branch')
            if branch not in (0, 1):
                continue
            expected = 0 if active[0] else 1
            require(pending is None and active[branch] and branch == expected,
                    'Pair started before primary stagnation or branch was inactive')
            require(event['absolute_start'] == original_start and event['budget_sec'] == budget,
                    'Rebased branch clock')
            require(event['round'] == rounds[branch] and event['policy'] == policies[branch],
                    'Wrong branch policy/round')
            require(event['before'] == len(paths[branch]) and
                    event['path_signature'] == fingerprint(paths[branch]), 'Changed own trajectory input')
            require(event['global_moves'] == best, 'Wrong global ledger at branch entry')
            pending = (branch, event['round'], event['before'])
            inputs.append(dict(branch=branch, round=event['round'], policy=event['policy'],
                before=event['before'], path_signature=event['path_signature']))
        elif kind == 'TRAJECTORY_END':
            branch = event['branch']
            require(pending == (branch, event['round'], event['before']), 'Unmatched branch completion')
            require(len(event['path']) == event['after'] <= event['before'], 'Invalid branch improvement')
            require(event['global_moves'] == best <= event['after'], 'Wrong branch/global costs')
            require(event['active'] == bool(event['after'] and event['after'] < event['before']),
                    'Incorrect stagnation rule')
            paths[branch] = event['path']
            rounds[branch] += 1
            active[branch] = event['active']
            pending = None
        elif kind == 'PHASE' and event.get('phase') == 'qubo_search':
            requests += 1
        elif kind == 'MODEL':
            model = event['record']
            models += 1
            branch = model.get('trajectory_branch')
            require(pending is not None and (branch, model.get('trajectory_round')) == pending[:2],
                    'Model from another active branch')
            if branch not in (0, 1):
                require(False, 'Model has unknown branch')
                continue
            policy = model.get('trajectory_policy')
            require(policy == policies[branch], 'Wrong model policy')
            seed = model['seed_metadata']
            is_pair = policy == 'pair'
            require(seed['seed_kind'] == ('four_start_pair' if is_pair else 'three_order_single'),
                    'Wrong seed kind')
            require(len(seed['single_trials']) == 3 and len(seed['pair_trials']) == (4 if is_pair else 0),
                    'Wrong actually computed seed starts')
            if is_pair:
                require(seed['selected_metadata']['pair_local_optimal'], 'Pair seed did not converge')
            require(model['nonoracle_milp_calls'] == 0, 'Unexpected MILP call')
            require(model['energy'] == model['after'] <= model['seed_moves'] <= model['before'],
                    'Wrong energy/full-path accounting')
            require(model['global_moves'] == best <= model['after'], 'Wrong model/global ledger')
            backend = model.get('backend') or {}
            require(not (backend.get('cutoff') and backend.get('optimal')), 'False local certificate')
            if policy == 'single_qubo':
                require(backend.get('local_time_limit_sec') == .05, 'Changed local Q time request')
            if model['energy'] < model['seed_moves']:
                strict += 1
                require(policy == 'single_qubo' and model['changed_groups_from_seed'] >= 2,
                        'Unsupported joint gain attribution')
    require(best == answer['moves'], 'Late/unpublished raw result credited')
    if mode == 'single_deferred_pair':
        require(requests == 0 and joint_publications == 0, 'Q used in the matched classical control')
    return dict(errors=errors, models=models, strict_models=strict, q_requests=requests,
        joint_publications=joint_publications, pair_publications=pair_publications,
        pair_started=any(item['branch'] == 1 for item in inputs), branch_inputs=inputs,
        completed_rounds=rounds)
