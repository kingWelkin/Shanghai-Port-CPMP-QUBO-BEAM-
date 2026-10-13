"""Independent accounting for any explicitly configured serial trajectory order."""
from __future__ import annotations

from production_selector_replay import fingerprint
from serial_classical_guards import POLICIES


def audit_answer(answer, initial, mode, budget):
    policies = POLICIES[mode]
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    events = [event for event in answer['messages'] if event['received_sec'] <= budget]
    acks = [event for event in events if event['type'] == 'START_ACK']
    require(len(acks) == 1 and acks[0]['budget_sec'] == budget, 'Wrong original clock')
    start = acks[0]['absolute_start'] if acks else None
    paths = [initial for _ in policies]
    active = [bool(initial) for _ in policies]
    rounds = [0 for _ in policies]
    best = len(initial)
    pending = None
    models = strict = joint_publications = requests = 0
    entries = []
    for event in events:
        kind = event['type']
        if kind == 'INCUMBENT' and event['ready_sec'] <= budget:
            require(event['moves'] == len(event['path']) < best, 'Invalid global publication')
            require(pending is not None and event.get('trajectory_branch') == pending[0],
                    'Publication from inactive trajectory')
            best = event['moves']
            joint_publications += event.get('original_source', event['source']) == 'qubo_joint'
        elif kind == 'TRAJECTORY_BEGIN':
            branch = event['branch']
            if branch not in range(len(policies)):
                require(False, 'Unknown trajectory')
                continue
            expected = next((index for index, available in enumerate(active) if available), None)
            require(pending is None and branch == expected and active[branch], 'Premature or inactive guard')
            require(event['absolute_start'] == start and event['budget_sec'] == budget, 'Rebased trajectory clock')
            require(event['policy'] == policies[branch] and event['round'] == rounds[branch], 'Wrong trajectory policy/round')
            require(event['before'] == len(paths[branch]) and event['path_signature'] == fingerprint(paths[branch]),
                    'Trajectory did not use its own actual path')
            require(event['global_moves'] == best, 'Wrong global ledger at branch entry')
            pending = (branch, event['round'], event['before'])
            entries.append(dict(branch=branch, round=event['round'], policy=event['policy'],
                path_signature=event['path_signature'], before=event['before']))
        elif kind == 'TRAJECTORY_END':
            branch = event['branch']
            require(pending == (branch, event['round'], event['before']), 'Unmatched branch completion')
            require(len(event['path']) == event['after'] <= event['before'], 'Invalid own-branch improvement')
            require(event['global_moves'] == best <= event['after'], 'Wrong branch/global cost')
            require(event['active'] == bool(event['after'] and event['after'] < event['before']), 'Wrong stall rule')
            paths[branch] = event['path']
            active[branch] = event['active']
            rounds[branch] += 1
            pending = None
        elif kind == 'PHASE' and event.get('phase') == 'qubo_search':
            requests += 1
        elif kind == 'MODEL':
            model = event['record']
            models += 1
            branch = model.get('trajectory_branch')
            require(pending is not None and (branch, model.get('trajectory_round')) == pending[:2],
                    'Wrong active model trajectory')
            if branch not in range(len(policies)):
                require(False, 'Unknown model branch')
                continue
            policy = policies[branch]
            require(model['trajectory_policy'] == policy, 'Wrong model policy')
            pair = policy == 'pair'
            seed = model['seed_metadata']
            require(seed['seed_kind'] == ('four_start_pair' if pair else 'three_order_single'), 'Wrong seed')
            require(len(seed['single_trials']) == 3 and len(seed['pair_trials']) == (4 if pair else 0), 'Wrong paid seed starts')
            require(model['nonoracle_milp_calls'] == 0, 'Unexpected oracle call')
            require(model['energy'] == model['after'] <= model['seed_moves'] <= model['before'], 'Wrong physical/model cost')
            require(model['global_moves'] == best <= model['after'], 'Wrong model/global ledger')
            backend = model.get('backend') or {}
            require(not (backend.get('optimal') and backend.get('cutoff')), 'False local certificate')
            if policy == 'single_qubo':
                require(backend.get('local_time_limit_sec') == .05, 'Changed Q time request')
            if pair:
                require(seed['selected_metadata']['pair_local_optimal'], 'Pair seed not converged')
            if model['energy'] < model['seed_moves']:
                strict += 1
                require(policy == 'single_qubo' and model['changed_groups_from_seed'] >= 2, 'Unsupported joint gain')
    require(best == answer['moves'], 'Late or unpublished result credited')
    return dict(errors=errors, models=models, strict_models=strict, q_requests=requests,
        joint_publications=joint_publications, branch_inputs=entries, completed_rounds=rounds,
        single_guard_started=any(entry['policy'] == 'single' for entry in entries),
        pair_guard_started=any(entry['policy'] == 'pair' for entry in entries))
