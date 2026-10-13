"""Frozen conditional screen for reducing the route-QUBO warm-seed cost."""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / 'scripts', ROOT / 'src'):
    sys.path.insert(0, str(folder))
import numpy as np
import research_paper_v43_panel as broad
from container_reshuffle.core import Move
from deadline_worker_protocol import run_worker
from paper_priority_group_contract import PriorityGroupContract
from production_selector_replay import REAL_TIME
from research_receiver_allocation import physical_replay
from run_beam_selector_research import ENVIRONMENT, digest, write_csv
from research_hardset_qubo_utility import write_json
from single_seed_conditioned_refinement import MODES, run as refine

BASE = ROOT / 'docs/experiments/action_coupling_qubo_20261011'
FULL = BASE / 'shared_prefix_full_hardset169'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def out(scope):
    return BASE / ('single_seed_screen_' + scope)


def freeze(scope):
    dest = out(scope)
    if (dest / 'manifest.json').exists():
        raise FileExistsError('Single-seed screen already frozen')
    if read(FULL / 'status.json')['status'] != 'COMPLETE':
        raise ValueError('Do not contend with the frozen full study')
    full = read(FULL / 'manifest.json')
    primary = read(BASE / 'shared_prefix_wall_pilot/manifest.json')['cases']
    cases = [dict(entry=item['entry'], role='primary') for item in primary]
    seen = {item['entry']['case'] for item in cases}
    tails = {item['case'] for item in read(FULL / 'summary.json')['contrasts']['single']['originals']
             if item['mean'] >= 20}
    for item in full['cases']:
        if item['entry']['case'] in tails and item['entry']['case'] not in seen:
            cases.append(dict(entry=item['entry'], role='posthoc_single_tail'))
    repeats, source_repeats, jobs, modes = 2, (0, 1), 4, MODES
    if scope == 'smoke':
        cases = [next(item for item in cases if item['entry']['dataset'] == dataset)
                 for dataset in ('BF', 'CV', 'Open120')]
        repeats, source_repeats, jobs = 1, (0,), 1
    else:
        check = read(out('smoke') / 'summary.json')
        if not check['budget_gate'] or not check['audit_pass']:
            raise ValueError('Smoke execution has not qualified')
    sources = dict(full['source_sha256'])
    for name in ('deadline_worker_protocol.py', 'single_seed_conditioned_refinement.py',
                 'research_single_seed_screen.py'):
        sources['scripts/' + name] = digest(ROOT / 'scripts' / name)
    inputs = []
    for index, item in enumerate(cases):
        full_index = next(i for i, old in enumerate(full['cases'])
                          if old['entry']['case'] == item['entry']['case'])
        for source_repeat in source_repeats:
            source = FULL / 'runs' / f'c{full_index}__r{source_repeat}.json'
            data = read(source)
            handoff = data.get('handoff')
            source_rel = str(source.relative_to(ROOT))
            sources[source_rel] = digest(source)
            snapshot = dict(case=item['entry']['case'], source=source_rel,
                            source_repeat=source_repeat, valid=handoff is not None,
                            path=handoff['path'] if handoff else None,
                            moves=handoff['moves'] if handoff else None,
                            prior_handoff_ready_sec=handoff['ready_sec'] if handoff else None)
            filename = f'inputs/c{index}__s{source_repeat}.json'
            write_json(dest / filename, snapshot)
            inputs.append(dict(original_index=index, source_repeat=source_repeat,
                               input_file=filename, input_sha256=digest(dest / filename)))
    protocol = (
        'Conditional 2s refinement screen,not a fresh total-time solve. Freeze the same upstream '
        'HANDOFF paths from BOTH full-study repetitions for the unchanged12 primary originals. '
        'Separately include ALL additional originals whose full-study Q-minus-single original '
        'mean is>=20 moves;these selected tails cannot establish independent performance. '
        'Keep missing handoffs without replacing them by a branch winner. Historical upstream '
        'cost,input loading andimports are excluded equally;after START charge input replay, '
        'release selection,exact domain/factor construction,actual seed computation,joint '
        'search,physical validation,FRG andpublication. Single seed means the actual best of '
        'three full single-coordinate descent orders,with NO Pair computation. Single-Q adds '
        'the unchanged50000-node/50ms Q-energy kernel;single-MILP uses the equivalent50ms '
        'solver request under the same parent2s hard deadline. Strong controls freshly compute '
        'the actual four-start Pair seed,with andwithout the same Q kernel;pure FRG is also '
        'fresh. No arm receives a solved seed or another arm answer. Retain only strictly '
        'shorter physically valid complete paths ready AND received by deadline. The unchanged '
        'parent terminates overruns;DONE is metadata,not a late answer channel. Four pinned '
        'lanes run independent input blocks;all six methods run sequentially on the same '
        'lane in case-cyclic,repeat-reversed order. Two fresh execution repetitions of each '
        'available source handoff;aggregate across source handoffs andexecutions WITHIN each '
        'original,never use best-of-repeats orsubmodels as independent cases. Missing source '
        'coverage,all losses/cuts,andcheckpoints0.5/1/2s are reported. Q gain over single is '
        'not automatically a gain over strong Pair. Report seeds,changed groups,actual joint '
        'publications andzero nonoracle MILP calls. Forward screen only:valid audits,at least '
        'two primary originals with strict single-relative gains at every available '
        'source/execution repeat andjoint publications ineach,andnegative primary totals in '
        'BOTH execution repeats against single AND strong Pair. CI andtails still reported. '
        'Passing only permits a fresh raw-bay total-time pilot;no default orgoal promotion. '
        'Smoke uses first BF/CV/Open primary originals,source repeat0,one execution repeat.')
    write_json(dest / 'manifest.json', dict(cases=cases, inputs=inputs, modes=modes,
        repeats=repeats, jobs=jobs, cpus=[0, 2, 4, 6][:jobs], budget_sec=2.,
        checkpoints=[.5, 1., 2.], source_sha256=sources, protocol=protocol,
        frozen_at=datetime.now().isoformat()))
    print(json.dumps(dict(scope=scope, originals=len(cases), inputs=len(inputs), modes=modes)))


def manifest(scope):
    frozen = read(out(scope) / 'manifest.json')
    for name, expected in frozen['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    return frozen


def load_input(scope, frozen, index):
    item = frozen['inputs'][index]
    path = out(scope) / item['input_file']
    if digest(path) != item['input_sha256']:
        raise ValueError('Frozen handoff snapshot changed')
    return frozen['cases'][item['original_index']], read(path)


def worker(scope, index, repeat, mode, cpu):
    frozen = manifest(scope)
    item, raw = load_input(scope, frozen, index)
    if digest(ROOT / item['entry']['path']) != item['entry']['sha256']:
        raise ValueError('Original bay changed')
    broad.pin_cpu(cpu)
    contract = PriorityGroupContract()
    contract.install()
    def publish(message):
        print(json.dumps(message, separators=(',', ':')), flush=True)
    try:
        instance = broad.load_case(item['entry'])
        path = tuple(Move(**value) for value in raw['path'])
        random.seed(37)
        np.random.seed(37)
        publish(dict(type='READY', case=raw['case'], mode=mode, cpu=cpu, initial=len(path)))
        clock = json.loads(sys.stdin.readline())
        result = refine(instance, path, mode, clock['start'], clock['budget'], publish)
        publish(dict(type='DONE', elapsed_sec=REAL_TIME() - clock['start'], moves=result['moves'],
                     seed_kind=result['seed_kind'], backend_end_sec=result['backend_end_sec']))
    except BaseException:
        publish(dict(type='ERROR', detail=traceback.format_exc()))
        raise
    finally:
        contract.restore()


def at(answer, checkpoint):
    return min([answer['initial']] + [event['moves'] for event in answer['messages']
        if event['type'] == 'INCUMBENT' and event['ready_sec'] <= checkpoint
        and event['received_sec'] <= checkpoint])


def execute(scope, index, repeat, cpu, frozen):
    item, raw = load_input(scope, frozen, index)
    data = dict(case=raw['case'], dataset=item['entry']['dataset'], role=item['role'],
                source_repeat=raw['source_repeat'], repeat=repeat)
    if not raw['valid']:
        return dict(data, status='NO_SOURCE_HANDOFF')
    modes = list(frozen['modes'])
    offset = index % len(modes)
    order = modes[offset:] + modes[:offset]
    if repeat % 2:
        order.reverse()
    answers = {}
    for mode in order:
        command = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--scope', scope,
                   '--input-index', str(index), '--repeat', str(repeat), '--mode', mode, '--cpu', str(cpu)]
        try:
            answer = run_worker(command, cwd=ROOT, env={**os.environ, **ENVIRONMENT},
                                budget=frozen['budget_sec'])
        except Exception as error:
            answer = dict(status='WORKER_ERROR', error=str(error), messages=[])
        answer['initial'] = raw['moves']
        eligible = [event for event in answer['messages'] if event['type'] == 'INCUMBENT'
                    and event['ready_sec'] <= frozen['budget_sec']
                    and event['received_sec'] <= frozen['budget_sec']]
        best = min(eligible, key=lambda event: event['moves']) if eligible else raw
        answer.update(moves=best['moves'], path=best['path'])
        answers[mode] = answer
    contract = PriorityGroupContract()
    contract.install()
    try:
        instance = broad.load_case(item['entry'])
        physical_replay(instance.stacks, instance.max_height,
                        tuple(Move(**value) for value in raw['path']), goal=True)
        for answer in answers.values():
            previous = raw['moves']
            for event in answer['messages']:
                if event['type'] == 'INCUMBENT':
                    physical_replay(instance.stacks, instance.max_height,
                                    tuple(Move(**value) for value in event['path']), goal=True)
                    if len(event['path']) != event['moves'] or event['moves'] >= previous:
                        raise ValueError('Nonmonotone or incorrectly scored physical path')
                    previous = event['moves']
    finally:
        contract.restore()
    return dict(data, status='COMPLETE', input=raw['moves'], answers=answers, order=order, cpu=cpu)


def analyze(scope):
    frozen = manifest(scope)
    rows, coverage, missing, failures, audits = [], [], [], [], []
    for index, spec in enumerate(frozen['inputs']):
        for repeat in range(frozen['repeats']):
            data = read(out(scope) / 'runs' / f'i{index}__r{repeat}.json')
            key = {k: data[k] for k in ('case', 'dataset', 'role', 'source_repeat', 'repeat')}
            if data['status'] != 'COMPLETE':
                missing.append(dict(key, status=data['status']))
                continue
            for mode, answer in data['answers'].items():
                if answer['status'] == 'WORKER_ERROR' or answer.get('termination_lag_sec', 0.) > .1:
                    failures.append(dict(key, mode=mode, status=answer['status'], error=answer.get('error'),
                                         lag=answer.get('termination_lag_sec')))
                messages = answer['messages']
                models = [event['record'] for event in messages if event['type'] == 'MODEL'
                          and event['received_sec'] <= frozen['budget_sec']]
                for model in models:
                    if model['status'] != 'SUPPORTED':
                        continue
                    meta = model['seed_metadata']
                    cheap = mode.startswith('single')
                    if cheap and meta['pair_trials']:
                        audits.append(dict(key, mode=mode, error='Hidden Pair seed'))
                    if not mode.endswith('oracle') and model['nonoracle_milp_calls']:
                        audits.append(dict(key, mode=mode, error='Nonoracle MILP call'))
                    backend = model.get('backend') or {}
                    if backend.get('wall_cutoff') and backend.get('optimal'):
                        audits.append(dict(key, mode=mode, error='False cutoff certificate'))
                    if model['energy'] < model['seed_moves'] and model['changed_groups_from_seed'] < (2 if cheap else 3):
                        audits.append(dict(key, mode=mode, error='Improvement contradicts seed local optimum'))
                phases = [event for event in messages if event['type'] == 'PHASE']
                joint = [event for event in messages if event['type'] == 'INCUMBENT'
                         and event.get('source') == 'qubo_joint' and event['ready_sec'] <= frozen['budget_sec']
                         and event['received_sec'] <= frozen['budget_sec']]
                coverage.append(dict(key, mode=mode, status=answer['status'], models=len(models),
                    strict_models=sum(model['energy'] < model['seed_moves'] for model in models),
                    seed_sec=sum(model['seed_metadata']['runtime_sec'] for model in models),
                    setup_sec=sum(model['setup_sec'] for model in models), joint_paths=len(joint),
                    last_phase=phases[-1]['phase'] if phases else None,
                    completed_sec=next((event['elapsed_sec'] for event in messages if event['type']=='DONE'), None)))
                for checkpoint in frozen['checkpoints']:
                    rows.append(dict(key, mode=mode, time_sec=checkpoint, input=data['input'],
                                     moves=at(answer, checkpoint), joint_paths=len([e for e in joint
                                     if e['ready_sec'] <= checkpoint and e['received_sec'] <= checkpoint])))
    table = {(r['case'], r['source_repeat'], r['repeat'], r['mode'], r['time_sec']): r for r in rows}
    contrasts = {}
    rng = random.Random(169101262)
    for role in sorted({item['role'] for item in frozen['cases']}):
        for checkpoint in frozen['checkpoints']:
            for reference in [mode for mode in frozen['modes'] if mode != 'single_qubo']:
                originals = []
                for item in frozen['cases']:
                    if item['role'] != role:
                        continue
                    case, delta_rows = item['entry']['case'], []
                    for spec in frozen['inputs']:
                        if frozen['cases'][spec['original_index']]['entry']['case'] != case:
                            continue
                        for repeat in range(frozen['repeats']):
                            a = table.get((case, spec['source_repeat'], repeat, 'single_qubo', checkpoint))
                            b = table.get((case, spec['source_repeat'], repeat, reference, checkpoint))
                            if a and b:
                                delta_rows.append(dict(source_repeat=spec['source_repeat'], repeat=repeat,
                                                       delta=a['moves'] - b['moves'], joint=a['joint_paths'] > 0))
                    if delta_rows:
                        originals.append(dict(case=case, mean=sum(v['delta'] for v in delta_rows)/len(delta_rows),
                            stable_win=all(v['delta'] < 0 for v in delta_rows),
                            joint_all=all(v['joint'] for v in delta_rows), deltas=delta_rows))
                values = [item['mean'] for item in originals]
                samples = sorted(sum(rng.choices(values, k=len(values)))/len(values)
                                 for _ in range(10000)) if values else []
                contrasts[f'{role}:{checkpoint}:{reference}'] = dict(originals=len(values),
                    wins=sum(v<0 for v in values), losses=sum(v>0 for v in values), ties=sum(v==0 for v in values),
                    net_original_mean=sum(values), worst=max(values, default=0),
                    stable_wins=sum(v['stable_win'] for v in originals),
                    stable_joint_wins=sum(v['stable_win'] and v['joint_all'] for v in originals),
                    by_repeat={repeat:sum(sum(v['delta'] for v in row['deltas'] if v['repeat']==repeat)
                        /sum(v['repeat']==repeat for v in row['deltas']) for row in originals)
                        for repeat in range(frozen['repeats'])},
                    bootstrap95_mean=[samples[250], samples[9749]] if samples else None, details=originals)
    valid = not failures and all(v['status']=='NO_SOURCE_HANDOFF' for v in missing)
    single = contrasts['primary:2.0:single']
    pair = contrasts['primary:2.0:pair']
    screen = (scope=='pilot' and valid and not audits and single['stable_joint_wins']>=2
              and all(v<0 for v in single['by_repeat'].values())
              and all(v<0 for v in pair['by_repeat'].values()))
    result = dict(scope=scope, attempted_originals=len(frozen['cases']), input_snapshots=len(frozen['inputs']),
        budget_gate=valid, audit_pass=not audits, budget_failures=failures, audit_errors=audits,
        missing=missing, contrasts=contrasts, forward_screen=screen,
        scope_boundary='Conditional2s after recorded common handoff;not fresh total-time evidence')
    write_json(out(scope)/'summary.json', result)
    write_csv(out(scope)/'results.csv', rows)
    write_csv(out(scope)/'coverage.csv', coverage)
    print(json.dumps(dict(scope=scope, budget_gate=valid, audit_pass=not audits, forward_screen=screen,
        missing=len(missing), final_contrasts={k:{a:b for a,b in v.items() if a!='details'}
        for k,v in contrasts.items() if ':2.0:' in k})), flush=True)


def run(scope):
    frozen = manifest(scope)
    destination = out(scope)
    if (destination/'status.json').exists() and read(destination/'status.json')['status']=='COMPLETE':
        raise FileExistsError('Study is already complete')
    write_json(destination/'status.json', dict(status='RUNNING', pid=os.getpid(), started=datetime.now().isoformat()))
    def lane(lane_index):
        for index in range(lane_index, len(frozen['inputs']), frozen['jobs']):
            for repeat in range(frozen['repeats']):
                filename = destination/'runs'/f'i{index}__r{repeat}.json'
                if not filename.exists():
                    write_json(filename, execute(scope,index,repeat,frozen['cpus'][lane_index],frozen))
            print(json.dumps(dict(scope=scope,input=index,complete=True)),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=frozen['jobs']) as pool:
        for future in concurrent.futures.as_completed([pool.submit(lane,i) for i in range(frozen['jobs'])]):
            future.result()
    analyze(scope)
    write_json(destination/'status.json',dict(status='COMPLETE',finished=datetime.now().isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('freeze','run','worker','analyze'))
    parser.add_argument('--scope',choices=('smoke','pilot'),required=True)
    parser.add_argument('--input-index',type=int)
    parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--mode',choices=MODES)
    parser.add_argument('--cpu',type=int,default=0)
    args = parser.parse_args()
    if args.command == 'worker':
        worker(args.scope,args.input_index,args.repeat,args.mode,args.cpu)
    else:
        globals()[args.command](args.scope)
