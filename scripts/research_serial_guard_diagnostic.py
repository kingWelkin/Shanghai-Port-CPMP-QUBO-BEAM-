"""Conditional remaining-budget diagnostic; historical upstream is excluded."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT/'scripts', ROOT/'src'):
    sys.path.insert(0, str(directory))
import research_paired_fresh_stream as stream
import research_paper_v43_panel as broad
from audit_deferred_pair import audit_answer as audit_deferred
from audit_serial_guards import audit_answer as audit_serial
from container_reshuffle.core import Move
from cpmp_cpu_layout import homogeneous_layout
from paper_priority_group_contract import PriorityGroupContract
from production_selector_replay import REAL_TIME, fingerprint
from research_receiver_allocation import physical_replay
from research_hardset_qubo_utility import write_json
from run_beam_selector_research import ENVIRONMENT, digest, write_csv
from serial_classical_guards import MODES as SERIAL
from serial_guard_shared_pipeline import run_continuation
from shared_wall_deadline_protocol_full import run_shared, credited_path

BASE = ROOT/'docs/experiments/v50_budget_tail_20261013'
SOURCE = BASE/'screen_t60'
MODES = ('qubo_deferred_pair', 'single_deferred_pair', 'qubo_qsp', 'qubo_qps')
ROLES = ('prefix',)+MODES
CPUS = (0, 2, 4, 6, 8)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def out(scope):
    if scope not in ('smoke', 'diagnostic'):
        raise ValueError(scope)
    return BASE/('serial_guard_'+scope)


def freeze(scope):
    folder = out(scope)
    if (folder/'manifest.json').exists():
        raise FileExistsError('Conditional protocol already frozen')
    parent = read(SOURCE/'manifest.json')
    summary = read(SOURCE/'summary.json')
    if read(SOURCE/'status.json')['status'] != 'COMPLETE' or not summary['budget_gate'] or not summary['audit_pass']:
        raise ValueError('Source study must be complete and valid')
    selected = {row['case']: 'classical_tail' for row in
        summary['contrasts']['all_screen:qubo_deferred_pair:single_deferred_pair']['originals'] if row['mean'] > 0}
    for family in ('BF', 'Open120'):
        case = next(item['entry']['case'] for item in parent['cases']
                    if 'primary' in item['cohorts'] and item['entry']['dataset'] == family)
        selected[case] = 'family_sentinel'
    for case in ('CV__10-6__data10-6-32', 'CV__10-6__data10-6-34'):
        selected[case] = 'pair_rescue_sentinel'
    cases = [dict(entry=item['entry'], role=selected[item['entry']['case']], source_index=index)
             for index, item in enumerate(parent['cases']) if item['entry']['case'] in selected]
    targets, source_repeats, repeats = (60., 120.), (0, 1), 2
    if scope == 'smoke':
        cases = [next(item for item in cases if item['entry']['dataset'] == 'BF')]
        targets, source_repeats, repeats = (60.,), (0,), 1
    else:
        smoke = read(out('smoke')/'summary.json')
        if not smoke['budget_gate'] or not smoke['audit_pass']:
            raise ValueError('Conditional wiring smoke has not passed')
    sources = dict(parent['source_sha256'])
    for name in ('serial_classical_guards.py', 'serial_guard_shared_pipeline.py',
                 'audit_serial_guards.py', 'research_serial_guard_diagnostic.py'):
        sources['scripts/'+name] = digest(ROOT/'scripts'/name)
    inputs = []
    for index, item in enumerate(cases):
        for source_repeat in source_repeats:
            source_file = SOURCE/'runs'/f'c{item["source_index"]}__r{source_repeat}.json'
            original = read(source_file)
            handoff = original['handoff']
            if handoff is None:
                raise ValueError('Selected conditional input is unavailable')
            snapshot = dict(case=item['entry']['case'], source=str(source_file.relative_to(ROOT)),
                source_sha256=digest(source_file), source_repeat=source_repeat,
                source_budget_sec=60., prior_handoff_ready_sec=handoff['ready_sec'],
                moves=handoff['moves'], path=handoff['path'], path_signature=handoff['path_signature'])
            filename = f'inputs/c{index}__s{source_repeat}.json'
            write_json(folder/filename, snapshot)
            for target in targets:
                residual = target-handoff['ready_sec']
                if residual <= 0:
                    raise ValueError('No residual allowance')
                inputs.append(dict(original_index=index, source_repeat=source_repeat,
                    nominal_total_sec=target, budget_sec=residual, input_file=filename,
                    input_sha256=digest(folder/filename)))
    offsets = [index % len(ROLES) for index in range(len(inputs)*repeats)]
    random.Random(2026101341).shuffle(offsets)
    protocol = (
        'CONDITIONAL DIAGNOSTIC, NOT FRESH RAW-BAY TOTAL-TIME EVIDENCE. Given only the saved '
        'native/common HANDOFF from the valid60s study, rerun all refinement work and guards. '
        'No historical winning refinement answer is supplied. Select ALL two originals with '
        'positive mean loss to deferredSingle, include BOTH source repeats, plus fixed BF/Open '
        'family sentinels and the two known Pair-rescue originals32/34. Report diagnostic and '
        'sentinel roles separately. Two fresh execution repeats at both nominal60/120 residual '
        'allowances: allowance=nominal target minus recorded historical HANDOFF ready time. '
        'Historical upstream is excluded;120 uses the SAME60s-produced path, not a new120s '
        'prefix. A fresh real START is the beginning of this conditional interval; no virtual '
        'request-time clock or free new computation. Input verification/imports precede READY; '
        'raw bay loading, input replay, IPC, all output cleanup, seeds, models, kernels, FRG '
        'and every independent trajectory are paid after START. Five actors on five distinct '
        'homogeneous cores, rotated roles. The prefix role is ONLY a given-input loader, not '
        'a fresh solver. Compare unchanged deferredQ->Pair and Single->Pair with Q->Single->Pair '
        '(predeclared repair target) and Q->Pair->Single (order control). Every trajectory '
        'starts from the actual given common path and retains its own strictly shorter path. '
        'Proceed to the next trajectory only after a full-pass stall, if time remains. '
        'No global or branch clock reset; unchanged local50ms/50000-node Q cap. Only reported '
        'INCUMBENT ready AND received by deadline counts. Audit raw/reported ledgers, every '
        'physical path, actual seed starts and branch input/order. Retain all losses/cutoffs. '
        'A mechanism repair requires repeated fresh strict gains versus deferredQ and actual '
        'Single guard publications on the originally losing source input; it does not promote '
        'a default or establish full169 generalization. Never overlap this timed diagnostic '
        'with the active fresh20/60/120 budget suite.')
    write_json(folder/'manifest.json', dict(cases=cases, inputs=inputs, repeats=repeats,
        modes=MODES, roles=ROLES, cpus=CPUS, cpu_layout=homogeneous_layout(CPUS),
        offsets=offsets, source_sha256=sources, protocol=protocol, frozen_at=datetime.now().isoformat(),
        experiment_kind='CONDITIONAL_HISTORICAL_HANDOFF', upstream_excluded=True))
    print(json.dumps(dict(scope=scope, originals=len(cases), inputs=len(inputs), repeats=repeats)), flush=True)


def manifest(scope):
    frozen = read(out(scope)/'manifest.json')
    for path, expected in frozen['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Frozen source changed: '+path)
    homogeneous_layout(frozen['cpus'], frozen['cpu_layout'])
    return frozen


def worker(scope, index, repeat, role, cpu):
    frozen = manifest(scope)
    spec = frozen['inputs'][index]
    item = frozen['cases'][spec['original_index']]
    path = out(scope)/spec['input_file']
    if digest(path) != spec['input_sha256'] or digest(ROOT/item['entry']['path']) != item['entry']['sha256']:
        raise ValueError('Conditional input or raw bay changed')
    raw = read(path)
    broad.pin_cpu(cpu)
    publish = stream.captured_publisher()
    publish(dict(type='READY', role=role, case=raw['case'], cpu=cpu,
        cpu_info=next(row for row in frozen['cpu_layout'] if row['logical_processor'] == cpu),
        historical_input=True))
    clock = json.loads(sys.stdin.readline())
    start, budget = clock['start'], clock['budget']
    if budget != spec['budget_sec']:
        raise ValueError('Changed conditional allowance')
    publish(dict(type='START_ACK', absolute_start=start, budget_sec=budget))
    contract = PriorityGroupContract()
    contract.install()
    try:
        instance = broad.load_case(item['entry'])
        if role == 'prefix':
            physical_replay(instance.stacks, instance.max_height, tuple(Move(**value) for value in raw['path']), goal=True)
            if fingerprint(raw['path']) != raw['path_signature']:
                raise ValueError('Historical input signature changed')
            publish(dict(type='INCUMBENT', source='given_input', moves=raw['moves'],
                path=raw['path'], ready_sec=REAL_TIME()-start))
            publish(dict(type='HANDOFF', path=raw['path'], moves=raw['moves'],
                path_signature=raw['path_signature'], ready_sec=REAL_TIME()-start,
                absolute_start=start, budget_sec=budget, historical_input=True))
            result = dict(given_input_only=True, elapsed_sec=REAL_TIME()-start)
        else:
            message = json.loads(sys.stdin.readline())
            if message['type'] == 'STOP':
                result = dict(skipped=True, reason=message['reason'])
            else:
                if message['start'] != start or message['budget'] != budget:
                    raise ValueError('Parent rebased the conditional clock')
                result = run_continuation(instance, role, message['handoff'], start, budget, publish)
        publish(dict(type='DONE', **result))
    except BaseException:
        publish(dict(type='ERROR', detail=traceback.format_exc(), elapsed_sec=REAL_TIME()-start))
        raise
    finally:
        contract.restore()


def execute(scope, index, repeat, frozen):
    spec = frozen['inputs'][index]
    item = frozen['cases'][spec['original_index']]
    offset = frozen['offsets'][index*frozen['repeats']+repeat]
    order = list(ROLES[offset:]+ROLES[:offset])
    assignments = {role: frozen['cpus'][position] for position, role in enumerate(order)}
    commands = {role: [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--scope', scope,
        '--input-index', str(index), '--repeat', str(repeat), '--role', role, '--cpu', str(cpu)]
        for role, cpu in assignments.items()}
    budget = spec['budget_sec']
    data = run_shared(commands, cwd=ROOT, env={**os.environ, **ENVIRONMENT}, budget=budget,
                      handoff_roles=MODES)
    instance = broad.load_case(item['entry'])
    checks = 0
    errors = []
    for role, actor in data['actors'].items():
        for event in actor['messages']:
            if event['type'] not in ('INCUMBENT', 'REFINEMENT_INCUMBENT', 'HANDOFF', 'TRAJECTORY_END'):
                continue
            try:
                physical_replay(instance.stacks, instance.max_height,
                    tuple(Move(**value) for value in event['path']), goal=True)
                checks += 1
                count = event['after'] if event['type'] == 'TRAJECTORY_END' else event['moves']
                if count != len(event['path']):
                    raise ValueError('Wrong physical cost')
            except (ValueError, IndexError) as error:
                errors.append(dict(role=role, reason=str(error)))
    answers = {}
    for mode in MODES:
        best = credited_path(data['actors']['prefix']['messages'], data['actors'][mode]['messages'], budget)
        answers[mode] = dict(valid=best is not None, moves=None if best is None else best['moves'],
            path=[] if best is None else best['path'], provenance=None if best is None else best.get('refinement_provenance'))
    return dict(case=item['entry']['case'], dataset=item['entry']['dataset'], role=item['role'],
        source_repeat=spec['source_repeat'], nominal_total_sec=spec['nominal_total_sec'],
        input_index=index, repeat=repeat, status='COMPLETE', cpu_assignments=assignments,
        answers=answers, physical_checks=checks, physical_errors=errors,
        experiment_kind='CONDITIONAL_HISTORICAL_HANDOFF', **data)


def analyze(scope):
    frozen = manifest(scope)
    rows, errors, failures, audits = [], [], [], []
    physical = 0
    for index, spec in enumerate(frozen['inputs']):
        for repeat in range(frozen['repeats']):
            data = read(out(scope)/'runs'/f'i{index}__r{repeat}.json')
            key = {name: data[name] for name in ('case', 'source_repeat', 'nominal_total_sec', 'repeat')}
            if data['status'] != 'COMPLETE':
                failures.append(dict(key, status=data['status']))
                continue
            budget = spec['budget_sec']
            physical += data['physical_checks']
            errors.extend(dict(key, **error) for error in data['physical_errors'])
            errors.extend(dict(key, **error) for error in stream.stream_errors(data, budget))
            if data['issues'] or data['termination_lag_sec'] > .1:
                failures.append(dict(key, issues=data['issues'], lag=data['termination_lag_sec']))
            if len(set(data['cpu_assignments'].values())) != len(ROLES):
                errors.append(dict(key, reason='Shared actor CPU'))
            expected = {row['logical_processor']: row for row in frozen['cpu_layout']}
            signatures = []
            for role, actor in data['actors'].items():
                ready = actor['ready']
                if ready['cpu_info'] != expected[data['cpu_assignments'][role]]:
                    errors.append(dict(key, mode=role, reason='Changed CPU layout'))
                acks = [event for event in actor['messages'] if event['type']=='START_ACK' and event['received_sec']<=budget]
                if len(acks)!=1 or any(event['absolute_start']!=data['absolute_start'] or event['budget_sec']!=budget for event in acks):
                    errors.append(dict(key, mode=role, reason='Changed original conditional clock'))
                if role == 'prefix':
                    continue
                view = stream.study.raw_ledger_view(actor, budget)
                if view is not None:
                    audit = audit_serial if role in SERIAL else audit_deferred
                    checked = audit(view, view['initial_path'], role, budget)
                    errors.extend(dict(key, mode=role, reason=reason) for reason in checked['errors'])
                    audits.append(dict(key, mode=role, **checked))
                entries = [event for event in actor['messages'] if event['type']=='BRANCH_INPUT' and event['received_sec']<=budget]
                for entry in entries:
                    if data['handoff'] is None or entry['path_signature']!=data['handoff']['path_signature']:
                        errors.append(dict(key, mode=role, reason='Changed given input'))
                    if entry['model_input_signature'] is not None:
                        signatures.append(entry['model_input_signature'])
                best = credited_path(data['actors']['prefix']['messages'], actor['messages'], budget)
                answer = data['answers'][role]
                if answer['moves'] != (None if best is None else best['moves']):
                    errors.append(dict(key, mode=role, reason='Late/unpublished answer credited'))
                rows.append(dict(key, dataset=data['dataset'], role=data['role'], mode=role,
                    budget_sec=budget, moves=answer['moves'], valid=answer['valid'], status=actor['status'],
                    final_policy=(answer.get('provenance') or {}).get('trajectory_policy')))
            if len(set(signatures)) > 1:
                errors.append(dict(key, reason='Different initial model inputs'))
    table = {(row['case'], row['source_repeat'], row['nominal_total_sec'], row['repeat'], row['mode']): row for row in rows}
    contrasts = []
    for item in frozen['cases']:
        case = item['entry']['case']
        for target in sorted({spec['nominal_total_sec'] for spec in frozen['inputs']}):
            for reference in ('qubo_deferred_pair', 'single_deferred_pair', 'qubo_qps'):
                details = []
                for spec in frozen['inputs']:
                    if frozen['cases'][spec['original_index']]['entry']['case']!=case or spec['nominal_total_sec']!=target:
                        continue
                    for repeat in range(frozen['repeats']):
                        k = (case, spec['source_repeat'], target, repeat)
                        q, p = table.get(k+('qubo_qsp',)), table.get(k+(reference,))
                        if q and p and q['valid'] and p['valid']:
                            details.append(dict(source_repeat=spec['source_repeat'], repeat=repeat,
                                qsp=q['moves'], reference=p['moves'], delta=q['moves']-p['moves'], final_policy=q['final_policy']))
                contrasts.append(dict(case=case, role=item['role'], nominal_total_sec=target,
                    reference=reference, mean_delta=sum(row['delta'] for row in details)/len(details) if details else None,
                    deltas=details))
    result = dict(scope=scope, originals=len(frozen['cases']), inputs=len(frozen['inputs']),
        repeats=frozen['repeats'], budget_gate=not failures and len(rows)==len(frozen['inputs'])*frozen['repeats']*len(MODES),
        audit_pass=not errors, failures=failures, errors=errors, physical_path_checks=physical,
        valid_runs=dict(Counter(row['mode'] for row in rows if row['valid'])), contrasts=contrasts,
        scope_boundary='Conditional only: historical upstream excluded; no fresh total-time or promotion claim')
    write_json(out(scope)/'summary.json', result)
    write_json(out(scope)/'branch_audit.json', dict(rows=audits, errors=errors))
    write_csv(out(scope)/'results.csv', rows)
    print(json.dumps({key: value for key, value in result.items() if key != 'contrasts'}), flush=True)


def run(scope):
    if read(BASE/'screen_t120/status.json')['status'] != 'COMPLETE':
        raise ValueError('Do not overlap timed studies')
    frozen = manifest(scope)
    if (out(scope)/'status.json').exists():
        raise FileExistsError('Conditional run already started; retain existing records')
    write_json(out(scope)/'status.json', dict(status='RUNNING', pid=os.getpid(), started=datetime.now().isoformat()))
    for index, spec in enumerate(frozen['inputs']):
        for repeat in range(frozen['repeats']):
            try:
                data = execute(scope, index, repeat, frozen)
            except Exception:
                data = dict(status='WORKER_ERROR', input_index=index, repeat=repeat, error=traceback.format_exc())
            write_json(out(scope)/'runs'/f'i{index}__r{repeat}.json', data)
            print(json.dumps(dict(scope=scope, input=index, repeat=repeat, status=data['status'])), flush=True)
            if data['status']!='COMPLETE' or data.get('issues') or data.get('physical_errors'):
                write_json(out(scope)/'status.json', dict(status='STOPPED_ON_FAILURE', input=index, repeat=repeat))
                return
    analyze(scope)
    write_json(out(scope)/'status.json', dict(status='COMPLETE', finished=datetime.now().isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'run', 'worker', 'analyze'))
    parser.add_argument('--scope', choices=('smoke', 'diagnostic'), required=True)
    parser.add_argument('--input-index', type=int)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--cpu', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'worker':
        worker(args.scope, args.input_index, args.repeat, args.role, args.cpu)
    else:
        globals()[args.command](args.scope)
