"""Separate frozen studies of longer native budgets and deferred Pair tails."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT/'scripts', ROOT/'src'):
    sys.path.insert(0, str(directory))

import research_paired_fresh_stream as stream
from audit_deferred_pair import audit_answer as audit_deferred
from deferred_pair_refinement import MODES as DEFERRED
from deferred_pair_shared_pipeline import run_continuation
from cpmp_cpu_layout import homogeneous_layout
from production_selector_replay import fingerprint
from research_paired_trajectory_screen import audit_answer as audit_old
from run_beam_selector_research import digest, write_csv
from shared_wall_deadline_protocol_full import credited_path

study = stream.study
LEGACY = study.BASE
BASE = ROOT/'docs/experiments/v50_budget_tail_20261013'
STANDALONE = ('native',)
CONTINUATIONS = ('qubo_cycle', 'qubo_pair', 'qubo_deferred_pair', 'single_deferred_pair')
MODES = STANDALONE + CONTINUATIONS
ROLES = ('prefix',) + MODES
read = study.read
write_json = study.write_json


def out(scope):
    if not re.fullmatch(r'(smoke|screen|hardset169)_t(20|60|120)', scope):
        raise ValueError('Unknown frozen scope: '+scope)
    return BASE/scope


def selected_cases(full):
    inventory = {item['entry']['case']: item['entry'] for item in full['cases']}
    cohorts = {}
    def add(case, cohort):
        if case not in inventory:
            raise ValueError('Selected case missing from hardset169')
        cohorts.setdefault(case, []).append(cohort)
    for item in read(LEGACY/'paired_fresh_stream_pilot/manifest.json')['cases']:
        add(item['entry']['case'], 'primary')
    summary = read(LEGACY/'paired_fresh_exploratory_hardset169/summary.json')
    for row in summary['contrasts']['native']['originals']:
        if row['mean'] > 0:
            add(row['case'], 'known_native_tail')
    tails = read(LEGACY/'paired_solo_fresh_hardset169/tail_mechanism_examples.json')
    for case in sorted({row['case'] for row in tails['rows']}):
        add(case, 'known_schedule_or_basin_tail')
    return [dict(entry=inventory[case], cohorts=labels) for case, labels in cohorts.items()]


def freeze(scope):
    folder = out(scope)
    if (folder/'manifest.json').exists():
        raise FileExistsError('Study already frozen')
    parent = read(LEGACY/'paired_solo_fresh_hardset169/manifest.json')
    cases = selected_cases(parent)
    budget = float(scope.rsplit('_t', 1)[1])
    repeats = 2
    if scope.startswith('smoke'):
        cases = [next(item for item in cases if 'primary' in item['cohorts'] and
                      item['entry']['dataset'] == family) for family in ('BF', 'CV', 'Open120')]
        repeats = 1
    elif scope.startswith('screen'):
        smoke = read(out('smoke_t20')/'summary.json')
        if not smoke['budget_gate'] or not smoke['audit_pass']:
            raise ValueError('Protocol smoke has not passed')
    else:
        screen = read(out('screen_t120')/'summary.json')
        if not screen['budget_gate'] or not screen['audit_pass']:
            raise ValueError('Long-budget screen is not valid')
        if not screen['candidate_development_screen']:
            raise ValueError('Deferred candidate did not justify full169 expansion')
        membership = {item['entry']['case']: item['cohorts'] for item in cases}
        cases = [dict(entry=item['entry'], cohorts=['full_hardset']+
                      membership.get(item['entry']['case'], [])) for item in parent['cases']]
    sources = dict(parent['source_sha256'])
    for name in ('deferred_pair_refinement.py', 'deferred_pair_shared_pipeline.py',
                 'audit_deferred_pair.py', 'research_v50_budget_tail.py'):
        sources['scripts/'+name] = digest(ROOT/'scripts'/name)
    offsets = [index % len(ROLES) for index in range(len(cases)*repeats)]
    random.Random(2026101320).shuffle(offsets)
    protocol = (
        'Frozen longer-budget and tail development study. Native v4.3 has its entire declared '
        '20/60/120-second budget and receives no handoff. Four continuations use one separately '
        'computed fresh v4.3 prefix under the same original clock, retaining the cooperative '
        '2-second reserve. All six actors use different homogeneous physical cores and rotate '
        'roles. Imports/source checks precede READY; instance loading, all computation, common '
        'physical validation/output cleanup and IPC are charged after START. Only reported '
        'INCUMBENT ready AND received by the original deadline is credited. No historical '
        'solution, free Pair answer, best-of-repeats or clock reset. Keep original solo Q and '
        'alternating Q+Pair unchanged. Candidate runs primary Q to a no-strict-improvement '
        'complete pass, then independent Pair from the original common path. Matched classical '
        'candidate substitutes three-order Single for Q with identical scheduling. All joint '
        'models/seeds/local50ms/50000-node caps remain unchanged. Physical replay every path, '
        'audit actual Beam, raw/reported ledgers, first-input identity, branch clocks and '
        'deferred switches. Retain all failures, cutoffs and losses. Screen combines the '
        'unchanged12 development originals, all9 old mean QP-native losses, and all saved '
        'mean-loss-at-least10 scheduling/basin tail originals. Preserve separate cohort labels; '
        'this selection is not independent generalization. Two repeats at every budget; '
        'original means only when both runs succeed; all attempts remain in coverage. '
        '60/120 screens are completed even after negative20s quality if protocol remains valid. '
        'Candidate expansion screen at120: valid audits; no feasibility loss against soloQ, '
        'oldQP or native; negative primary mean-total versus native; nonpositive primary and '
        'all-screen totals versus soloQ and oldQP; at least2 stable oldQP-relative tail wins. '
        'These are development criteria, not full-corpus promotion. On failure retain this '
        'candidate; do not modify the frozen sources or rescue it by filtering cases. '
        'No default promotion or hardware claim.')
    write_json(folder/'manifest.json', dict(cases=cases, modes=MODES, roles=ROLES,
        repeats=repeats, cpus=study.CPUS, cpu_layout=homogeneous_layout(study.CPUS),
        offsets=offsets, budget_sec=budget, reserve_sec=2., source_sha256=sources,
        frozen_at=datetime.now().isoformat(), protocol=protocol,
        cohorts=dict(Counter(label for item in cases for label in item['cohorts'])),
        selection_sources=['paired_fresh_stream_pilot/manifest.json',
            'paired_fresh_exploratory_hardset169/summary.json',
            'paired_solo_fresh_hardset169/tail_mechanism_examples.json']))
    print(json.dumps(dict(scope=scope, originals=len(cases), repeats=repeats, budget=budget,
        cohorts=dict(Counter(label for item in cases for label in item['cohorts'])))), flush=True)


def contrast(rows, cases, repeats, target, reference, seed=2026101321):
    table = {(row['case'], row['repeat'], row['mode']): row for row in rows}
    originals = []
    gained = []
    lost = []
    for item in cases:
        case = item['entry']['case']
        deltas = []
        exposed = []
        target_costs = []
        reference_costs = []
        for repeat in range(repeats):
            q, p = table.get((case, repeat, target)), table.get((case, repeat, reference))
            if q is None or p is None:
                continue
            if q['valid'] and not p['valid']:
                gained.append(dict(case=case, repeat=repeat))
            if p['valid'] and not q['valid']:
                lost.append(dict(case=case, repeat=repeat))
            if q['valid'] and p['valid']:
                deltas.append(q['moves']-p['moves'])
                target_costs.append(q['moves'])
                reference_costs.append(p['moves'])
                exposed.append(q['joint_publications'] > 0)
        if len(deltas) == repeats:
            originals.append(dict(case=case, deltas=deltas, mean=sum(deltas)/repeats,
                target_mean=sum(target_costs)/repeats, reference_mean=sum(reference_costs)/repeats,
                stable_win=all(value < 0 for value in deltas), joint_all=all(exposed)))
    values = [row['mean'] for row in originals]
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(values, k=len(values)))/len(values) for _ in range(10000)) if values else []
    return dict(target=target, reference=reference, common_originals=len(values),
        wins=sum(value < 0 for value in values), losses=sum(value > 0 for value in values),
        ties=sum(value == 0 for value in values), net_original_mean=sum(values),
        worst=max(values, default=0), mean_delta=sum(values)/len(values) if values else None,
        target_mean=sum(row['target_mean'] for row in originals)/len(values) if values else None,
        reference_mean=sum(row['reference_mean'] for row in originals)/len(values) if values else None,
        by_repeat={repeat: sum(row['deltas'][repeat] for row in originals) for repeat in range(repeats)},
        stable_wins=sum(row['stable_win'] for row in originals),
        stable_wins_with_joint=sum(row['stable_win'] and row['joint_all'] for row in originals),
        bootstrap95_mean=[samples[250], samples[9749]] if samples else None,
        gained_solutions=gained, lost_solutions=lost, originals=originals)


def analyze(scope):
    frozen = study.manifest(scope)
    budget = frozen['budget_sec']
    errors, failures, missing, rows, branch_audits = [], [], [], [], []
    physical = producer_beams = native_beams = 0
    max_lag = 0.
    allocations = Counter()
    expected_cpus = {row['logical_processor']: row for row in frozen['cpu_layout']}
    for index, item in enumerate(frozen['cases']):
        for repeat in range(frozen['repeats']):
            data = read(out(scope)/'runs'/f'c{index}__r{repeat}.json')
            key = dict(case=item['entry']['case'], repeat=repeat)
            if data['status'] != 'COMPLETE':
                failures.append(dict(key, status=data['status']))
                continue
            def require(condition, reason):
                if not condition:
                    errors.append(dict(key, reason=reason))
            physical += data['physical_checks']
            errors.extend(dict(key, **error) for error in data['physical_errors'])
            errors.extend(dict(key, **error) for error in stream.stream_errors(data, budget))
            max_lag = max(max_lag, data['termination_lag_sec'])
            if data['issues'] or data['termination_lag_sec'] > .1:
                failures.append(dict(key, issues=data['issues'], lag=data['termination_lag_sec']))
            require(len(set(data['cpu_assignments'].values())) == len(ROLES), 'Shared actor CPU slot')
            for role, cpu in data['cpu_assignments'].items():
                allocations[(role, cpu)] += 1
                ready = data['actors'][role]['ready']
                require(ready['cpu'] == cpu and ready['cpu_info'] == expected_cpus[cpu], 'Wrong actual CPU layout')
                acks = [event for event in data['actors'][role]['messages']
                        if event['type'] == 'START_ACK' and event['received_sec'] <= budget]
                require(len(acks) == 1 and all(event['absolute_start'] == data['absolute_start'] and
                        event['budget_sec'] == budget for event in acks), 'Missing or changed original clock')
            prefix = [event for event in data['actors']['prefix']['messages'] if event['received_sec'] <= budget]
            commits = [event for event in prefix if event['type'] == 'BEAM']
            producer_beams += len(commits)
            require(all(event['actual_frontier_verified'] for event in commits), 'Unverified producer Beam')
            if credited_path(prefix, [], budget) is None:
                missing.append(dict(key, status='NO_FRESH_PREFIX_SOLUTION'))
            signatures = []
            for mode in MODES:
                actor = data['actors'][mode]
                events = [event for event in actor['messages'] if event['received_sec'] <= budget]
                if mode == 'native':
                    require(mode not in data['handoff_sent_sec'], 'Native received a shared answer')
                    beams = [event for event in events if event['type'] == 'BEAM']
                    native_beams += len(beams)
                    require(all(event['actual_frontier_verified'] for event in beams), 'Unverified native Beam')
                for entry in [event for event in events if event['type'] == 'BRANCH_INPUT']:
                    require(data['handoff'] is not None and entry['path_signature'] == data['handoff']['path_signature']
                        and entry['absolute_start'] == data['absolute_start'] and entry['budget_sec'] == budget,
                        'Changed continuation input/clock')
                    if entry['model_input_signature'] is not None:
                        signatures.append(entry['model_input_signature'])
                if mode in CONTINUATIONS:
                    try:
                        view = study.raw_ledger_view(actor, budget)
                        if view is not None:
                            audit = audit_deferred if mode in DEFERRED else audit_old
                            checked = audit(view, view['initial_path'], mode, budget)
                            errors.extend(dict(key, mode=mode, reason=reason) for reason in checked['errors'])
                            branch_audits.append(dict(key, mode=mode, **checked))
                    except ValueError as error:
                        errors.append(dict(key, mode=mode, reason=str(error)))
                publications = [event for event in events if event['type'] == 'INCUMBENT' and event['ready_sec'] <= budget]
                for event in publications:
                    provenance = event.get('refinement_provenance')
                    if provenance:
                        require(event['moves'] <= provenance['raw_moves'], 'Reported result exceeds raw source cost')
                source = [] if mode == 'native' else prefix
                best = credited_path(source, actor['messages'], budget)
                answer = data['answers'][mode]
                require(answer['moves'] == (None if best is None else best['moves']), 'Credited raw/late/unpublished path')
                joint = sum((event.get('refinement_provenance') or {}).get('original_source') == 'qubo_joint'
                            for event in publications)
                rows.append(dict(key, dataset=data['dataset'], cohorts=';'.join(item['cohorts']),
                    mode=mode, budget_sec=budget, valid=answer['valid'], moves=answer['moves'],
                    joint_publications=joint, status=actor['status'], source=answer['source']))
            require(len(set(signatures)) <= 1, 'Different initial model inputs across continuations')
    pairs = [('qubo_deferred_pair', reference) for reference in ('native', 'qubo_cycle', 'qubo_pair', 'single_deferred_pair')]
    pairs += [('qubo_cycle', 'native'), ('qubo_pair', 'native')]
    cohorts = ['all_screen'] + sorted({label for item in frozen['cases'] for label in item['cohorts']})
    contrasts = {}
    for cohort in cohorts:
        cases = [item for item in frozen['cases'] if cohort == 'all_screen' or cohort in item['cohorts']]
        for target, reference in pairs:
            contrasts[f'{cohort}:{target}:{reference}'] = contrast(rows, cases, frozen['repeats'], target, reference)
    valid = not failures and len(rows) == len(frozen['cases'])*frozen['repeats']*len(MODES)
    candidate = valid and not errors and scope != 'smoke_t20'
    if candidate:
        primary_native = contrasts['primary:qubo_deferred_pair:native']
        candidate = primary_native['net_original_mean'] < 0
        for reference in ('native', 'qubo_cycle', 'qubo_pair'):
            candidate = candidate and not contrasts[f'all_screen:qubo_deferred_pair:{reference}']['lost_solutions']
        for cohort in ('primary', 'all_screen'):
            for reference in ('qubo_cycle', 'qubo_pair'):
                candidate = candidate and contrasts[f'{cohort}:qubo_deferred_pair:{reference}']['net_original_mean'] <= 0
        candidate = candidate and contrasts['known_schedule_or_basin_tail:qubo_deferred_pair:qubo_pair']['stable_wins'] >= 2
    summary = dict(scope=scope, budget_sec=budget, originals=len(frozen['cases']), repeats=frozen['repeats'],
        budget_gate=valid, audit_pass=not errors, budget_failures=failures, audit_errors=errors,
        missing_common_prefix=missing, max_parent_lag_sec=max_lag, physical_path_checks=physical,
        actual_beam_checks=producer_beams, standalone_beam_checks=native_beams,
        valid_runs={mode: sum(row['valid'] for row in rows if row['mode'] == mode) for mode in MODES},
        role_cpu_counts={role: {str(cpu): allocations[(role, cpu)] for cpu in frozen['cpus']} for role in ROLES},
        contrasts=contrasts, candidate_development_screen=bool(candidate),
        scope_boundary='Selected development/tail screen unless explicit full_hardset cohort; no production promotion')
    write_json(out(scope)/'summary.json', summary)
    write_json(out(scope)/'branch_audit.json', dict(rows=branch_audits, errors=errors))
    write_csv(out(scope)/'results.csv', rows)
    print(json.dumps({key: value for key, value in summary.items() if key not in ('contrasts', 'role_cpu_counts')},
                     ensure_ascii=False), flush=True)


study.STANDALONE = STANDALONE
study.CONTINUATIONS = CONTINUATIONS
study.MODES = MODES
study.ROLES = ROLES
study.out = out
study.__file__ = __file__
study.run_continuation = run_continuation
study.analyze = analyze


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'run', 'worker', 'analyze', 'suite'))
    parser.add_argument('--scope')
    parser.add_argument('--scopes', nargs='+')
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--cpu', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'suite':
        if not args.scopes:
            parser.error('suite requires --scopes')
        for scope in args.scopes:
            study.run(scope)
            summary = read(out(scope)/'summary.json') if (out(scope)/'summary.json').exists() else None
            if summary is None or not summary['budget_gate'] or not summary['audit_pass']:
                raise RuntimeError('Suite stopped on invalid execution: '+scope)
    else:
        if not args.scope:
            parser.error('--scope is required')
        if args.command == 'worker':
            stream.worker(args.scope, args.case_index, args.repeat, args.role, args.cpu)
        elif args.command == 'run':
            study.run(args.scope)
        else:
            globals()[args.command](args.scope)
