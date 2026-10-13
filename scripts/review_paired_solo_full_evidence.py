"""Independent complete-cohort scoring and solo joint-lineage review."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'scripts', ROOT / 'src'):
    sys.path.insert(0, str(directory))
from audit_cyclic_refinement_lineage import trace
from audit_shared_prefix_lineage import semantic
from container_reshuffle.core import Move
from research_paired_fresh import raw_ledger_view
import research_paper_v43_panel as broad
from research_receiver_allocation import physical_replay
from shared_wall_deadline_protocol_full import credited_path

FOLDER = ROOT / 'docs/experiments/action_coupling_qubo_20261011/paired_solo_fresh_hardset169'
MODES = ('single_cycle', 'pair_cycle', 'qubo_cycle', 'single_pair', 'qubo_pair')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def solo_lineage(data, budget):
    qview = raw_ledger_view(data['actors']['qubo_cycle'], budget)
    sview = raw_ledger_view(data['actors']['single_cycle'], budget)
    qe, qm, _ = trace(qview, qview['initial_path'], budget) if qview else ({}, [], 0)
    se, sm, _ = trace(sview, sview['initial_path'], budget) if sview else ({}, [], 0)
    lookup = {(m['cycle'], m['index']): m for m in sm}
    first = next((m for m in qm if m['energy'] < m['seed_moves']), None)
    checks, entries, violations = [], [], []
    if first:
        stop = first['cycle'], first['index']
        entries = [dict(cycle=c, matched=c in qe and c in se and qe[c] == se[c])
                   for c in range(first['cycle'] + 1)]
        for q in qm:
            key = q['cycle'], q['index']
            if key > stop:
                break
            s = lookup.get(key)
            matched = bool(s and all(q[k] == s[k] for k in ('before', 'seed_moves', 'variables'))
                and semantic(q['model']) == semantic(s['model'])
                and semantic(q['seed_metadata']) == semantic(s['seed_metadata']))
            if matched and key < stop:
                matched = q['selected'] == s['selected'] and q['energy'] == s['energy']
            if matched and key == stop:
                matched = q['seed_metadata']['actual_seed_selection'] == s['selected']
            checks.append(dict(cycle=key[0], index=key[1], matched=matched))
        if first['seed_kind'] != 'three_order_single' or first['changed_groups_from_seed'] < 2:
            violations.append('Solo Q joint gain contradicts its actual single seed')
    matched = bool(first and checks and all(c['matched'] for c in entries + checks))
    reported = sum(e['type'] == 'INCUMBENT' and e['received_sec'] <= budget and e['ready_sec'] <= budget
        and (e.get('refinement_provenance') or {}).get('original_source') == 'qubo_joint'
        for e in data['actors']['qubo_cycle']['messages'])
    return dict(case=data['case'], repeat=data['repeat'], matched_first_joint=matched,
        status='NO_RECEIVED_JOINT_MODEL' if first is None else 'MATCHED_CYCLE_AND_SINGLE_PREFIX'
            if matched else 'INCOMPLETE_PREFIX', reported_joint_publications=reported,
        violations=violations, cycle_entry_checks=entries, model_checks=checks)


def contrast(cases, runs, repeats, target, reference, rng):
    originals = []
    outcomes = Counter()
    for item in cases:
        case = item['entry']['case']
        deltas, case_outcomes = [], []
        for repeat in range(repeats):
            q, r = runs[(case, repeat, target)], runs[(case, repeat, reference)]
            delta = None
            if q['valid'] and r['valid']:
                delta = q['moves'] - r['moves']
                outcome = 'quality_win' if delta < 0 else 'quality_loss' if delta > 0 else 'quality_tie'
            elif q['valid']:
                outcome = 'feasibility_win'
            elif r['valid']:
                outcome = 'feasibility_loss'
            else:
                outcome = 'both_unresolved'
            deltas.append(delta)
            case_outcomes.append(outcome)
            outcomes[outcome] += 1
        complete = all(d is not None for d in deltas)
        originals.append(dict(case=case, dataset=item['entry']['dataset'], deltas=deltas,
            mean=sum(deltas) / repeats if complete else None, outcomes=case_outcomes,
            stable_win=complete and all(d < 0 for d in deltas)))
    common = [r for r in originals if r['mean'] is not None]
    values = [r['mean'] for r in common]
    samples = sorted(sum(rng.choices(values, k=len(values))) / len(values)
                     for _ in range(10000)) if values else []
    return dict(target=target, reference=reference, attempted_originals=len(cases),
        common_originals=len(common), wins=sum(v < 0 for v in values), losses=sum(v > 0 for v in values),
        ties=sum(v == 0 for v in values), net_original_mean=sum(values),
        by_repeat={str(rep): sum(row['deltas'][rep] for row in common) for rep in range(repeats)},
        worst=max(values, default=0), stable_wins=sum(r['stable_win'] for r in common),
        bootstrap95_mean=[samples[250], samples[9749]] if samples else None,
        execution_outcomes=dict(outcomes),
        family_net={family: sum(r['mean'] for r in common if r['dataset'] == family)
                    for family in ('BF', 'CV', 'Open120')},
        largest_losses=sorted([r for r in common if r['mean'] > 0], key=lambda r: -r['mean'])[:15],
        originals=originals)


def main():
    if read(FOLDER / 'status.json')['status'] != 'COMPLETE':
        raise ValueError('Do not analyze the incomplete timed cohort')
    manifest, summary = read(FOLDER / 'manifest.json'), read(FOLDER / 'summary.json')
    paired_lineage = read(FOLDER / 'lineage_audit.json')
    cases, repeats = manifest['cases'], manifest['repeats']
    if (len(cases) != 169 or len({c['entry']['case'] for c in cases}) != 169 or repeats != 2
        or Counter(c['entry']['dataset'] for c in cases) != dict(BF=73, CV=84, Open120=12)
        or tuple(manifest['modes']) != MODES):
        raise ValueError('The full confirmed corpus and control panel are required')
    checks, runs, solo_rows = [], {}, []
    coverage, models, final_sources = defaultdict(Counter), Counter(), Counter()
    path_messages = credited_replays = 0
    for index, item in enumerate(cases):
        instance = broad.load_case(item['entry'])
        for repeat in range(repeats):
            data = read(FOLDER / 'runs' / f'c{index}__r{repeat}.json')
            if (data['status'] != 'COMPLETE' or data['case'] != item['entry']['case']
                or data['repeat'] != repeat or data['dataset'] != item['entry']['dataset']):
                raise ValueError(f'Incomplete or mislabeled execution: {index}/{repeat}')
            budget, prefix = manifest['budget_sec'], data['actors']['prefix']['messages']
            counted = sum(e['type'] in ('INCUMBENT', 'HANDOFF', 'REFINEMENT_INCUMBENT', 'TRAJECTORY_END')
                          for actor in data['actors'].values() for e in actor['messages'])
            if counted != data['physical_checks'] or data['physical_errors'] or data['issues']:
                checks.append(f'Path/execution accounting mismatch: {index}/{repeat}')
            path_messages += counted
            for mode in MODES:
                answer, actor = data['answers'][mode], data['actors'][mode]
                best = credited_path(prefix, actor['messages'], budget)
                if (answer['valid'] != (best is not None)
                    or answer['moves'] != (None if best is None else best['moves'])
                    or answer['path'] != ([] if best is None else best['path'])):
                    checks.append(f'Receipt/path mismatch: {index}/{repeat}/{mode}')
                if best:
                    physical_replay(instance.stacks, instance.max_height,
                                    tuple(Move(**v) for v in best['path']), goal=True)
                    if best['moves'] != len(best['path']):
                        checks.append(f'Physical objective mismatch: {index}/{repeat}/{mode}')
                    credited_replays += 1
                runs[(data['case'], repeat, mode)] = dict(
                    valid=answer['valid'], moves=answer['moves'],
                    final_branch=(answer['provenance'] or {}).get('trajectory_branch', 'prefix_or_none'),
                    guard_started=any(e['type'] == 'TRAJECTORY_BEGIN' and e['branch'] == 1
                                      and e['received_sec'] <= budget for e in actor['messages']),
                    timely_reported_guard_improvements=sum(
                        e['type'] == 'INCUMBENT' and e['received_sec'] <= budget and e['ready_sec'] <= budget
                        and (e.get('refinement_provenance') or {}).get('trajectory_branch') == 1
                        for e in actor['messages']))
                cov = coverage[item['entry']['dataset'] + ':' + mode]
                cov['attempts'] += 1
                cov['feasible'] += int(answer['valid'])
                if mode == 'qubo_pair':
                    final_sources[str((answer['provenance'] or {}).get('trajectory_branch', 'prefix_or_none'))] += 1
                for event in actor['messages']:
                    if event['received_sec'] > budget or event['type'] != 'MODEL':
                        continue
                    record = event['record']
                    policy = record.get('trajectory_policy', dict(single_cycle='single', pair_cycle='pair',
                                                                  qubo_cycle='single_qubo').get(mode, 'unknown'))
                    key = mode + '/' + policy
                    models[key + ':completed'] += 1
                    models[key + ':strict_seed_gain'] += record['energy'] < record['seed_moves']
            row = solo_lineage(data, budget)
            checks.extend(f"Solo lineage: {data['case']}/{repeat}: {v}" for v in row['violations'])
            solo_rows.append(row)
    primary, rng = {}, random.Random(2026101302)
    for reference in (m for m in MODES if m != 'qubo_pair'):
        result = contrast(cases, runs, repeats, 'qubo_pair', reference, rng)
        declared = summary['contrasts'][reference]
        for key in ('common_originals', 'wins', 'losses', 'ties', 'net_original_mean', 'worst',
                    'by_repeat', 'stable_wins', 'bootstrap95_mean'):
            if result[key] != declared[key]:
                checks.append('Original-level reconciliation mismatch: ' + reference + '/' + key)
        primary[reference] = result
    secondary = {reference: contrast(cases, runs, repeats, 'qubo_cycle', reference,
                                      random.Random(2026101399 + index))
                 for index, reference in enumerate(('single_cycle', 'pair_cycle'))}
    grouped = defaultdict(list)
    for row in solo_rows:
        grouped[row['case']].append(row)
    stable_solo = {}
    for reference, result in secondary.items():
        stable_solo[reference] = sum(original['stable_win'] and all(
            r['matched_first_joint'] and r['reported_joint_publications'] > 0
            for r in grouped[original['case']]) for original in result['originals'])
    protection_rows = []
    protection_counts = Counter()
    for item in cases:
        case = item['entry']['case']
        for repeat in range(repeats):
            paired, solo = runs[(case, repeat, 'qubo_pair')], runs[(case, repeat, 'qubo_cycle')]
            if not (paired['valid'] and solo['valid']):
                continue
            delta = paired['moves'] - solo['moves']
            outcome = 'paired_better' if delta < 0 else 'paired_worse' if delta > 0 else 'tie'
            source = str(paired['final_branch'])
            protection_counts[outcome + '/final_branch=' + source] += 1
            protection_rows.append(dict(case=case, repeat=repeat, delta=delta, outcome=outcome,
                final_branch=paired['final_branch'], guard_started=paired['guard_started'],
                timely_reported_guard_improvements=paired['timely_reported_guard_improvements']))
    result = dict(scope='Independent complete fresh solo-control review', accounting_errors=checks,
        complete_corpus_and_accounting=not checks, attempts=len(runs), path_messages=path_messages,
        credited_paths_replayed_again=credited_replays,
        family_coverage={k: dict(v) for k, v in coverage.items()}, primary_comparisons=primary,
        descriptive_solo_comparisons=secondary, models_by_actual_policy=dict(models),
        final_paired_Q_sources=dict(final_sources),
        paired_stable_matched_joint=paired_lineage['primary_stable_wins_with_matched_joint'],
        paired_lineage_violations=paired_lineage['violations'],
        solo_lineage_statuses=dict(Counter(r['status'] for r in solo_rows)),
        solo_stable_matched_joint=stable_solo, solo_lineage_rows=solo_rows,
        paid_protection_outcomes_by_final_source=dict(protection_counts),
        paid_protection_execution_rows=protection_rows,
        declared_strong_solo_quality_rule=summary['strong_solo_quality_rule'],
        declared_anchor_component_rule=summary['inherited_anchor_component_rule'],
        completion_boundary=dict(global_goal_automatically_complete=False,
            independent_test_set=False, scores_spliced_with_other_panels=False,
            matrix_only_or_hardware_advantage_measured=False,
            interpretation='Primary comparisons preserve the frozen Q+Pair target. Solo-Q contrasts '
                'against Single/Pair are descriptive secondary analyses prepared during the run '
                'without inspecting partial outcomes; no retrospective performance gate is added. '
                'Model/seed lineage is not matrix reconstruction. Positive branch-protection '
                'deltas do not establish cost-free protection or per-instance dominance. '
                'A paired win ending on its primary branch is not automatically a direct Pair '
                'rescue; timing and changed later trajectories can also affect its output.'))
    (FOLDER / 'evidence_review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('accounting_errors', 'attempts', 'path_messages',
        'credited_paths_replayed_again', 'paired_stable_matched_joint', 'solo_lineage_statuses',
        'solo_stable_matched_joint', 'declared_strong_solo_quality_rule', 'declared_anchor_component_rule')}))
    if checks or paired_lineage['violations']:
        raise ValueError('Full solo-control evidence review failed')


if __name__ == '__main__':
    main()
