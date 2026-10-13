"""Fresh matched comparison against full-budget solo refiners; no algorithm change."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime
import io
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'scripts', ROOT / 'src'):
    sys.path.insert(0, str(directory))
import research_paired_fresh_stream as stream
from cpmp_cpu_layout import homogeneous_layout
from run_beam_selector_research import digest

study = stream.study
BASE = study.BASE
MODES = ('single_cycle', 'pair_cycle', 'qubo_cycle', 'single_pair', 'qubo_pair')
ROLES = ('prefix',) + MODES
FAMILY = 'paired_solo_fresh'
complete_analysis = stream.analyze


def out(scope):
    return BASE / (FAMILY + '_' + scope)


def freeze(scope):
    folder = out(scope)
    if (folder / 'manifest.json').exists():
        raise FileExistsError('Solo-control comparison already frozen')
    completed = BASE / 'paired_fresh_exploratory_hardset169'
    if study.read(completed / 'status.json')['status'] != 'COMPLETE':
        raise ValueError('Wait for the original full paired study')
    parent_summary = study.read(completed / 'summary.json')
    parent_lineage = study.read(completed / 'lineage_audit.json')
    if not parent_summary['budget_gate'] or not parent_summary['audit_pass'] or parent_lineage['violations']:
        raise ValueError('The prior execution evidence must be valid')
    previous = study.read(completed / 'manifest.json')
    primary = study.read(BASE / 'paired_fresh_stream_pilot/manifest.json')['cases']
    cases = [dict(entry=item['entry'], role='primary') for item in primary]
    repeats = 2
    if scope == 'smoke':
        cases = [next(item for item in cases if item['entry']['dataset'] == dataset)
                 for dataset in ('BF', 'CV', 'Open120')]
        repeats = 1
    else:
        prerequisite = out('smoke' if scope == 'pilot' else 'pilot')
        if study.read(prerequisite / 'status.json')['status'] != 'COMPLETE':
            raise ValueError('The preceding wiring stage is unfinished')
        checked = study.read(prerequisite / 'summary.json')
        lineage = study.read(prerequisite / 'lineage_audit.json')
        if not checked['protocol_only_expansion_pass'] or lineage['violations']:
            raise ValueError('Solo-control wiring or lineage is invalid')
        if scope == 'hardset169':
            cases = [dict(entry=item['entry'], role='full_hardset') for item in previous['cases']]
            counts = Counter(item['entry']['dataset'] for item in cases)
            if len(cases) != 169 or counts != dict(BF=73, CV=84, Open120=12):
                raise ValueError('The complete confirmed corpus is required')
    sources = dict(previous['source_sha256'])
    sources['scripts/research_paired_solo_fresh.py'] = digest(Path(__file__).resolve())
    offsets = [i % len(ROLES) for i in range(len(cases) * repeats)]
    random.Random(2026101313).shuffle(offsets)
    protocol = (
        'COMPLEMENTARY FRESH STRONG-SOLO COMPARISON. No algorithm change: compare the existing '
        'single_cycle, pair_cycle, qubo_cycle, single_pair and qubo_pair continuations. The first '
        'three spend the entire actual remaining allowance on one trajectory; the paired methods '
        'retain their original paid full-pass alternation. Every method includes the same outer '
        'physical validation and output cleanup. A dedicated sixth actor computes a fresh native '
        'v4.3 prefix with actual ordered Beam verification and the unchanged common cleanup. '
        'All six roles use distinct same-class physical cores, live-verified before READY, with '
        'frozen rotated role allocations. Imports and source checks precede READY; raw instance '
        'loading and all prefix/search/model/seed/kernel/FRG/validation/IPC/publication work are '
        'charged from the same absolute START to its original 20-second deadline. A feasible '
        'prefix may cooperatively hand off at 18 seconds, or on earlier natural completion. '
        'All five methods receive that same actually computed path and its unrebased clock. '
        'No historical solution input, answer sharing between methods, free classical guard, '
        'extra continuation time, or best-of-repeats credit. Only outer INCUMBENT ready AND '
        'received by deadline counts. Raw refinement events are diagnostic only. Independently '
        'replay every complete published/raw/branch-end path, including late uncredited paths. '
        'Preserve unavailable prefixes, cutoffs, failed blocks, negative outcomes and source '
        'lineage. Stop after worker/protocol/physical/hardware failure; never silently retry '
        'individual methods. The unchanged Q kernel requests 50ms and 50000 nodes per model. '
        'Smoke uses the first primary BF/CV/Open120 cases once; pilot keeps the unchanged12 '
        'primary originals twice; full measures ALL169 originals twice. Expansion from smoke '
        'and pilot is based ONLY on protocol validity, not a favorable quality result: this '
        'study fills the missing strong-solo control coverage for an already completed full '
        'paired study. Failed or uncertain pilot quality remains recorded. Original-level '
        'quality uses the mean of both repeats on common successful originals; all failures '
        'remain separately in complete-corpus feasibility accounting. For the full strong-solo '
        'assessment require valid audits, negative totals in both repeats, negative upper95% '
        'original-mean intervals and no Q+Pair feasibility loss versus BOTH single_cycle and '
        'pair_cycle. Report Q+Pair versus qubo_cycle separately to quantify the paid guard cost. '
        'The existing Q+Pair-versus-Single+Pair component rule and matched actual joint lineage '
        'remain separate. Do not splice scores or minima with the earlier native/BS/MILP panel. '
        'No automatic goal completion, default promotion, generic-matrix or hardware claim.')
    study.write_json(folder / 'manifest.json', dict(
        cases=cases, modes=MODES, roles=ROLES, repeats=repeats, cpus=study.CPUS,
        cpu_layout=homogeneous_layout(study.CPUS), offsets=offsets, budget_sec=20., reserve_sec=2.,
        source_sha256=sources, protocol=protocol, frozen_at=datetime.now().isoformat(),
        measurement_intent='COMPLEMENTARY_FRESH_STRONG_SOLO_CONTROLS', new_algorithm=False,
        expansion_rule='protocol_validity_only',
        parent_summary_sha256=digest(completed / 'summary.json'),
        parent_lineage_sha256=digest(completed / 'lineage_audit.json')))
    print(json.dumps(dict(scope=scope, originals=len(cases), repeats=repeats, modes=MODES,
                          expansion_rule='protocol_validity_only')))


def cycle_stream_errors(data):
    errors = []
    for role, actor in data['actors'].items():
        for line in actor.get('stderr', '').splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get('type') in ('CYCLE_BEGIN', 'CYCLE_END'):
                errors.append(dict(role=role, reason='Cycle protocol message in diagnostic stderr'))
    return errors


def analyze(scope):
    # Both wrappers must use this new family; the prior offline routing failure
    # is covered by a focused regression test and must not be repeated here.
    with redirect_stdout(io.StringIO()):
        complete_analysis(scope)
    frozen = study.manifest(scope)
    summary = study.read(out(scope) / 'summary.json')
    errors = []
    cycle_counts = Counter()
    for index, item in enumerate(frozen['cases']):
        for repeat in range(frozen['repeats']):
            data = study.read(out(scope) / 'runs' / f'c{index}__r{repeat}.json')
            errors.extend(dict(case=item['entry']['case'], repeat=repeat, **e)
                          for e in cycle_stream_errors(data))
            for role in ('single_cycle', 'pair_cycle', 'qubo_cycle'):
                cycle_counts[role] += sum(e['type'] == 'CYCLE_BEGIN' and e['received_sec'] <= frozen['budget_sec']
                                         for e in data['actors'][role]['messages'])
    summary['audit_errors'].extend(errors)
    summary['audit_pass'] = summary['audit_pass'] and not errors
    summary['cycle_stream_errors'] = errors
    summary['actual_solo_cycles_started'] = dict(cycle_counts)
    summary['inherited_anchor_component_rule'] = summary.pop('forward_screen') and not errors
    summary['protocol_only_expansion_pass'] = summary['budget_gate'] and summary['audit_pass']
    summary['strong_solo_quality_rule'] = scope == 'hardset169' and summary['protocol_only_expansion_pass'] and all(
        all(value < 0 for value in summary['contrasts'][mode]['by_repeat'].values())
        and summary['contrasts'][mode]['bootstrap95_mean'] is not None
        and summary['contrasts'][mode]['bootstrap95_mean'][1] < 0
        and not summary['contrasts'][mode]['lost_solutions']
        for mode in ('single_cycle', 'pair_cycle'))
    summary['measurement_intent'] = frozen['measurement_intent']
    summary['scope_boundary'] = ('Fresh complementary solo-control panel; scores stay separate from '
        'the earlier native/BS/MILP panel. Protocol-only expansion is not a quality pass. '
        'Neither component nor solo quality rules automatically complete the research goal.')
    study.write_json(out(scope) / 'summary.json', summary)
    print(json.dumps({**{k: v for k, v in summary.items() if k not in
        ('contrasts', 'role_cpu_counts', 'missing_common_prefix')}, 'contrasts':
        {mode: {key: value for key, value in values.items() if key not in
                ('originals', 'gained_solutions', 'lost_solutions')}
         for mode, values in summary['contrasts'].items()}}), flush=True)


def lineage(scope):
    from audit_paired_fresh_lineage import main
    main(scope, FAMILY)


study.STANDALONE = ()
study.CONTINUATIONS = MODES
study.MODES = MODES
study.ROLES = ROLES
study.out = out
stream.out = out
study.__file__ = __file__
study.analyze = analyze


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'run', 'worker', 'analyze', 'lineage'))
    parser.add_argument('--scope', choices=('smoke', 'pilot', 'hardset169'), required=True)
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--cpu', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'worker':
        stream.worker(args.scope, args.case_index, args.repeat, args.role, args.cpu)
    elif args.command == 'run':
        study.run(args.scope)
    else:
        globals()[args.command](args.scope)
