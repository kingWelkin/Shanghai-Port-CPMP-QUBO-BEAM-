"""Explicit full169 measurement after an uncertain, not passed, primary pilot."""
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

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'scripts',ROOT/'src'):sys.path.insert(0,str(p))
import research_paired_fresh_stream as stream
from cpmp_cpu_layout import homogeneous_layout
from run_beam_selector_research import digest

study=stream.study
BASE=study.BASE
complete_analysis=stream.analyze


def out(scope):
    if scope!='hardset169':raise ValueError('This protocol only measures the complete corpus')
    return BASE/('paired_fresh_exploratory_'+scope)


def freeze(scope):
    folder=out(scope)
    if (folder/'manifest.json').exists():raise FileExistsError('Full exploratory study already frozen')
    pilot_folder=BASE/'paired_fresh_stream_pilot'
    if study.read(pilot_folder/'status.json')['status']!='COMPLETE':raise ValueError('Fresh pilot is unfinished')
    pilot=study.read(pilot_folder/'summary.json');lineage=study.read(pilot_folder/'lineage_audit.json')
    if not pilot['budget_gate'] or not pilot['audit_pass'] or lineage['violations']:
        raise ValueError('Execution/provenance validity is required even for exploratory measurement')
    previous=study.read(pilot_folder/'manifest.json')
    inventory=study.read(BASE/'shared_prefix_full_hardset169/manifest.json')['cases']
    cases=[dict(entry=item['entry'],role='full_hardset') for item in inventory]
    if len(cases)!=169 or Counter(item['entry']['dataset'] for item in cases)!=dict(BF=73,CV=84,Open120=12):
        raise ValueError('The user-confirmed corpus must remain complete')
    sources=dict(previous['source_sha256'])
    sources['scripts/research_paired_fresh_exploratory.py']=digest(Path(__file__).resolve())
    offsets=[i%len(study.ROLES) for i in range(len(cases)*2)]
    random.Random(2026101301).shuffle(offsets)
    protocol=(
        'EXPLORATORY FULL-CORPUS MEASUREMENT,not confirmation ofapassed pilot. Thecorrected '
        'fresh12-original pilot has3 wins/1 loss/8 ties versusSingle+Pair,-19.5 mean-total '
        'moves(-28/-11),three matched repeatablejoint wins,andone+1.5 meanloss. Its95%mean '
        'interval[-4.08333,+0.125] crosseszero anditsoriginalforwardgate FAILED. Preserve '
        'thatfailedgate andallrecords. This NEW protocol deliberately gathers complete '
        'hardset169 evidence toresolve uncertainty under theuser\'s persistent full-corpus '
        'research objective. Itdoesnot tune thealgorithm,select an easier subset,delete '
        'losses,lower thefull-cohort assessment rule,orclaimthe pilotqualified. Itis a '
        'documented expansion ofmeasurement scope,not aproduction promotion. The entire '
        'BF73/CV84/Open12012 corpus isattempted twice,including allpastfailures. Cases and '
        'parameters arefrozen beforeanyfull-study outcome;no interim tuning orstopping for '
        'favorablequality results. Preserve incomplete/timeout/negative results andanalyze '
        'original-level pairedrepeats. TheFULLcomponent rule remains>=10 stable Q+Pair wins '
        'withreportedjoint exposure versusSingle+Pair,negative totals inbothrepeats,negative '
        '95%meanCI upperbound,noQ solutionlost versusSingle+Pair,andQ feasible-run count '
        'atleastnative/BS,withall execution/hardware/physical/stream audits valid. Passing '
        'thatcomponentrule alone stilldoesnot complete theglobalgoal. Standalonequality, '
        'tails,matchedjoint lineage andtheclassical structuralbackend scope mustbeassessed '
        'separately. No untouched-test-set,worldwide-priority orhardwareadvantage claim. '
        'All per-instance execution details areUNCHANGED fromthecorrected freshpilot: '
    )+previous['protocol']
    record=dict(cases=cases,modes=study.MODES,roles=study.ROLES,repeats=2,cpus=study.CPUS,
        cpu_layout=homogeneous_layout(study.CPUS),offsets=offsets,budget_sec=20.,reserve_sec=2.,
        source_sha256=sources,protocol=protocol,frozen_at=datetime.now().isoformat(),
        measurement_intent='EXPLORATORY_FULL_CORPUS_AFTER_UNCERTAIN_PILOT',
        original_pilot_forward_screen=pilot['forward_screen'],
        pilot_summary_sha256=digest(pilot_folder/'summary.json'),
        pilot_lineage_sha256=digest(pilot_folder/'lineage_audit.json'))
    study.write_json(folder/'manifest.json',record)
    print(json.dumps(dict(scope=scope,originals=len(cases),repeats=2,modes=study.MODES,
        measurement_intent=record['measurement_intent'],original_pilot_forward_screen=pilot['forward_screen'])))


def analyze(scope):
    with redirect_stdout(io.StringIO()):complete_analysis(scope)
    summary=study.read(out(scope)/'summary.json');frozen=study.manifest(scope)
    summary.update(measurement_intent=frozen['measurement_intent'],
        original_pilot_forward_screen=frozen['original_pilot_forward_screen'],
        confirmation_of_passed_pilot=False)
    study.write_json(out(scope)/'summary.json',summary)
    print(json.dumps({**{k:v for k,v in summary.items() if k not in ('contrasts','role_cpu_counts')},'contrasts':
        {k:{a:b for a,b in v.items() if a!='originals'} for k,v in summary['contrasts'].items()}}),flush=True)


study.out=out
study.__file__=__file__
study.analyze=analyze


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','run','worker','analyze'))
    parser.add_argument('--scope',choices=('hardset169',),default='hardset169')
    parser.add_argument('--case-index',type=int);parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--role',choices=study.ROLES);parser.add_argument('--cpu',type=int,default=0)
    args=parser.parse_args()
    if args.command=='worker':stream.worker(args.scope,args.case_index,args.repeat,args.role,args.cpu)
    elif args.command=='run':study.run(args.scope)
    else:globals()[args.command](args.scope)
