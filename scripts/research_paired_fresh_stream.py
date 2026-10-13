"""Correct native callback transport without changing the frozen algorithm."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'scripts',ROOT/'src'):sys.path.insert(0,str(p))
import research_paired_fresh as study
from paper_priority_group_contract import PriorityGroupContract
from production_selector_replay import REAL_TIME
from run_beam_selector_research import digest

BASE=study.BASE
original_freeze=study.freeze
original_analyze=study.analyze


def out(scope):return BASE/('paired_fresh_stream_'+scope)


def captured_publisher():
    output=sys.stdout
    def publish(event):
        output.write(json.dumps(event,separators=(',',':'))+'\n')
        output.flush()
    return publish


def freeze(scope):
    saved=study.write_json
    def write_once(path,value):
        if path.name=='manifest.json':
            value['source_sha256']['scripts/research_paired_fresh_stream.py']=digest(Path(__file__).resolve())
            value['protocol']+=(' Stream correction: the first paired_fresh_smoke misrouted native '
                'callback JSON into stderr because its publisher looked up sys.stdout inside '
                'redirect_stdout. That study is retained butits performance scores are unusable. '
                'This newimmutable wrapper captures the protocol output object before native '
                'execution. Algorithms,budgets,CPUlayout,roles,offset rule andgates are unchanged. '
                'Reject protocol JSON found in diagnostic stderr. For timely NATIVE_END,require '
                'received Beam count toagree withnative observer metadata;never creditstderr '
                'candidates using readytime alone. No rescore orsplice ofthe failed study.')
        saved(path,value)
    try:
        study.write_json=write_once
        original_freeze(scope)
    finally:study.write_json=saved


def worker(scope,index,repeat,role,cpu):
    frozen=study.manifest(scope);entry=frozen['cases'][index]['entry']
    if digest(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Raw instance changed')
    study.broad.pin_cpu(cpu);publish=captured_publisher()
    cpu_info=next(row for row in frozen['cpu_layout'] if row['logical_processor']==cpu)
    publish(dict(type='READY',role=role,case=entry['case'],cpu=cpu,cpu_info=cpu_info))
    clock=json.loads(sys.stdin.readline());start,budget=clock['start'],clock['budget']
    publish(dict(type='START_ACK',absolute_start=start,budget_sec=budget))
    try:
        if role=='prefix':result=study.run_prefix(entry,start,budget,publish,reserve=frozen['reserve_sec'])
        elif role in study.STANDALONE:result=study.native_pipeline.run(entry,role,start,budget,publish,reserve=0.)
        else:
            contract=PriorityGroupContract();contract.install()
            try:
                instance=study.broad.load_case(entry);message=json.loads(sys.stdin.readline())
                if message['type']=='STOP':result=dict(mode=role,skipped=True,reason=message['reason'],elapsed_sec=REAL_TIME()-start)
                else:
                    if message['start']!=start or message['budget']!=budget:raise ValueError('Parent rebased clock')
                    result=study.run_continuation(instance,role,message['handoff'],start,budget,publish)
            finally:contract.restore()
        publish(dict(type='DONE',**result))
    except BaseException:
        publish(dict(type='ERROR',detail=traceback.format_exc(),elapsed_sec=REAL_TIME()-start));raise


def stream_errors(data,budget):
    errors=[]
    if data['status']!='COMPLETE':return errors
    protocol_types={'READY','START_ACK','INCUMBENT','HANDOFF','BEAM','PHASE','MODEL','TRAJECTORY_BEGIN',
                    'TRAJECTORY_END','REFINEMENT_INCUMBENT','BRANCH_INPUT','NATIVE_END','COMMON_END','DONE','ERROR'}
    for role,actor in data['actors'].items():
        misplaced=[]
        for line in actor.get('stderr','').splitlines():
            try:event=json.loads(line)
            except json.JSONDecodeError:continue
            if isinstance(event,dict) and event.get('type') in protocol_types:misplaced.append(event['type'])
        if misplaced:errors.append(dict(role=role,reason='Protocol messages in diagnostic stderr',
                                        counts={kind:misplaced.count(kind) for kind in set(misplaced)}))
        events=[e for e in actor['messages'] if e['received_sec']<=budget]
        beams=[e for e in events if e['type']=='BEAM']
        endings=[e for e in events if e['type']=='NATIVE_END' and e['elapsed_sec']<=budget]
        for end in endings:
            if end['beam_checks']!=len(beams):errors.append(dict(role=role,reason='Native Beam metadata lacks received events',
                                                               recorded=end['beam_checks'],received=len(beams)))
    return errors


def analyze(scope):
    with redirect_stdout(io.StringIO()):original_analyze(scope)
    frozen=study.manifest(scope);summary=study.read(out(scope)/'summary.json');errors=[]
    for index,item in enumerate(frozen['cases']):
        for repeat in range(frozen['repeats']):
            data=study.read(out(scope)/'runs'/f'c{index}__r{repeat}.json')
            errors.extend(dict(case=item['entry']['case'],repeat=repeat,**e) for e in stream_errors(data,frozen['budget_sec']))
    summary['protocol_stream_errors']=errors
    summary['audit_errors'].extend(errors)
    summary['audit_pass']=summary['audit_pass'] and not errors
    summary['forward_screen']=summary['forward_screen'] and not errors
    study.write_json(out(scope)/'summary.json',summary)
    print(json.dumps({**{k:v for k,v in summary.items() if k not in ('contrasts','role_cpu_counts')},'contrasts':
        {k:{a:b for a,b in v.items() if a!='originals'} for k,v in summary['contrasts'].items()}}),flush=True)


study.out=out
study.__file__=__file__
study.worker=worker
study.analyze=analyze


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','run','worker','analyze'))
    parser.add_argument('--scope',choices=('smoke','pilot','hardset169'),required=True)
    parser.add_argument('--case-index',type=int);parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--role',choices=study.ROLES);parser.add_argument('--cpu',type=int,default=0)
    args=parser.parse_args()
    if args.command=='worker':worker(args.scope,args.case_index,args.repeat,args.role,args.cpu)
    elif args.command=='run':study.run(args.scope)
    else:globals()[args.command](args.scope)
