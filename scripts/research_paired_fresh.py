"""Fresh20s paired trajectories on verified homogeneous, distinct CPU cores."""
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

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'scripts',ROOT/'src'):sys.path.insert(0,str(p))
import research_paper_v43_panel as broad
import wall_budget_cpmp_pipeline as native_pipeline
from shared_prefix_cpmp_pipeline import run_prefix
from paired_shared_pipeline import run_continuation
from shared_wall_deadline_protocol_full import run_shared,credited_path
from cpmp_cpu_layout import homogeneous_layout
from paper_priority_group_contract import PriorityGroupContract
from container_reshuffle.core import Move
from research_receiver_allocation import physical_replay
from production_selector_replay import REAL_TIME,fingerprint
from run_beam_selector_research import ENVIRONMENT,digest,write_csv
from research_hardset_qubo_utility import write_json

BASE=ROOT/'docs/experiments/action_coupling_qubo_20261011'
STANDALONE=('native','bs_frg')
CONTINUATIONS=('single_pair','qubo_pair','oracle_pair')
MODES=STANDALONE+CONTINUATIONS
ROLES=('prefix',)+MODES
CPUS=(0,2,4,6,8,10)


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def out(scope):return BASE/('paired_fresh_'+scope)


def freeze(scope):
    folder=out(scope)
    if (folder/'manifest.json').exists():raise FileExistsError('Fresh paired study already frozen')
    conditional=read(BASE/'paired_trajectory_screen_pilot/summary.json')
    lineage=read(BASE/'paired_trajectory_screen_pilot/lineage_audit.json')
    if not conditional['forward_screen'] or lineage['primary_stable_classical_portfolio_wins_with_matched_joint']<2:
        raise ValueError('Conditional paid-portfolio evidence has not qualified')
    previous=read(BASE/'paired_trajectory_screen_pilot/manifest.json')
    primary=read(BASE/'shared_prefix_wall_pilot/manifest.json')['cases']
    cases=[dict(entry=item['entry'],role='primary') for item in primary];repeats=2
    if scope=='smoke':
        cases=[next(item for item in cases if item['entry']['dataset']==dataset) for dataset in ('BF','CV','Open120')]
        repeats=1
    elif scope=='pilot':
        smoke=read(out('smoke')/'summary.json')
        if not smoke['budget_gate'] or not smoke['audit_pass']:raise ValueError('Homogeneous-core smoke failed')
    else:
        pilot=read(out('pilot')/'summary.json');audit=read(out('pilot')/'lineage_audit.json')
        if not pilot['forward_screen'] or audit['primary_stable_wins_with_matched_joint']<2:
            raise ValueError('Fresh paired pilot has not qualified')
        full=read(BASE/'shared_prefix_full_hardset169/manifest.json')
        cases=[dict(entry=item['entry'],role='full_hardset') for item in full['cases']]
        counts=Counter(item['entry']['dataset'] for item in cases)
        if len(cases)!=169 or counts!=dict(BF=73,CV=84,Open120=12):raise ValueError('Wrong hardset169 corpus')
    # No historical solution files are read or hashed by timed workers.
    sources={p:h for p,h in previous['source_sha256'].items() if not p.replace('\\','/').startswith('docs/experiments/')}
    for name in ('shared_wall_deadline_protocol_full.py','shared_prefix_cpmp_pipeline.py',
                 'wall_budget_cpmp_pipeline.py','paired_shared_pipeline.py','cpmp_cpu_layout.py',
                 'research_paired_fresh.py','research_paired_trajectory_screen.py'):
        sources['scripts/'+name]=digest(ROOT/'scripts'/name)
    offsets=[i%len(ROLES) for i in range(len(cases)*repeats)]
    random.Random(2026101301).shuffle(offsets)
    layout=homogeneous_layout(CPUS)
    protocol=(
        'Fresh raw-bay20s study ofthe ungated paid Q+Pair controller. Five methods:nativev4.3, '
        'paper-pure BS-FRG,Single+Pair,Q+Pair,and50ms equivalentMILP+Pair. Aseparate common '
        'prefix producer runs unchangednative/common simplification for thethree continuations. '
        'Six roles useSIX DISTINCT physicalcores ofthe SAME WindowsEfficiencyClass,verifiedlive '
        'against frozenCPU-set intrinsicfields beforeREADY. Source hasadedicatedcore:nomethod '
        'shares itscore. Allsixrole assignments rotate byfrozen shuffledoffsets;the24-block '
        'primarypilot gives eachrole eachcorefourtimes. Oldmixed-class/core-sharing rows are '
        'notcombined withthisstudy. Imports/runtime checks/sourceverification precedeREADY; '
        'thenoneparent sends thesameabsoluteSTART and20sdeadline toallactors. Actualinstance '
        'loading begins afterSTART. Prefixsearch,physicalvalidation,actualorderedBeamchecks, '
        'paidcommoncleanup,IPC,inputcleanup,allmodels/seeds/kernels/FRG,branchswitches and '
        'outputcleanup occurinside thatoriginaldeadline. Producerrequests18s cooperative '
        'handoff onlywithavalidincumbent andno pendingBeamverification;earlynaturalcompletion '
        'leaves actualremainingtime. Never rebasetheclock orgivehistorical/cached answers. '
        'Standalone native/BS searchthe rawbay independently for20s andreceiveNO sharedpath. '
        'Allcontinuations receivethe same freshlycomputed handoff;their independentlypaid '
        'primary andPair trajectories then alternate unchangedfull route/FRG passes. No '
        'cross-method answer sharing. Preserve a globalbest andtwo own searchtrajectories; '
        'OnlyouterINCUMBENT ready ANDreceived bydeadline iscredited. RawREFINEMENT_INCUMBENT '
        'is telemetryonly,evenifarriving beforedeadline. Audit therawtrajectory ledger '
        'separately fromthe reportedledger aftercommonoutputgraphcleanup. Replay ALL full '
        'reported,raw andbranch-end paths,including lateuncreditedones. Missing initial '
        'solutions remain incoverage. Preserve model/seed/branch/sourceprovenance;actual '
        'primary joint paths and matchedinputs are requiredforQ attribution. TheQ kernel '
        'andmodels remain unchanged;one neighborhood requests50ms/50000nodes. Equivalent '
        'MILPconstruction andits50ms solverrequest arepaid. SoloSingle/Pair/Q controls exist '
        'inthepreceding conditionalstudy,not asmembers ofthis freshfive-method panel. '
        'SmokefirstBF/CV/Open primaryoriginals,oneblock each. Pilotunchanged12originals,two '
        'freshrepeats. FullconfirmationALL169originals,twofreshrepeats. Primaryorfull original '
        'means usebothrepeats;no best-of-repeats andnomodel-level independent sample counts. '
        'Forward pilot gate:valid execution,hardware,physical andbranch audits;>=2 stable '
        'Q+Pair wins withreportedprimaryjoint exposure versusSingle+Pair;negative totals in '
        'bothrepeats andnegative95%meanCI upperbound versusSingle+Pair;noQ solution lost '
        'versusSingle+Pair;Q feasible-run countatleastnative andBS counts. Fullcomponent '
        'gate uses>=10 suchstablewins andthesame remainingrequirements. Standalonequality '
        'andtails mustalso beassessed beforeanyglobalgoal completion. Stopandretain the '
        'wholefailedblock afterworker/protocol/physical/hardware failure;do notsplice rows '
        'or silentlyretry. No newdefaults,world-priority orquantum-hardwareclaims.')
    write_json(folder/'manifest.json',dict(cases=cases,modes=MODES,roles=ROLES,repeats=repeats,
        cpus=CPUS,cpu_layout=layout,offsets=offsets,budget_sec=20.,reserve_sec=2.,
        source_sha256=sources,protocol=protocol,frozen_at=datetime.now().isoformat()))
    print(json.dumps(dict(scope=scope,originals=len(cases),modes=MODES,repeats=repeats,cpu_layout=layout)))


def manifest(scope):
    frozen=read(out(scope)/'manifest.json')
    for path,expected in frozen['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Frozen source changed: '+path)
    homogeneous_layout(frozen['cpus'],frozen['cpu_layout'])
    return frozen


def worker(scope,index,repeat,role,cpu):
    frozen=manifest(scope);entry=frozen['cases'][index]['entry']
    if digest(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Raw instance changed')
    broad.pin_cpu(cpu)
    def publish(event):print(json.dumps(event,separators=(',',':')),flush=True)
    cpu_info=next(row for row in frozen['cpu_layout'] if row['logical_processor']==cpu)
    publish(dict(type='READY',role=role,case=entry['case'],cpu=cpu,cpu_info=cpu_info))
    clock=json.loads(sys.stdin.readline());start,budget=clock['start'],clock['budget']
    publish(dict(type='START_ACK',absolute_start=start,budget_sec=budget))
    try:
        if role=='prefix':result=run_prefix(entry,start,budget,publish,reserve=frozen['reserve_sec'])
        elif role in STANDALONE:result=native_pipeline.run(entry,role,start,budget,publish,reserve=0.)
        else:
            contract=PriorityGroupContract();contract.install()
            try:
                instance=broad.load_case(entry);message=json.loads(sys.stdin.readline())
                if message['type']=='STOP':result=dict(mode=role,skipped=True,reason=message['reason'],elapsed_sec=REAL_TIME()-start)
                else:
                    if message['start']!=start or message['budget']!=budget:raise ValueError('Parent rebased clock')
                    result=run_continuation(instance,role,message['handoff'],start,budget,publish)
            finally:contract.restore()
        publish(dict(type='DONE',**result))
    except BaseException:
        publish(dict(type='ERROR',detail=traceback.format_exc(),elapsed_sec=REAL_TIME()-start));raise


def execute(scope,index,repeat,frozen):
    item=frozen['cases'][index];offset=frozen['offsets'][index*frozen['repeats']+repeat]
    roles=list(frozen['roles']);order=roles[offset:]+roles[:offset]
    assignments={role:frozen['cpus'][position] for position,role in enumerate(order)}
    commands={role:[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--scope',scope,
        '--case-index',str(index),'--repeat',str(repeat),'--role',role,'--cpu',str(cpu)] for role,cpu in assignments.items()}
    data=run_shared(commands,cwd=ROOT,env={**os.environ,**ENVIRONMENT},budget=frozen['budget_sec'],handoff_roles=CONTINUATIONS)
    instance=broad.load_case(item['entry']);physical_errors=[];physical_checks=0
    for role,actor in data['actors'].items():
        for event in actor['messages']:
            if event['type'] not in ('INCUMBENT','HANDOFF','REFINEMENT_INCUMBENT','TRAJECTORY_END'):continue
            try:
                physical_replay(instance.stacks,instance.max_height,tuple(Move(**v) for v in event['path']),goal=True)
                physical_checks+=1;count=event['after'] if event['type']=='TRAJECTORY_END' else event['moves']
                if count!=len(event['path']):raise ValueError('Full physical cost mismatch')
            except (ValueError,IndexError) as error:physical_errors.append(dict(role=role,type=event['type'],detail=str(error)))
    answers={};prefix=data['actors']['prefix']['messages']
    for mode in frozen['modes']:
        source=[] if mode in STANDALONE else prefix
        best=credited_path(source,data['actors'][mode]['messages'],frozen['budget_sec'])
        answers[mode]=dict(valid=best is not None,moves=None if best is None else best['moves'],
            path=[] if best is None else best['path'],source=None if best is None else best['source'],
            provenance=None if best is None else best.get('refinement_provenance'))
    return dict(case=item['entry']['case'],dataset=item['entry']['dataset'],repeat=repeat,status='COMPLETE',
        cpu_assignments=assignments,offset=offset,answers=answers,physical_checks=physical_checks,
        physical_errors=physical_errors,**data)


def raw_ledger_view(actor,budget):
    """Build a separate diagnostic view without mutating or crediting raw events."""
    events=[e for e in actor['messages'] if e['received_sec']<=budget]
    entry=next((e for e in events if e['type']=='BRANCH_INPUT'),None)
    if entry is None or entry['model_input_signature'] is None:return None
    matches=[e for e in events if e['type']=='INCUMBENT' and e['received_sec']<=entry['received_sec']
             and fingerprint(e['path'])==entry['model_input_signature']]
    if not matches:raise ValueError('Model input has no actual reported physical path')
    initial=matches[-1]['path']
    messages=[]
    for event in events:
        if event['type']=='START_ACK':messages.append(event)
        elif event.get('stage')=='refinement':
            messages.append(dict(event,type='INCUMBENT') if event['type']=='REFINEMENT_INCUMBENT' else event)
    best=min([len(initial)]+[e['moves'] for e in messages if e['type']=='INCUMBENT' and e['ready_sec']<=budget])
    return dict(initial_path=initial,initial=len(initial),moves=best,messages=messages)


def analyze(scope):
    # The imported function checks the raw ledger only. Its conditional-runner
    # globals never drive this fresh experiment or contribute output credit.
    from research_paired_trajectory_screen import audit_answer
    frozen=manifest(scope);rows=[];failures=[];errors=[];missing=[];branch_audits=[]
    physical=beams=standalone_beams=0;allocations=Counter();max_lag=0.
    expected_cpus={row['logical_processor']:row for row in frozen['cpu_layout']}
    for index,item in enumerate(frozen['cases']):
        for repeat in range(frozen['repeats']):
            data=read(out(scope)/'runs'/f'c{index}__r{repeat}.json')
            key=dict(case=item['entry']['case'],repeat=repeat)
            if data['status']!='COMPLETE':failures.append(dict(key,detail=data));continue
            def require(condition,reason):
                if not condition:errors.append(dict(key,reason=reason))
            budget=frozen['budget_sec'];physical+=data['physical_checks']
            errors.extend(dict(key,**error) for error in data['physical_errors'])
            max_lag=max(max_lag,data['termination_lag_sec'])
            if data['issues'] or data['termination_lag_sec']>.1:
                failures.append(dict(key,issues=data['issues'],lag=data['termination_lag_sec']))
            require(len(set(data['cpu_assignments'].values()))==6,'Shared actor CPU slot')
            for role,cpu in data['cpu_assignments'].items():
                allocations[(role,cpu)]+=1;ready=data['actors'][role]['ready']
                require(ready['cpu']==cpu and ready['cpu_info']==expected_cpus[cpu],'Wrong actual CPU layout at READY')
                acks=[e for e in data['actors'][role]['messages'] if e['type']=='START_ACK' and e['received_sec']<=budget]
                require(len(acks)==1 and all(e['absolute_start']==data['absolute_start'] and e['budget_sec']==budget for e in acks),
                        'Missing or changed original clock')
            prefix=[e for e in data['actors']['prefix']['messages'] if e['received_sec']<=budget]
            commitments=[e for e in prefix if e['type']=='BEAM'];beams+=len(commitments)
            require(all(e['actual_frontier_verified'] for e in commitments),'Unverified source Beam')
            common=credited_path(prefix,[],budget)
            if common is None:missing.append(dict(key,status='NO_FRESH_PREFIX_SOLUTION'))
            signatures=[]
            for mode in frozen['modes']:
                actor=data['actors'][mode];events=[e for e in actor['messages'] if e['received_sec']<=budget]
                if mode in STANDALONE:
                    require(mode not in data['handoff_sent_sec'],'Standalone received a shared answer')
                    checks=[e for e in events if e['type']=='BEAM'];standalone_beams+=len(checks)
                    require(all(e['actual_frontier_verified'] for e in checks),'Unverified standalone Beam')
                inputs=[e for e in events if e['type']=='BRANCH_INPUT']
                for entry in inputs:
                    require(data['handoff'] is not None and entry['path_signature']==data['handoff']['path_signature']
                        and entry['absolute_start']==data['absolute_start'],'Changed continuation input/clock')
                    if entry['model_input_signature'] is not None:signatures.append(entry['model_input_signature'])
                if mode in CONTINUATIONS:
                    try:
                        view=raw_ledger_view(actor,budget)
                        if view is not None:
                            checked=audit_answer(view,view['initial_path'],mode,budget)
                            errors.extend(dict(key,mode=mode,reason=reason) for reason in checked['errors'])
                            branch_audits.append(dict(key,mode=mode,**checked))
                    except ValueError as error:errors.append(dict(key,mode=mode,reason=str(error)))
                publications=[e for e in events if e['type']=='INCUMBENT' and e['ready_sec']<=budget]
                joint=sum((e.get('refinement_provenance') or {}).get('original_source')=='qubo_joint' for e in publications)
                for event in publications:
                    provenance=event.get('refinement_provenance')
                    if provenance:require(event['moves']<=provenance['raw_moves'],'Reported cost exceeds originating raw path')
                source=[] if mode in STANDALONE else prefix
                best=credited_path(source,actor['messages'],budget);answer=data['answers'][mode]
                require(answer['moves']==(None if best is None else best['moves']),'Credited a raw/unpublished/late answer')
                rows.append(dict(key,dataset=data['dataset'],mode=mode,valid=answer['valid'],moves=answer['moves'],
                    joint_publications=joint,status=actor['status'],source=answer['source']))
            require(len(set(signatures))<=1,'Different initial model inputs across continuations')
    table={(r['case'],r['repeat'],r['mode']):r for r in rows};rng=random.Random(2026101302);contrasts={}
    for reference in [mode for mode in frozen['modes'] if mode!='qubo_pair']:
        pairs=[];gained=[];lost=[]
        for item in frozen['cases']:
            case=item['entry']['case'];deltas=[];exposed=[]
            for repeat in range(frozen['repeats']):
                q=table.get((case,repeat,'qubo_pair'));p=table.get((case,repeat,reference))
                if q is None or p is None:continue
                if q['valid'] and not p['valid']:gained.append(dict(case=case,repeat=repeat))
                if p['valid'] and not q['valid']:lost.append(dict(case=case,repeat=repeat))
                if q['valid'] and p['valid']:
                    deltas.append(q['moves']-p['moves']);exposed.append(q['joint_publications']>0)
            if len(deltas)==frozen['repeats']:pairs.append(dict(case=case,deltas=deltas,mean=sum(deltas)/len(deltas),
                stable_win=all(d<0 for d in deltas),joint_all=all(exposed)))
        values=[row['mean'] for row in pairs]
        samples=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(10000)) if values else []
        contrasts[reference]=dict(common_originals=len(values),wins=sum(v<0 for v in values),losses=sum(v>0 for v in values),
            ties=sum(v==0 for v in values),net_original_mean=sum(values),worst=max(values,default=0),
            by_repeat={rep:sum(row['deltas'][rep] for row in pairs) for rep in range(frozen['repeats'])},
            stable_wins=sum(row['stable_win'] for row in pairs),stable_wins_with_joint=sum(row['stable_win'] and row['joint_all'] for row in pairs),
            bootstrap95_mean=[samples[250],samples[9749]] if samples else None,gained_solutions=gained,lost_solutions=lost,originals=pairs)
    allocation_rows={role:{str(cpu):allocations[(role,cpu)] for cpu in frozen['cpus']} for role in frozen['roles']}
    if scope=='pilot' and any(set(counts.values())!={4} for counts in allocation_rows.values()):
        errors.append(dict(reason='Primary role/core allocation is not balanced',observed=allocation_rows))
    valid=not failures and len(rows)==len(frozen['cases'])*frozen['repeats']*len(frozen['modes'])
    valid_runs={mode:sum(r['valid'] for r in rows if r['mode']==mode) for mode in frozen['modes']}
    reference=contrasts['single_pair'];required=10 if scope=='hardset169' else 2
    screen=(scope!='smoke' and valid and not errors and reference['stable_wins_with_joint']>=required
        and all(v<0 for v in reference['by_repeat'].values()) and not reference['lost_solutions']
        and reference['bootstrap95_mean'] is not None and reference['bootstrap95_mean'][1]<0
        and all(valid_runs['qubo_pair']>=valid_runs[mode] for mode in STANDALONE))
    result=dict(scope=scope,originals=len(frozen['cases']),budget_gate=valid,audit_pass=not errors,
        budget_failures=failures,audit_errors=errors,missing_common_prefix=missing,max_parent_lag_sec=max_lag,
        valid_runs=valid_runs,physical_path_checks=physical,actual_beam_checks=beams,standalone_beam_checks=standalone_beams,
        role_cpu_counts=allocation_rows,contrasts=contrasts,forward_screen=screen,
        scope_boundary='Fresh commonprefix20s ondistinct homogeneouscores;component gate isnot automatic globalpromotion')
    write_json(out(scope)/'summary.json',result);write_csv(out(scope)/'results.csv',rows)
    write_json(out(scope)/'branch_audit.json',dict(rows=branch_audits,errors=errors))
    print(json.dumps({**{k:v for k,v in result.items() if k not in ('contrasts','role_cpu_counts')},'contrasts':
        {k:{a:b for a,b in v.items() if a!='originals'} for k,v in contrasts.items()}}),flush=True)


def run(scope):
    frozen=manifest(scope);folder=out(scope)
    if (folder/'status.json').exists() and read(folder/'status.json')['status']=='COMPLETE':raise FileExistsError('Study already complete')
    write_json(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),started=datetime.now().isoformat()))
    for index,item in enumerate(frozen['cases']):
        for repeat in range(frozen['repeats']):
            path=folder/'runs'/f'c{index}__r{repeat}.json'
            if path.exists():continue
            try:data=execute(scope,index,repeat,frozen)
            except Exception:data=dict(status='WORKER_ERROR',error=traceback.format_exc(),case=item['entry']['case'],repeat=repeat)
            write_json(path,data)
            print(json.dumps(dict(scope=scope,case=item['entry']['case'],repeat=repeat,status=data['status'])),flush=True)
            if data['status']!='COMPLETE' or data.get('issues') or data.get('physical_errors'):
                write_json(folder/'status.json',dict(status='STOPPED_ON_FAILURE',case=item['entry']['case'],repeat=repeat,
                    finished=datetime.now().isoformat()));return
    analyze(scope)
    write_json(folder/'status.json',dict(status='COMPLETE',finished=datetime.now().isoformat()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','run','worker','analyze'))
    parser.add_argument('--scope',choices=('smoke','pilot','hardset169'),required=True)
    parser.add_argument('--case-index',type=int);parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--role',choices=ROLES);parser.add_argument('--cpu',type=int,default=0)
    args=parser.parse_args()
    if args.command=='worker':worker(args.scope,args.case_index,args.repeat,args.role,args.cpu)
    else:globals()[args.command](args.scope)
