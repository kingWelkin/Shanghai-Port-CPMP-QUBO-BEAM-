"""Frozen paid two-trajectory comparison with branch-specific provenance."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'scripts',ROOT/'src'):sys.path.insert(0,str(p))
import numpy as np
import research_single_seed_screen as base
from paired_trajectory_refinement import MODES,SOLO,PRIMARY,run as improve
from container_reshuffle.core import Move
from paper_priority_group_contract import PriorityGroupContract
from production_selector_replay import REAL_TIME,fingerprint
from research_receiver_allocation import physical_replay
from deadline_worker_protocol import run_worker
from run_beam_selector_research import ENVIRONMENT,digest,write_csv
from research_hardset_qubo_utility import write_json

BASE=base.BASE


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def out(scope):return BASE/('paired_trajectory_screen_'+scope)


def freeze(scope):
    folder=out(scope)
    if (folder/'manifest.json').exists():raise FileExistsError('Paired trajectory study already frozen')
    previous=BASE/'cyclic_residual_budget_pilot'
    if read(previous/'status.json')['status']!='COMPLETE':raise ValueError('Do not overlap timed studies')
    old=read(previous/'manifest.json');result=read(previous/'summary.json')
    if not result['budget_gate'] or not result['audit_pass']:raise ValueError('Invalid source study')
    chosen=list(range(len(old['cases'])));repeats,jobs,source_repeats=2,4,(0,1)
    if scope=='smoke':
        chosen=[next(i for i,item in enumerate(old['cases']) if item['role']=='primary' and item['entry']['dataset']==dataset)
                for dataset in ('BF','CV','Open120')]
        repeats,jobs,source_repeats=1,1,(0,)
    else:
        smoke=read(out('smoke')/'summary.json')
        if not smoke['budget_gate'] or not smoke['audit_pass']:raise ValueError('Paired wiring smoke failed')
    sources=dict(old['source_sha256'])
    for name in ('paired_trajectory_refinement.py','research_paired_trajectory_screen.py'):
        sources['scripts/'+name]=digest(ROOT/'scripts'/name)
    inputs=[]
    for new_index,old_index in enumerate(chosen):
        for source_repeat in source_repeats:
            spec=next(i for i in old['inputs'] if i['original_index']==old_index and i['source_repeat']==source_repeat)
            src=previous/spec['input_file']
            if digest(src)!=spec['input_sha256']:raise ValueError('Frozen input changed')
            raw=read(src);filename=f'inputs/c{new_index}__s{source_repeat}.json'
            write_json(folder/filename,raw)
            inputs.append(dict(original_index=new_index,source_repeat=source_repeat,input_file=filename,
                input_sha256=digest(folder/filename),budget_sec=spec['budget_sec']))
    protocol=(
        'Conditional two-trajectory screen on the SAME14 originals,28 common handoffs andactual '
        'residual allowances as the completed residual-budget study. Historicalprefix generation '
        'isnot rerun;do notclaim fresh raw-bay20s evidence. Keep unchanged12 primary originals '
        'andthetwo separately labeled diagnostic losses. Both sourcehandoffs andtwo fresh '
        'execution repeats,seven methods,four pinned lanes,case-cyclic/repeat-reversed order. '
        'Only imports/source checks/input loading precedeREADY. AfteroriginalSTART charge every '
        'new model,actual seed,kernel,FRG,trajectory replay,publication andswitch. Each branch '
        'starts from theactual commoninput,never from another method answer. Q+Pair alternates '
        'full route/FRG passes on independent cheapQ andfour-startPair trajectories. Single+Pair '
        'is the matched no-Q controller withthe same schedule;equivalent50msMILP+Pair is a '
        'separate classical model/backend control. Solo cyclicSingle,Pair andQ are also freshly '
        'executed. Branches keep their own shorter paths,including a longer searchanchor when '
        'theother branch already improved theglobalbest. Only strictly shorter global paths '
        'arepublished asINCUMBENT andcredited ifready AND received bydeadline. Every published '
        'INCUMBENT andevery completeTRAJECTORY_END path is independently physically replayed, '
        'including lateuncredited messages. Branch inputs/signatures,order,rounds,originalclock, '
        'paidseed kinds,local modelenergy andglobal incumbent cost are checked separately. '
        'ThedeclaredPair branch in aQ actor isclassical work,not aQbackend seed. No MILPcall '
        'isallowed insideQ orclassical seed calculations. EquivalentMILP calls areexplicit. '
        'UngatedQ+Pair is thepredeclared primary target. Aseparately labeled gatedQ+Pair uses '
        'cyclicSingle alone ifactualentryremainingtime<2s;record zeroQ requests there anddo '
        'notcredit thisfallback asQ benefit. Do notselect thebettervariant afteroutcomes and '
        'present itasindependentconfirmation. Report bothvariations andalllosses/clockcuts. '
        'Original-level means average available source/execution runs withinoriginal,not '
        'best-of-repeats orsubmodels. Primary core screen requires valid physical/budget/seed '
        'audits,atleast2stable primary Q+Pair wins withactualprimary joint publications against '
        'Single+Pair,negative totals inboth execution repeats versusSingle+Pair ANDsoloPair, '
        'andnegative upper95%original-mean CI versusSingle+Pair. Aseparate declared tail '
        'mechanism check requiresQ+Pair tobe no worse thansoloPair on BOTH fresh execution '
        'repeats ofCV10-6case26 source1,withactualindependentPairbranch entry/publication. '
        'Core andtail checks onlypermit planning aNEW fresh total-time experiment;they do '
        'notpromote defaults orcomplete theglobalgoal. Gateeffects andequivalentMILP results '
        'areseparate. Smoke usesfirstBF/CV/Open primary originals,source0,one repeat. Preserve '
        'thefailed2s schedule screen andallpastresults. No hardware orworldwidepriority claim.')
    write_json(folder/'manifest.json',dict(cases=[old['cases'][i] for i in chosen],inputs=inputs,modes=MODES,
        repeats=repeats,jobs=jobs,cpus=[0,2,4,6][:jobs],budget_sec=20.,source_sha256=sources,
        protocol=protocol,frozen_at=datetime.now().isoformat()))
    print(json.dumps(dict(scope=scope,originals=len(chosen),inputs=len(inputs),modes=MODES)))


def worker(scope,index,repeat,mode,cpu):
    frozen=base.manifest(scope);item,raw=base.load_input(scope,frozen,index)
    if digest(ROOT/item['entry']['path'])!=item['entry']['sha256']:raise ValueError('Original bay changed')
    base.broad.pin_cpu(cpu);contract=PriorityGroupContract();contract.install()
    def publish(event):print(json.dumps(event,separators=(',',':')),flush=True)
    try:
        instance=base.broad.load_case(item['entry']);path=tuple(Move(**v) for v in raw['path'])
        random.seed(37);np.random.seed(37)
        publish(dict(type='READY',case=raw['case'],mode=mode,cpu=cpu,initial=len(path)))
        clock=json.loads(sys.stdin.readline())
        if clock['budget']!=raw['budget_sec']:raise ValueError('Parent changed the residual allowance')
        publish(dict(type='START_ACK',absolute_start=clock['start'],budget_sec=clock['budget']))
        result=improve(instance,path,mode,clock['start'],clock['budget'],publish)
        publish(dict(type='DONE',moves=result['moves'],elapsed_sec=REAL_TIME()-clock['start'],
            branch_moves=result.get('branch_moves'),branch_passes=result.get('branch_passes'),
            stop_reason=result.get('stop_reason'),gate=result.get('gate'),
            q_search_starts=result.get('q_search_starts')))
    except BaseException:
        publish(dict(type='ERROR',detail=traceback.format_exc()));raise
    finally:contract.restore()


def execute(scope,index,repeat,cpu,frozen):
    item,raw=base.load_input(scope,frozen,index);budget=frozen['inputs'][index]['budget_sec']
    header=dict(case=raw['case'],dataset=item['entry']['dataset'],role=item['role'],
                source_repeat=raw['source_repeat'],repeat=repeat,budget_sec=budget)
    if not raw['valid']:return dict(header,status='NO_SOURCE_HANDOFF')
    modes=list(frozen['modes']);offset=index%len(modes);order=modes[offset:]+modes[:offset]
    if repeat%2:order.reverse()
    answers={}
    for mode in order:
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--scope',scope,
            '--input-index',str(index),'--repeat',str(repeat),'--mode',mode,'--cpu',str(cpu)]
        try:answer=run_worker(command,cwd=ROOT,env={**os.environ,**ENVIRONMENT},budget=budget)
        except Exception as error:answer=dict(status='WORKER_ERROR',error=str(error),messages=[])
        answer['initial']=raw['moves']
        eligible=[e for e in answer['messages'] if e['type']=='INCUMBENT' and e['ready_sec']<=budget and e['received_sec']<=budget]
        best=min(eligible,key=lambda e:e['moves']) if eligible else raw
        answer.update(moves=best['moves'],path=best['path']);answers[mode]=answer
    instance=base.broad.load_case(item['entry']);audit_errors=[];physical_checks=0
    physical_replay(instance.stacks,instance.max_height,tuple(Move(**v) for v in raw['path']),goal=True)
    for mode,answer in answers.items():
        previous=raw['moves']
        for event in answer['messages']:
            if event['type'] not in ('INCUMBENT','TRAJECTORY_END'):continue
            try:
                physical_replay(instance.stacks,instance.max_height,tuple(Move(**v) for v in event['path']),goal=True)
                physical_checks+=1
                count=event['moves'] if event['type']=='INCUMBENT' else event['after']
                if len(event['path'])!=count:raise ValueError('Incorrect full-path cost')
                if event['type']=='INCUMBENT':
                    if count>=previous:raise ValueError('Nonmonotone global publication')
                    previous=count
            except (ValueError,IndexError) as error:
                audit_errors.append(dict(mode=mode,type=event['type'],detail=str(error)))
    return dict(header,status='COMPLETE',input=raw['moves'],answers=answers,order=order,cpu=cpu,
                physical_checks=physical_checks,physical_errors=audit_errors)


def audit_answer(answer,initial,mode,budget):
    errors=[];events=[e for e in answer['messages'] if e['received_sec']<=budget]
    def require(condition,message):
        if not condition:errors.append(message)
    acks=[e for e in events if e['type']=='START_ACK']
    require(len(acks)==1 and acks[0]['budget_sec']==budget,'Missing/wrong original clock')
    original_start=acks[0]['absolute_start'] if acks else None
    gates=[e for e in events if e['type']=='PORTFOLIO_GATE']
    fallback=bool(gates)
    if fallback:
        require(mode=='gated_qubo_pair' and len(gates)==1,'Unexpected gate')
        for gate in gates:
            require(gate['remaining_sec']<2. and gate['threshold_sec']==2. and gate['selected_policy']=='single_cycle'
                and gate['absolute_start']==original_start and gate['budget_sec']==budget,'Wrong gate policy/clock')
    branch_paths=[initial,initial];rounds=[0,0];active=[True,True];pending=None;expected_branch=0
    global_moves=len(initial);joint_publications=0;pair_publications=0;branch_inputs=[]
    q_requests=0;models=0;strict=0
    for event in events:
        kind=event['type']
        if kind=='INCUMBENT' and event['ready_sec']<=budget:
            global_moves=event['moves']
            joint_publications+=event.get('original_source',event['source'])=='qubo_joint'
            pair_publications+=event.get('trajectory_branch')==1
        elif kind in ('CYCLE_BEGIN','TRAJECTORY_BEGIN'):
            require(event['absolute_start']==original_start and event['budget_sec']==budget,'Rebased branch/cycle clock')
        if kind=='TRAJECTORY_BEGIN':
            branch=event['branch'];policies=(PRIMARY.get(mode),'pair')
            require(mode not in SOLO and not fallback,'Unexpected paired branch')
            require(pending is None and branch==expected_branch and active[branch],'Wrong trajectory alternation')
            require(event['round']==rounds[branch] and event['policy']==policies[branch],'Wrong branch policy/round')
            require(event['before']==len(branch_paths[branch]) and event['path_signature']==fingerprint(branch_paths[branch]),
                    'Branch did not use its own actual path')
            require(event['global_moves']==global_moves,'Wrong global ledger at branch entry')
            pending=(branch,event['round'],event['before'])
            branch_inputs.append(dict(branch=branch,round=event['round'],policy=event['policy'],
                                      path_signature=event['path_signature'],before=event['before']))
        elif kind=='TRAJECTORY_END':
            branch=event['branch']
            require(pending==(branch,event['round'],event['before']),'Unmatched branch completion')
            require(event['after']<=event['before'] and len(event['path'])==event['after'],'Invalid branch improvement')
            require(event['global_moves']==global_moves and global_moves<=event['after'],'Wrong global/branch costs')
            branch_paths[branch]=event['path'];rounds[branch]+=1;active[branch]=event['active'];pending=None
            require(active[branch]==bool(event['after'] and event['after']<event['before']),'Wrong branch stop rule')
            expected_branch=1-branch if active[1-branch] else branch
        elif kind=='PHASE' and event.get('phase')=='qubo_search':q_requests+=1
        elif kind=='MODEL':
            model=event['record'];models+=1
            policy=model.get('trajectory_policy',SOLO.get(mode,'single' if fallback else None))
            require(policy in ('single','single_qubo','single_oracle','pair'),'Missing actual model policy')
            if mode not in SOLO and not fallback:
                branch=model.get('trajectory_branch')
                require(pending is not None and (branch,model.get('trajectory_round'))==pending[:2],
                        'Model attributed to a different active branch')
                require(policy==(PRIMARY.get(mode) if branch==0 else 'pair'),'Model policy differs from declared branch')
            seed=model['seed_metadata'];pair=policy=='pair'
            require(seed['seed_kind']==('four_start_pair' if pair else 'three_order_single'),'Wrong model seed kind')
            require(len(seed['single_trials'])==3 and len(seed['pair_trials'])==(4 if pair else 0),'Wrong actual seed starts')
            if pair:require(seed['selected_metadata']['pair_local_optimal'],'Pair seed not locally converged')
            require(model['nonoracle_milp_calls']==0,'MILP used in Q/classical seed calculation')
            require(model['energy']==model['after']<=model['seed_moves']<=model['before'],'Wrong branch model energy')
            if 'global_moves' in model:require(model['global_moves']<=model['after'],'Model confuses branch/global costs')
            backend=model.get('backend') or {}
            require(not(backend.get('cutoff') and backend.get('optimal')),'False cutoff certificate')
            if policy=='single_qubo':require(backend.get('local_time_limit_sec')==.05,'Changed Q kernel limit')
            if model['energy']<model['seed_moves']:
                strict+=1;require(model['changed_groups_from_seed']>=(3 if pair else 2),'Contradicts seed local optimum')
    if fallback:require(q_requests==0 and not branch_inputs,'Gate performed hidden Q/Pair work')
    require(base.at(answer,budget)==answer['moves'],'Credited late or unpublished answer')
    return dict(errors=errors,gate=fallback,models=models,strict_models=strict,q_requests=q_requests,
        joint_publications=joint_publications,pair_publications=pair_publications,
        pair_started=any(e['branch']==1 for e in branch_inputs),branch_inputs=branch_inputs,
        completed_rounds=rounds)


def analyze(scope):
    frozen=base.manifest(scope);rows=[];audits=[];missing=[];failures=[];errors=[];physical=0;tail=[]
    for index,spec in enumerate(frozen['inputs']):
        source=read(out(scope)/spec['input_file'])
        for repeat in range(frozen['repeats']):
            data=read(out(scope)/'runs'/f'i{index}__r{repeat}.json')
            key={k:data[k] for k in ('case','dataset','role','source_repeat','repeat')}
            if data['status']!='COMPLETE':missing.append(dict(key,status=data['status']));continue
            budget=spec['budget_sec'];physical+=data['physical_checks']
            errors.extend(dict(key,**error) for error in data['physical_errors'])
            for mode,answer in data['answers'].items():
                if answer['status']=='WORKER_ERROR' or answer.get('termination_lag_sec',0.)>.1:
                    failures.append(dict(key,mode=mode,status=answer['status'],error=answer.get('error'),lag=answer.get('termination_lag_sec')))
                if answer.get('budget_sec')!=budget:errors.append(dict(key,mode=mode,error='Wrong parent allowance'))
                checked=audit_answer(answer,source['path'],mode,budget)
                errors.extend(dict(key,mode=mode,error=message) for message in checked['errors'])
                audits.append(dict(key,mode=mode,**checked))
                rows.append(dict(key,mode=mode,budget_sec=budget,moves=answer['moves'],input=data['input'],
                    status=answer['status'],gate=checked['gate'],joint_paths=checked['joint_publications'],
                    pair_started=checked['pair_started'],pair_publications=checked['pair_publications']))
            if data['case']=='CV__10-6__data10-6-26' and data['source_repeat']==1:
                q=data['answers']['qubo_pair'];p=data['answers']['pair_cycle']
                info=next(a for a in reversed(audits) if a['case']==data['case'] and a['source_repeat']==1
                          and a['repeat']==repeat and a['mode']=='qubo_pair')
                tail.append(dict(key,qubo_pair=q['moves'],solo_pair=p['moves'],
                    pair_started=info['pair_started'],pair_publications=info['pair_publications']))
    table={(r['case'],r['source_repeat'],r['repeat'],r['mode']):r for r in rows}
    rng=random.Random(2026101291);contrasts={}
    for target in ('qubo_pair','gated_qubo_pair'):
        for role in sorted({item['role'] for item in frozen['cases']}):
            for reference in [m for m in frozen['modes'] if m!=target]:
                originals=[]
                for item in frozen['cases']:
                    if item['role']!=role:continue
                    case=item['entry']['case'];deltas=[]
                    for spec in frozen['inputs']:
                        if frozen['cases'][spec['original_index']]['entry']['case']!=case:continue
                        for repeat in range(frozen['repeats']):
                            q=table.get((case,spec['source_repeat'],repeat,target));p=table.get((case,spec['source_repeat'],repeat,reference))
                            if q and p:deltas.append(dict(source_repeat=spec['source_repeat'],repeat=repeat,
                                delta=q['moves']-p['moves'],joint=q['joint_paths']>0,gate=q['gate']))
                    if deltas:originals.append(dict(case=case,mean=sum(d['delta'] for d in deltas)/len(deltas),
                        stable_win=all(d['delta']<0 for d in deltas),joint_all=all(d['joint'] for d in deltas),deltas=deltas))
                values=[r['mean'] for r in originals]
                samples=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(10000)) if values else []
                contrasts[f'{target}:{role}:{reference}']=dict(originals=len(values),wins=sum(v<0 for v in values),
                    losses=sum(v>0 for v in values),ties=sum(v==0 for v in values),net_original_mean=sum(values),
                    worst=max(values,default=0),stable_wins=sum(r['stable_win'] for r in originals),
                    stable_joint_wins=sum(r['stable_win'] and r['joint_all'] for r in originals),
                    by_repeat={rep:sum(sum(d['delta'] for d in r['deltas'] if d['repeat']==rep)/sum(d['repeat']==rep for d in r['deltas'])
                        for r in originals) for rep in range(frozen['repeats'])},
                    bootstrap95_mean=[samples[250],samples[9749]] if samples else None,details=originals)
    valid=not failures and all(m['status']=='NO_SOURCE_HANDOFF' for m in missing)
    paired=contrasts['qubo_pair:primary:single_pair'];solo=contrasts['qubo_pair:primary:pair_cycle']
    core=(scope=='pilot' and valid and not errors and paired['stable_joint_wins']>=2
        and all(v<0 for v in paired['by_repeat'].values()) and all(v<0 for v in solo['by_repeat'].values())
        and paired['bootstrap95_mean'] is not None and paired['bootstrap95_mean'][1]<0)
    tail_pass=(scope=='pilot' and len(tail)==2 and all(r['qubo_pair']<=r['solo_pair'] and r['pair_started']
                                                   and r['pair_publications']>0 for r in tail))
    result=dict(scope=scope,originals=len(frozen['cases']),budget_gate=valid,audit_pass=not errors,
        budget_failures=failures,audit_errors=errors,missing=missing,physical_path_checks=physical,
        contrasts=contrasts,primary_core_screen=core,tail_mechanism_screen=tail_pass,
        forward_screen=core and tail_pass,tail_mechanism=tail,
        gate_runs=sum(r['gate'] for r in rows),
        scope_boundary='Conditional paid trajectory portfolio;historical prefix excluded;no promotion')
    write_json(out(scope)/'summary.json',result);write_csv(out(scope)/'results.csv',rows)
    write_json(out(scope)/'branch_audit.json',dict(rows=audits,errors=errors))
    report_keys=[f'qubo_pair:primary:{m}' for m in ('single_pair','pair_cycle','qubo_cycle','oracle_pair')]
    report_keys+=['gated_qubo_pair:primary:qubo_pair','gated_qubo_pair:posthoc_conditional_loss:single_cycle',
                  'qubo_pair:posthoc_conditional_loss:pair_cycle']
    print(json.dumps({**{k:v for k,v in result.items() if k!='contrasts'},'contrasts':
        {k:{a:b for a,b in contrasts[k].items() if a!='details'} for k in report_keys if k in contrasts}}),flush=True)


base.out=out
base.MODES=MODES
base.__file__=__file__
base.execute=execute
base.analyze=analyze


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','run','worker','analyze'))
    parser.add_argument('--scope',choices=('smoke','pilot'),required=True)
    parser.add_argument('--input-index',type=int);parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--mode',choices=MODES);parser.add_argument('--cpu',type=int,default=0)
    args=parser.parse_args()
    if args.command=='worker':worker(args.scope,args.input_index,args.repeat,args.mode,args.cpu)
    elif args.command=='run':base.run(args.scope)
    else:globals()[args.command](args.scope)
