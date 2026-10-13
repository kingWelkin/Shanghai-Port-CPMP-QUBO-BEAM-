"""Check actual cycle-entry paths and first-joint metadata against cyclic single."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

from audit_shared_prefix_lineage import semantic

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/experiments/action_coupling_qubo_20261011'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def trace(answer,initial,budget):
    current=initial;entries={};models=[];joints=0
    for event in answer['messages']:
        if event['received_sec']>budget:continue
        if event['type']=='CYCLE_BEGIN':entries[event['cycle']]=current
        elif event['type']=='INCUMBENT' and event['ready_sec']<=budget:
            current=event['path'];joints+=event.get('source')=='qubo_joint'
        elif event['type']=='MODEL':models.append(event['record'])
    return entries,models,joints


def main(folder_name):
    folder=BASE/folder_name
    if read(folder/'status.json')['status']!='COMPLETE':raise ValueError('Timed study is not complete')
    frozen=read(folder/'manifest.json');rows=[];grouped=defaultdict(list);violations=[]
    for index,spec in enumerate(frozen['inputs']):
        source=read(folder/spec['input_file'])
        for repeat in range(frozen['repeats']):
            data=read(folder/'runs'/f'i{index}__r{repeat}.json')
            if data['status']!='COMPLETE':continue
            budget=data['answers']['single_qubo']['budget_sec']
            qe,qm,qj=trace(data['answers']['single_qubo'],source['path'],budget)
            se,sm,sj=trace(data['answers']['single'],source['path'],budget)
            bykey={(m['cycle'],m['index']):m for m in sm}
            first=next((m for m in qm if m['energy']<m['seed_moves']),None)
            checks=[];cycle_checks=[]
            if first is not None:
                stop=(first['cycle'],first['index'])
                for cycle in range(first['cycle']+1):
                    cycle_checks.append(dict(cycle=cycle,matched=cycle in qe and cycle in se and qe[cycle]==se[cycle]))
                for q in qm:
                    key=(q['cycle'],q['index'])
                    if key>stop:break
                    s=bykey.get(key)
                    match=bool(s and all(q[k]==s[k] for k in ('before','seed_moves','variables'))
                        and semantic(q['model'])==semantic(s['model'])
                        and semantic(q['seed_metadata'])==semantic(s['seed_metadata']))
                    if match and key<stop:match=q['selected']==s['selected'] and q['energy']==s['energy']
                    if match and key==stop:match=q['seed_metadata']['actual_seed_selection']==s['selected']
                    checks.append(dict(cycle=key[0],index=key[1],matched=match))
                if first['changed_groups_from_seed']<2:
                    violations.append(dict(case=data['case'],repeat=repeat,error='Improving only one single-optimal group'))
            matched=bool(first and checks and all(c['matched'] for c in checks+cycle_checks))
            row=dict(case=data['case'],role=data['role'],source_repeat=data['source_repeat'],repeat=repeat,
                status='NO_RECEIVED_JOINT_MODEL' if first is None else 'MATCHED_CYCLE_AND_SINGLE_PREFIX' if matched else 'INCOMPLETE_PREFIX',
                matched_first_joint=matched,joint_paths=qj,
                first_joint=None if first is None else dict(cycle=first['cycle'],index=first['index'],
                    seed=first['seed_moves'],energy=first['energy'],changed_groups=first['changed_groups_from_seed']),
                delta_single=data['answers']['single_qubo']['moves']-data['answers']['single']['moves'],
                delta_pair=data['answers']['single_qubo']['moves']-data['answers']['pair']['moves'],
                delta_once=data['answers']['single_qubo']['moves']-data['answers']['single_qubo_once']['moves'],
                cycle_entry_checks=cycle_checks,model_checks=checks)
            rows.append(row);grouped[row['case']].append(row)
    originals=[dict(case=case,role=rs[0]['role'],stable_single=all(r['delta_single']<0 for r in rs),
        stable_pair=all(r['delta_pair']<0 for r in rs),stable_once=all(r['delta_once']<0 for r in rs),
        matched_joint_all=all(r['matched_first_joint'] and r['joint_paths'] for r in rs)) for case,rs in grouped.items()]
    result=dict(folder=folder_name,blocks=len(rows),statuses=dict(Counter(r['status'] for r in rows)),
        violations=violations,primary_stable_matched_joint={kind:sum(r['role']=='primary' and r['stable_'+kind]
            and r['matched_joint_all'] for r in originals) for kind in ('single','pair','once')},
        originals=originals,rows=rows,
        scope='Actual cycle-entry path andreceived model/seed metadata beforefirstjoint change;not matrix reconstruction. '
              'Physical replay isperformed independently bythefrozen driver. Missingprefix records remainincomplete attribution.')
    (folder/'lineage_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('originals','rows','scope')}))
    if violations:raise ValueError('Cycle local-optimum invariant failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True)
    main(parser.parse_args().folder)
