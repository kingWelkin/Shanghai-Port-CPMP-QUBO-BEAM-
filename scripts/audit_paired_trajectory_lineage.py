"""Match the primary trajectory before its first joint gain in the paid portfolio."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

from audit_shared_prefix_lineage import semantic

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/experiments/action_coupling_qubo_20261011'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def primary(answer,budget):
    events=[e for e in answer['messages'] if e['received_sec']<=budget]
    entries={e['round']:e for e in events if e['type']=='TRAJECTORY_BEGIN' and e['branch']==0}
    models=[e['record'] for e in events if e['type']=='MODEL' and e['record'].get('trajectory_branch')==0]
    joints=[e for e in events if e['type']=='INCUMBENT' and e['ready_sec']<=budget
            and e.get('trajectory_branch')==0 and e.get('original_source')=='qubo_joint']
    return entries,models,joints


def main(scope):
    folder=BASE/('paired_trajectory_screen_'+scope)
    if read(folder/'status.json')['status']!='COMPLETE':raise ValueError('Wait for the complete timed cohort')
    frozen=read(folder/'manifest.json');rows=[];grouped=defaultdict(list);violations=[]
    for index,spec in enumerate(frozen['inputs']):
        for repeat in range(frozen['repeats']):
            data=read(folder/'runs'/f'i{index}__r{repeat}.json')
            if data['status']!='COMPLETE':continue
            budget=spec['budget_sec'];answers=data['answers']
            qe,qm,qj=primary(answers['qubo_pair'],budget)
            se,sm,sj=primary(answers['single_pair'],budget)
            bykey={(m['trajectory_round'],m['index']):m for m in sm}
            first=next((m for m in qm if m['energy']<m['seed_moves']),None)
            checks=[];entries=[]
            if first:
                stop=(first['trajectory_round'],first['index'])
                for round_index in range(first['trajectory_round']+1):
                    q=qe.get(round_index);s=se.get(round_index)
                    match=bool(q and s and q['path_signature']==s['path_signature'] and q['before']==s['before'])
                    entries.append(dict(round=round_index,matched=match))
                for q in qm:
                    key=(q['trajectory_round'],q['index'])
                    if key>stop:break
                    s=bykey.get(key)
                    match=bool(s and all(q[k]==s[k] for k in ('before','seed_moves','variables'))
                        and semantic(q['model'])==semantic(s['model'])
                        and semantic(q['seed_metadata'])==semantic(s['seed_metadata']))
                    if match and key<stop:match=q['selected']==s['selected'] and q['energy']==s['energy']
                    if match and key==stop:match=q['seed_metadata']['actual_seed_selection']==s['selected']
                    checks.append(dict(round=key[0],index=key[1],matched=match))
                if first['changed_groups_from_seed']<2 or first['seed_kind']!='three_order_single':
                    violations.append(dict(case=data['case'],source_repeat=data['source_repeat'],repeat=repeat,
                                          reason='Primary gain contradicts declared single seed'))
            matched=bool(first and entries and checks and all(e['matched'] for e in entries+checks))
            publications=[e for e in answers['qubo_pair']['messages'] if e['type']=='INCUMBENT'
                          and e['ready_sec']<=budget and e['received_sec']<=budget]
            final=min(publications,key=lambda e:e['moves']) if publications else None
            row=dict(case=data['case'],role=data['role'],source_repeat=data['source_repeat'],repeat=repeat,
                status='NO_RECEIVED_PRIMARY_JOINT_MODEL' if first is None else
                    'MATCHED_PRIMARY_TRAJECTORY_AND_SINGLE_PREFIX' if matched else 'INCOMPLETE_PREFIX',
                matched_first_joint=matched,joint_publications=len(qj),
                final_branch=None if final is None else final.get('trajectory_branch'),
                final_source='input' if final is None else final['source'],
                delta_classical_portfolio=answers['qubo_pair']['moves']-answers['single_pair']['moves'],
                delta_solo_pair=answers['qubo_pair']['moves']-answers['pair_cycle']['moves'],
                first_joint=None if first is None else dict(round=first['trajectory_round'],index=first['index'],
                    seed=first['seed_moves'],energy=first['energy'],changed_groups=first['changed_groups_from_seed']),
                entry_checks=entries,model_checks=checks)
            rows.append(row);grouped[row['case']].append(row)
    originals=[dict(case=case,role=rs[0]['role'],
        stable_classical_portfolio_win=all(r['delta_classical_portfolio']<0 for r in rs),
        stable_solo_pair_win=all(r['delta_solo_pair']<0 for r in rs),
        final_primary_all=all(r['final_branch']==0 for r in rs),
        final_pair_all=all(r['final_branch']==1 for r in rs),
        matched_joint_all=all(r['matched_first_joint'] and r['joint_publications'] for r in rs))
        for case,rs in grouped.items()]
    summary=dict(scope=scope,blocks=len(rows),statuses=dict(Counter(r['status'] for r in rows)),
        violations=violations,
        primary_stable_classical_portfolio_wins_with_matched_joint=sum(r['role']=='primary'
            and r['stable_classical_portfolio_win'] and r['matched_joint_all'] for r in originals),
        primary_stable_solo_pair_wins_with_matched_joint=sum(r['role']=='primary'
            and r['stable_solo_pair_win'] and r['matched_joint_all'] for r in originals),
        winning_execution_sources=dict(Counter('primary' if r['final_branch']==0 else
            'independent_pair' if r['final_branch']==1 else 'input' for r in rows if r['delta_classical_portfolio']<0)),
        originals=originals,rows=rows,
        scope_boundary='Primary branch entry signatures andreceived model/seed metadata arematched beforefirstjoint change. '
            'Independent Pair branches arepaid controls;their paths arechecked bythefrozen driver. Thisdoesnot '
            'reconstruct matrices orclaimwholehardset/native/hardware superiority. Winningexecution '
            'source counts areprovenance diagnostics,not independent-original sample sizes. Awin '
            'ending onPair mayreflect scheduling interactions;do notcallit adirectQ-generated finalpath.')
    (folder/'lineage_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('originals','rows','scope_boundary')}))
    if violations:raise ValueError('Portfolio primary lineage invariant failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=('smoke','pilot'),required=True)
    main(parser.parse_args().scope)
