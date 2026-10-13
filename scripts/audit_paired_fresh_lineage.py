"""Matched raw primary lineage plus actual reported joint exposure in fresh runs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from audit_shared_prefix_lineage import semantic
from audit_paired_trajectory_lineage import primary
from research_paired_fresh import raw_ledger_view

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/experiments/action_coupling_qubo_20261011'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main(scope,family='paired_fresh'):
    folder=BASE/(family+'_'+scope)
    if read(folder/'status.json')['status']!='COMPLETE':raise ValueError('Wait for the complete fresh study')
    frozen=read(folder/'manifest.json');rows=[];originals=[];violations=[]
    for index,item in enumerate(frozen['cases']):
        repeated=[]
        for repeat in range(frozen['repeats']):
            data=read(folder/'runs'/f'c{index}__r{repeat}.json')
            if data['status']!='COMPLETE':raise ValueError('Incomplete fresh block')
            budget=frozen['budget_sec'];actors=data['actors']
            qview=raw_ledger_view(actors['qubo_pair'],budget);sview=raw_ledger_view(actors['single_pair'],budget)
            qe,qm,qj=primary(qview,budget) if qview is not None else ({},[],[])
            se,sm,sj=primary(sview,budget) if sview is not None else ({},[],[])
            bykey={(m['trajectory_round'],m['index']):m for m in sm}
            first=next((m for m in qm if m['energy']<m['seed_moves']),None)
            entry_checks=[];model_checks=[]
            if first is not None:
                stop=(first['trajectory_round'],first['index'])
                for round_index in range(first['trajectory_round']+1):
                    q=qe.get(round_index);s=se.get(round_index)
                    match=bool(q and s and q['path_signature']==s['path_signature'] and q['before']==s['before'])
                    entry_checks.append(dict(round=round_index,matched=match))
                for q in qm:
                    key=(q['trajectory_round'],q['index'])
                    if key>stop:break
                    s=bykey.get(key)
                    match=bool(s and all(q[k]==s[k] for k in ('before','seed_moves','variables'))
                        and semantic(q['model'])==semantic(s['model'])
                        and semantic(q['seed_metadata'])==semantic(s['seed_metadata']))
                    if match and key<stop:match=q['selected']==s['selected'] and q['energy']==s['energy']
                    if match and key==stop:match=q['seed_metadata']['actual_seed_selection']==s['selected']
                    model_checks.append(dict(round=key[0],index=key[1],matched=match))
                if first['seed_kind']!='three_order_single' or first['changed_groups_from_seed']<2:
                    violations.append(dict(case=item['entry']['case'],repeat=repeat,reason='Wrong primary seed/gain'))
            matched=bool(first and entry_checks and model_checks and all(c['matched'] for c in entry_checks+model_checks))
            publications=[e for e in actors['qubo_pair']['messages'] if e['type']=='INCUMBENT'
                          and e['received_sec']<=budget and e['ready_sec']<=budget]
            reported_joint=sum((e.get('refinement_provenance') or {}).get('original_source')=='qubo_joint' for e in publications)
            q=data['answers']['qubo_pair'];s=data['answers']['single_pair']
            delta=q['moves']-s['moves'] if q['valid'] and s['valid'] else None
            row=dict(case=item['entry']['case'],repeat=repeat,
                status='NO_RECEIVED_PRIMARY_JOINT_MODEL' if first is None else 'MATCHED_PRIMARY_PREFIX' if matched else 'INCOMPLETE_PREFIX',
                matched_first_joint=matched,raw_joint_publications=len(qj),reported_joint_publications=reported_joint,
                delta=delta,final_source=q['source'],final_provenance=q['provenance'],
                first_joint=None if first is None else dict(round=first['trajectory_round'],index=first['index'],
                    seed=first['seed_moves'],energy=first['energy'],changed_groups=first['changed_groups_from_seed']),
                entry_checks=entry_checks,model_checks=model_checks)
            rows.append(row);repeated.append(row)
        valid=len(repeated)==frozen['repeats'] and all(r['delta'] is not None for r in repeated)
        originals.append(dict(case=item['entry']['case'],valid=valid,
            stable_win=valid and all(r['delta']<0 for r in repeated),
            matched_reported_joint_all=all(r['matched_first_joint'] and r['reported_joint_publications']>0 for r in repeated),
            deltas=[r['delta'] for r in repeated]))
    wins=[r for r in rows if r['delta'] is not None and r['delta']<0]
    result=dict(scope=scope,blocks=len(rows),statuses=dict(Counter(r['status'] for r in rows)),violations=violations,
        primary_stable_wins_with_matched_joint=sum(r['stable_win'] and r['matched_reported_joint_all'] for r in originals),
        winning_execution_sources=dict(Counter(str((r['final_provenance'] or {}).get('trajectory_branch','prefix')) for r in wins)),
        originals=originals,rows=rows,
        scope_boundary='Raw primary model/seed paths establish lineage;onlyouter reported joint publications '
            'count as actualsolution exposure. Raw telemetry neverchanges scores. Thisisnot matrix '
            'reconstruction,globalCPMP optimality orhardware advantage.')
    (folder/'lineage_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','originals','scope_boundary')}))
    if violations:raise ValueError('Fresh paired lineage invariant failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=('smoke','pilot','hardset169'),required=True)
    parser.add_argument('--family',choices=('paired_fresh','paired_fresh_stream','paired_fresh_exploratory'),default='paired_fresh')
    args=parser.parse_args();main(args.scope,args.family)
