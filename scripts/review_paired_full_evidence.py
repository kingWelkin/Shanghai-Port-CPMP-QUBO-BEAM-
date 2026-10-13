"""Post-completion accounting; never turns a component gate into goal completion."""
from __future__ import annotations

from collections import Counter,defaultdict
import json
from pathlib import Path

from shared_wall_deadline_protocol_full import credited_path

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'docs/experiments/action_coupling_qubo_20261011/paired_fresh_exploratory_hardset169'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    if read(FOLDER/'status.json')['status']!='COMPLETE':raise ValueError('Wait for the entire timed cohort')
    manifest=read(FOLDER/'manifest.json');summary=read(FOLDER/'summary.json')
    lineage=read(FOLDER/'lineage_audit.json')
    checks=[];runs={};coverage=defaultdict(lambda:Counter());path_messages=0
    actual_joint=0;models=Counter();final_sources=Counter()
    if len(manifest['cases'])!=169 or Counter(c['entry']['dataset'] for c in manifest['cases'])!=dict(BF=73,CV=84,Open120=12):
        checks.append('Corpus is not the complete confirmed hardset169')
    for index,item in enumerate(manifest['cases']):
        for repeat in range(manifest['repeats']):
            d=read(FOLDER/'runs'/f'c{index}__r{repeat}.json')
            if d['status']!='COMPLETE':
                checks.append(f'Incomplete block:{index}/{repeat}');continue
            prefix=d['actors']['prefix']['messages'];budget=manifest['budget_sec']
            counted=sum(e['type'] in ('INCUMBENT','HANDOFF','REFINEMENT_INCUMBENT','TRAJECTORY_END')
                        for a in d['actors'].values() for e in a['messages'])
            if counted!=d['physical_checks'] or d['physical_errors']:
                checks.append(f'Physical path accounting mismatch:{index}/{repeat}')
            path_messages+=counted
            for mode,answer in d['answers'].items():
                source=[] if mode in ('native','bs_frg') else prefix
                best=credited_path(source,d['actors'][mode]['messages'],budget)
                if answer['valid']!=(best is not None) or answer['moves']!=(None if best is None else best['moves']):
                    checks.append(f'Receipt-based score mismatch:{index}/{repeat}/{mode}')
                runs[(item['entry']['case'],repeat,mode)]=answer
                key=item['entry']['dataset']+':'+mode
                coverage[key]['attempts']+=1;coverage[key]['feasible']+=int(answer['valid'])
            q=d['answers']['qubo_pair'];final_sources[str((q['provenance'] or {}).get('trajectory_branch','prefix_or_none'))]+=1
            for e in d['actors']['qubo_pair']['messages']:
                if e['received_sec']>budget:continue
                if e['type']=='INCUMBENT' and e['ready_sec']<=budget:
                    actual_joint+=(e.get('refinement_provenance') or {}).get('original_source')=='qubo_joint'
                if e['type']=='MODEL':
                    m=e['record'];policy=m.get('trajectory_policy','unknown');models[policy+':completed']+=1
                    models[policy+':strict_seed_gain']+=m['energy']<m['seed_moves']
                    meta=m.get('backend') or {};models[policy+':certified']+=bool(meta.get('optimal'))
                    models[policy+':wall_cut']+=bool(meta.get('wall_cutoff'))
    comparisons={}
    for ref in (m for m in manifest['modes'] if m!='qubo_pair'):
        originals=[]
        for item in manifest['cases']:
            case=item['entry']['case'];outcomes=[];deltas=[];feasibility=[]
            for repeat in range(manifest['repeats']):
                q=runs[(case,repeat,'qubo_pair')];r=runs[(case,repeat,ref)]
                if q['valid'] and r['valid']:
                    delta=q['moves']-r['moves'];deltas.append(delta)
                    outcome='quality_win' if delta<0 else 'quality_loss' if delta>0 else 'quality_tie'
                elif q['valid']:outcome='feasibility_win'
                elif r['valid']:outcome='feasibility_loss'
                else:outcome='both_unresolved'
                outcomes.append(outcome);feasibility.append(int(q['valid'])-int(r['valid']))
            originals.append(dict(case=case,dataset=item['entry']['dataset'],outcomes=outcomes,
                mean_move_delta=sum(deltas)/len(deltas) if len(deltas)==manifest['repeats'] else None,
                feasibility_delta=sum(feasibility),robust_feasibility_loss=all(v=='feasibility_loss' for v in outcomes)))
        common=[r for r in originals if r['mean_move_delta'] is not None]
        recomputed=sum(r['mean_move_delta'] for r in common)
        declared=summary['contrasts'][ref]
        if len(common)!=declared['common_originals'] or recomputed!=declared['net_original_mean']:
            checks.append('Original-mean score aggregation mismatch:'+ref)
        comparisons[ref]=dict(attempted_originals=len(originals),common_originals=len(common),
            original_mean_move_total=recomputed,
            paired_execution_outcomes=dict(Counter(v for r in originals for v in r['outcomes'])),
            robust_feasibility_losses=[r['case'] for r in originals if r['robust_feasibility_loss']],
            largest_common_losses=sorted([r for r in common if r['mean_move_delta']>0],
                                         key=lambda r:(-r['mean_move_delta'],r['case']))[:15],originals=originals)
    joint_models=Counter();joint_rows=[]
    for item in manifest['cases']:
        case=item['entry']['case'];values={mode:[] for mode in ('single_pair','qubo_pair','oracle_pair')}
        for mode in values:
            for repeat in range(manifest['repeats']):
                a=runs[(case,repeat,mode)]
                if a['valid']:values[mode].append(a['moves'])
        if not all(len(v)==manifest['repeats'] for v in values.values()):continue
        means={mode:sum(v)/len(v) for mode,v in values.items()}
        qgain=means['qubo_pair']<means['single_pair'];mgain=means['oracle_pair']<means['single_pair']
        category='both_joint_backends_gain' if qgain and mgain else 'Q_only_gain' if qgain else 'MILP_only_gain' if mgain else 'neither_gains'
        joint_models[category]+=1;joint_rows.append(dict(case=case,category=category,means=means))
    result=dict(scope='complete exploratory169 evidence review',accounting_errors=checks,
        complete_corpus_and_accounting=not checks,attempts=len(runs),independently_checked_path_messages=path_messages,
        family_coverage={k:dict(v) for k,v in coverage.items()},comparisons=comparisons,
        Q_actor_models_by_actual_policy=dict(models),timely_reported_joint_publications=actual_joint,
        final_Q_actor_sources=dict(final_sources),joint_model_comparison=dict(joint_models),joint_model_rows=joint_rows,
        declared_full_component_gate=summary['forward_screen'],lineage_violations=lineage['violations'],
        stable_originals_with_matched_reported_joint=lineage['primary_stable_wins_with_matched_joint'],
        original_pilot_gate_passed=manifest['original_pilot_forward_screen'],
        completion_boundary=dict(global_goal_automatically_complete=False,untouched_test_set=False,
            generic_matrix_only_backend_measured=False,quantum_hardware_measured=False,
            fresh_solo_refinement_panel_measured=False,
            fresh_controls=list(manifest['modes']),
            interpretation='Shared Q/MILP gains support the joint model. Differences between their implemented bounded '
                'pipelines may support a measured backend/scheduling advantage,not inherent QUBO orhardware superiority. '
                'The Pair trajectory is paid classical protection. Native/BS gaps include shared preprocessing and '
                'cannot allbecredited toQ. Both-unresolved runs are reported separately,not hidden asquality ties. '
                'SoloSingle/Pair/Q were measured conditionally on14 originals,not inthisfull freshpanel. '
                'Do notinfer globalgoalcompletion ordefaultpromotion fromthecomponentgate alone.'))
    (FOLDER/'evidence_review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(accounting_errors=checks,attempts=len(runs),path_messages=path_messages,
        family_coverage=result['family_coverage'],joint_model_comparison=dict(joint_models),
        declared_full_component_gate=result['declared_full_component_gate'],
        stable_originals_with_matched_reported_joint=result['stable_originals_with_matched_reported_joint'])))
    if checks:raise ValueError('Full-corpus accounting review failed')


if __name__=='__main__':main()
