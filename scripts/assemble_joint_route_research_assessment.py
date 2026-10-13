"""Assemble the final empirical claim audit from completed, separate studies."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/experiments/action_coupling_qubo_20261011'
OUT = BASE / 'final_assessment_20261013'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def favorable(contrast):
    return (contrast['net_original_mean'] < 0
            and all(value < 0 for value in contrast['by_repeat'].values())
            and contrast['bootstrap95_mean'][1] < 0)


def main():
    a = BASE / 'paired_fresh_exploratory_hardset169'
    b = BASE / 'paired_solo_fresh_hardset169'
    sa, sb = read(a / 'summary.json'), read(b / 'summary.json')
    ea, eb = read(a / 'evidence_review.json'), read(b / 'evidence_review.json')
    la, lb = read(a / 'lineage_audit.json'), read(b / 'lineage_audit.json')
    ma, mb = read(a / 'manifest.json'), read(b / 'manifest.json')
    stable_a = {r['case'] for r in la['originals'] if r['stable_win'] and r['matched_reported_joint_all']}
    stable_b = {r['case'] for r in lb['originals'] if r['stable_win'] and r['matched_reported_joint_all']}
    core_files = (
        'scripts/paired_trajectory_refinement.py', 'scripts/paired_shared_pipeline.py',
        'scripts/single_seed_conditioned_refinement.py', 'scripts/categorical_budgeted_qubo.py',
        'scripts/fast_reduced_trajectory_domains.py', 'scripts/shared_prefix_cpmp_pipeline.py')
    source_identity = {name: ma['source_sha256'][name] == mb['source_sha256'][name]
                       == hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in core_files}
    witnesses = read(BASE / 'shared_prefix_wall_pilot/joint_witnesses.json')['witnesses']
    witness_rows = []
    for witness in witnesses:
        strict = [row for row in witness['subset_table']
                  if row['feasible'] and row['retained_move_cost'] < witness['seed_moves']]
        witness_rows.append(dict(case=witness['case'], seed_moves=witness['seed_moves'],
            joint_moves=witness['joint_moves'], changed_groups=witness['changed_groups'],
            tested_subsets=len(witness['subset_table']),
            minimum_changed_groups_for_strict_gain=min(row['changed_groups'] for row in strict)))
    factor = read(BASE / 'factorized_energy_equivalence/summary.json')
    reduced = read(BASE / 'fast_reduced_domains_validation/summary.json')
    tests = (OUT / 'focused_tests.log').read_text(encoding='utf-8-sig')
    checks = {
        'both_full169_studies_complete': all(read(folder / 'status.json')['status'] == 'COMPLETE'
                                            for folder in (a, b)),
        'full_corpus_and_independent_accounting': ea['complete_corpus_and_accounting']
                                                and eb['complete_corpus_and_accounting'],
        'frozen_budget_physical_cpu_and_stream_checks': all(s['budget_gate'] and s['audit_pass']
                                                           for s in (sa, sb)),
        'actual_producer_and_standalone_beam_evidence': sa['actual_beam_checks'] > 0
            and sa['standalone_beam_checks'] > 0 and sb['actual_beam_checks'] > 0,
        'same_current_core_sources_in_both_panels': all(source_identity.values()),
        'paired_core_gain_in_both_full_panels': sa['forward_screen']
            and sb['inherited_anchor_component_rule']
            and favorable(sa['contrasts']['single_pair']) and favorable(sb['contrasts']['single_pair']),
        'actual_matched_joint_lineage_in_both_panels': not la['violations'] and not lb['violations']
            and len(stable_a) >= 10 and len(stable_b) >= 10,
        'full_fresh_strong_classical_solo_controls': sb['strong_solo_quality_rule']
            and all(favorable(sb['contrasts'][mode]) and not sb['contrasts'][mode]['lost_solutions']
                    for mode in ('single_cycle', 'pair_cycle')),
        'standalone_native_and_bs_average_quality_and_aggregate_coverage':
            all(favorable(sa['contrasts'][mode]) and sa['valid_runs']['qubo_pair'] >= sa['valid_runs'][mode]
                for mode in ('native', 'bs_frg')),
        'equivalent_joint_milp_control_measured': 'oracle_pair' in ma['modes']
            and ea['comparisons']['oracle_pair']['attempted_originals'] == 169,
        'physical_multi_group_witnesses': len(witness_rows) == 2
            and all(row['minimum_changed_groups_for_strict_gain'] >= 3 for row in witness_rows),
        'exact_factor_and_reduced_construction_equivalence': factor['qualified'] and reduced['qualified'],
        'focused_energy_deadline_clock_and_pipeline_tests': '20 passed' in tests,
        'previous_negative_protocol_and_pilot_records_retained':
            read(BASE / 'paired_fresh_smoke/status.json')['status'] == 'INVALID_PROTOCOL'
            and not read(BASE / 'paired_fresh_stream_pilot/summary.json')['forward_screen'],
        'current_losses_and_unresolved_runs_retained': sa['contrasts']['single_pair']['losses'] > 0
            and sb['contrasts']['single_pair']['losses'] > 0
            and eb['primary_comparisons']['single_pair']['execution_outcomes']['both_unresolved'] > 0,
    }
    result = dict(
        assembled_at=datetime.now().isoformat(), requirements=checks,
        empirical_objective_evidence_complete=all(checks.values()),
        scope='Reproducible empirical joint-route QUBO module contribution on the specified hardset169, '
              'under the two frozen fresh20s protocols and their measured strong controls.',
        core_source_identity=source_identity,
        matched_stable_originals_per_panel=[len(stable_a), len(stable_b)],
        matched_stable_originals_in_all_four_executions=sorted(stable_a & stable_b),
        physical_coupling_witnesses=witness_rows,
        source_studies=[str(folder.relative_to(ROOT)) for folder in (a, b)],
        selected_test_log=str((OUT / 'focused_tests.log').relative_to(ROOT)),
        interpretation={
            'model': 'Joint complete-route choices produce replay-valid path gains beyond strong Single/Pair controls.',
            'backend': 'The executed backend searches exact QUBO factors with classical structure-aware bounds. '
                       'It does not call a CPMP/MILP oracle; factor storage is energy-equivalent to the explicit matrix. '
                       'Q-versus-equivalent-MILP superiority remains unproved.',
            'repair_and_supply': 'Q choices are physically replayed. Common cleanup/FRG and the unchanged Beam '
                                 'prefix are separately attributed; no new Beam-core selector or macro-supply claim.',
            'guard': 'All16 paired winning executions against SoloQ end on the paid Pair branch. '
                     'The paired strategy nevertheless costs117 original-mean total moves in this panel. '
                     'The overall paired-versus-soloQ interval crosses zero; protection is not uniformly better.',
            'complete_algorithm': 'The paired realization has fresh native/BS comparisons and fresh strong '
                                  'solo-control comparisons in separate panels. Do not splice their scores.',
        },
        limitations={
            'all_instances_solved': False, 'no_tail_losses': False,
            'global_cpmp_optimality': False, 'exclusive_qubo_encoding_advantage': False,
            'generic_matrix_only_backend_advantage': False, 'quantum_hardware_advantage': False,
            'worldwide_priority_proved': False, 'untouched_external_test_set': False,
            'solo_q_direct_native_bs_comparison_in_this_final_panel': False,
            'production_default_promotion': False,
        },
        future_optional_work='Improve initial feasibility and time allocation or test matrix-only/hardware solvers '
            'as separately scoped future work. These are not established by this empirical result.')
    OUT.mkdir(exist_ok=True)
    (OUT / 'completion_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(requirements=checks,
        empirical_objective_evidence_complete=result['empirical_objective_evidence_complete'],
        stable_in_all_four=len(stable_a & stable_b), witnesses=witness_rows)))
    if not all(checks.values()):
        raise ValueError('At least one empirical objective requirement is unproved')


if __name__ == '__main__':
    main()
