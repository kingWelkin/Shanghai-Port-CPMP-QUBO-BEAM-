"""Frozen full-corpus CPMP selector study; opt-in, without default promotion."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'scripts', ROOT / 'src'):
    sys.path.insert(0, str(p))

import numpy as np
import benchmark_compact_beam_qubo_n0 as beam
import open_cpmp_benchmarks as opened
from benchmark_bs_frg_paper import discover_paper_cases, _load_case, PaperCase
from beam_selector_research import SelectorObserver, encode
from container_reshuffle.core import future_disorder_count_stacks
from container_reshuffle.solvers.base import SolverConfig
from container_reshuffle.solvers.bs_frg import BsFrgSolver
from frg_rollout_coordinates import CanonicalTailContract
from production_route_provenance import solver_frame
from production_selector_factorial import set_factors, restore_factors
from production_selector_replay import fingerprint
from run_beam_selector_research import ENVIRONMENT, digest, write_json, write_csv
from run_rollout_contract import cache_audit, TELEMETRY
from benchmark_compact_paper_set import build_parser as paper_parser, _args_for_variant
from paper_priority_group_contract import PriorityGroupContract

OUT = ROOT / 'docs/experiments/beam_selector_paper_v43_20261007'
PROFILE = 'qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_wang_route_v43_polish'
ARMS = {
    'linear': ('none', 'linear'),
    'pair_greedy': ('pair_only', 'marginal'),
    'quota_greedy': ('quota_only', 'marginal'),
    'full_greedy': ('full', 'marginal'),
    'full_sa': ('full', 'sa'),
    'bs_frg': (None, None),
}
SOURCES = [
    'scripts/research_paper_v43_panel.py', 'scripts/paper_priority_group_contract.py',
    'scripts/benchmark_compact_paper_set.py',
    'scripts/benchmark_compact_beam_qubo_n0.py', 'scripts/beam_selector_research.py',
    'scripts/frg_rollout_coordinates.py', 'scripts/production_selector_factorial.py',
    'scripts/production_selector_replay.py', 'scripts/production_route_provenance.py',
    'scripts/run_beam_selector_research.py', 'scripts/run_rollout_contract.py',
    'scripts/benchmark_bs_frg_paper.py', 'scripts/open_cpmp_benchmarks.py',
    'container_reshuffle/core.py', 'container_reshuffle/solvers/bs_frg.py',
    'container_reshuffle/solvers/base.py', 'container_reshuffle/qubo/model.py',
    'container_reshuffle/qubo/sampler.py',
]


def make_args(cap, seed, instance=None, entry=None):
    cli = paper_parser().parse_args(['--variant',PROFILE,'--time-limit',str(cap),'--alpha','10','--no-business-cap'])
    # The source is an origin hint used by the established v4.3 route. Actual
    # input reads always use the immutable tracked path in the corpus inventory.
    cli.source = ROOT / ('external/DLTS' if entry and entry['dataset']=='Open120' else 'benchmarks/paper')
    cli.seed = seed
    args = _args_for_variant(cli,PROFILE,instance)
    args.seed = seed
    beam._apply_solver_profile(args)
    beam._apply_qubo_influence_profile(args)
    beam._apply_dynamic_profile(args)
    args.inner_jobs = args.post_polish_inner_jobs = 1
    return args


def load_case(entry):
    path = ROOT / entry['path']
    if digest(path) != entry['sha256']:
        raise ValueError('Frozen input changed: ' + entry['case'])
    if entry['dataset'] == 'Open120':
        return opened.load_dlts_dat(path, max_height=entry['max_height'])
    return _load_case(PaperCase(entry['dataset'], entry['family'], entry['source_case'], path))


def inventory():
    entries = []
    for dataset in ('CV', 'BF'):
        for c in discover_paper_cases(ROOT / 'benchmarks/paper', dataset=dataset):
            inst = _load_case(c)
            entries.append(dict(case=f'{dataset}__{c.family}__{c.case_id}', dataset=dataset,
                family=c.family, source_case=c.case_id, path=c.instance_path.relative_to(ROOT).as_posix(),
                sha256=digest(c.instance_path), max_height=inst.max_height, stacks=inst.stack_count,
                containers=inst.container_count, initial_n=future_disorder_count_stacks(inst.stacks),
                priority_groups=opened.requires_priority_group_semantics(inst),
                canonical_instance_signature=fingerprint((inst.max_height, sorted(map(tuple, inst.stacks))))))
    for c in opened.discover_dlts_cases(ROOT / 'benchmarks/open120', limit_per_family=9999):
        inst = opened.load_dlts_dat(c.instance_path, max_height=c.reference.max_height)
        entries.append(dict(case=f'Open120__{c.family}__{c.case_id}', dataset='Open120', family=c.family,
            source_case=c.case_id, path=c.instance_path.relative_to(ROOT).as_posix(), sha256=digest(c.instance_path),
            max_height=inst.max_height, stacks=inst.stack_count, containers=inst.container_count,
            initial_n=future_disorder_count_stacks(inst.stacks), reference_moves=c.reference.optimal_moves,
            priority_groups=opened.requires_priority_group_semantics(inst),
            canonical_instance_signature=fingerprint((inst.max_height, sorted(map(tuple, inst.stacks))))))
    return assign_split(entries)


def assign_split(entries):
    groups = collections.defaultdict(list)
    for e in entries:
        groups[e['dataset'], e['family']].append(e)
    for group in groups.values():
        dev = max(1, len(group) // 5)
        for i, e in enumerate(group):
            e.update(split='development' if i < dev else 'evaluation',
                diagnostic=i in {0, len(group)-1}, family_index=i)
    dev_signatures = {e['canonical_instance_signature'] for e in entries if e['split'] == 'development'}
    for e in entries:
        if e['canonical_instance_signature'] in dev_signatures:
            e['split'] = 'development'
    return entries


def pin_cpu(index):
    if os.name != 'nt':
        raise RuntimeError('This frozen Windows experiment requires process affinity')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 1 << index):
        raise ctypes.WinError(ctypes.get_last_error())


class PaperObserver(SelectorObserver):
    """Ablate only the selection and its repair, then verify actual filled Beam."""
    def __init__(self, branch):
        factor, sampler = ARMS[branch]
        super().__init__(sampler, capture_limit=0)
        self.branch = branch
        self.factor = factor
        self.saved = None
        self.active_args = None
        self.originals = {}
        self.pending_frontier = None
        self.commits = []
        self.snapshots = []
        self.current_layer = None
        self.pending_snapshot = None

    def install(self):
        for name, callback in {
            '_record_quality_reservoir_selected_telemetry': self.commit,
            '_select_next_frontier': self.check_next,
            'CompactBeamResult': self.result,
        }.items():
            self.originals[name] = getattr(beam, name)
            setattr(beam, name, callback)

    def context(self, ranked, core, best_n):
        self.current_layer = solver_frame().f_locals['layer_index']
        super().context(ranked, core, best_n)

    def sample(self, model, config, usable, target, linear, features, args, feature_sec, build_sec):
        self.active_args = args
        self.saved = set_factors(args, self.factor)
        self.usable = list(usable)
        self.linear = linear
        self.features = features
        result = super().sample(model, config, usable, target, linear, features, args, feature_sec, build_sec)
        self.current['layer'] = self.current_layer
        self.current['macro_pool'] = sum(beam._union_diversity_is_macro_feature(f) for f in features)
        self.pending_snapshot = None
        if len(usable) > target and len(self.snapshots) < 2:
            # Inputs were generated under unchanged native settings. Stored after solve.
            self.pending_snapshot = dict(schema='paper-selector-static/v1', usable=usable, features=features,
                linear=linear, model=model.to_dict(), sampler_config=asdict(config), target=target,
                settings={k:v for k,v in vars(args).items() if isinstance(v,(str,int,float,bool,type(None)))},
                context=self.layer_context)
        return result

    def finish(self, ordered, selector_sec):
        try:
            self.current['objective'] = beam._union_diversity_objective(set(ordered), self.linear, self.features, self.active_args)
            self.current['selected_codes'] = [self.usable[i][0].hex() for i in ordered]
            super().finish(ordered, selector_sec)
            if self.pending_snapshot is not None:
                self.pending_snapshot['stages'] = dict(self.current)
                self.snapshots.append(self.pending_snapshot)
        finally:
            if self.saved is not None:
                restore_factors(self.active_args, self.saved)
                self.saved = None

    def verify(self):
        if self.pending_frontier is None:
            return
        actual = solver_frame().f_locals['frontier']
        if actual != self.pending_frontier:
            raise RuntimeError('Actual full Beam differs from committed ordered frontier')
        self.commits[-1]['actual_frontier_verified'] = True
        self.pending_frontier = None

    def commit(self, selected_codes, args):
        if self.pending_frontier is not None:
            raise RuntimeError('A filled-Beam commitment was not verified')
        loc = solver_frame().f_locals
        candidates = loc['candidates']
        self.pending_frontier = [(c,candidates[c][0],candidates[c][1]) for c in selected_codes]
        indices = self.current.get('selected', []) if self.records and self.current_layer == loc['layer_index'] else []
        chosen = [self.usable[i][0] for i in indices]
        self.commits.append(dict(layer=loc['layer_index'], width=len(selected_codes),
            ordered_frontier_signature=fingerprint(self.pending_frontier), membership_signature=fingerprint(sorted(selected_codes)),
            supplement_survived=sum(c in selected_codes for c in chosen), selected_supplement=len(chosen),
            actual_frontier_verified=False))
        return self.originals['_record_quality_reservoir_selected_telemetry'](selected_codes, args)

    def check_next(self, candidates, args):
        self.verify()
        return self.originals['_select_next_frontier'](candidates, args)

    def result(self, *args, **kwargs):
        self.verify()
        return self.originals['CompactBeamResult'](*args, **kwargs)

    def restore(self):
        if self.saved is not None:
            restore_factors(self.active_args, self.saved)
            self.saved = None
        for name, original in self.originals.items():
            setattr(beam, name, original)


def freeze():
    if (OUT / 'manifest.json').exists():
        raise FileExistsError('Paper corpus protocol already frozen')
    entries = inventory()
    counts = dict(collections.Counter(e['dataset'] for e in entries))
    if counts != {'CV':840,'BF':721,'Open120':120}:
        raise ValueError('Corpus differs from tracked benchmark definition: ' + str(counts))
    args = make_args(20., 37)
    args.inner_jobs = 1
    args.post_polish_inner_jobs = 1
    m = dict(schema='paper-selector-corpus/v1', date='2026-10-07', cases=entries, dataset_counts=counts,
        families=len({(e['dataset'],e['family']) for e in entries}), arms=ARMS, request_sec=20.,
        seeds=[37], repeats=1, jobs=8, affinity_cpus=list(range(0,16,2)), watchdog_sec=100,
        profile=PROFILE, alpha=10., business_cap_enabled=False,
        settings={k:v for k,v in vars(args).items() if not k.startswith('_') and isinstance(v,(str,int,float,bool,type(None)))},
        source_sha256={p:digest(ROOT/p) for p in SOURCES}, environment=ENVIRONMENT,
        python=sys.version, platform=platform.platform(), logical_cpus=os.cpu_count(),
        dependencies={p:importlib.metadata.version(p) for p in ('numpy','dimod','dwave-neal','pytest')},
        parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        diagnostic='First and last natural-order case from EVERY dataset/family, singleton once; no outcome screening.',
        split='First floor(n/5), minimum1, per original family development. Exact canonical duplicates share development split. Remainder evaluation. Corpora previously exposed; evaluation is not claimed never-seen.',
        full_corpus='ALL1681 original tracked instances; remaining stage excludes diagnostic cases already run. All six arms, no skipping inactive or failed cases.',
        intervention='Official v4.3 per-instance route factory; only union selection and synchronized repair/sort factors ablated. Original generators, core, linear features, full model construction and later searches retained. All native union decisions treated. Canonical cached-tail and priority-group semantics shared by all five hybrid arms.',
        route_origin='Open120 retains original DLTS origin hint for the established v4.3 source-dependent router; BF/CV use paper origin. Actual input paths remain benchmarks/open120 and benchmarks/paper. Route chosen from instance/source only, never solver outcomes.',
        preceding_pilot='Earlier v4.2 general-route diagnostic retained in beam_selector_paper_corpus_20261007. Its strict unique-ID cached-tail audit rejects repeated labels; it is not formal current-method evidence and its remaining stage is not used. This new frozen contract corrects corpus semantics and the current-method entry point, without tuning objective weights.',
        reference='BS-FRG paper_pure=True, width1000, k3, depth1000, entropy_alpha10, same20s request. Cross-system reference cannot attribute selector contribution.',
        timing='Paired complete case blocks on same pinned logical CPU, serial arms rotated by case index,8 blocks concurrently. Solver plus input/configuration and final path/cache validation charged; Python dependency import and evidence serialization excluded from algorithm budget, whole subprocess wall separately reported. No inference of exclusive-machine speed superiority.',
        success='All elementary moves physically valid and terminal N=0; primary success additionally measured load/config+solver+validator/cache wall <=20s and cache contract clean. No tolerance, failures/timeouts/late returns retained.',
        mechanism='After native full_sa diagnostics, predeclare same-prefix captures and fixed-count macro-pair trials on development members; features/weights cannot use evaluation labels. Conditional fitting requires identifiable within-context pair outcomes.',
        promotion='NO_PROMOTION; one seed broad coverage, not a robustness or production gate')
    write_json(OUT/'manifest.json',m)
    write_csv(OUT/'corpus_inventory.csv',entries)
    print(json.dumps(dict(cases=len(entries),diagnostic=sum(e['diagnostic'] for e in entries),families=m['families'],arms=len(ARMS),total_runs=len(entries)*len(ARMS))),flush=True)


def manifest():
    m = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    for p, expected in m['source_sha256'].items():
        if digest(ROOT/p) != expected:
            raise ValueError('Frozen source changed: '+p)
    return m


def worker(case, arm, cpu):
    m = manifest()
    e = next(e for e in m['cases'] if e['case']==case)
    pin_cpu(cpu)
    random.seed(37)
    np.random.seed(37)
    started = time.perf_counter()
    inst = load_case(e)
    observer = None
    if arm == 'bs_frg':
        config = SolverConfig(max_depth=1000,beam_width=1000,candidate_limit_per_state=3,time_limit_sec=20.,entropy_alpha=10.,random_seed=37)
        args = None
        configured = time.perf_counter()
        result = BsFrgSolver(paper_pure=True).solve(inst,config)
    else:
        args = make_args(20.,37,inst,e)
        args.inner_jobs = args.post_polish_inner_jobs = 1
        observer = PaperObserver(arm)
        args._union_selector_observer = observer
        contract = CanonicalTailContract()
        group_contract = PriorityGroupContract()
        configured = time.perf_counter()
        try:
            group_contract.install()
            contract.install()
            observer.install()
            result = beam.run_compact_beam_instance(case,inst,args)
        finally:
            observer.restore()
            contract.restore()
            group_contract.restore()
    solved = time.perf_counter()
    business_ok, goal_ok, msg = opened._open_validate_instance_result(result,inst,10.,business_cap_enabled=False)
    if args is not None:
        group_contract.install()
        try:
            audit = cache_audit(args)
        finally:
            group_contract.restore()
    else:
        audit = {'contract_invalid':0}
    ended = time.perf_counter()
    path = [asdict(v) for v in result.moves]
    records = observer.records if observer else []
    commits = observer.commits if observer else []
    if observer and (observer.pending_frontier is not None or any(not c['actual_frontier_verified'] for c in commits)):
        raise RuntimeError('Filled Beam verification incomplete')
    row = dict(case=case,dataset=e['dataset'],family=e['family'],split=e['split'],diagnostic=e['diagnostic'],
        arm=arm,seed=37,cpu=cpu,actual_profile=getattr(args,'solver_profile','paper_bs_frg'),priority_groups=e['priority_groups'],
        valid=business_ok and goal_ok,moves=len(path) if goal_ok else None,
        primary_ok=business_ok and goal_ok and ended-started<=20. and audit['contract_invalid']==0,
        status=result.status,validation_message=msg,load_config_sec=configured-started,solver_wall_sec=solved-configured,
        validation_cache_sec=ended-solved,measured_end_to_end_sec=ended-started,within_budget=ended-started<=20.,
        cache_contract_invalid=audit['contract_invalid'],union_calls=len(records),
        nontrivial_union_calls=sum(r['n']>r['k'] for r in records),actual_beam_commits=len(commits),
        actual_beam_verified=len(commits),full_path_signature=fingerprint(path),
        full_search_signature=fingerprint(dict(path=path,commits=[{k:v for k,v in c.items() if k!='actual_frontier_verified'} for c in commits],
            selections=[dict(n=r['n'],k=r['k'],selected=r['selected']) for r in records])),
        **{k:getattr(result,k) for k in TELEMETRY if hasattr(result,k)})
    for key in ('feature_sec','build_sec','initialization_sec','cardinality_sec','swap_stable_sec','selector_sec'):
        row[key] = sum(r[key] for r in records)
    row['raw_count_violations'] = sum(r['raw_count_error']>0 for r in records)
    write_json(OUT/'runs'/f'{case}__{arm}.json',dict(row=row,path=path,selector_records=records,commits=commits,
        contract_stats=contract.stats if observer else None,cache_audit=audit))
    if observer:
        for i, snapshot in enumerate(observer.snapshots):
            write_json(OUT/'snapshots'/f'{case}__{arm}__{i}.json',encode(snapshot))
    print(json.dumps({k:row[k] for k in ('case','arm','valid','primary_ok','moves','nontrivial_union_calls','measured_end_to_end_sec')},ensure_ascii=False),flush=True)


def run_block(index, e, m):
    cpu = m['affinity_cpus'][index % m['jobs']]
    arms = list(ARMS)
    rotation = index % len(arms)
    arms = arms[rotation:] + arms[:rotation]
    block_path = OUT/'blocks'/f"{e['case']}.json"
    if block_path.exists():
        return {'case':e['case'],'status':'EXISTING_COMPLETE_BLOCK'}
    rows = []
    for arm in arms:
        path = OUT/'runs'/f"{e['case']}__{arm}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding='utf-8'))['row'])
            continue
        started = time.perf_counter()
        cmd = [sys.executable,str(Path(__file__).resolve()),'worker','--case',e['case'],'--arm',arm,'--cpu',str(cpu)]
        try:
            proc = subprocess.run(cmd,cwd=ROOT,env={**os.environ,**ENVIRONMENT},capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=m['watchdog_sec'])
            process_wall = time.perf_counter()-started
            if proc.returncode:
                raise RuntimeError(proc.stderr or proc.stdout)
            data = json.loads(path.read_text(encoding='utf-8'))
            data['row']['process_wall_sec'] = process_wall
            write_json(path,data)
        except subprocess.TimeoutExpired as ex:
            data = dict(row=dict(case=e['case'],dataset=e['dataset'],family=e['family'],split=e['split'],
                diagnostic=e['diagnostic'],arm=arm,seed=37,cpu=cpu,valid=False,primary_ok=False,moves=None,
                status='EXTERNAL_WATCHDOG',process_wall_sec=time.perf_counter()-started,within_budget=False),path=[])
            write_json(path,data)
            write_json(OUT/'errors'/f"{e['case']}__{arm}.json",dict(type='watchdog',stdout=str(ex.stdout),stderr=str(ex.stderr)))
        except Exception as ex:
            write_json(OUT/'errors'/f"{e['case']}__{arm}.json",dict(type=type(ex).__name__,error=str(ex)))
            raise
        rows.append(data['row'])
    write_json(block_path,dict(case=e['case'],cpu=cpu,arms=arms,complete=True,run_sha256={r['arm']:digest(OUT/'runs'/f"{e['case']}__{r['arm']}.json") for r in rows}))
    return dict(case=e['case'],status='COMPLETE_BLOCK',valid=sum(r['valid'] for r in rows),primary_ok=sum(r['primary_ok'] for r in rows),
        nontrivial_calls=sum(r.get('nontrivial_union_calls',0) for r in rows),moves={r['arm']:r['moves'] for r in rows})


def run(stage, slot=None):
    m = manifest()
    entries = [(i,e) for i,e in enumerate(m['cases']) if (stage=='all' or e['diagnostic']==(stage=='diagnostic'))]
    # One serial case stream per affinity slot: never schedule two blocks on one CPU.
    def stream(cpu_slot):
        for i,e in entries:
            if i % m['jobs'] != cpu_slot:
                continue
            print(json.dumps(run_block(i,e,m),ensure_ascii=False),flush=True)
    if slot is not None:
        stream(slot)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=m['jobs']) as executor:
            tasks = [executor.submit(stream,s) for s in range(m['jobs'])]
            for t in concurrent.futures.as_completed(tasks):
                t.result()
    summarize()


def compare(left, right, success='valid'):
    if left.get(success) and right.get(success):
        return 'win' if left['moves']<right['moves'] else 'loss' if left['moves']>right['moves'] else 'tie_solved'
    if left.get(success):
        return 'win'
    if right.get(success):
        return 'loss'
    return 'both_failed'


def summarize():
    m = manifest()
    rows = []
    for b in sorted((OUT/'blocks').glob('*.json')):
        block = json.loads(b.read_text(encoding='utf-8'))
        for arm in ARMS:
            p = OUT/'runs'/f"{block['case']}__{arm}.json"
            if digest(p) != block['run_sha256'][arm]:
                raise ValueError('Completed evidence block changed: '+str(p))
            rows.append(json.loads(p.read_text(encoding='utf-8'))['row'])
    write_csv(OUT/'results.csv',rows)
    by_case = collections.defaultdict(dict)
    for r in rows:
        by_case[r['case']][r['arm']] = r
    groups = []
    comparisons = []
    for dataset in ('ALL','CV','BF','Open120'):
        for arm in ARMS:
            rr = [r for r in rows if r['arm']==arm and (dataset=='ALL' or r['dataset']==dataset)]
            if not rr:
                continue
            valid = [r for r in rr if r['valid']]
            groups.append(dict(dataset=dataset,arm=arm,runs=len(rr),valid=len(valid),primary_ok=sum(r['primary_ok'] for r in rr),
                moves_sum_valid=sum(r['moves'] for r in valid),nontrivial_cases=sum(r.get('nontrivial_union_calls',0)>0 for r in rr),
                cache_contract_invalid=sum(r.get('cache_contract_invalid',0) for r in rr),
                median_measured_sec=statistics.median(r.get('measured_end_to_end_sec',r.get('process_wall_sec',0.)) for r in rr)))
        for left,right in [('full_sa','linear'),('pair_greedy','linear'),('quota_greedy','linear'),('full_greedy','linear'),('full_sa','full_greedy'),('full_sa','bs_frg')]:
            pairs = [(arms[left],arms[right]) for arms in by_case.values() if dataset=='ALL' or arms[left]['dataset']==dataset]
            counts = collections.Counter(compare(a,b) for a,b in pairs)
            primary = collections.Counter(compare(a,b,'primary_ok') for a,b in pairs)
            comparisons.append(dict(dataset=dataset,left=left,right=right,n=len(pairs),valid_outcomes=dict(counts),primary_outcomes=dict(primary),
                identical_physical_paths=sum('full_path_signature' in a and 'full_path_signature' in b and a['full_path_signature']==b['full_path_signature'] for a,b in pairs),
                identical_full_search=sum('full_search_signature' in a and 'full_search_signature' in b and a['full_search_signature']==b['full_search_signature'] for a,b in pairs)))
    summary = dict(total_corpus=len(m['cases']),expected_runs=len(m['cases'])*len(ARMS),complete_blocks=len(by_case),executed_runs=len(rows),
        executed_families=len({(r['dataset'],r['family']) for r in rows}),groups=groups,comparisons=comparisons,
        scope='Only complete predeclared atomic case blocks; final corpus conclusions require all1681 blocks. No default promotion.')
    write_json(OUT/'summary.json',summary)
    write_csv(OUT/'groups.csv',groups)
    write_json(OUT/'comparisons.json',comparisons)
    print(json.dumps(dict(complete_blocks=len(by_case),runs=len(rows),families=summary['executed_families'])),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command',choices=['freeze','run','worker','summarize'])
    p.add_argument('--case')
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--cpu',type=int,default=0)
    p.add_argument('--stage',choices=['diagnostic','remaining','all'],default='diagnostic')
    p.add_argument('--slot',type=int)
    a = p.parse_args()
    if a.command=='freeze':
        freeze()
    elif a.command=='run':
        run(a.stage,a.slot)
    elif a.command=='worker':
        worker(a.case,a.arm,a.cpu)
    else:
        summarize()


if __name__=='__main__':
    main()
