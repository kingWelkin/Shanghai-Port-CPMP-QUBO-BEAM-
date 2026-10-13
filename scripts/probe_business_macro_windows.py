"""Follow the historical Real30 exposure surface without changing beam/generators."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'scripts',ROOT/'src'):
    sys.path.insert(0,str(path))
import benchmark_compact_beam_qubo_n0 as beam
from container_reshuffle.core import Instance,ProblemVariant
from probe_production_macro_windows import MacroWindowObserver,OUT
from beam_selector_research import encode
from run_beam_selector_research import ENVIRONMENT,PROFILE,digest,write_csv,write_json
from run_exact_astar_real_until_done import _instance_from_single_bay


def business_args():
    args=beam.make_compact_args(solver_profile=PROFILE,time_limit=20.0,seed=37,alpha=1.3,business_cap_enabled=True)
    beam._apply_solver_profile(args);beam._apply_qubo_influence_profile(args);beam._apply_dynamic_profile(args)
    return args


def freeze():
    path=OUT/'business_manifest.json'
    if path.exists(): raise FileExistsError('Business panel already frozen')
    historical=ROOT/'docs/experiments/v43_full_gate_20260709/v43_real30_mainline_branch_t120.csv'
    with historical.open(encoding='utf-8-sig',newline='') as stream:
        records=list(csv.DictReader(stream))
    cases=[]
    for row in records:
        calls=float(row.get('union_diversity_qubo_calls') or 0)
        if not calls or float(row['union_diversity_qubo_vars'])/calls<=4: continue
        zone,bay,_=row['case'].split('_')
        instance=_instance_from_single_bay(zone,bay)
        data=asdict(instance);data['variant']=instance.variant.value
        cases.append(dict(case=row['case'],instance=data,historical_calls=int(calls),
                          historical_vars=int(row['union_diversity_qubo_vars'])))
    assert len(cases)==6
    args=business_args()
    sources=['scripts/probe_business_macro_windows.py','scripts/probe_production_macro_windows.py',
             'scripts/benchmark_compact_beam_qubo_n0.py','scripts/beam_selector_research.py']
    write_json(path,dict(cases=cases,selection='ALL Real30 rows with historical union_vars/union_calls > 4; no current outcome selection',
        historical_source=historical.relative_to(ROOT).as_posix(),historical_source_sha256=digest(historical),
        source_sha256={p:digest(ROOT/p) for p in sources},cap_sec=20.0,seed=37,alpha=1.3,business_cap_enabled=True,
        config={k:v for k,v in vars(args).items() if not k.startswith('_') and isinstance(v,(int,float,str,bool,type(None)))},
        scope='Separate business-constrained exposure panel, not pooled with the standard CPMP panel',environment=ENVIRONMENT))


def manifest():
    m=json.loads((OUT/'business_manifest.json').read_text(encoding='utf-8'))
    for p,sha in m['source_sha256'].items():
        if digest(ROOT/p)!=sha: raise ValueError(f'Source changed: {p}')
    return m


def load_instance(data):
    data=dict(data);data['variant']=ProblemVariant(data['variant'])
    return Instance(**data)


def worker(case_id):
    m=manifest();c=next(c for c in m['cases'] if c['case']==case_id)
    instance=load_instance(c['instance']);args=business_args()
    observer=MacroWindowObserver();args._union_selector_observer=observer
    started=time.perf_counter()
    result=beam.run_compact_beam_instance(case_id,instance,args)
    wall=time.perf_counter()-started
    business_ok,goal_ok,msg=beam._validate_instance_result(result,instance,business_alpha=1.3,business_cap_enabled=True)
    row=dict(case=case_id,surface='business',business_ok=business_ok,goal_ok=goal_ok,
        valid=business_ok and goal_ok,moves=len(result.moves) if goal_ok else None,
        solver_wall_sec=wall,within_budget=wall<=20.0,union_calls=len(observer.records),
        all_qubo_calls=result.qubo_calls,nontrivial_calls=sum(r['n']>r['k'] for r in observer.records),
        macro_calls=sum(r['macro_count']>0 for r in observer.records),eligible_calls=sum(r['eligible'] for r in observer.records),
        saved_windows=len(observer.windows),expanded=result.expanded,generated=result.generated,
        status=result.status,validation_message=msg)
    write_json(OUT/'business_runs'/f'{case_id}.json',dict(row=row,path=[asdict(move) for move in result.moves],calls=observer.records))
    for i,window in enumerate(observer.windows):
        window.update(case=case_id,surface='business')
        target=OUT/'business_windows'/f'{case_id}__{i}.json.gz'
        target.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(target,'wt',encoding='utf-8') as stream: json.dump(encode(window),stream,separators=(',',':'))
    print(json.dumps({k:v for k,v in row.items() if k not in ('status','validation_message')},ensure_ascii=False),flush=True)


def panel():
    m=manifest()
    for case in m['cases']:
        if (OUT/'business_runs'/f"{case['case']}.json").exists(): continue
        proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--case',case['case']],
            cwd=ROOT,env={**os.environ,**ENVIRONMENT},capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=50)
        if proc.returncode:
            write_json(OUT/'errors'/f"{case['case']}.json",dict(stdout=proc.stdout,stderr=proc.stderr))
            raise RuntimeError(proc.stderr)
        print(proc.stdout.strip(),flush=True)
    rows=[json.loads(p.read_text(encoding='utf-8'))['row'] for p in sorted((OUT/'business_runs').glob('*.json'))]
    write_csv(OUT/'business_activation_results.csv',rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run','worker']);p.add_argument('--case')
    args=p.parse_args()
    if args.command=='freeze': freeze()
    elif args.command=='run': panel()
    else: worker(args.case)

if __name__=='__main__': main()
