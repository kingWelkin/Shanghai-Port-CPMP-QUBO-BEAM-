"""Scoped component removals; no mutation of native sources or defaults."""
from __future__ import annotations
import collections
import inspect
import time

import benchmark_compact_beam_qubo_n0 as beam
from beam_selector_research import SelectorObserver
from research_paper_v43_panel import PaperObserver

GENERAL='scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_short_incumbent_v42_polish'
WANG_WEIGHTS={
    'wang_probe_supplement_width':0,'wang_feature_fixed_prefix_weight':0.,'wang_feature_fix_pair_weight':0.,
    'candidate_ledger_wang_fixed_weight':0.,'candidate_ledger_wang_pair_weight':0.,
    'union_diversity_qubo_wang_fixed_weight':0.,'union_diversity_qubo_wang_pair_weight':0.,
    'unified_wang_fixed_prefix_weight':0.,'unified_wang_fix_pair_weight':0.,
}
PAIR_KEYS=('same_parent','same_source','signature','reduce_signature','n_band','origin_similarity',
           'same_origin_parent_n_band','frg_witness_signature','basin_similarity')
ARMS={
    'native':dict(kind='baseline',updates={}),
    'no_wang_pair':dict(kind='component',updates={'wang_pair_qubo_enabled':False}),
    'no_wang_quality':dict(kind='signal_group',updates=WANG_WEIGHTS),
    'no_all_wang':dict(kind='dependency_group',updates={**WANG_WEIGHTS,'wang_pair_qubo_enabled':False,
        'wang_incumbent_enabled':False,'wang_rollout_enabled':False,'wang_cap_enabled':False,'unified_wang_qubo_enabled':False,'wang2017_task_enabled':False}),
    'no_frg_frontier':dict(kind='component',updates={'frg_frontier_child_enabled':False}),
    'no_macro_chain':dict(kind='component',updates={'macro_chain_witness_enabled':False}),
    'no_unified_supply':dict(kind='component',updates={'unified_macro_supply_enabled':False}),
    'no_chain_and_unified':dict(kind='dependency_group',updates={'macro_chain_witness_enabled':False,'unified_macro_supply_enabled':False}),
    'no_macro_future_credit':dict(kind='signal_group',updates={'union_diversity_qubo_macro_future_credit_enabled':False}),
    'no_macro_edge_cost':dict(kind='component',updates={'union_diversity_qubo_macro_edge_chain_cost_weight':0.}),
    'no_ancestry_cost':dict(kind='component',updates={'union_diversity_qubo_macro_ancestry_penalty_weight':0.}),
    'no_union_probe':dict(kind='component',updates={'union_rollout_probe_enabled':False}),
    'no_union_selector':dict(kind='component',updates={'union_diversity_qubo_enabled':False}),
    'no_state_scheduler':dict(kind='policy',updates={'selection_policy':'enhanced','scheduler_state_controller':'off'}),
    'no_core_reservation':dict(kind='policy',updates={},hooks=['core']),
    'no_cached_tail_graft':dict(kind='component',updates={},hooks=['graft']),
    'no_all_recovery':dict(kind='combined_reference',updates={'frg_frontier_child_enabled':False,
        'macro_chain_witness_enabled':False,'unified_macro_supply_enabled':False,'union_rollout_probe_enabled':False,
        'frg_seed_enabled':False,'frg_seed_beam_enabled':False,'frg_candidate_supplement_enabled':False,
        'frg_compound_escape_enabled':False,'bs_frg_qubo_enabled':False,'bs_frg_postcompress_enabled':False},hooks=['graft']),
    'no_postpolish':dict(kind='component',updates={'post_polish_qubo':False}),
    'no_bsfrg_postcompress':dict(kind='component',updates={'bs_frg_postcompress_enabled':False}),
    'linear_post_suffix':dict(kind='model',updates={},hooks=['suffix']),
    'linear_post_windows':dict(kind='model',updates={},hooks=['windows']),
    'no_shortcut_replay':dict(kind='component',updates={'post_polish_shortcut_replay_enabled':False}),
    'no_bridge_scan':dict(kind='component',updates={'post_polish_bridge_scan_enabled':False}),
    'no_successor_cache':dict(kind='component',updates={'post_polish_successor_cache':False}),
    'fixed_general_route':dict(kind='policy',updates={},route=GENERAL),
    'no_all_qubo':dict(kind='combined_reference',updates={'use_qubo':False,'wang_pair_qubo_enabled':False,
        'unified_wang_qubo_enabled':False,'union_diversity_qubo_enabled':False,'bs_frg_qubo_enabled':False,
        'pareto_union_enabled':False,'scarcity_group_qubo_enabled':False},hooks=['suffix','windows'],forbid_sampling=True),
    'classic_beam':dict(kind='combined_reference',updates={
        'selection_policy':'classic','scheduler_state_controller':'off','use_qubo':False,
        'wang_incumbent_enabled':False,'wang_rollout_enabled':False,'wang_cap_enabled':False,
        'wang_pair_qubo_enabled':False,'unified_wang_qubo_enabled':False,'union_diversity_qubo_enabled':False,
        'union_rollout_probe_enabled':False,'frg_frontier_child_enabled':False,'macro_chain_witness_enabled':False,
        'unified_macro_supply_enabled':False,'frg_seed_enabled':False,'frg_seed_beam_enabled':False,
        'frg_candidate_supplement_enabled':False,'frg_linear_score_enabled':False,'frg_rank_tiebreak_enabled':False,
        'frg_compound_escape_enabled':False,'bs_frg_qubo_enabled':False,'pareto_union_enabled':False,
        'post_polish_qubo':False,'bs_frg_postcompress_enabled':False,**WANG_WEIGHTS},forbid_sampling=True),
}
INACTIVE_COMPONENTS={
    'wang_incumbent':['wang_incumbent_enabled'], 'wang_rollout':['wang_rollout_enabled'],
    'frg_seed':['frg_seed_enabled','frg_seed_beam_enabled'], 'unified_wang':['unified_wang_qubo_enabled'],
    'pareto_union':['pareto_union_enabled'], 'bs_frg_selector':['bs_frg_qubo_enabled'],
    'quality_reservoir':['quality_reservoir_enabled'], 'debt_basin':['debt_basin_enabled'],
    'beam_feedback':['beam_feedback_enabled'], 'micro_rollout':['micro_rollout_enabled'],
    'receiver_safety':['receiver_safety_enabled'], 'feasibility_escape':['feasibility_escape_enabled'],
}


def scalar_settings(args):
    return {k:v for k,v in vars(args).items() if not k.startswith('_') and isinstance(v,(str,int,float,bool,type(None)))}


def apply_updates(args,arm):
    for key,value in ARMS[arm]['updates'].items():
        setattr(args,key,value)
    args.inner_jobs=args.post_polish_inner_jobs=1


def linear_suffixes(items,args):
    if not items:return [],0,0.
    target=max(1,min(args.post_polish_suffix_select,len(items)))
    usable=items[:max(1,min(args.post_polish_suffix_max_qubo_vars,len(items)))]
    if len(usable)<=target:return usable,0,0.
    chosen=[]
    for item in sorted(usable,key=lambda s:(-s.score,s.start)):
        if all(abs(item.start-old.start)>args.post_polish_suffix_near_gap for old in chosen):chosen.append(item)
        if len(chosen)>=target:break
    return sorted(chosen,key=lambda s:s.start),0,0.


def linear_windows(items,args,version):
    requested=args.post_polish_window_select if version==2 else args.post_polish_select_windows
    if not items or requested<=0:return [],0,0.
    if version==2:
        ratio=float(args.post_polish_window_min_score_ratio or 0.)
        best=max(w.score for w in items)
        if ratio>0 and best>0:items=[w for w in items if w.score>=best*ratio]
        if not items:return [],0,0.
    target=max(1,min(requested,len(items)))
    limit=args.post_polish_window_qubo_vars if version==2 else args.post_polish_max_qubo_vars
    usable=items[:max(1,min(limit,len(items)))]
    if len(usable)<=target:return usable,0,0.
    chosen=[]
    for item in sorted(usable,key=lambda w:(-w.score,w.start,w.end)):
        if all(not beam._windows_overlap(item,old) for old in chosen):chosen.append(item)
        if len(chosen)>=target:break
    return sorted(chosen,key=lambda w:w.start,reverse=True),0,0.


def effective_change(before,args,arm):
    after=scalar_settings(args)
    differences={k:dict(before=before.get(k),after=after.get(k)) for k in sorted(set(before)|set(after)) if before.get(k)!=after.get(k)}
    hooks=list(ARMS[arm].get('hooks',[]))
    # A graft replacement remains a callable intervention, even with an empty current cache.
    if 'suffix' in hooks and not args.post_polish_qubo:hooks.remove('suffix')
    if 'windows' in hooks and not args.post_polish_qubo:hooks.remove('windows')
    return differences,hooks


class ComponentIntervention:
    def __init__(self,arm):
        self.arm=arm;self.originals={};self.sampler_original=None
        self.calls=collections.Counter();self.sampler_calls=collections.Counter();self.applied=[]
        self.effective=None

    def patch(self,name,callback):
        original=getattr(beam,name);self.originals[name]=original
        def wrapped(*args,**kwargs):
            self.calls[name]+=1
            return callback(original,*args,**kwargs)
        setattr(beam,name,wrapped)

    def install(self):
        for name in ('_apply_solver_profile','_apply_qubo_influence_profile','_apply_dynamic_profile'):
            def configure(original,args,name=name):
                original(args);apply_updates(args,self.arm)
                self.applied.append(name);self.effective=scalar_settings(args)
            self.patch(name,configure)
        for name in ('_select_enhanced_qubo_frontier','_select_scheduler_state_qubo_frontier',
                     '_inject_macro_chain_witness_candidates','_inject_unified_macro_supply_candidates',
                     '_completion_witness_grafted_moves','_union_rollout_probe_grafted_moves',
                     '_select_post_polish_suffixes_qubo','_select_post_polish_windows_qubo',
                     '_select_post_polish_windows_v2_qubo','_post_polish_successors','_bs_frg_postcompress',
                     '_select_wang_pair_qubo_supplement'):
            self.patch(name,lambda original,*a,**kw:original(*a,**kw))
        hooks=ARMS[self.arm].get('hooks',[])
        if 'core' in hooks:
            original=self.originals['_select_enhanced_qubo_frontier']
            def core(ranked,count,args,beam_width=None,core_width=None,qubo_supplement_width=None):
                self.calls['core_removed']+=1
                return original(ranked,count,args,beam_width,0,qubo_supplement_width)
            beam._select_enhanced_qubo_frontier=core
        if 'graft' in hooks:
            for name in ('_completion_witness_grafted_moves','_union_rollout_probe_grafted_moves'):
                def disabled(*a,name=name,**kw):self.calls[name]+=1;return None
                setattr(beam,name,disabled)
        if 'suffix' in hooks:
            def suffix(items,args):self.calls['linear_suffix_calls']+=1;return linear_suffixes(items,args)
            beam._select_post_polish_suffixes_qubo=suffix
        if 'windows' in hooks:
            def windows(items,args,version=1):self.calls['linear_window_calls']+=1;return linear_windows(items,args,version)
            beam._select_post_polish_windows_qubo=windows
            beam._select_post_polish_windows_v2_qubo=lambda items,args:windows(items,args,2)
        cls=beam.SimulatedAnnealingQuboSampler
        self.sampler_original=cls.sample
        def sample(sampler,model,config):
            frame=inspect.currentframe().f_back
            while frame and not frame.f_code.co_name.startswith('_select_'):frame=frame.f_back
            owner=frame.f_code.co_name if frame else 'other'
            del frame
            self.sampler_calls[owner]+=1
            if ARMS[self.arm].get('forbid_sampling'):raise RuntimeError('Disabled QUBO sampler still called: '+owner)
            return self.sampler_original(sampler,model,config)
        cls.sample=sample

    def verify_effective(self,args):
        bad={k:(getattr(args,k,None),v) for k,v in ARMS[self.arm]['updates'].items() if getattr(args,k,None)!=v}
        if bad:raise RuntimeError('Component overwritten by native configuration: '+str(bad))
        if args.business_cap_enabled or args.inner_jobs!=1 or args.post_polish_inner_jobs!=1:
            raise RuntimeError('Common CPMP/resource contract changed')

    def restore(self):
        if self.sampler_original is not None:beam.SimulatedAnnealingQuboSampler.sample=self.sampler_original
        for name,original in self.originals.items():setattr(beam,name,original)


class ComponentObserver(PaperObserver):
    """Verify every committed next Beam without storing giant context snapshots."""
    def __init__(self):super().__init__('full_sa')

    def sample(self,model,config,usable,target,linear,features,args,feature_sec,build_sec):
        self.active_args=args;self.usable=list(usable);self.linear=linear;self.features=features
        result=SelectorObserver.sample(self,model,config,usable,target,linear,features,args,feature_sec,build_sec)
        self.current['layer']=self.current_layer
        self.current['macro_pool']=sum(beam._union_diversity_is_macro_feature(f) for f in features)
        self.pending_snapshot=None
        return result
