"""Use equivalent fast reduced construction in the frozen capped controller."""
from __future__ import annotations

import bounded_conditioned_refinement as original
from capped_conditioned_refinement import run as capped
from fast_reduced_trajectory_domains import FastReducedConditionedDomains


def run(instance, path, mode, start, budget, publish, **kwargs):
    saved = original.FactorizedConditionedDomains
    try:
        original.FactorizedConditionedDomains = FastReducedConditionedDomains
        return capped(instance, path, mode, start, budget, publish, **kwargs)
    finally:
        original.FactorizedConditionedDomains = saved
