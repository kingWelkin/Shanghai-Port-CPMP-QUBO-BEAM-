"""Opt-in deferred controller through the existing original-clock output ledger."""
from __future__ import annotations

import paired_shared_pipeline as shared
from deferred_pair_refinement import MODES, run as refine


def run_continuation(instance, mode, handoff, start, budget, publish):
    if mode not in MODES:
        return shared.run_continuation(instance, mode, handoff, start, budget, publish)
    original_modes, original_refine = shared.MODES, shared.refine
    try:
        shared.MODES = original_modes + MODES
        shared.refine = refine
        return shared.run_continuation(instance, mode, handoff, start, budget, publish)
    finally:
        shared.MODES, shared.refine = original_modes, original_refine
