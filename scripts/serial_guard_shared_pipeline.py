"""Three-trajectory opt-ins through the unchanged paid output cleanup ledger."""
from __future__ import annotations

import paired_shared_pipeline as shared
from deferred_pair_shared_pipeline import run_continuation as original
from serial_classical_guards import MODES, run as refine


def run_continuation(instance, mode, handoff, start, budget, publish):
    if mode not in MODES:
        return original(instance, mode, handoff, start, budget, publish)
    old_modes, old_refine = shared.MODES, shared.refine
    try:
        shared.MODES = old_modes + MODES
        shared.refine = refine
        return shared.run_continuation(instance, mode, handoff, start, budget, publish)
    finally:
        shared.MODES, shared.refine = old_modes, old_refine
