"""Local-search placeholders built on top of validated classical solutions."""

from __future__ import annotations

from .beam import BeamSearchSolver


class LocalSearchSolver(BeamSearchSolver):
    """First local-search baseline: reuse beam search as a validated fallback."""

    name = "local_search"
