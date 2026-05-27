from __future__ import annotations

from container_reshuffle.core import Instance, ProblemVariant, State
from container_reshuffle.qubo.beam_selection import QuboBeamSelectionBuilder
from container_reshuffle.qubo.model import QuboModel
from container_reshuffle.qubo.sampler import QuboSample, QuboSamplerConfig, SimulatedAnnealingQuboSampler
from container_reshuffle.solvers.beam import Candidate


def test_exactly_k_penalty_prefers_k_selected():
    model = QuboModel()
    vars_ = [model.add_binary_var(f"x{i}") for i in range(5)]
    model.add_exactly_k_penalty(vars_, 2, 5.0)
    assert model.evaluate({0: 1, 1: 1}) < model.evaluate({0: 1})


def test_qubo_evaluate():
    model = QuboModel()
    x = model.add_binary_var("x")
    y = model.add_binary_var("y")
    model.add_linear(x, 1.0)
    model.add_linear(y, 2.0)
    model.add_quadratic(x, y, 3.0)
    assert model.evaluate({x: 1, y: 1}) == 6.0


def test_sa_sampler_returns_samples():
    model = QuboModel()
    x = model.add_binary_var("x")
    model.add_linear(x, -1.0)
    ss = SimulatedAnnealingQuboSampler().sample(model, QuboSamplerConfig(num_reads=3, sa_iterations=50))
    assert ss.samples


def test_qubo_beam_selection_selects_b():
    inst = Instance("x", ProblemVariant.PREMARSHALLING_CPMP, 2, 2, 2, [[1, 2], []])
    candidates = [
        Candidate(State([[1], [2]], move_count=1), 1.0, None, (i,))
        for i in range(4)
    ]
    builder = QuboBeamSelectionBuilder()
    model, usable = builder.build(candidates, 2, 0.1, 10.0, 10)
    sample = QuboSample({0: 1, 1: 1}, model.evaluate({0: 1, 1: 1}))
    selected, repair = builder.decode_and_repair(sample, usable, 2)
    assert len(selected) == 2


def test_qubo_repair_too_many_too_few():
    candidates = [Candidate(State([[]]), float(i), None, (i,)) for i in range(5)]
    builder = QuboBeamSelectionBuilder()
    selected, repair = builder.decode_and_repair(QuboSample({0: 1, 1: 1, 2: 1}, 0), candidates, 2)
    assert len(selected) == 2 and repair
    selected, repair = builder.decode_and_repair(QuboSample({}, 0), candidates, 2)
    assert len(selected) == 2 and repair


def test_qubo_objective_aware_repair_improves_feasible_sample():
    candidates = [Candidate(State([[]]), float(i), None, (i,)) for i in [0, 1, 10, 11]]
    builder = QuboBeamSelectionBuilder()
    selected, repair = builder.decode_and_repair(
        QuboSample({2: 1, 3: 1}, 0),
        candidates,
        2,
        repair_hamming_slack=3,
    )
    assert [cand.score for cand in selected] == [0.0, 1.0]
    assert repair
