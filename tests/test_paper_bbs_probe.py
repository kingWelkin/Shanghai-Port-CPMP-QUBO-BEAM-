from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_paper_bbs_selects_frontier_by_ub_then_lb_and_prunes_incumbent_bound():
    from benchmark_paper_bounded_beam_cpmp import PaperBbsCandidate, _select_paper_bbs_frontier

    candidates = [
        PaperBbsCandidate(code=b"slow", n_value=3, depth=4, parent=b"p", ub_total=9, lb_total=7),
        PaperBbsCandidate(code=b"best", n_value=2, depth=4, parent=b"p", ub_total=7, lb_total=6),
        PaperBbsCandidate(code=b"tie", n_value=1, depth=5, parent=b"p", ub_total=7, lb_total=5),
        PaperBbsCandidate(code=b"cut", n_value=0, depth=8, parent=b"p", ub_total=8, lb_total=8),
    ]

    selected, pruned = _select_paper_bbs_frontier(candidates, beta=2, incumbent=8)

    assert [item.code for item in selected] == [b"tie", b"best"]
    assert pruned == 1


def test_paper_bbs_incumbent_successors_move_only_first_dirty_stack_top():
    from benchmark_paper_bounded_beam_cpmp import _paper_bbs_successors
    from run_business_exact_target_search_sqlite import _encode

    code, n_value = _encode([[1, 3], [2], [4]], stack_count=3, max_height=3)

    successors = _paper_bbs_successors(code, n_value, stack_count=3, max_height=3, branching="incumbent")

    assert len(successors) == 2
    assert {move[:2] for _nxt, _nxt_n, move in successors} == {(0, 1), (0, 2)}


def test_paper_bbs_cpmp_probe_solves_tiny_instance_with_legal_top_move():
    from benchmark_paper_bounded_beam_cpmp import make_paper_bbs_args, run_paper_bbs_cpmp_instance
    from open_cpmp_benchmarks import _open_validate_instance_result
    from container_reshuffle.core import Instance, ProblemVariant

    instance = Instance(
        instance_id="tiny_bbs",
        variant=ProblemVariant.PREMARSHALLING_CPMP,
        stack_count=3,
        max_height=3,
        container_count=3,
        stacks=[[1, 3], [2], []],
    )
    args = make_paper_bbs_args(beta=4, time_limit=1.0, alpha=10.0)

    result = run_paper_bbs_cpmp_instance(instance.instance_id, instance, args)
    business_ok, goal_ok, msg = _open_validate_instance_result(result, instance, business_alpha=10.0)

    assert result.final_n == 0
    assert len(result.moves) == 1
    assert result.moves[0].box == 3
    assert result.moves[0].from_stack == 0
    assert business_ok
    assert goal_ok, msg
