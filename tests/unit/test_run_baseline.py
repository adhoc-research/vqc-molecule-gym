import argparse
import random

from vqc_molecule_gym.baselines.beam_search_agent import BeamSearchAgent
from vqc_molecule_gym.baselines.greedy_agent import GreedyAgent
from vqc_molecule_gym.baselines.random_agent import RandomAgent
import pytest

from scripts.run_baseline import (
    _balanced_counts,
    _build_agent,
    _normalize_agent_name,
    _parse_angle_grid,
    _parse_refinement_deltas,
)


class DummyEvaluator:
    pass


def test_normalize_agent_name_accepts_beam_alias() -> None:
    assert _normalize_agent_name("beam") == "beam_search"
    assert _normalize_agent_name("greedy") == "greedy"


def test_build_agent_selects_random_greedy_and_beam_search() -> None:
    args = argparse.Namespace(seed=123, max_operators=2, candidate_limit=3, beam_width=5)
    evaluator = DummyEvaluator()

    random_agent = _build_agent("random", evaluator, args)
    greedy_agent = _build_agent("greedy", evaluator, args)
    beam_agent = _build_agent("beam_search", evaluator, args)

    assert isinstance(random_agent, RandomAgent)
    assert isinstance(random_agent.rng, random.Random)
    assert isinstance(greedy_agent, GreedyAgent)
    assert greedy_agent.max_operators == 2
    assert greedy_agent.candidate_limit == 3
    assert isinstance(beam_agent, BeamSearchAgent)
    assert beam_agent.beam_width == 5
    assert beam_agent.max_operators == 2
    assert beam_agent.candidate_limit == 3


def test_build_agent_passes_search_ranking_and_refinement_options() -> None:
    args = argparse.Namespace(
        seed=123,
        max_operators=2,
        candidate_limit=3,
        beam_width=5,
        angle_grid=(-0.1, 0.1),
        candidate_ranking="single_step",
        ranking_angle_grid=(-0.1,),
        refine_angles=True,
        refinement_candidates=2,
        refinement_deltas=(0.025,),
        refinement_passes=2,
    )
    evaluator = DummyEvaluator()

    greedy_agent = _build_agent("greedy", evaluator, args)
    beam_agent = _build_agent("beam_search", evaluator, args)

    assert greedy_agent.candidate_ranking == "single_step"
    assert greedy_agent.ranking_angle_grid == (-0.1,)
    assert greedy_agent.refine_angles is True
    assert greedy_agent.refinement_candidates == 2
    assert greedy_agent.refinement_deltas == (0.025,)
    assert greedy_agent.refinement_passes == 2
    assert beam_agent.candidate_ranking == "single_step"
    assert beam_agent.ranking_angle_grid == (-0.1,)
    assert beam_agent.refine_angles is True


def test_parse_angle_grid_validates_range_and_empty_values() -> None:
    assert _parse_angle_grid("-0.5, 0.0,0.25") == (-0.5, 0.0, 0.25)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_angle_grid("")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_angle_grid("0.6")


def test_parse_refinement_deltas_validates_nonzero_and_normalizes_sign() -> None:
    assert _parse_refinement_deltas("-0.025,0.0125") == (0.025, 0.0125)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_refinement_deltas("0.0")


def test_balanced_counts_handles_remainder() -> None:
    assert _balanced_counts(5, 3) == [2, 2, 1]
