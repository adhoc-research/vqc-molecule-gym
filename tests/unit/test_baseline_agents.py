from vqc_molecule_gym.baselines.beam_search_agent import BeamSearchAgent
from vqc_molecule_gym.baselines.greedy_agent import GreedyAgent
from vqc_molecule_gym.schemas.result import EvalResult
from vqc_molecule_gym.schemas.task import ActiveSpace, Constraints, Geometry, ReferenceEnergy, TaskSpec


class FakeEvaluator:
    def __init__(self, rewards: dict[tuple[str, ...], float], invalid: set[tuple[str, ...]] | None = None) -> None:
        self.rewards = rewards
        self.invalid = invalid or set()
        self.calls: list[tuple[str, ...]] = []

    def evaluate_payload(self, task: TaskSpec, action_payload: dict[str, object]) -> EvalResult:
        sequence = tuple(action_payload["operator_sequence"])
        self.calls.append(sequence)
        valid = sequence not in self.invalid
        return EvalResult(
            valid=valid,
            task_id=task.task_id,
            reward=self.rewards.get(sequence, -1.0),
            errors=[] if valid else [],
        )


class ParameterFakeEvaluator:
    def __init__(self, rewards: dict[tuple[tuple[str, ...], tuple[float, ...]], float]) -> None:
        self.rewards = rewards
        self.calls: list[tuple[tuple[str, ...], tuple[float, ...]]] = []

    def evaluate_payload(self, task: TaskSpec, action_payload: dict[str, object]) -> EvalResult:
        sequence = tuple(action_payload["operator_sequence"])
        parameters = tuple(action_payload.get("parameters", []))
        self.calls.append((sequence, parameters))
        return EvalResult(task_id=task.task_id, valid=True, reward=self.rewards.get((sequence, parameters), -1.0))


class EnergyFakeEvaluator:
    def __init__(self, values: dict[tuple[str, ...], tuple[float, float]]) -> None:
        self.values = values
        self.calls: list[tuple[str, ...]] = []

    def evaluate_payload(self, task: TaskSpec, action_payload: dict[str, object]) -> EvalResult:
        sequence = tuple(action_payload["operator_sequence"])
        self.calls.append(sequence)
        reward, error = self.values.get(sequence, (-1.0, 999.0))
        return EvalResult(task_id=task.task_id, valid=True, reward=reward, energy_error_mha=error)


class ParameterEnergyFakeEvaluator:
    def __init__(self, values: dict[tuple[tuple[str, ...], tuple[float, ...]], tuple[float, float]]) -> None:
        self.values = values
        self.calls: list[tuple[tuple[str, ...], tuple[float, ...]]] = []

    def evaluate_payload(self, task: TaskSpec, action_payload: dict[str, object]) -> EvalResult:
        sequence = tuple(action_payload["operator_sequence"])
        parameters = tuple(action_payload.get("parameters", []))
        self.calls.append((sequence, parameters))
        reward, error = self.values.get((sequence, parameters), (-1.0, 999.0))
        return EvalResult(task_id=task.task_id, valid=True, reward=reward, energy_error_mha=error)


def test_greedy_picks_best_improving_operator_and_stops() -> None:
    evaluator = FakeEvaluator(
        {
            (): 0.0,
            ("A",): 0.5,
            ("B",): 0.3,
            ("A", "A"): 0.4,
            ("A", "B"): 0.4,
        }
    )
    agent = GreedyAgent(evaluator, max_operators=3)

    action = agent.act(_task(max_operators=3), ["B", "A"])

    assert action.operator_sequence == ["A"]
    assert action.shots == 10000
    assert agent.last_search_evaluations == 5
    assert evaluator.calls == [(), ("A",), ("B",), ("A", "A"), ("A", "B")]


def test_greedy_tie_breaks_by_shorter_then_lexicographic_sequence() -> None:
    evaluator = FakeEvaluator({(): 0.0, ("B",): 0.5, ("A",): 0.5, ("A", "A"): 0.5, ("A", "B"): 0.5})
    agent = GreedyAgent(evaluator, max_operators=2)

    action = agent.act(_task(max_operators=2), ["B", "A"])

    assert action.operator_sequence == ["A"]


def test_beam_search_tracks_best_across_depths() -> None:
    evaluator = FakeEvaluator(
        {
            (): 0.0,
            ("A",): 0.1,
            ("B",): 0.2,
            ("B", "A"): 0.9,
            ("B", "B"): 0.1,
        }
    )
    agent = BeamSearchAgent(evaluator, beam_width=1, max_operators=2)

    action = agent.act(_task(max_operators=2), ["A", "B"])

    assert action.operator_sequence == ["B", "A"]
    assert agent.last_search_evaluations == 5


def test_greedy_returns_nonempty_even_when_empty_reward_is_higher() -> None:
    evaluator = FakeEvaluator({(): 1.0, ("A",): 0.1, ("B",): 0.2})
    agent = GreedyAgent(evaluator, max_operators=1)

    action = agent.act(_task(max_operators=1), ["A", "B"])

    assert action.operator_sequence == ["B"]
    assert agent.last_search_metadata["base_reward"] == 1.0
    assert agent.last_search_metadata["best_nonempty_sequence"] == ["B"]


def test_beam_returns_nonempty_even_when_empty_reward_is_higher() -> None:
    evaluator = FakeEvaluator({(): 1.0, ("A",): 0.1, ("B",): 0.2})
    agent = BeamSearchAgent(evaluator, beam_width=2, max_operators=1)

    action = agent.act(_task(max_operators=1), ["A", "B"])

    assert action.operator_sequence == ["B"]
    assert agent.last_search_metadata["base_reward"] == 1.0
    assert agent.last_search_metadata["best_nonempty_sequence"] == ["B"]


def test_greedy_prefers_energy_improvement_over_higher_reward() -> None:
    evaluator = EnergyFakeEvaluator({(): (1.0, 50.0), ("A",): (0.9, 40.0), ("B",): (0.1, 20.0)})
    agent = GreedyAgent(evaluator, max_operators=1)

    action = agent.act(_task(max_operators=1), ["A", "B"])

    assert action.operator_sequence == ["B"]
    assert agent.last_search_metadata["best_nonempty_sequence"] == ["B"]
    assert agent.last_search_metadata["best_nonempty_delta_error_mha"] == 30.0
    assert agent.last_best_result is not None
    assert agent.last_best_result.metadata["energy_improvement"]["delta_error_mha"] == 30.0


def test_beam_prefers_energy_improvement_over_higher_reward() -> None:
    evaluator = EnergyFakeEvaluator({(): (1.0, 50.0), ("A",): (0.9, 40.0), ("B",): (0.1, 20.0)})
    agent = BeamSearchAgent(evaluator, beam_width=2, max_operators=1)

    action = agent.act(_task(max_operators=1), ["A", "B"])

    assert action.operator_sequence == ["B"]
    assert agent.last_search_metadata["best_nonempty_sequence"] == ["B"]
    assert agent.last_search_metadata["best_nonempty_delta_error_mha"] == 30.0


def test_single_step_candidate_ranking_prefers_energy_improvement_over_reward() -> None:
    evaluator = ParameterEnergyFakeEvaluator(
        {
            ((), ()): (1.0, 50.0),
            (("A",), (0.1,)): (0.9, 40.0),
            (("C",), (0.1,)): (0.1, 20.0),
        }
    )
    agent = GreedyAgent(evaluator, max_operators=1, candidate_limit=1, angle_grid=[0.1], candidate_ranking="single_step")

    action = agent.act(_task(max_operators=1), ["A", "C"])

    assert action.operator_sequence == ["C"]
    assert action.parameters == [0.1]
    assert agent.last_search_metadata["selected_candidates"] == ["C"]


def test_parameterized_greedy_selects_operator_angle_pair() -> None:
    evaluator = ParameterFakeEvaluator({((), ()): 0.0, (("A",), (-0.1,)): 0.2, (("A",), (0.1,)): 0.8})
    agent = GreedyAgent(evaluator, max_operators=1, angle_grid=[-0.1, 0.1])

    action = agent.act(_task(max_operators=1), ["A"])

    assert action.operator_sequence == ["A"]
    assert action.parameters == [0.1]


def test_beam_search_respects_beam_width_and_candidate_limit() -> None:
    evaluator = FakeEvaluator({(): 0.0, ("A",): 0.1, ("B",): 0.2, ("C",): 1.0, ("B", "A"): 0.3, ("B", "B"): 0.4})
    agent = BeamSearchAgent(evaluator, beam_width=1, max_operators=2, candidate_limit=2)

    action = agent.act(_task(max_operators=2), ["C", "B", "A"])

    assert action.operator_sequence == ["B", "B"]
    assert ("C",) not in evaluator.calls
    assert agent.last_search_evaluations == 5


def test_single_step_candidate_ranking_finds_best_operator_before_limit() -> None:
    evaluator = ParameterFakeEvaluator(
        {
            ((), ()): 0.0,
            (("A",), (0.1,)): 0.1,
            (("B",), (0.1,)): 0.2,
            (("C",), (0.1,)): 0.9,
        }
    )
    agent = GreedyAgent(evaluator, max_operators=1, candidate_limit=1, angle_grid=[0.1], candidate_ranking="single_step")

    action = agent.act(_task(max_operators=1), ["A", "B", "C"])

    assert action.operator_sequence == ["C"]
    assert action.parameters == [0.1]
    assert agent.last_search_metadata["selected_candidates"] == ["C"]
    assert agent.last_search_metadata["ranking_evaluations"] == 3


def test_ranked_beam_can_cross_non_improving_single_operator_state() -> None:
    evaluator = ParameterFakeEvaluator(
        {
            ((), ()): 0.0,
            (("A",), (0.1,)): -0.2,
            (("C",), (0.1,)): -0.1,
            (("C", "C"), (0.1, 0.1)): 0.8,
        }
    )
    agent = BeamSearchAgent(
        evaluator,
        beam_width=1,
        max_operators=2,
        candidate_limit=1,
        angle_grid=[0.1],
        candidate_ranking="single_step",
    )

    action = agent.act(_task(max_operators=2), ["A", "C"])

    assert action.operator_sequence == ["C", "C"]
    assert action.parameters == [0.1, 0.1]
    assert agent.last_search_metadata["selected_candidates"] == ["C"]


def test_local_angle_refinement_can_turn_near_miss_into_nonempty_winner() -> None:
    evaluator = ParameterFakeEvaluator(
        {
            ((), ()): 0.0,
            (("A",), (0.1,)): -0.05,
            (("A",), (0.075,)): -0.1,
            (("A",), (0.125,)): 0.2,
        }
    )
    agent = GreedyAgent(
        evaluator,
        max_operators=1,
        angle_grid=[0.1],
        refine_angles=True,
        refinement_candidates=1,
        refinement_deltas=[0.025],
        refinement_passes=1,
    )

    action = agent.act(_task(max_operators=1), ["A"])

    assert action.operator_sequence == ["A"]
    assert action.parameters == [0.125]
    assert agent.last_search_metadata["refinement_enabled"] is True
    assert agent.last_search_metadata["refinement_evaluations"] == 2


def _task(max_operators: int = 8) -> TaskSpec:
    return TaskSpec(
        task_id="fake_task",
        benchmark_id="fake_benchmark",
        molecule="H2",
        geometry=Geometry(kind="cartesian", atoms=[("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)]),
        basis="sto-3g",
        active_space=ActiveSpace(electrons=2, orbitals=2, spin_orbitals=4, qubits=4),
        reference=ReferenceEnergy(method="fake", energy_hartree=-1.0),
        constraints=Constraints(max_operators=max_operators),
        operator_pool_id="fake_pool",
    )
