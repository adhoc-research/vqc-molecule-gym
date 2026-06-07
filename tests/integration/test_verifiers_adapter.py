import asyncio
import time
from typing import Any

import verifiers as vf

from vqc_molecule_gym.envs.verifiers_adapter import QChemEnvironmentConfig, load_environment


class DummyQChemClient(vf.Client[None, list[Any], vf.Response, dict[str, Any]]):
    def __init__(self) -> None:
        super().__init__(None)

    def setup_client(self, config: vf.ClientConfig) -> None:
        return None

    async def to_native_tool(self, tool: vf.Tool) -> dict[str, Any]:
        return tool.model_dump(mode="json")

    async def to_native_prompt(self, messages: vf.Messages) -> tuple[list[Any], dict[str, Any]]:
        return messages, {}

    async def get_native_response(
        self,
        prompt: list[Any],
        model: str,
        sampling_args: vf.SamplingArgs,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> vf.Response:
        return vf.Response(
            id="dummy-qchem-response",
            created=int(time.time()),
            model=model,
            message=vf.ResponseMessage(
                role="assistant",
                content='{"operator_sequence": [], "shots": 10000}',
                finish_reason="stop",
                is_truncated=False,
            ),
        )

    async def raise_from_native_response(self, response: vf.Response) -> None:
        return None

    async def from_native_response(self, response: vf.Response) -> vf.Response:
        return response

    async def close(self) -> None:
        return None


def test_load_environment_accepts_config_and_kwargs() -> None:
    env_from_config = load_environment(QChemEnvironmentConfig(benchmark_id="h2_tiny", max_turns=1))
    env_from_kwargs = load_environment(benchmark_id="h2_tiny", max_turns=1)

    assert isinstance(env_from_config, vf.Env)
    assert isinstance(env_from_kwargs, vf.Env)
    assert len(env_from_config.get_dataset()) > 0
    assert len(env_from_kwargs.get_eval_dataset()) > 0


def test_h2_tiny_dummy_model_rollout_real_evaluator() -> None:
    async def run_rollout():
        env = load_environment(benchmark_id="h2_tiny", max_turns=1)
        dataset = env.get_dataset()
        eval_dataset = env.get_eval_dataset()

        assert isinstance(env, vf.Env)
        assert len(dataset) > 0
        assert len(eval_dataset) > 0

        state = await env.rollout(dataset[0], client=DummyQChemClient(), model="dummy-qchem")
        await env.harness.teardown()
        return state

    state = asyncio.run(run_rollout())

    assert state["is_completed"] is True
    assert state["completion"]
    assert state["qchem_action"] == {"operator_sequence": [], "shots": 10000}
    assert "qchem_parse_error" not in state
    assert "qchem_eval_error" not in state
    assert state["qchem_eval"]["task_id"] == "h2_r0.50"
    assert isinstance(state["qchem_eval"]["reward"], float)
    assert isinstance(state["reward"], float)
