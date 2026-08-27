from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

from rollout import run_rollout
from task import ELTTask
from environment import ELTEnvironment


class SamplingBackend(Protocol):
    def sample(self, prompt: str) -> str: ...


class TinkerUnavailable(RuntimeError):
    pass


class TinkerSamplingAgent:
    """Agent bridge: Tinker sampling produces Python actions for the environment."""

    def __init__(self, sampler: SamplingBackend):
        self.sampler = sampler

    def act(self, observation: str, feedback: str | None = None) -> tuple[str, bool]:
        prompt = observation if feedback is None else f"{observation}\nFeedback: {feedback}"
        action = self.sampler.sample(prompt)
        return action, True


class TinkerMessageAgent:
    """Synchronous rollout adapter around Cookbook's real message completer."""

    def __init__(self, completer: Any):
        self.completer = completer

    def act(self, observation: str, feedback: str | None = None) -> tuple[str, bool]:
        prompt = observation if feedback is None else f"{observation}\nFeedback: {feedback}"
        message = asyncio.run(self.completer([{"role": "user", "content": prompt}]))
        return str(message["content"]), True


def create_tinker_message_agent(
    base_model: str = "Qwen/Qwen3.5-4B",
    renderer_name: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> TinkerMessageAgent:
    """Create a real Tinker-backed agent; construction may contact Tinker's service."""
    require_tinker()
    import tinker
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook.completers import TinkerMessageCompleter

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=base_model)
    tokenizer = get_tokenizer(base_model)
    renderer = renderers.get_renderer(
        renderer_name or model_info.get_recommended_renderer_name(base_model),
        tokenizer,
        model_name=base_model,
    )
    return TinkerMessageAgent(
        TinkerMessageCompleter(sampling_client, renderer, max_tokens, temperature=temperature)
    )


@dataclass
class EpisodeExample:
    prompt: str
    completion: str
    reward: float


class TinkerTrainingBackend(Protocol):
    def optimization_step(self, examples: Iterable[EpisodeExample]) -> Any: ...


class TinkerRLVRRecipe:
    """Collect execution-derived episodes and delegate one policy update to Tinker."""

    def __init__(self, env: ELTEnvironment, backend: TinkerTrainingBackend):
        self.env = env
        self.backend = backend

    def optimization_step(
        self, task: ELTTask, agent: TinkerSamplingAgent
    ) -> tuple[Any, EpisodeExample]:
        result = run_rollout(self.env, task, agent)
        first_prompt = result.transcript[0]["observation"]
        completion = "\n".join(turn["action"] for turn in result.transcript)
        example = EpisodeExample(first_prompt, completion, result.reward)
        return self.backend.optimization_step([example]), example


def require_tinker() -> Any:
    """Import the SDK only when a configured training run actually needs it."""
    try:
        import tinker
    except ImportError as error:
        raise TinkerUnavailable(
            "Install the Tinker SDK and configure its service credentials for training."
        ) from error
    return tinker


def build_tinker_training_client(factory: Callable[..., Any], **kwargs: Any) -> Any:
    """Construct a Tinker client through the caller's pinned SDK/cookbook factory."""
    require_tinker()
    return factory(**kwargs)
