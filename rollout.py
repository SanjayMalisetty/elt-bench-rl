from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from environment import ELTEnvironment
from task import ELTTask


class Agent(Protocol):
    def act(self, observation: str, feedback: str | None = None) -> tuple[str, bool]: ...


@dataclass
class RolloutResult:
    task_id: str
    reward: float
    done: bool
    turns: int
    transcript: list[dict[str, str]]


def run_rollout(env: ELTEnvironment, task: ELTTask, agent: Agent, timeout: int = 30) -> RolloutResult:
    state = env.reset(task=task)
    observation = state["observation"]
    feedback = None
    transcript: list[dict[str, str]] = []
    reward = 0.0
    done = False

    while not done:
        action, final = agent.act(observation, feedback)
        reward, done, info = env.step(action, timeout=timeout, final=final)
        transcript.append({"observation": observation, "action": action, "feedback": info["feedback"]})
        observation = info["observation"]
        feedback = info["feedback"]

    return RolloutResult(task.task_id, reward, done, len(transcript), transcript)
