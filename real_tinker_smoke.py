from __future__ import annotations

import os
import sys

import pandas as pd

from environment import ELTEnvironment
from task import ELTTask
from tinker_recipe import TinkerUnavailable, create_tinker_message_agent


TASK = ELTTask(
    task_id="real-local-smoke",
    prompt="Create target by copying every row from source_table.",
    source_df=pd.DataFrame({"id": [1, 2], "value": [10, 20]}),
    golden_df=pd.DataFrame({"id": [1, 2], "value": [10, 20]}),
    target_table="target",
)


def main() -> int:
    if os.environ.get("TINKER_RUN_REAL") != "1":
        print("Skipped: set TINKER_RUN_REAL=1 to make a paid model request.")
        return 0
    try:
        agent = create_tinker_message_agent()
    except TinkerUnavailable as error:
        print(error)
        return 2

    env = ELTEnvironment()
    try:
        state = env.reset(task=TASK)
        action, _ = agent.act(state["observation"])
        reward, done, info = env.step(action, final=True)
        print("Generated action:\n" + action)
        print("Execution feedback:\n" + info["feedback"])
        print(f"Reward: {reward}; done: {done}")
        return 0 if reward == 1.0 else 1
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
