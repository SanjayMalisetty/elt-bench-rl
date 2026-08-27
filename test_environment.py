import unittest

import pandas as pd

from environment import ELTEnvironment
from rollout import run_rollout
from task import ELTTask
from tinker_recipe import EpisodeExample, TinkerRLVRRecipe


class CopyAgent:
    def __init__(self):
        self.turn = 0

    def act(self, observation, feedback=None):
        self.turn += 1
        if self.turn == 1:
            return "print('inspected')", False
        return '''import os, duckdb
conn = duckdb.connect(os.environ["DUCKDB_PATH"])
conn.execute("CREATE TABLE target AS SELECT * FROM source_table")
conn.close()''', True


class RecordingBackend:
    def __init__(self):
        self.examples = None

    def optimization_step(self, examples):
        self.examples = list(examples)
        return {"updated": True}


class TestEnvironmentArchitecture(unittest.TestCase):
    def setUp(self):
        self.source = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        self.task = ELTTask(
            "copy-task", "Copy all source rows.", self.source, self.source.iloc[::-1], "target"
        )
        self.env = ELTEnvironment(max_steps=3)

    def tearDown(self):
        self.env.close()

    def test_multi_turn_rollout_and_training_example(self):
        backend = RecordingBackend()
        recipe = TinkerRLVRRecipe(self.env, backend)
        update, example = recipe.optimization_step(self.task, CopyAgent())
        self.assertEqual(update, {"updated": True})
        self.assertEqual(example.reward, 1.0)
        self.assertEqual(len(backend.examples), 1)
        self.assertIsInstance(backend.examples[0], EpisodeExample)

    def test_episode_limit_ends_without_final_reward(self):
        self.env = ELTEnvironment(max_steps=1)
        self.env.reset(task=self.task)
        reward, done, info = self.env.step("print('inspect')", final=False)
        self.assertEqual((reward, done), (0.0, True))
        self.assertEqual(info["step"], 1)


if __name__ == "__main__":
    unittest.main()
