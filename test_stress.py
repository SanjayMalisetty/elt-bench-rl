import unittest
import pandas as pd
import numpy as np
import time
from environment import ELTEnvironment
from verifier import ELTVerifier

class TestStress(unittest.TestCase):
    def test_stress_1_false_positive_trap(self):
        print("\n=== Running Test 1: The False Positive Trap ===")
        source_df = pd.DataFrame({
            "category": ["A", "B", "C"],
            "amount": [10, 20, 30]
        })
        golden_df = pd.DataFrame({
            "category": ["A", "B", "C"],
            "total_amount": [10, 20, 30]
        })

        env = ELTEnvironment(target_table_name="target_aggregate")
        env.reset(source_df, golden_df)

        # Creates target_aggregate but populates wrong data
        bad_agent_script = """
import os
import duckdb

db_path = os.environ["DUCKDB_PATH"]
tgt_table = os.environ["TARGET_TABLE"]

conn = duckdb.connect(db_path)
conn.execute(f"CREATE TABLE {tgt_table} AS SELECT 'A' as category, 999 as total_amount")
conn.close()
"""
        reward, done, info = env.step(bad_agent_script)
        env.close()

        print(f"Success state: {info['success']}")
        print(f"Reward received: {reward} (Expected: 0.0)")
        self.assertEqual(reward, 0.0)

    def test_stress_2_timeout_guardrail(self):
        print("\n=== Running Test 2: Infinite Loop & Timeout Guardrail ===")
        source_df = pd.DataFrame({"val": [1, 2]})
        golden_df = pd.DataFrame({"val": [1, 2]})

        env = ELTEnvironment(target_table_name="target_table")
        env.reset(source_df, golden_df)

        # Script containing infinite loop
        loop_script = """
import time
while True:
    time.sleep(0.1)
"""
        start_time = time.time()
        # Set a short timeout of 2 seconds for faster verification
        reward, done, info = env.step(loop_script, timeout=2)
        elapsed = time.time() - start_time
        env.close()

        print(f"Success state: {info['success']} (Expected: False)")
        print(f"Elapsed time: {elapsed:.2f} seconds (Expected: ~2 seconds)")
        print(f"Stderr context: {info['stderr']}")
        print(f"Reward received: {reward} (Expected: 0.0)")

        self.assertFalse(info["success"])
        self.assertEqual(reward, 0.0)
        self.assertLess(elapsed, 5.0) # Ensure it didn't block for the default 30s or longer

    def test_stress_3_data_perturbation(self):
        print("\n=== Running Test 3: Data Perturbation / Anti-Hacking ===")
        source_df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "value": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        golden_df = source_df.copy()

        env = ELTEnvironment()
        
        print("Original source dataframe:")
        print(source_df)

        # Perform resetting twice to observe perturbation behavior
        init_state_1 = env.reset(source_df, golden_df)
        perturbed_1 = env.source_df.copy()
        
        init_state_2 = env.reset(source_df, golden_df)
        perturbed_2 = env.source_df.copy()

        env.close()

        print("\nPerturbed dataframe (Run 1):")
        print(perturbed_1)
        print("\nPerturbed dataframe (Run 2):")
        print(perturbed_2)

        # Assert that row ordering or ID mapping has changed
        self.assertFalse(source_df.equals(perturbed_1), "Dataframe was not perturbed in Run 1")
        self.assertFalse(perturbed_1.equals(perturbed_2), "Perturbations should be dynamic across resets")

if __name__ == "__main__":
    unittest.main()
