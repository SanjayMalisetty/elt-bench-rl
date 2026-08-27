import unittest
import pandas as pd
import duckdb
import os
from environment import ELTEnvironment

class TestLocalIntegration(unittest.TestCase):
    def test_elt_pipeline_run(self):
        # Create dummy source data (no 'id' column here to keep verification invariant to row shuffling)
        source_df = pd.DataFrame({
            "category": ["A", "B", "A", "B", "C"],
            "amount": [10, 20, 30, 40, 50]
        })

        # Expected target warehouse state (aggregated sums)
        golden_df = pd.DataFrame({
            "category": ["A", "B", "C"],
            "total_amount": [40, 60, 50]
        })

        # Initialize environment
        env = ELTEnvironment(target_table_name="target_aggregate")
        init_state = env.reset(source_df, golden_df)
        
        print(f"\n--- Environment Reset ---")
        print(f"Database Path: {init_state['db_path']}")
        print(f"Source Table:  {init_state['source_table']}")
        print(f"Target Table:  {init_state['target_table']}")

        # Mock model transformation script
        agent_script = """
import os
import duckdb

db_path = os.environ["DUCKDB_PATH"]
src_table = os.environ["SOURCE_TABLE"]
tgt_table = os.environ["TARGET_TABLE"]

# Connect to the ephemeral warehouse and run ELT aggregation
conn = duckdb.connect(db_path)
conn.execute(f'''
    CREATE TABLE {tgt_table} AS
    SELECT category, SUM(amount) AS total_amount
    FROM {src_table}
    GROUP BY category
''')
conn.close()
"""

        # Execute step
        reward, done, info = env.step(agent_script)

        print(f"\n--- Step Execution Info ---")
        print(f"Success: {info['success']}")
        print(f"Stdout:  {info['stdout'].strip()}")
        print(f"Stderr:  {info['stderr'].strip()}")
        print(f"Reward:  {reward}")
        print(f"Done:    {done}")

        # Clean up database files
        env.close()

        # Assertions
        self.assertTrue(info["success"])
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)

if __name__ == "__main__":
    unittest.main()
