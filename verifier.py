import duckdb
import pandas as pd
import numpy as np

class ELTVerifier:
    def __init__(self, golden_df: pd.DataFrame):
        # Ground truth target state to compare against
        self.golden_df = golden_df.reset_index(drop=True)

    def verify_warehouse_state(self, conn: duckdb.DuckDBPyConnection, target_table_name: str) -> float:
        """
        Compare the agent's target table against the golden dataset.
        Returns 1.0 on exact match, 0.0 otherwise.
        """
        try:
            # Grab the agent's final table
            agent_df = conn.execute(f"SELECT * FROM {target_table_name}").df().reset_index(drop=True)

            if agent_df.shape != self.golden_df.shape:
                return 0.0

            # Ensure order-agnostic comparison by sorting columns
            agent_df = agent_df.reindex(sorted(agent_df.columns), axis=1)
            golden_sorted = self.golden_df.reindex(sorted(self.golden_df.columns), axis=1)

            pd.testing.assert_frame_equal(agent_df, golden_sorted, check_like=True, check_dtype=False)
            return 1.0
        except Exception:
            return 0.0

    @staticmethod
    def perturb_source_data(source_df: pd.DataFrame) -> pd.DataFrame:
        """
        Shuffle rows and scramble IDs on reset so agent can't memorize/hardcode outputs.
        """
        perturbed = source_df.sample(frac=1.0).reset_index(drop=True)
        if 'id' in perturbed.columns:
            perturbed['id'] = np.random.permutation(perturbed['id'].values)
        return perturbed
