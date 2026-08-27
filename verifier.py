import duckdb
import pandas as pd
from warehouse import quote_identifier

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
            agent_df = conn.execute(
                f"SELECT * FROM {quote_identifier(target_table_name)}"
            ).df().reset_index(drop=True)

            if agent_df.shape != self.golden_df.shape:
                return 0.0

            columns = sorted(agent_df.columns)
            if columns != sorted(self.golden_df.columns):
                return 0.0

            agent_df = agent_df.reindex(columns=columns)
            golden_sorted = self.golden_df.reindex(columns=columns)

            agent_df = agent_df.sort_values(
                by=columns, kind="mergesort", key=lambda values: values.map(repr)
            ).reset_index(drop=True)
            golden_sorted = golden_sorted.sort_values(
                by=columns, kind="mergesort", key=lambda values: values.map(repr)
            ).reset_index(drop=True)

            pd.testing.assert_frame_equal(agent_df, golden_sorted, check_like=True, check_dtype=False)
            return 1.0
        except Exception:
            return 0.0

    @staticmethod
    def perturb_source_data(source_df: pd.DataFrame) -> pd.DataFrame:
        """
        Shuffle rows on reset without changing relationships between columns.
        """
        return source_df.sample(frac=1.0).reset_index(drop=True)
