import os
import tempfile
import duckdb
import pandas as pd
from verifier import ELTVerifier
from guardrails import ExecutionSandbox

class ELTEnvironment:
    """
    Gym-style environment for evaluating ELT scripts against DuckDB and golden datasets.
    """
    def __init__(self, target_table_name: str = "target_table"):
        self.target_table_name = target_table_name
        self.db_path = None
        self.temp_dir_obj = None
        self.verifier = None
        self.source_df = None

    def reset(self, source_df: pd.DataFrame, golden_df: pd.DataFrame) -> dict:
        """
        Reset the environment state. Creates a fresh ephemeral database on disk,
        scrambles the source data, registers the source table, and returns basic info.
        """
        # Clean up any lingering DB files
        self.close()

        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir_obj.name, "elt_warehouse.db")

        # Initialize verifier with golden expectations
        self.verifier = ELTVerifier(golden_df)

        # Perturb source data to avoid hardcoding shortcuts
        self.source_df = ELTVerifier.perturb_source_data(source_df)

        # Ingest the perturbed data into the ephemeral warehouse
        conn = duckdb.connect(self.db_path)
        conn.register("source_df_view", self.source_df)
        conn.execute("CREATE TABLE source_table AS SELECT * FROM source_df_view")
        conn.close()

        return {
            "db_path": self.db_path,
            "source_table": "source_table",
            "target_table": self.target_table_name
        }

    def step(self, action_code: str, timeout: int = 30) -> tuple[float, bool, dict]:
        """
        Executes the agent's action script in the sandboxed subprocess,
        compares the resulting target table against the golden dataset, and returns (reward, done, info).
        """
        if not self.db_path or not os.path.exists(self.db_path):
            raise ValueError("Environment must be reset before taking a step.")

        # Provide connection details to the sandbox environment
        env_vars = {
            "DUCKDB_PATH": self.db_path,
            "SOURCE_TABLE": "source_table",
            "TARGET_TABLE": self.target_table_name
        }

        # Run script within sandbox
        success, stdout, stderr = ExecutionSandbox.run_script(action_code, timeout=timeout, env_vars=env_vars)

        reward = 0.0
        if success:
            conn = duckdb.connect(self.db_path)
            reward = self.verifier.verify_warehouse_state(conn, self.target_table_name)
            conn.close()

        info = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr
        }
        
        # Single-step environment (run code -> verify -> terminate)
        done = True

        return reward, done, info

    def close(self):
        """
        Cleans up the temporary warehouse directory.
        """
        if self.temp_dir_obj:
            try:
                self.temp_dir_obj.cleanup()
            except Exception:
                pass
            self.temp_dir_obj = None
            self.db_path = None
