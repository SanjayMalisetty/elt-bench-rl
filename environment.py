from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import pandas as pd

from guardrails import ExecutionSandbox
from task import ELTTask
from verifier import ELTVerifier
from warehouse import DuckDBWarehouse, Warehouse, quote_identifier


class ELTEnvironment:
    """Multi-turn, execution-backed environment for one ELT task episode."""

    def __init__(self, target_table_name: str = "target_table", warehouse: str = "duckdb", max_steps: int = 8, warehouse_factory: Callable[[str], Warehouse] | None = None):
        if warehouse != "duckdb" and warehouse_factory is None:
            raise ValueError(f"unsupported warehouse: {warehouse}")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.target_table_name = target_table_name
        self.max_steps = max_steps
        self.warehouse_factory = warehouse_factory or DuckDBWarehouse
        self.db_path: str | None = None
        self.temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
        self.warehouse: Warehouse | None = None
        self.verifier: ELTVerifier | None = None
        self.source_df: pd.DataFrame | None = None
        self.task: ELTTask | None = None
        self.step_count = 0

    def reset(self, source_df: pd.DataFrame | None = None, golden_df: pd.DataFrame | None = None, task: ELTTask | None = None) -> dict[str, str]:
        if task is None:
            if source_df is None or golden_df is None:
                raise ValueError("reset requires task or source_df and golden_df")
            task = ELTTask(
                task_id="local-inline",
                prompt="Transform the source data into the expected target table.",
                source_df=source_df,
                golden_df=golden_df,
                target_table=self.target_table_name,
            )
        self.close()
        self.task = task
        self.target_table_name = task.target_table
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir_obj.name) / "elt_warehouse.db")
        self.warehouse = self.warehouse_factory(self.db_path)
        self.verifier = ELTVerifier(task.golden_df)
        self.source_df = ELTVerifier.perturb_source_data(task.source_df)
        source_tables = task.source_tables or {"source_table": self.source_df}
        for table_name, frame in source_tables.items():
            self.warehouse.load_source(table_name, ELTVerifier.perturb_source_data(frame))
        self.warehouse.close()
        self.step_count = 0
        return {
            "task_id": task.task_id,
            "db_path": self.db_path,
            "source_table": "source_table",
            "target_table": self.target_table_name,
            "observation": task.observation(),
        }

    def step(self, action_code: str, timeout: int = 30, final: bool = True) -> tuple[float, bool, dict]:
        if not self.db_path or not self.warehouse or not self.task:
            raise ValueError("Environment must be reset before taking a step.")
        if not isinstance(action_code, str) or not action_code.strip():
            raise ValueError("action_code must be a non-empty string")

        self.step_count += 1
        env_vars = {
            "DUCKDB_PATH": self.db_path,
            "SOURCE_TABLE": "source_table",
            "TARGET_TABLE": self.target_table_name,
        }
        success, stdout, stderr = ExecutionSandbox.run_script(action_code, timeout=timeout, env_vars=env_vars)
        reward = 0.0
        verification_error = ""
        if success and final:
            verifier_connection = self.warehouse_factory(self.db_path)
            reward = self.verifier.verify_warehouse_state(verifier_connection.connection(), self.target_table_name)
            verifier_connection.close()
            if reward == 0.0:
                verification_error = f"Target table {quote_identifier(self.target_table_name)} does not match the expected warehouse state."

        done = final or self.step_count >= self.max_steps
        feedback = "Execution succeeded." if success else "Execution failed."
        if stdout.strip():
            feedback += f" stdout: {stdout.strip()}"
        if stderr.strip():
            feedback += f" stderr: {stderr.strip()}"
        if verification_error:
            feedback += f" {verification_error}"
        if not final and not done:
            feedback += " Continue iterating or submit a final action."
        info = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "step": self.step_count,
            "feedback": feedback,
            "observation": feedback,
        }
        return reward, done, info

    def close(self) -> None:
        if self.warehouse:
            self.warehouse.close()
            self.warehouse = None
        if self.temp_dir_obj:
            self.temp_dir_obj.cleanup()
            self.temp_dir_obj = None
        self.db_path = None
