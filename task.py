from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ELTTask:
    task_id: str
    prompt: str
    source_df: pd.DataFrame
    golden_df: pd.DataFrame
    target_table: str = "target_table"
    metadata: dict[str, Any] = field(default_factory=dict)
    source_tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def observation(self, source_table: str = "source_table") -> str:
        tables = self.source_tables or {source_table: self.source_df}
        schema = "; ".join(
            f"{table}: " + ", ".join(f"{name} ({dtype})" for name, dtype in frame.dtypes.items())
            for table, frame in tables.items()
        )
        return (
            f"Task: {self.prompt}\n"
            f"Source table: {source_table}\n"
            f"Source schema: {schema}\n"
            f"Target table: {self.target_table}\n"
            "Write Python code that connects to DUCKDB_PATH and creates the target table."
        )


def load_task(path: str | Path) -> ELTTask:
    """Load a credential-free task fixture or an exported ELT-Bench task."""
    payload = json.loads(Path(path).read_text())
    base_dir = Path(path).parent

    def dataframe(value: Any) -> pd.DataFrame:
        if isinstance(value, list):
            return pd.DataFrame(value)
        if isinstance(value, dict):
            return pd.DataFrame(value)
        if isinstance(value, str):
            return pd.read_csv(base_dir / value)
        raise TypeError("dataframe values must be a list, object, or CSV path")

    source = payload.get("source", payload.get("source_data"))
    golden = payload.get("golden", payload.get("golden_data", payload.get("expected")))
    if source is None or golden is None:
        raise ValueError("task must define source and golden data")
    return ELTTask(
        task_id=str(payload["task_id"]),
        prompt=str(payload["prompt"]),
        source_df=dataframe(source),
        golden_df=dataframe(golden),
        target_table=str(payload.get("target_table", "target_table")),
        metadata=dict(payload.get("metadata", {})),
    )
