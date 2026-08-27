from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from task import ELTTask


@dataclass
class ELTBenchTaskBundle:
    """Metadata bundle produced by the official ELT-Bench repository."""

    task: ELTTask
    config: dict[str, Any]
    data_model: dict[str, Any]
    evaluation_sql: dict[str, str]


def load_elt_bench_task(
    repository_root: str | Path,
    destination: str,
    task_name: str,
    ground_truth_root: str | Path | None = None,
) -> ELTBenchTaskBundle:
    """Load official task metadata and optional exported source/ground-truth CSVs.

    The benchmark provisions source data in a destination warehouse. For local
    runs, callers may export source tables and ground truth as CSV files using
    ``<task>/<table>.csv`` under the supplied roots.
    """
    root = Path(repository_root)
    task_root = root / "elt-bench" / destination / task_name
    config_path = task_root / "config.yaml"
    model_path = task_root / "data_model.yaml"
    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"missing ELT-Bench task metadata under {task_root}")

    config = yaml.safe_load(config_path.read_text()) or {}
    data_model = yaml.safe_load(model_path.read_text()) or {}
    source_root = root / "local_data" / destination / task_name
    truth_root = Path(ground_truth_root) / task_name if ground_truth_root else None
    source_tables = _load_tables(source_root)
    truth_tables = _load_tables(truth_root) if truth_root else {}
    if not source_tables or not truth_tables:
        raise FileNotFoundError(
            "local source and ground-truth CSVs are required; official ELT-Bench "
            "provisions these through its setup and Hugging Face download steps"
        )

    model_names = [str(model["name"]) for model in data_model.get("models", [])]
    if len(model_names) != 1:
        raise ValueError("local ELTTask adapter requires exactly one target model")
    target = model_names[0]
    sql_root = root / "evaluation" / "sql" / task_name
    evaluation_sql = {
        path.stem: path.read_text() for path in sorted(sql_root.glob("*.sql"))
    }
    task = ELTTask(
        task_id=f"{destination}/{task_name}",
        prompt=_prompt(data_model, destination, task_name, source_tables),
        source_df=next(iter(source_tables.values())),
        golden_df=truth_tables[target],
        target_table=target,
        metadata={"destination": destination, "config": config, "data_model": data_model},
        source_tables=source_tables,
    )
    return ELTBenchTaskBundle(task, config, data_model, evaluation_sql)


def _load_tables(root: Path | None) -> dict[str, pd.DataFrame]:
    if root is None or not root.exists():
        return {}
    return {path.stem: pd.read_csv(path) for path in sorted(root.glob("*.csv"))}


def _prompt(data_model: dict[str, Any], destination: str, task_name: str, tables: dict[str, pd.DataFrame]) -> str:
    models = data_model.get("models", [])
    descriptions = "\n".join(
        f"- {model.get('name')}: {model.get('description', '')}" for model in models
    )
    schemas = ", ".join(f"{name}({', '.join(frame.columns)})" for name, frame in tables.items())
    return (
        f"Implement ELT-Bench task {task_name} for {destination}.\n"
        f"Available source tables: {schemas}\nTarget models:\n{descriptions}\n"
        "Create the requested target model using the provided warehouse tools."
    )
