# ELT-Bench RLVR Reference Environment

This repository is a small, credential-free reference implementation of an execution-backed ELT RLVR environment. It provides the interfaces needed to connect an ELT-Bench task loader, an LLM rollout policy, and a Tinker training backend.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest -v
```

The local tests execute real Python transformation scripts in a temporary DuckDB warehouse. `test_environment.py` demonstrates a multi-turn rollout: the policy inspects the state, receives execution feedback, submits a final action, and sends its execution-derived reward to a training backend.

## Architecture

- `task.py` defines `ELTTask` and loads JSON task fixtures with inline data or CSV paths. An official ELT-Bench adapter can map its task specification into this contract.
- `warehouse.py` defines the destination boundary and provides `DuckDBWarehouse`. Another destination can implement the same `load_source`, `connection`, and `close` methods and be passed as `warehouse_factory`.
- `environment.py` creates an isolated episode, perturbs only row order, executes bounded actions, returns feedback, and verifies the final warehouse state.
- `rollout.py` turns observations and feedback into a multi-turn agent trajectory.
- `tinker_recipe.py` converts a trajectory into an execution-rewarded training example and delegates the optimization step to a Tinker/tinker-cookbook backend.
- `guardrails.py` limits the child process environment, working directory, memory, and wall-clock time. It is a process boundary for local evaluation, not a security boundary for hostile code.

## Tinker integration

The SDK is optional for local tests because it is not available on every machine and requires service credentials. In a configured training environment, install the pinned Tinker SDK and tinker-cookbook version used by your run, create a sampler and training backend, then pass them to `TinkerSamplingAgent` and `TinkerRLVRRecipe`. The recipe keeps SDK-specific `Datum` and model-input construction in the backend so SDK upgrades do not change the warehouse environment.

`require_tinker()` fails with an actionable error only when a configured run imports the SDK. No credentials are copied into the child process: warehouse connection details must be explicitly provided by a destination adapter.

## Task fixture shape

```json
{
  "task_id": "aggregate-orders",
  "prompt": "Aggregate amount by category.",
  "source": [{"category": "A", "amount": 10}],
  "golden": [{"category": "A", "total_amount": 10}],
  "target_table": "target_aggregate"
}
```

The official benchmark integration should supply extraction/loading and task-specific transformation metadata through an adapter rather than modifying `ELTEnvironment`.

## Real model smoke test

After `tinker auth login`, run the real sampler path explicitly:

```bash
TINKER_RUN_REAL=1 .venv/bin/python real_tinker_smoke.py
```

This contacts Tinker, asks the model for a DuckDB action, executes that generated action, and prints the reward. It is opt-in because sampling consumes account quota. Without `TINKER_RUN_REAL=1`, the command exits without making a model request.

For an official task, first run the benchmark setup and export source and ground-truth CSVs, then load it with `load_elt_bench_task(...)` from `elt_bench_adapter.py`. The official setup uses Docker/Airbyte and a configured Snowflake, Databricks, or Redshift destination; those services are not replaced by the local DuckDB test.
