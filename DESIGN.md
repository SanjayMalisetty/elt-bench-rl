# ELT-Bench RLVR Design

## Scope

The environment owns the episode lifecycle and warehouse verification. A task adapter owns benchmark-specific extraction, credentials, and task parsing. A policy owns language-model prompting and action generation. A Tinker backend owns SDK-specific tokenization, loss construction, and optimizer calls.

## Episode contract

`ELTTask` contains a task id, natural-language specification, source data, expected final relation, destination name, and optional metadata. `reset(task=...)` creates a fresh temporary warehouse and returns the task prompt plus source schema. The source is row-shuffled only; column relationships and values remain intact.

A rollout may call `step(action, final=False)` for inspection, loading, or transformation attempts. The result includes stdout, stderr, and feedback for the next model turn. A final call verifies the target relation and returns an outcome reward of `1.0` or `0.0`. The episode also terminates at `max_steps`.

## Verification and rewards

The verifier checks target existence, exact column names, row count, and a multiset comparison of rows. Sorting is applied before comparison so SQL result order is irrelevant and duplicate rows remain meaningful. The default reward is a terminal outcome reward. Process rewards can be added in a task adapter, for example for a successful extraction or a valid intermediate table, but should not replace final warehouse-state verification.

## Destination abstraction

`Warehouse` is the destination protocol. `DuckDBWarehouse` is the local implementation used by tests. `warehouse_factory` lets a credentialed adapter provide another destination without changing rollout or reward code. Credentials are never inherited by the script sandbox; a destination adapter must explicitly expose only the connection information required for its action.

## Policy and Tinker

`Agent.act(observation, feedback)` is the policy boundary. `run_rollout` records prompt, action, and execution feedback as a transcript. `TinkerRLVRRecipe` converts a completed transcript into an `EpisodeExample` and calls one `optimization_step` on a Tinker backend. SDK-specific `Datum`, renderer, sampling, and loss APIs stay in that backend because Tinker and tinker-cookbook versions evolve independently.

## Guardrails and limitations

Scripts run from a temporary working directory with a minimal environment, a wall-clock timeout, and a best-effort address-space limit. This is suitable for local integration testing, but it is not a hardened hostile-code sandbox: credentialed deployments should add container or VM isolation, network policy, filesystem policy, and an external secret broker.

## Efficiency

DuckDB uses a temporary local file and avoids network setup during local tests. Verification opens a short-lived read connection after the action process exits, preventing file-lock contention. A production adapter can batch task loading and Tinker sampling while retaining the same episode and reward contracts.
