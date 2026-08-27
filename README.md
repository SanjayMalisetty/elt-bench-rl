# ELT-Bench-RL

An RL environment and verification framework for executing and evaluating ELT (Extract, Load, Transform) tasks using reinforcement learning.

## Architecture

- **`environment.py`**: Handles the Gym environment, state management, and resets.
- **`verifier.py`**: Handles DuckDB state checks and relational verification.
- **`guardrails.py`**: Handles timeouts, memory limits, and sandboxing.
- **`test_local.py`**: Credential-free local integration test suite.
