# ELT-Bench-RL

A lightweight RL environment and verification rig for testing ELT agents. Built to safely run, sandbox, and score data transformation code using verifiable rewards.

## Repo Layout

- `environment.py` — Gym-style environment loop (handles resets, state tracking, and step execution).
- `verifier.py` — DuckDB backend checking outputs against golden datasets, plus anti-hacking data shuffling.
- `guardrails.py` — Subprocess execution sandbox with resource bounds and hard timeouts.
- `test_stress.py` — Core test suite covering timeout traps, false positives, and perturbation.
- `test_e2e_rl.py` — Full E2E training loop simulation checking multi-episode reward progression.