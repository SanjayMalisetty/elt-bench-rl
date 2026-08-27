# ELT-Bench-RL: Implementation Design & Architecture

## 1. ELT Task & Environment Analysis
- **What's going on here:** We ingest raw data, stage it, and run transformation code to build out the target analytical model.
- **State management:** Each episode gets its own clean slate! A new and isolated DuckDB database file spins up on disk during `reset()`, gets modified by the agent's code during `step()`, and is completely wiped away when it wraps up.

## 2. Reward Function Design (RLVR)
- **The reward setup:** We use a simple, clear outcome-based reward (binary $1.0$ for success, $0.0$ for failure).
- **How we check it:** Instead of messy string matching, the verifier does a proper relational diff using DuckDB and Pandas. It checks everything—row counts, column types, and aggregated values—against the golden dataset.

## 3. Anti-Reward Hacking & Security Guardrails
- **No cheating allowed:** To stop agents from hardcoding static answers, `ELTVerifier.perturb_source_data()` actively shuffles rows and keys on every single reset!
- **Safe sandboxing:** Agent code runs inside a restricted subprocess wrapper with strict wall-clock timeouts so infinite loops get caught immediately.

## 4. Training Efficiency
- **Fast execution:** Everything runs on lightweight, serverless DuckDB instances backed by temporary local directories, keeping environment resets and reward calculations lightning fast :)