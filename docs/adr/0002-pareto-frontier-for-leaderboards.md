# 2. N-Dimensional Pareto Frontier for Leaderboards

Date: 2026-08-07

## Status

Accepted

## Context

In standard benchmarking platforms, experiments are typically ranked using a single weighted objective score (e.g., `0.7 * Quality + 0.3 * Speed`). However, this collapses meaningful trade-offs, making it impossible to see the "fastest," "cheapest," and "highest quality" options simultaneously. 

## Decision

We will compute and return the **Pareto Frontier** dynamically based on user-selected objectives (e.g., Latency vs. Faithfulness vs. Cost). The leaderboard will expose the non-dominated set of configurations.

## Consequences

- **Pros:** Empowers ML engineers to make nuanced decisions based on their specific production constraints (e.g., finding the best accuracy for a strict latency budget). Avoids artificial ranking bias.
- **Cons:** Slightly higher computational overhead when resolving the leaderboard for thousands of runs, though negligible for standard experiment sizes.
