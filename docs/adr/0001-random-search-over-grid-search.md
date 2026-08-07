# 1. Random Search Over Grid Search for AutoML Optimization

Date: 2026-08-07

## Status

Accepted

## Context

As RagBench evolves into an optimization engine, it must search a hyperparameter space (chunk size, overlap, top_k, models, retrieval strategies) to find optimal configurations. Grid Search explores the Cartesian product of all possible parameter combinations.

## Decision

We will use **Random Search** (implemented in `RandomOptimizer`) over Grid Search as our default hyperparameter search strategy for V2. 

## Consequences

- **Pros:** Avoids combinatorial explosion. Grid searching 3 chunk sizes, 4 embeddings, 4 generators, 4 rerankers, 4 top_ks, and 3 overlaps results in 2304 experiments, which is computationally unfeasible. Random search explores the space much more efficiently, often yielding near-optimal results with an order of magnitude fewer trials.
- **Cons:** Does not guarantee finding the absolute global optimum within a finite set. May require integration with Bayesian Optimization (e.g., Optuna) in future iterations for more intelligent search space traversal.
