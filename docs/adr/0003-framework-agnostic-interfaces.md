# 3. Framework-Agnostic Interfaces

Date: 2026-08-07

## Status

Accepted

## Context

The RAG ecosystem moves rapidly. Tooling that is deeply coupled to a specific framework (e.g., LangChain, LlamaIndex, ChromaDB) risks obsolescence when newer paradigms or vendors emerge. RagBench aims to be an enduring orchestration layer for RAG experiments.

## Decision

We define formal abstract base classes (ABCs) for every component in the pipeline: `BaseRetriever`, `BaseEmbedder`, `BaseGenerator`, `BaseEvaluator`, `BaseOptimizer`, and `BaseExecutor`. All orchestration logic must program against these interfaces, rather than concrete implementations.

## Consequences

- **Pros:** Prevents vendor lock-in. We can swap ChromaDB for Qdrant, or LiteLLM for a direct API call, without rewriting the `ExperimentRunner`. Protects the core intellectual property (the optimization logic) from the churn of the underlying infrastructure.
- **Cons:** Requires writing adapter classes for any new dependency, adding boilerplate.
