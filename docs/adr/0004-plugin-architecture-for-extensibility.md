# 4. Plugin Architecture for Extensibility

Date: 2026-08-07

## Status

Accepted

## Context

As the platform scales to support new vector stores, embedding providers, or proprietary evaluation logic, modifying the core orchestration code for each addition violates the Open/Closed Principle. 

## Decision

We will implement a plugin discovery mechanism. Users can drop implementations of our core interfaces (e.g., `CustomQdrantRetriever`) into a `plugins/` directory. The application will automatically discover and register these components into the `SearchSpace` dimensions.

## Consequences

- **Pros:** Maximum extensibility. Community contributions or private enterprise integrations can be added without modifying the core RagBench repository. 
- **Cons:** Requires a robust registry and dynamic import system, which can complicate debugging if plugins fail to load silently.
