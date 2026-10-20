"""Regression tests for the RAGOS benchmark methodology fixes.

Covers:
- experiment dedup that ignores the unused generation dimension
- exact-token (no-substring) evidence recall / extractive groundedness
- shared winner-selection score for random search and RAGOS
- SLA feasibility helper used identically for both strategies
- full-set random-mean aggregation (previously crashed the summary table)
- ParetoCalculator rejecting runs with missing objective metrics
"""
import copy
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.services.optimization.pareto import ParetoCalculator
from benchmark.run_benchmark import (
    OBJECTIVES,
    SELECTION_WEIGHTS,
    ChunkItem,
    aggregate_mean_metrics,
    evaluate_evidence_recall,
    evaluate_extractive_groundedness,
    experiment_key,
    meets_sla,
    selection_score,
)


# ── Experiment dedup ─────────────────────────────────────────────────────────

def test_experiment_key_ignores_generation_model():
    """The generator is never invoked by the benchmark, so configs differing
    only by generation model are the same experiment."""
    a = {
        "chunking": {"chunk_size": 256, "overlap": 32, "strategy": "fixed"},
        "retrieval": {"strategy": "dense", "top_k": 3},
        "generation": {"model_name": "phi3"},
    }
    b = copy.deepcopy(a)
    b["generation"]["model_name"] = "gpt-4o"
    assert experiment_key(a) == experiment_key(b)


def test_experiment_key_distinguishes_effective_dimensions():
    a = {"chunking": {"chunk_size": 256}, "generation": {"model_name": "phi3"}}
    b = {"chunking": {"chunk_size": 512}, "generation": {"model_name": "phi3"}}
    assert experiment_key(a) != experiment_key(b)


def test_experiment_key_includes_embedding_model():
    """Embedding model IS executed by the benchmark and must stay in the key."""
    a = {"embedding": {"model_name": "all-MiniLM-L6-v2"}}
    b = {"embedding": {"model_name": "BAAI/bge-small-en-v1.5"}}
    assert experiment_key(a) != experiment_key(b)


# ── Exact-token metrics ──────────────────────────────────────────────────────

def test_evidence_recall_rejects_substring_false_positives():
    ctx = [ChunkItem("c0", "topic_0", "The comparison marketplace sells fresh produce.")]
    # "paris" is a substring of "comparison" but not an exact token
    assert evaluate_evidence_recall(ctx, "paris france") == 0.0


def test_evidence_recall_counts_exact_tokens():
    ctx = [ChunkItem("c0", "topic_0", "Paris is the capital of France.")]
    assert evaluate_evidence_recall(ctx, "paris france") == 1.0


def test_evidence_recall_empty_context_is_zero():
    assert evaluate_evidence_recall([], "paris france") == 0.0


def test_extractive_groundedness_rejects_substring_false_positives():
    ctx = [ChunkItem("c0", "topic_0", "A start menu opens applications.")]
    # "art" is a substring of "start" but not its own token
    assert evaluate_extractive_groundedness("art gallery", ctx) == 0.0


# ── Shared selection & feasibility ───────────────────────────────────────────

def _run(prec, recall, lat, cost, label):
    return {
        "label": label,
        "metrics": {
            "context_precision": prec,
            "evidence_recall": recall,
            "query_context_similarity": 0.5,
            "retrieval_latency_ms": lat,
            "cost_per_1k": cost,
            "extractive_groundedness": 0.9,
            "rouge_l": 0.4,
            "avg_prompt_tokens": 100.0,
        },
    }


def test_selection_score_is_weighted_precision_and_recall():
    r = _run(0.8, 0.5, 5.0, 0.01, "x")
    expected = 0.8 * SELECTION_WEIGHTS["context_precision"] + 0.5 * SELECTION_WEIGHTS["evidence_recall"]
    assert selection_score(r) == pytest.approx(expected)


def test_selection_score_ignores_rouge():
    """ROUGE must not influence winner selection."""
    a = _run(0.5, 0.5, 5.0, 0.01, "a")
    b = copy.deepcopy(a)
    b["metrics"]["rouge_l"] = 0.99
    b["label"] = "b"
    assert selection_score(a) == selection_score(b)


def test_meets_sla_applies_both_thresholds():
    ok = _run(0.5, 0.5, 19.9, 0.079, "ok")
    assert meets_sla(ok, 20.0, 0.080) is True
    slow = _run(0.5, 0.5, 20.1, 0.079, "slow")
    assert meets_sla(slow, 20.0, 0.080) is False
    pricey = _run(0.5, 0.5, 5.0, 0.081, "pricey")
    assert meets_sla(pricey, 20.0, 0.080) is False


# ── Mean aggregation (summary-table crash regression) ────────────────────────

def test_aggregate_mean_metrics_covers_every_reported_metric():
    """Regression: the summary table reads extractive_groundedness from every
    row; the random-search mean previously omitted it and crashed with KeyError."""
    runs = [_run(0.5, 0.5, 8.0, 0.02, "r1"), _run(0.7, 0.6, 6.0, 0.03, "r2")]
    mean = aggregate_mean_metrics(runs)
    reported_keys = set(runs[0]["metrics"].keys())
    assert reported_keys <= set(mean.keys())
    assert mean["extractive_groundedness"] == pytest.approx(0.9)
    assert mean["context_precision"] == pytest.approx(0.6)


def test_aggregate_mean_metrics_rejects_empty_list():
    with pytest.raises(ValueError):
        aggregate_mean_metrics([])


# ── Pareto missing-metric safety ─────────────────────────────────────────────

def test_pareto_frontier_rejects_missing_metrics():
    a = _run(0.9, 0.9, 9.0, 0.05, "a")
    b = _run(0.5, 0.5, 5.0, 0.01, "b")
    del b["metrics"]["evidence_recall"]
    with pytest.raises(ValueError, match="evidence_recall"):
        ParetoCalculator.get_pareto_frontier([a, b], OBJECTIVES)


def test_pareto_frontier_keeps_non_dominated_trades_off():
    a = _run(0.9, 0.9, 9.0, 0.05, "a")   # best quality
    b = _run(0.5, 0.5, 5.0, 0.01, "b")   # fastest & cheapest
    frontier = ParetoCalculator.get_pareto_frontier([a, b], OBJECTIVES)
    assert {r["label"] for r in frontier} == {"a", "b"}


def test_pareto_frontier_drops_dominated_run():
    a = _run(0.9, 0.9, 9.0, 0.05, "a")
    b = _run(0.5, 0.5, 12.0, 0.09, "b")  # worse on every objective
    frontier = ParetoCalculator.get_pareto_frontier([a, b], OBJECTIVES)
    assert {r["label"] for r in frontier} == {"a"}
