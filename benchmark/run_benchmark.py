"""
RAGOS Empirical Benchmark: Baseline vs Random Search vs RAGOS Optimizer
======================================================================
Empirically benchmarks RAG pipeline configurations on a controlled technical QA corpus:
  1. Default Baseline    – Standard unoptimized pipeline (chunk=512, dense, top_k=3, no-rerank)
  2. Random Search       – Unguided exploration across search dimensions (12 trials without constraint repair)
  3. RAGOS Optimizer     – Constraint-screened search via RandomOptimizer + ConstraintSolver + ParetoCalculator

Scope & Measurement Attribution:
--------------------------------
- Retrieval-Stage Latency: Measured wall-clock time (ms) for index query, dense/sparse/hybrid fusion,
  and CrossEncoder reranking. (Generation latency is excluded as it depends on external LLM provider round-trips).
- Context Token Footprint & Projected Input Cost: Calculated from prompt tokens of retrieved context chunks
  at standard input rates ($0.150 per 1M input tokens).
- IR Precision@K: Fraction of retrieved chunks originating from the ground-truth source document.
- Evidence Recall: Proportion of ground-truth content tokens retrieved in the context.
- Extractive Groundedness: Lexical overlap between an extractive diagnostic string and its source context.
  It is not an LLM faithfulness metric.
- Query-Context Similarity: Semantic cosine similarity between the query and the extractive context.
  It is not a generated-answer relevancy metric.

Outputs:
  benchmark/results/benchmark_results.json
  benchmark/results/summary_table.txt
"""

import copy, json, os, random, re, sys, time
from typing import List, Dict, Any, Tuple, Set

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rouge_score import rouge_scorer as rs_module

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.services.optimization.constraints import ConstraintSolver
from backend.services.optimization.pareto import ParetoCalculator
from backend.services.optimization.optimizer import RandomOptimizer
from backend.services.optimization.dimensions import get_default_dimensions

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH   = os.path.join(REPO_ROOT, "data", "qa_pairs.json")
RESULTS_DIR = os.path.join(REPO_ROOT, "benchmark", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Production SLA constraints: sub-20ms retrieval latency and <= $0.080 / 1k queries
SLA_CONSTRAINTS = ["latency <= 20ms", "cost <= 0.080"]

NUM_CANDIDATES = 12
ALPHA_HYBRID = 0.6  # weight for dense component in hybrid retrieval

# Optimization objectives for ParetoCalculator
OBJECTIVES: Dict[str, str] = {
    "context_precision":    "maximize",
    "evidence_recall":      "maximize",
    "query_context_similarity": "maximize",
    "retrieval_latency_ms": "minimize",
    "cost_per_1k":          "minimize",
}

DEFAULT_BASE_CONFIG: Dict[str, Any] = {
    "chunking":   {"chunk_size": 512, "overlap": 32, "strategy": "fixed"},
    "retrieval":  {"strategy": "dense", "top_k": 3},
    "embedding":  {"model_name": "all-MiniLM-L6-v2"},
    "reranker":   {"model_name": None},
    "generation": {"model_name": "phi3"},
}

# ── Technical Knowledge-Base Corpus ───────────────────────────────────────────
TOPIC_PASSAGES = [
    # 1. France / Paris
    ("topic_0", "Paris is the capital and most populous city of France, situated along the Seine River in the north-central region. "
     "Dating back over two millennia, Paris emerged as a leading European centre of commerce, international diplomacy, "
     "finance, science, and the arts. The city serves as the administrative seat of the French government and hosts major "
     "international organizations including UNESCO. Its metropolitan economy represents roughly one-third of France's GDP, "
     "making it an indispensable hub in European transit, industry, and higher education."),

    # 2. Quantum Computing
    ("topic_1", "Quantum computing uses quantum bits or qubits to perform calculations at speeds exponentially faster than classical computers "
     "for certain tasks. Unlike classical binary transistors that evaluate discrete bits as zero or one, quantum processors exploit "
     "superposition and quantum entanglement. This enables algorithms like Shor's and Grover's to evaluate massive search spaces "
     "simultaneously, unlocking breakthroughs in cryptographic factorization, molecular material simulation, and complex combinatorial optimization."),

    # 3. Hamlet / Shakespeare
    ("topic_2", "William Shakespeare wrote the play Hamlet, formally titled The Tragedy of Hamlet, Prince of Denmark, between 1599 and 1601. "
     "Set in the Kingdom of Denmark, the tragedy recounts Prince Hamlet's quest for vengeance against his uncle Claudius, who murdered "
     "the King to seize the throne and marry Hamlet's mother. It stands as Shakespeare's longest play and among the most celebrated works "
     "in English dramatic literature, deeply exploring themes of grief, betrayal, existential doubt, and madness."),

    # 4. RAG
    ("topic_3", "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by retrieving relevant information from a "
     "knowledge base and providing it as context to generate more accurate and grounded answers. Instead of relying strictly on parametric "
     "memory encoded during pre-training, a RAG system fetches non-parametric evidence from internal corpora, databases, or documentation. "
     "This architecture dramatically mitigates factual hallucinations, facilitates dynamic knowledge updates without retraining, and provides auditable citations."),

    # 5. Dense Embeddings
    ("topic_4", "Dense embeddings are low-dimensional, dense vectors representing the semantic meaning of text, typically generated by neural networks like BERT. "
     "In contrast to high-dimensional, sparse representations such as term-frequency vectors, dense embeddings map syntactically distinct phrases "
     "with analogous conceptual meanings to nearby coordinates in vector space. Dense encoders optimize contrastive loss functions to ensure "
     "semantic proximity translates directly into high cosine similarity scores during approximate nearest neighbor search."),

    # 6. BM25
    ("topic_5", "BM25 stands for Best Matching 25. It is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document. "
     "Building upon classical TF-IDF, BM25 introduces term-frequency saturation via parameter k1, ensuring that repeated occurrences of a keyword yield diminishing "
     "marginal score increases, alongside document length normalization via parameter b. BM25 remains the primary industry benchmark for lexical keyword search."),

    # 7. Hybrid Search
    ("topic_6", "Hybrid search combines multiple search strategies, such as dense vector similarity and sparse keyword matching (like BM25), to improve retrieval accuracy by "
     "capturing both semantic meaning and exact term matches. While dense retrieval excels at semantic paraphrasing and conceptual intent, it frequently struggles "
     "with exact part numbers, specialized acronyms, or rare domain entities. Hybrid pipelines merge normalized scores via linear fusion or Reciprocal Rank Fusion."),

    # 8. LLM-as-a-judge
    ("topic_7", "LLM-as-a-judge is an evaluation method where a Large Language Model is used to assess the quality, accuracy, or faithfulness of text generated by another system. "
     "Rather than relying strictly on lexical metrics like BLEU or ROUGE, a judge model evaluates complex attributes including contextual groundedness, adherence to "
     "system instructions, and reasoning soundness. To prevent evaluation bias, implementations often employ structured scoring rubrics and swap pairwise candidate order."),

    # 9. Hallucination
    ("topic_8", "Hallucination refers to instances where an LLM generates information that is factually incorrect, nonsensical, or not grounded in the provided source material. "
     "Because autoregressive language models maximize next-token probability rather than factual truth, they can construct plausible-sounding yet completely fabricated claims. "
     "Mitigating hallucinations in production requires rigorous retrieval grounding, constrained decoding, and automated faithfulness verification guardrails."),

    # 10. Cosine Similarity
    ("topic_9", "Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space, commonly used to determine the semantic similarity between two text embeddings. "
     "The calculation divides the dot product of two vectors by the product of their Euclidean magnitudes. Because it evaluates angular orientation rather than vector length, "
     "cosine similarity is invariant to text scale, making it ideal for comparing queries against embedded document passages."),

    # 11. Attention Mechanism
    ("topic_10", "The attention mechanism allows the model to weigh the importance of different words in an input sequence relative to a specific word, capturing context and dependencies regardless of distance. "
     "Introduced in the Transformer architecture, scaled dot-product multi-head attention computes Query, Key, and Value projections for each token. "
     "This eliminates sequential recurrent constraints, allowing parallelized processing of long-range syntactic and semantic relationships across sequences."),

    # 12. Vector Database
    ("topic_11", "A vector database is specialized software designed to store, manage, and index high-dimensional vector embeddings, enabling efficient similarity search. "
     "Conventional relational databases struggle with high-dimensional Euclidean or cosine distance operations. Vector databases implement specialized indexing "
     "algorithms such as Hierarchical Navigable Small World (HNSW) graphs and Inverted File with Product Quantization (IVF-PQ) to deliver sub-millisecond nearest neighbor search."),

    # 13. ROUGE Metric
    ("topic_12", "ROUGE measures the quality of a generated summary by comparing it to reference summaries, calculating the overlap of n-grams, word sequences, and word pairs. "
     "Variants include ROUGE-1 for unigram overlap, ROUGE-2 for bigram overlap, and ROUGE-L which evaluates the Longest Common Subsequence between hypothesis and ground truth. "
     "ROUGE scores provide automated, reproducible evaluations of lexical recall, though they do not capture semantic equivalencies."),

    # 14. Reciprocal Rank Fusion
    ("topic_13", "RRF combines document rankings from multiple retrieval systems by assigning a score based on the inverse of the document's rank in each system, placing highly ranked items from multiple lists near the top. "
     "The algorithm computes a fused score for each document as the sum of 1 / (k + rank_i), where k is a smoothing constant typically set to 60. "
     "Because RRF relies purely on relative ordinal ranking rather than raw scores, it cleanly circumvents the score normalization and scale calibration issues of hybrid retrievers."),

    # 15. Summarization
    ("topic_14", "Extractive summarization selects and concatenates key sentences directly from the source text, while abstractive summarization generates new sentences to convey the main ideas. "
     "Extractive systems prioritize factual fidelity and low compute cost by ranking and lifting salient sentences verbatim. Abstractive summarization uses generative sequence-to-sequence models "
     "to compress, rephrase, and synthesize ideas coherently, though it introduces risks of hallucinating information not present in the original document."),

    # 16. Zero-Shot Learning
    ("topic_15", "Zero-shot learning is the ability of a model to correctly perform a task or recognize a concept it has never seen during training, relying solely on provided descriptions or prompts. "
     "In foundation models, zero-shot capabilities emerge from web-scale pretraining across diverse natural language corpora, allowing models to generalize to novel classification, "
     "translation, or reasoning prompts without task-specific supervised fine-tuning or parameter updates."),

    # 17. Chunking in RAG
    ("topic_16", "Chunking breaks large documents into smaller, manageable pieces so that they fit within the embedding model's token limit and allow for more precise retrieval of relevant information. "
     "Oversized chunks dilute relevant facts with irrelevant peripheral text, degrading embedding specificity and ballooning LLM prompt token costs. "
     "Conversely, undersized chunks risk fragmenting concepts across boundaries. Choosing the optimal chunk size balances contextual completeness against retrieval precision."),

    # 18. Embedding Models
    ("topic_17", "An embedding model converts text into numerical vectors that capture semantic relationships, allowing computers to process and compare text based on meaning. "
     "By passing tokens through deep contextual encoders, embedding models generate fixed-width vector spaces where geometric distance reflects conceptual similarity. "
     "These embeddings power search engines, clustering algorithms, recommendation systems, and the retrieval phase of modern RAG architectures."),

    # 19. Context Window
    ("topic_18", "The context window is the maximum number of tokens (words or subwords) that an LLM can process in a single input, limiting how much prompt and context it can handle at once. "
     "While recent architectures have expanded context windows to hundreds of thousands of tokens, attention compute costs scale quadratically or require specialized attention approximations. "
     "Moreover, models exhibit 'lost-in-the-middle' retrieval degradation when relevant facts are buried deep within massive context inputs."),

    # 20. Temperature in Text Generation
    ("topic_19", "Temperature controls the randomness of the LLM's predictions. A higher temperature makes the output more diverse and creative, while a lower temperature makes it more deterministic and focused. "
     "Technically, temperature acts as a divisor applied to the output logits prior to the softmax normalization layer. A temperature approaching zero forces greedy argmax selection of the most probable token, "
     "which is optimal for factual QA, whereas higher values flatten the probability distribution to encourage creative variance."),
]

DISTRACTOR_PASSAGES = [
    ("dist_0", "Relational database normalization organizes tables to minimize data redundancy and dependency across schema relations using normal forms from 1NF to BCNF."),
    ("dist_1", "Operating system virtual memory uses hardware paging tables and Translation Lookaside Buffers (TLB) to map process address spaces to physical RAM pages."),
    ("dist_2", "Public-key cryptography uses asymmetric key pairs, enabling secure digital signatures and encrypted channel negotiation via RSA or elliptic curve algorithms."),
    ("dist_3", "Distributed consensus protocols like Raft ensure fault-tolerant state machine replication across unreliable network nodes via leader election and log replication."),
    ("dist_4", "Compilers utilize static single assignment (SSA) intermediate representations to perform dead code elimination, constant folding, and register allocation."),
    ("dist_5", "Transmission Control Protocol (TCP) guarantees ordered, error-checked packet delivery using congestion control, three-way handshakes, and sliding windows."),
    ("dist_6", "Linux cgroups and namespaces provide kernel resource isolation mechanisms that underpin modern containerization runtimes like Docker and containerd."),
    ("dist_7", "B-trees and Log-Structured Merge (LSM) trees represent two primary indexing structures balancing read latency against sequential write amplification in databases."),
    ("dist_8", "Cache coherence protocols like MESI maintain uniform memory state across multi-core processors sharing an interconnect bus via invalidate broadcasts."),
    ("dist_9", "Garbage collection algorithms track object graph reachability via mark-and-sweep or generational tracing to reclaim heap memory without explicit developer deallocation."),
]

# ── Model Cache ───────────────────────────────────────────────────────────────
_embedding_models: Dict[str, SentenceTransformer] = {}
_cross_encoders: Dict[str, CrossEncoder] = {}

def get_embedding_model(name: str) -> SentenceTransformer:
    if name not in _embedding_models:
        print(f"  [Loading embedding model: {name}]")
        _embedding_models[name] = SentenceTransformer(name)
    return _embedding_models[name]

def get_cross_encoder(name: str) -> CrossEncoder:
    if name not in _cross_encoders:
        print(f"  [Loading cross-encoder reranker: {name}]")
        _cross_encoders[name] = CrossEncoder(name)
    return _cross_encoders[name]

def warmup_models():
    """Warms up PyTorch and models so cold-start initialization doesn't distort latency timing."""
    print(">>> Warming up embedding and cross-encoder models...")
    _ = get_embedding_model("all-MiniLM-L6-v2").encode(["Warmup text."], convert_to_tensor=False)
    _ = get_embedding_model("BAAI/bge-small-en-v1.5").encode(["Warmup text."], convert_to_tensor=False)
    _ = get_cross_encoder("cross-encoder/ms-marco-MiniLM-L-6-v2").predict([["Warmup query", "Warmup passage text."]])
    print("    Warmup complete.\n")


# ── Chunking with Provenance ──────────────────────────────────────────────────
class ChunkItem:
    def __init__(self, chunk_id: str, doc_id: str, text: str):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.text = text

def build_chunked_corpus(corpus: List[Tuple[str, str]], chunk_size: int, overlap: int = 32) -> List[ChunkItem]:
    all_chunks = []
    chunk_counter = 0
    step = max(32, chunk_size - overlap)
    for doc_id, text in corpus:
        if len(text) <= chunk_size:
            all_chunks.append(ChunkItem(f"c_{chunk_counter}", doc_id, text.strip()))
            chunk_counter += 1
            continue
        for i in range(0, len(text), step):
            sub = text[i : i + chunk_size].strip()
            if len(sub) > 20:
                all_chunks.append(ChunkItem(f"c_{chunk_counter}", doc_id, sub))
                chunk_counter += 1
    return all_chunks

# ── Retrievers ────────────────────────────────────────────────────────────────
def dense_retrieve(query: str, chunks: List[ChunkItem], embs: np.ndarray,
                   model: SentenceTransformer, top_k: int) -> List[ChunkItem]:
    q_emb = model.encode(query, convert_to_tensor=False)
    sims = (embs @ q_emb) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb) + 1e-9)
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_idx]

def sparse_retrieve(query: str, chunks: List[ChunkItem], top_k: int) -> List[ChunkItem]:
    q_words = set(re.findall(r"\w+", query.lower()))
    scores = []
    for c in chunks:
        c_words = set(re.findall(r"\w+", c.text.lower()))
        overlap = len(q_words & c_words) / (len(q_words | c_words) + 1e-9)
        scores.append(overlap)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_idx]

def hybrid_retrieve(query: str, chunks: List[ChunkItem], embs: np.ndarray,
                    model: SentenceTransformer, top_k: int) -> List[ChunkItem]:
    q_emb = model.encode(query, convert_to_tensor=False)
    dense_sims = (embs @ q_emb) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb) + 1e-9)
    d_min, d_max = dense_sims.min(), dense_sims.max()
    d_norm = (dense_sims - d_min) / (d_max - d_min + 1e-9)

    q_words = set(re.findall(r"\w+", query.lower()))
    sparse_scores = np.array([
        len(q_words & set(re.findall(r"\w+", c.text.lower()))) / (len(q_words | set(re.findall(r"\w+", c.text.lower()))) + 1e-9)
        for c in chunks
    ])
    s_min, s_max = sparse_scores.min(), sparse_scores.max()
    s_norm = (sparse_scores - s_min) / (s_max - s_min + 1e-9)

    combined = ALPHA_HYBRID * d_norm + (1.0 - ALPHA_HYBRID) * s_norm
    top_idx = np.argsort(combined)[::-1][:top_k]
    return [chunks[i] for i in top_idx]

def cross_encoder_rerank(query: str, candidates: List[ChunkItem],
                         reranker_name: str, top_k: int) -> List[ChunkItem]:
    if not candidates:
        return candidates
    model = get_cross_encoder(reranker_name)
    pairs = [[query, c.text] for c in candidates]
    scores = model.predict(pairs)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [candidates[i] for i in top_idx]

# ── Grounded Metrics ──────────────────────────────────────────────────────────
_rouge_scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)

_STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "and",
    "but", "or", "as", "if", "while", "of", "that", "this", "it", "they"
}

def extract_content_tokens(text: str) -> Set[str]:
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}

def evaluate_context_precision(retrieved: List[ChunkItem], target_doc_id: str) -> float:
    """IR Precision@K: Fraction of retrieved chunks coming from the ground-truth document."""
    if not retrieved:
        return 0.0
    hits = sum(1 for c in retrieved if c.doc_id == target_doc_id)
    return round(hits / len(retrieved), 4)

def evaluate_evidence_recall(retrieved: List[ChunkItem], ground_truth: str) -> float:
    """Evidence Recall: Fraction of key ground-truth content words present as
    whole tokens in the retrieved context (exact-token match, no substrings)."""
    gt_tokens = extract_content_tokens(ground_truth)
    if not gt_tokens:
        return 1.0
    retrieved_tokens: Set[str] = set()
    for c in retrieved:
        retrieved_tokens |= extract_content_tokens(c.text)
    found = sum(1 for token in gt_tokens if token in retrieved_tokens)
    return round(found / len(gt_tokens), 4)

def evaluate_extractive_groundedness(answer: str, retrieved: List[ChunkItem]) -> float:
    """Diagnostic lexical overlap for an extractive answer, not LLM faithfulness.
    Uses whole-token matching against the retrieved context (no substrings)."""
    ans_tokens = extract_content_tokens(answer)
    if not ans_tokens:
        return 1.0
    retrieved_tokens: Set[str] = set()
    for c in retrieved:
        retrieved_tokens |= extract_content_tokens(c.text)
    grounded = sum(1 for token in ans_tokens if token in retrieved_tokens)
    return round(grounded / len(ans_tokens), 4)

def evaluate_query_context_similarity(query: str, answer: str, model: SentenceTransformer) -> float:
    """Semantic similarity between the query and extractive retrieved context."""
    if not answer.strip():
        return 0.0
    q_emb = model.encode(query, convert_to_tensor=False)
    a_emb = model.encode(answer, convert_to_tensor=False)
    sim = np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-9)
    return round(float(sim), 4)

def evaluate_rouge_l(answer: str, ground_truth: str) -> float:
    return round(_rouge_scorer.score(ground_truth, answer)["rougeL"].fmeasure, 4)

# ── Experiment Identity & Shared Selection ───────────────────────────────────
def experiment_key(config: Dict[str, Any]) -> str:
    """Dedup fingerprint over the dimensions this benchmark actually exercises.

    `generation` is excluded: the benchmark synthesizes extractive answers and
    never invokes an LLM generator, so candidates differing only by generation
    model are the same experiment and must not consume extra evaluation budget
    or inflate the unique-experiment count.
    """
    effective = {k: v for k, v in config.items() if k != "generation"}
    return json.dumps(effective, sort_keys=True, separators=(",", ":"))

# Shared, pre-declared winner-selection criterion. Both the random-search
# winner and the RAGOS winner are chosen with this score so the two strategies
# are compared on identical terms. ROUGE-L is reported but not used to select.
SELECTION_WEIGHTS: Dict[str, float] = {
    "context_precision": 0.6,
    "evidence_recall":   0.4,
}

def selection_score(run: Dict[str, Any]) -> float:
    """Shared winner-selection score: weighted precision + evidence recall."""
    return sum(run["metrics"][m] * w for m, w in SELECTION_WEIGHTS.items())

def meets_sla(run: Dict[str, Any], latency_limit_ms: float, cost_limit_1k: float) -> bool:
    m = run["metrics"]
    return m["retrieval_latency_ms"] <= latency_limit_ms and m["cost_per_1k"] <= cost_limit_1k

def aggregate_mean_metrics(runs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Mean of EVERY metric key present on the runs.

    Aggregating a hand-picked subset previously omitted `extractive_groundedness`,
    which the summary table reads from every row, so fresh benchmark runs
    crashed with a KeyError.
    """
    if not runs:
        raise ValueError("Cannot aggregate an empty run list.")
    keys = sorted({k for r in runs for k in r["metrics"]})
    return {
        m: round(sum(r["metrics"][m] for r in runs) / len(runs),
                 1 if m == "avg_prompt_tokens" else 4)
        for m in keys
    }

# ── Pipeline Runner ───────────────────────────────────────────────────────────
def run_pipeline(config_dict: Dict[str, Any],
                 qa_pairs: List[Dict],
                 corpus: List[Tuple[str, str]]) -> Dict[str, Any]:
    chunk_size = config_dict["chunking"]["chunk_size"]
    overlap = config_dict["chunking"].get("overlap", 32)
    strat = config_dict["retrieval"]["strategy"]
    k = config_dict["retrieval"]["top_k"]
    reranker_name = config_dict.get("reranker", {}).get("model_name")
    emb_model_name = config_dict.get("embedding", {}).get("model_name", "all-MiniLM-L6-v2")

    model = get_embedding_model(emb_model_name)
    chunks = build_chunked_corpus(corpus, chunk_size, overlap)
    chunk_texts = [c.text for c in chunks]
    embs = model.encode(chunk_texts, convert_to_tensor=False, show_progress_bar=False)

    rows = []
    for idx, item in enumerate(qa_pairs):
        q, gt = item["query"], item["ground_truth"]
        target_doc_id = f"topic_{idx}"

        # Measure RETRIEVAL-STAGE latency (search + rerank)
        t_start = time.perf_counter()
        initial_fetch = k * 2 if reranker_name else k

        if strat == "dense":
            cands = dense_retrieve(q, chunks, embs, model, initial_fetch)
        elif strat == "sparse":
            cands = sparse_retrieve(q, chunks, initial_fetch)
        else:  # hybrid
            cands = hybrid_retrieve(q, chunks, embs, model, initial_fetch)

        if reranker_name:
            final_chunks = cross_encoder_rerank(q, cands, reranker_name, k)
        else:
            final_chunks = cands[:k]

        retrieval_latency = (time.perf_counter() - t_start) * 1000  # ms

        # Extractive answer synthesis from top retrieved chunks
        answer = " ".join(c.text for c in final_chunks[:2])[:350]

        # Context prompt token footprint (~4 chars per token)
        prompt_tokens = sum(len(c.text) for c in final_chunks) // 4 + len(q) // 4
        # Standard input token pricing: $0.150 per 1M input tokens
        query_cost = prompt_tokens * (0.150 / 1_000_000)

        rows.append({
            "query":                q,
            "target_doc_id":        target_doc_id,
            "context_precision":    evaluate_context_precision(final_chunks, target_doc_id),
            "evidence_recall":      evaluate_evidence_recall(final_chunks, gt),
            "extractive_groundedness": evaluate_extractive_groundedness(answer, final_chunks),
            "query_context_similarity": evaluate_query_context_similarity(q, answer, model),
            "rouge_l":              evaluate_rouge_l(answer, gt),
            "retrieval_latency_ms": retrieval_latency,
            "prompt_tokens":        prompt_tokens,
            "cost_usd":             query_cost,
        })

    n = len(rows)
    agg = lambda key: round(sum(r[key] for r in rows) / n, 4)
    avg_prompt_tokens = round(sum(r["prompt_tokens"] for r in rows) / n, 1)
    avg_latency = round(sum(r["retrieval_latency_ms"] for r in rows) / n, 2)
    cost_per_1k = round((sum(r["cost_usd"] for r in rows) / n) * 1000, 4)

    return {
        "config": config_dict,
        "per_query": rows,
        "metrics": {
            "context_precision":    agg("context_precision"),
            "evidence_recall":      agg("evidence_recall"),
            "extractive_groundedness": agg("extractive_groundedness"),
            "query_context_similarity": agg("query_context_similarity"),
            "rouge_l":              agg("rouge_l"),
            "retrieval_latency_ms": avg_latency,
            "avg_prompt_tokens":    avg_prompt_tokens,
            "cost_per_1k":          cost_per_1k,
        },
    }

# ── Main Benchmark ────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print(" RAGOS Empirical Benchmark: Baseline vs Random Search vs RAGOS Optimizer")
    print("=" * 80)

    with open(DATA_PATH) as f:
        qa_pairs = json.load(f)

    corpus = list(TOPIC_PASSAGES[:len(qa_pairs)]) + DISTRACTOR_PASSAGES
    print(f"\nEvaluation Set : {len(qa_pairs)} QA pairs")
    print(f"Corpus Size    : {len(corpus)} technical passages")
    print(f"SLA Constraints: {SLA_CONSTRAINTS}")
    print(f"Seed           : {SEED}\n")

    warmup_models()

    # Shared feasibility thresholds — applied identically to the random-search
    # winner and the RAGOS winner so both strategies face the same SLA bar.
    solver = ConstraintSolver(SLA_CONSTRAINTS)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1: Default Baseline Configuration
    # ──────────────────────────────────────────────────────────────────────────
    print(">>> [Phase 1/3] Evaluating Default Baseline (chunk=512, dense, k=3, no-rerank) ...")
    def_run = run_pipeline(DEFAULT_BASE_CONFIG, qa_pairs, corpus)
    def_run["label"] = "Default Baseline"
    def_run["meets_sla"] = meets_sla(def_run, solver.latency_limit, solver.cost_limit)
    dm = def_run["metrics"]
    print(f"    Prec={dm['context_precision']:.4f}  EvidenceRecall={dm['evidence_recall']:.4f}  "
          f"Extractive={dm['extractive_groundedness']:.4f}  QueryCtx={dm['query_context_similarity']:.4f}  "
          f"Lat={dm['retrieval_latency_ms']:.1f}ms  Cost=${dm['cost_per_1k']:.4f}/1k\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 2: Unguided Random Search (Generated via RandomOptimizer with no constraints)
    # ──────────────────────────────────────────────────────────────────────────
    print(f">>> [Phase 2/3] Evaluating Unguided Random Search ({NUM_CANDIDATES} candidates) ...")
    optimizer = RandomOptimizer()
    
    # Generate unguided candidates (empty constraints)
    unguided_candidates = optimizer.generate_candidates(DEFAULT_BASE_CONFIG, n=NUM_CANDIDATES, constraints=[])
    random_runs = []
    seen_configs = set()

    for trial, cand in enumerate(unguided_candidates):
        cfg_key = experiment_key(cand)
        if cfg_key in seen_configs:
            continue
        seen_configs.add(cfg_key)

        r_name = "cross-enc" if cand.get("reranker", {}).get("model_name") else "none"
        print(f"  T{trial+1:02d}: chunk={cand['chunking']['chunk_size']:>4}  "
              f"strat={cand['retrieval']['strategy']:<6}  k={cand['retrieval']['top_k']:>2}  "
              f"rerank={r_name:<9}", end=" -> ")

        run_res = run_pipeline(cand, qa_pairs, corpus)
        run_res["label"] = f"Random-{trial+1}"
        rm = run_res["metrics"]
        print(f"prec={rm['context_precision']:.4f}  evidence_recall={rm['evidence_recall']:.4f}  "
              f"lat={rm['retrieval_latency_ms']:>5.1f}ms  cost=${rm['cost_per_1k']:.4f}/1k")
        random_runs.append(run_res)

    # Compute the Random-Search mean over the FULL per-run metric set.
    # Aggregating a hand-picked subset previously omitted `extractive_groundedness`,
    # which the summary table then read from every row — crashing fresh runs.
    if not random_runs:
        raise RuntimeError("Unguided random search produced no evaluable trials.")
    rand_mean_metrics = aggregate_mean_metrics(random_runs)

    rand_mean_run = {
        "label": "Random Search (Mean)",
        "config": {"note": f"Mean over {len(random_runs)} unguided trials"},
        "metrics": rand_mean_metrics,
    }

    # Best Random Trial selected with the SAME shared selection score and SLA
    # feasibility thresholds used for the RAGOS winner (previously this used
    # ROUGE-L alone, which made the random-vs-RAGOS comparison unfair).
    feasible_random = [r for r in random_runs if meets_sla(r, solver.latency_limit, solver.cost_limit)]
    if feasible_random:
        best_rand_run = max(feasible_random, key=selection_score)
        sla_note = "SLA-feasible"
    else:
        best_rand_run = max(random_runs, key=selection_score)
        sla_note = "no SLA-feasible random trial; best infeasible trial shown"
    best_rand_run = copy.deepcopy(best_rand_run)
    best_rand_run["label"] = "Random Search (Best)"
    best_rand_run["meets_sla"] = meets_sla(best_rand_run, solver.latency_limit, solver.cost_limit)
    brm = best_rand_run["metrics"]
    print(f"\n  Best Random Trial by shared selection score ({sla_note}; "
          f"{best_rand_run['config']['retrieval']['strategy']}, "
          f"rerank={bool(best_rand_run['config'].get('reranker', {}).get('model_name'))}): "
          f"prec={brm['context_precision']:.4f}  lat={brm['retrieval_latency_ms']:.1f}ms  "
          f"cost=${brm['cost_per_1k']:.4f}/1k\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 3: RAGOS Optimizer (Measured Constraint Screening + Pareto Pruning)
    # ──────────────────────────────────────────────────────────────────────────
    print(">>> [Phase 3/3] Executing RAGOS Optimizer (Measured Constraint Screening & Pareto Pruning) ...")

    # Generate an equal-budget candidate set; enforce measured constraints below.
    ragos_raw_candidates = optimizer.generate_candidates(DEFAULT_BASE_CONFIG, n=NUM_CANDIDATES, constraints=SLA_CONSTRAINTS)
    
    optimizer_evaluated = []
    seen_ragos = set()
    for cand in ragos_raw_candidates:
        cfg_key = experiment_key(cand)
        if cfg_key in seen_ragos:
            continue
        seen_ragos.add(cfg_key)

        res = run_pipeline(cand, qa_pairs, corpus)
        res["label"] = f"RAGOS-Candidate-{len(optimizer_evaluated)+1}"
        optimizer_evaluated.append(res)

    # Filter to candidates that satisfy SLA constraints
    valid_candidates = [r for r in optimizer_evaluated if meets_sla(r, solver.latency_limit, solver.cost_limit)]
    if not valid_candidates:
        raise RuntimeError(
            "No RAGOS candidate met the declared SLA; refusing to select an invalid configuration."
        )

    # Multi-objective Pareto Frontier calculation via backend ParetoCalculator
    frontier_candidates = ParetoCalculator.get_pareto_frontier(valid_candidates, OBJECTIVES)

    # Select the winner from the Pareto Frontier with the SAME shared
    # selection score and feasibility thresholds used for random search.
    opt_winner = max(frontier_candidates, key=selection_score)

    opt_run = copy.deepcopy(opt_winner)
    opt_run["label"] = "RAGOS-Optimized"
    opt_run["meets_sla"] = True
    om = opt_run["metrics"]

    print(f"  Optimizer Winner Config: {opt_run['config']['chunking']['chunk_size']} chunk, "
          f"{opt_run['config']['retrieval']['strategy']}, top_k={opt_run['config']['retrieval']['top_k']}, "
          f"reranker={opt_run['config'].get('reranker', {}).get('model_name')}")
    print(f"    Prec={om['context_precision']:.4f}  EvidenceRecall={om['evidence_recall']:.4f}  "
          f"Extractive={om['extractive_groundedness']:.4f}  QueryCtx={om['query_context_similarity']:.4f}  "
          f"Lat={om['retrieval_latency_ms']:.1f}ms  Cost=${om['cost_per_1k']:.4f}/1k\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 4: Full Pareto Frontier Validation Across All Strategies
    # ──────────────────────────────────────────────────────────────────────────
    comparison_runs = [def_run, best_rand_run, opt_run] + [
        r for r in random_runs if r["label"] != best_rand_run["label"]
    ]
    global_frontier = ParetoCalculator.get_pareto_frontier(comparison_runs, OBJECTIVES)
    frontier_labels = {r["label"] for r in global_frontier}

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 5: Structured Reporting
    # ──────────────────────────────────────────────────────────────────────────
    def delta(a: Dict, b: Dict, metric: str):
        va, vb = a["metrics"][metric], b["metrics"][metric]
        d = va - vb
        pct = d / (abs(vb) + 1e-9) * 100
        return round(d, 4), round(pct, 1)

    hdr = (f"\n{'Pipeline Strategy':<22} {'Prec':>6} {'EvRecall':>8} {'Extract':>7} "
           f"{'QryCtx':>7} {'RougeL':>7} {'Lat(ms)':>8} {'Cost/1k':>9} {'SLA':>5} {'Pareto':>7}")
    sep = "=" * len(hdr)
    lines = [hdr, sep]

    for run in [def_run, rand_mean_run, best_rand_run, opt_run]:
        mm = run["metrics"]
        pf = "YES" if run["label"] in frontier_labels else "NO"
        cost_str = f"${mm['cost_per_1k']:.4f}"
        if run["label"] == "Random Search (Mean)":
            sla_str = "-"  # mean row aggregates configs; not a real pipeline
        else:
            sla_str = "YES" if meets_sla(run, solver.latency_limit, solver.cost_limit) else "NO"
        lines.append(
            f"{run['label']:<22} "
            f"{mm['context_precision']:>6.4f} "
            f"{mm['evidence_recall']:>8.4f} "
            f"{mm['extractive_groundedness']:>7.4f} "
            f"{mm['query_context_similarity']:>7.4f} "
            f"{mm.get('rouge_l', 0.0):>7.4f} "
            f"{mm['retrieval_latency_ms']:>8.1f} "
            f"{cost_str:>9} "
            f"{sla_str:>5} "
            f"{pf:>7}"
        )
    lines.append(sep)
    lines.append("")
    lines.append("  Gains vs Default Baseline:")
    for run in [best_rand_run, opt_run]:
        lines.append(f"  {run['label']}:")
        for m in ["context_precision", "evidence_recall", "retrieval_latency_ms", "cost_per_1k"]:
            d, pct = delta(run, def_run, m)
            direction = OBJECTIVES.get(m, "maximize")
            good = (d >= 0 and direction == "maximize") or (d <= 0 and direction == "minimize")
            arrow = "↑" if good else "↓"
            sign  = "+" if d >= 0 else ""
            lines.append(f"    {m:<24}: {sign}{d:.4f}  ({sign}{pct:.1f}%) {arrow}")
        lines.append("")

    table = "\n".join(lines)
    print(table)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 6: Persist Artifacts
    # ──────────────────────────────────────────────────────────────────────────
    results_payload = {
        "benchmark_scope": {
            "latency_scope": "retrieval_stage_wall_clock_ms",
            "cost_scope": "projected_prompt_tokens_at_0.150_per_1m",
            "eval_pairs": len(qa_pairs),
            "corpus_size": len(corpus),
            "sla_constraints": SLA_CONSTRAINTS,
            "seed": SEED,
            "candidate_budget_per_strategy": NUM_CANDIDATES,
            "metric_notes": {
                "extractive_groundedness": "Lexical overlap diagnostic for extractive text; not LLM faithfulness.",
                "query_context_similarity": "Query-to-extractive-context cosine similarity; not generated-answer relevancy.",
                "sparse_retrieval": "Jaccard token-overlap scorer; not BM25.",
                "evidence_recall": "Whole-token content-word coverage; exact-token matching, no substrings.",
            },
            "selection": {
                "score": SELECTION_WEIGHTS,
                "feasibility": SLA_CONSTRAINTS,
                "applies_to": ["Random Search (Best)", "RAGOS-Optimized"],
            },
            "dedup_key": "experiment_key: config minus generation dimension (the generator is never invoked by this benchmark)",
        },
        "default_baseline": def_run,
        "random_search_mean": rand_mean_run,
        "random_search_best": best_rand_run,
        "ragos_optimized": opt_run,
        "all_random_trials": random_runs,
        "pareto_frontier_labels": list(frontier_labels),
        "objectives": OBJECTIVES,
    }

    json_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    txt_path  = os.path.join(RESULTS_DIR, "summary_table.txt")
    with open(json_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    with open(txt_path, "w") as f:
        f.write(table + "\n")

    print(f"{'='*80}")
    print(" RAGOS VERDICT")
    print(f"{'='*80}")
    for m, direction in OBJECTIVES.items():
        d, pct = delta(opt_run, def_run, m)
        good = (d >= 0 and direction == "maximize") or (d <= 0 and direction == "minimize")
        status = "BETTER ↑" if good else "WORSE  ↓"
        sign = "+" if d >= 0 else ""
        print(f"  {m:<24}: {sign}{pct:+.1f}%  [{status}]")
    print(f"  {'Pareto Frontier Status':<24}: {'YES (Optimal non-dominated)' if 'RAGOS-Optimized' in frontier_labels else 'NO'}")
    print(f"  {'Artifacts Saved':<24}: {RESULTS_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
