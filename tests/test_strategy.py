import sys
from unittest.mock import MagicMock
# Mock rank_bm25 before importing anything that uses it
sys.modules['rank_bm25'] = MagicMock()

import pytest
from unittest.mock import patch
from backend.services.rag.retriever import Retriever, RetrievedChunk

@patch('backend.services.rag.retriever.VectorStore')
@patch('backend.services.rag.retriever.Embedder')
def test_retriever_strategies(MockEmbedder, MockVectorStore):
    # Setup mock vector store
    mock_vs = MockVectorStore.return_value
    mock_vs.get_all.return_value = {
        "documents": [
            "This is a document about machine learning and artificial intelligence.",
            "Cats are very cute pets.",
            "Transformers use attention mechanism for NLP tasks.",
            "Dogs bark at strangers but are loyal."
        ],
        "metadatas": [
            {"source": "doc1.pdf", "doc_id": "1", "chunk_index": 0},
            {"source": "doc2.pdf", "doc_id": "2", "chunk_index": 0},
            {"source": "doc3.pdf", "doc_id": "3", "chunk_index": 0},
            {"source": "doc4.pdf", "doc_id": "4", "chunk_index": 0}
        ],
        "ids": ["1_0", "2_0", "3_0", "4_0"]
    }
    
    mock_vs.query.return_value = {
        "documents": [["Transformers use attention mechanism for NLP tasks."]],
        "metadatas": [[{"source": "doc3.pdf", "doc_id": "3", "chunk_index": 0}]],
        "distances": [[0.1]]
    }
    
    # Setup mock embedder
    mock_emb = MockEmbedder.return_value
    mock_emb.embed_single.return_value = [0.1, 0.2, 0.3]
    
    # Setup mocked BM25
    mock_bm25_instance = MagicMock()
    # Fake get_scores to return high score for 3rd doc when searching 'attention mechanism'
    # and 2nd doc when searching 'pets'
    def fake_get_scores(tokenized_query):
        if "pets" in tokenized_query:
            return [0.1, 10.0, 0.1, 0.1]
        elif "attention" in tokenized_query:
            return [0.1, 0.1, 10.0, 0.1]
        return [0.1, 0.1, 0.1, 0.1]
    
    mock_bm25_instance.get_scores.side_effect = fake_get_scores
    sys.modules['rank_bm25'].BM25Okapi.return_value = mock_bm25_instance
    
    retriever = Retriever()
    
    # Test dense strategy
    dense_chunks = retriever.retrieve("attention mechanism", strategy="dense", top_k=2)
    assert len(dense_chunks) == 1
    assert "attention mechanism" in dense_chunks[0].text
    
    # Test sparse strategy
    sparse_chunks = retriever.retrieve("pets", strategy="sparse", top_k=2)
    assert len(sparse_chunks) > 0
    assert "pets" in sparse_chunks[0].text
    
    # Test hybrid strategy
    hybrid_chunks = retriever.retrieve("attention mechanism pets", strategy="hybrid", top_k=2)
    assert len(hybrid_chunks) > 0
    
    # Check that both mock approaches were called in hybrid
    has_attention = any("attention mechanism" in c.text for c in hybrid_chunks)
    assert has_attention
