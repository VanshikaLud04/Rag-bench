from typing import List
from .pipeline import CitationEngine, Citation
from .retriever import RetrievedChunk
from sentence_transformers import SentenceTransformer, util
import nltk
import logging

logger = logging.getLogger(__name__)

class SemanticCitationEngine(CitationEngine):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.75):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        try:
            nltk.download('punkt', quiet=True)
            self.sent_tokenize = nltk.sent_tokenize
        except Exception:
            # Fallback naive tokenizer
            import re
            self.sent_tokenize = lambda text: [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]

    def ground(self, answer_text: str, chunks: List[RetrievedChunk]) -> List[Citation]:
        citations = []
        if not chunks or not answer_text:
            return citations
            
        answer_sentences = self.sent_tokenize(answer_text)
        if not answer_sentences:
            return citations
            
        # Precompute chunk embeddings for their sentences
        chunk_sentences = []
        chunk_sentence_map = [] # stores (chunk_index, chunk)
        
        for c in chunks:
            sents = self.sent_tokenize(c.text)
            for s in sents:
                chunk_sentences.append(s)
                chunk_sentence_map.append(c)
                
        if not chunk_sentences:
            return citations
            
        chunk_embs = self.model.encode(chunk_sentences, convert_to_tensor=True)
        ans_embs = self.model.encode(answer_sentences, convert_to_tensor=True)
        
        cos_scores = util.cos_sim(ans_embs, chunk_embs)
        
        for i, ans_sent in enumerate(answer_sentences):
            scores = cos_scores[i]
            max_score_idx = scores.argmax().item()
            max_score = scores[max_score_idx].item()
            
            if max_score >= self.threshold:
                matched_chunk = chunk_sentence_map[max_score_idx]
                citations.append(Citation(
                    answer_sentence=ans_sent,
                    chunk=matched_chunk,
                    similarity_score=max_score
                ))
                
        return citations
