from typing import List
from .pipeline import ContextCompressor
from .retriever import RetrievedChunk
from sentence_transformers import SentenceTransformer, util
import nltk
import logging
from ..llm.router import LLMRouter

logger = logging.getLogger(__name__)

# Basic sentence tokenizer fallback if nltk not downloaded
def basic_sent_tokenize(text: str) -> List[str]:
    # A naive fallback
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s for s in sentences if s.strip()]

class ExtractiveContextCompressor(ContextCompressor):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", llm_router: LLMRouter = None):
        self.model = SentenceTransformer(model_name)
        self.llm = llm_router or LLMRouter()
        try:
            nltk.download('punkt', quiet=True)
            self.sent_tokenize = nltk.sent_tokenize
        except Exception:
            self.sent_tokenize = basic_sent_tokenize

    def compress(self, query: str, chunks: List[RetrievedChunk], budget: int) -> List[RetrievedChunk]:
        # Count words as a rough token proxy (1 word ~ 1.3 tokens usually, we'll use words for simplicity)
        total_words = sum(len(c.text.split()) for c in chunks)
        
        if total_words <= budget:
            return chunks
            
        logger.info(f"Compressing context from {total_words} words to budget {budget}")
        
        query_emb = self.model.encode(query, convert_to_tensor=True)
        compressed_chunks = []
        
        current_words = 0
        
        for chunk in chunks:
            sentences = self.sent_tokenize(chunk.text)
            if not sentences:
                continue
                
            sent_embs = self.model.encode(sentences, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_emb, sent_embs)[0]
            
            # Pair sentences with scores
            sent_scores = [(sentences[i], cos_scores[i].item()) for i in range(len(sentences))]
            # Sort by highest score
            sent_scores.sort(key=lambda x: x[1], reverse=True)
            
            selected_sentences = []
            for sent, score in sent_scores:
                sent_words = len(sent.split())
                if current_words + sent_words <= budget:
                    selected_sentences.append((sent, chunk.text.index(sent))) # keep index for reordering
                    current_words += sent_words
                else:
                    # if we can't fit even the top sentence, we skip
                    pass
            
            if selected_sentences:
                # Reorder to original appearance
                selected_sentences.sort(key=lambda x: x[1])
                compressed_text = " ".join([s[0] for s in selected_sentences])
                
                # Clone chunk
                new_chunk = RetrievedChunk(
                    text=compressed_text,
                    score=chunk.score,
                    source=chunk.source,
                    doc_id=chunk.doc_id,
                    chunk_index=chunk.chunk_index,
                    metadata=chunk.metadata
                )
                compressed_chunks.append(new_chunk)
                
            if current_words >= budget:
                break
                
        return compressed_chunks
