import uuid
from typing import List, Dict
from .parsers import ParserRegistry, RawDocument

# Simple fixed-size chunker as a default
def chunk_text_fixed(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# Note: In a full production system, we'd add MarkdownHeadingChunker, ASTChunker, etc.
# For now, we wrap the fixed chunker or use specific strategies based on document type.
def chunk_document(doc: RawDocument, chunk_size: int, overlap: int) -> List[Dict]:
    doc_type = doc.metadata.get("type", "unknown")
    text = doc.text
    
    # Placeholder for structure-aware chunking based on doc_type
    # Example: if doc_type == "markdown": return chunk_text_markdown(text)
    
    chunks = chunk_text_fixed(text, chunk_size, overlap)
    doc_id = str(uuid.uuid4())
    
    return [
        {
            "id": f"{doc_id}_chunk_{i}",
            "text": chunk,
            "metadata": {
                **doc.metadata,
                "doc_id": doc_id,
                "chunk_index": i,
                "token_count": len(chunk.split()), # approximate token count
            }
        }
        for i, chunk in enumerate(chunks)
    ]

def process_document(source: str, chunk_size: int, overlap: int) -> List[Dict]:
    registry = ParserRegistry()
    parser = registry.get_parser(source)
    raw_docs = parser.parse(source)
    
    all_chunks = []
    for doc in raw_docs:
        all_chunks.extend(chunk_document(doc, chunk_size, overlap))
        
    return all_chunks