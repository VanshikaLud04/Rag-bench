import uuid
from pathlib import Path
from typing import List, Dict
import pypdf

def extract_text_from_pdf(file_path: str) -> str:
    reader = pypdf.PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def process_document(file_path: str, chunk_size: int, overlap: int) -> List[Dict]:
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text, chunk_size, overlap)
    doc_id = str(uuid.uuid4())
    source = Path(file_path).name

    return [
        {
            "id": f"{doc_id}_chunk_{i}",
            "text": chunk,
            "metadata": {
                "doc_id": doc_id,
                "source": source,
                "chunk_index": i,
            }
        }
        for i, chunk in enumerate(chunks)
    ]