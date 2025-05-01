from textsplit.tools import get_segments
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def split_text_into_chunks(text: str, max_chunk_tokens: int = 1000) -> list:
    segments = get_segments(text, model, ideal_segsize=max_chunk_tokens // 100)
    return [s.strip() for s in segments if len(s.strip()) > 50]
