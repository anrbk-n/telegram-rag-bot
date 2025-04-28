from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts: list) -> list:
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    return embeddings
