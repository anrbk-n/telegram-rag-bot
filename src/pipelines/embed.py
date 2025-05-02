from sentence_transformers import SentenceTransformer
import torch

embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def embed_texts(texts: list[str], convert_to_tensor: bool = True) -> torch.Tensor | list:
    return embed_model.encode(
        texts,
        convert_to_tensor=convert_to_tensor,
        normalize_embeddings=True,
        show_progress_bar=False
    )
