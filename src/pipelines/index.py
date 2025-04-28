import faiss
import torch

def create_faiss_index(embeddings: torch.Tensor):
   
    dim = embeddings.shape[1]  
    index = faiss.IndexFlatIP(dim)  

    faiss.normalize_L2(embeddings.cpu().numpy())
    index.add(embeddings.cpu().numpy())


    return index

def search_faiss(index, query_embedding: torch.Tensor, top_k: int = 5):
    query_np = query_embedding.cpu().numpy()  # сначала переносим на CPU
    faiss.normalize_L2(query_np)
    distances, indices = index.search(query_np, top_k)
    return indices[0]

