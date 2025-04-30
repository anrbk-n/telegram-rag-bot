import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from typing import List
from pipelines.chunk import split_text_into_chunks
from pipelines.embed import embed_texts
from pipelines.index import create_faiss_index, search_faiss

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    model="declare-lab/flan-alpaca-base",
    token=HF_TOKEN,
)

def build_prompt(question: str, context_chunks: List[str], language: str = "ru") -> str:
    context = "\n\n".join(context_chunks)
    if language == "ru":
        return f"""Ты отвечаешь на вопрос, основываясь исключительно на приведённом контексте. Если ответа нет в контексте, скажи об этом честно. Контекст:{context} Вопрос:{question}
Ответ:"""
    else:
        return f"""Answer the question strictly using the information from the context. If the answer is not present, say so. Context: {context} Question: {question}
Answer:"""

def generate_answer_with_hf(question: str, retrieved_chunks: List[str], language: str = "ru") -> str:
    prompt = build_prompt(question, retrieved_chunks, language)
    try:
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=300,
            temperature=0.5,
            repetition_penalty=1.1,
            stop=["Ответ:", "Answer:"],
            do_sample=True,
        )
        return response.strip()
    except Exception as e:
        return f"[HF API error] {e}"

def answer_question(question: str, document_text: str, language: str = "ru") -> str:
    chunks = split_text_into_chunks(document_text)
    chunk_embeddings = embed_texts(chunks)
    index = create_faiss_index(chunk_embeddings)

    query_embedding = embed_texts([question])
    top_k_indices = search_faiss(index, query_embedding)
    top_chunks = [chunks[i] for i in top_k_indices if i < len(chunks)]

    return generate_answer_with_hf(question, top_chunks, language)