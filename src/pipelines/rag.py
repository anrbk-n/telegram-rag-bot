import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List
from pipelines.chunk import split_text_into_chunks
from pipelines.embed import embed_texts
from pipelines.index import create_faiss_index, search_faiss

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("models/gemini-1.5-flash")

def build_prompt(question: str, context_chunks: List[str], language: str = "ru") -> str:
    context = "\n\n".join(context_chunks[:3]) 
    if language == "ru":
        return f"""Ты умный помощник. Отвечай строго по контексту. Если информации нет — честно скажи.

Контекст:
{context}

Вопрос:
{question}

Ответ:"""
    else:
        return f"""You are a helpful assistant. Answer strictly based on the context. If the answer is missing — say so honestly.

Context:
{context}

Question:
{question}

Answer:"""

def generate_answer_with_gemini(question: str, retrieved_chunks: List[str], language: str = "ru") -> str:
    prompt = build_prompt(question, retrieved_chunks, language)
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Gemini API error] {e}"

def answer_question(question: str, document_text: str, language: str = "ru") -> str:
    chunks = split_text_into_chunks(document_text)
    chunk_embeddings = embed_texts(chunks)
    index = create_faiss_index(chunk_embeddings)

    query_embedding = embed_texts([question])
    top_k_indices = search_faiss(index, query_embedding, top_k=3)
    top_chunks = [chunks[i] for i in top_k_indices if i < len(chunks)]

    return generate_answer_with_gemini(question, top_chunks, language)
