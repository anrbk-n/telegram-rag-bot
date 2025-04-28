import os
import time
import logging
import re
import requests
from dotenv import load_dotenv
from typing import List

load_dotenv()
API_KEY = os.getenv("GEM_API_TOKEN")
if not API_KEY:
    logging.warning("[Gemini] API key (GEM_API_TOKEN) not found")

API_URL = "https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText"
HEADERS = {"Content-Type": "application/json"}

def gemini_api_call(prompt: str, max_new_tokens: int = 300, timeout: int = 30, retries: int = 3, delay: float = 2.0) -> str:
    if not API_KEY:
        return ""
    payload = {
        "prompt": {"text": prompt},
        "temperature": 0.3,
        "maxOutputTokens": max_new_tokens
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(f"{API_URL}?key={API_KEY}", headers=HEADERS, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["output"]
        except requests.exceptions.HTTPError:
            if resp.status_code == 429:
                time.sleep(delay * attempt)
                continue
            break
        except Exception:
            break
    return ""

def optimize_question(raw_question: str, language: str = "ru") -> str:
    if language == "ru":
        return (
            "Ты профессиональный помощник. Переформулируй следующий пользовательский вопрос "
            "так, чтобы он был максимально чётким и логичным для поиска информации:\n\n"
            f"{raw_question}\n\nПереписанный вопрос:"
        )
    else:
        return (
            "You are a professional assistant. Rewrite the following user question "
            "to make it as clear and logical as possible for information retrieval:\n\n"
            f"{raw_question}\n\nRewritten question:"
        )

def generate_answer(question: str, retrieved_chunks: List[str], language: str = "ru") -> str:
    optimized_q = optimize_question(question, language)
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nQuestion:\n{optimized_q}\n\nAnswer:"
    raw = gemini_api_call(prompt)
    if not raw:
        return "❌ Не удалось получить ответ. Попробуйте позже."
    if raw.startswith(prompt):
        raw = raw[len(prompt):]
    raw = re.sub(re.escape(optimized_q), "", raw, flags=re.IGNORECASE).strip()
    answer = re.sub(r"^```[\w\n]*|```$", "", raw.strip(), flags=re.IGNORECASE).strip()
    return answer
