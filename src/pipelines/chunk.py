import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from langdetect import detect

def split_text_into_chunks(text: str, sentences_per_chunk: int = 10) -> list:

    try:
        lang = detect(text)
    except Exception:
        lang = "en"  
    if lang not in ["russian", "english"]:
        lang = "english"
    if lang == "ru":
        lang = "russian"
    elif lang == "en":
        lang = "english"

    sentences = sent_tokenize(text, language=lang)

    if len(sentences) < 3:
        return [text.strip()]

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk.strip())

    return [c for c in chunks if len(c) > 50]
