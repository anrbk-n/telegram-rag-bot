# Telegram RAG Bot (в процессе)

Бот на Python для “retrieval-augmented generation” по PDF/DOCX:  
– разбивает документ на чанки,  
– векторизует через Sentence-Transformers,  
– ищет релевантные куски в FAISS,  
– генерирует ответы через Google Generative Language API.

## 📋 Возможности

- **Мультиязычность:** русский / английский  
- **Поддержка PDF и DOCX**  
- **Чанкинг**   
- **Эмбеддинги** через `sentence-transformers`  
- **FAISS** для быстрого поиска  
- **Генерация ответов** через gemini
