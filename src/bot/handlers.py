import os
import tempfile
import logging

from telegram import Update
from telegram.ext import ContextTypes, CommandHandler, CallbackQueryHandler, MessageHandler, filters

from bot.keyboards import language_keyboard

from pipelines.docx import extract_text_from_docx
from pipelines.pdf import extract_text_from_pdf
from pipelines.chunk import split_text_into_chunks
from pipelines.embed import embed_texts
from pipelines.index import create_faiss_index, search_faiss
from pipelines.rag import generate_answer_with_gemini

user_language = {}
user_indices = {}
user_chunks = {}

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет!\n\nHello!\n\nВыберите язык / Select language:",
        reply_markup=language_keyboard()
    )

async def language_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    selected = query.data

    if selected == "lang_ru":
        user_language[user_id] = "ru"
        await query.edit_message_text(
            "✅ Язык установлен: Русский 🇷🇺\n\n📄 Теперь отправьте файл в формате PDF или DOCX."
        )
    elif selected == "lang_en":
        user_language[user_id] = "en"
        await query.edit_message_text(
            "✅ Language set to: English 🇬🇧\n\n📄 Now please send a PDF or DOCX file."
        )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id

    if user_id not in user_language:
        await update.message.reply_text("❗ Сначала выберите язык через /start.")
        return

    document = update.message.document
    file = await document.get_file()

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
        await file.download_to_drive(path)

    if document.file_name.lower().endswith(".pdf"):
        text = extract_text_from_pdf(path)
    elif document.file_name.lower().endswith(".docx"):
        text = extract_text_from_docx(path)
    else:
        await update.message.reply_text("❗ Пожалуйста, отправьте файл в формате PDF или DOCX.")
        os.remove(path)
        return

    os.remove(path)

    if not text.strip():
        await update.message.reply_text("❗ Не удалось извлечь текст из файла.")
        return

    chunks = split_text_into_chunks(text)
    embeddings = embed_texts(chunks)
    index = create_faiss_index(embeddings)

    user_chunks[user_id] = chunks
    user_indices[user_id] = index

    await update.message.reply_text("✅ Файл успешно обработан! Теперь отправьте ваш вопрос.")

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = update.message.text

    if user_id not in user_language:
        await update.message.reply_text("❗ Сначала выберите язык через /start.")
        return
    if user_id not in user_indices:
        await update.message.reply_text("❗ Сначала отправьте файл (PDF или DOCX).")
        return

    try:
        lang = user_language[user_id]
        embedding = embed_texts([text])[0].unsqueeze(0)
        top_idxs = search_faiss(user_indices[user_id], embedding, top_k=3)
        context_chunks = [user_chunks[user_id][i] for i in top_idxs]

        answer = generate_answer_with_gemini(text, context_chunks, language=lang)
        await update.message.reply_text(answer)
    except Exception as e:
        logging.exception("Ошибка при обработке вопроса:")
        await update.message.reply_text("❌ Что-то пошло не так. Попробуйте ещё раз.")


def register_handlers(app):
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CallbackQueryHandler(language_selection))
    app.add_handler(MessageHandler(filters.Document.PDF | filters.Document.DOCX, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
