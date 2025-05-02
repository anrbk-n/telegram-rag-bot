from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def language_keyboard(show_back=False, language: str = "ru"):
    keyboard = [
        [InlineKeyboardButton("Русский 🇷🇺", callback_data="lang_ru")],
        [InlineKeyboardButton("English 🇬🇧", callback_data="lang_en")]
    ]
    if show_back:
        back_label = "🔙 Назад" if language == "ru" else "🔙 Back"
        keyboard = [[InlineKeyboardButton(back_label, callback_data="lang_back")]]
    return InlineKeyboardMarkup(keyboard)
