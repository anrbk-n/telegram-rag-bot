from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def language_keyboard():
    keyboard = [
        [InlineKeyboardButton("Русский 🇷🇺", callback_data="lang_ru")],
        [InlineKeyboardButton("English 🇬🇧", callback_data="lang_en")]
    ]
    return InlineKeyboardMarkup(keyboard)
