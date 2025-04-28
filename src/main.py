import os
from dotenv import load_dotenv
from telegram.ext import ApplicationBuilder

from bot.handlers import register_handlers

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    register_handlers(app)

    app.run_polling()

if __name__ == "__main__":
    main()
    