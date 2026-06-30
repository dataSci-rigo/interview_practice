#!/usr/bin/env python3
"""
learn_bot.py — Telegram learning hub dispatcher.

/start → choose a game → plays that game in this conversation.

Available games:
  🎯  Leet Practice   (/quiz)   — FAANG coding / complexity / error questions
  🧠  ML Design Quiz  (/mlquiz) — ML system design for L4–L6

Token:  learn_bot env var
Run:    python learn_bot.py
"""

import logging
import os

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

import importlib.util
from pathlib import Path

def _import_hyphenated(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_HERE         = Path(__file__).parent
leet_telegram = _import_hyphenated("leet_telegram",    _HERE / "leet_telegram.py")
ml_quiz       = _import_hyphenated("ml_quiz_telegram", _HERE / "ml-quiz-telegram.py")
ml_code       = _import_hyphenated("ml_code_telegram", _HERE / "ml-code-telegram.py")

load_dotenv()

TOKEN    = os.getenv("learn_bot", "")
OWNER_ID = int(os.getenv("OWNER_CHAT_ID", "0") or "0")

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Menu ──────────────────────────────────────────────────────────────────────

GAMES = [
    ("🎯  Leet Practice",  "/quiz"),
    ("🧠  ML Design Quiz", "/mlquiz"),
    ("💻  ML Code Quiz",   "/mlcode"),
]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton(label, callback_data=f"launch_{cmd[1:]}")
    ] for label, cmd in GAMES])
    await update.message.reply_text(
        "👋 *Learning Hub*\n\nChoose a game to start:\n\n"
        "/quiz — FAANG coding practice\n"
        "/mlquiz — ML system design\n"
        "/mlcode — ML coding from scratch\n"
        "/progress — DS&A stats\n"
        "/mlcodeProgress — ML coding stats\n"
        "/cancel — exit current game\n"
        "/start — return to this menu",
        reply_markup=keyboard,
        parse_mode="Markdown",
    )


async def launch_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q    = update.callback_query
    await q.answer()
    game = q.data[len("launch_"):]   # "quiz" or "mlquiz"

    # Synthesise a /command message so the ConversationHandler picks it up
    await q.edit_message_text(
        f"Starting *{'Leet Practice' if game == 'quiz' else 'ML Design Quiz'}*…\n"
        f"Use /{game} to restart at any time.",
        parse_mode="Markdown",
    )
    # Send the slash command as a new message to trigger the ConversationHandler
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"/{game}",
    )


async def shutdown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Owner only.")
        return
    await update.message.reply_text("🔴 Shutting down.")
    import os, signal
    os.kill(os.getpid(), signal.SIGINT)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not TOKEN:
        print("learn_bot token not set in .env")
        return

    app = Application.builder().token(TOKEN).build()

    # Game conversation handlers
    app.add_handler(leet_telegram.build_handler())
    app.add_handler(ml_quiz.build_handler())
    app.add_handler(ml_code.build_handler())

    # Hub commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("progress", leet_telegram.progress_command))
    app.add_handler(CommandHandler("mlcodeProgress", ml_code.progress_command))
    app.add_handler(CallbackQueryHandler(launch_callback, pattern=r"^launch_"))
    app.add_handler(CommandHandler("shutdown", shutdown_command))

    logger.info("Learn Bot polling — games: %s", [g for _, g in GAMES])
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
