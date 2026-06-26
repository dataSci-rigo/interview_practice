#!/usr/bin/env python3
"""FAANG Coding Interview Prep — Telegram game module.

Imported by learn_bot.py (the main entry point).
Can also run standalone:  python leet_telegram.py
Token:                     learn_bot env var
Commands:                  /quiz
"""

import asyncio
import json
import logging
import os
import random
import signal
import threading
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    ContextTypes,
)

load_dotenv()

TELEGRAM_TOKEN    = os.getenv("learn_bot", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OWNER_ID          = int(os.getenv("OWNER_CHAT_ID", "0") or "0")
MODEL             = "claude-sonnet-4-6"
CACHE_FILE        = Path(__file__).parent / "questions_cache.json"

CATEGORIES      = ["Algorithms", "Complexity", "Errors"]
DIFFICULTIES    = ["easy", "medium", "hard"]
QUESTION_COUNTS = [5, 10, 15, 20]

SELECT_CATEGORIES, SELECT_DIFFICULTY, SELECT_COUNT, ANSWERING = range(4)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Offline bank ──────────────────────────────────────────────────────────────

_cache_lock = threading.Lock()


def _load_offline_bank() -> dict:
    try:
        data = json.loads(CACHE_FILE.read_text())
        bank: dict = {}
        for q in data.get("questions", []):
            bank.setdefault(q.get("category", "Unknown"), []).append(q)
        return bank
    except Exception:
        return {}


OFFLINE_BANK = _load_offline_bank()


def _save_to_cache(questions: list, categories: list, difficulty: str) -> None:
    with _cache_lock:
        try:
            data = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {"questions": []}
        except Exception:
            data = {"questions": []}
        ts = datetime.utcnow().isoformat()
        for q in questions:
            data["questions"].append({
                **q, "id": str(uuid.uuid4()),
                "generated_at": ts,
                "difficulty_used": difficulty,
                "categories_used": categories,
            })
        CACHE_FILE.write_text(json.dumps(data, indent=2))


# ── Question generation ───────────────────────────────────────────────────────

def _build_prompt(categories: list, difficulty: str, n: int) -> str:
    cats = ", ".join(categories)
    return f"""Generate exactly {n} coding interview questions.
- Categories: {cats}
- Difficulty: {difficulty}
- Format: Multiple choice, 4 options

Guidelines:
1. Algorithms — FAANG Python questions; most efficient algorithm. correctAnswer may be a list when multiple correct.
2. Complexity — time or space complexity in Big-O notation.
3. Errors — numbered Python code; correctAnswer is the first error line number, or 0-based index of "No error" option.

Respond ONLY with valid JSON:
{{
  "questions": [
    {{
      "question": "...",
      "options": ["A text", "B text", "C text", "D text"],
      "correctAnswer": 0,
      "category": "Algorithms"
    }}
  ]
}}"""


def _generate_live(categories: list, difficulty: str, n: int) -> list:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=MODEL, max_tokens=4000,
        messages=[{"role": "user", "content": _build_prompt(categories, difficulty, n)}],
    )
    text = msg.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    questions = json.loads(text).get("questions", [])
    _save_to_cache(questions, categories, difficulty)
    return questions


def _generate_offline(categories: list, n: int) -> list:
    pool = []
    for cat in categories:
        pool.extend(OFFLINE_BANK.get(cat, []))
    if not pool:
        return []
    random.shuffle(pool)
    result = []
    while len(result) < n:
        result.extend(pool)
    return result[:n]


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _is_correct(q: dict, idx: int) -> bool:
    c = q["correctAnswer"]
    return idx in c if isinstance(c, list) else idx == c


def _correct_text(q: dict) -> str:
    c = q["correctAnswer"]
    if isinstance(c, list):
        return " / ".join(q["options"][i] for i in c)
    return q["options"][c]


# ── Keyboards ─────────────────────────────────────────────────────────────────

def _cat_keyboard(selected: set) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(
        f"{'✅' if cat in selected else '⬜'} {cat}", callback_data=f"lcat_{cat}"
    )] for cat in CATEGORIES]
    rows.append([InlineKeyboardButton("✓ Done", callback_data="lcat_done")])
    return InlineKeyboardMarkup(rows)


def _diff_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(d.capitalize(), callback_data=f"ldiff_{d}") for d in DIFFICULTIES
    ]])


def _count_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(str(n), callback_data=f"lcount_{n}") for n in QUESTION_COUNTS
    ]])


def _answer_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(ch, callback_data=f"lans_{ch}") for ch in "ABCD"
    ]])


# ── Handlers ──────────────────────────────────────────────────────────────────

async def quiz_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    context.user_data["selected_cats"] = set()
    await update.message.reply_text(
        "🎯 *FAANG Interview Prep*\nSelect categories (tap to toggle, then Done):",
        reply_markup=_cat_keyboard(set()),
        parse_mode="Markdown",
    )
    return SELECT_CATEGORIES


async def cat_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query
    await q.answer()
    selected = context.user_data.setdefault("selected_cats", set())

    if q.data == "lcat_done":
        if not selected:
            await q.answer("Pick at least one category!", show_alert=True)
            return SELECT_CATEGORIES
        await q.edit_message_text(
            f"✅ Categories: *{', '.join(sorted(selected))}*\n\nSelect difficulty:",
            reply_markup=_diff_keyboard(),
            parse_mode="Markdown",
        )
        return SELECT_DIFFICULTY

    cat = q.data[len("lcat_"):]
    selected.symmetric_difference_update({cat})
    await q.edit_message_reply_markup(_cat_keyboard(selected))
    return SELECT_CATEGORIES


async def diff_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query
    await q.answer()
    diff = q.data[len("ldiff_"):]
    context.user_data["difficulty"] = diff
    await q.edit_message_text(
        f"✅ Difficulty: *{diff.capitalize()}*\n\nHow many questions?",
        reply_markup=_count_keyboard(),
        parse_mode="Markdown",
    )
    return SELECT_COUNT


async def count_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q    = update.callback_query
    await q.answer()
    n    = int(q.data[len("lcount_"):])
    cats = list(context.user_data["selected_cats"])
    diff = context.user_data["difficulty"]

    await q.edit_message_text(f"⏳ Generating {n} *{diff}* questions…", parse_mode="Markdown")

    if ANTHROPIC_API_KEY:
        try:
            questions = await asyncio.to_thread(_generate_live, cats, diff, n)
        except Exception as exc:
            await q.edit_message_text(f"❌ Generation failed: {exc}\nUsing offline bank.")
            questions = _generate_offline(cats, n)
    else:
        questions = _generate_offline(cats, n)

    if not questions:
        await q.edit_message_text("❌ No questions available. Try /quiz with different categories.")
        return ConversationHandler.END

    context.user_data.update(questions=questions, current=0, score=0, answers=[])
    await _post_question(q, context)
    return ANSWERING


async def _post_question(q, context: ContextTypes.DEFAULT_TYPE) -> None:
    qs      = context.user_data["questions"]
    current = context.user_data["current"]
    score   = context.user_data["score"]
    quest   = qs[current]
    opts    = "\n".join(f"*{chr(65+i)}.* {o}" for i, o in enumerate(quest["options"]))
    body    = f"```\n{quest['question']}\n```" if "\n" in quest["question"] else quest["question"]
    text    = (
        f"❓ *Q{current+1}/{len(qs)}*  |  Score: {score}\n"
        f"_{quest.get('category', '')}_ · {context.user_data.get('difficulty','').capitalize()}\n\n"
        f"{body}\n\n{opts}"
    )
    await q.edit_message_text(text, reply_markup=_answer_keyboard(), parse_mode="Markdown")


async def answer_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q      = update.callback_query
    await q.answer()
    chosen = ord(q.data[len("lans_"):]) - 65
    qs     = context.user_data["questions"]
    quest  = qs[context.user_data["current"]]

    correct = _is_correct(quest, chosen)
    if correct:
        context.user_data["score"] += 1
    context.user_data["answers"].append({"selected": chosen, "correct": correct})
    context.user_data["current"] += 1

    fb  = f"{'✅' if correct else '❌'} *{'Correct!' if correct else 'Wrong!'}*\n"
    fb += f"Your answer: {quest['options'][chosen]}\n"
    if not correct:
        fb += f"✅ Correct: *{_correct_text(quest)}*\n"
    fb += f"\nScore: {context.user_data['score']}/{context.user_data['current']}"

    next_i = context.user_data["current"]
    btn    = "📊 Results" if next_i >= len(qs) else "Next →"
    cb     = "lresults"  if next_i >= len(qs) else "lnext"
    await q.edit_message_text(
        fb, parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(btn, callback_data=cb)]]),
    )
    return ANSWERING


async def next_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query
    await q.answer()
    await _post_question(q, context)
    return ANSWERING


async def results_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q       = update.callback_query
    await q.answer()
    qs      = context.user_data["questions"]
    answers = context.user_data["answers"]
    score   = context.user_data["score"]
    total   = len(qs)
    pct     = round(score / total * 100) if total else 0
    verdict = ("🌟 Excellent!" if pct >= 80 else "👍 Good job!" if pct >= 60
               else "📚 Keep going!" if pct >= 40 else "💪 Keep studying!")
    lines   = [f"📊 *Results: {score}/{total} ({pct}%)*\n{verdict}\n"]
    for i, (quest, ans) in enumerate(zip(qs, answers)):
        short = quest["question"].split("\n")[0][:65]
        lines.append(f"{'✅' if ans['correct'] else '❌'} Q{i+1}: {short}")
        if not ans["correct"]:
            lines.append(f"   ↳ {_correct_text(quest)}")
    lines.append("\n/quiz to play again")
    await q.edit_message_text("\n".join(lines), parse_mode="Markdown")
    return ConversationHandler.END


async def shutdown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Owner only.")
        return
    await update.message.reply_text("🔴 Shutting down.")
    os.kill(os.getpid(), signal.SIGINT)


# ── Conversation handler (importable by learn_bot) ────────────────────────────

def build_handler() -> ConversationHandler:
    return ConversationHandler(
        entry_points=[CommandHandler("quiz", quiz_start)],
        states={
            SELECT_CATEGORIES: [CallbackQueryHandler(cat_callback,    pattern=r"^lcat_")],
            SELECT_DIFFICULTY:  [CallbackQueryHandler(diff_callback,   pattern=r"^ldiff_")],
            SELECT_COUNT:       [CallbackQueryHandler(count_callback,  pattern=r"^lcount_")],
            ANSWERING: [
                CallbackQueryHandler(answer_callback,  pattern=r"^lans_"),
                CallbackQueryHandler(next_callback,    pattern=r"^lnext$"),
                CallbackQueryHandler(results_callback, pattern=r"^lresults$"),
            ],
        },
        fallbacks=[CommandHandler("quiz", quiz_start)],
    )


def main() -> None:
    if not TELEGRAM_TOKEN:
        print("learn_bot token not set in .env")
        return
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(build_handler())
    app.add_handler(CommandHandler("shutdown", shutdown_command))
    logger.info("Leet Prep bot polling…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
