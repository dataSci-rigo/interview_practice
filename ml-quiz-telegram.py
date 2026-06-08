#!/usr/bin/env python3
"""ML Quiz Telegram Bot — clawbot group, ML topic thread (id=21)."""

import json
import logging
import os
import signal

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

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CHAT_ID = int(os.getenv("GROUP_CLAWBOT_CHAT_ID"))
THREAD_ID = 21
OWNER_ID = int(os.getenv("OWNER_CHAT_ID"))
MODEL = "claude-opus-4-20250514"

CATEGORIES = [
    "Video Recommendation", "Event Recommendation", "Ad Click Prediction",
    "Visual Search", "Video Search", "Personalized News Feed",
    "People You May Know", "Data Engineering", "Feature Engineering",
    "Model Selection", "Model Training", "Offline Metrics",
    "Online Metrics", "ML Serving",
]

LEVELS = [
    "L4 (Senior SWE)",
    "L5 (Staff SWE)",
    "L6 (Principal SWE)",
]

NUM_OPTIONS = [5, 10, 15, 20]

CATEGORY_CONTEXT = """\
- Video Recommendation: YouTube, Netflix, TikTok recommendation systems
- Event Recommendation: Facebook events, Eventbrite suggestions
- Ad Click Prediction: Google Ads, Facebook Ads CTR prediction
- Visual Search: Pinterest visual search, Google Lens
- Video Search: YouTube search, video content retrieval
- Personalized News Feed: Facebook feed, LinkedIn feed ranking
- People You May Know: LinkedIn, Facebook friend suggestions
- Data Engineering: Data pipelines, ETL, feature stores
- Feature Engineering: Feature selection, transformation, encoding
- Model Selection: Algorithm choice, ensemble methods
- Model Training: Training strategies, distributed training
- Offline Metrics: Evaluation metrics, A/B testing design
- Online Metrics: Production monitoring, real-time evaluation
- ML Serving: Model deployment, inference optimization"""

SELECT_CATEGORIES, SELECT_LEVEL, SELECT_NUM_QUESTIONS, ANSWERING = range(4)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)


# ---------------------------------------------------------------------------
# Keyboard builders
# ---------------------------------------------------------------------------

def category_keyboard(selected: set) -> InlineKeyboardMarkup:
    rows = []
    for i, cat in enumerate(CATEGORIES):
        mark = "✅" if i in selected else "⬜"
        rows.append([InlineKeyboardButton(f"{mark} {cat}", callback_data=f"cat_{i}")])
    rows.append([InlineKeyboardButton("✓ Done selecting", callback_data="cat_done")])
    return InlineKeyboardMarkup(rows)


def level_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(lvl, callback_data=f"lvl_{i}")]
         for i, lvl in enumerate(LEVELS)]
    )


def num_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(str(n), callback_data=f"num_{n}") for n in NUM_OPTIONS]]
    )


def answer_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(ch, callback_data=f"ans_{ch}") for ch in "ABCD"]]
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def post(context: ContextTypes.DEFAULT_TYPE, text: str, **kwargs):
    """Send a message to the ML thread."""
    return await context.bot.send_message(
        chat_id=CHAT_ID,
        message_thread_id=THREAD_ID,
        text=text,
        **kwargs,
    )


def generate_questions(categories: list, level: str, n: int) -> list:
    prompt = f"""Generate exactly {n} machine learning system design interview questions:
- Categories: {', '.join(categories)}
- Level: {level}
- Format: multiple choice, 4 options

{CATEGORY_CONTEXT}

Respond ONLY with valid JSON:
{{
  "questions": [
    {{
      "question": "...",
      "options": ["A text", "B text", "C text", "D text"],
      "correctAnswer": 0,
      "category": "..."
    }}
  ]
}}
correctAnswer is the 0-based index of the correct option."""

    msg = anthropic_client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text).get("questions", [])


async def send_question(context: ContextTypes.DEFAULT_TYPE, idx: int):
    q = context.user_data["questions"][idx]
    total = len(context.user_data["questions"])
    score = context.user_data["score"]
    opts = "\n".join(f"*{chr(65+i)}.* {o}" for i, o in enumerate(q["options"]))
    text = (
        f"❓ *Question {idx+1}/{total}*  |  Score: {score}\n"
        f"_{q.get('category', 'General')}_\n\n"
        f"{q['question']}\n\n{opts}"
    )
    await post(context, text, reply_markup=answer_keyboard(), parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Conversation handlers
# ---------------------------------------------------------------------------

async def quiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    context.user_data["selected"] = set()
    await post(
        context,
        "🧠 *ML System Design Quiz*\nSelect categories (tap to toggle, then press Done):",
        reply_markup=category_keyboard(set()),
        parse_mode="Markdown",
    )
    return SELECT_CATEGORIES


async def category_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query
    await q.answer()
    selected = context.user_data.setdefault("selected", set())

    if q.data == "cat_done":
        if not selected:
            await q.edit_message_text(
                "⚠️ Pick at least one category first!",
                reply_markup=category_keyboard(selected),
            )
            return SELECT_CATEGORIES
        context.user_data["categories"] = [CATEGORIES[i] for i in sorted(selected)]
        names = ", ".join(context.user_data["categories"])
        await q.edit_message_text(
            f"✅ *Categories:* {names}\n\nNow pick a difficulty level:",
            reply_markup=level_keyboard(),
            parse_mode="Markdown",
        )
        return SELECT_LEVEL

    idx = int(q.data[4:])
    selected.symmetric_difference_update({idx})
    await q.edit_message_reply_markup(reply_markup=category_keyboard(selected))
    return SELECT_CATEGORIES


async def level_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query
    await q.answer()
    level = LEVELS[int(q.data[4:])]
    context.user_data["level"] = level
    await q.edit_message_text(
        f"✅ *Level:* {level}\n\nHow many questions?",
        reply_markup=num_keyboard(),
        parse_mode="Markdown",
    )
    return SELECT_NUM_QUESTIONS


async def num_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query
    await q.answer()
    n = int(q.data[4:])
    categories = context.user_data["categories"]
    level = context.user_data["level"]

    await q.edit_message_text(f"⏳ Generating {n} questions for *{level}*…", parse_mode="Markdown")

    try:
        questions = generate_questions(categories, level, n)
    except Exception as exc:
        await post(context, f"❌ Failed to generate questions: {exc}\nTry /quiz again.")
        return ConversationHandler.END

    context.user_data.update(questions=questions, current=0, score=0, answers=[])
    await send_question(context, 0)
    return ANSWERING


async def answer_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query
    await q.answer()

    letter = q.data[4:]
    chosen = ord(letter) - 65

    questions = context.user_data["questions"]
    current = context.user_data["current"]
    question = questions[current]
    correct = question["correctAnswer"]
    context.user_data["answers"].append(chosen)

    if chosen == correct:
        context.user_data["score"] += 1
        feedback = f"✅ *Correct!*  {chr(65+correct)}. {question['options'][correct]}"
    else:
        feedback = (
            f"❌ *Incorrect*\n"
            f"Your answer: {letter}. {question['options'][chosen]}\n"
            f"✅ Correct: {chr(65+correct)}. {question['options'][correct]}"
        )

    await q.edit_message_reply_markup(reply_markup=None)
    await post(context, feedback, parse_mode="Markdown")

    current += 1
    context.user_data["current"] = current
    if current < len(questions):
        await send_question(context, current)
        return ANSWERING

    await show_results(context)
    return ConversationHandler.END


async def show_results(context: ContextTypes.DEFAULT_TYPE):
    questions = context.user_data["questions"]
    answers = context.user_data["answers"]
    score = context.user_data["score"]
    total = len(questions)
    pct = round(score / total * 100) if total else 0

    if pct >= 80:
        verdict = "🌟 Excellent! You're well prepared for senior ML interviews."
    elif pct >= 60:
        verdict = "👍 Good job! Solid foundation with room to sharpen."
    elif pct >= 40:
        verdict = "📚 Not bad! Keep reviewing the trade-offs."
    else:
        verdict = "💪 Keep studying! Focus on system design fundamentals."

    lines = [f"📊 *Results: {score}/{total} ({pct}%)*\n{verdict}\n"]
    for i, q in enumerate(questions):
        ok = answers[i] == q["correctAnswer"]
        short = q["question"][:70] + "…" if len(q["question"]) > 70 else q["question"]
        lines.append(f"{'✅' if ok else '❌'} {short}")
    lines.append("\nType /quiz to play again!")

    await post(context, "\n".join(lines), parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

async def shutdown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await post(context, "⛔ Only the bot owner can shut down the server.")
        return
    await post(context, "🔴 Shutting down ML Quiz bot. Goodbye!")
    os.kill(os.getpid(), signal.SIGINT)


# ---------------------------------------------------------------------------
# Startup welcome
# ---------------------------------------------------------------------------

async def on_startup(app: Application):
    await app.bot.send_message(
        chat_id=CHAT_ID,
        message_thread_id=THREAD_ID,
        text=(
            "🤖 *ML Quiz Bot is online!*\n\n"
            "Test your ML System Design knowledge for L4–L6 interviews.\n\n"
            "• /quiz — Start a new quiz session\n"
            "• /shutdown — Stop the bot _(owner only)_"
        ),
        parse_mode="Markdown",
    )
    logger.info("Welcome message sent to ML thread.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(on_startup).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("quiz", quiz_command)],
        states={
            SELECT_CATEGORIES: [CallbackQueryHandler(category_callback, pattern=r"^cat_")],
            SELECT_LEVEL:      [CallbackQueryHandler(level_callback,     pattern=r"^lvl_")],
            SELECT_NUM_QUESTIONS: [CallbackQueryHandler(num_callback,    pattern=r"^num_")],
            ANSWERING:         [CallbackQueryHandler(answer_callback,    pattern=r"^ans_")],
        },
        fallbacks=[CommandHandler("quiz", quiz_command)],
    )

    app.add_handler(conv)
    app.add_handler(CommandHandler("shutdown", shutdown_command))

    logger.info("ML Quiz bot polling…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
