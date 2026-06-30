#!/usr/bin/env python3
"""ML Coding From Scratch — Telegram game module.

Imported by learn_bot.py (the main entry point).
Can also run standalone:  python ml-code-telegram.py
Token:                     learn_bot env var
Commands:                  /mlcode, /mlcodeProgress
"""

import asyncio
import json
import logging
import os
import signal
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
PROGRESS_FILE     = Path(__file__).parent / "ml_code_progress.json"

CATEGORIES = [
    "From Scratch",    # k-means, k-NN, linear/logistic regression, backprop, softmax
    "Building Blocks", # self-attention, convolution, mini-batch sampler, train/val split
    "Eval & Metrics",  # precision/recall/F1, ROC-AUC, confusion matrix
    "Sampling",        # weighted sampling, reservoir sampling, probability
]

DIFFICULTIES    = ["beginner", "intermediate", "advanced"]
QUESTION_COUNTS = [5, 10, 15, 20]

MC_SELECT_CATEGORIES, MC_SELECT_DIFFICULTY, MC_SELECT_COUNT, MC_ANSWERING = range(4)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CATEGORY_CONTEXT = """\
- From Scratch: implement k-means, k-NN, linear regression, logistic regression, backprop, \
softmax/cross-entropy in NumPy from first principles
- Building Blocks: self-attention / scaled dot-product attention, convolution, mini-batch \
sampler, train/val split, basic data loader logic
- Eval & Metrics: precision, recall, F1, ROC-AUC, confusion matrix computed from raw \
predictions — including numerically correct implementations
- Sampling: weighted random sampling, reservoir sampling for streams, probability basics"""


# ── Progress tracking ─────────────────────────────────────────────────────────

def _load_progress() -> dict:
    try:
        return json.loads(PROGRESS_FILE.read_text())
    except Exception:
        return {"sessions": [], "topic_stats": {}, "seen_fingerprints": [], "wrong_snippets": []}


def _save_progress(p: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(p, indent=2))


def _fingerprint(q: dict) -> str:
    return q["question"][:80].lower().strip()


def _progress_context(p: dict) -> str:
    sessions = p.get("sessions", [])
    stats    = p.get("topic_stats", {})
    wrong    = p.get("wrong_snippets", [])[-10:]

    if not sessions:
        return ""

    lines = ["USER PROGRESS CONTEXT (use this to personalise and target weak areas):"]
    lines.append(f"- Total sessions completed: {len(sessions)}")

    recent     = sessions[-5:]
    recent_pct = round(sum(s["pct"] for s in recent) / len(recent)) if recent else 0
    lines.append(f"- Recent average score (last {len(recent)} sessions): {recent_pct}%")

    if stats:
        lines.append("- Per-category accuracy:")
        for cat, s in sorted(stats.items()):
            seen    = s.get("seen", 0)
            correct = s.get("correct", 0)
            pct     = round(correct / seen * 100) if seen else 0
            lines.append(f"    {cat}: {correct}/{seen} correct ({pct}%)")
        weakest = min(stats, key=lambda c: (stats[c].get("correct", 0) / max(stats[c].get("seen", 1), 1)))
        lines.append(f"- Weakest category: {weakest} — prioritise questions here.")

    if wrong:
        lines.append("- Recent questions the user got WRONG (drill similar concepts):")
        for w in wrong:
            lines.append(f"    • {w}")

    seen_fps = p.get("seen_fingerprints", [])
    if seen_fps:
        lines.append(f"- {len(seen_fps)} questions already seen — generate NEW questions not repeating these verbatim.")

    return "\n".join(lines)


def _record_session(categories: list, difficulty: str, questions: list, answers: list) -> None:
    p = _load_progress()

    score = sum(1 for a in answers if a["correct"])
    total = len(answers)
    pct   = round(score / total * 100) if total else 0

    wrong_snippets = [
        q["question"][:100]
        for q, a in zip(questions, answers) if not a["correct"]
    ]

    p["sessions"].append({
        "date":       datetime.utcnow().isoformat(),
        "categories": categories,
        "difficulty": difficulty,
        "score":      score,
        "total":      total,
        "pct":        pct,
    })

    stats = p.setdefault("topic_stats", {})
    for q, a in zip(questions, answers):
        cat = q.get("category", "Unknown")
        s   = stats.setdefault(cat, {"seen": 0, "correct": 0})
        s["seen"]    += 1
        s["correct"] += int(a["correct"])

    fps = p.setdefault("seen_fingerprints", [])
    for q in questions:
        fp = _fingerprint(q)
        if fp not in fps:
            fps.append(fp)
    p["seen_fingerprints"] = fps[-500:]

    existing_wrong = p.setdefault("wrong_snippets", [])
    existing_wrong.extend(wrong_snippets)
    p["wrong_snippets"] = existing_wrong[-30:]

    _save_progress(p)


# ── Question generation ───────────────────────────────────────────────────────

def _build_prompt(categories: list, difficulty: str, n: int) -> str:
    cats     = ", ".join(categories)
    progress = _load_progress()
    ctx      = _progress_context(progress)
    ctx_block = f"\n\n{ctx}\n" if ctx else ""

    diff_guide = {
        "beginner":     "shape/dtype reasoning, sklearn-API logic, simple metric formulas, basic NumPy ops",
        "intermediate": "gradient descent update steps, k-means convergence checks, loss function implementations, backprop through a single layer",
        "advanced":     "numerically stable softmax/cross-entropy, vectorised multi-head attention, backprop through stacked layers, efficient reservoir sampling",
    }.get(difficulty, difficulty)

    return f"""Generate exactly {n} ML coding interview questions.
- Categories: {cats}
- Difficulty: {difficulty} ({diff_guide})
- Focus: NumPy implementations, code correctness, numerical stability
{ctx_block}
{CATEGORY_CONTEXT}

Guidelines:
1. Each question MUST include a short Python/NumPy code snippet (4–15 lines) either in the question body or in the options.
2. Question styles: "Which implementation is correct?", "What does this output?", \
"What's wrong with this code?", "Which option is numerically stable?", "Fill in the blank."
3. correctAnswer is always a single 0-based index (integer).
4. Keep questions practical — things an ML engineer would actually write or debug.

Respond ONLY with valid JSON:
{{
  "questions": [
    {{
      "question": "...",
      "options": ["A text", "B text", "C text", "D text"],
      "correctAnswer": 0,
      "category": "From Scratch"
    }}
  ]
}}"""


def _generate_live(categories: list, difficulty: str, n: int) -> list:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=MODEL, max_tokens=4096,
        messages=[{"role": "user", "content": _build_prompt(categories, difficulty, n)}],
    )
    text = msg.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text).get("questions", [])


# ── Keyboards ─────────────────────────────────────────────────────────────────

def _cat_keyboard(selected: set) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(
        f"{'✅' if cat in selected else '⬜'} {cat}", callback_data=f"mccat_{cat}"
    )] for cat in CATEGORIES]
    rows.append([InlineKeyboardButton("✓ Done", callback_data="mccat_done")])
    return InlineKeyboardMarkup(rows)


def _diff_keyboard() -> InlineKeyboardMarkup:
    labels = {"beginner": "🟢 Beginner", "intermediate": "🟡 Intermediate", "advanced": "🔴 Advanced"}
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(labels[d], callback_data=f"mcdiff_{d}") for d in DIFFICULTIES
    ]])


def _count_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(str(n), callback_data=f"mccount_{n}") for n in QUESTION_COUNTS
    ]])


def _answer_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(ch, callback_data=f"mcans_{ch}") for ch in "ABCD"
    ]])


# ── Handlers ──────────────────────────────────────────────────────────────────

async def quiz_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    context.user_data["selected_cats"] = set()
    await update.message.reply_text(
        "💻 *ML Coding From Scratch*\nSelect categories (tap to toggle, then Done):",
        reply_markup=_cat_keyboard(set()),
        parse_mode="Markdown",
    )
    return MC_SELECT_CATEGORIES


async def cat_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query
    await q.answer()
    selected = context.user_data.setdefault("selected_cats", set())

    if q.data == "mccat_done":
        if not selected:
            await q.answer("Pick at least one category!", show_alert=True)
            return MC_SELECT_CATEGORIES
        await q.edit_message_text(
            f"✅ Categories: *{', '.join(sorted(selected))}*\n\nSelect difficulty:",
            reply_markup=_diff_keyboard(),
            parse_mode="Markdown",
        )
        return MC_SELECT_DIFFICULTY

    cat = q.data[len("mccat_"):]
    selected.symmetric_difference_update({cat})
    await q.edit_message_reply_markup(_cat_keyboard(selected))
    return MC_SELECT_CATEGORIES


async def diff_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query
    await q.answer()
    diff = q.data[len("mcdiff_"):]
    context.user_data["difficulty"] = diff
    await q.edit_message_text(
        f"✅ Difficulty: *{diff.capitalize()}*\n\nHow many questions?",
        reply_markup=_count_keyboard(),
        parse_mode="Markdown",
    )
    return MC_SELECT_COUNT


async def count_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q    = update.callback_query
    await q.answer()
    n    = int(q.data[len("mccount_"):])
    cats = list(context.user_data["selected_cats"])
    diff = context.user_data["difficulty"]

    await q.edit_message_text(f"⏳ Generating {n} *{diff}* ML coding questions…", parse_mode="Markdown")

    if ANTHROPIC_API_KEY:
        try:
            questions = await asyncio.to_thread(_generate_live, cats, diff, n)
        except Exception as exc:
            await q.edit_message_text(f"❌ Generation failed: {exc}\nTry /mlcode again.")
            return ConversationHandler.END
    else:
        await q.edit_message_text("❌ No API key configured. Set ANTHROPIC_API_KEY.")
        return ConversationHandler.END

    if not questions:
        await q.edit_message_text("❌ No questions returned. Try /mlcode again.")
        return ConversationHandler.END

    context.user_data.update(questions=questions, current=0, score=0, answers=[])
    await _post_question(q, context)
    return MC_ANSWERING


async def _post_question(q, context: ContextTypes.DEFAULT_TYPE) -> None:
    qs      = context.user_data["questions"]
    current = context.user_data["current"]
    score   = context.user_data["score"]
    quest   = qs[current]
    opts    = "\n".join(f"*{chr(65+i)}.* {o}" for i, o in enumerate(quest["options"]))
    body    = f"```\n{quest['question']}\n```" if "\n" in quest["question"] else quest["question"]
    text    = (
        f"💻 *Q{current+1}/{len(qs)}*  |  Score: {score}\n"
        f"_{quest.get('category', '')}_ · {context.user_data.get('difficulty','').capitalize()}\n\n"
        f"{body}\n\n{opts}"
    )
    await q.edit_message_text(text, reply_markup=_answer_keyboard(), parse_mode="Markdown")


async def answer_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q      = update.callback_query
    await q.answer()
    chosen = ord(q.data[len("mcans_"):]) - 65
    qs     = context.user_data["questions"]
    quest  = qs[context.user_data["current"]]

    correct = chosen == quest["correctAnswer"]
    if correct:
        context.user_data["score"] += 1
    context.user_data["answers"].append({"selected": chosen, "correct": correct})
    context.user_data["current"] += 1

    fb  = f"{'✅' if correct else '❌'} *{'Correct!' if correct else 'Wrong!'}*\n"
    fb += f"Your answer: {quest['options'][chosen]}\n"
    if not correct:
        fb += f"✅ Correct: *{quest['options'][quest['correctAnswer']]}*\n"
    fb += f"\nScore: {context.user_data['score']}/{context.user_data['current']}"

    next_i = context.user_data["current"]
    btn    = "📊 Results" if next_i >= len(qs) else "Next →"
    cb     = "mcresults"  if next_i >= len(qs) else "mcnext"
    await q.edit_message_text(
        fb, parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(btn, callback_data=cb)]]),
    )
    return MC_ANSWERING


async def next_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query
    await q.answer()
    await _post_question(q, context)
    return MC_ANSWERING


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
            lines.append(f"   ↳ {quest['options'][quest['correctAnswer']]}")
    lines.append("\n/mlcode to play again | /mlcodeProgress to see stats")

    cats = context.user_data.get("selected_cats", [])
    diff = context.user_data.get("difficulty", "unknown")
    _record_session(list(cats), diff, qs, answers)

    await q.edit_message_text("\n".join(lines), parse_mode="Markdown")
    return ConversationHandler.END


async def progress_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    p        = _load_progress()
    sessions = p.get("sessions", [])
    stats    = p.get("topic_stats", {})

    if not sessions:
        await update.message.reply_text("No ML coding sessions yet. Use /mlcode to get started!")
        return

    total_q  = sum(s["total"] for s in sessions)
    total_ok = sum(s["score"] for s in sessions)
    overall  = round(total_ok / total_q * 100) if total_q else 0
    streak   = 0
    for s in reversed(sessions):
        if s["pct"] >= 60:
            streak += 1
        else:
            break

    lines = [
        f"📈 *ML Coding Progress*\n",
        f"Sessions completed : {len(sessions)}",
        f"Total questions    : {total_q}",
        f"Overall accuracy   : {total_ok}/{total_q} ({overall}%)",
        f"Win streak (≥60%)  : {streak} sessions\n",
    ]

    if stats:
        lines.append("*By category:*")
        for cat, s in sorted(stats.items()):
            seen    = s.get("seen", 0)
            correct = s.get("correct", 0)
            pct     = round(correct / seen * 100) if seen else 0
            bar     = "█" * (pct // 10) + "░" * (10 - pct // 10)
            lines.append(f"  {cat}: {bar} {pct}% ({correct}/{seen})")

    recent = sessions[-3:]
    lines.append("\n*Last 3 sessions:*")
    for s in recent:
        date = s["date"][:10]
        cats = ", ".join(s["categories"])
        lines.append(f"  {date} · {cats} · {s['difficulty']} · {s['score']}/{s['total']} ({s['pct']}%)")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("❌ Quiz cancelled. Use /mlcode to start again or /start for the menu.")
    return ConversationHandler.END


# ── Conversation handler (importable by learn_bot) ────────────────────────────

def build_handler() -> ConversationHandler:
    return ConversationHandler(
        entry_points=[CommandHandler("mlcode", quiz_start)],
        states={
            MC_SELECT_CATEGORIES: [CallbackQueryHandler(cat_callback,     pattern=r"^mccat_")],
            MC_SELECT_DIFFICULTY: [CallbackQueryHandler(diff_callback,    pattern=r"^mcdiff_")],
            MC_SELECT_COUNT:      [CallbackQueryHandler(count_callback,   pattern=r"^mccount_")],
            MC_ANSWERING: [
                CallbackQueryHandler(answer_callback,  pattern=r"^mcans_"),
                CallbackQueryHandler(next_callback,    pattern=r"^mcnext$"),
                CallbackQueryHandler(results_callback, pattern=r"^mcresults$"),
            ],
        },
        fallbacks=[
            CommandHandler("mlcode", quiz_start),
            CommandHandler("cancel", cancel_command),
            CommandHandler("start",  cancel_command),
        ],
    )


def main() -> None:
    if not TELEGRAM_TOKEN:
        print("learn_bot token not set in .env")
        return
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(build_handler())
    app.add_handler(CommandHandler("mlcodeProgress", progress_command))
    logger.info("ML Code bot polling…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
