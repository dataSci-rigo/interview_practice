#!/usr/bin/env python3
"""
FAANG Coding Interview Prep — Discord Bot
==========================================
  python leet_discord.py

Setup:
  Set DISCORD_TOKEN in your .env file.
  Optional: DISCORD_OWNER_ID for /shutdown command.
  Optional: ANTHROPIC_API_KEY for live AI question generation.

Shutdown:
  /shutdown  (owner only)
"""

import asyncio
import json
import os
import random
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DISCORD_TOKEN     = os.environ.get("DISCORD_TOKEN", "")
DISCORD_OWNER_ID  = int(os.environ.get("DISCORD_OWNER_ID", "0") or "0")
CACHE_FILE        = Path(__file__).parent / "questions_cache.json"
MODEL             = "claude-sonnet-4-6"

try:
    import anthropic as _anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
CATEGORIES      = ["Algorithms", "Complexity", "Errors"]
DIFFICULTIES    = ["easy", "medium", "hard"]
QUESTION_COUNTS = [5, 10, 15, 20]

# ── Offline question bank ─────────────────────────────────────────────────────
def _load_offline_bank() -> dict:
    try:
        data = json.loads(CACHE_FILE.read_text())
        bank: dict = {}
        for q in data.get("questions", []):
            cat = q.get("category", "Unknown")
            bank.setdefault(cat, []).append(q)
        return bank
    except Exception:
        return {}

OFFLINE_BANK = _load_offline_bank()

# ── Question cache ────────────────────────────────────────────────────────────
_cache_lock = threading.Lock()

def _read_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return {"questions": []}

def save_questions_to_cache(questions: list, categories: list, difficulty: str) -> None:
    with _cache_lock:
        data = _read_cache()
        ts = datetime.utcnow().isoformat()
        for q in questions:
            data["questions"].append({
                **q,
                "id": str(uuid.uuid4()),
                "generated_at": ts,
                "difficulty_used": difficulty,
                "categories_used": categories,
            })
        CACHE_FILE.write_text(json.dumps(data, indent=2))

# ── Core logic ────────────────────────────────────────────────────────────────
def build_prompt(categories: list, difficulty: str, num_questions: int) -> str:
    cats = ", ".join(categories)
    return f"""Generate exactly {num_questions} coding interview questions with the following specifications:
- Categories: {cats}
- Difficulty: {difficulty}
- Format: Multiple choice with 4 options

Category Guidelines:
1. "Algorithms" - FAANG coding interview questions in Python. The answer should be the most efficient algorithm to solve the question. Some questions can have multiple correct algorithms - in that case set "correctAnswer" to a list of all correct option indices.

2. "Complexity" - FAANG coding interview questions in Python. The answer should be either the minimum time complexity OR space complexity in which the question can be solved.

3. "Errors" - FAANG coding interview questions in Python with a solution provided. Number every line of the solution starting at 1. The solution may or may not contain errors. If there are no errors, the correct option text is "No error". If there is an error, the correct option identifies the line number of the FIRST error. Include the numbered code inside the "question" field.

Respond ONLY with a valid JSON object in this exact format:
{{
  "questions": [
    {{
      "question": "Question text here (include numbered code for Errors questions)",
      "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
      "correctAnswer": 0,
      "category": "Algorithms"
    }}
  ]
}}

"correctAnswer" is the 0-based index of the correct option, or a list of indices when multiple options are correct (Algorithms only).
Make sure each question has exactly 4 plausible options.
DO NOT OUTPUT ANYTHING OTHER THAN VALID JSON. Your entire response must be a single, valid JSON object."""


def generate_questions_live(api_key: str, categories: list, difficulty: str, num_questions: int) -> list:
    client = _anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": build_prompt(categories, difficulty, num_questions)}],
    )
    text = "".join(b.text for b in message.content if b.type == "text")
    cleaned = text.replace("```json", "").replace("```", "").strip()
    data = json.loads(cleaned)
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Invalid response format from the model.")
    save_questions_to_cache(questions, categories, difficulty)
    return questions


def generate_questions_offline(categories: list, difficulty: str, num_questions: int) -> list:
    pool = []
    for cat in categories:
        pool.extend(OFFLINE_BANK.get(cat, []))
    random.shuffle(pool)
    chosen = []
    while len(chosen) < num_questions and pool:
        for q in pool:
            chosen.append(q)
            if len(chosen) >= num_questions:
                break
        if len(chosen) < num_questions:
            random.shuffle(pool)
    return chosen[:num_questions]


def is_correct(question: dict, selected_index: int) -> bool:
    correct = question["correctAnswer"]
    return selected_index in correct if isinstance(correct, (list, tuple)) else selected_index == correct


def correct_text(question: dict) -> str:
    correct = question["correctAnswer"]
    if isinstance(correct, (list, tuple)):
        return " / ".join(question["options"][i] for i in correct)
    return question["options"][correct]


def fmt_question(q: dict, idx: int, total: int, score: int) -> str:
    cat   = q.get("category", "")
    qtext = q["question"]
    lines = [f"**Q{idx + 1}/{total}** · *{cat}*", ""]
    if "\n" in qtext:
        lines.append(f"```\n{qtext}\n```")
    else:
        lines.append(qtext)
    lines.append("")
    for i, opt in enumerate(q["options"]):
        lines.append(f"{chr(65 + i)}. {opt}")
    lines.append(f"\n*Score: {score}*")
    return "\n".join(lines)


# ── Discord Bot ───────────────────────────────────────────────────────────────
def run_discord():
    try:
        import discord
        from discord import app_commands
    except ImportError:
        print("discord.py not installed. Run: pip install discord.py")
        sys.exit(1)

    if not DISCORD_TOKEN:
        print("DISCORD_TOKEN not set in .env")
        sys.exit(1)

    # Per-user game state
    _sessions: dict[int, dict] = {}

    def get_sess(uid: int) -> dict:
        return _sessions.get(uid, {})

    def set_sess(uid: int, data: dict) -> None:
        _sessions[uid] = data

    # ── Views ─────────────────────────────────────────────────────────────────

    class _OwnedView(discord.ui.View):
        """Base view that rejects interaction from non-owners."""
        def __init__(self, user_id: int, timeout: float = 120):
            super().__init__(timeout=timeout)
            self.user_id = user_id

        async def _check(self, interaction: discord.Interaction) -> bool:
            if interaction.user.id != self.user_id:
                await interaction.response.send_message(
                    "This is not your game!", ephemeral=True)
                return False
            return True

    # ── Category selection ──
    class CategoryView(_OwnedView):
        def __init__(self, user_id: int):
            super().__init__(user_id)
            self.selected: set[str] = set()
            for cat in CATEGORIES:
                self.add_item(_CatToggle(cat, self))
            self.add_item(_CatConfirm(self))

    class _CatToggle(discord.ui.Button):
        def __init__(self, cat: str, parent: "CategoryView"):
            super().__init__(label=cat, style=discord.ButtonStyle.secondary)
            self.cat    = cat
            self.parent = parent

        async def callback(self, interaction: discord.Interaction):
            if not await self.parent._check(interaction):
                return
            if self.cat in self.parent.selected:
                self.parent.selected.discard(self.cat)
                self.style = discord.ButtonStyle.secondary
            else:
                self.parent.selected.add(self.cat)
                self.style = discord.ButtonStyle.success
            await interaction.response.edit_message(view=self.parent)

    class _CatConfirm(discord.ui.Button):
        def __init__(self, parent: "CategoryView"):
            super().__init__(label="Confirm ✓", style=discord.ButtonStyle.primary, row=1)
            self.parent = parent

        async def callback(self, interaction: discord.Interaction):
            if not await self.parent._check(interaction):
                return
            if not self.parent.selected:
                await interaction.response.send_message(
                    "Select at least one category!", ephemeral=True)
                return
            uid = self.parent.user_id
            set_sess(uid, {"cats": list(self.parent.selected)})
            cats_str  = ", ".join(self.parent.selected)
            diff_view = DifficultyView(uid)
            await interaction.response.edit_message(
                content=f"✅ Categories: **{cats_str}**\n\n**Select difficulty:**",
                view=diff_view,
            )
            self.parent.stop()

    # ── Difficulty selection ──
    class DifficultyView(_OwnedView):
        def __init__(self, user_id: int):
            super().__init__(user_id)
            for d in DIFFICULTIES:
                self.add_item(_DiffButton(d, self))

    class _DiffButton(discord.ui.Button):
        def __init__(self, diff: str, parent: "DifficultyView"):
            super().__init__(label=diff.capitalize(), style=discord.ButtonStyle.secondary)
            self.diff   = diff
            self.parent = parent

        async def callback(self, interaction: discord.Interaction):
            if not await self.parent._check(interaction):
                return
            uid  = self.parent.user_id
            sess = get_sess(uid)
            sess["diff"] = self.diff
            set_sess(uid, sess)
            count_view = CountView(uid)
            await interaction.response.edit_message(
                content=f"✅ Difficulty: **{self.diff.capitalize()}**\n\n**How many questions?**",
                view=count_view,
            )
            self.parent.stop()

    # ── Question count selection ──
    class CountView(_OwnedView):
        def __init__(self, user_id: int):
            super().__init__(user_id)
            for n in QUESTION_COUNTS:
                self.add_item(_CountButton(n, self))

    class _CountButton(discord.ui.Button):
        def __init__(self, count: int, parent: "CountView"):
            super().__init__(label=str(count), style=discord.ButtonStyle.secondary)
            self.count  = count
            self.parent = parent

        async def callback(self, interaction: discord.Interaction):
            if not await self.parent._check(interaction):
                return
            uid   = self.parent.user_id
            sess  = get_sess(uid)
            cats  = sess.get("cats", [])
            diff  = sess.get("diff", "medium")
            count = self.count

            # Acknowledge and start generation
            await interaction.response.defer_update()
            self.parent.stop()
            await interaction.edit_original_response(
                content="⏳ **Generating questions...**", view=None)

            if ANTHROPIC_API_KEY and ANTHROPIC_AVAILABLE:
                try:
                    questions = await asyncio.to_thread(
                        generate_questions_live, ANTHROPIC_API_KEY, cats, diff, count)
                except Exception as exc:
                    await interaction.edit_original_response(
                        content=(f"❌ Generation failed: {exc}\n\n"
                                 "Falling back to offline questions."))
                    questions = generate_questions_offline(cats, diff, count)
            else:
                questions = generate_questions_offline(cats, diff, count)

            if not questions:
                await interaction.edit_original_response(
                    content="❌ No questions available for that selection.")
                return

            set_sess(uid, {
                "cats": cats, "diff": diff,
                "questions": questions, "current": 0, "score": 0, "answers": [],
            })
            q = questions[0]
            await interaction.edit_original_response(
                content=fmt_question(q, 0, len(questions), 0),
                view=AnswerView(uid),
            )

    # ── Answer buttons ──
    class AnswerView(_OwnedView):
        def __init__(self, user_id: int):
            super().__init__(user_id, timeout=300)
            for i in range(4):
                self.add_item(_AnswerButton(i, self))

    class _AnswerButton(discord.ui.Button):
        def __init__(self, idx: int, parent: "AnswerView"):
            super().__init__(label=chr(65 + idx), style=discord.ButtonStyle.secondary)
            self.idx    = idx
            self.parent = parent

        async def callback(self, interaction: discord.Interaction):
            if not await self.parent._check(interaction):
                return
            uid       = self.parent.user_id
            sess      = get_sess(uid)
            questions = sess["questions"]
            current   = sess["current"]
            score     = sess["score"]
            answers   = sess["answers"]
            q         = questions[current]
            total     = len(questions)

            correct = is_correct(q, self.idx)
            if correct:
                score += 1
            answers.append({"question_index": current,
                            "selected": self.idx, "correct": correct})
            sess.update({"score": score, "answers": answers})
            set_sess(uid, sess)

            icon = "✅" if correct else "❌"
            text  = f"{icon} **{'Correct!' if correct else 'Wrong!'}**\n\n"
            text += f"Your answer: {q['options'][self.idx]}\n"
            if not correct:
                text += f"Correct: **{correct_text(q)}**\n"
            text += f"\nScore: {score}/{current + 1}"

            next_view = NextView(uid, to_results=(current + 1 >= total))
            await interaction.response.edit_message(content=text, view=next_view)
            self.parent.stop()

    # ── Next / See Results button ──
    class NextView(_OwnedView):
        def __init__(self, user_id: int, to_results: bool):
            super().__init__(user_id, timeout=300)
            label = "📊 See Results" if to_results else "Next →"
            self.add_item(_NextButton(user_id, label, to_results, self))

    class _NextButton(discord.ui.Button):
        def __init__(self, uid: int, label: str, to_results: bool, parent: "NextView"):
            super().__init__(label=label, style=discord.ButtonStyle.primary)
            self.uid        = uid
            self.to_results = to_results
            self.parent     = parent

        async def callback(self, interaction: discord.Interaction):
            if not await self.parent._check(interaction):
                return
            self.parent.stop()
            sess = get_sess(self.uid)

            if self.to_results:
                questions = sess["questions"]
                answers   = sess["answers"]
                score     = sess["score"]
                total     = len(questions)
                pct       = round((score / total) * 100) if total else 0
                msg       = ("Excellent!" if pct >= 80 else "Good job!" if pct >= 60
                             else "Not bad!" if pct >= 40 else "Keep studying!")

                text = f"📊 **Results**\n\nScore: **{score}/{total}** ({pct}%)\n{msg}\n\n"
                for i, (q, ans) in enumerate(zip(questions, answers)):
                    icon  = "✅" if ans["correct"] else "❌"
                    first = q["question"].split("\n")[0][:65]
                    text += f"{icon} Q{i + 1}: {first}\n"
                    if not ans["correct"]:
                        text += f"   ↳ Correct: {correct_text(q)}\n"

                await interaction.response.edit_message(
                    content=text, view=PlayAgainView(self.uid))
            else:
                sess["current"] += 1
                current   = sess["current"]
                questions = sess["questions"]
                score     = sess["score"]
                set_sess(self.uid, sess)
                q = questions[current]
                await interaction.response.edit_message(
                    content=fmt_question(q, current, len(questions), score),
                    view=AnswerView(self.uid),
                )

    # ── Play Again button ──
    class PlayAgainView(_OwnedView):
        def __init__(self, user_id: int):
            super().__init__(user_id, timeout=300)
            self.add_item(_PlayAgainButton(user_id, self))

    class _PlayAgainButton(discord.ui.Button):
        def __init__(self, uid: int, parent: "PlayAgainView"):
            super().__init__(label="🔄 Play Again", style=discord.ButtonStyle.success)
            self.uid    = uid
            self.parent = parent

        async def callback(self, interaction: discord.Interaction):
            if not await self.parent._check(interaction):
                return
            set_sess(self.uid, {})
            cat_view = CategoryView(self.uid)
            await interaction.response.edit_message(
                content="**Select categories:**\n(tap to toggle, then Confirm)",
                view=cat_view,
            )
            self.parent.stop()

    # ── Client + commands ─────────────────────────────────────────────────────
    intents = discord.Intents.default()
    client  = discord.Client(intents=intents)
    tree    = app_commands.CommandTree(client)

    @tree.command(name="play", description="Start a FAANG coding interview prep session")
    async def play_cmd(interaction: discord.Interaction):
        set_sess(interaction.user.id, {})
        cat_view = CategoryView(interaction.user.id)
        await interaction.response.send_message(
            "**FAANG Coding Interview Prep**\n\n"
            "**Select categories:**\n(tap to toggle, then Confirm)",
            view=cat_view,
        )

    @tree.command(name="shutdown", description="Shut down the bot (owner only)")
    async def shutdown_cmd(interaction: discord.Interaction):
        if DISCORD_OWNER_ID and interaction.user.id != DISCORD_OWNER_ID:
            await interaction.response.send_message("❌ Not authorized.", ephemeral=True)
            return
        await interaction.response.send_message("🛑 Shutting down...")
        await client.close()

    @client.event
    async def on_ready():
        await tree.sync()
        live = ("Yes (API key set)" if (ANTHROPIC_API_KEY and ANTHROPIC_AVAILABLE)
                else "No (offline mode)")
        bank_total = sum(len(v) for v in OFFLINE_BANK.values())
        print(f"\nFANG Interview Prep — Discord Bot")
        print(f"  Logged in as: {client.user}")
        print(f"  Commands synced: /play, /shutdown")
        print(f"  Question bank: {bank_total} offline questions")
        print(f"  Live AI questions: {live}")
        print(f"  Press Ctrl+C to stop\n")

    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    run_discord()
