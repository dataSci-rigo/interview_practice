#!/usr/bin/env python3
"""
FAANG Coding Interview Prep
============================
  python leet_practice.py                    # desktop app (default)
  python leet_practice.py --mode flask       # web server on local network
  python leet_practice.py --mode telegram    # Telegram bot

Shutdown:
  Flask:    visit /shutdown in the browser
  Telegram: /shutdown  (owner only; requires OWNER_CHAT_ID in .env)
  App:      close the window
"""

import argparse
import asyncio
import html as _html_mod
import json
import os
import random
import secrets
import signal
import socket
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
ANTHROPIC_API_KEY        = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_TOKEN           = os.environ.get("TELEGRAM_TOKEN", "")
OWNER_CHAT_ID            = int(os.environ.get("OWNER_CHAT_ID", "0") or "0")
FLASK_PORT               = int(os.environ.get("FLASK_PORT", "8022"))
GROUP_CLAWBOT_CHAT_ID    = int(os.environ.get("GROUP_CLAWBOT_CHAT_ID", "0") or "0")
CLAWBOT_LEETCODE_TOPIC   = 14  # t.me/c/3807732966/14
CACHE_FILE               = Path(__file__).parent / "questions_cache.json"
MODEL                    = "claude-sonnet-4-6"

try:
    import anthropic as _anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
CATEGORIES      = ["Algorithms", "Complexity", "Errors"]
DIFFICULTIES    = ["easy", "medium", "hard"]
QUESTION_COUNTS = [5, 10, 15, 20]

C = {
    "bg":         "#111827",
    "card":       "#1f2937",
    "border":     "#374151",
    "border_hi":  "#4b5563",
    "green":      "#22c55e",
    "green_hi":   "#4ade80",
    "green_dark": "#16a34a",
    "red":        "#ef4444",
    "red_bg":     "#451a1a",
    "green_bg":   "#14532d",
    "text":       "#ffffff",
    "muted":      "#d1d5db",
    "muted2":     "#9ca3af",
    "black":      "#000000",
}
FONT = "Helvetica"

# ── Offline question bank ─────────────────────────────────────────────────────
# Loaded lazily from questions_cache.json; falls back to empty dict if the file
# is missing or unreadable.
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


# Legacy hard-coded bank kept only as a fallback constant (still used below).
_FALLBACK_BANK = {}

OFFLINE_BANK = _load_offline_bank()


# ── Question cache (JSON) ─────────────────────────────────────────────────────
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


# ── Mode: Desktop App (tkinter) ───────────────────────────────────────────────
def run_app():
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError:
        print("tkinter not available. Install: sudo apt-get install python3-tk")
        sys.exit(1)

    class ScrollFrame(tk.Frame):
        def __init__(self, parent):
            super().__init__(parent, bg=C["bg"])
            self.canvas = tk.Canvas(self, bg=C["bg"], highlightthickness=0)
            self.scrollbar = tk.Scrollbar(self, orient="vertical",
                                           command=self.canvas.yview)
            self.body = tk.Frame(self.canvas, bg=C["bg"])
            self.body.bind(
                "<Configure>",
                lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
            self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
            self.canvas.bind(
                "<Configure>",
                lambda e: self.canvas.itemconfig(self._win, width=e.width))
            self.canvas.configure(yscrollcommand=self.scrollbar.set)
            self.canvas.pack(side="left", fill="both", expand=True)
            self.scrollbar.pack(side="right", fill="y")
            self.canvas.bind_all("<MouseWheel>", self._on_wheel)
            self.canvas.bind_all("<Button-4>", self._on_wheel)
            self.canvas.bind_all("<Button-5>", self._on_wheel)

        def _on_wheel(self, event):
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")
            else:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    class TriviaApp:
        def __init__(self, root):
            self.root = root
            self.root.title("FAANG Coding Interview Prep")
            self.root.geometry("820x860")
            self.root.minsize(640, 600)
            self.root.configure(bg=C["bg"])

            self.selected_categories = set()
            self.difficulty = "medium"
            self.num_questions = 5
            self.offline_mode = tk.BooleanVar(value=not ANTHROPIC_AVAILABLE)
            self.api_key_var = tk.StringVar(value=ANTHROPIC_API_KEY)

            self.questions = []
            self.current = 0
            self.selected_answer = None
            self.show_answer = False
            self.score = 0
            self.answers = []

            self.container = tk.Frame(self.root, bg=C["bg"])
            self.container.pack(fill="both", expand=True)
            self.show_setup()

        def clear(self):
            for child in self.container.winfo_children():
                child.destroy()

        def make_button(self, parent, text, command, selected=False, big=False):
            return tk.Button(
                parent, text=text, command=command,
                bg=C["green"] if selected else C["card"],
                fg=C["black"] if selected else C["text"],
                activebackground=C["green_hi"], activeforeground=C["black"],
                relief="flat", bd=0, font=(FONT, 14 if big else 12, "bold"),
                padx=16, pady=12, cursor="hand2", highlightthickness=2,
                highlightbackground=C["green_hi"] if selected else C["border"])

        def show_setup(self):
            self.clear()
            wrap = ScrollFrame(self.container)
            wrap.pack(fill="both", expand=True)
            pad = tk.Frame(wrap.body, bg=C["bg"])
            pad.pack(fill="x", padx=40, pady=20)

            tk.Label(pad, text="FAANG Coding Interview Prep", bg=C["bg"],
                     fg=C["green_hi"], font=(FONT, 28, "bold")).pack(pady=(10, 4))
            tk.Label(pad, text="Practice algorithms, complexity analysis, and debugging.",
                     bg=C["bg"], fg=C["muted"], font=(FONT, 12)).pack(pady=(0, 20))

            card = tk.Frame(pad, bg=C["card"], highlightthickness=2,
                            highlightbackground=C["border"])
            card.pack(fill="x")
            inner = tk.Frame(card, bg=C["card"])
            inner.pack(fill="x", padx=24, pady=24)

            tk.Label(inner, text="Categories", bg=C["card"], fg=C["green_hi"],
                     font=(FONT, 16, "bold")).pack(anchor="w", pady=(0, 8))
            cat_row = tk.Frame(inner, bg=C["card"])
            cat_row.pack(fill="x", pady=(0, 18))
            self.cat_buttons = {}
            for cat in CATEGORIES:
                b = self.make_button(cat_row, cat, lambda c=cat: self.toggle_category(c))
                b.pack(side="left", padx=6, fill="x", expand=True)
                self.cat_buttons[cat] = b

            tk.Label(inner, text="Difficulty", bg=C["card"], fg=C["green_hi"],
                     font=(FONT, 16, "bold")).pack(anchor="w", pady=(0, 8))
            diff_row = tk.Frame(inner, bg=C["card"])
            diff_row.pack(fill="x", pady=(0, 18))
            self.diff_buttons = {}
            for d in DIFFICULTIES:
                b = self.make_button(diff_row, d.capitalize(),
                                     lambda dd=d: self.set_difficulty(dd),
                                     selected=(d == self.difficulty))
                b.pack(side="left", padx=6, fill="x", expand=True)
                self.diff_buttons[d] = b

            tk.Label(inner, text="Number of questions", bg=C["card"],
                     fg=C["green_hi"], font=(FONT, 16, "bold")).pack(anchor="w", pady=(0, 8))
            num_row = tk.Frame(inner, bg=C["card"])
            num_row.pack(fill="x", pady=(0, 18))
            self.num_buttons = {}
            for n in QUESTION_COUNTS:
                b = self.make_button(num_row, str(n), lambda nn=n: self.set_num(nn),
                                     selected=(n == self.num_questions))
                b.pack(side="left", padx=6, fill="x", expand=True)
                self.num_buttons[n] = b

            tk.Label(inner, text="Question source", bg=C["card"], fg=C["green_hi"],
                     font=(FONT, 16, "bold")).pack(anchor="w", pady=(4, 8))
            tk.Checkbutton(inner, text="Offline mode (use built-in question bank)",
                           variable=self.offline_mode, command=self._toggle_offline,
                           bg=C["card"], fg=C["muted"], selectcolor=C["border"],
                           activebackground=C["card"], activeforeground=C["text"],
                           font=(FONT, 11), anchor="w").pack(anchor="w")

            key_row = tk.Frame(inner, bg=C["card"])
            key_row.pack(fill="x", pady=(8, 4))
            tk.Label(key_row, text="Anthropic API key:", bg=C["card"], fg=C["muted"],
                     font=(FONT, 11)).pack(side="left")
            self.key_entry = tk.Entry(key_row, textvariable=self.api_key_var, show="*",
                                      bg=C["bg"], fg=C["text"], insertbackground=C["text"],
                                      relief="flat", font=(FONT, 11))
            self.key_entry.pack(side="left", fill="x", expand=True, padx=(8, 0), ipady=4)

            hint = ("Live generation needs the 'anthropic' package and a key. "
                    "Without them the app runs offline.")
            if not ANTHROPIC_AVAILABLE:
                hint = ("'anthropic' is not installed — running offline. "
                        "Run 'pip install anthropic' for live questions.")
            tk.Label(inner, text=hint, bg=C["card"], fg=C["muted2"], font=(FONT, 9),
                     wraplength=620, justify="left").pack(anchor="w", pady=(4, 16))

            tk.Button(inner, text="Start game", command=self.start_game,
                      bg=C["green"], fg=C["black"], activebackground=C["green_hi"],
                      activeforeground=C["black"], relief="flat", bd=0,
                      font=(FONT, 16, "bold"), pady=14, cursor="hand2").pack(fill="x")
            self._toggle_offline()

        def _toggle_offline(self):
            if hasattr(self, "key_entry"):
                self.key_entry.configure(
                    state="disabled" if self.offline_mode.get() else "normal")

        def toggle_category(self, cat):
            if cat in self.selected_categories:
                self.selected_categories.discard(cat)
            else:
                self.selected_categories.add(cat)
            sel = cat in self.selected_categories
            self.cat_buttons[cat].configure(
                bg=C["green"] if sel else C["card"],
                fg=C["black"] if sel else C["text"],
                highlightbackground=C["green_hi"] if sel else C["border"])

        def set_difficulty(self, d):
            self.difficulty = d
            for k, b in self.diff_buttons.items():
                sel = k == d
                b.configure(bg=C["green"] if sel else C["card"],
                            fg=C["black"] if sel else C["text"],
                            highlightbackground=C["green_hi"] if sel else C["border"])

        def set_num(self, n):
            self.num_questions = n
            for k, b in self.num_buttons.items():
                sel = k == n
                b.configure(bg=C["green"] if sel else C["card"],
                            fg=C["black"] if sel else C["text"],
                            highlightbackground=C["green_hi"] if sel else C["border"])

        def start_game(self):
            if not self.selected_categories:
                messagebox.showwarning("No categories",
                                       "Please select at least one category!")
                return
            categories = [c for c in CATEGORIES if c in self.selected_categories]

            if self.offline_mode.get() or not ANTHROPIC_AVAILABLE:
                self.questions = generate_questions_offline(
                    categories, self.difficulty, self.num_questions)
                if not self.questions:
                    messagebox.showerror("No questions",
                                         "No offline questions for that selection.")
                    return
                self._begin_play()
                return

            key = self.api_key_var.get().strip()
            if not key:
                messagebox.showwarning(
                    "API key needed",
                    "Enter your Anthropic API key, or check 'Offline mode'.")
                return
            self.show_loading()
            threading.Thread(
                target=self._generate_thread,
                args=(key, categories, self.difficulty, self.num_questions),
                daemon=True).start()

        def _generate_thread(self, key, categories, difficulty, num):
            try:
                qs = generate_questions_live(key, categories, difficulty, num)
                self.root.after(0, lambda: self._on_questions_ready(qs))
            except Exception as exc:
                self.root.after(0, lambda e=exc: self._on_generation_error(e))

        def _on_questions_ready(self, questions):
            self.questions = questions
            self._begin_play()

        def _on_generation_error(self, exc):
            messagebox.showerror(
                "Generation failed",
                f"Failed to generate questions:\n{exc}\n\nTry again or use Offline mode.")
            self.show_setup()

        def show_loading(self):
            self.clear()
            frame = tk.Frame(self.container, bg=C["bg"])
            frame.pack(expand=True)
            tk.Label(frame, text="Generating questions...", bg=C["bg"],
                     fg=C["green_hi"], font=(FONT, 20, "bold")).pack(pady=8)
            tk.Label(frame, text="Preparing your interview challenge", bg=C["bg"],
                     fg=C["muted"], font=(FONT, 12)).pack()

        def _begin_play(self):
            self.current = 0
            self.score = 0
            self.answers = []
            self.selected_answer = None
            self.show_answer = False
            self.show_playing()

        def show_playing(self):
            self.clear()
            q = self.questions[self.current]
            wrap = ScrollFrame(self.container)
            wrap.pack(fill="both", expand=True)
            pad = tk.Frame(wrap.body, bg=C["bg"])
            pad.pack(fill="x", padx=40, pady=20)

            head = tk.Frame(pad, bg=C["bg"])
            head.pack(fill="x", pady=(4, 12))
            tk.Label(head, text=f"Question {self.current + 1} of {len(self.questions)}",
                     bg=C["bg"], fg=C["green_hi"], font=(FONT, 13, "bold")).pack(side="left")
            tk.Label(head, text=f"Score: {self.score}", bg=C["bg"], fg=C["green_hi"],
                     font=(FONT, 13, "bold")).pack(side="right")

            card = tk.Frame(pad, bg=C["card"], highlightthickness=2,
                            highlightbackground=C["border"])
            card.pack(fill="x")
            inner = tk.Frame(card, bg=C["card"])
            inner.pack(fill="both", padx=24, pady=24)

            tk.Label(inner, text=f"  {q.get('category', '')}  ", bg=C["green"],
                     fg=C["black"], font=(FONT, 10, "bold")).pack(anchor="w", pady=(0, 12))

            has_code = "\n" in q["question"]
            q_font = ("Courier New", 12) if has_code else (FONT, 15, "bold")
            tk.Label(inner, text=q["question"], bg=C["card"], fg=C["text"], font=q_font,
                     justify="left", wraplength=660).pack(anchor="w", pady=(0, 16))

            self.option_buttons = []
            for i, opt in enumerate(q["options"]):
                b = tk.Button(inner, text=f"{chr(65 + i)}.  {opt}", anchor="w",
                              justify="left", command=lambda idx=i: self.select_answer(idx),
                              bg=C["border"], fg=C["text"], activebackground=C["border_hi"],
                              activeforeground=C["text"], relief="flat", bd=0,
                              font=(FONT, 12, "bold"), padx=16, pady=12, cursor="hand2",
                              highlightthickness=2, highlightbackground=C["border_hi"],
                              wraplength=620)
                b.pack(fill="x", pady=5)
                self.option_buttons.append(b)
            self._refresh_options()

            if not self.show_answer:
                label = "Check answer"
            elif self.current + 1 == len(self.questions):
                label = "Finish game"
            else:
                label = "Next question"
            self.action_btn = tk.Button(inner, text=label, command=self.advance,
                                        bg=C["green"], fg=C["black"],
                                        activebackground=C["green_hi"],
                                        activeforeground=C["black"], relief="flat", bd=0,
                                        font=(FONT, 15, "bold"), pady=14, cursor="hand2")
            self.action_btn.pack(fill="x", pady=(16, 0))

        def _refresh_options(self):
            q = self.questions[self.current]
            for i, b in enumerate(self.option_buttons):
                if self.show_answer:
                    b.configure(state="disabled")
                    if is_correct(q, i):
                        b.configure(bg=C["green"], fg=C["black"],
                                    disabledforeground=C["black"],
                                    highlightbackground=C["green_hi"])
                    elif i == self.selected_answer:
                        b.configure(bg=C["red"], fg=C["text"],
                                    disabledforeground=C["text"],
                                    highlightbackground=C["red"])
                    else:
                        b.configure(bg=C["border_hi"], fg=C["muted"],
                                    disabledforeground=C["muted"],
                                    highlightbackground=C["border_hi"])
                else:
                    sel = i == self.selected_answer
                    b.configure(state="normal",
                                bg=C["green"] if sel else C["border"],
                                fg=C["black"] if sel else C["text"],
                                highlightbackground=C["green_hi"] if sel else C["border_hi"])

        def select_answer(self, idx):
            if self.show_answer:
                return
            self.selected_answer = idx
            self._refresh_options()

        def advance(self):
            if self.selected_answer is None:
                messagebox.showwarning("No answer", "Please select an answer!")
                return
            if not self.show_answer:
                self.show_answer = True
                self._refresh_options()
                self.action_btn.configure(
                    text="Finish game" if self.current + 1 == len(self.questions)
                    else "Next question")
                return

            q = self.questions[self.current]
            correct = is_correct(q, self.selected_answer)
            self.answers.append({"question_index": self.current,
                                 "selected": self.selected_answer, "correct": correct})
            if correct:
                self.score += 1
            if self.current + 1 < len(self.questions):
                self.current += 1
                self.selected_answer = None
                self.show_answer = False
                self.show_playing()
            else:
                self.show_results()

        def show_results(self):
            self.clear()
            total = len(self.questions)
            pct = round((self.score / total) * 100) if total else 0
            wrap = ScrollFrame(self.container)
            wrap.pack(fill="both", expand=True)
            pad = tk.Frame(wrap.body, bg=C["bg"])
            pad.pack(fill="x", padx=40, pady=20)

            tk.Label(pad, text="Results", bg=C["bg"], fg=C["green_hi"],
                     font=(FONT, 30, "bold")).pack(pady=(8, 16))
            card = tk.Frame(pad, bg=C["card"], highlightthickness=2,
                            highlightbackground=C["border"])
            card.pack(fill="x")
            inner = tk.Frame(card, bg=C["card"])
            inner.pack(fill="both", padx=24, pady=24)

            tk.Label(inner, text=f"{self.score}/{total}", bg=C["card"], fg=C["green_hi"],
                     font=(FONT, 40, "bold")).pack()
            tk.Label(inner, text=f"{pct}% correct", bg=C["card"], fg=C["text"],
                     font=(FONT, 18, "bold")).pack(pady=(0, 8))
            msg = ("Excellent!" if pct >= 80 else "Good job!" if pct >= 60
                   else "Not bad!" if pct >= 40 else "Keep studying!")
            tk.Label(inner, text=msg, bg=C["card"], fg=C["muted"],
                     font=(FONT, 14)).pack(pady=(0, 16))

            for i, q in enumerate(self.questions):
                ans = self.answers[i] if i < len(self.answers) else None
                correct = ans["correct"] if ans else False
                box = tk.Frame(inner, bg=C["green_bg"] if correct else C["red_bg"],
                               highlightthickness=2,
                               highlightbackground=C["green_hi"] if correct else C["red"])
                box.pack(fill="x", pady=6)
                ib = tk.Frame(box, bg=box["bg"])
                ib.pack(fill="x", padx=12, pady=10)
                qtext = q["question"].split("\n")[0]
                tk.Label(ib, text=f"Q{i + 1}. {qtext}", bg=box["bg"], fg=C["text"],
                         font=(FONT, 11, "bold"), justify="left", wraplength=640,
                         anchor="w").pack(anchor="w")
                your = q["options"][ans["selected"]] if ans is not None else "-"
                tk.Label(ib, text=f"Your answer: {your}", bg=box["bg"],
                         fg=C["green_hi"] if correct else C["red"], font=(FONT, 10),
                         justify="left", wraplength=640, anchor="w").pack(anchor="w")
                if not correct:
                    tk.Label(ib, text=f"Correct answer: {correct_text(q)}", bg=box["bg"],
                             fg=C["green_hi"], font=(FONT, 10), justify="left",
                             wraplength=640, anchor="w").pack(anchor="w")

            tk.Button(inner, text="Play again", command=self.show_setup, bg=C["green"],
                      fg=C["black"], activebackground=C["green_hi"],
                      activeforeground=C["black"], relief="flat", bd=0,
                      font=(FONT, 15, "bold"), pady=14, cursor="hand2").pack(
                fill="x", pady=(16, 0))

    root = tk.Tk()
    TriviaApp(root)
    root.mainloop()


# ── Mode: Flask web server ────────────────────────────────────────────────────
_BASE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111827; color: #fff; font-family: Helvetica, Arial, sans-serif; min-height: 100vh; }
.container { max-width: 820px; margin: 0 auto; padding: 2rem 1.5rem; }
h1 { color: #4ade80; font-size: 2rem; margin-bottom: .4rem; }
.subtitle { color: #d1d5db; margin-bottom: 2rem; font-size: 1rem; }
.card { background: #1f2937; border: 1px solid #374151; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; }
.section-label { color: #4ade80; font-weight: bold; font-size: 1rem; margin-bottom: .6rem; }
.btn-group { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.btn { padding: .55rem 1.1rem; border: 2px solid #374151; border-radius: 6px; background: #1f2937;
       color: #fff; font-size: .92rem; font-weight: 600; cursor: pointer; transition: all .12s;
       text-decoration: none; display: inline-block; }
.btn:hover { border-color: #4ade80; background: #374151; }
.btn.sel { background: #22c55e; color: #000; border-color: #4ade80; }
.btn-primary { background: #22c55e; color: #000; border-color: #22c55e; padding: .8rem;
               font-size: 1rem; width: 100%; text-align: center; border-radius: 8px; font-weight: 700; }
.btn-primary:hover { background: #4ade80; border-color: #4ade80; }
.btn-danger { background: #ef4444; color: #fff; border-color: #ef4444; }
.btn-danger:hover { background: #f87171; }
input[type=password], input[type=text] { background: #111827; color: #fff; border: 1px solid #374151;
  border-radius: 6px; padding: .45rem .75rem; font-size: .92rem; width: 100%; }
input:focus { outline: none; border-color: #4ade80; }
.badge { display: inline-block; background: #22c55e; color: #000; font-size: .72rem;
         font-weight: bold; padding: .18rem .55rem; border-radius: 4px; margin-bottom: .7rem; }
.opt-btn { display: block; width: 100%; text-align: left; padding: .7rem 1rem;
           margin-bottom: .45rem; background: #374151; border: 2px solid #4b5563;
           border-radius: 8px; color: #fff; font-size: .92rem; font-weight: 600;
           cursor: pointer; transition: all .12s; }
.opt-btn:hover { border-color: #4ade80; background: #4b5563; }
.opt-btn.sel { background: #22c55e; color: #000; border-color: #4ade80; }
.opt-btn.correct { background: #14532d; border-color: #22c55e; color: #4ade80; }
.opt-btn.wrong { background: #451a1a; border-color: #ef4444; color: #ef4444; }
.opt-static { display: block; padding: .7rem 1rem; margin-bottom: .45rem;
              border: 2px solid #4b5563; border-radius: 8px; font-size: .92rem; font-weight: 600; }
.progress-bar { display: flex; justify-content: space-between; align-items: center;
                margin-bottom: 1rem; color: #9ca3af; font-size: .9rem; }
.score-big { font-size: 3rem; font-weight: bold; color: #4ade80; text-align: center; padding: 1rem 0; }
.pct-text { font-size: 1.2rem; color: #fff; text-align: center; margin-bottom: .5rem; }
.result-item { padding: .7rem 1rem; border-radius: 8px; margin-bottom: .45rem; }
.res-ok { background: #14532d; border: 1px solid #22c55e; }
.res-bad { background: #451a1a; border: 1px solid #ef4444; }
.q-line { font-weight: bold; margin-bottom: .2rem; font-size: .95rem; }
.a-line { font-size: .88rem; color: #9ca3af; }
.a-correct { color: #4ade80; }
pre { background: #0d1117; color: #e6edf3; padding: 1rem; border-radius: 6px;
      overflow-x: auto; font-size: .82rem; line-height: 1.5; margin-bottom: 1rem; }
.check-row { display: flex; align-items: center; gap: .5rem; margin-bottom: 1rem; }
.check-row input[type=checkbox] { width: 16px; height: 16px; accent-color: #22c55e; }
.loading { text-align: center; padding: 5rem 2rem; }
.loading h2 { font-size: 1.6rem; color: #4ade80; margin-bottom: 1rem; }
.loading p { color: #9ca3af; }
.row-end { text-align: right; margin-top: .75rem; }
"""

_SETUP_JS = """
<script>
let selCats = new Set();
function toggleCat(cat) {
    selCats.has(cat) ? selCats.delete(cat) : selCats.add(cat);
    document.getElementById('cat-' + cat).classList.toggle('sel', selCats.has(cat));
    document.getElementById('cats-input').value = [...selCats].join(',');
}
function setDiff(d) {
    document.querySelectorAll('.diff-btn').forEach(b => b.classList.remove('sel'));
    document.getElementById('diff-' + d).classList.add('sel');
    document.getElementById('diff-input').value = d;
}
function setCount(n) {
    document.querySelectorAll('.cnt-btn').forEach(b => b.classList.remove('sel'));
    document.getElementById('cnt-' + n).classList.add('sel');
    document.getElementById('cnt-input').value = n;
}
function toggleOffline(cb) {
    document.getElementById('key-row').style.display = cb.checked ? 'none' : '';
}
document.getElementById('setup-form').addEventListener('submit', function(e) {
    if (selCats.size === 0) { e.preventDefault(); alert('Please select at least one category!'); }
});
</script>
"""

_PLAY_JS = """
<script>
function selectOpt(i) {
    document.querySelectorAll('.opt-btn').forEach(b => b.classList.remove('sel'));
    document.querySelectorAll('.opt-btn')[i].classList.add('sel');
    document.getElementById('sel-input').value = i;
}
</script>
"""


def _flask_page(content: str, extra_head: str = "") -> str:
    return (
        "<!DOCTYPE html><html lang='en'><head>"
        "<meta charset='UTF-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>FAANG Interview Prep</title>"
        f"<style>{_BASE_CSS}</style>{extra_head}"
        "</head><body><div class='container'>"
        f"{content}"
        "</div></body></html>"
    )


def run_flask():
    try:
        from flask import Flask, request, session, redirect
    except ImportError:
        print("Flask not installed. Run: pip install flask")
        sys.exit(1)

    app = Flask(__name__)
    app.secret_key = secrets.token_hex(32)

    _game_sessions: dict = {}
    _pending_jobs: dict  = {}
    _shutdown_event      = threading.Event()

    def get_gs() -> dict:
        tok = session.get("tok")
        return _game_sessions.get(tok, {}) if tok else {}

    def set_gs(state: dict) -> None:
        tok = session.get("tok")
        if not tok:
            tok = secrets.token_hex(16)
            session["tok"] = tok
        _game_sessions[tok] = state

    # ── routes ──
    @app.route("/")
    def index():
        cat_btns = "".join(
            f"<button type='button' class='btn' id='cat-{c}' onclick=\"toggleCat('{c}')\">{c}</button>"
            for c in CATEGORIES
        )
        diff_btns = "".join(
            f"<button type='button' class='btn diff-btn{' sel' if d=='medium' else ''}' "
            f"id='diff-{d}' onclick=\"setDiff('{d}')\">{d.capitalize()}</button>"
            for d in DIFFICULTIES
        )
        cnt_btns = "".join(
            f"<button type='button' class='btn cnt-btn{' sel' if n==5 else ''}' "
            f"id='cnt-{n}' onclick='setCount({n})'>{n}</button>"
            for n in QUESTION_COUNTS
        )
        offline_checked = "checked" if not ANTHROPIC_AVAILABLE else ""
        key_display     = "display:none" if not ANTHROPIC_AVAILABLE else ""

        content = f"""
<h1>FAANG Coding Interview Prep</h1>
<p class='subtitle'>Practice algorithms, complexity analysis, and debugging.</p>
<div class='card'>
  <form method='POST' action='/start' id='setup-form'>
    <div class='section-label'>Categories</div>
    <div class='btn-group'>{cat_btns}</div>
    <input type='hidden' name='categories' id='cats-input' value=''>

    <div class='section-label'>Difficulty</div>
    <div class='btn-group'>{diff_btns}</div>
    <input type='hidden' name='difficulty' id='diff-input' value='medium'>

    <div class='section-label'>Number of Questions</div>
    <div class='btn-group'>{cnt_btns}</div>
    <input type='hidden' name='num_questions' id='cnt-input' value='5'>

    <div class='section-label'>Question Source</div>
    <div class='check-row'>
      <input type='checkbox' id='offline-cb' name='offline' {offline_checked}
             {'disabled' if not ANTHROPIC_AVAILABLE else ''} onchange='toggleOffline(this)'>
      <label for='offline-cb' style='color:#d1d5db'>Offline mode (built-in question bank)</label>
    </div>

    <div id='key-row' style='{key_display};margin-bottom:1rem'>
      <div class='section-label'>Anthropic API Key</div>
      <input type='password' name='api_key' value='{_html_mod.escape(ANTHROPIC_API_KEY)}' placeholder='sk-ant-...'>
    </div>

    <button type='submit' class='btn btn-primary'>Start Game</button>
  </form>
</div>
<div class='row-end'>
  <a href='/shutdown' class='btn btn-danger'
     onclick="return confirm('Shut down the server?')">Shut down server</a>
</div>
""" + _SETUP_JS
        return _flask_page(content)

    @app.route("/start", methods=["POST"])
    def start():
        cats_str      = request.form.get("categories", "")
        categories    = [c for c in cats_str.split(",") if c in CATEGORIES]
        if not categories:
            return redirect("/")
        difficulty    = request.form.get("difficulty", "medium")
        num_questions = int(request.form.get("num_questions", 5))
        offline       = bool(request.form.get("offline")) or not ANTHROPIC_AVAILABLE
        api_key       = request.form.get("api_key", ANTHROPIC_API_KEY).strip()

        if offline or not api_key:
            questions = generate_questions_offline(categories, difficulty, num_questions)
            set_gs({"questions": questions, "current": 0, "score": 0,
                    "answers": [], "selected": None, "show": False})
            return redirect("/play")

        job_id = secrets.token_hex(8)
        _pending_jobs[job_id] = {"status": "pending", "questions": None, "error": None}

        def worker():
            try:
                qs = generate_questions_live(api_key, categories, difficulty, num_questions)
                _pending_jobs[job_id].update({"status": "done", "questions": qs})
            except Exception as exc:
                _pending_jobs[job_id].update({"status": "error", "error": str(exc)})

        threading.Thread(target=worker, daemon=True).start()
        session["pending_job"] = job_id
        return redirect("/loading")

    @app.route("/loading")
    def loading():
        job_id = session.get("pending_job")
        if not job_id or job_id not in _pending_jobs:
            return redirect("/")
        job = _pending_jobs[job_id]

        if job["status"] == "done":
            set_gs({"questions": job["questions"], "current": 0, "score": 0,
                    "answers": [], "selected": None, "show": False})
            del _pending_jobs[job_id]
            session.pop("pending_job", None)
            return redirect("/play")

        if job["status"] == "error":
            err = _html_mod.escape(job["error"])
            del _pending_jobs[job_id]
            session.pop("pending_job", None)
            return _flask_page(f"<div class='loading'><h2>Generation failed</h2>"
                               f"<p>{err}</p><a href='/' class='btn' style='margin-top:1.5rem'>Try again</a></div>")

        return _flask_page(
            "<div class='loading'><h2>Generating questions...</h2>"
            "<p>Preparing your interview challenge with AI ✨</p></div>",
            extra_head="<meta http-equiv='refresh' content='2'>",
        )

    @app.route("/play")
    def play():
        gs = get_gs()
        if not gs.get("questions"):
            return redirect("/")

        questions = gs["questions"]
        current   = gs["current"]
        score     = gs["score"]
        show      = gs.get("show", False)
        selected  = gs.get("selected")

        if current >= len(questions):
            return redirect("/results")

        q     = questions[current]
        total = len(questions)

        if "\n" in q["question"]:
            q_html = f"<pre>{_html_mod.escape(q['question'])}</pre>"
        else:
            q_html = f"<p style='font-size:1.05rem;font-weight:bold;line-height:1.5;margin-bottom:1rem'>{_html_mod.escape(q['question'])}</p>"

        opts_html = ""
        if show:
            for i, opt in enumerate(q["options"]):
                if is_correct(q, i):
                    cls = "opt-static correct"
                elif i == selected:
                    cls = "opt-static wrong"
                else:
                    cls = "opt-static"
                    cls += ";color:#6b7280;border-color:#374151"
                opts_html += f"<div class='{cls}'>{chr(65+i)}. {_html_mod.escape(opt)}</div>"
            if current + 1 == total:
                action_label = "Finish Game"
            else:
                action_label = "Next Question →"
            form = (f"<form method='POST' action='/answer'>"
                    f"<input type='hidden' name='action' value='next'>"
                    f"<button type='submit' class='btn btn-primary' style='margin-top:1rem'>{action_label}</button>"
                    f"</form>")
        else:
            for i, opt in enumerate(q["options"]):
                sel_cls = " sel" if i == selected else ""
                opts_html += (
                    f"<button type='button' class='opt-btn{sel_cls}' onclick='selectOpt({i})'>"
                    f"{chr(65+i)}. {_html_mod.escape(opt)}</button>"
                )
            form = (f"<form method='POST' action='/answer' id='ans-form'>"
                    f"<input type='hidden' name='action' value='check'>"
                    f"<input type='hidden' name='selected' id='sel-input' value='{'' if selected is None else selected}'>"
                    f"{opts_html}"
                    f"<button type='submit' class='btn btn-primary' style='margin-top:1rem'>Check Answer</button>"
                    f"</form>")
            opts_html = ""  # included in form

        content = f"""
<div class='progress-bar'>
  <span>Question {current + 1} of {total}</span>
  <span style='color:#4ade80;font-weight:bold'>Score: {score}</span>
</div>
<div class='card'>
  <span class='badge'>{_html_mod.escape(q.get('category',''))}</span>
  {q_html}
  {opts_html if show else ''}
  {form}
</div>
""" + (_PLAY_JS if not show else "")
        return _flask_page(content)

    @app.route("/answer", methods=["POST"])
    def answer():
        gs = get_gs()
        if not gs.get("questions"):
            return redirect("/")

        action = request.form.get("action")

        if action == "check":
            sel_str = request.form.get("selected", "")
            if sel_str == "":
                return redirect("/play")
            gs["selected"] = int(sel_str)
            gs["show"]     = True
            set_gs(gs)

        elif action == "next":
            sel     = gs.get("selected")
            q       = gs["questions"][gs["current"]]
            correct = is_correct(q, sel) if sel is not None else False
            gs["answers"].append({"question_index": gs["current"],
                                  "selected": sel, "correct": correct})
            if correct:
                gs["score"] += 1
            gs["current"] += 1
            gs["show"]     = False
            gs["selected"] = None
            set_gs(gs)
            if gs["current"] >= len(gs["questions"]):
                return redirect("/results")

        return redirect("/play")

    @app.route("/results")
    def results():
        gs = get_gs()
        if not gs.get("questions"):
            return redirect("/")

        questions = gs["questions"]
        answers   = gs["answers"]
        score     = gs["score"]
        total     = len(questions)
        pct       = round((score / total) * 100) if total else 0
        msg       = ("Excellent!" if pct >= 80 else "Good job!" if pct >= 60
                     else "Not bad!" if pct >= 40 else "Keep studying!")

        items = ""
        for i, q in enumerate(questions):
            ans     = answers[i] if i < len(answers) else None
            correct = ans["correct"] if ans else False
            your    = _html_mod.escape(q["options"][ans["selected"]]) if ans else "-"
            first   = _html_mod.escape(q["question"].split("\n")[0])
            icon    = "✅" if correct else "❌"
            cls     = "res-ok" if correct else "res-bad"
            wrong   = (f"<div class='a-line'><span class='a-correct'>✓ {_html_mod.escape(correct_text(q))}</span></div>"
                       if not correct else "")
            items += (f"<div class='result-item {cls}'>"
                      f"<div class='q-line'>{icon} Q{i+1}. {first}</div>"
                      f"<div class='a-line'>Your answer: {your}</div>"
                      f"{wrong}</div>")

        content = f"""
<h1>Results</h1>
<div class='card'>
  <div class='score-big'>{score}/{total}</div>
  <div class='pct-text'>{pct}% correct &mdash; {msg}</div>
  <div style='margin:1.5rem 0'>{items}</div>
  <a href='/' class='btn btn-primary'>Play Again</a>
</div>
<div class='row-end'>
  <a href='/shutdown' class='btn btn-danger'
     onclick="return confirm('Shut down the server?')">Shut down server</a>
</div>
"""
        return _flask_page(content)

    @app.route("/shutdown")
    def shutdown():
        _shutdown_event.set()
        return _flask_page(
            "<div class='loading'><h2>Server shutting down...</h2>"
            "<p>You can close this tab.</p></div>"
        )

    # ── start server in thread, main thread monitors shutdown ──
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "127.0.0.1"

    print(f"\nFANG Interview Prep — Flask Server")
    print(f"  Local:   http://localhost:{FLASK_PORT}")
    print(f"  Network: http://{local_ip}:{FLASK_PORT}")
    print(f"  Shutdown: http://{local_ip}:{FLASK_PORT}/shutdown  or  Ctrl+C\n")

    server_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=FLASK_PORT,
                               debug=False, use_reloader=False),
        daemon=True,
    )
    server_thread.start()

    try:
        _shutdown_event.wait()
    except KeyboardInterrupt:
        pass

    print("\nFlask server stopped.")
    sys.exit(0)


# ── Mode: Telegram bot ────────────────────────────────────────────────────────
def run_telegram():
    try:
        from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
        from telegram.ext import (
            Application, CommandHandler, CallbackQueryHandler,
            ConversationHandler, ContextTypes,
        )
        from telegram.constants import ParseMode
    except ImportError:
        print("python-telegram-bot not installed. Run: pip install python-telegram-bot")
        sys.exit(1)

    if not TELEGRAM_TOKEN:
        print("TELEGRAM_TOKEN not set in .env")
        sys.exit(1)

    CHOOSING_CATS, CHOOSING_DIFF, CHOOSING_COUNT, PLAYING, RESULTS = range(5)

    # ── keyboard builders ──
    def kb_cats(selected: set) -> InlineKeyboardMarkup:
        row = [InlineKeyboardButton(
                   f"✅ {c}" if c in selected else c,
                   callback_data=f"cat:{c}")
               for c in CATEGORIES]
        return InlineKeyboardMarkup([row, [InlineKeyboardButton("Confirm ✓", callback_data="cat_done")]])

    def kb_diff() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([[
            InlineKeyboardButton(d.capitalize(), callback_data=f"diff:{d}")
            for d in DIFFICULTIES
        ]])

    def kb_count() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([[
            InlineKeyboardButton(str(n), callback_data=f"count:{n}")
            for n in QUESTION_COUNTS
        ]])

    def kb_answer() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([[
            InlineKeyboardButton(chr(65 + i), callback_data=f"ans:{i}")
            for i in range(4)
        ]])

    def fmt_question(q: dict, idx: int, total: int, score: int) -> str:
        cat  = _html_mod.escape(q.get("category", ""))
        text = f"<b>Q{idx+1}/{total}</b> · <i>{cat}</i>\n\n"
        qtext = q["question"]
        if "\n" in qtext:
            text += f"<pre>{_html_mod.escape(qtext)}</pre>\n"
        else:
            text += f"{_html_mod.escape(qtext)}\n\n"
        for i, opt in enumerate(q["options"]):
            text += f"{chr(65+i)}. {_html_mod.escape(opt)}\n"
        text += f"\n<i>Score: {score}</i>"
        return text

    # ── handlers ──
    async def play_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        context.user_data.clear()
        context.user_data["cats"] = set()
        await update.message.reply_text(
            "<b>Select categories:</b>\n(tap to toggle, then Confirm)",
            reply_markup=kb_cats(set()),
            parse_mode=ParseMode.HTML,
        )
        return CHOOSING_CATS

    async def toggle_cat(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        cat     = query.data.split(":", 1)[1]
        selected = context.user_data.get("cats", set())
        selected.discard(cat) if cat in selected else selected.add(cat)
        context.user_data["cats"] = selected
        await query.edit_message_reply_markup(reply_markup=kb_cats(selected))
        return CHOOSING_CATS

    async def confirm_cats(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query    = update.callback_query
        selected = context.user_data.get("cats", set())
        if not selected:
            await query.answer("Select at least one category!", show_alert=True)
            return CHOOSING_CATS
        await query.answer()
        cats_str = ", ".join(selected)
        await query.edit_message_text(
            f"✅ Categories: <b>{_html_mod.escape(cats_str)}</b>\n\n<b>Select difficulty:</b>",
            reply_markup=kb_diff(),
            parse_mode=ParseMode.HTML,
        )
        return CHOOSING_DIFF

    async def choose_diff(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        diff  = query.data.split(":", 1)[1]
        context.user_data["diff"] = diff
        await query.edit_message_text(
            f"✅ Difficulty: <b>{_html_mod.escape(diff.capitalize())}</b>\n\n<b>How many questions?</b>",
            reply_markup=kb_count(),
            parse_mode=ParseMode.HTML,
        )
        return CHOOSING_COUNT

    async def choose_count(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        count = int(query.data.split(":", 1)[1])
        cats  = list(context.user_data.get("cats", []))
        diff  = context.user_data.get("diff", "medium")

        context.user_data["count"] = count
        await query.edit_message_text("⏳ <b>Generating questions...</b>", parse_mode=ParseMode.HTML)

        if ANTHROPIC_API_KEY and ANTHROPIC_AVAILABLE:
            try:
                questions = await asyncio.to_thread(
                    generate_questions_live, ANTHROPIC_API_KEY, cats, diff, count
                )
            except Exception as exc:
                await query.edit_message_text(
                    f"❌ Generation failed: {_html_mod.escape(str(exc))}\n\nFalling back to offline questions.",
                    parse_mode=ParseMode.HTML,
                )
                questions = generate_questions_offline(cats, diff, count)
        else:
            questions = generate_questions_offline(cats, diff, count)

        if not questions:
            await query.edit_message_text("❌ No questions available for that selection.")
            return ConversationHandler.END

        context.user_data.update({"questions": questions, "current": 0,
                                   "score": 0, "answers": []})
        await query.edit_message_text(
            fmt_question(questions[0], 0, len(questions), 0),
            reply_markup=kb_answer(),
            parse_mode=ParseMode.HTML,
        )
        return PLAYING

    async def handle_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query     = update.callback_query
        await query.answer()
        ans_idx   = int(query.data.split(":", 1)[1])
        questions = context.user_data["questions"]
        current   = context.user_data["current"]
        score     = context.user_data["score"]
        answers   = context.user_data["answers"]
        q         = questions[current]

        correct = is_correct(q, ans_idx)
        if correct:
            score += 1
        answers.append({"question_index": current, "selected": ans_idx, "correct": correct})
        context.user_data["score"]   = score
        context.user_data["answers"] = answers

        icon     = "✅" if correct else "❌"
        your_ans = _html_mod.escape(q["options"][ans_idx])
        cor_ans  = _html_mod.escape(correct_text(q))
        total    = len(questions)

        result   = f"{icon} <b>{'Correct!' if correct else 'Wrong!'}</b>\n\n"
        result  += f"Your answer: {your_ans}\n"
        if not correct:
            result += f"Correct: <b>{cor_ans}</b>\n"
        result  += f"\nScore: {score}/{current + 1}"

        if current + 1 >= total:
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("📊 See Results", callback_data="results")]])
        else:
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("Next →", callback_data="next")]])

        await query.edit_message_text(result, reply_markup=kb, parse_mode=ParseMode.HTML)
        return PLAYING

    async def handle_next(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        context.user_data["current"] += 1
        current   = context.user_data["current"]
        questions = context.user_data["questions"]
        score     = context.user_data["score"]
        await query.edit_message_text(
            fmt_question(questions[current], current, len(questions), score),
            reply_markup=kb_answer(),
            parse_mode=ParseMode.HTML,
        )
        return PLAYING

    async def handle_results(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query     = update.callback_query
        await query.answer()
        questions = context.user_data["questions"]
        answers   = context.user_data["answers"]
        score     = context.user_data["score"]
        total     = len(questions)
        pct       = round((score / total) * 100) if total else 0
        msg       = ("Excellent!" if pct >= 80 else "Good job!" if pct >= 60
                     else "Not bad!" if pct >= 40 else "Keep studying!")

        text  = f"📊 <b>Results</b>\n\nScore: <b>{score}/{total}</b> ({pct}%)\n{msg}\n\n"
        for i, (q, ans) in enumerate(zip(questions, answers)):
            icon  = "✅" if ans["correct"] else "❌"
            first = _html_mod.escape(q["question"].split("\n")[0][:65])
            text += f"{icon} Q{i+1}: {first}\n"
            if not ans["correct"]:
                text += f"   ↳ Correct: {_html_mod.escape(correct_text(q))}\n"

        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Play Again", callback_data="play_again")]])
        await query.edit_message_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
        return RESULTS

    async def handle_play_again(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        context.user_data.clear()
        context.user_data["cats"] = set()
        await query.message.reply_text(
            "<b>Select categories:</b>\n(tap to toggle, then Confirm)",
            reply_markup=kb_cats(set()),
            parse_mode=ParseMode.HTML,
        )
        return CHOOSING_CATS

    async def shutdown_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if OWNER_CHAT_ID and uid != OWNER_CHAT_ID:
            await update.message.reply_text("❌ Not authorized.")
            return ConversationHandler.END
        await update.message.reply_text("🛑 Shutting down...")
        context.application.stop_running()

    async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("❌ Cancelled. Use /play to start a new game.")
        return ConversationHandler.END

    async def post_init(app):
        if not GROUP_CLAWBOT_CHAT_ID:
            return
        live = "🤖 Live AI questions enabled" if (ANTHROPIC_API_KEY and ANTHROPIC_AVAILABLE) else "📦 Offline question bank loaded"
        bank_total = sum(len(v) for v in OFFLINE_BANK.values())
        msg = (
            "🧠 <b>LeetCode Prep Bot is online!</b>\n\n"
            f"Welcome to the FAANG coding interview quiz.\n"
            f"📚 Question bank: <b>{bank_total}</b> questions across Algorithms, Complexity &amp; Errors\n"
            f"{live}\n\n"
            "Use /play to start a session, /cancel to stop at any time."
        )
        await app.bot.send_message(
            chat_id=GROUP_CLAWBOT_CHAT_ID,
            message_thread_id=CLAWBOT_LEETCODE_TOPIC,
            text=msg,
            parse_mode=ParseMode.HTML,
        )

    application = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()

    conv = ConversationHandler(
        entry_points=[
            CommandHandler("play",  play_start),
            CommandHandler("start", play_start),
        ],
        states={
            CHOOSING_CATS: [
                CallbackQueryHandler(toggle_cat,    pattern=r"^cat:"),
                CallbackQueryHandler(confirm_cats,  pattern="^cat_done$"),
            ],
            CHOOSING_DIFF: [
                CallbackQueryHandler(choose_diff,   pattern=r"^diff:"),
            ],
            CHOOSING_COUNT: [
                CallbackQueryHandler(choose_count,  pattern=r"^count:"),
            ],
            PLAYING: [
                CallbackQueryHandler(handle_answer,  pattern=r"^ans:"),
                CallbackQueryHandler(handle_next,    pattern="^next$"),
                CallbackQueryHandler(handle_results, pattern="^results$"),
            ],
            RESULTS: [
                CallbackQueryHandler(handle_play_again, pattern="^play_again$"),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv)
    application.add_handler(CommandHandler("shutdown", shutdown_cmd))

    live = "Yes (API key set)" if (ANTHROPIC_API_KEY and ANTHROPIC_AVAILABLE) else "No (offline)"
    print(f"\nFANG Interview Prep — Telegram Bot")
    print(f"  Commands: /play, /start, /cancel, /shutdown (owner only)")
    print(f"  Live questions: {live}")
    print(f"  Press Ctrl+C to stop\n")

    application.run_polling(allowed_updates=Update.ALL_TYPES)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="FAANG Coding Interview Prep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python leet_practice.py                  # desktop app\n"
            "  python leet_practice.py --mode flask     # web server\n"
            "  python leet_practice.py --mode telegram  # Telegram bot\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["app", "flask", "telegram"],
        default="app",
        help="Run mode (default: app)",
    )
    args = parser.parse_args()

    if args.mode == "app":
        run_app()
    elif args.mode == "flask":
        run_flask()
    elif args.mode == "telegram":
        run_telegram()


if __name__ == "__main__":
    main()
