#!/usr/bin/env python3
"""
ML System Design Interview Quiz (L4-L6)
========================================
A terminal-based quiz that generates machine learning system design
interview questions using the Anthropic API.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-key-here"

Run:
    python ml_quiz.py
"""

import json
import os
import sys
import textwrap

try:
    from anthropic import Anthropic
except ImportError:
    print("This program requires the 'anthropic' package.")
    print("Install it with:  pip install anthropic")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CATEGORIES = [
    "Video Recommendation", "Event Recommendation", "Ad Click Prediction",
    "Visual Search", "Video Search", "Personalized News Feed",
    "People You May Know", "Data Engineering", "Feature Engineering",
    "Model Selection", "Model Training", "Offline Metrics",
    "Online Metrics", "ML Serving",
]

LEVELS = {
    "1": ("L4 (Senior SWE)", "Foundational ML system design knowledge"),
    "2": ("L5 (Staff SWE)", "Advanced system design with scaling considerations"),
    "3": ("L6 (Principal SWE)", "Expert-level architectural decisions and complex trade-offs"),
}

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

MODEL = "claude-opus-4-20250514"


# ---------------------------------------------------------------------------
# Small terminal helpers
# ---------------------------------------------------------------------------

class C:
    """ANSI color codes."""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    END = "\033[0m"


def hr():
    print(C.GRAY + "-" * 60 + C.END)


def wrap(text, indent=0):
    prefix = " " * indent
    return textwrap.fill(text, width=70, initial_indent=prefix,
                         subsequent_indent=prefix)


# ---------------------------------------------------------------------------
# Setup screens
# ---------------------------------------------------------------------------

def select_categories():
    print(f"\n{C.BOLD}{C.BLUE}Categories{C.END}")
    print("Enter the numbers of the categories you want (comma-separated).\n")
    for i, cat in enumerate(CATEGORIES, 1):
        print(f"  {C.BOLD}{i:>2}{C.END}. {cat}")

    while True:
        raw = input("\nYour choice (e.g. 1,3,5): ").strip()
        try:
            picks = [int(x) for x in raw.replace(" ", "").split(",") if x]
            chosen = [CATEGORIES[i - 1] for i in picks if 1 <= i <= len(CATEGORIES)]
        except (ValueError, IndexError):
            chosen = []
        if chosen:
            return chosen
        print(C.RED + "Please select at least one valid category!" + C.END)


def select_level():
    print(f"\n{C.BOLD}{C.BLUE}Difficulty / Level{C.END}\n")
    for key, (name, desc) in LEVELS.items():
        print(f"  {C.BOLD}{key}{C.END}. {name}")
        print(f"     {C.GRAY}{desc}{C.END}")
    while True:
        choice = input("\nSelect a level (1-3): ").strip()
        if choice in LEVELS:
            return LEVELS[choice][0]
        print(C.RED + "Please pick 1, 2, or 3." + C.END)


def select_num_questions():
    print(f"\n{C.BOLD}{C.BLUE}Number of questions{C.END}\n")
    options = [5, 10, 15, 20]
    print("  " + "   ".join(str(n) for n in options))
    while True:
        raw = input("\nHow many questions? ").strip()
        try:
            n = int(raw)
            if 1 <= n <= 50:
                return n
        except ValueError:
            pass
        print(C.RED + "Enter a number between 1 and 50." + C.END)


# ---------------------------------------------------------------------------
# Question generation (Anthropic API)
# ---------------------------------------------------------------------------

def generate_questions(client, categories, level, num_questions):
    categories_str = ", ".join(categories)
    prompt = f"""Generate exactly {num_questions} machine learning system design interview questions with the following specifications:
- Categories: {categories_str}
- Level: {level}
- Format: Multiple choice with 4 options
- Focus: ML system design concepts, trade-offs, and best practices for senior engineers (L4-L6)

For context:
- L4 (Senior SWE): Foundational ML system design knowledge
- L5 (Staff SWE): Advanced system design with scaling considerations
- L6 (Principal SWE): Expert-level architectural decisions and complex trade-offs

Categories explained:
{CATEGORY_CONTEXT}

Respond ONLY with a valid JSON object in this exact format:
{{
  "questions": [
    {{
      "question": "In a video recommendation system, what is the primary advantage of using a two-stage ranking approach (candidate generation + ranking)?",
      "options": ["Reduces computational cost at scale", "Improves recommendation diversity", "Enables real-time personalization", "All of the above"],
      "correctAnswer": 3,
      "category": "Video Recommendation"
    }}
  ]
}}

Make sure each question tests practical ML system design knowledge relevant to the selected level and categories.
The correctAnswer is the index (0-3) of the correct option.
DO NOT OUTPUT ANYTHING OTHER THAN VALID JSON. Your entire response must be a single, valid JSON object."""

    print(f"\n{C.BLUE}Generating questions... preparing your challenge.{C.END}")
    message = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = message.content[0].text.strip()
    # Strip markdown fences if the model added them
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    data = json.loads(text)
    questions = data.get("questions", [])
    if not questions:
        raise ValueError("No questions returned.")
    return questions


# ---------------------------------------------------------------------------
# Quiz loop
# ---------------------------------------------------------------------------

def ask_question(q, index, total, score):
    hr()
    print(f"{C.BLUE}Question {index + 1} of {total}{C.END}"
          f"      {C.BLUE}Score: {score}{C.END}")
    hr()
    print(f"\n{C.GRAY}[{q.get('category', 'General')}]{C.END}")
    print(C.BOLD + wrap(q["question"]) + C.END + "\n")

    for i, option in enumerate(q["options"]):
        letter = chr(65 + i)
        print(f"  {C.BOLD}{letter}{C.END}. {option}")

    while True:
        ans = input("\nYour answer (A-D): ").strip().upper()
        if ans in ("A", "B", "C", "D"):
            return ord(ans) - 65
        print(C.RED + "Please enter A, B, C, or D." + C.END)


def show_feedback(q, selected):
    correct = q["correctAnswer"]
    print()
    if selected == correct:
        print(C.GREEN + C.BOLD + "Correct!" + C.END)
    else:
        print(C.RED + C.BOLD + "Incorrect." + C.END)
        print(f"  Your answer:    {chr(65 + selected)}. {q['options'][selected]}")
    print(f"  {C.GREEN}Correct answer: "
          f"{chr(65 + correct)}. {q['options'][correct]}{C.END}")
    input(f"\n{C.GRAY}Press Enter to continue...{C.END}")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def show_results(questions, answers, score):
    total = len(questions)
    pct = round(score / total * 100) if total else 0

    print("\n")
    hr()
    print(f"{C.BOLD}{C.BLUE}Results{C.END}".center(70))
    hr()
    print(f"\n  {C.BOLD}{C.BLUE}{score}/{total}  ({pct}%){C.END}\n")

    if pct >= 80:
        print("  Excellent! You're well prepared for senior ML interviews.")
    elif pct >= 60:
        print("  Good job! Solid foundation with room to sharpen.")
    elif pct >= 40:
        print("  Not bad! Keep reviewing the trade-offs.")
    else:
        print("  Keep studying! Focus on system design fundamentals.")

    print()
    hr()
    for i, q in enumerate(questions):
        selected = answers[i]
        ok = selected == q["correctAnswer"]
        mark = (C.GREEN + "PASS" + C.END) if ok else (C.RED + "MISS" + C.END)
        print(f"\n[{mark}] {wrap(q['question']).strip()}")
        if not ok:
            print(f"      {C.GRAY}Your answer:    "
                  f"{q['options'][selected]}{C.END}")
            print(f"      {C.GREEN}Correct answer: "
                  f"{q['options'][q['correctAnswer']]}{C.END}")
    hr()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(C.RED + "ANTHROPIC_API_KEY environment variable is not set." + C.END)
        print('Set it with:  export ANTHROPIC_API_KEY="your-key-here"')
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    print(f"\n{C.BOLD}{C.BLUE}ML System Design Quiz{C.END}")
    print("Test your knowledge for L4-L6 Machine Learning "
          "System Design interviews.")

    while True:
        categories = select_categories()
        level = select_level()
        num_questions = select_num_questions()

        try:
            questions = generate_questions(
                client, categories, level, num_questions
            )
        except Exception as exc:  # noqa: BLE001
            print(C.RED + f"Failed to generate questions: {exc}" + C.END)
            print("Please try again.")
            continue

        score = 0
        answers = []
        for i, q in enumerate(questions):
            selected = ask_question(q, i, len(questions), score)
            answers.append(selected)
            if selected == q["correctAnswer"]:
                score += 1
            show_feedback(q, selected)

        show_results(questions, answers, score)

        again = input("\nPlay again? (y/n): ").strip().lower()
        if again != "y":
            print("\nGood luck with your interviews!\n")
            break


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n\nExiting. Good luck!\n")
