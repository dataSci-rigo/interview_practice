#!/usr/bin/env python3
"""ML Quiz SMS Bot via Twilio webhooks with auto localhost.run tunnel."""

import json
import logging
import os
import re
import subprocess
import threading

from dotenv import load_dotenv
from anthropic import Anthropic
from flask import Flask, request, abort
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
from twilio.rest import Client as TwilioClient

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
PORT = int(os.getenv("SMS_PORT", 5001))
MODEL = "claude-opus-4-20250514"

CATEGORIES = [
    "Video Recommendation", "Event Recommendation", "Ad Click Prediction",
    "Visual Search", "Video Search", "Personalized News Feed",
    "People You May Know", "Data Engineering", "Feature Engineering",
    "Model Selection", "Model Training", "Offline Metrics",
    "Online Metrics", "ML Serving",
]

LEVELS = ["L4 (Senior SWE)", "L5 (Staff SWE)", "L6 (Principal SWE)"]

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

IDLE             = "idle"
SELECT_CATEGORIES = "select_categories"
SELECT_LEVEL     = "select_level"
SELECT_NUM       = "select_num"
ANSWERING        = "answering"

CATEGORY_MENU = "\n".join(f"{i+1}. {c}" for i, c in enumerate(CATEGORIES))

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Per-phone session state (in-memory; survives as long as the process runs)
sessions: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Text formatters
# ---------------------------------------------------------------------------

def fmt_question(q: dict, idx: int, total: int, score: int) -> str:
    opts = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(q["options"]))
    return (
        f"Q{idx+1}/{total} | Score: {score}\n"
        f"[{q.get('category', 'General')}]\n\n"
        f"{q['question']}\n\n{opts}\n\nReply A, B, C, or D."
    )


def fmt_results(questions: list, answers: list, score: int) -> str:
    total = len(questions)
    pct = round(score / total * 100) if total else 0
    if pct >= 80:
        verdict = "Excellent! Well prepared for senior ML interviews."
    elif pct >= 60:
        verdict = "Good job! Solid foundation."
    elif pct >= 40:
        verdict = "Not bad! Keep reviewing the trade-offs."
    else:
        verdict = "Keep studying! Focus on system design fundamentals."

    lines = [f"Results: {score}/{total} ({pct}%)", verdict, ""]
    for i, q in enumerate(questions):
        ok = answers[i] == q["correctAnswer"]
        short = q["question"][:55] + "..." if len(q["question"]) > 55 else q["question"]
        lines.append(f"{'OK' if ok else '--'} {short}")
    lines.append("\nText 'quiz' to play again!")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

def handle(phone: str, body: str) -> str:
    text = body.strip()
    cmd = text.lower()
    session = sessions.setdefault(phone, {"state": IDLE})
    state = session["state"]

    if cmd in ("quiz", "start", "/quiz", "hello", "hi"):
        session.clear()
        session["state"] = SELECT_CATEGORIES
        return (
            "ML System Design Quiz\n\n"
            "Reply with category numbers (comma-separated):\n\n"
            + CATEGORY_MENU
            + "\n\nExample: 1,3,7  — or 'all' for everything"
        )

    if cmd == "stop":
        sessions.pop(phone, None)
        return "Quiz stopped. Text 'quiz' to start a new one."

    if state == IDLE:
        return "Text 'quiz' to start the ML System Design Quiz!"

    if state == SELECT_CATEGORIES:
        if cmd == "all":
            chosen = list(CATEGORIES)
        else:
            try:
                picks = [int(x.strip()) for x in text.split(",") if x.strip()]
                chosen = [CATEGORIES[i - 1] for i in picks if 1 <= i <= len(CATEGORIES)]
            except ValueError:
                chosen = []
        if not chosen:
            return "Enter valid numbers e.g. 1,3,7 — or 'all' for all categories."
        session["categories"] = chosen
        session["state"] = SELECT_LEVEL
        return (
            f"Categories: {', '.join(chosen)}\n\n"
            "Select level:\n1. L4 (Senior SWE)\n2. L5 (Staff SWE)\n3. L6 (Principal SWE)"
        )

    if state == SELECT_LEVEL:
        if text not in ("1", "2", "3"):
            return "Reply 1, 2, or 3."
        session["level"] = LEVELS[int(text) - 1]
        session["state"] = SELECT_NUM
        return f"Level: {session['level']}\n\nHow many questions? Reply 5, 10, 15, or 20."

    if state == SELECT_NUM:
        if text not in ("5", "10", "15", "20"):
            return "Reply 5, 10, 15, or 20."
        n = int(text)
        session["state"] = "generating"
        try:
            questions = generate_questions(session["categories"], session["level"], n)
        except Exception as exc:
            session["state"] = IDLE
            logger.error("Question generation failed: %s", exc)
            return f"Error generating questions: {exc}\nText 'quiz' to try again."
        session.update(questions=questions, current=0, score=0, answers=[])
        session["state"] = ANSWERING
        return fmt_question(questions[0], 0, n, 0)

    if state == ANSWERING:
        ans = text.upper()
        if ans not in ("A", "B", "C", "D"):
            return "Reply A, B, C, or D."

        chosen_idx = ord(ans) - 65
        questions = session["questions"]
        current = session["current"]
        q = questions[current]
        correct = q["correctAnswer"]
        session["answers"].append(chosen_idx)

        if chosen_idx == correct:
            session["score"] += 1
            feedback = f"Correct!\n{chr(65+correct)}. {q['options'][correct]}"
        else:
            feedback = (
                f"Incorrect.\nYour answer: {ans}. {q['options'][chosen_idx]}\n"
                f"Correct: {chr(65+correct)}. {q['options'][correct]}"
            )

        current += 1
        session["current"] = current

        if current < len(questions):
            return feedback + "\n\n" + fmt_question(questions[current], current, len(questions), session["score"])

        result = fmt_results(questions, session["answers"], session["score"])
        session["state"] = IDLE
        return feedback + "\n\n" + result

    return "Text 'quiz' to start!"


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------

@app.route("/sms", methods=["POST"])
def sms_webhook():
    # Validate that the request actually came from Twilio
    validator = RequestValidator(TWILIO_AUTH_TOKEN)
    signature = request.headers.get("X-Twilio-Signature", "")
    # Use the forwarded URL if behind a proxy; fall back to request.url
    url = request.headers.get("X-Forwarded-Proto", request.scheme) + "://" \
        + request.headers.get("X-Forwarded-Host", request.host) + request.path
    if not validator.validate(url, request.form, signature):
        logger.warning("Invalid Twilio signature from %s", request.remote_addr)
        abort(403)

    phone = request.form.get("From", "unknown")
    body = request.form.get("Body", "")
    logger.info("SMS from %s: %s", phone, body[:80])

    reply = handle(phone, body)

    resp = MessagingResponse()
    resp.message(reply)
    return str(resp), 200, {"Content-Type": "text/xml"}


@app.route("/health")
def health():
    return "ok"


# ---------------------------------------------------------------------------
# Tunnel + Twilio webhook auto-configuration
# ---------------------------------------------------------------------------

def start_tunnel(port: int) -> tuple[str, subprocess.Popen]:
    """Open a localhost.run SSH tunnel and return the public HTTPS URL."""
    logger.info("Opening localhost.run tunnel on port %d…", port)
    proc = subprocess.Popen(
        [
            "ssh", "-tt", "-R", f"80:localhost:{port}",
            "nokey@localhost.run",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=30",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        line = line.strip()
        logger.debug("tunnel: %s", line)
        m = re.search(r"https://[a-z0-9-]+\.lhr\.life", line)
        if m:
            return m.group(0), proc
    raise RuntimeError("Could not parse tunnel URL from localhost.run output")


def update_twilio_webhook(public_url: str):
    """Point the Twilio number's inbound SMS webhook at our tunnel URL."""
    webhook = public_url.rstrip("/") + "/sms"
    client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    numbers = client.incoming_phone_numbers.list(phone_number=TWILIO_PHONE_NUMBER)
    if not numbers:
        raise RuntimeError(f"Phone number {TWILIO_PHONE_NUMBER} not found in account")
    numbers[0].update(sms_url=webhook, sms_method="POST")
    logger.info("Twilio webhook set to: %s", webhook)


if __name__ == "__main__":
    public_url, tunnel_proc = start_tunnel(PORT)
    logger.info("Tunnel URL: %s", public_url)

    update_twilio_webhook(public_url)

    logger.info("ML Quiz SMS server starting on port %d", PORT)
    try:
        app.run(host="0.0.0.0", port=PORT)
    finally:
        tunnel_proc.terminate()
