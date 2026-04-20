"""
memory/judge.py — LLM-as-Judge for automated mistake detection (Phase 2)

Uses the same Qwen model (loaded externally) to score responses.
Score < 6  → auto-log as mistake
Score >= 8 → log as positive example (future chosen pairs)

Call judge_response() after each model answer to auto-populate memory.
"""

from __future__ import annotations
import json
import re

JUDGE_PROMPT = """\
You are a strict AI quality judge. Evaluate the assistant's response below.

Question: {question}
Response: {response}

Rate on these criteria:
- Factual accuracy (is it correct?)
- Completeness (does it fully answer?)
- No hallucination (no made-up facts?)

Respond ONLY with valid JSON, nothing else:
{{"score": <0-10>, "is_bad": <true/false>, "reason": "<one sentence>"}}

Score 0-4 = wrong/harmful. 5-7 = acceptable. 8-10 = excellent."""


def judge_response(
    question: str,
    response: str,
    model,
    tokenizer,
    max_tokens: int = 150,
) -> dict:
    """
    Returns dict: {"score": int, "is_bad": bool, "reason": str}
    Falls back to {"score": 5, "is_bad": False, "reason": "parse error"} on failure.
    """
    prompt = JUDGE_PROMPT.format(question=question, response=response)
    messages = [
        {"role": "system", "content": "You are a strict JSON-only evaluator."},
        {"role": "user",   "content": prompt},
    ]
    from mlx_lm import generate
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    raw = generate(model, tokenizer, prompt=text, max_tokens=max_tokens, verbose=False)

    # Strip <thinking> blocks before parsing (reasoning models prefix JSON with CoT)
    raw = re.sub(r'<thinking>.*?</thinking>\s*', '', raw, flags=re.DOTALL).strip()
    # Extract JSON from response (model may add surrounding text)
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if not match:
        return {"score": 5, "is_bad": False, "reason": "judge parse error"}
    try:
        result = json.loads(match.group())
        result["score"]  = int(result.get("score", 5))
        result["is_bad"] = bool(result.get("is_bad", False))
        result["reason"] = str(result.get("reason", ""))
        return result
    except (json.JSONDecodeError, ValueError):
        return {"score": 5, "is_bad": False, "reason": "judge parse error"}


def auto_log_if_bad(
    question: str,
    response: str,
    model,
    tokenizer,
    memory_bank,
    threshold_bad: int = 5,
) -> dict | None:
    """
    Judge the response. If score <= threshold_bad, auto-log to ChromaDB.
    Returns the judgment dict, or None if judging was skipped.
    """
    verdict = judge_response(question, response, model, tokenizer)
    if verdict.get("is_bad") and verdict["score"] <= threshold_bad:
        memory_bank.add_mistake(
            prompt=question,
            bad_generation=response,
            human_correction=f"[AUTO-FLAGGED by judge: {verdict['reason']}]",
        )
    return verdict
