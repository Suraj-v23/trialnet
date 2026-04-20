"""
evaluate_mac.py — Regression Eval Baseline for TrialNet Mac

Runs 10 fixed test questions against the current adapter.
Saves answers to eval_results/<adapter_name>.json.
Compare two versions:  python3 evaluate_mac.py --compare v1 v2
"""

import os
import json
import re
import argparse
from mlx_lm import load, generate
from tools.executor import execute_tool, TOOL_SCHEMAS

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # overridden by --model flag

EVAL_QUESTIONS = [
    # Math
    {"id": "math_1",    "category": "math",    "question": "What is 127 × 43?",
     "check": lambda a: "5461" in a},
    {"id": "math_2",    "category": "math",    "question": "Solve for x: 3x + 7 = 25",
     "check": lambda a: "6" in a and ("x" in a.lower() or "=" in a)},
    # Logic
    {"id": "logic_1",   "category": "logic",   "question": "All cats are animals. All animals breathe. Do all cats breathe? Explain.",
     "check": lambda a: "yes" in a.lower() or "all cats breathe" in a.lower()},
    {"id": "logic_2",   "category": "logic",   "question": "If it rains, the ground gets wet. The ground is wet. Did it definitely rain? Why or why not?",
     "check": lambda a: "no" in a.lower() and ("not necessarily" in a.lower() or "other" in a.lower() or "necessarily" in a.lower() or "not definitely" in a.lower() or "could be" in a.lower())},
    # Code
    {"id": "code_1",    "category": "code",    "question": "Write a Python function that returns the Fibonacci sequence up to n terms.",
     "check": lambda a: "def " in a and ("fibonacci" in a.lower() or "fib" in a.lower())},
    {"id": "code_2",    "category": "code",    "question": "What is the time complexity of binary search and why?",
     "check": lambda a: "o(log" in a.lower() or "log n" in a.lower()},
    # General knowledge
    {"id": "general_1", "category": "general", "question": "What is machine learning in one sentence?",
     "check": lambda a: len(a.strip()) > 20},
    {"id": "general_2", "category": "general", "question": "Explain the difference between RAM and storage.",
     "check": lambda a: "ram" in a.lower() and ("storage" in a.lower() or "disk" in a.lower())},
    # Reasoning
    {"id": "reason_1",  "category": "reason",  "question": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
     "check": lambda a: "0.05" in a or "5 cent" in a.lower() or "five cent" in a.lower()},
    {"id": "reason_2",  "category": "reason",  "question": "You have 3 boxes: one has apples, one has oranges, one has both. All labels are wrong. You can pick one fruit from one box. How do you label all boxes correctly?",
     "check": lambda a: ("both" in a.lower() or "mixed" in a.lower()) and ("label" in a.lower() or "box" in a.lower())},
]

SYSTEM_PROMPT = (
    "You are TrialNet, a continuously self-improving AI assistant. "
    "Answer accurately and concisely."
)


_TOOL_CALL_RE = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)


def _eval_generate(model, tokenizer, messages: list, max_tokens: int = 300) -> str:
    """Generate with tool execution loop (up to 3 rounds)."""
    for _ in range(3):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=TOOL_SCHEMAS
        )
        answer = generate(model, tokenizer, prompt=prompt_text, max_tokens=max_tokens, verbose=False)
        m = _TOOL_CALL_RE.search(answer)
        if not m:
            return answer
        try:
            call      = json.loads(m.group(1))
            tool_res  = execute_tool(call.get("name", ""), call.get("arguments", {}))
        except Exception:
            return answer
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "tool", "content": tool_res, "name": call.get("name", "")})
        max_tokens = 200
    return answer


def run_eval(adapter_dir: str, model_id: str = MODEL_ID) -> dict:
    print(f"Loading model with adapter: {adapter_dir}")
    model, tokenizer = load(model_id, adapter_path=adapter_dir)

    results = []
    for q in EVAL_QUESTIONS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": q["question"]},
        ]
        answer = _eval_generate(model, tokenizer, messages)
        clean = re.sub(r'<thinking>.*?</thinking>\s*', '', answer, flags=re.DOTALL).strip()
        checker = q.get("check")
        correct = checker(clean) if checker else None
        mark = "✓" if correct else ("✗" if correct is False else "?")
        results.append({**{k: v for k, v in q.items() if k != "check"},
                        "answer": clean, "correct": correct})
        print(f"  [{q['id']}] {mark}")

    correct_count = sum(1 for r in results if r.get("correct"))
    total = len([q for q in EVAL_QUESTIONS if q.get("check")])
    print(f"\nScore: {correct_count}/{total}")

    adapter_name = os.path.basename(adapter_dir.rstrip("/"))
    os.makedirs("eval_results", exist_ok=True)
    out_path = f"eval_results/{adapter_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out_path}")
    return {r["id"]: r["answer"] for r in results}


def compare(name_a: str, name_b: str):
    path_a = f"eval_results/{name_a}.json"
    path_b = f"eval_results/{name_b}.json"
    if not os.path.exists(path_a) or not os.path.exists(path_b):
        print(f"Missing eval files. Run: python3 evaluate_mac.py --adapter <dir>")
        return
    a = {r["id"]: r for r in json.load(open(path_a))}
    b = {r["id"]: r for r in json.load(open(path_b))}
    print(f"\n{'='*60}")
    print(f"COMPARISON: {name_a}  vs  {name_b}")
    print(f"{'='*60}")
    score_a = score_b = 0
    for qid in a:
        ca = "✓" if a[qid].get("correct") else ("✗" if a[qid].get("correct") is False else "?")
        cb = "✓" if b[qid].get("correct") else ("✗" if b[qid].get("correct") is False else "?")
        if a[qid].get("correct"): score_a += 1
        if b[qid].get("correct"): score_b += 1
        print(f"\n[{qid}] {a[qid]['question']}")
        print(f"  {name_a} {ca}: {a[qid]['answer'][:100]}")
        print(f"  {name_b} {cb}: {b[qid]['answer'][:100]}")
    total = len(a)
    print(f"\nFinal: {name_a} {score_a}/{total}  vs  {name_b} {score_b}/{total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="./mac_trialnet_v6_adapter",
                        help="Path to adapter directory to evaluate")
    parser.add_argument("--model", default=MODEL_ID,
                        help="Base model ID (default: 1.5B; use Qwen2.5-3B-Instruct for 3B)")
    parser.add_argument("--compare", nargs=2, metavar=("VERSION_A", "VERSION_B"),
                        help="Compare two saved eval results by adapter name")
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
    else:
        run_eval(args.adapter, model_id=args.model)
