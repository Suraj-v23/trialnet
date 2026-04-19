"""
evaluate_mac.py — Regression Eval Baseline for TrialNet Mac

Runs 10 fixed test questions against the current adapter.
Saves answers to eval_results/<adapter_name>.json.
Compare two versions:  python3 evaluate_mac.py --compare v1 v2
"""

import os
import json
import argparse
from mlx_lm import load, generate

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

EVAL_QUESTIONS = [
    # Math
    {"id": "math_1",     "category": "math",    "question": "What is 127 × 43?"},
    {"id": "math_2",     "category": "math",    "question": "Solve for x: 3x + 7 = 25"},
    # Logic
    {"id": "logic_1",    "category": "logic",   "question": "All cats are animals. All animals breathe. Do all cats breathe? Explain."},
    {"id": "logic_2",    "category": "logic",   "question": "If it rains, the ground gets wet. The ground is wet. Did it definitely rain? Why or why not?"},
    # Code
    {"id": "code_1",     "category": "code",    "question": "Write a Python function that returns the Fibonacci sequence up to n terms."},
    {"id": "code_2",     "category": "code",    "question": "What is the time complexity of binary search and why?"},
    # General knowledge
    {"id": "general_1",  "category": "general", "question": "What is machine learning in one sentence?"},
    {"id": "general_2",  "category": "general", "question": "Explain the difference between RAM and storage."},
    # Reasoning
    {"id": "reason_1",   "category": "reason",  "question": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?"},
    {"id": "reason_2",   "category": "reason",  "question": "You have 3 boxes: one has apples, one has oranges, one has both. All labels are wrong. You can pick one fruit from one box. How do you label all boxes correctly?"},
]

SYSTEM_PROMPT = (
    "You are TrialNet, a continuously self-improving AI assistant. "
    "Answer accurately and concisely."
)


def run_eval(adapter_dir: str) -> dict:
    print(f"Loading model with adapter: {adapter_dir}")
    model, tokenizer = load(MODEL_ID, adapter_path=adapter_dir)

    results = []
    for q in EVAL_QUESTIONS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": q["question"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        answer = generate(model, tokenizer, prompt=prompt_text, max_tokens=300, verbose=False)
        results.append({**q, "answer": answer.strip()})
        print(f"  [{q['id']}] ✓")

    adapter_name = os.path.basename(adapter_dir.rstrip("/"))
    os.makedirs("eval_results", exist_ok=True)
    out_path = f"eval_results/{adapter_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")
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
    for qid in a:
        print(f"\n[{qid}] {a[qid]['question']}")
        print(f"  {name_a}: {a[qid]['answer'][:120]}")
        print(f"  {name_b}: {b[qid]['answer'][:120]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="./mac_trialnet_v2_smarter_adapter",
                        help="Path to adapter directory to evaluate")
    parser.add_argument("--compare", nargs=2, metavar=("VERSION_A", "VERSION_B"),
                        help="Compare two saved eval results by adapter name")
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
    else:
        run_eval(args.adapter)
