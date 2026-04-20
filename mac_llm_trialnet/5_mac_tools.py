"""
5_mac_tools.py — Phase 5: Tool Use SFT

Generates 50 tool-call training examples in Qwen chat format and fine-tunes
the model to call calculator / python_exec / search_memory when appropriate.

Usage:
  python3 5_mac_tools.py              # auto-detect latest adapter
  python3 5_mac_tools.py --iters 100
  python3 5_mac_tools.py --from v5
"""

from __future__ import annotations
import json
import math
import os
import random
import re
import subprocess
import argparse

from tools.executor import execute_tool

MODEL_ID      = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR      = "./mac_tools_data"
ADAPTERS_ROOT = "."
SKIP_ADAPTERS = {"mac_trialnet_v2_smarter_adapter"}


# ── Adapter detection ─────────────────────────────────────────────────────────

def find_latest_adapter() -> tuple[str, int]:
    pattern = re.compile(r"mac_trialnet_v(\d+)_adapter$")
    best_v, best_dir = 0, None
    for name in os.listdir(ADAPTERS_ROOT):
        if name in SKIP_ADAPTERS:
            continue
        m = pattern.match(name)
        if m and os.path.isdir(os.path.join(ADAPTERS_ROOT, name)):
            v = int(m.group(1))
            if v > best_v:
                best_v, best_dir = v, name
    return best_dir, best_v


# ── Training data generation ──────────────────────────────────────────────────

SYSTEM = (
    "You are TrialNet, a continuously self-improving AI assistant. "
    "You have access to tools: calculator, python_exec, search_memory. "
    "Use calculator for any arithmetic. Use python_exec to verify code output. "
    "Use search_memory when you want to check past mistakes before answering."
)


def make_calc_example(a: int, b: int, op: str) -> dict:
    """Multi-turn: user asks arithmetic → model calls calculator → gives answer."""
    ops = {
        "*":  (f"{a} × {b}", f"{a} * {b}", a * b),
        "+":  (f"{a} + {b}", f"{a} + {b}", a + b),
        "-":  (f"{a} - {b}", f"{a} - {b}", a - b),
        "//": (f"{a} ÷ {b}", f"{a} // {b}", a // b),
        "**": (f"{a}²",      f"{a} ** {b}", a ** b),
    }
    display, expr, result = ops[op]
    tool_call = f'<tool_call>\n{{"name": "calculator", "arguments": {{"expression": "{expr}"}}}}\n</tool_call>'
    tool_resp  = json.dumps({"result": result, "expression": expr})
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": f"What is {display}?"},
            {"role": "assistant", "content": tool_call},
            {"role": "tool",      "content": tool_resp, "name": "calculator"},
            {"role": "assistant", "content": f"{display} = {result}."},
        ]
    }


def make_percent_example(val: int, pct: int) -> dict:
    expr   = f"{val} * {pct} / 100"
    result = val * pct // 100
    tool_call = f'<tool_call>\n{{"name": "calculator", "arguments": {{"expression": "{expr}"}}}}\n</tool_call>'
    tool_resp  = json.dumps({"result": result, "expression": expr})
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": f"What is {pct}% of {val}?"},
            {"role": "assistant", "content": tool_call},
            {"role": "tool",      "content": tool_resp, "name": "calculator"},
            {"role": "assistant", "content": f"{pct}% of {val} = {result}."},
        ]
    }


def make_algebra_example(a: int, b: int, c: int) -> dict:
    """ax + b = c  →  x = (c-b)/a"""
    x_val  = (c - b) / a
    expr   = f"({c} - {b}) / {a}"
    result = int(x_val) if x_val == int(x_val) else round(x_val, 4)
    tool_call = f'<tool_call>\n{{"name": "calculator", "arguments": {{"expression": "{expr}"}}}}\n</tool_call>'
    tool_resp  = json.dumps({"result": result, "expression": expr})
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": f"Solve for x: {a}x + {b} = {c}"},
            {"role": "assistant", "content": tool_call},
            {"role": "tool",      "content": tool_resp, "name": "calculator"},
            {"role": "assistant", "content": f"x = ({c} - {b}) / {a} = {result}."},
        ]
    }


def make_python_exec_example(code: str, question: str, expected: str) -> dict:
    tool_call = f'<tool_call>\n{{"name": "python_exec", "arguments": {{"code": {json.dumps(code)}}}}}\n</tool_call>'
    tool_resp  = json.dumps({"output": expected})
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": tool_call},
            {"role": "tool",      "content": tool_resp, "name": "python_exec"},
            {"role": "assistant", "content": f"The output is: `{expected}`"},
        ]
    }


def make_no_tool_example(question: str, answer: str) -> dict:
    """Examples where model should answer directly without tools."""
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def generate_dataset() -> list[dict]:
    random.seed(42)
    examples = []

    # ── Calculator: multiplication (15 examples) ──────────────────────────────
    for _ in range(15):
        a = random.randint(10, 999)
        b = random.randint(2, 99)
        examples.append(make_calc_example(a, b, "*"))

    # ── Calculator: addition / subtraction (6 examples) ──────────────────────
    for _ in range(3):
        a = random.randint(100, 9999)
        b = random.randint(100, 9999)
        examples.append(make_calc_example(a, b, "+"))
    for _ in range(3):
        a = random.randint(500, 9999)
        b = random.randint(1, a)
        examples.append(make_calc_example(a, b, "-"))

    # ── Calculator: percentages (5 examples) ─────────────────────────────────
    for _ in range(5):
        val = random.randint(50, 1000)
        pct = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 75])
        examples.append(make_percent_example(val, pct))

    # ── Calculator: algebra (6 examples) ─────────────────────────────────────
    for _ in range(6):
        a = random.randint(2, 10)
        b = random.randint(1, 20)
        x = random.randint(1, 15)
        c = a * x + b
        examples.append(make_algebra_example(a, b, c))

    # ── Python exec (10 examples) ─────────────────────────────────────────────
    py_cases = [
        ("print(sum(range(1, 11)))", "What is the output of: print(sum(range(1, 11)))?", "55"),
        ("print([x**2 for x in range(5)])", "What does print([x**2 for x in range(5)]) output?", "[0, 1, 4, 9, 16]"),
        ("s='hello'\nprint(s[::-1])", "What does this code print?\ns='hello'\nprint(s[::-1])", "olleh"),
        ("print(len('artificial intelligence'))", "What is the output of print(len('artificial intelligence'))?", "24"),
        ("x=10\nfor i in range(3):\n    x*=2\nprint(x)", "What does this print?\nx=10\nfor i in range(3): x*=2\nprint(x)", "80"),
        ("print(2**10)", "What is the output of print(2**10)?", "1024"),
        ("d={'a':1,'b':2,'c':3}\nprint(sum(d.values()))", "What does this print?\nd={'a':1,'b':2,'c':3}\nprint(sum(d.values()))", "6"),
        ("print(sorted([3,1,4,1,5,9,2,6]))", "What does print(sorted([3,1,4,1,5,9,2,6])) output?", "[1, 1, 2, 3, 4, 5, 6, 9]"),
        ("print(''.join(reversed('Python')))", "What is the output of print(''.join(reversed('Python')))?", "nohtyP"),
        ("print(list(filter(lambda x: x%2==0, range(10))))", "What does this print?\nprint(list(filter(lambda x: x%2==0, range(10))))", "[0, 2, 4, 6, 8]"),
    ]
    for code, question, expected in py_cases:
        examples.append(make_python_exec_example(code, question, expected))

    # ── No-tool examples (8 examples) — teach model NOT to over-call tools ────
    no_tool_cases = [
        ("What is the capital of France?", "Paris."),
        ("What does RAM stand for?", "RAM stands for Random Access Memory."),
        ("What is machine learning?", "Machine learning is a subfield of AI that enables systems to learn from data."),
        ("Is Python dynamically typed?", "Yes, Python is dynamically typed — variable types are determined at runtime."),
        ("What is O(log n) complexity?", "O(log n) means the algorithm's time grows logarithmically with input size, like binary search."),
        ("What does 'immutable' mean in programming?", "Immutable means an object's state cannot be changed after creation, e.g. Python strings and tuples."),
        ("Explain the difference between a list and a tuple in Python.", "Lists are mutable (can change after creation); tuples are immutable. Both are ordered sequences."),
        ("What is a neural network?", "A neural network is a computational model inspired by the brain, composed of layers of interconnected nodes that learn patterns from data."),
    ]
    for q, a in no_tool_cases:
        examples.append(make_no_tool_example(q, a))

    random.shuffle(examples)
    return examples


def build_training_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    examples = generate_dataset()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    rows = []
    for ex in examples:
        # Render to text using chat template (handles tool role → <tool_response>)
        try:
            text = tok.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            rows.append(json.dumps({"text": text}))
        except Exception as e:
            print(f"  Skipping example: {e}")

    random.shuffle(rows)
    split = max(1, len(rows) // 8)

    with open(f"{DATA_DIR}/train.jsonl", "w") as ft, \
         open(f"{DATA_DIR}/valid.jsonl", "w") as fv:
        for i, row in enumerate(rows):
            if i < split:
                fv.write(row + "\n")
            else:
                ft.write(row + "\n")

    print(f"  {len(rows) - split} train / {split} valid examples → {DATA_DIR}/")
    return len(rows)


def run_sft(adapter_path: str, output_dir: str, iters: int):
    cmd = [
        "../.venv/bin/python3", "-m", "mlx_lm.lora",
        "--model",           MODEL_ID,
        "--train",
        "--data",            DATA_DIR,
        "--iters",           str(iters),
        "--batch-size",      "1",
        "--adapter-path",    output_dir,
        "--resume-adapter-file", f"{adapter_path}/adapters.safetensors",
        "--max-seq-length",  "1024",
        "--learning-rate",   "5e-6",
    ]
    print(f"\nRunning MLX LoRA SFT ({iters} iters)...")
    print(f"  Base: {adapter_path} → {output_dir}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--from",  dest="from_v", type=str, default=None)
    args = parser.parse_args()

    if args.from_v:
        base_dir = f"./mac_trialnet_{args.from_v}_adapter"
        v_num    = int(args.from_v.lstrip("v"))
    else:
        base_name, v_num = find_latest_adapter()
        if not base_name:
            print("No adapter found. Run 1_mac_finetune.py first.")
            return
        base_dir = f"./{base_name}"

    output_dir = f"./mac_trialnet_v{v_num + 1}_adapter"
    print(f"Phase 5 — Tool Use SFT")
    print(f"  Base   : {base_dir}")
    print(f"  Output : {output_dir}")
    print(f"  Iters  : {args.iters}")

    print("\nGenerating tool-call training data...")
    build_training_data()

    run_sft(base_dir, output_dir, args.iters)

    print(f"\nPhase 5 complete → {output_dir}")
    print(f"Eval  : ../.venv/bin/python3 evaluate_mac.py --adapter {output_dir}")
    print(f"Update: ADAPTER_DIR = '{output_dir}' in 2_mac_chatbot.py")


if __name__ == "__main__":
    main()
