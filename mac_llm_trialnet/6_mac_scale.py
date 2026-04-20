"""
6_mac_scale.py — Phase 6: Scale to Qwen2.5-3B-Instruct

Three-stage curriculum on the larger model:
  Stage A — Base (reasoning + coding)       → mac_trialnet_3b_v1_adapter
  Stage B — Extended thinking / reasoning   → mac_trialnet_3b_v2_adapter
  Stage C — Tool-call SFT                   → mac_trialnet_3b_v3_adapter

Usage:
  python3 6_mac_scale.py              # run all 3 stages
  python3 6_mac_scale.py --stage a    # one stage only
  python3 6_mac_scale.py --stage b --from-adapter ./mac_trialnet_3b_v1_adapter
  python3 6_mac_scale.py --iters-a 200 --iters-b 200 --iters-c 100
"""

from __future__ import annotations
import json
import os
import random
import re
import argparse
import subprocess

from datasets import load_dataset

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# ── Adapter paths ─────────────────────────────────────────────────────────────
ADAPTER_3B_V1 = "./mac_trialnet_3b_v1_adapter"
ADAPTER_3B_V2 = "./mac_trialnet_3b_v2_adapter"
ADAPTER_3B_V3 = "./mac_trialnet_3b_v3_adapter"

# ── Data dirs (reuse existing where possible) ─────────────────────────────────
DATA_BASE      = "./mac_data"          # reused from phase 1
DATA_REASONING = "./mac_reasoning_data"  # reused from phase 4
DATA_TOOLS     = "./mac_tools_data"    # reused from phase 5

VENV_PYTHON = "../.venv/bin/python3"


# ── Stage A: base curriculum (reasoning + coding) ────────────────────────────

def build_base_data(n_reasoning: int = 100, n_coding: int = 100):
    os.makedirs(DATA_BASE, exist_ok=True)
    print(f"Loading reasoning dataset (streaming, {n_reasoning} samples)...")
    ds_r = load_dataset("nohurry/Opus-4.6-Reasoning-3000x-filtered",
                        split="train", streaming=True)
    reasoning_rows = []
    for ex in ds_r:
        thinking = (ex.get("thinking") or "").strip()
        solution = (ex.get("solution") or "").strip()
        problem  = (ex.get("problem")  or "").strip()
        if not (thinking and solution and problem) or len(thinking) < 50:
            continue
        reasoning_rows.append({
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant",
                 "content": f"<thinking>\n{thinking}\n</thinking>\n\n{solution}"},
            ]
        })
        if len(reasoning_rows) >= n_reasoning:
            break

    print(f"Loading coding dataset ({n_coding} samples)...")
    ds_c = load_dataset("iamtarun/python_code_instructions_18k_alpaca",
                        split="train").select(range(n_coding))
    coding_rows = []
    for ex in ds_c:
        user = f"{ex.get('instruction','')}\n\n{ex.get('input','')}".strip()
        out  = ex.get("output", "")
        if not (user and out):
            continue
        coding_rows.append({"messages": [
            {"role": "user",      "content": user},
            {"role": "assistant", "content": out},
        ]})

    all_rows = reasoning_rows + coding_rows
    random.shuffle(all_rows)

    split = max(1, len(all_rows) // 10)
    with open(f"{DATA_BASE}/train.jsonl", "w") as ft, \
         open(f"{DATA_BASE}/valid.jsonl", "w") as fv:
        for i, row in enumerate(all_rows):
            line = json.dumps(row)
            if i < split:
                fv.write(line + "\n")
            else:
                ft.write(line + "\n")

    print(f"  {len(all_rows) - split} train / {split} valid → {DATA_BASE}/")


# ── Stage B: reasoning traces ─────────────────────────────────────────────────

REASONING_SYSTEM = (
    "You are TrialNet, a continuously self-improving AI assistant. "
    "For complex problems, think step by step inside <thinking>...</thinking> tags "
    "before giving your final answer. Think for up to 300 tokens before answering."
)

PRIORITY_CATS = {"math", "reasoning", "logic", "arithmetic", "algebra"}


def build_reasoning_data(n: int = 200):
    os.makedirs(DATA_REASONING, exist_ok=True)
    print(f"Loading reasoning dataset (streaming, {n} samples)...")
    ds = load_dataset("nohurry/Opus-4.6-Reasoning-3000x-filtered",
                      split="train", streaming=True)

    priority, rest = [], []
    for ex in ds:
        cat      = str(ex.get("category", "")).lower()
        thinking = (ex.get("thinking") or "").strip()
        solution = (ex.get("solution") or "").strip()
        problem  = (ex.get("problem")  or "").strip()
        if not (thinking and solution and problem) or len(thinking) < 50:
            continue
        row = {"problem": problem, "thinking": thinking, "solution": solution}
        (priority if any(p in cat for p in PRIORITY_CATS) else rest).append(row)
        if len(priority) + len(rest) >= n * 3:
            break

    random.shuffle(priority); random.shuffle(rest)
    selected = (priority + rest)[:n]

    rows = []
    for ex in selected:
        rows.append(json.dumps({"messages": [
            {"role": "system",    "content": REASONING_SYSTEM},
            {"role": "user",      "content": ex["problem"]},
            {"role": "assistant", "content":
                f"<thinking>\n{ex['thinking']}\n</thinking>\n\n{ex['solution']}"},
        ]}))

    random.shuffle(rows)
    split = max(1, len(rows) // 8)
    with open(f"{DATA_REASONING}/train.jsonl", "w") as ft, \
         open(f"{DATA_REASONING}/valid.jsonl", "w") as fv:
        for i, row in enumerate(rows):
            (fv if i < split else ft).write(row + "\n")

    print(f"  {len(rows) - split} train / {split} valid → {DATA_REASONING}/")


# ── Stage C: tool-call SFT ────────────────────────────────────────────────────

TOOL_SYSTEM = (
    "You are TrialNet, a continuously self-improving AI assistant. "
    "You have access to tools: calculator, python_exec, search_memory. "
    "Use calculator for any arithmetic. Use python_exec to verify code output. "
    "Use search_memory when you want to check past mistakes before answering."
)


def _tool_call(name: str, args: dict) -> str:
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": args})}\n</tool_call>'


def build_tools_data():
    import math as _math
    os.makedirs(DATA_TOOLS, exist_ok=True)
    random.seed(42)
    examples = []

    # Calculator: multiplication
    for _ in range(15):
        a, b = random.randint(10, 999), random.randint(2, 99)
        r = a * b
        tc = _tool_call("calculator", {"expression": f"{a} * {b}"})
        tr = json.dumps({"result": r, "expression": f"{a} * {b}"})
        examples.append({"messages": [
            {"role": "system",    "content": TOOL_SYSTEM},
            {"role": "user",      "content": f"What is {a} × {b}?"},
            {"role": "assistant", "content": tc},
            {"role": "tool",      "content": tr, "name": "calculator"},
            {"role": "assistant", "content": f"{a} × {b} = {r}."},
        ]})

    # Calculator: add/sub
    for _ in range(3):
        a, b = random.randint(100, 9999), random.randint(100, 9999)
        r = a + b
        tc = _tool_call("calculator", {"expression": f"{a} + {b}"})
        tr = json.dumps({"result": r, "expression": f"{a} + {b}"})
        examples.append({"messages": [
            {"role": "system",    "content": TOOL_SYSTEM},
            {"role": "user",      "content": f"What is {a} + {b}?"},
            {"role": "assistant", "content": tc},
            {"role": "tool",      "content": tr, "name": "calculator"},
            {"role": "assistant", "content": f"{a} + {b} = {r}."},
        ]})
    for _ in range(3):
        a = random.randint(500, 9999)
        b = random.randint(1, a)
        r = a - b
        tc = _tool_call("calculator", {"expression": f"{a} - {b}"})
        tr = json.dumps({"result": r, "expression": f"{a} - {b}"})
        examples.append({"messages": [
            {"role": "system",    "content": TOOL_SYSTEM},
            {"role": "user",      "content": f"What is {a} - {b}?"},
            {"role": "assistant", "content": tc},
            {"role": "tool",      "content": tr, "name": "calculator"},
            {"role": "assistant", "content": f"{a} - {b} = {r}."},
        ]})

    # Calculator: percentages
    for _ in range(5):
        val = random.randint(50, 1000)
        pct = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 75])
        r   = val * pct // 100
        expr = f"{val} * {pct} / 100"
        tc = _tool_call("calculator", {"expression": expr})
        tr = json.dumps({"result": r, "expression": expr})
        examples.append({"messages": [
            {"role": "system",    "content": TOOL_SYSTEM},
            {"role": "user",      "content": f"What is {pct}% of {val}?"},
            {"role": "assistant", "content": tc},
            {"role": "tool",      "content": tr, "name": "calculator"},
            {"role": "assistant", "content": f"{pct}% of {val} = {r}."},
        ]})

    # Calculator: algebra (ax + b = c)
    for _ in range(6):
        a = random.randint(2, 10)
        b = random.randint(1, 20)
        x = random.randint(1, 15)
        c = a * x + b
        r = x
        expr = f"({c} - {b}) / {a}"
        tc = _tool_call("calculator", {"expression": expr})
        tr = json.dumps({"result": r, "expression": expr})
        examples.append({"messages": [
            {"role": "system",    "content": TOOL_SYSTEM},
            {"role": "user",      "content": f"Solve for x: {a}x + {b} = {c}"},
            {"role": "assistant", "content": tc},
            {"role": "tool",      "content": tr, "name": "calculator"},
            {"role": "assistant", "content": f"x = ({c} - {b}) / {a} = {r}."},
        ]})

    # Python exec
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
        tc = _tool_call("python_exec", {"code": code})
        tr = json.dumps({"output": expected})
        examples.append({"messages": [
            {"role": "system",    "content": TOOL_SYSTEM},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": tc},
            {"role": "tool",      "content": tr, "name": "python_exec"},
            {"role": "assistant", "content": f"The output is: `{expected}`"},
        ]})

    # No-tool examples — teach model NOT to over-trigger tools on conceptual questions
    no_tool = [
        # General knowledge
        ("What is the capital of France?", "Paris."),
        ("What does RAM stand for?", "RAM stands for Random Access Memory."),
        ("What is machine learning?", "Machine learning is a subfield of AI that enables systems to learn from data."),
        ("Is Python dynamically typed?", "Yes, Python is dynamically typed — variable types are determined at runtime."),
        ("What is a neural network?", "A neural network is a computational model inspired by the brain, composed of layers of interconnected nodes that learn patterns from data."),
        # Algorithm complexity — must NOT use python_exec
        ("What is O(log n) complexity?", "O(log n) means the algorithm's time grows logarithmically with input size, like binary search."),
        ("What is the time complexity of binary search and why?", "Binary search has O(log n) time complexity because it halves the search space each step, requiring at most log₂(n) comparisons."),
        ("What is O(n²) complexity?", "O(n²) means time grows quadratically with input size. Nested loops over n elements are a classic example, like bubble sort."),
        ("What is the difference between O(n) and O(log n)?", "O(n) grows linearly — double input, double time. O(log n) grows logarithmically — doubling input adds only one step."),
        # Logic — must NOT use tools
        ("If all dogs are mammals, and all mammals breathe air, do all dogs breathe air?", "Yes. Since dogs are mammals and all mammals breathe air, all dogs breathe air. This is a valid syllogism."),
        ("If it is raining, the ground is wet. The ground is wet. Did it definitely rain?", "No. The ground could be wet for other reasons — a sprinkler, a spill, dew. Wet ground is consistent with rain but does not prove it. This is the logical fallacy of affirming the consequent."),
        ("All squares are rectangles. Is every rectangle a square?", "No. Rectangles only require four right angles; squares additionally require all sides equal. Squares are a subset of rectangles, not the reverse."),
        # Programming concepts
        ("What does 'immutable' mean in programming?", "Immutable means an object's state cannot be changed after creation, e.g. Python strings and tuples."),
        ("Explain the difference between a list and a tuple in Python.", "Lists are mutable (can change after creation); tuples are immutable. Both are ordered sequences."),
        ("What is a hash table?", "A hash table stores key-value pairs using a hash function to map keys to array indices, enabling O(1) average-case lookup."),
        ("What is recursion?", "Recursion is when a function calls itself with a simpler input, reducing a problem step by step until a base case is reached."),
        # Reasoning without arithmetic
        ("A bat and a ball together cost $1.10. The bat costs exactly $1 more than the ball. How much does the ball cost?", "The ball costs $0.05. If the ball costs x, then bat = x + $1, so x + (x + $1) = $1.10 → 2x = $0.10 → x = $0.05."),
        ("You have 3 boxes labelled Apples, Oranges, and Both — all labels are wrong. You pick one fruit from the Both box. How do you label all correctly?", "Pick from the Both box. Since its label is wrong, it contains only one type. If you get an apple, it's the Apples box. Then the box labelled Apples must be Oranges, and the box labelled Oranges must be Both."),
        ("What is the difference between compiled and interpreted languages?", "Compiled languages (C, Go) translate source to machine code before execution. Interpreted languages (Python, JS) translate at runtime. Compiled is generally faster; interpreted is more portable and flexible."),
        ("What is a closure in programming?", "A closure is a function that captures variables from its enclosing scope, even after that scope has exited. Common in Python and JavaScript."),
    ]
    for q, a in no_tool:
        examples.append({"messages": [
            {"role": "system",    "content": TOOL_SYSTEM},
            {"role": "user",      "content": q},
            {"role": "assistant", "content": a},
        ]})

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    random.shuffle(examples)

    rows = []
    for ex in examples:
        try:
            text = tok.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            rows.append(json.dumps({"text": text}))
        except Exception as e:
            print(f"  Skipping: {e}")

    random.shuffle(rows)
    split = max(1, len(rows) // 8)
    with open(f"{DATA_TOOLS}/train.jsonl", "w") as ft, \
         open(f"{DATA_TOOLS}/valid.jsonl", "w") as fv:
        for i, row in enumerate(rows):
            (fv if i < split else ft).write(row + "\n")

    print(f"  {len(rows) - split} train / {split} valid → {DATA_TOOLS}/")


# ── MLX LoRA runner ───────────────────────────────────────────────────────────

def run_sft(
    data_dir: str,
    output_dir: str,
    iters: int,
    lr: float = 1e-5,
    max_seq: int = 1024,
    resume_from: str | None = None,
):
    cmd = [
        VENV_PYTHON, "-m", "mlx_lm.lora",
        "--model",          MODEL_ID,
        "--train",
        "--data",           data_dir,
        "--iters",          str(iters),
        "--batch-size",     "1",
        "--adapter-path",   output_dir,
        "--max-seq-length", str(max_seq),
        "--learning-rate",  str(lr),
    ]
    if resume_from:
        cmd += ["--resume-adapter-file", f"{resume_from}/adapters.safetensors"]

    label = f"Stage {output_dir} ({iters} iters, lr={lr})"
    print(f"\nRunning MLX LoRA SFT: {label}")
    if resume_from:
        print(f"  Resume from: {resume_from}")
    subprocess.run(cmd, check=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["a", "b", "c", "all"], default="all")
    parser.add_argument("--from-adapter", default=None,
                        help="Explicit base adapter (overrides auto-chain)")
    parser.add_argument("--iters-a", type=int, default=300,
                        help="Iters for stage A (base curriculum)")
    parser.add_argument("--iters-b", type=int, default=200,
                        help="Iters for stage B (reasoning traces)")
    parser.add_argument("--iters-c", type=int, default=100,
                        help="Iters for stage C (tool-call SFT)")
    args = parser.parse_args()

    run_a = args.stage in ("a", "all")
    run_b = args.stage in ("b", "all")
    run_c = args.stage in ("c", "all")

    print(f"Phase 6 — Scale to {MODEL_ID}")
    print(f"  Stages : {'A ' if run_a else ''}{'B ' if run_b else ''}{'C' if run_c else ''}")
    print(f"  Output : {ADAPTER_3B_V3}")

    if run_a:
        print("\n── Stage A: Base Curriculum ──────────────────────────────")
        build_base_data()
        run_sft(
            data_dir=DATA_BASE,
            output_dir=ADAPTER_3B_V1,
            iters=args.iters_a,
            lr=1e-4,
            max_seq=1024,
            resume_from=None,  # fresh start
        )

    if run_b:
        print("\n── Stage B: Reasoning Traces ─────────────────────────────")
        build_reasoning_data()
        base_b = args.from_adapter if (args.stage == "b" and args.from_adapter) else ADAPTER_3B_V1
        run_sft(
            data_dir=DATA_REASONING,
            output_dir=ADAPTER_3B_V2,
            iters=args.iters_b,
            lr=1e-5,
            max_seq=1024,  # 2048 OOMs on 3B/16GB; reasoning traces truncated but still useful
            resume_from=base_b,
        )

    if run_c:
        print("\n── Stage C: Tool-Call SFT ────────────────────────────────")
        build_tools_data()
        base_c = args.from_adapter if (args.stage == "c" and args.from_adapter) else ADAPTER_3B_V2
        run_sft(
            data_dir=DATA_TOOLS,
            output_dir=ADAPTER_3B_V3,
            iters=args.iters_c,
            lr=5e-6,
            max_seq=1024,
            resume_from=base_c,
        )

    final = ADAPTER_3B_V3 if run_c else (ADAPTER_3B_V2 if run_b else ADAPTER_3B_V1)
    print(f"\nPhase 6 complete → {final}")
    print(f"Eval: ../.venv/bin/python3 evaluate_mac.py --adapter {final} --model {MODEL_ID}")
    print(f"Chat: set ADAPTER_DIR='{final}', MODEL_ID='{MODEL_ID}' in 2_mac_chatbot.py")


if __name__ == "__main__":
    main()
