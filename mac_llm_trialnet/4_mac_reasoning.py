"""
4_mac_reasoning.py — Phase 4: Extended Thinking / Reasoning SFT

Trains the model to use <thinking> blocks for multi-step math and logic.
Loads 200 curated reasoning traces from Opus-4.6-Reasoning dataset,
builds on top of latest non-bad adapter.

Usage:
  python3 4_mac_reasoning.py              # auto-detect latest adapter
  python3 4_mac_reasoning.py --iters 200  # default
  python3 4_mac_reasoning.py --from v4    # explicit base adapter
"""

from __future__ import annotations
import os
import re
import json
import random
import argparse
import subprocess

from datasets import load_dataset

MODEL_ID      = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_ID    = "nohurry/Opus-4.6-Reasoning-3000x-filtered"
DATA_DIR      = "./mac_reasoning_data"
ADAPTERS_ROOT = "."
SKIP_ADAPTERS = {"mac_trialnet_v2_smarter_adapter"}

N_SAMPLES   = 200
MAX_SEQ_LEN = 2048

# System prompt with budget control — teaches model to think before answering
SYSTEM_PROMPT = (
    "You are TrialNet, a continuously self-improving AI assistant. "
    "For complex problems, think step by step inside <thinking>...</thinking> tags "
    "before giving your final answer. Think for up to 300 tokens before answering."
)

PRIORITY_CATEGORIES = {"math", "reasoning", "logic", "arithmetic", "algebra"}


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


def next_adapter_dir(current_v: int) -> str:
    return f"./mac_trialnet_v{current_v + 1}_adapter"


def build_training_data(n: int = N_SAMPLES) -> int:
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Loading {DATASET_ID} (streaming)...")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    priority, rest = [], []
    for example in ds:
        cat = str(example.get("category", "")).lower()
        thinking = (example.get("thinking") or "").strip()
        solution = (example.get("solution") or "").strip()
        problem  = (example.get("problem")  or "").strip()

        if not thinking or not solution or not problem:
            continue
        if len(thinking) < 50:   # too shallow, skip
            continue

        row = {
            "problem":  problem,
            "thinking": thinking,
            "solution": solution,
        }
        if any(p in cat for p in PRIORITY_CATEGORIES):
            priority.append(row)
        else:
            rest.append(row)

        if len(priority) + len(rest) >= n * 3:  # collect 3× then trim
            break

    random.shuffle(priority)
    random.shuffle(rest)

    # Fill quota: prefer math/logic, pad with rest
    selected = (priority + rest)[:n]
    if len(selected) < n:
        print(f"  Warning: only found {len(selected)} samples (target {n})")

    rows = []
    for ex in selected:
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": ex["problem"]},
            {"role": "assistant", "content": f"<thinking>\n{ex['thinking']}\n</thinking>\n\n{ex['solution']}"},
        ]
        rows.append(json.dumps({"messages": messages}))

    random.shuffle(rows)
    split = max(1, len(rows) // 8)  # ~12.5% validation

    with open(f"{DATA_DIR}/train.jsonl", "w") as ft, \
         open(f"{DATA_DIR}/valid.jsonl", "w") as fv:
        for i, row in enumerate(rows):
            if i < split:
                fv.write(row + "\n")
            else:
                ft.write(row + "\n")

    train_n = len(rows) - split
    print(f"  {train_n} train / {split} valid samples written to {DATA_DIR}/")
    return train_n


def run_sft(adapter_path: str, output_dir: str, iters: int):
    cmd = [
        "../.venv/bin/python3", "-m", "mlx_lm.lora",
        "--model",          MODEL_ID,
        "--train",
        "--data",           DATA_DIR,
        "--iters",          str(iters),
        "--batch-size",     "1",
        "--adapter-path",   output_dir,
        "--resume-adapter-file", f"{adapter_path}/adapters.safetensors",
        "--max-seq-length", str(MAX_SEQ_LEN),
        "--learning-rate",  "1e-5",
    ]
    print(f"\nRunning MLX LoRA SFT ({iters} iters, max_seq={MAX_SEQ_LEN})...")
    print(f"  Base: {adapter_path} → {output_dir}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--from",  dest="from_v", type=str, default=None,
                        help="Base adapter version, e.g. v4")
    args = parser.parse_args()

    # Resolve base adapter
    if args.from_v:
        base_dir  = f"./mac_trialnet_{args.from_v}_adapter"
        v_num     = int(args.from_v.lstrip("v"))
    else:
        base_name, v_num = find_latest_adapter()
        if not base_name:
            print("No adapter found. Run 1_mac_finetune.py first.")
            return
        base_dir = f"./{base_name}"

    output_dir = next_adapter_dir(v_num)
    print(f"Phase 4 — Reasoning SFT")
    print(f"  Base adapter : {base_dir}")
    print(f"  New adapter  : {output_dir}")
    print(f"  Iters        : {args.iters}")

    # Build data
    n = build_training_data(N_SAMPLES)

    # Train
    run_sft(base_dir, output_dir, args.iters)

    print(f"\nPhase 4 complete. New adapter → {output_dir}")
    print(f"Eval: ../.venv/bin/python3 evaluate_mac.py --adapter {output_dir}")
    print(f"Chat: update ADAPTER_DIR in 2_mac_chatbot.py to '{output_dir}'")


if __name__ == "__main__":
    main()
