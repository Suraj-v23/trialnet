"""
3_mac_self_correct.py — TrialNet Self-Correction Loop (Apple Silicon)

Reads ChromaDB memory bank → exports correction pairs → runs MLX LoRA SFT
to inject corrections into a new versioned adapter.

Usage:
  python3 3_mac_self_correct.py                  # auto-detect latest adapter, create next version
  python3 3_mac_self_correct.py --from v2 --to v3
"""

import os
import re
import json
import argparse
import subprocess

from memory.chroma_bank import ChromaMemoryBank

# Defaults
DEFAULT_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTERS_ROOT  = "."
CORRECTION_DIR = "./mac_correction_data"
MIN_MISTAKES   = 10   
SKIP_ADAPTERS  = {"mac_trialnet_v2_smarter_adapter"}  

def find_latest_adapter(model_id: str) -> tuple[str, int]:
    """Find highest-versioned non-bad adapter directory for the given model size."""
    # Detect if we are looking for 3b or 1.5b
    is_3b = "3B" in model_id.upper()
    prefix = "mac_trialnet_3b_v" if is_3b else "mac_trialnet_v"
    
    pattern = re.compile(rf"{prefix}(\d+)")
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


def build_sft_data(pairs_path: str, out_dir: str):
    """Format exported DPO pairs + manual corrections as MLX SFT JSONL."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    # ChromaDB correction pairs
    if os.path.exists(pairs_path):
        with open(pairs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = json.loads(line)
                messages = [
                    {"role": "user",      "content": p["prompt"]},
                    {"role": "assistant", "content": p["chosen"]},
                ]
                rows.append(json.dumps({"messages": messages}))

    # Manual corrections
    manual_path = os.path.join(out_dir, "manual_corrections.jsonl")
    if os.path.exists(manual_path):
        with open(manual_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(line)

    import random as _random
    _random.shuffle(rows)

    with open(f"{out_dir}/train.jsonl", "w") as ft, \
         open(f"{out_dir}/valid.jsonl", "w") as fv:
        for i, row in enumerate(rows):
            ft.write(row + "\n")
            if i % max(1, len(rows) // 5) == 0:  
                fv.write(row + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model ID (default: Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--from", dest="from_adapter", default=None,
                        help="Source adapter dir (default: auto-detect latest)")
    parser.add_argument("--to", dest="to_adapter", default=None,
                        help="Output adapter dir (default: auto-increment version)")
    parser.add_argument("--iters", type=int, default=50,
                        help="Training iterations (default 50)")
    parser.add_argument("--lr", type=float, default=2e-6,
                        help="Learning rate (default 2e-6)")
    args = parser.parse_args()

    # --- Memory check ---
    memory = ChromaMemoryBank()
    count = memory.count()
    manual_path = os.path.join(CORRECTION_DIR, "manual_corrections.jsonl")
    manual_count = 0
    if os.path.exists(manual_path):
        with open(manual_path) as f:
            manual_count = sum(1 for _ in f)
            
    total_pairs = count + manual_count
    print(f"Model ID   : {args.model}")
    print(f"Memory bank: {count} ChromaDB mistake(s) + {manual_count} manual correction(s) = {total_pairs} total.")
    
    if total_pairs < MIN_MISTAKES:
        print(f"Need at least {MIN_MISTAKES} total pairs before self-correction is meaningful.")
        return

    # --- Adapter versioning ---
    latest_dir, latest_v = find_latest_adapter(args.model)
    from_dir = args.from_adapter or (f"./{latest_dir}" if latest_dir else None)
    to_v     = latest_v + 1
    
    is_3b = "3B" in args.model.upper()
    prefix = "mac_trialnet_3b_v" if is_3b else "mac_trialnet_v"
    to_dir = args.to_adapter or f"./{prefix}{to_v}_adapter"

    if not from_dir or not os.path.exists(from_dir):
        print(f"No source adapter found for {args.model}. Initializing first version...")
        # If no adapter exists, we don't use --resume-adapter-file
        from_dir = None
    else:
        print(f"Source adapter : {from_dir}")
        
    print(f"New adapter    : {to_dir}")

    # --- Export pairs from ChromaDB ---
    pairs_path = f"{CORRECTION_DIR}/dpo_pairs.jsonl"
    exported = memory.export_dpo_pairs(pairs_path)
    print(f"Exported {exported} correction pair(s) from ChromaDB.")

    # --- Format as MLX SFT data ---
    build_sft_data(pairs_path, CORRECTION_DIR)
    print(f"SFT data written to {CORRECTION_DIR}/")

    # --- Run MLX LoRA fine-tune ---
    cmd = [
        "python3", "-m", "mlx_lm.lora",
        "--model",               args.model,
        "--train",
        "--data",                CORRECTION_DIR,
        "--iters",               str(args.iters),
        "--batch-size",          "1",
        "--learning-rate",       str(args.lr),
        "--adapter-path",        to_dir,
        "--max-seq-length",      "1024",
    ]
    if from_dir:
        cmd += ["--resume-adapter-file", f"{from_dir}/adapters.safetensors"]

    print(f"\nRunning MLX LoRA ({args.iters} iters, lr={args.lr})...")
    try:
        subprocess.run(cmd, check=True)
        print(f"\nSelf-correction complete. New adapter → {to_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\nMLX training failed: {e}")

if __name__ == "__main__":
    main()
