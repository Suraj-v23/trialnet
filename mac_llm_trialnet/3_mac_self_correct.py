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

MODEL_ID       = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTERS_ROOT  = "."
CORRECTION_DIR = "./mac_correction_data"
MIN_MISTAKES   = 5   # don't run if fewer than this


def find_latest_adapter() -> tuple[str, int]:
    """Find highest-versioned adapter directory."""
    pattern = re.compile(r"mac_trialnet_v(\d+)")
    best_v, best_dir = 0, None
    for name in os.listdir(ADAPTERS_ROOT):
        m = pattern.match(name)
        if m and os.path.isdir(os.path.join(ADAPTERS_ROOT, name)):
            v = int(m.group(1))
            if v > best_v:
                best_v, best_dir = v, name
    return best_dir, best_v


def build_sft_data(pairs_path: str, out_dir: str):
    """Format exported DPO pairs as MLX SFT JSONL (prompt → correction)."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    with open(pairs_path) as f:
        for line in f:
            p = json.loads(line)
            messages = [
                {"role": "user",      "content": p["prompt"]},
                {"role": "assistant", "content": p["chosen"]},
            ]
            rows.append(json.dumps({"messages": messages}))

    with open(f"{out_dir}/train.jsonl", "w") as ft, \
         open(f"{out_dir}/valid.jsonl", "w") as fv:
        for i, row in enumerate(rows):
            ft.write(row + "\n")
            if i % max(1, len(rows) // 5) == 0:  # ~20% goes to valid
                fv.write(row + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_adapter", default=None,
                        help="Source adapter dir (default: auto-detect latest)")
    parser.add_argument("--to", dest="to_adapter", default=None,
                        help="Output adapter dir (default: auto-increment version)")
    parser.add_argument("--iters", type=int, default=50,
                        help="Training iterations (default 50)")
    args = parser.parse_args()

    # --- Memory check ---
    memory = ChromaMemoryBank()
    count = memory.count()
    print(f"Memory bank: {count} mistake(s) stored.")
    if count < MIN_MISTAKES:
        print(f"Need at least {MIN_MISTAKES} mistakes before self-correction is meaningful.")
        print("Keep chatting and use /correct to log more mistakes.")
        return

    # --- Adapter versioning ---
    latest_dir, latest_v = find_latest_adapter()
    from_dir = args.from_adapter or (f"./{latest_dir}" if latest_dir else None)
    to_v     = latest_v + 1
    to_dir   = args.to_adapter or f"./mac_trialnet_v{to_v}_adapter"

    if not from_dir or not os.path.exists(from_dir):
        print(f"No source adapter found. Check that a mac_trialnet_v* dir exists.")
        return

    print(f"Source adapter : {from_dir}")
    print(f"New adapter    : {to_dir}")

    # --- Export pairs from ChromaDB ---
    pairs_path = f"{CORRECTION_DIR}/dpo_pairs.jsonl"
    exported = memory.export_dpo_pairs(pairs_path)
    print(f"Exported {exported} correction pair(s) from ChromaDB.")
    if exported == 0:
        print("No pairs with both bad_generation and human_correction. Use /correct in chatbot.")
        return

    # --- Format as MLX SFT data ---
    build_sft_data(pairs_path, CORRECTION_DIR)
    print(f"SFT data written to {CORRECTION_DIR}/")

    # --- Run MLX LoRA fine-tune on corrections ---
    cmd = [
        "python3", "-m", "mlx_lm.lora",
        "--model",               MODEL_ID,
        "--train",
        "--data",                CORRECTION_DIR,
        "--iters",               str(args.iters),
        "--batch-size",          "1",
        "--learning-rate",       "1e-5",
        "--resume-adapter-file", f"{from_dir}/adapters.safetensors",
        "--adapter-path",        to_dir,
        "--max-seq-length",      "1024",
    ]

    print(f"\nRunning MLX LoRA ({args.iters} iters)...")
    try:
        subprocess.run(cmd, check=True)
        print(f"\nSelf-correction complete. New adapter → {to_dir}")
        print(f"Update ADAPTER_DIR in 2_mac_chatbot.py to '{to_dir}' to use it.")
        print(f"Run: python3 evaluate_mac.py --adapter {to_dir}")
        print(f"Then: python3 evaluate_mac.py --compare "
              f"{os.path.basename(from_dir)} {os.path.basename(to_dir)}")
    except subprocess.CalledProcessError as e:
        print(f"\nMLX training failed: {e}")


if __name__ == "__main__":
    main()
