"""
1_mac_finetune.py — Hybrid Foundational Training (Apple Silicon)

Merges Logic/Reasoning (Opus) and Code Generation (Python Alpaca) into 
a single high-quality curriculum for the Qwen 1.5B model.
"""

import os
import json
import random
import subprocess
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_REASONING = "nohurry/Opus-4.6-Reasoning-3000x-filtered"
DATASET_CODING = "iamtarun/python_code_instructions_18k_alpaca"

DATA_DIR = "./mac_data"
OUTPUT_DIR = "./mac_trialnet_v1_adapter"

os.makedirs(DATA_DIR, exist_ok=True)

print(f"📚 Loading Reasoning Dataset ({DATASET_REASONING})...")
ds_reasoning = load_dataset(DATASET_REASONING, split="train").select(range(100))

print(f"💻 Loading Coding Dataset ({DATASET_CODING})...")
ds_coding = load_dataset(DATASET_CODING, split="train").select(range(100))

print("⚙️ Formatting Hybrid Curriculum into MLX JSONL...")

# Prepare list of messages
hybrid_data = []

# 1. Process Reasoning
for example in ds_reasoning:
    thinking = example['thinking'] if example['thinking'] else ""
    solution = example['solution'] if example['solution'] else ""
    problem = example['problem']
    
    messages = [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": f"<thinking>\n{thinking}\n</thinking>\n\n{solution}"}
    ]
    hybrid_data.append(messages)

# 2. Process Coding
for example in ds_coding:
    instruction = example.get('instruction', '')
    inp = example.get('input', '')
    output = example.get('output', '')
    user_content = f"{instruction}\n\n{inp}".strip()
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output}
    ]
    hybrid_data.append(messages)

# Shuffle the curriculum so it learns both simultaneously!
random.shuffle(hybrid_data)

with open(f"{DATA_DIR}/train.jsonl", "w") as f_train, open(f"{DATA_DIR}/valid.jsonl", "w") as f_valid:
    for i, messages in enumerate(hybrid_data):
        row = json.dumps({"messages": messages})
        if i % 10 == 0:
            f_valid.write(row + "\n")
        else:
            f_train.write(row + "\n")

print("✅ Hybrid Data formatted natively for MLX!")
print("\n🚀 Triggering MLX LoRA Fine-Tuning... (This will run blazingly fast on your M4 GPU!)")

cmd = [
    "python3", "-m", "mlx_lm.lora",
    "--model", MODEL_ID,
    "--train",
    "--data", DATA_DIR,
    "--iters", "200", 
    "--batch-size", "1", 
    "--adapter-path", OUTPUT_DIR,
    "--max-seq-length", "1024"
]

try:
    subprocess.run(cmd, check=True)
    print(f"\n🎉 Training complete! Hybrid Adapter saved safely to {OUTPUT_DIR}")
except subprocess.CalledProcessError as e:
    print(f"\n❌ MLX Training Failed! Please ensure you pip installed requirements_mac.txt. Error: {e}")
