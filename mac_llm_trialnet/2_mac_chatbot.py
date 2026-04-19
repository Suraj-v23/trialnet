"""
2_mac_chatbot.py — TrialNet Chatbot with RAG Error Memory (Apple Silicon)

Commands:
  /correct [answer]   — log model's last mistake + your correction
  /memory             — show how many mistakes are stored
  /stats              — show current adapter + memory count
  quit                — exit
"""

import os
import json
from mlx_lm import load, generate
from memory.chroma_bank import ChromaMemoryBank

MODEL_ID    = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "./mac_trialnet_v2_smarter_adapter"
LEGACY_JSONL = "./mac_mistakes_memory.jsonl"

BASE_SYSTEM = (
    "You are TrialNet, a continuously self-improving AI assistant running on Apple Silicon. "
    "You learn from your mistakes. Answer accurately and concisely."
)

print("Initializing Error Memory Bank (ChromaDB)...")
memory = ChromaMemoryBank()

# One-time migration of legacy JSONL into ChromaDB
if os.path.exists(LEGACY_JSONL) and memory.count() == 0:
    n = memory.migrate_jsonl(LEGACY_JSONL)
    if n:
        print(f"  Migrated {n} legacy mistake(s) from JSONL → ChromaDB")

print(f"Loading TrialNet [{os.path.basename(ADAPTER_DIR)}] on MLX...")
model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER_DIR)

print("\n" + "=" * 55)
print(f"  TrialNet LLM — adapter: {os.path.basename(ADAPTER_DIR)}")
print(f"  Memory bank: {memory.count()} mistake(s) stored")
print("  /correct [answer]  |  /memory  |  /stats  |  quit")
print("=" * 55 + "\n")

last_prompt   = ""
last_response = ""


def build_system_prompt(user_prompt: str) -> str:
    injection = memory.build_system_injection(user_prompt)
    if injection:
        return BASE_SYSTEM + "\n\n" + injection
    return BASE_SYSTEM


def chat(user_prompt: str) -> str:
    system = build_system_prompt(user_prompt)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, prompt=prompt_text, max_tokens=400, verbose=False)


while True:
    try:
        user_input = input("\nYou: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        break

    if not user_input:
        continue

    if user_input.lower() == "quit":
        break

    if user_input.lower() == "/memory":
        print(f"  Memory bank: {memory.count()} mistake(s) stored in ChromaDB")
        continue

    if user_input.lower() == "/stats":
        print(f"  Adapter : {ADAPTER_DIR}")
        print(f"  Model   : {MODEL_ID}")
        print(f"  Memory  : {memory.count()} mistake(s)")
        continue

    if user_input.startswith("/correct "):
        correction = user_input[len("/correct "):].strip()
        if not last_prompt:
            print("  Nothing to correct yet.")
        elif not correction:
            print("  Usage: /correct [the right answer]")
        else:
            memory.add_mistake(last_prompt, last_response, correction)
            print(f"  Logged to memory bank. Total: {memory.count()} mistake(s).")
            print("  Run python3 3_mac_self_correct.py when you have 10+ mistakes.")
        continue

    print("\nTrialNet: ", end="", flush=True)
    response = chat(user_input)
    print(response)

    last_prompt   = user_input
    last_response = response
