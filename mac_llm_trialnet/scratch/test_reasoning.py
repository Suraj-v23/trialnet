
import json
import os
import re
from mlx_lm import load, generate

MODEL_ID     = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR  = "./mac_trialnet_3b_v4_adapter"

print(f"Loading TrialNet [{os.path.basename(ADAPTER_DIR)}] via MLX...")
model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER_DIR)

def chat(user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are TrialNet, a continuously self-improving AI assistant. Think step by step using <thinking> tags before answering."},
        {"role": "user", "content": user_prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Give it more tokens to think
    return generate(model, tokenizer, prompt=prompt_text, max_tokens=1500, verbose=False)

questions = [
    "Sally has 3 brothers. Each of her brothers has 2 sisters. How many sisters does Sally have?",
    "A man is looking at a photograph of someone. His friend asks who it is. The man replies: 'Brothers and sisters I have none, but that man's father is my father's son.' Who is in the photograph?",
    "If yesterday was tomorrow, today would be Friday. What day is it today?",
    "If you have a 3-gallon jug and a 5-gallon jug, how can you get exactly 4 gallons of water? (Assume you have an infinite supply of water and no markings on the jugs)"
]

for i, q in enumerate(questions, 1):
    print(f"\n--- Reasoning Challenge {i} ---\n{q}")
    response = chat(q)
    print(f"\n--- TrialNet Response ---\n{response}")
    print("-" * 60)
