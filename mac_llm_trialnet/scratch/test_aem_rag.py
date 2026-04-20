
import json
import os
import re
from mlx_lm import load, generate

MODEL_ID     = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR  = "./mac_trialnet_v8_adapter"

print(f"Loading TrialNet [{os.path.basename(ADAPTER_DIR)}] via MLX...")
model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER_DIR)

def chat(user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are TrialNet, a continuously self-improving AI assistant running on Apple Silicon. Answer accurately and concisely."},
        {"role": "user", "content": user_prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=prompt_text, max_tokens=1000, verbose=False)

questions = [
    "Write a complete JavaScript function for an AEM component that fetches the user query, sends it to a RAG API endpoint (/bin/trialnet/rag), and updates a <div> with the response. Include basic error handling.",
    "Implement a simple cosine similarity function in JavaScript to compare two vectors (arrays of numbers).",
    "Write an AEM clientlib JavaScript snippet that captures a search input, prevents default submission, and instead calls our local TrialNet inference engine via a POST request."
]

for i, q in enumerate(questions, 1):
    print(f"\n--- Question {i} ---\n{q}")
    response = chat(q)
    print(f"\n--- Response {i} ---\n{response}")
    print("-" * 40)
