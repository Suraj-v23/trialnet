from flask import Flask, request, jsonify, render_template
import json
import os
import threading
from mlx_lm import load, generate

app = Flask(__name__)

# Constants
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# Always point the UI to the smartest model
ADAPTER_DIR = "../mac_trialnet_v2_smarter_adapter"
MEMORY_FILE = "../mac_mistakes_memory.jsonl"

# Global model state
model = None
tokenizer = None
model_lock = threading.Lock()

print("🔥 Loading Native Apple MLX Model into memory for the web server...")
try:
    model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER_DIR)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load v2 adapter ({e}). Falling back to v1...")
    ADAPTER_DIR_FALLBACK = "../mac_trialnet_v1_adapter"
    model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER_DIR_FALLBACK)
    print("✅ V1 fallback Model loaded successfully!")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("prompt", "")
    
    if not user_input:
        return jsonify({"error": "No prompt provided"}), 400

    messages = [{"role": "user", "content": user_input}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    with model_lock:
        # Generate natively on M4 GPU
        output = generate(
            model, 
            tokenizer, 
            prompt=prompt_text, 
            verbose=False,
            max_tokens=400
        )
        
    return jsonify({"response": output})

@app.route("/correct", methods=["POST"])
def correct_mistake():
    data = request.json
    prompt = data.get("prompt", "")
    bad_generation = data.get("bad_generation", "")
    human_correction = data.get("human_correction", "")
    
    if not prompt or not human_correction:
        return jsonify({"error": "Missing data"}), 400

    log_entry = {
        "prompt": prompt,
        "bad_generation": bad_generation,
        "human_correction": human_correction
    }
    
    # Save the interaction state for later self-correction!
    with open(MEMORY_FILE, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
        
    return jsonify({"status": "success", "message": "Correction saved to TrialNet Error Memory Bank!"})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
