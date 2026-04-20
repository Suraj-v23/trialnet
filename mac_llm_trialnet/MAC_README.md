# TrialNet 1.5B LLM (Apple Silicon Native Edition)

This directory contains the completely refactored Try-and-Learn continuous learning pipeline, specifically designed to run on macOS using Apple's incredibly optimized **MLX Framework**. By dropping PyTorch and `bitsandbytes`, this model will dynamically share your 16GB of Unified memory to allow blazing fast training without crashing your Mac.

## Setup Instructions

### 1. Open Terminal and Navigate Here

```bash
cd trialnet/mac_llm_trialnet/
```

### 2. Install MLX

You can install the dependencies directly to your system or virtual environment:

```bash
pip install -r requirements_mac.txt
```

### 3. Start TrialNet Base Fine-Tuning

This script downloads a highly intelligent 1.5 Billion parameter model, automatically quantizes it utilizing MLX, formats the Opus logic dataset, and runs native Apple LoRA training:

```bash
python3 1_mac_finetune.py
```

*Note: Because you have an M4 chip, this will allocate memory directly to the GPU cores. You should hear your fans spin up as it learns!*

### 4. Interactive Chatbot (The Mistake Bank)

Once the training finishes, chat with your new creation!

```bash
python3 2_mac_chatbot.py
```

If the AI hallucinates, use the Try-and-Learn framework: type `/correct [your answer]` in the prompt. It will securely cache it to `mac_mistakes_memory.jsonl`.

### 5. Self-Correction Loop

Once you've corrected a few mistakes, run:

```bash
python3 3_mac_self_correct.py
```

This forces MLX to reload your adapter, inject your new human corrections, and aggressively reinforce them so the model never makes the same mistake again!
