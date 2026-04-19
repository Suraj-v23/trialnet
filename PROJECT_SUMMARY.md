# TrialNet: The Continuous Self-Learning AI Project 🧠

## 🌟 Project Overview
Traditional AI models are static: once trained, they freeze. If they make a mistake playing chess or writing code, they will repeat that exact same mistake unless a data scientist manually curates a new dataset and retrains the entire model.

This project was built to solve that flaw. We designed **TrialNet**, a novel architectural framework that allows an AI to continuously learn from its mistakes and human feedback *during* production inference. 

We successfully implemented this architecture in two distinct ecosystems across two phases.

---

## Phase 1: Developing the "Try and Learn" Core Architecture (NumPy)
To prove the math and mechanics of self-learning without relying on heavy frameworks, we built a complete Dual-Learning neural network from absolute scratch using pure Python and NumPy.

### Core Components Built:
1. **The Base Architecture**: Custom Tensors, Dense/Dropout Layers, and Activation functions (ReLU, Softmax).
2. **Traditional Learning Engine**: Standard SGD/Adam optimization with backpropagation.
3. **The Novel Ecosystem**:
   - **Error Memory Bank**: A prioritized short-term memory system that caches high-confidence mistakes made during evaluation.
   - **Mistake Pattern Analyzer**: Scans the memory bank for persistent confusion (e.g., repeatedly confusing `9` and `4`).
   - **Perturbation Explorer**: Tests microscopic weight changes on specific neurons without breaking the overarching baseline model.
   - **Try-and-Learn Orchestrator**: The conductor that blends the traditional gradients with the exploratory, targeted fixes for a hybrid training loop.

### Phase 1 Results:
- **Baseline**: A digit classifier (MNIST) trained from scratch.
- **Milestone**: The model achieved **93.9% accuracy**. The Try-and-Learn orchestrator successfully corrected exactly *20,587 mistakes* throughout the training run before they could become ingrained in the model weights.

---

## Phase 2: Scaling to Massive Large Language Models (PyTorch / HuggingFace)
Having proven the math on rigid image inputs, we pivoted the architecture to handle dynamic Generative AI. We applied the TrialNet concepts to a massive **7-Billion parameter LLM** (Large Language Model). 

Because fine-tuning billions of parameters requires exorbitant GPU clusters, we engineered a pipeline specifically for **Google Colab Free Tier (15GB VRAM)**.

### The Colab TrialNet Pipeline
1. **The LLM Brain**: Using `Qwen2.5-7B-Instruct`, squashed via **4-bit Quantization** (`bitsandbytes`) to fit inside limited consumer hardware.
2. **LoRA Fine-Tuning**: Hooked up Low-Rank Adapters (`PEFT`) to modify less than 0.1% of the model's weights while preserving its immense core knowledge.
3. **Advanced Reasoning**: Executed Supervised Fine-Tuning (SFT) using the `Opus-4.6-Reasoning` dataset, teaching the model to output `<thinking>` structures before reaching a final solution.
4. **The Live Mistake Capture System**: An interactive chatbot shell where a human user can interact with the LLM. If the LLM's logic fails, the user types `/correct [feedback]`, instantly caching the prompt and human correction into an updated `Error Memory Bank.jsonl`.
5. **The Self-Correction Loop**: Using HuggingFace's `DPOTrainer` (Direct Preference Optimization), we created an asynchronous loop that mathematically punishes the LLM's recorded hallucinations and heavily rewards the neural pathways that trace the user's manual correction, resulting in a newly minted "Smarter" Adapter.

---

## 🛠 Tech Stack
- **Math & Proof of Concept**: `Python`, `NumPy`, `Matplotlib`
- **Dashboard UI**: `Flask`, `Chart.js`, Vanilla `CSS` (Glassmorphism)
- **Generative AI Scalability**: `PyTorch`, `HuggingFace Transformers`, `PEFT (LoRA)`, `TRL (Transformer Reinforcement Learning)`, `Datasets`

## 🔮 Future Roadmap
- Automate the DPO evaluation cycle by using an "LLM-as-a-Judge" (a stronger model grading the 7B model) to remove the need for human `/correct` inputs entirely.
- Shift the JSON error-memory bank to a Vector Database (`ChromaDB` or `Milvus`) so the model can contextually query past mistakes via RAG before it answers a user prompt!
