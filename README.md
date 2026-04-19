# 🧠 TrialNet — A Self-Learning AI That Learns From Its Mistakes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon%20%7C%20Colab-lightgrey.svg)]()

**TrialNet** is a self-improving AI system with two layers:
1. **TrialNet Core** — A neural network built **from scratch with NumPy** that uses a novel *Try-and-Learn* engine to remember and correct its mistakes.
2. **TrialNet LLM** — A locally-running Large Language Model (Qwen2.5-1.5B) fine-tuned on Apple Silicon via MLX with a **ChromaDB-backed mistake memory** and an **automated LLM-as-Judge** for continuous self-correction.

---

## 🚀 What Makes This Different?

| Feature | Traditional Models | TrialNet |
|---|---|---|
| Error handling | Forgotten after weight update | **Error Memory Bank** stores & prioritizes mistakes |
| Learning signal | Loss gradient only | Gradient + **targeted mistake replay** |
| Self-awareness | None | **Mistake Pattern Analyzer** discovers failure patterns |
| Weight updates | Gradient descent only | Gradient + **Perturbation Explorer** |
| LLM Memory | Static weights | **ChromaDB RAG** + continuous LoRA self-correction |
| Feedback | Manual retraining | **Auto-judge scores every response**, logs bad ones |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        TrialNet System                          │
│                                                                │
│  ┌──────────────────────┐    ┌────────────────────────────┐   │
│  │   TrialNet Core      │    │   TrialNet LLM (Mac)       │   │
│  │   (NumPy from scratch)│    │   Qwen2.5-1.5B + MLX LoRA │   │
│  │                      │    │                            │   │
│  │  [Traditional SGD]   │    │  [ChromaDB Memory Bank]    │   │
│  │      +               │    │  ← stores mistakes         │   │
│  │  [Try-and-Learn]     │    │       ↓                    │   │
│  │  ┌──────────────┐    │    │  [LLM-as-Judge]            │   │
│  │  │ Error Memory │    │    │  ← scores every response   │   │
│  │  │ Bank         │    │    │       ↓                    │   │
│  │  ├──────────────┤    │    │  [MLX LoRA Self-Correction]│   │
│  │  │ Mistake      │    │    │  ← injects corrections     │   │
│  │  │ Analyzer     │    │    │    into new adapter        │   │
│  │  ├──────────────┤    │    └────────────────────────────┘   │
│  │  │ Perturbation │    │                                      │
│  │  │ Explorer     │    │                                      │
│  │  └──────────────┘    │                                      │
│  └──────────────────────┘                                      │
└────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### TrialNet Core (NumPy Engine)
```bash
git clone https://github.com/Suraj-v23/trialnet.git
cd trialnet
pip install -r requirements.txt
```

### TrialNet LLM (Apple Silicon — Requires M1/M2/M3/M4 Mac)
```bash
cd mac_llm_trialnet
pip install -r requirements_mac.txt
```

---

## 🎯 Quick Start

### 1. Train the Core (NumPy) Model
```bash
# Hybrid mode — recommended
python train.py --mode hybrid --epochs 15

# Compare all three learning modes
python train.py --mode compare --epochs 15

# With live dashboard at http://localhost:5050
cd dashboard && python server.py
python train.py --mode hybrid --epochs 15 --dashboard
```

### 2. Evaluate
```bash
python evaluate.py --model saved_models/hybrid
```

### 3. Run the Local LLM Chatbot (Apple Silicon only)
```bash
cd mac_llm_trialnet

# Step 1 — Fine-tune on hybrid logic + coding curriculum
python 1_mac_finetune.py

# Step 2 — Chat (Auto-judge runs on every response)
python 2_mac_chatbot.py

# Step 3 — Self-correct (run after 10+ logged mistakes)
bash run_self_correction.sh
```

---

## 📁 Project Structure

```
trialnet/
├── train.py                    # Core training script
├── evaluate.py                 # Evaluation script
├── requirements.txt            # Core dependencies
│
├── trialnet/                   # NumPy neural network library
│   ├── model.py                # Main TrialNet class
│   ├── utils.py                # Data loading utilities
│   ├── core/                   # Neural network fundamentals
│   │   ├── tensor.py           # Custom tensor ops (no PyTorch)
│   │   ├── layers.py           # Dense, Dropout, BatchNorm
│   │   ├── activations.py      # ReLU, Sigmoid, Softmax, etc.
│   │   └── losses.py           # CrossEntropy, MSE
│   └── learning/               # Learning engines
│       ├── traditional.py        # SGD, Adam optimizers
│       ├── error_memory.py       # Error Memory Bank ⭐ NOVEL
│       ├── mistake_analyzer.py   # Pattern discovery ⭐ NOVEL
│       ├── perturbation.py       # Weight exploration ⭐ NOVEL
│       └── trial_learner.py      # Orchestrator ⭐ NOVEL
│
├── mac_llm_trialnet/           # Apple Silicon LLM pipeline
│   ├── 1_mac_finetune.py         # Hybrid LoRA fine-tuning
│   ├── 2_mac_chatbot.py          # Chat + Auto-judge
│   ├── 3_mac_self_correct.py     # Self-correction loop
│   ├── evaluate_mac.py           # Regression eval baseline
│   ├── run_self_correction.sh    # Full pipeline runner
│   ├── requirements_mac.txt      # MLX + ChromaDB deps
│   └── memory/
│       ├── chroma_bank.py        # ChromaDB mistake banking
│       └── judge.py              # LLM-as-Judge scorer
│
├── colab_llm_trialnet/         # Google Colab pipeline
│   ├── 1_base_finetune.py
│   ├── 2_colab_chatbot.py
│   └── 3_self_correction_loop.py
│
├── dashboard/                  # Real-time training dashboard
│   ├── server.py               # Flask API
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── PROGRESS.md                 # Build progress log
├── ROADMAP.md                  # Planned features
├── LICENSE                     # MIT
└── README.md                   # This file
```

---

## 🔬 The Three Core Learning Modes

### 1. Traditional
Standard neural network — forward pass, loss, backpropagation, gradient descent.

### 2. Trial
Only the novel Try-and-Learn system — no gradient descent at all:
- **Error Memory Bank**: Stores mistakes with priority scoring (high-confidence wrong answers get highest priority)
- **Perturbation Explorer**: Random weight experiments — keep what helps, revert what hurts
- **Targeted Replay**: Spends more training time on the hardest, most-repeated mistakes

### 3. Hybrid *(recommended)*
Combines both. Traditional gradients provide the base learning signal; Try-and-Learn provides targeted corrections for stubborn mistakes.

---

## 🤖 LLM Self-Correction Pipeline (Apple Silicon)

```
You (user) ──► Chat ──► LLM Response
                              │
                         LLM Judge scores (0–10)
                              │
                    ┌─────────┴──────────┐
                score ≤ 5 (bad)      score ≥ 8 (good)
                    │
              Auto-logged to ChromaDB
                    │
              /correct [fix]  ← you provide the right answer
                    │
         (10 mistakes collected)
                    │
           bash run_self_correction.sh
                    │
            New LoRA adapter created
                    │
         Model permanently updated ✅
```

---

## 🔧 Built With

| Component | Technology |
|---|---|
| Core neural network | Pure NumPy (no PyTorch/TensorFlow) |
| LLM backbone | Qwen/Qwen2.5-1.5B-Instruct |
| Apple Silicon inference | MLX + mlx-lm |
| Mistake memory | ChromaDB (vector database) |
| Auto-judge | LLM-as-Judge (same local model) |
| Dashboard | Flask + Chart.js |

---

## 🕸️ Knowledge Graph (Graphify)

This project uses [Graphify](https://github.com/safishamsi/graphify) 
for AI-assisted code understanding.

To generate the knowledge graph locally:

```bash
pip install graphify
graphify .
```

Then open `graphify-out/graph.html` to explore the codebase visually.

---

## 🗺️ Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features including DPO training, GRPO alignment, and multimodal capabilities.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to:
1. Fork the repository
2. Create a branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Suraj Verma** · [@Suraj-v23](https://github.com/Suraj-v23)

*Building AI that learns the way humans do — by remembering and correcting its mistakes.*
