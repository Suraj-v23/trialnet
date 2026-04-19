# 🧠 TrialNet — A Neural Network That Learns From Its Mistakes

**TrialNet** is a novel AI model built entirely from scratch with NumPy. Unlike traditional neural networks that simply minimize a loss function, TrialNet uses a **Dual Learning Engine** that explicitly remembers, analyzes, and targets its mistakes.

## 🚀 What Makes This Different?

| Feature | Traditional Models | TrialNet |
|---------|-------------------|----------|
| Error handling | Errors forgotten after weight update | **Error Memory Bank** stores and prioritizes mistakes |
| Learning signal | Average loss gradient | Gradient + **targeted mistake correction** |
| Self-awareness | None | **Mistake Pattern Analyzer** discovers failure patterns |
| Weight updates | Gradient-only | Gradient + **perturbation exploration** (try random changes, keep what works) |
| Adaptation | Fixed curriculum | **Dynamic focus** — spends more time on hard examples |

## 🏗️ Architecture

```
┌──────────────────────────────────────────────┐
│                  TrialNet                     │
│                                              │
│  Input → [Dense → ReLU → Dropout] × N → Softmax
│                    │                         │
│         ┌─────────┴──────────┐               │
│         │                    │               │
│    [Path A]            [Path B]              │
│  Traditional        Try & Learn              │
│  Backpropagation   ┌──────────┐              │
│                    │ Error     │              │
│                    │ Memory    │              │
│                    │ Bank      │              │
│                    ├──────────┤              │
│                    │ Mistake   │              │
│                    │ Analyzer  │              │
│                    ├──────────┤              │
│                    │ Perturb   │              │
│                    │ Explorer  │              │
│                    └──────────┘              │
│         └─────────┬──────────┘               │
│              Weight Merge                    │
└──────────────────────────────────────────────┘
```

## 📦 Installation

```bash
cd /Users/suraj/Documents/custom-ai-model
pip install -r requirements.txt
```

## 🎯 Quick Start

### Train the Hybrid Model (Recommended)
```bash
python train.py --mode hybrid --epochs 15
```

### Compare All Three Modes
```bash
python train.py --mode compare --epochs 15
```

### Train with Dashboard
```bash
# Terminal 1: Start the dashboard server
cd dashboard && python server.py

# Terminal 2: Train with live metrics
python train.py --mode hybrid --epochs 15 --dashboard
```

### Evaluate a Trained Model
```bash
python evaluate.py --model saved_models/hybrid
```

## 🔬 The Three Learning Modes

### 1. Traditional (Path A only)
Standard neural network — forward pass, loss, backpropagation, gradient descent.

### 2. Trial (Path B only)
Only the novel Try-and-Learn system — no gradient descent at all. Uses:
- Error Memory Bank for mistake tracking
- Perturbation Explorer for weight experimentation
- Prioritized replay for targeted retraining

### 3. Hybrid (Both — recommended)
Combines both paths. Traditional gradients provide the main learning signal, while Try-and-Learn provides targeted corrections for stubborn mistakes.

## 📊 The Novel Components

### Error Memory Bank
Stores every mistake with full context: what the model predicted, what was correct, how confident it was, and which internal neurons were active. Prioritizes high-confidence mistakes (the most dangerous kind — the model was very sure but very wrong).

### Mistake Pattern Analyzer
Discovers systemic issues like:
- "Always confuses digit 3 with digit 8"
- "Overconfident when wrong (avg 87% confidence on mistakes)"
- "Class 7 accounts for 35% of all errors"

### Perturbation Explorer
Instead of only following gradients, it experiments:
1. Save current weights
2. Add random noise
3. Test on mistake samples
4. Keep the change if it helps, revert if it doesn't

This can find solutions that pure gradient descent misses.

## 🌐 Dashboard

A real-time web dashboard at `http://localhost:5050` showing:
- Training/validation loss and accuracy curves
- Error Memory Bank size and correction rate
- Mistake patterns and confusion analysis
- Learning rate schedule
- Side-by-side comparison of all three modes

## 📁 Project Structure

```
custom-ai-model/
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── requirements.txt      # Dependencies
├── trialnet/             # Core library
│   ├── model.py          # Main TrialNet class
│   ├── utils.py          # Data loading utilities
│   ├── core/             # Neural network fundamentals
│   │   ├── tensor.py     # Custom tensor operations
│   │   ├── layers.py     # Dense, Dropout, BatchNorm
│   │   ├── activations.py # ReLU, Sigmoid, Softmax, etc.
│   │   └── losses.py     # CrossEntropy, MSE
│   └── learning/         # Learning engines
│       ├── traditional.py    # SGD, Adam optimizers
│       ├── error_memory.py   # Error Memory Bank (NOVEL)
│       ├── mistake_analyzer.py # Pattern analysis (NOVEL)
│       ├── perturbation.py   # Weight exploration (NOVEL)
│       └── trial_learner.py  # Orchestrator (NOVEL)
└── dashboard/            # Web visualization
    ├── server.py         # Flask API server
    ├── index.html        # Dashboard UI
    ├── style.css         # Styling
    └── app.js            # Chart logic
```

## 🔧 Built With

- **NumPy** — All math operations (no TensorFlow, no PyTorch)
- **Flask** — Dashboard API server
- **Chart.js** — Dashboard charts
- **Pure Python** — Everything built from scratch

## 👨‍💻 Author

Built by Suraj — TrialNet v0.1
