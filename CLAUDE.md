# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

```bash
# Train (default: hybrid mode, 15 epochs)
python train.py
python train.py --mode traditional
python train.py --mode trial
python train.py --mode compare        # train all 3 and compare
python train.py --mode hybrid --epochs 20 --batch-size 128 --dashboard

# Evaluate a saved model
python evaluate.py                                    # hybrid (default)
python evaluate.py --model saved_models/traditional

# Dashboard (after training)
cd dashboard && python server.py      # http://localhost:5050

# Quick demo
python demo.py
```

## Architecture

Pure NumPy neural network with a novel **Dual Learning Engine** — combines standard backpropagation with a "Try-and-Learn" system that corrects mistakes in real-time during training.

### Core layers (`trialnet/core/`)
- `tensor.py` — `Tensor` wrapper: ndarray + grad + zero_grad; `xavier_init`, `he_init`
- `layers.py` — `DenseLayer`, `DropoutLayer`, `BatchNormLayer`; all share `Layer` base with `forward/backward/get_params/train/eval`
- `activations.py` — ReLU, Softmax, Sigmoid, Tanh
- `losses.py` — CrossEntropy, MSE

### Novel learning system (`trialnet/learning/`)
- `error_memory.py` — `ErrorMemoryBank`: priority queue caching high-confidence mistakes; deduplicates by cosine similarity
- `mistake_analyzer.py` — `MistakePatternAnalyzer`: scans memory bank for recurring confusion pairs (e.g. digit 9 vs 4), returns `MistakeReport`
- `perturbation.py` — `PerturbationExplorer`: tests microscopic weight nudges on specific neurons without touching baseline; keeps changes only if loss improves
- `trial_learner.py` — `TrialLearner` orchestrator: runs the 5-step loop (CAPTURE → ANALYZE → EXPLORE → REPLAY → MERGE) every N batches
- `traditional.py` — SGD / Adam optimizer + `LearningRateScheduler` (warmup, cosine, step)

### Model entry point
`trialnet/model.py` — `TrialNet` class. Three `learning_mode` values:
- `'traditional'` — backprop only
- `'trial'` — Try-and-Learn only (no gradient descent)
- `'hybrid'` — both combined (recommended); trial weight = 0.05, trial lr = main lr × 0.05

Trial learning starts at epoch 4 to let the model build a baseline first. Trial backward uses direct SGD with gradient clipping (not Adam) to avoid corrupting optimizer momentum.

### Saved model format (`saved_models/<mode>/`)
Three files: `weights.npz`, `config.json`, `layer_configs`, `history.json`.

### Dashboard (`dashboard/`)
Flask server on port 5050. Reads JSON from `dashboard/data/`. Supports SSE stream at `/api/stream/<mode>` for live updates during training (requires `--dashboard` flag).

## Data

MNIST gz files in `data/mnist/`. Loaded via `trialnet/utils.py::load_mnist()` which returns `(train, val, test)` splits as `(X, y)` tuples with X normalized to `[0, 1]`.
