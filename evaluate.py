#!/usr/bin/env python3
"""
evaluate.py — Evaluation and Mistake Analysis for TrialNet

Loads a trained model and generates a comprehensive evaluation report
including mistake analysis, confusion matrix, and performance breakdown.

Usage:
    python evaluate.py                           # Evaluate hybrid model
    python evaluate.py --model saved_models/traditional  # Evaluate specific model
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trialnet.model import TrialNet
from trialnet.utils import load_mnist, classification_report, confusion_matrix


def evaluate_model(model_path: str):
    """Run comprehensive evaluation on a saved model."""
    # Load model
    model = TrialNet.load(model_path)

    # Load test data
    _, _, (X_test, y_test) = load_mnist()

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print(f"\n{'='*60}")
    print(f"  Evaluation Report — {model.name}")
    print(f"{'='*60}")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

    # Detailed classification report
    predictions = model.predict(X_test)
    report = classification_report(predictions, y_test)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(predictions, y_test)
    print("\n  Confusion Matrix:")
    print("     " + "  ".join(f"{i:>4}" for i in range(10)))
    print("     " + "─" * 50)
    for i in range(10):
        row = "  ".join(f"{cm[i][j]:>4}" for j in range(10))
        print(f"  {i}: {row}")

    # Mistake analysis
    pred_classes = np.argmax(predictions, axis=1)
    mistake_mask = pred_classes != y_test.astype(int)
    n_mistakes = np.sum(mistake_mask)

    print(f"\n  Total Mistakes: {n_mistakes} / {len(y_test)} ({n_mistakes/len(y_test)*100:.1f}%)")

    # Per-class mistake breakdown
    print("\n  Mistakes by Class:")
    for cls in range(10):
        cls_mask = y_test.astype(int) == cls
        cls_mistakes = np.sum(mistake_mask & cls_mask)
        cls_total = np.sum(cls_mask)
        print(f"    Class {cls}: {cls_mistakes:>4} / {cls_total:>4} ({cls_mistakes/max(cls_total,1)*100:.1f}%)")

    # Top confusion pairs
    print("\n  Top Confusion Pairs:")
    confusions = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i][j] > 0:
                confusions.append((i, j, cm[i][j]))
    confusions.sort(key=lambda x: x[2], reverse=True)
    for correct, predicted, count in confusions[:10]:
        print(f"    {correct} → {predicted}: {count} mistakes")

    # Save report
    report_data = {
        'model_name': model.name,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'total_mistakes': int(n_mistakes),
        'confusion_matrix': cm.tolist(),
        'top_confusions': [(int(a), int(b), int(c)) for a, b, c in confusions[:20]],
    }

    report_path = os.path.join(model_path, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    print(f"\n  📄 Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate TrialNet')
    parser.add_argument('--model', type=str, default='./saved_models/hybrid',
                       help='Path to saved model directory')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Model not found at {args.model}")
        print("   Run train.py first to create a model.")
        return

    evaluate_model(args.model)


if __name__ == '__main__':
    main()
