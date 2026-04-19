#!/usr/bin/env python3
"""
train.py — Training Script for TrialNet

Trains three variants of the model for comparison:
1. Traditional only (standard backpropagation)
2. Trial-only (Try-and-Learn without backpropagation)
3. Hybrid (both combined — the recommended approach)

Usage:
    python train.py                     # Train hybrid (default)
    python train.py --mode traditional  # Traditional only
    python train.py --mode trial        # Trial-only
    python train.py --mode compare      # Train all 3 and compare
    python train.py --dashboard         # Enable real-time dashboard
"""

import argparse
import json
import os
import sys
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trialnet.model import TrialNet
from trialnet.utils import load_mnist, classification_report, confusion_matrix


def create_model(mode: str, num_classes: int = 10, epochs: int = 20) -> TrialNet:
    """Create and compile a TrialNet model."""
    model = TrialNet(learning_mode=mode, name=f"TrialNet-{mode}")

    # Architecture: 784 → 256 → 128 → 64 → 10
    model.add_dense(784, 256, activation='relu')
    model.add_dropout(0.3)
    model.add_dense(256, 128, activation='relu')
    model.add_dropout(0.2)
    model.add_dense(128, 64, activation='relu')
    model.add_dense(64, num_classes, activation='softmax')

    model.compile(
        optimizer='adam',
        loss='cross_entropy',
        learning_rate=0.001,
        lr_schedule='warmup',
        num_classes=num_classes,
        total_epochs=epochs,
    )

    return model


def save_metrics(history: dict, filepath: str):
    """Save training metrics to JSON for the dashboard."""

    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        return obj

    serializable = convert(history)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"📊 Metrics saved to {filepath}")


def train_single(mode: str, epochs: int = 15, batch_size: int = 64, dashboard: bool = False):
    """Train a single model."""
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist()

    # Create model
    model = create_model(mode, epochs=epochs)

    # Dashboard callback
    metrics_callback = None
    if dashboard:
        metrics_file = f'./dashboard/data/live_metrics_{mode}.json'
        os.makedirs('./dashboard/data', exist_ok=True)

        def callback(metrics):
            save_metrics(metrics, metrics_file)

        metrics_callback = callback

    # Train
    history = model.train(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1,
        metrics_callback=metrics_callback,
    )

    # Evaluate on test set
    print("\n" + "=" * 60)
    print(f"  📋 Final Evaluation — {model.name}")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

    # Detailed report
    predictions = model.predict(X_test)
    report = classification_report(predictions, y_test)
    print(report)

    # Mistake report (if using trial/hybrid)
    if model.trial_learner:
        report = model.get_mistake_report()
        if report:
            print("\n🔍 Mistake Analysis Report:")
            print(f"   Overall Severity: {report.overall_severity:.2f}")
            print(f"   Patterns Found: {len(report.patterns)}")
            if report.top_confusions:
                print(f"   Top Confusions:")
                for correct, predicted, count in report.top_confusions[:5]:
                    print(f"     {correct} → {predicted}: {count} times")

    # Save model
    save_dir = f'./saved_models/{mode}'
    model.save(save_dir)

    # Save full training metrics
    save_metrics(history, f'./dashboard/data/history_{mode}.json')

    return model, history, test_acc


def train_compare(epochs: int = 15, batch_size: int = 64):
    """Train all three modes and compare results."""
    print("\n" + "🔬" * 30)
    print("  COMPARATIVE TRAINING: Traditional vs Trial vs Hybrid")
    print("🔬" * 30 + "\n")

    # Load data once
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist()

    results = {}

    for mode in ['traditional', 'trial', 'hybrid']:
        print(f"\n{'─' * 60}")
        print(f"  Training: {mode.upper()}")
        print(f"{'─' * 60}")

        model = create_model(mode, epochs=epochs)

        start_time = time.time()
        history = model.train(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1,
        )
        train_time = time.time() - start_time

        test_loss, test_acc = model.evaluate(X_test, y_test)

        results[mode] = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'train_time': train_time,
            'final_train_acc': history['train_accuracy'][-1],
            'final_val_acc': history['val_accuracy'][-1],
            'history': history,
        }

        model.save(f'./saved_models/{mode}')
        save_metrics(history, f'./dashboard/data/history_{mode}.json')

    # ── Comparison Summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  📊 COMPARISON RESULTS")
    print("=" * 70)
    print(f"  {'Mode':<15} {'Test Acc':>10} {'Val Acc':>10} {'Train Time':>12}")
    print(f"  {'─'*15} {'─'*10} {'─'*10} {'─'*12}")

    for mode, res in results.items():
        print(f"  {mode:<15} {res['test_accuracy']:>10.4f} "
              f"{res['final_val_acc']:>10.4f} {res['train_time']:>10.1f}s")

    # Find winner
    best_mode = max(results.keys(), key=lambda m: results[m]['test_accuracy'])
    print(f"\n  🏆 Winner: {best_mode.upper()} with {results[best_mode]['test_accuracy']:.4f} test accuracy")

    # Save comparison
    comparison = {
        mode: {
            'test_accuracy': float(res['test_accuracy']),
            'test_loss': float(res['test_loss']),
            'train_time': float(res['train_time']),
            'final_train_acc': float(res['final_train_acc']),
            'final_val_acc': float(res['final_val_acc']),
        }
        for mode, res in results.items()
    }
    save_metrics(comparison, './dashboard/data/comparison.json')

    return results


def main():
    parser = argparse.ArgumentParser(description='Train TrialNet')
    parser.add_argument('--mode', type=str, default='hybrid',
                       choices=['traditional', 'trial', 'hybrid', 'compare'],
                       help='Training mode')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--dashboard', action='store_true',
                       help='Enable real-time dashboard metrics')

    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║    ████████╗██████╗ ██╗ █████╗ ██╗     ███╗   ██╗███████╗████████╗ ║
║    ╚══██╔══╝██╔══██╗██║██╔══██╗██║     ████╗  ██║██╔════╝╚══██╔══╝ ║
║       ██║   ██████╔╝██║███████║██║     ██╔██╗ ██║█████╗     ██║    ║
║       ██║   ██╔══██╗██║██╔══██║██║     ██║╚██╗██║██╔══╝     ██║    ║
║       ██║   ██║  ██║██║██║  ██║███████╗██║ ╚████║███████╗   ██║    ║
║       ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚══════╝   ╚═╝    ║
║                                                          ║
║    A Neural Network That Learns From Its Mistakes        ║
║    Built from scratch with NumPy                         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

    if args.mode == 'compare':
        train_compare(epochs=args.epochs, batch_size=args.batch_size)
    else:
        train_single(args.mode, epochs=args.epochs, batch_size=args.batch_size,
                     dashboard=args.dashboard)


if __name__ == '__main__':
    main()
