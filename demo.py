#!/usr/bin/env python3
"""
demo.py — Interactive Demo for TrialNet

Test your trained model interactively:
1. Predict random test samples and see confidence
2. Draw your own digits and test
3. See what the model gets wrong and why
4. Explore mistake patterns
5. Challenge the model with hard examples

Usage:
    python3 demo.py
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trialnet.model import TrialNet
from trialnet.utils import load_mnist, confusion_matrix


def print_digit(pixels, label=None, prediction=None):
    """Print a 28x28 digit as ASCII art in the terminal."""
    img = pixels.reshape(28, 28)
    chars = " .:-=+*#%@"

    if label is not None and prediction is not None:
        status = "✅" if label == prediction else "❌"
        print(f"\n  {status} True: {label} | Predicted: {prediction}")
    elif label is not None:
        print(f"\n  Label: {label}")

    print("  ┌" + "─" * 28 + "┐")
    for row in img:
        line = ""
        for pixel in row:
            idx = min(int(pixel * (len(chars) - 1)), len(chars) - 1)
            line += chars[idx]
        print(f"  │{line}│")
    print("  └" + "─" * 28 + "┘")


def print_confidence(probs):
    """Print a confidence bar chart for all 10 digits."""
    print("\n  Confidence for each digit:")
    print("  " + "─" * 45)
    for i in range(10):
        conf = probs[i]
        bar_len = int(conf * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        marker = " ◄── PREDICTION" if i == np.argmax(probs) else ""
        print(f"  {i}: [{bar}] {conf*100:5.1f}%{marker}")
    print()


def predict_random_samples(model, X_test, y_test, n=5):
    """Predict random samples from the test set."""
    print(f"\n{'='*55}")
    print(f"  🎲 Random Test Samples (out of {len(y_test)} test images)")
    print(f"{'='*55}")

    indices = np.random.choice(len(y_test), size=n, replace=False)
    correct = 0

    for idx in indices:
        x = X_test[idx:idx+1]
        true_label = int(y_test[idx])

        probs = model.predict(x)[0]
        predicted = int(np.argmax(probs))

        print_digit(X_test[idx], label=true_label, prediction=predicted)
        print_confidence(probs)

        if predicted == true_label:
            correct += 1

    print(f"  Score: {correct}/{n} correct ({correct/n*100:.0f}%)")


def predict_specific_digit(model, X_test, y_test, digit):
    """Find and predict a specific digit from the test set."""
    mask = y_test.astype(int) == digit
    indices = np.where(mask)[0]

    if len(indices) == 0:
        print(f"  No digit {digit} found in test set.")
        return

    idx = np.random.choice(indices)
    x = X_test[idx:idx+1]
    probs = model.predict(x)[0]
    predicted = int(np.argmax(probs))

    print(f"\n  Showing digit: {digit}")
    print_digit(X_test[idx], label=digit, prediction=predicted)
    print_confidence(probs)


def find_mistakes(model, X_test, y_test, n=5):
    """Find and display samples the model gets WRONG."""
    print(f"\n{'='*55}")
    print(f"  🔍 Model Mistakes — What It Gets Wrong")
    print(f"{'='*55}")

    predictions = model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = y_test.astype(int)

    mistake_mask = pred_classes != true_classes
    mistake_indices = np.where(mistake_mask)[0]

    total_mistakes = len(mistake_indices)
    print(f"\n  Total mistakes: {total_mistakes} out of {len(y_test)} ({total_mistakes/len(y_test)*100:.1f}%)")

    if total_mistakes == 0:
        print("  🎉 Perfect score! No mistakes!")
        return

    # Show a few mistakes
    selected = np.random.choice(mistake_indices, size=min(n, total_mistakes), replace=False)

    for idx in selected:
        true_label = int(true_classes[idx])
        pred_label = int(pred_classes[idx])
        probs = predictions[idx]

        print_digit(X_test[idx], label=true_label, prediction=pred_label)
        print_confidence(probs)
        conf = probs[pred_label] * 100
        print(f"  💭 Model was {conf:.1f}% confident it was a {pred_label}, but it's actually a {true_label}")
        print()


def show_confusion_analysis(model, X_test, y_test):
    """Show what digit pairs the model confuses most."""
    print(f"\n{'='*55}")
    print(f"  📊 Confusion Analysis — What Gets Mixed Up")
    print(f"{'='*55}")

    predictions = model.predict(X_test)
    cm = confusion_matrix(predictions, y_test)

    # Find top confusion pairs
    confusions = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i][j] > 0:
                confusions.append((i, j, cm[i][j]))
    confusions.sort(key=lambda x: x[2], reverse=True)

    print("\n  Top 10 Confusion Pairs:")
    print("  " + "─" * 45)
    for true_c, pred_c, count in confusions[:10]:
        bar = "█" * min(count, 30)
        print(f"  {true_c} → {pred_c}: {bar} ({count} times)")

    print("\n  Confusion Matrix:")
    print("       " + "  ".join(f"{i:>3}" for i in range(10)))
    print("      " + "─" * 43)
    for i in range(10):
        row = "  ".join(f"{cm[i][j]:>3}" for j in range(10))
        print(f"  {i}:  {row}")


def challenge_hard_examples(model, X_test, y_test):
    """Show the model's WORST predictions (highest confidence when wrong)."""
    print(f"\n{'='*55}")
    print(f"  💀 Hardest Mistakes — Most Confident AND Wrong")
    print(f"{'='*55}")

    predictions = model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = y_test.astype(int)

    mistake_mask = pred_classes != true_classes
    mistake_indices = np.where(mistake_mask)[0]

    if len(mistake_indices) == 0:
        print("  No mistakes found!")
        return

    # Get confidence of wrong predictions
    confidences = [predictions[idx][pred_classes[idx]] for idx in mistake_indices]
    sorted_idx = np.argsort(confidences)[::-1]  # Most confident first

    print("\n  These are the model's WORST mistakes — it was very confident, but WRONG:\n")

    for i in range(min(5, len(sorted_idx))):
        idx = mistake_indices[sorted_idx[i]]
        true_label = int(true_classes[idx])
        pred_label = int(pred_classes[idx])
        conf = predictions[idx][pred_label] * 100

        print_digit(X_test[idx], label=true_label, prediction=pred_label)
        print(f"  ⚠️  Confidence: {conf:.1f}% — thought it was {pred_label}, actually {true_label}")
        print()


def draw_digit_mode(model):
    """Let the user create a digit using a simple text grid."""
    print(f"\n{'='*55}")
    print(f"  ✏️  Draw Your Own Digit")
    print(f"{'='*55}")
    print("""
  Enter your digit as a 28x28 grid, or use a simplified mode:

  SIMPLIFIED MODE (7x7 grid, we'll upscale to 28x28):
  Enter 7 rows of 7 characters each.
  Use '#' for black (ink) and '.' for white (paper).

  Example (drawing a "1"):
    ..#....
    .##....
    ..#....
    ..#....
    ..#....
    ..#....
    .###...
  """)

    rows = []
    print("  Enter 7 rows (7 chars each, '#' = ink, '.' = paper):")
    print("  Type 'done' when finished, or 'cancel' to go back.\n")

    for i in range(7):
        while True:
            row = input(f"  Row {i+1}: ").strip()
            if row.lower() == 'cancel':
                return
            if row.lower() == 'done':
                break
            if len(row) >= 7:
                rows.append(row[:7])
                break
            else:
                print(f"    Need 7 characters, got {len(row)}. Try again.")

        if row.lower() == 'done':
            break

    if len(rows) < 7:
        # Pad with empty rows
        while len(rows) < 7:
            rows.append(".......")

    # Convert to 7x7 grid
    small_grid = np.zeros((7, 7))
    for i, row in enumerate(rows):
        for j, char in enumerate(row):
            if char == '#' or char == '*' or char == 'X':
                small_grid[i][j] = 1.0
            elif char == '+':
                small_grid[i][j] = 0.5

    # Upscale to 28x28 using nearest neighbor
    big_grid = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            si, sj = i // 4, j // 4
            big_grid[i][j] = small_grid[min(si, 6)][min(sj, 6)]

    # Flatten for prediction
    x = big_grid.reshape(1, 784)

    # Show what we're predicting
    print("\n  Your drawn digit:")
    print_digit(x[0])

    # Predict
    probs = model.predict(x)[0]
    predicted = int(np.argmax(probs))
    conf = probs[predicted] * 100

    print(f"\n  🤖 Model predicts: {predicted} (confidence: {conf:.1f}%)")
    print_confidence(probs)


def model_capabilities(model):
    """Explain what this model can and cannot do."""
    print(f"\n{'='*55}")
    print(f"  🧠 What Can TrialNet Do?")
    print(f"{'='*55}")

    n_params = sum(p.data.size for layer in model.layers for p in layer.get_params())

    print(f"""
  Model: {model.name}
  Mode:  {model.learning_mode}
  Parameters: {n_params:,}

  ✅ WHAT IT CAN DO:
  ─────────────────
  • Recognize handwritten digits (0-9)
  • Show confidence scores for each prediction
  • Analyze its own mistakes and find patterns
  • Learn from its mistakes using the Error Memory Bank
  • Experiment with weight changes (Perturbation Explorer)
  • Identify what digit pairs it confuses (e.g., 3↔8, 4↔9)

  ❌ WHAT IT CANNOT DO (yet):
  ──────────────────────────
  • Understand text or language (it's a vision model)
  • Recognize objects beyond digits
  • Generate images
  • Answer questions in natural language
  • Process color images (MNIST is grayscale 28x28)

  🔧 HOW TO EXTEND:
  ──────────────────
  • Replace MNIST with Fashion-MNIST for clothing classification
  • Add convolutional layers for better image recognition
  • Train on CIFAR-10 for object recognition (cars, planes, etc.)
  • Add more layers for more complex tasks
  • The Try-and-Learn engine works with ANY classification task!
  """)


def interactive_menu(model, X_test, y_test):
    """Main interactive menu."""
    while True:
        print(f"\n{'='*55}")
        print(f"  🧠 TrialNet Interactive Demo")
        print(f"{'='*55}")
        print("""
  Choose an option:

    [1] 🎲 Predict random test samples
    [2] 🔢 Predict a specific digit (0-9)
    [3] 🔍 Show model mistakes
    [4] 💀 Show hardest mistakes (most confident + wrong)
    [5] 📊 Confusion analysis
    [6] ✏️  Draw your own digit
    [7] 🧠 What can this model do?
    [8] 📈 Model statistics
    [0] 🚪 Exit
        """)

        choice = input("  Enter choice (0-8): ").strip()

        if choice == '1':
            n = input("  How many samples? (default: 5): ").strip()
            n = int(n) if n.isdigit() else 5
            predict_random_samples(model, X_test, y_test, n=n)

        elif choice == '2':
            d = input("  Which digit (0-9)? ").strip()
            if d.isdigit() and 0 <= int(d) <= 9:
                predict_specific_digit(model, X_test, y_test, int(d))
            else:
                print("  Please enter a digit 0-9.")

        elif choice == '3':
            find_mistakes(model, X_test, y_test)

        elif choice == '4':
            challenge_hard_examples(model, X_test, y_test)

        elif choice == '5':
            show_confusion_analysis(model, X_test, y_test)

        elif choice == '6':
            draw_digit_mode(model)

        elif choice == '7':
            model_capabilities(model)

        elif choice == '8':
            print(f"\n  {model}")
            test_loss, test_acc = model.evaluate(X_test, y_test)
            print(f"  Test Accuracy: {test_acc*100:.1f}%")
            print(f"  Test Loss: {test_loss:.4f}")
            if model.trial_learner:
                stats = model.trial_learner.get_comprehensive_stats()
                print(f"  Mistake Memory: {stats['memory']['total_stored']} stored")
                print(f"  Correction Rate: {stats['memory'].get('correction_rate', 0)*100:.1f}%")

        elif choice == '0':
            print("\n  👋 Goodbye!")
            break

        else:
            print("  ❓ Invalid choice. Please enter 0-8.")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║          TrialNet — Interactive Demo                     ║
║          Test your AI model!                             ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)

    # Load model
    model_path = './saved_models/hybrid'
    if not os.path.exists(model_path):
        print("  ❌ No trained model found!")
        print("  Run this first: python3 train.py --mode hybrid --epochs 15")
        return

    model = TrialNet.load(model_path)
    model.compile(optimizer='adam', loss='cross_entropy', learning_rate=0.001, num_classes=10)

    # Load test data
    _, _, (X_test, y_test) = load_mnist()

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\n  ✅ Model loaded! Test accuracy: {test_acc*100:.1f}%")

    # Start interactive menu
    interactive_menu(model, X_test, y_test)


if __name__ == '__main__':
    main()
