"""
utils.py — Utility Functions for TrialNet

Data loading, preprocessing, and helper functions.
Includes automatic MNIST dataset download and preprocessing.
"""

import numpy as np
import os
import gzip
import struct
from typing import Tuple, Optional
import urllib.request


# ── MNIST Dataset ──────────────────────────────────────────────────────

MNIST_URLS = {
    'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
}

MNIST_MIRROR_URLS = {
    'train_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
}


def download_mnist(data_dir: str = './data/mnist') -> dict:
    """
    Download the MNIST dataset if not already present.

    Returns dict with paths to downloaded files.
    """
    os.makedirs(data_dir, exist_ok=True)
    paths = {}

    for key, url in MNIST_URLS.items():
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        paths[key] = filepath

        if os.path.exists(filepath):
            continue

        print(f"  Downloading {filename}...", end=' ', flush=True)
        try:
            urllib.request.urlretrieve(url, filepath)
            print("✓")
        except Exception:
            # Try mirror
            mirror_url = MNIST_MIRROR_URLS[key]
            try:
                urllib.request.urlretrieve(mirror_url, filepath)
                print("✓ (mirror)")
            except Exception as e:
                print(f"✗ Error: {e}")
                raise

    return paths


def load_mnist_images(filepath: str) -> np.ndarray:
    """Load MNIST images from IDX file."""
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows * cols)
    return data.astype(np.float64)


def load_mnist_labels(filepath: str) -> np.ndarray:
    """Load MNIST labels from IDX file."""
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_mnist(
    data_dir: str = './data/mnist',
    normalize: bool = True,
    flatten: bool = True,
    validation_split: float = 0.1,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load and preprocess the MNIST dataset.

    Returns:
        ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    print("📦 Loading MNIST dataset...")
    paths = download_mnist(data_dir)

    # Load raw data
    X_train_full = load_mnist_images(paths['train_images'])
    y_train_full = load_mnist_labels(paths['train_labels'])
    X_test = load_mnist_images(paths['test_images'])
    y_test = load_mnist_labels(paths['test_labels'])

    # Normalize to [0, 1]
    if normalize:
        X_train_full = X_train_full / 255.0
        X_test = X_test / 255.0

    # Split training into train + validation
    n_total = X_train_full.shape[0]
    n_val = int(n_total * validation_split)
    indices = np.random.permutation(n_total)

    X_val = X_train_full[indices[:n_val]]
    y_val = y_train_full[indices[:n_val]]
    X_train = X_train_full[indices[n_val:]]
    y_train = y_train_full[indices[n_val:]]

    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {len(np.unique(y_train))}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ── Data Preprocessing ────────────────────────────────────────────────

def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels to one-hot encoding."""
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels.astype(int)] = 1.0
    return one_hot


def shuffle_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle X and y together."""
    indices = np.random.permutation(X.shape[0])
    return X[indices], y[indices]


def create_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True
) -> list:
    """Create mini-batches from data."""
    if shuffle:
        X, y = shuffle_data(X, y)

    n_samples = X.shape[0]
    batches = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batches.append((X[start:end], y[start:end]))

    return batches


# ── Metrics ────────────────────────────────────────────────────────────

def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute accuracy."""
    if predictions.ndim > 1:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = predictions

    if targets.ndim > 1:
        true_classes = np.argmax(targets, axis=1)
    else:
        true_classes = targets.astype(int)

    return float(np.mean(pred_classes == true_classes))


def confusion_matrix(predictions: np.ndarray, targets: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Compute confusion matrix."""
    if predictions.ndim > 1:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = predictions.astype(int)

    if targets.ndim > 1:
        true_classes = np.argmax(targets, axis=1)
    else:
        true_classes = targets.astype(int)

    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(true_classes, pred_classes):
        matrix[t][p] += 1

    return matrix


def classification_report(predictions: np.ndarray, targets: np.ndarray, num_classes: int = 10) -> str:
    """Generate a text classification report."""
    cm = confusion_matrix(predictions, targets, num_classes)

    report = f"\n{'Class':>8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n"
    report += "─" * 50 + "\n"

    total_correct = 0
    total_samples = 0

    for cls in range(num_classes):
        tp = cm[cls][cls]
        fp = sum(cm[i][cls] for i in range(num_classes)) - tp
        fn = sum(cm[cls]) - tp
        support = sum(cm[cls])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        report += f"{cls:>8} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}\n"

        total_correct += tp
        total_samples += support

    report += "─" * 50 + "\n"
    report += f"{'Accuracy':>8} {'':>10} {'':>10} {total_correct/max(total_samples,1):>10.4f} {total_samples:>10}\n"

    return report
