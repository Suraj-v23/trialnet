"""
losses.py — Loss Functions for TrialNet

Each loss function computes both the loss value (forward) and gradient (backward).
The gradient tells us how to adjust the model output to reduce the loss.
"""

import numpy as np
from typing import Tuple


class Loss:
    """Base class for loss functions."""

    def forward(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """Compute loss value."""
        raise NotImplementedError

    def backward(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Compute gradient of loss with respect to predicted."""
        raise NotImplementedError

    def __call__(self, predicted: np.ndarray, target: np.ndarray) -> float:
        return self.forward(predicted, target)

    @property
    def name(self) -> str:
        return self.__class__.__name__


class CrossEntropyLoss(Loss):
    """
    Cross-Entropy Loss for multi-class classification.

    Expects:
    - predicted: (batch_size, num_classes) — raw logits or softmax probabilities
    - target: (batch_size,) — integer class labels, OR
              (batch_size, num_classes) — one-hot encoded

    This implementation combines softmax + cross-entropy for numerical stability.
    """

    def __init__(self, from_logits: bool = True):
        self.from_logits = from_logits
        self._probabilities = None

    def _to_one_hot(self, target: np.ndarray, num_classes: int) -> np.ndarray:
        """Convert integer labels to one-hot encoding."""
        if target.ndim == 1:
            one_hot = np.zeros((target.shape[0], num_classes))
            one_hot[np.arange(target.shape[0]), target.astype(int)] = 1.0
            return one_hot
        return target

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """
        Compute cross-entropy loss.

        If from_logits=True, applies softmax first (more numerically stable).
        """
        batch_size = predicted.shape[0]
        num_classes = predicted.shape[1]

        # Convert target to one-hot if needed
        target_one_hot = self._to_one_hot(target, num_classes)

        # Apply softmax if input is raw logits
        if self.from_logits:
            self._probabilities = self._softmax(predicted)
        else:
            self._probabilities = predicted

        # Clip to prevent log(0)
        probs_clipped = np.clip(self._probabilities, 1e-12, 1.0 - 1e-12)

        # Cross-entropy: -sum(target * log(predicted))
        loss = -np.sum(target_one_hot * np.log(probs_clipped)) / batch_size

        # Store for backward
        self._target_one_hot = target_one_hot

        return float(loss)

    def backward(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Gradient of softmax + cross-entropy loss.

        The beautiful simplification: dL/dz = softmax(z) - target_one_hot
        This is much simpler and more stable than computing each part separately.
        """
        batch_size = predicted.shape[0]
        num_classes = predicted.shape[1]

        target_one_hot = self._to_one_hot(target, num_classes)

        if self._probabilities is None:
            self._probabilities = self._softmax(predicted)

        # The elegant gradient: predictions - targets
        grad = (self._probabilities - target_one_hot)

        return grad

    def get_per_sample_loss(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute loss for each sample individually (used by Error Memory Bank).
        Returns: (batch_size,) array of per-sample losses.
        """
        num_classes = predicted.shape[1]
        target_one_hot = self._to_one_hot(target, num_classes)
        probs = self._softmax(predicted) if self.from_logits else predicted
        probs_clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)

        # Per-sample cross-entropy
        per_sample = -np.sum(target_one_hot * np.log(probs_clipped), axis=1)
        return per_sample


class MSELoss(Loss):
    """
    Mean Squared Error Loss for regression.

    L = mean((predicted - target)^2)
    """

    def forward(self, predicted: np.ndarray, target: np.ndarray) -> float:
        self._predicted = predicted
        self._target = target
        diff = predicted - target
        return float(np.mean(diff ** 2))

    def backward(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        batch_size = predicted.shape[0]
        return 2.0 * (predicted - target) / (batch_size * predicted.shape[1])

    def get_per_sample_loss(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Per-sample MSE for Error Memory Bank."""
        diff = predicted - target
        return np.mean(diff ** 2, axis=1)


# ── Factory ────────────────────────────────────────────────────────────

LOSS_MAP = {
    'cross_entropy': CrossEntropyLoss,
    'crossentropy': CrossEntropyLoss,
    'mse': MSELoss,
    'mean_squared_error': MSELoss,
}


def get_loss(name: str, **kwargs) -> Loss:
    """Get loss function by name."""
    name = name.lower()
    if name not in LOSS_MAP:
        raise ValueError(f"Unknown loss: {name}. Choose from: {list(LOSS_MAP.keys())}")
    return LOSS_MAP[name](**kwargs)
