"""
activations.py — Activation Functions for TrialNet

All activation functions implement both forward and backward passes,
built entirely from scratch using NumPy.
"""

import numpy as np


class Activation:
    """Base class for all activation functions."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    @property
    def name(self) -> str:
        return self.__class__.__name__


class ReLU(Activation):
    """
    Rectified Linear Unit: f(x) = max(0, x)
    Simple, fast, and effective. The workhorse of deep learning.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x
        return np.maximum(0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (self._cache > 0).astype(np.float64)


class LeakyReLU(Activation):
    """
    Leaky ReLU: f(x) = x if x > 0, else alpha * x
    Prevents dead neurons by allowing small gradients when x < 0.
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * np.where(self._cache > 0, 1.0, self.alpha)


class Sigmoid(Activation):
    """
    Sigmoid: f(x) = 1 / (1 + exp(-x))
    Squashes output to (0, 1). Good for binary classification output.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Clip to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        self._output = 1.0 / (1.0 + np.exp(-x_clipped))
        return self._output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self._output * (1.0 - self._output)


class Tanh(Activation):
    """
    Tanh: f(x) = tanh(x)
    Squashes output to (-1, 1). Zero-centered, which can help training.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._output = np.tanh(x)
        return self._output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (1.0 - self._output ** 2)


class Softmax(Activation):
    """
    Softmax: f(x_i) = exp(x_i) / sum(exp(x_j))
    Converts logits to probabilities. Used for multi-class classification output.
    
    Note: The backward pass here computes the Jacobian-vector product.
    In practice, we often combine Softmax + CrossEntropy for a simpler gradient.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability (prevents exp overflow)
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        self._output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self._output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Jacobian of softmax is: diag(s) - s * s^T
        For batch: compute per-sample.
        """
        batch_size = grad_output.shape[0]
        grad_input = np.zeros_like(grad_output)

        for i in range(batch_size):
            s = self._output[i].reshape(-1, 1)
            jacobian = np.diagflat(s) - s @ s.T
            grad_input[i] = jacobian @ grad_output[i]

        return grad_input


class Linear(Activation):
    """
    Linear (identity) activation: f(x) = x
    Used when no activation is needed (e.g., regression output).
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output


# ── Factory ────────────────────────────────────────────────────────────

ACTIVATION_MAP = {
    'relu': ReLU,
    'leaky_relu': LeakyReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'softmax': Softmax,
    'linear': Linear,
    'none': Linear,
}


def get_activation(name: str) -> Activation:
    """Get activation function by name."""
    name = name.lower()
    if name not in ACTIVATION_MAP:
        raise ValueError(f"Unknown activation: {name}. Choose from: {list(ACTIVATION_MAP.keys())}")
    return ACTIVATION_MAP[name]()
