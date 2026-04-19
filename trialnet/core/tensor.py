"""
tensor.py — Custom Tensor Operations for TrialNet

Provides lightweight tensor utilities built on top of NumPy arrays.
Handles computation tracking, gradient accumulation, and weight snapshots
needed by the Try-and-Learn engine.
"""

import numpy as np
from typing import Optional, Tuple, List
import copy


class Tensor:
    """
    A lightweight wrapper around NumPy arrays that supports:
    - Gradient accumulation for backpropagation
    - Weight snapshot/restore for perturbation exploration
    - Computation history tracking
    """

    def __init__(self, data: np.ndarray, requires_grad: bool = False, name: str = ""):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self.name = name
        self._snapshot_stack: List[np.ndarray] = []

    @property
    def shape(self) -> Tuple:
        return self.data.shape

    @property
    def T(self) -> 'Tensor':
        """Transpose."""
        return Tensor(self.data.T, name=f"{self.name}.T")

    def zero_grad(self):
        """Reset gradient to zero."""
        self.grad = np.zeros_like(self.data)

    def accumulate_grad(self, grad: np.ndarray):
        """Accumulate gradient (supports multiple backward passes)."""
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad

    # ── Snapshot/Restore for Perturbation Explorer ──────────────────────

    def save_snapshot(self):
        """Save current weights to snapshot stack (for perturbation rollback)."""
        self._snapshot_stack.append(self.data.copy())

    def restore_snapshot(self):
        """Restore weights from last snapshot."""
        if self._snapshot_stack:
            self.data = self._snapshot_stack.pop()
        else:
            raise RuntimeError(f"No snapshot to restore for tensor '{self.name}'")

    def discard_snapshot(self):
        """Discard last snapshot (keep current weights)."""
        if self._snapshot_stack:
            self._snapshot_stack.pop()

    # ── Perturbation Helpers ───────────────────────────────────────────

    def perturb(self, scale: float = 0.01, mask: Optional[np.ndarray] = None):
        """
        Add random perturbation to weights.
        
        Args:
            scale: Standard deviation of perturbation noise
            mask: Optional mask to perturb only specific weights
        """
        noise = np.random.randn(*self.data.shape) * scale
        if mask is not None:
            noise *= mask
        self.data += noise

    def perturb_targeted(self, indices: np.ndarray, scale: float = 0.01):
        """Perturb only specific weight indices."""
        noise = np.random.randn(len(indices)) * scale
        flat = self.data.flatten()
        flat[indices] += noise
        self.data = flat.reshape(self.data.shape)

    # ── Statistics ─────────────────────────────────────────────────────

    def norm(self) -> float:
        """L2 norm of the tensor."""
        return float(np.linalg.norm(self.data))

    def mean(self) -> float:
        return float(np.mean(self.data))

    def std(self) -> float:
        return float(np.std(self.data))

    def max_abs(self) -> float:
        return float(np.max(np.abs(self.data)))

    # ── Arithmetic ─────────────────────────────────────────────────────

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        return Tensor(self.data + other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        return Tensor(self.data - other)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        return Tensor(self.data * other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        return Tensor(self.data / other)

    def __neg__(self):
        return Tensor(-self.data)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, name='{self.name}', grad={'yes' if self.grad is not None else 'no'})"

    def matmul(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication."""
        return Tensor(self.data @ other.data)

    def copy(self) -> 'Tensor':
        """Deep copy of tensor."""
        t = Tensor(self.data.copy(), requires_grad=self.requires_grad, name=self.name)
        if self.grad is not None:
            t.grad = self.grad.copy()
        return t


# ── Utility Functions ──────────────────────────────────────────────────

def xavier_init(fan_in: int, fan_out: int) -> Tensor:
    """
    Xavier/Glorot initialization for weights.
    Helps prevent vanishing/exploding gradients in deep networks.
    """
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    data = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
    return Tensor(data, requires_grad=True, name=f"weight_{fan_in}x{fan_out}")


def he_init(fan_in: int, fan_out: int) -> Tensor:
    """
    He initialization — better for ReLU activations.
    """
    std = np.sqrt(2.0 / fan_in)
    data = np.random.randn(fan_in, fan_out) * std
    return Tensor(data, requires_grad=True, name=f"weight_{fan_in}x{fan_out}")


def zeros(shape: Tuple, name: str = "") -> Tensor:
    """Create a zero tensor."""
    return Tensor(np.zeros(shape), requires_grad=True, name=name)


def ones(shape: Tuple, name: str = "") -> Tensor:
    """Create a ones tensor."""
    return Tensor(np.ones(shape), requires_grad=True, name=name)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a.flatten(), b.flatten()) / (norm_a * norm_b))
