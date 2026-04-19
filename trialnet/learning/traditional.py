"""
traditional.py — Traditional Learning Engine for TrialNet

Implements standard optimizers (SGD with momentum, Adam) from scratch.
These perform weight updates based on gradients computed via backpropagation.

This is "Path A" of the Dual Learning Engine.
"""

import numpy as np
from typing import List, Dict, Optional
from trialnet.core.tensor import Tensor


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, params: List[Tensor], learning_rate: float = 0.001):
        self.params = params
        self.learning_rate = learning_rate
        self.step_count = 0

    def step(self):
        """Update parameters using their gradients."""
        raise NotImplementedError

    def zero_grad(self):
        """Reset all parameter gradients to zero."""
        for param in self.params:
            param.zero_grad()


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.

    Update rule:
    - Without momentum: w = w - lr * grad
    - With momentum:    v = momentum * v - lr * grad
                       w = w + v
    """

    def __init__(
        self,
        params: List[Tensor],
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        super().__init__(params, learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Velocity buffers for momentum
        self._velocities: Dict[int, np.ndarray] = {}
        for i, param in enumerate(self.params):
            self._velocities[i] = np.zeros_like(param.data)

    def step(self):
        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.copy()

            # L2 regularization
            if self.weight_decay > 0:
                grad += self.weight_decay * param.data

            if self.momentum > 0:
                # Momentum update
                self._velocities[i] = (self.momentum * self._velocities[i] -
                                       self.learning_rate * grad)
                param.data += self._velocities[i]
            else:
                # Standard SGD
                param.data -= self.learning_rate * grad


class Adam(Optimizer):
    """
    Adam Optimizer — Adaptive Moment Estimation.

    Combines the benefits of:
    - Momentum (first moment)
    - RMSProp (second moment)

    Update rule:
        m = beta1 * m + (1 - beta1) * grad           (first moment)
        v = beta2 * v + (1 - beta2) * grad^2          (second moment)
        m_hat = m / (1 - beta1^t)                      (bias correction)
        v_hat = v / (1 - beta2^t)                      (bias correction)
        w = w - lr * m_hat / (sqrt(v_hat) + eps)       (update)
    """

    def __init__(
        self,
        params: List[Tensor],
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(params, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        # First moment (mean) buffers
        self._m: Dict[int, np.ndarray] = {}
        # Second moment (variance) buffers
        self._v: Dict[int, np.ndarray] = {}

        for i, param in enumerate(self.params):
            self._m[i] = np.zeros_like(param.data)
            self._v[i] = np.zeros_like(param.data)

    def step(self):
        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.copy()

            # L2 regularization (decoupled weight decay — AdamW style)
            if self.weight_decay > 0:
                param.data -= self.learning_rate * self.weight_decay * param.data

            # Update first moment (momentum)
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * grad
            # Update second moment (RMSProp)
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self._m[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self._v[i] / (1 - self.beta2 ** self.step_count)

            # Update weights
            param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


class LearningRateScheduler:
    """
    Learning rate scheduler with multiple strategies.
    
    Strategies:
    - 'constant': No change
    - 'step': Decay by factor every N epochs
    - 'cosine': Cosine annealing
    - 'warmup': Linear warmup then decay
    """

    def __init__(
        self,
        optimizer: Optimizer,
        strategy: str = 'constant',
        **kwargs
    ):
        self.optimizer = optimizer
        self.strategy = strategy
        self.initial_lr = optimizer.learning_rate
        self.kwargs = kwargs

    def step(self, epoch: int):
        """Update learning rate based on current epoch."""
        if self.strategy == 'constant':
            return

        elif self.strategy == 'step':
            step_size = self.kwargs.get('step_size', 30)
            gamma = self.kwargs.get('gamma', 0.1)
            self.optimizer.learning_rate = self.initial_lr * (gamma ** (epoch // step_size))

        elif self.strategy == 'cosine':
            total_epochs = self.kwargs.get('total_epochs', 100)
            min_lr = self.kwargs.get('min_lr', 1e-6)
            self.optimizer.learning_rate = min_lr + 0.5 * (self.initial_lr - min_lr) * (
                1 + np.cos(np.pi * epoch / total_epochs)
            )

        elif self.strategy == 'warmup':
            warmup_epochs = self.kwargs.get('warmup_epochs', 5)
            if epoch < warmup_epochs:
                self.optimizer.learning_rate = self.initial_lr * (epoch + 1) / warmup_epochs
            else:
                total_epochs = self.kwargs.get('total_epochs', 100)
                decay_epochs = total_epochs - warmup_epochs
                progress = (epoch - warmup_epochs) / max(decay_epochs, 1)
                self.optimizer.learning_rate = self.initial_lr * (1 - progress) + 1e-6

    def get_lr(self) -> float:
        return self.optimizer.learning_rate


# ── Factory ────────────────────────────────────────────────────────────

OPTIMIZER_MAP = {
    'sgd': SGD,
    'adam': Adam,
}


def get_optimizer(name: str, params: List[Tensor], **kwargs) -> Optimizer:
    """Get optimizer by name."""
    name = name.lower()
    if name not in OPTIMIZER_MAP:
        raise ValueError(f"Unknown optimizer: {name}. Choose from: {list(OPTIMIZER_MAP.keys())}")
    return OPTIMIZER_MAP[name](params, **kwargs)
