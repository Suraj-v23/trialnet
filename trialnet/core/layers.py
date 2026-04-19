"""
layers.py — Neural Network Layers for TrialNet

Dense (fully connected) and Dropout layers built from scratch.
Each layer manages its own weights, biases, forward cache, and gradients.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from trialnet.core.tensor import Tensor, xavier_init, he_init, zeros
from trialnet.core.activations import Activation, get_activation


class Layer:
    """Base class for all layers."""

    def __init__(self, name: str = ""):
        self.name = name
        self.trainable = True
        self._training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_params(self) -> list:
        """Return list of trainable Tensor parameters."""
        return []

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class DenseLayer(Layer):
    """
    Fully Connected (Dense) Layer.

    Performs: output = activation(input @ weights + bias)

    Features:
    - Xavier or He weight initialization
    - Stores forward cache for backpropagation
    - Stores activation history for the Error Memory Bank
    - Supports weight snapshot/restore for perturbation exploration
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = 'relu',
        init_method: str = 'he',
        name: str = ""
    ):
        super().__init__(name=name or f"dense_{input_size}x{output_size}")
        self.input_size = input_size
        self.output_size = output_size

        # Initialize weights
        if init_method == 'xavier':
            self.weights = xavier_init(input_size, output_size)
        else:
            self.weights = he_init(input_size, output_size)

        self.weights.name = f"{self.name}.weights"
        self.bias = zeros((1, output_size), name=f"{self.name}.bias")

        # Activation function
        self.activation: Activation = get_activation(activation)
        self.activation_name = activation

        # Forward cache (stored during forward pass, used in backward pass)
        self._input_cache: Optional[np.ndarray] = None
        self._pre_activation: Optional[np.ndarray] = None
        self._post_activation: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: z = x @ W + b, output = activation(z)
        """
        self._input_cache = x

        # Linear transformation
        z = x @ self.weights.data + self.bias.data

        # Clip to prevent numerical overflow (critical for NumPy stability)
        z = np.clip(z, -500, 500)
        z = np.nan_to_num(z, nan=0.0, posinf=500.0, neginf=-500.0)
        self._pre_activation = z

        # Apply activation
        output = self.activation.forward(z)
        self._post_activation = output

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass — computes gradients for weights, bias, and input.

        Chain rule:
        dL/dW = input^T @ (dL/dz)
        dL/db = sum(dL/dz, axis=0)
        dL/dinput = (dL/dz) @ W^T

        where dL/dz = activation.backward(dL/doutput)
        """
        batch_size = grad_output.shape[0]

        # Sanitize incoming gradient
        grad_output = np.nan_to_num(grad_output, nan=0.0, posinf=1.0, neginf=-1.0)

        # Gradient through activation
        grad_z = self.activation.backward(grad_output)

        # Clip gradients to prevent explosion
        grad_z = np.clip(grad_z, -10.0, 10.0)

        # Gradient for weights: dL/dW = X^T @ dL/dz
        grad_weights = self._input_cache.T @ grad_z / batch_size
        grad_weights = np.clip(grad_weights, -5.0, 5.0)
        self.weights.accumulate_grad(grad_weights)

        # Gradient for bias: dL/db = mean(dL/dz)
        grad_bias = np.mean(grad_z, axis=0, keepdims=True)
        grad_bias = np.clip(grad_bias, -5.0, 5.0)
        self.bias.accumulate_grad(grad_bias)

        # Gradient for input (to pass to previous layer)
        grad_input = grad_z @ self.weights.data.T
        grad_input = np.nan_to_num(grad_input, nan=0.0, posinf=1.0, neginf=-1.0)

        return grad_input

    def get_params(self) -> list:
        return [self.weights, self.bias]

    def get_activation_stats(self) -> Dict:
        """Get statistics about layer activations (used by Mistake Analyzer)."""
        if self._post_activation is None:
            return {}
        return {
            'mean': float(np.mean(self._post_activation)),
            'std': float(np.std(self._post_activation)),
            'max': float(np.max(self._post_activation)),
            'min': float(np.min(self._post_activation)),
            'dead_ratio': float(np.mean(self._post_activation == 0)),
            'sparsity': float(np.mean(np.abs(self._post_activation) < 0.01)),
        }

    def __repr__(self):
        return (f"DenseLayer({self.input_size} → {self.output_size}, "
                f"activation={self.activation_name}, name='{self.name}')")


class DropoutLayer(Layer):
    """
    Dropout Layer — Regularization technique.

    During training: randomly sets a fraction of inputs to zero.
    During evaluation: passes through unchanged (scaled).
    """

    def __init__(self, rate: float = 0.5, name: str = ""):
        super().__init__(name=name or f"dropout_{rate}")
        self.rate = rate
        self.trainable = False
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self._training:
            # Generate dropout mask
            self._mask = (np.random.rand(*x.shape) > self.rate).astype(np.float64)
            # Scale to maintain expected values
            return x * self._mask / (1.0 - self.rate)
        else:
            return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._training and self._mask is not None:
            return grad_output * self._mask / (1.0 - self.rate)
        return grad_output

    def __repr__(self):
        return f"DropoutLayer(rate={self.rate})"


class BatchNormLayer(Layer):
    """
    Batch Normalization Layer.

    Normalizes layer inputs to have zero mean and unit variance,
    then applies learnable scale (gamma) and shift (beta).
    Helps with training stability and speed.
    """

    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-8, name: str = ""):
        super().__init__(name=name or f"batchnorm_{num_features}")
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Learnable parameters
        self.gamma = Tensor(np.ones((1, num_features)), requires_grad=True, name=f"{self.name}.gamma")
        self.beta = Tensor(np.zeros((1, num_features)), requires_grad=True, name=f"{self.name}.beta")

        # Running statistics for evaluation mode
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

        # Cache for backward pass
        self._x_norm = None
        self._std = None
        self._x_centered = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self._training:
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            self._x_centered = x - mean
            self._std = np.sqrt(var + self.eps)
            self._x_norm = self._x_centered / self._std
        else:
            self._x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma.data * self._x_norm + self.beta.data

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        batch_size = grad_output.shape[0]

        # Gradients for gamma and beta
        grad_gamma = np.sum(grad_output * self._x_norm, axis=0, keepdims=True)
        grad_beta = np.sum(grad_output, axis=0, keepdims=True)
        self.gamma.accumulate_grad(grad_gamma / batch_size)
        self.beta.accumulate_grad(grad_beta / batch_size)

        # Gradient for input
        dx_norm = grad_output * self.gamma.data
        dvar = np.sum(dx_norm * self._x_centered * -0.5 * self._std ** (-3), axis=0, keepdims=True)
        dmean = np.sum(dx_norm * -1.0 / self._std, axis=0, keepdims=True)

        grad_input = (dx_norm / self._std +
                      dvar * 2.0 * self._x_centered / batch_size +
                      dmean / batch_size)

        return grad_input

    def get_params(self) -> list:
        return [self.gamma, self.beta]

    def __repr__(self):
        return f"BatchNormLayer({self.num_features})"
