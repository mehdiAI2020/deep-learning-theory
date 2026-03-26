"""
Normalization Techniques from First Principles

Normalization standardizes inputs or activations to improve training stability
and convergence speed.

Key insight: By centering and scaling activations, we prevent the internal
covariate shift problem, where the distribution of layer inputs changes during
training, slowing convergence.

References:
- BatchNorm: Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training"
- LayerNorm: Ba et al., "Layer Normalization"
"""

import numpy as np
from typing import Tuple


class BatchNorm1d:
    """
    Batch Normalization for 1D inputs (per-feature normalization).

    During training: normalize using batch statistics
    During inference: normalize using running statistics

    Normalization formula:
        x̂ = (x - μ_batch) / √(σ²_batch + ε)
        y = γ * x̂ + β

    where:
    - μ_batch, σ²_batch: Mean and variance computed over batch
    - γ, β: Learnable scale and shift parameters
    - ε: Small constant for numerical stability
    """

    def __init__(self, num_features: int, epsilon: float = 1e-5, momentum: float = 0.1):
        """
        Args:
            num_features: Number of features (input dimension)
            epsilon: Small constant for stability (default: 1e-5)
            momentum: Momentum for running statistics (default: 0.1)
        """
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Running statistics (for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # Gradients
        self.gamma_grad = np.zeros(num_features)
        self.beta_grad = np.zeros(num_features)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass of batch normalization.

        Args:
            x: Input of shape (batch_size, num_features)
            training: Whether in training or inference mode

        Returns:
            Normalized output of same shape
        """
        if training:
            # Compute batch statistics
            mu = np.mean(x, axis=0)  # (num_features,)
            var = np.var(x, axis=0)  # (num_features,)

            # Normalize
            x_normalized = (x - mu) / np.sqrt(var + self.epsilon)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            # Store for backward pass
            self.x_normalized = x_normalized
            self.mu = mu
            self.var = var
        else:
            # Use running statistics
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        # Scale and shift
        y = self.gamma * x_normalized + self.beta

        return y

    def backward(self, dL_dy: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Backward pass through batch normalization.

        Args:
            dL_dy: Gradient from next layer (batch_size, num_features)
            x: Original input

        Returns:
            Gradient w.r.t. input
        """
        batch_size = x.shape[0]

        # Gradients w.r.t. parameters
        self.beta_grad = np.sum(dL_dy, axis=0)
        self.gamma_grad = np.sum(dL_dy * self.x_normalized, axis=0)

        # Gradient w.r.t. normalized input
        dL_dx_norm = dL_dy * self.gamma

        # Gradient w.r.t. variance
        dL_dvar = np.sum(dL_dx_norm * (x - self.mu) * -0.5 * (self.var + self.epsilon) ** -1.5, axis=0)

        # Gradient w.r.t. mean
        dL_dmu = np.sum(dL_dx_norm * -1 / np.sqrt(self.var + self.epsilon), axis=0)
        dL_dmu += dL_dvar * np.sum(-2 * (x - self.mu), axis=0) / batch_size

        # Gradient w.r.t. input
        dL_dx = dL_dx_norm / np.sqrt(self.var + self.epsilon)
        dL_dx += dL_dvar * 2 * (x - self.mu) / batch_size
        dL_dx += dL_dmu / batch_size

        return dL_dx


class LayerNorm:
    """
    Layer Normalization (normalizes across feature dimension).

    Unlike BatchNorm which normalizes across batch dimension,
    LayerNorm normalizes across the feature dimension.

    Formula:
        x̂ = (x - μ_feature) / √(σ²_feature + ε)
        y = γ * x̂ + β

    Advantages:
    - Works with batch size 1 (no batch statistics needed)
    - Often better for Transformers and RNNs
    - Independent of batch composition
    """

    def __init__(self, num_features: int, epsilon: float = 1e-5):
        """
        Args:
            num_features: Number of features
            epsilon: Small constant for stability
        """
        self.num_features = num_features
        self.epsilon = epsilon

        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Gradients
        self.gamma_grad = np.zeros(num_features)
        self.beta_grad = np.zeros(num_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of layer normalization.

        Args:
            x: Input of shape (batch_size, num_features) or (num_features,)

        Returns:
            Normalized output of same shape
        """
        # Normalize across feature dimension
        if x.ndim == 1:
            mu = np.mean(x)
            var = np.var(x)
        else:
            mu = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)

        # Normalize
        x_normalized = (x - mu) / np.sqrt(var + self.epsilon)

        # Scale and shift
        y = self.gamma * x_normalized + self.beta

        # Store for backward
        self.x_normalized = x_normalized
        self.mu = mu
        self.var = var

        return y

    def backward(self, dL_dy: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Backward pass through layer normalization.

        Args:
            dL_dy: Gradient from next layer
            x: Original input

        Returns:
            Gradient w.r.t. input
        """
        # Gradients w.r.t. parameters
        self.beta_grad = np.sum(dL_dy, axis=tuple(range(dL_dy.ndim - 1)))
        self.gamma_grad = np.sum(dL_dy * self.x_normalized, axis=tuple(range(dL_dy.ndim - 1)))

        # Gradient w.r.t. normalized input
        dL_dx_norm = dL_dy * self.gamma

        # Gradient w.r.t. variance and mean
        N = x.shape[-1]  # Number of features
        dL_dvar = np.sum(
            dL_dx_norm * (x - self.mu) * -0.5 * (self.var + self.epsilon) ** -1.5,
            axis=-1,
            keepdims=True,
        )
        dL_dmu = np.sum(dL_dx_norm * -1 / np.sqrt(self.var + self.epsilon), axis=-1, keepdims=True)
        dL_dmu += dL_dvar * np.sum(-2 * (x - self.mu), axis=-1, keepdims=True) / N

        # Gradient w.r.t. input
        dL_dx = dL_dx_norm / np.sqrt(self.var + self.epsilon)
        dL_dx += dL_dvar * 2 * (x - self.mu) / N
        dL_dx += dL_dmu / N

        return dL_dx
