"""
Optimization Algorithms from First Principles

All optimization algorithms follow the same principle:
    θ_{t+1} = θ_t - learning_rate * ∂L/∂θ

But different algorithms compute different effective learning rates based on
the history of gradients.

References:
- SGD: Stochastic Gradient Descent
- Momentum: Nesterov Accelerated Gradient
- Adam: Adaptive Moment Estimation
"""

import numpy as np
from typing import List, Dict


class SGD:
    """
    Stochastic Gradient Descent (SGD).

    Simple update rule:
        θ ← θ - α * ∇L(θ)

    where α is the learning rate and ∇L(θ) is the gradient.
    """

    def __init__(self, learning_rate: float = 0.01):
        """
        Args:
            learning_rate: Learning rate (default: 0.01)
        """
        self.learning_rate = learning_rate

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict:
        """
        Update parameters using SGD.

        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients

        Returns:
            Updated parameters
        """
        updated = {}
        for name, param in params.items():
            grad = grads.get(name, np.zeros_like(param))
            updated[name] = param - self.learning_rate * grad

        return updated


class Momentum:
    """
    Momentum Optimizer (Nesterov Accelerated Gradient).

    Accumulates gradient history to accelerate convergence and reduce oscillations.

    Update rules:
        v_t = β * v_{t-1} + ∇L(θ)
        θ = θ - α * v_t

    where:
    - β is momentum coefficient (typically 0.9)
    - v_t is the accumulated velocity/momentum
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """
        Args:
            learning_rate: Learning rate
            momentum: Momentum coefficient (default: 0.9)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict:
        """
        Update parameters with momentum.

        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients

        Returns:
            Updated parameters
        """
        updated = {}

        for name, param in params.items():
            grad = grads.get(name, np.zeros_like(param))

            # Initialize velocity if first time
            if name not in self.velocities:
                self.velocities[name] = np.zeros_like(param)

            # v_t = β * v_{t-1} + ∇L(θ)
            self.velocities[name] = self.momentum * self.velocities[name] + grad

            # θ = θ - α * v_t
            updated[name] = param - self.learning_rate * self.velocities[name]

        return updated


class Adam:
    """
    Adaptive Moment Estimation (Adam).

    Combines advantages of:
    - Momentum: Accumulates gradient history
    - RMSProp: Adapts learning rate per parameter

    Update rules:
        m_t = β₁ * m_{t-1} + (1 - β₁) * ∇L
        v_t = β₂ * v_{t-1} + (1 - β₂) * ∇L²
        m̂_t = m_t / (1 - β₁^t)        (bias correction)
        v̂_t = v_t / (1 - β₂^t)        (bias correction)
        θ = θ - α * m̂_t / (√v̂_t + ε)

    where:
    - m_t: First moment (mean) of gradients
    - v_t: Second moment (variance) of gradients
    - β₁, β₂: Exponential decay rates (default: 0.9, 0.999)
    - ε: Small constant for numerical stability (default: 1e-8)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """
        Args:
            learning_rate: Learning rate (default: 0.001)
            beta1: Exponential decay rate for first moment (default: 0.9)
            beta2: Exponential decay rate for second moment (default: 0.999)
            epsilon: Small constant for numerical stability (default: 1e-8)
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first_moments = {}
        self.second_moments = {}
        self.t = 0  # Time step counter

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict:
        """
        Update parameters using Adam.

        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients

        Returns:
            Updated parameters
        """
        self.t += 1
        updated = {}

        for name, param in params.items():
            grad = grads.get(name, np.zeros_like(param))

            # Initialize moments if first time
            if name not in self.first_moments:
                self.first_moments[name] = np.zeros_like(param)
                self.second_moments[name] = np.zeros_like(param)

            # m_t = β₁ * m_{t-1} + (1 - β₁) * ∇L
            self.first_moments[name] = (
                self.beta1 * self.first_moments[name] + (1 - self.beta1) * grad
            )

            # v_t = β₂ * v_{t-1} + (1 - β₂) * ∇L²
            self.second_moments[name] = (
                self.beta2 * self.second_moments[name] + (1 - self.beta2) * (grad ** 2)
            )

            # Bias correction
            m_hat = self.first_moments[name] / (1 - self.beta1 ** self.t)
            v_hat = self.second_moments[name] / (1 - self.beta2 ** self.t)

            # θ = θ - α * m̂_t / (√v̂_t + ε)
            updated[name] = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updated
