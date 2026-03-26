"""
Backpropagation from First Principles

Backpropagation is the chain rule applied systematically to compute gradients
of a loss function with respect to all parameters.

Mathematical Foundation:
    ∂L/∂θ = ∂L/∂ŷ · ∂ŷ/∂θ₂ · ∂θ₂/∂θ₁ · ... (chain rule)

Key insight: We can compute ∂L/∂ layer_input from ∂L/∂layer_output
using the same gradient computation rules, working backwards through the network.
"""

import numpy as np
from typing import List, Callable, Optional


class Tensor:
    """
    A tensor with automatic differentiation support.

    Stores:
    - data: The numerical value
    - grad: Gradient with respect to loss
    - _children: Tensors that contributed to this value
    - _op: The operation that created this tensor
    """

    def __init__(self, data: np.ndarray, _children=(), _op: str = ""):
        """
        Args:
            data: Numpy array containing values
            _children: Tuple of (previous tensor, operation)
            _op: Name of operation that created this tensor
        """
        self.data = data
        self.grad = np.zeros_like(data, dtype=np.float32)
        self._children = _children
        self._op = _op

    def __repr__(self) -> str:
        return f"Tensor({self.data.shape}, op={self._op})"

    def __add__(self, other: "Tensor") -> "Tensor":
        """Addition: d(a+b)/da = 1, d(a+b)/db = 1"""
        if not isinstance(other, Tensor):
            other = Tensor(np.array([other]))

        output = Tensor(self.data + other.data, _children=(self, other), _op="add")
        return output

    def __sub__(self, other: "Tensor") -> "Tensor":
        """Subtraction: d(a-b)/da = 1, d(a-b)/db = -1"""
        if not isinstance(other, Tensor):
            other = Tensor(np.array([other]))

        output = Tensor(self.data - other.data, _children=(self, other), _op="sub")
        return output

    def __mul__(self, other: "Tensor") -> "Tensor":
        """Multiplication: d(a*b)/da = b, d(a*b)/db = a"""
        if not isinstance(other, Tensor):
            other = Tensor(np.array([other]))

        output = Tensor(self.data * other.data, _children=(self, other), _op="mul")
        return output

    def __truediv__(self, other: "Tensor") -> "Tensor":
        """Division: d(a/b)/da = 1/b, d(a/b)/db = -a/b²"""
        if not isinstance(other, Tensor):
            other = Tensor(np.array([other]))

        output = Tensor(self.data / other.data, _children=(self, other), _op="div")
        return output

    def __pow__(self, exp: float) -> "Tensor":
        """Power: d(a^n)/da = n*a^(n-1)"""
        output = Tensor(self.data ** exp, _children=(self,), _op=f"pow({exp})")
        return output

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return Tensor(np.array([other])) - self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * Tensor(np.array([-1.0]))

    def relu(self) -> "Tensor":
        """ReLU: max(0, x). Derivative: 1 if x > 0 else 0"""
        output = Tensor(np.maximum(0, self.data), _children=(self,), _op="relu")
        return output

    def sigmoid(self) -> "Tensor":
        """Sigmoid: σ(x) = 1/(1+e^(-x)). Derivative: σ(x)(1-σ(x))"""
        sig = 1 / (1 + np.exp(-self.data))
        output = Tensor(sig, _children=(self,), _op="sigmoid")
        return output

    def tanh(self) -> "Tensor":
        """Hyperbolic tangent. Derivative: 1 - tanh²(x)"""
        t = np.tanh(self.data)
        output = Tensor(t, _children=(self,), _op="tanh")
        return output

    def log(self) -> "Tensor":
        """Natural logarithm. Derivative: 1/x"""
        output = Tensor(np.log(self.data + 1e-8), _children=(self,), _op="log")
        return output

    def sum(self) -> "Tensor":
        """Sum all elements. Derivative: ones"""
        output = Tensor(np.array([np.sum(self.data)]), _children=(self,), _op="sum")
        return output

    def mean(self) -> "Tensor":
        """Mean of all elements."""
        return self.sum() / Tensor(np.array([self.data.size]))

    def backward(self, gradient=None):
        """
        Compute gradients using reverse-mode automatic differentiation (backpropagation).

        This implements the chain rule:
            ∂L/∂a = ∂L/∂output · ∂output/∂a
        """
        if gradient is None:
            # If no gradient provided, assume this is the loss (gradient = 1)
            gradient = np.ones_like(self.data, dtype=np.float32)

        self.grad = gradient

        # Process each operation in reverse
        if self._op == "add":
            self._children[0].backward(self.grad)
            self._children[1].backward(self.grad)

        elif self._op == "sub":
            self._children[0].backward(self.grad)
            self._children[1].backward(-self.grad)

        elif self._op == "mul":
            # d(a*b)/da = b, d(a*b)/db = a
            self._children[0].backward(self.grad * self._children[1].data)
            self._children[1].backward(self.grad * self._children[0].data)

        elif self._op == "div":
            # d(a/b)/da = 1/b, d(a/b)/db = -a/b²
            self._children[0].backward(self.grad / self._children[1].data)
            self._children[1].backward(-self.grad * self._children[0].data / (self._children[1].data ** 2))

        elif self._op.startswith("pow"):
            exp = float(self._op.split("(")[1].rstrip(")"))
            self._children[0].backward(self.grad * exp * (self._children[0].data ** (exp - 1)))

        elif self._op == "relu":
            mask = self._children[0].data > 0
            self._children[0].backward(self.grad * mask)

        elif self._op == "sigmoid":
            sig = 1 / (1 + np.exp(-self._children[0].data))
            self._children[0].backward(self.grad * sig * (1 - sig))

        elif self._op == "tanh":
            t = np.tanh(self._children[0].data)
            self._children[0].backward(self.grad * (1 - t ** 2))

        elif self._op == "log":
            self._children[0].backward(self.grad / (self._children[0].data + 1e-8))

        elif self._op == "sum":
            self._children[0].backward(np.full_like(self._children[0].data, self.grad.item()))

    def zero_grad(self):
        """Clear gradients."""
        self.grad = np.zeros_like(self.data, dtype=np.float32)


def scalar_to_tensor(value: float) -> Tensor:
    """Convert scalar to Tensor."""
    return Tensor(np.array([value], dtype=np.float32))


def backward(loss: Tensor):
    """Compute gradients from loss."""
    loss.backward()
