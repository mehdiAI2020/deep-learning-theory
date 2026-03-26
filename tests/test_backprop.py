"""Unit tests for backpropagation implementation."""

import pytest
import numpy as np
from src.backprop import Tensor, backward


class TestTensorOperations:
    """Test basic tensor operations."""

    def test_tensor_creation(self):
        """Test tensor creation."""
        x = Tensor(np.array([1, 2, 3]))
        assert x.data.shape == (3,)
        assert x.grad.shape == (3,)

    def test_addition(self):
        """Test tensor addition."""
        x = Tensor(np.array([1, 2, 3]))
        y = Tensor(np.array([4, 5, 6]))
        z = x + y

        assert np.allclose(z.data, np.array([5, 7, 9]))

    def test_multiplication(self):
        """Test tensor multiplication."""
        x = Tensor(np.array([2, 3]))
        y = Tensor(np.array([4, 5]))
        z = x * y

        assert np.allclose(z.data, np.array([8, 15]))

    def test_division(self):
        """Test tensor division."""
        x = Tensor(np.array([10.0, 20.0]))
        y = Tensor(np.array([2.0, 4.0]))
        z = x / y

        assert np.allclose(z.data, np.array([5.0, 5.0]))

    def test_power(self):
        """Test tensor power."""
        x = Tensor(np.array([2.0, 3.0]))
        z = x ** 2

        assert np.allclose(z.data, np.array([4.0, 9.0]))

    def test_relu(self):
        """Test ReLU activation."""
        x = Tensor(np.array([-1, 0, 1, 2]))
        y = x.relu()

        assert np.allclose(y.data, np.array([0, 0, 1, 2]))

    def test_sigmoid(self):
        """Test sigmoid activation."""
        x = Tensor(np.array([0.0]))
        y = x.sigmoid()

        # sigmoid(0) = 0.5
        assert np.isclose(y.data[0], 0.5)


class TestBackpropagation:
    """Test backpropagation through various operations."""

    def test_simple_backprop(self):
        """Test basic backpropagation: y = 2x, dy/dx should be 2."""
        x = Tensor(np.array([3.0]))
        y = x * Tensor(np.array([2.0]))
        y.backward()

        assert np.isclose(x.grad[0], 2.0)

    def test_chain_rule(self):
        """Test chain rule: z = (x*2)² = 4x²"""
        x = Tensor(np.array([3.0]))
        y = x * Tensor(np.array([2.0]))
        z = y ** 2
        z.backward()

        # dz/dx = 8*x = 8*3 = 24
        assert np.isclose(x.grad[0], 24.0)

    def test_sum_gradient(self):
        """Test gradient through sum operation."""
        x = Tensor(np.array([1.0, 2.0, 3.0]))
        y = x.sum()
        y.backward()

        # Gradient w.r.t. all elements should be 1
        assert np.allclose(x.grad, np.ones(3))

    def test_mean_gradient(self):
        """Test gradient through mean operation."""
        x = Tensor(np.array([1.0, 2.0, 3.0]))
        y = x.mean()
        y.backward()

        # Gradient should be 1/3 for each element
        assert np.allclose(x.grad, np.ones(3) / 3)

    def test_multiple_paths(self):
        """Test gradient with multiple computational paths."""
        x = Tensor(np.array([2.0]))

        # y = x + x = 2x, so dy/dx should be 2
        y = x + x
        y.backward()

        # Both paths should contribute to gradient
        assert np.isclose(x.grad[0], 2.0)

    def test_quadratic_loss(self):
        """Test backprop on quadratic loss: L = (y-t)²"""
        y = Tensor(np.array([3.0]))  # Prediction
        t = Tensor(np.array([2.0]))  # Target
        loss = (y - t) ** 2
        loss.backward()

        # dL/dy = 2*(y-t) = 2*(3-2) = 2
        assert np.isclose(y.grad[0], 2.0)

    def test_complex_expression(self):
        """Test complex computation graph."""
        x = Tensor(np.array([2.0]))
        w = Tensor(np.array([3.0]))

        # y = wx + 1, then z = y²
        # dz/dx = 2*y*w = 2*(2*3 + 1)*3 = 42
        y = w * x + Tensor(np.array([1.0]))
        z = y ** 2
        z.backward()

        assert np.isclose(x.grad[0], 42.0)


class TestActivationGradients:
    """Test gradients through activation functions."""

    def test_relu_gradient(self):
        """Test ReLU gradient: d(ReLU)/dx = 1 if x > 0 else 0."""
        x = Tensor(np.array([-1.0, 0.5, 2.0]))
        y = x.relu()
        loss = y.sum()
        loss.backward()

        expected_grad = np.array([0.0, 1.0, 1.0])
        assert np.allclose(x.grad, expected_grad)

    def test_sigmoid_gradient(self):
        """Test sigmoid gradient at x=0 (should be 0.25)."""
        x = Tensor(np.array([0.0]))
        y = x.sigmoid()
        loss = y.sum()
        loss.backward()

        # d(sigmoid)/dx at x=0 = sigmoid(0)*(1-sigmoid(0)) = 0.5*0.5 = 0.25
        assert np.isclose(x.grad[0], 0.25)

    def test_tanh_gradient(self):
        """Test tanh gradient at x=0 (should be 1)."""
        x = Tensor(np.array([0.0]))
        y = x.tanh()
        loss = y.sum()
        loss.backward()

        # d(tanh)/dx at x=0 = 1 - tanh(0)² = 1
        assert np.isclose(x.grad[0], 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
