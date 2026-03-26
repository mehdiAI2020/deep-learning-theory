"""Unit tests for optimization algorithms."""

import pytest
import numpy as np
from src.optimizers import SGD, Momentum, Adam


class TestOptimizers:
    """Test optimizer algorithms."""

    def test_sgd_basic(self):
        """Test basic SGD update."""
        sgd = SGD(learning_rate=0.1)

        params = {"w": np.array([1.0, 2.0])}
        grads = {"w": np.array([0.1, 0.2])}

        updated = sgd.step(params, grads)

        expected = params["w"] - 0.1 * grads["w"]
        assert np.allclose(updated["w"], expected)

    def test_sgd_convergence(self):
        """Test SGD on simple quadratic: L = x²"""
        sgd = SGD(learning_rate=0.1)

        x = np.array([1.0])

        for _ in range(10):
            # Gradient of x² is 2x
            grad = 2 * x
            params = {"x": x}
            grads = {"x": np.array([grad])}
            updated = sgd.step(params, grads)
            x = updated["x"]

        # Should converge towards 0
        assert abs(x[0]) < 0.01

    def test_momentum_accumulation(self):
        """Test that momentum accumulates gradients."""
        momentum = Momentum(learning_rate=0.1, momentum=0.9)

        params = {"w": np.array([1.0])}
        grads = {"w": np.array([1.0])}

        # First step
        updated1 = momentum.step(params, grads)

        # Second step (same gradient)
        updated2 = momentum.step(updated1, grads)

        # Momentum should make second step larger
        delta1 = params["w"] - updated1["w"]
        delta2 = updated1["w"] - updated2["w"]

        assert delta2 > delta1

    def test_momentum_convergence(self):
        """Test momentum convergence on quadratic."""
        momentum = Momentum(learning_rate=0.1, momentum=0.9)

        x = np.array([1.0])

        for _ in range(20):
            grad = 2 * x
            params = {"x": x}
            grads = {"x": np.array([grad])}
            updated = momentum.step(params, grads)
            x = updated["x"]

        assert abs(x[0]) < 0.01

    def test_adam_initialization(self):
        """Test Adam initializes moments correctly."""
        adam = Adam(learning_rate=0.001)

        params = {"w": np.array([1.0, 2.0])}
        grads = {"w": np.array([0.1, 0.2])}

        # First step
        updated = adam.step(params, grads)

        # Moments should be initialized and updated
        assert "w" in adam.first_moments
        assert "w" in adam.second_moments
        assert adam.t == 1

    def test_adam_convergence(self):
        """Test Adam convergence on quadratic."""
        adam = Adam(learning_rate=0.01)

        x = np.array([1.0])

        for _ in range(50):
            grad = 2 * x
            params = {"x": x}
            grads = {"x": np.array([grad])}
            updated = adam.step(params, grads)
            x = updated["x"]

        assert abs(x[0]) < 0.01

    def test_adam_bias_correction(self):
        """Test that Adam performs bias correction in early steps."""
        adam = Adam(learning_rate=0.01)

        params = {"w": np.array([1.0])}
        grads = {"w": np.array([1.0])}

        # First step: m_hat = m / (1 - 0.9^1) = m / 0.1
        updated1 = adam.step(params, grads)

        # Without bias correction, would move too little
        # With bias correction, should move appropriately
        assert updated1["w"] < params["w"]

    def test_adam_adaptive_learning(self):
        """Test that Adam adapts learning rate based on gradient magnitudes."""
        adam = Adam(learning_rate=0.01)

        # Large gradient
        params1 = {"w": np.array([1.0])}
        grads1 = {"w": np.array([10.0])}
        updated1 = adam.step(params1, grads1)
        delta1 = params1["w"] - updated1["w"]

        # Small gradient
        adam2 = Adam(learning_rate=0.01)
        params2 = {"w": np.array([1.0])}
        grads2 = {"w": np.array([0.1])}
        updated2 = adam2.step(params2, grads2)
        delta2 = params2["w"] - updated2["w"]

        # Both should move in same direction but with adapted scales
        assert delta1 > 0 and delta2 > 0


class TestMultipleParameters:
    """Test optimizers on multiple parameters."""

    def test_sgd_multiple_params(self):
        """Test SGD with multiple parameters."""
        sgd = SGD(learning_rate=0.01)

        params = {"w": np.array([1.0, 2.0]), "b": np.array([0.5])}
        grads = {"w": np.array([0.1, 0.2]), "b": np.array([0.05])}

        updated = sgd.step(params, grads)

        assert "w" in updated and "b" in updated
        assert updated["w"].shape == params["w"].shape
        assert updated["b"].shape == params["b"].shape

    def test_adam_multiple_params(self):
        """Test Adam with multiple parameters."""
        adam = Adam()

        params = {"w": np.ones((3, 4)), "b": np.zeros(4)}
        grads = {"w": np.ones((3, 4)) * 0.01, "b": np.ones(4) * 0.01}

        for _ in range(5):
            updated = adam.step(params, grads)
            params = updated

        # Should converge
        assert np.all(np.abs(params["w"]) < 0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
