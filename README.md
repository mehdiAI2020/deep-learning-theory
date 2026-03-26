# Deep Learning Theory from Scratch

> **Implementing Core DL Concepts from Mathematical First Principles**

A structured collection of implementations revealing the mathematical foundations of deep learning:
- ✅ **Backpropagation**: Gradient computation from chain rule
- ✅ **Optimizers**: SGD, Momentum, Adam derived from first principles
- ✅ **Normalization**: BatchNorm, LayerNorm mathematical foundations
- ✅ **Generalization**: PAC learning, bias-variance tradeoff, double descent
- ✅ **Information Theory**: KL divergence, mutual information, entropy

## 📋 Modules

### 1. Backpropagation
```python
# src: backpropagation/
├── scalar_backprop.py      # Single scalar: first principles
├── vector_backprop.py      # Jacobian matrices
├── autograd_engine.py      # Minimal autodiff system
└── gradient_flow.py        # Analysis of gradient flow
```

Key insight: Backprop is **reverse-mode automatic differentiation** computing gradients via the chain rule.

### 2. Optimization
```python
# src: optimization/
├── gradient_descent.py     # Basic GD convergence analysis
├── momentum.py             # Momentum & Nesterov
├── adaptive_learning.py    # AdaGrad, RMSprop
├── adam_optimizer.py       # Adam: first and second moments
└── convergence_analysis.py # When optimizers converge/diverge
```

Key insight: Different optimizers balance **first-order** (gradient) and **second-order** (curvature) information differently.

### 3. Normalization
```python
# src: normalization/
├── batch_norm.py           # Reduce internal covariate shift
├── layer_norm.py           # Stabilize activations
├── instance_norm.py        # Per-instance statistics
└── normalization_theory.py # Why normalization helps
```

Key insight: Normalization keeps activations in well-conditioned ranges, **stabilizing training**.

### 4. Generalization
```python
# src: generalization/
├── pac_learning.py         # PAC bounds on generalization
├── rademacher_complexity.py # Empirical Rademacher complexity
├── double_descent.py       # The double descent phenomenon
└── implicit_regularization.py # Why overparameterized nets generalize
```

Key insight: **Overparameterized networks generalize well** due to implicit regularization — a surprising empirical discovery.

### 5. Information Theory
```python
# src: information_theory/
├── entropy.py              # Shannon entropy
├── kl_divergence.py        # KL divergence & divergence measures
├── mutual_information.py   # MI between random variables
├── variational_bounds.py   # ELBO for VAEs
└── information_bottleneck.py # Information theoretic perspective
```

Key insight: Deep learning can be viewed through an **information-theoretic lens** (compression + prediction).

## 🚀 Quick Start

```python
# Backprop from scratch
from backpropagation import ScalarGraph

x = ScalarGraph(3.0)
y = ScalarGraph(2.0)
z = x * y + x ** 2

z.backward()  # Compute dz/dx, dz/dy
print(f"dz/dx = {x.grad}, dz/dy = {y.grad}")  # dz/dx=5.0, dz/dy=3.0

# Adam optimizer
from optimization import Adam

model = SimpleNet()
optimizer = Adam(lr=0.001, beta1=0.9, beta2=0.999)

for epoch in range(100):
    loss = model(X_train)
    loss.backward()
    optimizer.step(model)  # Update with adaptive learning rates

# Batch normalization
from normalization import BatchNorm

bn = BatchNorm(num_features=64)
x = torch.randn(32, 64)  # batch_size=32, features=64

normalized = bn(x)  # Normalize to mean≈0, std≈1
```

## 📚 Mathematical Highlights

### Backpropagation Chain Rule
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w}$$

### Adam Update Rule
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad \text{(first moment)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \quad \text{(second moment)}$$
$$\theta_t = \theta_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}$$

### Batch Norm
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \quad \text{(normalize within batch)}$$
$$y_i = \gamma \hat{x}_i + \beta \quad \text{(learned rescaling)}$$

### Double Descent
When models are **very overparameterized** (parameter count >> training examples), test error can **decrease again** — the "double descent" phenomenon (Belkin et al., 2019).

---

## 🧪 Tests & Verification

```bash
pytest tests/ -v
```

Each module includes tests verifying:
- ✅ Numerical correctness (matches NumPy/PyTorch)
- ✅ Gradient computation
- ✅ Convergence properties
- ✅ Edge cases

---

## 📖 Educational Purpose

This is **not** meant to replace PyTorch. It's meant to:
1. **Clarify understanding** of DL fundamentals
2. **Show derivations** — not just formulas
3. **Serve as reference** for interviews, teaching, research

---

## 🎓 Use Cases

- **Job interviews**: "Explain backprop from first principles" ← you have the code
- **Teaching**: Show students where formulas come from
- **Research**: Verify theoretical claims with simple implementations
- **Publications**: Reference correct mathematical formulations

---

**Made for deep understanding, not production use. See PyTorch/JAX for production code.**
