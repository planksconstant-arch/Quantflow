# Neural SDEs in QuantFlow

## Overview

Neural Stochastic Differential Equations (Neural SDEs) replace traditional parametric models (Black-Scholes, Heston) with **learnable dynamics**. Instead of assuming market behavior follows a specific equation, we train neural networks to learn the drift and diffusion directly from historical data.

## Mathematical Framework

### Standard SDE
A classical SDE for asset price $S_t$:
$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

### Neural SDE
We replace the constants $\mu$ and $\sigma$ with neural networks:
$$dS_t = f_\theta(t, S_t) dt + g_\phi(t, S_t) dW_t$$

Where:
- $f_\theta$: Drift network (learned)
- $g_\phi$: Diffusion network (learned)
- $\theta, \phi$: Neural network weights

## Architecture

### Drift Network
```python
Input: [time, price] → MLP(3 layers, 64 hidden) → Output: drift
```

### Diffusion Network
```python
Input: [time, price] → MLP(3 layers, 64 hidden) → Softplus → Output: diffusion
```
*Note: Softplus ensures positive diffusion*

### Training (WGAN-GP)

1. **Generator**: Neural SDE produces synthetic price paths
2. **Discriminator**: Neural network classifies real vs. fake paths
3. **Adversarial Training**: Generator learns to fool discriminator
4. **Loss**: Wasserstein distance with gradient penalty

## Usage

### Training a Neural SDE

```python
from models.neural_sde import NeuralSDE, NeuralSDETrainer
from models.neural_sde.utils import generate_gbm_paths

# Generate or load historical paths
historical_paths = ...  # (num_samples, num_steps, 1)

# Create model
sde = NeuralSDE(state_size=1, noise_size=1, hidden_size=64)

# Train
trainer = NeuralSDETrainer(sde, historical_paths, device='cuda')
trainer.train(epochs=1000)

# Generate synthetic paths
synthetic_paths = trainer.generate_paths(num_paths=10000)
```

### Option Pricing

```python
from examples.neural_sde_pricing import price_option_with_neural_sde

# Price option using trained SDE
option_price = price_option_with_neural_sde(
    sde=trained_sde,
    S0=100,
    K=105,
    T=0.25,
    r=0.05,
    option_type='call',
    num_sims=10000
)
```

## Advantages Over Classical Models

| Feature | Black-Scholes | Neural SDE |
|---------|---------------|------------|
| Assumptions | Log-normal returns | **Data-driven** |
| Volatility | Constant | **Time-varying & path-dependent** |
| Jumps | No | **Learned automatically** |
| Regimes | No | **Captured via training** |
| Calibration | Find σ | **End-to-end learning** |

## GPU Acceleration

Neural SDEs leverage PyTorch for GPU training:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = NeuralSDETrainer(sde, data, device=device)
```

On CUDA GPUs, training is **10-50x faster** than CPU.

## References

- Chen et al. (2018): "Neural Ordinary Differential Equations"
- Kidger et al. (2020): "Neural Controlled Differential Equations"
- Li et al. (2020): "Scalable Gradients for Stochastic Differential Equations"

## Files

- `models/neural_sde/neural_sde.py`: Core implementation
- `models/neural_sde/trainer.py`: GAN training loop
- `models/neural_sde/utils.py`: Brownian motion utilities
- `examples/neural_sde_pricing.py`: Complete usage example
