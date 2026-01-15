"""
Neural SDE Core Implementation
===============================

Implements Neural Stochastic Differential Equations using PyTorch and torchsde.

Mathematical Form:
    dX_t = f_θ(t, X_t) dt + g_φ(t, X_t) dW_t

Where:
    - f_θ: Drift network (MLP)
    - g_φ: Diffusion network (MLP)
    - W_t: Brownian motion
"""

import torch
import torch.nn as nn
import torchsde


class DriftNet(nn.Module):
    """Neural network for drift function f_θ(t, X_t)"""
    
    def __init__(self, state_size, hidden_size=64, num_layers=3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(state_size + 1, hidden_size))  # +1 for time
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_size, state_size))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, t, x):
        """
        Args:
            t: Time tensor (batch_size,)
            x: State tensor (batch_size, state_size)
        Returns:
            Drift (batch_size, state_size)
        """
        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        t_expanded = t.unsqueeze(-1)
        tx = torch.cat([t_expanded, x], dim=-1)
        return self.net(tx)


class DiffusionNet(nn.Module):
    """Neural network for diffusion function g_φ(t, X_t)"""
    
    def __init__(self, state_size, noise_size, hidden_size=64, num_layers=3):
        super().__init__()
        
        self.state_size = state_size
        self.noise_size = noise_size
        
        layers = []
        layers.append(nn.Linear(state_size + 1, hidden_size))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        
        # Output: state_size x noise_size matrix flattened
        layers.append(nn.Linear(hidden_size, state_size * noise_size))
        layers.append(nn.Softplus())  # Ensure positive diffusion
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, t, x):
        """
        Args:
            t: Time tensor (batch_size,)
            x: State tensor (batch_size, state_size)
        Returns:
            Diffusion matrix (batch_size, state_size, noise_size)
        """
        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        t_expanded = t.unsqueeze(-1)
        tx = torch.cat([t_expanded, x], dim=-1)
        
        flat_diffusion = self.net(tx)
        batch_size = x.shape[0]
        
        return flat_diffusion.view(batch_size, self.state_size, self.noise_size)


class NeuralSDE(nn.Module):
    """
    Neural Stochastic Differential Equation
    
    Learns market dynamics from historical data using neural networks
    for both drift and diffusion components.
    
    Usage:
        >>> sde = NeuralSDE(state_size=1, noise_size=1)
        >>> x0 = torch.tensor([[100.0]])  # Initial price
        >>> ts = torch.linspace(0, 1, 100)  # Time grid
        >>> paths = sde(x0, ts)  # Generate price paths
    """
    
    noise_type = "diagonal"  # Diagonal noise (uncorrelated Brownian motions)
    sde_type = "ito"  # Ito interpretation
    
    def __init__(self, state_size=1, noise_size=1, hidden_size=64, num_layers=3):
        super().__init__()
        
        self.state_size = state_size
        self.noise_size = noise_size
        
        # Neural networks for drift and diffusion
        self.drift_net = DriftNet(state_size, hidden_size, num_layers)
        self.diffusion_net = DiffusionNet(state_size, noise_size, hidden_size, num_layers)
    
    def f(self, t, x):
        """Drift function"""
        return self.drift_net(t, x)
    
    def g(self, t, x):
        """Diffusion function"""
        return self.diffusion_net(t, x)
    
    def forward(self, x0, ts, dt=1e-3, method='euler'):
        """
        Solve the Neural SDE from x0 over time steps ts
        
        Args:
            x0: Initial state (batch_size, state_size)
            ts: Time tensor (num_steps,)
            dt: Time step for numerical integration
            method: 'euler' or 'heun' (Heun is more accurate)
        
        Returns:
            Simulated paths (batch_size, num_steps, state_size)
        """
        # Use torchsde for efficient SDE integration
        xs = torchsde.sdeint(self, x0, ts, dt=dt, method=method)
        
        # Transpose to (batch_size, time, state_size)
        return xs.transpose(0, 1)
    
    def sample_paths(self, x0, num_paths, num_steps, T=1.0, device='cpu'):
        """
        Generate multiple sample paths for Monte Carlo pricing
        
        Args:
            x0: Initial value (scalar or tensor)
            num_paths: Number of paths to simulate
            num_steps: Number of time steps
            T: Terminal time
            device: 'cpu' or 'cuda'
        
        Returns:
            Paths tensor (num_paths, num_steps, state_size)
        """
        if isinstance(x0, (int, float)):
            x0 = torch.tensor([[x0]] * num_paths, dtype=torch.float32, device=device)
        elif x0.dim() == 1:
            x0 = x0.unsqueeze(0).repeat(num_paths, 1).to(device)
        
        ts = torch.linspace(0, T, num_steps, device=device)
        
        with torch.no_grad():
            paths = self.forward(x0, ts)
        
        return paths


if __name__ == "__main__":
    # Test Neural SDE
    print("Testing Neural SDE...")
    
    # Create model
    sde = NeuralSDE(state_size=1, noise_size=1, hidden_size=32)
    
    # Initial condition: S0 = 100
    x0 = torch.tensor([[100.0]])
    
    # Time grid: 0 to 1 year, 252 steps (daily)
    ts = torch.linspace(0, 1.0, 252)
    
    # Generate path
    paths = sde(x0, ts)
    
    print(f"Generated path shape: {paths.shape}")
    print(f"Initial value: {paths[0, 0, 0]:.2f}")
    print(f"Final value: {paths[0, -1, 0]:.2f}")
    print("✓ Neural SDE test passed")
