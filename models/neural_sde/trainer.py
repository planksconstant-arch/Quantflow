"""
Neural SDE GAN Trainer
======================

Trains Neural SDEs using Generative Adversarial Networks
to match historical market path distributions.

Architecture:
    - Generator: Neural SDE
    - Discriminator: Path Classifier (CNN or MLP)
    - Loss: Wasserstein GAN with Gradient Penalty
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .neural_sde import NeuralSDE
from .utils import compute_path_signatures, generate_gbm_paths


class PathDiscriminator(nn.Module):
    """
    Discriminates between real and generated price paths
    
    Uses path signatures (statistical features) for classification
    """
    
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_size // 2, 1)  # Single output (real/fake score)
        )
    
    def forward(self, paths):
        """
        Args:
            paths: (batch_size, num_steps, state_size)
        Returns:
            Scores (batch_size, 1)
        """
        # Extract features
        features = compute_path_signatures(paths)
        return self.net(features)


class NeuralSDETrainer:
    """
    Trainer for Neural SDEs using WGAN-GP
    
    Usage:
        >>> trainer = NeuralSDETrainer(sde, historical_paths)
        >>> trainer.train(epochs=1000)
        >>> trained_sde = trainer.sde
    """
    
    def __init__(self, sde, historical_paths, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            sde: NeuralSDE instance
            historical_paths: Real market paths (num_samples, num_steps, state_size)
            device: 'cpu' or 'cuda'
        """
        self.sde = sde.to(device)
        self.device = device
        
        # Convert historical paths to tensor
        if isinstance(historical_paths, np.ndarray):
            historical_paths = torch.from_numpy(historical_paths).float()
        
        self.real_paths = historical_paths.to(device)
        self.num_steps = historical_paths.shape[1]
        self.T = 1.0  # Normalized time
        
        # Create discriminator
        # Input size = number of signature features
        sample_sig = compute_path_signatures(self.real_paths[:1])
        sig_size = sample_sig.shape[1]
        
        self.discriminator = PathDiscriminator(sig_size).to(device)
        
        # Optimizers
        self.optimizer_G = optim.Adam(self.sde.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        
        # Training history
        self.history = {
            'd_loss': [],
            'g_loss': [],
            'wasserstein_distance': []
        }
    
    def gradient_penalty(self, real_paths, fake_paths):
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_paths.shape[0]
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, device=self.device)
        interpolated = alpha * real_paths + (1 - alpha) * fake_paths
        interpolated.requires_grad_(True)
        
        # Discriminator output
        d_interpolated = self.discriminator(interpolated)
        
        # Gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty
    
    def train_discriminator(self, real_batch, fake_batch, lambda_gp=10):
        """Train discriminator for one step"""
        self.discriminator.train()
        self.optimizer_D.zero_grad()
        
        # Scores
        real_score = self.discriminator(real_batch)
        fake_score = self.discriminator(fake_batch.detach())
        
        # Wasserstein loss
        d_loss = fake_score.mean() - real_score.mean()
        
        # Gradient penalty
        gp = self.gradient_penalty(real_batch, fake_batch)
        
        # Total loss
        total_loss = d_loss + lambda_gp * gp
        total_loss.backward()
        self.optimizer_D.step()
        
        return d_loss.item(), -d_loss.item()  # Return (loss, Wasserstein distance)
    
    def train_generator(self, batch_size):
        """Train generator (Neural SDE) for one step"""
        self.sde.train()
        self.optimizer_G.zero_grad()
        
        # Generate fake paths
        x0 = torch.tensor([[self.real_paths[0, 0, 0].item()]] * batch_size, device=self.device)
        ts = torch.linspace(0, self.T, self.num_steps, device=self.device)
        
        fake_paths = self.sde(x0, ts)
        
        # Generator loss: fool the discriminator
        fake_score = self.discriminator(fake_paths)
        g_loss = -fake_score.mean()
        
        g_loss.backward()
        self.optimizer_G.step()
        
        return g_loss.item()
    
    def train(self, epochs=1000, batch_size=64, n_critic=5, verbose=True):
        """
        Train the Neural SDE
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            n_critic: Number of discriminator updates per generator update
            verbose: Print progress
        """
        # Create dataloader
        dataset = TensorDataset(self.real_paths)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        for epoch in range(epochs):
            epoch_d_loss = 0
            epoch_g_loss = 0
            epoch_w_dist = 0
            num_batches = 0
            
            for real_batch, in dataloader:
                # Train Discriminator
                x0 = torch.tensor([[real_batch[0, 0, 0].item()]] * batch_size, device=self.device)
                ts = torch.linspace(0, self.T, self.num_steps, device=self.device)
                
                with torch.no_grad():
                    fake_batch = self.sde(x0, ts)
                
                d_loss, w_dist = self.train_discriminator(real_batch, fake_batch)
                epoch_d_loss += d_loss
                epoch_w_dist += w_dist
                
                # Train Generator (every n_critic steps)
                if num_batches % n_critic == 0:
                    g_loss = self.train_generator(batch_size)
                    epoch_g_loss += g_loss
                
                num_batches += 1
            
            # Record history
            self.history['d_loss'].append(epoch_d_loss / num_batches)
            self.history['g_loss'].append(epoch_g_loss / (num_batches // n_critic))
            self.history['wasserstein_distance'].append(epoch_w_dist / num_batches)
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"D Loss: {self.history['d_loss'][-1]:.4f} | "
                      f"G Loss: {self.history['g_loss'][-1]:.4f} | "
                      f"W-Dist: {self.history['wasserstein_distance'][-1]:.4f}")
    
    def generate_paths(self, num_paths, S0=None):
        """Generate synthetic paths after training"""
        self.sde.eval()
        
        if S0 is None:
            S0 = self.real_paths[0, 0, 0].item()
        
        with torch.no_grad():
            paths = self.sde.sample_paths(S0, num_paths, self.num_steps, self.T, self.device)
        
        return paths.cpu().numpy()


if __name__ == "__main__":
    print("Testing Neural SDE Trainer...")
    
    # Generate synthetic "historical" data (GBM)
    np.random.seed(42)
    torch.manual_seed(42)
    
    historical = generate_gbm_paths(
        S0=100, mu=0.05, sigma=0.2, T=1.0,
        num_paths=500, num_steps=100, device='cpu'
    )
    
    print(f"Historical paths shape: {historical.shape}")
    
    # Create and train Neural SDE
    sde = NeuralSDE(state_size=1, noise_size=1, hidden_size=32, num_layers=2)
    trainer = NeuralSDETrainer(sde, historical, device='cpu')
    
    print("\nTraining Neural SDE (demo: 10 epochs)...")
    trainer.train(epochs=10, batch_size=32, verbose=True)
    
    # Generate paths
    generated = trainer.generate_paths(num_paths=100)
    print(f"\nGenerated paths shape: {generated.shape}")
    
    print("✓ Trainer test passed")
