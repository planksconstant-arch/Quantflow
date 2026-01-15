"""
Tests for Neural SDE Module
============================
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.neural_sde.neural_sde import NeuralSDE
from models.neural_sde.utils import generate_brownian_paths, generate_gbm_paths
from models.neural_sde.trainer import NeuralSDETrainer


class TestNeuralSDE:
    """Test suite for Neural SDE"""
    
    def test_initialization(self):
        """Test Neural SDE can be initialized"""
        sde = NeuralSDE(state_size=1, noise_size=1)
        assert sde is not None
        assert sde.state_size == 1
        assert sde.noise_size == 1
   
    def test_forward_pass(self):
        """Test forward pass generates paths"""
        sde = NeuralSDE(state_size=1, noise_size=1, hidden_size=16)
        
        x0 = torch.tensor([[100.0]])
        ts = torch.linspace(0, 1.0, 50)
        
        paths = sde(x0, ts)
        
        assert paths.shape == (1, 50, 1)
        assert torch.isfinite(paths).all()
    
    def test_sample_paths(self):
        """Test batch path generation"""
        sde = NeuralSDE(state_size=1, noise_size=1, hidden_size=16)
        
        paths = sde.sample_paths(x0=100.0, num_paths=10, num_steps=50)
        
        assert paths.shape == (10, 50, 1)
        assert paths[:, 0, 0].mean().item() == pytest.approx(100.0, abs=1e-5)
    
    def test_determinism(self):
        """Test paths are different (stochastic)"""
        sde = NeuralSDE(state_size=1, noise_size=1, hidden_size=16)
        
        paths1 = sde.sample_paths(x0=100.0, num_paths=2, num_steps=50)
        paths2 = sde.sample_paths(x0=100.0, num_paths=2, num_steps=50)
        
        # Paths should be different (stochastic)
        assert not torch.allclose(paths1, paths2)


class TestUtils:
    """Test suite for utility functions"""
    
    def test_brownian_motion(self):
        """Test Brownian motion generation"""
        W, t = generate_brownian_paths(num_paths=1000, num_steps=100)
        
        assert W.shape == (1000, 100, 1)
        assert t.shape == (100,)
        
        # Brownian motion properties
        assert W[:, 0, :].abs().max() < 1e-6  # Starts at zero
        assert W.mean().abs() < 0.1  # Mean ~0
        assert 0.8 < W.std() < 1.2  # Std ~sqrt(T)
    
    def test_gbm_paths(self):
        """Test GBM path generation"""
        S = generate_gbm_paths(S0=100, mu=0.0, sigma=0.2, T=1.0,
                               num_paths=1000, num_steps=252)
        
        assert S.shape == (1000, 252, 1)
        assert S[:, 0, 0].mean().item() == pytest.approx(100.0, abs=1e-5)
        assert (S > 0).all()  # Prices always positive


class TestTrainer:
    """Test suite for Neural SDE Trainer"""
    
    def test_trainer_initialization(self):
        """Test trainer can be created"""
        historical = generate_gbm_paths(S0=100, mu=0.05, sigma=0.2, T=1.0,
                                        num_paths=100, num_steps=50, device='cpu')
        
        sde = NeuralSDE(state_size=1, noise_size=1, hidden_size=16)
        trainer = NeuralSDETrainer(sde, historical, device='cpu')
        
        assert trainer is not None
        assert trainer.sde is not None
        assert trainer.discriminator is not None
    
    def test_training_loop(self):
        """Test training runs without errors"""
        torch.manual_seed(42)
        
        historical = generate_gbm_paths(S0=100, mu=0.05, sigma=0.2, T=1.0,
                                        num_paths=50, num_steps=30, device='cpu')
        
        sde = NeuralSDE(state_size=1, noise_size=1, hidden_size=16, num_layers=2)
        trainer = NeuralSDETrainer(sde, historical, device='cpu')
        
        # Train for few epochs
        trainer.train(epochs=5, batch_size=16, verbose=False)
        
        assert len(trainer.history['d_loss']) == 5
        assert len(trainer.history['g_loss']) == 5
    
    def test_path_generation_after_training(self):
        """Test can generate paths after training"""
        torch.manual_seed(42)
        
        historical = generate_gbm_paths(S0=100, mu=0.05, sigma=0.2, T=1.0,
                                        num_paths=50, num_steps=30, device='cpu')
        
        sde = NeuralSDE(state_size=1, noise_size=1, hidden_size=16, num_layers=2)
        trainer = NeuralSDETrainer(sde, historical, device='cpu')
        trainer.train(epochs=2, batch_size=16, verbose=False)
        
        # Generate paths
        generated = trainer.generate_paths(num_paths=10, S0=100)
        
        assert generated.shape == (10, 30, 1)
        assert (generated > 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
