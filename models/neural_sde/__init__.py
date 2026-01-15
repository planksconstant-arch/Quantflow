"""
Neural Stochastic Differential Equations Module
================================================

Implements learnable stochastic processes using neural networks.
Replaces parametric models (Heston, GBM) with data-driven dynamics.
"""

from .neural_sde import NeuralSDE
from .trainer import NeuralSDETrainer
from .utils import generate_brownian_paths

__all__ = ['NeuralSDE', 'NeuralSDETrainer', 'generate_brownian_paths']
