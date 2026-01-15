"""
Deep Hedging Policy Networks
=============================

LSTM-based policy for dynamic hedging with partial observability.
Compatible with Stable-Baselines3.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class LSTMExtractor(BaseFeaturesExtractor):
    """
    LSTM Feature Extractor for policies
    
    Maintains hidden state across observations to capture market regimes.
    Compatible with SB3's ActorCriticPolicy.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64, lstm_hidden_size: int =32):
        super().__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0]
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=n_input,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Post-LSTM processing
        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.ReLU()
        )
        
        # Hidden state (will be reset between episodes)
        self.lstm_hidden = None
    
    def forward(self, observations):
        """
        Args:
            observations: (batch_size, n_features)
        Returns:
            features: (batch_size, features_dim)
        """
        batch_size = observations.shape[0]
        
        # Add sequence dimension
        obs_seq = observations.unsqueeze(1)  # (batch, 1, features)
        
        # Initialize hidden if needed
        if self.lstm_hidden is None or self.lstm_hidden[0].shape[1] != batch_size:
            device = observations.device
            self.lstm_hidden = (
                torch.zeros(1, batch_size, self.lstm.hidden_size, device=device),
                torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
            )
        
        # LSTM forward
        lstm_out, self.lstm_hidden = self.lstm(obs_seq, self.lstm_hidden)
        
        # Detach hidden state (don't backprop through time across batches)
        self.lstm_hidden = (self.lstm_hidden[0].detach(), self.lstm_hidden[1].detach())
        
        # Extract last output
        features = lstm_out[:, -1, :]
        
        # Post-process
        return self.linear(features)
    
    def reset_hidden(self):
        """Reset LSTM hidden state (call at episode start)"""
        self.lstm_hidden = None


class MLPExtractor(BaseFeaturesExtractor):
    """
    Simple MLP extractor (non-recurrent baseline)
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0]
        
        self.net = nn.Sequential(
            nn.Linear(n_input, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        return self.net(observations)


if __name__ == "__main__":
    print("Testing Policy Networks...")
    
    # Create mock observation space
    obs_space = spaces.Box(low=-10, high=10, shape=(3,), dtype=float)
    
    # LSTM extractor
    lstm_extractor = LSTMExtractor(obs_space, features_dim=32, lstm_hidden_size=16)
    
    # Test forward pass
    obs = torch.randn(4, 3)  # Batch of 4 observations
    features = lstm_extractor(obs)
    
    print(f"LSTM features shape: {features.shape}")
    assert features.shape == (4, 32)
    
    # MLP extractor
    mlp_extractor = MLPExtractor(obs_space, features_dim=32)
    features_mlp = mlp_extractor(obs)
    
    print(f"MLP features shape: {features_mlp.shape}")
    assert features_mlp.shape == (4, 32)
    
    print("✓ Policy network tests passed")
