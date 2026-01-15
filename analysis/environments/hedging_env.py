"""
Option Hedging Environment (Gymnasium)
=======================================

Reinforcement Learning environment for learning optimal hedging strategies
under transaction costs and realistic market dynamics.

State: [Moneyness, Time-to-Expiry, IV, Current Position]
Action: Hedge ratio (continuous)
Reward: Negative of risk measure (entropic or CVaR)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch


class OptionHedgingEnv(gym.Env):
    """
    Gym environment for option hedging with transaction costs
    
    The agent learns to dynamically hedge an option position by trading
    the underlying asset, balancing hedging error against transaction costs.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 S0=100.0,
                 K=100.0,
                 T=0.25,
                 r=0.05,
                 sigma=0.2,
                 option_type='call',
                 num_steps=50,
                 dt=None,
                 cost_bps=5.0,
                 neural_sde=None,
                 risk_aversion=0.1):
        """
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (used if no neural_sde)
            option_type: 'call' or 'put'
            num_steps: Number of rebalancing steps
            dt: Time step (auto-computed if None)
            cost_bps: Transaction cost (basis points)
            neural_sde: Optional Neural SDE for market simulation
            risk_aversion: Risk aversion parameter for entropic risk
        """
        super().__init__()
        
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.num_steps = num_steps
        self.dt = dt if dt is not None else T / num_steps
        self.cost_bps = cost_bps
        self.cost_rate = cost_bps / 10000.0
        self.risk_aversion = risk_aversion
        
        # Market model
        self.neural_sde = neural_sde
        self.use_neural = neural_sde is not None
        
        # State space: [log_moneyness, time_remaining, position]
        self.observation_space = spaces.Box(
            low=np.array([-2.0, 0.0, -10.0]),
            high=np.array([2.0, 1.0, 10.0]),
            dtype=np.float32
        )
        
        # Action space: hedge ratio (number of shares, normalized)
        self.action_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Episode state
        self.reset()
    
    def _simulate_price_step(self, S_current):
        """Simulate one time step of stock price"""
        if self.use_neural:
            # Use Neural SDE
            with torch.no_grad():
                x0 = torch.tensor([[S_current]], dtype=torch.float32)
                ts = torch.tensor([0.0, self.dt])
                path = self.neural_sde(x0, ts)
                S_next = path[0, -1, 0].item()
        else:
            # Geometric Brownian Motion
            dW = np.random.normal(0, np.sqrt(self.dt))
            dS = self.r * S_current * self.dt + self.sigma * S_current * dW
            S_next = S_current + dS
        
        return max(S_next, 0.01)  # Prevent negative prices
    
    def _option_payoff(self, S_T):
        """Calculate option payoff at expiration"""
        if self.option_type == 'call':
            return max(S_T - self.K, 0)
        else:
            return max(self.K - S_T, 0)
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.S = self.S0
        self.time_remaining = self.T
        self.position = 0.0  # Shares of underlying held
        self.pnl = 0.0
        self.option_sold = True  # Short 1 option
        
        return self._get_state(), {}
    
    def _get_state(self):
        """Get current state observation"""
        log_moneyness = np.log(self.S / self.K)
        time_frac = self.time_remaining / self.T
        
        return np.array([
            log_moneyness,
            time_frac,
            self.position
        ], dtype=np.float32)
    
    def step(self, action):
        """
        Execute one hedging step
        
        Args:
            action: Desired hedge ratio (shares to hold)
        
        Returns:
            state, reward, done, truncated, info
        """
        # Parse action
        target_position = float(action[0])
        
        # Compute trade
        trade_size = target_position - self.position
        
        # Transaction cost
        cost = self.cost_rate * abs(trade_size) * self.S
        
        # Update position and PnL
        self.position = target_position
        self.pnl -= cost
        
        # Simulate market
        S_old = self.S
        self.S = self._simulate_price_step(self.S)
        
        # Update time
        self.current_step += 1
        self.time_remaining -= self.dt
        
        # Mark-to-market PnL from holding position
        delta_S = self.S - S_old
        mtm_pnl = self.position * delta_S
        self.pnl += mtm_pnl
        
        # Check if terminal
        done = self.current_step >= self.num_steps
        
        if done:
            # Settle option at expiration
            option_payoff = self._option_payoff(self.S)
            self.pnl -= option_payoff  # We're short the option
            
            # Reward: negative of entropic risk measure
            # For simplicity, use negative squared PnL (quadratic utility)
            reward = -self.pnl ** 2 * self.risk_aversion
        else:
            # Intermediate reward: penalize costs
            reward = -cost
        
        return self._get_state(), reward, done, False, {
            'pnl': self.pnl,
            'S': self.S,
            'position': self.position,
            'cost': cost
        }
    
    def render(self, mode='human'):
        """Render current state"""
        print(f"Step {self.current_step}/{self.num_steps}")
        print(f"  S=${self.S:.2f}, Position={self.position:.2f}, PnL=${self.pnl:.2f}")


if __name__ == "__main__":
    print("Testing Option Hedging Environment...")
    
    # Create environment
    env = OptionHedgingEnv(
        S0=100, K=100, T=0.25, r=0.05, sigma=0.2,
        option_type='call', num_steps=20, cost_bps=5.0
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test episode
    state, _ = env.reset()
    print(f"\nInitial state: {state}")
    
    total_reward = 0
    for i in range(5):
        # Random action (for testing)
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\nStep {i+1}:")
        print(f"  Action: {action[0]:.2f}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Info: {info}")
        
        if done:
            break
    
    print(f"\nEpisode complete. Total reward: {total_reward:.4f}")
    print("✓ Environment test passed")
