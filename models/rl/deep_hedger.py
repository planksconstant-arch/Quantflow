"""
Deep Hedging Agent Trainer
===========================

Trains PPO agent for dynamic option hedging.
GPU-accelerated, uses Stable-Baselines3.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from analysis.environments.hedging_env import OptionHedgingEnv
from models.rl.policy_networks import LSTMExtractor
from models.neural_sde import NeuralSDE


class DeepHedger:
    """
    Deep Reinforcement Learning Hedger
    
    Learns optimal hedging strategy via PPO with LSTM policy.
    """
    
    def __init__(self, 
                 S0=100,
                 K=100,
                 T=0.25,
                 r=0.05,
                 sigma=0.2,
                 option_type='call',
                 num_steps=50,
                 cost_bps=5.0,
                 neural_sde=None,
                 device='auto'):
        """
        Args:
            S0, K, T, r, sigma: Option parameters
            option_type: 'call' or 'put'
            num_steps: Rebalancing frequency
            cost_bps: Transaction cost (basis points)
            neural_sde: Optional trained Neural SDE
            device: 'cpu', 'cuda', or 'auto'
        """
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create environment
        def make_env():
            return OptionHedgingEnv(
                S0=S0, K=K, T=T, r=r, sigma=sigma,
                option_type=option_type,
                num_steps=num_steps,
                cost_bps=cost_bps,
                neural_sde=neural_sde
            )
        
        self.env = DummyVecEnv([make_env])
        self.eval_env = DummyVecEnv([make_env])
        
        # Policy kwargs
        policy_kwargs = dict(
            features_extractor_class=LSTMExtractor,
            features_extractor_kwargs=dict(features_dim=64, lstm_hidden_size=32),
            net_arch=[dict(pi=[64, 64], vf=[64, 64])]
        )
        
        # Create PPO agent
        self.model = PPO(
            "MlpPolicy",
            self.env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=1,
            device=self.device,
            tensorboard_log="./quantflow_tensorboard/"
        )
    
    def train(self, total_timesteps=100000, eval_freq=10000):
        """
        Train the hedging agent
        
        Args:
            total_timesteps: Number of environment steps
            eval_freq: Evaluation frequency
        """
        print(f"Training Deep Hedger on {self.device}...")
        print(f"Total timesteps: {total_timesteps}")
        
        # Evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path='./models/rl/checkpoints/',
            log_path='./models/rl/logs/',
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback
        )
        
        print("✓ Training complete!")
    
    def evaluate(self, num_episodes=100):
        """
        Evaluate trained agent
        
        Returns:
            mean_pnl, std_pnl, sharpe_ratio
        """
        pnls = []
        
        for _ in range(num_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_pnl = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                
                if done:
                    episode_pnl = info[0]['pnl']
            
            pnls.append(episode_pnl)
        
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        sharpe = mean_pnl / (std_pnl + 1e-8)
        
        return mean_pnl, std_pnl, sharpe
    
    def hedge(self, S, time_remaining, current_position):
        """
        Get hedge action for current state
        
        Args:
            S: Current stock price
            time_remaining: Time to expiration
            current_position: Current inventory
        
        Returns:
            action: Recommended hedge ratio
        """
        # Construct state
        log_moneyness = np.log(S / self.env.envs[0].K)
        time_frac = time_remaining / self.env.envs[0].T
        state = np.array([log_moneyness, time_frac, current_position], dtype=np.float32)
        
        # Get action
        action, _ = self.model.predict(state.reshape(1, -1), deterministic=True)
        
        return float(action[0])
    
    def save(self, path="./models/rl/deep_hedger.zip"):
        """Save trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path="./models/rl/deep_hedger.zip"):
        """Load trained model"""
        self.model = PPO.load(path, env=self.env, device=self.device)
        print(f"Model loaded from {path}")


def compare_with_delta_hedge(hedger, num_episodes=100):
    """
    Benchmark Deep Hedger against classical delta hedging
    """
    from models.pricing.black_scholes import BlackScholesModel
    
    rl_pnls = []
    delta_pnls = []
    
    env_params = hedger.env.envs[0]
    
    for _ in range(num_episodes):
        # RL Agent
        obs = hedger.eval_env.reset()
        done = False
        while not done:
            action, _ = hedger.model.predict(obs, deterministic=True)
            obs, _, done, info = hedger.eval_env.step(action)
            if done:
                rl_pnls.append(info[0]['pnl'])
        
        # Delta Hedging
        env = OptionHedgingEnv(
            S0=env_params.S0, K=env_params.K, T=env_params.T,
            r=env_params.r, sigma=env_params.sigma,
            option_type=env_params.option_type,
            num_steps=env_params.num_steps,
            cost_bps=env_params.cost_bps
        )
        
        obs, _ = env.reset()
        done = False
        while not done:
            # Compute Black-Scholes delta
            bs = BlackScholesModel(env.S, env.K, env.time_remaining, env.r, env.sigma)
            delta = bs.delta(env.option_type)
            
            action = np.array([delta])
            obs, _, done, info = env.step(action)
            if done:
                delta_pnls.append(info['pnl'])
    
    # Statistics
    print("\n" + "="*60)
    print("Hedging Strategy Comparison")
    print("="*60)
    
    print(f"\nDeep RL Hedger:")
    print(f"  Mean PnL: ${np.mean(rl_pnls):.2f}")
    print(f"  Std PnL: ${np.std(rl_pnls):.2f}")
    print(f"  Sharpe: {np.mean(rl_pnls) / (np.std(rl_pnls) + 1e-8):.3f}")
    
    print(f"\nDelta Hedging:")
    print(f"  Mean PnL: ${np.mean(delta_pnls):.2f}")
    print(f"  Std PnL: ${np.std(delta_pnls):.2f}")
    print(f"  Sharpe: {np.mean(delta_pnls) / (np.std(delta_pnls) + 1e-8):.3f}")
    
    improvement = (np.mean(rl_pnls) - np.mean(delta_pnls)) / abs(np.mean(delta_pnls)) * 100
    print(f"\nImprovement: {improvement:+.1f}%")
    
    return rl_pnls, delta_pnls


if __name__ == "__main__":
    print("Training Deep Hedging Agent...")
    
    # Create hedger
    hedger = DeepHedger(
        S0=100, K=100, T=0.25, r=0.05, sigma=0.2,
        option_type='call', num_steps=20, cost_bps=5.0,
        device='auto'
    )
    
    # Train (demo: short run)
    hedger.train(total_timesteps=10000, eval_freq=5000)
    
    # Evaluate
    mean_pnl, std_pnl, sharpe = hedger.evaluate(num_episodes=50)
    print(f"\nEvaluation Results:")
    print(f"  Mean PnL: ${mean_pnl:.2f}")
    print(f"  Std PnL: ${std_pnl:.2f}")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    
    # Compare with delta hedging
    compare_with_delta_hedge(hedger, num_episodes=50)
    
    print("\n✓ Deep Hedger demo complete!")
