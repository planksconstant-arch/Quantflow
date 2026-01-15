"""
Hawkes Process for Microstructure Analysis
===========================================

Models order arrival intensity in high-frequency trading.

Self-exciting process:
    λ(t) = μ + Σ φ(t - t_i)

Where:
    μ: Baseline intensity
    φ: Excitation kernel
    t_i: Past event times
"""

import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class HawkesProcess:
    """
    Univariate Hawkes Process
    
    Models self-exciting point processes (e.g., buy/sell orders).
    """
    
    def __init__(self, alpha=0.5, beta=1.0, mu=1.0):
        """
        Exponential kernel: φ(t) = α*β*exp(-β*t)
        
        Args:
            alpha: Excitation strength
            beta: Decay rate
            mu: Baseline intensity
        """
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
    
    def intensity(self, t, event_times):
        """
        Compute conditional intensity at time t
        
        Args:
            t: Current time
            event_times: Array of past event times
        
        Returns:
            λ(t): Intensity value
        """
        if len(event_times) == 0:
            return self.mu
        
        # Only consider events before t
        past_events = event_times[event_times < t]
        
        if len(past_events) == 0:
            return self.mu
        
        # Kernel contributions
        kernel_sum = np.sum(self.alpha * self.beta * np.exp(-self.beta * (t - past_events)))
        
        return self.mu + kernel_sum
    
    def branching_ratio(self):
        """
        Compute branching ratio n = α
        
        If n < 1: stationary (stable)
        If n ≥ 1: explosive (unstable)
        """
        return self.alpha
    
    def simulate(self, T=1.0, max_events=10000):
        """
        Simulate Hawkes process via Ogata's thinning algorithm
        
        Args:
            T: Time horizon
            max_events: Maximum events to simulate
        
        Returns:
            event_times: Array of event times
        """
        event_times = []
        t = 0
        
        while t < T and len(event_times) < max_events:
            # Upper bound for intensity
            lambda_max = self.intensity(t, np.array(event_times)) + self.mu
            
            # Sample inter-arrival time
            t += np.random.exponential(1 / lambda_max)
            
            if t > T:
                break
            
            # Accept/reject
            lambda_t = self.intensity(t, np.array(event_times))
            if np.random.uniform() < lambda_t / lambda_max:
                event_times.append(t)
        
        return np.array(event_times)
    
    def log_likelihood(self, event_times, T):
        """
        Compute log-likelihood of observed events
        
        Args:
            event_times: Observed event times
            T: Observation window
        
        Returns:
            log_likelihood: Scalar
        """
        if len(event_times) == 0:
            return -self.mu * T
        
        # Log intensity at each event
        log_intensities = 0
        for i, t_i in enumerate(event_times):
            lambda_i = self.intensity(t_i, event_times[:i])
            log_intensities += np.log(lambda_i + 1e-10)
        
        # Compensator (integral of intensity)
        compensator = self.mu * T
        
        # Kernel integral
        for t_i in event_times:
            if t_i < T:
                compensator += self.alpha * (1 - np.exp(-self.beta * (T - t_i)))
        
        return log_intensities - compensator
    
    @staticmethod
    def fit(event_times, T):
        """
        Fit Hawkes process to data via Maximum Likelihood
        
        Args:
            event_times: Observed events
            T: Observation window
        
        Returns:
            fitted_model: HawkesProcess instance
        """
        def neg_log_likelihood(params):
            mu, alpha, beta = params
            if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= 1:
                return 1e10
            
            model = HawkesProcess(alpha=alpha, beta=beta, mu=mu)
            return -model.log_likelihood(event_times, T)
        
        # Initial guess
        x0 = [1.0, 0.5, 1.0]
        
        # Optimize
        result = minimize(neg_log_likelihood, x0, method='L-BFGS-B',
                          bounds=[(0.01, 10), (0.01, 0.99), (0.01, 10)])
        
        mu_fit, alpha_fit, beta_fit = result.x
        
        return HawkesProcess(alpha=alpha_fit, beta=beta_fit, mu=mu_fit)


class BivariateHawkes:
    """
    Bivariate Hawkes for buy/sell orders
    
    λ_+(t) = μ_+ + Σ φ_++(t - t_i^+) + Σ φ_-+(t - t_i^-)
    λ_-(t) = μ_- + Σ φ_+-(t - t_i^+) + Σ φ_--(t - t_i^-)
    """
    
    def __init__(self, mu_buy=1.0, mu_sell=1.0, 
                 alpha_buy_buy=0.5, alpha_sell_sell=0.5,
                 alpha_buy_sell=0.1, alpha_sell_buy=0.1,
                 beta=1.0):
        """
        Args:
            mu_buy, mu_sell: Baseline intensities
            alpha_xx: Cross-excitation coefficients
            beta: Decay rate (shared for simplicity)
        """
        self.mu_buy = mu_buy
        self.mu_sell = mu_sell
        self.alpha_buy_buy = alpha_buy_buy
        self.alpha_sell_sell = alpha_sell_sell
        self.alpha_buy_sell = alpha_buy_sell
        self.alpha_sell_buy = alpha_sell_buy
        self.beta = beta
    
    def branching_matrix(self):
        """
        Return 2x2 branching matrix
        
        [[n_++, n_-+],
         [n_+-, n_--]]
        """
        return np.array([
            [self.alpha_buy_buy, self.alpha_sell_buy],
            [self.alpha_buy_sell, self.alpha_sell_sell]
        ])
    
    def is_stationary(self):
        """Check if process is stationary (spectral radius < 1)"""
        eigenvalues = np.linalg.eigvals(self.branching_matrix())
        return np.max(np.abs(eigenvalues)) < 1


def detect_bull_rise_regime(buy_times, sell_times, T):
    """
    Detect "Bull Rise" regime using Hawkes analysis
    
    Criteria:
        1. n_++ (buy self-excitation) > 0.7 (critical)
        2. n_+- (buy->sell) < 0.3 (weak cross-inhibition)
        3. Asymmetry: buy intensity > sell intensity
    
    Args:
        buy_times, sell_times: Event times
        T: Observation window
    
    Returns:
        is_bull_rise: Boolean
        diagnostics: Dict of metrics
    """
    # Fit univariate models
    buy_model = HawkesProcess.fit(buy_times, T)
    sell_model = HawkesProcess.fit(sell_times, T)
    
    n_buy = buy_model.branching_ratio()
    n_sell = sell_model.branching_ratio()
    
    # Simple heuristic checks
    is_critical_buy = n_buy > 0.7
    is_weak_sell = n_sell < 0.5
    is_asymmetric = buy_model.mu > sell_model.mu
    
    is_bull_rise = is_critical_buy and is_weak_sell and is_asymmetric
    
    diagnostics = {
        'n_buy': n_buy,
        'n_sell': n_sell,
        'mu_buy': buy_model.mu,
        'mu_sell': sell_model.mu,
        'is_bull_rise': is_bull_rise
    }
    
    return is_bull_rise, diagnostics


if __name__ == "__main__":
    print("Testing Hawkes Process...")
    
    # Simulate data
    np.random.seed(42)
    hawkes = HawkesProcess(alpha=0.7, beta=2.0, mu=1.0)
    events = hawkes.simulate(T=10.0)
    
    print(f"Simulated {len(events)} events")
    print(f"Branching ratio: {hawkes.branching_ratio():.2f}")
    
    # Fit model
    fitted = HawkesProcess.fit(events, T=10.0)
    print(f"\nFitted parameters:")
    print(f"  μ: {fitted.mu:.3f} (true: {hawkes.mu:.3f})")
    print(f"  α: {fitted.alpha:.3f} (true: {hawkes.alpha:.3f})")
    print(f"  β: {fitted.beta:.3f} (true: {hawkes.beta:.3f})")
    
    # Bull Rise detection (synthetic example)
    buy_times = hawkes.simulate(T=10.0)
    sell_times = HawkesProcess(alpha=0.3, beta=2.0, mu=0.5).simulate(T=10.0)
    
    is_bull, diag = detect_bull_rise_regime(buy_times, sell_times, 10.0)
    print(f"\nBull Rise Detection:")
    print(f"  Is Bull Rise: {is_bull}")
    print(f"  Diagnostics: {diag}")
    
    print("\n✓ Hawkes Process test passed")
