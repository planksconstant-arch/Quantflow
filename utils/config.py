"""
QuantFlow Configuration Management
Handles environment variables and runtime configuration
"""

import os
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class Config:
    """Main configuration class for QuantFlow"""
    
    # Target option specification
    TICKER: str = "NVDA"
    OPTION_TYPE: str = "call"  # 'call' or 'put'
    STRIKE: float = 140.0
    EXPIRY: str = "2026-04-17"
    
    # Data settings
    HISTORICAL_DAYS: int = 365  # 12 months
    CACHE_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
    CACHE_TTL_HOURS: int = 24
    
    # Pricing model parameters
    MC_SIMULATIONS: int = 100000
    BINOMIAL_STEPS: int = 50
    RISK_FREE_RATE: Optional[float] = None  # Will fetch from FRED if None
    
    # Greeks calculation
    GREEKS_SPOT_RANGE: tuple = (0.7, 1.3)  # 70% to 130% of current spot
    GREEKS_SPOT_POINTS: int = 100
    GREEKS_TIME_POINTS: int = 50
    
    # Visualization settings
    CHART_OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "..", "outputs", "charts")
    CHART_FORMAT: str = "png"  # 'png' or 'html'
    PLOTLY_THEME: str = "plotly_dark"
    
    # ML Model parameters (Phase 2)
    VOL_LOOKBACK_WINDOWS: list = None  # [10, 20, 60]
    GARCH_P: int = 1
    GARCH_Q: int = 1
    REGIME_HIDDEN_STATES: int = 4
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    def __post_init__(self):
        """Initialize after dataclass creation"""
        if self.VOL_LOOKBACK_WINDOWS is None:
            self.VOL_LOOKBACK_WINDOWS = [10, 20, 60]
            
        # Create cache directory if it doesn't exist
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.CHART_OUTPUT_DIR, exist_ok=True)
    
    @property
    def expiry_date(self) -> datetime:
        """Convert expiry string to datetime"""
        return datetime.strptime(self.EXPIRY, "%Y-%m-%d")
    
    @property
    def cache_file_path(self) -> str:
        """Path to cached market data"""
        return os.path.join(self.CACHE_DIR, f"{self.TICKER}_data.pkl")
    
    def get_option_identifier(self) -> str:
        """Human-readable option identifier"""
        return f"{self.TICKER} ${self.STRIKE} {self.OPTION_TYPE.upper()} {self.EXPIRY}"


# Global configuration instance
config = Config()
