"""
High-Frequency Trading Execution & Strategy Module
==================================================
- Swarm-Skewed Avellaneda-Stoikov Market Making
- Almgren-Chriss Optimal Execution & Algorithmic Router
- Event-Driven HFT Simulator and Backtest Harness
"""

from models.hft_execution.avellaneda_stoikov import (
    SwarmAvellanedaStoikov,
    MarketMakerQuotes,
)
from models.hft_execution.optimal_execution import (
    AlmgrenChrissExecution,
    AlgorithmicRouter,
    ExecutionTrajectory,
)
from models.hft_execution.hft_simulator import (
    HFTSimulator,
    SimulationResult,
)

__all__ = [
    "SwarmAvellanedaStoikov",
    "MarketMakerQuotes",
    "AlmgrenChrissExecution",
    "AlgorithmicRouter",
    "ExecutionTrajectory",
    "HFTSimulator",
    "SimulationResult",
]
