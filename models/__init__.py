"""
QuantFlow Institutional Models Suite
====================================
- Classical & Ensemble Options Pricing (Black-Scholes, Binomial, Monte Carlo, Greeks)
- Biomimetic Mormyrid Swarm Consensus Intelligence
- Market Microstructure Engine (Level 2/3 LOB, OFI, VPIN, Hawkes, Stoikov)
- High-Frequency Trading Execution (Avellaneda-Stoikov, Almgren-Chriss, HFT Simulator)
- Real-Time Risk & Telemetry Engine
"""

from .pricing import BlackScholesModel, BinomialTreeModel, MonteCarloSimulation
from .greeks import GreeksCalculator
from .swarm import (
    MormyridSwarmConsensusEngine,
    SwarmConsensusSignal,
    FishAgent,
    AgentRole,
    JammingAvoidanceEngine,
    ArtificialFishSwarmOptimizer,
)
from .microstructure import (
    LimitOrderBook,
    L2Snapshot,
    OrderSide,
    generate_synthetic_lob_stream,
    MultiLevelOFI,
    VPIN,
    StoikovMicroPrice,
    HawkesProcess,
    BivariateHawkes,
)
from .hft_execution import (
    SwarmAvellanedaStoikov,
    MarketMakerQuotes,
    AlmgrenChrissExecution,
    AlgorithmicRouter,
    HFTSimulator,
    SimulationResult,
)
from .risk.hft_risk import HFTRiskEngine, RiskMetricsSnapshot

__all__ = [
    "BlackScholesModel",
    "BinomialTreeModel",
    "MonteCarloSimulation",
    "GreeksCalculator",
    "MormyridSwarmConsensusEngine",
    "SwarmConsensusSignal",
    "FishAgent",
    "AgentRole",
    "JammingAvoidanceEngine",
    "ArtificialFishSwarmOptimizer",
    "LimitOrderBook",
    "L2Snapshot",
    "OrderSide",
    "generate_synthetic_lob_stream",
    "MultiLevelOFI",
    "VPIN",
    "StoikovMicroPrice",
    "HawkesProcess",
    "BivariateHawkes",
    "SwarmAvellanedaStoikov",
    "MarketMakerQuotes",
    "AlmgrenChrissExecution",
    "AlgorithmicRouter",
    "HFTSimulator",
    "SimulationResult",
    "HFTRiskEngine",
    "RiskMetricsSnapshot",
]
